"""GPU rendering backend: OpenGL 3.3 core via moderngl.

Mirrors the software `Renderer`'s deferred-shading math per fragment instead
of per pixel-buffer-sample, so the two paths should look nearly identical:
same directional/ambient-cube light, same point/spot falloff + cone + IES
curve, same fog blend. The one thing that stays on the CPU is shadowing --
`raytrace.ShadowTracer` still ray-casts per face against the triangle soup
(that logic is agnostic to how the result gets shaded), and its per-face
factors are uploaded into a small texture the fragment shader samples by a
global face id (`entity's face offset in the scene + local face index`).

Geometry is a per-entity, non-indexed "vertex soup" in LOCAL space (position,
flat-shaded face normal, face color, local face index), cached by
`id(entity.mesh)`; only the color buffer is rebuilt when the material editor
swaps `mesh.face_colors` for a new array. Model/normal matrices are per-draw
uniforms -- one draw call per entity, which is plenty for the entity counts
this engine deals with.

Two ways to get a GLRenderer: attach to the window's context
(`GLRenderer(ctx)`, targets `ctx.screen`) or `GLRenderer.standalone(w, h)`
for headless unit tests (its own context + an off-screen FBO).
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from .gpu_geometry import _build_color, _build_geometry, _entity_world_faces, _scene_environment
from .lighting import _IES_CURVES
from .math3d import rotation_x, rotation_y
from .renderer import _face_light_strength, _gather_lights

_MAX_LIGHTS = 16

# ---------------------------------------------------------------------------
# shaders
# ---------------------------------------------------------------------------
_MESH_VS = """
#version 330 core
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
in float in_faceid;

uniform mat4 mvp;
uniform mat4 model;
uniform mat3 normalMat;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vColor;
flat out int vFaceId;

void main() {
    vec4 world = model * vec4(in_pos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(normalMat * in_normal);
    vColor = in_color;
    vFaceId = int(in_faceid + 0.5);
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

_MESH_FS = f"""
#version 330 core
#define MAX_LIGHTS {_MAX_LIGHTS}

in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vColor;
flat in int vFaceId;

uniform vec3 cameraPos;
uniform int faceOffset;

uniform vec3 dlDir;
uniform vec3 dlColor;
uniform float dlAmbient;
uniform int useEnv;
uniform vec3 ambientCube[6];
uniform float envStrength;

uniform int nLights;
uniform vec3 lightPos[MAX_LIGHTS];
uniform vec3 lightColor[MAX_LIGHTS];
uniform vec3 lightAxis[MAX_LIGHTS];
uniform float lightIntensity[MAX_LIGHTS];
uniform float lightRange[MAX_LIGHTS];
uniform float lightCosIn[MAX_LIGHTS];
uniform float lightCosOut[MAX_LIGHTS];
uniform int lightIesRow[MAX_LIGHTS];

uniform sampler2D shadowTex;
uniform sampler2D iesTex;

uniform int fogEnabled;
uniform vec3 fogColor;
uniform float fogStart;
uniform float fogEnd;

out vec4 fragColor;

// Same cosine-weighted cube lookup as Environment.ambient(): axes are
// (+X, -X, +Y, -Y, +Z, -Z), weighted by the clamped normal component.
vec3 evalAmbientCube(vec3 n) {{
    vec3 pos = max(n, 0.0);
    vec3 neg = max(-n, 0.0);
    float w0 = pos.x, w1 = neg.x, w2 = pos.y, w3 = neg.y, w4 = pos.z, w5 = neg.z;
    float wsum = max(w0 + w1 + w2 + w3 + w4 + w5, 1e-9);
    vec3 c = ambientCube[0] * w0 + ambientCube[1] * w1 + ambientCube[2] * w2
           + ambientCube[3] * w3 + ambientCube[4] * w4 + ambientCube[5] * w5;
    return (c / wsum) * envStrength;
}}

void main() {{
    vec3 n = normalize(vNormal);
    vec3 toLight = normalize(-dlDir);
    float lambert = clamp(dot(n, toLight), 0.0, 1.0);
    vec3 lum;
    if (useEnv != 0) {{
        lum = evalAmbientCube(n) + dlColor * lambert;
    }} else {{
        lum = vec3(dlAmbient) + dlColor * ((1.0 - dlAmbient) * lambert);
    }}

    for (int i = 0; i < nLights; i++) {{
        vec3 delta = lightPos[i] - vWorldPos;
        float dist = max(length(delta), 1e-6);
        float atten = clamp(1.0 - dist / lightRange[i], 0.0, 1.0);
        atten *= atten;
        float lambertL = clamp(dot(n, delta) / dist, 0.0, 1.0);
        float strength = lightIntensity[i] * atten * lambertL;

        bool isSpot = lightCosIn[i] > -1.5;
        bool hasIes = lightIesRow[i] >= 0;
        if (isSpot || hasIes) {{
            vec3 toFrag = -delta / dist;
            float cosAng = dot(toFrag, lightAxis[i]);
            if (isSpot) {{
                float cone = clamp((cosAng - lightCosOut[i])
                    / max(lightCosIn[i] - lightCosOut[i], 1e-6), 0.0, 1.0);
                strength *= cone * cone;
            }}
            if (hasIes) {{
                float ang = degrees(acos(clamp(cosAng, -1.0, 1.0)));
                float mul = texelFetch(iesTex, ivec2(int(ang), lightIesRow[i]), 0).r;
                strength *= mul;
            }}
        }}

        float shadow = texelFetch(shadowTex, ivec2(i, faceOffset + vFaceId), 0).r;
        strength *= shadow;

        lum += lightColor[i] * strength;
    }}

    vec3 outColor = vColor * lum;
    if (fogEnabled != 0) {{
        float dist = length(vWorldPos - cameraPos);
        float f = clamp((dist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
        outColor = mix(outColor, fogColor, f);
    }}
    fragColor = vec4(clamp(outColor, 0.0, 1.0), 1.0);
}}
"""

_SKY_VS = """
#version 330 core
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_SKY_FS = """
#version 330 core
uniform vec3 camRight, camUp, camFwd;
uniform vec2 viewSize;
uniform float focalK;
uniform int useEnv;
uniform sampler2D envTex;
uniform int useGradient;
uniform vec3 skyTop, skyHorizon, bgColor;
out vec4 fragColor;
const float PI = 3.14159265359;

void main() {
    if (useEnv != 0) {
        float dcx = (gl_FragCoord.x - viewSize.x * 0.5) / focalK;
        float dcy = (gl_FragCoord.y - viewSize.y * 0.5) / focalK;
        vec3 dir = normalize(dcx * camRight + dcy * camUp + camFwd);
        float theta = acos(clamp(dir.y, -1.0, 1.0));
        float phi = atan(dir.z, dir.x);
        if (phi < 0.0) phi += 2.0 * PI;
        vec2 uv = vec2(phi / (2.0 * PI), theta / PI);
        vec3 radiance = texture(envTex, uv).rgb;
        fragColor = vec4(clamp(radiance, 0.0, 1.0), 1.0);
    } else if (useGradient != 0) {
        float f = clamp((viewSize.y - gl_FragCoord.y) / viewSize.y, 0.0, 1.0);
        fragColor = vec4(mix(skyTop, skyHorizon, f), 1.0);
    } else {
        fragColor = vec4(bgColor, 1.0);
    }
}
"""


def _projection(camera, w: int, h: int) -> np.ndarray:
    """Standard right-handed perspective, vertical fov, OpenGL [-1,1] depth."""
    aspect = w / max(h, 1)
    f = 1.0 / math.tan(math.radians(camera.fov) * 0.5)
    near, far = camera.near, camera.far
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _write_mat(uniform, mat: np.ndarray) -> None:
    """Upload a numpy row-major matrix (3x3 or 4x4) as a column-major GLSL uniform."""
    uniform.write(np.ascontiguousarray(mat.T, dtype=np.float32).tobytes())


class GLRenderer:
    MAX_LIGHTS = _MAX_LIGHTS
    _ENV_UNIT = 0
    _SHADOW_UNIT = 1
    _IES_UNIT = 2

    def __init__(self, ctx: "moderngl.Context", target=None):
        self.ctx = ctx
        self.target = target if target is not None else ctx.screen
        self.stats = {"mode": "gpu", "triangles": 0, "shadow_lights": 0,
                      "wireframe": False}
        self._cull_face = "back"  # flipped to "front" if winding test shows holes
        self._geo_cache: dict[int, dict] = {}
        self._env_tex_cache: dict[int, "moderngl.Texture"] = {}
        self._shadow_tex_obj = None
        self._shadow_tex_size = None

        self._mesh_prog = ctx.program(vertex_shader=_MESH_VS, fragment_shader=_MESH_FS)
        self._sky_prog = ctx.program(vertex_shader=_SKY_VS, fragment_shader=_SKY_FS)
        verts = np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32)
        self._sky_vbo = ctx.buffer(verts.tobytes())
        self._sky_vao = ctx.vertex_array(self._sky_prog, [(self._sky_vbo, "2f", "in_pos")])

        names = list(_IES_CURVES.keys())
        curves = np.stack([_IES_CURVES[n] for n in names]).astype(np.float32)
        self._ies_tex = ctx.texture((curves.shape[1], curves.shape[0]), 1, dtype="f4")
        self._ies_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._ies_tex.write(np.ascontiguousarray(curves).tobytes())
        self._ies_row_of_id = {id(_IES_CURVES[n]): i for i, n in enumerate(names)}

    @classmethod
    def standalone(cls, width: int, height: int) -> "GLRenderer":
        """Headless context + its own color+depth FBO, for unit tests."""
        ctx = moderngl.create_context(standalone=True, require=330)
        color = ctx.texture((width, height), 4)
        depth = ctx.depth_texture((width, height))
        fbo = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
        renderer = cls(ctx, target=fbo)
        renderer._owns_ctx = True
        return renderer

    @property
    def wireframe(self) -> bool:
        return self.ctx.wireframe

    @wireframe.setter
    def wireframe(self, value: bool) -> None:
        self.ctx.wireframe = bool(value)

    # ------------------------------------------------------------------
    def render(self, scene, camera, size, tracer=None) -> None:
        ctx = self.ctx
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0:
            return
        self.target.use()
        ctx.viewport = (0, 0, w, h)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        wf = ctx.wireframe
        ctx.wireframe = False
        self._draw_sky(scene, camera, w, h)
        ctx.wireframe = wf

        env = _scene_environment(scene)
        lights = _gather_lights(scene)[: self.MAX_LIGHTS]
        self.stats["shadow_lights"] = sum(1 for li in lights if li.light.cast_shadows)

        live = [e for e in scene.entities if e.mesh is not None and e.visible]
        face_counts = [int(e.mesh.faces.shape[0]) for e in live]
        total_faces = int(sum(face_counts))
        offsets = np.cumsum([0] + face_counts[:-1]).tolist() if face_counts else []

        shadow_data = np.ones((max(total_faces, 1), self.MAX_LIGHTS), dtype=np.float32)
        if tracer is not None and total_faces > 0:
            for entity, off, m in zip(live, offsets, face_counts):
                if not any(li.light.cast_shadows for li in lights):
                    break
                centroids_w, normals_w = _entity_world_faces(entity)
                for li_idx, info in enumerate(lights):
                    if not info.light.cast_shadows:
                        continue
                    strength = _face_light_strength(info, normals_w, centroids_w)
                    active = strength > 1e-3
                    factors = tracer.shadow_factors(entity, info.light, info.pos,
                                                     centroids_w, normals_w, active)
                    shadow_data[off:off + m, li_idx] = factors
        self._upload_shadow_tex(shadow_data, total_faces)

        view = camera.view_matrix()
        proj = _projection(camera, w, h)

        prog = self._mesh_prog
        self._set_light_uniforms(prog, scene, lights, env)
        prog["cameraPos"].value = tuple(camera.position.to_array().astype(np.float32))
        fog = scene.fog
        prog["fogEnabled"].value = 1 if fog is not None else 0
        if fog is not None:
            prog["fogColor"].value = tuple(np.asarray(fog.color, dtype=np.float32) / 255.0)
            prog["fogStart"].value = float(fog.start)
            prog["fogEnd"].value = float(fog.end)

        self._shadow_tex_obj.use(location=self._SHADOW_UNIT)
        prog["shadowTex"].value = self._SHADOW_UNIT
        self._ies_tex.use(location=self._IES_UNIT)
        prog["iesTex"].value = self._IES_UNIT

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = self._cull_face

        triangles = 0
        for entity, off in zip(live, offsets):
            if entity.mesh.faces.shape[0] == 0:
                continue
            cache = self._get_geo_cache(entity.mesh)
            model = entity.transform.matrix()
            mvp = proj @ view @ model
            _write_mat(prog["mvp"], mvp)
            _write_mat(prog["model"], model)
            try:
                nmat = np.linalg.inv(model[:3, :3]).T
            except np.linalg.LinAlgError:
                nmat = np.eye(3)
            _write_mat(prog["normalMat"], nmat)
            prog["faceOffset"].value = int(off)
            cache["vao"].render(moderngl.TRIANGLES, vertices=cache["count"])
            triangles += cache["count"] // 3

        self._prune_geo_cache(live)
        self.stats["triangles"] = triangles
        self.stats["wireframe"] = ctx.wireframe

    # ------------------------------------------------------------------
    def _draw_sky(self, scene, camera, w: int, h: int) -> None:
        prog = self._sky_prog
        k = 0.5 * h / math.tan(math.radians(camera.fov) * 0.5)
        rot = (rotation_y(camera.yaw) @ rotation_x(camera.pitch))[:3, :3]
        right = rot @ np.array([1.0, 0.0, 0.0])
        up = rot @ np.array([0.0, 1.0, 0.0])
        fwd = rot @ np.array([0.0, 0.0, -1.0])
        prog["camRight"].value = tuple(right.astype(np.float32))
        prog["camUp"].value = tuple(up.astype(np.float32))
        prog["camFwd"].value = tuple(fwd.astype(np.float32))
        prog["viewSize"].value = (float(w), float(h))
        prog["focalK"].value = float(k)

        env = _scene_environment(scene)
        if env is not None:
            tex = self._get_env_tex(env)
            tex.use(location=self._ENV_UNIT)
            prog["envTex"].value = self._ENV_UNIT
            prog["useEnv"].value = 1
            prog["useGradient"].value = 0
        else:
            prog["useEnv"].value = 0
            if scene.sky is not None:
                prog["useGradient"].value = 1
                prog["skyTop"].value = tuple(np.asarray(scene.sky[0], dtype=np.float32) / 255.0)
                prog["skyHorizon"].value = tuple(
                    np.asarray(scene.sky[1], dtype=np.float32) / 255.0)
            else:
                prog["useGradient"].value = 0
                prog["bgColor"].value = tuple(
                    np.asarray(scene.background, dtype=np.float32) / 255.0)

        self._sky_vao.render(moderngl.TRIANGLES)

    def _set_light_uniforms(self, prog, scene, lights, env) -> None:
        dl = scene.light
        prog["dlDir"].value = tuple(dl.direction.to_array().astype(np.float32))
        dl_color = (np.asarray(dl.color, dtype=np.float32) / 255.0) * dl.intensity
        prog["dlColor"].value = tuple(dl_color)
        prog["dlAmbient"].value = float(dl.ambient)
        prog["useEnv"].value = 1 if env is not None else 0
        if env is not None:
            prog["ambientCube"].write(
                np.ascontiguousarray(env.ambient_cube, dtype=np.float32).tobytes())
            prog["envStrength"].value = float(env.strength)

        n = len(lights)
        prog["nLights"].value = n
        pos = np.zeros((self.MAX_LIGHTS, 3), dtype=np.float32)
        color = np.zeros((self.MAX_LIGHTS, 3), dtype=np.float32)
        axis = np.zeros((self.MAX_LIGHTS, 3), dtype=np.float32)
        intensity = np.zeros(self.MAX_LIGHTS, dtype=np.float32)
        rng = np.ones(self.MAX_LIGHTS, dtype=np.float32)
        cos_in = np.full(self.MAX_LIGHTS, -2.0, dtype=np.float32)
        cos_out = np.zeros(self.MAX_LIGHTS, dtype=np.float32)
        ies_row = np.full(self.MAX_LIGHTS, -1, dtype=np.int32)
        for i, info in enumerate(lights):
            pos[i] = info.pos
            color[i] = info.colorf
            axis[i] = info.axis
            intensity[i] = info.light.intensity
            rng[i] = max(info.light.range, 1e-6)
            if info.cos_in is not None:
                cos_in[i] = info.cos_in
                cos_out[i] = info.cos_out
            if info.curve is not None:
                ies_row[i] = self._ies_row_of_id.get(id(info.curve), -1)
        prog["lightPos"].write(pos.tobytes())
        prog["lightColor"].write(color.tobytes())
        prog["lightAxis"].write(axis.tobytes())
        prog["lightIntensity"].write(intensity.tobytes())
        prog["lightRange"].write(rng.tobytes())
        prog["lightCosIn"].write(cos_in.tobytes())
        prog["lightCosOut"].write(cos_out.tobytes())
        prog["lightIesRow"].write(ies_row.tobytes())

    # ------------------------------------------------------------------
    def _get_geo_cache(self, mesh) -> dict:
        key = id(mesh)
        cache = self._geo_cache.get(key)
        if cache is None:
            pos, nrm, fid, face_id_tri, m = _build_geometry(mesh)
            geom = np.concatenate([pos, nrm, fid[:, None]], axis=1).astype(np.float32)
            vbo_geom = self.ctx.buffer(np.ascontiguousarray(geom).tobytes())
            color = _build_color(mesh, face_id_tri)
            vbo_color = self.ctx.buffer(color.tobytes())
            vao = self.ctx.vertex_array(self._mesh_prog, [
                (vbo_geom, "3f 3f 1f", "in_pos", "in_normal", "in_faceid"),
                (vbo_color, "3f", "in_color"),
            ])
            cache = {"vbo_geom": vbo_geom, "vbo_color": vbo_color, "vao": vao,
                     "count": pos.shape[0], "num_faces": m,
                     "face_id_tri": face_id_tri, "color_id": id(mesh.face_colors)}
            self._geo_cache[key] = cache
        elif cache["color_id"] != id(mesh.face_colors):
            color = _build_color(mesh, cache["face_id_tri"])
            cache["vbo_color"].write(color.tobytes())
            cache["color_id"] = id(mesh.face_colors)
        return cache

    def _prune_geo_cache(self, live_entities) -> None:
        live_ids = {id(e.mesh) for e in live_entities}
        for key in [k for k in self._geo_cache if k not in live_ids]:
            c = self._geo_cache.pop(key)
            c["vao"].release()
            c["vbo_geom"].release()
            c["vbo_color"].release()

    def _get_env_tex(self, env) -> "moderngl.Texture":
        key = id(env.image)
        tex = self._env_tex_cache.get(key)
        if tex is None:
            for old in self._env_tex_cache.values():
                old.release()
            self._env_tex_cache.clear()
            h, w = env.image.shape[:2]
            tex = self.ctx.texture((w, h), 3, dtype="f4")
            tex.write(np.ascontiguousarray(env.image, dtype=np.float32).tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.repeat_y = False
            self._env_tex_cache[key] = tex
        return tex

    def _upload_shadow_tex(self, shadow_data: np.ndarray, total_faces: int) -> None:
        size = (self.MAX_LIGHTS, max(total_faces, 1))
        if self._shadow_tex_obj is None or self._shadow_tex_size != size:
            if self._shadow_tex_obj is not None:
                self._shadow_tex_obj.release()
            self._shadow_tex_obj = self.ctx.texture(size, 1, dtype="f4")
            self._shadow_tex_obj.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._shadow_tex_size = size
        self._shadow_tex_obj.write(np.ascontiguousarray(shadow_data, dtype=np.float32).tobytes())

    # ------------------------------------------------------------------
    def release(self) -> None:
        """Free GL resources (buffers/textures/programs). Does not free `ctx`."""
        self._prune_geo_cache([])
        for tex in self._env_tex_cache.values():
            tex.release()
        self._env_tex_cache.clear()
        if self._shadow_tex_obj is not None:
            self._shadow_tex_obj.release()
            self._shadow_tex_obj = None
        self._ies_tex.release()
        self._sky_vao.release()
        self._sky_vbo.release()
        self._mesh_prog.release()
        self._sky_prog.release()
