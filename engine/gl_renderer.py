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

from .gpu_geometry import (_build_color, _build_geometry, _build_pbr,
                           _entity_world_faces, _scene_environment)
from .lighting import _IES_CURVES
from .math3d import rotation_x, rotation_y
from .raytrace import GITracer
from .renderer import (_face_light_strength, _fog_volumes, _gather_lights,
                       _gi_direct_lighting, _gi_receiver_geometry, _scene_sun)

_MAX_LIGHTS = 16
_MAX_FOG_VOL = 4
_FOG_SKY_FAR = 260.0  # path-length clip for fog volumes behind the sky

# ---------------------------------------------------------------------------
# shaders
# ---------------------------------------------------------------------------
_MESH_VS = """
#version 330 core
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
in float in_faceid;
in vec2 in_rm;
in vec3 in_emissive;

uniform mat4 mvp;
uniform mat4 model;
uniform mat3 normalMat;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vColor;
flat out int vFaceId;
out float vRoughness;
out float vMetallic;
out vec3 vEmissive;

void main() {
    vec4 world = model * vec4(in_pos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(normalMat * in_normal);
    vColor = in_color;
    vFaceId = int(in_faceid + 0.5);
    vRoughness = in_rm.x;
    vMetallic = in_rm.y;
    vEmissive = in_emissive;
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
in float vRoughness;
in float vMetallic;
in vec3 vEmissive;

const float PI = 3.14159265359;

uniform vec3 cameraPos;
uniform int faceOffset;

uniform vec3 dlDir;
uniform vec3 dlColor;
uniform vec3 dlColorRaw;
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
uniform sampler2D dlShadowTex;
uniform sampler2D giTex;
uniform sampler2D iesTex;

uniform int fogEnabled;
uniform vec3 fogColor;
uniform float fogStart;
uniform float fogEnd;
uniform float fogHeightFalloff;
uniform float fogSunScatter;

#define MAX_FOG_VOL {_MAX_FOG_VOL}
uniform int fogVolCount;
uniform vec3 fogVolLo[MAX_FOG_VOL];
uniform vec3 fogVolHi[MAX_FOG_VOL];
uniform float fogVolDensity[MAX_FOG_VOL];
uniform vec3 fogVolColor[MAX_FOG_VOL];
uniform float fogVolHeightFalloff[MAX_FOG_VOL];

out vec4 fragColor;

// Ray-AABB slab intersection clipped to [tNear, tFar]; returns the segment
// (t0, t1) inside the box (t1 < t0 means no overlap).
vec2 fogVolSegment(vec3 origin, vec3 dir, vec3 lo, vec3 hi, float tNear, float tFar) {{
    vec3 invDir = 1.0 / dir;
    vec3 t1v = (lo - origin) * invDir;
    vec3 t2v = (hi - origin) * invDir;
    vec3 tmin = min(t1v, t2v);
    vec3 tmax = max(t1v, t2v);
    float t0 = max(max(max(tmin.x, tmin.y), tmin.z), tNear);
    float t1 = min(min(min(tmax.x, tmax.y), tmax.z), tFar);
    return vec2(t0, t1);
}}

vec3 applyFogVolumes(vec3 color, vec3 origin, vec3 dir, float tNear, float tFar) {{
    for (int i = 0; i < fogVolCount; i++) {{
        vec2 seg = fogVolSegment(origin, dir, fogVolLo[i], fogVolHi[i], tNear, tFar);
        float segLen = max(seg.y - seg.x, 0.0);
        float density = fogVolDensity[i];
        if (fogVolHeightFalloff[i] > 0.0) {{
            float midH = origin.y + dir.y * (0.5 * (seg.x + seg.y));
            density *= exp(-max(midH, 0.0) * fogVolHeightFalloff[i]);
        }}
        float T = exp(-density * segLen);
        color = color * T + fogVolColor[i] * (1.0 - T);
    }}
    return color;
}}

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

// Cook-Torrance/GGX specular BRDF, mirroring renderer.py's `_ggx_specular`
// exactly: Smith-GGX geometry with the direct-lighting k=(a+1)^2/8 remap
// (Karis/UE4), Schlick Fresnel. Caller multiplies by NdotL*radiance.
vec3 ggxSpecular(vec3 n, vec3 v, vec3 l, float alpha, vec3 f0) {{
    vec3 h = normalize(v + l);
    float ndoth = clamp(dot(n, h), 0.0, 1.0);
    float ndotv = clamp(dot(n, v), 1e-4, 1.0);
    float ndotl = clamp(dot(n, l), 1e-4, 1.0);
    float vdoth = clamp(dot(v, h), 0.0, 1.0);

    float a2 = alpha * alpha;
    float denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    float d = a2 / max(PI * denom * denom, 1e-8);

    float k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
    float g1v = ndotv / max(ndotv * (1.0 - k) + k, 1e-8);
    float g1l = ndotl / max(ndotl * (1.0 - k) + k, 1e-8);
    float g = g1v * g1l;

    vec3 f = f0 + (1.0 - f0) * pow(1.0 - vdoth, 5.0);
    return (d * g / max(4.0 * ndotv * ndotl, 1e-4)) * f;
}}

void main() {{
    vec3 n = normalize(vNormal);
    vec3 toLight = normalize(-dlDir);
    float lambert = clamp(dot(n, toLight), 0.0, 1.0);
    float dlShadow = texelFetch(dlShadowTex, ivec2(0, faceOffset + vFaceId), 0).r;
    lambert *= dlShadow;
    vec3 lum;
    if (useEnv != 0) {{
        lum = evalAmbientCube(n) + dlColor * lambert;
    }} else {{
        lum = vec3(dlAmbient) + dlColor * ((1.0 - dlAmbient) * lambert);
    }}
    lum += texelFetch(giTex, ivec2(0, faceOffset + vFaceId), 0).rgb;

    // PBR setup -- alpha/f0/specScale mirror renderer.py's per-pixel gather
    // exactly. specScale = 1 - roughness*(1-metallic) is the backward-compat
    // gate: zero at the default params (roughness=1, metallic=0).
    float alpha = clamp(vRoughness, 0.02, 1.0);
    alpha *= alpha;
    vec3 f0 = mix(vec3(0.04), vColor, vMetallic);
    float specScale = 1.0 - vRoughness * (1.0 - vMetallic);
    vec3 viewDir = normalize(cameraPos - vWorldPos);
    vec3 spec = vec3(0.0);

    if (specScale > 1e-6 && lambert > 1e-4) {{
        vec3 brdf = ggxSpecular(n, viewDir, toLight, alpha, f0);
        spec += dlColor * brdf * (lambert * specScale);
    }}

    for (int i = 0; i < nLights; i++) {{
        vec3 delta = lightPos[i] - vWorldPos;
        float dist = max(length(delta), 1e-6);
        float atten = clamp(1.0 - dist / lightRange[i], 0.0, 1.0);
        atten *= atten;
        float lambertL = clamp(dot(n, delta) / dist, 0.0, 1.0);
        float radiance = lightIntensity[i] * atten;

        bool isSpot = lightCosIn[i] > -1.5;
        bool hasIes = lightIesRow[i] >= 0;
        if (isSpot || hasIes) {{
            vec3 toFrag = -delta / dist;
            float cosAng = dot(toFrag, lightAxis[i]);
            if (isSpot) {{
                float cone = clamp((cosAng - lightCosOut[i])
                    / max(lightCosIn[i] - lightCosOut[i], 1e-6), 0.0, 1.0);
                radiance *= cone * cone;
            }}
            if (hasIes) {{
                float ang = degrees(acos(clamp(cosAng, -1.0, 1.0)));
                float mul = texelFetch(iesTex, ivec2(int(ang), lightIesRow[i]), 0).r;
                radiance *= mul;
            }}
        }}

        float shadow = texelFetch(shadowTex, ivec2(i, faceOffset + vFaceId), 0).r;
        radiance *= shadow;

        lum += lightColor[i] * (radiance * lambertL);

        if (specScale > 1e-6 && radiance > 1e-4) {{
            vec3 lDir = delta / dist;
            vec3 brdf = ggxSpecular(n, viewDir, lDir, alpha, f0);
            spec += lightColor[i] * brdf * (radiance * lambertL * specScale);
        }}
    }}

    // Metallic diffuse gate applies to the whole accumulated diffuse-ish
    // term at once (ambient+directional+GI+point/spot lambert) -- a single
    // scalar multiplier distributes linearly over the sum, matching
    // renderer.py's per-term gating exactly.
    lum *= (1.0 - vMetallic);
    vec3 outColor = vColor * lum + spec;

    vec3 viewDelta = vWorldPos - cameraPos;
    float fragDist = length(viewDelta);
    vec3 rayDir = viewDelta / max(fragDist, 1e-6);
    if (fogVolCount > 0) {{
        outColor = applyFogVolumes(outColor, cameraPos, rayDir, 0.0, fragDist);
    }}

    if (fogEnabled != 0) {{
        float f = clamp((fragDist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
        if (fogHeightFalloff > 0.0) {{
            f = clamp(f * exp(-max(vWorldPos.y, 0.0) * fogHeightFalloff), 0.0, 1.0);
        }}
        vec3 fogCol = fogColor;
        if (fogSunScatter > 0.0) {{
            float align = clamp(dot(rayDir, normalize(-dlDir)), 0.0, 1.0);
            float scatter = pow(align, 8.0) * fogSunScatter;
            fogCol = mix(fogColor, dlColorRaw, scatter);
        }}
        outColor = mix(outColor, fogCol, f);
    }}

    // Emissive is unconditional -- visible even unlit/in shadow, added
    // after fog, matching renderer.py's deferred pass.
    outColor += vEmissive;
    fragColor = vec4(clamp(outColor, 0.0, 1.0), 1.0);
}}
"""

_SKY_VS = """
#version 330 core
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_SKY_FS = f"""
#version 330 core
uniform vec3 camRight, camUp, camFwd;
uniform vec3 cameraPos;
uniform vec2 viewSize;
uniform float focalK;
uniform int useEnv;
uniform sampler2D envTex;
uniform int useGradient;
uniform vec3 skyTop, skyHorizon, bgColor;

uniform int sunEnabled;
uniform vec3 sunDir;
uniform vec3 sunColor;
uniform float sunDiscSize;
uniform float sunDiscSoftness;
uniform float sunGlow;

#define MAX_FOG_VOL {_MAX_FOG_VOL}
uniform int fogVolCount;
uniform vec3 fogVolLo[MAX_FOG_VOL];
uniform vec3 fogVolHi[MAX_FOG_VOL];
uniform float fogVolDensity[MAX_FOG_VOL];
uniform vec3 fogVolColor[MAX_FOG_VOL];
uniform float fogVolHeightFalloff[MAX_FOG_VOL];
uniform float fogVolFar;

out vec4 fragColor;
const float PI = 3.14159265359;

void main() {{
    float dcx = (gl_FragCoord.x - viewSize.x * 0.5) / focalK;
    float dcy = (gl_FragCoord.y - viewSize.y * 0.5) / focalK;
    vec3 dir = normalize(dcx * camRight + dcy * camUp + camFwd);

    vec3 base;
    if (useEnv != 0) {{
        float theta = acos(clamp(dir.y, -1.0, 1.0));
        float phi = atan(dir.z, dir.x);
        if (phi < 0.0) phi += 2.0 * PI;
        vec2 uv = vec2(phi / (2.0 * PI), theta / PI);
        base = texture(envTex, uv).rgb;
    }} else if (useGradient != 0) {{
        float f = clamp((viewSize.y - gl_FragCoord.y) / viewSize.y, 0.0, 1.0);
        base = mix(skyTop, skyHorizon, f);
    }} else {{
        base = bgColor;
    }}

    if (sunEnabled != 0) {{
        float cosAng = clamp(dot(dir, sunDir), -1.0, 1.0);
        float ang = degrees(acos(cosAng));
        float soft = max(sunDiscSize * sunDiscSoftness, 0.001);
        float e0 = sunDiscSize - soft;
        float e1 = sunDiscSize + soft;
        float t = clamp((e1 - ang) / max(e1 - e0, 0.0001), 0.0, 1.0);
        float disc = t * t * (3.0 - 2.0 * t);
        float haloDeg = sunDiscSize * 12.0 + 3.0;
        float g = clamp(1.0 - ang / haloDeg, 0.0, 1.0);
        float glowAmt = sunGlow * g * g * g;
        base += sunColor * disc + sunColor * glowAmt * 0.5;
    }}

    for (int i = 0; i < fogVolCount; i++) {{
        vec3 invDir = 1.0 / dir;
        vec3 t1v = (fogVolLo[i] - cameraPos) * invDir;
        vec3 t2v = (fogVolHi[i] - cameraPos) * invDir;
        vec3 tmin = min(t1v, t2v);
        vec3 tmax = max(t1v, t2v);
        float t0 = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
        float t1 = min(min(min(tmax.x, tmax.y), tmax.z), fogVolFar);
        float segLen = max(t1 - t0, 0.0);
        float density = fogVolDensity[i];
        if (fogVolHeightFalloff[i] > 0.0) {{
            float midH = cameraPos.y + dir.y * (0.5 * (t0 + t1));
            density *= exp(-max(midH, 0.0) * fogVolHeightFalloff[i]);
        }}
        float T = exp(-density * segLen);
        base = base * T + fogVolColor[i] * (1.0 - T);
    }}

    fragColor = vec4(clamp(base, 0.0, 1.0), 1.0);
}}
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
    _DL_SHADOW_UNIT = 3
    _GI_UNIT = 4

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
        self._dl_shadow_tex_obj = None
        self._dl_shadow_tex_size = None
        self._gi_tex_obj = None
        self._gi_tex_size = None
        self._gi = GITracer()

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

        # directional (sun) shadow -- reserved single-column texture, same
        # face-id indexing as shadowTex above
        sun = _scene_sun(scene)
        dl_shadow_data = np.ones((max(total_faces, 1), 1), dtype=np.float32)
        if tracer is not None and sun is not None and sun.shadow_depth > 1e-6 and total_faces > 0:
            dl_dir = scene.light.direction.to_array()
            to_light = -dl_dir / max(np.linalg.norm(dl_dir), 1e-12)
            for entity, off, m in zip(live, offsets, face_counts):
                centroids_w, normals_w = _entity_world_faces(entity)
                lambert = np.clip(normals_w @ to_light, 0.0, 1.0)
                active = lambert > 1e-3
                raw = tracer.directional_shadow_factors(
                    entity, dl_dir, sun.shadow_softness, sun.shadow_samples,
                    centroids_w, normals_w, active)
                dl_shadow_data[off:off + m, 0] = 1.0 - sun.shadow_depth * (1.0 - raw)
        self._upload_dl_shadow_tex(dl_shadow_data, total_faces)

        # one-bounce GI -- baked/cached by GITracer, zero per-frame cost once static
        gi_data = np.zeros((max(total_faces, 1), 3), dtype=np.float32)
        gi_cfg = getattr(scene, "gi", None)
        if tracer is not None and gi_cfg and gi_cfg.get("enabled") and total_faces > 0:
            gi_map = self._gi.compute(scene, tracer,
                                      lambda casters: _gi_direct_lighting(scene, casters, tracer),
                                      _gi_receiver_geometry,
                                      gi_cfg.get("samples", 16), gi_cfg.get("intensity", 1.0))
            for entity, off, m in zip(live, offsets, face_counts):
                gi = gi_map.get(id(entity))
                if gi is not None:
                    gi_data[off:off + m] = gi
        self._upload_gi_tex(gi_data, total_faces)

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
            prog["fogHeightFalloff"].value = float(fog.height_falloff)
            prog["fogSunScatter"].value = float(fog.sun_scatter)
            prog["dlColorRaw"].value = tuple(
                np.asarray(scene.light.color, dtype=np.float32) / 255.0)
        else:
            prog["fogHeightFalloff"].value = 0.0
            prog["fogSunScatter"].value = 0.0
            prog["dlColorRaw"].value = (0.0, 0.0, 0.0)
        self._set_fog_volume_uniforms(prog, _fog_volumes(scene))

        self._shadow_tex_obj.use(location=self._SHADOW_UNIT)
        prog["shadowTex"].value = self._SHADOW_UNIT
        self._dl_shadow_tex_obj.use(location=self._DL_SHADOW_UNIT)
        prog["dlShadowTex"].value = self._DL_SHADOW_UNIT
        self._gi_tex_obj.use(location=self._GI_UNIT)
        prog["giTex"].value = self._GI_UNIT
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
        prog["cameraPos"].value = tuple(camera.position.to_array().astype(np.float32))
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

        sun = _scene_sun(scene)
        if sun is not None and sun.enabled:
            dl_dir = scene.light.direction.to_array()
            n = np.linalg.norm(dl_dir)
            prog["sunEnabled"].value = 1 if n > 1e-9 else 0
            if n > 1e-9:
                prog["sunDir"].value = tuple((-dl_dir / n).astype(np.float32))
                dl_col = (np.asarray(scene.light.color, dtype=np.float32) / 255.0
                         * scene.light.intensity)
                prog["sunColor"].value = tuple(dl_col)
                prog["sunDiscSize"].value = float(max(sun.disc_size, 0.05))
                prog["sunDiscSoftness"].value = float(np.clip(sun.disc_softness, 0.0, 1.0))
                prog["sunGlow"].value = float(np.clip(sun.glow, 0.0, 1.0))
        else:
            prog["sunEnabled"].value = 0

        prog["fogVolFar"].value = float(_FOG_SKY_FAR)
        self._set_fog_volume_uniforms(prog, _fog_volumes(scene))

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

    def _set_fog_volume_uniforms(self, prog, vols) -> None:
        """Upload up to `_MAX_FOG_VOL` fog volumes; shared by the mesh and
        sky programs (both declare the same uniform names)."""
        prog["fogVolCount"].value = len(vols)
        lo = np.zeros((_MAX_FOG_VOL, 3), dtype=np.float32)
        hi = np.zeros((_MAX_FOG_VOL, 3), dtype=np.float32)
        density = np.zeros(_MAX_FOG_VOL, dtype=np.float32)
        color = np.zeros((_MAX_FOG_VOL, 3), dtype=np.float32)
        height_falloff = np.zeros(_MAX_FOG_VOL, dtype=np.float32)
        for i, (vlo, vhi, fv) in enumerate(vols):
            lo[i] = vlo
            hi[i] = vhi
            density[i] = fv.density
            color[i] = np.asarray(fv.color, dtype=np.float32) / 255.0
            height_falloff[i] = fv.height_falloff
        prog["fogVolLo"].write(lo.tobytes())
        prog["fogVolHi"].write(hi.tobytes())
        prog["fogVolDensity"].write(density.tobytes())
        prog["fogVolColor"].write(color.tobytes())
        prog["fogVolHeightFalloff"].write(height_falloff.tobytes())

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
            rm, emissive = _build_pbr(mesh, face_id_tri)
            pbr = np.concatenate([rm, emissive], axis=1)
            vbo_pbr = self.ctx.buffer(np.ascontiguousarray(pbr).tobytes())
            vao = self.ctx.vertex_array(self._mesh_prog, [
                (vbo_geom, "3f 3f 1f", "in_pos", "in_normal", "in_faceid"),
                (vbo_color, "3f", "in_color"),
                (vbo_pbr, "2f 3f", "in_rm", "in_emissive"),
            ])
            cache = {"vbo_geom": vbo_geom, "vbo_color": vbo_color,
                     "vbo_pbr": vbo_pbr, "vao": vao,
                     "count": pos.shape[0], "num_faces": m,
                     "face_id_tri": face_id_tri, "color_id": id(mesh.face_colors),
                     "pbr_id": (id(mesh.face_roughness), id(mesh.face_metallic),
                                id(mesh.face_emissive))}
            self._geo_cache[key] = cache
        else:
            if cache["color_id"] != id(mesh.face_colors):
                color = _build_color(mesh, cache["face_id_tri"])
                cache["vbo_color"].write(color.tobytes())
                cache["color_id"] = id(mesh.face_colors)
            pbr_id = (id(mesh.face_roughness), id(mesh.face_metallic),
                     id(mesh.face_emissive))
            if cache["pbr_id"] != pbr_id:
                rm, emissive = _build_pbr(mesh, cache["face_id_tri"])
                pbr = np.concatenate([rm, emissive], axis=1)
                cache["vbo_pbr"].write(np.ascontiguousarray(pbr).tobytes())
                cache["pbr_id"] = pbr_id
        return cache

    def _prune_geo_cache(self, live_entities) -> None:
        live_ids = {id(e.mesh) for e in live_entities}
        for key in [k for k in self._geo_cache if k not in live_ids]:
            c = self._geo_cache.pop(key)
            c["vao"].release()
            c["vbo_geom"].release()
            c["vbo_color"].release()
            c["vbo_pbr"].release()

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

    def _upload_dl_shadow_tex(self, data: np.ndarray, total_faces: int) -> None:
        """Reserved single-column texture for the directional (sun) shadow
        factor, indexed like `shadowTex` but with a fixed light index of 0."""
        size = (1, max(total_faces, 1))
        if self._dl_shadow_tex_obj is None or self._dl_shadow_tex_size != size:
            if self._dl_shadow_tex_obj is not None:
                self._dl_shadow_tex_obj.release()
            self._dl_shadow_tex_obj = self.ctx.texture(size, 1, dtype="f4")
            self._dl_shadow_tex_obj.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._dl_shadow_tex_size = size
        self._dl_shadow_tex_obj.write(np.ascontiguousarray(data, dtype=np.float32).tobytes())

    def _upload_gi_tex(self, data: np.ndarray, total_faces: int) -> None:
        """Per-face RGB indirect-light texture, same face-id indexing."""
        size = (1, max(total_faces, 1))
        if self._gi_tex_obj is None or self._gi_tex_size != size:
            if self._gi_tex_obj is not None:
                self._gi_tex_obj.release()
            self._gi_tex_obj = self.ctx.texture(size, 3, dtype="f4")
            self._gi_tex_obj.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._gi_tex_size = size
        self._gi_tex_obj.write(np.ascontiguousarray(data, dtype=np.float32).tobytes())

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
        if self._dl_shadow_tex_obj is not None:
            self._dl_shadow_tex_obj.release()
            self._dl_shadow_tex_obj = None
        if self._gi_tex_obj is not None:
            self._gi_tex_obj.release()
            self._gi_tex_obj = None
        self._ies_tex.release()
        self._sky_vao.release()
        self._sky_vbo.release()
        self._mesh_prog.release()
        self._sky_prog.release()
