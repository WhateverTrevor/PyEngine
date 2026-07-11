"""GPU rendering backend: DirectX 12 / Vulkan via wgpu-py (WebGPU).

Presentation model (deliberately simple): unlike `GLRenderer`, which renders
straight into the window's own GL-backed framebuffer, `WgpuRenderer` renders
OFFSCREEN into its own rgba8unorm color texture + depth24plus depth texture
sized to the window, then reads the color texture back to the CPU every
frame (`device.queue.read_texture`, which handles wgpu's 256-byte
copy-row-alignment requirement internally -- see the gotcha notes below) and
blits the result straight into `Engine.screen` -- the real pygame window
surface -- with `pygame.image.frombuffer`. From `Engine.run()`'s
perspective this backend therefore behaves exactly like the software
renderer: it fills `self.screen`, HUD/editor overlay drawing happens on top
completely unchanged, and `pygame.display.flip()` presents normally. There
is no OPENGL window flag and no GPU-side compositing pass, unlike the
moderngl path in `gl_renderer.py`. The known cost of this design is the
per-frame GPU->CPU texture readback (a full frame copy over the bus every
tick) -- acceptable for a first cut, but it is the reason this path will not
out-run `GLRenderer`'s in-framebuffer compositing at high resolutions; see
`read_frame()`.

wgpu (https://github.com/pygfx/wgpu-py) is an OPTIONAL dependency exactly
like moderngl, and independently so -- this module (and `gpu_geometry.py`,
which it shares with `gl_renderer.py`) never imports `moderngl`, so
requesting "dx12"/"vulkan" works on a machine that has wgpu but not
moderngl installed, and vice versa. `import wgpu` happens lazily inside
functions/methods here, never at module load, so the rest of the engine
imports fine without it installed. `WgpuRenderer("dx12")` /
`WgpuRenderer("vulkan")` set the `WGPU_BACKEND_TYPE` environment variable
("D3D12" / "Vulkan") *before* that lazy import -- wgpu-native reads the var
when it lazily creates its instance, and (verified empirically on this
machine) re-reads it on every `request_adapter_sync()` call, so constructing
a "dx12" renderer and then a "vulkan" one in the same process each get the
backend they asked for.

Shading mirrors `gl_renderer.py`'s GLSL as closely as WGSL allows: the same
flat-per-face-normal geometry soup (`_build_geometry`/`_build_color`,
imported from the shared, GPU-library-free `gpu_geometry.py` so all three
renderers -- software, GL, wgpu -- build triangle soup and world-space face
data from one source of truth), the same light-gathering
(`_gather_lights`/`_face_light_strength` from `renderer.py`), and the same
(total_faces, MAX_LIGHTS) float32 shadow-factor layout produced by
`raytrace.ShadowTracer` -- uploaded here as an r32float texture and sampled
with `textureLoad` (no sampler -- an exact texel fetch, like GL's
`texelFetch`).

wgpu/WebGPU gotchas handled here (each also called out at its call site):
  - `queue.read_texture()` already pads/strips the 256-byte copy-row
    alignment internally: verified with a 401px-wide readback (401*4=1604 B
    is *not* a multiple of 256) that still comes back as an exact
    W*H*4-byte contiguous buffer. No manual bytes_per_row padding/stripping
    needed on the Python side here, unlike a raw `copy_texture_to_buffer` +
    mapped-buffer readback would require.
  - WGSL struct/uniform alignment: every field in every uniform struct below
    is packed as vec4s (arrays-of-vec4 for lights, explicit-pad vec4 columns
    for matrices/normals) specifically to dodge vec3's 16-byte-alignment
    trap. Each Python packing helper sits right next to the WGSL struct it
    matches, with a field-order comment.
  - Readback row order: verified empirically (render an asymmetric
    NDC-top-left-corner triangle, read it back) that row 0 of the returned
    buffer is the TOP row of the image -- wgpu needs no GL-style vertical
    flip before `pygame.image.frombuffer`.
  - Clip space: WebGPU/D3D NDC z is [0, 1], not OpenGL's [-1, 1] --
    `_projection` below uses the D3D-style depth mapping (see its
    docstring).
  - Front-face winding: verified empirically that wgpu's default
    `front_face="ccw"` + `cull_mode="back"` keeps a CCW-in-NDC (x-right,
    y-up) triangle visible, i.e. it lines up with this engine's
    CCW-from-outside mesh winding with no extra flip -- same convention
    GLRenderer relies on.
  - Per-draw uniforms and write-then-submit ordering: `queue.write_buffer()`
    calls all execute (in call order) before a later `queue.submit()`, which
    means a render pass's draw calls all see whatever a *shared* buffer
    holds at submit time, not whatever it held when `set_bind_group` was
    recorded. To give each entity distinct per-draw data (mvp/model/normal
    matrix/face offset) safely, each entity gets its OWN small uniform
    buffer + bind group (cached like the geometry buffers below) instead of
    one shared buffer with dynamic offsets -- simpler, and correct because
    each entity's buffer only ever receives one write per frame.

`wireframe` is a stored, harmless no-op attribute: wgpu's line polygon mode
needs the non-guaranteed `polygon-mode-line` device feature (it happens to
be available on this dev machine's adapter, but this renderer does not
request it, to keep device creation portable across vendors/backends) -- so
there is no wireframe rendering path in this backend. F1 is a documented
no-op on dx12/vulkan in `engine/core.py`.
"""
from __future__ import annotations

import math

import numpy as np

from .gpu_geometry import _build_color, _build_geometry, _entity_world_faces, _scene_environment
from .lighting import _IES_CURVES
from .math3d import rotation_x, rotation_y
from .renderer import _face_light_strength, _gather_lights

_MAX_LIGHTS = 16
_BACKEND_ENV = {"dx12": "D3D12", "vulkan": "Vulkan"}
_BACKEND_MODE = {"D3D12": "dx12", "Vulkan": "vulkan"}

# ---------------------------------------------------------------------------
# WGSL shaders
# ---------------------------------------------------------------------------
# FrameUniforms / LightData field order+size must match `_pack_frame_uniforms`
# / `_pack_lights` below exactly -- every field is a vec4 (or array of vec4)
# specifically to avoid WGSL's vec3-aligns-to-16-bytes trap.
_MESH_WGSL = f"""
const MAX_LIGHTS: i32 = {_MAX_LIGHTS};

struct FrameUniforms {{
    camera_pos: vec4<f32>,        // xyz = camera world position
    dl_dir: vec4<f32>,            // xyz = directional light direction (travel dir)
    dl_color_ambient: vec4<f32>,  // xyz = dlColor * intensity, w = ambient
    flags: vec4<f32>,             // x=useEnv y=envStrength z=fogEnabled w=nLights
    fog_color_start: vec4<f32>,   // xyz = fog color (0..1), w = fog start
    fog_end: vec4<f32>,           // x = fog end
    ambient_cube: array<vec4<f32>, 6>,  // xyz per cube axis (+X -X +Y -Y +Z -Z)
}};

struct LightData {{
    pos: vec4<f32>,       // xyz = world position, w = range
    color: vec4<f32>,     // xyz = color (0..1), w = intensity
    axis_ies: vec4<f32>,  // xyz = axis, w = ies row (-1 if none)
    cone: vec4<f32>,      // x = cosIn (<=-1.5 sentinel for point lights), y = cosOut
}};

struct LightArray {{ items: array<LightData, MAX_LIGHTS> }};

// EntityUniforms field order+size must match `_pack_entity_uniforms` below.
struct EntityUniforms {{
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_mat0: vec4<f32>,  // normal matrix column 0, xyz used
    normal_mat1: vec4<f32>,  // normal matrix column 1, xyz used
    normal_mat2: vec4<f32>,  // normal matrix column 2, xyz used
    face_offset: vec4<f32>,  // x = this entity's face offset into the shadow texture
}};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<uniform> light_buf: LightArray;
@group(0) @binding(2) var shadow_tex: texture_2d<f32>;
@group(0) @binding(3) var ies_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> entity: EntityUniforms;

struct VOut {{
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) @interpolate(flat) face_id: i32,
}};

@vertex
fn vs_main(@location(0) in_pos: vec3<f32>, @location(1) in_normal: vec3<f32>,
           @location(2) in_faceid: f32, @location(3) in_color: vec3<f32>) -> VOut {{
    var out: VOut;
    let world = entity.model * vec4<f32>(in_pos, 1.0);
    out.world_pos = world.xyz;
    let nmat = mat3x3<f32>(entity.normal_mat0.xyz, entity.normal_mat1.xyz, entity.normal_mat2.xyz);
    out.normal = normalize(nmat * in_normal);
    out.color = in_color;
    out.face_id = i32(in_faceid + 0.5);
    out.clip_pos = entity.mvp * vec4<f32>(in_pos, 1.0);
    return out;
}}

// Same cosine-weighted cube lookup as Environment.ambient()/GLRenderer's
// evalAmbientCube(): axes are (+X, -X, +Y, -Y, +Z, -Z), weighted by the
// clamped normal component.
fn eval_ambient_cube(n: vec3<f32>) -> vec3<f32> {{
    let pos = max(n, vec3<f32>(0.0));
    let neg = max(-n, vec3<f32>(0.0));
    let w0 = pos.x; let w1 = neg.x; let w2 = pos.y;
    let w3 = neg.y; let w4 = pos.z; let w5 = neg.z;
    let wsum = max(w0 + w1 + w2 + w3 + w4 + w5, 1e-9);
    let c = frame.ambient_cube[0].xyz * w0 + frame.ambient_cube[1].xyz * w1
          + frame.ambient_cube[2].xyz * w2 + frame.ambient_cube[3].xyz * w3
          + frame.ambient_cube[4].xyz * w4 + frame.ambient_cube[5].xyz * w5;
    return (c / wsum) * frame.flags.y;
}}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    let to_light = normalize(-frame.dl_dir.xyz);
    let lambert = clamp(dot(n, to_light), 0.0, 1.0);
    var lum: vec3<f32>;
    if (frame.flags.x > 0.5) {{
        lum = eval_ambient_cube(n) + frame.dl_color_ambient.xyz * lambert;
    }} else {{
        let amb = frame.dl_color_ambient.w;
        lum = vec3<f32>(amb) + frame.dl_color_ambient.xyz * ((1.0 - amb) * lambert);
    }}

    let n_lights = i32(frame.flags.w + 0.5);
    let face_offset = i32(entity.face_offset.x + 0.5);
    for (var i: i32 = 0; i < n_lights; i = i + 1) {{
        let li = light_buf.items[i];
        let delta = li.pos.xyz - in.world_pos;
        let dist = max(length(delta), 1e-6);
        var atten = clamp(1.0 - dist / li.pos.w, 0.0, 1.0);
        atten = atten * atten;
        let lambert_l = clamp(dot(n, delta) / dist, 0.0, 1.0);
        var strength = li.color.w * atten * lambert_l;

        let is_spot = li.cone.x > -1.5;
        let has_ies = li.axis_ies.w >= 0.0;
        if (is_spot || has_ies) {{
            let to_frag = -delta / dist;
            let cos_ang = dot(to_frag, li.axis_ies.xyz);
            if (is_spot) {{
                let cone = clamp((cos_ang - li.cone.y) / max(li.cone.x - li.cone.y, 1e-6), 0.0, 1.0);
                strength = strength * cone * cone;
            }}
            if (has_ies) {{
                let ang = degrees(acos(clamp(cos_ang, -1.0, 1.0)));
                let row = i32(li.axis_ies.w + 0.5);
                let mul = textureLoad(ies_tex, vec2<i32>(i32(ang), row), 0).r;
                strength = strength * mul;
            }}
        }}

        let shadow = textureLoad(shadow_tex, vec2<i32>(i, face_offset + in.face_id), 0).r;
        strength = strength * shadow;
        lum = lum + li.color.xyz * strength;
    }}

    var out_color = in.color * lum;
    if (frame.flags.z > 0.5) {{
        let dist = length(in.world_pos - frame.camera_pos.xyz);
        let f = clamp((dist - frame.fog_color_start.w) / (frame.fog_end.x - frame.fog_color_start.w),
                      0.0, 1.0);
        out_color = mix(out_color, frame.fog_color_start.xyz, f);
    }}
    return vec4<f32>(clamp(out_color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}}
"""

# SkyUniforms field order+size must match `_pack_sky_uniforms` below.
_SKY_WGSL = """
const PI: f32 = 3.14159265359;

struct SkyUniforms {
    cam_right: vec4<f32>,   // xyz
    cam_up: vec4<f32>,      // xyz
    cam_fwd: vec4<f32>,     // xyz
    view_size: vec4<f32>,   // x=w y=h z=focalK
    flags: vec4<f32>,       // x=useEnv y=useGradient
    sky_top: vec4<f32>,     // xyz (0..1)
    sky_horizon: vec4<f32>, // xyz (0..1)
    bg_color: vec4<f32>,    // xyz (0..1)
};

@group(0) @binding(0) var<uniform> sky: SkyUniforms;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // big-triangle fullscreen trick -- no vertex buffer needed
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    if (sky.flags.x > 0.5) {
        // wgpu's @builtin(position) origin is TOP-left with y increasing
        // DOWNWARD (unlike GL's gl_FragCoord, which is bottom-left/y-up) --
        // flip the y term here so +dcy still means "up" in camera space.
        let dcx = (frag.x - sky.view_size.x * 0.5) / sky.view_size.z;
        let dcy = (sky.view_size.y * 0.5 - frag.y) / sky.view_size.z;
        let dir = normalize(dcx * sky.cam_right.xyz + dcy * sky.cam_up.xyz + sky.cam_fwd.xyz);
        let theta = acos(clamp(dir.y, -1.0, 1.0));
        var phi = atan2(dir.z, dir.x);
        if (phi < 0.0) { phi = phi + 2.0 * PI; }
        let uv = vec2<f32>(phi / (2.0 * PI), theta / PI);
        let radiance = textureSample(env_tex, env_samp, uv).rgb;
        return vec4<f32>(clamp(radiance, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    } else if (sky.flags.y > 0.5) {
        // top of screen (frag.y=0) -> sky_top, bottom (frag.y=view_size.y) -> sky_horizon;
        // no inversion needed here since wgpu's y-down frag coord already
        // runs top-to-bottom in the same order we want top-to-horizon.
        let f = clamp(frag.y / sky.view_size.y, 0.0, 1.0);
        return vec4<f32>(mix(sky.sky_top.xyz, sky.sky_horizon.xyz, f), 1.0);
    } else {
        return vec4<f32>(sky.bg_color.xyz, 1.0);
    }
}
"""


def _projection(camera, w: int, h: int) -> np.ndarray:
    """Right-handed perspective matching `gl_renderer._projection` in x/y,
    but with WebGPU/D3D-style [0, 1] NDC depth instead of OpenGL's [-1, 1].

    Derivation (camera looks down -Z, so camera-space z is negative in
    front of the eye): with m[3,2]=-1 the clip-space w is -z_cam (positive
    in front of the eye). Solving z_clip/w_clip = 0 at z_cam=-near and = 1
    at z_cam=-far for m[2,2] and m[2,3] gives the two lines below -- swap
    them back to `(far+near)/(near-far)` and `2*far*near/(near-far)` to
    recover GLRenderer's OpenGL [-1,1] mapping.
    """
    aspect = w / max(h, 1)
    f = 1.0 / math.tan(math.radians(camera.fov) * 0.5)
    near, far = camera.near, camera.far
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = far / (near - far)
    m[2, 3] = near * far / (near - far)
    m[3, 2] = -1.0
    return m


class WgpuRenderer:
    MAX_LIGHTS = _MAX_LIGHTS
    _COLOR_FORMAT = "rgba8unorm"
    _DEPTH_FORMAT = "depth24plus"
    _FRAME_UBO_SIZE = 192   # FrameUniforms: 6 header vec4 + 6 ambient_cube vec4
    _LIGHT_UBO_SIZE = 64 * _MAX_LIGHTS  # LightData = 4 vec4 = 64B, x16 lights
    _ENTITY_UBO_SIZE = 192  # 2x mat4x4 (64B) + 3x vec4 (48B) + 1x vec4 (16B)
    _SKY_UBO_SIZE = 128     # SkyUniforms: 8 vec4

    def __init__(self, backend: str):
        if backend not in _BACKEND_ENV:
            raise ValueError(f"unknown wgpu backend {backend!r}; expected 'dx12' or 'vulkan'")
        import os
        os.environ["WGPU_BACKEND_TYPE"] = _BACKEND_ENV[backend]
        import wgpu  # lazy: keeps the rest of the engine importable without wgpu installed
        self._wgpu = wgpu

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.backend_type = adapter.info["backend_type"]
        self.device = adapter.request_device_sync()
        self._wireframe = False  # stored but inert -- see module docstring
        mode = _BACKEND_MODE.get(self.backend_type, self.backend_type.lower())
        self.stats = {"mode": mode, "triangles": 0, "shadow_lights": 0, "wireframe": False}

        self._size = None
        self._color_tex = self._depth_tex = None
        self._color_view = self._depth_view = None
        self._geo_cache: dict[int, dict] = {}
        self._entity_uniform_cache: dict[int, dict] = {}
        self._env_tex_cache: dict[int, dict] = {}
        self._last_sky_env_view = None

        self._build_pipelines()
        self._build_static_resources()

    @property
    def wireframe(self) -> bool:
        """Stored but inert: wgpu's line-polygon-mode fill needs the
        non-guaranteed `polygon-mode-line` device feature, which this
        renderer does not request (to keep device creation portable), so
        there is no wireframe rendering path here. Setting this attribute
        is remembered (and reflected in `stats`) but has no visual effect."""
        return self._wireframe

    @wireframe.setter
    def wireframe(self, value: bool) -> None:
        self._wireframe = bool(value)
        self.stats["wireframe"] = self._wireframe

    # ------------------------------------------------------------------
    # one-time setup
    # ------------------------------------------------------------------
    def _build_pipelines(self) -> None:
        device = self.device
        mesh_shader = device.create_shader_module(code=_MESH_WGSL)
        self._mesh_pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": mesh_shader, "entry_point": "vs_main",
                "buffers": [
                    {  # pos(3f) normal(3f) faceid(1f) -- matches _build_geometry's concat
                        "array_stride": 4 * 7, "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32", "offset": 24, "shader_location": 2},
                        ],
                    },
                    {  # color(3f) -- matches _build_color's output
                        "array_stride": 4 * 3, "step_mode": "vertex",
                        "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 3}],
                    },
                ],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "back"},
            depth_stencil={"format": self._DEPTH_FORMAT, "depth_write_enabled": True,
                           "depth_compare": "less"},
            fragment={"module": mesh_shader, "entry_point": "fs_main",
                      "targets": [{"format": self._COLOR_FORMAT}]},
        )
        self._mesh_bgl0 = self._mesh_pipeline.get_bind_group_layout(0)
        self._mesh_bgl1 = self._mesh_pipeline.get_bind_group_layout(1)

        sky_shader = device.create_shader_module(code=_SKY_WGSL)
        self._sky_pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={"module": sky_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": self._DEPTH_FORMAT, "depth_write_enabled": False,
                           "depth_compare": "always"},
            fragment={"module": sky_shader, "entry_point": "fs_main",
                      "targets": [{"format": self._COLOR_FORMAT}]},
        )
        self._sky_bgl0 = self._sky_pipeline.get_bind_group_layout(0)

    def _build_static_resources(self) -> None:
        wgpu = self._wgpu
        device = self.device

        names = list(_IES_CURVES.keys())
        curves = np.stack([_IES_CURVES[n] for n in names]).astype(np.float32)  # (n_profiles, 181)
        self._ies_row_of_id = {id(_IES_CURVES[n]): i for i, n in enumerate(names)}
        w_ies, h_ies = curves.shape[1], curves.shape[0]
        self._ies_tex = device.create_texture(
            size=(w_ies, h_ies, 1), format="r32float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        device.queue.write_texture(
            {"texture": self._ies_tex}, np.ascontiguousarray(curves, dtype=np.float32).tobytes(),
            {"bytes_per_row": w_ies * 4, "rows_per_image": h_ies}, (w_ies, h_ies, 1))
        self._ies_view = self._ies_tex.create_view()

        self._frame_ubo = device.create_buffer(
            size=self._FRAME_UBO_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._light_ubo = device.create_buffer(
            size=self._LIGHT_UBO_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._sky_ubo = device.create_buffer(
            size=self._SKY_UBO_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # 1x1 transparent-black dummy env texture, always bound so the sky
        # bind group layout is stable whether or not a real HDRI is loaded
        # (WebGPU bind groups can't have "optional" bindings).
        self._dummy_env_tex = device.create_texture(
            size=(1, 1, 1), format="rgba16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        device.queue.write_texture(
            {"texture": self._dummy_env_tex}, np.zeros(4, dtype=np.float16).tobytes(),
            {"bytes_per_row": 8, "rows_per_image": 1}, (1, 1, 1))
        self._dummy_env_view = self._dummy_env_tex.create_view()
        # repeat horizontally (around the equirect seam), clamp vertically
        # (poles) -- matches GLRenderer's env texture wrap settings.
        self._env_sampler = device.create_sampler(
            address_mode_u="repeat", address_mode_v="clamp-to-edge",
            mag_filter="linear", min_filter="linear")

        self._shadow_tex_size = (self.MAX_LIGHTS, 1)
        self._shadow_tex = device.create_texture(
            size=(self.MAX_LIGHTS, 1, 1), format="r32float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self._shadow_view = self._shadow_tex.create_view()

        self._frame_bind_group = self._make_frame_bind_group()
        self._sky_bind_group = self._make_sky_bind_group(self._dummy_env_view)
        self._last_sky_env_view = self._dummy_env_view

    def _make_frame_bind_group(self):
        return self.device.create_bind_group(layout=self._mesh_bgl0, entries=[
            {"binding": 0, "resource": {"buffer": self._frame_ubo, "offset": 0,
                                        "size": self._FRAME_UBO_SIZE}},
            {"binding": 1, "resource": {"buffer": self._light_ubo, "offset": 0,
                                        "size": self._LIGHT_UBO_SIZE}},
            {"binding": 2, "resource": self._shadow_view},
            {"binding": 3, "resource": self._ies_view},
        ])

    def _make_sky_bind_group(self, env_view):
        return self.device.create_bind_group(layout=self._sky_bgl0, entries=[
            {"binding": 0, "resource": {"buffer": self._sky_ubo, "offset": 0,
                                        "size": self._SKY_UBO_SIZE}},
            {"binding": 1, "resource": env_view},
            {"binding": 2, "resource": self._env_sampler},
        ])

    # ------------------------------------------------------------------
    # size-dependent resources
    # ------------------------------------------------------------------
    def _ensure_size(self, w: int, h: int) -> None:
        if self._size == (w, h):
            return
        wgpu = self._wgpu
        if self._color_tex is not None:
            self._color_tex.destroy()
            self._depth_tex.destroy()
        self._color_tex = self.device.create_texture(
            size=(w, h, 1), format=self._COLOR_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self._depth_tex = self.device.create_texture(
            size=(w, h, 1), format=self._DEPTH_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        self._color_view = self._color_tex.create_view()
        self._depth_view = self._depth_tex.create_view()
        self._size = (w, h)

    def _upload_shadow_tex(self, shadow_data: np.ndarray, total_faces: int) -> None:
        wgpu = self._wgpu
        size = (self.MAX_LIGHTS, max(total_faces, 1))
        if self._shadow_tex_size != size:
            self._shadow_tex.destroy()
            self._shadow_tex = self.device.create_texture(
                size=(size[0], size[1], 1), format="r32float",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self._shadow_view = self._shadow_tex.create_view()
            self._shadow_tex_size = size
            self._frame_bind_group = self._make_frame_bind_group()  # view changed
        w_s, h_s = size
        # shadow_data is (total_faces, MAX_LIGHTS) row-major, i.e. already
        # (height, width) order matching a (MAX_LIGHTS, total_faces) texture
        # -- no transpose needed (same layout GLRenderer writes verbatim).
        data = np.ascontiguousarray(shadow_data, dtype=np.float32)
        self.device.queue.write_texture(
            {"texture": self._shadow_tex}, data.tobytes(),
            {"bytes_per_row": w_s * 4, "rows_per_image": h_s}, (w_s, h_s, 1))

    def _get_env_view(self, env):
        wgpu = self._wgpu
        key = id(env.image)
        cache = self._env_tex_cache.get(key)
        if cache is None:
            for old in self._env_tex_cache.values():
                old["tex"].destroy()
            self._env_tex_cache.clear()
            h, w = env.image.shape[:2]
            tex = self.device.create_texture(
                size=(w, h, 1), format="rgba16float",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            rgba[..., :3] = env.image
            half = np.ascontiguousarray(rgba, dtype=np.float32).astype(np.float16)
            self.device.queue.write_texture(
                {"texture": tex}, np.ascontiguousarray(half).tobytes(),
                {"bytes_per_row": w * 8, "rows_per_image": h}, (w, h, 1))
            cache = {"tex": tex, "view": tex.create_view()}
            self._env_tex_cache[key] = cache
        return cache["view"]

    # ------------------------------------------------------------------
    # per-mesh / per-entity GPU buffer caches (mirrors GLRenderer)
    # ------------------------------------------------------------------
    def _get_geo_cache(self, mesh) -> dict:
        wgpu = self._wgpu
        key = id(mesh)
        cache = self._geo_cache.get(key)
        if cache is None:
            pos, nrm, fid, face_id_tri, m = _build_geometry(mesh)
            geom = np.concatenate([pos, nrm, fid[:, None]], axis=1).astype(np.float32)
            geom_buf = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(geom).tobytes(), usage=wgpu.BufferUsage.VERTEX)
            color = _build_color(mesh, face_id_tri)
            color_buf = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(color).tobytes(), usage=wgpu.BufferUsage.VERTEX)
            cache = {"geom_buf": geom_buf, "color_buf": color_buf, "count": pos.shape[0],
                     "face_id_tri": face_id_tri, "color_id": id(mesh.face_colors)}
            self._geo_cache[key] = cache
        elif cache["color_id"] != id(mesh.face_colors):
            # buffers from create_buffer_with_data aren't COPY_DST, so a
            # recolor (material editor) rebuilds the small color buffer
            # rather than writing into it -- this is a rare, not per-frame,
            # path (unlike GL's mutable-in-place vbo.write()).
            cache["color_buf"].destroy()
            color = _build_color(mesh, cache["face_id_tri"])
            cache["color_buf"] = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(color).tobytes(), usage=wgpu.BufferUsage.VERTEX)
            cache["color_id"] = id(mesh.face_colors)
        return cache

    def _prune_geo_cache(self, live_entities) -> None:
        live_mesh_ids = {id(e.mesh) for e in live_entities}
        for key in [k for k in self._geo_cache if k not in live_mesh_ids]:
            c = self._geo_cache.pop(key)
            c["geom_buf"].destroy()
            c["color_buf"].destroy()
        live_ent_ids = {id(e) for e in live_entities}
        for key in [k for k in self._entity_uniform_cache if k not in live_ent_ids]:
            self._entity_uniform_cache.pop(key)["ubo"].destroy()

    def _get_entity_uniforms(self, entity) -> dict:
        wgpu = self._wgpu
        key = id(entity)
        cache = self._entity_uniform_cache.get(key)
        if cache is None:
            ubo = self.device.create_buffer(
                size=self._ENTITY_UBO_SIZE,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            bg = self.device.create_bind_group(layout=self._mesh_bgl1, entries=[
                {"binding": 0, "resource": {"buffer": ubo, "offset": 0,
                                            "size": self._ENTITY_UBO_SIZE}}])
            cache = {"ubo": ubo, "bind_group": bg}
            self._entity_uniform_cache[key] = cache
        return cache

    # ------------------------------------------------------------------
    # per-frame uniform packing -- each helper's comment mirrors the WGSL
    # struct it feeds (see _MESH_WGSL / _SKY_WGSL above)
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_entity_uniforms(mvp: np.ndarray, model: np.ndarray, normal_mat: np.ndarray,
                              face_offset: int) -> bytes:
        """EntityUniforms: mvp(64B) + model(64B) + 3x normal_mat column
        vec4(48B) + face_offset vec4(16B) = 192B. Matrices are written
        `.T`-flattened (row i of the transpose = column i of the original)
        to match WGSL's column-major mat4x4/vec4-column storage, same trick
        as GLRenderer's `_write_mat`."""
        buf = np.zeros(48, dtype=np.float32)
        buf[0:16] = np.ascontiguousarray(mvp.T, dtype=np.float32).ravel()
        buf[16:32] = np.ascontiguousarray(model.T, dtype=np.float32).ravel()
        nmat_t = np.ascontiguousarray(normal_mat.T, dtype=np.float32)  # nmat_t[c] = column c
        buf[32:35] = nmat_t[0]
        buf[36:39] = nmat_t[1]
        buf[40:43] = nmat_t[2]
        buf[44] = float(face_offset)
        return buf.tobytes()

    def _pack_frame_uniforms(self, scene, camera, env, lights) -> bytes:
        """FrameUniforms: 6 header vec4 (96B) + 6 ambient_cube vec4 (96B) = 192B."""
        dl = scene.light
        fog = scene.fog
        buf = np.zeros(48, dtype=np.float32)
        buf[0:3] = camera.position.to_array().astype(np.float32)
        buf[4:7] = dl.direction.to_array().astype(np.float32)
        dl_color = (np.asarray(dl.color, dtype=np.float32) / 255.0) * dl.intensity
        buf[8:11] = dl_color
        buf[11] = float(dl.ambient)
        buf[12] = 1.0 if env is not None else 0.0
        buf[13] = float(env.strength) if env is not None else 0.0
        buf[14] = 1.0 if fog is not None else 0.0
        buf[15] = float(len(lights))
        if fog is not None:
            buf[16:19] = np.asarray(fog.color, dtype=np.float32) / 255.0
            buf[19] = float(fog.start)
            buf[20] = float(fog.end)
        if env is not None:
            cube = np.zeros((6, 4), dtype=np.float32)
            cube[:, :3] = env.ambient_cube
            buf[24:48] = cube.ravel()
        return buf.tobytes()

    def _pack_lights(self, lights) -> bytes:
        """array<LightData, MAX_LIGHTS>: 4 vec4 (64B) per light."""
        buf = np.zeros((self.MAX_LIGHTS, 16), dtype=np.float32)
        for i, info in enumerate(lights[:self.MAX_LIGHTS]):
            buf[i, 0:3] = info.pos
            buf[i, 3] = max(info.light.range, 1e-6)
            buf[i, 4:7] = info.colorf
            buf[i, 7] = info.light.intensity
            buf[i, 8:11] = info.axis
            buf[i, 11] = float(self._ies_row_of_id.get(id(info.curve), -1)
                               if info.curve is not None else -1)
            buf[i, 12] = info.cos_in if info.cos_in is not None else -2.0
            buf[i, 13] = info.cos_out if info.cos_out is not None else 0.0
        return buf.tobytes()

    def _pack_sky_uniforms(self, scene, camera, env, w: int, h: int) -> bytes:
        """SkyUniforms: 8 vec4 = 128B."""
        k = 0.5 * h / math.tan(math.radians(camera.fov) * 0.5)
        rot = (rotation_y(camera.yaw) @ rotation_x(camera.pitch))[:3, :3]
        right = rot @ np.array([1.0, 0.0, 0.0])
        up = rot @ np.array([0.0, 1.0, 0.0])
        fwd = rot @ np.array([0.0, 0.0, -1.0])
        buf = np.zeros(32, dtype=np.float32)
        buf[0:3] = right
        buf[4:7] = up
        buf[8:11] = fwd
        buf[12], buf[13], buf[14] = float(w), float(h), float(k)
        buf[16] = 1.0 if env is not None else 0.0
        buf[17] = 1.0 if (env is None and scene.sky is not None) else 0.0
        if env is None:
            if scene.sky is not None:
                buf[20:23] = np.asarray(scene.sky[0], dtype=np.float32) / 255.0
                buf[24:27] = np.asarray(scene.sky[1], dtype=np.float32) / 255.0
            else:
                buf[28:31] = np.asarray(scene.background, dtype=np.float32) / 255.0
        return buf.tobytes()

    # ------------------------------------------------------------------
    def render(self, scene, camera, size, tracer=None) -> None:
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0:
            return
        self._ensure_size(w, h)

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

        # Frame-global uniforms: one write per buffer this frame, so all
        # draws in the render pass recorded below see consistent data once
        # the command buffer actually executes after submit() (see module
        # docstring's write-then-submit-ordering note).
        self.device.queue.write_buffer(self._frame_ubo, 0,
                                       self._pack_frame_uniforms(scene, camera, env, lights))
        self.device.queue.write_buffer(self._light_ubo, 0, self._pack_lights(lights))
        self.device.queue.write_buffer(self._sky_ubo, 0,
                                       self._pack_sky_uniforms(scene, camera, env, w, h))

        env_view = self._get_env_view(env) if env is not None else self._dummy_env_view
        if env_view is not self._last_sky_env_view:
            self._sky_bind_group = self._make_sky_bind_group(env_view)
            self._last_sky_env_view = env_view

        view = camera.view_matrix()
        proj = _projection(camera, w, h)

        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(
            color_attachments=[{
                "view": self._color_view, "clear_value": (0.0, 0.0, 0.0, 1.0),
                "load_op": "clear", "store_op": "store",
            }],
            depth_stencil_attachment={
                "view": self._depth_view, "depth_clear_value": 1.0,
                "depth_load_op": "clear", "depth_store_op": "store",
            },
        )
        # sky first, with depth write disabled at the pipeline level, so the
        # mesh pass below still depth-tests against a clean 1.0 (far) buffer
        pass_enc.set_pipeline(self._sky_pipeline)
        pass_enc.set_bind_group(0, self._sky_bind_group)
        pass_enc.draw(3)

        pass_enc.set_pipeline(self._mesh_pipeline)
        pass_enc.set_bind_group(0, self._frame_bind_group)
        triangles = 0
        for entity, off in zip(live, offsets):
            if entity.mesh.faces.shape[0] == 0:
                continue
            geo = self._get_geo_cache(entity.mesh)
            ent = self._get_entity_uniforms(entity)
            model = entity.transform.matrix()
            mvp = proj @ view @ model
            try:
                nmat = np.linalg.inv(model[:3, :3]).T
            except np.linalg.LinAlgError:
                nmat = np.eye(3)
            self.device.queue.write_buffer(
                ent["ubo"], 0, self._pack_entity_uniforms(mvp, model, nmat, off))
            pass_enc.set_bind_group(1, ent["bind_group"])
            pass_enc.set_vertex_buffer(0, geo["geom_buf"])
            pass_enc.set_vertex_buffer(1, geo["color_buf"])
            pass_enc.draw(geo["count"])
            triangles += geo["count"] // 3
        pass_enc.end()
        self.device.queue.submit([encoder.finish()])

        self._prune_geo_cache(live)
        self.stats["triangles"] = triangles

    def read_frame(self) -> bytes:
        """RGBA8 bytes of the last rendered frame, exactly W*H*4 -- wgpu-py's
        `read_texture` handles the 256-byte copy-row alignment internally
        (verified empirically, see module docstring). Caller (engine/core.py)
        wraps this straight in `pygame.image.frombuffer(..., "RGBA")`; no
        vertical flip needed (also verified empirically)."""
        w, h = self._size
        return self.device.queue.read_texture(
            {"texture": self._color_tex, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))

    # ------------------------------------------------------------------
    def release(self) -> None:
        """Free wgpu-side resources. Not currently called by Engine (which
        also never calls GLRenderer.release() -- the context/device is left
        to process teardown), provided for unit tests / symmetry."""
        self._prune_geo_cache([])
        for cache in self._env_tex_cache.values():
            cache["tex"].destroy()
        self._env_tex_cache.clear()
        if self._color_tex is not None:
            self._color_tex.destroy()
            self._depth_tex.destroy()
        self._shadow_tex.destroy()
        self._ies_tex.destroy()
        self._dummy_env_tex.destroy()
