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
flat-per-face-normal geometry soup (`_build_geometry`/`_build_color`/
`_build_pbr`, imported from the shared, GPU-library-free `gpu_geometry.py`
so all three renderers -- software, GL, wgpu -- build triangle soup,
per-face PBR params, and world-space face data from one source of truth),
metallic-roughness Cook-Torrance/GGX specular identical to the CPU/GL model
(see `ggx_specular()` in `_MESH_WGSL`), the same light-gathering
(`_gather_lights`/`_face_light_strength` from `renderer.py`), and the same
(total_faces, MAX_LIGHTS) float32 shadow-factor layout produced by
`raytrace.ShadowTracer` -- uploaded here as an r32float texture and sampled
with `textureLoad` (no sampler -- an exact texel fetch, like GL's
`texelFetch`). Directional (sun) shadow attenuation mirrors GL's
`dlShadowTex` exactly: a reserved single-column (1, total_faces) r32float
texture (`dl_shadow_tex`, modeled on `gi_tex`'s lifecycle) holds
`1 - shadow_depth*(1 - raw)` from `raytrace.ShadowTracer.
directional_shadow_factors`, gathered onto the render mesh via
`_lod_gather` exactly like GI/point-light shadows; the fragment shader
multiplies it into `lambert` at its point of computation (before both the
diffuse and specular uses), same as GL's `lambert *= dlShadow`.

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

Translucent materials (`MaterialGraph.blend_mode == "translucent"`) draw in
a second render-pass pipeline (`_mesh_translucent_pipeline`) sharing the
opaque pipeline's shader + explicit pipeline layout (bind groups from one
work with the other -- see `_build_pipelines`'s comment on why two
`layout="auto"` pipelines from the same shader are NOT bind-group-
compatible): depth test on / depth write off, standard alpha blend, drawn
after all opaque geometry sorted back-to-front by per-entity distance to
camera. Per-face opacity is a new one-float vertex attribute
(`_build_opacity`), same vertex-soup pattern as color/PBR.

`wireframe` is a stored, harmless no-op attribute: wgpu's line polygon mode
needs the non-guaranteed `polygon-mode-line` device feature (it happens to
be available on this dev machine's adapter, but this renderer does not
request it, to keep device creation portable across vendors/backends) -- so
there is no wireframe rendering path in this backend. F1 is a documented
no-op on dx12/vulkan in `engine/core.py`.

Per-pixel texturing (run 4/4 of the slate, mirroring `gl_renderer.py`'s run
3): `gpu_geometry._build_uv(mesh)` supplies a per-vertex UV vertex buffer
(`shader_location=7`), and a material's `direct_texture_bindings()` (see
materials.py) is sampled per pixel in WGSL for whichever Output channels a
`tex_sample` node feeds directly -- everything else keeps reading the
per-face-baked color/roughness/metallic/emissive attributes untouched, so
an untextured draw is a byte-identical passthrough. Texture sampling here
uses `textureLoad` (see `tex_texel()` in `_MESH_WGSL`), NOT `textureSample`
+ a `sampler`: a `sampler`-based path would need the texture's sample type
to be "float" (filterable), but an rgba32float texture is "unfilterable-
float" without requesting the non-portable `float32-filterable` device
feature -- `textureLoad` has no such restriction and needs no sampler
object at all, exactly like this file's existing `shadow_tex`/`gi_tex`/
`ies_tex` exact-texel reads. Because sampling is hand-rolled, the V-FLIP
question (`texture.sample_texture`'s `yi = clip((1-v)*h, ...)`, which maps
v=0 to the image's BOTTOM row) is answered by literally reproducing that
formula in `tex_texel()` against an UNFLIPPED upload (row 0 of the CPU
array = texel row 0), rather than by flipping the array to match some
hardware sampler convention the way `GLRenderer._get_material_tex` does --
there is no implicit v-convention to fight when every index is computed by
hand. Verified by the wgpu-vs-CPU pixel parity gate in `tests/wgpu_checks.
py` (a flipped upload would mirror every textured pixel and blow the parity
gate wide open). Per-entity texture views live in bind group 2 (4 fixed
slots: base_color/roughness/metallic/emissive, always dummy-filled --
`_dummy_tex_view` -- when a channel has no binding, since WebGPU bind
groups can't have "optional" entries), cached like the entity uniform bind
group and rebuilt only when the resolved view identities change; per-
channel has/port/tiling metadata rides in `EntityUniforms.tex_meta` (see
`_pack_entity_uniforms`), grown by one `array<vec4<f32>, 4>` (verify
`_ENTITY_UBO_SIZE` if this struct changes again). Translucent entities
always get `{}` bindings here too (never even resolved), same rule as GL/
CPU -- opacity is not a sampleable channel on this shader.
"""
from __future__ import annotations

import math

import numpy as np

from . import texture as texture_mod
from .gpu_geometry import (_build_color, _build_geometry, _build_opacity,
                           _build_pbr, _build_uv, _entity_world_faces, _lod_gather,
                           _scene_environment)
from .lighting import _IES_CURVES
from .math3d import rotation_x, rotation_y
from .raytrace import GITracer
from .renderer import (_face_light_strength, _fog_volumes, _gather_lights,
                       _gi_direct_lighting, _gi_receiver_geometry, _is_translucent,
                       _scene_sun, _sun_sky_info)

_MAX_LIGHTS = 16
_MAX_FOG_VOL = 4
_FOG_SKY_FAR = 260.0  # path-length clip for fog volumes behind the sky, matches GLRenderer
_BACKEND_ENV = {"dx12": "D3D12", "vulkan": "Vulkan"}
_BACKEND_MODE = {"D3D12": "dx12", "Vulkan": "vulkan"}

# Port name -> int code, mirroring materials.py's TextureBinding.port /
# GLRenderer's `_PORT_CODE` / renderer.py's `_extract_channel` -- read by
# both sample_port_vec3/sample_port_scalar in the fragment shader below.
_PORT_CODE = {"RGB": 0, "R": 1, "G": 2, "B": 3, "A": 4}

# ---------------------------------------------------------------------------
# WGSL shaders
# ---------------------------------------------------------------------------
# FrameUniforms / LightData field order+size must match `_pack_frame_uniforms`
# / `_pack_lights` below exactly -- every field is a vec4 (or array of vec4)
# specifically to avoid WGSL's vec3-aligns-to-16-bytes trap.
_MESH_WGSL = f"""
const PI: f32 = 3.14159265359;
const MAX_LIGHTS: i32 = {_MAX_LIGHTS};
const MAX_FOG_VOL: i32 = {_MAX_FOG_VOL};

struct FrameUniforms {{
    camera_pos: vec4<f32>,        // xyz = camera world position
    dl_dir: vec4<f32>,            // xyz = directional light direction (travel dir)
    dl_color_ambient: vec4<f32>,  // xyz = dlColor * intensity, w = ambient
    flags: vec4<f32>,             // x=useEnv y=envStrength z=fogEnabled w=nLights
    fog_color_start: vec4<f32>,   // xyz = fog color (0..1), w = fog start
    fog_end: vec4<f32>,           // x = fog end, y = fog volume count
    ambient_cube: array<vec4<f32>, 6>,  // xyz per cube axis (+X -X +Y -Y +Z -Z)
    fog_vol_lo: array<vec4<f32>, MAX_FOG_VOL>,     // xyz = box lo, w = density
    fog_vol_hi: array<vec4<f32>, MAX_FOG_VOL>,     // xyz = box hi, w = height falloff
    fog_vol_color: array<vec4<f32>, MAX_FOG_VOL>,  // xyz = color (0..1)
}};

struct LightData {{
    pos: vec4<f32>,       // xyz = world position, w = range
    color: vec4<f32>,     // xyz = color (0..1), w = intensity
    axis_ies: vec4<f32>,  // xyz = axis, w = ies row (-1 if none)
    cone: vec4<f32>,      // x = cosIn (<=-1.5 sentinel for point lights), y = cosOut
}};

struct LightArray {{ items: array<LightData, MAX_LIGHTS> }};

// EntityUniforms field order+size must match `_pack_entity_uniforms` below.
// tex_meta[i] = (has, port_code, u_tiling, v_tiling) for channel i, fixed
// order (base_color, roughness, metallic, emissive) matching `_TEX_CHANNELS`
// and bind group 2's 4 texture slots below -- grown onto the end of the
// struct (run 4/4 of the per-pixel texturing slate; see module docstring).
struct EntityUniforms {{
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_mat0: vec4<f32>,  // normal matrix column 0, xyz used
    normal_mat1: vec4<f32>,  // normal matrix column 1, xyz used
    normal_mat2: vec4<f32>,  // normal matrix column 2, xyz used
    face_offset: vec4<f32>,  // x = this entity's face offset into the shadow texture
    tex_meta: array<vec4<f32>, 4>,
}};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<uniform> light_buf: LightArray;
@group(0) @binding(2) var shadow_tex: texture_2d<f32>;
@group(0) @binding(3) var ies_tex: texture_2d<f32>;
@group(0) @binding(4) var gi_tex: texture_2d<f32>;
@group(0) @binding(5) var dl_shadow_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> entity: EntityUniforms;
// Per-pixel texturing: 4 fixed texture slots (base_color/roughness/
// metallic/emissive -- "opacity" is excluded, see module docstring), always
// bound to SOMETHING (a 1x1 dummy when a channel has no direct binding --
// WebGPU bind groups can't have optional entries), read via `textureLoad`
// (no sampler -- see module docstring on why `textureSample` doesn't fit
// an unfilterable-float rgba32float texture here).
@group(2) @binding(0) var uBaseTex: texture_2d<f32>;
@group(2) @binding(1) var uRoughTex: texture_2d<f32>;
@group(2) @binding(2) var uMetalTex: texture_2d<f32>;
@group(2) @binding(3) var uEmisTex: texture_2d<f32>;

struct VOut {{
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) @interpolate(flat) face_id: i32,
    @location(4) roughness: f32,
    @location(5) metallic: f32,
    @location(6) emissive: vec3<f32>,
    @location(7) opacity: f32,
    @location(8) uv: vec2<f32>,
}};

@vertex
fn vs_main(@location(0) in_pos: vec3<f32>, @location(1) in_normal: vec3<f32>,
           @location(2) in_faceid: f32, @location(3) in_color: vec3<f32>,
           @location(4) in_rm: vec2<f32>, @location(5) in_emissive: vec3<f32>,
           @location(6) in_opacity: f32, @location(7) in_uv: vec2<f32>) -> VOut {{
    var out: VOut;
    let world = entity.model * vec4<f32>(in_pos, 1.0);
    out.world_pos = world.xyz;
    let nmat = mat3x3<f32>(entity.normal_mat0.xyz, entity.normal_mat1.xyz, entity.normal_mat2.xyz);
    out.normal = normalize(nmat * in_normal);
    out.color = in_color;
    out.face_id = i32(in_faceid + 0.5);
    out.roughness = in_rm.x;
    out.metallic = in_rm.y;
    out.emissive = in_emissive;
    out.opacity = in_opacity;
    out.uv = in_uv;
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

// Ray-AABB slab intersection clipped to [tNear, tFar] -- same math as
// GLRenderer's fogVolSegment(); returns (t0, t1), t1 < t0 means no overlap.
fn fog_vol_segment(origin: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>,
                   t_near: f32, t_far: f32) -> vec2<f32> {{
    let inv_dir = vec3<f32>(1.0) / dir;
    let t1v = (lo - origin) * inv_dir;
    let t2v = (hi - origin) * inv_dir;
    let tmin = min(t1v, t2v);
    let tmax = max(t1v, t2v);
    let t0 = max(max(max(tmin.x, tmin.y), tmin.z), t_near);
    let t1 = min(min(min(tmax.x, tmax.y), tmax.z), t_far);
    return vec2<f32>(t0, t1);
}}

fn apply_fog_volumes(color_in: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>,
                     t_near: f32, t_far: f32) -> vec3<f32> {{
    var color = color_in;
    let count = i32(frame.fog_end.y + 0.5);
    for (var i: i32 = 0; i < count; i = i + 1) {{
        let lo = frame.fog_vol_lo[i];
        let hi = frame.fog_vol_hi[i];
        let seg = fog_vol_segment(origin, dir, lo.xyz, hi.xyz, t_near, t_far);
        let seg_len = max(seg.y - seg.x, 0.0);
        var density = lo.w;
        if (hi.w > 0.0) {{
            let mid_h = origin.y + dir.y * (0.5 * (seg.x + seg.y));
            density = density * exp(-max(mid_h, 0.0) * hi.w);
        }}
        let t_ext = exp(-density * seg_len);
        color = color * t_ext + frame.fog_vol_color[i].xyz * (1.0 - t_ext);
    }}
    return color;
}}

// Cook-Torrance/GGX specular BRDF, mirroring renderer.py's `_ggx_specular` /
// GLRenderer's `ggxSpecular` exactly: Smith-GGX geometry with the
// direct-lighting k=(a+1)^2/8 remap (Karis/UE4), Schlick Fresnel. Caller
// multiplies the result by NdotL*radiance.
fn ggx_specular(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, alpha: f32, f0: vec3<f32>) -> vec3<f32> {{
    let h = normalize(v + l);
    let ndoth = clamp(dot(n, h), 0.0, 1.0);
    let ndotv = clamp(dot(n, v), 1e-4, 1.0);
    let ndotl = clamp(dot(n, l), 1e-4, 1.0);
    let vdoth = clamp(dot(v, h), 0.0, 1.0);

    let a2 = alpha * alpha;
    let denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    let d = a2 / max(PI * denom * denom, 1e-8);

    let k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
    let g1v = ndotv / max(ndotv * (1.0 - k) + k, 1e-8);
    let g1l = ndotl / max(ndotl * (1.0 - k) + k, 1e-8);
    let g = g1v * g1l;

    let f = f0 + (1.0 - f0) * pow(1.0 - vdoth, 5.0);
    return (d * g / max(4.0 * ndotv * ndotl, 1e-4)) * f;
}}

// Nearest-neighbor exact-texel fetch with REPEAT wrap, replicating
// texture.sample_texture's formula exactly (see module docstring):
// `u = mod(uv.x, 1)`, `v = mod(uv.y, 1)`, `xi = clip(u*w, 0, w-1)`,
// `yi = clip((1-v)*h, 0, h-1)` -- the (1-v) is the V-FLIP: v=0 lands on the
// array's LAST row (image bottom), v=1 on row 0 (image top), matching an
// UNFLIPPED upload of `texture.load_texture`'s (row 0 = image top) array.
fn tex_texel(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {{
    let dims = vec2<i32>(textureDimensions(tex));
    let u = uv.x - floor(uv.x);
    let v = uv.y - floor(uv.y);
    let xi = clamp(i32(u * f32(dims.x)), 0, dims.x - 1);
    let yi = clamp(i32((1.0 - v) * f32(dims.y)), 0, dims.y - 1);
    return textureLoad(tex, vec2<i32>(xi, yi), 0);
}}

// Port -> shape coercion, mirroring GLRenderer's sampleTexVec3/Scalar /
// renderer.py's `_extract_channel` exactly (0=RGB, 1=R, 2=G, 3=B, 4=A --
// see materials.py's TextureBinding.port). Vec3 form is for vector
// channels (base_color/emissive): a scalar port replicates across R=G=B.
// Scalar form is for roughness/metallic: a vector port (RGB) mean-reduces.
fn sample_port_vec3(s: vec4<f32>, port: i32) -> vec3<f32> {{
    if (port == 0) {{ return s.rgb; }}
    if (port == 1) {{ return vec3<f32>(s.r); }}
    if (port == 2) {{ return vec3<f32>(s.g); }}
    if (port == 3) {{ return vec3<f32>(s.b); }}
    return vec3<f32>(s.a);
}}

fn sample_port_scalar(s: vec4<f32>, port: i32) -> f32 {{
    if (port == 0) {{ return (s.r + s.g + s.b) / 3.0; }}
    if (port == 1) {{ return s.r; }}
    if (port == 2) {{ return s.g; }}
    if (port == 3) {{ return s.b; }}
    return s.a;
}}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    // Per-pixel texturing overrides -- replace the per-face-baked value for
    // just the channels a material directly binds a texture to (REPLACE,
    // not blend/multiply -- mirrors renderer.py's `channel_overrides` and
    // GLRenderer's uHasXTex block exactly). Untextured draws leave every
    // tex_meta[i].x at 0, so this is a pure passthrough -- same values as
    // before this run.
    var base_color = in.color;
    var roughness_v = in.roughness;
    var metallic_v = in.metallic;
    var emissive_v = in.emissive;
    if (entity.tex_meta[0].x > 0.5) {{
        let s = tex_texel(uBaseTex, in.uv * entity.tex_meta[0].zw);
        base_color = sample_port_vec3(s, i32(entity.tex_meta[0].y + 0.5));
    }}
    if (entity.tex_meta[1].x > 0.5) {{
        let s = tex_texel(uRoughTex, in.uv * entity.tex_meta[1].zw);
        roughness_v = sample_port_scalar(s, i32(entity.tex_meta[1].y + 0.5));
    }}
    if (entity.tex_meta[2].x > 0.5) {{
        let s = tex_texel(uMetalTex, in.uv * entity.tex_meta[2].zw);
        metallic_v = sample_port_scalar(s, i32(entity.tex_meta[2].y + 0.5));
    }}
    if (entity.tex_meta[3].x > 0.5) {{
        let s = tex_texel(uEmisTex, in.uv * entity.tex_meta[3].zw);
        emissive_v = sample_port_vec3(s, i32(entity.tex_meta[3].y + 0.5));
    }}

    let n = normalize(in.normal);
    let to_light = normalize(-frame.dl_dir.xyz);
    let dl_shadow = textureLoad(
        dl_shadow_tex, vec2<i32>(0, i32(entity.face_offset.x + 0.5) + in.face_id), 0).r;
    let lambert = clamp(dot(n, to_light), 0.0, 1.0) * dl_shadow;
    var lum: vec3<f32>;
    if (frame.flags.x > 0.5) {{
        lum = eval_ambient_cube(n) + frame.dl_color_ambient.xyz * lambert;
    }} else {{
        let amb = frame.dl_color_ambient.w;
        lum = vec3<f32>(amb) + frame.dl_color_ambient.xyz * ((1.0 - amb) * lambert);
    }}
    lum = lum + textureLoad(gi_tex, vec2<i32>(0, i32(entity.face_offset.x + 0.5) + in.face_id), 0).rgb;

    // PBR setup -- alpha/f0/specScale mirror renderer.py's/GLRenderer's
    // per-pixel gather exactly. specScale = 1 - roughness*(1-metallic) is the
    // backward-compat gate: zero at the default params (roughness=1, metallic=0).
    var alpha = clamp(roughness_v, 0.02, 1.0);
    alpha = alpha * alpha;
    let f0 = mix(vec3<f32>(0.04), base_color, metallic_v);
    let spec_scale = 1.0 - roughness_v * (1.0 - metallic_v);
    let view_dir = normalize(frame.camera_pos.xyz - in.world_pos);
    var spec = vec3<f32>(0.0);

    if (spec_scale > 1e-6 && lambert > 1e-4) {{
        let brdf = ggx_specular(n, view_dir, to_light, alpha, f0);
        spec = spec + frame.dl_color_ambient.xyz * brdf * (lambert * spec_scale);
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
        var radiance = li.color.w * atten;

        let is_spot = li.cone.x > -1.5;
        let has_ies = li.axis_ies.w >= 0.0;
        if (is_spot || has_ies) {{
            let to_frag = -delta / dist;
            let cos_ang = dot(to_frag, li.axis_ies.xyz);
            if (is_spot) {{
                let cone = clamp((cos_ang - li.cone.y) / max(li.cone.x - li.cone.y, 1e-6), 0.0, 1.0);
                radiance = radiance * cone * cone;
            }}
            if (has_ies) {{
                let ang = degrees(acos(clamp(cos_ang, -1.0, 1.0)));
                let row = i32(li.axis_ies.w + 0.5);
                let mul = textureLoad(ies_tex, vec2<i32>(i32(ang), row), 0).r;
                radiance = radiance * mul;
            }}
        }}

        let shadow = textureLoad(shadow_tex, vec2<i32>(i, face_offset + in.face_id), 0).r;
        radiance = radiance * shadow;
        lum = lum + li.color.xyz * (radiance * lambert_l);

        if (spec_scale > 1e-6 && radiance > 1e-4) {{
            let l_dir = delta / dist;
            let brdf = ggx_specular(n, view_dir, l_dir, alpha, f0);
            spec = spec + li.color.xyz * brdf * (radiance * lambert_l * spec_scale);
        }}
    }}

    // Metallic diffuse gate applies to the whole accumulated diffuse-ish
    // term at once (ambient+directional+GI+point/spot lambert) -- a single
    // scalar multiplier distributes linearly over the sum, matching
    // renderer.py's/GLRenderer's per-term gating exactly.
    lum = lum * (1.0 - metallic_v);
    var out_color = base_color * lum + spec;

    let view_delta = in.world_pos - frame.camera_pos.xyz;
    let frag_dist = length(view_delta);
    let ray_dir = view_delta / max(frag_dist, 1e-6);
    if (i32(frame.fog_end.y + 0.5) > 0) {{
        out_color = apply_fog_volumes(out_color, frame.camera_pos.xyz, ray_dir, 0.0, frag_dist);
    }}

    if (frame.flags.z > 0.5) {{
        let f = clamp((frag_dist - frame.fog_color_start.w) / (frame.fog_end.x - frame.fog_color_start.w),
                      0.0, 1.0);
        out_color = mix(out_color, frame.fog_color_start.xyz, f);
    }}

    // Emissive is unconditional -- visible even unlit/in shadow, added
    // after fog, matching renderer.py's/GLRenderer's ordering.
    out_color = out_color + emissive_v;
    return vec4<f32>(clamp(out_color, vec3<f32>(0.0), vec3<f32>(1.0)), in.opacity);
}}
"""

# SkyUniforms field order+size must match `_pack_sky_uniforms` below.
_SKY_WGSL = f"""
const PI: f32 = 3.14159265359;
const MAX_FOG_VOL: i32 = {_MAX_FOG_VOL};

struct SkyUniforms {{
    cam_right: vec4<f32>,   // xyz
    cam_up: vec4<f32>,      // xyz
    cam_fwd: vec4<f32>,     // xyz
    view_size: vec4<f32>,   // x=w y=h z=focalK
    flags: vec4<f32>,       // x=useEnv y=useGradient z=sunEnabled w=fogVolCount
    sky_top: vec4<f32>,     // xyz (0..1)
    sky_horizon: vec4<f32>, // xyz (0..1)
    bg_color: vec4<f32>,    // xyz (0..1)
    camera_pos: vec4<f32>,  // xyz -- fog-volume ray origin
    sun_dir: vec4<f32>,     // xyz, w = disc size (deg)
    sun_color: vec4<f32>,   // xyz (already * intensity), w = disc softness
    sun_extra: vec4<f32>,   // x = glow, y = fog-volume far clip
    fog_vol_lo: array<vec4<f32>, MAX_FOG_VOL>,     // xyz = box lo, w = density
    fog_vol_hi: array<vec4<f32>, MAX_FOG_VOL>,     // xyz = box hi, w = height falloff
    fog_vol_color: array<vec4<f32>, MAX_FOG_VOL>,  // xyz = color (0..1)
}};

@group(0) @binding(0) var<uniform> sky: SkyUniforms;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    // big-triangle fullscreen trick -- no vertex buffer needed
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[vi], 0.0, 1.0);
}}

fn sky_fog_vol_segment(origin: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>,
                       t_near: f32, t_far: f32) -> vec2<f32> {{
    let inv_dir = vec3<f32>(1.0) / dir;
    let t1v = (lo - origin) * inv_dir;
    let t2v = (hi - origin) * inv_dir;
    let tmin = min(t1v, t2v);
    let tmax = max(t1v, t2v);
    let t0 = max(max(max(tmin.x, tmin.y), tmin.z), t_near);
    let t1 = min(min(min(tmax.x, tmax.y), tmax.z), t_far);
    return vec2<f32>(t0, t1);
}}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {{
    // wgpu's @builtin(position) origin is TOP-left with y increasing
    // DOWNWARD (unlike GL's gl_FragCoord, which is bottom-left/y-up) --
    // flip the y term here so +dcy still means "up" in camera space.
    let dcx = (frag.x - sky.view_size.x * 0.5) / sky.view_size.z;
    let dcy = (sky.view_size.y * 0.5 - frag.y) / sky.view_size.z;
    let dir = normalize(dcx * sky.cam_right.xyz + dcy * sky.cam_up.xyz + sky.cam_fwd.xyz);

    var base: vec3<f32>;
    if (sky.flags.x > 0.5) {{
        let theta = acos(clamp(dir.y, -1.0, 1.0));
        var phi = atan2(dir.z, dir.x);
        if (phi < 0.0) {{ phi = phi + 2.0 * PI; }}
        let uv = vec2<f32>(phi / (2.0 * PI), theta / PI);
        base = textureSample(env_tex, env_samp, uv).rgb;
    }} else if (sky.flags.y > 0.5) {{
        // top of screen (frag.y=0) -> sky_top, bottom (frag.y=view_size.y) -> sky_horizon;
        // no inversion needed here since wgpu's y-down frag coord already
        // runs top-to-bottom in the same order we want top-to-horizon.
        let f = clamp(frag.y / sky.view_size.y, 0.0, 1.0);
        base = mix(sky.sky_top.xyz, sky.sky_horizon.xyz, f);
    }} else {{
        base = sky.bg_color.xyz;
    }}

    if (sky.flags.z > 0.5) {{
        let cos_ang = clamp(dot(dir, sky.sun_dir.xyz), -1.0, 1.0);
        let ang = degrees(acos(cos_ang));
        let disc_size = sky.sun_dir.w;
        let soft = max(disc_size * sky.sun_color.w, 0.001);
        let e0 = disc_size - soft;
        let e1 = disc_size + soft;
        let t = clamp((e1 - ang) / max(e1 - e0, 0.0001), 0.0, 1.0);
        let disc = t * t * (3.0 - 2.0 * t);
        let halo_deg = disc_size * 12.0 + 3.0;
        let g = clamp(1.0 - ang / halo_deg, 0.0, 1.0);
        let glow_amt = sky.sun_extra.x * g * g * g;
        base = base + sky.sun_color.xyz * disc + sky.sun_color.xyz * glow_amt * 0.5;
    }}

    let fog_vol_count = i32(sky.flags.w + 0.5);
    let fog_far = sky.sun_extra.y;
    for (var i: i32 = 0; i < fog_vol_count; i = i + 1) {{
        let lo = sky.fog_vol_lo[i];
        let hi = sky.fog_vol_hi[i];
        let seg = sky_fog_vol_segment(sky.camera_pos.xyz, dir, lo.xyz, hi.xyz, 0.0, fog_far);
        let seg_len = max(seg.y - seg.x, 0.0);
        var density = lo.w;
        if (hi.w > 0.0) {{
            let mid_h = sky.camera_pos.y + dir.y * (0.5 * (seg.x + seg.y));
            density = density * exp(-max(mid_h, 0.0) * hi.w);
        }}
        let t_ext = exp(-density * seg_len);
        base = base * t_ext + sky.fog_vol_color[i].xyz * (1.0 - t_ext);
    }}

    return vec4<f32>(clamp(base, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}}
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
    # FrameUniforms: 6 header vec4 + 6 ambient_cube vec4 + 3x MAX_FOG_VOL fog-vol vec4
    _FRAME_UBO_SIZE = 192 + 48 * _MAX_FOG_VOL
    _LIGHT_UBO_SIZE = 64 * _MAX_LIGHTS  # LightData = 4 vec4 = 64B, x16 lights
    # 2x mat4x4 (64B) + 3x vec4 (48B) + 1x vec4 (16B) + 4x vec4 tex_meta (64B)
    _ENTITY_UBO_SIZE = 256
    # SkyUniforms: 12 header vec4 (incl. camera_pos/sun_*) + 3x MAX_FOG_VOL fog-vol vec4
    _SKY_UBO_SIZE = 192 + 48 * _MAX_FOG_VOL

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
        self._gi = GITracer()
        # Per-pixel texturing (run 4/4): GPU texture objects for a
        # material's direct texture binding, cached by the material's rel
        # path -- same key space as `texture_mod.load_texture_rel`'s own
        # CPU-side float-array cache and GLRenderer's `_material_tex_cache`
        # -- loaded/uploaded once, reused every draw/frame.
        self._material_tex_cache: dict[str, dict] = {}

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
                    {  # roughness/metallic(2f) + emissive(3f) -- matches _build_pbr's output
                        "array_stride": 4 * 5, "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x2", "offset": 0, "shader_location": 4},
                            {"format": "float32x3", "offset": 8, "shader_location": 5},
                        ],
                    },
                    {  # opacity(1f) -- matches _build_opacity's output
                        "array_stride": 4, "step_mode": "vertex",
                        "attributes": [{"format": "float32", "offset": 0, "shader_location": 6}],
                    },
                    {  # uv(2f) -- matches _build_uv's per-vertex-corner output (run 4/4)
                        "array_stride": 4 * 2, "step_mode": "vertex",
                        "attributes": [{"format": "float32x2", "offset": 0, "shader_location": 7}],
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
        # Per-pixel texturing (run 4/4): group 2 is the 4 fixed texture
        # slots (base_color/roughness/metallic/emissive), auto-inferred from
        # the fragment shader's @group(2) declarations same as bgl0/bgl1.
        self._mesh_bgl2 = self._mesh_pipeline.get_bind_group_layout(2)
        # Translucent pass: same shader + vertex layout, sharing an explicit
        # pipeline layout (built from the opaque pipeline's auto-inferred
        # bind group layouts) so bind groups created once work for either
        # pipeline -- two independent layout="auto" pipelines from the same
        # shader are NOT guaranteed bind-group-compatible per the WebGPU
        # spec, only structurally identical, so this can't be layout="auto"
        # again. Differs from the opaque pipeline only in depth-write
        # (off, so back-to-front translucent faces blend instead of
        # occluding each other -- occlusion by opaque geometry still
        # happens via depth-test-on) and standard alpha blending. Group 2
        # must be included even though translucent entities always pass
        # `{}` bindings (see module docstring) -- the shared fragment shader
        # references @group(2) structurally, so every draw through this
        # pipeline needs SOME bind group 2 set (the all-dummy one).
        mesh_pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._mesh_bgl0, self._mesh_bgl1, self._mesh_bgl2])
        self._mesh_translucent_pipeline = device.create_render_pipeline(
            layout=mesh_pipeline_layout,
            vertex={
                "module": mesh_shader, "entry_point": "vs_main",
                "buffers": [
                    {"array_stride": 4 * 7, "step_mode": "vertex", "attributes": [
                        {"format": "float32x3", "offset": 0, "shader_location": 0},
                        {"format": "float32x3", "offset": 12, "shader_location": 1},
                        {"format": "float32", "offset": 24, "shader_location": 2},
                    ]},
                    {"array_stride": 4 * 3, "step_mode": "vertex", "attributes": [
                        {"format": "float32x3", "offset": 0, "shader_location": 3}]},
                    {"array_stride": 4 * 5, "step_mode": "vertex", "attributes": [
                        {"format": "float32x2", "offset": 0, "shader_location": 4},
                        {"format": "float32x3", "offset": 8, "shader_location": 5},
                    ]},
                    {"array_stride": 4, "step_mode": "vertex", "attributes": [
                        {"format": "float32", "offset": 0, "shader_location": 6}]},
                    {"array_stride": 4 * 2, "step_mode": "vertex", "attributes": [
                        {"format": "float32x2", "offset": 0, "shader_location": 7}]},
                ],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "back"},
            depth_stencil={"format": self._DEPTH_FORMAT, "depth_write_enabled": False,
                           "depth_compare": "less"},
            fragment={"module": mesh_shader, "entry_point": "fs_main",
                      "targets": [{
                          "format": self._COLOR_FORMAT,
                          "blend": {
                              "color": {"src_factor": "src-alpha",
                                        "dst_factor": "one-minus-src-alpha",
                                        "operation": "add"},
                              "alpha": {"src_factor": "one",
                                        "dst_factor": "one-minus-src-alpha",
                                        "operation": "add"},
                          },
                      }]},
        )

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

        # per-face one-bounce GI (see GITracer): rgba32float, alpha unused --
        # same (1, total_faces) layout as GLRenderer's giTex, sampled by
        # textureLoad(gi_tex, (0, faceOffset + faceId)) in the mesh shader.
        self._gi_tex_size = (1, 1)
        self._gi_tex = device.create_texture(
            size=(1, 1, 1), format="rgba32float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self._gi_view = self._gi_tex.create_view()

        # directional (sun) shadow -- reserved single-column r32float
        # texture, same (1, total_faces) layout as gi_tex above, indexed
        # like shadowTex but with a fixed light index of 0. Mirrors
        # GLRenderer's dlShadowTex exactly (see module docstring).
        self._dl_shadow_tex_size = (1, 1)
        self._dl_shadow_tex = device.create_texture(
            size=(1, 1, 1), format="r32float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self._dl_shadow_view = self._dl_shadow_tex.create_view()

        # Per-pixel texturing (run 4/4): 1x1 opaque-white dummy, always
        # bound to whichever of the 4 channel slots (group 2) a material
        # doesn't directly bind a texture to -- mirrors GLRenderer's
        # `_dummy_tex`. rgba32float (not rgba8unorm) so it matches
        # `_get_material_tex_view`'s upload format exactly -- a bind group
        # can't mix sample types across draws using the same layout.
        self._dummy_tex = device.create_texture(
            size=(1, 1, 1), format="rgba32float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        device.queue.write_texture(
            {"texture": self._dummy_tex}, np.ones(4, dtype=np.float32).tobytes(),
            {"bytes_per_row": 16, "rows_per_image": 1}, (1, 1, 1))
        self._dummy_tex_view = self._dummy_tex.create_view()

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
            {"binding": 4, "resource": self._gi_view},
            {"binding": 5, "resource": self._dl_shadow_view},
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

    def _upload_gi_tex(self, gi_data: np.ndarray, total_faces: int) -> None:
        """Per-face RGB indirect-light texture, same (1, total_faces)
        face-id-indexed layout as GLRenderer's giTex."""
        wgpu = self._wgpu
        size = (1, max(total_faces, 1))
        if self._gi_tex_size != size:
            self._gi_tex.destroy()
            self._gi_tex = self.device.create_texture(
                size=(size[0], size[1], 1), format="rgba32float",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self._gi_view = self._gi_tex.create_view()
            self._gi_tex_size = size
            self._frame_bind_group = self._make_frame_bind_group()  # view changed
        w_s, h_s = size
        rgba = np.zeros((h_s, w_s, 4), dtype=np.float32)
        rgba[:, 0, :3] = gi_data
        data = np.ascontiguousarray(rgba, dtype=np.float32)
        self.device.queue.write_texture(
            {"texture": self._gi_tex}, data.tobytes(),
            {"bytes_per_row": w_s * 16, "rows_per_image": h_s}, (w_s, h_s, 1))

    def _upload_dl_shadow_tex(self, data: np.ndarray, total_faces: int) -> None:
        """Reserved single-column texture for the directional (sun) shadow
        factor, same (1, total_faces) layout as `_upload_gi_tex`, indexed
        like `shadowTex` but with a fixed light index of 0. Mirrors
        GLRenderer's `_upload_dl_shadow_tex` exactly."""
        wgpu = self._wgpu
        size = (1, max(total_faces, 1))
        if self._dl_shadow_tex_size != size:
            self._dl_shadow_tex.destroy()
            self._dl_shadow_tex = self.device.create_texture(
                size=(size[0], size[1], 1), format="r32float",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self._dl_shadow_view = self._dl_shadow_tex.create_view()
            self._dl_shadow_tex_size = size
            self._frame_bind_group = self._make_frame_bind_group()  # view changed
        w_s, h_s = size
        buf = np.ascontiguousarray(data, dtype=np.float32)
        self.device.queue.write_texture(
            {"texture": self._dl_shadow_tex}, buf.tobytes(),
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

    def _get_material_tex_view(self, rel_path: str) -> "object | None":
        """GPU texture view for a material's direct texture binding, cached
        by rel path (see `_material_tex_cache`) -- mirrors GLRenderer's
        `_get_material_tex`. Loads through `texture_mod.load_texture_rel`,
        the SAME float-array cache the CPU renderer's `_pixel_uv`/
        `sample_texture` path uses, so this never re-decodes an image the
        CPU path already has in memory. Returns None if the texture can't
        be resolved (missing at render time) -- caller falls back to the
        per-face bake exactly like GL's/CPU's `if img is None`.

        Uploaded as plain linear rgba32float (no sRGB, no flip) -- see
        `tex_texel()` in `_MESH_WGSL` for why no flip is needed here: this
        renderer computes its own nearest-neighbor texel index by hand
        (`textureLoad`, not `textureSample`), reproducing `texture.
        sample_texture`'s v-to-row formula directly against an unflipped
        upload, instead of fighting a hardware sampler's implicit
        v-convention the way GLRenderer's flip-before-upload does."""
        cached = self._material_tex_cache.get(rel_path)
        if cached is not None:
            return cached["view"]
        img = texture_mod.load_texture_rel(rel_path)  # (H, W, 4) float64 0..1, row 0 = image top
        if img is None:
            return None
        wgpu = self._wgpu
        data = np.ascontiguousarray(img, dtype=np.float32)
        h, w = data.shape[:2]
        tex = self.device.create_texture(
            size=(w, h, 1), format="rgba32float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture(
            {"texture": tex}, data.tobytes(),
            {"bytes_per_row": w * 16, "rows_per_image": h}, (w, h, 1))
        cache = {"tex": tex, "view": tex.create_view()}
        self._material_tex_cache[rel_path] = cache
        return cache["view"]

    # Output channels a texture can bind to directly on the mesh shader,
    # in FIXED order matching both `EntityUniforms.tex_meta`'s 4 slots and
    # bind group 2's 4 texture bindings -- "opacity" is excluded (translucent
    # entities never resolve real bindings here, see module docstring).
    _TEX_CHANNELS = ("base_color", "roughness", "metallic", "emissive")

    def _tex_meta_and_views(self, bindings: dict):
        """(meta (4,4) float32, views [4]) for one draw call's texture
        channel state, from `MaterialGraph.direct_texture_bindings()`'s
        result -- mirrors GLRenderer's `_set_texture_channel_uniforms`.
        meta[i] = (has, port_code, u_tiling, v_tiling); a channel with no
        binding, or whose texture can't be resolved, gets has=0 and the
        1x1 dummy view -- the shader's `tex_meta[i].x > 0.5` guard means
        the dummy is never actually sampled, but bind group 2 must always
        point at SOME texture in every slot."""
        meta = np.zeros((4, 4), dtype=np.float32)
        meta[:, 2:4] = 1.0  # default tiling, harmless when has=0
        views = [self._dummy_tex_view] * 4
        for i, channel in enumerate(self._TEX_CHANNELS):
            binding = bindings.get(channel)
            if binding is None:
                continue
            view = self._get_material_tex_view(binding.texture)
            if view is None:
                continue
            views[i] = view
            meta[i] = (1.0, float(_PORT_CODE.get(binding.port, 0)),
                      float(binding.u_tiling), float(binding.v_tiling))
        return meta, views

    def _get_entity_tex_bind_group(self, entity, views):
        """Bind group 2 (the 4 texture channel slots) for one entity, cached
        like `_get_entity_uniforms`'s bind group and rebuilt only when the
        resolved view identities change (rare -- a material edit, not a
        per-frame event); `views` identities are stable across frames since
        `_material_tex_cache`/`_dummy_tex_view` entries are permanent once
        created, so an unchanged material's key never rebuilds this."""
        cache = self._get_entity_uniforms(entity)
        key = tuple(id(v) for v in views)
        if cache.get("tex_key") != key:
            cache["tex_bind_group"] = self.device.create_bind_group(
                layout=self._mesh_bgl2,
                entries=[{"binding": i, "resource": v} for i, v in enumerate(views)])
            cache["tex_key"] = key
        return cache["tex_bind_group"]

    # ------------------------------------------------------------------
    # per-mesh / per-entity GPU buffer caches (mirrors GLRenderer)
    # ------------------------------------------------------------------
    def _get_geo_cache(self, mesh) -> dict:
        # keyed on the mesh's own monotonic serial (`Mesh._cache_id`), not
        # id(mesh) -- CPython recycles freed addresses, so a scene rebuild
        # could land a new Mesh on a previous one's address and an
        # id()-keyed cache would return the WRONG entry (mirrors GLRenderer's
        # identical fix; see gl_renderer.py's module docstring).
        wgpu = self._wgpu
        key = mesh._cache_id
        cache = self._geo_cache.get(key)
        if cache is None:
            pos, nrm, fid, face_id_tri, m = _build_geometry(mesh)
            geom = np.concatenate([pos, nrm, fid[:, None]], axis=1).astype(np.float32)
            geom_buf = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(geom).tobytes(), usage=wgpu.BufferUsage.VERTEX)
            color = _build_color(mesh, face_id_tri)
            color_buf = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(color).tobytes(), usage=wgpu.BufferUsage.VERTEX)
            rm, emissive = _build_pbr(mesh, face_id_tri)
            pbr = np.concatenate([rm, emissive], axis=1)
            pbr_buf = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(pbr).tobytes(), usage=wgpu.BufferUsage.VERTEX)
            opacity = _build_opacity(mesh, face_id_tri)
            opacity_buf = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(opacity).tobytes(), usage=wgpu.BufferUsage.VERTEX)
            # UV attribute for per-pixel texturing (run 4/4) -- unlike color/
            # pbr/opacity, corner_uvs is fixed at mesh construction (never
            # re-baked by the material editor), so this buffer is built once
            # here and never rewritten in the cache-hit branch below (same
            # as GLRenderer's vbo_uv).
            uv = _build_uv(mesh)
            uv_buf = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(uv).tobytes(), usage=wgpu.BufferUsage.VERTEX)
            cache = {"geom_buf": geom_buf, "color_buf": color_buf, "pbr_buf": pbr_buf,
                     "opacity_buf": opacity_buf, "uv_buf": uv_buf,
                     "count": pos.shape[0], "face_id_tri": face_id_tri,
                     "color_id": id(mesh.face_colors),
                     "pbr_id": (id(mesh.face_roughness), id(mesh.face_metallic),
                                id(mesh.face_emissive)),
                     "opacity_id": id(mesh.face_opacity)}
            self._geo_cache[key] = cache
        else:
            if cache["color_id"] != id(mesh.face_colors):
                # buffers from create_buffer_with_data aren't COPY_DST, so a
                # recolor (material editor) rebuilds the small color buffer
                # rather than writing into it -- this is a rare, not per-frame,
                # path (unlike GL's mutable-in-place vbo.write()).
                cache["color_buf"].destroy()
                color = _build_color(mesh, cache["face_id_tri"])
                cache["color_buf"] = self.device.create_buffer_with_data(
                    data=np.ascontiguousarray(color).tobytes(), usage=wgpu.BufferUsage.VERTEX)
                cache["color_id"] = id(mesh.face_colors)
            pbr_id = (id(mesh.face_roughness), id(mesh.face_metallic), id(mesh.face_emissive))
            if cache["pbr_id"] != pbr_id:
                cache["pbr_buf"].destroy()
                rm, emissive = _build_pbr(mesh, cache["face_id_tri"])
                pbr = np.concatenate([rm, emissive], axis=1)
                cache["pbr_buf"] = self.device.create_buffer_with_data(
                    data=np.ascontiguousarray(pbr).tobytes(), usage=wgpu.BufferUsage.VERTEX)
                cache["pbr_id"] = pbr_id
            if cache["opacity_id"] != id(mesh.face_opacity):
                cache["opacity_buf"].destroy()
                opacity = _build_opacity(mesh, cache["face_id_tri"])
                cache["opacity_buf"] = self.device.create_buffer_with_data(
                    data=np.ascontiguousarray(opacity).tobytes(), usage=wgpu.BufferUsage.VERTEX)
                cache["opacity_id"] = id(mesh.face_opacity)
        return cache

    def _prune_geo_cache(self, live_entities) -> None:
        # keep EVERY LOD level's buffers alive for a live entity, not just
        # its currently-selected one -- see gl_renderer.py's identical
        # comment on its _prune_geo_cache (mirrors this one).
        live_mesh_ids = set()
        for e in live_entities:
            live_mesh_ids.add(e.mesh._cache_id)
            for m in e.lod_meshes:
                live_mesh_ids.add(m._cache_id)
        for key in [k for k in self._geo_cache if k not in live_mesh_ids]:
            c = self._geo_cache.pop(key)
            c["geom_buf"].destroy()
            c["color_buf"].destroy()
            c["pbr_buf"].destroy()
            c["opacity_buf"].destroy()
            c["uv_buf"].destroy()
        # entity._cache_id, not id(entity) -- same recycled-address hazard
        # as the geometry cache above, but for the per-entity uniform/bind
        # group cache (`_get_entity_uniforms`).
        live_ent_ids = {e._cache_id for e in live_entities}
        for key in [k for k in self._entity_uniform_cache if k not in live_ent_ids]:
            self._entity_uniform_cache.pop(key)["ubo"].destroy()

    def _get_entity_uniforms(self, entity) -> dict:
        wgpu = self._wgpu
        key = entity._cache_id
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
                              face_offset: int, tex_meta: np.ndarray) -> bytes:
        """EntityUniforms: mvp(64B) + model(64B) + 3x normal_mat column
        vec4(48B) + face_offset vec4(16B) + 4x tex_meta vec4(64B) = 256B.
        Matrices are written `.T`-flattened (row i of the transpose =
        column i of the original) to match WGSL's column-major
        mat4x4/vec4-column storage, same trick as GLRenderer's
        `_write_mat`. `tex_meta` is `_tex_meta_and_views`'s (4, 4) array,
        one (has, port_code, u_tiling, v_tiling) row per channel, in the
        fixed order `_TEX_CHANNELS` / bind group 2's slots agree on."""
        buf = np.zeros(64, dtype=np.float32)
        buf[0:16] = np.ascontiguousarray(mvp.T, dtype=np.float32).ravel()
        buf[16:32] = np.ascontiguousarray(model.T, dtype=np.float32).ravel()
        nmat_t = np.ascontiguousarray(normal_mat.T, dtype=np.float32)  # nmat_t[c] = column c
        buf[32:35] = nmat_t[0]
        buf[36:39] = nmat_t[1]
        buf[40:43] = nmat_t[2]
        buf[44] = float(face_offset)
        buf[48:64] = np.ascontiguousarray(tex_meta, dtype=np.float32).ravel()
        return buf.tobytes()

    def _pack_frame_uniforms(self, scene, camera, env, lights, vols) -> bytes:
        """FrameUniforms: 6 header vec4 (96B) + 6 ambient_cube vec4 (96B)
        + 3x MAX_FOG_VOL fog-vol vec4 arrays (48B * MAX_FOG_VOL) = 384B."""
        dl = scene.light
        fog = scene.fog
        buf = np.zeros(self._FRAME_UBO_SIZE // 4, dtype=np.float32)
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
        buf[21] = float(len(vols))
        if env is not None:
            cube = np.zeros((6, 4), dtype=np.float32)
            cube[:, :3] = env.ambient_cube
            buf[24:48] = cube.ravel()
        lo = np.zeros((_MAX_FOG_VOL, 4), dtype=np.float32)
        hi = np.zeros((_MAX_FOG_VOL, 4), dtype=np.float32)
        color = np.zeros((_MAX_FOG_VOL, 4), dtype=np.float32)
        for i, (vlo, vhi, fv) in enumerate(vols):
            lo[i, :3] = vlo
            lo[i, 3] = fv.density
            hi[i, :3] = vhi
            hi[i, 3] = fv.height_falloff
            color[i, :3] = np.asarray(fv.color, dtype=np.float32) / 255.0
        buf[48:48 + 16] = lo.ravel()
        buf[64:64 + 16] = hi.ravel()
        buf[80:80 + 16] = color.ravel()
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

    def _pack_sky_uniforms(self, scene, camera, env, w: int, h: int, vols) -> bytes:
        """SkyUniforms: 12 header vec4 (192B) + 3x MAX_FOG_VOL fog-vol vec4
        arrays (48B * MAX_FOG_VOL) = 384B."""
        k = 0.5 * h / math.tan(math.radians(camera.fov) * 0.5)
        rot = (rotation_y(camera.yaw) @ rotation_x(camera.pitch))[:3, :3]
        right = rot @ np.array([1.0, 0.0, 0.0])
        up = rot @ np.array([0.0, 1.0, 0.0])
        fwd = rot @ np.array([0.0, 0.0, -1.0])
        buf = np.zeros(self._SKY_UBO_SIZE // 4, dtype=np.float32)
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
        buf[32:35] = camera.position.to_array().astype(np.float32)  # camera_pos
        sun = _sun_sky_info(scene)
        if sun is not None:
            buf[18] = 1.0  # flags.z = sunEnabled
            buf[36:39] = sun.dir
            buf[39] = float(sun.disc_size)
            buf[40:43] = sun.color
            buf[43] = float(sun.disc_softness)
            buf[44] = float(sun.glow)
        buf[45] = _FOG_SKY_FAR
        buf[19] = float(len(vols))  # flags.w = fogVolCount
        lo = np.zeros((_MAX_FOG_VOL, 4), dtype=np.float32)
        hi = np.zeros((_MAX_FOG_VOL, 4), dtype=np.float32)
        color = np.zeros((_MAX_FOG_VOL, 4), dtype=np.float32)
        for i, (vlo, vhi, fv) in enumerate(vols):
            lo[i, :3] = vlo
            lo[i, 3] = fv.density
            hi[i, :3] = vhi
            hi[i, 3] = fv.height_falloff
            color[i, :3] = np.asarray(fv.color, dtype=np.float32) / 255.0
        base = 48
        buf[base:base + 16] = lo.ravel()
        buf[base + 16:base + 32] = hi.ravel()
        buf[base + 32:base + 48] = color.ravel()
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

        # render_meshes[i] is live[i]'s SELECTED LOD -- see gl_renderer.py's
        # identical comment on its render() (mirrors this one): shadow/GI
        # values are always ray-traced at `entity.shadow_mesh()` granularity
        # then gathered onto whichever LOD is actually drawn, via
        # `_lod_gather`.
        live = [e for e in scene.entities if e.mesh is not None and e.visible]
        render_meshes = [e.render_mesh() for e in live]
        face_counts = [int(m.faces.shape[0]) for m in render_meshes]
        total_faces = int(sum(face_counts))
        offsets = np.cumsum([0] + face_counts[:-1]).tolist() if face_counts else []

        shadow_data = np.ones((max(total_faces, 1), self.MAX_LIGHTS), dtype=np.float32)
        if tracer is not None and total_faces > 0:
            for entity, rmesh, off, m in zip(live, render_meshes, offsets, face_counts):
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
                    shadow_data[off:off + m, li_idx] = _lod_gather(entity, rmesh, factors)
        self._upload_shadow_tex(shadow_data, total_faces)

        # directional (sun) shadow -- reserved single-column texture, same
        # face-id indexing as shadowTex above (mirrors GLRenderer.render()'s
        # dl_shadow_data block exactly).
        sun = _scene_sun(scene)
        dl_shadow_data = np.ones((max(total_faces, 1), 1), dtype=np.float32)
        if tracer is not None and sun is not None and sun.shadow_depth > 1e-6 and total_faces > 0:
            dl_dir = scene.light.direction.to_array()
            to_light = -dl_dir / max(np.linalg.norm(dl_dir), 1e-12)
            for entity, rmesh, off, m in zip(live, render_meshes, offsets, face_counts):
                centroids_w, normals_w = _entity_world_faces(entity)
                lambert = np.clip(normals_w @ to_light, 0.0, 1.0)
                active = lambert > 1e-3
                raw = tracer.directional_shadow_factors(
                    entity, dl_dir, sun.shadow_softness, sun.shadow_samples,
                    centroids_w, normals_w, active)
                dshadow = 1.0 - sun.shadow_depth * (1.0 - raw)
                dl_shadow_data[off:off + m, 0] = _lod_gather(entity, rmesh, dshadow)
        self._upload_dl_shadow_tex(dl_shadow_data, total_faces)

        # one-bounce GI -- baked/cached by GITracer, zero per-frame cost once
        # static (mirrors GLRenderer.render()'s GI block exactly).
        gi_data = np.zeros((max(total_faces, 1), 3), dtype=np.float32)
        gi_cfg = getattr(scene, "gi", None)
        if tracer is not None and gi_cfg and gi_cfg.get("enabled") and total_faces > 0:
            gi_map = self._gi.compute(scene, tracer,
                                      lambda casters: _gi_direct_lighting(scene, casters, tracer),
                                      _gi_receiver_geometry,
                                      gi_cfg.get("samples", 16), gi_cfg.get("intensity", 1.0))
            for entity, rmesh, off, m in zip(live, render_meshes, offsets, face_counts):
                gi = gi_map.get(id(entity))
                if gi is not None:
                    gi_data[off:off + m] = _lod_gather(entity, rmesh, gi)
        self._upload_gi_tex(gi_data, total_faces)

        vols = _fog_volumes(scene)

        # Frame-global uniforms: one write per buffer this frame, so all
        # draws in the render pass recorded below see consistent data once
        # the command buffer actually executes after submit() (see module
        # docstring's write-then-submit-ordering note).
        self.device.queue.write_buffer(self._frame_ubo, 0,
                                       self._pack_frame_uniforms(scene, camera, env, lights, vols))
        self.device.queue.write_buffer(self._light_ubo, 0, self._pack_lights(lights))
        self.device.queue.write_buffer(self._sky_ubo, 0,
                                       self._pack_sky_uniforms(scene, camera, env, w, h, vols))

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

        pass_enc.set_bind_group(0, self._frame_bind_group)

        def _draw(entity, rmesh, off, bindings) -> int:
            geo = self._get_geo_cache(rmesh)
            ent = self._get_entity_uniforms(entity)
            model = entity.transform.matrix()
            mvp = proj @ view @ model
            try:
                nmat = np.linalg.inv(model[:3, :3]).T
            except np.linalg.LinAlgError:
                nmat = np.eye(3)
            tex_meta, tex_views = self._tex_meta_and_views(bindings)
            tex_bg = self._get_entity_tex_bind_group(entity, tex_views)
            self.device.queue.write_buffer(
                ent["ubo"], 0, self._pack_entity_uniforms(mvp, model, nmat, off, tex_meta))
            pass_enc.set_bind_group(1, ent["bind_group"])
            pass_enc.set_bind_group(2, tex_bg)
            pass_enc.set_vertex_buffer(0, geo["geom_buf"])
            pass_enc.set_vertex_buffer(1, geo["color_buf"])
            pass_enc.set_vertex_buffer(2, geo["pbr_buf"])
            pass_enc.set_vertex_buffer(3, geo["opacity_buf"])
            pass_enc.set_vertex_buffer(4, geo["uv_buf"])
            pass_enc.draw(geo["count"])
            return geo["count"] // 3

        triangles = 0
        opaque_pairs = [(e, m, off) for e, m, off in zip(live, render_meshes, offsets)
                        if m.faces.shape[0] > 0 and not _is_translucent(e)]
        translucent_pairs = [(e, m, off) for e, m, off in zip(live, render_meshes, offsets)
                             if m.faces.shape[0] > 0 and _is_translucent(e)]

        pass_enc.set_pipeline(self._mesh_pipeline)
        for entity, rmesh, off in opaque_pairs:
            # Per-pixel texturing bindings come from the entity's own
            # material, exactly as GLRenderer's opaque-pass loop reads them
            # -- translucent entities never resolve real bindings (see
            # translucent_pairs below, which always passes {} to match the
            # CPU path never populating `tex_entities` for a translucent
            # entity at all).
            material = getattr(entity, "material", None)
            bindings = material.direct_texture_bindings() if material is not None else {}
            triangles += _draw(entity, rmesh, off, bindings)

        if translucent_pairs:
            # Back-to-front per-entity order, mirroring GLRenderer's
            # translucent pass (see its render() comment for why per-entity,
            # not per-triangle, is the practical granularity here).
            cam_pos = camera.position.to_array()
            translucent_pairs.sort(
                key=lambda p: -float(np.linalg.norm(
                    p[0].transform.matrix()[:3, 3] - cam_pos)))
            pass_enc.set_pipeline(self._mesh_translucent_pipeline)
            for entity, rmesh, off in translucent_pairs:
                triangles += _draw(entity, rmesh, off, {})
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
        for cache in self._material_tex_cache.values():
            cache["tex"].destroy()
        self._material_tex_cache.clear()
        if self._color_tex is not None:
            self._color_tex.destroy()
            self._depth_tex.destroy()
        self._shadow_tex.destroy()
        self._gi_tex.destroy()
        self._dl_shadow_tex.destroy()
        self._ies_tex.destroy()
        self._dummy_env_tex.destroy()
        self._dummy_tex.destroy()
