"""Judge checks for the GPU renderer: parity, IES, cone, shadows, depth.

Sections 7-10 cover per-pixel texturing run 3/4 (OpenGL): a byte-identical-
with-main regression gate for untextured scenes (same pattern as texture_
checks.py's CPU golden, engine/gl_renderer.py's UV attribute + texture
sampling must be a pure no-op when no material directly binds a texture),
real per-pixel variation within one GPU-rasterized face (not vacuous),
GL-vs-CPU textured parity with a numerically-justified tolerance, and
tiling reaching the GLSL sampler.
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.gettempdir()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, REPO)

import numpy as np
import pygame

import engine
from engine import mesh as mesh_mod
from engine import texture as texture_mod
from engine.gl_renderer import GLRenderer
from engine.renderer import Renderer

pygame.init()
pygame.display.set_mode((64, 64))

W, H = 400, 300
gl = GLRenderer.standalone(W, H)


def gl_frame(scene, cam, tracer=None):
    gl.render(scene, cam, (W, H), tracer)
    raw = gl.target.read(components=3) if hasattr(gl, "target") and gl.target else None
    if raw is None:  # fall back to fbo attr naming
        raw = gl.fbo.read(components=3)
    img = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
    return img[::-1]  # GL rows are bottom-up


def build_scene(ies="uniform"):
    sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.4, -1, -0.3),
                                                    ambient=0.15))
    sc.add(engine.Entity("floor", mesh=engine.checkerboard(8, 1.5)))
    sc.add(engine.Entity("cube", mesh=engine.cube(1.0, color=(255, 30, 30)),
                         position=engine.Vec3(0, 0.5, 0)))
    lamp = engine.Entity("lamp", light=engine.PointLight(
        intensity=2.0, range=15, radius=0.3, ies=ies, cast_shadows=False),
        position=engine.Vec3(2, 4, 2))
    sc.add(lamp)
    return sc


cam = engine.Camera(position=engine.Vec3(4, 3.2, 7), yaw=0.45, pitch=-0.3)

# 1. coarse GPU-vs-CPU parity on the same scene
scene = build_scene()
img_gpu = gl_frame(scene, cam).astype(float)
r = Renderer()
r.render_scale = 1
surf = pygame.Surface((W, H))
r.render(surf, scene, cam)
img_cpu = pygame.surfarray.array3d(surf).transpose(1, 0, 2).astype(float)
diff = abs(img_gpu.mean() - img_cpu.mean())
assert diff < 12.0, f"GPU/CPU mean brightness diverges: {diff:.1f}"
red_gpu = ((img_gpu[..., 0] > 100) & (img_gpu[..., 0] > 2 * img_gpu[..., 1])).sum()
assert red_gpu > 500, red_gpu
print(f"parity OK: mean brightness gpu={img_gpu.mean():.1f} cpu={img_cpu.mean():.1f} "
      f"(diff {diff:.1f}), red cube pixels={red_gpu}")

# 2. IES profile changes the image (downlight vs uniform)
img_down = gl_frame(build_scene("downlight"), cam).astype(float)
assert abs(img_down.mean() - img_gpu.mean()) > 1.0, "IES profile had no effect"
print(f"IES OK: uniform mean={img_gpu.mean():.1f}, downlight mean={img_down.mean():.1f}")

# 3. spotlight cone: lit patch under the spot, darker away from it
sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.4, -1, -0.3),
                                                ambient=0.04, intensity=0.05))
sc.add(engine.Entity("floor", mesh=engine.checkerboard(8, 1.5)))
spot_e = engine.Entity("spot", light=engine.SpotLight(
    intensity=3.0, range=18, inner=12, outer=25, cast_shadows=False),
    position=engine.Vec3(0, 4, 0), rotation=engine.Vec3(-1.5707, 0, 0))
sc.add(spot_e)
img = gl_frame(sc, engine.Camera(position=engine.Vec3(0, 5, 6), pitch=-0.65)).astype(float)
center = img[int(H*0.55):int(H*0.75), int(W*0.4):int(W*0.6)].mean()
edge = img[int(H*0.55):int(H*0.75), :int(W*0.12)].mean()
assert center > edge * 2.0, (center, edge)
print(f"spot cone OK: center={center:.1f} edge={edge:.1f}")

# 4. ray-traced shadow texture: blocker darkens floor under it
sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.4, -1, -0.3),
                                                ambient=0.10))
floor = engine.Entity("floor", mesh=engine.checkerboard(8, 1.0))
floor.casts_shadow = False
sc.add(floor)
sc.add(engine.Entity("blocker", mesh=engine.cube(1.6),
                     position=engine.Vec3(0, 2.0, 0)))
sc.add(engine.Entity("lamp", light=engine.PointLight(
    intensity=2.5, range=20, radius=0.3, shadow_samples=8),
    position=engine.Vec3(0, 5, 0)))
tracer = engine.ShadowTracer()
tracer.refresh(sc)
shadow_cam = engine.Camera(position=engine.Vec3(0.0, 6.5, 5.5), pitch=-0.9)
img_sh = gl_frame(sc, shadow_cam, tracer).astype(float)
img_no = gl_frame(sc, shadow_cam, None).astype(float)
delta = img_no - img_sh          # shadows only remove light
assert delta.min() > -2.0, delta.min()
assert img_sh.mean() < img_no.mean() - 1.5, (img_sh.mean(), img_no.mean())
assert (delta.mean(axis=-1) > 25).sum() > 300  # a real dark shadow region exists
print(f"gpu shadows OK: mean {img_no.mean():.1f} -> {img_sh.mean():.1f}, "
      f"{(delta.mean(axis=-1) > 25).sum()} strongly shadowed pixels")

# 5. depth buffer: giant floor cannot erase the cube
sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.4, -1, -0.3),
                                                ambient=0.6))
sc.add(engine.Entity("floor", mesh=engine.checkerboard(1, 20.0)))
sc.add(engine.Entity("cube", mesh=engine.cube(0.5, color=(255, 0, 0)),
                     position=engine.Vec3(0, 0.3, -2)))
img = gl_frame(sc, engine.Camera(position=engine.Vec3(0.5, 0.5, 0.5),
                                 pitch=-0.25, yaw=0.1)).astype(float)
red = ((img[..., 0] > 120) & (img[..., 0] > 2 * img[..., 1])).sum()
assert red > 1500, red
print(f"gpu depth OK: {red} red pixels")

# 6. PBR: metallic/roughness/emissive parity CPU vs GL, highlight + emissive
# on GL specifically. Camera looks straight down -z at a single box lit
# mostly by a point light placed near the camera so its reflection off the
# box's front face lands almost dead-on -- a real specular lobe to see
# (checks 1-5's plain diffuse scenes never touch the new PBR path at all;
# spec_scale is 0 there). Ambient/directional kept dim so a diffuse-only
# box never saturates, while the metallic+shiny highlight does.
def build_pbr_scene(box_roughness=1.0, box_metallic=0.0, box_emissive=(0.0, 0.0, 0.0),
                    pt_intensity=0.6):
    sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.4, -1.0, -0.3),
                                                    ambient=0.02, intensity=0.05))
    box_mesh = engine.cube(size=1.6, color=(200, 60, 60))
    box_mesh.face_roughness[:] = box_roughness
    box_mesh.face_metallic[:] = box_metallic
    box_mesh.face_emissive[:] = box_emissive
    box_e = engine.Entity("box", mesh=box_mesh, position=engine.Vec3(0, 0, 0))
    sc.add(box_e)
    sc.add(engine.Entity("lamp", light=engine.PointLight(
        intensity=pt_intensity, range=15, cast_shadows=False),
        position=engine.Vec3(0.3, 0.3, 5.0)))
    return sc

pbr_cam = engine.Camera(position=engine.Vec3(0.0, 0.0, 5.0))

# 6a. CPU-vs-GL mean-brightness parity on a metallic+rough-varying PBR scene
sc_pbr = build_pbr_scene(box_roughness=0.15, box_metallic=1.0)
img_pbr_gpu = gl_frame(sc_pbr, pbr_cam).astype(float)
r2 = Renderer()
r2.render_scale = 1
surf2 = pygame.Surface((W, H))
r2.render(surf2, sc_pbr, pbr_cam)
img_pbr_cpu = pygame.surfarray.array3d(surf2).transpose(1, 0, 2).astype(float)
pbr_diff = abs(img_pbr_gpu.mean() - img_pbr_cpu.mean())
assert pbr_diff < 12.0, f"PBR GPU/CPU mean brightness diverges: {pbr_diff:.1f}"
print(f"pbr parity OK: mean brightness gpu={img_pbr_gpu.mean():.1f} "
      f"cpu={img_pbr_cpu.mean():.1f} (diff {pbr_diff:.1f})")

# 6b. GL specular highlight: metallic+shiny box shows very-bright pixels
# that the same scene at default params (roughness=1, metallic=0, spec_scale
# gated to 0 -- no highlight possible) never reaches.
sc_default = build_pbr_scene()  # defaults: legacy diffuse-only look
img_default_gpu = gl_frame(sc_default, pbr_cam).astype(float)
bright_pbr = int((img_pbr_gpu.max(axis=-1) > 240).sum())
bright_default = int((img_default_gpu.max(axis=-1) > 240).sum())
assert bright_pbr > bright_default, (bright_pbr, bright_default)
assert bright_default == 0, ("default-param scene unexpectedly saturated -- "
                             "test scene isn't isolating the highlight", bright_default)
print(f"gl highlight OK: bright pixels pbr={bright_pbr} default={bright_default}")

# 6c. GL emissive-in-the-dark: emissive face visible with all lights off
def build_pbr_dark_scene(emissive):
    sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.4, -1.0, -0.3),
                                                    ambient=0.0, color=(0, 0, 0),
                                                    intensity=0.0),
                      background=(0, 0, 0))
    box_mesh = engine.cube(size=1.6, color=(10, 10, 10))
    box_mesh.face_emissive[:] = emissive
    sc.add(engine.Entity("box", mesh=box_mesh, position=engine.Vec3(0, 0, 0)))
    return sc

img_dark_off = gl_frame(build_pbr_dark_scene((0.0, 0.0, 0.0)), pbr_cam).astype(float)
img_dark_emis = gl_frame(build_pbr_dark_scene((220.0, 40.0, 40.0)), pbr_cam).astype(float)
assert img_dark_off.max() <= 2, f"expected near-black with no lights/emissive: {img_dark_off.max()}"
assert img_dark_emis.max() > 100, f"emissive face not bright in the dark on GL: {img_dark_emis.max()}"
print(f"gl emissive-in-dark OK: off max={img_dark_off.max():.0f} emissive max={img_dark_emis.max():.0f}")

# 7. UNTEXTURED REGRESSION GATE: this run's shader/geometry changes (UV
#    attribute in `_get_geo_cache`, texture-sampling uniforms and the
#    baseColor/roughnessV/metallicV/emissiveV override locals in _MESH_FS)
#    must not move a single pixel on a scene with no direct texture
#    bindings. `perpixel_gl_untextured_golden.npy` was captured from a
#    `git worktree add ... main` at 01a55c8 (this run's branch point),
#    rendering the SAME `build_scene()`/`cam` as check #1 above through
#    `GLRenderer.standalone` at the same 400x300 -- confirmed reproducible
#    (rendered twice in one worktree process, 0 diff) before being treated
#    as golden, same discipline as texture_checks.py's CPU golden.
_golden_path = os.path.join(REPO, "tests", "fixtures", "perpixel_gl_untextured_golden.npy")
assert os.path.exists(_golden_path), f"no golden at {_golden_path}"
_golden = np.load(_golden_path).astype(np.int32)
_gcurrent = img_gpu.astype(np.int32)  # same scene/cam/resolution as check #1, reused as-is
_gdiff = np.abs(_gcurrent - _golden)
assert _gdiff.max() == 0 and _gdiff.mean() == 0.0, (
    f"UNTEXTURED REGRESSION: GL render differs from main by up to {_gdiff.max()} "
    f"(mean {_gdiff.mean()}) on a scene with no direct texture bindings")
print(f"7. untextured regression gate OK: 0 of {_golden.shape[0] * _golden.shape[1]} pixels "
     "differ vs a main (01a55c8) GL render of the identical scene")

# ---------------------------------------------------------------------
# Per-pixel texturing (run 3/4): same quad-scene model as texture_checks.py
# #17 (direct-bind vs multiply-wrapped fallback control), extended with a
# GL-vs-CPU pixel comparison and a tiling check. See engine/gl_renderer.py's
# `_get_material_tex` / `_set_texture_channel_uniforms` / the new _MESH_FS
# uniforms + sampleTexVec3/sampleTexScalar helpers.
# ---------------------------------------------------------------------
_tex_tmp = os.path.join(TMP, "judge_perpixel_run3_assets")
_tex_dir = os.path.join(_tex_tmp, "textures")
os.makedirs(_tex_dir, exist_ok=True)
_tex_path = os.path.join(_tex_dir, "grad.png")
_grad_surf = pygame.Surface((8, 8))
for _yy in range(8):
    for _xx in range(8):
        _grad_surf.set_at((_xx, _yy), (_xx * 32 % 256, _yy * 32 % 256, (_xx * 7 + _yy * 13) % 256))
pygame.image.save(_grad_surf, _tex_path)
texture_mod.set_texture_root(_tex_tmp)
texture_mod.clear_cache()

_quad_verts = [(-2, -2, -5), (2, -2, -5), (2, 2, -5), (-2, 2, -5)]  # box-projected UVs vary 0..1 across it
_quad_cam = engine.Camera(position=engine.Vec3(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0)


def _build_quad_scene(direct_bind: bool, u_tiling: float = 1.0, v_tiling: float = 1.0):
    scene9 = engine.Scene(light=engine.DirectionalLight(
        engine.Vec3(0, -1, 0), ambient=1.0, color=(255, 255, 255), intensity=0.0))
    ent9 = engine.Entity("quad", mesh=mesh_mod.Mesh(_quad_verts, [(0, 1, 2, 3)]))
    g9 = engine.MaterialGraph()
    ts9 = g9.add("tex_sample", (0, 0))
    g9.nodes[ts9]["texture"] = "textures/grad.png"
    if direct_bind:
        if u_tiling != 1.0 or v_tiling != 1.0:
            tc9 = g9.add("tex_coord", (0, 0))
            g9.nodes[tc9]["params"]["u_tiling"] = u_tiling
            g9.nodes[tc9]["params"]["v_tiling"] = v_tiling
            assert g9.connect(tc9, ts9, "uv")
        assert g9.connect(ts9, g9.output_id(), "base_color", "RGB")
    else:
        mul9 = g9.add("multiply", (0, 0))
        one9 = g9.add("constant3vector", (0, 0))
        assert g9.connect(ts9, mul9, "a", "RGB")
        assert g9.connect(one9, mul9, "b")
        assert g9.connect(mul9, g9.output_id(), "base_color")
    ent9.material = g9
    g9.apply(ent9)
    scene9.add(ent9)
    return scene9


_QW, _QH = 200, 150
_gl_quad = GLRenderer.standalone(_QW, _QH)


def _gl_quad_frame(scene9):
    _gl_quad.render(scene9, _quad_cam, (_QW, _QH), None)
    raw9 = _gl_quad.target.read(components=3)
    return np.frombuffer(raw9, dtype=np.uint8).reshape(_QH, _QW, 3)[::-1]


_bg9 = np.asarray(engine.Scene().background)
_img_gl_direct = _gl_quad_frame(_build_quad_scene(True)).astype(np.int32)
_mask9 = ~np.all(_img_gl_direct == _bg9[None, None, :], axis=2)
_distinct_gl_direct = len(np.unique(_img_gl_direct[_mask9].reshape(-1, 3), axis=0))

_img_gl_fallback = _gl_quad_frame(_build_quad_scene(False)).astype(np.int32)
_mask9b = ~np.all(_img_gl_fallback == _bg9[None, None, :], axis=2)
_distinct_gl_fallback = len(np.unique(_img_gl_fallback[_mask9b].reshape(-1, 3), axis=0))

assert _mask9.sum() > 1000, "quad should cover a substantial part of the frame"
assert _distinct_gl_direct > 50, f"expected many distinct colors from GL per-pixel sampling, got {_distinct_gl_direct}"
assert _distinct_gl_fallback == 1, f"expected exactly 1 color from the per-face bake control, got {_distinct_gl_fallback}"
print(f"8. GL per-pixel variation OK (not vacuous): direct-bind base_color shows "
     f"{_distinct_gl_direct} distinct colors across {int(_mask9.sum())} pixels of ONE "
     f"face on the GPU; multiply-wrapped fallback (same graph, same face) gives exactly "
     f"{_distinct_gl_fallback} -- confirms the GLSL sampler, not lighting, is the source "
     "of variation")

# 9. NEW GATE: GL-vs-CPU textured parity. render_scale MUST be forced to 1
#    for a fair full-resolution comparison -- the default (3) renders the
#    CPU path at 1/3 internal resolution then upscales, which alone
#    produces a much larger, meaningless diff at texel-cell boundaries
#    (caught during implementation: 1587/30000 differing px at the default
#    scale vs 173/30000 at scale=1, same scene/camera/texture).
_r9 = Renderer()
_r9.render_scale = 1
_surf_cpu9 = pygame.Surface((_QW, _QH))
_r9.render(_surf_cpu9, _build_quad_scene(True), _quad_cam)
_img_cpu_direct = pygame.surfarray.array3d(_surf_cpu9).transpose(1, 0, 2).astype(np.int32)
_pdiff = np.abs(_img_gl_direct - _img_cpu_direct)
_pndiff = int((_pdiff.sum(axis=2) > 0).sum())
_pfrac = _pndiff / (_QW * _QH)
# Tolerance, justified by measurement (this exact scene, numbers found
# during implementation): mean diff 0.27, max diff 212, 173/30000 (0.58%)
# differing pixels. The nonzero diffs are NEAREST-neighbor texel-boundary
# aliasing: GL's hardware perspective-correct interpolation and the CPU's
# numpy barycentric world-space math are two independently-implemented,
# both-correct interpolants that can disagree by one texel index right at
# a shared cell edge; this test texture is a deliberately maximal-contrast
# 8x8 gradient so a single-texel edge miss shows up as a big color jump.
# Bounding at mean<3.0 / differing-fraction<3% is >5x the measured
# 0.27 / 0.58% -- tight enough that a real sampling bug (wrong channel,
# wrong UV, missing tiling, wrong V orientation -- any of which would
# desync entire regions, not just edge pixels) would blow through it,
# loose enough not to be a coin-flip on rasterizer rounding.
assert _pdiff.mean() < 3.0, f"GL/CPU textured mean diff too high: {_pdiff.mean():.3f}"
assert _pfrac < 0.03, f"GL/CPU textured differing-pixel fraction too high: {_pfrac:.4f}"
print(f"9. textured parity OK: mean diff={_pdiff.mean():.3f} (tolerance <3.0), "
     f"max diff={_pdiff.max()}, {_pndiff}/{_QW * _QH} differing px "
     f"({100 * _pfrac:.2f}%, tolerance <3%) -- diffs are nearest-neighbor "
     "texel-boundary aliasing between two independent rasterizers, not a "
     "systematic bug (both rendered at render_scale=1 -- see comment above)")

# 10. tiling reaches the GLSL sampler: u_tiling/v_tiling actually change the
#     GL render (still samples the same 8x8 palette -- repeats, doesn't
#     distort or go out of range -- while visibly differing from untiled).
_img_gl_tiled = _gl_quad_frame(_build_quad_scene(True, u_tiling=3.0, v_tiling=3.0)).astype(np.int32)
_mask9c = ~np.all(_img_gl_tiled == _bg9[None, None, :], axis=2)
_distinct_gl_tiled = len(np.unique(_img_gl_tiled[_mask9c].reshape(-1, 3), axis=0))
assert _distinct_gl_tiled > 50, f"tiled GL render should still sample the full palette, got {_distinct_gl_tiled}"
assert not np.array_equal(_img_gl_tiled, _img_gl_direct), "tiling=3 must render differently than tiling=1 on GL"
print(f"10. GL tiling OK: {_distinct_gl_tiled} distinct colors (still the full palette, "
     "repeated), differs from the untiled GL render")

_gl_quad.release()
texture_mod.clear_cache()
texture_mod.set_texture_root(os.path.join(REPO, "assets"))
import shutil as _shutil9
_shutil9.rmtree(_tex_tmp, ignore_errors=True)

# 11. CACHE IDENTITY REGRESSION: `_get_geo_cache` keys off `mesh._cache_id`
# (a monotonic per-Mesh serial, see engine/mesh.py), not `id(mesh)`. The
# `_geo_cache` dict holds no strong reference to the Mesh it caches (only
# derived VBOs/ids), so once a scene is dropped and gc'd, CPython is free to
# reuse that Mesh's address for a brand-new, unrelated Mesh. An id()-keyed
# cache would then return the OLD entry on lookup -- IndexError if face
# counts differ (see tests/gl_checks.py -- covered live by repro_cache.py
# during development), or, worse, SILENTLY WRONG geometry if they coincide,
# since a cache HIT never rebuilds the geometry VBO (only color/pbr/opacity
# get a per-hit staleness check). This reproduces exactly that silent case:
# two 6-quad cubes of the SAME topology but wildly different SIZE, so a
# false hit renders the WRONG (stale) size with no exception at all -- the
# case the bug report says matters more than the crash. Verified against
# pre-fix code (a `git worktree` of main @ c46d706) during implementation:
# this exact loop measured min-large-count == max-small-count (3844 == 3844
# -- a "small" iteration rendering fully as the LARGE cube's stale
# geometry), so it is a real regression gate, not a vacuous one.
import gc as _gc11

_CR_W, _CR_H = 160, 120
_cr_cam = engine.Camera(position=engine.Vec3(0, 0, 6), yaw=0.0, pitch=0.0)
_cr_gl = GLRenderer.standalone(_CR_W, _CR_H)


def _cr_scene(size):
    sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.3, -1, -0.2), ambient=0.6))
    sc.add(engine.Entity("cube", mesh=engine.cube(size, color=(255, 40, 40))))
    return sc


def _cr_frame(scene):
    _cr_gl.render(scene, _cr_cam, (_CR_W, _CR_H))
    raw = _cr_gl.target.read(components=3)
    return np.frombuffer(raw, dtype=np.uint8).reshape(_CR_H, _CR_W, 3)[::-1]


def _cr_red_count(img):
    r, g = img[..., 0].astype(int), img[..., 1].astype(int)
    return int(((r > 100) & (r > 2 * g)).sum())


_cr_addrs_seen, _cr_recycled = {}, False
_cr_small, _cr_large = [], []
for _cr_i in range(40):
    _cr_size = 0.6 if _cr_i % 2 == 0 else 3.2
    _cr_sc = _cr_scene(_cr_size)
    _cr_mesh = _cr_sc.entities[0].mesh
    _cr_key = id(_cr_mesh)  # raw address, purely to detect recycling for the assert below
    _cr_prev = _cr_addrs_seen.get(_cr_key)
    if _cr_prev is not None and _cr_prev != _cr_size:
        _cr_recycled = True
    _cr_addrs_seen[_cr_key] = _cr_size
    _cr_count = _cr_red_count(_cr_frame(_cr_sc))
    (_cr_small if _cr_size < 1.0 else _cr_large).append(_cr_count)
    del _cr_sc, _cr_mesh
    _gc11.collect()

assert _cr_recycled, ("address recycling never occurred in 40 iterations -- "
                      "this run cannot validate the fix; re-run or raise the iteration count")
_cr_min_large, _cr_max_small = min(_cr_large), max(_cr_small)
assert _cr_min_large > _cr_max_small * 3, (
    f"CACHE IDENTITY REGRESSION: a false cache hit rendered a stale (wrong-size) mesh -- "
    f"min large-cube red px={_cr_min_large}, max small-cube red px={_cr_max_small} "
    "(expected the large cube to always dominate)")
_cr_gl.release()
print(f"11. cache identity regression OK: address recycling confirmed over 40 iterations; "
     f"small-cube red px range [{min(_cr_small)}, {max(_cr_small)}], large-cube range "
     f"[{min(_cr_large)}, {max(_cr_large)}] -- no false cache hit smeared sizes together")

# 12. VERSION-STAMP REGRESSION: `_get_geo_cache`'s cache-HIT branch decides
# whether to re-upload the color/pbr/opacity VBOs by comparing a per-array
# stamp against the mesh's CURRENT arrays. The stamp is now `mesh.
# _color_version` (a monotonic counter bumped by the `face_colors` property
# setter, see engine/mesh.py), not `id(mesh.face_colors)`. This is a
# DIFFERENT hazard than #11's cache-KEY fix: here it's the SAME live mesh,
# replaced TWICE with no render between (mesh.face_colors = B, then = C --
# mirrors MaterialGraph.apply being called twice in a row while dragging a
# material slider, materials.py `apply`/editor.py's MaterialEditorUI.update,
# if the fixed-step accumulator fires update() twice before the next
# render -- see core.py's `while accumulator >= self.fixed_dt`). The first
# replacement (B) frees the array the cache last stamped (A); the second
# (C) can then be allocated at A's now-freed address, so an id()-keyed
# stamp would false-match "unchanged" and skip the VBO rebuild -- SILENTLY
# leaving the GPU showing A's stale colour, no exception at all (inverted
# vs #11: a false MATCH, not a false HIT). This drives exactly that shape
# and asserts the RENDERED colour reflects the latest assignment, not that
# a version field merely exists. Verified against pre-fix code (a `git
# worktree` of main @ 5d68517) during implementation: this exact loop hit
# the false match at iteration 1 and rendered the STALE red colour.
_VS_W, _VS_H = 160, 120
_vs_cam = engine.Camera(position=engine.Vec3(0, 0, 4.5), yaw=0.0, pitch=0.0)
_vs_gl = GLRenderer.standalone(_VS_W, _VS_H)
_vs_sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(0, -1, 0),
                                                    ambient=1.0, intensity=0.0))
_vs_ent = engine.Entity("cube", mesh=engine.cube(2.0))
_vs_sc.add(_vs_ent)
_vs_mesh = _vs_ent.mesh
_vs_n = len(_vs_mesh.faces)
_VS_RED = (220.0, 20.0, 20.0)
_VS_GREEN = (20.0, 220.0, 20.0)


def _vs_paint(rgb):
    return np.tile(np.array(rgb), (_vs_n, 1))


def _vs_frame():
    _vs_gl.render(_vs_sc, _vs_cam, (_VS_W, _VS_H))
    raw = _vs_gl.target.read(components=3)
    return np.frombuffer(raw, dtype=np.uint8).reshape(_VS_H, _VS_W, 3)[::-1].astype(np.int64)


def _vs_lit_mean(img):
    lit = img[img.sum(axis=2) > 30]
    return lit.mean(axis=0) if len(lit) else np.zeros(3)


_vs_mesh.face_colors = _vs_paint(_VS_RED)
_vs_frame()                                  # establishes the cache's stamp for the
                                              # array now in mesh.face_colors (A)
_vs_target_id = id(_vs_mesh.face_colors)     # A's id -- what the cache is now stamped with
# Tight loop, NO render (and so no GPU-pipeline allocation churn) between attempts --
# see wgpu_checks.py's identical check #13 for why: a render-per-attempt design still
# converges here (this suite is short enough that CPython's free lists stay shallow),
# but holding the render off between attempts removes allocator noise and converges
# faster and more robustly regardless of how much test-suite history precedes this.
_vs_recycled = False
_vs_attempts = 0
for _vs_i in range(20000):
    _vs_attempts += 1
    _vs_mesh.face_colors = _vs_paint(_VS_GREEN)   # replacement 1: frees the live array
    _vs_mesh.face_colors = _vs_paint(_VS_GREEN)   # replacement 2: may reuse a freed slot
    if id(_vs_mesh.face_colors) == _vs_target_id:
        _vs_recycled = True
        break

assert _vs_recycled, (
    f"address recycling onto the cache-stamped array's original address never occurred "
    f"in {_vs_attempts} no-render replacement attempts -- this run cannot validate the "
    f"fix; re-run or raise the iteration count")
_vs_m = _vs_lit_mean(_vs_frame())
assert _vs_m[1] > _vs_m[0], (
    f"VERSION-STAMP REGRESSION: after {_vs_attempts} no-render replacements, the current "
    f"face_colors array landed back on the address the cache had stamped for the "
    f"ORIGINAL (red) array, and the colour VBO was NOT rebuilt -- rendered mean RGB "
    f"{_vs_m.round(1)} is still RED (stale), not the GREEN just assigned")
_vs_gl.release()
print(f"12. version-stamp regression OK: address recycling onto the cache-stamped "
     f"array's original address confirmed after {_vs_attempts} no-render replacement "
     f"attempts; colour VBO correctly rebuilt (no stale render)")

print("JUDGE GPU CHECKS PASSED")
