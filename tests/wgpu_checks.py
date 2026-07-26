"""Judge checks for the wgpu backend: DX12 parity vs GL/CPU + correctness.

The directional (sun) shadow attenuation check (right after "dx12 sun disc
OK") renders a Sun entity WITH a tracer -- the sun disc check above it
never does, so it only exercises the sky disc, not `dl_shadow_tex`. Parity
vs GL is asserted exact (not toleranced): both backends index the same
`tracer.directional_shadow_factors` output at face granularity, with no
per-pixel resampling to introduce float divergence. Non-vacuous by
construction: forcing `shadow_depth` to 0 on the same scene/tracer must
make the wall's shadow disappear, or the exact-parity assertion would pass
trivially on a scene that doesn't actually shadow anything.

Sections 8-11 cover per-pixel texturing run 4/4 (wgpu/dx12), mirroring
gl_checks.py's sections 7-10 (its run 3/4 equivalent): a byte-identical-
with-main regression gate for untextured scenes (engine/wgpu_renderer.py's
UV vertex buffer + WGSL texture sampling must be a pure no-op when no
material directly binds a texture), real per-pixel variation within one
GPU-rasterized face (not vacuous) with a multiply-wrapped fallback control,
wgpu-vs-CPU textured parity with a numerically-justified tolerance PLUS a
discrimination check (the fallback control must blow past that tolerance),
and tiling reaching the WGSL sampler.
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
from engine.wgpu_renderer import WgpuRenderer

pygame.init()
pygame.display.set_mode((64, 64))
W, H = 400, 300


def build_scene():
    sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.4, -1, -0.3),
                                                    ambient=0.15))
    sc.add(engine.Entity("floor", mesh=engine.checkerboard(8, 1.5)))
    sc.add(engine.Entity("cube", mesh=engine.cube(1.0, color=(255, 30, 30)),
                         position=engine.Vec3(0, 0.5, 0)))
    sc.add(engine.Entity("lamp", light=engine.PointLight(
        intensity=2.0, range=15, radius=0.3, cast_shadows=False),
        position=engine.Vec3(2, 4, 2)))
    return sc


cam = engine.Camera(position=engine.Vec3(4, 3.2, 7), yaw=0.45, pitch=-0.3)
wr = WgpuRenderer("dx12")
scene = build_scene()

wr.render(scene, cam, (W, H))
img_dx = np.frombuffer(wr.read_frame(), dtype=np.uint8).reshape(H, W, 4)[..., :3].astype(float)
assert wr.stats["mode"] == "dx12", wr.stats["mode"]

gl = GLRenderer.standalone(W, H)
gl.render(scene, cam, (W, H))
raw = (gl.target if getattr(gl, "target", None) else gl.fbo).read(components=3)
img_gl = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)[::-1].astype(float)

r = Renderer()
r.render_scale = 1
surf = pygame.Surface((W, H))
r.render(surf, scene, cam)
img_cpu = pygame.surfarray.array3d(surf).transpose(1, 0, 2).astype(float)

d_gl = abs(img_dx.mean() - img_gl.mean())
d_cpu = abs(img_dx.mean() - img_cpu.mean())
assert d_gl < 6.0 and d_cpu < 12.0, (d_gl, d_cpu)
red = ((img_dx[..., 0] > 100) & (img_dx[..., 0] > 2 * img_dx[..., 1])).sum()
assert red > 500, red
print(f"parity OK: dx12={img_dx.mean():.1f} gl={img_gl.mean():.1f} "
      f"cpu={img_cpu.mean():.1f} | red cube px={red}")

# shadows: differential with a blocker + real tracer, plus sky sanity
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
lib = engine.AssetLibrary(os.path.join(REPO, "assets"))
sc.add(lib.instantiate("Sky Sphere"))
tracer = engine.ShadowTracer()
tracer.refresh(sc)
cam2 = engine.Camera(position=engine.Vec3(0.0, 6.5, 5.5), pitch=-0.9)
wr.render(sc, cam2, (W, H), tracer)
img_sh = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
wr.render(sc, cam2, (W, H), None)
img_no = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
delta = img_no - img_sh
assert delta.min() > -2.0 and img_sh.mean() < img_no.mean() - 1.5
assert (delta.mean(axis=-1) > 25).sum() > 300
cam3 = engine.Camera(position=engine.Vec3(0, 3, 8), pitch=0.5)  # look at sky
wr.render(sc, cam3, (W, H), None)
img_sky = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
top = img_sky[:H // 3]
assert top.mean() > 0.5 and top[..., 2].mean() >= top[..., 1].mean()  # bluish night sky
print(f"dx12 shadows OK ({(delta.mean(axis=-1) > 25).sum()} shadowed px), "
      f"HDRI sky OK (top mean {top.mean():.1f})")

# ---------------------------------------------------------------------
# sun disc: 3-way parity vs GL + tracks the Sun entity's rotation (dx12 only
# for the rotation half, since only dx12/GL both render a rasterized sky disc)
# ---------------------------------------------------------------------
import math


class _EngStub:
    def __init__(self, scene):
        self.scene = scene
        self.input = None


def make_sun_scene(yaw):
    s = engine.Scene(light=engine.DirectionalLight(engine.Vec3(0, -1, 0), ambient=0.1),
                     sky=((5, 8, 16), (20, 24, 40)))
    sun = lib.instantiate("Sun")
    sun.transform.rotation = engine.Vec3(-0.35, yaw, 0.0)
    s.add(sun)
    s.add(engine.Entity("floor", mesh=engine.checkerboard(8, 2.0)))
    return s


def bright_x(img):
    lum = img.mean(axis=-1)
    sky = lum[: H // 2]
    yx = np.unravel_index(np.argmax(sky), sky.shape)
    return yx[1], sky[yx]


cam_sun = engine.Camera(position=engine.Vec3(0, 2, 8), yaw=math.pi, pitch=0.15)
sun_sc = make_sun_scene(0.0)
sun_sc.update(1 / 60, _EngStub(sun_sc))
wr.render(sun_sc, cam_sun, (W, H))
img_sun_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
gl.render(sun_sc, cam_sun, (W, H))
img_sun_gl = np.frombuffer(gl.target.read(components=3), np.uint8).reshape(H, W, 3)[::-1].astype(float)
xa_dx, la_dx = bright_x(img_sun_dx)
xa_gl, la_gl = bright_x(img_sun_gl)
assert la_dx > 200, f"no bright sun disc in dx12 (peak {la_dx:.0f})"
assert abs(la_dx - la_gl) < 15 and abs(xa_dx - xa_gl) < 15, (la_dx, la_gl, xa_dx, xa_gl)

sun_sc2 = make_sun_scene(0.55)
sun_sc2.update(1 / 60, _EngStub(sun_sc2))
wr.render(sun_sc2, cam_sun, (W, H))
img_sun_dx2 = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
xb_dx, _ = bright_x(img_sun_dx2)
assert abs(xa_dx - xb_dx) > 25, f"dx12 sun disc did not track rotation ({xa_dx} -> {xb_dx})"
print(f"dx12 sun disc OK: peak {la_dx:.0f} (gl {la_gl:.0f}), moved x {xa_dx} -> {xb_dx}")

# ---------------------------------------------------------------------
# directional (sun) SHADOW ATTENUATION (dl_shadow_tex): the sun disc test
# above renders with NO tracer, so it exercises the sky disc only -- this
# is the first check that renders a Sun entity WITH a tracer, i.e. the
# actual dl_shadow_tex path added to WgpuRenderer. Floor + wall occluder +
# a low sun, same scene/camera used to independently measure the fix (a
# `git worktree` of main @29c5ab0 gave GL-vs-dx12 mean=1.628/4.16% px
# differing here, pre-fix; this branch gives byte-identical).
# ---------------------------------------------------------------------
sun_shadow_sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(0.6, -1.0, 0.0),
                                                            ambient=0.12))
sun_ent = lib.instantiate("Sun")
sun_ent.transform.rotation = engine.Vec3(-0.7, 1.6, 0.0)
sun_shadow_sc.add(sun_ent)
sun_floor = engine.Entity("floor", mesh=engine.checkerboard(16, 0.5))
sun_floor.casts_shadow = False
sun_shadow_sc.add(sun_floor)
sun_shadow_sc.add(engine.Entity("wall", mesh=engine.box(0.4, 2.5, 4.0),
                                position=engine.Vec3(0, 1.25, 0)))
sun_shadow_sc.update(1 / 60, _EngStub(sun_shadow_sc))  # sync scene.light.direction from the Sun entity
cam_sunshadow = engine.Camera(position=engine.Vec3(0, 4.0, 6.5), pitch=-0.55)
sun_tracer = engine.ShadowTracer()
sun_tracer.refresh(sun_shadow_sc)

wr.render(sun_shadow_sc, cam_sunshadow, (W, H), sun_tracer)
img_sunshadow_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
gl.render(sun_shadow_sc, cam_sunshadow, (W, H), sun_tracer)
img_sunshadow_gl = np.frombuffer(gl.target.read(components=3), np.uint8).reshape(H, W, 3)[::-1].astype(float)

# both backends index the SAME tracer.directional_shadow_factors output at
# face granularity (no per-pixel resampling, unlike textured sampling) --
# there is no float-divergence source, so parity is exact, not toleranced.
d_sunshadow = np.abs(img_sunshadow_dx - img_sunshadow_gl)
assert d_sunshadow.max() == 0.0, (
    f"dx12 sun-shadow render is not byte-identical to GL: max diff "
    f"{d_sunshadow.max():.1f}, mean {d_sunshadow.mean():.4f} -- dl_shadow_tex "
    f"regression")

# non-vacuous: shadow_depth only scales the tracer's raw factors inside
# WgpuRenderer.render()/GLRenderer.render() (the tracer itself is untouched
# by it, so no re-refresh is needed) -- forcing it to 0 must make the wall's
# shadow disappear. Without this, the byte-identical assertion above would
# pass trivially if a future change silently disabled sun shadows on BOTH
# backends at once.
sun_ent.sun.shadow_depth = 0.0
wr.render(sun_shadow_sc, cam_sunshadow, (W, H), sun_tracer)
img_noshadow_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
d_effect = np.abs(img_sunshadow_dx - img_noshadow_dx)
shadow_px = int((d_effect.max(axis=-1) > 5).sum())
assert d_effect.mean() > 1.0 and shadow_px > 3000, (
    f"sun-shadow test scene doesn't exercise a visible shadow: mean diff "
    f"{d_effect.mean():.3f}, {shadow_px} of {W * H} px changed (want > 3000)")
print(f"dx12 sun SHADOW ATTENUATION OK: byte-identical to GL (0 px differ), "
      f"shadow covers {shadow_px}/{W * H} px ({100 * shadow_px / (W * H):.1f}%) vs shadow_depth=0")

# ---------------------------------------------------------------------
# GI: red wall bleeds red light onto a neighboring white floor, 3-way parity
# ---------------------------------------------------------------------
gi_sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.3, -1, -0.2), ambient=0.05))
gi_sc.gi = {"enabled": True, "samples": 24, "intensity": 1.5}
gi_floor = engine.Entity("floor", mesh=engine.cube(1, color=(230, 230, 230)),
                         position=engine.Vec3(0, -0.55, 0), scale=engine.Vec3(3, 0.1, 3))
gi_wall = engine.Entity("wall", mesh=engine.cube(1, color=(255, 20, 20)),
                        position=engine.Vec3(-1.5, 0.5, 0), scale=engine.Vec3(0.1, 1, 3))
gi_sc.add(gi_floor)
gi_sc.add(gi_wall)
gi_tracer = engine.ShadowTracer()
gi_tracer.refresh(gi_sc)
cam_gi = engine.Camera(position=engine.Vec3(0.5, 1.5, 3), pitch=-0.3, yaw=-0.2)

wr.render(gi_sc, cam_gi, (W, H), gi_tracer)
img_gi_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
gl.render(gi_sc, cam_gi, (W, H), gi_tracer)
img_gi_gl = np.frombuffer(gl.target.read(components=3), np.uint8).reshape(H, W, 3)[::-1].astype(float)
gi_sc.gi = {"enabled": False}
wr.render(gi_sc, cam_gi, (W, H), gi_tracer)
img_gi_off = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
red_bleed_dx = (img_gi_dx[..., 0] - img_gi_dx[..., 1]) - (img_gi_off[..., 0] - img_gi_off[..., 1])
assert red_bleed_dx.max() > 3.0, f"expected red GI bounce on dx12 (max {red_bleed_dx.max():.1f})"
d_gi = abs(img_gi_dx.mean() - img_gi_gl.mean())
assert d_gi < 8.0, f"dx12/gl GI parity mismatch ({d_gi:.1f})"
print(f"dx12 GI OK: red bleed max {red_bleed_dx.max():.1f}, gl parity delta {d_gi:.1f}")

# ---------------------------------------------------------------------
# fog volumes: dense colored box tints a wall behind it, 3-way parity
# ---------------------------------------------------------------------
fog_sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.3, -1, -0.2), ambient=0.2))
fog_wall = engine.Entity("wall2", mesh=engine.cube(1, color=(255, 255, 255)),
                         position=engine.Vec3(0, 0, -6), scale=engine.Vec3(4, 4, 0.2))
fog_sc.add(fog_wall)
fog_vol_ent = lib.instantiate("Fog Volume")
fog_vol_ent.transform.position = engine.Vec3(0, 0, -3)
fog_vol_ent.transform.scale = engine.Vec3(2, 2, 2)
fog_vol_ent.fog_volume.density = 3.0
fog_vol_ent.fog_volume.color = (0, 255, 0)
fog_sc.add(fog_vol_ent)
cam_fog = engine.Camera(position=engine.Vec3(0, 0, 2), yaw=0, pitch=0)

wr.render(fog_sc, cam_fog, (W, H))
img_fog_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
gl.render(fog_sc, cam_fog, (W, H))
img_fog_gl = np.frombuffer(gl.target.read(components=3), np.uint8).reshape(H, W, 3)[::-1].astype(float)
fog_sc.entities.remove(fog_vol_ent)
wr.render(fog_sc, cam_fog, (W, H))
img_nofog_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
c = slice(H // 2 - 20, H // 2 + 20), slice(W // 2 - 20, W // 2 + 20)
g_dx, g_dx0, g_gl = img_fog_dx[c][..., 1].mean(), img_nofog_dx[c][..., 1].mean(), img_fog_gl[c][..., 1].mean()
assert g_dx > g_dx0 + 20, f"expected green fog volume tint on dx12 ({g_dx:.1f} vs {g_dx0:.1f})"
assert abs(g_dx - g_gl) < 15, f"dx12/gl fog volume parity mismatch ({g_dx:.1f} vs {g_gl:.1f})"
print(f"dx12 fog volume OK: center green {g_dx0:.1f} -> {g_dx:.1f} (gl {g_gl:.1f})")

# ---------------------------------------------------------------------
# PBR: metallic/roughness/emissive 3-way parity (cpu/gl/dx12), dx12
# specular highlight, dx12 emissive-in-the-dark -- mirrors gl_checks.py's
# PBR cases 6a/6b/6c exactly, same scene geometry/camera so the dx12
# numbers are directly comparable to the already-judged GL ones.
# ---------------------------------------------------------------------
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

# 7a. CPU-vs-GL-vs-dx12 mean-brightness parity on a metallic+rough-varying
# PBR scene.
sc_pbr = build_pbr_scene(box_roughness=0.15, box_metallic=1.0)
wr.render(sc_pbr, pbr_cam, (W, H))
img_pbr_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
gl.render(sc_pbr, pbr_cam, (W, H))
img_pbr_gl = np.frombuffer(gl.target.read(components=3), np.uint8).reshape(H, W, 3)[::-1].astype(float)
r3 = Renderer()
r3.render_scale = 1
surf3 = pygame.Surface((W, H))
r3.render(surf3, sc_pbr, pbr_cam)
img_pbr_cpu = pygame.surfarray.array3d(surf3).transpose(1, 0, 2).astype(float)
d_pbr_gl = abs(img_pbr_dx.mean() - img_pbr_gl.mean())
d_pbr_cpu = abs(img_pbr_dx.mean() - img_pbr_cpu.mean())
assert d_pbr_gl < 12.0, f"dx12/gl PBR mean brightness diverges: {d_pbr_gl:.1f}"
assert d_pbr_cpu < 15.0, f"dx12/cpu PBR mean brightness diverges: {d_pbr_cpu:.1f}"
print(f"dx12 pbr parity OK: dx12={img_pbr_dx.mean():.1f} gl={img_pbr_gl.mean():.1f} "
      f"cpu={img_pbr_cpu.mean():.1f} (diffs {d_pbr_gl:.1f}/{d_pbr_cpu:.1f})")

# 7b. dx12 specular highlight: metallic+shiny box shows very-bright pixels
# that the same scene at default params (spec_scale gated to 0) never reaches.
sc_default = build_pbr_scene()  # defaults: legacy diffuse-only look
wr.render(sc_default, pbr_cam, (W, H))
img_default_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
bright_pbr_dx = int((img_pbr_dx.max(axis=-1) > 240).sum())
bright_default_dx = int((img_default_dx.max(axis=-1) > 240).sum())
assert bright_pbr_dx > bright_default_dx, (bright_pbr_dx, bright_default_dx)
assert bright_default_dx == 0, ("default-param scene unexpectedly saturated on dx12 -- "
                                "test scene isn't isolating the highlight", bright_default_dx)
print(f"dx12 highlight OK: bright pixels pbr={bright_pbr_dx} default={bright_default_dx}")

# 7c. dx12 emissive-in-the-dark: emissive face visible with all lights off
def build_pbr_dark_scene(emissive):
    sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.4, -1.0, -0.3),
                                                    ambient=0.0, color=(0, 0, 0),
                                                    intensity=0.0),
                      background=(0, 0, 0))
    box_mesh = engine.cube(size=1.6, color=(10, 10, 10))
    box_mesh.face_emissive[:] = emissive
    sc.add(engine.Entity("box", mesh=box_mesh, position=engine.Vec3(0, 0, 0)))
    return sc


wr.render(build_pbr_dark_scene((0.0, 0.0, 0.0)), pbr_cam, (W, H))
img_dark_off_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
wr.render(build_pbr_dark_scene((220.0, 40.0, 40.0)), pbr_cam, (W, H))
img_dark_emis_dx = np.frombuffer(wr.read_frame(), np.uint8).reshape(H, W, 4)[..., :3].astype(float)
assert img_dark_off_dx.max() <= 2, f"expected near-black with no lights/emissive: {img_dark_off_dx.max()}"
assert img_dark_emis_dx.max() > 100, f"emissive face not bright in the dark on dx12: {img_dark_emis_dx.max()}"
print(f"dx12 emissive-in-dark OK: off max={img_dark_off_dx.max():.0f} "
      f"emissive max={img_dark_emis_dx.max():.0f}")

# ---------------------------------------------------------------------
# 8. UNTEXTURED REGRESSION GATE: this run's shader/geometry changes (UV
#    vertex buffer in `_get_geo_cache`, bind group 2 + tex_meta uniforms,
#    the base_color/roughness_v/metallic_v/emissive_v override locals in
#    fs_main) must not move a single pixel on a scene with no direct
#    texture bindings. `perpixel_wgpu_untextured_golden.npy` was captured
#    from a `git worktree add ... main` at e298d80 (this run's branch
#    point), rendering the SAME `build_scene()`/`cam` as the parity check
#    at the top of this file through `WgpuRenderer("dx12")` at the same
#    400x300 -- confirmed reproducible (rendered twice in the SAME worktree
#    process, 0 diff, AND re-rendered in a SEPARATE fresh process against
#    the saved .npy, 0 diff -- not just "captured carefully") before being
#    treated as golden, same discipline as texture_checks.py's CPU golden
#    and gl_checks.py's GL golden.
# ---------------------------------------------------------------------
_golden_path = os.path.join(REPO, "tests", "fixtures", "perpixel_wgpu_untextured_golden.npy")
assert os.path.exists(_golden_path), f"no golden at {_golden_path}"
_golden = np.load(_golden_path).astype(np.int32)
_wcurrent = img_dx.astype(np.int32)  # same scene/cam/resolution as the top-of-file parity check
_wdiff = np.abs(_wcurrent - _golden)
assert _wdiff.max() == 0 and _wdiff.mean() == 0.0, (
    f"UNTEXTURED REGRESSION: dx12 render differs from main by up to {_wdiff.max()} "
    f"(mean {_wdiff.mean()}) on a scene with no direct texture bindings")
print(f"8. untextured regression gate OK: 0 of {_golden.shape[0] * _golden.shape[1]} pixels "
     "differ vs a main (e298d80) dx12 render of the identical scene")

# ---------------------------------------------------------------------
# Per-pixel texturing (run 4/4): same quad-scene model as gl_checks.py's
# sections 8-10 (its own run 3/4 equivalent) and texture_checks.py's #17,
# extended with a wgpu-vs-CPU pixel comparison, a discrimination check, and
# a tiling check. See engine/wgpu_renderer.py's `_get_material_tex_view` /
# `_tex_meta_and_views` / `tex_texel()`+sample_port_vec3/scalar in _MESH_WGSL.
# ---------------------------------------------------------------------
_tex_tmp = os.path.join(TMP, "judge_perpixel_run4_assets")
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


def _wgpu_quad_frame(scene9):
    wr.render(scene9, _quad_cam, (_QW, _QH))
    return np.frombuffer(wr.read_frame(), np.uint8).reshape(_QH, _QW, 4)[..., :3]


_bg9 = np.asarray(engine.Scene().background)
_img_dx_direct = _wgpu_quad_frame(_build_quad_scene(True)).astype(np.int32)
_mask9 = ~np.all(_img_dx_direct == _bg9[None, None, :], axis=2)
_distinct_dx_direct = len(np.unique(_img_dx_direct[_mask9].reshape(-1, 3), axis=0))

_img_dx_fallback = _wgpu_quad_frame(_build_quad_scene(False)).astype(np.int32)
_mask9b = ~np.all(_img_dx_fallback == _bg9[None, None, :], axis=2)
_distinct_dx_fallback = len(np.unique(_img_dx_fallback[_mask9b].reshape(-1, 3), axis=0))

assert _mask9.sum() > 1000, "quad should cover a substantial part of the frame"
assert _distinct_dx_direct > 50, f"expected many distinct colors from wgpu per-pixel sampling, got {_distinct_dx_direct}"
assert _distinct_dx_fallback == 1, f"expected exactly 1 color from the per-face bake control, got {_distinct_dx_fallback}"
print(f"9. dx12 per-pixel variation OK (not vacuous): direct-bind base_color shows "
     f"{_distinct_dx_direct} distinct colors across {int(_mask9.sum())} pixels of ONE "
     f"face on the GPU; multiply-wrapped fallback (same graph, same face) gives exactly "
     f"{_distinct_dx_fallback} -- confirms the WGSL tex_texel() sampler, not lighting, is "
     "the source of variation")

# 10. dx12-vs-CPU textured parity + discrimination. render_scale MUST be
#     forced to 1 for a fair full-resolution comparison -- the default (3)
#     renders the CPU path at 1/3 internal resolution then upscales, which
#     alone produces a much larger, meaningless diff at texel-cell
#     boundaries (see gl_checks.py's identical note; measured here too:
#     forcing scale=1 is what keeps these numbers meaningful).
_r10 = Renderer()
_r10.render_scale = 1
_surf_cpu10 = pygame.Surface((_QW, _QH))
_r10.render(_surf_cpu10, _build_quad_scene(True), _quad_cam)
_img_cpu_direct = pygame.surfarray.array3d(_surf_cpu10).transpose(1, 0, 2).astype(np.int32)
_pdiff = np.abs(_img_dx_direct - _img_cpu_direct)
_pndiff = int((_pdiff.sum(axis=2) > 0).sum())
_pfrac = _pndiff / (_QW * _QH)
# Tolerance, justified by measurement (this exact scene, numbers found
# during implementation): mean diff 0.266, max diff 212, 173/30000 (0.58%)
# differing pixels -- essentially identical to gl_checks.py's own run-3
# numbers for the same scene/texture. The nonzero diffs are NEAREST-
# neighbor texel-boundary aliasing: dx12's hardware-rasterized, per-vertex-
# interpolated UV and the CPU's numpy barycentric world-space math are two
# independently-implemented, both-correct interpolants that can disagree by
# one texel index right at a shared cell edge; this test texture is a
# deliberately maximal-contrast 8x8 gradient so a single-texel edge miss
# shows up as a big color jump. Bounding at mean<3.0 / differing-fraction<3%
# is >10x the measured 0.266 / 0.58% -- tight enough that a real sampling
# bug (wrong channel, wrong UV, missing tiling, wrong V orientation -- any
# of which would desync entire regions, not just edge pixels) would blow
# through it, loose enough not to be a coin-flip on rasterizer rounding.
assert _pdiff.mean() < 3.0, f"dx12/CPU textured mean diff too high: {_pdiff.mean():.3f}"
assert _pfrac < 0.03, f"dx12/CPU textured differing-pixel fraction too high: {_pfrac:.4f}"
print(f"10. textured parity OK: mean diff={_pdiff.mean():.3f} (tolerance <3.0), "
     f"max diff={_pdiff.max()}, {_pndiff}/{_QW * _QH} differing px "
     f"({100 * _pfrac:.2f}%, tolerance <3%) -- diffs are nearest-neighbor "
     "texel-boundary aliasing between two independent rasterizers, not a "
     "systematic bug (both rendered at render_scale=1 -- see comment above)")

# 10b. DISCRIMINATION CHECK: the gate above is only meaningful if a real
# sampling bug would actually blow through it. Compare the wgpu FALLBACK
# render (multiply-wrapped tex_sample -> falls back to the per-face bake,
# same failure mode as a broken direct-binding implementation) against the
# CPU's correct direct-bind reference -- this simulates exactly what
# tolerance #10 exists to catch.
_disc_diff = np.abs(_img_dx_fallback - _img_cpu_direct)
_disc_ndiff = int((_disc_diff.sum(axis=2) > 0).sum())
_disc_frac = _disc_ndiff / (_QW * _QH)
assert _disc_diff.mean() > 3.0 and _disc_frac > 0.03, (
    f"discrimination check FAILED: a broken (flat per-face) render only differs by "
    f"mean={_disc_diff.mean():.3f} frac={_disc_frac:.4f} -- tolerance #10 would not catch it")
print(f"10b. discrimination check OK: a flat per-face fallback (simulating a broken "
     f"per-pixel implementation) diverges from the CPU reference by mean="
     f"{_disc_diff.mean():.3f} ({100*_disc_frac:.2f}% differing px) -- both well past the "
     "mean<3.0/3% gate above, confirming it isn't a rubber stamp")

# 11. tiling reaches the WGSL sampler: u_tiling/v_tiling actually change the
#     dx12 render (still samples the same 8x8 palette -- repeats, doesn't
#     distort or go out of range -- while visibly differing from untiled).
_img_dx_tiled = _wgpu_quad_frame(_build_quad_scene(True, u_tiling=3.0, v_tiling=3.0)).astype(np.int32)
_mask9c = ~np.all(_img_dx_tiled == _bg9[None, None, :], axis=2)
_distinct_dx_tiled = len(np.unique(_img_dx_tiled[_mask9c].reshape(-1, 3), axis=0))
assert _distinct_dx_tiled > 50, f"tiled dx12 render should still sample the full palette, got {_distinct_dx_tiled}"
assert not np.array_equal(_img_dx_tiled, _img_dx_direct), "tiling=3 must render differently than tiling=1 on dx12"
print(f"11. dx12 tiling OK: {_distinct_dx_tiled} distinct colors (still the full palette, "
     "repeated), differs from the untiled dx12 render")

texture_mod.clear_cache()
texture_mod.set_texture_root(os.path.join(REPO, "assets"))
import shutil as _shutil10
_shutil10.rmtree(_tex_tmp, ignore_errors=True)

# 12. CACHE IDENTITY REGRESSION: `_get_geo_cache`/`_get_entity_uniforms` key
# off `mesh._cache_id`/`entity._cache_id` (monotonic per-instance serials,
# see engine/mesh.py and engine/scene.py), not `id(mesh)`/`id(entity)`. The
# `_geo_cache`/`_entity_uniform_cache` dicts hold no strong reference to what
# they cache (only derived GPU buffers/ids), so once a scene is dropped and
# gc'd, CPython is free to reuse a dead Mesh/Entity's address for a
# brand-new, unrelated one. An id()-keyed cache would then return the OLD
# entry on lookup -- IndexError if face counts differ (covered live by
# repro_cache.py during development, ~1 iteration on this backend), or,
# worse, SILENTLY WRONG geometry if they coincide, since a cache HIT never
# rebuilds the geometry buffer (only color/pbr/opacity get a per-hit
# staleness check). This mirrors gl_checks.py's check #11, reproducing
# exactly the silent case: two 6-quad cubes of the SAME topology but wildly
# different SIZE, so a false hit renders the WRONG (stale) size with no
# exception at all -- the case the bug report says matters more than the
# crash. Verified against pre-fix code (a `git worktree` of main @ c46d706)
# during implementation: this exact loop measured min-large-count ==
# max-small-count (a "small" iteration rendering fully as the LARGE cube's
# stale geometry), so it is a real regression gate, not a vacuous one.
import gc as _gc12

_CR_W, _CR_H = 160, 120
_cr_cam = engine.Camera(position=engine.Vec3(0, 0, 6), yaw=0.0, pitch=0.0)


def _cr_scene(size):
    sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.3, -1, -0.2), ambient=0.6))
    sc.add(engine.Entity("cube", mesh=engine.cube(size, color=(255, 40, 40))))
    return sc


def _cr_frame(scene):
    wr.render(scene, _cr_cam, (_CR_W, _CR_H))
    return np.frombuffer(wr.read_frame(), np.uint8).reshape(_CR_H, _CR_W, 4)[..., :3]


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
    _gc12.collect()

assert _cr_recycled, ("address recycling never occurred in 40 iterations -- "
                      "this run cannot validate the fix; re-run or raise the iteration count")
_cr_min_large, _cr_max_small = min(_cr_large), max(_cr_small)
assert _cr_min_large > _cr_max_small * 3, (
    f"CACHE IDENTITY REGRESSION: a false cache hit rendered a stale (wrong-size) mesh -- "
    f"min large-cube red px={_cr_min_large}, max small-cube red px={_cr_max_small} "
    "(expected the large cube to always dominate)")
print(f"12. cache identity regression OK: address recycling confirmed over 40 iterations; "
     f"small-cube red px range [{min(_cr_small)}, {max(_cr_small)}], large-cube range "
     f"[{min(_cr_large)}, {max(_cr_large)}] -- no false cache hit smeared sizes together")

print("JUDGE DX12 CHECKS PASSED")
