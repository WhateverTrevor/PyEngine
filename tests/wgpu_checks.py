"""Judge checks for the wgpu backend: DX12 parity vs GL/CPU + correctness."""
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

print("JUDGE DX12 CHECKS PASSED")
