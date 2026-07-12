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
print("JUDGE DX12 CHECKS PASSED")
