"""Judge checks for the GPU renderer: parity, IES, cone, shadows, depth."""
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
print("JUDGE GPU CHECKS PASSED")
