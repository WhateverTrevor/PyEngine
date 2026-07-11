"""PyEngine demo: a lit, fogged scene you can fly through in real time.

    py demo.py                          interactive (GPU if available, else CPU)
    py demo.py --frames 300             benchmark 300 frames, print avg FPS
    py demo.py --frames 60 --headless --screenshot out.png
    py demo.py --api dx12               force a backend: cpu / gl / dx12 / vulkan
                                         (--gpu/--cpu still work as gl/cpu aliases)
"""
from __future__ import annotations

import argparse
import colorsys
import math
import os


def build_scene(scene_mod):
    e = scene_mod  # the imported engine package
    scene = e.Scene(
        light=e.DirectionalLight(e.Vec3(-0.55, -1.0, -0.35), ambient=0.32),
        sky=((18, 26, 52), (116, 135, 170)),
        fog=e.Fog(color=(116, 135, 170), start=40.0, end=130.0),
    )

    scene.add(e.Entity("ground", mesh=e.checkerboard(squares=24, square_size=2.5)))

    # Ring of spinning, bobbing cubes around the center, hue-swept.
    count = 8
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        r, g, b = colorsys.hsv_to_rgb(i / count, 0.65, 0.95)
        color = (int(r * 255), int(g * 255), int(b * 255))
        scene.add(
            e.Entity(f"cube{i}", mesh=e.cube(2.0, color=color),
                     position=e.Vec3(math.cos(angle) * 10.0, 1.6, math.sin(angle) * 10.0))
            .add_behavior(e.behaviors.Spin(e.Vec3(0.9, 1.3, 0.0)))
            .add_behavior(e.behaviors.Bob(amplitude=0.45, speed=1.4, phase=angle * 2.0))
        )

    scene.add(
        e.Entity("sphere", mesh=e.icosphere(2.2, subdivisions=2, color=(70, 200, 140)),
                 position=e.Vec3(0.0, 3.2, 0.0))
        .add_behavior(e.behaviors.Spin(e.Vec3(0.0, 0.6, 0.0)))
        .add_behavior(e.behaviors.Bob(amplitude=0.5, speed=0.9))
    )

    scene.add(
        e.Entity("halo", mesh=e.torus(4.6, 0.42, ring_segments=28, tube_segments=14,
                                      color=(235, 160, 70)),
                 position=e.Vec3(0.0, 3.2, 0.0), rotation=e.Vec3(0.35, 0.0, 0.15))
        .add_behavior(e.behaviors.Spin(e.Vec3(0.0, 0.8, 0.0)))
    )

    scene.add(
        e.Entity("moon", mesh=e.icosphere(0.7, subdivisions=1, color=(200, 205, 230)),
                 position=e.Vec3(7.5, 3.2, 0.0))
        .add_behavior(e.behaviors.Orbit(e.Vec3(0.0, 0.0, 0.0), radius=7.5, speed=0.8))
        .add_behavior(e.behaviors.Bob(amplitude=1.2, speed=0.8))
    )

    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description="PyEngine demo scene")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=None,
                        help="run exactly N frames then exit (benchmark mode)")
    parser.add_argument("--screenshot", default=None, help="save last frame to this path")
    parser.add_argument("--headless", action="store_true",
                        help="render without a window (SDL dummy driver)")
    parser.add_argument("--api", choices=["auto", "cpu", "gl", "dx12", "vulkan"], default=None,
                        help="force a rendering backend (default: auto)")
    parser.add_argument("--gpu", action="store_true",
                        help="alias for --api gl (force the OpenGL/moderngl renderer)")
    parser.add_argument("--cpu", action="store_true",
                        help="alias for --api cpu (force the software renderer)")
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    api_mode = "auto"
    if args.api:
        api_mode = args.api
    elif args.gpu:
        api_mode = "gl"
    elif args.cpu:
        api_mode = "cpu"
    if args.headless:
        api_mode = "cpu"  # the SDL dummy driver has no GL surface / wgpu window to attach to

    import engine  # after the SDL driver decision, since importing initializes pygame

    eng = engine.Engine(args.width, args.height, title="PyEngine Demo", api=api_mode)
    eng.hud_text = ("WASD move | Q/E or Space/Ctrl up/down | Shift fast | "
                    "hold LMB/RMB = mouse look | F1 wireframe | H hud | Esc quit")

    eng.loading_step("building scene", 0.5)
    scene = build_scene(engine)
    eng.loading_step("opening world", 0.9)
    camera = engine.Camera(position=engine.Vec3(0.0, 4.5, 19.0), pitch=-0.1, fov=70.0)
    scene.add(engine.Entity("player").add_behavior(
        engine.behaviors.FlyController(camera)))

    eng.run(scene, camera, max_frames=args.frames, screenshot_path=args.screenshot)


if __name__ == "__main__":
    main()
