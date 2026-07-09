"""Engine core: window, fixed-timestep game loop, HUD."""
from __future__ import annotations

import time

import pygame

from .input import InputManager
from .raytrace import ShadowTracer
from .renderer import Renderer


class Engine:
    def __init__(self, width: int = 1280, height: int = 720, title: str = "PyEngine",
                 max_fps: int = 120, fixed_dt: float = 1.0 / 60.0):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.input = InputManager()
        self.renderer = Renderer()
        self.tracer = ShadowTracer()
        self.max_fps = max_fps
        self.fixed_dt = fixed_dt
        self.show_hud = True
        self.hud_text: str | None = None  # optional controls line, set by the app
        self._font = pygame.font.SysFont("consolas,couriernew,monospace", 15)
        self._hud_cache: dict[str, pygame.Surface] = {}

    def run(self, scene, camera, max_frames: int | None = None,
            screenshot_path: str | None = None, overlay=None) -> None:
        """Run the game loop until quit.

        Updates run on a fixed timestep (deterministic behaviors); rendering
        runs as fast as the frame allows, capped at max_fps. `overlay(engine)`
        is called after the 3D render, before flip — used by the editor UI.
        `max_frames` + `screenshot_path` support benchmarking.
        """
        clock = pygame.time.Clock()
        last = time.perf_counter()
        start_time = last
        accumulator = 0.0
        frames = 0
        running = True

        while running:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
            self.input.process(events)

            now = time.perf_counter()
            frame_dt = min(now - last, 0.25)
            last = now
            if max_frames is not None:
                frame_dt = self.fixed_dt  # deterministic benchmark runs
            accumulator += frame_dt
            steps = 0
            while accumulator >= self.fixed_dt:
                scene.update(self.fixed_dt, self)
                accumulator -= self.fixed_dt
                steps += 1
            if steps:
                # engine hotkeys consume edges alongside behaviors
                if self.input.pressed(pygame.K_F1):
                    self.renderer.wireframe = not self.renderer.wireframe
                if self.input.pressed(pygame.K_h):
                    self.show_hud = not self.show_hud
                self.input.consume_edges()

            tracer = None
            if scene.enable_shadows:
                self.tracer.refresh(scene)
                tracer = self.tracer
            self.renderer.render(self.screen, scene, camera, tracer)
            if self.show_hud:
                self._draw_hud(clock.get_fps(), scene)
            if overlay is not None:
                overlay(self)
            pygame.display.flip()
            clock.tick(self.max_fps)

            frames += 1
            if max_frames is not None and frames >= max_frames:
                if screenshot_path:
                    pygame.image.save(self.screen, screenshot_path)
                running = False

        elapsed = time.perf_counter() - start_time
        if max_frames is not None and elapsed > 0:
            print(f"{frames} frames in {elapsed:.2f}s -> {frames / elapsed:.1f} FPS avg "
                  f"({self.renderer.stats['triangles']} triangles in final frame)")
        pygame.quit()

    def _draw_hud(self, fps: float, scene) -> None:
        stats = self.renderer.stats
        lines = [f"{fps:5.1f} FPS | {stats['triangles']} tris | "
                 f"{stats['shadow_lights']} shadow lights | {len(scene.entities)} entities"]
        if self.hud_text:
            lines.append(self.hud_text)
        y = 8
        for text in lines:
            surf = self._hud_cache.get(text)
            if surf is None:
                if len(self._hud_cache) > 64:
                    self._hud_cache.clear()
                shadow = self._font.render(text, True, (0, 0, 0))
                front = self._font.render(text, True, (235, 235, 235))
                surf = pygame.Surface((front.get_width() + 1, front.get_height() + 1),
                                      pygame.SRCALPHA)
                surf.blit(shadow, (1, 1))
                surf.blit(front, (0, 0))
                self._hud_cache[text] = surf
            self.screen.blit(surf, (10, y))
            y += surf.get_height() + 2
