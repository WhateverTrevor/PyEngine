"""Engine core: window, splash/loading screen, fixed-timestep game loop, HUD.

Rendering backend: software (numpy, always available), OpenGL 3.3 core via
moderngl ("gl", optional), or DirectX 12 / Vulkan via wgpu-py ("dx12" /
"vulkan", optional) -- see `api` on `Engine.__init__`. GL mode keeps
`self.screen` alive as an SRCALPHA pygame Surface used purely as a UI
overlay canvas -- all existing HUD/editor drawing code targets it unchanged
-- and composites it over the 3D frame each tick as an alpha-blended
fullscreen quad. wgpu mode is simpler: `WgpuRenderer` renders offscreen and
this file reads the frame back and blits it straight into `self.screen` (a
normal, non-OPENGL window surface), so HUD/editor drawing works exactly as
in software mode -- no compositing pass. See gl_renderer.py / wgpu_renderer.py
for the GPU pipelines themselves; this file only owns window/context
lifecycle, the api fallback chain, and the per-frame composite/blit.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pygame

from .input import InputManager
from .lod import update_scene_lods
from .raytrace import ShadowTracer
from .renderer import Renderer

_SPLASH_SIZE = (460, 260)

_UI_VS = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_UI_FS = """
#version 330 core
in vec2 v_uv;
uniform sampler2D tex;
out vec4 fragColor;
void main() { fragColor = texture(tex, v_uv); }
"""


class Engine:
    def __init__(self, width: int = 1280, height: int = 720, title: str = "PyEngine",
                 max_fps: int = 120, fixed_dt: float = 1.0 / 60.0, splash: bool = True,
                 api: str = "dx12", gpu: "str | bool | None" = None,
                 fullscreen: bool = False):
        """`api`: "dx12" (WebGPU via wgpu-py, see wgpu_renderer.py; the
        default -- CPU rendering is opt-in only), "vulkan" (WebGPU, other
        adapter), "auto" (alias for "dx12"), "gl" (OpenGL 3.3 via moderngl),
        "cpu" (software only). Headless/dummy driver always forces "cpu"
        regardless of `api`. Each GPU choice falls back one step at a time,
        dx12/vulkan -> "gl" -> "cpu", printing one warning line per step on
        failure -- see `_init_display`.

        `gpu` is a DEPRECATED alias for `api`, kept for old call sites:
        True -> "auto", False -> "cpu", any string -> passed through as-is.
        It takes precedence over `api` when given (so old `gpu=True/False`
        callers don't need to change anything).

        GPU window/context creation is deferred to `_end_splash()` (or
        immediately if `splash=False`) since the splash screen is a plain
        software surface."""
        if gpu is not None:
            api = "auto" if gpu is True else "cpu" if gpu is False else gpu
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        self._size = (width, height)
        self._windowed_size = (width, height)  # remembered across a fullscreen toggle
        self.fullscreen = False
        self._fullscreen_pref = fullscreen  # applied once the real window opens
        pygame.display.set_caption(title)
        self.input = InputManager()
        self.renderer = Renderer()
        self.gl_renderer = None          # GLRenderer, or None unless api ended up "gl"
        self.wgpu_renderer = None        # WgpuRenderer, or None unless api ended up dx12/vulkan
        self._api_mode = api
        self._gl_ctx = None
        self._ui_prog = self._ui_vbo = self._ui_vao = self._ui_tex = None
        self.tracer = ShadowTracer()
        self.max_fps = max_fps
        self.fixed_dt = fixed_dt
        self.show_hud = True
        self.hud_text: str | None = None  # optional controls line, set by the app
        self._font = pygame.font.SysFont("consolas,couriernew,monospace", 15)
        self._small_font = pygame.font.SysFont("consolas,couriernew,monospace", 12)
        self._title_font = pygame.font.SysFont("consolas,couriernew,monospace", 32, bold=True)
        self._hud_cache: dict[str, pygame.Surface] = {}
        self.esc_handler = None  # callable returning True to consume an Escape press
        self._splash_active = False
        if splash:
            self.screen = pygame.display.set_mode(_SPLASH_SIZE, pygame.NOFRAME)
            self._splash_active = True
            self.loading_step("starting engine", 0.05)
        else:
            self._init_display()

    def loading_step(self, message: str, progress: float) -> None:
        """Advance the startup splash: corner status text + progress bar.

        Call between loading phases (assets, thumbnails, scene, shadows).
        Becomes a no-op once the main window is open.
        """
        if not self._splash_active:
            return
        pygame.event.pump()  # keep the window responsive during long steps
        s = self.screen
        w, h = s.get_size()
        s.fill((16, 18, 23))
        pygame.draw.rect(s, (58, 62, 72), (0, 0, w, h), 1)

        title = self._title_font.render("PyEngine", True, (235, 235, 240))
        s.blit(title, ((w - title.get_width()) // 2, 72))
        sub = self._small_font.render("pure-python real-time 3d", True, (120, 124, 134))
        s.blit(sub, ((w - sub.get_width()) // 2, 76 + title.get_height()))

        bar_x, bar_w, bar_h = 24, w - 48, 6
        bar_y = h - 46
        pygame.draw.rect(s, (40, 43, 51), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        fill = int(bar_w * min(max(progress, 0.0), 1.0))
        if fill > 0:
            pygame.draw.rect(s, (255, 170, 60), (bar_x, bar_y, fill, bar_h),
                             border_radius=3)
        status = self._small_font.render(message, True, (150, 154, 163))
        s.blit(status, (bar_x, bar_y + 13))
        pygame.display.flip()

    def _end_splash(self) -> None:
        if self._splash_active:
            self.loading_step("ready", 1.0)
            self._splash_active = False
            self._init_display()

    # ------------------------------------------------------------------
    # display / GPU context lifecycle
    # ------------------------------------------------------------------
    def _gl_window_attribs(self) -> None:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

    def _init_display(self) -> None:
        """Open the real window per `self._api_mode`, falling back one step
        at a time -- requested api -> "gl" -> "cpu" -- printing one warning
        line per failed step. Headless/dummy-driver runs always force "cpu"
        (no GL surface or wgpu adapter is expected to make sense there)."""
        headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
        want = "cpu" if headless else self._api_mode
        if want == "auto":
            want = "dx12"

        if want in ("dx12", "vulkan"):
            try:
                self._create_wgpu_window(want)
                return
            except Exception as ex:
                print(f"{want} renderer unavailable ({ex!r}); "
                      f"falling back to the OpenGL renderer.")
                self.wgpu_renderer = None
                want = "gl"

        if want == "gl":
            try:
                self._create_gpu_window()
                self._apply_fullscreen_pref()
                return
            except Exception as ex:
                print(f"GPU renderer unavailable ({ex!r}); "
                      f"falling back to the software renderer.")
                self.gl_renderer = None

        self.screen = pygame.display.set_mode(self._size, pygame.RESIZABLE)
        self._apply_fullscreen_pref()

    def _apply_fullscreen_pref(self) -> None:
        """Applied once after the real window opens (fullscreen= at __init__,
        or a saved settings.json preference the app chooses to honor)."""
        if self._fullscreen_pref and not self.fullscreen:
            self.set_fullscreen(True)

    def _create_wgpu_window(self, backend: str) -> None:
        from .wgpu_renderer import WgpuRenderer

        self.wgpu_renderer = WgpuRenderer(backend)  # raises -> caller falls back to gl/cpu
        self.screen = pygame.display.set_mode(self._size, pygame.RESIZABLE)  # plain window
        self._apply_fullscreen_pref()

    def _create_gpu_window(self) -> None:
        import moderngl

        from .gl_renderer import GLRenderer

        self._gl_window_attribs()
        pygame.display.set_mode(self._size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        ctx = moderngl.create_context()
        ctx.viewport = (0, 0, self._size[0], self._size[1])
        self._gl_ctx = ctx
        self.gl_renderer = GLRenderer(ctx)
        self.screen = pygame.Surface(self._size, pygame.SRCALPHA)  # UI overlay canvas
        self._build_ui_pipeline()
        self._resize_ui_texture()

    def _build_ui_pipeline(self) -> None:
        """Compile the fullscreen-quad shader that composites the UI overlay
        (the `self.screen` pygame Surface) over the 3D frame. Built once;
        only the texture (sized to the window) is rebuilt on resize."""
        ctx = self._gl_ctx
        self._ui_prog = ctx.program(vertex_shader=_UI_VS, fragment_shader=_UI_FS)
        # (pos.xy, uv) — v=0 at the top row to match pygame's top-down surface
        # bytes, v=1 at the bottom, so NDC top (+1) samples uv.y=0.
        verts = np.array([-1, -1, 0, 1, 1, -1, 1, 1, 1, 1, 1, 0,
                          -1, -1, 0, 1, 1, 1, 1, 0, -1, 1, 0, 0], dtype=np.float32)
        self._ui_vbo = ctx.buffer(verts.tobytes())
        self._ui_vao = ctx.vertex_array(
            self._ui_prog, [(self._ui_vbo, "2f 2f", "in_pos", "in_uv")])

    def _resize_ui_texture(self) -> None:
        import moderngl
        if self._ui_tex is not None:
            self._ui_tex.release()
        self._ui_tex = self._gl_ctx.texture(self._size, 4)
        self._ui_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _composite_gpu_overlay(self) -> None:
        """Upload `self.screen` (UI overlay) and alpha-blend it over the 3D
        frame already drawn into ctx.screen by gl_renderer.render()."""
        import moderngl
        ctx = self._gl_ctx
        ctx.screen.use()
        ctx.viewport = (0, 0, self._size[0], self._size[1])
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._ui_tex.write(pygame.image.tobytes(self.screen, "RGBA"))
        self._ui_tex.use(location=0)
        self._ui_prog["tex"].value = 0
        self._ui_vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)

    def _rebuild_window(self, size: tuple[int, int], fullscreen: bool) -> None:
        """Re-open the window at `size`, in or out of fullscreen.

        Shared by `set_resolution`, `set_fullscreen`, and the VIDEORESIZE
        handler in `run()` so all three window-size paths behave the same
        way. The renderer's deferred-pass cache and HUD text cache key off
        the current screen size already, so they self-invalidate next frame
        -- no further bookkeeping needed here. In GL mode the window is
        re-created with the GL attributes/flags and the UI overlay surface
        + texture are rebuilt at the new size; the GL viewport is set
        explicitly afterward since SDL does not update it on a resize (nor,
        empirically, does moderngl.Context.screen's cached size). In wgpu
        mode (and software mode) the window is just a plain surface, so
        re-creating it is enough -- `WgpuRenderer._ensure_size` rebuilds its
        size-dependent offscreen textures lazily on the next `render()` call.
        """
        self._size = size
        extra = pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE
        if self.gl_renderer is not None:
            self._gl_window_attribs()
            pygame.display.set_mode(self._size, pygame.OPENGL | pygame.DOUBLEBUF | extra)
            self._gl_ctx.viewport = (0, 0, size[0], size[1])
            self.screen = pygame.Surface(self._size, pygame.SRCALPHA)
            self._resize_ui_texture()
        else:
            self.screen = pygame.display.set_mode(self._size, extra)

    def set_resolution(self, width: int, height: int) -> None:
        """Resize the window (e.g. from an editor Settings panel)."""
        if not self.fullscreen:
            self._windowed_size = (width, height)
        self._rebuild_window((width, height), self.fullscreen)

    def set_fullscreen(self, enabled: bool) -> None:
        """Toggle OS fullscreen; the layout adapts to whatever resolution
        that ends up being (see editor.py's proportional dock sizing) --
        this call only owns window/context lifecycle, not panel layout."""
        if enabled == self.fullscreen:
            return
        if enabled:
            self._windowed_size = self._size
            # NOT pygame.display.Info() -- that reports the *current window's*
            # size once a mode is set, not the monitor's native resolution;
            # get_desktop_sizes() is independent of window state.
            size = pygame.display.get_desktop_sizes()[0]
        else:
            size = self._windowed_size
        self.fullscreen = enabled
        self._rebuild_window(size, enabled)

    def run(self, scene, camera, max_frames: int | None = None,
            screenshot_path: str | None = None, overlay=None) -> None:
        """Run the game loop until quit.

        Updates run on a fixed timestep (deterministic behaviors); rendering
        runs as fast as the frame allows, capped at max_fps. `overlay(engine)`
        is called after the 3D render, before flip — used by the editor UI.
        `max_frames` + `screenshot_path` support benchmarking.
        """
        self._end_splash()
        self.scene = scene  # behaviors (e.g. collision) may query the live scene
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
                    if not (self.esc_handler is not None and self.esc_handler()):
                        running = False
                elif e.type == pygame.VIDEORESIZE and not self.fullscreen:
                    # dragging the OS window edge -- SDL already resized the
                    # surface but GL/wgpu targets need to follow explicitly
                    # (see _rebuild_window); ignored in fullscreen since SDL
                    # can send a spurious VIDEORESIZE during that transition
                    if (e.w, e.h) != self._size:
                        self._windowed_size = (e.w, e.h)
                        self._rebuild_window((e.w, e.h), False)
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
                    if self.gl_renderer is not None:
                        self.gl_renderer.wireframe = self.renderer.wireframe
                    # wgpu_renderer.wireframe exists but is a documented,
                    # harmless no-op (see wgpu_renderer.py) -- F1 has no
                    # visible effect on the dx12/vulkan backends.
                if (self.input.pressed(pygame.K_F2) and self.gl_renderer is None
                        and self.wgpu_renderer is None):
                    self.renderer.per_pixel = not self.renderer.per_pixel  # software-only
                if self.input.pressed(pygame.K_h):
                    self.show_hud = not self.show_hud
                self.input.consume_edges()

            tracer = None
            if scene.enable_shadows:
                self.tracer.refresh(scene)
                tracer = self.tracer

            # Distance-based LOD selection: once per rendered frame (not
            # per fixed-timestep update), before whichever backend renders,
            # so CPU/GL/wgpu all draw the SAME selected LOD per entity this
            # frame (see engine/lod.py -- hysteresis lives in entity.lod_index).
            update_scene_lods(scene, camera)

            if self.gl_renderer is not None:
                self.gl_renderer.render(scene, camera, self._size, tracer)
                self.screen.fill((0, 0, 0, 0))
                if self.show_hud:
                    self._draw_hud(clock.get_fps(), scene)
                if overlay is not None:
                    overlay(self)
                self._composite_gpu_overlay()
            elif self.wgpu_renderer is not None:
                self.wgpu_renderer.render(scene, camera, self._size, tracer)
                # offscreen frame -> CPU readback -> blit into the real
                # window surface (see wgpu_renderer.py's module docstring
                # for why this design and its cost); HUD/overlay then draw
                # on self.screen exactly as in software mode.
                frame = self.wgpu_renderer.read_frame()
                surf = pygame.image.frombuffer(frame, self._size, "RGBA")
                self.screen.blit(surf, (0, 0))
                if self.show_hud:
                    self._draw_hud(clock.get_fps(), scene)
                if overlay is not None:
                    overlay(self)
            else:
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
                    self._save_screenshot(screenshot_path)
                running = False

        elapsed = time.perf_counter() - start_time
        if max_frames is not None and elapsed > 0:
            stats = self._active_stats()
            print(f"{frames} frames in {elapsed:.2f}s -> {frames / elapsed:.1f} FPS avg "
                  f"({stats['triangles']} triangles in final frame)")
        pygame.quit()

    def _active_stats(self) -> dict:
        """The currently-active renderer's stats dict (mode/triangles/
        shadow_lights) -- whichever of gl_renderer/wgpu_renderer/renderer is
        actually running, not whichever api was requested."""
        if self.gl_renderer is not None:
            return self.gl_renderer.stats
        if self.wgpu_renderer is not None:
            return self.wgpu_renderer.stats
        return self.renderer.stats

    def _save_screenshot(self, path: str) -> None:
        if self.gl_renderer is None:
            # both software mode and wgpu mode already have the full
            # composited frame sitting in self.screen (wgpu blits its
            # offscreen readback there every tick -- see run()), so a plain
            # surface save just works for either.
            pygame.image.save(self.screen, path)
            return
        # self.screen is only the UI overlay in GL mode -- read the composited
        # frame back from the default framebuffer instead. glReadPixels rows
        # are bottom-up, so flip before saving. An explicit viewport avoids a
        # moderngl quirk where ctx.screen's cached size goes stale after a
        # pygame.display.set_mode() resize (see set_resolution).
        w, h = self._size
        data = self._gl_ctx.screen.read(viewport=(0, 0, w, h), components=3)
        surf = pygame.image.frombuffer(data, (w, h), "RGB")
        surf = pygame.transform.flip(surf, False, True)
        pygame.image.save(surf, path)

    def _draw_hud(self, fps: float, scene) -> None:
        stats = self._active_stats()
        lines = [f"{fps:5.1f} FPS | {stats['mode']} | {stats['triangles']} faces | "
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
