"""PyEngine Editor — world outliner, content browser, drag-and-drop placement.

    py editor.py                     open scenes/scene.json (or a starter scene)
    py editor.py --scene my.json     work on a specific scene file

Controls:
    RMB hold        mouse look + WASD/Space/Ctrl fly (Shift = fast)
    LMB             select in viewport / outliner; drag assets from the browser
    Z               focus camera on selection
    Ctrl+D          duplicate selection        Del  delete selection
    Ctrl+S          save scene                 F    toggle flashlight
    F1 wireframe    H toggle HUD               Esc  quit
"""
from __future__ import annotations

import argparse
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTLINER_W = 260
BROWSER_H = 118
DETAILS_H = 322
ROW_H = 20
DETAIL_ROW_H = 24
TILE_W, TILE_H, ICON = 84, 100, 64
PANEL_BG = (22, 24, 29)
PANEL_EDGE = (58, 62, 72)
TEXT = (210, 212, 218)
TEXT_DIM = (140, 143, 152)
SELECT_BG = (47, 66, 102)
HOVER_BG = (36, 39, 47)
ACCENT = (255, 170, 60)


def base_height(entity) -> float:
    """Lift so the mesh's lowest vertex sits on the placement point."""
    if entity.mesh is None:
        return 0.0
    return -float(entity.mesh.vertices[:, 1].min()) * entity.transform.scale.y


def build_starter_scene(engine, lib):
    """A small haunted courtyard so the editor never opens onto nothing."""
    Vec3 = engine.Vec3
    scene = engine.Scene(
        light=engine.DirectionalLight(Vec3(-0.4, -1.0, -0.25), ambient=0.07,
                                      color=(90, 105, 150), intensity=0.35),
        fog=engine.Fog(color=(7, 8, 12), start=10.0, end=42.0),
        sky=((3, 4, 8), (14, 16, 26)),
        background=(5, 6, 9),
    )

    def put(asset, x, z, ry=0.0, y=None):
        e = lib.instantiate(asset)
        e.transform.position = Vec3(x, base_height(e) if y is None else y, z)
        e.transform.rotation.y = ry
        return scene.add(e)

    put("Sky Sphere", 0, 0)
    put("Stone Floor", 0, 0)
    for x in (-4.0, 0.0, 4.0):
        put("Wall Segment", x, -8.0)
    for z in (-4.0, 0.0, 4.0):
        put("Wall Segment", -8.0, z, ry=math.pi / 2)
    put("Stone Pillar", -7.4, -7.4)
    put("Stone Pillar", 4.2, -7.4)
    put("Torch", -4.0, -7.3)
    put("Torch", 4.0, -7.3)
    put("Torch", -7.3, 2.0)
    put("Monolith", 0.0, -3.0, ry=0.4)
    put("Crate", 2.4, 1.8, ry=0.3)
    put("Barrel", 5.2, -4.6)
    put("Barrel", 5.9, -3.9, ry=1.1)
    put("Lantern", 3.4, 2.2)
    ghost = put("Ghost", -3.5, -1.5)
    ghost.transform.position.y = 1.5
    return scene


def make_icon(engine, asset, size=ICON):
    """Render a small 3D preview of an asset for the content browser."""
    import numpy as np
    import pygame

    entity = asset.instantiate()
    surf = pygame.Surface((size, size))
    surf.fill((29, 31, 37))
    if entity.environment is not None:  # panorama thumbnail, exposure-boosted
        img = entity.environment.image
        step = max(1, img.shape[0] // size)
        small = np.clip(img[::step, ::step] * 255.0 * 6.0, 0, 255).astype(np.uint8)
        pano = pygame.surfarray.make_surface(np.transpose(small, (1, 0, 2)))
        pygame.transform.smoothscale(pano, (size, size), surf)
        return surf
    if entity.mesh is not None:
        mini = engine.Scene(
            light=engine.DirectionalLight(engine.Vec3(-0.5, -0.9, -0.6), ambient=0.42),
            background=(29, 31, 37))
        mini.add(entity)
        bound = max(float(np.max(np.linalg.norm(entity.mesh.vertices, axis=1))), 0.1)
        cam = engine.Camera(yaw=0.65, pitch=-0.5)
        fwd = cam.forward()
        cam.position = engine.Vec3(-fwd.x, -fwd.y, -fwd.z) * (2.6 * bound)
        from engine.renderer import Renderer
        Renderer().render(surf, mini, cam)
    if entity.light is not None:
        color = tuple(entity.light.color)
        pygame.draw.circle(surf, color, (size // 2, int(size * 0.30)), 6)
        pygame.draw.circle(surf, color, (size // 2, int(size * 0.30)), 10, 1)
    return surf


class Editor:
    def __init__(self, engine_mod, eng, scene, camera, lib, scene_path):
        import pygame
        self.engine_mod = engine_mod
        self.eng = eng
        self.scene = scene
        self.camera = camera
        self.lib = lib
        self.scene_path = scene_path
        self.selected = None
        self.flashlight = None      # set by main(); hidden from glyphs
        self.fly = None             # the viewport FlyController, for C toggle
        self.dirty = False
        self.outliner_scroll = 0
        self.browser_scroll = 0
        self.drag_asset = None
        self.active_slider = None   # index into _details_rows while dragging
        self.gizmo_drag = None      # active gizmo drag state dict
        self.gizmo_mode = "translate"  # G cycles translate / rotate / scale
        self.mat_ui = None          # open MaterialEditorUI, or None
        self.status = ("", 0.0)     # transient message near the content browser
        self.save_flash = 0.0
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 14)
        self.font_small = pygame.font.SysFont("consolas,couriernew,monospace", 12)
        self.icons = {}
        count = max(len(lib.assets), 1)
        for i, a in enumerate(lib.assets):
            eng.loading_step(f"rendering thumbnail: {a.name}", 0.25 + 0.45 * i / count)
            self.icons[a.name] = make_icon(engine_mod, a)

    # ---- layout ----
    def outliner_rect(self, w, h):
        import pygame
        return pygame.Rect(w - OUTLINER_W, 0, OUTLINER_W, h - BROWSER_H - DETAILS_H)

    def details_rect(self, w, h):
        import pygame
        return pygame.Rect(w - OUTLINER_W, h - BROWSER_H - DETAILS_H,
                           OUTLINER_W, DETAILS_H)

    def browser_rect(self, w, h):
        import pygame
        return pygame.Rect(0, h - BROWSER_H, w, BROWSER_H)

    def over_ui(self, pos) -> bool:
        if self.mat_ui is not None:
            return True
        w, h = self.eng.screen.get_size()
        return (self.outliner_rect(w, h).collidepoint(pos)
                or self.details_rect(w, h).collidepoint(pos)
                or self.browser_rect(w, h).collidepoint(pos))

    def _outliner_rows(self):
        return [e for e in self.scene.entities
                if e.mesh is not None or e.light is not None
                or e.environment is not None]

    # ---- transform gizmo: G cycles translate / rotate / scale ----
    _GIZMO_AXES = (((1.0, 0.0, 0.0), (225, 85, 85)),
                   ((0.0, 1.0, 0.0), (105, 215, 105)),
                   ((0.0, 0.0, 1.0), (95, 145, 250)))
    _GIZMO_MODES = ("translate", "rotate", "scale")

    def _gizmo_center(self, w, h):
        e = self.selected
        if e is None:
            return None, None, None
        p = e.transform.position
        dist = (p - self.camera.position).length()
        length = max(0.6, dist * 0.14)
        s0 = self.camera.project(p, w, h)
        return p, s0, length

    def _gizmo_handles(self, w, h):
        """Axis segments for translate/scale: [(i, axis, s0, s1, color, length)]."""
        p, s0, length = self._gizmo_center(w, h)
        if s0 is None:
            return []
        Vec3 = self.engine_mod.Vec3
        handles = []
        for i, (axis, color) in enumerate(self._GIZMO_AXES):
            tip = Vec3(p.x + axis[0] * length, p.y + axis[1] * length,
                       p.z + axis[2] * length)
            s1 = self.camera.project(tip, w, h)
            if s1 is not None:
                handles.append((i, axis, (s0[0], s0[1]), (s1[0], s1[1]),
                                color, length))
        return handles

    def _gizmo_rings(self, w, h, steps=28):
        """Projected axis circles for rotate mode: [(i, axis, points, color)]."""
        import numpy as np
        p, s0, length = self._gizmo_center(w, h)
        if s0 is None:
            return []
        Vec3 = self.engine_mod.Vec3
        radius = length * 0.85
        rings = []
        for i, (axis, color) in enumerate(self._GIZMO_AXES):
            a = np.array(axis, dtype=float)
            u = np.cross(a, [0.0, 0.0, 1.0])
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(a, [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            v = np.cross(a, u)
            pts = []
            for k in range(steps + 1):
                t = 2.0 * math.pi * k / steps
                wp = (u * math.cos(t) + v * math.sin(t)) * radius
                sp = self.camera.project(Vec3(p.x + wp[0], p.y + wp[1],
                                              p.z + wp[2]), w, h)
                pts.append((sp[0], sp[1]) if sp is not None else None)
            rings.append((i, axis, pts, color))
        return rings

    @staticmethod
    def _segment_distance(p, a, b):
        ax, ay = b[0] - a[0], b[1] - a[1]
        seg2 = ax * ax + ay * ay
        if seg2 < 1e-9:
            return math.hypot(p[0] - a[0], p[1] - a[1])
        t = max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / seg2))
        return math.hypot(p[0] - (a[0] + ax * t), p[1] - (a[1] + ay * t))

    def _try_grab_gizmo(self, mp, w, h) -> bool:
        e = self.selected
        if e is None:
            return False
        mode = self.gizmo_mode
        t, s = e.transform, e.transform.scale
        if mode == "rotate":
            _p, s0, _len = self._gizmo_center(w, h)
            best = None
            for i, axis, pts, _c in self._gizmo_rings(w, h):
                for a, b in zip(pts, pts[1:]):
                    if a is None or b is None:
                        continue
                    d = self._segment_distance(mp, a, b)
                    if d < 9.0 and (best is None or d < best[0]):
                        best = (d, i, axis)
            if best is None:
                return False
            _d, i, axis = best
            to_cam = self.camera.position - t.position
            toward = (axis[0] * to_cam.x + axis[1] * to_cam.y
                      + axis[2] * to_cam.z) > 0
            self.gizmo_drag = {
                "mode": "rotate", "axis_i": i, "center": (s0[0], s0[1]),
                "a0": math.atan2(mp[1] - s0[1], mp[0] - s0[0]),
                "sign": -1.0 if toward else 1.0,
                "start": (t.rotation.x, t.rotation.y, t.rotation.z)}
            return True

        handles = self._gizmo_handles(w, h)
        if mode == "scale":
            _p, s0, _len = self._gizmo_center(w, h)
            if s0 is not None and math.hypot(mp[0] - s0[0], mp[1] - s0[1]) < 10:
                self.gizmo_drag = {"mode": "scale", "axis_i": -1, "press": mp,
                                   "start": (s.x, s.y, s.z)}
                return True
        best = None
        for i, axis, s0, s1, _color, length in handles:
            d = self._segment_distance(mp, s0, s1)
            if d < 9.0 and (best is None or d < best[0]):
                best = (d, i, axis, s0, s1, length)
        if best is None:
            return False
        _d, i, axis, s0, s1, length = best
        self.gizmo_drag = {
            "mode": mode, "axis_i": i, "axis": axis, "press": mp,
            "dpx": (s1[0] - s0[0], s1[1] - s0[1]), "length": length,
            "start": ((t.position.x, t.position.y, t.position.z)
                      if mode == "translate" else (s.x, s.y, s.z))}
        return True

    def _update_gizmo_drag(self, mp) -> None:
        g = self.gizmo_drag
        Vec3 = self.engine_mod.Vec3
        e = self.selected
        if g["mode"] == "rotate":
            cx, cy = g["center"]
            ang = math.atan2(mp[1] - cy, mp[0] - cx)
            delta = (ang - g["a0"]) * g["sign"]
            r = list(g["start"])
            r[g["axis_i"]] += delta
            e.transform.rotation = Vec3(*r)
            self.dirty = True
            return
        if g["mode"] == "scale" and g["axis_i"] == -1:
            factor = max(0.05, 1.0 + (mp[0] - g["press"][0]) * 0.004)
            s = g["start"]
            e.transform.scale = Vec3(s[0] * factor, s[1] * factor, s[2] * factor)
            self.dirty = True
            return
        dx, dy = g["dpx"]
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-9:
            return
        t = ((mp[0] - g["press"][0]) * dx + (mp[1] - g["press"][1]) * dy) / seg2
        if g["mode"] == "translate":
            move = t * g["length"]
            ax, s = g["axis"], g["start"]
            e.transform.position = Vec3(s[0] + ax[0] * move, s[1] + ax[1] * move,
                                        s[2] + ax[2] * move)
        else:  # per-axis scale
            s = list(g["start"])
            s[g["axis_i"]] = s[g["axis_i"]] * max(0.05, 1.0 + t)
            e.transform.scale = Vec3(*s)
        self.dirty = True

    # ---- details panel rows for the selected light ----
    def _details_rows(self):
        e = self.selected
        if e is None:
            return []
        rows = []
        if e.mesh is not None:
            rows.append({"kind": "button", "label": "material",
                         "text": "open node editor  (M)",
                         "action": lambda: setattr(self, "mat_ui",
                                                   MaterialEditorUI(self, e))})
        if e.environment is not None:
            env = e.environment
            return rows + [{"kind": "slider", "label": "env strength", "min": 0.0,
                            "max": 3.0, "get": lambda: env.strength,
                            "set": lambda v: setattr(env, "strength", v),
                            "fmt": "{:.2f}"}]
        if e.light is None:
            return rows
        light = e.light
        Flicker = self.engine_mod.behaviors.Flicker
        SpotLight = self.engine_mod.SpotLight

        def get_intensity():
            for b in e.behaviors:
                if isinstance(b, Flicker) and hasattr(b, "base"):
                    return b.base
            return light.intensity

        def set_intensity(v):
            light.intensity = v
            for b in e.behaviors:
                if isinstance(b, Flicker):
                    b.base = v

        def color_setter(i):
            def setter(v):
                c = list(light.color)
                c[i] = int(v)
                light.color = tuple(c)
            return setter

        def slider(label, lo, hi, get, set_, fmt="{:.2f}"):
            return {"kind": "slider", "label": label, "min": lo, "max": hi,
                    "get": get, "set": set_, "fmt": fmt}

        rows += [
            slider("brightness", 0.0, 5.0, get_intensity, set_intensity),
            slider("red", 0, 255, lambda: light.color[0], color_setter(0), "{:.0f}"),
            slider("green", 0, 255, lambda: light.color[1], color_setter(1), "{:.0f}"),
            slider("blue", 0, 255, lambda: light.color[2], color_setter(2), "{:.0f}"),
            slider("throw", 2.0, 40.0, lambda: light.range,
                   lambda v: setattr(light, "range", v), "{:.1f}"),
            slider("shadow soft", 0.02, 1.0, lambda: light.radius,
                   lambda v: setattr(light, "radius", v)),
        ]
        if isinstance(light, SpotLight):
            rows.append(slider("cone inner", 1.0, 80.0, lambda: light.inner,
                               lambda v: setattr(light, "inner",
                                                 min(v, light.outer - 1.0)), "{:.0f}°"))
            rows.append(slider("penumbra", 2.0, 85.0, lambda: light.outer,
                               lambda v: setattr(light, "outer",
                                                 max(v, light.inner + 1.0)), "{:.0f}°"))
        rows.append({"kind": "cycle", "label": "IES profile",
                     "get": lambda: light.ies,
                     "set": lambda v: setattr(light, "ies", v),
                     "options": self.engine_mod.IES_PROFILES})
        rows.append({"kind": "toggle", "label": "enabled",
                     "get": lambda: light.enabled,
                     "set": lambda v: setattr(light, "enabled", v)})
        rows.append({"kind": "toggle", "label": "cast shadows",
                     "get": lambda: light.cast_shadows,
                     "set": lambda v: setattr(light, "cast_shadows", v)})
        return rows

    def _detail_row_rect(self, rect, i):
        import pygame
        return pygame.Rect(rect.x + 6, rect.y + 56 + i * DETAIL_ROW_H,
                           rect.width - 12, DETAIL_ROW_H - 2)

    def _slider_track(self, row_rect):
        return (row_rect.x + 96, row_rect.right - 52)

    def _apply_slider(self, row, row_rect, mx):
        x0, x1 = self._slider_track(row_rect)
        f = min(max((mx - x0) / max(x1 - x0, 1), 0.0), 1.0)
        row["set"](row["min"] + f * (row["max"] - row["min"]))
        self.dirty = True

    # ---- per-frame logic (runs in a fixed update step) ----
    def update(self, engine, dt: float) -> None:
        import pygame
        inp = engine.input
        mp = inp.mouse_pos
        w, h = engine.screen.get_size()
        orect = self.outliner_rect(w, h)
        drect = self.details_rect(w, h)
        brect = self.browser_rect(w, h)
        self.save_flash = max(0.0, self.save_flash - dt)
        if self.status[1] > 0:
            self.status = (self.status[0], self.status[1] - dt)

        if self.mat_ui is not None:  # node editor captures all editor input
            self.mat_ui.update(engine, dt)
            return

        if inp.wheel:
            if orect.collidepoint(mp):
                self.outliner_scroll = max(0, self.outliner_scroll - int(inp.wheel) * 3)
            elif brect.collidepoint(mp):
                self.browser_scroll = max(0, self.browser_scroll - int(inp.wheel) * 70)

        if inp.mouse_button_pressed(1):
            if orect.collidepoint(mp):
                self.active_slider = None
                self._click_outliner(mp, orect)
            elif drect.collidepoint(mp):
                self._click_details(mp, drect)
            elif brect.collidepoint(mp):
                if self._import_btn_rect(brect).collidepoint(mp):
                    self._import_fbx_dialog()
                else:
                    asset = self._tile_at(mp, brect)
                    if asset is not None:
                        self.drag_asset = asset
            else:
                self.active_slider = None
                if not self._try_grab_gizmo(mp, w, h):
                    self._click_viewport(mp, w, h)

        # gizmo drag
        if self.gizmo_drag is not None:
            if inp.mouse_held(1) and self.selected is not None:
                self._update_gizmo_drag(mp)
            else:
                self.gizmo_drag = None

        # live slider drag
        if self.active_slider is not None:
            rows = self._details_rows()
            if inp.mouse_held(1) and self.active_slider < len(rows):
                row = rows[self.active_slider]
                if row["kind"] == "slider":
                    self._apply_slider(row, self._detail_row_rect(drect,
                                                                  self.active_slider),
                                       mp[0])
            else:
                self.active_slider = None

        if self.drag_asset is not None and inp.mouse_button_released(1):
            if not self.over_ui(mp):
                self._place_asset(self.drag_asset, mp, w, h)
            self.drag_asset = None

        ctrl = inp.held(pygame.K_LCTRL) or inp.held(pygame.K_RCTRL)
        if inp.pressed(pygame.K_DELETE) and self.selected is not None \
                and self.selected.asset_name is not None:
            self.scene.remove(self.selected)
            self.selected = None
            self.dirty = True
        if ctrl and inp.pressed(pygame.K_d) and self.selected is not None \
                and self.selected.asset_name is not None:
            src = self.selected
            dup = self.lib.instantiate(src.asset_name)
            t, s = dup.transform, src.transform
            t.position = self.engine_mod.Vec3(s.position.x + 0.8, s.position.y,
                                              s.position.z + 0.8)
            t.rotation = self.engine_mod.Vec3(s.rotation.x, s.rotation.y, s.rotation.z)
            t.scale = self.engine_mod.Vec3(s.scale.x, s.scale.y, s.scale.z)
            self.scene.add(dup)
            self.selected = dup
            self.dirty = True
        if ctrl and inp.pressed(pygame.K_s):
            os.makedirs(os.path.dirname(self.scene_path) or ".", exist_ok=True)
            self.engine_mod.save_scene(self.scene, self.camera, self.scene_path)
            self.dirty = False
            self.save_flash = 1.5
        if inp.pressed(pygame.K_z) and self.selected is not None:
            self._focus(self.selected)
        if inp.pressed(pygame.K_c) and self.fly is not None:
            self.fly.collide = not self.fly.collide
        if inp.pressed(pygame.K_g):
            modes = self._GIZMO_MODES
            self.gizmo_mode = modes[(modes.index(self.gizmo_mode) + 1) % len(modes)]
            self.gizmo_drag = None
        if inp.pressed(pygame.K_m) and self.selected is not None \
                and self.selected.mesh is not None:
            self.mat_ui = MaterialEditorUI(self, self.selected)

        # rotate / scale the selection
        if self.selected is not None:
            step = math.pi / 12.0
            if inp.pressed(pygame.K_COMMA):
                self.selected.transform.rotation.y += step
                self.dirty = True
            if inp.pressed(pygame.K_PERIOD):
                self.selected.transform.rotation.y -= step
                self.dirty = True
            if inp.pressed(pygame.K_MINUS) or inp.pressed(pygame.K_EQUALS):
                f = 1.1 if inp.pressed(pygame.K_EQUALS) else 1.0 / 1.1
                sc = self.selected.transform.scale
                self.selected.transform.scale = self.engine_mod.Vec3(
                    sc.x * f, sc.y * f, sc.z * f)
                self.dirty = True

    def handle_escape(self) -> bool:
        """Engine Esc hook: closes the material editor instead of quitting."""
        if self.mat_ui is not None:
            self.mat_ui.close()
            return True
        return False

    def _import_btn_rect(self, brect):
        import pygame
        return pygame.Rect(brect.right - 130, brect.y + 4, 120, 20)

    def _import_fbx_dialog(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Import FBX model",
                filetypes=[("FBX models", "*.fbx"), ("All files", "*.*")])
            root.destroy()
        except Exception as ex:
            self.status = (f"file dialog unavailable: {ex}", 5.0)
            return
        if not path:
            return
        try:
            name = self.engine_mod.import_fbx(path, self.lib.directory)
            self.lib.reload()
            self.icons[name] = make_icon(self.engine_mod, self.lib.by_name[name])
            self.status = (f"imported '{name}' — drag it from the browser", 5.0)
        except Exception as ex:
            self.status = (f"import failed: {ex}", 6.0)

    def _click_details(self, mp, drect) -> None:
        rows = self._details_rows()
        i = (mp[1] - (drect.y + 56)) // DETAIL_ROW_H
        if not (0 <= i < len(rows)):
            return
        row = rows[i]
        if row["kind"] == "slider":
            self.active_slider = i
            self._apply_slider(row, self._detail_row_rect(drect, i), mp[0])
        elif row["kind"] == "cycle":
            options = row["options"]
            current = options.index(row["get"]()) if row["get"]() in options else 0
            row["set"](options[(current + 1) % len(options)])
            self.dirty = True
        elif row["kind"] == "toggle":
            row["set"](not row["get"]())
            self.dirty = True
        elif row["kind"] == "button":
            row["action"]()

    def _click_outliner(self, mp, orect) -> None:
        rows = self._outliner_rows()
        i = (mp[1] - orect.y - 30) // ROW_H + self.outliner_scroll
        if 0 <= mp[1] - orect.y - 30 and 0 <= i < len(rows):
            self.selected = rows[i]

    def _tile_at(self, mp, brect):
        x0 = brect.x + 10 - self.browser_scroll
        for asset in self.lib.assets:
            if x0 <= mp[0] < x0 + TILE_W and brect.y + 26 <= mp[1] < brect.y + 26 + TILE_H - 8:
                return asset
            x0 += TILE_W + 8
        return None

    def _mouse_hit(self, mp, w, h):
        """Ray from the camera through the mouse; returns a world point."""
        import numpy as np
        origin = self.camera.position.to_array()
        direction = self.camera.mouse_ray(mp[0], mp[1], w, h)
        entity, t = self.engine_mod.pick_entity(self.scene, origin, direction)
        if entity is not None:
            return entity, origin + direction * t
        if direction[1] < -1e-4:  # fall back to the ground plane y=0
            t = -origin[1] / direction[1]
            if 0 < t < 400:
                return None, origin + direction * t
        return None, origin + direction * 8.0

    def _click_viewport(self, mp, w, h) -> None:
        entity, _ = self._mouse_hit(mp, w, h)
        self.selected = entity

    def _place_asset(self, asset, mp, w, h) -> None:
        _, point = self._mouse_hit(mp, w, h)
        entity = asset.instantiate()
        entity.transform.position = self.engine_mod.Vec3(
            float(point[0]), float(point[1]) + base_height(entity), float(point[2]))
        self.scene.add(entity)
        self.selected = entity
        self.dirty = True

    def _focus(self, entity) -> None:
        import numpy as np
        bound = 1.0
        if entity.mesh is not None:
            bound = max(float(np.max(np.linalg.norm(entity.mesh.vertices, axis=1))), 0.5)
        dist = max(3.0, bound * 3.0)
        fwd = self.camera.forward()
        p = entity.transform.position
        self.camera.position = self.engine_mod.Vec3(
            p.x - fwd.x * dist, p.y - fwd.y * dist, p.z - fwd.z * dist)

    # ---- drawing (engine overlay callback) ----
    def draw(self, eng) -> None:
        import pygame
        surf = eng.screen
        w, h = surf.get_size()
        self._draw_markers(surf, w, h)
        self._draw_outliner(surf, self.outliner_rect(w, h))
        self._draw_details(surf, self.details_rect(w, h))
        self._draw_browser(surf, self.browser_rect(w, h))
        if self.drag_asset is not None:
            icon = self.icons.get(self.drag_asset.name)
            if icon is not None:
                ghost = icon.copy()
                ghost.set_alpha(150)
                mp = eng.input.mouse_pos
                surf.blit(ghost, (mp[0] - ICON // 2, mp[1] - ICON // 2))
        if self.mat_ui is not None:
            self.mat_ui.draw(surf)

    def _draw_markers(self, surf, w, h) -> None:
        import numpy as np
        import pygame
        # light glyphs
        for e in self.scene.entities:
            if e.light is None or e is self.flashlight:
                continue
            m = e.transform.matrix()
            off = e.light_offset
            pos = m[:3, :3] @ np.array([off.x, off.y, off.z]) + m[:3, 3]
            pt = self.camera.project(self.engine_mod.Vec3(*pos), w, h)
            if pt is None:
                continue
            x, y = int(pt[0]), int(pt[1])
            color = tuple(e.light.color) if e.light.enabled else (70, 70, 70)
            pygame.draw.circle(surf, color, (x, y), 4)
            pygame.draw.circle(surf, color, (x, y), 8, 1)
        # selection brackets + transform gizmo
        e = self.selected
        if e is None:
            return
        drag = self.gizmo_drag
        if self.gizmo_mode == "rotate":
            for i, _axis, pts, color in self._gizmo_rings(w, h):
                active = drag is not None and drag.get("axis_i") == i
                c = (255, 255, 255) if active else color
                for a, b in zip(pts, pts[1:]):
                    if a is not None and b is not None:
                        pygame.draw.line(surf, c, (int(a[0]), int(a[1])),
                                         (int(b[0]), int(b[1])), 2)
        else:
            for i, _axis, s0, s1, color, _length in self._gizmo_handles(w, h):
                active = drag is not None and drag.get("axis_i") == i
                c = (255, 255, 255) if active else color
                pygame.draw.line(surf, c, (int(s0[0]), int(s0[1])),
                                 (int(s1[0]), int(s1[1])), 3)
                tip = (int(s1[0]), int(s1[1]))
                if self.gizmo_mode == "translate":
                    pygame.draw.circle(surf, c, tip, 6)
                else:
                    pygame.draw.rect(surf, c, (tip[0] - 5, tip[1] - 5, 10, 10))
            if self.gizmo_mode == "scale":
                _p, s0, _l = self._gizmo_center(w, h)
                if s0 is not None:
                    active = drag is not None and drag.get("axis_i") == -1
                    c = (255, 255, 255) if active else (200, 200, 205)
                    pygame.draw.rect(surf, c, (int(s0[0]) - 6, int(s0[1]) - 6,
                                               12, 12), 2)
        _p, s0, _l = self._gizmo_center(w, h)
        if s0 is not None:
            mode_label = self.font_small.render(self.gizmo_mode, True, TEXT_DIM)
            surf.blit(mode_label, (int(s0[0]) + 12, int(s0[1]) + 10))
        pt = self.camera.project(e.transform.position, w, h)
        if pt is None:
            return
        bound = 0.8
        if e.mesh is not None:
            bound = max(float(np.max(np.linalg.norm(e.mesh.vertices, axis=1))), 0.3)
        k = 0.5 * h / math.tan(math.radians(self.camera.fov) * 0.5)
        r = max(14, int(k * bound / pt[2]))
        x, y = int(pt[0]), int(pt[1])
        s = max(6, r // 3)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cx, cy = x + dx * r, y + dy * r
            pygame.draw.line(surf, ACCENT, (cx, cy), (cx - dx * s, cy), 2)
            pygame.draw.line(surf, ACCENT, (cx, cy), (cx, cy - dy * s), 2)
        label = self.font_small.render(e.name, True, ACCENT)
        surf.blit(label, (x - label.get_width() // 2, y - r - 16))

    def _draw_outliner(self, surf, rect) -> None:
        import pygame
        pygame.draw.rect(surf, PANEL_BG, rect)
        pygame.draw.line(surf, PANEL_EDGE, rect.topleft, rect.bottomleft)
        name = os.path.basename(self.scene_path) + (" *" if self.dirty else "")
        if self.save_flash > 0:
            name = "saved ✓"
        title = self.font.render(f"World Outliner — {name}", True, TEXT)
        surf.blit(title, (rect.x + 10, rect.y + 8))

        rows = self._outliner_rows()
        visible = (rect.height - 30 - 22) // ROW_H
        self.outliner_scroll = max(0, min(self.outliner_scroll, max(0, len(rows) - visible)))
        mp = pygame.mouse.get_pos()
        y = rect.y + 30
        for e in rows[self.outliner_scroll:self.outliner_scroll + visible]:
            row = pygame.Rect(rect.x + 1, y, rect.width - 2, ROW_H)
            if e is self.selected:
                pygame.draw.rect(surf, SELECT_BG, row)
            elif row.collidepoint(mp):
                pygame.draw.rect(surf, HOVER_BG, row)
            x = rect.x + 10
            if e.mesh is not None:
                c = tuple(int(v) for v in e.mesh.face_colors.mean(axis=0))
                pygame.draw.rect(surf, c, (x, y + 6, 8, 8))
            if e.environment is not None:
                pygame.draw.circle(surf, (120, 190, 235), (x + 4, y + 10), 5, 1)
                pygame.draw.line(surf, (120, 190, 235), (x, y + 10), (x + 8, y + 10))
            x += 12
            if e.light is not None:
                c = tuple(e.light.color) if e.light.enabled else (80, 80, 80)
                pygame.draw.circle(surf, c, (x + 4, y + 10), 4)
            x += 12
            text = self.font.render(e.name[:24], True, TEXT)
            surf.blit(text, (x, y + 3))
            y += ROW_H
        hint = self.font_small.render("Del del · Ctrl+D dup · Z focus · Ctrl+S save",
                                      True, TEXT_DIM)
        surf.blit(hint, (rect.x + 10, rect.bottom - 18))

    def _draw_details(self, surf, rect) -> None:
        import pygame
        pygame.draw.rect(surf, PANEL_BG, rect)
        pygame.draw.line(surf, PANEL_EDGE, rect.topleft, rect.topright)
        pygame.draw.line(surf, PANEL_EDGE, rect.topleft, rect.bottomleft)
        surf.blit(self.font.render("Details", True, TEXT), (rect.x + 10, rect.y + 8))

        e = self.selected
        if e is None:
            surf.blit(self.font_small.render("select an entity", True, TEXT_DIM),
                      (rect.x + 10, rect.y + 32))
            return
        head = f"{e.name}" + (f"  ({e.asset_name})" if e.asset_name else "")
        surf.blit(self.font_small.render(head[:34], True, TEXT), (rect.x + 10, rect.y + 28))
        p = e.transform.position
        pos_text = f"x {p.x:.1f}  y {p.y:.1f}  z {p.z:.1f}"
        surf.blit(self.font_small.render(pos_text, True, TEXT_DIM),
                  (rect.x + 10, rect.y + 42))

        rows = self._details_rows()
        if not rows:
            surf.blit(self.font_small.render("no light on this entity", True, TEXT_DIM),
                      (rect.x + 10, rect.y + 64))
            return
        for i, row in enumerate(rows):
            rr = self._detail_row_rect(rect, i)
            label = self.font_small.render(row["label"], True, TEXT_DIM)
            surf.blit(label, (rr.x + 2, rr.y + 5))
            if row["kind"] == "slider":
                x0, x1 = self._slider_track(rr)
                cy = rr.y + rr.height // 2
                f = (row["get"]() - row["min"]) / (row["max"] - row["min"])
                f = min(max(f, 0.0), 1.0)
                pygame.draw.line(surf, (48, 51, 60), (x0, cy), (x1, cy), 4)
                knob_x = int(x0 + f * (x1 - x0))
                pygame.draw.line(surf, ACCENT, (x0, cy), (knob_x, cy), 4)
                pygame.draw.circle(surf, (235, 235, 240), (knob_x, cy), 5)
                value = self.font_small.render(row["fmt"].format(row["get"]()),
                                               True, TEXT)
                surf.blit(value, (x1 + 8, rr.y + 5))
            elif row["kind"] == "cycle":
                value = self.font_small.render(f"< {row['get']()} >", True, ACCENT)
                surf.blit(value, (rr.x + 96, rr.y + 5))
            elif row["kind"] == "toggle":
                on = row["get"]()
                box = pygame.Rect(rr.x + 96, rr.y + 5, 12, 12)
                pygame.draw.rect(surf, (48, 51, 60), box, border_radius=2)
                if on:
                    pygame.draw.rect(surf, ACCENT, box.inflate(-4, -4), border_radius=2)
                state = self.font_small.render("on" if on else "off", True, TEXT)
                surf.blit(state, (box.right + 8, rr.y + 5))
            elif row["kind"] == "button":
                value = self.font_small.render(row["text"], True, ACCENT)
                surf.blit(value, (rr.x + 96, rr.y + 5))

    def _draw_browser(self, surf, rect) -> None:
        import pygame
        pygame.draw.rect(surf, PANEL_BG, rect)
        pygame.draw.line(surf, PANEL_EDGE, rect.topleft, rect.topright)
        title = self.font.render("Content Browser — drag into the world  (assets/)",
                                 True, TEXT)
        surf.blit(title, (rect.x + 10, rect.y + 6))
        mp = pygame.mouse.get_pos()
        btn = self._import_btn_rect(rect)
        pygame.draw.rect(surf, HOVER_BG if btn.collidepoint(mp) else (33, 36, 44),
                         btn, border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, btn, 1, border_radius=4)
        label = self.font_small.render("+ Import FBX", True, ACCENT)
        surf.blit(label, (btn.x + (btn.width - label.get_width()) // 2, btn.y + 4))
        if self.status[1] > 0:
            msg = self.font_small.render(self.status[0][:80], True, (235, 210, 140))
            surf.blit(msg, (btn.x - msg.get_width() - 14, rect.y + 8))
        x = rect.x + 10 - self.browser_scroll
        for asset in self.lib.assets:
            tile = pygame.Rect(x, rect.y + 26, TILE_W, TILE_H - 12)
            if tile.right > rect.x and tile.left < rect.right:
                hovered = tile.collidepoint(mp)
                pygame.draw.rect(surf, HOVER_BG if hovered else (30, 32, 39), tile,
                                 border_radius=4)
                icon = self.icons.get(asset.name)
                if icon is not None:
                    surf.blit(icon, (x + (TILE_W - ICON) // 2, rect.y + 30))
                label = self.font_small.render(asset.name[:12], True,
                                               TEXT if hovered else TEXT_DIM)
                surf.blit(label, (x + (TILE_W - label.get_width()) // 2,
                                  rect.y + 30 + ICON + 3))
            x += TILE_W + 8


NODE_W = 150


class MaterialEditorUI:
    """Node-based material editor: drag ports to connect, drag params to tune.

    The graph bakes to the entity mesh's per-face colors on every change, so
    the 3D viewport behind the panel is a live preview.
    """

    PALETTE = ("color", "position", "normal", "checker", "noise",
               "gradient", "mix", "multiply")

    def __init__(self, editor: Editor, entity):
        self.editor = editor
        self.entity = entity
        if entity.material is None:
            entity.material = editor.engine_mod.MaterialGraph()
        self.graph = entity.material
        self.drag_node = None    # (node_id, grab_dx, grab_dy)
        self.drag_link = None    # source node id while dragging a new wire
        self.drag_param = None   # (node_id, param_name)
        self._spawn_i = 0

    def close(self) -> None:
        self.editor.mat_ui = None

    # ---- geometry ----
    def rect(self, w, h):
        import pygame
        return pygame.Rect(30, 30, w - OUTLINER_W - 60, h - BROWSER_H - 60)

    def node_rect(self, nid, panel):
        import pygame
        node = self.graph.nodes[nid]
        inputs, params = self.editor.engine_mod.NODE_DEFS[node["type"]]
        height = 24 + len(inputs) * 18 + len(node["params"]) * 18 + 6
        return pygame.Rect(int(panel.x + node["pos"][0]),
                           int(panel.y + node["pos"][1]), NODE_W, height)

    def input_pos(self, nid, index, panel):
        r = self.node_rect(nid, panel)
        return (r.x, r.y + 24 + index * 18 + 9)

    def output_pos(self, nid, panel):
        r = self.node_rect(nid, panel)
        return (r.right, r.y + r.height // 2)

    def _param_row(self, nid, j, panel):
        import pygame
        node = self.graph.nodes[nid]
        inputs, _ = self.editor.engine_mod.NODE_DEFS[node["type"]]
        r = self.node_rect(nid, panel)
        return pygame.Rect(r.x + 6, r.y + 24 + (len(inputs) + j) * 18,
                           NODE_W - 12, 16)

    def _palette_rects(self, panel):
        import pygame
        out = []
        x = panel.x + 10
        for t in self.PALETTE:
            w = 24 + 7 * len(t)
            out.append((t, pygame.Rect(x, panel.y + 6, w, 20)))
            x += w + 6
        return out

    # ---- interaction ----
    def apply(self) -> None:
        self.graph.apply(self.entity.mesh)
        self.editor.dirty = True

    def update(self, engine, dt: float) -> None:
        import pygame
        inp = engine.input
        mp = inp.mouse_pos
        w, h = engine.screen.get_size()
        panel = self.rect(w, h)

        if inp.pressed(pygame.K_m):
            self.close()
            return
        if inp.mouse_button_pressed(1):
            self._press(mp, panel)
        if inp.mouse_held(1):
            if self.drag_node is not None:
                nid, dx, dy = self.drag_node
                if nid in self.graph.nodes:
                    self.graph.nodes[nid]["pos"] = [mp[0] - panel.x - dx,
                                                    mp[1] - panel.y - dy]
            if self.drag_param is not None:
                nid, pname = self.drag_param
                if nid in self.graph.nodes:
                    node = self.graph.nodes[nid]
                    inputs, _ = self.editor.engine_mod.NODE_DEFS[node["type"]]
                    j = list(node["params"]).index(pname)
                    rr = self._param_row(nid, j, panel)
                    lo, hi = self.editor.engine_mod.PARAM_RANGES.get(pname, (0, 1))
                    f = min(max((mp[0] - (rr.x + 46)) / max(rr.width - 52, 1), 0.0), 1.0)
                    node["params"][pname] = lo + f * (hi - lo)
                    self.apply()
        else:
            if self.drag_link is not None:
                self._finish_link(mp, panel)
            self.drag_node = self.drag_param = self.drag_link = None

    def _press(self, mp, panel) -> None:
        import pygame
        close = pygame.Rect(panel.right - 26, panel.y + 5, 20, 20)
        if close.collidepoint(mp):
            self.close()
            return
        for t, r in self._palette_rects(panel):
            if r.collidepoint(mp):
                self._spawn_i += 1
                self.graph.add(t, (30 + (self._spawn_i % 5) * 40,
                                   60 + (self._spawn_i % 7) * 30))
                return
        NODE_DEFS = self.editor.engine_mod.NODE_DEFS
        for nid in reversed(list(self.graph.nodes)):
            node = self.graph.nodes[nid]
            r = self.node_rect(nid, panel)
            # output port
            ox, oy = self.output_pos(nid, panel)
            if node["type"] != "output" and math.hypot(mp[0] - ox, mp[1] - oy) < 9:
                self.drag_link = nid
                return
            # input ports: click to unplug (and grab the wire), or nothing
            inputs, _ = NODE_DEFS[node["type"]]
            for i, name in enumerate(inputs):
                ix, iy = self.input_pos(nid, i, panel)
                if math.hypot(mp[0] - ix, mp[1] - iy) < 9:
                    src = self.graph.link_into(nid, name)
                    if src is not None:
                        self.graph.disconnect(nid, name)
                        self.drag_link = src  # re-route the existing wire
                        self.apply()
                    return
            if not r.collidepoint(mp):
                continue
            # delete button
            if node["type"] != "output" and pygame.Rect(
                    r.right - 18, r.y + 3, 15, 15).collidepoint(mp):
                self.graph.remove(nid)
                self.apply()
                return
            # param rows
            for j, pname in enumerate(node["params"]):
                if self._param_row(nid, j, panel).collidepoint(mp):
                    self.drag_param = (nid, pname)
                    return
            # body drag
            self.drag_node = (nid, mp[0] - r.x, mp[1] - r.y)
            return

    def _finish_link(self, mp, panel) -> None:
        NODE_DEFS = self.editor.engine_mod.NODE_DEFS
        for nid, node in self.graph.nodes.items():
            inputs, _ = NODE_DEFS[node["type"]]
            for i, name in enumerate(inputs):
                ix, iy = self.input_pos(nid, i, panel)
                if math.hypot(mp[0] - ix, mp[1] - iy) < 12:
                    if self.graph.connect(self.drag_link, nid, name):
                        self.apply()
                    return

    # ---- drawing ----
    def draw(self, surf) -> None:
        import pygame
        w, h = surf.get_size()
        panel = self.rect(w, h)
        pygame.draw.rect(surf, (18, 20, 25), panel)
        pygame.draw.rect(surf, PANEL_EDGE, panel, 1)
        title = self.editor.font.render(
            f"Material — {self.entity.name}   (drag ports to wire, M/Esc close)",
            True, TEXT)
        surf.blit(title, (panel.x + 10, panel.bottom - 24))
        for t, r in self._palette_rects(panel):
            hov = r.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(surf, HOVER_BG if hov else (30, 33, 40), r,
                             border_radius=4)
            lab = self.editor.font_small.render("+" + t, True, TEXT)
            surf.blit(lab, (r.x + 6, r.y + 4))
        close = pygame.Rect(panel.right - 26, panel.y + 5, 20, 20)
        pygame.draw.rect(surf, (60, 34, 34), close, border_radius=4)
        x_lab = self.editor.font_small.render("X", True, (230, 160, 160))
        surf.blit(x_lab, (close.x + 7, close.y + 4))

        NODE_DEFS = self.editor.engine_mod.NODE_DEFS
        # wires
        for src, dst, name in self.graph.links:
            if src not in self.graph.nodes or dst not in self.graph.nodes:
                continue
            inputs, _ = NODE_DEFS[self.graph.nodes[dst]["type"]]
            if name not in inputs:
                continue
            a = self.output_pos(src, panel)
            b = self.input_pos(dst, inputs.index(name), panel)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            pygame.draw.lines(surf, (150, 160, 185), False,
                              [a, (a[0] + 18, a[1]), mid, (b[0] - 18, b[1]), b], 2)
        if self.drag_link is not None:
            a = self.output_pos(self.drag_link, panel)
            pygame.draw.line(surf, ACCENT, a, pygame.mouse.get_pos(), 2)

        # nodes
        for nid, node in self.graph.nodes.items():
            r = self.node_rect(nid, panel)
            pygame.draw.rect(surf, (33, 36, 44), r, border_radius=5)
            pygame.draw.rect(surf, (70, 75, 88), r, 1, border_radius=5)
            name = self.editor.font_small.render(node["type"], True, TEXT)
            surf.blit(name, (r.x + 8, r.y + 5))
            if node["type"] != "output":
                pygame.draw.line(surf, (120, 80, 80), (r.right - 16, r.y + 6),
                                 (r.right - 7, r.y + 15), 2)
                pygame.draw.line(surf, (120, 80, 80), (r.right - 7, r.y + 6),
                                 (r.right - 16, r.y + 15), 2)
                ox, oy = self.output_pos(nid, panel)
                pygame.draw.circle(surf, (210, 190, 120), (ox, oy), 5)
            inputs, _ = NODE_DEFS[node["type"]]
            for i, iname in enumerate(inputs):
                ix, iy = self.input_pos(nid, i, panel)
                pygame.draw.circle(surf, (140, 170, 210), (ix, iy), 5)
                lab = self.editor.font_small.render(iname, True, TEXT_DIM)
                surf.blit(lab, (ix + 10, iy - 7))
            for j, (pname, value) in enumerate(node["params"].items()):
                rr = self._param_row(nid, j, panel)
                lab = self.editor.font_small.render(pname, True, TEXT_DIM)
                surf.blit(lab, (rr.x, rr.y + 2))
                lo, hi = self.editor.engine_mod.PARAM_RANGES.get(pname, (0, 1))
                f = min(max((value - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
                track_x0, track_x1 = rr.x + 46, rr.right - 6
                cy = rr.y + 8
                pygame.draw.line(surf, (52, 56, 66), (track_x0, cy), (track_x1, cy), 3)
                kx = int(track_x0 + f * (track_x1 - track_x0))
                pygame.draw.line(surf, ACCENT, (track_x0, cy), (kx, cy), 3)
                pygame.draw.circle(surf, (230, 230, 235), (kx, cy), 4)
            if node["type"] == "color":
                p = node["params"]
                sw = (int(p["r"] * 255), int(p["g"] * 255), int(p["b"] * 255))
                pygame.draw.rect(surf, sw, (r.x + 60, r.y + 4, 40, 12))


class EditorBehavior:
    """Bridges the editor into the scene's fixed-step update."""

    _started = True

    def start(self, entity, engine) -> None:
        pass

    def __init__(self, editor: Editor):
        self.editor = editor

    def update(self, entity, dt: float, engine) -> None:
        self.editor.update(engine, dt)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyEngine editor")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=810)
    parser.add_argument("--scene", default=os.path.join(BASE_DIR, "scenes", "scene.json"))
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--screenshot", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--pixel-scale", type=int, default=4,
                        help="per-pixel lighting internal resolution divisor "
                             "(lower = sharper, slower; default 4)")
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import engine

    eng = engine.Engine(args.width, args.height, title="PyEngine Editor")
    eng.renderer.render_scale = max(1, args.pixel_scale)
    eng.loading_step("loading asset library", 0.12)
    lib = engine.AssetLibrary(os.path.join(BASE_DIR, "assets"))
    camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08,
                           far=200.0)

    eng.loading_step("loading scene", 0.18)
    if os.path.exists(args.scene):
        scene = engine.load_scene(args.scene, lib, camera)
    else:
        scene = build_starter_scene(engine, lib)

    editor = Editor(engine, eng, scene, camera, lib, args.scene)

    flashlight = engine.Entity("flashlight", light=engine.SpotLight(
        color=(255, 244, 214), intensity=2.0, range=24.0, radius=0.25,
        inner=13.0, outer=27.0, shadow_samples=2, shadow_interval=2))
    flashlight.casts_shadow = False
    flashlight.add_behavior(engine.behaviors.FlashlightController(camera))
    scene.add(flashlight)
    editor.flashlight = flashlight

    fly = engine.behaviors.FlyController(
        camera, look_buttons=(3,), look_guard=lambda pos: not editor.over_ui(pos),
        collide=True)
    editor.fly = fly
    scene.add(engine.Entity("__camera").add_behavior(fly))
    scene.add(engine.Entity("__editor").add_behavior(EditorBehavior(editor)))

    # trace the static lights' shadows now so the first frame doesn't hitch
    eng.loading_step("pre-tracing shadows", 0.8)
    import pygame
    eng.tracer.refresh(scene)
    eng.renderer.render(pygame.Surface((320, 180)), scene, camera, eng.tracer)

    eng.loading_step("opening world", 0.95)
    eng.esc_handler = editor.handle_escape
    eng.hud_text = ("RMB: look/fly | LMB: select, drag assets & gizmo | G gizmo mode | "
                    "M material | F flashlight | C collision | Z focus | Ctrl+D dup | "
                    "Del del | Ctrl+S save | F2 shading | Esc quit")
    eng.run(scene, camera, max_frames=args.frames, screenshot_path=args.screenshot,
            overlay=editor.draw)


if __name__ == "__main__":
    main()
