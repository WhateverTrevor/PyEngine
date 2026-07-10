"""PyEngine Editor — menu bar, dockable panels, outliner, content browser.

    py editor.py                     open scenes/scene.json (or a starter scene)
    py editor.py --scene my.json     work on a specific scene file
    py editor.py --gpu / --cpu       force the OpenGL or software renderer
                                      (default: auto, or Settings > Renderer)

Controls (Help > Controls in the editor shows the full list):
    RMB hold        mouse look + WASD/QE/Space/Ctrl fly (Unreal-style: these
                    movement keys only act while RMB is held), Shift = fast
    LMB             select in viewport/outliner; drag assets; drag the gizmo;
                    drag panel title bars to move/dock/float them
    W/E/R           gizmo translate/rotate/scale (only while not looking)
    F               focus camera on selection (only while not looking)
    Ctrl+D          duplicate selection        Del  delete selection
    Ctrl+S          save scene                 L    toggle flashlight
    F1 wireframe    H toggle HUD               Esc  close UI / deselect / quit
"""
from __future__ import annotations

import argparse
import json
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")

MENU_H = 26
PANEL_TITLE_H = 18
EDGE_SNAP = 48

OUTLINER_W = 260          # docked left/right panel width
BROWSER_H = 118           # docked bottom panel height
DETAILS_H = 322           # default floating height for the details panel
ROW_H = 20
DETAIL_ROW_H = 24
DETAILS_ROWS_TOP = 34     # y-offset (within a panel's content rect) of row 0
TILE_W, TILE_H, ICON = 84, 100, 64
PANEL_BG = (22, 24, 29)
PANEL_EDGE = (58, 62, 72)
TEXT = (210, 212, 218)
TEXT_DIM = (140, 143, 152)
SELECT_BG = (47, 66, 102)
HOVER_BG = (36, 39, 47)
ACCENT = (255, 170, 60)

PANEL_DEFAULT_FLOAT = {   # (w, h) used the first time a panel floats
    "outliner": (OUTLINER_W, 360),
    "details": (OUTLINER_W, DETAILS_H),
    "browser": (760, BROWSER_H),
}
RESOLUTIONS = ((1280, 720), (1440, 810), (1600, 900), (1920, 1080))
SETTINGS_SIZE = (380, 226)


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


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
    _MENU_NAMES = ("File", "Edit", "Window", "Help")
    _WINDOW_PANEL_LABELS = {"Outliner": "outliner", "Details": "details",
                            "Content Browser": "browser"}
    _MENU_HOTKEYS = {
        "File": {"Save": "Ctrl+S"},
        "Edit": {"Duplicate": "Ctrl+D", "Delete": "Del", "Focus Selection": "F"},
    }

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
        self.gizmo_mode = "translate"  # W/E/R select translate / rotate / scale
        self.mat_ui = None          # open MaterialEditorUI, or None
        self.status = ("", 0.0)     # transient message near the content browser
        self.save_flash = 0.0
        # renderer preference for *next launch* (live switching is out of
        # scope); defaults to whatever this session actually ended up using
        self.renderer_pref = "gpu" if eng.gl_renderer is not None else "cpu"

        # ---- dockable panel state ----
        self.dock_order = {"left": [], "right": ["outliner", "details"],
                           "bottom": ["browser"]}
        self.floating = []          # panel ids currently floating, z-order (front=last)
        self.panel_visible = {"outliner": True, "details": True, "browser": True}
        self.float_rect = {}        # panel id -> pygame.Rect, used only while floating
        self.panel_drag = None      # {"id","dx","dy","w","h"} while dragging a title bar

        # ---- menu bar / dialogs ----
        self.open_menu = None       # name of the open dropdown, or None
        self.settings_open = False
        self.settings_drag = None   # "pixel" / "max_fps" while dragging a settings slider
        self.show_controls_overlay = False

        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 14)
        self.font_small = pygame.font.SysFont("consolas,couriernew,monospace", 12)
        self.icons = {}
        count = max(len(lib.assets), 1)
        for i, a in enumerate(lib.assets):
            eng.loading_step(f"rendering thumbnail: {a.name}", 0.25 + 0.45 * i / count)
            self.icons[a.name] = make_icon(engine_mod, a)

    # ---- layout: the single source of truth for panel/viewport rects ----
    def _float_rect_for(self, pid, w, h):
        import pygame
        r = self.float_rect.get(pid)
        if r is None:
            fw, fh = PANEL_DEFAULT_FLOAT[pid]
            r = pygame.Rect(max(0, (w - fw) // 2), MENU_H + 40, fw, fh)
            self.float_rect[pid] = r
        r.width = min(r.width, max(160, w - 20))
        r.height = min(r.height, max(100, h - MENU_H - 20))
        r.x = min(max(r.x, 0), max(0, w - r.width))
        r.y = min(max(r.y, MENU_H), max(MENU_H, h - r.height))
        return r

    def _layout(self, w, h):
        import pygame
        menu = pygame.Rect(0, 0, w, MENU_H)
        left_ids = [p for p in self.dock_order["left"] if self.panel_visible.get(p, True)]
        right_ids = [p for p in self.dock_order["right"] if self.panel_visible.get(p, True)]
        bottom_ids = [p for p in self.dock_order["bottom"] if self.panel_visible.get(p, True)]
        left_w = OUTLINER_W if left_ids else 0
        right_w = OUTLINER_W if right_ids else 0
        bottom_h = BROWSER_H if bottom_ids else 0
        top = MENU_H
        stack_bottom = h - bottom_h
        panels = {}

        def stack(ids, x, width):
            n = len(ids)
            if n == 0:
                return
            avail = max(0, stack_bottom - top)
            share = avail // n
            y = top
            for i, pid in enumerate(ids):
                hh = share if i < n - 1 else avail - share * (n - 1)
                panels[pid] = pygame.Rect(x, y, width, max(0, hh))
                y += hh

        stack(left_ids, 0, left_w)
        stack(right_ids, w - right_w, right_w)
        if bottom_ids:
            bw_total = max(0, w - left_w - right_w)
            n = len(bottom_ids)
            share = bw_total // n
            x = left_w
            for i, pid in enumerate(bottom_ids):
                ww = share if i < n - 1 else bw_total - share * (n - 1)
                panels[pid] = pygame.Rect(x, stack_bottom, max(0, ww), bottom_h)
                x += ww

        for pid in ("outliner", "details", "browser"):
            if not self.panel_visible.get(pid, True) or pid in panels:
                continue
            panels[pid] = self._float_rect_for(pid, w, h)

        viewport = pygame.Rect(left_w, top, max(0, w - left_w - right_w),
                               max(0, stack_bottom - top))
        return {"menu": menu, "viewport": viewport, "panels": panels,
                "left_w": left_w, "right_w": right_w, "bottom_h": bottom_h}

    def _panel_content_rect(self, pid, layout):
        import pygame
        r = layout["panels"].get(pid)
        if r is None:
            return None
        return pygame.Rect(r.x, r.y + PANEL_TITLE_H, r.width,
                           max(0, r.height - PANEL_TITLE_H))

    def _hit_panel(self, pos, layout):
        panels = layout["panels"]
        for pid in reversed(self.floating):
            r = panels.get(pid)
            if r is not None and r.collidepoint(pos):
                return pid
        for pid, r in panels.items():
            if pid in self.floating:
                continue
            if r.collidepoint(pos):
                return pid
        return None

    def over_ui(self, pos) -> bool:
        if self.mat_ui is not None or self.settings_open:
            return True
        if pos[1] < MENU_H:
            return True
        w, h = self.eng.screen.get_size()
        if self.open_menu is not None:
            drop, _rows = self._dropdown_geom(self.open_menu, w)
            if drop.collidepoint(pos):
                return True
        return self._hit_panel(pos, self._layout(w, h)) is not None

    def _outliner_rows(self):
        return [e for e in self.scene.entities
                if e.mesh is not None or e.light is not None
                or e.environment is not None]

    def _copy_entity_state(self, src, dst) -> None:
        """Carry per-entity edited state from src onto a freshly instantiated dst.

        Duplication (Ctrl+D) re-instantiates the asset from scratch, so anything
        tuned afterwards in the Details panel (light overrides) or the material
        editor (node graph) has to be copied over explicitly. This is the one
        place that happens, so future per-entity state has a single home.
        """
        dst.visible = src.visible
        dst.casts_shadow = src.casts_shadow
        dst.collidable = src.collidable

        if src.light is not None and dst.light is not None:
            # a Flicker overwrites intensity every frame; its captured base is
            # the value the user actually tuned (see engine.assets._entity_dict)
            flicker_base = None
            for b in src.behaviors:
                if isinstance(b, self.engine_mod.behaviors.Flicker) \
                        and hasattr(b, "base"):
                    flicker_base = b.base
            for key in self.engine_mod.assets._LIGHT_PROPS:
                if hasattr(src.light, key) and hasattr(dst.light, key):
                    setattr(dst.light, key, getattr(src.light, key))
            if flicker_base is not None and hasattr(dst.light, "intensity"):
                dst.light.intensity = flicker_base

        if src.material is not None and dst.mesh is not None:
            dst.material = self.engine_mod.MaterialGraph.from_dict(
                src.material.to_dict())
            dst.material.apply(dst.mesh)

    # ---- transform gizmo: W/E/R select translate / rotate / scale ----
    _GIZMO_AXES = (((1.0, 0.0, 0.0), (225, 85, 85)),
                   ((0.0, 1.0, 0.0), (105, 215, 105)),
                   ((0.0, 0.0, 1.0), (95, 145, 250)))

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
        return pygame.Rect(rect.x + 6, rect.y + DETAILS_ROWS_TOP + i * DETAIL_ROW_H,
                           rect.width - 12, DETAIL_ROW_H - 2)

    def _slider_track(self, row_rect):
        return (row_rect.x + 96, row_rect.right - 52)

    def _apply_slider(self, row, row_rect, mx):
        x0, x1 = self._slider_track(row_rect)
        f = min(max((mx - x0) / max(x1 - x0, 1), 0.0), 1.0)
        row["set"](row["min"] + f * (row["max"] - row["min"]))
        self.dirty = True

    # ---- menu bar ----
    def _menu_title_rects(self, w):
        import pygame
        rects = {}
        x = 8
        for name in self._MENU_NAMES:
            tw = self.font_small.size(name)[0] + 20
            rects[name] = pygame.Rect(x, 0, tw, MENU_H)
            x += tw
        return rects

    def _menu_defs(self):
        return {
            "File": [
                ("New Scene", self._new_scene, True),
                ("Open Scene...", self._open_scene_dialog, True),
                ("Save", self._save_scene, True),
                ("Save As...", self._save_scene_as_dialog, True),
                ("Import FBX...", self._import_fbx_dialog, True),
                ("Exit", self._quit, True),
            ],
            "Edit": [
                ("Duplicate", self._duplicate_selected, True),
                ("Delete", self._delete_selected, True),
                ("Focus Selection", self._focus_selection, True),
            ],
            "Window": [
                ("Outliner", lambda: self._toggle_panel("outliner"), True),
                ("Details", lambda: self._toggle_panel("details"), True),
                ("Content Browser", lambda: self._toggle_panel("browser"), True),
                ("Settings...", self._open_settings, True),
                ("Reset Layout", self._reset_layout, True),
            ],
            "Help": [
                ("Controls", self._toggle_controls, True),
                ("About", self._show_about, True),
            ],
        }

    def _dropdown_geom(self, name, w):
        import pygame
        items = self._menu_defs()[name]
        tr = self._menu_title_rects(w)[name]
        item_h = 22
        width = 210
        x, y = tr.x, MENU_H
        dropdown_rect = pygame.Rect(x, y, width, 6 + item_h * len(items))
        rows = []
        for i, (label, action, enabled) in enumerate(items):
            r = pygame.Rect(x + 2, y + 3 + i * item_h, width - 4, item_h)
            rows.append((label, r, action, enabled))
        return dropdown_rect, rows

    def _handle_menu_click(self, mp, w) -> bool:
        title_rects = self._menu_title_rects(w)
        if self.open_menu is not None:
            _drop, rows = self._dropdown_geom(self.open_menu, w)
            for _label, r, action, enabled in rows:
                if r.collidepoint(mp):
                    self.open_menu = None
                    if enabled:
                        action()
                    return True
            for name, r in title_rects.items():
                if r.collidepoint(mp):
                    self.open_menu = None if name == self.open_menu else name
                    return True
            self.open_menu = None
            return True
        for name, r in title_rects.items():
            if r.collidepoint(mp):
                self.open_menu = name
                return True
        return False

    # ---- panel docking / floating ----
    def _dock_panel(self, pid, side) -> None:
        for s in ("left", "right", "bottom"):
            if pid in self.dock_order[s]:
                self.dock_order[s].remove(pid)
        if pid in self.floating:
            self.floating.remove(pid)
        if side == "float":
            self.floating.append(pid)
        else:
            self.dock_order[side].append(pid)
        self._save_settings()

    def _begin_panel_drag(self, pid, mp, rect) -> None:
        self.panel_drag = {"id": pid, "dx": mp[0] - rect.x, "dy": mp[1] - rect.y,
                           "w": rect.width, "h": rect.height}

    def _finish_panel_drag(self, mp, w, h) -> None:
        import pygame
        g = self.panel_drag
        pid = g["id"]
        gx, gy = mp[0] - g["dx"], mp[1] - g["dy"]
        if pid in ("outliner", "details") and mp[0] <= EDGE_SNAP:
            self._dock_panel(pid, "left")
        elif pid in ("outliner", "details") and mp[0] >= w - EDGE_SNAP:
            self._dock_panel(pid, "right")
        elif pid == "browser" and mp[1] >= h - EDGE_SNAP:
            self._dock_panel(pid, "bottom")
        else:
            self.float_rect[pid] = pygame.Rect(gx, gy, g["w"], g["h"])
            self._dock_panel(pid, "float")
            self._float_rect_for(pid, w, h)  # clamp on-screen
        self.panel_drag = None

    def _toggle_panel(self, pid) -> None:
        self.panel_visible[pid] = not self.panel_visible.get(pid, True)
        self._save_settings()

    def _reset_layout(self) -> None:
        self.dock_order = {"left": [], "right": ["outliner", "details"],
                           "bottom": ["browser"]}
        self.floating = []
        self.panel_visible = {"outliner": True, "details": True, "browser": True}
        self.float_rect = {}
        self._save_settings()

    # ---- settings dialog ----
    def _open_settings(self) -> None:
        self.settings_open = True
        self.settings_drag = None

    def _toggle_controls(self) -> None:
        self.show_controls_overlay = not self.show_controls_overlay

    def _show_about(self) -> None:
        self.status = ("PyEngine 0.1 — pure-Python real-time 3D engine", 4.0)

    def _settings_rect(self, w, h):
        import pygame
        sw, sh = SETTINGS_SIZE
        return pygame.Rect((w - sw) // 2, max(MENU_H + 20, (h - sh) // 2), sw, sh)

    def _settings_res_buttons(self, rect):
        import pygame
        out = []
        x, y = rect.x + 12, rect.y + 54
        bw, bh, gap = 82, 24, 8
        for rw, rh in RESOLUTIONS:
            out.append(((rw, rh), pygame.Rect(x, y, bw, bh)))
            x += bw + gap
        return out

    def _settings_slider_row(self, rect, which):
        import pygame
        y = rect.y + (92 if which == "pixel" else 126)
        return pygame.Rect(rect.x + 12, y, rect.width - 24, 24)

    def _settings_slider_track(self, row):
        return (row.x + 100, row.right - 50)

    def _settings_renderer_buttons(self, rect):
        import pygame
        x, y = rect.x + 100, rect.y + 160
        bw, bh, gap = 60, 22, 8
        return [("gpu", pygame.Rect(x, y, bw, bh)),
                ("cpu", pygame.Rect(x + bw + gap, y, bw, bh))]

    def _apply_settings_slider(self, which, row, mx) -> None:
        x0, x1 = self._settings_slider_track(row)
        f = min(max((mx - x0) / max(x1 - x0, 1), 0.0), 1.0)
        if which == "pixel":
            v = int(round(1 + f * 5))
            self.eng.renderer.render_scale = max(1, v)
        else:
            v = int(round(30 + f * 210))
            self.eng.max_fps = max(1, v)

    def _update_settings(self, engine, w, h) -> None:
        import pygame
        inp = engine.input
        mp = inp.mouse_pos
        rect = self._settings_rect(w, h)
        close = pygame.Rect(rect.right - 26, rect.y + 5, 20, 20)
        if inp.mouse_button_pressed(1):
            if close.collidepoint(mp):
                self.settings_open = False
                return
            for (rw, rh), btn in self._settings_res_buttons(rect):
                if btn.collidepoint(mp):
                    self.eng.set_resolution(rw, rh)
                    self._save_settings()
                    return
            for key, btn in self._settings_renderer_buttons(rect):
                if btn.collidepoint(mp):
                    self.renderer_pref = key
                    self._save_settings()
                    self.status = ("Renderer preference saved — restart to apply", 3.0)
                    return
            for which in ("pixel", "max_fps"):
                if which == "pixel" and self.eng.gl_renderer is not None:
                    continue  # pixel scale only affects the software per-pixel pass
                row = self._settings_slider_row(rect, which)
                if row.collidepoint(mp):
                    self.settings_drag = which
                    self._apply_settings_slider(which, row, mp[0])
                    return
        if self.settings_drag is not None:
            if inp.mouse_held(1):
                row = self._settings_slider_row(rect, self.settings_drag)
                self._apply_settings_slider(self.settings_drag, row, mp[0])
            else:
                self.settings_drag = None
                self._save_settings()

    # ---- settings.json persistence ----
    def _settings_dict(self) -> dict:
        w, h = self.eng.screen.get_size()
        return {
            "width": w, "height": h,
            "pixel_scale": int(self.eng.renderer.render_scale),
            "max_fps": int(self.eng.max_fps),
            "renderer": self.renderer_pref,
            "panel_visible": dict(self.panel_visible),
            "dock_order": {side: list(ids) for side, ids in self.dock_order.items()},
            "floating": list(self.floating),
            "float_rect": {pid: [r.x, r.y, r.width, r.height]
                          for pid, r in self.float_rect.items()},
        }

    def _save_settings(self) -> None:
        save_settings(self._settings_dict())

    def _apply_layout_settings(self, data: dict) -> None:
        import pygame
        valid = {"outliner", "details", "browser"}
        dock_order = data.get("dock_order", {})
        left = [p for p in dock_order.get("left", []) if p in ("outliner", "details")]
        right = [p for p in dock_order.get("right", []) if p in ("outliner", "details")]
        bottom = [p for p in dock_order.get("bottom", []) if p == "browser"]
        floating = [p for p in data.get("floating", []) if p in valid]
        placed = left + right + bottom + floating
        if set(placed) != valid or len(placed) != len(valid):
            return  # partial/corrupt layout data: keep the built-in default
        self.dock_order = {"left": left, "right": right, "bottom": bottom}
        self.floating = floating
        pv = data.get("panel_visible", {})
        for pid in valid:
            if pid in pv:
                self.panel_visible[pid] = bool(pv[pid])
        for pid, v in data.get("float_rect", {}).items():
            if pid in valid and isinstance(v, list) and len(v) == 4:
                self.float_rect[pid] = pygame.Rect(*v)

    # ---- File / Edit menu actions (shared with hotkeys) ----
    def _duplicate_selected(self) -> None:
        src = self.selected
        if src is None or src.asset_name is None:
            return
        Vec3 = self.engine_mod.Vec3
        dup = self.lib.instantiate(src.asset_name)
        t, s = dup.transform, src.transform
        t.position = Vec3(s.position.x + 0.8, s.position.y, s.position.z + 0.8)
        t.rotation = Vec3(s.rotation.x, s.rotation.y, s.rotation.z)
        t.scale = Vec3(s.scale.x, s.scale.y, s.scale.z)
        self._copy_entity_state(src, dup)
        self.scene.add(dup)
        self.selected = dup
        self.dirty = True

    def _delete_selected(self) -> None:
        if self.selected is None or self.selected.asset_name is None:
            return
        self.scene.remove(self.selected)
        self.selected = None
        self.dirty = True

    def _focus_selection(self) -> None:
        if self.selected is not None:
            self._focus(self.selected)

    def _save_scene(self) -> None:
        os.makedirs(os.path.dirname(self.scene_path) or ".", exist_ok=True)
        self.engine_mod.save_scene(self.scene, self.camera, self.scene_path)
        self.dirty = False
        self.save_flash = 1.5

    def _quit(self) -> None:
        import pygame
        pygame.event.post(pygame.event.Event(pygame.QUIT))

    def _replace_scene_content(self, new_scene) -> None:
        """Mutate the LIVE scene in place with new_scene's content.

        Engine.run() holds the Scene object passed to it in a local variable
        and behaviors/renderer iterate that exact object, so New/Open Scene
        can't just rebind self.scene — the entity list has to be cleared and
        refilled, and the editor-owned entities (flashlight, __camera,
        __editor — identified by asset_name is None) carried over untouched.
        """
        editor_owned = [e for e in self.scene.entities if e.asset_name is None]
        self.scene.entities.clear()
        self.scene.entities.extend(new_scene.entities)
        self.scene.entities.extend(editor_owned)
        self.scene.light = new_scene.light
        self.scene.fog = new_scene.fog
        self.scene.sky = new_scene.sky
        self.scene.background = new_scene.background
        self.scene.enable_shadows = new_scene.enable_shadows
        self.selected = None
        self.dirty = False

    def _new_scene(self) -> None:
        Vec3 = self.engine_mod.Vec3
        self._replace_scene_content(self.engine_mod.Scene())
        self.scene_path = os.path.join(BASE_DIR, "scenes", "untitled.json")
        self.camera.position = Vec3(6.0, 2.6, 9.0)
        self.camera.yaw, self.camera.pitch = 0.45, -0.08
        self.dirty = True
        self.status = ("new scene", 3.0)

    def _open_scene_dialog(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Open scene", initialdir=os.path.join(BASE_DIR, "scenes"),
                filetypes=[("Scene files", "*.json"), ("All files", "*.*")])
            root.destroy()
        except Exception as ex:
            self.status = (f"file dialog unavailable: {ex}", 5.0)
            return
        if not path:
            return
        try:
            loaded = self.engine_mod.load_scene(path, self.lib, self.camera)
        except Exception as ex:
            self.status = (f"open failed: {ex}", 6.0)
            return
        self._replace_scene_content(loaded)
        self.scene_path = path
        self.status = (f"opened '{os.path.basename(path)}'", 3.0)

    def _save_scene_as_dialog(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.asksaveasfilename(
                title="Save scene as", initialdir=os.path.join(BASE_DIR, "scenes"),
                defaultextension=".json", filetypes=[("Scene files", "*.json")])
            root.destroy()
        except Exception as ex:
            self.status = (f"file dialog unavailable: {ex}", 5.0)
            return
        if not path:
            return
        self.scene_path = path
        self._save_scene()

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

    # ---- per-frame logic (runs in a fixed update step) ----
    def update(self, engine, dt: float) -> None:
        import pygame
        inp = engine.input
        mp = inp.mouse_pos
        w, h = engine.screen.get_size()
        self.save_flash = max(0.0, self.save_flash - dt)
        if self.status[1] > 0:
            self.status = (self.status[0], self.status[1] - dt)

        if self.mat_ui is not None:  # node editor captures all editor input
            self.mat_ui.update(engine, dt)
            return
        if self.settings_open:      # settings dialog captures all editor input
            self._update_settings(engine, w, h)
            return

        layout = self._layout(w, h)
        looking = self.fly is not None and self.fly.looking

        if inp.wheel and not looking:
            target = self._hit_panel(mp, layout)
            if target == "outliner":
                self.outliner_scroll = max(0, self.outliner_scroll - int(inp.wheel) * 3)
            elif target == "browser":
                self.browser_scroll = max(0, self.browser_scroll - int(inp.wheel) * 70)

        if inp.mouse_button_pressed(1):
            if not self._handle_menu_click(mp, w):
                target = self._hit_panel(mp, layout)
                if target is not None:
                    if target in self.floating:
                        self.floating.remove(target)
                        self.floating.append(target)
                    rect = layout["panels"][target]
                    title_rect = pygame.Rect(rect.x, rect.y, rect.width, PANEL_TITLE_H)
                    if title_rect.collidepoint(mp):
                        self._begin_panel_drag(target, mp, rect)
                    else:
                        content = self._panel_content_rect(target, layout)
                        self._route_panel_click(target, mp, content)
                elif layout["viewport"].collidepoint(mp):
                    self.active_slider = None
                    if not self._try_grab_gizmo(mp, w, h):
                        self._click_viewport(mp, w, h)

        if self.panel_drag is not None and not inp.mouse_held(1):
            self._finish_panel_drag(mp, w, h)

        # gizmo drag
        if self.gizmo_drag is not None:
            if inp.mouse_held(1) and self.selected is not None:
                self._update_gizmo_drag(mp)
            else:
                self.gizmo_drag = None

        # live slider drag (details panel)
        if self.active_slider is not None:
            rows = self._details_rows()
            content = self._panel_content_rect("details", layout)
            if inp.mouse_held(1) and content is not None and self.active_slider < len(rows):
                row = rows[self.active_slider]
                if row["kind"] == "slider":
                    self._apply_slider(row, self._detail_row_rect(content,
                                                                   self.active_slider),
                                       mp[0])
            else:
                self.active_slider = None

        if self.drag_asset is not None and inp.mouse_button_released(1):
            if not self.over_ui(mp):
                self._place_asset(self.drag_asset, mp, w, h)
            self.drag_asset = None

        ctrl = inp.held(pygame.K_LCTRL) or inp.held(pygame.K_RCTRL)
        if inp.pressed(pygame.K_DELETE):
            self._delete_selected()
        if ctrl and inp.pressed(pygame.K_d):
            self._duplicate_selected()
        if ctrl and inp.pressed(pygame.K_s):
            self._save_scene()
        if inp.pressed(pygame.K_f) and not looking:
            self._focus_selection()
        if inp.pressed(pygame.K_c) and self.fly is not None:
            self.fly.collide = not self.fly.collide
        if not looking:
            if inp.pressed(pygame.K_w):
                self.gizmo_mode, self.gizmo_drag = "translate", None
            elif inp.pressed(pygame.K_e):
                self.gizmo_mode, self.gizmo_drag = "rotate", None
            elif inp.pressed(pygame.K_r):
                self.gizmo_mode, self.gizmo_drag = "scale", None
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
        """Engine Esc hook: dropdown/settings, then material editor, then
        deselect, and only then let the engine quit."""
        if self.open_menu is not None:
            self.open_menu = None
            return True
        if self.settings_open:
            self.settings_open = False
            return True
        if self.mat_ui is not None:
            self.mat_ui.close()
            return True
        if self.selected is not None:
            self.selected = None
            return True
        return False

    def _route_panel_click(self, pid, mp, content) -> None:
        if pid == "outliner":
            self.active_slider = None
            self._click_outliner(mp, content)
        elif pid == "details":
            self._click_details(mp, content)
        elif pid == "browser":
            if self._import_btn_rect(content).collidepoint(mp):
                self._import_fbx_dialog()
            else:
                asset = self._tile_at(mp, content)
                if asset is not None:
                    self.drag_asset = asset

    def _click_details(self, mp, rect) -> None:
        rows = self._details_rows()
        i = (mp[1] - (rect.y + DETAILS_ROWS_TOP)) // DETAIL_ROW_H
        if not (0 <= i < len(rows)):
            return
        row = rows[i]
        if row["kind"] == "slider":
            self.active_slider = i
            self._apply_slider(row, self._detail_row_rect(rect, i), mp[0])
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

    def _click_outliner(self, mp, rect) -> None:
        rows = self._outliner_rows()
        i = (mp[1] - rect.y - 6) // ROW_H + self.outliner_scroll
        if 0 <= mp[1] - rect.y - 6 and 0 <= i < len(rows):
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
        layout = self._layout(w, h)
        self._draw_markers(surf, w, h)

        # backdrop so gaps between a side dock and the bottom dock (if both
        # are present) read as UI, not a hole showing the 3D scene through
        if layout["left_w"]:
            pygame.draw.rect(surf, PANEL_BG,
                             pygame.Rect(0, MENU_H, layout["left_w"], h - MENU_H))
        if layout["right_w"]:
            pygame.draw.rect(surf, PANEL_BG, pygame.Rect(
                w - layout["right_w"], MENU_H, layout["right_w"], h - MENU_H))
        if layout["bottom_h"]:
            pygame.draw.rect(surf, PANEL_BG, pygame.Rect(
                layout["left_w"], h - layout["bottom_h"],
                w - layout["left_w"] - layout["right_w"], layout["bottom_h"]))

        panels = dict(layout["panels"])
        drag_pid = self.panel_drag["id"] if self.panel_drag else None
        if drag_pid is not None and drag_pid in panels:
            mp = eng.input.mouse_pos
            g = self.panel_drag
            panels[drag_pid] = pygame.Rect(mp[0] - g["dx"], mp[1] - g["dy"],
                                           g["w"], g["h"])

        for pid in ("outliner", "details", "browser"):
            if pid in panels and pid not in self.floating and pid != drag_pid:
                self._draw_panel(surf, pid, panels[pid])
        for pid in self.floating:
            if pid in panels and pid != drag_pid:
                self._draw_panel(surf, pid, panels[pid])
        if drag_pid is not None and drag_pid in panels:
            self._draw_panel(surf, drag_pid, panels[drag_pid])

        if self.drag_asset is not None:
            icon = self.icons.get(self.drag_asset.name)
            if icon is not None:
                ghost = icon.copy()
                ghost.set_alpha(150)
                mp = eng.input.mouse_pos
                surf.blit(ghost, (mp[0] - ICON // 2, mp[1] - ICON // 2))

        self._draw_menu_bar(surf, w)
        if self.show_controls_overlay:
            self._draw_controls_overlay(surf, w, h)
        if self.settings_open:
            self._draw_settings(surf, w, h)
        if self.mat_ui is not None:
            self.mat_ui.draw(surf)

    def _panel_title(self, pid) -> str:
        if pid == "outliner":
            if self.save_flash > 0:
                return "World Outliner — saved ✓"
            name = os.path.basename(self.scene_path) + (" *" if self.dirty else "")
            return f"World Outliner — {name}"
        if pid == "details":
            return "Details"
        return "Content Browser"

    def _draw_panel(self, surf, pid, rect) -> None:
        import pygame
        pygame.draw.rect(surf, PANEL_BG, rect)
        pygame.draw.rect(surf, PANEL_EDGE, rect, 1)
        title_rect = pygame.Rect(rect.x, rect.y, rect.width, PANEL_TITLE_H)
        pygame.draw.rect(surf, (30, 33, 40), title_rect)
        pygame.draw.line(surf, PANEL_EDGE, (rect.x, rect.y + PANEL_TITLE_H),
                         (rect.right, rect.y + PANEL_TITLE_H))
        lab = self.font_small.render(self._panel_title(pid)[:44], True, TEXT)
        surf.blit(lab, (rect.x + 8, rect.y + 3))
        content = pygame.Rect(rect.x, rect.y + PANEL_TITLE_H, rect.width,
                              max(0, rect.height - PANEL_TITLE_H))
        if pid == "outliner":
            self._draw_outliner(surf, content)
        elif pid == "details":
            self._draw_details(surf, content)
        elif pid == "browser":
            self._draw_browser(surf, content)

    def _draw_menu_bar(self, surf, w) -> None:
        import pygame
        bar = pygame.Rect(0, 0, w, MENU_H)
        pygame.draw.rect(surf, (26, 28, 34), bar)
        pygame.draw.line(surf, PANEL_EDGE, (0, MENU_H), (w, MENU_H))
        mp = pygame.mouse.get_pos()
        title_rects = self._menu_title_rects(w)
        for name, r in title_rects.items():
            if name == self.open_menu:
                pygame.draw.rect(surf, SELECT_BG, r)
            elif r.collidepoint(mp):
                pygame.draw.rect(surf, HOVER_BG, r)
            lab = self.font_small.render(name, True, TEXT)
            surf.blit(lab, (r.x + 10, r.y + 6))

        if self.open_menu is None:
            return
        checks = self._WINDOW_PANEL_LABELS if self.open_menu == "Window" else {}
        hints = self._MENU_HOTKEYS.get(self.open_menu, {})
        drop, rows = self._dropdown_geom(self.open_menu, w)
        pygame.draw.rect(surf, (24, 26, 32), drop)
        pygame.draw.rect(surf, PANEL_EDGE, drop, 1)
        for label, r, _action, enabled in rows:
            if r.collidepoint(mp):
                pygame.draw.rect(surf, HOVER_BG, r)
            color = TEXT if enabled else TEXT_DIM
            if label in checks:
                pid = checks[label]
                box = pygame.Rect(r.x + 4, r.y + 5, 12, 12)
                pygame.draw.rect(surf, (48, 51, 60), box, border_radius=2)
                if self.panel_visible.get(pid, True):
                    pygame.draw.rect(surf, ACCENT, box.inflate(-4, -4), border_radius=2)
                surf.blit(self.font_small.render(label, True, color), (r.x + 22, r.y + 4))
            else:
                surf.blit(self.font_small.render(label, True, color), (r.x + 8, r.y + 4))
            hint = hints.get(label)
            if hint:
                hl = self.font_small.render(hint, True, TEXT_DIM)
                surf.blit(hl, (r.right - hl.get_width() - 8, r.y + 4))

    _CONTROLS_LINES = (
        "RMB (hold) - mouse look + fly: WASD move, Q/E or Space/Ctrl down/up, "
        "wheel = fly speed, Shift = fast",
        "LMB - select / drag assets & gizmo / drag panel title bars to move them",
        "W / E / R - gizmo mode: translate / rotate / scale  (only while not looking)",
        ", / . - rotate selection 15 deg        - / = - scale selection",
        "F - focus camera on selection  (only while not looking)",
        "Ctrl+D - duplicate selection           Del - delete selection",
        "Ctrl+S - save scene",
        "L - toggle flashlight                  C - toggle player collision",
        "M - open material editor for the selected mesh",
        "F1 - wireframe   F2 - per-pixel/flat shading   H - toggle HUD",
        "Esc - close menu/dialog, else deselect, else quit",
    )

    def _draw_controls_overlay(self, surf, w, h) -> None:
        import pygame
        pad, line_h = 16, 20
        box_w = min(w - 80, 640)
        box_h = pad * 2 + 28 + len(self._CONTROLS_LINES) * line_h
        rect = pygame.Rect((w - box_w) // 2, max(MENU_H + 10, (h - box_h) // 2),
                           box_w, box_h)
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 140))
        surf.blit(shade, (0, 0))
        pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(surf, PANEL_EDGE, rect, 1, border_radius=6)
        surf.blit(self.font.render("Controls", True, TEXT), (rect.x + pad, rect.y + pad))
        y = rect.y + pad + 28
        for line in self._CONTROLS_LINES:
            surf.blit(self.font_small.render(line, True, TEXT_DIM), (rect.x + pad, y))
            y += line_h

    def _draw_settings(self, surf, w, h) -> None:
        import pygame
        rect = self._settings_rect(w, h)
        pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(surf, PANEL_EDGE, rect, 1, border_radius=6)
        surf.blit(self.font.render("Settings", True, TEXT), (rect.x + 12, rect.y + 8))
        close = pygame.Rect(rect.right - 26, rect.y + 5, 20, 20)
        mp = pygame.mouse.get_pos()
        pygame.draw.rect(surf, (60, 34, 34) if not close.collidepoint(mp) else (90, 44, 44),
                         close, border_radius=4)
        surf.blit(self.font_small.render("X", True, (230, 160, 160)),
                  (close.x + 7, close.y + 4))

        surf.blit(self.font_small.render("Resolution", True, TEXT_DIM),
                  (rect.x + 12, rect.y + 38))
        cur_size = self.eng.screen.get_size()
        for (rw, rh), btn in self._settings_res_buttons(rect):
            active = (cur_size == (rw, rh))
            pygame.draw.rect(surf, SELECT_BG if active else (33, 36, 44), btn,
                             border_radius=4)
            pygame.draw.rect(surf, PANEL_EDGE, btn, 1, border_radius=4)
            lab = self.font_small.render(f"{rw}x{rh}", True, TEXT)
            surf.blit(lab, (btn.x + (btn.width - lab.get_width()) // 2, btn.y + 5))

        if self.eng.gl_renderer is None:
            self._draw_settings_slider(surf, rect, "pixel", "pixel scale",
                                       self.eng.renderer.render_scale, 1, 6)
        self._draw_settings_slider(surf, rect, "max_fps", "max fps",
                                   self.eng.max_fps, 30, 240)

        surf.blit(self.font_small.render("Renderer (restart)", True, TEXT_DIM),
                  (rect.x + 12, rect.y + 165 - 22))
        for key, btn in self._settings_renderer_buttons(rect):
            active = self.renderer_pref == key
            pygame.draw.rect(surf, SELECT_BG if active else (33, 36, 44), btn,
                             border_radius=4)
            pygame.draw.rect(surf, PANEL_EDGE, btn, 1, border_radius=4)
            lab = self.font_small.render(key.upper(), True, TEXT)
            surf.blit(lab, (btn.x + (btn.width - lab.get_width()) // 2, btn.y + 4))

    def _draw_settings_slider(self, surf, rect, which, label, value, lo, hi) -> None:
        import pygame
        row = self._settings_slider_row(rect, which)
        surf.blit(self.font_small.render(label, True, TEXT_DIM), (row.x, row.y + 5))
        x0, x1 = self._settings_slider_track(row)
        cy = row.y + row.height // 2
        f = min(max((value - lo) / (hi - lo), 0.0), 1.0)
        pygame.draw.line(surf, (48, 51, 60), (x0, cy), (x1, cy), 4)
        kx = int(x0 + f * (x1 - x0))
        pygame.draw.line(surf, ACCENT, (x0, cy), (kx, cy), 4)
        pygame.draw.circle(surf, (235, 235, 240), (kx, cy), 5)
        surf.blit(self.font_small.render(str(int(round(value))), True, TEXT),
                  (x1 + 8, row.y + 5))

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
        rows = self._outliner_rows()
        top_pad = 6
        visible = max(0, (rect.height - top_pad - 20)) // ROW_H
        self.outliner_scroll = max(0, min(self.outliner_scroll, max(0, len(rows) - visible)))
        mp = pygame.mouse.get_pos()
        y = rect.y + top_pad
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
        hint = self.font_small.render("Del delete · Ctrl+D dup · F focus · Ctrl+S save",
                                      True, TEXT_DIM)
        surf.blit(hint, (rect.x + 10, rect.bottom - 18))

    def _draw_details(self, surf, rect) -> None:
        import pygame
        e = self.selected
        if e is None:
            surf.blit(self.font_small.render("select an entity", True, TEXT_DIM),
                      (rect.x + 10, rect.y + 8))
            return
        head = f"{e.name}" + (f"  ({e.asset_name})" if e.asset_name else "")
        surf.blit(self.font_small.render(head[:34], True, TEXT), (rect.x + 10, rect.y + 6))
        p = e.transform.position
        pos_text = f"x {p.x:.1f}  y {p.y:.1f}  z {p.z:.1f}"
        surf.blit(self.font_small.render(pos_text, True, TEXT_DIM),
                  (rect.x + 10, rect.y + 20))

        rows = self._details_rows()
        if not rows:
            surf.blit(self.font_small.render("no light on this entity", True, TEXT_DIM),
                      (rect.x + 10, rect.y + DETAILS_ROWS_TOP + 8))
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
        mp = pygame.mouse.get_pos()
        btn = self._import_btn_rect(rect)
        pygame.draw.rect(surf, HOVER_BG if btn.collidepoint(mp) else (33, 36, 44),
                         btn, border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, btn, 1, border_radius=4)
        label = self.font_small.render("+ Import FBX", True, ACCENT)
        surf.blit(label, (btn.x + (btn.width - label.get_width()) // 2, btn.y + 4))
        if self.status[1] > 0:
            msg = self.font_small.render(self.status[0][:60], True, (235, 210, 140))
            surf.blit(msg, (rect.x + 10, rect.y + 6))
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
    the 3D viewport behind the panel is a live preview. Floating only — drag
    its 18px title bar to move it, click X (or M/Esc) to close.
    """

    PALETTE = ("color", "position", "normal", "checker", "noise",
               "gradient", "mix", "multiply")
    DEFAULT_SIZE = (900, 560)

    def __init__(self, editor: Editor, entity):
        self.editor = editor
        self.entity = entity
        if entity.material is None:
            entity.material = editor.engine_mod.MaterialGraph()
        self.graph = entity.material
        self.pos = [60, 50]
        self.size = list(self.DEFAULT_SIZE)
        self.drag_node = None    # (node_id, grab_dx, grab_dy)
        self.drag_link = None    # source node id while dragging a new wire
        self.drag_param = None   # (node_id, param_name)
        self.drag_title = None   # (grab_dx, grab_dy) while dragging the title bar
        self._spawn_i = 0

    def close(self) -> None:
        self.editor.mat_ui = None

    # ---- geometry ----
    def rect(self, w, h):
        import pygame
        sw = min(self.size[0], max(300, w - 40))
        sh = min(self.size[1], max(200, h - 40))
        x = min(max(self.pos[0], 0), max(0, w - sw))
        y = min(max(self.pos[1], MENU_H), max(MENU_H, h - sh))
        self.pos = [x, y]
        return pygame.Rect(x, y, sw, sh)

    def content_rect(self, w, h):
        import pygame
        outer = self.rect(w, h)
        return pygame.Rect(outer.x, outer.y + PANEL_TITLE_H, outer.width,
                           max(0, outer.height - PANEL_TITLE_H))

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
        outer = self.rect(w, h)
        panel = self.content_rect(w, h)
        title_bar = pygame.Rect(outer.x, outer.y, outer.width, PANEL_TITLE_H)
        close = pygame.Rect(outer.right - 24, outer.y + 2, 16, 16)

        if inp.pressed(pygame.K_m):
            self.close()
            return
        if inp.mouse_button_pressed(1):
            if close.collidepoint(mp):
                self.close()
                return
            if title_bar.collidepoint(mp):
                self.drag_title = (mp[0] - outer.x, mp[1] - outer.y)
            else:
                self._press(mp, panel)
        if inp.mouse_held(1):
            if self.drag_title is not None:
                dx, dy = self.drag_title
                self.pos = [mp[0] - dx, mp[1] - dy]
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
            self.drag_node = self.drag_param = self.drag_link = self.drag_title = None

    def _press(self, mp, panel) -> None:
        import pygame
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
        outer = self.rect(w, h)
        panel = self.content_rect(w, h)
        pygame.draw.rect(surf, (18, 20, 25), outer)
        pygame.draw.rect(surf, PANEL_EDGE, outer, 1)
        title_bar = pygame.Rect(outer.x, outer.y, outer.width, PANEL_TITLE_H)
        pygame.draw.rect(surf, (30, 33, 40), title_bar)
        pygame.draw.line(surf, PANEL_EDGE, (outer.x, outer.y + PANEL_TITLE_H),
                         (outer.right, outer.y + PANEL_TITLE_H))
        title = self.editor.font_small.render(
            f"Material — {self.entity.name}   (drag ports to wire, M/Esc close)",
            True, TEXT)
        surf.blit(title, (outer.x + 8, outer.y + 3))
        close = pygame.Rect(outer.right - 24, outer.y + 2, 16, 16)
        pygame.draw.rect(surf, (60, 34, 34), close, border_radius=3)
        x_lab = self.editor.font_small.render("X", True, (230, 160, 160))
        surf.blit(x_lab, (close.x + 4, close.y + 1))

        for t, r in self._palette_rects(panel):
            hov = r.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(surf, HOVER_BG if hov else (30, 33, 40), r,
                             border_radius=4)
            lab = self.editor.font_small.render("+" + t, True, TEXT)
            surf.blit(lab, (r.x + 6, r.y + 4))

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
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--scene", default=os.path.join(BASE_DIR, "scenes", "scene.json"))
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--screenshot", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--pixel-scale", type=int, default=None,
                        help="per-pixel lighting internal resolution divisor "
                             "(lower = sharper, slower; default 4, or from "
                             "settings.json if present)")
    parser.add_argument("--gpu", action="store_true",
                        help="force the GPU (moderngl) renderer")
    parser.add_argument("--cpu", action="store_true",
                        help="force the software renderer")
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import engine
    import pygame

    settings = load_settings()
    width = args.width if args.width is not None else settings.get("width", 1440)
    height = args.height if args.height is not None else settings.get("height", 810)
    pixel_scale = (args.pixel_scale if args.pixel_scale is not None
                   else settings.get("pixel_scale", 4))
    max_fps = settings.get("max_fps", 120)

    gpu_mode = "auto"
    renderer_pref = settings.get("renderer")
    if renderer_pref == "gpu":
        gpu_mode = True
    elif renderer_pref == "cpu":
        gpu_mode = False
    if args.gpu:
        gpu_mode = True
    elif args.cpu:
        gpu_mode = False
    if args.headless:
        gpu_mode = False  # the SDL dummy driver has no GL surface to attach to

    eng = engine.Engine(width, height, title="PyEngine Editor", max_fps=max_fps, gpu=gpu_mode)
    eng.renderer.render_scale = max(1, pixel_scale)
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
    editor._apply_layout_settings(settings)
    if renderer_pref in ("gpu", "cpu"):  # show the saved preference even if it
        editor.renderer_pref = renderer_pref  # didn't match what actually ran

    flashlight = engine.Entity("flashlight", light=engine.SpotLight(
        color=(255, 244, 214), intensity=2.0, range=24.0, radius=0.25,
        inner=13.0, outer=27.0, shadow_samples=2, shadow_interval=2))
    flashlight.casts_shadow = False
    flashlight.add_behavior(engine.behaviors.FlashlightController(camera, toggle_key=pygame.K_l))
    scene.add(flashlight)
    editor.flashlight = flashlight

    fly = engine.behaviors.FlyController(
        camera, look_buttons=(3,), look_guard=lambda pos: not editor.over_ui(pos),
        collide=True, move_requires_look=True)
    editor.fly = fly
    scene.add(engine.Entity("__camera").add_behavior(fly))
    scene.add(engine.Entity("__editor").add_behavior(EditorBehavior(editor)))

    # trace the static lights' shadows now so the first frame doesn't hitch
    eng.loading_step("pre-tracing shadows", 0.8)
    eng.tracer.refresh(scene)
    eng.renderer.render(pygame.Surface((320, 180)), scene, camera, eng.tracer)

    eng.loading_step("opening world", 0.95)
    eng.esc_handler = editor.handle_escape
    eng.hud_text = ("RMB: look/fly (WASD/QE/Space/Ctrl, wheel=speed) | LMB: select/gizmo/panels | "
                    "W/E/R gizmo mode | M material | L flashlight | C collision | F focus | "
                    "Ctrl+D dup | Del delete | Ctrl+S save | F1/F2 shading | H hud | Esc back/quit")
    eng.run(scene, camera, max_frames=args.frames, screenshot_path=args.screenshot,
            overlay=editor.draw)


if __name__ == "__main__":
    main()
