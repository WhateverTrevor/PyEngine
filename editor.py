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
ROW_H = 20
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
        self.dirty = False
        self.outliner_scroll = 0
        self.browser_scroll = 0
        self.drag_asset = None
        self.save_flash = 0.0
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 14)
        self.font_small = pygame.font.SysFont("consolas,couriernew,monospace", 12)
        self.icons = {a.name: make_icon(engine_mod, a) for a in lib.assets}

    # ---- layout ----
    def outliner_rect(self, w, h):
        import pygame
        return pygame.Rect(w - OUTLINER_W, 0, OUTLINER_W, h - BROWSER_H)

    def browser_rect(self, w, h):
        import pygame
        return pygame.Rect(0, h - BROWSER_H, w, BROWSER_H)

    def over_ui(self, pos) -> bool:
        w, h = self.eng.screen.get_size()
        return (self.outliner_rect(w, h).collidepoint(pos)
                or self.browser_rect(w, h).collidepoint(pos))

    def _outliner_rows(self):
        return [e for e in self.scene.entities
                if e.mesh is not None or e.light is not None]

    # ---- per-frame logic (runs in a fixed update step) ----
    def update(self, engine, dt: float) -> None:
        import pygame
        inp = engine.input
        mp = inp.mouse_pos
        w, h = engine.screen.get_size()
        orect = self.outliner_rect(w, h)
        brect = self.browser_rect(w, h)
        self.save_flash = max(0.0, self.save_flash - dt)

        if inp.wheel:
            if orect.collidepoint(mp):
                self.outliner_scroll = max(0, self.outliner_scroll - int(inp.wheel) * 3)
            elif brect.collidepoint(mp):
                self.browser_scroll = max(0, self.browser_scroll - int(inp.wheel) * 70)

        if inp.mouse_button_pressed(1):
            if orect.collidepoint(mp):
                self._click_outliner(mp, orect)
            elif brect.collidepoint(mp):
                asset = self._tile_at(mp, brect)
                if asset is not None:
                    self.drag_asset = asset
            else:
                self._click_viewport(mp, w, h)

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
        self._draw_browser(surf, self.browser_rect(w, h))
        if self.drag_asset is not None:
            icon = self.icons.get(self.drag_asset.name)
            if icon is not None:
                ghost = icon.copy()
                ghost.set_alpha(150)
                mp = eng.input.mouse_pos
                surf.blit(ghost, (mp[0] - ICON // 2, mp[1] - ICON // 2))

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
        # selection brackets
        e = self.selected
        if e is None:
            return
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

    def _draw_browser(self, surf, rect) -> None:
        import pygame
        pygame.draw.rect(surf, PANEL_BG, rect)
        pygame.draw.line(surf, PANEL_EDGE, rect.topleft, rect.topright)
        title = self.font.render("Content Browser — drag into the world  (assets/)",
                                 True, TEXT)
        surf.blit(title, (rect.x + 10, rect.y + 6))
        mp = pygame.mouse.get_pos()
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
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import engine

    eng = engine.Engine(args.width, args.height, title="PyEngine Editor")
    lib = engine.AssetLibrary(os.path.join(BASE_DIR, "assets"))
    camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08,
                           far=200.0)

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

    player = engine.Entity("__camera").add_behavior(engine.behaviors.FlyController(
        camera, look_buttons=(3,), look_guard=lambda pos: not editor.over_ui(pos)))
    scene.add(player)
    scene.add(engine.Entity("__editor").add_behavior(EditorBehavior(editor)))

    eng.hud_text = ("RMB hold: look/fly | LMB: select & drag assets | F flashlight | "
                    "Z focus | Ctrl+D dup | Del delete | Ctrl+S save | Esc quit")
    eng.run(scene, camera, max_frames=args.frames, screenshot_path=args.screenshot,
            overlay=editor.draw)


if __name__ == "__main__":
    main()
