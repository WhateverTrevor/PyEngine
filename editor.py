"""PyEngine Editor — menu bar, dockable panels, outliner, content browser.

    py editor.py                     open scenes/scene.json (or a starter scene)
    py editor.py --scene my.json     work on a specific scene file
    py editor.py --api cpu           force a rendering backend: dx12 / vulkan /
                                      gl / cpu (default: dx12, or
                                      Settings > Graphics API); CPU is opt-in
                                      only. --gpu/--cpu still work as aliases
                                      for gl/cpu

Controls (Help > Controls in the editor shows the full list):
    RMB hold        mouse look + WASD/QE/Space/Ctrl fly (Unreal-style: these
                    movement keys only act while RMB is held), Shift = fast
    LMB             select in viewport/outliner (Shift+click extends/toggles
                    a multi-selection, last-clicked = active); drag assets;
                    drag the gizmo; drag panel title bars to move/dock/float
    W/E/R           gizmo translate/rotate/scale (only while not looking)
    F               focus camera on selection (only while not looking)
    Ctrl+D          duplicate selection        Del  delete selection
    Ctrl+S          save scene                 L    toggle flashlight
    K               place 3D Cursor at surface under mouse (not RMB -- that's fly-look)
    Shift+C         reset 3D Cursor to origin
    F1 wireframe    H toggle HUD               Esc  close UI / deselect / quit
"""
from __future__ import annotations

import argparse
import json
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")


def default_settings_path() -> str:
    """Repo-root settings.json, unless PYENGINE_SETTINGS points elsewhere.

    Headless test suites set PYENGINE_SETTINGS to a temp-dir file so they
    never read/write the user's real settings.json — every UI action that
    tweaks layout calls _save_settings(), and a test run driving hundreds
    of those would otherwise clobber the user's saved window layout with
    mid-test fixture state.
    """
    return os.environ.get("PYENGINE_SETTINGS", SETTINGS_PATH)

MENU_H = 26
PANEL_TITLE_H = 18
EDGE_SNAP = 48

OUTLINER_W = 260          # docked left/right panel width (factory default)
MIN_DOCK_W = 150          # side-dock width when every panel on it is minimized
BROWSER_H = 118           # docked bottom panel height (factory default)
MIN_PANEL_W = 150         # smallest a dragged side-dock or floating panel can go
MIN_PANEL_H = 90          # smallest a dragged floating panel or bottom dock can go
SPLITTER_PX = 6           # hit-test/hover band around a dock splitter, in pixels
SIDE_TOOLBAR_W_EXPANDED = 128   # labeled-button width, docked to the window's LEFT
SIDE_TOOLBAR_W_COLLAPSED = 30   # icon-only width when collapsed
SIDE_TOOLBAR_BTN_H = 28
_LAYOUT_REF_W, _LAYOUT_REF_H = 1440, 810  # resolution the factory fractions assume
DOCK_FRAC_DEFAULT = {"left": OUTLINER_W / _LAYOUT_REF_W,
                     "right": OUTLINER_W / _LAYOUT_REF_W,
                     "bottom": BROWSER_H / _LAYOUT_REF_H}
DETAILS_H = 322           # default floating height for the details panel
ROW_H = 20
DETAIL_ROW_H = 24
TRANSFORM_ROWS_TOP = 26   # y-offset of the Position/Rotation/Scale grid
TRANSFORM_ROW_H = 20
TRANSFORM_BLOCK_H = TRANSFORM_ROWS_TOP + 3 * TRANSFORM_ROW_H + 6  # + gap before rows
DETAILS_ROWS_TOP = TRANSFORM_BLOCK_H  # y-offset (within content rect) of row 0,
                          # i.e. below the transform grid that's always shown
TILE_W, TILE_H, ICON = 84, 100, 64
BROWSER_TOPBAR_H = 26     # content-browser top bar (New Folder / Import)
VIEWPORT_TOOLBAR_H = 26   # slim bar docked at the top of the viewport rect
TREE_ROW_H = 18           # content-browser folder-tree row height
TREE_W = 130              # factory folder-tree column width
_NO_RENAME = object()     # sentinel for "not renaming" -- distinct from every
                          # real folder id AND from the root folder id (None),
                          # so the root row can't be mistaken for a rename target
PANEL_BG = (22, 24, 29)
PANEL_EDGE = (58, 62, 72)
TEXT = (210, 212, 218)
TEXT_DIM = (140, 143, 152)
SELECT_BG = (47, 66, 102)        # ACTIVE element -- unchanged from pre-multiselect
SELECT_BG_MULTI = (33, 45, 66)   # additional (non-active) members of a multi-selection
HOVER_BG = (36, 39, 47)
ACCENT = (255, 170, 60)
ACCENT_DIM = (150, 108, 45)      # viewport selection bracket for non-active members
SNAP_FACE_COLOR = (90, 220, 230)  # highlighted face during Shift-drag snap-to-mesh
MARQUEE_COLOR = (140, 180, 255)  # box/marquee-select drag rectangle
MARQUEE_THRESHOLD = 4            # px: shorter press-release drags stay a plain click

PANEL_DEFAULT_FLOAT = {   # (w, h) used the first time a panel floats
    "outliner": (OUTLINER_W, 360),
    "details": (OUTLINER_W, DETAILS_H),
    "browser": (760, BROWSER_H),
    "console": (520, BROWSER_H),
}
TAB_LABELS = {"outliner": "Outliner", "details": "Details", "browser": "Browser",
             "console": "Console"}
CONSOLE_ROW_H = 15       # monospace line height in the console panel
CONSOLE_LEVEL_COLOR = {"info": (190, 193, 200), "warn": (225, 175, 70),
                       "error": (230, 100, 100)}
RESOLUTIONS = ((1280, 720), (1440, 810), (1600, 900), (1920, 1080))
SETTINGS_SIZE = (380, 246)
IMPORT_SIZE = (420, 356)   # Unreal-style import-options modal (see _open_import_dialog)
# extension -> (internal kind, detected-type display label) for the import
# dialog's read-only "Type" header; scale/up-axis only apply to "mesh"
_IMPORT_TYPE_LABELS = {
    ".fbx": ("mesh", "Static Mesh"),
    ".hdr": ("hdri", "HDRI Environment"),
    ".png": ("texture", "Texture"),
    ".jpg": ("texture", "Texture"),
    ".jpeg": ("texture", "Texture"),
    ".bmp": ("texture", "Texture"),
}


def load_settings(path: str | None = None) -> dict:
    path = path or default_settings_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(data: dict, path: str | None = None) -> None:
    path = path or default_settings_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
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
        fog=engine.Fog(color=(7, 8, 12), start=10.0, end=42.0,
                       height_falloff=0.12, sun_scatter=0.4),
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

    # Sun: pale moonlight, its rotation matches the DirectionalLight above so
    # the first frame doesn't jump; the rotate gizmo becomes a time-of-day
    # control from here on. Soft directional shadows, disc/glow visible in
    # the sky, GI left off by default (opt in from the Details panel).
    sun = lib.instantiate("Sun")
    sun.transform.rotation = Vec3(-1.13, 1.01, 0.0)
    scene.add(sun)

    # Fog Volume: a low ground-hugging haze bank across the courtyard floor.
    fog_vol = lib.instantiate("Fog Volume")
    fog_vol.fog_volume.density = 0.12
    fog_vol.fog_volume.color = (60, 66, 78)
    fog_vol.transform.position = Vec3(0.0, 1.1, -2.0)
    fog_vol.transform.scale = Vec3(7.0, 1.6, 6.0)
    scene.add(fog_vol)
    return scene


def make_icon(engine, asset, size=ICON):
    """Render a small 3D preview of an asset for the content browser."""
    import numpy as np
    import pygame

    entity = asset.instantiate()
    surf = pygame.Surface((size, size))
    surf.fill((29, 31, 37))
    if "texture" in asset.data:  # texture asset: thumbnail is the image itself
        from engine.texture import load_texture_rel
        img = load_texture_rel(asset.data["texture"]["path"])
        if img is not None:
            arr = np.clip(img[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
            tsurf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
            pygame.transform.smoothscale(tsurf, (size, size), surf)
        return surf
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


def _draw_checker_bg(surf, size, cell=8):
    """Fill `surf` with a light/dark checker -- transparency-preview backdrop
    (UE-style) so a translucent material's alpha is visible against
    something other than a flat color."""
    a, b = (60, 60, 66), (40, 40, 46)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            c = a if ((x // cell) + (y // cell)) % 2 == 0 else b
            surf.fill(c, (x, y, cell, cell))


def make_material_icon(engine, graph, size=ICON):
    """Bake `graph` onto a small preview sphere and render it -- the
    thumbnail for a material asset tile / Details material slot swatch.
    Translucent materials preview against a checker backdrop (visible
    through the sphere's alpha) instead of the flat background color."""
    import pygame

    surf = pygame.Surface((size, size))
    translucent = graph.blend_mode == "translucent"
    if translucent:
        _draw_checker_bg(surf, size)
    else:
        surf.fill((29, 31, 37))
    sphere = engine.icosphere(radius=1.0, subdivisions=2)
    (sphere.face_colors, sphere.face_roughness,
     sphere.face_metallic, sphere.face_emissive, opacity) = graph.evaluate_pbr(sphere)
    if opacity is not None:
        sphere.face_opacity = opacity
    mini = engine.Scene(
        light=engine.DirectionalLight(engine.Vec3(-0.5, -0.9, -0.6), ambient=0.42),
        background=(29, 31, 37))
    entity = engine.Entity("preview", mesh=sphere)
    entity.material = graph
    mini.add(entity)
    cam = engine.Camera(yaw=0.65, pitch=-0.5)
    fwd = cam.forward()
    cam.position = engine.Vec3(-fwd.x, -fwd.y, -fwd.z) * 2.6
    from engine.renderer import Renderer
    Renderer().render(surf, mini, cam)
    return surf


BLUEPRINT_TILE_EDGE = (70, 150, 90)  # distinct from materials' purple (90, 60, 140)


def make_blueprint_icon(size=ICON):
    """Placeholder thumbnail for a Blueprint asset tile -- a script glyph.
    No posed-mesh preview yet (`components` is always [] this run); run 2
    can render the posed components here instead once they exist."""
    import pygame

    surf = pygame.Surface((size, size))
    surf.fill((22, 28, 24))
    pygame.draw.rect(surf, BLUEPRINT_TILE_EDGE, surf.get_rect(), 2, border_radius=6)
    font = pygame.font.SysFont("consolas,couriernew,monospace", int(size * 0.4), bold=True)
    label = font.render("</>", True, (130, 210, 150))
    surf.blit(label, ((size - label.get_width()) // 2, (size - label.get_height()) // 2))
    return surf


class Editor:
    _MENU_NAMES = ("File", "Edit", "Window", "Help")
    _WINDOW_PANEL_LABELS = {"Outliner": "outliner", "Details": "details",
                            "Content Browser": "browser", "Console": "console"}
    _MENU_HOTKEYS = {
        "File": {"Save": "Ctrl+S"},
        "Edit": {"Duplicate": "Ctrl+D", "Delete": "Del", "Focus Selection": "F",
                 "Snap to Floor": "End", "Reset 3D Cursor": "Shift+C"},
        "Window": {"Fullscreen": "F11"},
    }

    def __init__(self, engine_mod, eng, scene, camera, lib, scene_path,
                settings_path=None):
        import pygame
        self.engine_mod = engine_mod
        self.eng = eng
        self.scene = scene
        self.camera = camera
        self.lib = lib
        self.scene_path = scene_path
        # settings_path is injectable (else PYENGINE_SETTINGS env var, else
        # repo-root settings.json) so tests can point it at a temp file and
        # never touch the user's real one -- see default_settings_path().
        self.settings_path = settings_path or default_settings_path()
        self._selected = None
        self.selection = []         # ordered selection; self.selected (a property,
                                     # defined below) is always its last entry --
                                     # the ACTIVE element, Blender's term -- or None
                                     # when the selection is empty
        self.flashlight = None      # set by main(); hidden from glyphs
        self.fly = None             # the viewport FlyController, for C toggle
        self.dirty = False
        self.outliner_scroll = 0
        self.browser_scroll = 0
        self.console_scroll = 0     # lines scrolled UP from the latest entry;
                                     # 0 == pinned to the bottom (auto-scroll
                                     # follows new entries only while pinned)
        self.drag_asset = None
        self.selected_asset = None  # AssetDef of the last-clicked grid tile, for Export
        # ---- content-browser folder tree ----
        self.selected_folder = None   # folder id, or None == root
        self.tree_scroll = 0
        self.renaming_folder = _NO_RENAME  # folder id being renamed, or _NO_RENAME;
                                            # NOT None -- None is the root folder id,
                                            # so that can't double as "not renaming"
        self.rename_buffer = ""       # editable text while renaming_folder is set
        self.active_slider = None   # index into _details_rows while dragging
        self.gizmo_drag = None      # active gizmo drag state dict
        self.marquee = None         # {"start","cur","shift"} while an LMB press-drag
                                     # that started on empty viewport space is in
                                     # progress (see _begin_viewport_press /
                                     # _finish_marquee) -- box/marquee select
        self.gizmo_mode = "translate"  # W/E/R select translate / rotate / scale
        self.gizmo_space = "world"  # translate axes: "world" or "local" (viewport toolbar)
        self.pivot_mode = "median"  # rotate/scale pivot for multi-selections (viewport
                                     # toolbar): "median" / "bbox" / "active" / "individual" /
                                     # "cursor" -- see _pivot_point. Translate is pivot-independent.
        self.cursor3d = engine_mod.Vec3(0.0, 0.0, 0.0)  # Blender-style 3D cursor: a
                                     # placeable world-space point (K places it on the
                                     # surface under the mouse, Shift+C resets it to the
                                     # origin), persisted in settings.json. Doubles as the
                                     # "cursor" pivot_mode's rotate/scale anchor.
        self.snap_enabled = False   # viewport toolbar snap toggle (grid/interval snapping)
        self.snap_index = {"translate": 0, "rotate": 0, "scale": 0}  # cycled per mode
        self.snap_feedback = None   # (other_entity, axis_idx, plane, lo2, hi2) while a
                                     # Shift-held translate drag is flush against a face
        self.editing_field = None   # (row_label, axis) of the transform field being
                                     # typed into, e.g. ("Position", "x"), or None
        self.edit_buffer = ""       # editable text while editing_field is set
        self.mat_ui = None          # open MaterialEditorUI, or None
        self.script_ui = None       # open ScriptEditorUI (blueprint script editor), or None
        self._status = ("", 0.0)    # transient message near the content browser --
                                     # backing field for the `status` property below
        self.save_flash = 0.0
        # graphics-api preference for *next launch* (live switching is out
        # of scope); defaults to whatever this session actually ended up
        # using (see `_active_api`)
        self.api_pref = self._active_api()

        # ---- dockable panel state ----
        # dock_order[side] is an ordered list of GROUPS -- {"ids": [pid, ...],
        # "active": pid} -- so panels dropped onto each other tab together
        # (see "---- panel docking / floating ----" below). A lone panel is
        # a group of one; that degenerate case renders/behaves exactly like
        # the old flat {side: [pid, ...]} model this replaced.
        self.dock_order = {"left": [], "right": [self._solo_group("outliner"),
                                                 self._solo_group("details")],
                           "bottom": [self._solo_group("browser"),
                                     self._solo_group("console")]}
        self.floating = []          # panel ids currently floating, z-order (front=last)
        self.panel_visible = {"outliner": True, "details": True, "browser": True,
                              "console": True}
        self.panel_minimized = {"outliner": False, "details": False, "browser": False,
                                "console": False}
        self.float_rect = {}        # panel id -> pygame.Rect, used only while floating
        self.panel_drag = None      # {"id","dx","dy","w","h"} while dragging a title bar
        self.dock_frac = dict(DOCK_FRAC_DEFAULT)  # side -> fraction of window w/h;
                                     # proportional so a window resize/fullscreen
                                     # toggle rescales docks instead of freezing pixels
        self.splitter_drag = None   # "left" / "right" / "bottom" while dragging a splitter
        self.panel_resize = None    # {"id","w","h","mx","my"} while resizing a float corner

        # ---- menu bar / dialogs ----
        self.open_menu = None       # name of the open dropdown, or None
        self.settings_open = False
        self.settings_drag = None   # "pixel" / "max_fps" while dragging a settings slider
        self.show_controls_overlay = False
        self.import_dialog = None   # dict (path/ext/kind/label/name/folder/scale_text/
                                     # up_axis) while the import-options modal is open,
                                     # else None -- see _open_import_dialog
        self.import_field = None    # "name" / "scale" -- which dialog field has focus,
                                     # or None (see _update_import_text)

        # ---- collapsible side toolbar: a slim strip docked to the window's
        # LEFT edge, distinct from the "left" dock zone panels can drop
        # into (which starts AFTER this strip -- see _layout). Declarative
        # button list (_side_toolbar_buttons) so future tools just append.
        self.side_toolbar_collapsed = False

        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 14)
        self.font_small = pygame.font.SysFont("consolas,couriernew,monospace", 12)
        self.icons = {}
        count = max(len(lib.assets), 1)
        for i, a in enumerate(lib.assets):
            eng.loading_step(f"rendering thumbnail: {a.name}", 0.25 + 0.45 * i / count)
            self.icons[a.name] = make_icon(engine_mod, a)
        self.mat_icons = {}          # material asset name -> thumbnail Surface
        for m in lib.materials:
            self.mat_icons[m.name] = make_material_icon(engine_mod, m.graph())
        self.mat_icon_cache = {}     # id(entity) -> live Details-slot preview Surface
        self.bp_icons = {}           # blueprint asset name -> thumbnail Surface
        for bp in lib.blueprints:
            self.bp_icons[bp.name] = make_blueprint_icon()

    # ---- selection model ----
    # self.selection is the ordered, authoritative set. self.selected (below)
    # is a property exposing its ACTIVE element -- Blender's term for the
    # last-clicked object, the one typed transform edits / rotate / scale /
    # the material editor operate on for this run (run 2b adds real
    # per-entity pivot-mode support on top of this foundation). Plain
    # assignment (`editor.selected = e`) always collapses to a single-
    # element selection -- this matches every pre-existing call site (both
    # in this file and in tests that poke `editor.selected` directly to set
    # up a single-select fixture) so nothing single-select breaks. Multi-
    # entity selections are only ever produced by _set_selection /
    # _toggle_selection, which write the backing fields directly and
    # bypass the setter's collapse-to-one behavior. Invariant maintained
    # everywhere: self.selected is None iff self.selection == [].
    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value) -> None:
        self._selected = value
        self.selection = [] if value is None else [value]

    # ---- transient status strip -- mirrored into the engine console log ----
    # `status` is (message, ttl); every one of the ~30 call sites across this
    # file that does `self.status = (msg, ttl)` now also appends to the
    # console log, so the log doubles as a persistent history of every status
    # message ever shown, with NO call sites needing to change. The per-frame
    # ttl countdown (`self.status = (self.status[0], self.status[1] - dt)`)
    # re-assigns the SAME text every tick, so the setter only logs when the
    # text actually changes -- otherwise the log would fill with duplicates
    # of one message every frame until its ttl expired.
    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value) -> None:
        text, _ttl = value
        if text and text != self._status[0]:
            from engine import console_log
            level = "error" if "failed" in text.lower() else "info"
            console_log.log(level, text)
        self._status = value

    def _set_selection(self, entities, active=None) -> None:
        """Replace the selection with an explicit ordered list (shift-click
        toggle, batch duplicate, ...). `active` becomes the ACTIVE element
        if it's a member of `entities`; otherwise (or if omitted) the active
        element falls back to the list's last entry, or None when `entities`
        is empty.
        """
        self.selection = list(entities)
        self._selected = (active if active is not None and active in self.selection
                          else (self.selection[-1] if self.selection else None))

    def _toggle_selection(self, entity) -> None:
        """Shift+click semantics (viewport or outliner): an entity already
        in the selection is removed (active falls back to the new last
        member, or None); otherwise it's appended and becomes active."""
        if entity is None:
            return
        sel = list(self.selection)
        if entity in sel:
            sel.remove(entity)
            self._set_selection(sel)
        else:
            sel.append(entity)
            self._set_selection(sel, active=entity)

    def _selection_pivot(self):
        """World-space gizmo anchor for the current selection: the
        arithmetic mean of each selected entity's world position -- v1's
        only pivot mode. This is what Blender itself calls "Median Point"
        (despite the name it's a mean, not a per-axis statistical median);
        run 2b adds Active Element / Individual Origins pivot modes
        alongside it. Single-selection callers get exactly that entity's
        position, unchanged from pre-multiselect behavior. Caller must
        ensure self.selection is non-empty.
        """
        Vec3 = self.engine_mod.Vec3
        n = len(self.selection)
        x = sum(e.transform.position.x for e in self.selection) / n
        y = sum(e.transform.position.y for e in self.selection) / n
        z = sum(e.transform.position.z for e in self.selection) / n
        return Vec3(x, y, z)

    def _selection_bbox_center(self):
        """Center of the combined world AABB across every selected entity
        that has a mesh -- (min+max)/2 of the union of each entity's
        _world_aabb (Blender's "Bounding Box Center" pivot mode). None if
        no selected entity has a mesh, in which case _pivot_point falls
        back to Median Point.
        """
        import numpy as np
        mins, maxs = [], []
        for e in self.selection:
            aabb = self._world_aabb(e)
            if aabb is None:
                continue
            lo, hi = aabb
            mins.append(lo)
            maxs.append(hi)
        if not mins:
            return None
        lo = np.min(np.array(mins), axis=0)
        hi = np.max(np.array(maxs), axis=0)
        c = (lo + hi) / 2.0
        return self.engine_mod.Vec3(float(c[0]), float(c[1]), float(c[2]))

    def _pivot_point(self):
        """World-space anchor for the gizmo widget AND the rigid-group
        orbit/scale center used by rotate/scale drags (translate is pivot-
        independent), chosen by self.pivot_mode:
          median     -- mean of selected origins (_selection_pivot; v1's
                         only mode, still the default)
          bbox       -- center of the selection's combined world AABB
                         (_selection_bbox_center)
          active     -- the active (last-selected) entity's own origin
          individual -- no single group pivot; each entity rotates/scales
                         about ITSELF (see _update_gizmo_drag /
                         _apply_group_scale) -- the widget itself is still
                         drawn at the median, matching Blender
          cursor     -- the placeable 3D cursor (self.cursor3d), regardless
                         of selection size -- see below

        A single-entity selection always reduces to that entity's own
        origin for every OTHER mode -- median/individual/active trivially
        do (the lone entity IS the mean AND the active element), but bbox
        needs an explicit special case: a mesh's local AABB is not
        generally centered on the entity's origin (e.g. base_height meshes
        sit with their BOTTOM, not their center, at local y=0), so without
        this a single selected entity's gizmo would jump off its origin
        under Bounding Box Center, breaking "single-select is unchanged".

        "cursor" mode is deliberately exempt from that single-select
        collapse: the whole point of the 3D cursor pivot is that even a
        lone selected object orbits/scales about an external point instead
        of spinning in place, so this checks pivot_mode BEFORE the
        single-selection shortcut below.
        """
        if not self.selection:
            return None
        if self.pivot_mode == "cursor":
            c = self.cursor3d
            return self.engine_mod.Vec3(c.x, c.y, c.z)
        if len(self.selection) == 1:
            p = self.selection[0].transform.position
            return self.engine_mod.Vec3(p.x, p.y, p.z)
        if self.pivot_mode == "active" and self.selected is not None:
            p = self.selected.transform.position
            return self.engine_mod.Vec3(p.x, p.y, p.z)
        if self.pivot_mode == "bbox":
            c = self._selection_bbox_center()
            if c is not None:
                return c
        return self._selection_pivot()

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

    def _all_minimized(self, ids) -> bool:
        return len(ids) > 0 and all(self.panel_minimized.get(p, False) for p in ids)

    # ---- dock groups: a group is {"ids": [pid, ...], "active": pid} --
    # see "---- panel docking / floating ----" for the drag/drop mutations.
    # These tiny helpers are used by both _layout (rendering) and the drop
    # logic, so the notion of "which tab is showing" never drifts.
    @staticmethod
    def _solo_group(pid) -> dict:
        return {"ids": [pid], "active": pid}

    def _group_active(self, group):
        """Effective active id for `group`: its stored active if that
        member is still visible, else the first visible member, else
        whatever's stored (nothing visible -- callers won't render it
        anyway). Deliberately never WRITES back to group["active"] when a
        hidden tab is skipped over -- closing/reopening a tab is then
        self-healing (see _apply_layout_settings/_toggle_panel) without
        every visibility flip needing to hunt down and patch its group."""
        ids = group["ids"]
        vis = [p for p in ids if self.panel_visible.get(p, True)]
        if group["active"] in vis:
            return group["active"]
        if vis:
            return vis[0]
        return group["active"] if group["active"] in ids else (ids[0] if ids else None)

    def _group_for_pid(self, pid, dock_order=None):
        """(side, group) containing pid, or None if pid is floating/absent
        from `dock_order` (defaults to self.dock_order)."""
        dock_order = self.dock_order if dock_order is None else dock_order
        for side in ("left", "right", "bottom"):
            for group in dock_order[side]:
                if pid in group["ids"]:
                    return side, group
        return None

    def _panel_side(self, pid):
        g = self._group_for_pid(pid)
        return g[0] if g else None

    def _dock_side_w(self, side, w):
        """Docked left/right width from the stored fraction -- proportional
        to the window, clamped to a sane range so it can't be dragged (or
        resized into) unusably small/huge."""
        px = self.dock_frac.get(side, DOCK_FRAC_DEFAULT[side]) * w
        return int(max(MIN_PANEL_W, min(px, w * 0.6)))

    def _dock_bottom_h(self, h):
        px = self.dock_frac.get("bottom", DOCK_FRAC_DEFAULT["bottom"]) * h
        return int(max(MIN_PANEL_H, min(px, h * 0.6)))

    def _layout(self, w, h, dock_order=None):
        import pygame
        dock_order = self.dock_order if dock_order is None else dock_order
        menu = pygame.Rect(0, 0, w, MENU_H)
        tb_w = self._side_toolbar_w()  # side toolbar claims the window's left
                                        # edge; the "left" dock zone below
                                        # starts AFTER it, not at x=0

        def side_ids(side):
            """Active id per visible group on `side`, in group order -- a
            lone-panel group's active IS that panel, so this list is
            byte-identical to the old flat per-side pid list whenever every
            group on this side is still degenerate (the pre-tabs case)."""
            out = []
            for group in dock_order[side]:
                if any(self.panel_visible.get(p, True) for p in group["ids"]):
                    out.append(self._group_active(group))
            return out

        left_ids = side_ids("left")
        right_ids = side_ids("right")
        bottom_ids = side_ids("bottom")
        left_w = 0
        if left_ids:
            left_w = MIN_DOCK_W if self._all_minimized(left_ids) else self._dock_side_w("left", w)
        right_w = 0
        if right_ids:
            right_w = MIN_DOCK_W if self._all_minimized(right_ids) else self._dock_side_w("right", w)
        if left_w + right_w > max(0, w - tb_w - MIN_PANEL_W):  # keep a usable viewport strip
            scale = max(0, w - tb_w - MIN_PANEL_W) / max(1, left_w + right_w)
            left_w, right_w = int(left_w * scale), int(right_w * scale)
        bottom_h = 0
        if bottom_ids:
            bottom_h = PANEL_TITLE_H if self._all_minimized(bottom_ids) else self._dock_bottom_h(h)
        top = MENU_H
        stack_bottom = h - bottom_h
        panels = {}

        def stack(ids, x, width):
            n = len(ids)
            if n == 0:
                return
            avail = max(0, stack_bottom - top)
            minimized = [p for p in ids if self.panel_minimized.get(p, False)]
            normal = [p for p in ids if not self.panel_minimized.get(p, False)]
            remain = max(0, avail - len(minimized) * PANEL_TITLE_H)
            n_normal = len(normal)
            share = remain // n_normal if n_normal else 0
            y = top
            for pid in ids:
                if pid in minimized:
                    hh = PANEL_TITLE_H
                else:
                    idx = normal.index(pid)
                    hh = share if idx < n_normal - 1 else remain - share * (n_normal - 1)
                panels[pid] = pygame.Rect(x, y, width, max(0, hh))
                y += hh

        stack(left_ids, tb_w, left_w)
        stack(right_ids, w - right_w, right_w)
        if bottom_ids:
            bw_total = max(0, w - tb_w - left_w - right_w)
            n = len(bottom_ids)
            share = bw_total // n
            x = tb_w + left_w
            for i, pid in enumerate(bottom_ids):
                ww = share if i < n - 1 else bw_total - share * (n - 1)
                hh = PANEL_TITLE_H if self.panel_minimized.get(pid, False) else bottom_h
                panels[pid] = pygame.Rect(x, stack_bottom, max(0, ww), hh)
                x += ww

        grouped = {p for side in ("left", "right", "bottom")
                  for group in dock_order[side] for p in group["ids"]}
        for pid in ("outliner", "details", "browser", "console"):
            # `pid in panels` skips anything a stack() call already placed;
            # `pid in grouped` additionally skips an INACTIVE tab of a real
            # group (visible, but not the one currently showing -- it must
            # NOT be auto-floated just because it has no entry in `panels`)
            if not self.panel_visible.get(pid, True) or pid in panels or pid in grouped:
                continue
            r = self._float_rect_for(pid, w, h)
            if self.panel_minimized.get(pid, False):
                r = pygame.Rect(r.x, r.y, r.width, PANEL_TITLE_H)
            panels[pid] = r

        viewport = pygame.Rect(tb_w + left_w, top, max(0, w - tb_w - left_w - right_w),
                               max(0, stack_bottom - top))

        splitters = {}
        half = SPLITTER_PX // 2
        if left_ids and not self._all_minimized(left_ids):
            splitters["left"] = pygame.Rect(tb_w + left_w - half, top, SPLITTER_PX,
                                            max(0, stack_bottom - top))
        if right_ids and not self._all_minimized(right_ids):
            splitters["right"] = pygame.Rect(w - right_w - half, top, SPLITTER_PX,
                                             max(0, stack_bottom - top))
        if bottom_ids and not self._all_minimized(bottom_ids):
            splitters["bottom"] = pygame.Rect(tb_w + left_w, stack_bottom - half,
                                              max(0, w - tb_w - left_w - right_w), SPLITTER_PX)

        return {"menu": menu, "viewport": viewport, "panels": panels,
                "left_w": left_w, "right_w": right_w, "bottom_h": bottom_h,
                "splitters": splitters, "side_toolbar_w": tb_w}

    def _panel_content_rect(self, pid, layout):
        import pygame
        r = layout["panels"].get(pid)
        if r is None:
            return None
        return pygame.Rect(r.x, r.y + PANEL_TITLE_H, r.width,
                           max(0, r.height - PANEL_TITLE_H))

    def _panel_title_buttons(self, rect):
        """[minimize][close] rects at the right end of a panel's title bar,
        used identically for drawing and hit-testing."""
        import pygame
        close = pygame.Rect(rect.right - 18, rect.y + 3, 13, 13)
        minimize = pygame.Rect(rect.right - 34, rect.y + 3, 13, 13)
        return {"minimize": minimize, "close": close}

    @staticmethod
    def _panel_resize_handle(rect):
        """Bottom-right resize grip rect for a floating panel — same rect
        used for drawing the grip glyph and hit-testing the resize drag."""
        import pygame
        return pygame.Rect(rect.right - 10, rect.bottom - 10, 10, 10)

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
        if (self.mat_ui is not None or self.script_ui is not None
                or self.settings_open or self.import_dialog is not None):
            return True
        if pos[1] < MENU_H:
            return True
        w, h = self.eng.screen.get_size()
        if self.open_menu is not None:
            drop, _rows = self._dropdown_geom(self.open_menu, w)
            if drop.collidepoint(pos):
                return True
        if self._side_toolbar_rect(w, h).collidepoint(pos):
            return True
        layout = self._layout(w, h)
        if layout["viewport"].collidepoint(pos) \
                and self._viewport_toolbar_rect(layout["viewport"]).collidepoint(pos):
            return True
        return self._hit_panel(pos, layout) is not None

    def _outliner_rows(self):
        return [e for e in self.scene.entities
                if e.mesh is not None or e.light is not None
                or e.environment is not None or e.sun is not None
                or e.fog_volume is not None]

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

        if src.material is not None and (dst.mesh is not None or dst.environment is not None):
            dst.material = self.engine_mod.MaterialGraph.from_dict(
                src.material.to_dict())
            dst.material.apply(dst)

        if src.sun is not None and dst.sun is not None:
            for key in self.engine_mod.assets._SUN_PROPS:
                setattr(dst.sun, key, getattr(src.sun, key))

        if src.fog_volume is not None and dst.fog_volume is not None:
            for key in self.engine_mod.assets._FOG_VOLUME_PROPS:
                setattr(dst.fog_volume, key, getattr(src.fog_volume, key))

    # ---- world-space AABB + floor/mesh snapping ----
    @staticmethod
    def _world_aabb(entity):
        """World-space (min, max) numpy arrays for entity.mesh's local AABB,
        transformed by transform.matrix() -- 8 corners then min/max, same
        local-AABB-via-matrix approach as engine/behaviors.py
        resolve_collisions (which does the inverse: world point -> local).
        None if the entity has no mesh.
        """
        import numpy as np
        if entity is None or entity.mesh is None:
            return None
        m = entity.transform.matrix()
        lo, hi = entity.mesh.aabb_min, entity.mesh.aabb_max
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = corners @ m[:3, :3].T + m[:3, 3]
        return world.min(axis=0), world.max(axis=0)

    def _snap_to_floor(self) -> None:
        """End key: drop every selected entity independently (see
        _snap_entity_to_floor). Single-selection callers get exactly the
        old one-entity behavior since self.selection has one member."""
        for e in self.selection:
            self._snap_entity_to_floor(e)

    def _snap_entity_to_floor(self, e) -> None:
        """Drop `e` straight down onto the highest other-entity world-AABB
        surface below its footprint (XZ overlap of the world AABBs); rests
        on y=0 if nothing is underneath it.
        """
        if e is None or e.mesh is None:
            return
        my_min, my_max = self._world_aabb(e)
        x0, x1, z0, z1 = my_min[0], my_max[0], my_min[2], my_max[2]
        cur_bottom = float(my_min[1])
        best_y = 0.0  # y=0 fallback
        eps = 1e-4
        for other in self.scene.entities:
            if other is e or other.mesh is None or not other.visible:
                continue
            o_min, o_max = self._world_aabb(other)
            if o_max[0] < x0 or o_min[0] > x1 or o_max[2] < z0 or o_min[2] > z1:
                continue  # no XZ footprint overlap -- not underneath
            top = float(o_max[1])
            if top <= cur_bottom + eps and top > best_y:
                best_y = top
        delta = best_y - cur_bottom
        if abs(delta) > 1e-9:
            p = e.transform.position
            e.transform.position = self.engine_mod.Vec3(p.x, p.y + delta, p.z)
            self.dirty = True

    def _find_mesh_snap(self, entity, axis_idx, proposed_min, proposed_max):
        """Nearest other entity whose world-AABB face is flush (within a
        threshold) against the dragged entity's leading/trailing face along
        axis_idx, given the entity's PROPOSED (not yet applied) world AABB.
        v1: axis-aligned face-to-face only, no rotation matching (per spec).
        Returns (delta_along_axis, other_entity, plane_value, lo2, hi2) where
        lo2/hi2 are the overlap rectangle bounds on the two other axes (used
        to draw the highlighted face), or None if nothing is within threshold.
        """
        threshold = 0.5  # world units
        other_axes = [k for k in range(3) if k != axis_idx]
        best = None
        for other in self.scene.entities:
            if other is entity or other.mesh is None or not other.visible:
                continue
            o_min, o_max = self._world_aabb(other)
            for delta, plane in ((o_min[axis_idx] - proposed_max[axis_idx], o_min[axis_idx]),
                                 (o_max[axis_idx] - proposed_min[axis_idx], o_max[axis_idx])):
                if abs(delta) > threshold:
                    continue
                # the other two axes don't move with this axis-aligned shift,
                # so the overlap rectangle is just the two boxes' intersection
                lo2 = [max(proposed_min[k], o_min[k]) for k in other_axes]
                hi2 = [min(proposed_max[k], o_max[k]) for k in other_axes]
                if any(lo2[i] > hi2[i] for i in range(2)):
                    continue  # no overlap on the other two axes -- not a face contact
                if best is None or abs(delta) < abs(best[0]):
                    best = (delta, other, plane, lo2, hi2)
        return best

    # ---- viewport toolbar: mode buttons + world/local toggle, declarative so
    # future controls (snapping, more modes, ...) just append to the list ----
    def _set_gizmo_mode(self, mode) -> None:
        self.gizmo_mode, self.gizmo_drag = mode, None

    def _toggle_gizmo_space(self) -> None:
        self.gizmo_space = "local" if self.gizmo_space == "world" else "world"

    # pivot mode: Blender's Median Point / Bounding Box Center / Active
    # Element / Individual Origins, cycled by one toolbar button (see
    # _toolbar_buttons) and applied by _pivot_point / _update_gizmo_drag /
    # _apply_group_scale.
    _PIVOT_MODES = ("median", "bbox", "active", "individual", "cursor")
    _PIVOT_LABELS = {"median": "Median Point", "bbox": "Bounding Box",
                     "active": "Active Element", "individual": "Individual Origins",
                     "cursor": "3D Cursor"}

    def _cycle_pivot_mode(self) -> None:
        i = self._PIVOT_MODES.index(self.pivot_mode)
        self.pivot_mode = self._PIVOT_MODES[(i + 1) % len(self._PIVOT_MODES)]
        self._save_settings()

    def _pivot_label(self) -> str:
        return self._PIVOT_LABELS[self.pivot_mode]

    # increments cyclable per gizmo mode: translate/scale are linear units,
    # rotate is degrees (converted to radians where applied to Transform.rotation)
    _SNAP_INCREMENTS = {
        "translate": (0.1, 0.25, 0.5, 1.0),
        "rotate": (5.0, 15.0, 45.0, 90.0),
        "scale": (0.1, 0.25),
    }

    def _snap_increment(self, mode=None) -> float:
        mode = mode or self.gizmo_mode
        incs = self._SNAP_INCREMENTS.get(mode, self._SNAP_INCREMENTS["translate"])
        return incs[self.snap_index.get(mode, 0) % len(incs)]

    def _cycle_snap_increment(self) -> None:
        mode = self.gizmo_mode
        incs = self._SNAP_INCREMENTS.get(mode, self._SNAP_INCREMENTS["translate"])
        self.snap_index[mode] = (self.snap_index.get(mode, 0) + 1) % len(incs)
        self._save_settings()

    def _toggle_snap(self) -> None:
        self.snap_enabled = not self.snap_enabled
        self._save_settings()

    @staticmethod
    def _quantize(value: float, increment: float) -> float:
        if increment <= 0:
            return value
        return round(value / increment) * increment

    def _snap_active(self, inp) -> bool:
        """Effective snap state for the in-progress drag: the toolbar toggle,
        with Ctrl held during the drag temporarily inverting it (UE/Blender
        convention -- Ctrl toggles snap the opposite of whatever it's set to)."""
        import pygame
        ctrl = inp.held(pygame.K_LCTRL) or inp.held(pygame.K_RCTRL)
        return self.snap_enabled != ctrl

    def _snap_label(self) -> str:
        inc = self._snap_increment()
        return f"{inc:g}°" if self.gizmo_mode == "rotate" else f"{inc:g}"

    def _toolbar_buttons(self):
        return [
            {"id": "translate", "label": "Translate",
             "active": lambda: self.gizmo_mode == "translate",
             "action": lambda: self._set_gizmo_mode("translate")},
            {"id": "rotate", "label": "Rotate",
             "active": lambda: self.gizmo_mode == "rotate",
             "action": lambda: self._set_gizmo_mode("rotate")},
            {"id": "scale", "label": "Scale",
             "active": lambda: self.gizmo_mode == "scale",
             "action": lambda: self._set_gizmo_mode("scale")},
            {"id": "space", "label": lambda: "World" if self.gizmo_space == "world"
                                     else "Local", "group_gap": True,
             "active": lambda: self.gizmo_space == "local",
             "action": self._toggle_gizmo_space},
            {"id": "pivot_mode", "label": self._pivot_label, "group_gap": True,
             "active": lambda: False,
             "action": self._cycle_pivot_mode},
            {"id": "snap_toggle", "label": "Snap", "group_gap": True,
             "active": lambda: self.snap_enabled,
             "action": self._toggle_snap},
            {"id": "snap_inc", "label": self._snap_label,
             "active": lambda: False,
             "action": self._cycle_snap_increment},
        ]

    def _viewport_toolbar_rect(self, viewport_rect):
        import pygame
        return pygame.Rect(viewport_rect.x, viewport_rect.y,
                           viewport_rect.width, VIEWPORT_TOOLBAR_H)

    def _toolbar_button_rects(self, toolbar_rect):
        """[(button_def, rect), ...] -- the single source both the toolbar
        draw call and its click router use, so they can't disagree."""
        import pygame
        rects = []
        x = toolbar_rect.x + 6
        y = toolbar_rect.y + 3
        h = toolbar_rect.height - 6
        for btn in self._toolbar_buttons():
            if btn.get("group_gap"):
                x += 10
            label = btn["label"]() if callable(btn["label"]) else btn["label"]
            bw = self.font_small.size(label)[0] + 16
            rects.append((btn, pygame.Rect(x, y, bw, h)))
            x += bw + 4
        return rects

    def _click_viewport_toolbar(self, mp, toolbar_rect) -> bool:
        for btn, r in self._toolbar_button_rects(toolbar_rect):
            if r.collidepoint(mp):
                btn["action"]()
                return True
        return False

    def _draw_viewport_toolbar(self, surf, toolbar_rect, mp) -> None:
        import pygame
        pygame.draw.rect(surf, PANEL_BG, toolbar_rect)
        pygame.draw.line(surf, PANEL_EDGE, (toolbar_rect.x, toolbar_rect.bottom),
                         (toolbar_rect.right, toolbar_rect.bottom))
        for btn, r in self._toolbar_button_rects(toolbar_rect):
            active = btn["active"]()
            if active:
                bg = ACCENT
            elif r.collidepoint(mp):
                bg = HOVER_BG
            else:
                bg = (33, 36, 44)
            pygame.draw.rect(surf, bg, r, border_radius=4)
            pygame.draw.rect(surf, PANEL_EDGE, r, 1, border_radius=4)
            label = btn["label"]() if callable(btn["label"]) else btn["label"]
            color = (20, 20, 24) if active else ACCENT
            text = self.font_small.render(label, True, color)
            surf.blit(text, (r.x + (r.width - text.get_width()) // 2,
                             r.y + (r.height - text.get_height()) // 2))

    # ---- collapsible side toolbar: docked to the window's LEFT edge, a
    # NEW distinct element from the docked panels/viewport toolbar above --
    # declarative button list so future tools just append (mirrors
    # _toolbar_buttons' pattern, stacked vertically instead of horizontal). ----
    def _side_toolbar_w(self) -> int:
        return SIDE_TOOLBAR_W_COLLAPSED if self.side_toolbar_collapsed \
            else SIDE_TOOLBAR_W_EXPANDED

    def _side_toolbar_rect(self, w, h):
        import pygame
        return pygame.Rect(0, MENU_H, self._side_toolbar_w(), max(0, h - MENU_H))

    def _toggle_side_toolbar(self) -> None:
        self.side_toolbar_collapsed = not self.side_toolbar_collapsed
        self._save_settings()

    def _side_toolbar_buttons(self):
        """[{"id","label","icon","active","action"}, ...] -- `label` shows
        when expanded, `icon` (a short glyph) when collapsed. Growable: a
        future tool just appends another entry here."""
        return [
            {"id": "console", "label": "Console", "icon": "C",
             "active": lambda: self.panel_visible.get("console", False),
             "action": lambda: self._toggle_panel("console")},
        ]

    def _side_toolbar_collapse_rect(self, rect):
        import pygame
        return pygame.Rect(rect.x + 3, rect.y + 3, rect.width - 6, SIDE_TOOLBAR_BTN_H)

    def _side_toolbar_button_rects(self, rect):
        """[(button_def, rect), ...] below the collapse toggle -- the single
        source both the draw call and its click router use."""
        import pygame
        out = []
        y = rect.y + SIDE_TOOLBAR_BTN_H + 9
        for btn in self._side_toolbar_buttons():
            out.append((btn, pygame.Rect(rect.x + 3, y, rect.width - 6, SIDE_TOOLBAR_BTN_H)))
            y += SIDE_TOOLBAR_BTN_H + 4
        return out

    def _click_side_toolbar(self, mp, rect) -> bool:
        if self._side_toolbar_collapse_rect(rect).collidepoint(mp):
            self._toggle_side_toolbar()
            return True
        for btn, r in self._side_toolbar_button_rects(rect):
            if r.collidepoint(mp):
                btn["action"]()
                return True
        return True  # any other click on the strip is still consumed (over-UI)

    def _draw_side_toolbar(self, surf, rect, mp) -> None:
        import pygame
        pygame.draw.rect(surf, PANEL_BG, rect)
        pygame.draw.line(surf, PANEL_EDGE, (rect.right - 1, rect.y),
                         (rect.right - 1, rect.bottom))
        collapsed = self.side_toolbar_collapsed
        cr = self._side_toolbar_collapse_rect(rect)
        hov = cr.collidepoint(mp)
        pygame.draw.rect(surf, HOVER_BG if hov else (33, 36, 44), cr, border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, cr, 1, border_radius=4)
        glyph = ">" if collapsed else "<"
        lab = self.font_small.render(glyph, True, TEXT)
        surf.blit(lab, (cr.x + (cr.width - lab.get_width()) // 2,
                        cr.y + (cr.height - lab.get_height()) // 2))
        for btn, r in self._side_toolbar_button_rects(rect):
            active = btn["active"]()
            if active:
                bg = ACCENT
            elif r.collidepoint(mp):
                bg = HOVER_BG
            else:
                bg = (33, 36, 44)
            pygame.draw.rect(surf, bg, r, border_radius=4)
            pygame.draw.rect(surf, PANEL_EDGE, r, 1, border_radius=4)
            text_color = (20, 20, 24) if active else ACCENT
            label = btn["icon"] if collapsed else btn["label"]
            lab2 = self.font_small.render(label, True, text_color)
            surf.blit(lab2, (r.x + (r.width - lab2.get_width()) // 2,
                             r.y + (r.height - lab2.get_height()) // 2))

    # ---- transform gizmo: W/E/R select translate / rotate / scale ----
    _GIZMO_AXES = (((1.0, 0.0, 0.0), (225, 85, 85)),
                   ((0.0, 1.0, 0.0), (105, 215, 105)),
                   ((0.0, 0.0, 1.0), (95, 145, 250)))

    def _local_axis_defs(self, e):
        """_GIZMO_AXES rotated into the entity's local (object) space -- the
        same composition order as Transform.matrix()'s rotation (Ry @ Rx @ Rz),
        computed independently so nothing here touches the memoized matrix."""
        import numpy as np
        r = e.transform.rotation
        cx, sx = math.cos(r.x), math.sin(r.x)
        cy, sy = math.cos(r.y), math.sin(r.y)
        cz, sz = math.cos(r.z), math.sin(r.z)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        rot = ry @ rx @ rz
        return [((float(rot[0, i]), float(rot[1, i]), float(rot[2, i])),
                 self._GIZMO_AXES[i][1]) for i in range(3)]

    @staticmethod
    def _axis_rotation_matrix(axis_i, angle):
        """3x3 rotation matrix for `angle` radians about world axis `axis_i`
        (0=X / 1=Y / 2=Z) -- the same per-axis forms as _local_axis_defs's
        rx/ry/rz. Used to orbit a rigid-group multi-selection's positions
        about the gizmo pivot in lockstep with the identical Euler-component
        delta each entity's own rotation gets (see _update_gizmo_drag);
        rotate rings are always world-axis (_gizmo_rings uses _GIZMO_AXES
        directly, never _axis_defs), so this never needs a local variant.
        """
        import numpy as np
        c, s = math.cos(angle), math.sin(angle)
        if axis_i == 0:
            return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        if axis_i == 1:
            return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _axis_defs(self, e):
        """Axis set to use for the current mode/space. Local space only makes
        sense for translate: scale already operates on transform.scale, which
        is inherently object-local regardless of this toggle, and rotate drags
        Euler components directly (see _update_gizmo_drag) so a world/local
        split there would need a quaternion rewrite -- out of scope here."""
        if self.gizmo_mode == "translate" and self.gizmo_space == "local" and e is not None:
            return self._local_axis_defs(e)
        return self._GIZMO_AXES

    def _gizmo_center(self, w, h):
        if not self.selection:
            return None, None, None
        p = self._pivot_point()
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
        for i, (axis, color) in enumerate(self._axis_defs(self.selected)):
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

    def _alt_duplicate_for_drag(self, inp):
        """Alt held while grabbing a translate/rotate gizmo handle duplicates
        the selection IN PLACE (offset=False -- the drag itself is what
        moves/rotates the copy) and returns the new entity to drag; the
        original is left completely untouched. No-op (returns self.selected
        unchanged) if Alt isn't held or inp wasn't supplied (existing test
        call sites that don't care about Alt pass inp=None).
        """
        import pygame
        if inp is None or not (inp.held(pygame.K_LALT) or inp.held(pygame.K_RALT)):
            return self.selected
        self._duplicate_selected(offset=False)
        return self.selected

    def _try_grab_gizmo(self, mp, w, h, inp=None) -> bool:
        e = self.selected
        if e is None:
            return False
        mode = self.gizmo_mode
        t, s = e.transform, e.transform.scale
        if mode == "rotate":
            p, s0, _len = self._gizmo_center(w, h)
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
            # duplicate now that a ring hit is confirmed -- the dup starts at
            # the exact same transform, so the already-computed center/rings
            # (projected off the original) stay valid for it unchanged. Alt
            # on a multi-selection duplicates the WHOLE selection (see
            # _duplicate_selected), so self.selection below is already the
            # dup list -- the group rotates together in _update_gizmo_drag.
            e = self._alt_duplicate_for_drag(inp)
            t = e.transform
            _d, i, axis = best
            to_cam = self.camera.position - t.position
            toward = (axis[0] * to_cam.x + axis[1] * to_cam.y
                      + axis[2] * to_cam.z) > 0
            self.gizmo_drag = {
                "mode": "rotate", "axis_i": i, "center": (s0[0], s0[1]),
                "a0": math.atan2(mp[1] - s0[1], mp[0] - s0[0]),
                "sign": -1.0 if toward else 1.0,
                "start": (t.rotation.x, t.rotation.y, t.rotation.z),
                "pivot": (p.x, p.y, p.z),
                # per-entity (position, rotation) at grab time -- lets a
                # multi-selection rotate as a rigid group about the pivot
                # (see _update_gizmo_drag); absent on hand-built gizmo_drag
                # dicts in older direct-call tests, which fall back to
                # active-only, matching pre-pivot-mode behavior exactly
                "starts": {id(ent): (
                    (ent.transform.position.x, ent.transform.position.y,
                     ent.transform.position.z),
                    (ent.transform.rotation.x, ent.transform.rotation.y,
                     ent.transform.rotation.z)) for ent in self.selection}}
            return True

        handles = self._gizmo_handles(w, h)
        if mode == "scale":
            p, s0, _len = self._gizmo_center(w, h)
            if s0 is not None and math.hypot(mp[0] - s0[0], mp[1] - s0[1]) < 10:
                self.gizmo_drag = {
                    "mode": "scale", "axis_i": -1, "press": mp,
                    "start": (s.x, s.y, s.z), "pivot": (p.x, p.y, p.z),
                    "starts": {id(ent): (
                        (ent.transform.position.x, ent.transform.position.y,
                         ent.transform.position.z),
                        (ent.transform.scale.x, ent.transform.scale.y,
                         ent.transform.scale.z)) for ent in self.selection}}
                return True
        best = None
        for i, axis, s0, s1, _color, length in handles:
            d = self._segment_distance(mp, s0, s1)
            if d < 9.0 and (best is None or d < best[0]):
                best = (d, i, axis, s0, s1, length)
        if best is None:
            return False
        if mode == "translate":  # Alt-duplicate is out of scope for scale (spec)
            e = self._alt_duplicate_for_drag(inp)
            t, s = e.transform, e.transform.scale
        _d, i, axis, s0, s1, length = best
        self.gizmo_drag = {
            "mode": mode, "axis_i": i, "axis": axis, "press": mp,
            "dpx": (s1[0] - s0[0], s1[1] - s0[1]), "length": length,
            "start": ((t.position.x, t.position.y, t.position.z)
                      if mode == "translate" else (s.x, s.y, s.z))}
        if mode == "translate":
            # every selected entity's own start position, so the drag can
            # move the whole selection by one shared world delta (see
            # _update_gizmo_drag) -- captured post-duplicate, so an Alt-copy
            # of the whole selection gets its own starts, not the originals'.
            # Absent on hand-built gizmo_drag dicts (older direct-call
            # tests) -- _update_gizmo_drag falls back to active-only there.
            self.gizmo_drag["starts"] = {
                id(ent): (ent.transform.position.x, ent.transform.position.y,
                          ent.transform.position.z) for ent in self.selection}
        elif mode == "scale":
            # per-entity (position, scale) at grab time, plus the pivot `p`
            # already computed above for the center-hit-test -- lets a
            # multi-selection scale as a rigid group (see
            # _apply_group_scale). Scale never Alt-duplicates (out of spec
            # scope, unlike translate/rotate), so no post-dup recapture
            # needed here.
            self.gizmo_drag["pivot"] = (p.x, p.y, p.z)
            self.gizmo_drag["starts"] = {id(ent): (
                (ent.transform.position.x, ent.transform.position.y,
                 ent.transform.position.z),
                (ent.transform.scale.x, ent.transform.scale.y,
                 ent.transform.scale.z)) for ent in self.selection}
        return True

    def _apply_group_scale(self, g, factor, uniform) -> None:
        """Apply a scale `factor` (uniform: all 3 axes; per-axis: only
        g['axis_i']) to every entity captured in g['starts'] at grab time,
        honoring the active pivot mode: Individual Origins scales each
        entity about its OWN origin (positions untouched); every other
        mode (or a single-entity selection, which always reduces to
        "about your own origin" regardless of mode -- see _pivot_point)
        scales the whole selection as a rigid group about g['pivot']
        (p' = pivot + factor*(p - pivot); for a per-axis drag only the
        scaled coordinate moves, since scale's world-space handles are
        always axis-aligned -- see _axis_defs). Falls back to the
        pre-pivot-mode active-only behavior when 'starts'/'pivot' are
        absent (hand-built gizmo_drag dicts in older direct-call tests).

        "cursor" pivot mode is rigid even for a single-entity selection --
        the whole point of the 3D cursor pivot is that a lone object still
        scales relative to an external point instead of about itself (see
        _pivot_point's matching exemption).
        """
        Vec3 = self.engine_mod.Vec3
        starts = g.get("starts")
        pivot = g.get("pivot")
        axis_i = g.get("axis_i", -1)
        if not starts or pivot is None:
            e = self.selected
            s = list(g["start"])
            if uniform:
                e.transform.scale = Vec3(s[0] * factor, s[1] * factor, s[2] * factor)
            else:
                s[axis_i] = s[axis_i] * factor
                e.transform.scale = Vec3(*s)
            return
        rigid = self.pivot_mode == "cursor" or (
            len(self.selection) > 1 and self.pivot_mode != "individual")
        for ent in self.selection:
            st = starts.get(id(ent))
            if st is None:
                continue  # joined the selection after the drag began
            pos, sc = st
            if uniform:
                new_scale = [sc[0] * factor, sc[1] * factor, sc[2] * factor]
            else:
                new_scale = list(sc)
                new_scale[axis_i] = sc[axis_i] * factor
            ent.transform.scale = Vec3(*new_scale)
            if rigid:
                new_pos = list(pos)
                if uniform:
                    new_pos = [pivot[i] + factor * (pos[i] - pivot[i]) for i in range(3)]
                else:
                    new_pos[axis_i] = pivot[axis_i] + factor * (pos[axis_i] - pivot[axis_i])
                ent.transform.position = Vec3(*new_pos)

    def _update_gizmo_drag(self, mp, inp) -> None:
        import pygame
        import numpy as np
        g = self.gizmo_drag
        Vec3 = self.engine_mod.Vec3
        e = self.selected
        snap = self._snap_active(inp)
        if g["mode"] == "rotate":
            cx, cy = g["center"]
            ang = math.atan2(mp[1] - cy, mp[0] - cx)
            delta = (ang - g["a0"]) * g["sign"]
            axis_i = g["axis_i"]
            start = g["start"]
            new_val = start[axis_i] + delta
            if snap:
                new_val = self._quantize(new_val, math.radians(self._snap_increment("rotate")))
                # re-derive delta from the snapped value BEFORE the group loop
                # below so every entity's orbit + own-rotation increment uses
                # the exact same final (snapped) delta as the active entity
                delta = new_val - start[axis_i]
            starts = g.get("starts")
            if not starts:
                # hand-built gizmo_drag dict (older direct-call test): active
                # entity only, matching pre-pivot-mode behavior exactly
                r = list(start)
                r[axis_i] = new_val
                e.transform.rotation = Vec3(*r)
                self.dirty = True
                return
            pivot = g["pivot"]
            # rigid-group orbit applies for every mode except Individual
            # Origins, and only when there's more than one entity to begin
            # with -- a single-entity selection always reduces to "rotate
            # about your own origin", i.e. no position change at all, for
            # every pivot mode (see _pivot_point's single-select shortcut)
            # EXCEPT "cursor": that mode is exempt from the single-select
            # collapse (_pivot_point returns cursor3d regardless of
            # selection size), so a lone selected object must still orbit
            # the cursor instead of spinning in place.
            rigid = self.pivot_mode == "cursor" or (
                len(self.selection) > 1 and self.pivot_mode != "individual")
            R = self._axis_rotation_matrix(axis_i, delta) if rigid else None
            for ent in self.selection:
                st = starts.get(id(ent))
                if st is None:
                    continue  # joined the selection after the drag began
                pos, rot = st
                nr = list(rot)
                nr[axis_i] = rot[axis_i] + delta
                ent.transform.rotation = Vec3(*nr)
                if rigid:
                    off = np.array(pos) - np.array(pivot)
                    new_off = R @ off
                    ent.transform.position = Vec3(*(np.array(pivot) + new_off))
            self.dirty = True
            return
        if g["mode"] == "scale" and g["axis_i"] == -1:
            factor = max(0.05, 1.0 + (mp[0] - g["press"][0]) * 0.004)
            if snap:
                factor = max(0.05, self._quantize(factor, self._snap_increment("scale")))
            self._apply_group_scale(g, factor, uniform=True)
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
            new_pos = [s[0] + ax[0] * move, s[1] + ax[1] * move, s[2] + ax[2] * move]
            # Shift-held snap-to-mesh takes precedence over grid snap (a
            # separate mechanic, independent of the toolbar toggle -- see the
            # HUD hint). Only fires when the drag axis is (numerically) world-
            # axis-aligned: local-space drags on a rotated entity fall through
            # to grid snap only, per spec ("no rotation matching" in v1).
            import numpy as np
            snapped_to_mesh = False
            shift_held = inp.held(pygame.K_LSHIFT) or inp.held(pygame.K_RSHIFT)
            if shift_held:
                axis_idx = int(np.argmax(np.abs(ax)))
                if abs(ax[axis_idx]) > 0.999:
                    cur_min, cur_max = self._world_aabb(e)
                    p = e.transform.position
                    d = np.array(new_pos) - np.array([p.x, p.y, p.z])
                    found = self._find_mesh_snap(e, axis_idx, cur_min + d, cur_max + d)
                    if found is not None:
                        fdelta, other, plane, lo2, hi2 = found
                        new_pos[axis_idx] += fdelta
                        self.snap_feedback = (other, axis_idx, plane, lo2, hi2)
                        snapped_to_mesh = True
            if not snapped_to_mesh:
                self.snap_feedback = None
                if snap:
                    # quantize the position's coordinate along the drag axis
                    # (world or local, whichever _axis_defs handed us) to grid
                    # increments measured from the world origin -- not the
                    # delta, so the result lands on absolute grid lines
                    # regardless of start position
                    inc = self._snap_increment("translate")
                    proj = new_pos[0] * ax[0] + new_pos[1] * ax[1] + new_pos[2] * ax[2]
                    shift = self._quantize(proj, inc) - proj
                    new_pos = [new_pos[i] + ax[i] * shift for i in range(3)]
            # apply the SAME world delta (raw move + whatever snap adjusted)
            # to every selected entity's own start position, so the whole
            # selection moves together. "starts" is set by _try_grab_gizmo;
            # hand-built gizmo_drag dicts (older direct-call tests) fall
            # back to moving just the active entity, matching old behavior.
            delta = (new_pos[0] - s[0], new_pos[1] - s[1], new_pos[2] - s[2])
            starts = g.get("starts") or {id(e): s}
            for ent in self.selection:
                es = starts.get(id(ent))
                if es is None:
                    continue  # joined the selection after the drag began
                ent.transform.position = Vec3(es[0] + delta[0], es[1] + delta[1],
                                              es[2] + delta[2])
        else:  # per-axis scale
            factor = max(0.05, 1.0 + t)
            if snap:
                factor = max(0.05, self._quantize(factor, self._snap_increment("scale")))
            self._apply_group_scale(g, factor, uniform=False)
        self.dirty = True

    # ---- details panel rows for the selected light / sun / fog volume ----
    @staticmethod
    def _slider_row(label, lo, hi, get, set_, fmt="{:.2f}"):
        return {"kind": "slider", "label": label, "min": lo, "max": hi,
                "get": get, "set": set_, "fmt": fmt}

    @staticmethod
    def _color_rows(get_color, set_color):
        def setter(i):
            def _set(v):
                c = list(get_color())
                c[i] = int(v)
                set_color(tuple(c))
            return _set
        return [Editor._slider_row("red", 0, 255, lambda: get_color()[0], setter(0), "{:.0f}"),
                Editor._slider_row("green", 0, 255, lambda: get_color()[1], setter(1), "{:.0f}"),
                Editor._slider_row("blue", 0, 255, lambda: get_color()[2], setter(2), "{:.0f}")]

    def _sun_rows(self, e):
        sun, scene, dl = e.sun, self.scene, self.scene.light
        slider = self._slider_row
        rows = [
            slider("intensity", 0.0, 8.0, lambda: dl.intensity,
                   lambda v: setattr(dl, "intensity", v)),
            *self._color_rows(lambda: dl.color, lambda c: setattr(dl, "color", c)),
            slider("ambient", 0.0, 1.0, lambda: dl.ambient,
                   lambda v: setattr(dl, "ambient", v)),
            slider("disc size", 0.2, 10.0, lambda: sun.disc_size,
                   lambda v: setattr(sun, "disc_size", v), "{:.1f}°"),
            slider("disc softness", 0.0, 1.0, lambda: sun.disc_softness,
                   lambda v: setattr(sun, "disc_softness", v)),
            slider("glow", 0.0, 1.0, lambda: sun.glow, lambda v: setattr(sun, "glow", v)),
            {"kind": "toggle", "label": "disc enabled", "get": lambda: sun.enabled,
             "set": lambda v: setattr(sun, "enabled", v)},
            slider("shadow softness", 0.0, 5.0, lambda: sun.shadow_softness,
                   lambda v: setattr(sun, "shadow_softness", v), "{:.1f}°"),
            slider("shadow depth", 0.0, 1.0, lambda: sun.shadow_depth,
                   lambda v: setattr(sun, "shadow_depth", v)),
            {"kind": "toggle", "label": "GI enabled",
             "get": lambda: scene.gi.get("enabled", False),
             "set": lambda v: scene.gi.__setitem__("enabled", v)},
            slider("GI intensity", 0.0, 4.0, lambda: scene.gi.get("intensity", 1.0),
                   lambda v: scene.gi.__setitem__("intensity", v)),
            slider("GI samples", 4.0, 64.0, lambda: scene.gi.get("samples", 16),
                   lambda v: scene.gi.__setitem__("samples", int(round(v))), "{:.0f}"),
        ]
        if scene.fog is not None:
            fog = scene.fog
            rows += [
                slider("haze height falloff", 0.0, 2.0, lambda: fog.height_falloff,
                       lambda v: setattr(fog, "height_falloff", v)),
                slider("haze sun scatter", 0.0, 1.0, lambda: fog.sun_scatter,
                       lambda v: setattr(fog, "sun_scatter", v)),
            ]
        return rows

    def _fog_volume_rows(self, e):
        fv = e.fog_volume
        slider = self._slider_row
        return [
            slider("density", 0.0, 3.0, lambda: fv.density,
                   lambda v: setattr(fv, "density", v)),
            *self._color_rows(lambda: fv.color, lambda c: setattr(fv, "color", c)),
            slider("height falloff", 0.0, 2.0, lambda: fv.height_falloff,
                   lambda v: setattr(fv, "height_falloff", v)),
            {"kind": "toggle", "label": "enabled", "get": lambda: fv.enabled,
             "set": lambda v: setattr(fv, "enabled", v)},
        ]

    # ---- details panel: editable Position/Rotation/Scale XYZ fields ----
    _TRANSFORM_AXES = ("x", "y", "z")

    def _transform_rows(self, e):
        """[{"label", "fields": [{"get","set"}, ...3]}, ...] for Position,
        Rotation (stored in radians, edited in degrees), and Scale. Rebuilt
        fresh on every call (like _details_rows) so gizmo-driven live edits
        and Vec3 objects the gizmo replaced never go stale."""
        t = e.transform

        def block(label, obj, to_ui=lambda v: v, from_ui=lambda v: v):
            fields = []
            for ax in self._TRANSFORM_AXES:
                def get(obj=obj, ax=ax, to_ui=to_ui):
                    return to_ui(getattr(obj, ax))

                def set_(v, obj=obj, ax=ax, from_ui=from_ui):
                    setattr(obj, ax, from_ui(v))
                fields.append({"get": get, "set": set_})
            return {"label": label, "fields": fields}

        return [block("Position", t.position),
                block("Rotation", t.rotation, math.degrees, math.radians),
                block("Scale", t.scale)]

    @staticmethod
    def _fmt_num(v: float) -> str:
        s = f"{v:.3f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s or "0"

    def _transform_row_rect(self, rect, i):
        import pygame
        return pygame.Rect(rect.x + 6, rect.y + TRANSFORM_ROWS_TOP + i * TRANSFORM_ROW_H,
                           rect.width - 12, TRANSFORM_ROW_H - 2)

    @staticmethod
    def _transform_field_rects(row_rect):
        """3 XYZ field rects within a transform row -- shared by draw + hit-test."""
        import pygame
        label_w = 62
        avail = max(0, row_rect.width - label_w)
        fw = max(20, (avail - 8) // 3)
        rects = []
        x = row_rect.x + label_w
        for _ in range(3):
            rects.append(pygame.Rect(x, row_rect.y, fw, row_rect.height))
            x += fw + 4
        return rects

    def _begin_edit_field(self, key, value) -> None:
        self._commit_edit_field()
        self.editing_field = key
        self.edit_buffer = self._fmt_num(value)

    def _commit_edit_field(self) -> None:
        """Enter, Tab, or a click elsewhere calls this. Invalid text reverts
        (the field simply never gets written) rather than raising."""
        if self.editing_field is None:
            return
        label, axis = self.editing_field
        buf = self.edit_buffer
        self.editing_field = None
        e = self.selected
        if e is None:
            return
        try:
            value = float(buf)
        except ValueError:
            return
        for row in self._transform_rows(e):
            if row["label"] == label:
                row["fields"][self._TRANSFORM_AXES.index(axis)]["set"](value)
                self.dirty = True
                return

    def _cancel_edit_field(self) -> None:
        self.editing_field = None

    def _update_edit_field(self, inp) -> None:
        import pygame
        for ch in inp.take_text():
            if ch.isdigit() or ch in "-.":
                self.edit_buffer += ch
        if inp.pressed(pygame.K_BACKSPACE):
            self.edit_buffer = self.edit_buffer[:-1]
        if inp.pressed(pygame.K_RETURN) or inp.pressed(pygame.K_KP_ENTER) \
                or inp.pressed(pygame.K_TAB):
            self._commit_edit_field()

    def _click_transform_fields(self, mp, rect, e) -> bool:
        for i, row in enumerate(self._transform_rows(e)):
            rr = self._transform_row_rect(rect, i)
            for j, fr in enumerate(self._transform_field_rects(rr)):
                if fr.collidepoint(mp):
                    self._begin_edit_field((row["label"], self._TRANSFORM_AXES[j]),
                                           row["fields"][j]["get"]())
                    return True
        return False

    def _details_rows(self):
        e = self.selected
        if e is None:
            return []
        rows = []
        if e.mesh is not None or e.environment is not None:
            rows.append({"kind": "button", "label": "material",
                         "text": "open node editor  (M)",
                         "action": lambda: setattr(self, "mat_ui",
                                                   MaterialEditorUI(self, e))})
        if e.mesh is not None:
            rows.append({"kind": "material_slot", "label": "material slot", "entity": e})
        if e.environment is not None:
            env = e.environment
            return rows + [{"kind": "slider", "label": "env strength", "min": 0.0,
                            "max": 3.0, "get": lambda: env.strength,
                            "set": lambda v: setattr(env, "strength", v),
                            "fmt": "{:.2f}"}]
        if e.sun is not None:
            return rows + self._sun_rows(e)
        if e.fog_volume is not None:
            return rows + self._fog_volume_rows(e)
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
                ("Import HDRI...", self._import_hdri_dialog, True),
                ("Exit", self._quit, True),
            ],
            "Edit": [
                ("Duplicate", self._duplicate_selected, True),
                ("Delete", self._delete_selected, True),
                ("Focus Selection", self._focus_selection, True),
                ("Snap to Floor", self._snap_to_floor, True),
                ("Reset 3D Cursor", self._reset_cursor3d, True),
            ],
            "Window": [
                ("Outliner", lambda: self._toggle_panel("outliner"), True),
                ("Details", lambda: self._toggle_panel("details"), True),
                ("Content Browser", lambda: self._toggle_panel("browser"), True),
                ("Console", lambda: self._toggle_panel("console"), True),
                ("Material Editor", self._toggle_material_editor, True),
                ("Fullscreen", self._toggle_fullscreen, True),
                ("Settings...", self._open_settings, True),
                ("Reset Layout", self._reset_layout, True),
            ],
            "Help": [
                ("Controls", self._toggle_controls, True),
                ("About", self._show_about, True),
            ],
        }

    def _menu_checked(self, label):
        """True/False for a checkmarked Window-menu item, None otherwise."""
        pid = self._WINDOW_PANEL_LABELS.get(label)
        if pid is not None:
            return self.panel_visible.get(pid, True)
        if label == "Material Editor":
            return self.mat_ui is not None
        if label == "Fullscreen":
            return self.eng.fullscreen
        return None

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
    # A panel dropped onto another docked panel's title/tab-strip joins its
    # GROUP as an extra tab (only one docked side, left/right/bottom -- a
    # floating panel is never a tab-join target, out of scope for this
    # feature); dropped over a plain edge/band zone it still side-by-side
    # stacks as a fresh solo group, exactly like the pre-tabs flat-list
    # model. `_PANEL_ALLOWED_SIDES` is unchanged from the pre-tabs
    # restriction (outliner/details are side-dock-only, browser is
    # bottom-dock-only) and now gates BOTH kinds of drop.
    _PANEL_ALLOWED_SIDES = {"outliner": ("left", "right"), "details": ("left", "right"),
                            "browser": ("bottom",),
                            # "bottom" listed first: it's this panel's
                            # fallback side in _apply_layout_settings when
                            # migrating a settings.json saved before the
                            # console panel existed (index [0] of this
                            # tuple), matching where it lands by default in
                            # a fresh install (Editor.__init__/_reset_layout)
                            "console": ("bottom", "left", "right")}

    @staticmethod
    def _remove_pid_from(dock_order, floating, pid) -> None:
        """Strip `pid` out of wherever it currently lives in `dock_order`/
        `floating` -- a group loses just that id (reassigning "active" if it
        was the one removed; dropped entirely once empty) or the floating
        list loses the id. Takes explicit dock_order/floating arguments (not
        self.*) so `_simulate_drop` can replay the exact same mutation
        against a scratch clone for the drag-preview -- the no-drift
        requirement."""
        for side in ("left", "right", "bottom"):
            for group in list(dock_order[side]):
                if pid in group["ids"]:
                    group["ids"].remove(pid)
                    if group["active"] == pid:
                        group["active"] = group["ids"][0] if group["ids"] else pid
                    if not group["ids"]:
                        dock_order[side].remove(group)
        if pid in floating:
            floating.remove(pid)

    def _clone_dock_state(self):
        """Deep-enough copy of dock_order/floating for `_simulate_drop` to
        mutate freely without touching live state."""
        order = {side: [{"ids": list(g["ids"]), "active": g["active"]} for g in groups]
                for side, groups in self.dock_order.items()}
        return order, list(self.floating)

    def _apply_panel_drop(self, dock_order, floating, pid, target) -> None:
        """The one mutation both a real drop (`_finish_panel_drag`) and the
        preview simulation (`_simulate_drop`) perform against whichever
        dock_order/floating they're given -- live state or a scratch clone
        -- so the preview can never show a result the real drop wouldn't
        produce. `target` is whatever `_panel_drag_target` returned."""
        if target["kind"] == "tab":
            side, anchor = target["side"], target["anchor"]
            group = next((g for g in dock_order[side] if anchor in g["ids"]), None)
            if group is not None and pid in group["ids"]:
                group["active"] = pid  # already a tab here -- just switch focus
                return
            self._remove_pid_from(dock_order, floating, pid)
            if group is None:  # anchor WAS `pid` and vanished when its
                                # (degenerate) group emptied out above
                group = next((g for g in dock_order[side] if anchor in g["ids"]), None)
            if group is None:  # nothing left to join -- land as a fresh solo slot
                dock_order[side].append(self._solo_group(pid))
            else:
                group["ids"].append(pid)
                group["active"] = pid
            return
        self._remove_pid_from(dock_order, floating, pid)
        if target["kind"] == "band":
            dock_order[target["side"]].append(self._solo_group(pid))
        else:
            floating.append(pid)

    def _dock_panel(self, pid, side) -> None:
        self._remove_pid_from(self.dock_order, self.floating, pid)
        if side == "float":
            self.floating.append(pid)
        else:
            self.dock_order[side].append(self._solo_group(pid))
        self._save_settings()

    def _full_panel_size(self, pid, rect):
        """The panel's un-minimized (w, h) — used so a drag on a minimized
        title bar doesn't bake the collapsed height into float_rect."""
        stored = self.float_rect.get(pid)
        if stored is not None:
            return stored.width, stored.height
        if self.panel_minimized.get(pid, False):
            return PANEL_DEFAULT_FLOAT[pid]
        return rect.width, rect.height

    def _begin_panel_drag(self, pid, mp, rect) -> None:
        fw, fh = self._full_panel_size(pid, rect)
        self.panel_drag = {"id": pid, "dx": mp[0] - rect.x, "dy": mp[1] - rect.y,
                           "w": fw, "h": fh}

    def _tab_header_rects(self, ids, rect):
        """[left-packed header rect per visible tab id] within `rect`'s top
        PANEL_TITLE_H band, leaving room for the minimize/close buttons at
        the right end -- shared by `_draw_tab_strip` (drawing) and the
        mousedown router (hit-testing), per the house rule against
        hit-test/draw drift."""
        import pygame
        out = {}
        x = rect.x + 2
        limit = rect.right - 38  # minimize+close buttons live past this
        for pid in ids:
            if x >= limit:
                break
            label = TAB_LABELS.get(pid, pid)
            tw = self.font_small.size(label)[0] + 16
            x2 = min(x + tw, limit)
            out[pid] = pygame.Rect(x, rect.y + 1, max(0, x2 - x), PANEL_TITLE_H - 2)
            x = x2
        return out

    def _panel_tab_strip(self, pid, rect):
        """{member_pid: header_rect} for `pid`'s tab strip if it's currently
        in a REAL dock group (>=2 VISIBLE members) -- None if `pid` is
        floating or its group is degenerate (a lone panel), in which case
        callers must fall back to the plain single-title-bar rendering so
        that case stays byte-identical to the pre-tabs layout."""
        info = self._group_for_pid(pid)
        if info is None:
            return None
        _, group = info
        vis = [p for p in group["ids"] if self.panel_visible.get(p, True)]
        if len(vis) < 2:
            return None
        return self._tab_header_rects(vis, rect)

    def _dock_zone_rect(self, side, w, h, layout):
        """Rect used for BOTH drop-zone hit-testing (`_finish_panel_drag`) and
        the hover-highlight drawn while dragging (`_draw_panel_drag_zone`) --
        one source of truth, per the house rule against hit-test/draw drift.

        If `side` already has a dock, the zone is that dock's full band --
        drop anywhere over an existing side/bottom dock and it re-docks there,
        not just within a thin edge strip. If the side has no dock yet, the
        zone is an edge strip sized to whichever is larger: EDGE_SNAP or 4% of
        the window's relevant dimension, so a big/fullscreen window doesn't
        make docking feel broken (a real user complaint -- EDGE_SNAP=48px is
        a sliver on a 2560px-wide window)."""
        import pygame
        tb_w = layout.get("side_toolbar_w", 0)
        if side == "left":
            width = layout["left_w"] or max(EDGE_SNAP, int(w * 0.04))
            return pygame.Rect(tb_w, MENU_H, width, max(0, h - MENU_H))
        if side == "right":
            width = layout["right_w"] or max(EDGE_SNAP, int(w * 0.04))
            return pygame.Rect(w - width, MENU_H, width, max(0, h - MENU_H))
        if side == "bottom":
            height = layout["bottom_h"] or max(EDGE_SNAP, int(h * 0.04))
            return pygame.Rect(tb_w, h - height, max(0, w - tb_w), height)
        raise ValueError(side)

    def _panel_drag_target_side(self, pid, mp, w, h, layout):
        """Which dock zone (if any) a drag of `pid` is currently over -- used
        both to decide where a "band" drop docks and to draw the hover
        highlight; `_panel_drag_target` checks this AFTER the more specific
        tab-join rects."""
        for side in self._PANEL_ALLOWED_SIDES.get(pid, ()):
            if self._dock_zone_rect(side, w, h, layout).collidepoint(mp):
                return side
        return None

    def _panel_drag_target(self, pid, mp, w, h, layout):
        """Single source of truth for where a title/tab-strip drag of `pid`
        will land if released right now -- used by BOTH `_finish_panel_drag`
        (the real drop) and the ghost preview drawn while dragging
        (`_simulate_drop`), so the preview can never show a spot the drop
        wouldn't actually honor. Returns one of:
          {"kind": "tab", "side": side, "anchor": apid}  -- join apid's group
          {"kind": "band", "side": side}                 -- new stacked group
          {"kind": "float"}                              -- floats

        Checked most-specific first: mp over an existing DOCKED panel's
        title/tab-strip rect (any currently-active pid in `layout["panels"]`
        that belongs to a dock group, gated by `_PANEL_ALLOWED_SIDES` so a
        browser can't tab into a side-dock slot or vice versa) beats the
        coarser edge/band zones -- exactly the "as opposed to the dock
        band's edge zones" distinction from the spec. A floating panel is
        never a tab-join target (out of scope)."""
        import pygame
        for apid, rect in layout["panels"].items():
            info = self._group_for_pid(apid)
            if info is None:
                continue  # floating -- not a tab-join target
            side = info[0]
            if side not in self._PANEL_ALLOWED_SIDES.get(pid, ()):
                continue
            title_rect = pygame.Rect(rect.x, rect.y, rect.width, PANEL_TITLE_H)
            if title_rect.collidepoint(mp):
                return {"kind": "tab", "side": side, "anchor": apid}
        side = self._panel_drag_target_side(pid, mp, w, h, layout)
        if side is not None:
            return {"kind": "band", "side": side}
        return {"kind": "float"}

    def _simulate_drop(self, pid, target, w, h):
        """The rect `pid` would occupy immediately after finishing this
        drag: replays the exact drop mutation (`_apply_panel_drop`) on a
        scratch clone of the dock state and re-runs the real `_layout` math
        -- so the drag-preview ghost is computed from the same layout math
        a real drop would produce, never hand-approximated. None if `pid`
        wouldn't end up placed (shouldn't happen for "tab"/"band" targets)."""
        order, floating = self._clone_dock_state()
        self._apply_panel_drop(order, floating, pid, target)
        sim = self._layout(w, h, dock_order=order)
        return sim["panels"].get(pid)

    def _finish_panel_drag(self, mp, w, h) -> None:
        import pygame
        g = self.panel_drag
        pid = g["id"]
        gx, gy = mp[0] - g["dx"], mp[1] - g["dy"]
        layout = self._layout(w, h)
        target = self._panel_drag_target(pid, mp, w, h, layout)
        if target["kind"] == "float":
            self.float_rect[pid] = pygame.Rect(gx, gy, g["w"], g["h"])
            self._apply_panel_drop(self.dock_order, self.floating, pid, target)
            self._float_rect_for(pid, w, h)  # clamp on-screen
        else:
            self._apply_panel_drop(self.dock_order, self.floating, pid, target)
        self._save_settings()
        self.panel_drag = None

    # ---- splitter drag (docked panel resize) ----
    def _update_splitter_drag(self, side, mp, w, h) -> None:
        """Drag updates `dock_frac[side]` directly (as a fraction of the
        *current* window size) so the dock stays proportional across a
        later resize/fullscreen toggle instead of freezing at this pixel
        width. Clamped the same way `_dock_side_w`/`_dock_bottom_h` clamp on
        read, so the splitter can't be dragged past the min/max it enforces."""
        if side == "left":
            px = max(MIN_PANEL_W, min(mp[0], int(w * 0.6)))
            self.dock_frac["left"] = px / w
        elif side == "right":
            px = max(MIN_PANEL_W, min(w - mp[0], int(w * 0.6)))
            self.dock_frac["right"] = px / w
        elif side == "bottom":
            px = max(MIN_PANEL_H, min(h - mp[1], int(h * 0.6)))
            self.dock_frac["bottom"] = px / h

    # ---- floating-panel corner resize ----
    def _begin_panel_resize(self, pid, mp, rect) -> None:
        self.panel_resize = {"id": pid, "w": rect.width, "h": rect.height,
                             "mx": mp[0], "my": mp[1]}

    def _update_panel_resize(self, mp) -> None:
        import pygame
        g = self.panel_resize
        pid = g["id"]
        nw = max(MIN_PANEL_W, g["w"] + (mp[0] - g["mx"]))
        nh = max(MIN_PANEL_H, g["h"] + (mp[1] - g["my"]))
        r = self.float_rect.get(pid)
        if r is None:
            r = pygame.Rect(0, 0, nw, nh)
            self.float_rect[pid] = r
        r.width, r.height = nw, nh

    # ---- hover affordance / fullscreen ----
    def _update_cursor(self, mp, layout) -> None:
        import pygame
        side = self.splitter_drag
        if side is None:
            for s, r in layout["splitters"].items():
                if r.collidepoint(mp):
                    side = s
                    break
        if side in ("left", "right"):
            cursor = pygame.SYSTEM_CURSOR_SIZEWE
        elif side == "bottom":
            cursor = pygame.SYSTEM_CURSOR_SIZENS
        elif self.panel_resize is not None:
            cursor = pygame.SYSTEM_CURSOR_SIZENWSE
        else:
            cursor = pygame.SYSTEM_CURSOR_ARROW
        try:  # the SDL dummy video driver (headless benchmarking/tests) has
            pygame.mouse.set_cursor(cursor)  # no real cursor to set
        except pygame.error:
            pass

    def _toggle_fullscreen(self) -> None:
        self.eng.set_fullscreen(not self.eng.fullscreen)
        self._save_settings()

    def _toggle_panel(self, pid) -> None:
        self.panel_visible[pid] = not self.panel_visible.get(pid, True)
        self._save_settings()

    def _toggle_minimize(self, pid) -> None:
        self.panel_minimized[pid] = not self.panel_minimized.get(pid, False)
        self._save_settings()

    def _toggle_material_editor(self) -> None:
        if self.mat_ui is not None:
            self.mat_ui.close()
            return
        e = self.selected
        if e is None or e.mesh is None:
            self.status = ("select a mesh entity first", 3.0)
            return
        self.mat_ui = MaterialEditorUI(self, e)

    def _reset_layout(self) -> None:
        self.dock_order = {"left": [], "right": [self._solo_group("outliner"),
                                                 self._solo_group("details")],
                           "bottom": [self._solo_group("browser"),
                                     self._solo_group("console")]}
        self.floating = []
        self.panel_visible = {"outliner": True, "details": True, "browser": True,
                              "console": True}
        self.panel_minimized = {"outliner": False, "details": False, "browser": False,
                                "console": False}
        self.float_rect = {}
        self.dock_frac = dict(DOCK_FRAC_DEFAULT)
        self._save_settings()

    # ---- settings dialog ----
    def _open_settings(self) -> None:
        self.settings_open = True
        self.settings_drag = None

    def _toggle_controls(self) -> None:
        self.show_controls_overlay = not self.show_controls_overlay

    def _show_about(self) -> None:
        self.status = ("PyEngine 0.1 — pure-Python real-time 3D engine", 4.0)

    def _active_api(self) -> str:
        """Which backend this session actually ended up running, read from
        the live engine (not from any saved preference) -- "gl", "dx12",
        "vulkan", or "cpu"."""
        eng = self.eng
        if eng.gl_renderer is not None:
            return "gl"
        if eng.wgpu_renderer is not None:
            return eng.wgpu_renderer.stats.get("mode", "dx12")
        return "cpu"

    def _software_active(self) -> bool:
        """True while the software (CPU) renderer is the one actually
        running -- gates the pixel-scale slider, which only affects that
        renderer's per-pixel pass."""
        return self.eng.gl_renderer is None and self.eng.wgpu_renderer is None

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

    def _settings_api_buttons(self, rect):
        import pygame
        x, y = rect.x + 12, rect.y + 180
        bw, bh, gap = 62, 22, 6
        out = []
        for key in ("dx12", "vulkan", "gl", "auto", "cpu"):
            out.append((key, pygame.Rect(x, y, bw, bh)))
            x += bw + gap
        return out

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
            for key, btn in self._settings_api_buttons(rect):
                if btn.collidepoint(mp):
                    self.api_pref = key
                    self._save_settings()
                    self.status = ("Graphics API preference saved — restart to apply", 3.0)
                    return
            for which in ("pixel", "max_fps"):
                if which == "pixel" and not self._software_active():
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
        # the *windowed* size, not the live one -- while fullscreen, the live
        # size is just the desktop resolution, and saving that would make
        # "un-fullscreen" on next launch pointless (see Engine.set_fullscreen)
        w, h = getattr(self.eng, "_windowed_size", self.eng.screen.get_size())
        return {
            "width": w, "height": h,
            "pixel_scale": int(self.eng.renderer.render_scale),
            "max_fps": int(self.eng.max_fps),
            "api": self.api_pref,
            "fullscreen": bool(self.eng.fullscreen),
            "panel_visible": dict(self.panel_visible),
            "panel_minimized": dict(self.panel_minimized),
            "dock_order": {side: [{"ids": list(g["ids"]), "active": g["active"]}
                                  for g in groups]
                          for side, groups in self.dock_order.items()},
            "floating": list(self.floating),
            "float_rect": {pid: [r.x, r.y, r.width, r.height]
                          for pid, r in self.float_rect.items()},
            "dock_frac": dict(self.dock_frac),
            "snap_enabled": self.snap_enabled,
            "snap_index": dict(self.snap_index),
            "pivot_mode": self.pivot_mode,
            "cursor3d": [self.cursor3d.x, self.cursor3d.y, self.cursor3d.z],
            "side_toolbar_collapsed": self.side_toolbar_collapsed,
        }

    def _save_settings(self) -> None:
        save_settings(self._settings_dict(), self.settings_path)

    def _apply_snap_settings(self, data: dict) -> None:
        self.snap_enabled = bool(data.get("snap_enabled", self.snap_enabled))
        idx = data.get("snap_index", {})
        if isinstance(idx, dict):
            for mode, incs in self._SNAP_INCREMENTS.items():
                v = idx.get(mode)
                if isinstance(v, int) and 0 <= v < len(incs):
                    self.snap_index[mode] = v

    def _apply_pivot_settings(self, data: dict) -> None:
        mode = data.get("pivot_mode")
        if mode in self._PIVOT_MODES:
            self.pivot_mode = mode

    def _apply_cursor_settings(self, data: dict) -> None:
        c = data.get("cursor3d")
        if (isinstance(c, list) and len(c) == 3
                and all(isinstance(v, (int, float)) for v in c)):
            self.cursor3d = self.engine_mod.Vec3(float(c[0]), float(c[1]), float(c[2]))

    def _apply_side_toolbar_settings(self, data: dict) -> None:
        v = data.get("side_toolbar_collapsed")
        if isinstance(v, bool):
            self.side_toolbar_collapsed = v

    def _migrate_dock_order(self, raw: dict, valid: set) -> dict:
        """Build fresh dock groups from a saved `dock_order`, accepting
        EITHER the pre-tabs flat format ({side: [pid, ...]}) or the current
        group format ({side: [{"ids": [...], "active": pid}, ...]} written
        by `_settings_dict`). An old-format entry becomes a solo (single-
        tab) group -- the same "lone panel is a group of one" degenerate
        case _layout already renders identically to the flat-list era --
        so a settings.json saved before this feature loads exactly as it
        looked before."""
        out = {"left": [], "right": [], "bottom": []}
        for side in ("left", "right", "bottom"):
            for item in raw.get(side, []):
                if isinstance(item, str):  # old flat {side: [pid, ...]} format
                    if side in self._PANEL_ALLOWED_SIDES.get(item, ()) and item in valid:
                        out[side].append(self._solo_group(item))
                elif isinstance(item, dict):  # current {"ids", "active"} group format
                    ids = [p for p in item.get("ids", [])
                          if side in self._PANEL_ALLOWED_SIDES.get(p, ()) and p in valid]
                    ids = list(dict.fromkeys(ids))  # de-dupe, keep first occurrence's order
                    if not ids:
                        continue
                    active = item.get("active")
                    if active not in ids:
                        active = ids[0]
                    out[side].append({"ids": ids, "active": active})
        return out

    def _apply_layout_settings(self, data: dict) -> None:
        import pygame
        self._apply_snap_settings(data)
        self._apply_pivot_settings(data)
        self._apply_cursor_settings(data)
        self._apply_side_toolbar_settings(data)
        valid = {"outliner", "details", "browser", "console"}
        migrated = self._migrate_dock_order(data.get("dock_order", {}), valid)
        floating = [p for p in data.get("floating", []) if p in valid]
        placed = ([p for side in ("left", "right", "bottom")
                  for g in migrated[side] for p in g["ids"]] + floating)
        if not placed:
            return  # no saved layout at all (fresh install / blank temp
                     # settings in a test) -- keep the constructor's
                     # hand-built default untouched, exactly like before
        if len(placed) != len(set(placed)):
            return  # a pid placed twice somewhere: corrupt data, keep default
        # a panel introduced AFTER whatever version wrote `data` (e.g.
        # "console", added post-hoc) won't be in `placed` -- default-dock it
        # instead of discarding the WHOLE saved layout just because one new
        # id is absent, which would otherwise reset every other panel's
        # carefully-arranged position on every settings.json written before
        # this panel existed.
        for pid in valid - set(placed):
            side = self._PANEL_ALLOWED_SIDES.get(pid, ("bottom",))[0]
            migrated[side].append(self._solo_group(pid))
            placed.append(pid)
        if set(placed) != valid:
            return  # still partial/corrupt: keep the built-in default
        self.dock_order = migrated
        self.floating = floating
        pv = data.get("panel_visible", {})
        for pid in valid:
            if pid in pv:
                self.panel_visible[pid] = bool(pv[pid])
        pm = data.get("panel_minimized", {})
        for pid in valid:
            if pid in pm:
                self.panel_minimized[pid] = bool(pm[pid])
        for pid, v in data.get("float_rect", {}).items():
            if pid in valid and isinstance(v, list) and len(v) == 4:
                self.float_rect[pid] = pygame.Rect(*v)
        df = data.get("dock_frac", {})
        for side in ("left", "right", "bottom"):
            v = df.get(side)
            if isinstance(v, (int, float)) and 0.02 <= v <= 0.8:
                self.dock_frac[side] = float(v)

    # ---- File / Edit menu actions (shared with hotkeys) ----
    def _duplicate_selected(self, offset: bool = True) -> None:
        """Ctrl+D (and Alt-drag, via _alt_duplicate_for_drag) duplicates
        every entity in the selection in place, each with a visible
        +0.8/+0.8 nudge so the copies don't sit exactly under their
        originals. Alt-drag passes offset=False: the drag itself is what
        moves the copies, so they must start exactly where the originals
        were. The duplicates become the new selection; the duplicate of the
        previous ACTIVE element becomes the new active element (falls back
        to the last duplicate if the previous active had no asset_name and
        was skipped). Single-selection callers get exactly the old
        one-entity behavior.
        """
        Vec3 = self.engine_mod.Vec3
        nudge = 0.8 if offset else 0.0
        srcs = [e for e in self.selection if e.asset_name is not None]
        if not srcs:
            return
        prev_active = self.selected
        dups, active_dup = [], None
        for src in srcs:
            dup = self.lib.instantiate(src.asset_name)
            t, s = dup.transform, src.transform
            t.position = Vec3(s.position.x + nudge, s.position.y, s.position.z + nudge)
            t.rotation = Vec3(s.rotation.x, s.rotation.y, s.rotation.z)
            t.scale = Vec3(s.scale.x, s.scale.y, s.scale.z)
            self._copy_entity_state(src, dup)
            self.scene.add(dup)
            dups.append(dup)
            if src is prev_active:
                active_dup = dup
        self._set_selection(dups, active=active_dup)
        self.dirty = True

    def _delete_selected(self) -> None:
        targets = [e for e in self.selection if e.asset_name is not None]
        if not targets:
            return
        remaining = [e for e in self.selection if e not in targets]
        for e in targets:
            self.scene.remove(e)
        self._set_selection(remaining)
        self.dirty = True

    def _focus_selection(self) -> None:
        if self.selection:
            self._focus(self.selection)

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

    # ---- content browser: top bar / folder tree / grid layout ----
    def _browser_layout(self, content):
        """topbar / tree / grid sub-rects of the browser panel's content
        rect -- the single source both `_draw_browser` and the browser
        click router use, so drawing and hit-testing never disagree."""
        import pygame
        topbar = pygame.Rect(content.x, content.y, content.width, BROWSER_TOPBAR_H)
        below = pygame.Rect(content.x, topbar.bottom, content.width,
                            max(0, content.height - BROWSER_TOPBAR_H))
        tree_w = min(TREE_W, max(60, below.width // 3))
        tree = pygame.Rect(below.x, below.y, tree_w, below.height)
        grid = pygame.Rect(below.x + tree_w, below.y,
                           max(0, below.width - tree_w), below.height)
        return {"topbar": topbar, "tree": tree, "grid": grid}

    def _new_folder_btn_rect(self, topbar):
        import pygame
        return pygame.Rect(topbar.x + 4, topbar.y + 3, 90, 20)

    def _new_blueprint_btn_rect(self, topbar):
        import pygame
        nfb = self._new_folder_btn_rect(topbar)
        return pygame.Rect(nfb.right + 4, topbar.y + 3, 96, 20)

    def _import_btn_rect(self, topbar):
        import pygame
        return pygame.Rect(topbar.right - 74, topbar.y + 3, 70, 20)

    def _export_btn_rect(self, topbar):
        import pygame
        imp = self._import_btn_rect(topbar)
        return pygame.Rect(imp.x - 78, topbar.y + 3, 70, 20)

    # ---- content browser: folder tree ----
    def _folder_tree_rows(self):
        """Flattened [(folder_id_or_None, depth, name)], root first, then a
        depth-first walk of `lib.folders` sorted by name at each level."""
        rows = [(None, 0, "Assets")]

        def rec(parent, depth):
            for fid in self.lib.folder_children(parent):
                rows.append((fid, depth, self.lib.folders[fid]["name"]))
                rec(fid, depth + 1)

        rec(None, 1)
        return rows

    def _tree_row_rect(self, tree_rect, i):
        import pygame
        y = tree_rect.y + 4 + i * TREE_ROW_H - self.tree_scroll
        return pygame.Rect(tree_rect.x + 2, y, tree_rect.width - 4, TREE_ROW_H - 1)

    @staticmethod
    def _tree_rows_clip(tree_rect):
        """Rows area excluding the bottom "F2 rename" hint strip -- shared by
        drawing and hit-testing so a row half-hidden behind the hint can't be
        clicked, and the hint never draws over a fully visible row."""
        import pygame
        return pygame.Rect(tree_rect.x, tree_rect.y, tree_rect.width,
                           max(0, tree_rect.height - 16))

    def _tree_row_at(self, mp, tree_rect):
        """(index, folder_id) of the tree row under mp, or (None, None)."""
        clip = self._tree_rows_clip(tree_rect)
        if not clip.collidepoint(mp):
            return None, None
        for i, (fid, _depth, _name) in enumerate(self._folder_tree_rows()):
            r = self._tree_row_rect(tree_rect, i)
            if r.bottom <= clip.bottom and r.collidepoint(mp):
                return i, fid
        return None, None

    def _new_folder(self) -> None:
        parent = self.selected_folder
        existing = {self.lib.folders[c]["name"] for c in self.lib.folder_children(parent)}
        base, name, n = "New Folder", "New Folder", 2
        while name in existing:
            name = f"{base} {n}"
            n += 1
        fid = self.lib.create_folder(name, parent)
        self.lib.save_folders()
        self.selected_folder = fid
        self._begin_rename(fid)

    def _new_blueprint(self) -> None:
        """"+ Blueprint" topbar button: create a blueprint asset (starter
        script that already compiles clean), file it into the currently
        selected folder like a new folder would be, and open it straight
        into the script editor -- the natural next step after creating one."""
        bp = self.lib.new_blueprint()
        self.lib.set_asset_folder(bp.name, self.selected_folder)
        self.lib.save_folders()
        self.bp_icons[bp.name] = make_blueprint_icon()
        self.selected_asset = bp
        self.script_ui = ScriptEditorUI(self, bp)

    def _begin_rename(self, folder_id) -> None:
        """No-ops for the root folder (id None) and unknown ids -- root has
        no name to edit, and there's no "F2 renamed root" state to enter."""
        if folder_id is None or folder_id not in self.lib.folders:
            return
        self.renaming_folder = folder_id
        self.rename_buffer = self.lib.folders[folder_id]["name"]

    def _commit_rename(self) -> None:
        fid, name = self.renaming_folder, self.rename_buffer.strip()
        self.renaming_folder = _NO_RENAME
        if fid is not _NO_RENAME and fid is not None and name:
            self.lib.rename_folder(fid, name)
            self.lib.save_folders()

    def _cancel_rename(self) -> None:
        self.renaming_folder = _NO_RENAME

    def _update_rename(self, inp) -> None:
        import pygame
        self.rename_buffer += inp.take_text()
        if inp.pressed(pygame.K_BACKSPACE):
            self.rename_buffer = self.rename_buffer[:-1]
        if inp.pressed(pygame.K_RETURN) or inp.pressed(pygame.K_KP_ENTER):
            self._commit_rename()

    # ---- content browser: import routed into the selected folder ----
    def _import_dialog_to_folder(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Import asset",
                filetypes=[("Supported", "*.fbx *.hdr *.png *.jpg *.jpeg *.bmp"),
                          ("FBX models", "*.fbx"), ("Radiance HDR", "*.hdr"),
                          ("Textures", "*.png *.jpg *.jpeg *.bmp"),
                          ("All files", "*.*")])
            root.destroy()
        except Exception as ex:
            self.status = (f"file dialog unavailable: {ex}", 5.0)
            return
        if not path:
            return
        self._open_import_dialog(path)

    def _open_import_dialog(self, path: str) -> None:
        """Open the Unreal-style import-options modal for `path` instead of
        importing immediately -- lets the user rename, refile, and (for
        meshes) rescale/reorient before anything is written to disk.
        Cancel writes nothing; Import routes to `_do_import` (see
        `_import_confirm`). Kept separate from `_import_path_to_folder`,
        the dialog-free instant-import path tests and any future Explorer
        drag-drop use.
        """
        ext = os.path.splitext(path)[1].lower()
        kind_label = _IMPORT_TYPE_LABELS.get(ext)
        if kind_label is None:
            self.status = (f"unsupported file type: {ext}", 5.0)
            return
        kind, label = kind_label
        stem = os.path.splitext(os.path.basename(path))[0]
        default_name = stem.replace("_", " ").strip().title() or "Imported Asset"
        self.import_dialog = {
            "path": path, "ext": ext, "kind": kind, "label": label,
            "name": default_name, "folder": self.selected_folder,
            "scale_text": "1", "up_axis": "y",
            # "Generate LODs" (mesh only): on by default -- inert for meshes
            # at or below lod.LOD_FACE_THRESHOLD (generate_lods no-ops) and
            # for texture/HDRI kinds (grayed out like Scale/Up Axis below).
            "generate_lods": True,
        }
        self.import_field = None

    def _face_count_suffix(self, asset_name: str) -> str:
        """" (<N> faces)" for a mesh asset, "" otherwise -- used to shape the
        console-log import-complete message; cheap (one extra instantiate,
        same cost `make_icon` already pays for the thumbnail)."""
        asset = self.lib.by_name.get(asset_name)
        if asset is None:
            return ""
        entity = asset.instantiate()
        if entity.mesh is None:
            return ""
        return f" ({entity.mesh.faces.shape[0]} faces)"

    def _import_path_to_folder(self, path: str) -> None:
        """Import an .fbx/.hdr file and assign it to `self.selected_folder`.

        Split out from `_import_dialog_to_folder` so tests (and any future
        drag-and-drop from Explorer) can drive the import with a real path,
        no tkinter dialog involved.
        """
        from engine import console_log
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".fbx":
                console_log.log_info(f"Importing '{os.path.basename(path)}'...")
                name = self.engine_mod.import_fbx(path, self.lib.directory)
            elif ext == ".hdr":
                console_log.log_info(f"Importing '{os.path.basename(path)}'...")
                name = self.engine_mod.import_hdri(path, self.lib.directory)
            elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
                console_log.log_info(f"Importing '{os.path.basename(path)}'...")
                name = self.engine_mod.import_texture(path, self.lib.directory)
            else:
                self.status = (f"unsupported file type: {ext}", 5.0)
                return
            self.lib.reload()
            self.lib.set_asset_folder(name, self.selected_folder)
            self.lib.save_folders()
            self.icons[name] = make_icon(self.engine_mod, self.lib.by_name[name])
            console_log.log_info(f"imported '{name}'{self._face_count_suffix(name)}")
            self.status = (f"imported '{name}' — drag it from the browser", 5.0)
        except Exception as ex:
            console_log.log_error(f"import failed: {ex}")
            self.status = (f"import failed: {ex}", 6.0)

    def _rename_new_asset(self, old_name: str, new_name: str) -> str:
        """Apply the import dialog's Name field to a just-imported asset's
        JSON, de-duped against existing names (same pattern `_new_folder`
        uses). Called before `set_asset_folder` so the folder assignment
        lands on the final name. Returns the name actually in effect (the
        original name if `new_name` is blank or unchanged)."""
        new_name = (new_name or "").strip()
        if not new_name or new_name == old_name:
            return old_name
        asset = self.lib.by_name.get(old_name)
        if asset is None:
            return old_name
        existing = set(self.lib.by_name) - {old_name}
        final, n = new_name, 2
        while final in existing:
            final = f"{new_name} {n}"
            n += 1
        asset.data["name"] = final
        with open(asset.path, "w", encoding="utf-8") as f:
            json.dump(asset.data, f, indent=2)
        self.lib.reload()
        return final

    def _do_import(self, path: str, name: str = "", folder=None,
                   scale: float = 1.0, up_axis: str = "y",
                   generate_lods: bool = True) -> None:
        """Dialog-free import with options: applies the chosen name/folder
        and, for a mesh, bakes the uniform scale + up-axis conversion into
        the stored vertices (Unreal-style "bake import transform"), then
        NAVIGATES the browser to the result (selected_folder +
        selected_asset) so the new tile is immediately visible and
        selected -- the "imported but hidden in another folder" problem
        this dialog exists to fix. The import dialog's Import button
        (`_import_confirm`) and tests both call this; a future Explorer
        drag-drop could call it directly with the defaults to skip the
        dialog entirely, same contract `_import_path_to_folder` established.
        """
        from engine import console_log
        ext = os.path.splitext(path)[1].lower()
        try:
            console_log.log_info(f"Importing '{os.path.basename(path)}'...")
            if ext == ".fbx":
                asset_name = self.engine_mod.import_fbx(
                    path, self.lib.directory, scale=scale, up_axis=up_axis,
                    generate_lods=generate_lods)
            elif ext == ".hdr":
                asset_name = self.engine_mod.import_hdri(path, self.lib.directory)
            elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
                asset_name = self.engine_mod.import_texture(path, self.lib.directory)
            else:
                self.status = (f"unsupported file type: {ext}", 5.0)
                return
            self.lib.reload()
            asset_name = self._rename_new_asset(asset_name, name)
            self.lib.set_asset_folder(asset_name, folder)
            self.lib.save_folders()
            self.icons[asset_name] = make_icon(self.engine_mod, self.lib.by_name[asset_name])
            self.selected_folder = folder
            self.selected_asset = self.lib.by_name[asset_name]
            console_log.log_info(
                f"imported '{asset_name}'{self._face_count_suffix(asset_name)}")
            self.status = (f"imported '{asset_name}'", 5.0)
        except Exception as ex:
            console_log.log_error(f"import failed: {ex}")
            self.status = (f"import failed: {ex}", 6.0)

    # ---- content browser: import-options dialog (rects, styled like Settings) ----
    def _import_rect(self, w, h):
        import pygame
        iw, ih = IMPORT_SIZE
        return pygame.Rect((w - iw) // 2, max(MENU_H + 20, (h - ih) // 2), iw, ih)

    def _import_close_rect(self, rect):
        import pygame
        return pygame.Rect(rect.right - 26, rect.y + 5, 20, 20)

    def _import_name_rect(self, rect):
        import pygame
        return pygame.Rect(rect.x + 12, rect.y + 76, rect.width - 24, 24)

    def _import_folder_rects(self, rect):
        """(prev_btn, label_rect, next_btn) for the target-folder cycle row."""
        import pygame
        row = pygame.Rect(rect.x + 12, rect.y + 124, rect.width - 24, 24)
        prev_btn = pygame.Rect(row.x, row.y, 26, row.height)
        next_btn = pygame.Rect(row.right - 26, row.y, 26, row.height)
        label = pygame.Rect(prev_btn.right + 4, row.y,
                            next_btn.x - prev_btn.right - 8, row.height)
        return prev_btn, label, next_btn

    def _import_scale_rect(self, rect):
        import pygame
        return pygame.Rect(rect.x + 12, rect.y + 172, 100, 24)

    def _import_fit_btn_rect(self, rect):
        import pygame
        s = self._import_scale_rect(rect)
        return pygame.Rect(s.right + 10, s.y, rect.right - 12 - (s.right + 10), 24)

    def _import_axis_rects(self, rect):
        """(y_btn, z_btn) for the Up Axis toggle."""
        import pygame
        y0 = rect.y + 220
        y_btn = pygame.Rect(rect.x + 12, y0, 60, 24)
        z_btn = pygame.Rect(rect.x + 12 + 68, y0, 60, 24)
        return y_btn, z_btn

    def _import_lod_checkbox_rect(self, rect):
        import pygame
        return pygame.Rect(rect.x + 12, rect.y + 268, 18, 18)

    def _import_cancel_rect(self, rect):
        import pygame
        return pygame.Rect(rect.right - 12 - 90 - 10 - 90, rect.bottom - 34, 90, 26)

    def _import_ok_rect(self, rect):
        import pygame
        return pygame.Rect(rect.right - 12 - 90, rect.bottom - 34, 90, 26)

    def _import_folder_options(self):
        """[(folder_id_or_None, indented display name)] in the same
        flattened depth-first order the folder-tree panel shows, so cycling
        through it lands on the same folders in the same order a user
        would see there."""
        return [(fid, ("  " * depth) + name) for fid, depth, name in self._folder_tree_rows()]

    def _import_cycle_folder(self, direction: int) -> None:
        opts = self._import_folder_options()
        ids = [fid for fid, _name in opts]
        cur = self.import_dialog["folder"]
        idx = ids.index(cur) if cur in ids else 0
        self.import_dialog["folder"] = ids[(idx + direction) % len(ids)]

    # ---- content browser: import-options dialog (update / text entry) ----
    def _update_import_text(self, inp) -> None:
        import pygame
        d = self.import_dialog
        field = self.import_field
        typed = inp.take_text()
        if field == "name":
            d["name"] += typed
            if inp.pressed(pygame.K_BACKSPACE):
                d["name"] = d["name"][:-1]
        elif field == "scale":
            for ch in typed:
                if ch.isdigit() or ch in "-.":
                    d["scale_text"] += ch
            if inp.pressed(pygame.K_BACKSPACE):
                d["scale_text"] = d["scale_text"][:-1]
        if inp.pressed(pygame.K_RETURN) or inp.pressed(pygame.K_KP_ENTER) \
                or inp.pressed(pygame.K_TAB):
            self.import_field = None

    def _import_apply_fit_scale(self) -> None:
        """"Fit to ~1 unit" button: re-derive the scale that would bring the
        mesh's largest bbox dimension to ~1 unit, given the currently
        selected up-axis (so toggling axis then fitting uses the correct
        orientation's extents)."""
        d = self.import_dialog
        if d is None or d["kind"] != "mesh":
            return
        try:
            scale = self.engine_mod.fbx_fit_scale(d["path"], up_axis=d["up_axis"])
        except Exception as ex:
            self.status = (f"fit-to-unit failed: {ex}", 5.0)
            return
        d["scale_text"] = self._fmt_num(scale)

    def _import_confirm(self) -> None:
        d = self.import_dialog
        if d is None:
            return
        try:
            scale = float(d["scale_text"])
            if scale <= 0:
                raise ValueError
        except ValueError:
            scale = 1.0
        self.import_dialog = None
        self.import_field = None
        self._do_import(d["path"], name=d["name"], folder=d["folder"],
                        scale=scale, up_axis=d["up_axis"],
                        generate_lods=d.get("generate_lods", True))

    def _update_import_dialog(self, engine, w, h) -> None:
        import pygame
        inp = engine.input
        mp = inp.mouse_pos
        d = self.import_dialog
        rect = self._import_rect(w, h)

        if self.import_field is not None:
            self._update_import_text(inp)

        if not inp.mouse_button_pressed(1):
            return
        if self._import_close_rect(rect).collidepoint(mp):
            self.import_dialog = None
            self.import_field = None
            return
        if self._import_name_rect(rect).collidepoint(mp):
            self.import_field = "name"
            return
        prev_btn, _label, next_btn = self._import_folder_rects(rect)
        if prev_btn.collidepoint(mp):
            self.import_field = None
            self._import_cycle_folder(-1)
            return
        if next_btn.collidepoint(mp):
            self.import_field = None
            self._import_cycle_folder(1)
            return
        if d["kind"] == "mesh":
            if self._import_scale_rect(rect).collidepoint(mp):
                self.import_field = "scale"
                return
            if self._import_fit_btn_rect(rect).collidepoint(mp):
                self.import_field = None
                self._import_apply_fit_scale()
                return
            y_btn, z_btn = self._import_axis_rects(rect)
            if y_btn.collidepoint(mp):
                d["up_axis"] = "y"
                self.import_field = None
                return
            if z_btn.collidepoint(mp):
                d["up_axis"] = "z"
                self.import_field = None
                return
            if self._import_lod_checkbox_rect(rect).collidepoint(mp):
                self.import_field = None
                d["generate_lods"] = not d["generate_lods"]
                return
        if self._import_cancel_rect(rect).collidepoint(mp):
            self.import_dialog = None
            self.import_field = None
            return
        if self._import_ok_rect(rect).collidepoint(mp):
            self._import_confirm()
            return
        # click elsewhere inside the dialog defocuses any active text field
        self.import_field = None

    # ---- content browser: export selected tile's mesh to FBX ----
    def _export_asset_to_path(self, asset, path: str) -> None:
        """Dialog-free export handler so tests can drive it with a real path,
        no tkinter dialog involved -- same split as `_import_path_to_folder`.
        """
        try:
            self.engine_mod.export_asset_fbx(asset, path)
            self.status = (f"exported '{asset.name}' -> {os.path.basename(path)}", 5.0)
        except Exception as ex:
            self.status = (f"export failed: {ex}", 6.0)

    def _export_fbx_dialog(self) -> None:
        asset = self.selected_asset
        if asset is None or not self.engine_mod.has_mesh(asset):
            return
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            default_name = asset.name.replace(" ", "_") + ".fbx"
            path = filedialog.asksaveasfilename(
                title="Export FBX model", initialfile=default_name,
                defaultextension=".fbx", filetypes=[("FBX models", "*.fbx")])
            root.destroy()
        except Exception as ex:
            self.status = (f"file dialog unavailable: {ex}", 5.0)
            return
        if not path:
            return
        self._export_asset_to_path(asset, path)

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

    def _import_hdri_dialog(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Import HDRI",
                filetypes=[("Radiance HDR", "*.hdr"), ("All files", "*.*")])
            root.destroy()
        except Exception as ex:
            self.status = (f"file dialog unavailable: {ex}", 5.0)
            return
        if not path:
            return
        try:
            name = self.engine_mod.import_hdri(path, self.lib.directory)
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

        if inp.pressed(pygame.K_F11):
            self._toggle_fullscreen()

        if self.mat_ui is not None:  # node editor captures all editor input
            self.mat_ui.update(engine, dt)
            return
        if self.script_ui is not None:  # blueprint script editor captures all editor input
            self.script_ui.update(engine, dt)
            return
        if self.settings_open:      # settings dialog captures all editor input
            self._update_settings(engine, w, h)
            return
        if self.import_dialog is not None:  # import-options dialog captures all input
            self._update_import_dialog(engine, w, h)
            return
        if self.renaming_folder is not _NO_RENAME:  # inline folder rename captures input
            self._update_rename(inp)
            return
        if self.editing_field is not None:  # transform field text entry (see _update_edit_field)
            self._update_edit_field(inp)

        layout = self._layout(w, h)
        looking = self.fly is not None and self.fly.looking
        self._update_cursor(mp, layout)

        if inp.wheel and not looking:
            from engine import console_log
            target = self._hit_panel(mp, layout)
            if target is not None and self.panel_minimized.get(target, False):
                target = None
            if target == "outliner":
                self.outliner_scroll = max(0, self.outliner_scroll - int(inp.wheel) * 3)
            elif target == "browser":
                content = self._panel_content_rect("browser", layout)
                blay = self._browser_layout(content) if content is not None else None
                if blay is not None and blay["tree"].collidepoint(mp):
                    n_rows = len(self._folder_tree_rows())
                    visible = max(0, blay["tree"].height - 4) // TREE_ROW_H
                    self.tree_scroll = max(0, min(
                        self.tree_scroll - int(inp.wheel) * 2,
                        max(0, (n_rows - visible) * TREE_ROW_H)))
                else:
                    self.browser_scroll = max(0, self.browser_scroll - int(inp.wheel) * 70)
            elif target == "console":
                # scroll UP (positive wheel) moves AWAY from the latest entry
                # (console_scroll counts up from the pinned-to-bottom anchor,
                # the opposite sense from outliner/browser's top-anchored
                # scroll) -- clamped for real at draw time (_draw_console),
                # this clamp is just a sane upper bound so it can't grow
                # unbounded while the panel isn't even visible/rendering
                self.console_scroll = max(0, min(self.console_scroll + int(inp.wheel) * 3,
                                                 console_log.get_log().entries.maxlen))

        if inp.mouse_button_pressed(1):
            self._commit_edit_field()
            hit_splitter = None
            for side, r in layout["splitters"].items():
                if r.collidepoint(mp):
                    hit_splitter = side
                    break
            if hit_splitter is not None:
                self.splitter_drag = hit_splitter
            elif not self._handle_menu_click(mp, w):
                target = self._hit_panel(mp, layout)
                if target is not None:
                    if target in self.floating:
                        self.floating.remove(target)
                        self.floating.append(target)
                    rect = layout["panels"][target]
                    title_rect = pygame.Rect(rect.x, rect.y, rect.width, PANEL_TITLE_H)
                    if title_rect.collidepoint(mp):
                        btns = self._panel_title_buttons(rect)
                        if btns["close"].collidepoint(mp):
                            # close/minimize apply to the ACTIVE tab -- `target`
                            # is always that active pid, tabbed or not
                            self.panel_visible[target] = False
                            self._save_settings()
                        elif btns["minimize"].collidepoint(mp):
                            self._toggle_minimize(target)
                        else:
                            drag_pid = target
                            tabs = self._panel_tab_strip(target, rect)
                            if tabs is not None:
                                for tpid, trect in tabs.items():
                                    if trect.collidepoint(mp):
                                        drag_pid = tpid
                                        break
                            # begin_panel_drag on whichever tab was actually
                            # clicked (may differ from `target`, the slot's
                            # current active) -- release re-runs the same
                            # drop math regardless, so a plain click that
                            # lands back on this same group just switches
                            # active (see _apply_panel_drop's fast path)
                            # while an actual drag can pull it elsewhere or
                            # out to floating
                            self._begin_panel_drag(drag_pid, mp, rect)
                    elif not self.panel_minimized.get(target, False):
                        if target in self.floating \
                                and self._panel_resize_handle(rect).collidepoint(mp):
                            self._begin_panel_resize(target, mp, rect)
                        else:
                            content = self._panel_content_rect(target, layout)
                            self._route_panel_click(target, mp, content, inp)
                elif self._side_toolbar_rect(w, h).collidepoint(mp):
                    # checked AFTER _hit_panel (so a floating panel that
                    # happens to overlap the strip still wins, same
                    # precedence the viewport toolbar gets below) but BEFORE
                    # the viewport branch -- a click here must never start a
                    # marquee or place/deselect in the 3D view
                    self._click_side_toolbar(mp, self._side_toolbar_rect(w, h))
                elif layout["viewport"].collidepoint(mp):
                    toolbar_rect = self._viewport_toolbar_rect(layout["viewport"])
                    if not (toolbar_rect.collidepoint(mp)
                            and self._click_viewport_toolbar(mp, toolbar_rect)):
                        self.active_slider = None
                        if not self._try_grab_gizmo(mp, w, h, inp):
                            self._begin_viewport_press(mp, w, h, inp)

        if self.panel_drag is not None and not inp.mouse_held(1):
            self._finish_panel_drag(mp, w, h)

        if self.splitter_drag is not None:
            if inp.mouse_held(1):
                self._update_splitter_drag(self.splitter_drag, mp, w, h)
            else:
                self.splitter_drag = None
                self._save_settings()

        if self.panel_resize is not None:
            if inp.mouse_held(1):
                self._update_panel_resize(mp)
            else:
                self.panel_resize = None
                self._save_settings()

        # gizmo drag
        if self.gizmo_drag is not None:
            if inp.mouse_held(1) and self.selected is not None:
                self._update_gizmo_drag(mp, inp)
            else:
                self.gizmo_drag = None
                self.snap_feedback = None

        # marquee (box) select drag -- started by _begin_viewport_press on an
        # empty-space press; the rect is redrawn live from marquee["cur"]
        # (_draw_marquee), the selection change itself is deferred to
        # release (_finish_marquee) so a tiny in-place press-release still
        # collapses to the old plain-click behavior
        if self.marquee is not None:
            if inp.mouse_held(1):
                self.marquee["cur"] = mp
            else:
                self._finish_marquee(mp, w, h)
                self.marquee = None

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
            if not self._try_drop_material_slot(mp, layout):
                if not self.over_ui(mp):
                    self._place_asset(self.drag_asset, mp, w, h)
            self.drag_asset = None

        if inp.pressed(pygame.K_F2) and self.selected_folder is not None:
            self._begin_rename(self.selected_folder)

        # a details-panel transform field being typed into owns the keyboard:
        # letters like w/e/r/f/c/k must not also fire viewport hotkeys below
        editing_text = self.editing_field is not None
        ctrl = inp.held(pygame.K_LCTRL) or inp.held(pygame.K_RCTRL)
        shift_held = inp.held(pygame.K_LSHIFT) or inp.held(pygame.K_RSHIFT)
        if not editing_text:
            if inp.pressed(pygame.K_DELETE):
                self._delete_selected()
            if inp.pressed(pygame.K_END):
                self._snap_to_floor()
            if ctrl and inp.pressed(pygame.K_d):
                self._duplicate_selected()
            if ctrl and inp.pressed(pygame.K_s):
                self._save_scene()
            if inp.pressed(pygame.K_f) and not looking:
                self._focus_selection()
            if inp.pressed(pygame.K_c):
                # Shift+C (Blender: reset 3D cursor) takes priority over the
                # plain-C fly-collision toggle so the two never both fire
                if shift_held:
                    self._reset_cursor3d()
                elif self.fly is not None:
                    self.fly.collide = not self.fly.collide
            if inp.pressed(pygame.K_k) and not looking \
                    and layout["viewport"].collidepoint(mp) \
                    and not self._viewport_toolbar_rect(layout["viewport"]).collidepoint(mp):
                self._place_cursor3d(mp, w, h)
            if not looking:
                if inp.pressed(pygame.K_w):
                    self._set_gizmo_mode("translate")
                elif inp.pressed(pygame.K_e):
                    self._set_gizmo_mode("rotate")
                elif inp.pressed(pygame.K_r):
                    self._set_gizmo_mode("scale")
            if inp.pressed(pygame.K_m) and self.selected is not None \
                    and (self.selected.mesh is not None or self.selected.environment is not None):
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
        """Engine Esc hook: folder rename, then dropdown/settings, then
        material editor, then deselect, and only then let the engine quit."""
        if self.editing_field is not None:
            self._cancel_edit_field()
            return True
        if self.renaming_folder is not _NO_RENAME:
            self._cancel_rename()
            return True
        if self.open_menu is not None:
            self.open_menu = None
            return True
        if self.settings_open:
            self.settings_open = False
            return True
        if self.import_dialog is not None:
            self.import_dialog = None
            self.import_field = None
            return True
        if self.mat_ui is not None:
            if self.mat_ui.ctx_menu is not None:
                self.mat_ui.ctx_menu = None
                return True
            self.mat_ui.close()
            return True
        if self.script_ui is not None:
            self.script_ui.close()
            return True
        if self.selected is not None:
            self.selected = None
            return True
        return False

    def _route_panel_click(self, pid, mp, content, inp=None) -> None:
        if pid == "outliner":
            self.active_slider = None
            self._click_outliner(mp, content, inp)
        elif pid == "details":
            self._click_details(mp, content)
        elif pid == "browser":
            blay = self._browser_layout(content)
            if self._new_folder_btn_rect(blay["topbar"]).collidepoint(mp):
                self._new_folder()
            elif self._new_blueprint_btn_rect(blay["topbar"]).collidepoint(mp):
                self._new_blueprint()
            elif self._import_btn_rect(blay["topbar"]).collidepoint(mp):
                self._import_dialog_to_folder()
            elif self._export_btn_rect(blay["topbar"]).collidepoint(mp):
                if (self.selected_asset is not None
                        and not isinstance(self.selected_asset, self.engine_mod.MaterialAsset)
                        and not isinstance(self.selected_asset, self.engine_mod.BlueprintAsset)
                        and self.engine_mod.has_mesh(self.selected_asset)):
                    self._export_fbx_dialog()
            elif blay["tree"].collidepoint(mp):
                _i, fid = self._tree_row_at(mp, blay["tree"])
                if _i is not None:
                    if self.renaming_folder is not _NO_RENAME and self.renaming_folder != fid:
                        self._commit_rename()
                    self.selected_folder = fid
            elif blay["grid"].collidepoint(mp):
                if self.renaming_folder is not _NO_RENAME:
                    self._commit_rename()
                asset = self._tile_at(mp, blay["grid"])
                if asset is not None:
                    self.selected_asset = asset
                    is_mat = isinstance(asset, self.engine_mod.MaterialAsset)
                    is_bp = isinstance(asset, self.engine_mod.BlueprintAsset)
                    if is_bp:
                        # no double-click in this codebase -- a single click
                        # both selects the tile AND opens the script editor
                        # (mirrors the M-key-opens-material-editor precedent)
                        if self.script_ui is None or self.script_ui.blueprint is not asset:
                            if self.script_ui is not None:
                                self.script_ui._save()  # persist any pending
                                                         # edits before switching
                            self.script_ui = ScriptEditorUI(self, asset)
                    elif is_mat or "texture" not in asset.data:  # textures aren't placeable
                        self.drag_asset = asset

    def _try_drop_material_slot(self, mp, layout) -> bool:
        """Handle a drag_asset release over the Details panel's material
        slot. Returns True if it was consumed here (whether or not it
        resulted in an assignment -- e.g. a non-material asset dropped on
        the slot is rejected but the drop still counts as "handled", so it
        doesn't fall through and get placed in the world instead)."""
        import pygame
        content = self._panel_content_rect("details", layout)
        if content is None or not content.collidepoint(mp):
            return False
        rows = self._details_rows()
        i = (mp[1] - (content.y + DETAILS_ROWS_TOP)) // DETAIL_ROW_H
        if not (0 <= i < len(rows)) or rows[i]["kind"] != "material_slot":
            return False
        if not isinstance(self.drag_asset, self.engine_mod.MaterialAsset):
            self.status = (f"'{self.drag_asset.name}' is not a material asset", 4.0)
            return True
        ent = rows[i]["entity"]
        mat_asset = self.drag_asset
        ent.material = mat_asset.graph()
        ent.material_asset = mat_asset.name
        ent.material.apply(ent)
        self.mat_icon_cache[id(ent)] = self.mat_icons.get(
            mat_asset.name) or make_material_icon(self.engine_mod, ent.material)
        if self.mat_ui is not None and self.mat_ui.entity is ent:
            self.mat_ui.graph = ent.material
        self.dirty = True
        self.status = (f"assigned material '{mat_asset.name}'", 4.0)
        return True

    def _click_details(self, mp, rect) -> None:
        e = self.selected
        if e is not None and self._click_transform_fields(mp, rect, e):
            return
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

    def _click_outliner(self, mp, rect, inp=None) -> None:
        import pygame
        rows = self._outliner_rows()
        i = (mp[1] - rect.y - 6) // ROW_H + self.outliner_scroll
        if 0 <= mp[1] - rect.y - 6 and 0 <= i < len(rows):
            if inp is not None and (inp.held(pygame.K_LSHIFT) or inp.held(pygame.K_RSHIFT)):
                self._toggle_selection(rows[i])
            else:
                self.selected = rows[i]

    def _tile_at(self, mp, grid_rect):
        x0 = grid_rect.x + 10 - self.browser_scroll
        for asset in self._tiles_in(self.selected_folder):
            if x0 <= mp[0] < x0 + TILE_W and grid_rect.y + 6 <= mp[1] < grid_rect.y + 6 + TILE_H - 8:
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

    def _place_cursor3d(self, mp, w, h) -> None:
        """K key: place the 3D cursor at the surface under the mouse --
        reuses _mouse_hit's exact raycast (pick_entity against scene
        geometry; y=0 ground-plane fallback if nothing is hit; a fixed
        distance along the ray as the last resort) so cursor placement
        always agrees with what a translate drag's Shift-snap / End-key
        floor-snap would read as "the surface here". RMB is already the
        fly-look toggle in this editor, so a dedicated key avoids any
        conflict (see the Controls overlay / _CONTROLS_LINES)."""
        _entity, pt = self._mouse_hit(mp, w, h)
        self.cursor3d = self.engine_mod.Vec3(float(pt[0]), float(pt[1]), float(pt[2]))
        self._save_settings()

    def _reset_cursor3d(self) -> None:
        """Shift+C / Edit > Reset 3D Cursor: Blender's cursor-to-origin reset."""
        self.cursor3d = self.engine_mod.Vec3(0.0, 0.0, 0.0)
        self._save_settings()

    def _pick_marker(self, mp, w, h):
        """Screen-space proximity pick for mesh-less entities (Sun, Fog
        Volume) that a raycast can't hit: whichever marker's projected
        center is within a small pixel radius, nearest wins."""
        best, best_d = None, 14.0
        for e in self.scene.entities:
            if e.sun is None and e.fog_volume is None:
                continue
            pt = self.camera.project(e.transform.position, w, h)
            if pt is None:
                continue
            d = math.hypot(pt[0] - mp[0], pt[1] - mp[1])
            if d < best_d:
                best, best_d = e, d
        return best

    def _viewport_hit(self, mp, w, h):
        """Whatever the mouse would pick in the viewport right now -- a
        marker entity (Sun/Fog Volume, proximity-picked since a raycast
        can't hit them) takes priority, else a mesh raycast; None on empty
        space. Shared by _click_viewport and _begin_viewport_press so both
        agree on what counts as 'empty' (i.e. eligible to start a marquee)."""
        marker = self._pick_marker(mp, w, h)
        return marker if marker is not None else self._mouse_hit(mp, w, h)[0]

    def _click_viewport(self, mp, w, h, inp=None) -> None:
        """Plain click replaces the selection with the hit entity (or clears
        it, on empty space). Shift+click toggles the hit entity in/out of
        the selection and makes it active; Shift+click on empty space is a
        no-op (Blender doesn't deselect-all on that either) -- it neither
        clears nor extends the existing selection."""
        import pygame
        hit = self._viewport_hit(mp, w, h)
        shift = inp is not None and (inp.held(pygame.K_LSHIFT) or inp.held(pygame.K_RSHIFT))
        if hit is None:
            if not shift:
                self.selected = None
            return
        if shift:
            self._toggle_selection(hit)
        else:
            self.selected = hit

    def _begin_viewport_press(self, mp, w, h, inp=None) -> None:
        """LMB press in the viewport that didn't grab a gizmo handle (see
        the update() call site). A press landing on an entity keeps the
        exact pre-marquee behavior -- click-select fires immediately, same
        as _click_viewport always has. A press on empty space is NOT
        resolved yet: it becomes a pending marquee: _finish_marquee decides
        on release whether the drag distance was big enough to be a box-
        select, or should collapse back to the old click-empty behavior
        (clear the selection, or no-op under Shift) -- see
        MARQUEE_THRESHOLD."""
        import pygame
        if self._viewport_hit(mp, w, h) is not None:
            self._click_viewport(mp, w, h, inp)
            return
        shift = inp is not None and (inp.held(pygame.K_LSHIFT) or inp.held(pygame.K_RSHIFT))
        self.marquee = {"start": mp, "cur": mp, "shift": shift}

    def _entities_in_rect(self, rect, w, h):
        """Entities for marquee/box select: v1 scope is mesh-bearing,
        visible entities (see _world_aabb) -- projects each one's 8 world-
        AABB corners and keeps whichever ones actually project (some may
        fall behind the camera); an entity is included if the screen-space
        bounding box of its projectable corners intersects `rect`. An
        entity with NO projectable corner (fully behind the camera) is
        excluded entirely."""
        import pygame
        hits = []
        for e in self.scene.entities:
            if e.mesh is None or not e.visible:
                continue
            aabb = self._world_aabb(e)
            if aabb is None:
                continue
            lo, hi = aabb
            corners = [(x, y, z) for x in (lo[0], hi[0])
                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            pts = [self.camera.project(self.engine_mod.Vec3(*c), w, h) for c in corners]
            xs = [p[0] for p in pts if p is not None]
            ys = [p[1] for p in pts if p is not None]
            if not xs:
                continue
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            bbox = pygame.Rect(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0)))
            if bbox.colliderect(rect):
                hits.append(e)
        return hits

    def _finish_marquee(self, mp, w, h) -> None:
        """LMB release ending a press that started on empty viewport space
        (see _begin_viewport_press). A drag distance under
        MARQUEE_THRESHOLD is treated as the original plain click-on-empty
        behavior: clear the selection, or (Shift) do nothing -- exactly
        _click_viewport's empty-space case. Past the threshold, every
        entity from _entities_in_rect is selected: a plain marquee
        REPLACES the selection; Shift+marquee ADDS the rect's entities to
        the current one, keeping any already-selected entities that fall
        outside the rect. The active element becomes the last NEWLY added
        entity; if nothing new qualified (e.g. Shift+marquee over empty
        space, or every hit was already selected) the previous active
        element is left unchanged."""
        import pygame
        sx, sy = self.marquee["start"]
        shift = self.marquee["shift"]
        if math.hypot(mp[0] - sx, mp[1] - sy) < MARQUEE_THRESHOLD:
            if not shift:
                self.selected = None
            return
        rect = pygame.Rect(min(sx, mp[0]), min(sy, mp[1]),
                           abs(mp[0] - sx), abs(mp[1] - sy))
        hits = self._entities_in_rect(rect, w, h)
        if shift:
            sel = list(self.selection)
            active = self.selected
            for e in hits:
                if e not in sel:
                    sel.append(e)
                    active = e
            self._set_selection(sel, active=active)
        else:
            self._set_selection(hits, active=(hits[-1] if hits else None))

    def _place_asset(self, asset, mp, w, h) -> None:
        if isinstance(asset, self.engine_mod.MaterialAsset):
            # materials aren't placeable in the world -- drag them onto a
            # mesh entity's Details material slot instead.
            return
        _, point = self._mouse_hit(mp, w, h)
        entity = asset.instantiate()
        entity.transform.position = self.engine_mod.Vec3(
            float(point[0]), float(point[1]) + base_height(entity), float(point[2]))
        self.scene.add(entity)
        self.selected = entity
        self.dirty = True

    def _focus(self, entities) -> None:
        """Pull the camera back along its current forward direction to
        frame `entities` (F key): a single-entity list reproduces the old
        one-entity framing exactly (bound from that entity's local mesh
        extent, centered on its position). Multiple entities frame their
        combined bound around the mean of their positions, with `spread`
        (how far apart they are) widening the distance so a wide
        multi-selection doesn't clip.
        """
        import numpy as np
        Vec3 = self.engine_mod.Vec3
        positions = np.array([[e.transform.position.x, e.transform.position.y,
                               e.transform.position.z] for e in entities])
        center = positions.mean(axis=0)
        spread = float(np.max(np.linalg.norm(positions - center, axis=1))) \
            if len(entities) > 1 else 0.0
        bound = 0.5
        for e in entities:
            if e.mesh is not None:
                s = e.transform.scale
                max_scale = max(abs(s.x), abs(s.y), abs(s.z))
                b = float(np.max(np.linalg.norm(e.mesh.vertices, axis=1))) * max_scale
                bound = max(bound, b)
        dist = max(3.0, (bound + spread) * 3.0)
        fwd = self.camera.forward()
        self.camera.position = Vec3(center[0] - fwd.x * dist, center[1] - fwd.y * dist,
                                    center[2] - fwd.z * dist)

    # ---- drawing (engine overlay callback) ----
    def draw(self, eng) -> None:
        import pygame
        surf = eng.screen
        w, h = surf.get_size()
        layout = self._layout(w, h)
        mp = eng.input.mouse_pos
        self._draw_markers(surf, w, h)
        self._draw_marquee(surf, layout)

        # backdrop so gaps between a side dock and the bottom dock (if both
        # are present) read as UI, not a hole showing the 3D scene through --
        # docked panels start AFTER the side toolbar (tb_w), not at x=0
        tb_w = layout["side_toolbar_w"]
        if layout["left_w"]:
            pygame.draw.rect(surf, PANEL_BG,
                             pygame.Rect(tb_w, MENU_H, layout["left_w"], h - MENU_H))
        if layout["right_w"]:
            pygame.draw.rect(surf, PANEL_BG, pygame.Rect(
                w - layout["right_w"], MENU_H, layout["right_w"], h - MENU_H))
        if layout["bottom_h"]:
            pygame.draw.rect(surf, PANEL_BG, pygame.Rect(
                tb_w + layout["left_w"], h - layout["bottom_h"],
                w - tb_w - layout["left_w"] - layout["right_w"], layout["bottom_h"]))

        self._draw_side_toolbar(surf, self._side_toolbar_rect(w, h), mp)
        if layout["viewport"].width > 0 and layout["viewport"].height > 0:
            self._draw_viewport_toolbar(surf, self._viewport_toolbar_rect(layout["viewport"]), mp)
        for side, r in layout["splitters"].items():
            hov = self.splitter_drag == side or r.collidepoint(mp)
            if hov:
                pygame.draw.rect(surf, ACCENT, r)

        panels = dict(layout["panels"])
        drag_pid = self.panel_drag["id"] if self.panel_drag else None
        if drag_pid is not None:
            # unconditional: a dragged tab may be an INACTIVE member of its
            # group (so it has no entry of its own in layout["panels"] --
            # only the active tab does) yet must still get a ghost rect that
            # follows the mouse, same as the plain single-panel drag case
            mp = eng.input.mouse_pos
            g = self.panel_drag
            dh = PANEL_TITLE_H if self.panel_minimized.get(drag_pid, False) else g["h"]
            panels[drag_pid] = pygame.Rect(mp[0] - g["dx"], mp[1] - g["dy"],
                                           g["w"], dh)

        for pid in ("outliner", "details", "browser", "console"):
            if pid in panels and pid not in self.floating and pid != drag_pid:
                self._draw_panel(surf, pid, panels[pid])
        for pid in self.floating:
            if pid in panels and pid != drag_pid:
                self._draw_panel(surf, pid, panels[pid])
        if drag_pid is not None:
            # docking preview drawn FIRST (underneath), the flying ghost
            # panel on top of it -- the preview rect can be as large as an
            # entire merged tab group's slot and may fully contain the
            # ghost's own on-screen position, so the ghost has to win that
            # layering fight to stay legible (title text, buttons, ...).
            # The preview is the ACTUAL rect (and, for a tab-join, the
            # tab-strip band) `pid` will occupy after the drop -- run
            # through the exact same math `_finish_panel_drag` uses, so
            # this can never show a spot the drop wouldn't honor
            target = self._panel_drag_target(drag_pid, mp, w, h, layout)
            if target["kind"] != "float":
                preview = self._simulate_drop(drag_pid, target, w, h)
                if preview is not None:
                    hl = pygame.Surface((preview.width, preview.height), pygame.SRCALPHA)
                    hl.fill((*ACCENT, 55))
                    surf.blit(hl, preview.topleft)
                    pygame.draw.rect(surf, ACCENT, preview, 2)
                    if target["kind"] == "tab":  # emphasize: joins as a tab,
                                                  # not a side-by-side split
                        strip = pygame.Rect(preview.x, preview.y,
                                            preview.width, PANEL_TITLE_H)
                        pygame.draw.rect(surf, ACCENT, strip, 3)
            # force_plain: the flying ghost previews just the ONE panel
            # being pulled, never its (still-intact, until release) tab
            # strip -- that would look like it dragged its whole group
            self._draw_panel(surf, drag_pid, panels[drag_pid], force_plain=True)

        if self.drag_asset is not None:
            is_mat = isinstance(self.drag_asset, self.engine_mod.MaterialAsset)
            icon = (self.mat_icons if is_mat else self.icons).get(self.drag_asset.name)
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
        if self.import_dialog is not None:
            self._draw_import_dialog(surf, w, h)
        if self.mat_ui is not None:
            self.mat_ui.draw(surf)
        if self.script_ui is not None:
            self.script_ui.draw(surf)

    def _panel_title(self, pid) -> str:
        if pid == "outliner":
            if self.save_flash > 0:
                return "World Outliner — saved ✓"
            name = os.path.basename(self.scene_path) + (" *" if self.dirty else "")
            return f"World Outliner — {name}"
        if pid == "details":
            return "Details"
        if pid == "console":
            return "Console"
        return "Content Browser"

    def _draw_title_buttons(self, surf, rect) -> None:
        import pygame
        mp = pygame.mouse.get_pos()
        for key, r in self._panel_title_buttons(rect).items():
            hov = r.collidepoint(mp)
            pygame.draw.rect(surf, HOVER_BG if hov else (44, 47, 56), r, border_radius=2)
            glyph = "-" if key == "minimize" else "x"
            lab = self.font_small.render(glyph, True, TEXT)
            surf.blit(lab, (r.x + (r.width - lab.get_width()) // 2,
                            r.y + (r.height - lab.get_height()) // 2))

    def _draw_tab_strip(self, surf, active_pid, tabs) -> None:
        """Tab headers for a REAL (>=2 visible member) dock group -- active
        tab brighter with an accent underline, inactive tabs dimmer, styled
        consistently with the plain panel title bar it replaces (UE-ish)."""
        import pygame
        mp = pygame.mouse.get_pos()
        for pid, r in tabs.items():
            active = pid == active_pid
            if active:
                bg = (40, 44, 54)
            elif r.collidepoint(mp):
                bg = (34, 37, 45)
            else:
                bg = (27, 29, 35)
            pygame.draw.rect(surf, bg, r)
            color = TEXT if active else TEXT_DIM
            label = TAB_LABELS.get(pid, pid)
            lab = self.font_small.render(label, True, color)
            surf.blit(lab, (r.x + (r.width - lab.get_width()) // 2,
                            r.y + (r.height - lab.get_height()) // 2))
            if active:
                pygame.draw.line(surf, ACCENT, (r.x, r.bottom - 1),
                                 (r.right, r.bottom - 1), 2)

    def _draw_panel(self, surf, pid, rect, force_plain=False) -> None:
        """`force_plain` skips the tab strip even if `pid`'s group has other
        visible members -- used only for the flying drag-ghost, which
        previews just the one panel being pulled, not its (still-intact
        until release) tab group."""
        import pygame
        pygame.draw.rect(surf, PANEL_BG, rect)
        pygame.draw.rect(surf, PANEL_EDGE, rect, 1)
        title_rect = pygame.Rect(rect.x, rect.y, rect.width, PANEL_TITLE_H)
        pygame.draw.rect(surf, (30, 33, 40), title_rect)
        pygame.draw.line(surf, PANEL_EDGE, (rect.x, rect.y + PANEL_TITLE_H),
                         (rect.right, rect.y + PANEL_TITLE_H))
        tabs = None if force_plain else self._panel_tab_strip(pid, rect)
        if tabs is not None:
            self._draw_tab_strip(surf, pid, tabs)
        else:
            lab = self.font_small.render(self._panel_title(pid)[:40], True, TEXT)
            surf.blit(lab, (rect.x + 8, rect.y + 3))
        self._draw_title_buttons(surf, rect)
        if self.panel_minimized.get(pid, False):
            return
        content = pygame.Rect(rect.x, rect.y + PANEL_TITLE_H, rect.width,
                              max(0, rect.height - PANEL_TITLE_H))
        if pid == "outliner":
            self._draw_outliner(surf, content)
        elif pid == "details":
            self._draw_details(surf, content)
        elif pid == "browser":
            self._draw_browser(surf, content)
        elif pid == "console":
            self._draw_console(surf, content)
        if pid in self.floating:
            grip = self._panel_resize_handle(rect)
            corner = (ACCENT if self.panel_resize is not None
                                and self.panel_resize["id"] == pid else PANEL_EDGE)
            for i in range(3):
                off = 3 + i * 3
                pygame.draw.line(surf, corner, (grip.right - off, grip.bottom),
                                 (grip.right, grip.bottom - off))

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
        hints = self._MENU_HOTKEYS.get(self.open_menu, {})
        drop, rows = self._dropdown_geom(self.open_menu, w)
        pygame.draw.rect(surf, (24, 26, 32), drop)
        pygame.draw.rect(surf, PANEL_EDGE, drop, 1)
        for label, r, _action, enabled in rows:
            if r.collidepoint(mp):
                pygame.draw.rect(surf, HOVER_BG, r)
            color = TEXT if enabled else TEXT_DIM
            checked = self._menu_checked(label)
            if checked is not None:
                box = pygame.Rect(r.x + 4, r.y + 5, 12, 12)
                pygame.draw.rect(surf, (48, 51, 60), box, border_radius=2)
                if checked:
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
        "Shift+LMB - extend/toggle selection (viewport or outliner); last click = active",
        "W / E / R - gizmo mode: translate / rotate / scale  (only while not looking)",
        ", / . - rotate selection 15 deg        - / = - scale selection",
        "F - focus camera on selection  (only while not looking)",
        "Ctrl+D - duplicate selection           Del - delete selection",
        "End - snap selection to floor          Shift+drag - snap to nearby mesh face",
        "Alt+drag gizmo (translate/rotate) - duplicate and move/rotate the copy",
        "Ctrl+S - save scene",
        "L - toggle flashlight                  C - toggle player collision",
        "K - place 3D Cursor at surface under mouse (not RMB, which is fly-look)",
        "Shift+C - reset 3D Cursor to origin    Pivot toolbar - \"3D Cursor\" mode "
        "orbits/scales about it",
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

        if self._software_active():
            self._draw_settings_slider(surf, rect, "pixel", "pixel scale",
                                       self.eng.renderer.render_scale, 1, 6)
        self._draw_settings_slider(surf, rect, "max_fps", "max fps",
                                   self.eng.max_fps, 30, 240)

        surf.blit(self.font_small.render("Graphics API (restart)", True, TEXT_DIM),
                  (rect.x + 12, rect.y + 160))
        for key, btn in self._settings_api_buttons(rect):
            active = self.api_pref == key
            pygame.draw.rect(surf, SELECT_BG if active else (33, 36, 44), btn,
                             border_radius=4)
            pygame.draw.rect(surf, PANEL_EDGE, btn, 1, border_radius=4)
            lab = self.font_small.render(key.upper(), True, TEXT)
            surf.blit(lab, (btn.x + (btn.width - lab.get_width()) // 2, btn.y + 4))

        active_lab = self.font_small.render(f"Active: {self._active_api()}", True, TEXT_DIM)
        surf.blit(active_lab, (rect.x + 12, rect.y + 210))

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

    def _draw_import_dialog(self, surf, w, h) -> None:
        import pygame
        d = self.import_dialog
        rect = self._import_rect(w, h)
        mp = pygame.mouse.get_pos()
        pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(surf, PANEL_EDGE, rect, 1, border_radius=6)
        surf.blit(self.font.render("Import Options", True, TEXT), (rect.x + 12, rect.y + 8))

        close = self._import_close_rect(rect)
        pygame.draw.rect(surf, (60, 34, 34) if not close.collidepoint(mp) else (90, 44, 44),
                         close, border_radius=4)
        surf.blit(self.font_small.render("X", True, (230, 160, 160)),
                  (close.x + 7, close.y + 4))

        is_mesh = d["kind"] == "mesh"
        dim_or_off = TEXT_DIM if is_mesh else (75, 77, 84)
        box_bg = (33, 36, 44) if is_mesh else (26, 28, 33)
        box_edge = PANEL_EDGE if is_mesh else (44, 47, 54)
        val_col = TEXT if is_mesh else (90, 92, 98)

        surf.blit(self.font_small.render(f"Type: {d['label']}", True, TEXT_DIM),
                  (rect.x + 12, rect.y + 34))

        # ---- name ----
        surf.blit(self.font_small.render("Name", True, TEXT_DIM), (rect.x + 12, rect.y + 60))
        name_r = self._import_name_rect(rect)
        editing_name = self.import_field == "name"
        pygame.draw.rect(surf, (33, 36, 44), name_r, border_radius=4)
        pygame.draw.rect(surf, ACCENT if editing_name else PANEL_EDGE, name_r, 1, border_radius=4)
        surf.blit(self.font_small.render(d["name"] + ("_" if editing_name else ""), True, TEXT),
                  (name_r.x + 6, name_r.y + 5))

        # ---- target folder (cycle) ----
        surf.blit(self.font_small.render("Target Folder", True, TEXT_DIM),
                  (rect.x + 12, rect.y + 108))
        prev_btn, label_r, next_btn = self._import_folder_rects(rect)
        for btn, glyph in ((prev_btn, "<"), (next_btn, ">")):
            pygame.draw.rect(surf, HOVER_BG if btn.collidepoint(mp) else (33, 36, 44),
                             btn, border_radius=4)
            pygame.draw.rect(surf, PANEL_EDGE, btn, 1, border_radius=4)
            gl = self.font_small.render(glyph, True, TEXT)
            surf.blit(gl, (btn.x + (btn.width - gl.get_width()) // 2, btn.y + 4))
        pygame.draw.rect(surf, (33, 36, 44), label_r, border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, label_r, 1, border_radius=4)
        fname = dict(self._import_folder_options()).get(d["folder"], "Assets").strip() or "Assets"
        fl = self.font_small.render(fname, True, TEXT)
        surf.blit(fl, (label_r.x + max(4, (label_r.width - fl.get_width()) // 2), label_r.y + 5))

        # ---- uniform scale + fit button (mesh only) ----
        surf.blit(self.font_small.render(
            "Uniform Scale" + ("" if is_mesh else " (mesh only)"), True, dim_or_off),
            (rect.x + 12, rect.y + 156))
        scale_r = self._import_scale_rect(rect)
        editing_scale = self.import_field == "scale"
        pygame.draw.rect(surf, box_bg, scale_r, border_radius=4)
        pygame.draw.rect(surf, (ACCENT if editing_scale else box_edge) if is_mesh else box_edge,
                         scale_r, 1, border_radius=4)
        surf.blit(self.font_small.render(d["scale_text"] + ("_" if editing_scale else ""),
                                         True, val_col), (scale_r.x + 6, scale_r.y + 5))
        fit_r = self._import_fit_btn_rect(rect)
        fit_bg = (HOVER_BG if fit_r.collidepoint(mp) else (33, 36, 44)) if is_mesh else box_bg
        pygame.draw.rect(surf, fit_bg, fit_r, border_radius=4)
        pygame.draw.rect(surf, box_edge, fit_r, 1, border_radius=4)
        fit_lab = self.font_small.render("Fit to ~1 unit", True, ACCENT if is_mesh else val_col)
        surf.blit(fit_lab, (fit_r.x + (fit_r.width - fit_lab.get_width()) // 2, fit_r.y + 5))

        # ---- up axis toggle (mesh only) ----
        surf.blit(self.font_small.render(
            "Up Axis" + ("" if is_mesh else " (mesh only)"), True, dim_or_off),
            (rect.x + 12, rect.y + 204))
        y_btn, z_btn = self._import_axis_rects(rect)
        for btn, axis in ((y_btn, "y"), (z_btn, "z")):
            active = is_mesh and d["up_axis"] == axis
            pygame.draw.rect(surf, SELECT_BG if active else box_bg, btn, border_radius=4)
            pygame.draw.rect(surf, box_edge, btn, 1, border_radius=4)
            lab = self.font_small.render(axis.upper(), True, val_col)
            surf.blit(lab, (btn.x + (btn.width - lab.get_width()) // 2, btn.y + 5))

        # ---- Generate LODs checkbox (mesh only) ----
        lod_r = self._import_lod_checkbox_rect(rect)
        pygame.draw.rect(surf, box_bg, lod_r, border_radius=3)
        pygame.draw.rect(surf, box_edge, lod_r, 1, border_radius=3)
        if is_mesh and d["generate_lods"]:
            inner = lod_r.inflate(-6, -6)
            pygame.draw.rect(surf, ACCENT, inner, border_radius=2)
        lod_lab = self.font_small.render(
            "Generate LODs" + ("" if is_mesh else " (mesh only)"), True, dim_or_off)
        surf.blit(lod_lab, (lod_r.right + 8, lod_r.y - 2))

        # ---- Cancel / Import ----
        cancel_r = self._import_cancel_rect(rect)
        ok_r = self._import_ok_rect(rect)
        pygame.draw.rect(surf, HOVER_BG if cancel_r.collidepoint(mp) else (33, 36, 44),
                         cancel_r, border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, cancel_r, 1, border_radius=4)
        clab = self.font_small.render("Cancel", True, TEXT)
        surf.blit(clab, (cancel_r.x + (cancel_r.width - clab.get_width()) // 2, cancel_r.y + 6))
        pygame.draw.rect(surf, (60, 110, 60) if ok_r.collidepoint(mp) else (44, 84, 44),
                         ok_r, border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, ok_r, 1, border_radius=4)
        ilab = self.font_small.render("Import", True, (210, 245, 210))
        surf.blit(ilab, (ok_r.x + (ok_r.width - ilab.get_width()) // 2, ok_r.y + 6))

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
        # sun glyph (a small sunburst at the entity position)
        for e in self.scene.entities:
            if e.sun is None:
                continue
            pt = self.camera.project(e.transform.position, w, h)
            if pt is None:
                continue
            x, y = int(pt[0]), int(pt[1])
            color = (255, 224, 150) if e.sun.enabled else (110, 108, 96)
            pygame.draw.circle(surf, color, (x, y), 5, 1)
            for k in range(8):
                a = k * math.pi / 4.0
                x0, y0 = x + math.cos(a) * 7, y + math.sin(a) * 7
                x1, y1 = x + math.cos(a) * 11, y + math.sin(a) * 11
                pygame.draw.line(surf, color, (x0, y0), (x1, y1))
        # fog volume: wireframe AABB + center marker (world-axis-aligned;
        # rotation is not applied to the box, see engine/lighting.FogVolume)
        for e in self.scene.entities:
            if e.fog_volume is None:
                continue
            p, s = e.transform.position, e.transform.scale
            lo = (p.x - abs(s.x), p.y - abs(s.y), p.z - abs(s.z))
            hi = (p.x + abs(s.x), p.y + abs(s.y), p.z + abs(s.z))
            corners = [(x, y, z) for x in (lo[0], hi[0])
                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            pts = [self.camera.project(self.engine_mod.Vec3(*c), w, h) for c in corners]
            color = tuple(e.fog_volume.color) if e.fog_volume.enabled else (90, 90, 95)
            edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                     (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
            for a, b in edges:
                if pts[a] is not None and pts[b] is not None:
                    pygame.draw.line(surf, color, (int(pts[a][0]), int(pts[a][1])),
                                     (int(pts[b][0]), int(pts[b][1])), 1)
            center = self.camera.project(p, w, h)
            if center is not None:
                cx, cy = int(center[0]), int(center[1])
                pygame.draw.circle(surf, color, (cx, cy), 5, 1)
                pygame.draw.line(surf, color, (cx - 7, cy), (cx + 7, cy))
                pygame.draw.line(surf, color, (cx, cy - 7), (cx, cy + 7))
        # 3D cursor: Blender-style red/white dashed ring + crosshair ticks at
        # self.cursor3d, drawn only when projectable (e.g. not behind camera)
        pt = self.camera.project(self.cursor3d, w, h)
        if pt is not None:
            x, y = int(pt[0]), int(pt[1])
            r, segs = 9, 16
            for k in range(segs):
                a0 = 2 * math.pi * k / segs
                a1 = 2 * math.pi * (k + 0.6) / segs
                color = (222, 40, 40) if k % 2 == 0 else (240, 240, 240)
                p0 = (x + math.cos(a0) * r, y + math.sin(a0) * r)
                p1 = (x + math.cos(a1) * r, y + math.sin(a1) * r)
                pygame.draw.line(surf, color, p0, p1, 2)
            tick = 4
            pygame.draw.line(surf, (240, 240, 240), (x - r - tick, y), (x - r + 2, y))
            pygame.draw.line(surf, (240, 240, 240), (x + r - 2, y), (x + r + tick, y))
            pygame.draw.line(surf, (240, 240, 240), (x, y - r - tick), (x, y - r + 2))
            pygame.draw.line(surf, (240, 240, 240), (x, y + r - 2), (x, y + r + tick))
        # snap-to-mesh feedback: highlight the AABB face the drag is
        # currently flush against (see _find_mesh_snap / _update_gizmo_drag)
        if self.snap_feedback is not None:
            other, axis_idx, plane, lo2, hi2 = self.snap_feedback
            other_axes = [k for k in range(3) if k != axis_idx]
            corners = []
            for a in (lo2[0], hi2[0]):
                for b in (lo2[1], hi2[1]):
                    c = [0.0, 0.0, 0.0]
                    c[axis_idx] = plane
                    c[other_axes[0]] = a
                    c[other_axes[1]] = b
                    corners.append(c)
            loop = (0, 1, 3, 2)  # corners are (lo,lo)(lo,hi)(hi,lo)(hi,hi) -> quad order
            pts = [self.camera.project(self.engine_mod.Vec3(*corners[i]), w, h) for i in loop]
            if all(p is not None for p in pts):
                poly = [(int(p[0]), int(p[1])) for p in pts]
                pygame.draw.polygon(surf, SNAP_FACE_COLOR, poly, 3)

        # selection brackets + transform gizmo
        if not self.selection:
            return
        e = self.selected
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
            label_text = self.gizmo_mode
            if self.gizmo_mode == "translate":
                label_text += f" ({self.gizmo_space})"
            mode_label = self.font_small.render(label_text, True, TEXT_DIM)
            surf.blit(mode_label, (int(s0[0]) + 12, int(s0[1]) + 10))
        # corner brackets around every selected entity -- ACCENT for the
        # ACTIVE element (identical to the pre-multiselect single-select
        # look), a dimmer shade for the rest (Blender distinguishes the
        # active object the same way)
        for ent in self.selection:
            self._draw_selection_bracket(surf, w, h, ent, ACCENT if ent is e else ACCENT_DIM)

    def _draw_selection_bracket(self, surf, w, h, e, color) -> None:
        import numpy as np
        import pygame
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
            pygame.draw.line(surf, color, (cx, cy), (cx - dx * s, cy), 2)
            pygame.draw.line(surf, color, (cx, cy), (cx, cy - dy * s), 2)
        label = self.font_small.render(e.name, True, color)
        surf.blit(label, (x - label.get_width() // 2, y - r - 16))

    def _draw_marquee(self, surf, layout) -> None:
        """Live drag rectangle for box/marquee select (see
        _begin_viewport_press / _finish_marquee): thin outline + faint
        fill, clipped to the viewport rect so a drag that wanders over a
        docked/floating panel never paints on top of it. Drawn only past
        MARQUEE_THRESHOLD -- a still-pending tiny press shows nothing,
        matching it staying a plain click on release."""
        if self.marquee is None:
            return
        import pygame
        sx, sy = self.marquee["start"]
        cx, cy = self.marquee["cur"]
        if math.hypot(cx - sx, cy - sy) < MARQUEE_THRESHOLD:
            return
        rect = pygame.Rect(min(sx, cx), min(sy, cy), abs(cx - sx), abs(cy - sy))
        prev_clip = surf.get_clip()
        surf.set_clip(layout["viewport"])
        fill = pygame.Surface((max(1, rect.width), max(1, rect.height)), pygame.SRCALPHA)
        fill.fill((*MARQUEE_COLOR, 40))
        surf.blit(fill, rect.topleft)
        pygame.draw.rect(surf, MARQUEE_COLOR, rect, 1)
        surf.set_clip(prev_clip)

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
            elif e in self.selection:
                pygame.draw.rect(surf, SELECT_BG_MULTI, row)
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
        hint = self.font_small.render(
            "Del delete · Ctrl+D dup · F focus · End floor · Ctrl+S save",
            True, TEXT_DIM)
        surf.blit(hint, (rect.x + 10, rect.bottom - 18))

    def _draw_transform_fields(self, surf, rect, e) -> None:
        import pygame
        axis_labels = ("X", "Y", "Z")
        for i, row in enumerate(self._transform_rows(e)):
            rr = self._transform_row_rect(rect, i)
            label = self.font_small.render(row["label"], True, TEXT_DIM)
            surf.blit(label, (rr.x, rr.y + 4))
            field_rects = self._transform_field_rects(rr)
            for j, fr in enumerate(field_rects):
                editing = self.editing_field == (row["label"], self._TRANSFORM_AXES[j])
                bg = (16, 17, 21) if editing else (30, 32, 39)
                pygame.draw.rect(surf, bg, fr, border_radius=2)
                pygame.draw.rect(surf, ACCENT if editing else PANEL_EDGE, fr, 1,
                                 border_radius=2)
                text = self.edit_buffer if editing else self._fmt_num(row["fields"][j]["get"]())
                color = TEXT if editing else self._GIZMO_AXES[j][1]
                glyph = self.font_small.render(f"{axis_labels[j]} {text}", True, color)
                surf.blit(glyph, (fr.x + 3, fr.y + (fr.height - glyph.get_height()) // 2))

    def _draw_details(self, surf, rect) -> None:
        import pygame
        e = self.selected
        if e is None:
            surf.blit(self.font_small.render("select an entity", True, TEXT_DIM),
                      (rect.x + 10, rect.y + 8))
            return
        head = f"{e.name}" + (f"  ({e.asset_name})" if e.asset_name else "")
        avail = rect.width - 20
        tag = None
        if len(self.selection) > 1:
            # active element's transform/rows are shown below (typed edits,
            # rotate/scale hotkeys, and the material editor all stay
            # active-only for this run -- see _duplicate_selected docstring
            # and CLAUDE.md for the run 2b pivot-mode followup)
            tag = self.font_small.render(f"{len(self.selection)} selected", True, TEXT_DIM)
            avail -= tag.get_width() + 8
        while head and self.font_small.size(head)[0] > avail:
            head = head[:-1]  # pixel-precise truncation -- never overlaps the tag
        surf.blit(self.font_small.render(head, True, TEXT), (rect.x + 10, rect.y + 6))
        if tag is not None:
            surf.blit(tag, (rect.right - tag.get_width() - 10, rect.y + 6))
        self._draw_transform_fields(surf, rect, e)

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
            elif row["kind"] == "material_slot":
                ent = row["entity"]
                swatch = pygame.Rect(rr.x + 96, rr.y + 2, 20, 20)
                if ent.material is None:
                    pygame.draw.rect(surf, (26, 27, 32), swatch, border_radius=3)
                    pygame.draw.rect(surf, PANEL_EDGE, swatch, 1, border_radius=3)
                    txt = self.font_small.render("empty — drop material here", True, TEXT_DIM)
                else:
                    icon = self.mat_icon_cache.get(id(ent))
                    if icon is None:
                        icon = make_material_icon(self.engine_mod, ent.material)
                        self.mat_icon_cache[id(ent)] = icon
                    if icon is not None:
                        surf.blit(pygame.transform.smoothscale(icon, (20, 20)), swatch)
                    else:
                        pygame.draw.rect(surf, (90, 90, 100), swatch, border_radius=3)
                    pygame.draw.rect(surf, PANEL_EDGE, swatch, 1, border_radius=3)
                    name = ent.material_asset or "(unsaved graph)"
                    txt = self.font_small.render(name[:22], True, TEXT)
                surf.blit(txt, (swatch.right + 6, rr.y + 5))

    def _draw_browser_topbar(self, surf, topbar, mp) -> None:
        import pygame
        pygame.draw.line(surf, PANEL_EDGE, (topbar.x, topbar.bottom),
                         (topbar.right, topbar.bottom))
        nfb = self._new_folder_btn_rect(topbar)
        pygame.draw.rect(surf, HOVER_BG if nfb.collidepoint(mp) else (33, 36, 44),
                         nfb, border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, nfb, 1, border_radius=4)
        nflabel = self.font_small.render("+ Folder", True, ACCENT)
        surf.blit(nflabel, (nfb.x + (nfb.width - nflabel.get_width()) // 2, nfb.y + 4))
        nbb = self._new_blueprint_btn_rect(topbar)
        pygame.draw.rect(surf, HOVER_BG if nbb.collidepoint(mp) else (33, 36, 44),
                         nbb, border_radius=4)
        pygame.draw.rect(surf, BLUEPRINT_TILE_EDGE, nbb, 1, border_radius=4)
        nblabel = self.font_small.render("+ Blueprint", True, (130, 210, 150))
        surf.blit(nblabel, (nbb.x + (nbb.width - nblabel.get_width()) // 2, nbb.y + 4))
        btn = self._import_btn_rect(topbar)
        pygame.draw.rect(surf, HOVER_BG if btn.collidepoint(mp) else (33, 36, 44),
                         btn, border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, btn, 1, border_radius=4)
        label = self.font_small.render("Import", True, ACCENT)
        surf.blit(label, (btn.x + (btn.width - label.get_width()) // 2, btn.y + 4))
        exp = self._export_btn_rect(topbar)
        exportable = (self.selected_asset is not None
                     and not isinstance(self.selected_asset, self.engine_mod.MaterialAsset)
                     and not isinstance(self.selected_asset, self.engine_mod.BlueprintAsset)
                     and self.engine_mod.has_mesh(self.selected_asset))
        exp_color = ACCENT if exportable else TEXT_DIM
        pygame.draw.rect(surf, HOVER_BG if exportable and exp.collidepoint(mp)
                         else (33, 36, 44), exp, border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, exp, 1, border_radius=4)
        exp_label = self.font_small.render("Export", True, exp_color)
        surf.blit(exp_label, (exp.x + (exp.width - exp_label.get_width()) // 2, exp.y + 4))
        if self.status[1] > 0:
            avail = exp.x - (nbb.right + 8)
            msg = self.font_small.render(self.status[0][:60], True, (235, 210, 140))
            if avail > 20:
                surf.blit(msg, (nbb.right + 8, topbar.y + 6))

    def _draw_browser_tree(self, surf, tree_rect, mp) -> None:
        import pygame
        pygame.draw.line(surf, PANEL_EDGE, (tree_rect.right, tree_rect.y),
                         (tree_rect.right, tree_rect.bottom))
        clip = self._tree_rows_clip(tree_rect)
        rows = self._folder_tree_rows()
        for i, (fid, depth, name) in enumerate(rows):
            rr = self._tree_row_rect(tree_rect, i)
            if rr.bottom < tree_rect.y or rr.bottom > clip.bottom:
                continue
            selected = fid == self.selected_folder
            if selected:
                pygame.draw.rect(surf, SELECT_BG, rr, border_radius=2)
            elif rr.collidepoint(mp):
                pygame.draw.rect(surf, HOVER_BG, rr, border_radius=2)
            if self.renaming_folder is not _NO_RENAME and self.renaming_folder == fid:
                pygame.draw.rect(surf, (16, 17, 21), rr.inflate(-2, -2), border_radius=2)
                pygame.draw.rect(surf, ACCENT, rr.inflate(-2, -2), 1, border_radius=2)
                text = self.rename_buffer
            else:
                text = name
            lab = self.font_small.render(text[:20], True, TEXT if selected else TEXT_DIM)
            surf.blit(lab, (rr.x + 4 + depth * 12, rr.y + 3))
        hint = self.font_small.render("F2 rename", True, TEXT_DIM)
        surf.blit(hint, (tree_rect.x + 4, tree_rect.bottom - 16))

    def _tiles_in(self, folder_id):
        """Regular assets, then blueprints (both participate in the folder
        tree), then material assets (materials only live at the browser
        root -- they aren't part of the folder tree)."""
        tiles = list(self.lib.assets_in(folder_id)) + list(self.lib.blueprints_in(folder_id))
        if folder_id is None:
            tiles += list(self.lib.materials)
        return tiles

    def _draw_browser_grid(self, surf, grid_rect, mp) -> None:
        import pygame
        x = grid_rect.x + 10 - self.browser_scroll
        for asset in self._tiles_in(self.selected_folder):
            is_mat = isinstance(asset, self.engine_mod.MaterialAsset)
            is_bp = isinstance(asset, self.engine_mod.BlueprintAsset)
            tile = pygame.Rect(x, grid_rect.y + 6, TILE_W, TILE_H - 12)
            if tile.right > grid_rect.x and tile.left < grid_rect.right:
                hovered = tile.collidepoint(mp)
                selected = asset is self.selected_asset
                if selected:
                    pygame.draw.rect(surf, SELECT_BG, tile, border_radius=4)
                else:
                    pygame.draw.rect(surf, HOVER_BG if hovered else (30, 32, 39), tile,
                                     border_radius=4)
                if selected:
                    pygame.draw.rect(surf, ACCENT, tile, 1, border_radius=4)
                if is_mat:
                    pygame.draw.rect(surf, (90, 60, 140), tile, 1, border_radius=4)
                elif is_bp:
                    pygame.draw.rect(surf, BLUEPRINT_TILE_EDGE, tile, 1, border_radius=4)
                icon = (self.mat_icons if is_mat else
                       (self.bp_icons if is_bp else self.icons)).get(asset.name)
                if icon is not None:
                    surf.blit(icon, (x + (TILE_W - ICON) // 2, grid_rect.y + 10))
                label = self.font_small.render(asset.name[:12], True,
                                               TEXT if (hovered or selected) else TEXT_DIM)
                surf.blit(label, (x + (TILE_W - label.get_width()) // 2,
                                  grid_rect.y + 10 + ICON + 3))
            x += TILE_W + 8

    def _draw_browser(self, surf, rect) -> None:
        import pygame
        mp = pygame.mouse.get_pos()
        blay = self._browser_layout(rect)
        self._draw_browser_topbar(surf, blay["topbar"], mp)
        self._draw_browser_tree(surf, blay["tree"], mp)
        self._draw_browser_grid(surf, blay["grid"], mp)

    def _draw_console(self, surf, rect) -> None:
        """Engine message log -- newest at the bottom, auto-scrolling to
        follow new entries UNLESS the user has scrolled up into history (see
        `console_scroll`'s docstring in __init__), level-colored, monospace.
        """
        import time as _time

        import pygame
        from engine import console_log

        pygame.draw.rect(surf, (16, 17, 21), rect)
        entries = list(console_log.get_log().entries)
        n = len(entries)
        top_pad = 4
        visible = max(0, (rect.height - top_pad * 2)) // CONSOLE_ROW_H
        # a scrolled-up viewport must not get yanked back to the tail just
        # because new entries arrived -- advance the offset by exactly how
        # many new entries showed up so the SAME historical lines stay put
        prev_n = getattr(self, "_console_last_len", n)
        if self.console_scroll > 0:
            self.console_scroll += max(0, n - prev_n)
        self._console_last_len = n
        max_scroll = max(0, n - visible)
        self.console_scroll = max(0, min(self.console_scroll, max_scroll))
        end = n - self.console_scroll
        start = max(0, end - visible)
        prev_clip = surf.get_clip()
        surf.set_clip(rect)
        y = rect.y + top_pad
        for entry in entries[start:end]:
            color = CONSOLE_LEVEL_COLOR.get(entry["level"], TEXT)
            ts = _time.strftime("%H:%M:%S", _time.localtime(entry["time"]))
            line = f"[{ts}] {entry['text']}"
            lab = self.font_small.render(line, True, color)
            surf.blit(lab, (rect.x + 6, y))
            y += CONSOLE_ROW_H
        surf.set_clip(prev_clip)
        if self.console_scroll > 0:
            hint = self.font_small.render(
                f"-- scrolled up ({self.console_scroll}) -- scroll down to follow --",
                True, TEXT_DIM)
            surf.blit(hint, (rect.right - hint.get_width() - 6,
                             rect.bottom - hint.get_height() - 2))


NODE_W = 150


# node type -> UE-ish display name (search matches anywhere in this string)
NODE_DISPLAY = {
    "constant": "Constant", "constant2vector": "Constant2Vector",
    "constant3vector": "Constant3Vector", "constant4vector": "Constant4Vector",
    "position": "Position", "normal": "Normal", "checker": "Checker",
    "noise": "Noise", "gradient": "Gradient", "mix": "Mix", "lerp": "Lerp",
    "multiply": "Multiply", "add": "Add", "subtract": "Subtract",
    "divide": "Divide", "power": "Power", "clamp": "Clamp",
    "one_minus": "OneMinus", "abs": "Abs", "floor": "Floor", "frac": "Frac",
    "sine": "Sine", "cosine": "Cosine", "dot_product": "DotProduct",
    "vmax": "Max", "vmin": "Min", "component_mask": "ComponentMask",
    "hdri": "HDRI", "tex_coord": "TexCoord", "tex_sample": "TextureSample",
}
# node type -> title-bar category, driving CATEGORY_COLOR (incremental UE-ish
# restyle -- doesn't touch the rest of the node body drawing)
NODE_CATEGORY = {"output": "output",
                 **{t: "constant" for t in ("constant", "constant2vector",
                                            "constant3vector", "constant4vector",
                                            "position", "normal")},
                 **{t: "texture" for t in ("checker", "noise", "gradient", "hdri",
                                          "tex_coord", "tex_sample")}}
CATEGORY_COLOR = {"output": (68, 38, 38), "constant": (36, 56, 64),
                  "texture": (38, 60, 42), "math": (42, 42, 60)}
# seed usage counts (descending) so the right-click "add node" menu's top-10
# is useful before any real usage history has accumulated
DEFAULT_NODE_USAGE = {"constant3vector": 100, "multiply": 90, "add": 80,
                      "lerp": 70, "clamp": 60, "noise": 50, "tex_sample": 40,
                      "power": 30, "one_minus": 20, "checker": 10}


class MaterialEditorUI:
    """Node-based material editor: drag ports to connect, drag params to tune.

    The graph bakes to the entity mesh's per-face colors on every change, so
    the 3D viewport behind the panel is a live preview. Floating only — drag
    its 18px title bar to move it, click X (or M/Esc) to close.

    UE-style workflow: nodes are added via a right-click context menu on the
    canvas (search bar + top-10-by-usage), not a top-bar palette. Right-click
    on a node opens Delete/Disconnect/Duplicate/Preview actions. A preview
    panel docks to the left, rendering the current (or isolated-node) graph
    output on a sphere.
    """

    ADDABLE_TYPES = ("constant", "constant2vector", "constant3vector", "constant4vector",
                     "position", "normal", "checker", "noise", "gradient",
                     "mix", "lerp", "multiply", "add", "subtract", "divide", "power",
                     "clamp", "one_minus", "abs", "floor", "frac", "sine", "cosine",
                     "dot_product", "vmax", "vmin", "component_mask", "hdri",
                     "tex_coord", "tex_sample")
    DEFAULT_SIZE = (900, 560)
    PREVIEW_W = 150       # left preview-panel strip width, inside content_rect
    PREVIEW_SIZE = 108    # rendered preview sphere resolution
    CTX_MENU_W = 220
    CTX_ROW_H = 20
    CTX_SEARCH_H = 22

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
        self.minimized = False
        self._spawn_i = 0
        self.ctx_menu = None      # dict, see _open_add_menu/_open_node_menu
        self.preview_nid = None   # None == showing the real Output; else isolated node id
        self._preview_surf = None
        self._preview_dirty = True
        self._preview_stop_rect = None
        self._blend_opaque_rect = None
        self._blend_translucent_rect = None

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

    def outer_rect(self, w, h):
        """The on-screen box — collapsed to the title bar while minimized."""
        import pygame
        r = self.rect(w, h)
        if self.minimized:
            return pygame.Rect(r.x, r.y, r.width, PANEL_TITLE_H)
        return r

    def content_rect(self, w, h):
        import pygame
        outer = self.rect(w, h)
        return pygame.Rect(outer.x, outer.y + PANEL_TITLE_H, outer.width,
                           max(0, outer.height - PANEL_TITLE_H))

    def preview_rect(self, w, h):
        """Left preview-panel strip (UE-style), inside content_rect."""
        import pygame
        content = self.content_rect(w, h)
        pw = min(self.PREVIEW_W, max(0, content.width - 200))
        return pygame.Rect(content.x, content.y, pw, content.height)

    def graph_panel(self, w, h):
        """Node-graph canvas -- content_rect minus the preview strip. All node
        positions/hit-tests are relative to THIS rect's origin, not content_rect's."""
        import pygame
        content = self.content_rect(w, h)
        prev = self.preview_rect(w, h)
        return pygame.Rect(content.x + prev.width, content.y,
                           max(0, content.width - prev.width), content.height)

    def _out_ports(self, node_type):
        return self.editor.engine_mod.NODE_OUTPUTS.get(node_type, ("out",))

    def _rows_top(self, nid):
        """Row count reserved for input/output ports (whichever has more) --
        a node with only the implicit single output doesn't reserve a row for
        it (drawn centered on the node instead), so this only grows for
        multi-output nodes like tex_sample."""
        node = self.graph.nodes[nid]
        inputs, _ = self.editor.engine_mod.NODE_DEFS[node["type"]]
        outputs = self._out_ports(node["type"])
        n_out = len(outputs) if len(outputs) > 1 else 0
        return max(len(inputs), n_out)

    def node_rect(self, nid, panel):
        import pygame
        node = self.graph.nodes[nid]
        rows_top = self._rows_top(nid)
        extra = 1 if node["type"] == "tex_sample" else 0  # texture-picker row
        height = 24 + rows_top * 18 + (len(node["params"]) + extra) * 18 + 6
        return pygame.Rect(int(panel.x + node["pos"][0]),
                           int(panel.y + node["pos"][1]), NODE_W, height)

    def input_pos(self, nid, index, panel):
        r = self.node_rect(nid, panel)
        return (r.x, r.y + 24 + index * 18 + 9)

    def output_pos(self, nid, panel, port=None):
        r = self.node_rect(nid, panel)
        node = self.graph.nodes[nid]
        outputs = self._out_ports(node["type"])
        if len(outputs) <= 1 or port is None:
            return (r.right, r.y + r.height // 2)
        return (r.right, r.y + 24 + outputs.index(port) * 18 + 9)

    def _param_row(self, nid, j, panel):
        import pygame
        r = self.node_rect(nid, panel)
        rows_top = self._rows_top(nid)
        return pygame.Rect(r.x + 6, r.y + 24 + (rows_top + j) * 18,
                           NODE_W - 12, 16)

    def _texture_row_rect(self, nid, panel):
        import pygame
        node = self.graph.nodes[nid]
        return self._param_row(nid, len(node["params"]), panel)

    # ---- node usage tracking (drives the add-node menu's top-10) ----
    def _usage_path(self) -> str:
        d = os.path.dirname(self.editor.settings_path) or "."
        return os.path.join(d, "mat_node_usage.json")

    def _load_usage(self) -> dict:
        data = load_settings(self._usage_path())
        return data if data else dict(DEFAULT_NODE_USAGE)

    def _bump_usage(self, node_type: str) -> None:
        usage = self._load_usage()
        usage[node_type] = usage.get(node_type, 0) + 1
        save_settings(usage, self._usage_path())

    def _top10(self) -> list[str]:
        usage = self._load_usage()
        return sorted(self.ADDABLE_TYPES, key=lambda t: -usage.get(t, 0))[:10]

    # ---- interaction ----
    def apply(self, draft: bool = False) -> None:
        """Re-bake the graph. `draft=True` (used while dragging a param
        slider) bakes a sky material at quarter resolution -- cheap enough
        for every frame; the final release re-bakes at full resolution."""
        self.graph.apply(self.entity, draft=draft)
        self.editor.dirty = True
        self._preview_dirty = True
        if not draft and self.entity.mesh is not None:
            # keep the Details-panel material slot preview in sync with
            # live node-editor edits (requirement: "slot preview updates
            # when the assigned material is edited in the node editor")
            self.editor.mat_icon_cache[id(self.entity)] = make_material_icon(
                self.editor.engine_mod, self.graph)

    def _save_as_asset(self) -> None:
        """Save the current graph as a reusable material asset in the
        content browser (Unreal: right-click a material instance -> Save
        as Asset). Auto-names from the owning entity, de-duping on clash."""
        lib = self.editor.lib
        base = f"{self.entity.name} Material"
        name = base
        i = 2
        while name in lib.material_by_name:
            name = f"{base} {i}"
            i += 1
        mat = lib.save_material(name, self.graph)
        self.editor.mat_icons[name] = make_material_icon(self.editor.engine_mod, self.graph)
        self.entity.material_asset = name
        self.editor.status = (f"saved material asset '{name}'", 4.0)

    def update(self, engine, dt: float) -> None:
        import pygame
        inp = engine.input
        mp = inp.mouse_pos
        w, h = engine.screen.get_size()
        outer = self.outer_rect(w, h)
        title_bar = pygame.Rect(outer.x, outer.y, outer.width, PANEL_TITLE_H)
        close = pygame.Rect(outer.right - 24, outer.y + 2, 16, 16)
        minimize = pygame.Rect(outer.right - 44, outer.y + 2, 16, 16)
        save_asset = pygame.Rect(outer.right - 140, outer.y + 2, 90, 16)

        if self.ctx_menu is not None:
            self._update_ctx_menu(inp, mp)
            return
        if inp.pressed(pygame.K_m):
            self.close()
            return
        if inp.mouse_button_pressed(3) and not self.minimized:
            panel = self.graph_panel(w, h)
            if panel.collidepoint(mp):
                self._open_context_menu(mp, panel)
                return
        if inp.mouse_button_pressed(1):
            if close.collidepoint(mp):
                self.close()
                return
            if minimize.collidepoint(mp):
                self.minimized = not self.minimized
                return
            if save_asset.collidepoint(mp):
                self._save_as_asset()
                return
            if not self.minimized and self._blend_opaque_rect is not None \
                    and self._blend_opaque_rect.collidepoint(mp):
                self.graph.set_blend_mode("opaque")
                self.apply(draft=False)
                return
            if not self.minimized and self._blend_translucent_rect is not None \
                    and self._blend_translucent_rect.collidepoint(mp):
                self.graph.set_blend_mode("translucent")
                self.apply(draft=False)
                return
            if title_bar.collidepoint(mp):
                self.drag_title = (mp[0] - outer.x, mp[1] - outer.y)
            elif not self.minimized and self._preview_stop_rect is not None \
                    and self._preview_stop_rect.collidepoint(mp):
                self.preview_nid = None
                self._preview_dirty = True
            elif not self.minimized:
                self._press(mp, self.graph_panel(w, h))
        if inp.mouse_held(1):
            if self.drag_title is not None:
                dx, dy = self.drag_title
                self.pos = [mp[0] - dx, mp[1] - dy]
            if not self.minimized:
                panel = self.graph_panel(w, h)
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
                        self.apply(draft=True)
        else:
            if self.drag_link is not None and not self.minimized:
                self._finish_link(mp, self.graph_panel(w, h))
            if self.drag_param is not None:
                self.apply(draft=False)  # full-res bake once the drag releases
            self.drag_node = self.drag_param = self.drag_link = self.drag_title = None

    def _node_at(self, mp, panel):
        """Topmost node whose body rect contains `mp`, or None."""
        for nid in reversed(list(self.graph.nodes)):
            if self.node_rect(nid, panel).collidepoint(mp):
                return nid
        return None

    # ---- right-click context menu (UE-style add-node / node-actions) ----
    def _open_context_menu(self, mp, panel) -> None:
        nid = self._node_at(mp, panel)
        if nid is not None:
            self._open_node_menu(mp, nid, panel)
        else:
            self._open_add_menu(mp, panel)

    def _open_add_menu(self, mp, panel) -> None:
        self.ctx_menu = {"kind": "add", "screen_pos": mp,
                         "graph_pos": (mp[0] - panel.x, mp[1] - panel.y),
                         "search": "", "top10": self._top10()}

    def _open_node_menu(self, mp, nid, panel) -> None:
        NODE_DEFS = self.editor.engine_mod.NODE_DEFS
        node = self.graph.nodes[nid]
        inputs, _ = NODE_DEFS[node["type"]]
        connected = [name for name in inputs if self.graph.link_into(nid, name) is not None]
        items = []  # (label, action_key, payload)
        is_output = node["type"] == "output"
        if not is_output:
            items.append(("Delete", "delete", None))
        if connected:
            items.append(("Break All Node Links", "disconnect_all", None))
            for name in connected:
                items.append((f"Break Link: {name}", "disconnect_one", name))
        if not is_output:
            items.append(("Duplicate", "duplicate", None))
            label = ("Stop Previewing Node" if self.preview_nid == nid
                    else "Start Previewing Node")
            items.append((label, "toggle_preview", None))
        self.ctx_menu = {"kind": "node", "screen_pos": mp, "nid": nid, "items": items}

    def _update_ctx_menu(self, inp, mp) -> None:
        import pygame
        menu = self.ctx_menu
        if inp.pressed(pygame.K_ESCAPE):
            self.ctx_menu = None
            return
        if menu["kind"] == "add":
            for ch in inp.take_text():
                if ch.isprintable():
                    menu["search"] += ch
            if inp.pressed(pygame.K_BACKSPACE):
                menu["search"] = menu["search"][:-1]
            matches = self._ctx_search_matches(menu)
            if inp.pressed(pygame.K_RETURN) or inp.pressed(pygame.K_KP_ENTER):
                if matches:
                    self._add_node_from_menu(matches[0], menu["graph_pos"])
                self.ctx_menu = None
                return
        if inp.mouse_button_pressed(1):
            self._click_ctx_menu(mp)
            return
        if inp.mouse_button_pressed(3):
            self.ctx_menu = None  # right-clicking elsewhere dismisses it too

    def _ctx_search_matches(self, menu) -> list[str]:
        """Matches for the add-menu's current search text, computed purely
        from `menu` (never `self.ctx_menu` -- this is called from
        `_ctx_menu_rows` with a locally-captured menu dict after
        `self.ctx_menu` may already have been cleared, e.g. mid-click)."""
        search = menu["search"]
        if not search:
            return menu["top10"]
        s = search.lower()
        return [t for t in self.ADDABLE_TYPES
               if s in NODE_DISPLAY.get(t, t).lower() or s in t][:20]

    def _add_node_from_menu(self, node_type: str, graph_pos) -> None:
        self.graph.add(node_type, graph_pos)
        self._bump_usage(node_type)
        self.apply()

    def _click_ctx_menu(self, mp) -> None:
        """Hit-test the click against the SAME total rect `_draw_ctx_menu`
        draws (search bar + header + rows), handle a hit entry, THEN close.
        A click truly outside that rect closes the menu without running any
        entry action. Closing only after the action keeps `menu` (the local
        capture) the sole source of truth for `_ctx_menu_rows` /
        `_ctx_search_matches` while they run -- `self.ctx_menu` must not go
        None out from under them mid-click."""
        menu = self.ctx_menu
        if not self._ctx_menu_total_rect(menu).collidepoint(mp):
            self.ctx_menu = None
            return
        for _label, rect, payload in self._ctx_menu_rows(menu):
            if not rect.collidepoint(mp):
                continue
            if menu["kind"] == "add":
                self._add_node_from_menu(payload, menu["graph_pos"])
            else:
                self._run_node_action(menu["nid"], payload)
            break
        self.ctx_menu = None

    def _run_node_action(self, nid, action) -> None:
        kind, arg = action
        if nid not in self.graph.nodes:
            return
        if kind == "delete":
            if self.preview_nid == nid:
                self.preview_nid = None
            self.graph.remove(nid)
            self.apply()
        elif kind == "disconnect_all":
            NODE_DEFS = self.editor.engine_mod.NODE_DEFS
            inputs, _ = NODE_DEFS[self.graph.nodes[nid]["type"]]
            for name in inputs:
                self.graph.disconnect(nid, name)
            self.graph.links = [l for l in self.graph.links if l[0] != nid]
            self.apply()
        elif kind == "disconnect_one":
            self.graph.disconnect(nid, arg)
            self.apply()
        elif kind == "duplicate":
            src = self.graph.nodes[nid]
            new_id = self.graph.add(src["type"],
                                    (src["pos"][0] + 24, src["pos"][1] + 24))
            self.graph.nodes[new_id]["params"] = dict(src["params"])
            for k, v in src.items():
                if k not in ("type", "pos", "params"):
                    self.graph.nodes[new_id][k] = v
            self.apply()
        elif kind == "toggle_preview":
            self.preview_nid = None if self.preview_nid == nid else nid
            self._preview_dirty = True

    CTX_HEADER_H = 16  # "Common" section-label row, add-menu only

    def _ctx_header_rect(self, menu):
        """The "Common" section-label row (add-menu, empty search, non-empty
        top-10 only) -- its own row, stacked below the search bar and above
        the entries, never overlapping either (house rule: same rect for
        draw + layout math)."""
        import pygame
        if menu["kind"] != "add" or menu["search"] or not menu["top10"]:
            return None
        x, y = menu["screen_pos"]
        return pygame.Rect(x, y + self.CTX_SEARCH_H, self.CTX_MENU_W, self.CTX_HEADER_H)

    def _ctx_menu_total_rect(self, menu):
        """The full menu rect (search bar + header + rows) -- the single
        source of truth for both drawing and click hit-testing, so they can
        never disagree."""
        import pygame
        x, y = menu["screen_pos"]
        search_h = self.CTX_SEARCH_H if menu["kind"] == "add" else 0
        header_h = self.CTX_HEADER_H if self._ctx_header_rect(menu) is not None else 0
        total_h = search_h + header_h + len(self._ctx_menu_rows(menu)) * self.CTX_ROW_H
        return pygame.Rect(x, y, self.CTX_MENU_W, total_h)

    def _ctx_menu_rows(self, menu):
        """[(label, rect, payload), ...] -- same list drives draw + hit-test.
        `payload` is a node type string for the "add" menu, or an
        (action_kind, arg) pair for the "node" menu."""
        import pygame
        x, y = menu["screen_pos"]
        rows = []
        if menu["kind"] == "add":
            y += self.CTX_SEARCH_H  # search bar itself isn't a clickable row
            header = self._ctx_header_rect(menu)
            if header is not None:
                y += self.CTX_HEADER_H  # section label isn't a clickable row either
            for t in self._ctx_search_matches(menu):
                rows.append((NODE_DISPLAY.get(t, t),
                            pygame.Rect(x, y, self.CTX_MENU_W, self.CTX_ROW_H), t))
                y += self.CTX_ROW_H
        else:
            for label, action_kind, payload in menu["items"]:
                rows.append((label, pygame.Rect(x, y, self.CTX_MENU_W, self.CTX_ROW_H),
                            (action_kind, payload)))
                y += self.CTX_ROW_H
        return rows

    def _press(self, mp, panel) -> None:
        import pygame
        NODE_DEFS = self.editor.engine_mod.NODE_DEFS
        for nid in reversed(list(self.graph.nodes)):
            node = self.graph.nodes[nid]
            r = self.node_rect(nid, panel)
            # output port(s) -- most node types have one; tex_sample has five
            if node["type"] != "output":
                outputs = self._out_ports(node["type"])
                for oport in outputs:
                    ox, oy = self.output_pos(nid, panel, oport if len(outputs) > 1 else None)
                    if math.hypot(mp[0] - ox, mp[1] - oy) < 9:
                        self.drag_link = (nid, oport)
                        return
            # input ports: click to unplug (and grab the wire), or nothing
            inputs, _ = NODE_DEFS[node["type"]]
            for i, name in enumerate(inputs):
                ix, iy = self.input_pos(nid, i, panel)
                if math.hypot(mp[0] - ix, mp[1] - iy) < 9:
                    link = self.graph.link_into(nid, name)
                    if link is not None:
                        self.graph.disconnect(nid, name)
                        self.drag_link = link  # re-route the existing wire
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
            # texture picker row (tex_sample): click cycles the assigned asset
            if node["type"] == "tex_sample" and self._texture_row_rect(nid, panel
                                                                        ).collidepoint(mp):
                self._cycle_texture(nid)
                self.apply()
                return
            # body drag
            self.drag_node = (nid, mp[0] - r.x, mp[1] - r.y)
            return

    def _cycle_texture(self, nid) -> None:
        """Cycle a tex_sample node's assigned texture through every texture
        asset in the library (plus "" for none), sorted by rel path."""
        rel_paths = sorted(a.data["texture"]["path"] for a in self.editor.lib.assets
                           if "texture" in a.data)
        options = [""] + rel_paths
        node = self.graph.nodes[nid]
        cur = node.get("texture", "")
        i = (options.index(cur) + 1) % len(options) if cur in options else 0
        node["texture"] = options[i]

    def _finish_link(self, mp, panel) -> None:
        NODE_DEFS = self.editor.engine_mod.NODE_DEFS
        src_id, src_port = self.drag_link
        for nid, node in self.graph.nodes.items():
            inputs, _ = NODE_DEFS[node["type"]]
            for i, name in enumerate(inputs):
                ix, iy = self.input_pos(nid, i, panel)
                if math.hypot(mp[0] - ix, mp[1] - iy) < 12:
                    if self.graph.connect(src_id, nid, name, src_port):
                        self.apply()
                    return

    # ---- drawing ----
    def draw(self, surf) -> None:
        import pygame
        w, h = surf.get_size()
        outer = self.outer_rect(w, h)
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
        minimize = pygame.Rect(outer.right - 44, outer.y + 2, 16, 16)
        pygame.draw.rect(surf, (40, 44, 54), minimize, border_radius=3)
        m_lab = self.editor.font_small.render("-", True, TEXT)
        surf.blit(m_lab, (minimize.x + 5, minimize.y - 1))
        save_asset = pygame.Rect(outer.right - 140, outer.y + 2, 90, 16)
        pygame.draw.rect(surf, (40, 44, 54), save_asset, border_radius=3)
        s_lab = self.editor.font_small.render("Save Asset", True, TEXT)
        surf.blit(s_lab, (save_asset.x + 6, save_asset.y + 1))
        if self.minimized:
            return

        content = self.content_rect(w, h)
        prev_r = self.preview_rect(w, h)
        panel = self.graph_panel(w, h)
        self._draw_preview_panel(surf, prev_r)
        pygame.draw.line(surf, PANEL_EDGE, (panel.x, content.y), (panel.x, content.bottom))

        NODE_DEFS = self.editor.engine_mod.NODE_DEFS
        # wires
        for src, dst, name, port in self.graph.links:
            if src not in self.graph.nodes or dst not in self.graph.nodes:
                continue
            inputs, _ = NODE_DEFS[self.graph.nodes[dst]["type"]]
            if name not in inputs:
                continue
            outputs = self._out_ports(self.graph.nodes[src]["type"])
            a = self.output_pos(src, panel, port if len(outputs) > 1 else None)
            b = self.input_pos(dst, inputs.index(name), panel)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            pygame.draw.lines(surf, (150, 160, 185), False,
                              [a, (a[0] + 18, a[1]), mid, (b[0] - 18, b[1]), b], 2)
        if self.drag_link is not None:
            src_id, src_port = self.drag_link
            outputs = self._out_ports(self.graph.nodes[src_id]["type"])
            a = self.output_pos(src_id, panel, src_port if len(outputs) > 1 else None)
            pygame.draw.line(surf, ACCENT, a, pygame.mouse.get_pos(), 2)

        # nodes
        for nid, node in self.graph.nodes.items():
            r = self.node_rect(nid, panel)
            pygame.draw.rect(surf, (33, 36, 44), r, border_radius=5)
            category = NODE_CATEGORY.get(node["type"], "math")
            title_r = pygame.Rect(r.x, r.y, r.width, 18)
            pygame.draw.rect(surf, CATEGORY_COLOR[category], title_r,
                             border_top_left_radius=5, border_top_right_radius=5)
            edge_color = ACCENT if nid == self.preview_nid else (70, 75, 88)
            pygame.draw.rect(surf, edge_color, r, 2 if nid == self.preview_nid else 1,
                             border_radius=5)
            name = self.editor.font_small.render(
                NODE_DISPLAY.get(node["type"], node["type"]), True, TEXT)
            surf.blit(name, (r.x + 8, r.y + 5))
            if node["type"] != "output":
                pygame.draw.line(surf, (120, 80, 80), (r.right - 16, r.y + 6),
                                 (r.right - 7, r.y + 15), 2)
                pygame.draw.line(surf, (120, 80, 80), (r.right - 7, r.y + 6),
                                 (r.right - 16, r.y + 15), 2)
                outputs = self._out_ports(node["type"])
                for oport in outputs:
                    ox, oy = self.output_pos(nid, panel, oport if len(outputs) > 1 else None)
                    pygame.draw.circle(surf, (210, 190, 120), (ox, oy), 5)
                    if len(outputs) > 1:
                        lab = self.editor.font_small.render(oport, True, TEXT_DIM)
                        surf.blit(lab, (ox - 8 - lab.get_width(), oy - 7))
            inputs, _ = NODE_DEFS[node["type"]]
            for i, iname in enumerate(inputs):
                ix, iy = self.input_pos(nid, i, panel)
                # Output.Opacity is only connectable in translucent mode
                # (see MaterialGraph.connect) -- grey the pin out here so the
                # UI reflects that gating instead of just silently refusing
                # the drag, UE-style.
                gated = (node["type"] == "output" and iname == "opacity"
                        and self.graph.blend_mode != "translucent")
                pin_color = (70, 74, 82) if gated else (140, 170, 210)
                lab_color = (90, 92, 98) if gated else TEXT_DIM
                pygame.draw.circle(surf, pin_color, (ix, iy), 5)
                lab = self.editor.font_small.render(iname, True, lab_color)
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
            if node["type"] == "tex_sample":
                rr = self._texture_row_rect(nid, panel)
                shown = node.get("texture", "") or "<none>"
                lab = self.editor.font_small.render(f"tex: {shown}", True, TEXT_DIM)
                surf.blit(lab, (rr.x, rr.y + 2))
            if node["type"] in ("color", "constant3vector", "constant4vector"):
                p = node["params"]
                sw = (int(p["r"] * 255), int(p["g"] * 255), int(p["b"] * 255))
                pygame.draw.rect(surf, sw, (r.x + 60, r.y + 4, 40, 12))

        if self.ctx_menu is not None:
            self._draw_ctx_menu(surf)

    def _draw_preview_panel(self, surf, prev_r) -> None:
        import pygame
        pygame.draw.rect(surf, (26, 28, 34), prev_r)
        pygame.draw.line(surf, PANEL_EDGE, (prev_r.x, prev_r.y), (prev_r.right, prev_r.y))
        if self._preview_dirty or self._preview_surf is None:
            self._preview_surf = self._render_preview()
            self._preview_dirty = False
        sw = min(self.PREVIEW_SIZE, prev_r.width - 16)
        sx = prev_r.x + (prev_r.width - sw) // 2
        sy = prev_r.y + 10
        if sw > 0:
            scaled = pygame.transform.smoothscale(self._preview_surf, (sw, sw))
            surf.blit(scaled, (sx, sy))
            pygame.draw.rect(surf, PANEL_EDGE, (sx, sy, sw, sw), 1)
        # Blend-mode selector (UE-style opaque/translucent two-button toggle)
        # -- lives in the preview strip since it's a graph-level property,
        # not a node. Opacity pin gating (materials.py's `connect()`) is
        # driven off `self.graph.blend_mode`; this is the UI that flips it.
        blend_y = sy + sw + 8
        half = (prev_r.width - 16 - 4) // 2
        is_translucent = self.graph.blend_mode == "translucent"
        opaque_btn = pygame.Rect(prev_r.x + 8, blend_y, half, 18)
        translucent_btn = pygame.Rect(opaque_btn.right + 4, blend_y, half, 18)
        pygame.draw.rect(surf, ACCENT if not is_translucent else (40, 44, 54),
                         opaque_btn, border_radius=3)
        pygame.draw.rect(surf, ACCENT if is_translucent else (40, 44, 54),
                         translucent_btn, border_radius=3)
        o_lab = self.editor.font_small.render("Opaque", True, TEXT)
        t_lab = self.editor.font_small.render("Translucent", True, TEXT)
        surf.blit(o_lab, (opaque_btn.x + (opaque_btn.width - o_lab.get_width()) // 2,
                         opaque_btn.y + 2))
        surf.blit(t_lab, (translucent_btn.x + (translucent_btn.width - t_lab.get_width()) // 2,
                         translucent_btn.y + 2))
        self._blend_opaque_rect = opaque_btn
        self._blend_translucent_rect = translucent_btn

        label = ("Output" if self.preview_nid is None
                else NODE_DISPLAY.get(self.graph.nodes.get(self.preview_nid, {}).get("type", ""),
                                     "?"))
        lab = self.editor.font_small.render(f"Preview: {label}", True, TEXT_DIM)
        surf.blit(lab, (prev_r.x + 8, blend_y + 24))
        if self.preview_nid is not None:
            stop = pygame.Rect(prev_r.x + 8, blend_y + 42, prev_r.width - 16, 18)
            pygame.draw.rect(surf, (40, 44, 54), stop, border_radius=3)
            slab = self.editor.font_small.render("Stop Previewing", True, TEXT)
            surf.blit(slab, (stop.x + 6, stop.y + 2))
            self._preview_stop_rect = stop
        else:
            self._preview_stop_rect = None

    def _render_preview(self):
        """Bake a small preview sphere with the current graph (or, while
        isolating a node, that node's own output) -- cheap and only
        re-rendered when the graph or the isolated node changes."""
        import pygame
        eng = self.editor.engine_mod
        size = self.PREVIEW_SIZE
        surf = pygame.Surface((size, size))
        translucent = self.graph.blend_mode == "translucent"
        if translucent:
            _draw_checker_bg(surf, size)
        else:
            surf.fill((29, 31, 37))
        sphere = eng.icosphere(radius=1.0, subdivisions=2)
        entity = eng.Entity("preview", mesh=sphere)
        if self.preview_nid is not None and self.preview_nid in self.graph.nodes:
            colors = self.graph.preview_value(sphere, self.preview_nid)
            sphere.face_colors = colors
        else:
            (sphere.face_colors, sphere.face_roughness, sphere.face_metallic,
             sphere.face_emissive, opacity) = self.graph.evaluate_pbr(sphere)
            if opacity is not None:
                sphere.face_opacity = opacity
            entity.material = self.graph
        mini = eng.Scene(
            light=eng.DirectionalLight(eng.Vec3(-0.5, -0.9, -0.6), ambient=0.42),
            background=(29, 31, 37))
        mini.add(entity)
        cam = eng.Camera(yaw=0.65, pitch=-0.5)
        fwd = cam.forward()
        cam.position = eng.Vec3(-fwd.x, -fwd.y, -fwd.z) * 2.6
        from engine.renderer import Renderer
        Renderer().render(surf, mini, cam)
        return surf

    def _draw_ctx_menu(self, surf) -> None:
        import pygame
        menu = self.ctx_menu
        rows = self._ctx_menu_rows(menu)
        x, y = menu["screen_pos"]
        mp = pygame.mouse.get_pos()
        header = self._ctx_header_rect(menu)
        total = self._ctx_menu_total_rect(menu)
        total_h = total.height
        pygame.draw.rect(surf, (24, 26, 32), (x, y, self.CTX_MENU_W, total_h))
        if menu["kind"] == "add":
            search_r = pygame.Rect(x, y, self.CTX_MENU_W, self.CTX_SEARCH_H)
            pygame.draw.rect(surf, (16, 18, 22), search_r)
            pygame.draw.rect(surf, PANEL_EDGE, search_r, 1)
            txt = menu["search"] or "Search nodes..."
            color = TEXT if menu["search"] else TEXT_DIM
            lab = self.editor.font_small.render(txt, True, color)
            surf.blit(lab, (search_r.x + 6, search_r.y + 4))
            if header is not None:
                hdr = self.editor.font_small.render("Common", True, TEXT_DIM)
                surf.blit(hdr, (header.x + 6, header.y + 2))
        pygame.draw.rect(surf, PANEL_EDGE, (x, y, self.CTX_MENU_W, total_h), 1)
        for label, rect, _payload in rows:
            if rect.collidepoint(mp):
                pygame.draw.rect(surf, HOVER_BG, rect)
            lab = self.editor.font_small.render(label, True, TEXT)
            surf.blit(lab, (rect.x + 8, rect.y + 3))


class ScriptEditorUI:
    """Blueprint Python-script editor: a real multi-line text buffer with a
    line-number gutter, arrow-key/Home/End caret navigation, and an
    in-engine Compile step (engine.blueprint.compile_blueprint) that
    catches SyntaxError, definition-time exceptions, and a missing
    Behavior subclass without ever crashing or hanging the editor.

    Floating window, same chrome as MaterialEditorUI: drag the title bar
    to move it, X/Esc closes (both auto-save, see `close()`). No
    selection/clipboard/undo this run -- typing, Enter/Backspace/Delete,
    arrow + Home/End caret movement, Tab-inserts-4-spaces, and vertical
    scroll-follows-caret are what's needed to write and bug-check a
    Behavior script; those are noted as future work.

    Save policy (all three call `_save`, which just writes `self.lines`
    joined by "\\n" to the asset JSON): Compile always saves (script text +
    the new compile_result, since the result must be persisted); the Save
    button / Ctrl+S persists script text without recompiling (an existing
    compile_result is left as-is and may go stale -- the status strip's
    dirty-flag hint covers that); closing the window auto-saves so edits
    are never silently lost.
    """

    DEFAULT_SIZE = (760, 560)
    TOOLBAR_H = 26
    STATUS_H = 40
    TAB_SPACES = "    "

    def __init__(self, editor: Editor, blueprint):
        self.editor = editor
        self.blueprint = blueprint
        self.lines = blueprint.script.split("\n") if blueprint.script else [""]
        self.caret_row = 0
        self.caret_col = 0
        self.scroll_row = 0
        self.pos = [80, 60]
        self.size = list(self.DEFAULT_SIZE)
        self.minimized = False
        self.drag_title = None    # (grab_dx, grab_dy) while dragging the title bar
        self.dirty = False        # unsaved edits since the last Save/Compile
        self.compile_result = blueprint.compile_result
        self.error_line = (self.compile_result.get("line")
                           if self.compile_result and not self.compile_result.get("ok")
                           else None)

    def close(self) -> None:
        self._save()
        self.editor.script_ui = None

    # ---- geometry (same recompute-from-self.pos/size pattern as MaterialEditorUI) ----
    def rect(self, w, h):
        import pygame
        sw = min(self.size[0], max(420, w - 40))
        sh = min(self.size[1], max(260, h - 40))
        x = min(max(self.pos[0], 0), max(0, w - sw))
        y = min(max(self.pos[1], MENU_H), max(MENU_H, h - sh))
        self.pos = [x, y]
        return pygame.Rect(x, y, sw, sh)

    def outer_rect(self, w, h):
        """The on-screen box -- collapsed to the title bar while minimized."""
        import pygame
        r = self.rect(w, h)
        if self.minimized:
            return pygame.Rect(r.x, r.y, r.width, PANEL_TITLE_H)
        return r

    def content_rect(self, w, h):
        import pygame
        outer = self.rect(w, h)
        return pygame.Rect(outer.x, outer.y + PANEL_TITLE_H, outer.width,
                           max(0, outer.height - PANEL_TITLE_H))

    def _toolbar_rect(self, w, h):
        import pygame
        content = self.content_rect(w, h)
        return pygame.Rect(content.x, content.y, content.width, self.TOOLBAR_H)

    def _status_rect(self, w, h):
        import pygame
        content = self.content_rect(w, h)
        return pygame.Rect(content.x, content.bottom - self.STATUS_H,
                           content.width, self.STATUS_H)

    def _text_area_rect(self, w, h):
        import pygame
        content = self.content_rect(w, h)
        top = self._toolbar_rect(w, h).bottom
        status_y = self._status_rect(w, h).y
        return pygame.Rect(content.x, top, content.width, max(0, status_y - top))

    def _char_w(self) -> int:
        return self.editor.font_small.size("0")[0]

    def _line_h(self) -> int:
        return self.editor.font_small.get_height() + 4

    def _gutter_w(self) -> int:
        digits = max(2, len(str(len(self.lines))))
        return 10 + digits * self._char_w()

    def _gutter_rect(self, w, h):
        import pygame
        area = self._text_area_rect(w, h)
        return pygame.Rect(area.x, area.y, self._gutter_w(), area.height)

    def _code_rect(self, w, h):
        import pygame
        area = self._text_area_rect(w, h)
        gw = self._gutter_w()
        return pygame.Rect(area.x + gw, area.y, max(0, area.width - gw), area.height)

    def _visible_rows(self, w, h) -> int:
        return max(1, self._code_rect(w, h).height // self._line_h())

    def _compile_btn_rect(self, w, h):
        import pygame
        tb = self._toolbar_rect(w, h)
        return pygame.Rect(tb.x + 6, tb.y + 3, 78, 20)

    def _save_btn_rect(self, w, h):
        import pygame
        tb = self._toolbar_rect(w, h)
        cb = self._compile_btn_rect(w, h)
        return pygame.Rect(cb.right + 6, tb.y + 3, 60, 20)

    # ---- buffer editing ----
    def _clamp_caret(self) -> None:
        self.caret_row = max(0, min(self.caret_row, len(self.lines) - 1))
        self.caret_col = max(0, min(self.caret_col, len(self.lines[self.caret_row])))

    def _ensure_caret_visible(self, w, h) -> None:
        visible = self._visible_rows(w, h)
        if self.caret_row < self.scroll_row:
            self.scroll_row = self.caret_row
        elif self.caret_row >= self.scroll_row + visible:
            self.scroll_row = self.caret_row - visible + 1
        self.scroll_row = max(0, min(self.scroll_row, max(0, len(self.lines) - 1)))

    def _insert_text(self, text: str) -> None:
        if not text:
            return
        line = self.lines[self.caret_row]
        self.lines[self.caret_row] = line[:self.caret_col] + text + line[self.caret_col:]
        self.caret_col += len(text)
        self.dirty = True

    def _insert_newline(self) -> None:
        line = self.lines[self.caret_row]
        before, after = line[:self.caret_col], line[self.caret_col:]
        self.lines[self.caret_row:self.caret_row + 1] = [before, after]
        self.caret_row += 1
        self.caret_col = 0
        self.dirty = True

    def _backspace(self) -> None:
        if self.caret_col > 0:
            line = self.lines[self.caret_row]
            self.lines[self.caret_row] = line[:self.caret_col - 1] + line[self.caret_col:]
            self.caret_col -= 1
            self.dirty = True
        elif self.caret_row > 0:
            prev = self.lines[self.caret_row - 1]
            cur = self.lines.pop(self.caret_row)
            self.caret_row -= 1
            self.caret_col = len(prev)
            self.lines[self.caret_row] = prev + cur
            self.dirty = True

    def _delete_forward(self) -> None:
        line = self.lines[self.caret_row]
        if self.caret_col < len(line):
            self.lines[self.caret_row] = line[:self.caret_col] + line[self.caret_col + 1:]
            self.dirty = True
        elif self.caret_row < len(self.lines) - 1:
            nxt = self.lines.pop(self.caret_row + 1)
            self.lines[self.caret_row] = line + nxt
            self.dirty = True

    def _move_left(self) -> None:
        if self.caret_col > 0:
            self.caret_col -= 1
        elif self.caret_row > 0:
            self.caret_row -= 1
            self.caret_col = len(self.lines[self.caret_row])

    def _move_right(self) -> None:
        line = self.lines[self.caret_row]
        if self.caret_col < len(line):
            self.caret_col += 1
        elif self.caret_row < len(self.lines) - 1:
            self.caret_row += 1
            self.caret_col = 0

    def _move_up(self) -> None:
        if self.caret_row > 0:
            self.caret_row -= 1
            self.caret_col = min(self.caret_col, len(self.lines[self.caret_row]))

    def _move_down(self) -> None:
        if self.caret_row < len(self.lines) - 1:
            self.caret_row += 1
            self.caret_col = min(self.caret_col, len(self.lines[self.caret_row]))

    def _caret_from_mouse(self, mp, w, h) -> None:
        code = self._code_rect(w, h)
        line_h = self._line_h()
        char_w = max(1, self._char_w())
        row = self.scroll_row + max(0, mp[1] - code.y) // line_h
        row = max(0, min(row, len(self.lines) - 1))
        col = round((mp[0] - code.x) / char_w)
        col = max(0, min(col, len(self.lines[row])))
        self.caret_row, self.caret_col = row, col

    # ---- persistence + compile ----
    def _source(self) -> str:
        return "\n".join(self.lines)

    def _save(self) -> None:
        self.blueprint.script = self._source()
        self.blueprint.save()
        self.dirty = False

    def _compile(self) -> None:
        from engine import console_log
        source = self._source()
        result = self.editor.engine_mod.compile_blueprint(source, self.blueprint.name)
        self.compile_result = result
        self.blueprint.script = source
        self.blueprint.compile_result = result
        self.blueprint.save()
        self.dirty = False
        self.error_line = result.get("line") if not result.get("ok") else None
        if self.error_line:
            row = max(0, min(len(self.lines) - 1, self.error_line - 1))
            self.caret_row, self.caret_col = row, len(self.lines[row])
        if result.get("ok"):
            console_log.log_info(
                f"blueprint '{self.blueprint.name}' compiled OK "
                f"-- Behavior subclass '{result['class_name']}'")
        else:
            loc = f" (line {result['line']})" if result.get("line") else ""
            console_log.log_error(
                f"blueprint '{self.blueprint.name}' {result['stage']} "
                f"error{loc}: {result['message']}")

    def _status_text(self):
        r = self.compile_result
        if r is None:
            return "not compiled yet", TEXT_DIM
        if r.get("ok"):
            return f"compiled OK -- Behavior subclass '{r['class_name']}'", (140, 220, 150)
        if r.get("line") and r.get("col"):
            loc = f" (line {r['line']}, col {r['col']})"
        elif r.get("line"):
            loc = f" (line {r['line']})"
        else:
            loc = ""
        return f"{r['stage']} error{loc}: {r['message']}", (230, 140, 140)

    # ---- interaction ----
    def update(self, engine, dt: float) -> None:
        import pygame
        inp = engine.input
        mp = inp.mouse_pos
        w, h = engine.screen.get_size()
        outer = self.outer_rect(w, h)
        title_bar = pygame.Rect(outer.x, outer.y, outer.width, PANEL_TITLE_H)
        close = pygame.Rect(outer.right - 24, outer.y + 2, 16, 16)
        minimize = pygame.Rect(outer.right - 44, outer.y + 2, 16, 16)

        ctrl = inp.held(pygame.K_LCTRL) or inp.held(pygame.K_RCTRL)
        if ctrl and inp.pressed(pygame.K_s):
            self._save()
        compiled_this_step = False
        if ctrl and inp.pressed(pygame.K_RETURN):
            self._compile()
            compiled_this_step = True

        if inp.mouse_button_pressed(1):
            if close.collidepoint(mp):
                self.close()
                return
            if minimize.collidepoint(mp):
                self.minimized = not self.minimized
                return
            if not self.minimized:
                if self._compile_btn_rect(w, h).collidepoint(mp):
                    self._compile()
                elif self._save_btn_rect(w, h).collidepoint(mp):
                    self._save()
                elif title_bar.collidepoint(mp):
                    self.drag_title = (mp[0] - outer.x, mp[1] - outer.y)
                elif self._code_rect(w, h).collidepoint(mp):
                    self._caret_from_mouse(mp, w, h)

        if inp.mouse_held(1):
            if self.drag_title is not None:
                dx, dy = self.drag_title
                self.pos = [mp[0] - dx, mp[1] - dy]
        else:
            self.drag_title = None

        if self.minimized:
            return

        for ch in inp.take_text():
            self._insert_text(ch)
        if not compiled_this_step and (inp.pressed(pygame.K_RETURN)
                                       or inp.pressed(pygame.K_KP_ENTER)):
            self._insert_newline()
        if inp.pressed(pygame.K_BACKSPACE):
            self._backspace()
        if inp.pressed(pygame.K_DELETE):
            self._delete_forward()
        if inp.pressed(pygame.K_TAB):
            self._insert_text(self.TAB_SPACES)
        if inp.pressed(pygame.K_LEFT):
            self._move_left()
        if inp.pressed(pygame.K_RIGHT):
            self._move_right()
        if inp.pressed(pygame.K_UP):
            self._move_up()
        if inp.pressed(pygame.K_DOWN):
            self._move_down()
        if inp.pressed(pygame.K_HOME):
            self.caret_col = 0
        if inp.pressed(pygame.K_END):
            self.caret_col = len(self.lines[self.caret_row])
        self._clamp_caret()
        self._ensure_caret_visible(w, h)
        if inp.wheel and self._code_rect(w, h).collidepoint(mp):
            max_scroll = max(0, len(self.lines) - 1)
            self.scroll_row = max(0, min(self.scroll_row - int(inp.wheel) * 3, max_scroll))

    # ---- drawing ----
    def draw(self, surf) -> None:
        import pygame
        w, h = surf.get_size()
        outer = self.outer_rect(w, h)
        pygame.draw.rect(surf, (18, 20, 25), outer)
        pygame.draw.rect(surf, PANEL_EDGE, outer, 1)
        title_bar = pygame.Rect(outer.x, outer.y, outer.width, PANEL_TITLE_H)
        pygame.draw.rect(surf, (30, 33, 40), title_bar)
        pygame.draw.line(surf, PANEL_EDGE, (outer.x, outer.y + PANEL_TITLE_H),
                         (outer.right, outer.y + PANEL_TITLE_H))
        star = " *" if self.dirty else ""
        title = self.editor.font_small.render(
            f"Script — {self.blueprint.name}{star}   "
            "(Ctrl+Enter compile, Ctrl+S save, Esc close)", True, TEXT)
        surf.blit(title, (outer.x + 8, outer.y + 3))
        close = pygame.Rect(outer.right - 24, outer.y + 2, 16, 16)
        pygame.draw.rect(surf, (60, 34, 34), close, border_radius=3)
        x_lab = self.editor.font_small.render("X", True, (230, 160, 160))
        surf.blit(x_lab, (close.x + 4, close.y + 1))
        minimize = pygame.Rect(outer.right - 44, outer.y + 2, 16, 16)
        pygame.draw.rect(surf, (40, 44, 54), minimize, border_radius=3)
        m_lab = self.editor.font_small.render("-", True, TEXT)
        surf.blit(m_lab, (minimize.x + 5, minimize.y - 1))
        if self.minimized:
            return

        mp = pygame.mouse.get_pos()

        tb = self._toolbar_rect(w, h)
        pygame.draw.rect(surf, (26, 28, 34), tb)
        pygame.draw.line(surf, PANEL_EDGE, (tb.x, tb.bottom), (tb.right, tb.bottom))
        cb = self._compile_btn_rect(w, h)
        pygame.draw.rect(surf, HOVER_BG if cb.collidepoint(mp) else (33, 46, 36), cb,
                         border_radius=4)
        pygame.draw.rect(surf, BLUEPRINT_TILE_EDGE, cb, 1, border_radius=4)
        clab = self.editor.font_small.render("Compile", True, (150, 220, 165))
        surf.blit(clab, (cb.x + (cb.width - clab.get_width()) // 2, cb.y + 4))
        sb = self._save_btn_rect(w, h)
        pygame.draw.rect(surf, HOVER_BG if sb.collidepoint(mp) else (33, 36, 44), sb,
                         border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, sb, 1, border_radius=4)
        slab = self.editor.font_small.render("Save", True, ACCENT)
        surf.blit(slab, (sb.x + (sb.width - slab.get_width()) // 2, sb.y + 4))

        gutter = self._gutter_rect(w, h)
        code = self._code_rect(w, h)
        pygame.draw.rect(surf, (24, 26, 31), gutter)
        pygame.draw.line(surf, PANEL_EDGE, (gutter.right, code.y), (gutter.right, code.bottom))
        line_h = self._line_h()
        visible = self._visible_rows(w, h)
        clip = surf.get_clip()
        surf.set_clip(code.union(gutter))
        for i in range(self.scroll_row, min(len(self.lines), self.scroll_row + visible + 1)):
            y = code.y + (i - self.scroll_row) * line_h
            line_no = i + 1
            is_caret_line = i == self.caret_row
            is_error_line = self.error_line is not None and line_no == self.error_line
            if is_error_line:
                pygame.draw.rect(surf, (58, 28, 30), (code.x, y, code.width, line_h))
                pygame.draw.rect(surf, (58, 28, 30), (gutter.x, y, gutter.width, line_h))
            elif is_caret_line:
                pygame.draw.rect(surf, (30, 33, 42), (code.x, y, code.width, line_h))
            num_color = (220, 130, 130) if is_error_line else (TEXT if is_caret_line else TEXT_DIM)
            num = self.editor.font_small.render(str(line_no), True, num_color)
            surf.blit(num, (gutter.right - 6 - num.get_width(), y + 2))
            text_surf = self.editor.font_small.render(self.lines[i], True, TEXT)
            surf.blit(text_surf, (code.x + 4, y + 2))
        surf.set_clip(clip)

        if self.scroll_row <= self.caret_row < self.scroll_row + visible + 1:
            cy = code.y + (self.caret_row - self.scroll_row) * line_h
            cx = code.x + 4 + self._char_w() * self.caret_col
            pygame.draw.line(surf, ACCENT, (cx, cy + 1), (cx, cy + line_h - 3), 2)

        sr = self._status_rect(w, h)
        pygame.draw.rect(surf, (22, 24, 29), sr)
        pygame.draw.line(surf, PANEL_EDGE, (sr.x, sr.y), (sr.right, sr.y))
        msg, color = self._status_text()
        msg_surf = self.editor.font_small.render(msg[:120], True, color)
        surf.blit(msg_surf, (sr.x + 8, sr.y + 6))
        if self.dirty:
            edited = self.editor.font_small.render("(edited since last save)", True, TEXT_DIM)
            surf.blit(edited, (sr.right - edited.get_width() - 8, sr.y + 6))


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
    parser.add_argument("--api", choices=["auto", "cpu", "gl", "dx12", "vulkan"], default=None,
                        help="force a rendering backend (default: dx12, or "
                             "Settings > Graphics API); CPU is opt-in only")
    parser.add_argument("--gpu", action="store_true",
                        help="alias for --api gl (force the OpenGL/moderngl renderer)")
    parser.add_argument("--cpu", action="store_true",
                        help="alias for --api cpu (force the software renderer)")
    parser.add_argument("--settings-path", default=None,
                        help="override settings.json location (else "
                             "PYENGINE_SETTINGS env var, else repo-root "
                             "settings.json); use for tests/headless runs so "
                             "they don't touch the user's real settings")
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import engine
    import pygame

    settings_path = args.settings_path or default_settings_path()
    settings = load_settings(settings_path)
    width = args.width if args.width is not None else settings.get("width", 1440)
    height = args.height if args.height is not None else settings.get("height", 810)
    pixel_scale = (args.pixel_scale if args.pixel_scale is not None
                   else settings.get("pixel_scale", 4))
    max_fps = settings.get("max_fps", 120)
    fullscreen = bool(settings.get("fullscreen", False)) and not args.headless

    api_mode = "dx12"
    settings_api = settings.get("api")
    if settings_api in ("auto", "cpu", "gl", "dx12", "vulkan"):
        api_mode = settings_api
    if args.api:
        api_mode = args.api
    elif args.gpu:
        api_mode = "gl"
    elif args.cpu:
        api_mode = "cpu"
    if args.headless:
        api_mode = "cpu"  # the SDL dummy driver has no GL surface / wgpu window to attach to

    eng = engine.Engine(width, height, title="PyEngine Editor", max_fps=max_fps,
                        api=api_mode, fullscreen=fullscreen)
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

    editor = Editor(engine, eng, scene, camera, lib, args.scene,
                   settings_path=settings_path)
    editor._apply_layout_settings(settings)
    if settings_api in ("auto", "cpu", "gl", "dx12", "vulkan"):
        # show the saved preference even if it didn't match what actually ran
        editor.api_pref = settings_api

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
    import time as _time
    _rebuilt = eng.tracer.refresh(scene)
    if _rebuilt:
        engine.console_log.log_info(
            f"Baking lighting ({eng.tracer.occluder_triangle_count()} "
            f"occluder triangles)...")
        _t0 = _time.perf_counter()
    eng.renderer.render(pygame.Surface((320, 180)), scene, camera, eng.tracer)
    if _rebuilt:
        engine.console_log.log_info(
            f"Lighting baked in {_time.perf_counter() - _t0:.1f}s")

    eng.loading_step("opening world", 0.95)
    eng.esc_handler = editor.handle_escape
    eng.hud_text = ("RMB: look/fly (WASD/QE/Space/Ctrl, wheel=speed) | LMB: select/gizmo/panels | "
                    "W/E/R gizmo mode | M material | L flashlight | C collision | F focus | "
                    "Ctrl+D dup | Alt+drag dup | Del delete | End floor snap | Shift+drag mesh snap | "
                    "Ctrl+S save | F1/F2 shading | H hud | Esc back/quit")
    eng.run(scene, camera, max_frames=args.frames, screenshot_path=args.screenshot,
            overlay=editor.draw)


if __name__ == "__main__":
    main()
