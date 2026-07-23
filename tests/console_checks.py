"""Judge checks: engine console log/bus + dockable Console panel +
collapsible side toolbar.

Covers engine/console_log.py (bounded ring buffer, levels), its real
producers (FBX import via `_do_import`, blueprint compile via
ScriptEditorUI._compile), the Console panel (dockable like outliner/
details/browser, Window-menu toggle, scroll with auto-follow, level
colors), and the new side toolbar (collapse/expand, persistence, houses
the Console toggle, must not conflict with marquee/docktab/viewport).

Per the project's hard rule for UI/interaction tests, every genuinely
interactive check (menu clicks, side-toolbar clicks, panel toggles) drives
the REAL event path (eng.input.process + editor.update), patching
pygame.mouse.get_pos/get_pressed and pygame.key.get_pressed at the OS
boundary -- the idiom tests/docktab_checks.py and tests/marquee_checks.py
established. The FBX fixture builder is copied from tests/import_checks.py
(same minimal binary-FBX construction browser_checks.py/smoke_test.py also
each keep their own copy of -- no shared test-utils module exists in this
codebase)."""
import json
import os
import struct
import sys
import tempfile
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np
import pygame

import engine
from engine import console_log
from editor import Editor, EditorBehavior, ScriptEditorUI, build_starter_scene

# Guard against the real settings.json (several actions below save settings).
REAL_SETTINGS = os.path.join(WT, "settings.json")
_real_before = (open(REAL_SETTINGS, "rb").read()
               if os.path.exists(REAL_SETTINGS) else None)
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_console_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)
TMP = tempfile.gettempdir()
OUT = os.path.join(TMP, "judge_console.png")

# assets/folders.json backup -- the import producer test writes a real
# asset into assets/ (AssetLibrary always points there), cleaned up in `finally`
FOLDERS_JSON = os.path.join(WT, "assets", "folders.json")
_folders_before = (open(FOLDERS_JSON, encoding="utf-8").read()
                   if os.path.exists(FOLDERS_JSON) else None)

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
W, H = eng.screen.get_size()


def new_editor():
    return Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
                 settings_path=TEST_SETTINGS)


class FakeKeys:
    def __init__(self, held=()):
        self._held = set(held)
    def __getitem__(self, key):
        return key in self._held


def step(ed, events):
    eng.input.process(events)
    ed.update(eng, 1 / 60)
    eng.input.consume_edges()


def click(ed, pos):
    """A plain press-then-release at `pos`, driven through the real event
    path (mirrors docktab_checks.py's `click`)."""
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step(ed, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)])
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
        step(ed, [])


# ----------------------------------------------------------------------------
# FBX fixture builder (same minimal binary-FBX construction tests/import_checks.py,
# browser_checks.py and smoke_test.py each keep their own copy of)
# ----------------------------------------------------------------------------
def _enc_prop(v):
    if isinstance(v, np.ndarray) and v.dtype == np.float64:
        return b"d" + struct.pack("<III", len(v), 0, len(v) * 8) + v.tobytes()
    if isinstance(v, np.ndarray) and v.dtype == np.int32:
        return b"i" + struct.pack("<III", len(v), 0, len(v) * 4) + v.tobytes()
    if isinstance(v, str):
        raw = v.encode()
        return b"S" + struct.pack("<I", len(raw)) + raw
    if isinstance(v, int):
        return b"L" + struct.pack("<q", v)
    raise TypeError(v)


def _build_node(node, start):
    name, props, children = node
    prop_data = b"".join(_enc_prop(p) for p in props)
    pos = start + 13 + len(name) + len(prop_data)
    child_data = b""
    if children:
        for ch in children:
            cb = _build_node(ch, pos)
            child_data += cb
            pos += len(cb)
        child_data += b"\x00" * 13
        pos += 13
    header = struct.pack("<IIIB", pos, len(props), len(prop_data), len(name))
    return header + name.encode() + prop_data + child_data


def build_box_fbx(path, hx, hy, hz, gid=9101, model_id=9102):
    verts = np.array(
        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
         [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
        dtype=np.float64).reshape(-1)
    quads = [(4, 5, 6, 7), (1, 0, 3, 2), (0, 4, 7, 3), (5, 1, 2, 6), (7, 6, 2, 3), (0, 1, 5, 4)]
    pvi = []
    for q in quads:
        pvi += [q[0], q[1], q[2], -(q[3] + 1)]
    geometry = ("Geometry", [gid, "Geometry::box", "Mesh"], [
        ("Vertices", [verts], []),
        ("PolygonVertexIndex", [np.array(pvi, dtype=np.int32)], []),
    ])
    objects = ("Objects", [], [geometry, ("Model", [model_id, "Model::box", "Mesh"], [])])
    connections = ("Connections", [], [("C", ["OO", gid, model_id], [])])
    header = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400)
    body = _build_node(objects, len(header))
    body += _build_node(connections, len(header) + len(body))
    with open(path, "wb") as fh:
        fh.write(header + body + b"\x00" * 13)


created_asset_name = None
try:
    # ========================================================================
    # 1. ring buffer bounds + levels
    # ========================================================================
    console_log.reset()
    for i in range(console_log.MAX_ENTRIES + 50):
        console_log.log_info(f"msg {i}")
    entries = list(console_log.get_log().entries)
    assert len(entries) == console_log.MAX_ENTRIES
    assert entries[-1]["text"] == f"msg {console_log.MAX_ENTRIES + 49}"
    assert entries[0]["text"] == "msg 50"  # oldest 50 evicted
    print(f"ring-buffer bound OK: capped at {console_log.MAX_ENTRIES}, oldest evicted")

    console_log.reset()
    console_log.log_info("an info line")
    console_log.log_warn("a warn line")
    console_log.log_error("an error line")
    levels = [e["level"] for e in console_log.get_log().entries]
    assert levels == ["info", "warn", "error"]
    assert all(isinstance(e["time"], float) for e in console_log.get_log().entries)
    print("levels + timestamped entries OK:", levels)

    # ========================================================================
    # 2. import producer actually appends (drive _do_import for real)
    # ========================================================================
    fbx_path = os.path.join(TMP, "judge_console_cube.fbx")
    build_box_fbx(fbx_path, 1, 1, 1)
    ed = new_editor()
    console_log.reset()
    ed._do_import(fbx_path, name="Judge Console Cube", folder=None,
                 scale=1.0, up_axis="y", generate_lods=False)
    created_asset_name = "Judge Console Cube"
    texts = [e["text"] for e in console_log.get_log().entries]
    assert any("Importing" in t for t in texts), texts
    assert any("imported 'Judge Console Cube'" in t and "faces" in t for t in texts), texts
    print("import producer OK (_do_import, real path):", texts[:2])

    console_log.reset()
    ed._do_import(os.path.join(TMP, "judge_console_missing.fbx"))
    entries = [(e["level"], e["text"]) for e in console_log.get_log().entries]
    assert any(lvl == "error" and "import failed" in t for lvl, t in entries), entries
    print("import failure producer OK:", entries)

    # ========================================================================
    # 3. blueprint compile producer (scratch asset -- never touches the real
    # assets/blueprints/ dir, per this task's hard constraint)
    # ========================================================================
    scratch_bp_path = os.path.join(TMP, "judge_console_blueprint.json")
    bp = engine.BlueprintAsset(
        {"name": "Judge Console Blueprint", "components": [],
         "script": engine.DEFAULT_BLUEPRINT_SCRIPT, "compile_result": None},
        scratch_bp_path)
    console_log.reset()
    ui = ScriptEditorUI(ed, bp)
    ui.lines = ["def broken(:"]
    ui._compile()
    entries = [(e["level"], e["text"]) for e in console_log.get_log().entries]
    assert any(lvl == "error" and "error" in t for lvl, t in entries), entries
    print("blueprint compile error producer OK:", entries)

    console_log.reset()
    ui.lines = engine.DEFAULT_BLUEPRINT_SCRIPT.split("\n")
    ui._compile()
    entries = [(e["level"], e["text"]) for e in console_log.get_log().entries]
    assert any(lvl == "info" and "compiled OK" in t for lvl, t in entries), entries
    print("blueprint compile OK producer OK:", entries)
    if os.path.exists(scratch_bp_path):
        os.remove(scratch_bp_path)

    # ========================================================================
    # 4. console panel toggles via the Window menu (real clicks)
    # ========================================================================
    ed2 = new_editor()
    assert ed2.panel_visible["console"] is True
    title_rects = ed2._menu_title_rects(W)
    click(ed2, title_rects["Window"].center)
    assert ed2.open_menu == "Window"
    _drop, rows = ed2._dropdown_geom("Window", W)
    console_row = next(r for label, r, _a, _e in rows if label == "Console")
    click(ed2, console_row.center)
    assert ed2.open_menu is None
    assert ed2.panel_visible["console"] is False
    print("Window menu > Console toggle OK (real clicks)")
    # toggle back on for the rendering checks below
    click(ed2, title_rects["Window"].center)
    _drop, rows = ed2._dropdown_geom("Window", W)
    console_row = next(r for label, r, _a, _e in rows if label == "Console")
    click(ed2, console_row.center)
    assert ed2.panel_visible["console"] is True

    # ========================================================================
    # 5. console panel renders + auto-scrolls + scroll-anchors when scrolled up
    # ========================================================================
    console_log.reset()
    for i in range(40):
        console_log.log_info(f"line {i}")
    lay = ed2._layout(W, H)
    content = ed2._panel_content_rect("console", lay)
    assert content is not None, "console panel should have a rect (docked, visible)"
    surf = pygame.Surface((W, H))
    ed2._draw_console(surf, content)
    assert ed2.console_scroll == 0, "pinned to latest by default"
    ed2.console_scroll = 5
    ed2._draw_console(surf, content)
    console_log.log_info("new while scrolled up")
    ed2._draw_console(surf, content)
    assert ed2.console_scroll == 6, (
        "a scrolled-up viewport must advance by exactly the new-entry count, "
        "not get yanked back to the tail", ed2.console_scroll)
    print("console panel draw + scroll + auto-follow-unless-scrolled-up OK")

    # wheel-scroll over the console panel via the real wheel-routing path
    ed2.console_scroll = 0
    console_rect = lay["panels"]["console"]
    with um.patch.object(pygame.mouse, "get_pos", return_value=console_rect.center):
        eng.input.wheel = 1.0
        ed2.update(eng, 1 / 60)
        eng.input.wheel = 0.0
    assert ed2.console_scroll > 0, "wheel-up over the console panel should scroll into history"
    print("console wheel-scroll routing OK (real path)")

    # ========================================================================
    # 6. side toolbar: collapse/expand (real click) + settings round-trip
    # ========================================================================
    def collapse_click_pos(ed):
        # the collapse button's top-left corner sits at a fixed (rect.x+3,
        # rect.y+3) regardless of collapsed/expanded state -- only its WIDTH
        # changes, so a `.center` click position goes stale across a toggle;
        # a small fixed offset from the toolbar's origin stays valid either way
        r = ed._side_toolbar_rect(W, H)
        return (r.x + 8, r.y + 8)

    ed3 = new_editor()
    assert ed3.side_toolbar_collapsed is False
    tb_rect0 = ed3._side_toolbar_rect(W, H)
    click(ed3, collapse_click_pos(ed3))
    assert ed3.side_toolbar_collapsed is True
    assert ed3._side_toolbar_rect(W, H).width < tb_rect0.width
    with open(TEST_SETTINGS, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["side_toolbar_collapsed"] is True
    ed4 = new_editor()
    ed4._apply_layout_settings(saved)
    assert ed4.side_toolbar_collapsed is True
    print("side toolbar collapse/expand + settings round-trip OK (real click)")
    click(ed3, collapse_click_pos(ed3))  # expand back for the checks below
    assert ed3.side_toolbar_collapsed is False

    # ========================================================================
    # 7. side toolbar houses a Console toggle button (real click)
    # ========================================================================
    tb_rect = ed3._side_toolbar_rect(W, H)
    console_btn_rect = ed3._side_toolbar_button_rects(tb_rect)[0][1]
    was_visible = ed3.panel_visible["console"]
    click(ed3, console_btn_rect.center)
    assert ed3.panel_visible["console"] == (not was_visible)
    click(ed3, console_btn_rect.center)
    assert ed3.panel_visible["console"] == was_visible
    print("side toolbar 'Console' button toggles the panel OK (real click)")

    # ========================================================================
    # 8. a click on the side toolbar must NOT start a marquee or place/
    # deselect in the viewport (the task's explicit hard requirement)
    # ========================================================================
    ed3.selected = None
    ed3.marquee = None
    click(ed3, (10, 200))  # empty strip area, below the buttons
    assert ed3.marquee is None, "side-toolbar click must never start a marquee"
    assert ed3.selected is None, "side-toolbar click must never touch viewport selection"
    assert ed3.over_ui((10, 200)) is True
    print("side-toolbar click never starts a marquee / falls through to the viewport OK")

    # ========================================================================
    # 9. every side-toolbar + console control clicked doesn't crash
    # ========================================================================
    for pos in (collapse_click_pos(ed3), console_btn_rect.center, (5, 5), (tb_rect.width - 1, 400),
               console_rect.center, (console_rect.right - 5, console_rect.bottom - 5)):
        click(ed3, pos)
        ed3.draw(eng)
    print("every side-toolbar + console control clicked -- no crash")

    # ========================================================================
    # 10. existing panels/docktab still work alongside the new strip
    # ========================================================================
    ed5 = new_editor()
    lay5 = ed5._layout(W, H)
    outliner_rect = lay5["panels"]["outliner"]
    details_rect = lay5["panels"]["details"]
    ed5._begin_panel_drag("details", (details_rect.x + 5, details_rect.y + 5), details_rect)
    ed5._finish_panel_drag((outliner_rect.x + 10, outliner_rect.y + 5), W, H)
    side, group = ed5._group_for_pid("details")
    assert side == "right" and group["ids"] == ["outliner", "details"], group
    ed5._toggle_minimize("browser")
    assert ed5.panel_minimized["browser"] is True
    lay5b = ed5._layout(W, H)
    assert lay5b["panels"]["browser"].height == 18  # PANEL_TITLE_H, collapsed
    print("existing docktab/minimize behavior unaffected by the side toolbar OK")

    # ---- final screenshot with everything visible ----
    fly = engine.behaviors.FlyController(camera, look_buttons=(3,),
                                         look_guard=lambda p: not ed5.over_ui(p))
    ed5.flashlight = None
    scene.add(engine.Entity("__camera_console").add_behavior(fly))
    scene.add(engine.Entity("__editor_console").add_behavior(EditorBehavior(ed5)))
    eng.esc_handler = ed5.handle_escape
    eng.run(scene, camera, max_frames=20, screenshot_path=OUT, overlay=ed5.draw)
    print("screenshot saved:", OUT)

    print("ALL CONSOLE CHECKS PASSED")
finally:
    if created_asset_name is not None:
        try:
            lib2 = engine.AssetLibrary(os.path.join(WT, "assets"))
            asset = lib2.by_name.get(created_asset_name)
            if asset is not None and os.path.exists(asset.path):
                os.remove(asset.path)
        except Exception:
            pass
    if _folders_before is not None:
        with open(FOLDERS_JSON, "w", encoding="utf-8") as f:
            f.write(_folders_before)
    elif os.path.exists(FOLDERS_JSON):
        os.remove(FOLDERS_JSON)
    _real_after = (open(REAL_SETTINGS, "rb").read()
                  if os.path.exists(REAL_SETTINGS) else None)
    assert _real_after == _real_before, (
        "console_checks touched the real settings.json -- an Editor() in "
        "this suite is missing settings_path=TEST_SETTINGS")
    print("no-pollution guard OK: real settings.json and assets/ untouched")
