"""Judge checks: Blueprint asset + Python script editor + in-engine
compile/bug-check (run 1 of 2 -- see engine/blueprint.py and
engine/assets.py's BlueprintAsset for the schema/design notes).

Covers: asset round-trip (script text verbatim, incl. newlines/tabs/
indentation, plus compile_result); the content browser's "+ Blueprint"
button and tile-click-opens-editor, driven through the REAL click router
(_route_panel_click, same code path a mouse click takes); ScriptEditorUI's
full multi-line editing vocabulary (typing, Enter, Backspace, Tab-indent,
arrow-key caret movement) via REAL pygame KEYDOWN events through
eng.input.process + editor.update, per the project's hard rule for
UI/interaction tests (the idiom is tests/docktab_checks.py's `step`/`click`
helpers); all four compile_blueprint outcomes (ok, SyntaxError, exec-time
NameError, no-Behavior-subclass); every control in the script window
clicked without crashing (this project has a documented ctx-menu-crash
history -- a UI click must never raise); and the has_mesh() AttributeError
this run's editor.py integration had to guard against (BlueprintAsset has
no `.data`, unlike AssetDef/its Export-button gating assumed).

Isolation: unlike some older suites here that mutate the real assets/
folder and restore it in `finally` (e.g. browser_checks.py's folders.json
dance), this suite is told NOT to touch assets/folders.json (or
gat.json/gat.npz) at all -- so it works against a TEMP COPY of the whole
assets/ tree instead of the real one. Zero risk even if the process is
killed mid-run; nothing here ever opens a real repo file for writing.
"""
import os
import shutil
import sys
import tempfile
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import pygame

import engine
from editor import Editor, EditorBehavior, ScriptEditorUI, build_starter_scene

REAL_SETTINGS = os.path.join(WT, "settings.json")
_real_settings_before = (open(REAL_SETTINGS, "rb").read()
                         if os.path.exists(REAL_SETTINGS) else None)
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_blueprint_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

# ---- isolated temp copy of assets/ -- see module docstring ----
TMP_ASSETS_ROOT = tempfile.mkdtemp(prefix="pyengine_blueprint_checks_")
ASSETS_DIR = os.path.join(TMP_ASSETS_ROOT, "assets")
shutil.copytree(os.path.join(WT, "assets"), ASSETS_DIR)
assert ASSETS_DIR != os.path.join(WT, "assets"), "must never point at the real assets dir"

eng = engine.Engine(1280, 800, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(ASSETS_DIR)
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
               settings_path=TEST_SETTINGS)
W, H = eng.screen.get_size()


class FakeKeys:
    """pygame.key.get_pressed() stand-in -- held-key state isn't exercised
    by any check here (only single KEYDOWN presses), this just needs to
    never IndexError on pygame's huge SDLK_* values for special keys."""

    def __init__(self, held=()):
        self._held = set(held)

    def __getitem__(self, key):
        return key in self._held


_keys_patch = um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys())
_keys_patch.start()


def step(events):
    eng.input.process(events)
    editor.update(eng, 1 / 60)
    eng.input.consume_edges()


def click(pos):
    """A plain press-then-release at `pos`, through the real event path --
    same idiom as tests/docktab_checks.py's `click`."""
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)])
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
        step([])


def key(k, unicode="") -> None:
    """One real KEYDOWN event through eng.input.process + editor.update,
    consumed the same way any engine hotkey/text-entry field does."""
    step([pygame.event.Event(pygame.KEYDOWN, key=k, unicode=unicode, mod=0)])


def type_text(s: str) -> None:
    for ch in s:
        key(pygame.K_a, unicode=ch)  # the physical key doesn't matter, only e.unicode


try:
    # ========================================================================
    # 1. asset round-trip: script text verbatim (incl. newlines/tabs/blank
    #    lines/indentation) + compile_result, through save -> reload
    # ========================================================================
    bp = lib.new_blueprint("RoundTrip BP")
    assert bp.name in lib.blueprint_by_name
    assert os.path.exists(bp.path)
    assert bp.category == "blueprints"
    assert bp.components == []
    tricky_script = (
        "class Weird(Behavior):\n"
        "    def update(self, entity, dt, engine):\n"
        "\tpass  # a literal tab-indented line\n"
        "\n"
        "    # trailing blank line above, and this comment, are part of the file\n"
    )
    bp.script = tricky_script
    bp.compile_result = {"ok": False, "stage": "syntax", "message": "x",
                         "line": 3, "col": 1}
    bp.save()
    lib.reload()
    bp_reloaded = lib.blueprint_by_name["RoundTrip BP"]
    assert bp_reloaded.script == tricky_script, repr(bp_reloaded.script)
    assert bp_reloaded.compile_result == {"ok": False, "stage": "syntax",
                                          "message": "x", "line": 3, "col": 1}
    print("asset round-trip OK: script text verbatim (tabs/blank lines/indentation) "
         "+ compile_result survive save -> reload")

    # ========================================================================
    # 2. folder tree participation (blueprints, unlike materials, are filed
    #    into folders like a regular AssetDef)
    # ========================================================================
    folder = lib.create_folder("Scripts", None)
    lib.set_asset_folder("RoundTrip BP", folder)
    lib.save_folders()
    lib.reload()
    assert lib.folder_of.get("RoundTrip BP") == folder
    assert any(b.name == "RoundTrip BP" for b in lib.blueprints_in(folder))
    editor.selected_folder = folder
    assert any(t.name == "RoundTrip BP" for t in editor._tiles_in(folder)), \
        "blueprint must appear in the content-browser grid for its folder"
    editor.selected_folder = None
    print("folder-tree participation OK: blueprint filed + retrieved via blueprints_in/_tiles_in")

    # ========================================================================
    # 3. "+ Blueprint" topbar button, through the REAL click router
    #    (_route_panel_click -- the same code a mouse click takes)
    # ========================================================================
    layout = editor._layout(W, H)
    content = editor._panel_content_rect("browser", layout)
    blay = editor._browser_layout(content)
    nbb = editor._new_blueprint_btn_rect(blay["topbar"])
    assert editor.script_ui is None
    click((nbb.centerx, nbb.centery))
    assert editor.script_ui is not None, "+ Blueprint click must create + open a script editor"
    new_bp = editor.script_ui.blueprint
    assert new_bp.name in lib.blueprint_by_name
    assert os.path.exists(new_bp.path)
    editor.draw(eng)  # must not crash with the window open
    print(f"+ Blueprint button OK: created '{new_bp.name}', ScriptEditorUI open, draw() clean")

    # ========================================================================
    # 4. regression guard: selecting a blueprint tile must not crash the
    #    has_mesh() Export-button gating (BlueprintAsset has no .data,
    #    unlike AssetDef -- this WAS an AttributeError before the fix).
    #    Close the script editor first so the click actually reaches
    #    _route_panel_click's export branch (while open, script_ui captures
    #    all input) -- the draw() call below is what exercises the
    #    topbar's `exportable` calc either way, every frame it's visible.
    # ========================================================================
    editor.selected_asset = new_bp
    editor.script_ui.close()
    editor.draw(eng)  # topbar's `exportable` calc runs on every draw frame
    exp_rect = editor._export_btn_rect(blay["topbar"])
    click((exp_rect.centerx, exp_rect.centery))  # routes through _route_panel_click now
    assert editor.status[0] == "" or "export" not in editor.status[0].lower(), (
        "clicking Export on a selected blueprint must no-op, not attempt an export")
    editor.script_ui = ScriptEditorUI(editor, new_bp)  # reopen for the editing checks below
    print("has_mesh() regression guard OK: blueprint selected, Export click + draw() both clean")

    # ========================================================================
    # 5. multi-line editing via REAL KEYDOWN events: typing, Enter, Tab
    #    indent, Backspace, arrow-key + Home/End caret movement
    # ========================================================================
    sui = editor.script_ui
    sui.lines = [""]
    sui.caret_row = sui.caret_col = 0
    type_text("class Foo(Behavior):")
    assert sui.lines == ["class Foo(Behavior):"], sui.lines
    assert sui.caret_row == 0 and sui.caret_col == len(sui.lines[0])

    key(pygame.K_RETURN, unicode="\r")
    type_text("def update(self, entity, dt, engine):")
    key(pygame.K_RETURN, unicode="\r")
    key(pygame.K_TAB)
    type_text("pass")
    assert sui.lines == [
        "class Foo(Behavior):",
        "def update(self, entity, dt, engine):",
        "    pass",
    ], sui.lines
    print("typing + Enter (newline split) + Tab (4-space indent) OK:", sui.lines)

    # Home / End
    sui.caret_row, sui.caret_col = 0, 3
    key(pygame.K_END)
    assert sui.caret_col == len(sui.lines[0])
    key(pygame.K_HOME)
    assert sui.caret_col == 0
    print("Home/End caret movement OK")

    # arrow keys: full traversal incl. wrapping across line boundaries
    sui.caret_row, sui.caret_col = 0, 0
    key(pygame.K_DOWN)
    key(pygame.K_DOWN)
    assert sui.caret_row == 2
    key(pygame.K_UP)
    assert sui.caret_row == 1
    sui.caret_col = 0
    key(pygame.K_LEFT)  # wraps to end of the previous line
    assert sui.caret_row == 0 and sui.caret_col == len(sui.lines[0])
    key(pygame.K_RIGHT)  # wraps back down to the start of line 1
    assert sui.caret_row == 1 and sui.caret_col == 0
    print("arrow-key caret movement OK (incl. line-boundary wrap)")

    # Backspace merges a line into the previous one at col 0; mid-line
    # Backspace deletes the char before the caret
    before_merge = list(sui.lines)
    sui.caret_row, sui.caret_col = 1, 0
    key(pygame.K_BACKSPACE)
    assert sui.lines == [before_merge[0] + before_merge[1], before_merge[2]], sui.lines
    assert sui.caret_row == 0 and sui.caret_col == len(before_merge[0])
    sui.caret_col = 5
    line_before = sui.lines[0]
    key(pygame.K_BACKSPACE)
    assert sui.lines[0] == line_before[:4] + line_before[5:], sui.lines[0]
    print("Backspace OK (mid-line delete + line-merge at column 0)")

    # Delete (forward) mirrors Backspace at the far end
    sui.lines = ["abc", "def"]
    sui.caret_row, sui.caret_col = 0, 3
    key(pygame.K_DELETE)  # merges "def" onto "abc" (caret at end of line 0)
    assert sui.lines == ["abcdef"], sui.lines
    sui.caret_col = 1
    key(pygame.K_DELETE)  # removes the char AT the caret ('b'), not before it
    assert sui.lines == ["acdef"], sui.lines
    print("Delete (forward) OK")

    # ========================================================================
    # 6. compile: valid Behavior subclass -> success + discovered class name
    # ========================================================================
    sui.lines = ["from engine.scene import Behavior", "", "class MyBehavior(Behavior):",
                "    def update(self, entity, dt, engine):", "        pass"]
    sui._compile()
    assert sui.compile_result == {"ok": True, "class_name": "MyBehavior"}, sui.compile_result
    assert new_bp.compile_result == sui.compile_result, "compile must persist to the asset"
    assert new_bp.script == "\n".join(sui.lines), "compile must persist the script text too"
    print("compile: valid Behavior subclass OK ->", sui.compile_result)

    # ========================================================================
    # 7. compile: SyntaxError -> correct line + message, handler doesn't raise
    # ========================================================================
    sui.lines = ["class Bad(Behavior):", "    def update(self, entity, dt, engine)", "        pass"]
    sui._compile()  # missing ':' on line 2 -- must not raise out of this call
    r = sui.compile_result
    assert r["ok"] is False and r["stage"] == "syntax", r
    assert r["line"] == 2, r
    assert isinstance(r["message"], str) and r["message"], r
    assert sui.error_line == 2
    print("compile: SyntaxError OK -> line", r["line"], "message:", r["message"])

    # ========================================================================
    # 8. compile: raises at definition time (undefined name at module level)
    # ========================================================================
    sui.lines = ["from engine.scene import Behavior", "", "class Ok(Behavior):", "    pass",
                "", "totally_undefined_name.frobnicate()"]
    sui._compile()
    r = sui.compile_result
    assert r["ok"] is False and r["stage"] == "exec", r
    assert "NameError" in r["message"] and "totally_undefined_name" in r["message"], r
    assert r["line"] == 6, r
    print("compile: definition-time NameError OK -> line", r["line"], "message:", r["message"])

    # ========================================================================
    # 9. compile: no Behavior subclass -> reported clearly
    # ========================================================================
    sui.lines = ["x = 1", "y = 2"]
    sui._compile()
    r = sui.compile_result
    assert r["ok"] is False and r["stage"] == "validate", r
    assert "Behavior" in r["message"], r
    print("compile: no-Behavior-subclass OK ->", r["message"])

    # ========================================================================
    # 10. click every control in the script window: compile, save, minimize,
    #     title-bar drag, code-area (caret placement), close -- none may crash
    # ========================================================================
    sui.lines = ["from engine.scene import Behavior", "class Foo(Behavior):",
                "    def update(self, entity, dt, engine): pass"]
    w, h = eng.screen.get_size()
    cb = sui._compile_btn_rect(w, h)
    click((cb.centerx, cb.centery))
    assert sui.compile_result["ok"] is True
    sb = sui._save_btn_rect(w, h)
    click((sb.centerx, sb.centery))
    print("Compile + Save button clicks OK")

    outer = sui.outer_rect(w, h)
    minimize_pt = (outer.right - 44 + 8, outer.y + 2 + 8)
    click(minimize_pt)
    assert sui.minimized is True
    editor.draw(eng)  # draw while minimized must not crash either
    click(minimize_pt)
    assert sui.minimized is False
    print("minimize toggle OK (incl. draw() while minimized)")

    outer = sui.outer_rect(w, h)
    title_pt = (outer.x + 100, outer.y + 8)
    with um.patch.object(pygame.mouse, "get_pos", return_value=title_pt), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=title_pt)])
        drag_pt = (title_pt[0] + 30, title_pt[1] + 20)
        with um.patch.object(pygame.mouse, "get_pos", return_value=drag_pt):
            step([])
        assert sui.drag_title is not None
    with um.patch.object(pygame.mouse, "get_pos", return_value=drag_pt), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
        step([])
    assert sui.drag_title is None
    print("title-bar drag OK, window moved to", sui.pos)

    code = sui._code_rect(*eng.screen.get_size())
    click((code.x + 30, code.y + 5))
    assert sui.caret_row == 0
    print("code-area click (caret placement) OK, caret at", (sui.caret_row, sui.caret_col))

    outer = sui.outer_rect(*eng.screen.get_size())  # recompute -- the drag moved it
    close_pt = (outer.right - 24 + 8, outer.y + 2 + 8)
    click(close_pt)
    assert editor.script_ui is None, "X must close the script editor"
    print("close button OK")

    # closing must have persisted the buffer (auto-save on close)
    lib.reload()
    reloaded = lib.blueprint_by_name[new_bp.name]
    assert "class Foo(Behavior):" in reloaded.script, reloaded.script
    print("close-auto-saves OK: script persisted, confirmed via a fresh reload")

    # ========================================================================
    # 11. screenshot: content browser with a blueprint tile + the script
    #     editor open (visual sanity check)
    # ========================================================================
    editor.selected_folder = None
    editor.script_ui = ScriptEditorUI(editor, new_bp)
    fly = engine.behaviors.FlyController(camera, look_buttons=(3,),
                                         look_guard=lambda p: not editor.over_ui(p))
    editor.fly = fly
    scene.add(engine.Entity("__camera_bp").add_behavior(fly))
    scene.add(engine.Entity("__editor_bp").add_behavior(EditorBehavior(editor)))
    eng.esc_handler = editor.handle_escape
    OUT = os.path.join(tempfile.gettempdir(), "judge_blueprint.png")
    eng.run(scene, camera, max_frames=15, screenshot_path=OUT, overlay=editor.draw)
    print(f"screenshot saved: {OUT}")

    # ========================================================================
    # no-pollution guards
    # ========================================================================
    _real_settings_after = (open(REAL_SETTINGS, "rb").read()
                            if os.path.exists(REAL_SETTINGS) else None)
    assert _real_settings_after == _real_settings_before, (
        "blueprint_checks touched the real settings.json -- an Editor() in "
        "this suite is missing settings_path=TEST_SETTINGS")
    real_assets_dir = os.path.join(WT, "assets")
    assert not os.path.exists(os.path.join(real_assets_dir, "blueprints",
                                           "roundtrip_bp.json")), \
        "a blueprint leaked into the REAL assets/blueprints dir"
    assert lib.directory == ASSETS_DIR and lib.directory != real_assets_dir
    print("no-pollution guards OK: real settings.json and real assets/ untouched")

    print("ALL BLUEPRINT CHECKS PASSED")

finally:
    _keys_patch.stop()
    shutil.rmtree(TMP_ASSETS_ROOT, ignore_errors=True)
