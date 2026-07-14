"""Judge checks: multi-selection foundation (self.selection + self.selected
as its ACTIVE/last-clicked element), gizmo pivot anchoring, and batch ops.

Per the project's hard rule for UI/interaction tests, selection changes that
happen via a click (viewport or outliner, plain or Shift-held) and hotkeys
(Ctrl+D, Del, End, F) are driven through the REAL event path
(eng.input.process + editor.update), not direct handler calls -- matching
the precedent in tests/snap_checks.py and tests/mat_ui_checks.py. Held
modifier keys (Shift/Ctrl/Alt) aren't reachable via synthetic SDL events, so
those sections patch pygame.key.get_pressed / pygame.mouse at the OS
boundary while eng.input.process()/editor.update() themselves stay real.
Pure math (gizmo pivot, _set_selection/_toggle_selection bookkeeping) is
checked via direct calls, matching the toolbar_checks.py/snap_checks.py
precedent for non-interactive logic.
"""
import os
import sys
import tempfile
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import pygame

import engine
from editor import Editor, ROW_H, build_starter_scene

TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_multiselect_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
Vec3 = engine.Vec3
W, H = eng.screen.get_size()

# ---- setup A: the oblique camera + starter scene shared by every gizmo
# section below -- same camera pose proven (in snap_checks.py) to project
# near-origin entities into the viewport reliably for handle grabbing ----
camera = engine.Camera(position=Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
               settings_path=TEST_SETTINGS)


class FakeKeys:
    """pygame.key.get_pressed() stand-in (see snap_checks.py precedent)."""
    def __init__(self, held=()):
        self._held = set(held)
    def __getitem__(self, key):
        return key in self._held


def step(ed, events):
    """Drive one frame of `ed` through the REAL input + update path."""
    eng.input.process(events)
    ed.update(eng, 1 / 60)
    eng.input.consume_edges()


def assert_active_valid(ed, msg=""):
    """Invariant that must hold after every selection-affecting op."""
    if ed.selection:
        assert ed.selected in ed.selection, f"active not in selection: {msg}"
    else:
        assert ed.selected is None, f"selected must be None when selection is empty: {msg}"


# ==== 1. _set_selection / _toggle_selection bookkeeping (pure, direct) ====
pA = lib.instantiate("Crate"); pA.transform.position = Vec3(0.0, 1.0, 0.0); scene.add(pA)
pB = lib.instantiate("Crate"); pB.transform.position = Vec3(2.0, 1.0, 0.0); scene.add(pB)
pC = lib.instantiate("Crate"); pC.transform.position = Vec3(4.0, 1.0, 0.0); scene.add(pC)

editor.selected = pA
assert editor.selection == [pA] and editor.selected is pA, "plain assignment must single-select"
editor._toggle_selection(pB)
assert editor.selection == [pA, pB] and editor.selected is pB, "toggle-add must append + activate"
editor._toggle_selection(pC)
assert editor.selection == [pA, pB, pC] and editor.selected is pC
editor._toggle_selection(pB)  # remove a middle, non-active member
assert editor.selection == [pA, pC] and editor.selected is pC, "removing a non-active member keeps active"
editor._toggle_selection(pC)  # remove the active member itself
assert editor.selection == [pA] and editor.selected is pA, "removing active falls back to new last member"
editor._toggle_selection(pA)
assert editor.selection == [] and editor.selected is None, "removing the last member clears everything"
assert_active_valid(editor, "toggle bookkeeping")
print("_set_selection / _toggle_selection bookkeeping OK")

# ==== 2. gizmo pivot = mean of selected world positions (v1's "Median
#          Point"); single-selection callers get exactly that entity ====
editor.selected = pA
p, s0, _len = editor._gizmo_center(W, H)
assert abs(p.x - 0.0) < 1e-9 and abs(p.y - 1.0) < 1e-9, "single-select pivot must equal that entity"
editor._set_selection([pA, pB, pC], active=pC)
pivot = editor._selection_pivot()
assert abs(pivot.x - 2.0) < 1e-9 and abs(pivot.y - 1.0) < 1e-9 and abs(pivot.z) < 1e-9, pivot
p2, s02, _l2 = editor._gizmo_center(W, H)
assert abs(p2.x - pivot.x) < 1e-9 and abs(p2.y - pivot.y) < 1e-9
assert s02 is not None, "pivot must project into the viewport for the oblique test camera"
print(f"gizmo pivot OK: mean of (0,2,4)->x={pivot.x}, single-select pivot == entity position")

# ==== 3. translate drag moves every selected entity by the SAME world
#          delta (real event path), including a grid-snap case ====
editor._set_selection([pA, pB, pC], active=pC)
editor._set_gizmo_mode("translate")
editor.gizmo_space = "world"
editor.snap_enabled = False
handles = editor._gizmo_handles(W, H)
i0, axis0, s0h, s1h, _color, _length = handles[0]
assert axis0 == (1.0, 0.0, 0.0), axis0
starts = {id(e): (e.transform.position.x, e.transform.position.y, e.transform.position.z)
         for e in (pA, pB, pC)}
grab_pos = (int(s1h[0]), int(s1h[1]))
with um.patch.object(pygame.mouse, "get_pos", return_value=grab_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step(editor, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=grab_pos)])
    assert editor.gizmo_drag is not None and editor.gizmo_drag["mode"] == "translate"
    assert set(editor.gizmo_drag["starts"].keys()) == {id(pA), id(pB), id(pC)}
    dxg, dyg = s1h[0] - s0h[0], s1h[1] - s0h[1]
    move_pos = (int(s1h[0] + dxg * 1.3), int(s1h[1] + dyg * 1.3))
    with um.patch.object(pygame.mouse, "get_pos", return_value=move_pos):
        step(editor, [])  # continuation frame, mouse still "held"
deltas = {name: (e.transform.position.x - starts[id(e)][0],
                 e.transform.position.y - starts[id(e)][1],
                 e.transform.position.z - starts[id(e)][2])
         for name, e in (("A", pA), ("B", pB), ("C", pC))}
assert abs(deltas["A"][0]) > 1e-6, "the drag must have actually moved things"
assert abs(deltas["A"][0] - deltas["B"][0]) < 1e-9, deltas
assert abs(deltas["A"][0] - deltas["C"][0]) < 1e-9, deltas
assert all(abs(d[1]) < 1e-9 and abs(d[2]) < 1e-9 for d in deltas.values()), \
    "only the dragged (X) axis should move"
print(f"multi-translate (no snap) OK: all three moved by the same delta x={deltas['A'][0]:.4f}")
editor.gizmo_drag = None

# grid-snap case: reset to a deliberately OFF-grid start (0.13/2.13/4.13,
# none a multiple of 0.5) so a successful snap is unambiguous -- same drag,
# snap enabled -- every entity still moves by the identical (now-quantized)
# delta, and the active element (the snap reference) lands exactly on-grid
snap_starts = {}
for e, ox in ((pA, 0.13), (pB, 2.13), (pC, 4.13)):
    e.transform.position = Vec3(ox, 1.0, 0.0)
    snap_starts[id(e)] = ox
editor.snap_index["translate"] = 2  # 0.5
editor.snap_enabled = True
handles = editor._gizmo_handles(W, H)
i0, axis0, s0h, s1h, _color, _length = handles[0]
grab_pos = (int(s1h[0]), int(s1h[1]))
with um.patch.object(pygame.mouse, "get_pos", return_value=grab_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step(editor, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=grab_pos)])
    dxg, dyg = s1h[0] - s0h[0], s1h[1] - s0h[1]
    move_pos = (int(s1h[0] + dxg * 0.3), int(s1h[1] + dyg * 0.3))
    with um.patch.object(pygame.mouse, "get_pos", return_value=move_pos):
        step(editor, [])
deltas2 = {name: e.transform.position.x - snap_starts[id(e)] for name, e in
          (("A", pA), ("B", pB), ("C", pC))}
assert abs(deltas2["A"]) > 1e-6, "the snapped drag must have actually moved things"
assert abs(deltas2["A"] - deltas2["B"]) < 1e-9 and abs(deltas2["A"] - deltas2["C"]) < 1e-9, deltas2
active_x = editor.selected.transform.position.x  # active (pC) is the snap reference entity
assert abs(active_x / 0.5 - round(active_x / 0.5)) < 1e-6, active_x
assert abs(active_x - 4.13) > 1e-6, "must have actually snapped away from the off-grid start"
print(f"multi-translate (grid-snap) OK: uniform delta, active snapped 4.13->{active_x}")
editor.gizmo_drag = None
editor.snap_enabled = False
for e in (pA, pB, pC):
    e.transform.position = Vec3(*starts[id(e)])
assert_active_valid(editor, "post multi-translate")

# ==== 4. batch Ctrl+D duplicate: N copies become the new selection, the
#          duplicate of the previous active becomes the new active ====
dA = lib.instantiate("Crate"); dA.transform.position = Vec3(0.0, 1.0, 0.0); scene.add(dA)
dB = lib.instantiate("Crate"); dB.transform.position = Vec3(2.0, 1.0, 0.0); scene.add(dB)
editor._set_selection([dA, dB], active=dB)
count_before = len(scene.entities)
with um.patch.object(pygame.mouse, "get_pos", return_value=(700, 400)), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LCTRL])):
    step(editor, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, unicode="d", mod=0)])
assert len(scene.entities) == count_before + 2, "Ctrl+D must duplicate every selected entity"
assert len(editor.selection) == 2 and dA not in editor.selection and dB not in editor.selection
dup_dA = next(e for e in editor.selection if abs(e.transform.position.x - 0.8) < 1e-6)
dup_dB = next(e for e in editor.selection if abs(e.transform.position.x - 2.8) < 1e-6)
assert editor.selected is dup_dB, "duplicate of the previous ACTIVE element must become active"
assert_active_valid(editor, "post batch duplicate")
print(f"batch Ctrl+D duplicate OK: {count_before}->{len(scene.entities)} entities, "
     f"new selection is the {len(editor.selection)} duplicates, active is dup of previous active")

# ==== 5. batch delete removes every selected entity (real Del key) ====
count_before2 = len(scene.entities)
with um.patch.object(pygame.mouse, "get_pos", return_value=(700, 400)):
    step(editor, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, unicode="", mod=0)])
assert len(scene.entities) == count_before2 - 2, "Del must remove every selected entity"
assert dup_dA not in scene.entities and dup_dB not in scene.entities
assert editor.selection == [] and editor.selected is None
print(f"batch Del delete OK: {count_before2}->{len(scene.entities)} entities, selection cleared")

# ==== 6. Alt-drag gizmo duplicate duplicates the WHOLE selection (real
#          event path) -- mirrors snap_checks.py's single-entity version ====
altA = lib.instantiate("Crate"); altA.transform.position = Vec3(4.0, 1.0, 0.0); scene.add(altA)
altB = lib.instantiate("Crate"); altB.transform.position = Vec3(5.0, 1.0, 0.0); scene.add(altB)
editor._set_selection([altA, altB], active=altB)
editor._set_gizmo_mode("translate")
editor.gizmo_space = "world"
count_before3 = len(scene.entities)
handles = editor._gizmo_handles(W, H)
i0, axis0, s0h, s1h, _color, _length = handles[0]
grab_pos = (int(s1h[0]), int(s1h[1]))
with um.patch.object(pygame.mouse, "get_pos", return_value=grab_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LALT])), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step(editor, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=grab_pos)])
    assert len(scene.entities) == count_before3 + 2, "Alt-grab must duplicate the whole selection"
    dup_altA = next(e for e in editor.selection if abs(e.transform.position.x - 4.0) < 1e-6)
    dup_altB = editor.selected
    assert dup_altB is not altB and abs(dup_altB.transform.position.x - 5.0) < 1e-6
    dxg, dyg = s1h[0] - s0h[0], s1h[1] - s0h[1]
    move_pos = (int(s1h[0] + dxg), int(s1h[1] + dyg))
    with um.patch.object(pygame.mouse, "get_pos", return_value=move_pos):
        step(editor, [])
assert dup_altA.transform.position.x != 4.0 and dup_altB.transform.position.x != 5.0, \
    "both duplicates must have moved"
assert (altA.transform.position.x, altB.transform.position.x) == (4.0, 5.0), \
    "the ORIGINALS must be untouched"
assert_active_valid(editor, "post alt-drag duplicate")
print(f"Alt-drag duplicates whole selection OK: entities {count_before3}->{len(scene.entities)}, "
     f"both duplicates moved, originals untouched")
editor.gizmo_drag = None

# ==== 7. floor snap (End) drops every selected entity independently ====
floor_x = lib.instantiate("Crate"); floor_x.transform.position = Vec3(50.0, 0.65, 50.0)
scene.add(floor_x)  # a "floor" resting on y=0
drop1 = lib.instantiate("Crate"); drop1.transform.position = Vec3(50.0, 6.0, 50.0)
scene.add(drop1)    # floating above floor_x
drop2 = lib.instantiate("Crate"); drop2.transform.position = Vec3(-50.0, 9.0, -50.0)
scene.add(drop2)    # nothing underneath -> y=0 fallback
editor._set_selection([drop1, drop2], active=drop2)
with um.patch.object(pygame.mouse, "get_pos", return_value=(700, 400)):
    step(editor, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_END, unicode="", mod=0)])
lo1, _ = editor._world_aabb(drop1)
_, hi_floor = editor._world_aabb(floor_x)
lo2, _ = editor._world_aabb(drop2)
assert abs(lo1[1] - hi_floor[1]) < 1e-6, (lo1[1], hi_floor[1])
assert abs(lo2[1] - 0.0) < 1e-6, lo2[1]
assert editor.selection == [drop1, drop2], "floor snap must not disturb the selection"
print(f"batch End floor-snap OK: drop1 bottom={lo1[1]:.6f} (on floor_x top {hi_floor[1]:.6f}), "
     f"drop2 bottom={lo2[1]:.6f} (y=0 fallback)")

# ==== 8. Focus (F) frames the whole selection (real event path) ====
editor._set_selection([drop1, drop2], active=drop2)
cam_before = (camera.position.x, camera.position.y, camera.position.z)
with um.patch.object(pygame.mouse, "get_pos", return_value=(700, 400)), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()):
    step(editor, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, unicode="f", mod=0)])
cam_after = (camera.position.x, camera.position.y, camera.position.z)
assert cam_after != cam_before, "F must move the camera"
print(f"Focus-selection (F) OK: camera moved {cam_before} -> {cam_after}")

# ==== 9. outliner shift-click extends/toggles + sets active; plain click
#          replaces (REAL event path) ====
layout = editor._layout(W, H)
content = editor._panel_content_rect("outliner", layout)


def outliner_click_pos(entity):
    rows = editor._outliner_rows()
    i = rows.index(entity)
    return (content.x + 10, content.y + 6 + i * ROW_H + ROW_H // 2 - editor.outliner_scroll * ROW_H)


editor.outliner_scroll = 0
crate0 = next(e for e in scene.entities if e.asset_name == "Crate")
floor0 = next(e for e in scene.entities if e.asset_name == "Stone Floor")
mp_crate = outliner_click_pos(crate0)
mp_floor = outliner_click_pos(floor0)

with um.patch.object(pygame.mouse, "get_pos", return_value=mp_crate), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()):
    step(editor, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=mp_crate)])
assert editor.selection == [crate0] and editor.selected is crate0, "plain outliner click must single-select"

with um.patch.object(pygame.mouse, "get_pos", return_value=mp_floor), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LSHIFT])):
    step(editor, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=mp_floor)])
assert editor.selection == [crate0, floor0] and editor.selected is floor0, \
    "Shift+click outliner must extend + activate"

with um.patch.object(pygame.mouse, "get_pos", return_value=mp_floor), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LSHIFT])):
    step(editor, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=mp_floor)])
assert editor.selection == [crate0] and editor.selected is crate0, \
    "Shift+click an already-selected outliner row must toggle it off, active falls back"
assert_active_valid(editor, "post outliner shift-click")
print("outliner shift-click extend/toggle + active OK (real event path)")

# ==== 10. viewport click: plain replaces, Shift+click extends/toggles,
#          click-empty clears (REAL event path, dedicated straight-on
#          camera + clean scene so ray-picks are unambiguous) ====
camera_b = engine.Camera(position=Vec3(0.0, 2.0, 10.0), yaw=0.0, pitch=0.0)
scene_b = engine.Scene()
vcA = lib.instantiate("Crate"); vcA.transform.position = Vec3(-3.0, 1.0, 0.0); scene_b.add(vcA)
vcB = lib.instantiate("Crate"); vcB.transform.position = Vec3(3.0, 1.0, 0.0); scene_b.add(vcB)
editor_b = Editor(engine, eng, scene_b, camera_b, lib, "scenes/scene.json",
                  settings_path=TEST_SETTINGS)
layout_b = editor_b._layout(W, H)
vp_b = layout_b["viewport"]


def project_click(entity):
    pt = camera_b.project(entity.transform.position, W, H)
    assert pt is not None
    return (int(round(pt[0])), int(round(pt[1])))


mp_a, mp_b = project_click(vcA), project_click(vcB)
# bottom-right corner: below the viewport toolbar strip, far from both
# crates' screen columns/rows -- clear sky
mp_empty = (vp_b.right - 30, vp_b.bottom - 30)

with um.patch.object(pygame.mouse, "get_pos", return_value=mp_a), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()):
    step(editor_b, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=mp_a)])
assert editor_b.selection == [vcA] and editor_b.selected is vcA, "plain viewport click must single-select"

with um.patch.object(pygame.mouse, "get_pos", return_value=mp_b), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LSHIFT])):
    step(editor_b, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=mp_b)])
assert editor_b.selection == [vcA, vcB] and editor_b.selected is vcB, \
    "Shift+click viewport must extend + activate"

with um.patch.object(pygame.mouse, "get_pos", return_value=mp_a), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LSHIFT])):
    step(editor_b, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=mp_a)])
assert editor_b.selection == [vcB] and editor_b.selected is vcB, \
    "Shift+click an already-selected viewport entity must toggle it off"
assert_active_valid(editor_b, "post viewport shift-click")

with um.patch.object(pygame.mouse, "get_pos", return_value=mp_empty), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LSHIFT])):
    step(editor_b, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=mp_empty)])
assert editor_b.selection == [vcB], "Shift+click on empty space must be a no-op, not a clear"

with um.patch.object(pygame.mouse, "get_pos", return_value=mp_empty), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()):
    step(editor_b, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=mp_empty)])
assert editor_b.selection == [] and editor_b.selected is None, \
    "plain click on empty space must clear the selection"
print("viewport click: single/Shift-extend/Shift-toggle/empty-clear OK (real event path)")

# ==== 11. visual check: active vs. non-active highlight colors differ in
#          the outliner and in the viewport bracket, "N selected" shown ====
editor._set_selection([crate0, floor0], active=floor0)
w, h = eng.screen.get_size()
surf = eng.screen
surf.fill((15, 16, 20))
editor.draw(eng)
out_png = os.path.join(tempfile.gettempdir(), "judge_multiselect.png")
pygame.image.save(surf, out_png)
print(f"screenshot saved: {out_png}")

print("ALL MULTISELECT CHECKS PASSED")
