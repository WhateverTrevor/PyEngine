"""Judge checks: box/marquee selection (Blender-style drag-rectangle select).

Per the project's hard rule for UI/interaction tests, every press/drag/
release sequence below is driven through the REAL event path
(eng.input.process + editor.update), not direct handler calls. Held mouse
button / modifier-key state isn't reachable via synthetic SDL events under
the dummy video driver, so those frames patch pygame.mouse.get_pos AND
pygame.mouse.get_pressed AND pygame.key.get_pressed at the OS boundary --
the exact idiom already proven in tests/multiselect_checks.py and
tests/snap_checks.py (editor.update()/eng.input.process() themselves stay
completely real). A MOUSEBUTTONDOWN is a real injectable edge event; the
"release" frame that follows re-patches get_pressed to all-False (mouse no
longer held) rather than injecting a MOUSEBUTTONUP, since the production
code (see editor.py update()'s marquee block) resolves the drag off
inp.mouse_held(1), matching the gizmo_drag/panel_drag/splitter_drag
precedent already in editor.py.
"""
import math
import os
import sys
import tempfile
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import pygame

import engine
from editor import Editor, MARQUEE_THRESHOLD, build_starter_scene

TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_marquee_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
Vec3 = engine.Vec3
W, H = eng.screen.get_size()


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


def press(ed, pos, mods=(), release_pos=None):
    """Press LMB at `pos` (real MOUSEBUTTONDOWN), optionally drag to
    `release_pos` over a continuation frame, then release (mouse_held ->
    False). Returns nothing; caller inspects `ed` state afterward. Mirrors
    the drag idiom in snap_checks.py sections 3/6/11/12."""
    end = release_pos if release_pos is not None else pos
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys(mods)), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step(ed, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)])
        yield "pressed"
        if release_pos is not None:
            with um.patch.object(pygame.mouse, "get_pos", return_value=release_pos):
                step(ed, [])
                yield "dragged"
    with um.patch.object(pygame.mouse, "get_pos", return_value=end), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
        step(ed, [])
    yield "released"


def do_drag(ed, start, end, mods=()):
    """Run a full press -> drag -> release cycle, checking self.marquee is
    live (non-None) during the held phase -- proves the state machine
    actually entered marquee mode instead of silently no-opping."""
    gen = press(ed, start, mods=mods, release_pos=end)
    assert next(gen) == "pressed"
    assert next(gen) == "dragged"
    assert ed.marquee is not None, "a drag past the threshold must be tracked as a live marquee"
    assert list(gen) == ["released"]
    assert ed.marquee is None, "marquee state must be cleared after release"


def entity_screen_bbox(ed, e, w, h):
    lo, hi = ed._world_aabb(e)
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    pts = [ed.camera.project(Vec3(*c), w, h) for c in corners]
    xs = [p[0] for p in pts if p is not None]
    ys = [p[1] for p in pts if p is not None]
    assert xs and ys, "entity must be projectable for this test's camera setup"
    return min(xs), min(ys), max(xs), max(ys)


# ============================================================================
# Setup A: shared oblique camera + starter scene (matches multiselect_checks.py
# / snap_checks.py) -- used for the conflict-avoidance checks that need real
# panel layout + a real gizmo, not entity screen geometry.
# ============================================================================
camera = engine.Camera(position=Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
               settings_path=TEST_SETTINGS)

# ==== 1. a press starting on a gizmo handle still grabs the gizmo, not a
#          marquee (must-not-conflict requirement) ====
crate0 = next(e for e in scene.entities if e.asset_name == "Crate")
editor.selected = crate0
editor._set_gizmo_mode("translate")
editor.gizmo_space = "world"
handles = editor._gizmo_handles(W, H)
i0, axis0, s0h, s1h, _color, _length = handles[0]
grab_pos = (int(s1h[0]), int(s1h[1]))
with um.patch.object(pygame.mouse, "get_pos", return_value=grab_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step(editor, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=grab_pos)])
assert editor.gizmo_drag is not None and editor.gizmo_drag["mode"] == "translate", \
    "pressing a gizmo handle must still grab the gizmo"
assert editor.marquee is None, "a gizmo-handle grab must never also start a marquee"
editor.gizmo_drag = None
print("gizmo-handle press grabs the gizmo, not a marquee OK")

# ==== 2. a press over a UI panel does not start a marquee ====
layout = editor._layout(W, H)
content = editor._panel_content_rect("outliner", layout)
ui_pos = (content.x + 10, content.y + 10)
editor.marquee = None
with um.patch.object(pygame.mouse, "get_pos", return_value=ui_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step(editor, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=ui_pos)])
assert editor.marquee is None, "a press over a UI panel must never start a marquee"
print("UI-panel press does not start a marquee OK")


# ============================================================================
# Setup B: dedicated straight-on camera + a clean 3-entity scene (matches
# multiselect_checks.py section 10's viewport-click precedent) -- unambiguous
# screen geometry for the actual entity-in-rect selection logic.
# ============================================================================
camera_m = engine.Camera(position=Vec3(0.0, 2.0, 20.0), yaw=0.0, pitch=0.0)
scene_m = engine.Scene()
eA = lib.instantiate("Crate"); eA.transform.position = Vec3(-8.0, 1.0, 0.0); scene_m.add(eA)
eB = lib.instantiate("Crate"); eB.transform.position = Vec3(0.0, 1.0, 0.0); scene_m.add(eB)
eC = lib.instantiate("Crate"); eC.transform.position = Vec3(8.0, 1.0, 0.0); scene_m.add(eC)
editor_m = Editor(engine, eng, scene_m, camera_m, lib, "scenes/scene.json",
                  settings_path=TEST_SETTINGS)
layout_m = editor_m._layout(W, H)
vp = layout_m["viewport"]

bbox_a = entity_screen_bbox(editor_m, eA, W, H)
bbox_b = entity_screen_bbox(editor_m, eB, W, H)
bbox_c = entity_screen_bbox(editor_m, eC, W, H)
rect_a = pygame.Rect(int(bbox_a[0]), int(bbox_a[1]),
                     max(1, int(bbox_a[2] - bbox_a[0])), max(1, int(bbox_a[3] - bbox_a[1])))
rect_b = pygame.Rect(int(bbox_b[0]), int(bbox_b[1]),
                     max(1, int(bbox_b[2] - bbox_b[0])), max(1, int(bbox_b[3] - bbox_b[1])))
rect_c = pygame.Rect(int(bbox_c[0]), int(bbox_c[1]),
                     max(1, int(bbox_c[2] - bbox_c[0])), max(1, int(bbox_c[3] - bbox_c[1])))
for r in (rect_a, rect_b, rect_c):
    assert vp.collidepoint(r.center), f"test geometry: entity bbox {r} must land inside the viewport"
# sanity: the 3 crates project to well-separated, non-overlapping screen
# footprints -- otherwise a rect built from A+B's bbox could accidentally
# reach C, invalidating the exclusion checks below
assert not rect_a.colliderect(rect_b) and not rect_b.colliderect(rect_c), \
    "test geometry: the 3 crates must not overlap on screen"
print(f"setup B geometry OK: A={tuple(rect_a)}, B={tuple(rect_b)}, C={tuple(rect_c)}")

# ==== 3. tiny-movement press-release ON an entity still single-selects it
#          (not a marquee) ====
editor_m.selected = None
pt = camera_m.project(eA.transform.position, W, H)
press_pos = (int(round(pt[0])), int(round(pt[1])))
release_pos = (press_pos[0] + 2, press_pos[1] + 1)  # well under MARQUEE_THRESHOLD
assert math.hypot(*(a - b for a, b in zip(release_pos, press_pos))) < MARQUEE_THRESHOLD
with um.patch.object(pygame.mouse, "get_pos", return_value=press_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step(editor_m, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=press_pos)])
    assert editor_m.marquee is None, "a press landing on an entity must never start a marquee"
    assert editor_m.selected is eA, "entity click still selects immediately, on press"
    with um.patch.object(pygame.mouse, "get_pos", return_value=release_pos):
        step(editor_m, [])
with um.patch.object(pygame.mouse, "get_pos", return_value=release_pos), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
    step(editor_m, [])
assert editor_m.selection == [eA] and editor_m.selected is eA
assert editor_m.marquee is None
assert_active_valid(editor_m, "tiny click on entity")
print("tiny-movement press-release on an entity OK: unchanged single-click select")

# ==== 4. tiny-movement press-release on EMPTY space stays the old plain-
#          click behavior: clear (plain) / no-op (Shift) -- NOT a marquee ====
empty_pos = (vp.x + 20, vp.y + 30)  # top strip of the viewport, above every
                                    # crate's projected footprint (see bboxes)
for r in (rect_a, rect_b, rect_c):
    assert empty_pos[1] < r.top - 5, "test geometry: empty_pos must clear every crate's bbox"

editor_m.selected = eA
tiny_end = (empty_pos[0] + 1, empty_pos[1] + 2)
with um.patch.object(pygame.mouse, "get_pos", return_value=empty_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step(editor_m, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=empty_pos)])
    assert editor_m.marquee is not None, "a press on empty space is a PENDING marquee until release"
    with um.patch.object(pygame.mouse, "get_pos", return_value=tiny_end):
        step(editor_m, [])
with um.patch.object(pygame.mouse, "get_pos", return_value=tiny_end), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
    step(editor_m, [])
assert editor_m.selection == [] and editor_m.selected is None, \
    "tiny click on empty space must clear the selection exactly like the old plain click"
assert editor_m.marquee is None
print("tiny-movement press-release on empty space (plain) clears OK")

editor_m.selected = eA
with um.patch.object(pygame.mouse, "get_pos", return_value=empty_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LSHIFT])), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step(editor_m, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=empty_pos)])
    with um.patch.object(pygame.mouse, "get_pos", return_value=tiny_end):
        step(editor_m, [])
with um.patch.object(pygame.mouse, "get_pos", return_value=tiny_end), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
    step(editor_m, [])
assert editor_m.selection == [eA] and editor_m.selected is eA, \
    "Shift + tiny click on empty space must be a no-op, matching Shift+click-empty"
assert editor_m.marquee is None
assert_active_valid(editor_m, "tiny click on empty space")
print("tiny-movement press-release on empty space (Shift) is a no-op OK")

# ==== 5. drag a rect over 2 of 3 spread-out entities -> exactly those 2
#          selected (plain marquee REPLACES the selection) ====
editor_m.selected = None
margin = 20
start_ab = (rect_a.left - margin, min(rect_a.top, rect_b.top) - margin)
end_ab = (rect_b.right + margin, max(rect_a.bottom, rect_b.bottom) + margin)
drag_ab_rect = pygame.Rect(start_ab[0], start_ab[1],
                           end_ab[0] - start_ab[0], end_ab[1] - start_ab[1])
assert not drag_ab_rect.colliderect(rect_c), \
    "test geometry: the A+B drag rect must not reach entity C"
do_drag(editor_m, start_ab, end_ab)
assert set(editor_m.selection) == {eA, eB}, editor_m.selection
assert editor_m.selected in (eA, eB), "active must be a member of the new selection"
assert editor_m.selected is eB, "active should be the last entity added (scene order A,B,C)"
assert_active_valid(editor_m, "marquee over A+B")
print(f"marquee drag over A+B (excluding C) OK: selection={[e is eA and 'A' or 'B' for e in editor_m.selection]}, "
     f"active={'A' if editor_m.selected is eA else 'B'}")

# ==== 6. Shift+drag adds a third without dropping the first two ====
margin_c = 20
start_c = (rect_c.left - margin_c, rect_c.top - margin_c)
end_c = (rect_c.right + margin_c, rect_c.bottom + margin_c)
drag_c_rect = pygame.Rect(start_c[0], start_c[1], end_c[0] - start_c[0], end_c[1] - start_c[1])
assert not drag_c_rect.colliderect(rect_a) and not drag_c_rect.colliderect(rect_b), \
    "test geometry: the C-only drag rect must not reach A or B"
do_drag(editor_m, start_c, end_c, mods=(pygame.K_LSHIFT,))
assert set(editor_m.selection) == {eA, eB, eC}, editor_m.selection
assert eA in editor_m.selection and eB in editor_m.selection, \
    "Shift+marquee must ADD to the selection, not replace it"
assert editor_m.selected is eC, "active should be the newly-added entity"
assert_active_valid(editor_m, "Shift+marquee adding C")
print("Shift+drag over C OK: added without dropping A/B, active is the newly-added C")

# ==== 7. marquee over empty area with no entities: clears (plain) / leaves
#          selection intact (Shift) ====
empty_start = (vp.x + 20, vp.y + 30)
empty_end = (vp.x + 140, vp.y + 90)
empty_rect = pygame.Rect(empty_start[0], empty_start[1],
                         empty_end[0] - empty_start[0], empty_end[1] - empty_start[1])
for r in (rect_a, rect_b, rect_c):
    assert not empty_rect.colliderect(r), "test geometry: empty_rect must contain no entity"

editor_m._set_selection([eA, eB], active=eB)
do_drag(editor_m, empty_start, empty_end)
assert editor_m.selection == [] and editor_m.selected is None, \
    "plain marquee over empty area must clear the selection"
print("marquee over empty area (plain) clears OK")

editor_m._set_selection([eA, eB], active=eB)
do_drag(editor_m, empty_start, empty_end, mods=(pygame.K_LSHIFT,))
assert editor_m.selection == [eA, eB] and editor_m.selected is eB, \
    "Shift+marquee over empty area must leave the selection untouched"
assert_active_valid(editor_m, "Shift+marquee over empty area")
print("marquee over empty area (Shift) leaves selection intact OK")

# ==== 8. visual check: the marquee rectangle is actually drawn mid-drag ====
editor_m._set_selection([], active=None)
gen = press(editor_m, start_ab, release_pos=end_ab)
assert next(gen) == "pressed"
assert next(gen) == "dragged"
assert editor_m.marquee is not None
surf = eng.screen
surf.fill((15, 16, 20))
editor_m.draw(eng)
out_png = os.path.join(tempfile.gettempdir(), "judge_marquee.png")
pygame.image.save(surf, out_png)
print(f"screenshot saved: {out_png}")
assert list(gen) == ["released"]
assert set(editor_m.selection) == {eA, eB}
assert editor_m.marquee is None

print("ALL MARQUEE CHECKS PASSED")
