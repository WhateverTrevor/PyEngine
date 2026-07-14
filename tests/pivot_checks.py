"""Judge checks: Blender-style pivot modes (Median Point / Bounding Box
Center / Active Element / Individual Origins) for multi-selection rotate
and scale gizmo drags, run 2b of the pivot slate (foundation: run 2a's
multi-selection, see tests/multiselect_checks.py).

Per the project's hard rule for UI/interaction tests: the pivot-mode
toolbar button click and one rotate drag are driven through the REAL event
path (eng.input.process + editor.update), matching the precedent in
tests/snap_checks.py and tests/multiselect_checks.py. Held modifier keys
(Alt) aren't reachable via synthetic SDL events, so those sections patch
pygame.key.get_pressed / pygame.mouse.get_pos/get_pressed at the OS
boundary while eng.input.process()/editor.update() themselves stay real.
Pure orbit/scale math (the core of this task) is checked via direct calls
to _try_grab_gizmo/_update_gizmo_drag with numeric asserts, matching the
snap_checks.py precedent for gizmo-drag math -- driving every one of the
4 modes x 2 gizmo ops through real mouse motion would mostly re-test
already-proven ring/handle hit-testing, not the new orbit/scale math.
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
from editor import Editor, build_starter_scene, load_settings

TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_pivot_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
Vec3 = engine.Vec3
camera = engine.Camera(position=Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
               settings_path=TEST_SETTINGS)
W, H = eng.screen.get_size()


class FakeInput:
    def __init__(self, alt=False, shift=False, ctrl=False):
        self._alt, self._shift, self._ctrl = alt, shift, ctrl
    def held(self, key):
        if key in (pygame.K_LALT, pygame.K_RALT):
            return self._alt
        if key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
            return self._shift
        if key in (pygame.K_LCTRL, pygame.K_RCTRL):
            return self._ctrl
        return False


class FakeKeys:
    def __init__(self, held=()):
        self._held = set(held)
    def __getitem__(self, key):
        return key in self._held


def step(events):
    eng.input.process(events)
    editor.update(eng, 1 / 60)
    eng.input.consume_edges()


def robust_y_ring_point(pts1):
    """A screen point on the Y (yaw) ring that hit-tests to axis_i == 1
    even after int-truncation/jitter -- see snap_checks.py precedent."""
    for p in pts1:
        if p is None:
            continue
        ip = (int(p[0]), int(p[1]))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                got = editor._try_grab_gizmo((ip[0] + dx, ip[1] + dy), W, H)
                axis_i = editor.gizmo_drag["axis_i"] if got else None
                editor.gizmo_drag = None
                if not got or axis_i != 1:
                    ok = False
        if ok:
            return ip
    raise AssertionError("no Y-ring screen point robustly hit-tests to axis_i==1")


def fresh_trio():
    """3 crates at x=0,2,4 (y=1, z=0), unit scale, zero rotation -- the
    shared fixture for every group rotate/scale section below."""
    a = lib.instantiate("Crate"); a.transform.position = Vec3(0.0, 1.0, 0.0); scene.add(a)
    b = lib.instantiate("Crate"); b.transform.position = Vec3(2.0, 1.0, 0.0); scene.add(b)
    c = lib.instantiate("Crate"); c.transform.position = Vec3(4.0, 1.0, 0.0); scene.add(c)
    for e in (a, b, c):
        e.transform.rotation = Vec3(0.0, 0.0, 0.0)
        e.transform.scale = Vec3(1.0, 1.0, 1.0)
    return a, b, c


# ==== 1. pivot-mode selector: toolbar button cycles + labels the 5 modes
#          (run 3 added "3D Cursor" -- see tests/cursor_checks.py for its
#          dedicated coverage), driven via a REAL click event ====
assert editor.pivot_mode == "median", "Median Point must be the default"
layout = editor._layout(W, H)
toolbar_rect = editor._viewport_toolbar_rect(layout["viewport"])
rects = editor._toolbar_button_rects(toolbar_rect)
pivot_rect = next(r for b, r in rects if b["id"] == "pivot_mode")
mp = (pivot_rect.centerx, pivot_rect.centery)
seen = [editor.pivot_mode]
for _ in range(5):
    with um.patch.object(pygame.mouse, "get_pos", return_value=mp), \
         um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()):
        step([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=mp)])
    seen.append(editor.pivot_mode)
assert seen == ["median", "bbox", "active", "individual", "cursor", "median"], seen
assert editor._pivot_label() == "Median Point"
print(f"pivot-mode toolbar button OK (real click path): cycled {seen}")

# ==== 2. pivot mode persists through a settings round-trip (never touches
#          the real settings.json -- settings_path is the injected temp
#          file, matching snap_checks.py section 6) ====
editor.pivot_mode = "bbox"
editor._save_settings()
assert os.path.exists(TEST_SETTINGS)
saved = load_settings(TEST_SETTINGS)
assert saved.get("pivot_mode") == "bbox"
editor2 = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
                 settings_path=TEST_SETTINGS)
editor2._apply_layout_settings(load_settings(TEST_SETTINGS))
assert editor2.pivot_mode == "bbox"
print("pivot mode settings round-trip OK (temp path, real settings.json untouched)")
editor.pivot_mode = "median"

# ==== 3. single-select reduction: rotate+scale on ONE entity is BYTE-
#          IDENTICAL across all 4 pivot modes, and matches the exact
#          pre-pivot-mode formula (golden compare) -- position untouched,
#          the dragged Euler/scale component set exactly as before ====
for mode in ("median", "bbox", "active", "individual"):
    solo = lib.instantiate("Crate")
    solo.transform.position = Vec3(7.0, 1.0, 3.0)
    solo.transform.rotation = Vec3(0.0, 0.0, 0.0)
    solo.transform.scale = Vec3(1.0, 1.0, 1.0)
    scene.add(solo)
    editor._set_selection([solo], active=solo)
    editor.pivot_mode = mode

    # rotate: golden formula is exactly the pre-pivot-mode single-Euler-
    # component increment (r[axis_i] = start[axis_i] + delta)
    editor._set_gizmo_mode("rotate")
    _i1, _axis1, pts1, _c1 = editor._gizmo_rings(W, H)[1]
    grab_pt = robust_y_ring_point(pts1)
    assert editor._try_grab_gizmo(grab_pt, W, H)
    cx, cy = editor.gizmo_drag["center"]
    a0 = editor.gizmo_drag["a0"]
    sign = editor.gizmo_drag["sign"]
    far = (grab_pt[0] + 80, grab_pt[1] - 80)
    ang = math.atan2(far[1] - cy, far[0] - cx)
    expected_delta = (ang - a0) * sign
    editor._update_gizmo_drag(far, FakeInput())
    assert abs(solo.transform.rotation.y - expected_delta) < 1e-9, \
        (mode, solo.transform.rotation.y, expected_delta)
    assert (solo.transform.position.x, solo.transform.position.y,
            solo.transform.position.z) == (7.0, 1.0, 3.0), \
        f"single-select rotate must never move position ({mode})"
    editor.gizmo_drag = None

    # scale: golden formula is exactly the pre-pivot-mode per-axis multiply
    editor._set_gizmo_mode("scale")
    handles = editor._gizmo_handles(W, H)
    i0, axis0, s0, s1, _c, _l = handles[0]
    assert editor._try_grab_gizmo(s1, W, H)
    dx, dy = s1[0] - s0[0], s1[1] - s0[1]
    drag_to = (s1[0] + dx * 0.7, s1[1] + dy * 0.7)
    # t is measured from "press" (== s1, the grab point _try_grab_gizmo was
    # called with above), NOT s0 -- see _update_gizmo_drag's `g["press"]`
    t = ((drag_to[0] - s1[0]) * dx + (drag_to[1] - s1[1]) * dy) / (dx * dx + dy * dy)
    expected_factor = max(0.05, 1.0 + t)
    editor._update_gizmo_drag(drag_to, FakeInput())
    assert abs(solo.transform.scale.x - expected_factor) < 1e-6, \
        (mode, solo.transform.scale.x, expected_factor)
    assert abs(solo.transform.scale.y - 1.0) < 1e-9, "only the dragged axis scales"
    assert (solo.transform.position.x, solo.transform.position.y,
            solo.transform.position.z) == (7.0, 1.0, 3.0), \
        f"single-select scale must never move position ({mode})"
    editor.gizmo_drag = None
    print(f"single-select reduction OK ({mode}): rotate.y={solo.transform.rotation.y:.4f} "
         f"(golden={expected_delta:.4f}), scale.x={solo.transform.scale.x:.4f} "
         f"(golden={expected_factor:.4f}), position untouched")

# ==== 4. rotate about MEDIAN POINT: rigid group orbits the mean; single
#          gizmo Y-drag, driven via REAL mouse motion for at least this
#          one case per spec ====
a, b, c = fresh_trio()
editor._set_selection([a, b, c], active=c)
editor.pivot_mode = "median"
editor._set_gizmo_mode("rotate")
pivot = editor._pivot_point()
assert abs(pivot.x - 2.0) < 1e-9 and abs(pivot.z - 0.0) < 1e-9, (pivot.x, pivot.z)
_i1, _axis1, pts1, _c1 = editor._gizmo_rings(W, H)[1]
grab_pt = robust_y_ring_point(pts1)
grab_screen = (int(grab_pt[0]), int(grab_pt[1]))
far_screen = (int(grab_pt[0] + 80), int(grab_pt[1] - 80))
with um.patch.object(pygame.mouse, "get_pos", return_value=grab_screen), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=grab_screen)])
    assert editor.gizmo_drag is not None and editor.gizmo_drag["mode"] == "rotate"
    with um.patch.object(pygame.mouse, "get_pos", return_value=far_screen):
        step([])
editor.gizmo_drag = None
starts = {"a": 0.0, "b": 2.0, "c": 4.0}
deltas = {n: e.transform.rotation.y for n, e in (("a", a), ("b", b), ("c", c))}
assert abs(deltas["a"] - deltas["b"]) < 1e-9 and abs(deltas["b"] - deltas["c"]) < 1e-9, deltas
assert abs(deltas["a"]) > math.radians(1.0), "must be a non-trivial rotation"
for n, e in (("a", a), ("b", b), ("c", c)):
    p = e.transform.position
    d_before = abs(starts[n] - pivot.x)
    d_after = math.hypot(p.x - pivot.x, p.z - pivot.z)
    assert abs(d_before - d_after) < 1e-4, (n, d_before, d_after)
    assert abs(p.y - 1.0) < 1e-9, "Y-axis rotate must not move Y"
print(f"rotate about Median Point OK (REAL event drag): pivot={pivot.x:.2f}, "
     f"shared delta={deltas['a']:.4f} rad, orbit radii preserved")

# ==== 5. rotate about BOUNDING BOX CENTER: pivot = combined AABB center,
#          rigid group orbits it (direct-call, math-focused) ====
a, b, c = fresh_trio()
editor._set_selection([a, b, c], active=c)
editor.pivot_mode = "bbox"
editor._set_gizmo_mode("rotate")
pivot = editor._pivot_point()
lo, hi = None, None
for e in (a, b, c):
    elo, ehi = editor._world_aabb(e)
    lo = elo if lo is None else [min(lo[i], elo[i]) for i in range(3)]
    hi = ehi if hi is None else [max(hi[i], ehi[i]) for i in range(3)]
expected = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
assert abs(pivot.x - expected[0]) < 1e-6 and abs(pivot.z - expected[2]) < 1e-6, \
    (pivot.x, pivot.z, expected)
_i1, _axis1, pts1, _c1 = editor._gizmo_rings(W, H)[1]
grab_pt = robust_y_ring_point(pts1)
assert editor._try_grab_gizmo(grab_pt, W, H)
far = (grab_pt[0] + 80, grab_pt[1] - 80)
editor._update_gizmo_drag(far, FakeInput())
deltas = {n: e.transform.rotation.y for n, e in (("a", a), ("b", b), ("c", c))}
assert abs(deltas["a"] - deltas["b"]) < 1e-9 and abs(deltas["b"] - deltas["c"]) < 1e-9
for n, e, ox in (("a", a, 0.0), ("b", b, 2.0), ("c", c, 4.0)):
    d_after = math.hypot(e.transform.position.x - pivot.x, e.transform.position.z - pivot.z)
    assert abs(abs(ox - pivot.x) - d_after) < 1e-4, (n, ox, pivot.x, d_after)
editor.gizmo_drag = None
print(f"rotate about Bounding Box Center OK: pivot=({pivot.x:.4f},{pivot.z:.4f}), "
     f"shared delta={deltas['a']:.4f} rad, orbit radii preserved")

# ==== 6. rotate about ACTIVE ELEMENT: pivot = active entity's own origin;
#          the active entity itself must not move ====
a, b, c = fresh_trio()
editor._set_selection([a, b, c], active=c)  # active = c @ x=4
editor.pivot_mode = "active"
editor._set_gizmo_mode("rotate")
pivot = editor._pivot_point()
assert abs(pivot.x - 4.0) < 1e-9, pivot.x
_i1, _axis1, pts1, _c1 = editor._gizmo_rings(W, H)[1]
grab_pt = robust_y_ring_point(pts1)
assert editor._try_grab_gizmo(grab_pt, W, H)
far = (grab_pt[0] + 80, grab_pt[1] - 80)
editor._update_gizmo_drag(far, FakeInput())
assert abs(c.transform.position.x - 4.0) < 1e-6 and abs(c.transform.position.z - 0.0) < 1e-6, \
    "the ACTIVE (pivot) entity itself must not move"
assert a.transform.position.x != 0.0 or a.transform.position.z != 0.0, "others must orbit"
deltas = {n: e.transform.rotation.y for n, e in (("a", a), ("b", b), ("c", c))}
assert abs(deltas["a"] - deltas["c"]) < 1e-9, "even the pivot entity gets the own-rotation delta"
editor.gizmo_drag = None
print(f"rotate about Active Element OK: active entity stayed at pivot "
     f"({c.transform.position.x:.2f},{c.transform.position.z:.2f}), others orbited")

# ==== 7. rotate about INDIVIDUAL ORIGINS: positions stay FIXED, only each
#          entity's own orientation changes ====
a, b, c = fresh_trio()
editor._set_selection([a, b, c], active=c)
editor.pivot_mode = "individual"
editor._set_gizmo_mode("rotate")
starts_pos = {id(e): (e.transform.position.x, e.transform.position.z) for e in (a, b, c)}
_i1, _axis1, pts1, _c1 = editor._gizmo_rings(W, H)[1]
grab_pt = robust_y_ring_point(pts1)
assert editor._try_grab_gizmo(grab_pt, W, H)
far = (grab_pt[0] + 80, grab_pt[1] - 80)
editor._update_gizmo_drag(far, FakeInput())
for e in (a, b, c):
    sx, sz = starts_pos[id(e)]
    assert abs(e.transform.position.x - sx) < 1e-12 and abs(e.transform.position.z - sz) < 1e-12, \
        "Individual Origins must NEVER move positions"
deltas = {n: e.transform.rotation.y for n, e in (("a", a), ("b", b), ("c", c))}
assert abs(deltas["a"] - deltas["b"]) < 1e-9 and abs(deltas["b"] - deltas["c"]) < 1e-9
assert abs(deltas["a"]) > math.radians(1.0)
editor.gizmo_drag = None
print(f"rotate about Individual Origins OK: positions fixed, shared own-rotation "
     f"delta={deltas['a']:.4f} rad")

# ==== 8. scale about MEDIAN POINT (uniform, center-drag) ====
a, b, c = fresh_trio()
editor._set_selection([a, b, c], active=c)
editor.pivot_mode = "median"
editor._set_gizmo_mode("scale")
pivot = editor._pivot_point()
_p, s0c, _l = editor._gizmo_center(W, H)
assert editor._try_grab_gizmo((int(s0c[0]), int(s0c[1])), W, H)
assert editor.gizmo_drag["axis_i"] == -1
editor._update_gizmo_drag((s0c[0] + 50, s0c[1]), FakeInput())
factor = a.transform.scale.x
for n, e, ox in (("a", a, 0.0), ("b", b, 2.0), ("c", c, 4.0)):
    expected_x = pivot.x + factor * (ox - pivot.x)
    assert abs(e.transform.position.x - expected_x) < 1e-6, (n, e.transform.position.x, expected_x)
    assert abs(e.transform.scale.x - factor) < 1e-9 and abs(e.transform.scale.y - factor) < 1e-9
editor.gizmo_drag = None
print(f"scale about Median Point OK: pivot.x={pivot.x:.2f}, factor={factor:.4f}, "
     f"positions match pivot+factor*(pos-pivot)")

# ==== 9. scale about BOUNDING BOX CENTER (per-axis drag) ====
a, b, c = fresh_trio()
editor._set_selection([a, b, c], active=c)
editor.pivot_mode = "bbox"
editor._set_gizmo_mode("scale")
pivot = editor._pivot_point()
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _c, _l = handles[0]
assert axis0 == (1.0, 0.0, 0.0)
assert editor._try_grab_gizmo(s1, W, H)
assert editor.gizmo_drag["axis_i"] == 0
dx, dy = s1[0] - s0[0], s1[1] - s0[1]
editor._update_gizmo_drag((s1[0] + dx * 0.7, s1[1] + dy * 0.7), FakeInput())
factor = a.transform.scale.x
for n, e, ox in (("a", a, 0.0), ("b", b, 2.0), ("c", c, 4.0)):
    expected_x = pivot.x + factor * (ox - pivot.x)
    assert abs(e.transform.position.x - expected_x) < 1e-6, (n, e.transform.position.x, expected_x)
    assert abs(e.transform.position.z - 0.0) < 1e-9, "only the scaled (X) axis should move position"
    assert abs(e.transform.scale.x - factor) < 1e-9
    assert abs(e.transform.scale.y - 1.0) < 1e-9, "only X scale should change"
editor.gizmo_drag = None
print(f"scale about Bounding Box Center OK: pivot.x={pivot.x:.4f}, factor={factor:.4f}, "
     f"only X moved/scaled")

# ==== 10. scale about ACTIVE ELEMENT ====
a, b, c = fresh_trio()
editor._set_selection([a, b, c], active=c)
editor.pivot_mode = "active"
editor._set_gizmo_mode("scale")
pivot = editor._pivot_point()
assert abs(pivot.x - 4.0) < 1e-9
_p, s0c, _l = editor._gizmo_center(W, H)
assert editor._try_grab_gizmo((int(s0c[0]), int(s0c[1])), W, H)
editor._update_gizmo_drag((s0c[0] + 50, s0c[1]), FakeInput())
assert abs(c.transform.position.x - 4.0) < 1e-6, "active (pivot) entity must not move"
factor = a.transform.scale.x
expected_x = pivot.x + factor * (0.0 - pivot.x)
assert abs(a.transform.position.x - expected_x) < 1e-6
editor.gizmo_drag = None
print(f"scale about Active Element OK: active stayed at {c.transform.position.x:.2f}, "
     f"factor={factor:.4f}")

# ==== 11. scale about INDIVIDUAL ORIGINS: positions fixed, only each
#          entity's own scale changes ====
a, b, c = fresh_trio()
editor._set_selection([a, b, c], active=c)
editor.pivot_mode = "individual"
editor._set_gizmo_mode("scale")
starts_pos = {id(e): e.transform.position.x for e in (a, b, c)}
_p, s0c, _l = editor._gizmo_center(W, H)
assert editor._try_grab_gizmo((int(s0c[0]), int(s0c[1])), W, H)
editor._update_gizmo_drag((s0c[0] + 50, s0c[1]), FakeInput())
for e in (a, b, c):
    assert abs(e.transform.position.x - starts_pos[id(e)]) < 1e-12, \
        "Individual Origins scale must NEVER move positions"
    assert abs(e.transform.scale.x - 1.0) > 1e-6, "scale must still change per entity"
editor.gizmo_drag = None
print(f"scale about Individual Origins OK: positions fixed, scale.x={a.transform.scale.x:.4f} per entity")

# ==== 12. Alt+rotate-drag with a multi-selection rotates ALL duplicated
#          copies about the pivot (the 2a-flagged gap this run fixes),
#          driven via the REAL event path ====
altA = lib.instantiate("Crate"); altA.transform.position = Vec3(4.0, 1.0, 0.0); scene.add(altA)
altB = lib.instantiate("Crate"); altB.transform.position = Vec3(6.0, 1.0, 0.0); scene.add(altB)
editor._set_selection([altA, altB], active=altB)
editor.pivot_mode = "median"  # pivot = (5, 1, 0)
editor._set_gizmo_mode("rotate")
count_before = len(scene.entities)
_i1, _axis1, pts1, _c1 = editor._gizmo_rings(W, H)[1]
grab_pos = robust_y_ring_point(pts1)
with um.patch.object(pygame.mouse, "get_pos", return_value=grab_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LALT])), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=grab_pos)])
    assert len(scene.entities) == count_before + 2, "Alt-grab must duplicate the WHOLE selection"
    dup_A = next(e for e in editor.selection if abs(e.transform.position.x - 4.0) < 1e-6)
    dup_B = editor.selected
    assert dup_B is not altB and abs(dup_B.transform.position.x - 6.0) < 1e-6
    far_pos = (int(grab_pos[0] + 80), int(grab_pos[1] - 80))
    with um.patch.object(pygame.mouse, "get_pos", return_value=far_pos):
        step([])
assert dup_A.transform.position.x != 4.0, "dup_A must have ORBITED the pivot, not just spun in place"
assert abs(dup_A.transform.rotation.y - dup_B.transform.rotation.y) < 1e-9, \
    "both duplicates must rotate by the SAME delta"
assert abs(dup_A.transform.rotation.y) > math.radians(1.0)
assert (altA.transform.position.x, altA.transform.position.z, altA.transform.rotation.y) == (4.0, 0.0, 0.0)
assert (altB.transform.position.x, altB.transform.position.z, altB.transform.rotation.y) == (6.0, 0.0, 0.0)
editor.gizmo_drag = None
print(f"Alt+rotate-multi fix OK (REAL events): both duplicates orbited pivot together, "
     f"delta={dup_A.transform.rotation.y:.4f} rad, originals untouched")

print("ALL PIVOT CHECKS PASSED")
