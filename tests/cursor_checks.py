"""Judge checks: Blender-style 3D cursor (a placeable world-space point) and
its "3D Cursor" pivot mode, run 3 of the pivot slate (foundation: run 2b's
pivot modes, see tests/pivot_checks.py).

Per the project's hard rule for UI/interaction tests: cursor placement (K)
and reset (Shift+C) are driven through the REAL event path
(eng.input.process + editor.update), matching the precedent in
tests/marquee_checks.py and tests/pivot_checks.py. Held modifier keys
(Shift) aren't reachable via synthetic SDL events under the dummy video
driver, so those frames patch pygame.key.get_pressed / pygame.mouse.get_pos
at the OS boundary while eng.input.process()/editor.update() themselves
stay completely real. Rotate-about-cursor orbit math (this run's core
addition, and the semantic differentiator vs every other pivot mode) is
driven via REAL mouse-drag events for both a multi- and a single-selection
(matching pivot_checks.py section 4's precedent); the scale case follows
pivot_checks.py's direct-call precedent for pure orbit/scale math.
"""
import math
import os
import sys
import tempfile
import types
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import pygame

import engine
from editor import Editor, build_starter_scene, load_settings

TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_cursor_settings.json")
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
    """pygame.key.get_pressed() stand-in (see pivot_checks.py precedent)."""
    def __init__(self, held=()):
        self._held = set(held)
    def __getitem__(self, key):
        return key in self._held


def step(ed, events):
    eng.input.process(events)
    ed.update(eng, 1 / 60)
    eng.input.consume_edges()


def robust_y_ring_point(ed, pts1, w, h):
    """A screen point on the Y (yaw) ring that hit-tests to axis_i == 1
    even after int-truncation/jitter -- see pivot_checks.py precedent."""
    for p in pts1:
        if p is None:
            continue
        ip = (int(p[0]), int(p[1]))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                got = ed._try_grab_gizmo((ip[0] + dx, ip[1] + dy), w, h)
                axis_i = ed.gizmo_drag["axis_i"] if got else None
                ed.gizmo_drag = None
                if not got or axis_i != 1:
                    ok = False
        if ok:
            return ip
    raise AssertionError("no Y-ring screen point robustly hit-tests to axis_i==1")


def fresh_trio(scn):
    """3 crates at x=0,2,4 (y=1, z=0) -- the shared multi-select fixture."""
    a = lib.instantiate("Crate"); a.transform.position = Vec3(0.0, 1.0, 0.0); scn.add(a)
    b = lib.instantiate("Crate"); b.transform.position = Vec3(2.0, 1.0, 0.0); scn.add(b)
    c = lib.instantiate("Crate"); c.transform.position = Vec3(4.0, 1.0, 0.0); scn.add(c)
    for e in (a, b, c):
        e.transform.rotation = Vec3(0.0, 0.0, 0.0)
        e.transform.scale = Vec3(1.0, 1.0, 1.0)
    return a, b, c


def real_rotate_y_drag(ed, w, h):
    """Grab the Y ring and drag it via REAL mouse events; returns the pivot
    _pivot_point() reported at grab time (before the drag moved anything)."""
    pivot = ed._pivot_point()
    _i1, _axis1, pts1, _c1 = ed._gizmo_rings(w, h)[1]
    grab_pt = robust_y_ring_point(ed, pts1, w, h)
    grab_screen = (int(grab_pt[0]), int(grab_pt[1]))
    far_screen = (int(grab_pt[0] + 80), int(grab_pt[1] - 80))
    with um.patch.object(pygame.mouse, "get_pos", return_value=grab_screen), \
         um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step(ed, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=grab_screen)])
        assert ed.gizmo_drag is not None and ed.gizmo_drag["mode"] == "rotate"
        with um.patch.object(pygame.mouse, "get_pos", return_value=far_screen):
            step(ed, [])
    ed.gizmo_drag = None
    return pivot


# ==== 1. default cursor is the world origin ====
assert (editor.cursor3d.x, editor.cursor3d.y, editor.cursor3d.z) == (0.0, 0.0, 0.0)
print("default 3D cursor at origin OK")

# ==== 2. "cursor" is a 5th pivot mode; _pivot_point() returns cursor3d for
#          BOTH single- and multi-selections -- the key semantic difference
#          from median/bbox/active/individual, which all collapse a single
#          selection down to its own origin ====
assert "cursor" in editor._PIVOT_MODES
crate_solo = lib.instantiate("Crate")
crate_solo.transform.position = Vec3(7.0, 1.0, 3.0)
scene.add(crate_solo)
editor.cursor3d = Vec3(2.0, 0.5, -1.0)
editor.pivot_mode = "cursor"
assert editor._pivot_label() == "3D Cursor"
editor._set_selection([crate_solo], active=crate_solo)
p = editor._pivot_point()
assert abs(p.x - 2.0) < 1e-9 and abs(p.y - 0.5) < 1e-9 and abs(p.z - (-1.0)) < 1e-9, \
    (p.x, p.y, p.z)
print("_pivot_point() == cursor3d for a SINGLE selection in cursor mode OK")

a, b, c = fresh_trio(scene)
editor._set_selection([a, b, c], active=c)
p = editor._pivot_point()
assert abs(p.x - 2.0) < 1e-9 and abs(p.z - (-1.0)) < 1e-9
print("_pivot_point() == cursor3d for a MULTI selection in cursor mode OK")

# ==== 3. the other 4 modes are unaffected by a non-origin cursor3d: a
#          single selection still reduces to its own origin (regression
#          guard for _pivot_point's single-select shortcut) ====
for mode in ("median", "bbox", "active", "individual"):
    editor.pivot_mode = mode
    editor._set_selection([crate_solo], active=crate_solo)
    p = editor._pivot_point()
    assert abs(p.x - 7.0) < 1e-9 and abs(p.z - 3.0) < 1e-9, (mode, p.x, p.z)
print("other 4 pivot modes unchanged OK: single-select still own-origin "
      "(a non-origin cursor3d had no effect)")
editor.pivot_mode = "cursor"

# ============================================================================
# Setup P: dedicated straight-down camera for exact, hand-verifiable
# placement raycasts (matches marquee_checks.py's "Setup B" precedent).
# ============================================================================
camera_p = engine.Camera(position=Vec3(0.0, 10.0, 0.0), yaw=0.0, pitch=-math.pi / 2.0)
scene_p = engine.Scene()
crate_p = lib.instantiate("Crate")
crate_p.transform.position = Vec3(0.0, 1.0, 0.0)
scene_p.add(crate_p)
editor_p = Editor(engine, eng, scene_p, camera_p, lib, "scenes/scene.json",
                  settings_path=TEST_SETTINGS)
mp_center = (W // 2, H // 2)  # mouse ray at screen center == straight down

# ==== 4. K places the 3D cursor on the surface under the mouse (a real
#          raycast hit), and falls back to the y=0 ground plane when
#          nothing is hit -- driven via a REAL K keydown event ====
lo, hi = editor_p._world_aabb(crate_p)
expected_top_y = float(hi[1])
editor_p.cursor3d = Vec3(999.0, 999.0, 999.0)  # sentinel -- must get overwritten
with um.patch.object(pygame.mouse, "get_pos", return_value=mp_center):
    step(editor_p, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_k, unicode="k", mod=0)])
assert abs(editor_p.cursor3d.x - 0.0) < 1e-6 and abs(editor_p.cursor3d.z - 0.0) < 1e-6
assert abs(editor_p.cursor3d.y - expected_top_y) < 1e-6, (editor_p.cursor3d.y, expected_top_y)
print(f"K placement (REAL event): surface hit OK -> "
      f"({editor_p.cursor3d.x:.3f}, {editor_p.cursor3d.y:.3f}, {editor_p.cursor3d.z:.3f}) "
      f"== crate top ({expected_top_y:.3f})")

camera_p.position = Vec3(5.0, 10.0, 3.0)  # off the crate's small XZ footprint
editor_p.cursor3d = Vec3(999.0, 999.0, 999.0)
with um.patch.object(pygame.mouse, "get_pos", return_value=mp_center):
    step(editor_p, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_k, unicode="k", mod=0)])
assert abs(editor_p.cursor3d.x - 5.0) < 1e-6 and abs(editor_p.cursor3d.z - 3.0) < 1e-6 \
       and abs(editor_p.cursor3d.y - 0.0) < 1e-6, \
    (editor_p.cursor3d.x, editor_p.cursor3d.y, editor_p.cursor3d.z)
print(f"K placement (REAL event): y=0 ground-plane fallback OK -> "
      f"({editor_p.cursor3d.x:.3f}, {editor_p.cursor3d.y:.3f}, {editor_p.cursor3d.z:.3f})")

# ==== 5. Shift+C resets the cursor to the origin (Blender convention), via
#          a REAL event; plain C (no Shift) still toggles fly-collision and
#          does NOT reset the cursor -- proves K/Shift+C don't collide with
#          any existing binding (RMB stays the fly-look toggle, untouched) ====
editor_p.fly = types.SimpleNamespace(collide=True, looking=False)
editor_p.cursor3d = Vec3(4.0, 2.0, -6.0)
with um.patch.object(pygame.mouse, "get_pos", return_value=mp_center), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()):
    step(editor_p, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c, unicode="c", mod=0)])
assert editor_p.fly.collide is False, "plain C (no Shift) must still toggle fly collision"
assert (editor_p.cursor3d.x, editor_p.cursor3d.y, editor_p.cursor3d.z) == (4.0, 2.0, -6.0), \
    "plain C must NOT reset the cursor"
print("plain C (no Shift) toggles fly-collision only OK (no conflict with cursor reset)")

with um.patch.object(pygame.mouse, "get_pos", return_value=mp_center), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LSHIFT])):
    step(editor_p, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c, unicode="c", mod=0)])
assert (editor_p.cursor3d.x, editor_p.cursor3d.y, editor_p.cursor3d.z) == (0.0, 0.0, 0.0), \
    "Shift+C must reset the cursor to the origin"
assert editor_p.fly.collide is False, "Shift+C must NOT also toggle fly-collision"
print("Shift+C resets 3D cursor to origin OK (REAL event), no double-fire of collide toggle")

# Edit menu action does the same thing (menu entry wired to the same method)
editor_p.cursor3d = Vec3(9.0, 9.0, 9.0)
editor_p._reset_cursor3d()
assert (editor_p.cursor3d.x, editor_p.cursor3d.y, editor_p.cursor3d.z) == (0.0, 0.0, 0.0)
assert dict(editor_p._menu_defs())["Edit"] == editor_p._menu_defs()["Edit"]  # sanity
assert any(label == "Reset 3D Cursor" for label, _action, _en in editor_p._menu_defs()["Edit"])
print("Edit > Reset 3D Cursor menu entry present and wired OK")

# ==== 6. cursor3d + pivot_mode "cursor" round-trip through settings.json
#          (temp path only -- the real settings.json is never touched) ====
editor_p.pivot_mode = "cursor"
editor_p.cursor3d = Vec3(1.5, -2.25, 8.0)
editor_p._save_settings()
saved = load_settings(TEST_SETTINGS)
assert saved.get("pivot_mode") == "cursor"
assert saved.get("cursor3d") == [1.5, -2.25, 8.0], saved.get("cursor3d")
editor_reload = Editor(engine, eng, scene_p, camera_p, lib, "scenes/scene.json",
                       settings_path=TEST_SETTINGS)
editor_reload._apply_layout_settings(load_settings(TEST_SETTINGS))
assert editor_reload.pivot_mode == "cursor"
rc = editor_reload.cursor3d
assert abs(rc.x - 1.5) < 1e-9 and abs(rc.y - (-2.25)) < 1e-9 and abs(rc.z - 8.0) < 1e-9
print("cursor3d + pivot_mode='cursor' settings round-trip OK (temp path, real settings.json untouched)")
editor_p.pivot_mode = "median"
editor_p.cursor3d = Vec3(0.0, 0.0, 0.0)

# ==== 7. rotate about the 3D cursor orbits a MULTI-selection, via REAL
#          mouse-drag events (matches pivot_checks.py section 4's precedent) ====
a, b, c = fresh_trio(scene)
editor._set_selection([a, b, c], active=c)
editor.pivot_mode = "cursor"
editor.cursor3d = Vec3(2.0, 0.0, 0.0)
editor._set_gizmo_mode("rotate")
pivot = real_rotate_y_drag(editor, W, H)
assert abs(pivot.x - 2.0) < 1e-9 and abs(pivot.z - 0.0) < 1e-9
deltas = {n: e.transform.rotation.y for n, e in (("a", a), ("b", b), ("c", c))}
assert abs(deltas["a"] - deltas["b"]) < 1e-9 and abs(deltas["b"] - deltas["c"]) < 1e-9
assert abs(deltas["a"]) > math.radians(1.0), "must be a non-trivial rotation"
for n, e, ox in (("a", a, 0.0), ("b", b, 2.0), ("c", c, 4.0)):
    d_before = abs(ox - pivot.x)
    d_after = math.hypot(e.transform.position.x - pivot.x, e.transform.position.z - pivot.z)
    assert abs(d_before - d_after) < 1e-4, (n, d_before, d_after)
print(f"rotate about 3D Cursor OK (MULTI selection, REAL events): pivot={pivot.x:.2f}, "
     f"shared delta={deltas['a']:.4f} rad, orbit radii preserved")

# ==== 8. rotate about the 3D cursor orbits a SINGLE selection -- position
#          MOVES -- the key differentiator vs every other pivot mode (which
#          all reduce a lone object to spinning about its own origin), via
#          REAL mouse-drag events ====
solo = lib.instantiate("Crate")
solo.transform.position = Vec3(0.0, 1.0, 0.0)
solo.transform.rotation = Vec3(0.0, 0.0, 0.0)
scene.add(solo)
editor._set_selection([solo], active=solo)
editor.pivot_mode = "cursor"
editor.cursor3d = Vec3(2.0, 0.0, 0.0)
editor._set_gizmo_mode("rotate")
pivot = real_rotate_y_drag(editor, W, H)
assert abs(pivot.x - 2.0) < 1e-9, "single-select must NOT collapse to own origin in cursor mode"
assert solo.transform.position.x != 0.0 or solo.transform.position.z != 0.0, \
    "SINGLE selection must ORBIT the cursor, not spin in place -- the differentiator"
d_after = math.hypot(solo.transform.position.x - 2.0, solo.transform.position.z - 0.0)
assert abs(d_after - 2.0) < 1e-4, ("orbit radius must be preserved", d_after)
assert abs(solo.transform.rotation.y) > math.radians(1.0), "own orientation still changes too"
print(f"rotate about 3D Cursor OK (SINGLE selection, REAL events, the differentiator): "
     f"pos moved to ({solo.transform.position.x:.3f},{solo.transform.position.z:.3f}), "
     f"orbit radius={d_after:.3f} (preserved), rot.y={solo.transform.rotation.y:.4f} rad")

# ==== 9. scale about the 3D cursor moves a SINGLE selection's position
#          relative to the cursor (direct-call, math-focused -- matches
#          pivot_checks.py's precedent for scale sections) ====
solo.transform.position = Vec3(0.0, 1.0, 0.0)
solo.transform.scale = Vec3(1.0, 1.0, 1.0)
editor._set_selection([solo], active=solo)
editor.pivot_mode = "cursor"
editor.cursor3d = Vec3(2.0, 0.0, 0.0)
editor._set_gizmo_mode("scale")
pivot = editor._pivot_point()
assert abs(pivot.x - 2.0) < 1e-9
_p, s0c, _l = editor._gizmo_center(W, H)
assert editor._try_grab_gizmo((int(s0c[0]), int(s0c[1])), W, H)
assert editor.gizmo_drag["axis_i"] == -1
editor._update_gizmo_drag((s0c[0] + 50, s0c[1]), FakeInput())
factor = solo.transform.scale.x
expected_x = pivot.x + factor * (0.0 - pivot.x)
assert abs(factor - 1.0) > 1e-6, "scale factor must actually change"
assert abs(solo.transform.position.x - expected_x) < 1e-6, (solo.transform.position.x, expected_x)
editor.gizmo_drag = None
print(f"scale about 3D Cursor OK (SINGLE selection): pivot.x={pivot.x:.2f}, factor={factor:.4f}, "
     f"position matches pivot+factor*(pos-pivot)")

# ==== 10. marker draw doesn't crash + visual screenshot (matches
#           marquee_checks.py's screenshot precedent) ====
editor.cursor3d = Vec3(2.0, 1.0, 0.0)
surf = eng.screen
surf.fill((15, 16, 20))
editor.draw(eng)
out_png = os.path.join(tempfile.gettempdir(), "judge_cursor.png")
pygame.image.save(surf, out_png)
print(f"screenshot saved: {out_png}")

print("ALL CURSOR CHECKS PASSED")
