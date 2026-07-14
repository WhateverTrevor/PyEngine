"""Judge checks: interval/grid snap, floor snap, snap-to-mesh, Alt-drag dup.

Quantization math and gizmo-drag application are tested via direct calls
with a FakeInput stand-in, matching the existing tests/toolbar_checks.py
precedent for gizmo-drag math. Alt+translate-drag and Alt+rotate-drag are
tested via the REAL pygame-event path (eng.input.process + editor.update)
per the project's hard rule for UI/interaction tests -- the ctx-menu crash
(tests/mat_ui_checks.py, event-driven regression section) proved direct
handler calls can miss real dispatch bugs. inp.held()/mouse_held()/mouse_pos
read raw SDL hardware state and cannot be driven by posted events alone (no
OS input backend in headless/dummy-driver mode -- verified empirically), so
the real-event sections patch pygame.mouse.get_pos/get_pressed and
pygame.key.get_pressed at the OS boundary -- the same idiom mat_ui_checks.py
already uses for pygame.mouse.get_pos in its ctx-menu regression tests --
while eng.input.process()/editor.update() themselves stay completely real.
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
from editor import Editor, build_starter_scene

TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_snap_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
               settings_path=TEST_SETTINGS)
Vec3 = engine.Vec3
W, H = eng.screen.get_size()


class FakeInput:
    """Stand-in for the held()/mouse_held() modifier checks in direct-call
    tests -- matches the FakeInput precedent in tests/toolbar_checks.py."""
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
    """pygame.key.get_pressed() returns a ScancodeWrapper that supports
    arbitrary large keycode indices (K_LALT == 1073742050) -- a plain list
    IndexErrors. This is the minimal stand-in for patching it."""
    def __init__(self, held=()):
        self._held = set(held)
    def __getitem__(self, key):
        return key in self._held


def step(events):
    """Drive one frame through the REAL input + update path."""
    eng.input.process(events)
    editor.update(eng, 1 / 60)
    eng.input.consume_edges()


def robust_y_ring_point(pts1):
    """A screen point on the Y (yaw) ring that hit-tests to axis_i == 1
    even after int-truncation (real mouse positions are ints) and a +/-1px
    jitter. The three rotation rings are projected circles that can cross
    close to each other on screen, so an arbitrary point taken from the Y
    ring's own polyline can still hit-test closest to a DIFFERENT ring --
    caller must have editor.selected/gizmo_mode="rotate" already set.
    """
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


# ==== 1. quantization math (pure) ====
assert Editor._quantize(0.37, 0.25) == 0.25   # 0.37/0.25=1.48 -> round 1
assert Editor._quantize(0.40, 0.25) == 0.5    # 0.40/0.25=1.6  -> round 2
assert Editor._quantize(1.24, 0.5) == 1.0
assert Editor._quantize(1.26, 0.5) == 1.5
assert Editor._quantize(5.0, 0.0) == 5.0      # increment<=0 passthrough
print("quantize math OK")

# ==== 2. snap toggle, per-mode increment cycling, Ctrl-invert ====
editor.snap_enabled = False
editor.snap_index = {"translate": 0, "rotate": 0, "scale": 0}
assert editor._snap_increment("translate") == 0.1
assert editor._snap_increment("rotate") == 5.0
assert editor._snap_increment("scale") == 0.1

editor.gizmo_mode = "translate"
editor._cycle_snap_increment()
assert editor._snap_increment("translate") == 0.25
editor.gizmo_mode = "rotate"
editor._cycle_snap_increment()
assert editor._snap_increment("rotate") == 15.0
assert editor._snap_increment("translate") == 0.25, "cycling rotate must not touch translate"
print("per-mode increment cycling OK (independent counters)")

assert editor._snap_active(FakeInput(ctrl=False)) is False
assert editor._snap_active(FakeInput(ctrl=True)) is True   # Ctrl inverts off->on
editor._toggle_snap()
assert editor.snap_enabled is True
assert editor._snap_active(FakeInput(ctrl=False)) is True
assert editor._snap_active(FakeInput(ctrl=True)) is False  # Ctrl inverts on->off
print("Ctrl temporary invert OK")

# ==== 3. translate grid-snap drag: result lands on absolute grid lines,
#          both world and local space (following the toolbar_checks.py
#          local/world axis drag precedent) ====
crate = next(e for e in scene.entities if e.asset_name == "Crate")
crate.transform.rotation.y = 0.0
editor.selected = crate
editor._set_gizmo_mode("translate")
editor.snap_index["translate"] = 2  # 0.5
editor.snap_enabled = True

crate.transform.position = Vec3(1.13, 0.0, 0.0)
editor.gizmo_space = "world"
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _color, length = handles[0]
assert axis0 == (1.0, 0.0, 0.0)
assert editor._try_grab_gizmo(s1, W, H)
dx, dy = s1[0] - s0[0], s1[1] - s0[1]
# small nudge so the raw (unsnapped) target is off-grid but within a snap of 2.5
editor._update_gizmo_drag((s1[0] + dx * 0.12, s1[1] + dy * 0.12), FakeInput())
gx = crate.transform.position.x
assert abs(gx / 0.5 - round(gx / 0.5)) < 1e-6, gx
print(f"world-space translate grid-snap OK: landed on grid at x={gx}")
editor.gizmo_drag = None

crate.transform.rotation.y = math.pi / 2  # 90 deg, local +X -> world -Z
crate.transform.position = Vec3(0.0, 1.0, 0.13)
editor.gizmo_space = "local"
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _color, length = handles[0]
assert editor._try_grab_gizmo(s1, W, H)
dx, dy = s1[0] - s0[0], s1[1] - s0[1]
editor._update_gizmo_drag((s1[0] + dx * 0.9, s1[1] + dy * 0.9), FakeInput())
gz = crate.transform.position.z
assert abs(gz / 0.5 - round(gz / 0.5)) < 1e-6, gz
print(f"local-space translate grid-snap OK: landed on grid at z={gz}")
editor.gizmo_drag = None
crate.transform.rotation.y = 0.0

# ==== 4. rotate grid-snap (degree increment) ====
crate.transform.position = Vec3(0.0, 1.0, 0.0)
crate.transform.rotation = Vec3(0.0, 0.0, 0.0)
editor.selected = crate
editor._set_gizmo_mode("rotate")
editor.snap_index["rotate"] = 1  # 15 deg
_i1, _axis1, pts1, _c1 = editor._gizmo_rings(W, H)[1]  # Y ring (yaw)
grab_pt = robust_y_ring_point(pts1)
assert editor._try_grab_gizmo(grab_pt, W, H)
assert editor.gizmo_drag["axis_i"] == 1
cx, cy = editor.gizmo_drag["center"]
mp_far = (cx + (grab_pt[0] - cx) + 80, cy + (grab_pt[1] - cy) - 80)
editor._update_gizmo_drag(mp_far, FakeInput())
deg = math.degrees(crate.transform.rotation.y)
assert abs(deg) > 1.0, f"drag must produce a non-trivial angle before quantizing, got {deg}"
assert abs(deg % 15.0) < 1e-4 or abs(deg % 15.0 - 15.0) < 1e-4, deg
print(f"rotate grid-snap OK: landed on a 15-deg multiple ({deg:.4f} deg)")
editor.gizmo_drag = None

# ==== 5. scale grid-snap (factor quantize -- both uniform-center and
#          per-axis handle drags) ====
crate.transform.position = Vec3(0.0, 1.0, 0.0)
crate.transform.scale = Vec3(1.0, 1.0, 1.0)
editor.selected = crate
editor._set_gizmo_mode("scale")
editor.snap_index["scale"] = 0  # 0.1
_p, s0c, _l = editor._gizmo_center(W, H)
assert editor._try_grab_gizmo((int(s0c[0]), int(s0c[1])), W, H)
assert editor.gizmo_drag["axis_i"] == -1
editor._update_gizmo_drag((s0c[0] + 37, s0c[1]), FakeInput())
fac = crate.transform.scale.x  # uniform, all axes equal
assert abs(fac / 0.1 - round(fac / 0.1)) < 1e-6, fac
print(f"uniform scale grid-snap OK: factor quantized to {fac}")
editor.gizmo_drag = None

crate.transform.scale = Vec3(1.0, 1.0, 1.0)
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _color, length = handles[0]
assert editor._try_grab_gizmo(s1, W, H)
assert editor.gizmo_drag["mode"] == "scale" and editor.gizmo_drag["axis_i"] == 0
dx, dy = s1[0] - s0[0], s1[1] - s0[1]
editor._update_gizmo_drag((s1[0] + dx * 0.7, s1[1] + dy * 0.7), FakeInput())
fac_x = crate.transform.scale.x
assert abs(fac_x / 0.1 - round(fac_x / 0.1)) < 1e-6, fac_x
print(f"per-axis scale grid-snap OK: X factor quantized to {fac_x}")
editor.gizmo_drag = None
crate.transform.scale = Vec3(1.0, 1.0, 1.0)
editor.snap_enabled = False

# ==== 6. snap settings persist through a round trip (never touches the
#          real settings.json -- settings_path is the injected temp file) ====
editor.snap_index["translate"] = 3
editor.snap_enabled = True
editor._save_settings()
assert os.path.exists(TEST_SETTINGS)
import json
saved = json.load(open(TEST_SETTINGS))
assert saved.get("snap_enabled") is True
assert saved.get("snap_index", {}).get("translate") == 3
editor2 = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
                 settings_path=TEST_SETTINGS)
from editor import load_settings
editor2._apply_layout_settings(load_settings(TEST_SETTINGS))
assert editor2.snap_enabled is True
assert editor2.snap_index["translate"] == 3
print("snap settings round-trip OK (temp path, real settings.json untouched)")
editor.snap_enabled = False
editor.snap_index = {"translate": 0, "rotate": 0, "scale": 0}

# ==== 7. _world_aabb sanity (unrotated cube matches position +/- half-extent) ====
sanity_crate = lib.instantiate("Crate")
sanity_crate.transform.position = Vec3(30.0, 4.0, 30.0)
scene.add(sanity_crate)
lo, hi = editor._world_aabb(sanity_crate)
half = 0.65
p = sanity_crate.transform.position
assert abs((hi[0] - lo[0]) - 1.3) < 1e-6, (lo, hi)
assert abs(lo[1] - (p.y - half)) < 1e-6
assert abs(hi[1] - (p.y + half)) < 1e-6
print("_world_aabb OK: unrotated crate world AABB matches position +/- half-extent")

# ==== 8. floor snap: mesh below -> lands exactly on its top; y=0 fallback;
#          a "surface" that isn't below the current footprint is ignored ====
floor_a = next(e for e in scene.entities if e.asset_name == "Crate" and e is not sanity_crate)
floor_a.transform.rotation.y = 0.0
floor_b = lib.instantiate("Crate")
floor_b.transform.position = Vec3(40.0, 0.65, 40.0)  # resting on y=0
scene.add(floor_b)

floor_a.transform.position = Vec3(40.0, 6.0, 40.0)  # floating above floor_b
editor.selected = floor_a
editor._snap_to_floor()
lo_a, hi_a = editor._world_aabb(floor_a)
_lo_b, hi_b = editor._world_aabb(floor_b)
assert abs(lo_a[1] - hi_b[1]) < 1e-6, (lo_a[1], hi_b[1])
print(f"floor snap (mesh below) OK: bottom {lo_a[1]:.6f} == neighbor top {hi_b[1]:.6f}")

floor_a.transform.position = Vec3(-90.0, 9.0, -90.0)  # nothing underneath
editor._snap_to_floor()
lo_a2, _ = editor._world_aabb(floor_a)
assert abs(lo_a2[1] - 0.0) < 1e-6, lo_a2[1]
print(f"floor snap (y=0 fallback) OK: bottom={lo_a2[1]:.6f}")

floor_c = lib.instantiate("Crate")
floor_c.transform.position = Vec3(-95.0, 0.65, -95.0)
scene.add(floor_c)
floor_a.transform.position = Vec3(-95.0, 0.325, -95.0)  # bottom is BELOW floor_c's top
editor._snap_to_floor()
lo_a3, _ = editor._world_aabb(floor_a)
assert abs(lo_a3[1] - 0.0) < 1e-6, lo_a3[1]  # nothing qualifies below -> y=0 fallback
print(f"floor snap (surface above current bottom ignored -> y=0 fallback) OK: "
      f"bottom={lo_a3[1]:.6f}")

# ==== 9. floor snap dispatch via the REAL End-hotkey event path (proves the
#          update() hotkey block actually routes End -> _snap_to_floor,
#          K_END uses inp.pressed() which IS injectable) ====
floor_a.transform.position = Vec3(40.0, 6.0, 40.0)
editor.selected = floor_a
with um.patch.object(pygame.mouse, "get_pos", return_value=(700, 400)):
    step([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_END, unicode="", mod=0)])
lo_a4, _ = editor._world_aabb(floor_a)
assert abs(lo_a4[1] - hi_b[1]) < 1e-6, lo_a4[1]
print("floor snap via REAL End-key event dispatch OK")

# ==== 10. AABB face-flush snap-to-mesh (Shift-held translate drag) ====
mesh_a = lib.instantiate("Crate")
mesh_a.transform.position = Vec3(0.0, 0.65, 50.0)
scene.add(mesh_a)
mesh_d = lib.instantiate("Crate")
mesh_d.transform.position = Vec3(3.0, 0.65, 50.0)  # 3 units away along +X
scene.add(mesh_d)
editor.selected = mesh_a
editor._set_gizmo_mode("translate")
editor.gizmo_space = "world"
editor.gizmo_drag = {
    "mode": "translate", "axis_i": 0, "axis": (1.0, 0.0, 0.0),
    "press": (0, 0), "dpx": (1.0, 0.0), "length": 1.0,
    "start": (0.0, 0.65, 50.0),
}
# raw target x=1.55 is within the 0.5 threshold of the flush position 1.70
editor._update_gizmo_drag((1.55, 0), FakeInput(shift=True))
got_x = mesh_a.transform.position.x
assert abs(got_x - 1.70) < 1e-6, got_x
assert editor.snap_feedback is not None
other, axis_idx, plane, lo2, hi2 = editor.snap_feedback
assert other is mesh_d and axis_idx == 0
assert abs(plane - 2.35) < 1e-6, plane  # mesh_d's -X face (3.0 - 0.65)
print(f"AABB face-flush mesh snap (within threshold) OK: x={got_x}, "
      f"feedback face at {plane}")

mesh_a.transform.position = Vec3(0.0, 0.65, 50.0)
editor.gizmo_drag["start"] = (0.0, 0.65, 50.0)
editor._update_gizmo_drag((0.4, 0), FakeInput(shift=True))  # far from 1.70
got_x2 = mesh_a.transform.position.x
assert abs(got_x2 - 0.4) < 1e-6, got_x2
assert editor.snap_feedback is None
print(f"AABB face-flush mesh snap (beyond threshold, ignored) OK: x={got_x2}")

mesh_a.transform.position = Vec3(0.0, 0.65, 50.0)
editor.gizmo_drag["start"] = (0.0, 0.65, 50.0)
editor.snap_enabled = False
editor._update_gizmo_drag((1.55, 0), FakeInput(shift=False))  # no Shift -> no face-snap
got_x3 = mesh_a.transform.position.x
assert abs(got_x3 - 1.55) < 1e-6, got_x3
assert editor.snap_feedback is None
print(f"no Shift held -> face-snap does not fire OK: x={got_x3}")
editor.gizmo_drag = None

# ==== 11. Alt+translate-drag via REAL pygame event injection ====
# a dedicated entity, positioned in-view of the fixed test camera (matching
# the z~=0 area sections 3-5 already proved projects to valid screen coords
# -- _gizmo_handles/_gizmo_rings return [] for anything the camera can't
# project, which would otherwise silently no-op the grab below)
alt_crate = lib.instantiate("Crate")
scene.add(alt_crate)
alt_crate.transform.rotation.y = 0.0
alt_crate.transform.position = Vec3(0.0, 1.0, 0.0)
editor.selected = alt_crate
editor._set_gizmo_mode("translate")
editor.gizmo_space = "world"
count_before = len(scene.entities)
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _color, length = handles[0]
grab_pos = (int(s1[0]), int(s1[1]))

with um.patch.object(pygame.mouse, "get_pos", return_value=grab_pos), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LALT])), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=grab_pos)])
    assert len(scene.entities) == count_before + 1, "Alt-grab must duplicate via the real event path"
    dup_t = editor.selected
    assert dup_t is not alt_crate
    assert (dup_t.transform.position.x, dup_t.transform.position.y,
            dup_t.transform.position.z) == (0.0, 1.0, 0.0), "dup must start at the SAME transform"
    dx, dy = s1[0] - s0[0], s1[1] - s0[1]
    move_pos = (int(s1[0] + dx), int(s1[1] + dy))
    with um.patch.object(pygame.mouse, "get_pos", return_value=move_pos):
        step([])  # continuation frame: mouse still "held" per the get_pressed patch
    assert (dup_t.transform.position.x, dup_t.transform.position.y,
            dup_t.transform.position.z) != (0.0, 1.0, 0.0), "the duplicate must have moved"
    assert (alt_crate.transform.position.x, alt_crate.transform.position.y,
            alt_crate.transform.position.z) == (0.0, 1.0, 0.0), "the ORIGINAL must be untouched"
print(f"Alt+translate-drag via REAL events OK: entities {count_before}->{len(scene.entities)}, "
      f"duplicate moved to x={dup_t.transform.position.x:.3f}, original unmoved")
editor.gizmo_drag = None

# ==== 12. Alt+rotate-drag via REAL pygame event injection ====
alt_crate.transform.position = Vec3(4.0, 1.0, 0.0)
alt_crate.transform.rotation = Vec3(0.0, 0.0, 0.0)
editor.selected = alt_crate
editor._set_gizmo_mode("rotate")
count_before2 = len(scene.entities)
_i1, _axis1, pts1, _c1 = editor._gizmo_rings(W, H)[1]  # Y ring
grab_pos2 = robust_y_ring_point(pts1)

with um.patch.object(pygame.mouse, "get_pos", return_value=grab_pos2), \
     um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys([pygame.K_LALT])), \
     um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
    step([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=grab_pos2)])
    assert len(scene.entities) == count_before2 + 1, "Alt-grab (rotate) must duplicate via real events"
    dup_r = editor.selected
    assert dup_r is not alt_crate
    assert (dup_r.transform.rotation.x, dup_r.transform.rotation.y,
            dup_r.transform.rotation.z) == (0.0, 0.0, 0.0)
    cx, cy = editor.gizmo_drag["center"]
    far_pos = (int(grab_pos2[0] + 80), int(grab_pos2[1] - 80))
    with um.patch.object(pygame.mouse, "get_pos", return_value=far_pos):
        step([])
    assert abs(dup_r.transform.rotation.y) > math.radians(1.0), \
        f"drag must produce a non-trivial rotation, got {dup_r.transform.rotation.y}"
    assert (alt_crate.transform.rotation.x, alt_crate.transform.rotation.y,
            alt_crate.transform.rotation.z) == (0.0, 0.0, 0.0), "the ORIGINAL must be untouched"
print(f"Alt+rotate-drag via REAL events OK: entities {count_before2}->{len(scene.entities)}, "
      f"duplicate rotated to y={dup_r.transform.rotation.y:.4f} rad, original unmoved")
editor.gizmo_drag = None

# ==== 13. Alt-duplicate composes with grid snap; scale is out of scope;
#          inp=None back-compat (direct-call sites) ====
alt_crate.transform.position = Vec3(8.13, 1.0, 0.0)
editor.selected = alt_crate
editor._set_gizmo_mode("translate")
editor.gizmo_space = "world"
editor.snap_index["translate"] = 2  # 0.5
editor.snap_enabled = True
count_before3 = len(scene.entities)
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _c, _l = handles[0]
assert editor._try_grab_gizmo(s1, W, H, FakeInput(alt=True))
dup_snap = editor.selected
assert len(scene.entities) == count_before3 + 1
dxg, dyg = s1[0] - s0[0], s1[1] - s0[1]
editor._update_gizmo_drag((s1[0] + dxg * 3, s1[1] + dyg * 3), FakeInput(alt=True))
gxs = dup_snap.transform.position.x
assert abs(gxs / 0.5 - round(gxs / 0.5)) < 1e-9, gxs
print(f"Alt-drag + grid-snap compose OK: duplicate landed on grid at x={gxs}")
editor.gizmo_drag = None
editor.snap_enabled = False

alt_crate.transform.position = Vec3(10.0, 1.0, 0.0)
alt_crate.transform.scale = Vec3(1.0, 1.0, 1.0)
editor.selected = alt_crate
editor._set_gizmo_mode("scale")
count_before4 = len(scene.entities)
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _c, _l = handles[0]
assert editor._try_grab_gizmo(s1, W, H, FakeInput(alt=True))
assert len(scene.entities) == count_before4, "Alt+scale must NOT duplicate (out of spec scope)"
assert editor.selected is alt_crate
print("Alt+scale grab OK: no duplicate (out of milestone-3 scope)")
editor.gizmo_drag = None

alt_crate.transform.position = Vec3(12.0, 1.0, 0.0)
editor.selected = alt_crate
editor._set_gizmo_mode("translate")
count_before5 = len(scene.entities)
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _c, _l = handles[0]
assert editor._try_grab_gizmo(s1, W, H)  # no inp arg at all -- back-compat
assert len(scene.entities) == count_before5
print("inp omitted (back-compat with pre-existing direct-call test sites) OK")
editor.gizmo_drag = None

print("ALL SNAP CHECKS PASSED")
