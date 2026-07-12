"""Judge checks: viewport toolbar (mode/space buttons) + Details transform fields."""
import math
import os
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import pygame

import engine
from editor import Editor, build_starter_scene

OUT = os.path.join(tempfile.gettempdir(), "judge_toolbar.png")

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json")
W, H = eng.screen.get_size()

crate = next(e for e in scene.entities if e.asset_name == "Crate")
editor.selected = crate

# ---- 1. toolbar rect math + hit-test consistency ----
layout = editor._layout(W, H)
toolbar_rect = editor._viewport_toolbar_rect(layout["viewport"])
assert toolbar_rect.x == layout["viewport"].x and toolbar_rect.y == layout["viewport"].y
assert toolbar_rect.width == layout["viewport"].width
rects = editor._toolbar_button_rects(toolbar_rect)
labels = [b["id"] for b, _ in rects]
assert labels == ["translate", "rotate", "scale", "space"]
# rects are strictly left-to-right, non-overlapping
for (b0, r0), (b1, r1) in zip(rects, rects[1:]):
    assert r1.x >= r0.right, (b0["id"], b1["id"], r0, r1)
print(f"toolbar layout OK: {[ (b['id'], tuple(r)) for b, r in rects ]}")

# click-does-not-pick-through: clicking a toolbar button must not also select
# a viewport entity underneath it
editor.selected = None
translate_rect = dict(rects)[{"id": "translate"}["id"]] if False else None
translate_rect = next(r for b, r in rects if b["id"] == "rotate")
mp = (translate_rect.centerx, translate_rect.centery)
assert layout["viewport"].collidepoint(mp)  # the toolbar sits inside the viewport rect
hit = editor._click_viewport_toolbar(mp, toolbar_rect)
assert hit is True
assert editor.gizmo_mode == "rotate"
assert editor.selected is None, "toolbar click must not fall through to viewport picking"
print("toolbar click-through guard OK: mode set, no entity selected")

# ---- 2. mode sync both ways: hotkey -> toolbar state, click -> gizmo mode ----
editor._set_gizmo_mode("translate")
assert editor.gizmo_mode == "translate"
active_ids = [b["id"] for b, r in editor._toolbar_button_rects(toolbar_rect) if b["active"]()]
assert active_ids == ["translate"], active_ids
scale_rect = next(r for b, r in rects if b["id"] == "scale")
assert editor._click_viewport_toolbar((scale_rect.centerx, scale_rect.centery), toolbar_rect)
assert editor.gizmo_mode == "scale"
active_ids = [b["id"] for b, r in editor._toolbar_button_rects(toolbar_rect) if b["active"]()]
assert active_ids == ["scale"], active_ids
print("mode sync OK: click <-> gizmo_mode both directions")

# world/local toggle
assert editor.gizmo_space == "world"
space_rect = next(r for b, r in rects if b["id"] == "space")
assert editor._click_viewport_toolbar((space_rect.centerx, space_rect.centery), toolbar_rect)
assert editor.gizmo_space == "local"
assert editor._click_viewport_toolbar((space_rect.centerx, space_rect.centery), toolbar_rect)
assert editor.gizmo_space == "world"
print("world/local toggle OK")

# ---- 3. local-vs-world translation math on a rotated entity ----
editor.selected = crate  # restore selection after the click-through test above
editor._set_gizmo_mode("translate")
crate.transform.rotation.y = math.pi / 2  # 90 degrees about Y
crate.transform.position = engine.Vec3(0.0, 1.0, 0.0)

# World space: axis 0 (red, nominally +X) must be the world +X unit vector
editor.gizmo_space = "world"
w_axes = editor._axis_defs(crate)
ax0 = w_axes[0][0]
assert abs(ax0[0] - 1.0) < 1e-9 and abs(ax0[2]) < 1e-9, ax0

# Local space: rotated 90 deg about Y turns local +X into world -Z (since
# rotation_y(pi/2) maps (1,0,0) -> (cos, 0, -sin) = (0,0,-1))
editor.gizmo_space = "local"
l_axes = editor._axis_defs(crate)
lax0 = l_axes[0][0]
assert abs(lax0[0]) < 1e-6 and abs(lax0[2] - (-1.0)) < 1e-6, lax0
print(f"local axis rotation OK: world +X -> local handle {lax0} at yaw=90deg")

# Drive an actual drag on axis 0 and confirm the position delta follows the
# expected direction in each space.
def drag_axis0(space):
    editor.gizmo_space = space
    crate.transform.position = engine.Vec3(0.0, 1.0, 0.0)
    ok = editor._try_grab_gizmo((0, 0), W, H)  # placeholder, replaced below
    return ok

# _try_grab_gizmo needs real screen-projected handle positions; drive it
# through the same handles list the gizmo drawing uses instead of guessing
# screen coords, so this exercises the real hit-test + drag math.
editor.gizmo_space = "local"
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _color, length = handles[0]
assert editor._try_grab_gizmo(s1, W, H)
assert editor.gizmo_drag["mode"] == "translate" and editor.gizmo_drag["axis_i"] == 0
start_pos = (crate.transform.position.x, crate.transform.position.y, crate.transform.position.z)
# drag the mouse further along the same screen-space handle direction
dx, dy = s1[0] - s0[0], s1[1] - s0[1]
mp2 = (s1[0] + dx, s1[1] + dy)
editor._update_gizmo_drag(mp2)
delta = (crate.transform.position.x - start_pos[0],
        crate.transform.position.y - start_pos[1],
        crate.transform.position.z - start_pos[2])
# delta should be parallel to the local axis (0,0,-1ish), i.e. dominant -Z, ~0 X
assert abs(delta[0]) < 1e-6, delta
assert delta[2] < -1e-6, delta
print(f"local-space drag OK: delta={delta} follows rotated local +X axis")

crate.transform.position = engine.Vec3(0.0, 1.0, 0.0)
editor.gizmo_space = "world"
handles = editor._gizmo_handles(W, H)
i0, axis0, s0, s1, _color, length = handles[0]
assert axis0 == (1.0, 0.0, 0.0)
assert editor._try_grab_gizmo(s1, W, H)
dx, dy = s1[0] - s0[0], s1[1] - s0[1]
mp2 = (s1[0] + dx, s1[1] + dy)
start_pos = (crate.transform.position.x, crate.transform.position.y, crate.transform.position.z)
editor._update_gizmo_drag(mp2)
delta = (crate.transform.position.x - start_pos[0],
        crate.transform.position.y - start_pos[1],
        crate.transform.position.z - start_pos[2])
assert delta[0] > 1e-6, delta
assert abs(delta[2]) < 1e-6, delta
print(f"world-space drag OK: delta={delta} follows world +X axis")
editor.gizmo_drag = None

# ---- 4. Details XYZ field parse/commit/cancel/revert + rotation degrees ----
crate.transform.position = engine.Vec3(1.5, 2.0, -3.25)
crate.transform.rotation = engine.Vec3(0.0, math.pi / 2, 0.0)  # 90 deg yaw
crate.transform.scale = engine.Vec3(1.0, 1.0, 1.0)

content = editor._panel_content_rect("details", layout)
if content is None:
    content = pygame.Rect(0, 0, 260, 400)

rows = editor._transform_rows(crate)
assert rows[0]["label"] == "Position"
assert rows[1]["label"] == "Rotation"
assert rows[2]["label"] == "Scale"
# rotation displayed in degrees, round-trips back to radians
rot_deg = rows[1]["fields"][1]["get"]()  # y axis
assert abs(rot_deg - 90.0) < 1e-6, rot_deg
rows[1]["fields"][1]["set"](45.0)
assert abs(crate.transform.rotation.y - math.radians(45.0)) < 1e-9
print(f"rotation degree round-trip OK: 90deg stored as {math.pi/2:.4f} rad, "
      f"set(45) -> {crate.transform.rotation.y:.4f} rad")

# click-to-edit -> commit via Enter
crate.transform.rotation.y = math.pi / 2
rr = editor._transform_row_rect(content, 0)  # Position row
fr = editor._transform_field_rects(rr)[0]    # X field
assert editor._click_transform_fields(fr.center, content, crate)
assert editor.editing_field == ("Position", "x")
assert editor.edit_buffer == "1.5"
editor.edit_buffer = "9.25"
editor._commit_edit_field()
assert editor.editing_field is None
assert abs(crate.transform.position.x - 9.25) < 1e-9
print("commit OK: typed '9.25' -> position.x == 9.25")

# cancel (Esc) leaves the value untouched
rr = editor._transform_row_rect(content, 0)
fr = editor._transform_field_rects(rr)[1]  # Y field
assert editor._click_transform_fields(fr.center, content, crate)
before = crate.transform.position.y
editor.edit_buffer = "123"
editor._cancel_edit_field()
assert editor.editing_field is None
assert crate.transform.position.y == before
print("cancel OK: Esc leaves value untouched")

# invalid text reverts (no exception, old value kept)
fr = editor._transform_field_rects(rr)[2]  # Z field
assert editor._click_transform_fields(fr.center, content, crate)
before = crate.transform.position.z
editor.edit_buffer = "-3.--"
editor._commit_edit_field()
assert editor.editing_field is None
assert crate.transform.position.z == before, "invalid text must not write a value"
print("invalid-text revert OK: unparsable buffer leaves old value")

# clicking another field commits the current one (simulated: begin edit A,
# then begin edit B without an explicit commit call -- _begin_edit_field
# commits the outgoing field itself)
rrp = editor._transform_row_rect(content, 0)
frx = editor._transform_field_rects(rrp)[0]
assert editor._click_transform_fields(frx.center, content, crate)
editor.edit_buffer = "42"
fry = editor._transform_field_rects(rrp)[1]
assert editor._click_transform_fields(fry.center, content, crate)  # switches field
assert editor.editing_field == ("Position", "y")
assert abs(crate.transform.position.x - 42.0) < 1e-9, \
    "switching fields must commit the previously-open field"
print("field-switch commit OK: opening field Y committed pending edit on X")
editor.editing_field = None

# typed-character filter: only digits/minus/dot accepted
class FakeInput:
    def __init__(self, text="", keys=()):
        self.text_typed = text
        self._keys = set(keys)
    def pressed(self, k):
        return k in self._keys

editor.editing_field = ("Position", "x")
editor.edit_buffer = ""
editor._update_edit_field(FakeInput(text="1a2.-3xb"))
assert editor.edit_buffer == "12.-3", editor.edit_buffer
print(f"typed-char filter OK: '1a2.-3xb' -> '{editor.edit_buffer}'")
editor.editing_field = None

# ---- 5. render + screenshot for visual confirmation ----
crate.transform.position = engine.Vec3(1.5, 2.0, -3.25)
crate.transform.rotation = engine.Vec3(0.0, math.radians(30.0), 0.0)
editor.selected = crate
editor.gizmo_mode = "translate"
editor.gizmo_space = "local"
surf = eng.screen
editor.draw(type("E", (), {"screen": surf, "input": eng.input})())
pygame.image.save(surf, OUT)
print(f"screenshot saved: {OUT}")

print("ALL TOOLBAR/DETAILS CHECKS PASSED")
