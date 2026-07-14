# No task in flight

Box/marquee selection was judged and squash-merged to main on 2026-07-14.
An LMB drag starting on empty viewport space draws a selection rectangle;
release selects every entity whose world-AABB screen bbox intersects it
(plain replaces, Shift adds). Does not conflict with gizmo grab, entity
click, fly-look, or UI.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

Next queued: 3D-cursor pivot — a scene-level 3D cursor point the user can
place (Blender: Shift+right-click in the viewport places it on the surface
under the cursor / a ground plane), shown as a marker, added as a FIFTH
pivot mode ("3D Cursor") so rotate/scale on the selection pivot about the
cursor. Persist the cursor position + a reset (to origin). The pivot-mode
system already has 4 modes (median/bbox/active/individual) in a
declarative toolbar cycler + _pivot_point()/_pivot_mode — extend both.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. UI/interaction tests MUST drive the real
event path; held modifier keys need pygame.key.get_pressed + get_pos +
get_pressed patched at the OS boundary (see marquee_checks.py). DX12 is
the default renderer (CPU opt-in).

Backlog: wgpu directional sun-shadow attenuation; per-pixel texturing;
flat-mode translucency; triangle-precise floor snap; rotation matching in
mesh snap; marquee for non-mesh marker glyphs (Sun/Fog); material assets
in browser folders; folder deletion.
