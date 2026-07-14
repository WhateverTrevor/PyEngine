# No task in flight

The full snapping + pivot slate is merged to main (2026-07-14):
- Snapping + Alt-drag duplicate (abf4a00)
- Multi-selection foundation (d5eefef)
- Blender-style pivot modes (this run)

The editor now has: grid/interval snap, floor snap (End), Shift-drag mesh
snap, Alt-drag gizmo duplicate (translate+rotate); multi-selection with an
active element; and four pivot modes (Median Point, Bounding Box Center,
Active Element, Individual Origins) honored by rotate AND scale on
multi-selections, selectable in the viewport toolbar and persisted.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. UI/interaction tests MUST drive the real
event path; held modifier keys need pygame.key.get_pressed + get_pos
patched at the OS boundary. DX12 is the default renderer (CPU opt-in).

Backlog: wgpu directional sun-shadow attenuation; per-pixel texturing;
flat-mode translucency; triangle-precise floor snap; rotation matching in
mesh snap; box/marquee multi-select; 3D-cursor pivot; material assets in
browser folders; folder deletion.
