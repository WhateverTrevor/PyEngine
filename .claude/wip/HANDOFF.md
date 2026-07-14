# No task in flight

3D-cursor pivot was judged and squash-merged to main on 2026-07-14. The
editor now has a placeable 3D cursor (K places it on the surface under the
mouse / y=0 fallback; Shift+C resets to origin; Edit-menu "Reset 3D
Cursor"; drawn as a ringed crosshair marker), added as a FIFTH pivot mode
("3D Cursor") — rotate/scale orbit the selection about the cursor for ANY
selection size, including a single object (unlike the other four modes
which reduce single-select to own-origin). Persisted in settings.json.

This completes the marquee + 3D-cursor follow-up pair, and the whole
snapping/multiselect/pivot editing-workflow slate.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. UI/interaction tests MUST drive the real
event path; held modifier keys need pygame.key.get_pressed +
mouse.get_pos + mouse.get_pressed patched at the OS boundary. DX12 is the
default renderer (CPU opt-in). The full battery is now SEVENTEEN suites.

Backlog: wgpu directional sun-shadow attenuation; per-pixel texturing;
flat-mode translucency; triangle-precise floor snap; rotation matching in
mesh snap; marquee for non-mesh marker glyphs; material assets in browser
folders; folder deletion.
