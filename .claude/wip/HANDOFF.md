# No task in flight

Snapping + Alt-drag duplicate (grid/interval snap, floor snap on End,
Shift-drag mesh face snap, Alt-drag gizmo duplicate for translate+rotate)
was judged and squash-merged to main on 2026-07-14. The task ran across
two agents: first checkpointed cleanly at milestone 1, a FRESH agent
finished 2-4 from branch+HANDOFF (checkpoint protocol working as intended).

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

Next queued (run 2 of this slate): Blender-style pivot modes (median
point, individual origins, active/last-selected, etc). This requires
MULTI-SELECTION first (the editor is single-select today), since pivot
mode only matters with several objects selected — scope multi-select as
part of that run.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. UI/interaction tests MUST drive the real
event path (pygame event injection) — direct handler calls miss crashes
(the ctx-menu crash proved it). DX12 is the default renderer (CPU
opt-in). Held modifier keys (Alt/Shift/Ctrl) aren't reachable by
synthetic SDL events — patch pygame.key.get_pressed / pygame.mouse at the
OS boundary in tests (see snap_checks / mat_ui_checks idiom).

Backlog: wgpu directional sun-shadow attenuation; per-pixel texturing;
flat-mode translucency; triangle-precise floor snap (v1 is AABB); rotation
matching in mesh snap; material assets in browser folders; folder deletion.
