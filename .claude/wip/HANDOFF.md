# No task in flight

Nothing is mid-implementation. The viewport-toolbar + editable transform
vectors task was judged and squash-merged to main on 2026-07-12; its wip
branch is deleted. (Earlier the same day: content-browser folder tree,
panel resizing + fullscreen, wgpu visual parity.)

When a task IS in flight, this file holds its resume state (task, DONE with
evidence, NEXT, known issues, temp-artifact paths) per the checkpoint
protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

Backlog pointers live in CLAUDE.md ("Known gaps / natural backlog") — the
top queued item is directional (sun) shadow attenuation on mesh faces in
the wgpu backend. Natural follow-ups from the toolbar task: local-space
rotate (needs quaternion rewrite of the ring drag), folder deletion in
the content browser.

BENCHMARKING NOTE (supersedes earlier worktree advice): in-place runs
with the per-user settings.json set aside match the CLAUDE.md reference
numbers reliably; worktree-vs-in-place gaps have been observed in BOTH
directions on identical code (disk/AV noise). Benchmark in place, always
set settings.json aside (it can pin pixel_scale/api and skew numbers),
and always restore it afterward.
