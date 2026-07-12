# No task in flight

Nothing is mid-implementation. The last two tasks (environment assets,
window management) were judged and squash-merged to main on 2026-07-11;
their wip branches and worktrees are deleted.

When a task IS in flight, this file holds its resume state (task, DONE with
evidence, NEXT, known issues, temp-artifact paths) per the checkpoint
protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

Backlog pointers live in CLAUDE.md ("Known gaps / natural backlog") — the
top queued item is wgpu (DX12/Vulkan) visual parity for the sun disc, GI,
and fog volumes.
