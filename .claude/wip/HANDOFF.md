# No task in flight

Nothing is mid-implementation. The content-browser folder tree task was
judged (1 finding — root/no-rename None collision — fixed on send-back)
and squash-merged to main on 2026-07-12; its wip branch is deleted.

When a task IS in flight, this file holds its resume state (task, DONE with
evidence, NEXT, known issues, temp-artifact paths) per the checkpoint
protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

Backlog pointers live in CLAUDE.md ("Known gaps / natural backlog").
Queued next by the user: viewport toolbar (Translate/Rotate/Scale +
World/Local toggle) and editable transform vectors in the Details panel.

NOTE for benchmarking: running in this long-lived working directory gives
~35% lower FPS than a fresh `git worktree` of identical code (confirmed
2026-07-12 on both branch and main; cProfile shows uniform slowdown in
unmodified renderer.py — likely AV/indexing). Benchmark FPS from a clean
worktree, or treat in-place numbers as ~35% pessimistic.
