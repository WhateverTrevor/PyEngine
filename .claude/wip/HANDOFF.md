# No task in flight

Nothing is mid-implementation. The settings-isolation + dock-zones task
was judged and squash-merged to main on 2026-07-12; its wip branch is
deleted.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT for all future test/benchmark work: test suites now isolate
settings via PYENGINE_SETTINGS / Editor(settings_path=...) — any NEW test
that constructs an Editor must do the same (window_checks has a
no-pollution guard that will catch violations). The user's real
settings.json must never be written by automation.

Backlog pointers live in CLAUDE.md ("Known gaps / natural backlog") — the
top queued item is directional (sun) shadow attenuation on mesh faces in
the wgpu backend. Other follow-ups: local-space rotate (quaternion ring
drag), content-browser folder deletion.
