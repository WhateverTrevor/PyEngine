# No task in flight

Nothing is mid-implementation. The wgpu visual parity task (sun disc, GI,
fog volumes) was judged and squash-merged to main on 2026-07-12; its wip
branch is deleted.

When a task IS in flight, this file holds its resume state (task, DONE with
evidence, NEXT, known issues, temp-artifact paths) per the checkpoint
protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

Backlog pointers live in CLAUDE.md ("Known gaps / natural backlog") — the
top queued item is now directional (sun) shadow attenuation on mesh faces
in the wgpu backend (mirror GL's `_upload_dl_shadow_tex` + `dlShadowTex`
texelFetch pattern), flagged during the parity task as the remaining gap.
