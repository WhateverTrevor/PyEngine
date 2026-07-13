# No task in flight

The DX12-default run was judged and squash-merged to main (2df73cd) on
2026-07-13; its wip branch is deleted. DX12 is now the default renderer
everywhere; CPU is opt-in (user preference — never reintroduce cpu as a
default). Headless still forces CPU by design.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. Working dir intermittently benches 2-10x
slow (environmental) — verify FPS in a fresh worktree before chasing.
User should manually verify one exported FBX imports into real Blender.

Backlog: wgpu directional sun-shadow attenuation (top); per-pixel
texturing (unlocks texture-mapped PBR); material assets in browser
folders; local-space rotate; folder deletion.
