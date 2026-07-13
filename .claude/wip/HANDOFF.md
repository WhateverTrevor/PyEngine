# No task in flight

The 3-run PBR arc (CPU foundation b97f515, OpenGL 53db243, wgpu c314422)
is fully judged and squash-merged to main on 2026-07-12; all wip branches
deleted. All three render paths now share the metallic-roughness model
(GGX Cook-Torrance; per-face roughness/metallic/emissive; defaults render
byte-identical to legacy via the spec_scale gate — preserve it in any
future shading change).

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. Working dir intermittently benches 2-10x
slow (environmental) — verify FPS in a fresh worktree before chasing.
User should manually verify one exported FBX imports into real Blender.

Backlog: wgpu directional sun-shadow attenuation (top); per-pixel
texturing (would unlock texture-mapped roughness/metallic too);
material assets filed into browser folders; local-space rotate;
folder deletion.
