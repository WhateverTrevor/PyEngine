# No task in flight

The material-editor UX overhaul (79dc8a9) and material types +
transparency (a284f06) were judged and squash-merged to main on
2026-07-13; wip branches deleted. The transparency task demonstrated the
checkpoint protocol working: first agent stopped cleanly at milestone 2,
a FRESH agent finished 3-4 from branch+HANDOFF.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. DX12 is the default renderer (user
preference — CPU is opt-in only). Working dir intermittently benches
slow (environmental) — verify FPS in a fresh worktree before chasing.
User should manually verify one exported FBX imports into real Blender.

The full battery is now TWELVE suites: smoke, browser, toolbar, window,
texture, material, mat_ui, pbr, transparency, gl, wgpu, env (+ fbx
cases inside browser_checks). Name all of them in every brief.

Backlog: wgpu directional sun-shadow attenuation (top); per-pixel
texturing; flat-mode translucency; per-face GPU translucent sort;
material assets in browser folders; local-space rotate; folder deletion.
