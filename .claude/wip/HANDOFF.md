# No task in flight

Nothing is mid-implementation. The texture/material slate (FBX export;
texture assets + UV foundation + TexCoord/TextureSample; UE node overhaul
+ drag-drop material assignment) is fully judged and squash-merged to
main on 2026-07-12; all wip branches are deleted.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS /
Editor(settings_path=...) — any NEW test constructing an Editor must do
the same. The user's real settings.json must never be written by
automation. The user should manually verify one exported FBX imports
into real Blender (unverifiable in this environment).

ENVIRONMENT WARNING: this long-lived working directory intermittently
benches 2-10x slow on identical code (AV/indexing); if FPS looks
regressed, verify in a fresh `git worktree` before chasing it.

Backlog: wgpu directional sun-shadow attenuation (top); per-pixel
texturing; material assets filed into browser folders (root-only today);
ComponentMask real checkboxes; local-space rotate; folder deletion.
