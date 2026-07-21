# No task in flight

Blueprint asset + Python script editor + in-engine compile/bug-check (run
1 of 2) was judged and squash-merged to main on 2026-07-14.

engine/blueprint.py: compile_blueprint(source, name) -> plain dict, NEVER
raises (SyntaxError with line/col; BaseException during exec, so even
SystemExit is contained; validates a Behavior subclass exists).
engine/assets.py: BlueprintAsset {name, category, components[], script,
compile_result} persisted to assets/blueprints/*.json, folder-tree aware.
editor.py: "+ Blueprint" browser button, blueprint tiles, and
ScriptEditorUI — a MaterialEditorUI-style in-app window with a
line-numbered gutter, caret + arrows/Home/End, Enter/Backspace/Delete,
Tab=4 spaces, scroll-follows-caret, Compile (Ctrl+Enter) / Save (Ctrl+S),
error-line highlight + status strip. Compile saves; close auto-saves.

NEXT (run 2 of 2): POSED MESHES + world instantiation. The
`components` field already exists and is an empty list — fill it with
{asset_name, position, rotation, scale} entries, add UI to compose/pose
mesh components inside the blueprint (the transform gizmo can likely be
reused), and make instantiating a blueprint into the world build the
composed entity with the compiled Behavior attached and running.
Behavior runtime errors during update must be caught per-frame and
surfaced, not crash the game loop.

KNOWN LIMITATION to address eventually: there is NO infinite-loop/hang
guard on exec — a `while True:` in a user script WILL hang the editor
(verified). A worker-thread + timeout sandbox is the fix. Also no
selection/clipboard/undo in the script editor yet.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: settings isolate via PYENGINE_SETTINGS; UI tests drive the real
event path; DX12 default; DO NOT touch assets/gat.* or folders.json. Full
battery is TWENTY-ONE suites. FPS here is 2-10x slow + high variance —
same-environment A/B only.

Backlog: QEM decimation; LOD tuning knobs; carry face_uvs through
decimation; File-menu import paths still folder-unaware; wgpu directional
sun-shadow; per-pixel texturing; folder deletion.
