# No task in flight

Unreal-style import-options dialog was judged and squash-merged to main on
2026-07-14. Importing an FBX/HDR/texture now opens a dark-theme modal:
detected type header, editable Name, target-folder cycle picker, and (for
meshes) Uniform Scale + "Fit to ~1 unit" + Up Axis Y/Z — baked into the
stored mesh on import. After Import the browser navigates to the
destination folder and selects the new tile (fixes the "hidden tile /
invisible speck" problem the user hit with their 0.12-unit "Gat" FBX).
engine/fbx.py import_fbx gained scale/up_axis kwargs (defaults reproduce
prior output); new engine.fbx_fit_scale. Dialog-free _do_import underneath
for tests/future Explorer drag-drop.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

Next queued (run 2 of this slate): distance-based LOD. The user's real
imported mesh is 10,448 faces; the engine's built-ins are 6-80. Generate
decimated LOD levels for high-poly meshes (offline/at import) and select
the LOD by camera distance at render time across CPU/GL/wgpu. A "Generate
LODs" option belongs in the import dialog (leave a hook). Mesh has
face_colors/face_uvs/face_roughness/metallic/emissive per-face arrays that
any decimation must keep consistent (or drop gracefully to per-mesh).

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; UI tests
drive the real event path; DX12 default. Full battery is NINETEEN suites.
DO NOT touch the user's assets/gat.json, assets/models/gat.npz,
assets/folders.json. FPS in this working dir is 2-10x slow + high variance
— only same-environment A/B is valid.

Backlog: File-menu's older _import_fbx_dialog/_import_hdri_dialog are still
folder-unaware (same bug, out of scope this run); wgpu directional
sun-shadow; per-pixel texturing; flat-mode translucency; folder deletion.
