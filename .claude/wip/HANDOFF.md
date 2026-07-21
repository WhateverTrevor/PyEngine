# No task in flight

Distance-based LOD was judged and squash-merged to main on 2026-07-14,
completing the import/LOD slate (import dialog + LOD).

engine/lod.py does vertex-clustering decimation (generate_lods -> [LOD0,
LOD1..]) and per-frame distance selection with hysteresis
(update_scene_lods, called once in core.py before render dispatch). LODs
generate at import (import_fbx generate_lods=True, dialog checkbox) for
meshes >200 faces and store in the .npz; Entity gains lod_meshes /
lod_index / render_mesh(). All three rasterizers draw render_mesh(); the
ray-traced ShadowTracer/GITracer stay on entity.mesh (LOD0) so shadow/GI
caches never invalidate on an LOD switch and shadows don't pop. Built-ins
(<=200 faces) carry no LOD data and render byte-identical.

CONTRACT for future work: entity.lod_meshes = generate_lods()[1:]
(decimated only; LOD0 IS entity.mesh); render_mesh() indexes
lod_meshes[lod_index-1]. Constants in engine/lod.py: LOD_FACE_THRESHOLD
200, DEFAULT_LOD_RATIOS (0.5,0.25,0.12), LOD_DISTANCE_FACTORS
(8,20,45)*bound, LOD_HYSTERESIS 0.2. face_uvs are box-reprojected on
decimation (not carried); LOD thresholds are fixed constants, not tunable.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: settings isolate via PYENGINE_SETTINGS; UI tests drive the real
event path; DX12 default; DO NOT touch assets/gat.* or folders.json. Full
battery is TWENTY suites. FPS in this working dir is 2-10x slow + high
variance — only same-environment A/B is valid.

Backlog: QEM decimation (higher quality than clustering); LOD tuning knobs
in the import dialog; carry face_uvs through decimation; File-menu import
paths still folder-unaware; wgpu directional sun-shadow; per-pixel
texturing; folder deletion.
