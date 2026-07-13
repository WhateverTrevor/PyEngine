# No task in flight

Nothing is mid-implementation. FBX export (run 1 of the texture/material
slate) was judged and squash-merged to main on 2026-07-12; its wip branch
is deleted.

Remaining runs in the user's current slate:
2. Texture assets in the content browser (import png/jpg, grid previews)
   + TextureCoordinate / TextureSample material nodes (UE semantics) —
   requires building per-face UV evaluation (FBX import currently strips
   UVs; box-projection fallback for meshes without them).
3. Unreal-style node overhaul (Material Attributes, math, noise, scaling;
   UE input/output pin conventions) + drag-and-drop material assignment
   from browser onto the Details material slot with a material preview.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS /
Editor(settings_path=...) — any NEW test constructing an Editor must do
the same. The user's real settings.json must never be written by
automation. User should manually verify one exported FBX imports into
real Blender (unverifiable in this environment).

Backlog: wgpu directional sun-shadow attenuation; local-space rotate;
folder deletion.
