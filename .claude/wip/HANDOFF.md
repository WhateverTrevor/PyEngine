# No task in flight

Nothing is mid-implementation. Texture assets + UV foundation +
TexCoord/TextureSample nodes (run 2 of the texture/material slate) was
judged and squash-merged to main on 2026-07-12; its wip branch is deleted.

Remaining run in the user's current slate:
3. Unreal-style node overhaul (Material Attributes, math, noise, scaling;
   UE input/output pin conventions applied to ALL nodes) + drag-and-drop
   material assignment from the content browser onto the Details panel's
   material slot, with a material preview swatch in Details.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS /
Editor(settings_path=...) — any NEW test constructing an Editor must do
the same. The user's real settings.json must never be written by
automation. User should manually verify one exported FBX imports into
real Blender (unverifiable here).

Backlog: wgpu directional sun-shadow attenuation; per-pixel texturing;
local-space rotate; folder deletion.
