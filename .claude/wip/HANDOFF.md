# No task in flight

PBR run 1/3 (engine foundation + CPU deferred renderer) was judged and
squash-merged to main on 2026-07-12; its wip branch is deleted.

Remaining PBR runs:
2. OpenGL backend (engine/gl_renderer.py): same metallic-roughness model
   in GLSL — per-face roughness/metallic/emissive fed like face colors
   (texture or attribute), GGX + Smith + Schlick matching renderer.py's
   _ggx_specular math, emissive after fog, default-param fast path not
   needed on GPU but default output must match legacy within existing
   parity tolerances. Extend gl_checks with a PBR-scene CPU-vs-GL parity
   case.
3. wgpu backend (engine/wgpu_renderer.py): same, in WGSL; extend
   wgpu_checks 3-way parity with the PBR scene.

Contract from run 1: defaults (roughness=1, metallic=0, emissive=0)
render byte-identical to legacy on CPU (golden fixture
tests/fixtures/pbr_golden.npy); spec_scale = 1 - roughness*(1-metallic)
zeroes specular at the default combo — GPU implementations must replicate
this gate or default-param parity breaks.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. Working dir intermittently benches 2-10x
slow (environmental) — verify FPS in a fresh worktree before chasing.

Backlog: wgpu directional sun-shadow attenuation; per-pixel texturing;
material assets in browser folders; local-space rotate; folder deletion.
