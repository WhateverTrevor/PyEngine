# PBR materials run 2/3: OpenGL backend parity -- DONE, awaiting review

Branch: wip/pbr-gl (from main at b97f515). All milestones complete.

## DONE

1. Per-face PBR data plumbing (engine/gpu_geometry.py, engine/gl_renderer.py):
   `_build_pbr(mesh, face_id_tri)` builds per-vertex (roughness, metallic)
   and emissive (0..1) arrays, repeated per triangle vertex like
   `_build_color`. Fed to the mesh VAO as two new vertex attributes
   (`in_rm` vec2, `in_emissive` vec3) via a third VBO, cached and rebuilt
   on `(id(face_roughness), id(face_metallic), id(face_emissive))` change,
   same pattern as the existing `color_id` cache-invalidation for
   `face_colors`.
2. GLSL shading (_MESH_FS in engine/gl_renderer.py): `ggxSpecular()` mirrors
   renderer.py's `_ggx_specular` exactly (D/G/F terms, Smith-GGX Karis
   k=(a+1)^2/8 remap, Schlick Fresnel, F0 = mix(0.04, albedo, metallic)).
   Applied to the sun (reusing the same shadowed `lambert` the diffuse term
   uses, matching the CPU's shared shadowed-NdotL reuse) and to every
   point/spot/IES light (reusing `radiance` which already has the point-
   light shadow factor folded in before the specular term reads it, so
   shadows darken specular exactly like the CPU). Diffuse gate: `lum *=
   (1.0 - vMetallic)` applied ONCE to the whole accumulated ambient+
   directional+GI+lights sum -- verified algebraically equivalent to the
   CPU's per-term gating (linear distribution of a scalar multiplier).
   `spec_scale = 1 - roughness*(1-metallic)` gates both sun and point-light
   specular to exactly 0 at default params. Emissive (`vEmissive`) added
   unconditionally after fog, matching CPU order.
3. tests/gl_checks.py extended with 3 new PBR cases (all passing):
   - 6a: CPU-vs-GL mean-brightness parity on a metallic=1.0/roughness=0.15
     box (diff 0.0).
   - 6b: GL specular highlight -- a metallic+shiny box shows 396 pixels
     >240 brightness where the same scene at default params shows 0 (test
     scene deliberately dims ambient/directional so default-param diffuse
     never saturates, isolating the highlight; camera looks straight at
     the box with the point light placed near the camera so the reflection
     lands almost dead-on).
   - 6c: GL emissive-in-the-dark -- lights all off/zero, emissive box hits
     max=220 vs off max=0.

## Verification (all real output, see below)

- `py tests/gl_checks.py` -- PASSED (parity, IES, cone, shadow, depth, +
  new PBR parity/highlight/emissive cases).
- Full battery, all 10 suites, individually run: smoke_test, browser_checks,
  toolbar_checks, window_checks, texture_checks, material_checks,
  pbr_checks, gl_checks, wgpu_checks, env_checks -- ALL PASSED.
- FPS: `demo.py --api gl` (real window, not headless -- headless forces
  CPU per demo.py's own `--headless` handling) = 111.9 FPS vs the ~116
  reference in CLAUDE.md -- no regression.
  `editor.py --api gl` (real window) = 31.6 FPS (972 tris).
  `editor.py --headless` (forced CPU) = 2.3-2.4 FPS, reproduced IDENTICALLY
  with this run's diff `git stash`-ed out -- confirmed pre-existing/
  environmental slowness (matches the "working dir intermittently benches
  slow" note), not a regression from this run.
  `demo.py --headless` (forced CPU) = 56.4 FPS, no regression vs the
  ~44-52 CPU reference.
- Working tree: only engine/gpu_geometry.py, engine/gl_renderer.py,
  tests/gl_checks.py touched. No stray files.

## Commits on this branch

- db20fbf [wip] PBR materials (2/3): OpenGL metallic-roughness shading
  (plumbing + GLSL shading; gl_checks legacy cases still passing)
- (pending) tests + full battery checkpoint -- push before reporting to
  supervisor if not already pushed.

## Known limitations / decisions for the reviewer to weigh

- wgpu (run 3) is untouched by design -- its own PBR parity is the next run.
- No per-fragment PBR texture maps (roughness/metallic/emissive textures);
  this run only extends the existing per-face baked-color model to GL,
  matching run 1's CPU scope exactly.
- The PBR vertex data is baked per-face at mesh-build/material-bake time
  (same as face_colors), so editing materials at runtime triggers the same
  cache invalidation path already proven for face_colors.

## Remaining PBR work

3. wgpu backend (engine/wgpu_renderer.py): same model in WGSL; extend
   wgpu_checks 3-way parity with a PBR scene. wgpu is also missing sun
   disc/GI/fog volumes (pre-existing gap, not in scope for the PBR runs).
