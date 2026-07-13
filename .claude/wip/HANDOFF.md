# PBR materials run 3/3: wgpu (DX12/Vulkan) backend parity -- DONE, awaiting review

Branch: wip/pbr-wgpu (from main at 53db243). All milestones complete in one pass.

## DONE

1. Per-face PBR data plumbing (engine/wgpu_renderer.py):
   Reused `gpu_geometry._build_pbr(mesh, face_id_tri)` (already shared with
   GL, no changes needed to gpu_geometry.py) to build a third per-vertex
   VBO (`in_rm` vec2 roughness/metallic + `in_emissive` vec3, one combined
   buffer, `array_stride 4*5`, `shader_location`s 4/5) alongside the
   existing pos/normal/faceid and color buffers, added as a 3rd vertex
   buffer layout entry on the mesh pipeline. Cache invalidation mirrors the
   existing color-buffer pattern: `create_buffer_with_data` buffers aren't
   COPY_DST, so a PBR-param edit (material editor) destroys and rebuilds
   the small pbr buffer, keyed on `(id(face_roughness), id(face_metallic),
   id(face_emissive))` -- same non-per-frame path as the color_id cache.
2. WGSL shading (`_MESH_WGSL` in engine/wgpu_renderer.py): added a `PI`
   const (was missing from this shader module -- only `_SKY_WGSL` had one,
   caused a shader-compile validation error on first run, fixed) and
   `ggx_specular()`, a line-for-line WGSL port of GLRenderer's
   `ggxSpecular()`/renderer.py's `_ggx_specular` -- same D/G/F terms,
   Smith-GGX Karis k=(a+1)^2/8 remap, Schlick Fresnel, F0 = mix(0.04,
   albedo, metallic). Applied to the sun (reusing the shared `lambert`) and
   to every point/spot/IES light (reusing `radiance`, which already has the
   per-light shadow factor folded in before the specular term reads it, so
   shadows darken specular exactly like CPU/GL). `spec_scale = 1 -
   roughness*(1-metallic)` gates both terms to exactly 0 at default params.
   Diffuse gate `lum *= (1.0 - metallic)` applied once to the whole
   accumulated sum, matching GL/CPU. Emissive added unconditionally after
   fog.
3. Default-param scenes: no literal Python-side fast-path branch in the
   WGSL fragment shader (unlike the CPU's per-frame `pbr_active` gate) --
   the GPU runs the same instructions for every pixel regardless, but
   `spec_scale` algebraically zeroes the specular term at defaults
   (roughness=1, metallic=0) so the numeric *output* is identical; the
   `demo.py --api dx12` FPS check below (98.6 FPS, default-param scene)
   confirms no measurable GPU-side cost regression from the extra math.
4. tests/wgpu_checks.py extended with 3 new PBR cases mirroring gl_checks.py's
   6a/6b/6c exactly (same scene builders/camera, so dx12 numbers are
   directly comparable to the already-judged GL numbers):
   - 7a: 3-way (cpu/gl/dx12) mean-brightness parity on a metallic=1.0/
     roughness=0.15 box -- diffs 0.0/0.0 (exact match to GL and CPU).
   - 7b: dx12 specular highlight -- pbr scene shows 396 bright (>240) pixels
     (identical count to GL's 396), default-param scene shows 0.
   - 7c: dx12 emissive-in-the-dark -- lights all off/zero, emissive box hits
     max=220 vs off max=0.
5. Stale-doc check (per task item 5): verified engine/wgpu_renderer.py
   already has sun disc (`_SKY_WGSL`'s disc/glow math, `sun_extra`/
   `sun_dir` uniforms), one-bounce GI (`gi_tex`, `GITracer`,
   `_upload_gi_tex`), and fog volumes (`fog_vol_*` uniforms,
   `apply_fog_volumes`/`sky_fog_vol_segment`) -- all shipped and covered by
   wgpu_checks.py's existing sun-disc/GI/fog-volume cases (which pass, see
   verification below). The only place the "wgpu missing sun disc/GI/fog
   volumes" claim was found was the *prior* wip/pbr-gl branch's now-
   superseded HANDOFF.md (this file, overwritten) -- CLAUDE.md itself does
   not currently make that claim, so no CLAUDE.md edit was needed. Flagging
   to the supervisor: this agent's own system-prompt architecture map
   (engine-coder briefing) still says wgpu is "MISSING sun disc/GI/fog
   volumes -- parity is the top backlog item" and should be corrected at
   the source (`.claude/agents/engine-coder.md` per CLAUDE.md), since I
   cannot edit that file from here.

## Verification (all real output, see below)

- `py tests/wgpu_checks.py` -- PASSED: original parity/shadows/sun-disc/GI/
  fog-volume cases + new PBR parity/highlight/emissive cases, e.g.:
  `dx12 pbr parity OK: dx12=15.6 gl=15.6 cpu=15.6 (diffs 0.0/0.0)`
  `dx12 highlight OK: bright pixels pbr=396 default=0`
  `dx12 emissive-in-dark OK: off max=0 emissive max=220`
- Full battery, all 10 suites, individually run: smoke_test, browser_checks,
  toolbar_checks, window_checks, texture_checks, material_checks,
  pbr_checks, gl_checks, wgpu_checks, env_checks -- ALL PASSED (exit 0,
  each printed its own PASSED banner).
- FPS: `py demo.py --api dx12 --frames 150` = 98.6 FPS (2432 tris final
  frame), default-param scene, vs ~85-92 reference -- no regression.
  `py demo.py --headless --frames 90` (forced CPU) = 56.6 FPS, no
  regression vs the ~44-56 CPU reference range.
  `py editor.py --headless --frames 120` (forced CPU) = 2.6 FPS -- matches
  the documented "working dir intermittently benches slow" environmental
  CPU-editor-headless number (2.3-2.6 range from the prior wip/pbr-gl run),
  not a regression from this run's diff.
- Working tree: only engine/wgpu_renderer.py and tests/wgpu_checks.py
  touched (gpu_geometry.py's `_build_pbr` was already shared/reusable from
  run 2, no changes needed there). `git status --short` clean otherwise.
  Temp screenshot written to the OS temp dir, not the repo.

## Commits on this branch

- (pending this checkpoint) [wip] PBR materials (3/3): wgpu WGSL
  metallic-roughness shading + parity tests

## Known limitations / decisions for the reviewer to weigh

- No per-fragment PBR texture maps (roughness/metallic/emissive textures);
  this run only extends the existing per-face baked-color model to wgpu,
  matching runs 1 (CPU) and 2 (GL)'s scope exactly.
- All three backends (CPU/GL/wgpu) now have full PBR metallic-roughness
  parity; this closes the 3-run PBR materials arc.
