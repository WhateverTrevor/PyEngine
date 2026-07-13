# Material Types + Transparency (run B of 2) -- wip/mat-transparency

## Task
Add material blend_mode (opaque/translucent), an Output Opacity pin gated by
blend_mode, per-face opacity bake, and translucent rendering across CPU/GL/wgpu
+ editor UI. See full spec in the original task brief (supervisor has it).

## DONE (milestone 1: blend_mode + opacity pin gating + bake)
- `engine/materials.py`:
  - `MaterialGraph.blend_mode` ("opaque" default / "translucent"), persisted
    via `to_dict`/`from_dict` (`"blend_mode"` key, defaults to "opaque" on
    load for old graphs -- backward compat).
  - `set_blend_mode(mode)`: switches mode; opaque<-translucent transition
    cleanly disconnects any Opacity link (UE-consistent: opaque has no
    opacity concept, no ghost link kept -- documented in the docstring).
  - Output node gained an "opacity" input pin + "opacity" param (default 1.0,
    the inline/fallback value UE-style).
  - `connect()`: rejects connecting into Output.opacity unless
    `blend_mode == "translucent"` (the "greyed out, refuses connections"
    requirement).
  - `_evaluate_common`'s output branch always computes `opacity` (harmless
    when opaque) but `evaluate_pbr` only surfaces it (non-None 5th return
    value) when `blend_mode == "translucent"`.
  - `apply()`: writes `entity.mesh.face_opacity` only when translucent;
    opaque bakes explicitly leave/reset it to all-1.0 (backward-compat
    contract, mirrors the roughness/metallic/emissive pattern).
- `engine/mesh.py`: `Mesh.face_opacity` (M,) float, default all-1.0,
  constructor param `face_opacity=None` alongside the other PBR arrays.
- Scene/material-asset persistence needed NO changes: `entity.material` and
  `MaterialAsset.graph_dict` already round-trip through `MaterialGraph.
  to_dict`/`from_dict` generically (verified by reading `assets.py` lines
  ~298-377), and `entity.material.apply(entity)` re-bakes (incl. face_opacity)
  on every scene/asset load -- same pattern as PBR roughness/metallic/emissive.
- Verified via throwaway script (scratchpad `test_m1.py`, not in repo):
  opaque rejects opacity connect, translucent accepts it, switching back to
  opaque disconnects, blend_mode round-trips through to_dict/from_dict,
  opaque bake leaves face_opacity at 1.0, translucent bake with unconnected
  Opacity pin uses the inline param value (tested 0.3 -> face_opacity==0.3).
- `py tests/smoke_test.py` PASSED (PYENGINE_SETTINGS pointed at a temp path,
  not the real settings.json) -- no regression in existing material graph
  round-trip test.

## DONE (milestone 2: CPU deferred translucent pass + shadow/GI gating)
- `engine/raytrace.py`: `_is_translucent(entity)` helper (material present +
  `blend_mode == "translucent"`); `ShadowTracer.refresh` and `GITracer.
  compute`'s caster-list comprehensions now also exclude translucent
  entities, mirroring the existing `casts_shadow` gate -- translucent meshes
  never occlude shadow rays or source GI bounce, but remain normal GI/shadow
  *receivers* (untouched receiver-list comprehensions).
- `engine/renderer.py`:
  - `_is_translucent(entity)` module helper (same check, renderer-local to
    avoid an engine.raytrace import from renderer for a one-line check).
  - Both `_render_flat`'s and `_render_deferred`'s main per-entity loops now
    skip translucent entities entirely (no face-ID-buffer entry, no PBR
    array contribution) -- translucent faces never occlude opaque geometry.
  - New `Renderer._render_translucent(surface, scene, camera, tracer, env,
    lights)`: full-window-resolution back-to-front painter pass, called
    after every opaque-blit exit point in `_render_deferred` (including the
    two early-return "nothing visible" branches) and once at the end of
    `_render_flat`. Per-face lighting at the face centroid (v1 approximation
    -- same one flat mode already uses for everything, documented in the
    method's docstring). Alpha-composites via a reusable `pygame.SRCALPHA`
    overlay surface (`self._translucent_overlay`): for each poly back-to-
    front, fill only its screen-space bbox transparent, draw the poly
    (r,g,b,face_opacity*255) into that bbox, blit the bbox onto `surface`
    (pygame's default blit alpha-composites SRCALPHA source over opaque
    dest -- standard Porter-Duff "over"), then re-clear the bbox so the next
    poly's blit doesn't pick up stale alpha. Bbox-bounded cost, not
    full-frame, so idle cost (no translucent entities) is a single
    early-return.
  - Flat mode (F2) draws translucent faces fully OPAQUE after the opaque
    polys (documented pre-existing limitation: flat mode has no per-pixel
    alpha path; v1 transparency targets the deferred/default renderer + GL/
    wgpu, not the flat debug view).
  - `editor.py`'s two material-preview call sites (`make_material_icon`,
    `MaterialEditorUI._render_preview`) updated for the new 5-tuple
    `evaluate_pbr` return (`..., opacity`); both now checker-background the
    preview when `graph.blend_mode == "translucent"` (new `_draw_checker_bg`
    helper) and set `entity.material = graph` so the renderer's translucent
    path actually activates for the preview sphere (previously baked
    face_colors directly onto the mesh without ever setting `entity.
    material`, which is also why translucent bakes wouldn't render
    translucent in ANY preview context without this).
  - `tests/pbr_checks.py`: updated two `evaluate_pbr` unpacks for the new
    5-tuple (opaque -> `opac is None` asserted).
- Verified via throwaway script (scratchpad `test_m2.py`, not in repo):
  numeric blend-equation check (red translucent quad @ alpha=0.5 over blue
  background: center pixel matches `bg*0.5 + red*0.5` within atol=12,
  corner/background-only pixel untouched) and back-to-front stacked-quad
  ordering (green quad behind + red quad in front, both alpha=0.5: center
  matches the correctly-ordered composite `((bg*.5+green*.5)*.5+red*.5)`
  within atol=15 -- confirms farthest-drawn-first, nearest-composited-last).
- Ran individually, all PASSED: smoke_test, material_checks, pbr_checks,
  mat_ui_checks.
- FPS spot-check (PYENGINE_SETTINGS pointed at temp paths): editor.py
  headless 120f in 5.06s = 23.7 FPS (matches ~23 CPU reference, no
  regression -- idle/no-translucent-entities cost is the early-return in
  `_render_translucent`, confirmed structurally not just measured); demo.py
  headless 90f in 1.69s = 53.3 FPS (within the 44-52 reference band).

## DONE (milestone 3: GL + wgpu translucent passes)
- `engine/gpu_geometry.py`: new `_build_opacity(mesh, face_id_tri)` -- same
  per-face-vertex repeat pattern as `_build_color`/`_build_pbr`, returns
  (T*3, 1) float32 from `mesh.face_opacity` (shared source of truth for both
  GPU backends, same file that already holds `_build_color`/`_build_pbr`).
- `engine/gl_renderer.py`:
  - Added `in_opacity` vertex attribute -> `vOpacity` varying -> fragColor.a
    = vOpacity (was hardcoded 1.0). Opaque meshes always carry
    face_opacity==1.0 so this is a no-op for them.
  - `_get_geo_cache` builds/caches a new `vbo_opacity` (mutable, rewritten
    in place like the PBR vbo when `mesh.face_opacity` identity changes --
    e.g. after a material re-bake); `_prune_geo_cache` releases it.
  - `render()`'s draw loop split into `opaque_pairs` (rendered exactly as
    before -- unmodified code path, this is why gl_checks.py parity is
    byte-identical) and `translucent_pairs` (entities where
    `renderer._is_translucent(entity)`, imported from `renderer.py`, is
    true). Translucent pairs sorted back-to-front by per-entity distance
    from camera to `transform.matrix()[:3,3]`, drawn with `ctx.enable(BLEND)`
    + `ctx.blend_func = (SRC_ALPHA, ONE_MINUS_SRC_ALPHA)` +
    `self.target.depth_mask = False` (moderngl Framebuffer property,
    confirmed present on both `ctx.screen` and the standalone FBO target);
    depth TEST stays on throughout (opaque occlusion still correct
    per-pixel). Blend disabled + depth_mask restored to True after.
  - Documented in the module docstring why per-entity (not per-triangle)
    sort is the practical GPU granularity -- the CPU path sorts per-face
    globally (see `renderer._render_translucent`), but a cross-entity
    per-triangle GPU sort would mean per-triangle draw calls; per-entity is
    the standard approximation and is exact for the common "one translucent
    volume in front of opaque geometry" case (verified numerically, below).
- `engine/wgpu_renderer.py`:
  - Same `in_opacity`/`vOpacity`-equivalent WGSL attribute (`@location(6)`
    vertex input, `@location(7)` VOut field), fragment returns
    `vec4(color, in.opacity)` instead of hardcoded 1.0.
  - New `_mesh_translucent_pipeline`: same shader module + vertex layout as
    `_mesh_pipeline`, `depth_write_enabled: false`, standard alpha blend
    target state. Built with an EXPLICIT shared `pipeline_layout` (from the
    opaque pipeline's auto-inferred bind group layouts via
    `device.create_pipeline_layout`) rather than its own `layout="auto"` --
    two independently-`"auto"`-built pipelines from the same shader are only
    structurally identical, not bind-group-compatible per the WebGPU spec;
    sharing the layout lets one set of frame/entity bind groups serve both
    pipelines. Documented in `_build_pipelines`'s comment.
  - `_get_geo_cache` builds/caches a 4th vertex buffer (`opacity_buf`,
    rebuilt via destroy+recreate on `face_opacity` identity change, matching
    the existing color/pbr rebuild pattern since wgpu buffers from
    `create_buffer_with_data` aren't COPY_DST); `_prune_geo_cache` destroys
    it. `render()`'s draw loop split the same way as GL's (opaque pairs
    with `_mesh_pipeline`, then back-to-front-sorted translucent pairs with
    `_mesh_translucent_pipeline`, both drawn via one shared `_draw` closure
    inside the same render pass -- `set_pipeline` mid-pass is valid WebGPU).
- Verified via throwaway script (scratchpad `test_m3_translucent.py`, not in
  repo): built a scene with an opaque blue floor cube behind a translucent
  red cube (alpha=0.5) via `MaterialGraph.set_blend_mode("translucent")` +
  inline Output.opacity param + `graph.apply(entity)`. Center-patch RGB
  compared across CPU/GL/DX12: all three came back byte-identical
  (`[83.4, 20.6, 23.6]`, diff 0.0 for both GL and DX12 vs CPU), and clearly
  distinct from the same scene rendered with alpha=1.0 (`[155.8, 28.2,
  28.2]`) -- confirms blending is actually applied, not silently a no-op,
  and that GL/CPU/DX12 agree exactly on this scene (single translucent
  entity, so the per-entity-sort approximation is exact here).
- Regression checks, all PASSED unchanged: `gl_checks.py` (opaque golden
  parity byte-for-byte, e.g. "parity OK: mean brightness gpu=70.4 cpu=70.1"
  -- same numbers as pre-milestone-3), `wgpu_checks.py` (dx12=70.4 gl=70.4
  cpu=70.1, all sub-checks incl. shadows/GI/fog/sun disc/PBR unchanged).
- FPS spot-check (PYENGINE_SETTINGS temp paths, no translucent entities in
  demo/editor default content so this exercises the early-return/idle-cost
  path for both new pipelines): demo.py `--api gl --frames 120` (real
  window, not --headless since headless forces CPU) = 117.5 FPS (ref
  112-116, effectively on-band); demo.py `--api dx12 --frames 120` = 96.4
  FPS (ref 92-98); editor.py `--headless --frames 120` (CPU) = 24.6 FPS
  (ref ~23, unaffected as expected -- editor CPU path untouched by this
  milestone). No regression.

## DONE (milestone 4: editor UI blend-mode selector + tests/transparency_checks.py)
- `editor.py` (`MaterialEditorUI`):
  - New blend-mode selector: two buttons ("Opaque"/"Translucent") in the
    preview-panel strip, drawn in `_draw_preview_panel` just below the
    preview sphere (shifts the existing "Preview: X" label / "Stop
    Previewing" button down to make room, same strip, no new panel). Active
    mode highlighted with `ACCENT`, inactive with the existing dim button
    color. Rects stored as `self._blend_opaque_rect`/`_blend_translucent_rect`
    (init'd to `None` in `__init__`, same pattern as `_preview_stop_rect`).
  - `update()` handles clicks on the two rects: calls
    `self.graph.set_blend_mode(...)` then `self.apply(draft=False)` (full
    re-bake + preview refresh + Details-panel icon refresh, same call the
    param-slider release path already uses).
  - Opacity pin greyed when opaque: in the node `draw()` loop's input-pin
    rendering, `node["type"] == "output" and iname == "opacity" and
    self.graph.blend_mode != "translucent"` now draws the pin dot + label in
    a dim grey instead of the normal pin blue/TEXT_DIM -- the underlying
    `connect()` gating already existed (milestone 1); this is purely the
    draw-side reflection the task brief asked for. Node-graph "drag to
    connect" itself already silently refuses (via `connect()` returning
    False) -- unchanged, no new refusal logic needed.
- New `tests/transparency_checks.py` (10 numbered checks, same
  isolate-settings.json / no-pollution-guard pattern as the other judge
  suites): opacity pin gating (opaque rejects / translucent accepts /
  opaque<-translucent disconnects); blend_mode round-trip through
  `to_dict`/`from_dict` incl. old-graph backward-compat default; blend_mode
  + face_opacity round-trip through a full scene save/load cycle (not just
  the graph dict in isolation); opaque bake resets `face_opacity` to 1.0;
  CPU blend-equation numeric check (ported from milestone-2's `test_m2.py`
  scratch script) plus a new "doesn't bleed outside its own footprint"
  check; back-to-front stacking-order numeric check (ported from the same
  scratch script); `ShadowTracer.refresh` excludes a translucent entity
  from `_caster_mats` while keeping an opaque one (direct introspection,
  not just predicate-function testing); `raytrace._is_translucent` and
  `renderer._is_translucent` agree on both entities; opaque golden parity
  (starter scene still renders non-blank, explicit per the task brief even
  though gl_checks.py/wgpu_checks.py already cover CPU/GL/DX12 parity
  numerically); 3-way CPU/GL/DX12 parity on a translucent scene (ported
  from milestone-3's `test_m3_translucent.py` scratch script -- byte-
  identical center-patch RGB across all three, same as the scratch result);
  editor blend-mode-selector UI (rects populated after `draw()` for both
  states, and a real click on the Opaque button driven through
  `MaterialEditorUI.update()` -- monkeypatches `pygame.mouse.get_pos` since
  the SDL dummy driver doesn't support cursor warps -- actually flips
  `blend_mode` back, exercising the real click-handling wiring not just the
  underlying graph API).
- Full 12-suite battery run individually (each with its own
  `PYENGINE_SETTINGS` temp path), all exit 0 / all PASSED: smoke_test,
  browser_checks, toolbar_checks, window_checks, texture_checks,
  material_checks, mat_ui_checks, pbr_checks, transparency_checks,
  gl_checks, wgpu_checks, env_checks.
- Final FPS spot-check (PYENGINE_SETTINGS temp paths, default content --
  no translucent entities in demo/editor defaults, so idle-cost paths):
  editor.py `--headless --frames 120` (CPU) = 24.4 FPS (ref ~23); demo.py
  `--api dx12 --frames 120` (real window) = 96.4 FPS (ref 92-98); demo.py
  `--api gl --frames 120` = 116.4 FPS (ref 112-116). No regression anywhere.
- `git status` clean of strays after this milestone (only `editor.py`
  modified + `tests/transparency_checks.py` added, both intended).

## TASK COMPLETE
All four milestones landed: blend_mode + gated Opacity pin + face_opacity
bake (m1), CPU deferred translucent pass + shadow/GI exclusion (m2), GL +
wgpu translucent passes (m3), editor UI selector + transparency_checks.py +
full battery (m4). Nothing outstanding from the original task brief.

## Known issues / decisions to weigh
- Flat mode (F2) renders translucent faces as fully opaque (see above) --
  a scoping call, not an oversight; flag to the supervisor if flat-mode
  transparency turns out to matter for the horror game's debug workflow.
- Translucent per-face lighting is centroid-based (v1, matches flat mode's
  existing approximation) -- no per-pixel reconstruction for translucent
  faces even though the surrounding opaque pass is per-pixel deferred.

## Temp artifacts
- scratchpad test script: `C:\Users\tseit\AppData\Local\Temp\claude\D--ClaudeCode-PyEngine\fff00280-50d8-4931-b697-241c3b13c260\scratchpad\test_m1.py`
  (not part of the repo, safe to ignore/regenerate)
