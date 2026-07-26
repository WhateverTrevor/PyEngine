# PyEngine — supervisor notes

Pure-Python 3D engine + world editor (pygame + numpy) with optional GPU
backends (OpenGL via moderngl; DirectX 12 / Vulkan via wgpu). End goal: a
survival horror game. Repo: https://github.com/WhateverTrevor/PyEngine.

## Your role — established workflow, follow it

- **All engine coding is delegated to the `engine-coder` agent** (Sonnet 5;
  `.claude/agents/engine-coder.md` is its briefing AND the architecture map —
  read it before briefing a task). You write the brief, review the diff,
  verify INDEPENDENTLY (add checks the agent didn't run — its own tests have
  twice missed the judgment-critical cases), judge, squash-merge to main,
  push. Commit messages: body notes `Implemented-By: engine-coder (Sonnet 5)`
  and end with the Claude co-author line. Pass commit messages to git via
  `-F <file>` (PowerShell 5.1 mangles embedded quotes).
- Agents work on `wip/<task>` branches with pushed `[wip]` checkpoints per
  milestone + `.claude/wip/HANDOFF.md`. If an agent dies (usage limits have
  killed two): checkpoint any uncommitted tree work yourself immediately,
  then start a FRESH agent from branch+HANDOFF — never resume a large
  transcript (it re-reads everything at cold rates).
- **Token economy:** update `.claude/token-ledger.md` after EVERY agent run
  (the usage block reports subagent_tokens). Target <=150k per run; split
  bigger features. Delegate token-heavy/judgment-light work (running test
  batteries, doc edits) to a haiku-model agent. Launch big runs right after
  the user's usage window resets, never near its end. Account quota is not
  queryable — the ledger is the only signal.

## Environment facts (hard-won)

- Windows 11, RTX 5070 Ti. Run Python via `py` (`python` is not on PATH).
- pip needs a CA bundle (Avast TLS interception):
  `py -m pip install <pkg> --cert D:\ClaudeCode\Spotidownload\win-ca-bundle.pem`
- Headless: `py editor.py --frames 120 --headless [--screenshot x.png]`
  (forces the CPU renderer). GPU verification needs a real window — a brief
  flash is fine — except moderngl/wgpu standalone contexts, which work
  headless (see tests/). A per-user `settings.json` (gitignored) overrides
  resolution/pixel_scale — account for it when benchmarking.

## Verification (run these as judge; all must pass before merging)

The battery is now **27 suites** (`py tests\<name>.py`). Name ALL of them in
every engine-coder brief and run every one before merging:
smoke_test, gl_checks, wgpu_checks, env_checks, window_checks, browser_checks,
toolbar_checks, texture_checks, material_checks, mat_ui_checks, pbr_checks,
transparency_checks, snap_checks, multiselect_checks, pivot_checks,
marquee_checks, cursor_checks, docktab_checks, import_checks, lod_checks,
blueprint_checks, console_checks, fps_checks, async_bake_checks, bvh_checks,
bp_component_checks, bp_runtime_checks.

- `bp_runtime_checks` takes ~80s, most of it one 15-frame render — it holds
  a live runaway thread on purpose (see below) and a moving shadow caster.
  That's expected, not a hang. Anything that leaves an extra ANIMATED
  caster in its scene makes that render ~8x worse; remove such entities
  once they've proved their point.

- **Exit code is the pass signal.** Most suites also print an
  "ALL ... PASSED"/"JUDGE ... PASSED" line, but `window_checks`,
  `browser_checks` and `import_checks` print NO such line — they end on a
  no-pollution guard / cleanup and signal success purely via exit 0. A
  battery runner that greps for "PASSED" will report those three as false
  failures (this bit one supervisor run).

- UI/interaction tests MUST drive the real event path (pygame event
  injection through eng.input.process + editor.update), NOT direct handler
  calls — a ctx-menu click crash slipped past handler-only tests. Held
  modifier keys / mouse aren't reachable by synthetic SDL events; patch
  pygame.key.get_pressed + mouse.get_pos + mouse.get_pressed at the OS
  boundary (see marquee_checks / snap_checks / async_bake_checks).
- Tests isolate settings via PYENGINE_SETTINGS; a no-pollution guard asserts
  the real settings.json is untouched.
- **FPS BENCHMARKING IS UNRELIABLE HERE.** This working dir benches 2-10x
  slow AND high-variance (identical code has swung e.g. demo dx12 61-102).
  Only same-environment A/B (branch vs a `git worktree` of main, same run)
  is meaningful — never trust a single sample vs an absolute reference.
  ALSO: the repo-root settings.json holds the user's real 5120x1369 /
  pixel_scale 3 — a headless bench without PYENGINE_SETTINGS isolation looks
  catastrophically slow (1-2 FPS). Rough in-place references w/ settings
  aside: editor cpu ~23 / dx12 ~49; demo cpu ~52 / gl ~116 / dx12 ~85-98.

## Known gaps / natural backlog

- Blueprint feature is COMPLETE (runs 2a + 2b): posed-mesh `components`,
  composite instantiation, compose UI, drag-place, runtime Behavior attach
  with per-entity error isolation, and the exec timeout guard. Residual
  limitations, in rough priority order:
  - **A timed-out script is abandoned, not stopped** — CPython cannot kill
    a thread. Each survivor burns a core and contends for the GIL: measured
    8.8 FPS clean → 0.66 with one alive (13x) on the starter scene's CPU
    renderer. Capped at `MAX_RUNAWAY_THREADS` = 2 so it can't accumulate,
    and the user is told to restart, but the first one's cost is real. The
    only true fix is OS-process isolation (subprocess + kill) — a genuine
    feature, and the natural run 3 if this ever bites in practice.
  - An infinite loop inside a Behavior's `update()` (as opposed to module
    level) still hangs the frame — the guard covers script exec only.
  - Blueprint instantiation is MESH-ONLY: a component asset's light / sun /
    fog_volume / environment aspects are ignored. A lamp component won't
    actually light anything.
  - An already-placed instance does not auto-update when its blueprint
    changes; re-placing picks up edits (matches every other asset type).
- wgpu directional (sun) shadow attenuation: DONE. `dl_shadow_tex` in
  `wgpu_renderer.py` mirrors GL's `dlShadowTex`; parity vs GL is asserted
  EXACT in wgpu_checks (both backends index the same per-face
  `directional_shadow_factors` output — no per-pixel resampling, so no
  float-divergence source). **wgpu is now at full visual parity with GL.**
- GPU cache identity: FULLY FIXED, both levels. Cache KEYS use monotonic
  `Mesh._cache_id`/`Entity._cache_id`; per-array VERSION STAMPS use
  `Mesh._color_version`/`_pbr_version`/`_opacity_version` (bumped by
  property setters on the five per-face arrays) and `Environment._image_id`.
  Regression tests: gl_checks #11/#12, wgpu_checks #12/#13, all four proven
  to fail against their pre-fix commits.
  **Do not "simplify" any of these back to `id()`** — CPython recycles
  addresses, and both failure modes were reproduced on both backends: a
  false KEY hit crashes or renders wrong geometry, a false STAMP match
  silently skips the rebuild (dragging a material slider could leave the
  change not showing). Note the per-face arrays are PROPERTIES now: an
  in-place write (`face_colors[:] = ...`) bypasses the setter and will NOT
  bump the version. Nothing in engine/editor does that today (only test
  setup) — keep it that way, or bump the counter explicitly.
- Per-pixel texturing — PRE-SPLIT INTO 4 RUNS, run 1/4 DONE:
  1. [done] Per-corner UV foundation. `Mesh.corner_uvs` (M,4,2) parallel to
     `faces`; FBX import now preserves the per-polygon-vertex UVs it used
     to discard. No rendering change.
  2. [done] CPU renderer per-pixel sampling. `MaterialGraph.
     direct_texture_bindings()` reports which output channels a texture
     feeds directly; `renderer._pixel_uv` interpolates `corner_uvs` at the
     already-reconstructed world position. Byte-identical gate for
     untextured scenes lives in `tests/fixtures/perpixel_starter_golden.npy`
     (captured from main @76cb82b — supervisor re-verified it against a real
     main render, 0 px, so it is NOT circular; keep it that way if you
     regenerate it).
  3. [done] OpenGL parity. `gpu_geometry._build_uv` gives per-vertex UVs in
     `_build_geometry`'s triangle-vertex order; GLSL samples per channel.
     Untextured golden at `tests/fixtures/perpixel_gl_untextured_golden.npy`
     (supervisor re-verified vs a real main render, 0 px — not circular).
  4. [done] wgpu/DX12 parity, in WGSL. Notably did NOT copy GL's V-flip:
     it uses `textureLoad` (no sampler — `rgba32float` is unfilterable-float
     and can't back `textureSample` without a non-portable feature) and
     hand-rolls the index to reproduce `sample_texture`'s own formula, so
     no flip is needed. Golden at
     `tests/fixtures/perpixel_wgpu_untextured_golden.npy`.
  **The slate is COMPLETE across CPU / OpenGL / wgpu.**

  Cross-backend facts worth keeping: NEAREST filter + REPEAT wrap + no sRGB
  on upload are what reach CPU parity; force `render_scale = 1` for any
  GPU-vs-CPU pixel comparison (scale 3 gives 1587/30000 differing px vs
  173/30000 at scale 1); textured GPU-vs-CPU parity is mean 0.266 /
  173 of 30000 px on BOTH GPU backends, and a silent flat-per-face
  regression measures mean ~13.3 / ~25% — that gap is what makes the
  mean<3.0 / <3% gates real gates.
  Consequences of runs 1-2 now live, NOT yet addressed:
  - **LOD swim**: decimated levels get a fresh box projection, so a textured
    mesh's pattern visibly JUMPS at the LOD switch distance. The most
    user-visible loose end of the slate.
  - GI/shadow bounce colour still reads per-face albedo, not the per-pixel
    texture (different cost shape: per bounce-source face, not per visible
    pixel).
  - Translucent entities never enter the deferred face-ID buffer, so
    `opacity` texture bindings are structurally unreachable in the CPU path.
  - Sampling is nearest-neighbour; real filtering needs bilinear AND mip
    generation together to help.
  - Worst-case cost is ~2x per-call when a textured surface fills the frame
    (scales with textured visible-pixel count). Untextured is unchanged.
  **Architectural decision already made for runs 2-4:** the material graph
  KEEPS baking per-face. Only a texture feeding a channel directly goes
  per-pixel — one array index per pixel per channel, which maps 1:1 onto
  GPU sampling. Evaluating the whole graph per pixel is ~285x more samples
  per frame (≈114k visible pixels vs ~400 faces) on an engine already at
  ~10 FPS; baking to a UV atlas instead would need real UV unwrapping,
  which doesn't exist here and breaks on overlapping box-projected UVs.
  Documented limitation: graph math *around* a texture still bakes per-face.
  Also still per-face and untouched: `export_fbx` writes one UV repeated
  per polygon vertex, so a round-trip through our own exporter flattens
  corner UVs.
- LOD/decimation is vertex-clustering (not QEM); BVH split is median (not
  SAH); shadow/GI still one-bounce, per-face.
- No undo system; folder deletion in the content browser; flat-mode (F2)
  translucency; import status double-logs (harmless).
- Longer-horizon game features: walking player (gravity), interactions
  (doors/pickups), enemy AI.

## State at last supervisor retirement (2026-07-14)

Big multi-day push, all merged + pushed through `e477b49`, working tree
clean, no wip branches, HANDOFF is the no-task stub. Shipped since the
2026-07-11 retirement:
- **Rendering:** full PBR (metallic-roughness GGX) across CPU/GL/wgpu;
  wgpu visual parity (sun disc, GI, fog); material transparency/opacity
  (blend modes) on all three backends; DX12 is now the DEFAULT renderer
  (CPU opt-in — user preference, in supervisor memory).
- **Editor UX:** resizable panels + fullscreen adaptive; dockable panels
  with drop-previews + TABBED docking; viewport toolbar (gizmo modes,
  World/Local); editable transform vectors; snapping (grid/floor/mesh) +
  Alt-drag gizmo duplicate; multi-selection + 5 Blender pivot modes
  (incl. 3D cursor); box/marquee select; collapsible left side toolbar +
  dockable engine CONSOLE (logs imports/compiles/lighting bakes).
- **Content:** content-browser folder tree; texture assets w/ previews +
  UV pipeline + TexCoord/TextureSample nodes; UE-style material node
  overhaul + drag-drop material assignment; FBX EXPORT; Unreal-style
  import-options dialog (scale/up-axis/fit-to-unit/folder nav); distance
  LOD w/ decimation; Python BLUEPRINT assets + in-engine script editor
  with compile/bug-check (posed meshes still TODO — see backlog).
- **Performance (the big one):** the user's 10k-face FBX import went from a
  ~5-MINUTE lighting-bake FREEZE to ~1.1s, via three merged changes —
  coarse-LOD shadow/GI occluder proxy (incl. on-demand decimation for
  meshes lacking LODs), ASYNC non-blocking bake (background thread, atomic
  install), and a BVH ray-tracer acceleration structure. Uncapped FPS by
  default + optional clamp + smoothed readout.

Process notes for the next supervisor: engine-coder runs THIS SESSION
consistently ran 1.5-2.7x the 150k token budget on large features — split
big/cross-cutting work (esp. anything touching all 3 renderers, threading,
or big refactors) into MORE, smaller runs, and enforce per-milestone
checkpoints. Several review send-backs and one supervisor-implemented fix
(the on-demand shadow proxy — the agent's version missed the user's
LOD-less asset) are recorded in the ledger. The user's real imported asset
files (assets/gat.json, assets/models/gat.npz, assets/folders.json,
assets/blueprints/) are UNTRACKED local data — never modify or commit them.
