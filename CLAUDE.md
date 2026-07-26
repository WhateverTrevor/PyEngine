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
- wgpu directional (sun) shadow attenuation on mesh faces (GL's dlShadowTex)
  — the last wgpu visual gap.
- Per-pixel texturing (materials bake per-face; would unlock texture-mapped
  roughness/metallic and higher-fidelity textures).
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
