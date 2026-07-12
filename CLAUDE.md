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

- `py tests\smoke_test.py`     — full engine/editor battery (CPU paths)
- `py tests\gl_checks.py`      — OpenGL backend: CPU parity, IES, cone, shadows, depth
- `py tests\wgpu_checks.py`    — DX12 backend: 3-way parity, shadows, HDRI sky
- `py tests\env_checks.py`     — sun disc tracking, directional shadows, GI
  green-bleed, fog volumes, sky-material bake, HDRI import
- `py tests\window_checks.py`  — panel minimize/close/reset layout math,
  settings round-trip, material-editor bake
- FPS reference (post environment features, 2026-07-11): editor cpu ~23 /
  dx12 ~53; demo cpu ~44-52 / gl ~116 / dx12 ~81 / vulkan ~85. If a number
  regresses >20%, investigate before merging.

## Known gaps / natural backlog

- **Top queued task: wgpu directional (sun) shadow attenuation on mesh
  faces** — the only remaining wgpu visual gap after the parity merge
  (52f0601); mirror GL's `_upload_dl_shadow_tex` + `dlShadowTex`
  texelFetch pattern.
- Software flat mode (F2) is painter-only (approximate depth); GPU paths cap
  at 16 lights; wgpu path = offscreen+readback, no wireframe.
- Fog Volume boxes are world-axis-aligned (rotation ignored, v1). GI is
  one-bounce, per-face, static-scene cached (~300ms starter-scene bake).
- No undo system. FBX import is geometry+diffuse colors only (no textures/UVs).
- Next obvious features: walking player (gravity), interactions
  (doors/pickups), enemy AI, texture/UV pipeline.

## State at last supervisor retirement (2026-07-11)

Everything requested by the user is merged and pushed through `8ef3dc2`:
GPU backends (gl/dx12/vulkan + settings API selector), Sun (time-of-day
rotation, disc, directional ray-traced shadows), HDRI import + editable sky
materials (node editor, Unreal-vocabulary nodes), one-bounce GI, atmospheric
fog + Fog Volume assets, and full window management (minimize/close on every
panel, Window-menu registry, factory Reset Layout). No task in flight; no
wip branches; `.claude/wip/HANDOFF.md` is the no-task stub. Token ledger is
current through the window-management run.
