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

- `py tests\smoke_test.py`   — full engine/editor battery (CPU paths)
- `py tests\gl_checks.py`    — OpenGL backend: CPU parity, IES, cone, shadows, depth
- `py tests\wgpu_checks.py`  — DX12 backend: 3-way parity, shadows, HDRI sky
- FPS reference (regressions): editor cpu ~33 / gl ~58-80 / dx12 ~45;
  demo cpu ~54 / gl ~117 / dx12 ~88 / vulkan ~85.

## Known gaps / natural backlog

- Software flat mode (F2) is painter-only (approximate depth); GPU paths cap
  at 16 lights; wgpu path = offscreen+readback, no wireframe.
- No undo system. FBX import is geometry+diffuse colors only (no textures/UVs).
- Next obvious features: walking player (gravity), interactions
  (doors/pickups), enemy AI, texture/UV pipeline.
