# No task in flight — supervisor retirement 2026-07-14

Everything the user asked for is merged and pushed through `e477b49`.
Working tree clean, no wip branches, no agent runs in flight.

Read `CLAUDE.md` first — its "Verification", "Known gaps / natural
backlog", and "State at last supervisor retirement (2026-07-14)" sections
were refreshed at this retirement and are the authoritative summary of
what shipped, what's queued, and the process lessons.

## If you are the next supervisor, know these five things

1. **The test battery is 25 suites** (listed in CLAUDE.md). Name every one
   in every brief; agents have skipped the suites most coupled to their
   own changes and shipped a break that way.
2. **UI tests must drive the real pygame event path**, not handlers
   directly (a basic click crash slipped through handler-only tests).
   Held modifiers need OS-boundary patching — see marquee_checks.
3. **FPS numbers here are untrustworthy**: 2-10x slow, high variance, and
   the repo-root settings.json (user's real 5120x1369) makes headless
   benches look catastrophic. Same-environment A/B only, with
   PYENGINE_SETTINGS isolation.
4. **Budget**: engine-coder runs this session ran 1.5-2.7x the 150k
   target on large features (worst: 405k threading rewrite, and one
   ~575k thrash across two stops). Split cross-cutting work (3 render
   backends / threading / big refactors) into more, smaller runs and
   enforce a pushed checkpoint per milestone.
5. **Never touch the user's untracked asset data**: assets/gat.json,
   assets/models/gat.npz, assets/folders.json, assets/blueprints/.

## Immediate candidates (user's own priorities first)

- Blueprint POSED MESHES — run 2 of the blueprint feature the user
  explicitly scoped ("assets that house their own posed meshes and
  code"). Schema field `components` already exists and is empty, so the
  work is additive. Pair it with the missing infinite-loop guard on
  script exec (a `while True:` currently hangs the editor).
- wgpu directional sun-shadow attenuation (last wgpu visual gap).
- Per-pixel texturing (unlocks texture-mapped PBR).

When a task IS in flight, this file holds its resume state (task, DONE
with evidence, NEXT, known issues, temp-artifact paths) per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.
