# No task in flight — blueprint run 2a merged 2026-07-25

Run 2a (blueprint posed meshes: `components`, composite instantiation,
compose UI, drag-place) is judged, squash-merged to main and pushed.
Working tree clean, `wip/blueprint-components` deleted.

Read `CLAUDE.md` first — "Verification" (the battery is now **26** suites),
"Known gaps / natural backlog", and the retirement-state section are the
authoritative summary.

## Next task, already scoped: blueprint run 2b

The other half of the feature the user scoped as "assets that house their
own posed meshes and code". 2a did the meshes; 2b does the running code:

1. Attach the compiled Behavior (`engine/blueprint.py`'s
   `compile_blueprint` already returns the class name) to an instantiated
   blueprint entity and run it per-frame, with PER-ENTITY error isolation:
   catch exceptions out of `update`, log once to the console, and disable a
   repeatedly-failing behavior instead of spamming every frame.
2. The infinite-loop guard on script exec. A `while True:` in a blueprint
   script currently HANGS THE EDITOR — `engine/blueprint.py`'s module
   docstring documents this as a known, deliberately-unsolved limitation
   from run 1. Worker-thread timeout is the intended fix. Note honestly
   that a thread cannot be force-killed in Python: a timeout can abandon
   the worker and report the failure without hanging the UI, but the
   runaway thread survives until process exit. Say so in the docstring
   rather than implying real sandboxing.

Small follow-ups from 2a's review, fold into 2b's brief:
- Blueprint instantiation is MESH-ONLY — a component asset's light / sun /
  fog_volume / environment aspects are ignored. Decide whether 2b honors
  lights (a lamp component that actually lights) or documents it as
  permanent.
- An already-placed instance does not auto-update when its blueprint
  changes; re-placing picks up edits. Matches other asset types.
- `tests/bp_component_checks.py` drives Ctrl+D / Del by calling
  `_duplicate_selected` / `_delete_selected` directly rather than through
  the real key event path. Low risk (the bindings are untouched and covered
  elsewhere) but it violates the standing UI-test rule — tighten it.

## If you are the next supervisor, know these five things

1. **The battery is 26 suites** (listed in CLAUDE.md). Name every one in
   every brief; agents have skipped the suites most coupled to their own
   changes and shipped a break that way. Exit code is the pass signal —
   three suites print no "PASSED" line at all (see CLAUDE.md).
2. **UI tests must drive the real pygame event path**, not handlers
   directly. Held modifiers need OS-boundary patching — see marquee_checks.
3. **FPS numbers here are untrustworthy**: 2-10x slow, high variance, and
   the repo-root settings.json (user's real 5120x1369) makes headless
   benches look catastrophic. Same-environment A/B only, with
   PYENGINE_SETTINGS isolation. Run 2a's A/B is a worked example.
4. **Budget**: see `.claude/token-ledger.md`. Large features consistently
   run 1.5-2.7x the 150k target. Split cross-cutting work and enforce a
   pushed checkpoint per milestone. STEP 0 (push a branch + HANDOFF
   skeleton before writing any code) is cheap insurance — it turned a
   second zero-output limit death into a recoverable run.
5. **Never touch the user's untracked asset data**: assets/gat.json,
   assets/models/gat.npz, assets/folders.json, assets/blueprints/.

When a task IS in flight, this file holds its resume state (task, DONE
with evidence, NEXT, known issues, temp-artifact paths) per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.
