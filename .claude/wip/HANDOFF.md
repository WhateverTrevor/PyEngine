# No task in flight — blueprint feature COMPLETE 2026-07-25

Runs 2a (posed meshes) and 2b (runtime behavior + exec timeout guard) are
both judged, squash-merged to main and pushed. Working tree clean, both
wip branches deleted. The blueprint feature the user scoped as "assets
that house their own posed meshes and code" is done end to end.

Read `CLAUDE.md` first — "Verification" (the battery is now **27** suites),
"Known gaps / natural backlog", and the retirement-state section are the
authoritative summary.

## Next candidates (user's own priority order)

1. **wgpu directional sun-shadow attenuation** — the last wgpu visual gap
   (GL has `dlShadowTex`; wgpu lacks it). Was #2 on the user's list.
2. **Per-pixel texturing** — materials currently bake per-face; this would
   unlock texture-mapped roughness/metallic and higher-fidelity textures.
   Was #3 on the user's list.
3. Blueprint follow-ups, only if they bite in practice — see CLAUDE.md's
   backlog for the residual limits (abandoned runaway threads, no guard on
   an infinite loop inside `update()`, mesh-only components). The real fix
   for the first is OS-process isolation (subprocess + kill), a genuine
   feature-sized run.

Both #1 and #2 touch render backends. #1 is wgpu-only so it is naturally
single-backend; **#2 is cross-cutting across CPU/GL/wgpu and MUST be
pre-split** — that is the single most repeated lesson in the ledger.

## If you are the next supervisor, know these six things

1. **The battery is 27 suites** (listed in CLAUDE.md). Name every one in
   every brief. **Exit code is the pass signal** — `window_checks`,
   `browser_checks` and `import_checks` print no "PASSED" line at all, so a
   runner that greps for one reports three false failures (this happened).
2. **UI tests must drive the real pygame event path**, not handlers
   directly. Held modifiers need OS-boundary patching (`pygame.key.
   get_pressed`) — see marquee_checks, or `ctrl_key()` in
   bp_component_checks.
3. **FPS numbers here are untrustworthy**: 2-10x slow, high variance, and
   the repo-root settings.json (user's real 5120x1369) makes headless
   benches look catastrophic. Same-environment A/B only. Run 2a is a worked
   example: an apparent regression vs the reference numbers was disproved
   by interleaving branch and a main worktree.
4. **Agents stop at "waiting for the battery."** Both blueprint runs ended
   with the agent launching the battery in the background, reporting that
   it was waiting, and stopping — leaving its new suite UNCOMMITTED and the
   battery unconfirmed. Budget for the supervisor checkpointing loose files
   and running the final battery itself. Telling the agent not to do this
   (2b's brief did, explicitly) did not prevent it.
5. **STEP 0 works.** Requiring a pushed branch + HANDOFF skeleton before
   any code turned a zero-output usage-limit death into a recoverable run.
   Keep it in every brief.
6. **Never touch the user's untracked asset data**: assets/gat.json,
   assets/models/gat.npz, assets/folders.json, assets/blueprints/.

## Verification habits that earned their keep this session

Both merged runs passed their own tests and the full battery, and both
still had real defects found only by INDEPENDENT checks the agent hadn't
written:
- Prove a claimed invariant with a CONTROL. 2a's negative-scale winding
  flip was verified by also showing the naive mirror is 100% inward-facing
  — without that, "normals look fine" proves nothing.
- Measure what the brief said to measure. 2b's agent left the shadow-cache
  cost unmeasured; measuring it (2.44 vs 6.30 FPS) is what validated its
  design decision, and probing the runaway-thread cost found unbounded
  accumulation (8.83 -> 0.66 -> 0.31 FPS) that no test covered.
- Check the cache/invalidation KEY, not the obvious field. 2b warned on
  position changes, but ShadowTracer keys on the full transform matrix, so
  rotation-only animation thrashed silently.

When a task IS in flight, this file holds its resume state (task, DONE
with evidence, NEXT, known issues, temp-artifact paths) per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.
