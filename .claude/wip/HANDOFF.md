# No task in flight — GPU cache-identity fix merged 2026-07-26

The `id()`-keyed geometry/uniform cache bug is fixed on both GPU backends,
judged, squash-merged and pushed. Working tree clean, `wip/cache-identity`
deleted.

Read `CLAUDE.md` first — "Verification" (27 suites) and "Known gaps /
natural backlog".

## Next task: wgpu directional sun-shadow attenuation

**Still the only item left from the user's own stated priority list.** They
ranked it second (above per-pixel texturing, which is now complete); it has
been deferred twice by their own "go ahead" instructions on other work. Do
it next unless told otherwise.

GL has `dlShadowTex`; wgpu lacks directional (sun) shadow attenuation on
mesh faces — the last wgpu visual gap. Single-backend, so unlike the
texturing slate it does NOT need pre-splitting; budget one run.

Reusable pattern now well established for wgpu visual work: an
unaffected-scene golden regression gate captured from a real main worktree,
plus a dx12-vs-CPU (or vs-GL) parity gate whose tolerance is justified by
measured numbers AND a discrimination check showing what the failure
actually measures.

Then, from the backlog: the remaining id()-as-version-stamp risks
(`color_id`/`pbr_id`/`opacity_id`, `_get_env_tex`) — same bug class as the
one just fixed but with an inverted, silent failure mode; LOD swim (the
most user-visible loose end of the texturing slate); per-pixel GI bounce
colour; bilinear/mipmap filtering.

## If you are the next supervisor, know these six things

1. **The battery is 27 suites** (listed in CLAUDE.md). Name every one in
   every brief. **Exit code is the pass signal** — `window_checks`,
   `browser_checks` and `import_checks` print no "PASSED" line at all.
   `bp_runtime_checks` takes ~80s by design.
2. **UI tests must drive the real pygame event path**, not handlers
   directly. Held modifiers need OS-boundary patching — see marquee_checks
   or `ctrl_key()` in bp_component_checks.
3. **FPS numbers here are untrustworthy**: 2-10x slow, high variance, and
   they DRIFT over a long session. Interleaved same-environment A/B against
   a `git worktree` of main is the only meaningful form; remove the
   worktree before merging, since one holding `main` blocks
   `git checkout main`. Where a byte-identical gate exists it is STRONGER
   evidence than any FPS number that a code path is untouched — there are
   now four such gates (three per-pixel goldens + the CPU starter golden).
4. **Agents stop at "waiting for the battery"** — three of the eight runs
   this session did, twice leaving files uncommitted. The last five
   complied when told explicitly; keep the instruction, budget for doing it
   yourself anyway.
5. **STEP 0 works** and is now in every brief: push a branch + HANDOFF
   skeleton before writing any code.
6. **Never touch the user's untracked asset data**: assets/gat.json,
   assets/models/gat.npz, assets/folders.json, assets/blueprints/.

## The single highest-leverage supervisor habit

**Reproduce the bug yourself before briefing it.** The cache-identity run
was the cheapest of the session (176k, no send-back) precisely because the
brief carried a real traceback and a working repro script. It also changed
the DESIGN: reproducing it showed the defect was silent corruption rather
than a crash, which ruled out the count-sidecar fix that would otherwise
have looked reasonable. The ledger shows the same pattern on the
text-input and ctx-menu bugs.

## Verification habits that keep finding real things

Eight runs merged this session. Every one passed its own tests AND the full
battery; several still had defects or unproven claims found only by
independent checks:
- **Never trust a golden fixture's provenance.** All three per-pixel
  goldens would have proven nothing if captured from their own branch;
  re-render from a real main worktree and diff. All came back 0 px.
- **Never trust "this test would have caught it."** Run the new test
  against pre-fix code in a worktree and confirm it FAILS. Done for the
  cache fix; both suites failed pre-fix with an exact signature.
- **Test the tolerance, not just the value.** A parity gate is only a gate
  if a real bug exceeds it — measure the failure mode's magnitude.
- **Prove the feature isn't vacuous.** Corner UVs all equal, or a texture
  rendering one flat colour, would pass most assertions.
- **Find the case where a wrong implementation coincidentally passes.** A
  box-projected quad is globally affine, so an always-use-triangle-1
  barycentric bug looks correct until you build a non-affine quad.
- **Check the invalidation KEY, not the obvious field.** Run 2b warned on
  position changes; ShadowTracer keys on the full transform matrix.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.
