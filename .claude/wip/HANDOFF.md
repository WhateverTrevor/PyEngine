# No task in flight — per-pixel texturing slate COMPLETE 2026-07-26

All four runs (per-corner UV foundation, CPU sampling, OpenGL parity,
wgpu/DX12 parity) are judged, squash-merged to main and pushed. Working
tree clean, all wip branches deleted. Per-pixel texturing works on all
three backends.

Read `CLAUDE.md` first — "Verification" (27 suites) and "Known gaps /
natural backlog", which now carries the cross-backend parity facts and the
remaining work.

## Next task: wgpu directional sun-shadow attenuation

**This is the only item left from the user's own stated priority list.**
They ranked it second, above per-pixel texturing; it was skipped only
because they said "go ahead" on the texturing split. Do it next unless
told otherwise.

GL has `dlShadowTex`; wgpu lacks directional (sun) shadow attenuation on
mesh faces — the last wgpu visual gap. It is single-backend, so unlike the
texturing slate it does NOT need pre-splitting; budget one run.

Useful precedent now on disk: the texturing slate established a working
pattern for wgpu visual work — an untextured/unaffected golden regression
gate captured from a real main worktree, plus a dx12-vs-CPU (or vs-GL)
parity gate with a tolerance justified by measured numbers AND a
discrimination check. Reuse it.

After that, from the backlog: LOD swim (the most user-visible loose end of
the texturing slate — a textured pattern jumps at the LOD switch distance,
because decimated levels get a fresh box projection), the id()-keyed cache
trap, per-pixel GI bounce colour, bilinear/mipmap filtering.

## If you are the next supervisor, know these six things

1. **The battery is 27 suites** (listed in CLAUDE.md). Name every one in
   every brief. **Exit code is the pass signal** — `window_checks`,
   `browser_checks` and `import_checks` print no "PASSED" line at all.
   `bp_runtime_checks` takes ~80s by design.
2. **UI tests must drive the real pygame event path**, not handlers
   directly. Held modifiers need OS-boundary patching — see marquee_checks
   or `ctrl_key()` in bp_component_checks.
3. **FPS numbers here are untrustworthy**: 2-10x slow, high variance, and
   they DRIFT over a long session (the same demo measured 14 FPS early and
   9 FPS late on identical code). Interleaved same-environment A/B against
   a `git worktree` of main is the only meaningful form; remove the
   worktree before merging, since one holding `main` blocks
   `git checkout main`. Where a byte-identical gate exists it is STRONGER
   evidence than any FPS number that a code path is untouched.
4. **Agents stop at "waiting for the battery"** — three of the seven runs
   this session did, twice leaving files uncommitted. The last four
   complied when told explicitly; keep the instruction, budget for doing it
   yourself anyway.
5. **STEP 0 works.** A pushed branch + HANDOFF skeleton before any code
   turned a zero-output usage-limit death into a recoverable run. It is now
   in every brief.
6. **Never touch the user's untracked asset data**: assets/gat.json,
   assets/models/gat.npz, assets/folders.json, assets/blueprints/.

## Verification habits that earned their keep

Seven runs merged this session. Every one passed its own tests AND the full
battery; several still had defects or unproven claims found only by
independent checks:
- **Never trust a golden fixture's provenance.** All three per-pixel
  goldens would have proven nothing if captured from their own branch.
  Re-render from a real main worktree and diff. All three came back 0 px —
  the check has never failed, which is exactly why it stays cheap to run
  and worth running.
- **Test the tolerance, not just the value.** A parity gate is only a gate
  if a real bug exceeds it. Measure the failure mode's magnitude: a silent
  flat-per-face fallback is mean ~13.3 / ~25% differing against a 3.0 / 3%
  gate and 0.266 / 0.58% actual.
- **Prove the feature isn't vacuous.** Corner UVs all equal, or a texture
  rendering one flat colour, would pass most assertions. Count distinct
  colours within a single face; measure UV spread.
- **Find the case where a wrong implementation coincidentally passes.** A
  box-projected quad is globally affine, so an always-use-triangle-1
  barycentric bug looks correct until you build a non-affine quad.
- **Check the invalidation KEY, not the obvious field.** Run 2b warned on
  position changes; ShadowTracer keys on the full transform matrix.
- **Measure what the brief said to measure**, and probe next to it — run
  2b's shadow measurement validated its design, and probing the adjacent
  cost found unbounded thread accumulation no test covered.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.
