# No task in flight — per-pixel texturing run 1/4 merged 2026-07-25

Run 1/4 of the per-pixel texturing slate (per-corner UV foundation) is
judged, squash-merged to main and pushed. Working tree clean,
`wip/corner-uvs` deleted. No rendering behaviour changed — that starts in
run 2.

Read `CLAUDE.md` first — "Verification" (27 suites), "Known gaps / natural
backlog" (which carries the 4-run split and the architectural decision
behind it), and the retirement-state section.

## Next task: per-pixel texturing run 2/4 — CPU renderer

The foundation is in: `Mesh.corner_uvs` (M,4,2) parallel to `faces`, real
per-corner detail preserved through FBX import, npz round-trip with legacy
fallback, LOD deliberately dropping them.

Run 2 makes the CPU renderer sample per pixel:
- `engine/renderer.py`'s deferred pass already builds a face-ID buffer and
  reconstructs per-pixel world positions by ray-plane intersection, then
  indexes per-face arrays (`f_roughness` etc.) by face id. Per-pixel UV =
  barycentric interpolation of that face's `corner_uvs` at the
  reconstructed world position.
- Sample the texture per pixel for channels a texture feeds DIRECTLY. The
  material graph keeps baking per-face (see CLAUDE.md for why: per-pixel
  graph evaluation is ~285x more samples per frame).
- `engine/texture.py`'s `sample_texture` is already vectorised over an
  (M,2) array of UVs and nearest-neighbour — it should serve per-pixel
  sampling unchanged, but its module docstring still claims "this engine
  bakes materials to per-face colors... so there is no per-pixel footprint
  to filter against". That justification expires in run 2; revisit whether
  nearest-neighbour is still acceptable (it probably is for now — say so
  explicitly rather than leaving a stale rationale).
- MUST be gated so an untextured material stays byte-identical. That gate
  is what makes the run reviewable.

Then run 3 (OpenGL) and run 4 (wgpu). Keep them separate.

## If you are the next supervisor, know these six things

1. **The battery is 27 suites** (listed in CLAUDE.md). Name every one in
   every brief. **Exit code is the pass signal** — `window_checks`,
   `browser_checks` and `import_checks` print no "PASSED" line at all.
   `bp_runtime_checks` takes ~80s by design (it holds a deliberate runaway
   thread and a moving shadow caster); that's not a hang.
2. **UI tests must drive the real pygame event path**, not handlers
   directly. Held modifiers need OS-boundary patching — see marquee_checks
   or `ctrl_key()` in bp_component_checks.
3. **FPS numbers here are untrustworthy**: 2-10x slow and high variance,
   and this working dir drifts over a long session (the same demo measured
   14 FPS early and 9 FPS late on identical code). Same-environment A/B
   against a `git worktree` of main, interleaved, is the only meaningful
   form. NOTE: a worktree holding `main` blocks `git checkout main` in the
   primary tree — remove it before merging.
4. **Agents stop at "waiting for the battery."** Three runs in a row have
   ended with the agent launching the battery in the background, reporting
   that it was waiting, and stopping — twice leaving new files uncommitted.
   Budget for the supervisor checkpointing loose files and running the
   final battery itself. Explicitly forbidding it in the brief did not
   work; run 1/4's agent did complete it in the foreground when told, so
   keep the instruction, just don't rely on it.
5. **STEP 0 works.** A pushed branch + HANDOFF skeleton before any code
   turned a zero-output usage-limit death into a recoverable run.
6. **Never touch the user's untracked asset data**: assets/gat.json,
   assets/models/gat.npz, assets/folders.json, assets/blueprints/.
   `gat.npz` is a genuinely useful READ-ONLY subject: 10448 faces, no
   face_uvs/corner_uvs/PBR keys at all, so it exercises every legacy
   fallback path.

## Verification habits that keep paying off

Every merged run this session passed its own tests AND the full battery,
and several still had real defects found only by independent checks:
- **Prove the feature isn't vacuous.** Run 1/4's corner UVs would satisfy
  every shape/padding assertion if the implementation copied one UV to all
  four corners — so measure the actual spread (it's 100% varying).
  Likewise 2a's winding flip needed a control showing the naive mirror is
  100% inward-facing.
- **Measure what the brief said to measure.** 2b's agent skipped the
  shadow-cache measurement; taking it validated its design decision, and
  probing the neighbouring cost found unbounded thread accumulation that
  no test covered.
- **Check the invalidation KEY, not the obvious field.** 2b warned on
  position changes; ShadowTracer keys on the full transform matrix.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.
