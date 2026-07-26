# No task in flight — per-pixel texturing run 3/4 merged 2026-07-26

Runs 1/4 (per-corner UV foundation), 2/4 (CPU per-pixel sampling) and 3/4
(OpenGL parity) are judged, squash-merged to main and pushed. Working tree
clean, all wip branches deleted. Textures show real per-pixel detail on the
CPU and OpenGL renderers; wgpu still bakes per-face.

Read `CLAUDE.md` first — "Verification" (27 suites) and the per-pixel
texturing entry in "Known gaps / natural backlog", which carries the 4-run
split, the architectural decision, the live consequences, and the two traps
run 4 must re-derive.

## Next task: per-pixel texturing run 4/4 — wgpu (LAST of the slate)

`engine/wgpu_renderer.py`, WGSL. It has been byte-untouched through runs
1-3, so this is a clean port of run 3's GL work.

Ready for it:
- `MaterialGraph.direct_texture_bindings()` — renderer-agnostic, same call
  CPU and GL both make.
- `gpu_geometry._build_uv(mesh)` — per-vertex UVs in `_build_geometry`'s
  triangle-vertex row order, ready to import unchanged.
- `TextureBinding.port` -> WGSL swizzle, mirroring
  `renderer._extract_channel` (scalar port to a vector channel replicates
  across RGB; vector port to a scalar channel reduces by mean).
- Do NOT port `Mesh.is_tri` or `renderer._pixel_uv`'s barycentric math —
  CPU-only, for resolving face-vs-triangle from the CPU face-ID buffer.

**Verify independently, do not assume they carry over from GL:**
- The V-flip (CPU maps v=0 to the image's bottom row).
- NEAREST filter, REPEAT wrap, no sRGB on upload.
- Force `render_scale = 1` for any GPU-vs-CPU pixel comparison.

Gates run 4 should have: an untextured regression gate (its own golden,
captured from a real main worktree — verify it isn't circular), and a
textured wgpu-vs-CPU parity gate with a tolerance justified by numbers. The
supervisor checked run 3's tolerance by measuring what a silent flat-
fallback bug looks like (mean 13.3 / 25.2% differing, vs a 3.0 / 3% gate
and 0.266 / 0.58% actual) — do the same rather than picking a round number.

Known limitation to expect: translucent entities never enter the deferred
face-ID buffer on CPU and always pass `{}` bindings on GL, so `opacity`
texture bindings stay structurally unreachable. Mirror that, don't "fix" it.

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
4. **Agents stop at "waiting for the battery"** — three of the six runs
   this session did, twice leaving files uncommitted. The last three
   complied when told explicitly, so keep the instruction; budget for doing
   it yourself anyway.
5. **STEP 0 works.** A pushed branch + HANDOFF skeleton before any code
   turned a zero-output usage-limit death into a recoverable run.
6. **Never touch the user's untracked asset data**: assets/gat.json,
   assets/models/gat.npz, assets/folders.json, assets/blueprints/.

## Verification habits that keep finding real things

Every merged run this session passed its own tests AND the full battery;
several still had defects or unproven claims found only by independent
checks:
- **Never trust a golden fixture's provenance.** Both per-pixel goldens
  would have proven nothing if captured from their own branch. Re-render
  from a real main worktree; both came back 0 px, which is what made them
  evidence rather than assertion.
- **Test the tolerance, not just the value.** A parity gate is only a gate
  if a real bug exceeds it — measure the failure mode's actual magnitude.
- **Prove the feature isn't vacuous.** Corner UVs that were all equal, or a
  texture rendering one flat colour, would pass most assertions. Count
  distinct colours within a single face.
- **Find the case where a wrong implementation coincidentally passes.** A
  box-projected quad is globally affine, so an always-use-triangle-1
  barycentric bug looks correct until you build a non-affine quad.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.
