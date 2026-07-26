# No task in flight — per-pixel texturing run 2/4 merged 2026-07-26

Runs 1/4 (per-corner UV foundation) and 2/4 (CPU renderer per-pixel
sampling) are judged, squash-merged to main and pushed. Working tree clean,
both wip branches deleted. Textures now show real detail across a face on
the CPU renderer; the GPU backends still bake per-face.

Read `CLAUDE.md` first — "Verification" (27 suites) and the per-pixel
texturing entry in "Known gaps / natural backlog", which carries the 4-run
split, the architectural decision behind it, and the live consequences of
runs 1-2.

## Next task: per-pixel texturing run 3/4 — OpenGL

`engine/gl_renderer.py`. What run 2 leaves ready:
- `MaterialGraph.direct_texture_bindings()` is renderer-agnostic — call it
  exactly as the CPU path does to decide, per material, which output
  channels need a sampler bound versus which read the existing per-face
  baked attribute.
- `TextureBinding.port` says which channel of the sampled texture routes to
  the target output slot — the same semantics as `renderer._extract_channel`,
  which maps onto GLSL swizzles.
- `Mesh.corner_uvs` (M,4,2) is the vertex attribute to upload.
  **`Mesh.is_tri` is almost certainly NOT needed on the GPU**: the
  face-vs-triangle ambiguity that `_pixel_uv` exists to resolve is specific
  to the CPU face-ID buffer. `gpu_geometry.py` already expands to a real
  triangle list, so each of a quad's 6 vertices carries its own corner UV
  and the rasteriser interpolates natively. Don't port the barycentric math.
- The byte-identical fixture `tests/fixtures/perpixel_starter_golden.npy` is
  CPU-only. Run 3 wants its OWN gate: GPU-matches-CPU for an untextured
  scene (an existing pattern in gl_checks/wgpu_checks), plus a NEW
  GPU-matches-CPU-for-a-TEXTURED-scene parity check.

Then run 4 (wgpu, same shape in WGSL). Keep them separate.

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
   a `git worktree` of main is the only meaningful form. Remove the
   worktree before merging — one holding `main` blocks `git checkout main`.
   Where a byte-identical gate exists, it is STRONGER evidence than any FPS
   number that a code path is untouched.
4. **Agents stop at "waiting for the battery"** — three of the five runs
   this session did, twice leaving files uncommitted. Runs 1/4 and 2/4 both
   completed it in the foreground when told to explicitly, so keep the
   instruction; just budget for doing it yourself.
5. **STEP 0 works.** A pushed branch + HANDOFF skeleton before any code
   turned a zero-output usage-limit death into a recoverable run.
6. **Never touch the user's untracked asset data**: assets/gat.json,
   assets/models/gat.npz, assets/folders.json, assets/blueprints/.

## Verification habits that keep finding real things

Every merged run this session passed its own tests AND the full battery;
several still had defects or unproven claims found only by independent
checks:
- **Never trust a golden fixture's provenance.** Run 2's gate would prove
  nothing if the fixture had been captured from the branch. Re-rendering
  from a real main worktree (0 px, 948 distinct colours) is what made it
  evidence.
- **Prove the feature isn't vacuous.** Corner UVs that were all equal would
  pass every shape assertion; measure the spread. A texture that renders
  one flat colour would pass "it rendered"; count distinct colours per face.
- **Find the case where a wrong implementation coincidentally passes.** A
  box-projected quad is globally affine, so always-use-triangle-1 looks
  correct until you build a non-affine quad.
- **Measure what the brief said to measure**, and probe next to it — run
  2b's shadow measurement validated its design, and probing the adjacent
  cost found unbounded thread accumulation no test covered.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.
