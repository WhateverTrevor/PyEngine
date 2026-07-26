# No task in flight — GPU cache identity fully fixed 2026-07-26

The `id()`-as-version-stamp staleness fix is judged, squash-merged and
pushed. Working tree clean, `wip/stamp-identity` deleted.

Both levels of the GPU cache-identity bug class are now closed: the cache
KEYS (`Mesh._cache_id`/`Entity._cache_id`) and the per-array VERSION STAMPS
(`_color_version`/`_pbr_version`/`_opacity_version`, `Environment._image_id`).
Four regression tests, each proven to fail against its pre-fix commit.

The user's own stated priority list was cleared earlier today (blueprints,
per-pixel texturing across all three backends, wgpu sun-shadow attenuation).
Everything since has been follow-up work this session generated.

Read `CLAUDE.md` first — "Verification" (27 suites) and "Known gaps /
natural backlog".

## Next candidates — none are user-stated. Ask before picking.

1. **LOD swim** — decimated LOD levels get a fresh box projection, so a
   textured pattern visibly jumps at the switch distance. The most
   user-VISIBLE loose end of the per-pixel texturing slate.
2. **Per-pixel GI bounce colour** — the ray tracer still reads per-face
   albedo. Different cost shape (per bounce-source face, not per visible
   pixel), so it is its own design question, not a small extension.
3. **Bilinear + mipmap texture filtering** — sampling is nearest-neighbour
   on all three backends; real filtering needs both together to help.

Longer-horizon game features from the original backlog remain untouched and
are a different kind of work from the last several days of renderer
internals: walking player with gravity, interactions (doors/pickups),
enemy AI. Worth asking whether the user wants to switch tracks.

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
   `git checkout main`. Where a deterministic check exists, prefer it —
   the stamp fix was proved by build-function CALL COUNTS (exactly 1 over
   300-600 renders of an unchanged mesh), which beats any FPS number.
4. **Agents stop at "waiting for the battery"** — three of the ten runs
   this session did, twice leaving files uncommitted. The rest complied
   when told explicitly; keep the instruction, budget for doing it anyway.
5. **STEP 0 works** and is in every brief: push a branch + HANDOFF skeleton
   before writing any code.
6. **Never touch the user's untracked asset data**: assets/gat.json,
   assets/models/gat.npz, assets/folders.json, assets/blueprints/.

## The two highest-leverage supervisor habits

**Reproduce the bug yourself before briefing it.** Both cache-identity runs
prove it. The first was the cheapest of the session (176k, no send-back)
because the brief carried a traceback and a repro script. In BOTH cases
reproducing also corrected the DESIGN or the premise:
- the key bug turned out to be silent corruption rather than a crash, which
  ruled out the count-sidecar fix that looked reasonable on paper;
- the stamp bug turned out NOT to be reachable by a single array
  replacement (`apply` allocates the new array before freeing the old), only
  by two replacements with no render between — which is the exact shape the
  regression test had to take.

**Check that a claimed gate is a real gate.** Every one of these has caught
something, or would have:
- Re-render a golden fixture from a real main worktree; a fixture captured
  from its own branch proves nothing. All four came back 0 px.
- Run a new regression test against PRE-FIX code and confirm it FAILS. Done
  four times now; all four failed with exact signatures. One run shipped
  with NO committed test at all until review caught it.
- Measure what the failure mode actually looks like, so you know a
  tolerance discriminates (a silent flat-per-face texture fallback is mean
  ~13.3 / ~25% differing against a mean<3.0 / <3% gate).
- Write the CONTROL. "It stayed red" only means something once you have
  shown the scene turns green under an ordinary change.

And one that cost time: **look at the render, not just the number.** A
supervisor scene built to measure the sun-shadow gap put the shadow out of
frame and understated it as 0.06% against the real 4.16%. The images
settled it instantly. When building a sun scene: instantiate the real `Sun`
asset, call `scene.update` so `SunController` drives
`scene.light.direction`, and set the floor's `casts_shadow = False`.

## Agents pushing back is a good sign — twice it was right

- Run 1/4 of the texturing slate was told to derive `face_uvs` from corner
  UVs for a single source of truth; it found ~1 ULP drift would break the
  byte-identical gate, kept two projections, and added a 1e-9 agreement
  assertion instead.
- The stamp fix was told to prefer holding a strong reference to the
  stamped array. It implemented that, found it structurally prevents the
  address recycling a non-vacuous test must observe (the test would fail its
  own sanity assertion against FIXED code), and switched to version
  counters. The supervisor's recommendation was wrong.

When an agent contradicts the brief with evidence, check the evidence
rather than the instruction.

## Other verification habits worth keeping

- **Prove the feature isn't vacuous.** Corner UVs all equal, or a texture
  rendering one flat colour, would pass most assertions.
- **Find the case where a wrong implementation coincidentally passes.** A
  box-projected quad is globally affine, so an always-use-triangle-1
  barycentric bug looks correct until you build a non-affine quad.
- **Check the invalidation KEY, not the obvious field.**
- **Don't pad a tolerance you don't need.** The sun-shadow GL/dx12 parity
  is asserted exact because there is no per-pixel resampling.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.
