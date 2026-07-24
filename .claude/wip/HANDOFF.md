# No task in flight

Asynchronous non-blocking lighting bake was judged and squash-merged to
main on 2026-07-14 (run 1 of a 2-run RT responsiveness effort; BVH is
run 2). The shadow/GI bake now runs on a background daemon thread:
engine/lighting_bake.py has snapshot_scene() -> bake() (pure, from an
immutable snapshot) + LightingBakeManager (dispatch/poll/install/coalesce),
driven once per frame in core.py's run() right after tracer.refresh(). The
frame that triggers a bake costs ~1.3ms (just snapshots + starts the
worker); the render keeps drawing with the last-ready lighting until the
worker finishes, then the result installs via an atomic reference swap
(pre-warming the existing ShadowTracer/GITracer caches, so renderer.py /
gl_renderer.py / wgpu_renderer.py needed ZERO changes and GPU parity is
preserved). tracer._bake_pending suppresses lazy blocking mid-bake.
force_sync (max_frames set OR synchronous=True) blocks exactly like before
— benchmarks/tests are deterministic and byte-identical.

Verified: sync path byte-identical to main (starter A/B 0.0/0.0); async
CONVERGES to the sync result exactly (0.0/0.0) delivered a couple frames
later; dispatch non-blocking (1.3ms); all 24 suites pass.

REMAINING (run 2 of the RT effort): BVH acceleration. The bake is now
non-blocking but the raw compute is still O(rays x occluder-triangles) —
the hot path is the dense Moller-Trumbore tests in engine/raytrace.py
(_intersect_any / _nearest_hit_faces / shadow_factors, called by
ShadowTracer.shadow_factors and GITracer.compute). Add a bounding-volume
hierarchy (or spatial grid) so each ray only tests nearby triangles
(O(log n)). Correctness bar: results match the current brute-force within
tolerance; it plugs into the now-pure bake() step. This makes the bake
itself fast (the async layer already keeps it off the main thread).

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: settings isolate via PYENGINE_SETTINGS; UI tests drive the real
event path; DX12 default; DO NOT touch assets/gat.*, folders.json,
blueprints/. Full battery is TWENTY-FOUR suites. FPS here is 2-10x slow +
high variance — same-environment A/B only; never run multi-minute benches.
Pre-existing tiny run-to-run non-determinism near the sky/sun-disc region
(present on main too) — not from these changes.

Backlog: BVH (above); blueprint posed meshes (run 2 of blueprint) +
infinite-loop guard on script exec; QEM decimation; per-pixel texturing;
folder deletion; import status double-log cleanup.
