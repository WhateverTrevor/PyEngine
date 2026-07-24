# No task in flight

BVH ray-trace acceleration was judged and squash-merged to main on
2026-07-14, completing the 2-run RT responsiveness effort (async bake +
BVH). engine/bvh.py is a pure-numpy flat BVH: build_bvh(v0,e1,e2),
any_hit(bvh,origins,dirs,t_max), nearest(bvh,origins,dirs). Traversal
advances a (ray_id,node_id) frontier one tree LEVEL per Python iteration —
no per-ray Python loop. raytrace.py's _intersect_any/_nearest_hit_faces
take an optional bvh= and route through it only above BVH_THRESHOLD (256);
at/below threshold the exact brute-force path runs, so all low-poly
content is byte-identical. The BVH is built once per occluder-soup version
(cached on the tracer, snapshot into lighting_bake for the worker thread).

RESULTS (RT effort complete): the user's 10,448-face Gat first-render
lighting bake went 295.63s (original freeze) -> 8.9s (coarse shadow proxy)
-> 1.14s (BVH), and that 1.14s runs on the async worker thread so the
frame never blocks (1.3ms dispatch). Supervisor independently verified:
BVH any_hit EXACT vs brute-force (0 mismatches / 1500 rays), nearest exact
(t-diff 0.0), 11x isolation speedup; starter scene byte-identical main
(brute) vs branch (BVH) — 0 differing pixels even though it's above the
threshold; all 25 suites pass.

NOTE: the starter scene (~368 occluder tris) is just above BVH_THRESHOLD
so it exercises the BVH in normal runs — verified byte-identical, but if a
future high-poly scene ever shows tie-break pixel diffs vs brute-force,
that's the expected sub-pixel nearest-hit tie-break behavior (raise the
threshold if it ever matters). The repo-root settings.json holds the
user's real 5120x1369 / pixel_scale 3 — headless benchmarks MUST use
PYENGINE_SETTINGS isolation or numbers look catastrophically slow.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: settings isolate via PYENGINE_SETTINGS; UI tests drive the real
event path; DX12 default; DO NOT touch assets/gat.*, folders.json,
blueprints/. Full battery is TWENTY-FIVE suites. FPS here is 2-10x slow +
high variance — same-environment A/B only; never run multi-minute benches.

Backlog: blueprint posed meshes (run 2 of blueprint) + infinite-loop guard
on script exec; QEM decimation; per-pixel texturing; folder deletion;
import status double-log cleanup; SAH BVH split (currently median).
