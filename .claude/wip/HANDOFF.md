# No task in flight

The high-poly shadow/GI freeze fix was judged and squash-merged to main on
2026-07-14. Root cause: ShadowTracer/GITracer built their ray-traced
occluder soup + receiver geometry from the full mesh, so placing a
10,448-face import made the first bake take ~295s (total engine lockup).
Fix: shadow/GI geometry routes through Entity.shadow_mesh() — the coarsest
LOD when the mesh has precomputed LODs, OR an on-demand decimated proxy
(cached, lod.generate_lods) for any mesh above lod.SHADOW_PROXY_THRESHOLD
(1000) that has NO LODs. The gather map (lod.shadow_gather_map) maps
proxy-face shadow/GI values back onto the rasterized faces across
CPU/GL/wgpu. Real Gat: 295s -> 8.9s. Built-ins (<=256 faces) keep
shadow_mesh() == mesh, byte-identical shadows.

SUPERVISOR NOTE: the agent's first pass only handled meshes WITH
precomputed LODs and missed the actual user asset (imported pre-LOD, no
LOD data) — the supervisor caught it (real Gat still 283s), the agent
thrashed (~575k tokens, stopped mid-benchmark), and the supervisor
finished the on-demand-proxy fix directly. 8.9s is still a one-time hitch
on placement; run 2's async bake makes it non-blocking.

Two features REMAIN from the user's request (2026-07-14):
1. Collapsible side toolbar + a minimizable/tabbable CONSOLE that reports
   what the engine is doing ("baking lighting…", import/compile progress,
   errors). Move the shadow/GI bake onto a BACKGROUND THREAD so even the
   8.9s hitch never blocks the UI; show progress in the console.
2. FPS: uncapped smooth frame pacing with an OPTIONAL clamp setting (the
   user reports the frame rate "jumping around"). Investigate the
   fixed-60Hz loop + present timing in engine/core.py.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: settings isolate via PYENGINE_SETTINGS; UI tests drive the real
event path; DX12 default; DO NOT touch assets/gat.* or folders.json /
blueprints. Full battery is TWENTY-ONE suites. FPS here is 2-10x slow +
high variance — same-environment A/B only.

Backlog: BVH for the ray tracer (real long-term shadow/GI perf); QEM
decimation; blueprint posed-mesh components (run 2 of blueprint) +
infinite-loop guard on script exec; per-pixel texturing; folder deletion.
