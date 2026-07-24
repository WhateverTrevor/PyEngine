# No task in flight

Uncapped FPS + optional clamp was judged and squash-merged to main on
2026-07-14. Engine max_fps defaults to 0 (uncapped; clock.tick(0) = no
cap); the editor Settings dialog has an FPS-cap row (Uncapped/30/60/120/
144) persisted to settings.json (old integer values still load). The HUD
FPS readout is a wall-clock EMA (FpsSmoother, tau 0.5s) so it stops
jumping. The fixed-60Hz scene.update accumulator and the deterministic
max_frames benchmark path are untouched. A static scene does NOT re-bake
per frame (tracer short-circuit verified).

The user's console + FPS requests from 2026-07-14 are now BOTH done
(console/side-toolbar + uncapped/clamp FPS).

REMAINING follow-up (the async lighting bake): move the shadow/GI bake
onto a BACKGROUND THREAD so an edit-triggered re-bake never stalls a
frame. The console already shows "Baking lighting (N)…"/"Lighting baked
in X.Xs"; make it non-blocking (compute into a fresh result on a worker,
swap in atomically, render prior/no lighting until ready, mind tracer-
cache thread-safety). This is the single riskiest remaining change — scope
it alone, checkpoint hard. The per-placement hitch is currently ~8.9s for
the user's 10k Gat (down from 295s) but still synchronous.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

Minor: a stale off-preset max_fps (e.g. 87) works but highlights no cap
button. import status double-logs (harmless). console_checks lacks a
bake-log assertion (supervisor verified manually).

IMPORTANT: settings isolate via PYENGINE_SETTINGS; UI tests drive the real
event path; DX12 default; DO NOT touch assets/gat.*, folders.json,
blueprints/. Full battery is TWENTY-THREE suites. FPS here is 2-10x slow +
high variance — same-environment A/B only; never run multi-minute benches.

Backlog: async lighting bake (above); BVH for the ray tracer; blueprint
posed meshes (run 2 of blueprint) + infinite-loop guard on script exec;
QEM decimation; per-pixel texturing; folder deletion.
