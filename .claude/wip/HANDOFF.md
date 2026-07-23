# No task in flight

Collapsible side toolbar + engine console was judged and squash-merged to
main on 2026-07-14. engine/console_log.py is a bounded ring-buffer log
(info/warn/error, timestamped) with module helpers log_info/warn/error +
get_log(). Producers wired: FBX/HDRI/texture import, blueprint compile,
the shadow/GI bake ("Baking lighting (N occluder triangles)..." /
"Lighting baked in X.Xs" in core.py's run loop), and Editor.status mirrors
into it. The Console is a first-class dockable/tabbable/minimizable panel
(id "console") in the Window menu; a new collapsible LEFT side toolbar
houses its toggle (declarative growable button list; collapse state
persists). All 22 suites pass; the agent also fixed a real forward-compat
bug (saved layout was discarded when a new panel id appeared).

REMAINING (run 2 of the user's request): FPS. The user reports the frame
rate "jumping around" and wants smooth UNCAPPED FPS with an OPTIONAL clamp
setting. Investigate engine/core.py's loop (fixed 60Hz update + present
timing). FOLD IN the async lighting bake here: move the shadow/GI bake
onto a BACKGROUND THREAD so it never stalls a frame (the console already
shows "Baking lighting…"; make it non-blocking + show progress). The bake
result must swap in atomically; render prior/no lighting until ready;
mind thread-safety on the tracer cache.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

KNOWN minor issue (not blocking): import status double-logs (redundant,
harmless) — clean up opportunistically. console_checks has NO bake-log
assertion (supervisor verified the bake logs via the real run loop
manually) — add one if touching console tests.

IMPORTANT: settings isolate via PYENGINE_SETTINGS; UI tests drive the real
event path; DX12 default; DO NOT touch assets/gat.*, folders.json,
blueprints/. Full battery is TWENTY-TWO suites. FPS here is 2-10x slow +
high variance — same-environment A/B only; never run multi-minute benches.

Backlog: BVH for the ray tracer (real shadow/GI perf); blueprint posed
meshes (run 2 of blueprint) + infinite-loop guard on script exec; QEM
decimation; per-pixel texturing; folder deletion.
