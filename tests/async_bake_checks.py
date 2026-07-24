"""Judge checks: asynchronous, non-blocking lighting bake (engine/lighting_bake.py).

Covers the Milestone 1 snapshot -> pure compute -> result refactor (purity,
determinism, no scene/tracer mutation, synchronous path byte-identical to
the original lazy per-(entity,light) `ShadowTracer`/`GITracer` API) and the
Milestone 2 background-worker manager (dispatch, poll-until-ready, install,
supersede/coalescing, determinism under `synchronous=True`). Threads are
always explicitly joined before assertions -- no wall-clock racing.
"""
import math
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np
import pygame

import engine
from engine.lighting_bake import LightingBakeManager, bake, snapshot_scene
from engine.raytrace import GITracer, ShadowTracer
from engine.renderer import Renderer, _face_light_strength, _gather_lights, _world_face_geometry


class _EngStub:
    def __init__(self, scene):
        self.scene = scene
        self.input = None


def build_scene(n_pillars=4, gi=True):
    scene = engine.Scene(
        light=engine.DirectionalLight(engine.Vec3(-0.4, -1.0, -0.25), ambient=0.15))
    scene.gi = {"enabled": gi, "intensity": 1.3, "samples": 10}
    lib = engine.AssetLibrary(os.path.join(WT, "assets"))
    sun = lib.instantiate("Sun")
    sun.transform.rotation = engine.Vec3(-0.4, 0.6, 0.0)
    scene.add(sun)
    scene.add(engine.Entity("floor", mesh=engine.checkerboard(8, 1.0)))
    scene.add(engine.Entity("wall", mesh=engine.box(6.0, 3.0, 0.3, color=(30, 200, 60)),
                            position=engine.Vec3(0, 1.5, -2.0)))
    for i in range(n_pillars):
        angle = 2.0 * math.pi * i / max(n_pillars, 1)
        scene.add(engine.Entity(f"pillar{i}", mesh=engine.cube(1.0, color=(200, 90, 60)),
                                position=engine.Vec3(math.cos(angle) * 3.0, 0.5,
                                                     math.sin(angle) * 3.0)))
    scene.add(engine.Entity("lamp", light=engine.PointLight(intensity=4.0, range=20,
                                                            cast_shadows=True),
                            position=engine.Vec3(1.5, 3.0, 1.5)))
    scene.update(1 / 60, _EngStub(scene))  # sync SunController once
    return scene


def lazy_reference(scene, tracer, r):
    """Every (entity, sun/light) shadow factor + GI array via the original,
    untouched lazy `ShadowTracer`/`GITracer` API -- what render() consumed
    before this task, and what it still falls back to for anything a bake
    hasn't covered yet. Used as the ground truth for byte-identical checks."""
    out = {}
    dl_dir = scene.light.direction.to_array()
    to_light = -dl_dir / np.linalg.norm(dl_dir)
    for e in scene.entities:
        if e.mesh is None or not e.visible:
            continue
        sm = e.shadow_mesh()
        c, n = _world_face_geometry(e, sm)
        lam = np.clip(n @ to_light, 0.0, 1.0)
        active = lam > 1e-3
        out[("sun", e)] = tracer.directional_shadow_factors(e, dl_dir, 2.0, 8, c, n, active).copy()
        for info in _gather_lights(scene):
            strength = _face_light_strength(info, n, c)
            act2 = strength > 1e-3
            if act2.any() and info.light.cast_shadows:
                out[("light", e, info.light)] = tracer.shadow_factors(
                    e, info.light, info.pos, c, n, act2).copy()
    gi_map = r._gi_contrib(scene, tracer)
    for e in scene.entities:
        gi = gi_map.get(id(e))
        if gi is not None:
            out[("gi", e)] = gi.copy()
    return out


def assert_all_equal(a: dict, b: dict, label: str) -> None:
    assert set(a) == set(b), (label, set(a) ^ set(b))
    for k in a:
        assert np.array_equal(a[k], b[k]), (label, k, np.abs(a[k] - b[k]).max())


# ============================================================================
# 1. bake() is pure: deterministic (same snapshot -> identical output twice)
#    and mutates neither the snapshot's own arrays nor the source scene.
# ============================================================================
scene = build_scene()
tracer = ShadowTracer()
tracer.refresh(scene)
snap = snapshot_scene(scene, tracer)

positions_before = [(e.name, e.transform.position.x, e.transform.position.y,
                    e.transform.position.z) for e in scene.entities]
occ_before = tuple(a.copy() for a in snap.occ)

result1 = bake(snap)
result2 = bake(snap)

positions_after = [(e.name, e.transform.position.x, e.transform.position.y,
                   e.transform.position.z) for e in scene.entities]
assert positions_before == positions_after, "bake() mutated entity transforms"
assert all(np.array_equal(a, b) for a, b in zip(occ_before, snap.occ)), \
    "bake() mutated the snapshot's occluder soup in place"
assert set(result1.sun_factors) == set(result2.sun_factors)
assert all(np.array_equal(result1.sun_factors[k], result2.sun_factors[k]) for k in result1.sun_factors)
assert set(result1.light_factors) == set(result2.light_factors)
assert all(np.array_equal(result1.light_factors[k], result2.light_factors[k]) for k in result1.light_factors)
assert np.array_equal(result1.gi_concat, result2.gi_concat)
print(f"bake() purity OK: {len(result1.sun_factors)} sun + {len(result1.light_factors)} "
     f"light entries deterministic, no scene/snapshot mutation")

# `bake` takes no scene/tracer/entity argument at all -- structurally cannot
# touch shared mutable state beyond what's already baked into the snapshot's
# plain numpy arrays and read-only records.
import inspect
params = list(inspect.signature(bake).parameters)
assert params == ["snapshot"], params
print("bake() signature OK: pure function of a single immutable snapshot")

# ============================================================================
# 2. synchronous path (LightingBakeManager, no threading involved) produces
#    results BYTE-IDENTICAL to the original lazy per-(entity,light) API.
# ============================================================================
scene = build_scene()
r = Renderer()
tracer_lazy = ShadowTracer()
tracer_lazy.refresh(scene)
reference = lazy_reference(scene, tracer_lazy, r)

# SAME scene (so entity/light object identities match `reference`'s keys),
# but a fresh ShadowTracer/Renderer/GITracer driven entirely through the
# bake manager instead of ad hoc lazy calls.
tracer_baked = ShadowTracer()
tracer_baked.refresh(scene)
r2 = Renderer()
mgr = LightingBakeManager()
assert mgr.needs_bake(scene, tracer_baked)
status = mgr.update(scene, tracer_baked, r2._gi, synchronous=True)
assert status["dispatched"] and status["installed"] is not None, status
secs = status["installed"].bake_seconds
assert mgr.needs_bake(scene, tracer_baked) is False, "manager should be settled right after a sync update()"
baked = lazy_reference(scene, tracer_baked, r2)  # now reads from the warmed cache
assert_all_equal(reference, baked, "sync bake vs lazy reference")
print(f"synchronous bake path OK: byte-identical to the lazy reference "
     f"({len(reference)} entries, baked in {secs:.3f}s)")

# ============================================================================
# 3. max_frames / benchmark determinism: engine.run() with max_frames set
#    forces LightingBakeManager.update()'s `synchronous` path (see engine/
#    core.py's `force_sync = max_frames is not None or synchronous`) so a
#    deterministic headless run never leaves a bake in flight.
# ============================================================================
scene3 = build_scene(n_pillars=2)
camera = engine.Camera(position=engine.Vec3(0.0, 3.0, 8.0), pitch=-0.2)
eng = engine.Engine(160, 120, title="async_bake_checks", api="cpu")
eng.run(scene3, camera, max_frames=5)
assert eng.tracer._world_version >= 0, "no bake ran during a max_frames run"
assert eng.bake_manager._last_world_version == eng.tracer._world_version, \
    "bake manager did not settle by the end of a deterministic max_frames run"
assert not eng.bake_manager.busy() and not eng.tracer._bake_pending, \
    "max_frames must never leave an async bake in flight"
print("max_frames determinism OK: bake settled synchronously within 5 frames, no stray pending bake")

# the explicit `synchronous=True` flag (no max_frames) must force the same
# inline behavior -- run a handful of frames via the plain run() loop with
# an overlay that stops it, standing in for "max_frames" without setting it.
scene3b = build_scene(n_pillars=2)
eng2 = engine.Engine(160, 120, title="async_bake_checks_sync_flag", api="cpu")


class _StopAfter:
    def __init__(self, n):
        self.n = n

    def __call__(self, e):
        self.n -= 1
        if self.n <= 0:
            pygame.event.post(pygame.event.Event(pygame.QUIT))


eng2.run(scene3b, camera, max_frames=None, synchronous=True, overlay=_StopAfter(5))
assert not eng2.bake_manager.busy() and not eng2.tracer._bake_pending, \
    "synchronous=True (no max_frames) must also never leave an async bake in flight"
print("synchronous=True flag OK: forces the same inline bake path as max_frames")

# ============================================================================
# 4. async dispatch does not block the caller; poll-until-ready (the worker
#    is explicitly .join()-ed -- deterministic, no wall-clock racing)
#    installs and matches the lazy reference exactly, same as sync.
# ============================================================================
scene4 = build_scene()
r4 = Renderer()
tracer_lazy4 = ShadowTracer()
tracer_lazy4.refresh(scene4)
reference4 = lazy_reference(scene4, tracer_lazy4, r4)

tracer4 = ShadowTracer()
tracer4.refresh(scene4)
r4b = Renderer()
mgr4 = LightingBakeManager()
status = mgr4.update(scene4, tracer4, r4b._gi, synchronous=False)
assert status["dispatched"] and status["installed"] is None, \
    "async dispatch must return immediately with nothing installed yet"
assert mgr4.busy(), "manager should report a bake in flight right after async dispatch"
assert tracer4._bake_pending, "tracer should be marked pending while the async bake runs"

# a cache miss while a bake is pending must return the safe default (fully
# lit), not trace synchronously -- that would be exactly the stall this
# task exists to eliminate.
floor4 = next(e for e in scene4.entities if e.name == "floor")
lamp_info = _gather_lights(scene4)[0]
untouched = tracer4.shadow_factors(floor4, lamp_info.light, lamp_info.pos,
                                   np.zeros((3, 3)), np.array([[0.0, 1.0, 0.0]] * 3),
                                   np.array([True, True, True]))
assert np.array_equal(untouched, np.ones(3)), \
    "a cache miss while a bake is pending must return the safe default, not trace"
gi_pending = r4b._gi_contrib(scene4, tracer4)
assert gi_pending == {}, "GI must not recompute inline while a bake is pending"

mgr4._thread.join()  # deterministic: wait for the real background bake to finish
status2 = mgr4.update(scene4, tracer4, r4b._gi, synchronous=False)
assert status2["installed"] is not None, "poll after join should install the finished bake"
assert not mgr4.busy() and not tracer4._bake_pending
baked4 = lazy_reference(scene4, tracer4, r4b)
assert_all_equal(reference4, baked4, "async bake vs lazy reference")
print(f"async dispatch/poll/install OK: non-blocking dispatch, safe-default cache "
     f"misses while pending, byte-identical after join+install ({len(reference4)} entries)")

# ============================================================================
# 5. rapid successive scene changes: never crashes, and the FINAL installed
#    result matches the FINAL world -- coalescing (intermediate versions
#    while a bake is in flight are superseded, never separately baked).
# ============================================================================
scene5 = build_scene(n_pillars=6)  # bake takes tens of ms -- plenty of time
tracer5 = ShadowTracer()           # for the mutation loop below to overlap it
tracer5.refresh(scene5)
r5 = Renderer()
mgr5 = LightingBakeManager()


def settle(mgr, scene, tracer, gi_tracer):
    """Drain any in-flight bake to completion (deterministic join, no
    wall-clock racing), re-dispatching if the world moved on meanwhile,
    until nothing is pending."""
    for _ in range(10):
        if mgr._thread is not None:
            mgr._thread.join()
        status = mgr.update(scene, tracer, gi_tracer, synchronous=False)
        if not status["dispatched"] and mgr._thread is None:
            return
    raise AssertionError("bake manager did not settle within 10 update() cycles")


saw_overlap = False
for i in range(5):
    scene5.add(engine.Entity(f"extra{i}", mesh=engine.cube(0.5, color=(10 * i, 40, 90)),
                             position=engine.Vec3(i * 0.7, 0.25, 4.0)))
    tracer5.refresh(scene5)
    status = mgr5.update(scene5, tracer5, r5._gi, synchronous=False)  # must never raise
    if mgr5.busy() and not status["dispatched"]:
        saw_overlap = True  # a change landed while a prior bake was still in flight

settle(mgr5, scene5, tracer5, r5._gi)
assert not mgr5.busy() and not tracer5._bake_pending
assert mgr5._last_world_version == tracer5._world_version, \
    "manager did not converge to the final world version"

tracer5_lazy = ShadowTracer()  # ground truth: fresh lazy tracer, final scene state
tracer5_lazy.refresh(scene5)
r5_lazy = Renderer()
reference5 = lazy_reference(scene5, tracer5_lazy, r5_lazy)
baked5 = lazy_reference(scene5, tracer5, r5)
assert_all_equal(reference5, baked5, "coalesced final bake vs lazy reference")
print(f"coalescing OK: 5 rapid scene changes (overlap observed: {saw_overlap}), no crash, "
     f"final installed result matches the final world ({len(reference5)} entries)")

print("ALL ASYNC-BAKE CHECKS PASSED")
