"""Whole-scene lighting bake: a pure snapshot -> result step, plus the
manager that runs it (on a background thread) and installs the result.

`bake()` is a pure function of a single `LightingSnapshot` -- no scene,
entity, tracer, pygame, or GPU access once the snapshot is built -- so it
is safe to run off the main thread. `snapshot_scene` does all the
scene-reading up front, on the calling (main) thread, into plain numpy
arrays and read-only records; entity/light OBJECT REFERENCES do end up in
the snapshot, but only ever as opaque, never-dereferenced dict keys (mirrors
how `ShadowTracer._cache`/`GITracer` already key off entity/light identity)
-- the worker thread touches no attribute of any live scene object.

`LightingBakeManager.update()` is engine/core.py's one call per frame,
right after `ShadowTracer.refresh()`. When `synchronous` (max_frames set,
or the caller passes `synchronous=True` -- benchmarks and the deterministic
test battery) it calls `bake()` inline and installs immediately, exactly
where the lazy per-(entity,light) recompute used to happen -- this is the
whole of Milestone 1, and it is byte-identical to that lazy path because
`install_bake` (see raytrace.py) seeds the exact same cache dicts those
lazy calls already check: the very next lazy call for each pair -- CPU
renderer, gl_renderer, or wgpu_renderer, none of which change at all --
finds a warm, matching entry and returns instantly.

Otherwise (interactive play) it dispatches a daemon `threading.Thread`
running the SAME `bake()` and returns immediately; `ShadowTracer.
_bake_pending` is set for the duration so the lazy calls the render loop
keeps making meanwhile serve stale data instead of tracing (see their
guards in raytrace.py) -- the frame never blocks. The finished result is
handed back through a lock-guarded mailbox and installed ONLY by the next
`update()` call on the main thread (an atomic reference swap, never
in-place mutation of a result already in use).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .raytrace import _directional_shadow_math, _gi_math, _point_shadow_math
from .renderer import _face_light_strength, _gather_lights, _is_translucent, _scene_sun, _world_face_geometry


@dataclass
class LightingSnapshot:
    """Immutable inputs for one whole-scene bake, built by `snapshot_scene`
    on the main thread. `bake()` reads only this -- no scene/tracer access."""
    geom_version: int
    occ: tuple                  # (v0, e1, e2, occ_centroids) world occluder soup
    occ_bvh: object             # BVH over occ (engine.bvh.BVH), or None at/below
                                 # BVH_THRESHOLD -- built by ShadowTracer.refresh(),
                                 # read-only here (immutable numpy arrays, safe to
                                 # hand to the worker thread untouched)
    occ_face_ids: np.ndarray    # per-occluder-triangle -> coarse caster-face id
    receivers: list             # [(entity, centroids, normals), ...] shadow_mesh() granularity
    caster_ids: set             # {id(entity), ...} subset of receivers that cast shadows/bounce GI
    caster_albedo: dict         # id(entity) -> shadow_mesh().face_colors, casters only
    sun: dict | None            # {"dir","color","softness","samples","shadow_depth"} or None
    lights: list                # [_LightInfo, ...] (see renderer.py) enabled point/spot lights
    gi_cfg: dict | None         # {"samples","intensity"} or None if GI is off
    mkeys: dict                 # entity -> transform bytes at snapshot time
    light_lkeys: dict           # light -> rounded (pos,radius,samples,range,inner,outer,ies) tuple
    sun_lkey: tuple | None      # rounded (dir, softness, samples) tuple, or None


@dataclass
class BakeResult:
    """Pure output of `bake()`: per-face shadow/GI factor arrays for every
    receiver in the snapshot, ready to install into ShadowTracer's and
    GITracer's caches (see their `install_bake` methods)."""
    geom_version: int
    sun_factors: dict = field(default_factory=dict)      # entity -> (M,) raw factor
    light_factors: dict = field(default_factory=dict)    # (entity, light) -> (M,) factor
    gi: dict = field(default_factory=dict)                # id(entity) -> (M, 3) float32
    gi_concat: np.ndarray = None                          # (R, 3) float32, GITracer._result format
    gi_entity_ranges: list = field(default_factory=list)  # [(entity, start, count), ...]
    gi_caster_list: list = field(default_factory=list)
    gi_key: tuple | None = None
    mkeys: dict = field(default_factory=dict)
    light_lkeys: dict = field(default_factory=dict)
    sun_lkey: tuple | None = None
    bake_seconds: float = 0.0


def snapshot_scene(scene, tracer) -> LightingSnapshot:
    """Read the scene + tracer ONCE, on the calling thread, into a
    `LightingSnapshot`. Cheap (geometry gathers, no ray tracing) -- safe to
    call every frame the world might have changed; `bake()` is the
    expensive part."""
    receiver_entities = [e for e in scene.entities if e.mesh is not None and e.visible]
    caster_entities = [e for e in receiver_entities
                       if e.casts_shadow and not _is_translucent(e)]
    caster_ids = {id(e) for e in caster_entities}

    receivers = []
    mkeys = {}
    caster_albedo = {}
    for e in receiver_entities:
        sm = e.shadow_mesh()
        centroids, normals = _world_face_geometry(e, sm)
        receivers.append((e, centroids, normals))
        mkeys[e] = e.transform.matrix().tobytes()
        if id(e) in caster_ids:
            caster_albedo[id(e)] = sm.face_colors

    dl = scene.light
    dl_dir = dl.direction.to_array()
    sun_disc = _scene_sun(scene)
    sun = None
    sun_lkey = None
    if sun_disc is not None and sun_disc.shadow_depth > 1e-6:
        sun = {"dir": dl_dir,
              "color": (np.asarray(dl.color, dtype=np.float64) / 255.0) * dl.intensity,
              "softness": sun_disc.shadow_softness, "samples": sun_disc.shadow_samples,
              "shadow_depth": sun_disc.shadow_depth}
        sun_lkey = (round(float(dl_dir[0]), 4), round(float(dl_dir[1]), 4),
                   round(float(dl_dir[2]), 4), round(float(sun_disc.shadow_softness), 3),
                   int(sun_disc.shadow_samples))

    lights = _gather_lights(scene)
    light_lkeys = {}
    for info in lights:
        light = info.light
        light_lkeys[light] = (round(float(info.pos[0]), 4), round(float(info.pos[1]), 4),
                              round(float(info.pos[2]), 4), light.radius, light.shadow_samples,
                              light.range, getattr(light, "inner", 0.0),
                              getattr(light, "outer", 0.0), light.ies)

    gi_cfg_raw = getattr(scene, "gi", None)
    gi_cfg = None
    if gi_cfg_raw and gi_cfg_raw.get("enabled"):
        gi_cfg = {"samples": gi_cfg_raw.get("samples", 16),
                  "intensity": gi_cfg_raw.get("intensity", 1.0)}

    return LightingSnapshot(
        geom_version=tracer._world_version, occ=tracer._occ, occ_bvh=tracer._occ_bvh,
        occ_face_ids=tracer._occ_face_ids,
        receivers=receivers, caster_ids=caster_ids, caster_albedo=caster_albedo,
        sun=sun, lights=lights, gi_cfg=gi_cfg,
        mkeys=mkeys, light_lkeys=light_lkeys, sun_lkey=sun_lkey)


def bake(snapshot: LightingSnapshot) -> BakeResult:
    """Pure: compute every receiver's shadow/GI factors from `snapshot`
    alone. Touches no scene, entity, tracer, pygame, or GPU state -- safe
    to run on a worker thread (Milestone 2). Mirrors exactly what today's
    lazy `ShadowTracer.shadow_factors`/`directional_shadow_factors` calls
    and `GITracer.compute`'s direct-lighting + hemisphere pass compute,
    just eagerly, for every receiver/light pair in the snapshot at once."""
    occ = snapshot.occ
    sun_factors: dict = {}
    light_factors: dict = {}
    to_light = None
    if snapshot.sun is not None:
        d = snapshot.sun["dir"]
        to_light = -d / max(np.linalg.norm(d), 1e-12)

    for entity, centroids, normals in snapshot.receivers:
        if snapshot.sun is not None:
            lambert = np.clip(normals @ to_light, 0.0, 1.0)
            active = lambert > 1e-3
            sun_factors[entity] = _directional_shadow_math(
                occ, snapshot.sun["dir"], snapshot.sun["softness"], snapshot.sun["samples"],
                centroids, normals, active, bvh=snapshot.occ_bvh)
        for info in snapshot.lights:
            strength = _face_light_strength(info, normals, centroids)
            active = strength > 1e-3
            if active.any() and info.light.cast_shadows:
                light_factors[(entity, info.light)] = _point_shadow_math(
                    occ, info.pos, info.light.radius, info.light.range,
                    info.light.shadow_samples, centroids, normals, active,
                    bvh=snapshot.occ_bvh)

    gi = {}
    gi_concat = np.zeros((0, 3), dtype=np.float32)
    gi_entity_ranges = []
    gi_caster_list = []
    gi_key = None
    if snapshot.gi_cfg is not None:
        gi_key = (max(snapshot.gi_cfg["samples"], 1), round(float(snapshot.gi_cfg["intensity"]), 4))
        gi_caster_list = [e for e, _c, _n in snapshot.receivers if id(e) in snapshot.caster_ids]

        offset = 0
        for entity, centroids, _normals in snapshot.receivers:
            m = len(centroids)
            gi_entity_ranges.append((entity, offset, m))
            offset += m

        if offset > 0 and gi_caster_list:
            # casters' own direct lighting (lambert*sun-shadow + point/spot*
            # shadow), reusing the factors just computed above for the SAME
            # (entity, light)/(entity, "sun") pairs -- mirrors renderer.py's
            # `_gi_direct_lighting`.
            c_list, a_list, d_list = [], [], []
            for entity, centroids, normals in snapshot.receivers:
                if id(entity) not in snapshot.caster_ids:
                    continue
                if snapshot.sun is not None:
                    lambert = np.clip(normals @ to_light, 0.0, 1.0)
                    raw = sun_factors.get(entity, np.ones(len(centroids)))
                    dshadow = 1.0 - snapshot.sun["shadow_depth"] * (1.0 - raw)
                    direct = snapshot.sun["color"][None, :] * (lambert * dshadow)[:, None]
                else:
                    direct = np.zeros((len(centroids), 3))
                for info in snapshot.lights:
                    strength = _face_light_strength(info, normals, centroids)
                    factors = light_factors.get((entity, info.light))
                    if factors is not None:
                        strength = strength * factors
                    direct = direct + info.colorf[None, :] * strength[:, None]
                c_list.append(centroids)
                a_list.append(snapshot.caster_albedo[id(entity)])
                d_list.append(direct)

            receivers_c = np.concatenate([c for _e, c, _n in snapshot.receivers])
            receivers_n = np.concatenate([n for _e, _c, n in snapshot.receivers])
            albedo_c = np.concatenate(a_list)
            direct_c = np.concatenate(d_list)

            gi_concat = _gi_math(occ, snapshot.occ_face_ids, albedo_c, direct_c,
                                 receivers_c, receivers_n,
                                 snapshot.gi_cfg["samples"], snapshot.gi_cfg["intensity"],
                                 bvh=snapshot.occ_bvh)
            gi = {id(e): gi_concat[start:start + m] for e, start, m in gi_entity_ranges}

    return BakeResult(
        geom_version=snapshot.geom_version,
        sun_factors=sun_factors, light_factors=light_factors,
        gi=gi, gi_concat=gi_concat, gi_entity_ranges=gi_entity_ranges,
        gi_caster_list=gi_caster_list, gi_key=gi_key,
        mkeys=snapshot.mkeys, light_lkeys=snapshot.light_lkeys, sun_lkey=snapshot.sun_lkey)


class LightingBakeManager:
    """Decides when the whole scene needs re-baking and runs it.

    Milestone 1: `run_bake` always blocks (calls `bake()` inline) -- this
    class exists so engine/core.py's run loop already has its final shape:
    `update()` is the only entry point core.py calls, every frame, right
    after `tracer.refresh(scene)`. Change detection covers every trigger
    that the lazy caches it primes independently recompute on: occluder
    geometry (`tracer._world_version`), the GI receiver SET (any visible
    mesh entity, not just casters -- `GITracer.compute`'s own `same_
    receivers` check), and the GI samples/intensity key.

    Milestone 2 (background worker): when `synchronous` is False, `update()`
    dispatches at most ONE bake at a time onto a daemon `threading.Thread`
    running the pure `bake()` -- safe off-thread since it touches only its
    own `LightingSnapshot`. `tracer._bake_pending` is set for the duration
    so the lazy `shadow_factors`/`directional_shadow_factors`/`GITracer.
    compute` calls the render loop keeps making meanwhile serve stale data
    instead of tracing (see their guards in raytrace.py) -- the frame never
    blocks. The finished `BakeResult` is handed back through `_lock`-guarded
    `_finished` and installed ONLY from the main thread, in `update()`,
    never from the worker -- an atomic reference swap at a safe point (top
    of frame), per the task's threading requirement. Coalescing: if the
    world changes again while a bake is in flight, `needs_bake()` keeps
    returning True but `update()` won't start a second thread until the
    first one's result has been installed -- the very next `update()` call
    after that dispatches fresh against whatever is newest, so any
    intermediate versions in between are simply never baked (superseded).
    """

    def __init__(self):
        self._last_world_version = -2   # forces a bake on the very first call
        self._last_gi_key = None
        self._last_receivers_key = None
        self._thread: "_BakeWorker | None" = None
        self._lock = threading.Lock()
        self._finished: BakeResult | None = None   # worker -> main-thread mailbox

    def _gi_key(self, scene) -> tuple | None:
        gi_cfg = getattr(scene, "gi", None)
        if not gi_cfg or not gi_cfg.get("enabled"):
            return None
        return (gi_cfg.get("samples", 16), round(float(gi_cfg.get("intensity", 1.0)), 4))

    def _receivers_key(self, scene) -> tuple:
        return tuple(id(e) for e in scene.entities if e.mesh is not None and e.visible)

    def needs_bake(self, scene, tracer) -> bool:
        """Cheap (no tracing): call AFTER `tracer.refresh(scene)`. True iff
        anything a whole-scene bake would need to reflect has changed since
        the last dispatch."""
        return (tracer._world_version != self._last_world_version
               or self._gi_key(scene) != self._last_gi_key
               or self._receivers_key(scene) != self._last_receivers_key)

    def busy(self) -> bool:
        """True while a background bake is in flight (worker thread alive,
        result not yet installed)."""
        return self._thread is not None

    def _install(self, tracer, gi_tracer, result: BakeResult) -> None:
        tracer.install_bake(result)
        tracer._bake_pending = False
        if gi_tracer is not None:
            gi_tracer.install_bake(result)

    def update(self, scene, tracer, gi_tracer, synchronous: bool = False) -> dict:
        """Call once per frame, right after `tracer.refresh(scene)`.

        Returns {"dispatched": bool, "installed": BakeResult | None} for
        engine/core.py's console logging: log "Baking lighting (N occluder
        triangles)..." when `dispatched`, and "Lighting baked in X.Xs"
        (using `installed.bake_seconds`) whenever `installed` is not None
        -- both may fire in the SAME call (synchronous path: dispatch and
        install happen back to back, exactly like before this task) or in
        DIFFERENT calls, frames apart (async path: `dispatched` fires at
        the start, `installed` only once the worker finishes).
        """
        installed = self._poll_finished(tracer, gi_tracer)

        dispatched = False
        if self._thread is None and self.needs_bake(scene, tracer):
            dispatched = True
            snapshot = snapshot_scene(scene, tracer)
            self._last_world_version = tracer._world_version
            self._last_gi_key = self._gi_key(scene)
            self._last_receivers_key = self._receivers_key(scene)
            if synchronous:
                t0 = time.perf_counter()
                result = bake(snapshot)
                result.bake_seconds = time.perf_counter() - t0
                self._install(tracer, gi_tracer, result)
                installed = result
            else:
                tracer._bake_pending = True
                self._thread = _BakeWorker(snapshot, self)
                self._thread.start()

        return {"dispatched": dispatched, "installed": installed}

    def _poll_finished(self, tracer, gi_tracer) -> BakeResult | None:
        """Main-thread-only: if the worker thread has a result waiting,
        install it (atomic reference swaps into `tracer`/`gi_tracer`,
        never in-place mutation of anything still in use) and clear the
        slot so the next `update()` may dispatch a fresh bake."""
        if self._thread is None or self._thread.is_alive():
            return None
        self._thread.join()  # already finished; reaps the thread object
        self._thread = None
        with self._lock:
            result, self._finished = self._finished, None
        if result is None:
            # the worker raised (see _BakeWorker.run) -- give up on this
            # bake rather than leaving the render loop suppressed forever
            tracer._bake_pending = False
            return None
        self._install(tracer, gi_tracer, result)
        return result


class _BakeWorker(threading.Thread):
    """Runs the pure `bake()` off the main thread. Reads only its own
    `snapshot` (never the live scene/tracer/entities) and, on success,
    hands the result back through the manager's lock-guarded mailbox --
    it never touches `ShadowTracer`/`GITracer` state itself; only the main
    thread's `_poll_finished` installs."""

    def __init__(self, snapshot: LightingSnapshot, manager: "LightingBakeManager"):
        super().__init__(daemon=True)
        self._snapshot = snapshot
        self._manager = manager

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            result = bake(self._snapshot)
            result.bake_seconds = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - never let a worker crash the process
            console_log_error(exc)
            result = None
        with self._manager._lock:
            self._manager._finished = result


def console_log_error(exc: Exception) -> None:
    """Best-effort error surfacing for a failed background bake -- import
    is local to avoid a hard dependency from this pure-compute module on
    the console/pygame-adjacent logging subsystem at import time."""
    try:
        from . import console_log
        console_log.log_error(f"background lighting bake failed: {exc}")
    except Exception:
        pass
