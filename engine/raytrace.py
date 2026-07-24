"""Real-time ray tracing: soft shadows, global illumination, and mouse picking.

Point/spot shadow rays are cast from face centroids toward jittered sample
points on each light's spherical area; directional (sun) shadow rays instead
jitter across a disk perpendicular to the light's direction, sized by an
angular softness. Both use Moller-Trumbore intersection, vectorized over
rays x occluder triangles, and the fraction of unblocked rays gives a soft
shadow factor per face. Results are cached per (entity, light) and only
recomputed when the receiver, the light, or any shadow caster actually moved,
so static scenes pay the tracing cost once; moving lights (the flashlight)
retrace on a configurable frame interval.

`GITracer` (bottom of the file) reuses the same occluder soup for one-bounce
global illumination: cosine-weighted hemisphere rays from every occluder
face, cached with identical world-version invalidation.
"""
from __future__ import annotations

import math

import numpy as np

from .bvh import any_hit as _bvh_any_hit
from .bvh import build_bvh
from .bvh import nearest as _bvh_nearest

_RAY_CHUNK = 128       # rays per Moller-Trumbore batch (bounds temp memory)
_ORIGIN_OFFSET = 0.02  # push ray origins off the surface to avoid self-hits
_RECEIVER_INTERVAL = 3  # frames a moving receiver may reuse its cached shadows
BVH_THRESHOLD = 256    # occluder triangle count above which _intersect_any/
                       # _nearest_hit_faces route through engine/bvh.py
                       # instead of the dense (rays x triangles) arrays below;
                       # <= threshold keeps every existing low-poly scene on
                       # the untouched brute-force path (byte-identical).


def _is_translucent(entity) -> bool:
    """True when `entity`'s material is a translucent-blend-mode graph.
    Translucent meshes don't occlude shadow rays and don't source GI bounce
    light (mirrors the `casts_shadow=False` treatment below) -- they still
    receive lighting/shadows/GI normally, only the occluder/bounce-source
    role is gated. v1 semantics: a see-through mesh casting a hard shadow or
    bouncing full-opacity light would look wrong."""
    mat = getattr(entity, "material", None)
    return mat is not None and getattr(mat, "blend_mode", "opaque") == "translucent"


def sphere_samples(n: int) -> np.ndarray:
    """Deterministic points on the unit sphere (golden spiral)."""
    if n <= 1:
        return np.zeros((1, 3))
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=-1)


def disk_samples(n: int) -> np.ndarray:
    """Deterministic points in the unit disk (Vogel/Fibonacci spiral): (n, 2)."""
    if n <= 1:
        return np.zeros((1, 2))
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)


def hemisphere_samples(n: int) -> np.ndarray:
    """Deterministic cosine-weighted points on the local +Z hemisphere: (n, 3)."""
    if n <= 1:
        return np.array([[0.0, 0.0, 1.0]])
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    x, y = r * np.cos(theta), r * np.sin(theta)
    z = np.sqrt(np.maximum(1.0 - x * x - y * y, 0.0))
    return np.stack([x, y, z], axis=-1)


def _intersect_any(origins, dirs, max_t, v0, e1, e2, bvh=None) -> np.ndarray:
    """For each ray, is anything hit before max_t? (R,) bool.

    When `bvh` is given (occluder count above BVH_THRESHOLD -- see
    `build_bvh_for_occ`), traversal routes through engine/bvh.py instead of
    the dense (rays x triangles) arrays below; `v0`/`e1`/`e2` go unused in
    that branch (kept for call-site symmetry). `bvh is None` leaves this
    function's body completely untouched, so the <=threshold brute-force
    path stays byte-identical to before this module had a BVH at all."""
    if bvh is not None:
        return _bvh_any_hit(bvh, origins, dirs, max_t)
    blocked = np.zeros(len(origins), dtype=bool)
    if len(v0) == 0:
        return blocked
    for i in range(0, len(origins), _RAY_CHUNK):
        o = origins[i:i + _RAY_CHUNK]
        d = dirs[i:i + _RAY_CHUNK]
        mt = max_t[i:i + _RAY_CHUNK]
        p = np.cross(d[:, None, :], e2[None, :, :])          # (r,T,3)
        det = np.einsum("rtk,tk->rt", p, e1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            tv = o[:, None, :] - v0[None, :, :]
            u = np.einsum("rtk,rtk->rt", tv, p) * inv
            q = np.cross(tv, e1[None, :, :])
            v = np.einsum("rk,rtk->rt", d, q) * inv
            t = np.einsum("tk,rtk->rt", e2, q) * inv
            hit = ((np.abs(det) > 1e-12) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
                   & (t > 1e-4) & (t < mt[:, None]))
        blocked[i:i + _RAY_CHUNK] = hit.any(axis=1)
    return blocked


def _intersect_nearest(origin, direction, v0, e1, e2) -> float | None:
    """Nearest hit distance of a single ray, or None."""
    if len(v0) == 0:
        return None
    p = np.cross(direction, e2)
    det = np.einsum("tk,tk->t", p, e1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        tv = origin - v0
        u = np.einsum("tk,tk->t", tv, p) * inv
        q = np.cross(tv, e1)
        v = q @ direction * inv
        t = np.einsum("tk,tk->t", e2, q) * inv
        hit = ((np.abs(det) > 1e-12) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
               & (t > 1e-4))
    if not hit.any():
        return None
    return float(t[hit].min())


def _point_shadow_math(occ, light_pos: np.ndarray, radius: float, rng: float,
                       samples_n: int, centroids: np.ndarray, normals: np.ndarray,
                       active: np.ndarray, bvh=None) -> np.ndarray:
    """Pure core of point/spot soft-shadow tracing -- no cache, no `self`.
    `occ` is the (v0, e1, e2, centroids) occluder soup. Extracted verbatim
    from `ShadowTracer.shadow_factors` so that method's cache wrapper and
    `lighting_bake.bake`'s whole-scene, off-render-loop pass share one
    implementation (see lighting_bake.py's module docstring).

    `bvh` (built once per occluder-soup version above BVH_THRESHOLD -- see
    `build_bvh_for_occ`) is `None` for every scene at/below the threshold,
    in which case this function is EXACTLY what it always was (the `else`
    branch below, byte-identical). Above threshold, the distance-based
    `near` occluder prefilter is skipped in favor of querying the BVH
    against the full soup directly: the BVH's own spatial traversal already
    prunes far-away triangles faster than that linear cull approximates, so
    re-deriving the same prefilter first would be redundant work."""
    factors = np.ones(len(centroids))
    idx = np.nonzero(active)[0]
    v0, e1, e2, occ_c = occ
    if len(idx) == 0 or len(v0) == 0:
        return factors
    origins = centroids[idx] + normals[idx] * _ORIGIN_OFFSET
    samples = light_pos + sphere_samples(samples_n) * radius
    hits = np.zeros(len(idx))
    if bvh is not None:
        for s in samples:
            d = s - origins
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            blocked = _intersect_any(origins, d / dist[:, None], dist - 0.05,
                                     v0, e1, e2, bvh=bvh)
            hits += blocked
    else:
        reach = rng + radius + 1.0
        near = np.linalg.norm(occ_c - light_pos, axis=1) < reach
        ov0, oe1, oe2 = v0[near], e1[near], e2[near]
        if len(ov0) == 0:
            return factors
        for s in samples:
            d = s - origins
            dist = np.linalg.norm(d, axis=1)
            dist = np.maximum(dist, 1e-9)
            blocked = _intersect_any(origins, d / dist[:, None], dist - 0.05, ov0, oe1, oe2)
            hits += blocked
    factors[idx] = 1.0 - hits / len(samples)
    return factors


def _directional_shadow_math(occ, direction: np.ndarray, softness_deg: float, samples_n: int,
                             centroids: np.ndarray, normals: np.ndarray, active: np.ndarray,
                             max_t: float = 500.0, bvh=None) -> np.ndarray:
    """Pure core of directional (sun) soft-shadow tracing -- no cache, no
    `self`. Extracted verbatim from `ShadowTracer.directional_shadow_factors`
    (see `_point_shadow_math`'s docstring for why, and for the `bvh`
    contract -- this function already used the full soup with no distance
    prefilter, so passing `bvh` through changes no behavior beyond routing
    the intersection test itself)."""
    factors = np.ones(len(centroids))
    idx = np.nonzero(active)[0]
    v0, e1, e2, _occ_c = occ
    if len(idx) == 0 or len(v0) == 0:
        return factors
    to_light = -direction / max(np.linalg.norm(direction), 1e-12)
    up = np.array([0.0, 1.0, 0.0]) if abs(to_light[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    tx = np.cross(up, to_light)
    tx /= max(np.linalg.norm(tx), 1e-12)
    ty = np.cross(to_light, tx)
    radius = math.tan(math.radians(max(softness_deg, 0.0)))
    origins = centroids[idx] + normals[idx] * _ORIGIN_OFFSET
    disk = disk_samples(max(samples_n, 1))
    hits = np.zeros(len(idx))
    max_t_arr = np.full(len(idx), max_t)
    for ox, oy in disk:
        d = to_light + tx * (ox * radius) + ty * (oy * radius)
        d = d / max(np.linalg.norm(d), 1e-12)
        d_b = np.broadcast_to(d, (len(idx), 3))
        blocked = _intersect_any(origins, d_b, max_t_arr, v0, e1, e2, bvh=bvh)
        hits += blocked
    factors[idx] = 1.0 - hits / max(samples_n, 1)
    return factors


def _gi_math(occ, tri_face_id: np.ndarray, albedo_c: np.ndarray, direct_c: np.ndarray,
            centroids_r: np.ndarray, normals_r: np.ndarray, samples_n: int,
            intensity: float, bvh=None) -> np.ndarray:
    """Pure core of one-bounce GI hemisphere sampling -- no cache, no
    `self`. Extracted verbatim from `GITracer.compute` (see
    `_point_shadow_math`'s docstring for why); `tri_face_id`/`albedo_c`/
    `direct_c` describe the bounce-source (caster) soup, `centroids_r`/
    `normals_r` the receivers being lit. Returns (R, 3) float32."""
    offset = len(centroids_r)
    v0, e1, e2, _occ_c = occ
    local = hemisphere_samples(max(samples_n, 1))
    up = np.where(np.abs(normals_r[:, 1:2]) < 0.99,
                 np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    tangent = np.cross(up, normals_r)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bitangent = np.cross(normals_r, tangent)
    origins = centroids_r + normals_r * _ORIGIN_OFFSET
    accum = np.zeros((offset, 3), dtype=np.float64)
    for lx, ly, lz in local:
        dirs = tangent * lx + bitangent * ly + normals_r * lz
        hit_face = _nearest_hit_faces(origins, dirs, v0, e1, e2, tri_face_id, bvh=bvh)
        valid = hit_face >= 0
        if valid.any():
            hf = hit_face[valid]
            accum[valid] += (albedo_c[hf] / 255.0) * direct_c[hf]
    return (intensity * accum / max(samples_n, 1)).astype(np.float32)


def _nearest_hit_faces(origins, dirs, v0, e1, e2, tri_face_id, max_t=200.0, bvh=None):
    """Nearest-hit face id per ray (mapped through `tri_face_id`), or -1.

    `bvh`/byte-identical-when-None contract mirrors `_intersect_any` above."""
    n = len(origins)
    if bvh is not None:
        tri_idx, _t = _bvh_nearest(bvh, origins, dirs, max_t)
        return np.where(tri_idx >= 0, tri_face_id[np.maximum(tri_idx, 0)], -1)
    out = np.full(n, -1, dtype=np.int64)
    if len(v0) == 0:
        return out
    for i in range(0, n, _RAY_CHUNK):
        o = origins[i:i + _RAY_CHUNK]
        d = dirs[i:i + _RAY_CHUNK]
        p = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum("rtk,tk->rt", p, e1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            tv = o[:, None, :] - v0[None, :, :]
            u = np.einsum("rtk,rtk->rt", tv, p) * inv
            q = np.cross(tv, e1[None, :, :])
            v = np.einsum("rk,rtk->rt", d, q) * inv
            t = np.einsum("tk,rtk->rt", e2, q) * inv
            hit = ((np.abs(det) > 1e-12) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
                   & (t > 1e-4) & (t < max_t))
        t_masked = np.where(hit, t, np.inf)
        j = np.argmin(t_masked, axis=1)
        valid = np.isfinite(t_masked[np.arange(len(o)), j])
        out[i:i + _RAY_CHUNK] = np.where(valid, tri_face_id[j], -1)
    return out


def build_bvh_for_occ(occ):
    """Build (or skip) a BVH for the `occ` = (v0, e1, e2, centroids) soup,
    gated on BVH_THRESHOLD: low-poly soups return None so every caller's
    `bvh=None` byte-identical brute-force path stays in effect and no build
    cost is ever paid for scenes that don't need one."""
    v0, e1, e2, _centroids = occ
    if len(v0) <= BVH_THRESHOLD:
        return None
    return build_bvh(v0, e1, e2)


def _world_triangles(entity, mesh=None):
    """World-space (v0, e1, e2) triangle-edge arrays for `mesh` (default
    `entity.mesh`/LOD0 -- exact geometry, used by mouse-picking below).
    `ShadowTracer.refresh` passes `entity.shadow_mesh()` explicitly to build
    the occluder soup from a coarse LOD proxy on high-poly casters instead
    (see scene.py's `Entity.shadow_mesh`)."""
    mesh = mesh if mesh is not None else entity.mesh
    m = entity.transform.matrix()
    wv = mesh.vertices @ m[:3, :3].T + m[:3, 3]
    tri = wv[mesh.tri_faces]
    return tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]


def pick_entity(scene, origin: np.ndarray, direction: np.ndarray):
    """Nearest entity hit by a world-space ray. Returns (entity, t) or (None, None)."""
    best, best_t = None, None
    for e in scene.entities:
        if e.mesh is None or not e.visible:
            continue
        t = _intersect_nearest(origin, direction, *_world_triangles(e))
        if t is not None and (best_t is None or t < best_t):
            best, best_t = e, t
    return best, best_t


class ShadowTracer:
    def __init__(self):
        self.frame = 0
        self._occ = None            # (v0, e1, e2, centroids) world-space soup
        self._occ_bvh = None        # BVH over _occ, or None at/below BVH_THRESHOLD
        self._occ_face_ids = None   # per-triangle -> global caster-face id (for GI)
        self._caster_mats = {}      # id(entity) -> matrix bytes
        self._world_version = -1
        self._cache = {}            # (entity, light) -> cache dict
        self._bake_pending = False  # True while LightingBakeManager has an async
                                    # bake for the CURRENT world in flight -- see
                                    # shadow_factors'/directional_shadow_factors'
                                    # guard against tracing (and stalling) again

    def refresh(self, scene) -> bool:
        """Rebuild the world-space occluder soup if any shadow caster moved.

        Returns True iff a rebuild actually happened (world_version bumped).
        The rebuild itself is cheap; callers (engine/core.py) use this
        return value to bracket a console-log message around the render()
        call that follows, since THAT'S where the expensive per-pixel
        shadow_factors/GI recompute actually happens (only on a world-
        version change -- see shadow_factors' cache check below)."""
        self.frame += 1
        casters = [e for e in scene.entities
                   if e.mesh is not None and e.visible and e.casts_shadow
                   and not _is_translucent(e)]
        mats = {id(e): e.transform.matrix().tobytes() for e in casters}
        if mats == self._caster_mats and self._occ is not None:
            return False
        v0s, e1s, e2s, face_ids = [], [], [], []
        face_offset = 0
        for e in casters:
            # occluder geometry uses the entity's shadow mesh (a coarse LOD
            # proxy for high-poly meshes, else `entity.mesh` unchanged --
            # see `Entity.shadow_mesh`) so occlusion cost scales with the
            # coarse triangle count, not the full-detail import; face ids
            # below index into this SAME mesh's faces, matching
            # `_gi_direct_lighting`'s per-caster arrays one-for-one.
            occ_mesh = e.shadow_mesh()
            v0, e1, e2 = _world_triangles(e, occ_mesh)
            v0s.append(v0)
            e1s.append(e1)
            e2s.append(e2)
            faces = occ_mesh.faces
            is_tri = (faces[:, 3] == faces[:, 2]) | (faces[:, 3] == faces[:, 0])
            counts = np.where(is_tri, 1, 2)  # tris contribute 1 tri, quads 2
            face_ids.append(np.repeat(np.arange(len(faces)), counts) + face_offset)
            face_offset += len(faces)
        if v0s:
            v0 = np.concatenate(v0s)
            e1 = np.concatenate(e1s)
            e2 = np.concatenate(e2s)
            occ_face_ids = np.concatenate(face_ids)
        else:
            v0 = e1 = e2 = np.zeros((0, 3))
            occ_face_ids = np.zeros((0,), dtype=np.int64)
        centroids = v0 + (e1 + e2) / 3.0
        self._occ = (v0, e1, e2, centroids)
        self._occ_bvh = build_bvh_for_occ(self._occ)  # built once per soup version
        self._occ_face_ids = occ_face_ids
        self._caster_mats = mats
        self._world_version += 1
        # drop cache entries for entities no longer in the scene
        live = set(scene.entities)
        self._cache = {k: v for k, v in self._cache.items() if k[0] in live}
        return True

    def occluder_triangle_count(self) -> int:
        """Triangle count of the current occluder soup, 0 before the first
        refresh() -- sizes the "Baking lighting (...)" console message."""
        return 0 if self._occ is None else int(self._occ[0].shape[0])

    def shadow_factors(self, entity, light, light_pos: np.ndarray,
                       centroids: np.ndarray, normals: np.ndarray,
                       active: np.ndarray) -> np.ndarray:
        """Per-face soft shadow factor in [0, 1] for one light.

        Only faces marked `active` (facing and in range of the light) are
        traced; everything else keeps factor 1 (their contribution is zero
        anyway). Cached until the receiver, light, or world geometry changes.
        """
        key = (entity, light)
        mkey = entity.transform.matrix().tobytes()
        lkey = (round(float(light_pos[0]), 4), round(float(light_pos[1]), 4),
                round(float(light_pos[2]), 4), light.radius, light.shadow_samples,
                light.range, getattr(light, "inner", 0.0), getattr(light, "outer", 0.0),
                light.ies)
        cached = self._cache.get(key)
        if cached is not None:
            age = self.frame - cached["frame"]
            if cached["world"] == self._world_version and cached["m"] == mkey \
                    and cached["l"] == lkey:
                return cached["factors"]
            # only this receiver moved: its own shadows may lag a few frames
            if cached["world"] == self._world_version and cached["l"] == lkey \
                    and age < _RECEIVER_INTERVAL:
                return cached["factors"]
            # a moving light retraces on its declared interval (flashlight)
            if light.shadow_interval > 1 and age < light.shadow_interval:
                return cached["factors"]

        if self._bake_pending:
            # a whole-scene bake for the CURRENT world is already computing
            # off-thread (LightingBakeManager, engine/lighting_bake.py) --
            # don't trace here too (that would be exactly the stall async
            # mode exists to avoid). Serve the best data on hand and let
            # the manager's eventual install_bake() refresh this entry;
            # this can only under-shadow a BRAND NEW (entity, light) pair
            # that has never been cached at all (falls back to fully lit
            # until the next bake covers it).
            return cached["factors"] if cached is not None else np.ones(len(centroids))

        factors = _point_shadow_math(self._occ, light_pos, light.radius, light.range,
                                     light.shadow_samples, centroids, normals, active,
                                     bvh=self._occ_bvh)

        self._cache[key] = {"factors": factors, "world": self._world_version,
                            "m": mkey, "l": lkey, "frame": self.frame}
        return factors

    def directional_shadow_factors(self, entity, direction: np.ndarray, softness_deg: float,
                                   samples: int, centroids: np.ndarray, normals: np.ndarray,
                                   active: np.ndarray, max_t: float = 500.0) -> np.ndarray:
        """Per-face soft shadow factor in [0, 1] for the directional (sun) light.

        Rays leave each active face's centroid along the light's reverse
        direction, jittered across a disk perpendicular to it (radius =
        tan(softness_deg)) -- the sun is treated as infinitely far away with
        an angular size, unlike the point/spot spherical-area model above.
        Cached exactly like `shadow_factors`: keyed on the rounded direction/
        softness/samples, invalidated by world geometry or receiver movement.
        """
        key = (entity, "sun")
        mkey = entity.transform.matrix().tobytes()
        lkey = (round(float(direction[0]), 4), round(float(direction[1]), 4),
                round(float(direction[2]), 4), round(float(softness_deg), 3), int(samples))
        cached = self._cache.get(key)
        if cached is not None:
            age = self.frame - cached["frame"]
            if cached["world"] == self._world_version and cached["m"] == mkey \
                    and cached["l"] == lkey:
                return cached["factors"]
            if cached["world"] == self._world_version and cached["l"] == lkey \
                    and age < _RECEIVER_INTERVAL:
                return cached["factors"]

        if self._bake_pending:
            # see the identical guard in shadow_factors() above
            return cached["factors"] if cached is not None else np.ones(len(centroids))

        factors = _directional_shadow_math(self._occ, direction, softness_deg, samples,
                                           centroids, normals, active, max_t,
                                           bvh=self._occ_bvh)

        self._cache[key] = {"factors": factors, "world": self._world_version,
                            "m": mkey, "l": lkey, "frame": self.frame}
        return factors

    def install_bake(self, result: "BakeResult") -> None:
        """Seed `self._cache` with a whole-scene bake result (see
        engine/lighting_bake.py's `bake`) so the very next `shadow_factors`/
        `directional_shadow_factors` call for each (entity, light) pair in
        the result finds a warm, matching cache entry and returns instantly
        instead of tracing itself -- exactly as if that pair had just been
        lazily computed and cached, only eagerly, for the whole scene at
        once. Pairs NOT in the result (e.g. an entity added after this bake
        was dispatched) are left alone -- their next lazy call is a normal
        cache miss, bounded to just that one pair (see raytrace.py's
        module docstring on cache invalidation).

        Only ever called from the main thread (mutates `self._cache`); a
        background bake worker (Milestone 2) computes `result` purely and
        hands it back for THIS call to install at a safe point."""
        frame = self.frame
        world = self._world_version
        for entity, factors in result.sun_factors.items():
            self._cache[(entity, "sun")] = {"factors": factors, "world": world,
                                            "m": result.mkeys.get(entity),
                                            "l": result.sun_lkey, "frame": frame}
        for (entity, light), factors in result.light_factors.items():
            self._cache[(entity, light)] = {"factors": factors, "world": world,
                                            "m": result.mkeys.get(entity),
                                            "l": result.light_lkeys.get(light), "frame": frame}


class GITracer:
    """One-bounce global illumination, cached like ShadowTracer's shadow
    factors: keyed to the same world-version invalidation (a static scene
    bakes once, then costs nothing per frame until something moves).

    Bounce *sources* (what hemisphere rays can hit, and whose direct
    radiance gets reflected) are exactly the shadow-casting mesh entities --
    the same occluder soup `ShadowTracer` traces against -- so animated
    entities (`casts_shadow = False`) don't emit bounce light, for the same
    reason they don't cast shadows: their movement would otherwise thrash
    the bake every frame. *Receivers*, however, are every visible mesh
    entity regardless of `casts_shadow` -- mirroring `ShadowTracer.
    shadow_factors`, which lights every visible entity's faces the same way
    -- so a non-casting floor (a common asset default) still gets bounce
    light landing on it; it just can't bounce light onward itself.
    """

    def __init__(self):
        self._world_version = -1
        self._key = None
        self._result = None          # (R, 3) indirect contribution, per receiver face
        self._entity_ranges = []     # [(entity, start, count), ...] receiver order
        self._caster_list = []       # casters backing the bounce-source soup, for cache validity

    def compute(self, scene, tracer, direct_fn, receiver_fn,
               samples: int, intensity: float) -> dict:
        """Return {id(entity): (M, 3) indirect-light array} for every visible
        mesh entity (the receiver set).

        `direct_fn(casters)` -> (centroids, normals, albedo, direct) arrays
        for the bounce-source soup (shadow casters only), concatenated in
        order. `receiver_fn(receivers)` -> (centroids, normals) for every
        visible mesh entity. Both are supplied by the caller (Renderer)
        since the direct-lighting/shadow machinery already lives there.

        Receiver ranges are sized to each entity's `shadow_mesh()` face
        count (LOD0, or a coarse LOD proxy for high-poly meshes -- see
        scene.py), matching what `receiver_fn` actually returns per entity;
        the renderer gathers this coarse result onto whichever LOD is
        rasterized via `_lod_gather`.
        """
        casters = [e for e in scene.entities
                  if e.mesh is not None and e.visible and e.casts_shadow
                  and not _is_translucent(e)]
        receivers = [e for e in scene.entities
                    if e.mesh is not None and e.visible]
        key = (max(samples, 1), round(float(intensity), 4))
        same_receivers = [e for e, _s, _c in self._entity_ranges] == receivers
        same_casters = self._caster_list == casters
        if (self._result is not None and same_receivers and same_casters
                and self._world_version == tracer._world_version and self._key == key):
            return self._to_dict()

        if tracer._bake_pending:
            # a whole-scene bake for the CURRENT world is already computing
            # off-thread (LightingBakeManager) -- don't recompute GI here
            # too (the receivers/casters-list change above would otherwise
            # force exactly the synchronous stall async mode exists to
            # avoid, e.g. when a caster is freshly added). Serve whatever
            # was last installed -- entities added/removed since simply
            # won't appear in `_to_dict()`'s output (renderer treats a
            # missing entry as "no GI yet", see `_gi_contrib`) until the
            # in-flight bake's `install_bake()` lands.
            return self._to_dict() if self._result is not None else {}

        ranges = []
        offset = 0
        for e in receivers:
            m = int(e.shadow_mesh().faces.shape[0])
            ranges.append((e, offset, m))
            offset += m

        if offset == 0 or not casters:
            self._result = np.zeros((0, 3), dtype=np.float32)
            self._entity_ranges = ranges
            self._caster_list = casters
            self._world_version = tracer._world_version
            self._key = key
            return {}

        albedo_c, direct_c = direct_fn(casters)[2:]
        centroids_r, normals_r = receiver_fn(receivers)

        self._result = _gi_math(tracer._occ, tracer._occ_face_ids, albedo_c, direct_c,
                                centroids_r, normals_r, samples, intensity,
                                bvh=tracer._occ_bvh)
        self._entity_ranges = ranges
        self._caster_list = casters
        self._world_version = tracer._world_version
        self._key = key
        return self._to_dict()

    def _to_dict(self) -> dict:
        return {id(e): self._result[start:start + m]
               for e, start, m in self._entity_ranges}

    def install_bake(self, result: "BakeResult") -> None:
        """Seed this GI cache with a whole-scene bake result (see
        engine/lighting_bake.py) -- mirrors exactly what a successful
        `compute()` assigns at its own tail, so the very next `compute()`
        call (from whichever renderer backend owns this GITracer instance)
        finds a warm, matching cache and returns instantly instead of
        recomputing. A no-op when the bake ran with GI disabled (leaves any
        prior GI cache untouched, matching `compute()` never being called
        at all while `scene.gi["enabled"]` is False)."""
        if result.gi_key is None:
            return
        self._result = result.gi_concat
        self._entity_ranges = result.gi_entity_ranges
        self._caster_list = result.gi_caster_list
        self._world_version = result.geom_version
        self._key = result.gi_key
