"""Real-time ray tracing: soft shadows and mouse picking.

Shadow rays are cast from face centroids toward jittered sample points on each
light's spherical area (Moller-Trumbore intersection, vectorized over rays x
occluder triangles). The fraction of unblocked rays gives a soft shadow factor
per face. Results are cached per (entity, light) and only recomputed when the
receiver, the light, or any shadow caster actually moved, so static scenes pay
the tracing cost once; moving lights (the flashlight) retrace on a configurable
frame interval.
"""
from __future__ import annotations

import numpy as np

_RAY_CHUNK = 128       # rays per Moller-Trumbore batch (bounds temp memory)
_ORIGIN_OFFSET = 0.02  # push ray origins off the surface to avoid self-hits
_RECEIVER_INTERVAL = 3  # frames a moving receiver may reuse its cached shadows


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


def _intersect_any(origins, dirs, max_t, v0, e1, e2) -> np.ndarray:
    """For each ray, is anything hit before max_t? (R,) bool."""
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


def _world_triangles(entity):
    m = entity.transform.matrix()
    wv = entity.mesh.vertices @ m[:3, :3].T + m[:3, 3]
    tri = wv[entity.mesh.faces]
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
        self._caster_mats = {}      # id(entity) -> matrix bytes
        self._world_version = -1
        self._cache = {}            # (entity, light) -> cache dict

    def refresh(self, scene) -> None:
        """Rebuild the world-space occluder soup if any shadow caster moved."""
        self.frame += 1
        casters = [e for e in scene.entities
                   if e.mesh is not None and e.visible and e.casts_shadow]
        mats = {id(e): e.transform.matrix().tobytes() for e in casters}
        if mats == self._caster_mats and self._occ is not None:
            return
        v0s, e1s, e2s = [], [], []
        for e in casters:
            v0, e1, e2 = _world_triangles(e)
            v0s.append(v0)
            e1s.append(e1)
            e2s.append(e2)
        if v0s:
            v0 = np.concatenate(v0s)
            e1 = np.concatenate(e1s)
            e2 = np.concatenate(e2s)
        else:
            v0 = e1 = e2 = np.zeros((0, 3))
        centroids = v0 + (e1 + e2) / 3.0
        self._occ = (v0, e1, e2, centroids)
        self._caster_mats = mats
        self._world_version += 1
        # drop cache entries for entities no longer in the scene
        live = set(scene.entities)
        self._cache = {k: v for k, v in self._cache.items() if k[0] in live}

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
                round(float(light_pos[2]), 4), light.radius, light.shadow_samples)
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

        factors = np.ones(len(centroids))
        idx = np.nonzero(active)[0]
        v0, e1, e2, occ_c = self._occ
        if len(idx) > 0 and len(v0) > 0:
            # only occluders near the light can block its rays
            reach = light.range + light.radius + 1.0
            near = np.linalg.norm(occ_c - light_pos, axis=1) < reach
            ov0, oe1, oe2 = v0[near], e1[near], e2[near]
            if len(ov0) > 0:
                origins = centroids[idx] + normals[idx] * _ORIGIN_OFFSET
                samples = light_pos + sphere_samples(light.shadow_samples) * light.radius
                hits = np.zeros(len(idx))
                for s in samples:
                    d = s - origins
                    dist = np.linalg.norm(d, axis=1)
                    dist = np.maximum(dist, 1e-9)
                    blocked = _intersect_any(origins, d / dist[:, None],
                                             dist - 0.05, ov0, oe1, oe2)
                    hits += blocked
                factors[idx] = 1.0 - hits / len(samples)

        self._cache[key] = {"factors": factors, "world": self._world_version,
                            "m": mkey, "l": lkey, "frame": self.frame}
        return factors
