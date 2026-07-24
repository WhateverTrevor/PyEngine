"""Vectorized numpy BVH for ray/triangle acceleration.

Builds a flat bounding volume hierarchy (BVH) over a triangle soup given as
Moller-Trumbore edge vectors (v0, e1, e2) -- the same representation
raytrace.py's occluder soup already uses -- and traverses it batched over a
whole ray array at once, cutting the per-ray triangle test cost from O(T)
to ~O(log T).

TRAVERSAL DESIGN (the key constraint): a Python loop over individual RAYS
would defeat the purpose (numpy overhead per ray). Instead, `any_hit`/
`nearest` maintain a single "frontier" of (ray_index, node_index) pairs as
flat numpy arrays and advance it one TREE LEVEL per Python-loop iteration:
each iteration vectorizes a ray-box test over the whole frontier at once,
resolves any (ray, LEAF) pairs with a batched, padded Moller-Trumbore test,
and expands (ray, INTERNAL) pairs into their two children for the next
iteration. Any single ray visits ~O(log T) nodes, but the loop itself is
bounded by tree DEPTH (~11 iterations at 10k triangles / leaf size 8), not
by ray count -- this is the numpy analogue of a GPU stackless BVH walk.

Node layout (flat structure-of-arrays -- no per-node Python objects, so the
traversal loop above only ever touches numpy arrays):
  bounds_min/bounds_max : (N, 3) float64 -- world-space AABB
  left/right            : (N,) int32     -- child node indices, -1 if leaf
  tri_start/tri_count   : (N,) int32     -- leaf triangle range into
                                             `tri_order`/`v0`/`e1`/`e2`
                                             (tri_count == 0 => internal node)
  tri_order             : (T,) int64     -- original triangle index for each
                                             slot; leaves are the contiguous
                                             range [tri_start, tri_start+
                                             tri_count) into this array
  v0/e1/e2              : (T, 3) float64 -- input triangles REORDERED to
                                             `tri_order`, so a leaf's
                                             triangles are one contiguous
                                             slice (no per-query gather)

Split: median split on the longest axis of the node's triangle CENTROID
bounds (an O(T) `argpartition`, no full sort) -- simple, deterministic, and
plenty good at the moderate triangle counts (~1e3-1e4) this module targets.
Leaves stop subdividing at <= `leaf_size` triangles (default 8).
"""
from __future__ import annotations

import numpy as np

DEFAULT_LEAF_SIZE = 8


class BVH:
    """Flat BVH node arrays + the triangle soup reordered to match them.
    Build with `build_bvh`; query with `any_hit`/`nearest` below."""

    __slots__ = ("bounds_min", "bounds_max", "left", "right",
                 "tri_start", "tri_count", "tri_order", "v0", "e1", "e2",
                 "leaf_size")

    def __init__(self, bounds_min, bounds_max, left, right, tri_start, tri_count,
                 tri_order, v0, e1, e2, leaf_size):
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.left = left
        self.right = right
        self.tri_start = tri_start
        self.tri_count = tri_count
        self.tri_order = tri_order
        self.v0 = v0
        self.e1 = e1
        self.e2 = e2
        self.leaf_size = leaf_size


def build_bvh(v0: np.ndarray, e1: np.ndarray, e2: np.ndarray,
             leaf_size: int = DEFAULT_LEAF_SIZE) -> BVH:
    """Build a BVH over triangles given as Moller-Trumbore edge vectors
    (v0, e1, e2), each (T, 3). T == 0 is handled (an empty single-leaf
    tree; `any_hit`/`nearest` short-circuit on it without traversing)."""
    v0 = np.asarray(v0, dtype=np.float64)
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    T = len(v0)
    if T == 0:
        z = np.zeros((0, 3))
        return BVH(np.zeros((1, 3)), np.zeros((1, 3)), np.array([-1], dtype=np.int32),
                   np.array([-1], dtype=np.int32), np.array([0], dtype=np.int32),
                   np.array([0], dtype=np.int32), np.zeros((0,), dtype=np.int64),
                   z, z, z, leaf_size)

    v1 = v0 + e1
    v2 = v0 + e2
    tri_min = np.minimum(v0, np.minimum(v1, v2))
    tri_max = np.maximum(v0, np.maximum(v1, v2))
    centroids = (tri_min + tri_max) * 0.5

    bmins, bmaxs, lefts, rights, tstarts, tcounts = [], [], [], [], [], []
    tri_order: list = []

    def build(indices: np.ndarray) -> int:
        node_id = len(bmins)
        bmins.append(tri_min[indices].min(axis=0))
        bmaxs.append(tri_max[indices].max(axis=0))
        lefts.append(-1)
        rights.append(-1)
        tstarts.append(0)
        tcounts.append(0)

        c = centroids[indices]
        extent = c.max(axis=0) - c.min(axis=0)
        axis = int(np.argmax(extent))
        # leaf if small enough, or centroids coincide on every axis (further
        # splitting would never terminate / wouldn't separate anything)
        if len(indices) <= leaf_size or extent[axis] <= 1e-12:
            tstarts[node_id] = len(tri_order)
            tcounts[node_id] = len(indices)
            tri_order.extend(int(i) for i in indices)
            return node_id

        key = c[:, axis]
        mid = len(indices) // 2
        part = np.argpartition(key, mid)
        left_id = build(indices[part[:mid]])
        right_id = build(indices[part[mid:]])
        lefts[node_id] = left_id
        rights[node_id] = right_id
        return node_id

    build(np.arange(T))

    order = np.asarray(tri_order, dtype=np.int64)
    return BVH(
        np.asarray(bmins), np.asarray(bmaxs),
        np.asarray(lefts, dtype=np.int32), np.asarray(rights, dtype=np.int32),
        np.asarray(tstarts, dtype=np.int32), np.asarray(tcounts, dtype=np.int32),
        order, v0[order], e1[order], e2[order], leaf_size)


def _ray_aabb(o, d, bmin, bmax, t_hi):
    """Slab test, vectorized over a (m,) batch. Returns (hit bool (m,),
    tnear (m,)); `t_hi` bounds the far end (per-ray current max_t)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d = 1.0 / d
        t0 = (bmin - o) * inv_d
        t1 = (bmax - o) * inv_d
        tsmall = np.minimum(t0, t1)
        tbig = np.maximum(t0, t1)
        tnear = np.max(tsmall, axis=1)
        tfar = np.min(tbig, axis=1)
    tnear = np.maximum(tnear, 1e-4)
    tfar = np.minimum(tfar, t_hi)
    return tnear <= tfar, tnear


def _gather_leaf(bvh: BVH, node_ids: np.ndarray):
    """(v0, e1, e2, valid) for the (m, leaf_size) padded triangle slots of
    each leaf node in `node_ids` -- `valid` masks slots past that leaf's
    actual triangle count so every leaf can be gathered at a uniform
    `leaf_size` width without a ragged per-leaf Python loop."""
    starts = bvh.tri_start[node_ids]
    counts = bvh.tri_count[node_ids]
    offsets = np.arange(bvh.leaf_size)
    slots = starts[:, None] + offsets[None, :]
    valid = offsets[None, :] < counts[:, None]
    slots = np.minimum(slots, max(len(bvh.v0) - 1, 0))
    return bvh.v0[slots], bvh.e1[slots], bvh.e2[slots], valid, slots


def _leaf_moller_trumbore(o, d, v0, e1, e2, valid, t_hi):
    """Batched Moller-Trumbore over (m, leaf_size) triangle slots against
    one ray per batch row `o`/`d` (m, 3). Mirrors raytrace.py's
    `_intersect_any`/`_nearest_hit_faces` math exactly, just with the
    triangle axis bounded to `leaf_size` instead of the whole soup."""
    p = np.cross(d[:, None, :], e2)
    det = np.einsum("mlk,mlk->ml", p, e1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        tv = o[:, None, :] - v0
        u = np.einsum("mlk,mlk->ml", tv, p) * inv
        q = np.cross(tv, e1)
        v = np.einsum("mk,mlk->ml", d, q) * inv
        t = np.einsum("mlk,mlk->ml", e2, q) * inv
        hit = (valid & (np.abs(det) > 1e-12) & (u >= 0.0) & (v >= 0.0)
               & (u + v <= 1.0) & (t > 1e-4) & (t < t_hi[:, None]))
    return hit, t


def any_hit(bvh: BVH, origins: np.ndarray, dirs: np.ndarray,
           t_max: np.ndarray) -> np.ndarray:
    """Per ray, is ANY triangle in `bvh` hit before `t_max`? (R,) bool.
    `t_max` may be a scalar or (R,) array (broadcast either way)."""
    n = len(origins)
    blocked = np.zeros(n, dtype=bool)
    if n == 0 or len(bvh.v0) == 0:
        return blocked
    t_max_full = np.broadcast_to(np.asarray(t_max, dtype=np.float64), (n,)).copy()

    frontier_ray = np.arange(n, dtype=np.int64)
    frontier_node = np.zeros(n, dtype=np.int64)
    while len(frontier_ray) > 0:
        r, nd = frontier_ray, frontier_node
        hit_box, _tnear = _ray_aabb(origins[r], dirs[r], bvh.bounds_min[nd],
                                    bvh.bounds_max[nd], t_max_full[r])
        r, nd = r[hit_box], nd[hit_box]
        if len(r) == 0:
            break
        is_leaf = bvh.tri_count[nd] > 0
        if is_leaf.any():
            lr, lnd = r[is_leaf], nd[is_leaf]
            v0, e1, e2, valid, _slots = _gather_leaf(bvh, lnd)
            hit, _t = _leaf_moller_trumbore(origins[lr], dirs[lr], v0, e1, e2,
                                            valid, t_max_full[lr])
            blocked[lr[hit.any(axis=1)]] = True
        if (~is_leaf).any():
            ir, ind = r[~is_leaf], nd[~is_leaf]
            keep = ~blocked[ir]   # don't keep descending for already-blocked rays
            ir, ind = ir[keep], ind[keep]
            frontier_ray = np.concatenate([ir, ir])
            frontier_node = np.concatenate([bvh.left[ind], bvh.right[ind]])
        else:
            frontier_ray = np.zeros(0, dtype=np.int64)
            frontier_node = np.zeros(0, dtype=np.int64)
    return blocked


def _update_nearest(best_t, best_tri, ray_ids, cand_t, cand_tri):
    """Compare-and-swap (ray_ids, cand_t, cand_tri) candidates into the
    full-size `best_t`/`best_tri` arrays, in place. A ray can legitimately
    appear more than once in `ray_ids` within one traversal step (it may
    pass the box test against several sibling/cousin nodes at once), so a
    naive parallel `best_t[ray_ids] = cand_t` would be last-write-wins and
    could pair a losing t with a winning candidate's triangle -- lexsort by
    (ray_id primary, t secondary) so each ray's group is t-ascending, then
    keep just the first row of each group (that group's true minimum)
    before comparing against the running best."""
    valid = cand_t < np.inf
    if not valid.any():
        return
    ray_ids, cand_t, cand_tri = ray_ids[valid], cand_t[valid], cand_tri[valid]
    order = np.lexsort((cand_t, ray_ids))  # primary key = last arg = ray_ids
    r, ct, ci = ray_ids[order], cand_t[order], cand_tri[order]
    first = np.empty(len(r), dtype=bool)
    first[0] = True
    if len(r) > 1:
        first[1:] = r[1:] != r[:-1]
    r, ct, ci = r[first], ct[first], ci[first]
    better = ct < best_t[r]
    r, ct, ci = r[better], ct[better], ci[better]
    best_t[r] = ct
    best_tri[r] = ci


def nearest(bvh: BVH, origins: np.ndarray, dirs: np.ndarray,
           max_t: float = 200.0):
    """Nearest-hit (tri_index, t) per ray. tri_index == -1 for a miss (t is
    left at `max_t` in that case, same convention as brute-force callers
    that mask on tri_index)."""
    n = len(origins)
    best_t = np.full(n, float(max_t), dtype=np.float64)
    best_tri = np.full(n, -1, dtype=np.int64)
    if n == 0 or len(bvh.v0) == 0:
        return best_tri, best_t

    frontier_ray = np.arange(n, dtype=np.int64)
    frontier_node = np.zeros(n, dtype=np.int64)
    while len(frontier_ray) > 0:
        r, nd = frontier_ray, frontier_node
        hit_box, _tnear = _ray_aabb(origins[r], dirs[r], bvh.bounds_min[nd],
                                    bvh.bounds_max[nd], best_t[r])
        r, nd = r[hit_box], nd[hit_box]
        if len(r) == 0:
            break
        is_leaf = bvh.tri_count[nd] > 0
        if is_leaf.any():
            lr, lnd = r[is_leaf], nd[is_leaf]
            v0, e1, e2, valid, slots = _gather_leaf(bvh, lnd)
            hit, t = _leaf_moller_trumbore(origins[lr], dirs[lr], v0, e1, e2,
                                           valid, best_t[lr])
            t_masked = np.where(hit, t, np.inf)
            local_best = t_masked.min(axis=1)
            local_arg = np.argmin(t_masked, axis=1)
            global_tri = bvh.tri_order[slots[np.arange(len(lr)), local_arg]]
            _update_nearest(best_t, best_tri, lr, local_best, global_tri)
        if (~is_leaf).any():
            ir, ind = r[~is_leaf], nd[~is_leaf]
            frontier_ray = np.concatenate([ir, ir])
            frontier_node = np.concatenate([bvh.left[ind], bvh.right[ind]])
        else:
            frontier_ray = np.zeros(0, dtype=np.int64)
            frontier_node = np.zeros(0, dtype=np.int64)
    return best_tri, best_t
