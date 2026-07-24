"""Judge checks: BVH ray/triangle acceleration (engine/bvh.py), Milestone 1.

Standalone/isolated from raytrace.py on purpose (Milestone 1 is meant to
de-risk the BVH on its own before anything is wired into the renderer): a
brute-force Moller-Trumbore reference is reimplemented locally here (not
imported from raytrace.py) so a shared bug in that module couldn't hide a
BVH bug behind matching wrong answers.

Correctness bar (per the task spec): `any_hit` must match brute-force
EXACTLY (same bool per ray -- shadow rays are a boolean question). `nearest`
must match brute-force's hit distance `t` within float tolerance; the
triangle INDEX may legitimately differ on exact coplanar ties (both are
correct, the spec explicitly allows this), so we only assert on t/miss
agreement, never on tri-id equality.
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np

from engine.bvh import any_hit, build_bvh, nearest

RNG = np.random.default_rng(12345)


def random_triangle_soup(n: int):
    """n small, randomly placed/oriented triangles scattered through a
    [-5, 5]^3 cube -- mimics a scattered high-poly mesh well enough to
    exercise a real spatial split without depending on engine/mesh.py."""
    centers = RNG.uniform(-5.0, 5.0, size=(n, 3))
    v0 = centers + RNG.uniform(-0.3, 0.3, size=(n, 3))
    e1 = RNG.uniform(-0.4, 0.4, size=(n, 3))
    e2 = RNG.uniform(-0.4, 0.4, size=(n, 3))
    return v0, e1, e2


def random_rays(n: int, spread: float = 6.0):
    origins = RNG.uniform(-spread, spread, size=(n, 3))
    dirs = RNG.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return origins, dirs


def brute_any_hit(origins, dirs, t_max, v0, e1, e2, chunk=128):
    """Independent (not raytrace.py) reference: dense (rays x tris)
    Moller-Trumbore, chunked over rays to bound memory."""
    n = len(origins)
    blocked = np.zeros(n, dtype=bool)
    if len(v0) == 0:
        return blocked
    t_max = np.broadcast_to(np.asarray(t_max, dtype=np.float64), (n,))
    for i in range(0, n, chunk):
        o, d, mt = origins[i:i + chunk], dirs[i:i + chunk], t_max[i:i + chunk]
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
                   & (t > 1e-4) & (t < mt[:, None]))
        blocked[i:i + chunk] = hit.any(axis=1)
    return blocked


def brute_nearest(origins, dirs, v0, e1, e2, max_t=200.0, chunk=128):
    n = len(origins)
    best_t = np.full(n, float(max_t))
    best_tri = np.full(n, -1, dtype=np.int64)
    if len(v0) == 0:
        return best_tri, best_t
    for i in range(0, n, chunk):
        o, d = origins[i:i + chunk], dirs[i:i + chunk]
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
        best_t[i:i + chunk] = np.where(valid, t_masked[np.arange(len(o)), j], max_t)
        best_tri[i:i + chunk] = np.where(valid, j, -1)
    return best_tri, best_t


# --- correctness: any_hit exact match -------------------------------------
v0, e1, e2 = random_triangle_soup(3000)
origins, dirs = random_rays(1200)
t_max = RNG.uniform(2.0, 40.0, size=len(origins))

bvh = build_bvh(v0, e1, e2, leaf_size=8)
bvh_blocked = any_hit(bvh, origins, dirs, t_max)
ref_blocked = brute_any_hit(origins, dirs, t_max, v0, e1, e2)
assert bvh_blocked.dtype == np.bool_
n_mismatch = int((bvh_blocked != ref_blocked).sum())
assert n_mismatch == 0, f"any_hit mismatched brute-force on {n_mismatch}/{len(origins)} rays"
print(f"any_hit: exact match with brute-force on {len(origins)} rays "
     f"({int(ref_blocked.sum())} blocked, {len(v0)} triangles)")

# --- correctness: nearest matches within tolerance ------------------------
bvh_tri, bvh_t = nearest(bvh, origins, dirs, max_t=200.0)
ref_tri, ref_t = brute_nearest(origins, dirs, v0, e1, e2, max_t=200.0)
bvh_hit, ref_hit = bvh_tri >= 0, ref_tri >= 0
assert np.array_equal(bvh_hit, ref_hit), "nearest hit/miss disagreement vs brute-force"
assert np.allclose(bvh_t[ref_hit], ref_t[ref_hit], atol=1e-6, rtol=1e-6), \
    "nearest t disagreement vs brute-force beyond float tolerance"
# where both hit, the actual world-space hit point must match even on ties
# (tri-id may legitimately differ on exact coplanar ties -- assert on the
# point in space, not which triangle claimed it, per the task's tolerance)
hp_bvh = origins[ref_hit] + dirs[ref_hit] * bvh_t[ref_hit, None]
hp_ref = origins[ref_hit] + dirs[ref_hit] * ref_t[ref_hit, None]
assert np.allclose(hp_bvh, hp_ref, atol=1e-5), "nearest hit POINT disagreement vs brute-force"
tri_mismatches = int((bvh_tri[ref_hit] != ref_tri[ref_hit]).sum())
print(f"nearest: t/hit-point match within tolerance on {int(ref_hit.sum())} hits "
     f"({tri_mismatches} tie-break tri-id differences, expected-acceptable)")

# --- build determinism -----------------------------------------------------
bvh2 = build_bvh(v0, e1, e2, leaf_size=8)
for name in ("bounds_min", "bounds_max", "left", "right", "tri_start", "tri_count", "tri_order"):
    assert np.array_equal(getattr(bvh, name), getattr(bvh2, name)), \
        f"build_bvh not deterministic: '{name}' differs across identical builds"
print("build determinism: two builds from identical input produced identical node arrays")

# --- empty-soup edge case ---------------------------------------------------
empty_v0 = np.zeros((0, 3))
empty_bvh = build_bvh(empty_v0, empty_v0, empty_v0)
eo, ed = random_rays(20)
assert not any_hit(empty_bvh, eo, ed, 50.0).any()
etri, _et = nearest(empty_bvh, eo, ed)
assert (etri == -1).all()
print("empty soup: any_hit all-False, nearest all-miss (matches brute-force convention)")

# --- speedup vs brute-force at a few thousand triangles --------------------
v0b, e1b, e2b = random_triangle_soup(6000)
originsb, dirsb = random_rays(600)
t_maxb = np.full(len(originsb), 100.0)

t0 = time.perf_counter()
bvh_big = build_bvh(v0b, e1b, e2b)
build_s = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(5):
    any_hit(bvh_big, originsb, dirsb, t_maxb)
bvh_any_s = (time.perf_counter() - t0) / 5

t0 = time.perf_counter()
for _ in range(5):
    brute_any_hit(originsb, dirsb, t_maxb, v0b, e1b, e2b)
brute_any_s = (time.perf_counter() - t0) / 5

t0 = time.perf_counter()
for _ in range(5):
    nearest(bvh_big, originsb, dirsb)
bvh_near_s = (time.perf_counter() - t0) / 5

t0 = time.perf_counter()
for _ in range(5):
    brute_nearest(originsb, dirsb, v0b, e1b, e2b)
brute_near_s = (time.perf_counter() - t0) / 5

print(f"speedup @ {len(v0b)} tris, {len(originsb)} rays: build={build_s * 1000:.1f}ms  "
     f"any_hit bvh={bvh_any_s * 1000:.2f}ms vs brute={brute_any_s * 1000:.2f}ms "
     f"({brute_any_s / max(bvh_any_s, 1e-9):.1f}x)  "
     f"nearest bvh={bvh_near_s * 1000:.2f}ms vs brute={brute_near_s * 1000:.2f}ms "
     f"({brute_near_s / max(bvh_near_s, 1e-9):.1f}x)")
assert bvh_any_s < brute_any_s, "BVH any_hit should be faster than brute-force at 6k triangles"
assert bvh_near_s < brute_near_s, "BVH nearest should be faster than brute-force at 6k triangles"

print("ALL BVH CHECKS PASSED (Milestone 1: isolation)")

# ===========================================================================
# Milestone 2: integration into raytrace.py / lighting_bake.py
# ===========================================================================
import engine
from engine.raytrace import (BVH_THRESHOLD, ShadowTracer, _directional_shadow_math,
                             _gi_math, _point_shadow_math, _world_triangles,
                             build_bvh_for_occ)
from engine.renderer import _world_face_geometry

# --- low-poly scene: <=threshold path must be byte-identical to main -------
scene_lo = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.3, -1.0, -0.2)))
scene_lo.add(engine.Entity("floor", mesh=engine.checkerboard(6, 1.0)))
scene_lo.add(engine.Entity("box", mesh=engine.cube(1.0), position=engine.Vec3(0, 0.5, 0)))
tracer_lo = ShadowTracer()
tracer_lo.refresh(scene_lo)
assert tracer_lo.occluder_triangle_count() <= BVH_THRESHOLD, \
    "low-poly fixture must stay at/below BVH_THRESHOLD for this check to mean anything"
assert tracer_lo._occ_bvh is None, "low-poly occluder soup must not build a BVH"
centroids_lo, normals_lo = _world_face_geometry(scene_lo.entities[0])
active_lo = np.ones(len(centroids_lo), dtype=bool)
factors_lo_a = _directional_shadow_math(tracer_lo._occ, np.array([-0.3, -1.0, -0.2]),
                                        1.0, 6, centroids_lo, normals_lo, active_lo,
                                        bvh=tracer_lo._occ_bvh)
factors_lo_b = _directional_shadow_math(tracer_lo._occ, np.array([-0.3, -1.0, -0.2]),
                                        1.0, 6, centroids_lo, normals_lo, active_lo,
                                        bvh=None)
assert np.array_equal(factors_lo_a, factors_lo_b), \
    "low-poly path (bvh=None from refresh()) must be byte-identical to explicit bvh=None"
print(f"low-poly integration: {tracer_lo.occluder_triangle_count()} occluder tris, "
     f"BVH skipped (None), directional shadow factors byte-identical")

# --- high-poly scene: subdivided icosphere (~5k faces), in-memory only ----
sphere_mesh = engine.icosphere(radius=2.0, subdivisions=4)
n_faces = len(sphere_mesh.faces)
assert n_faces > BVH_THRESHOLD, f"expected a high-poly sphere, got {n_faces} faces"
sphere_entity = engine.Entity("hp_sphere", mesh=sphere_mesh)

occ_v0, occ_e1, occ_e2 = _world_triangles(sphere_entity)
occ_centroids = occ_v0 + (occ_e1 + occ_e2) / 3.0
occ = (occ_v0, occ_e1, occ_e2, occ_centroids)
hp_bvh = build_bvh_for_occ(occ)
assert hp_bvh is not None, "high-poly soup must build a BVH"

centroids_all, normals_all = _world_face_geometry(sphere_entity)
# bound the RECEIVER side to a modest subset for a fast, controllable-scale
# test (per task instructions: profile at a few thousand triangles, not a
# multi-minute bench) -- the OCCLUDER side (what the BVH actually
# accelerates) stays the full ~5k-triangle sphere throughout.
rsub = slice(0, 250)
centroids_r, normals_r = centroids_all[rsub], normals_all[rsub]
active_r = np.ones(len(centroids_r), dtype=bool)

sun_dir = np.array([-0.4, -1.0, -0.3])
sun_brute = _directional_shadow_math(occ, sun_dir, 1.5, 8, centroids_r, normals_r,
                                     active_r, bvh=None)
sun_bvh = _directional_shadow_math(occ, sun_dir, 1.5, 8, centroids_r, normals_r,
                                   active_r, bvh=hp_bvh)
sun_diff = np.abs(sun_bvh - sun_brute)
assert sun_diff.max() < 0.05, f"directional shadow factors diverged too far: max diff {sun_diff.max()}"
print(f"high-poly directional shadow ({n_faces} occluder tris, {len(centroids_r)} receivers): "
     f"BVH vs brute max diff {sun_diff.max():.4f}, mean diff {sun_diff.mean():.5f}")

light_pos = np.array([0.0, 0.0, 4.0])
pt_brute = _point_shadow_math(occ, light_pos, 0.3, 10.0, 8, centroids_r, normals_r,
                              active_r, bvh=None)
pt_bvh = _point_shadow_math(occ, light_pos, 0.3, 10.0, 8, centroids_r, normals_r,
                            active_r, bvh=hp_bvh)
pt_diff = np.abs(pt_bvh - pt_brute)
# the BVH path deliberately skips the point-light near-occluder prefilter
# (see raytrace.py's _point_shadow_math docstring) and queries the full
# soup directly -- geometrically equivalent, but allow a slightly wider
# tolerance than the sun path per the task's explicit "small tolerance"
# bar for this path.
assert pt_diff.max() < 0.08, f"point shadow factors diverged too far: max diff {pt_diff.max()}"
print(f"high-poly point-light shadow: BVH vs brute max diff {pt_diff.max():.4f}, "
     f"mean diff {pt_diff.mean():.5f}")

tri_face_id = np.arange(n_faces, dtype=np.int64)  # sphere is all-triangle: face id == tri id
albedo_c = sphere_mesh.face_colors
direct_c = np.tile(np.array([180.0, 180.0, 180.0]) / 255.0, (n_faces, 1))
gi_brute = _gi_math(occ, tri_face_id, albedo_c, direct_c, centroids_r, normals_r,
                    12, 1.0, bvh=None)
gi_bvh = _gi_math(occ, tri_face_id, albedo_c, direct_c, centroids_r, normals_r,
                  12, 1.0, bvh=hp_bvh)
gi_diff = np.abs(gi_bvh - gi_brute)
assert gi_diff.max() < 8.0, f"GI factors diverged too far: max diff {gi_diff.max()}"
print(f"high-poly GI: BVH vs brute max diff {gi_diff.max():.4f}, mean diff {gi_diff.mean():.5f}")

# --- bake-time improvement bound: BVH < 0.5x brute at ~5k faces -----------
def _one_pass(bvh):
    _directional_shadow_math(occ, sun_dir, 1.5, 8, centroids_r, normals_r, active_r, bvh=bvh)
    _point_shadow_math(occ, light_pos, 0.3, 10.0, 8, centroids_r, normals_r, active_r, bvh=bvh)
    _gi_math(occ, tri_face_id, albedo_c, direct_c, centroids_r, normals_r, 12, 1.0, bvh=bvh)


t0 = time.perf_counter()
_one_pass(None)
brute_bake_s = time.perf_counter() - t0

t0 = time.perf_counter()
_one_pass(hp_bvh)
bvh_bake_s = time.perf_counter() - t0

print(f"bake-time @ {n_faces} occluder tris, {len(centroids_r)} receivers: "
     f"brute={brute_bake_s:.3f}s  bvh={bvh_bake_s:.3f}s  "
     f"({brute_bake_s / max(bvh_bake_s, 1e-9):.1f}x)")
assert bvh_bake_s < 0.5 * brute_bake_s, \
    f"BVH bake ({bvh_bake_s:.3f}s) should be < 0.5x brute-force bake ({brute_bake_s:.3f}s)"

print("ALL BVH CHECKS PASSED (Milestone 2: integration)")
