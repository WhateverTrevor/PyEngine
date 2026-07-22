"""Distance-based level of detail: pure-numpy decimation + runtime selection.

Decimation is VERTEX CLUSTERING: snap every vertex to the nearest point on a
grid sized off the mesh's own bounding box, weld vertices that land on the
same grid point, rebuild triangles through the welded indices, and drop any
triangle that collapsed to <3 distinct vertices. This is fast (a handful of
vectorized numpy passes, no per-triangle Python loop) and robust for
arbitrary geometry, at the cost of edge-collapse (QEM) quality -- acceptable
for background/distant LODs, out of scope for this engine's budget.

Per-face attributes (`face_colors`/`face_roughness`/`face_metallic`/
`face_emissive`/`face_opacity`) are carried onto each output face from ONE
representative source triangle (the first surviving source triangle that
welds to that output triangle's vertex triple) -- not an area-weighted
average. Documented tradeoff: cheap (falls out of the same `np.unique` call
that dedupes triangles) and exact for uniformly-colored regions (the common
case for hard-surface art), at the cost of picking a somewhat arbitrary
source color at a boundary between two differently-colored regions instead
of blending them. `face_uvs` are NOT carried -- a decimated face's box
projection is regenerated from scratch by `Mesh.__init__` (its normal
default-UV fallback), which is more coherent for a totally different face
layout than reusing a stale per-source-face UV would be.

Each LOD mesh returned by `generate_lods` (levels 1+, not LOD0) carries a
bolted-on `.lod_source_faces` attribute: an (M,) int array mapping every
output face back to its representative index into the ORIGINAL mesh's
`.faces`. This isn't used by decimation itself -- it's for the renderer
(see renderer.py's `_lod_gather`): ray-traced shadow/GI values are always
computed at LOD0 face granularity (so the shadow/GI world-version cache
never invalidates on an LOD switch, see raytrace.py), then gathered onto
whichever LOD is actually being rasterized via this same mapping.
"""
from __future__ import annotations

import math

import numpy as np

from .mesh import Mesh

# meshes at or below this face count skip LOD generation entirely -- every
# built-in primitive (6-80 faces) and any small import stays LOD-free, so it
# renders through the exact same code path as before this feature existed.
LOD_FACE_THRESHOLD = 200

# Above this face count, a mesh that carries NO precomputed LODs gets an
# on-demand coarse shadow/GI occluder proxy (see Entity.shadow_mesh). Kept
# well above the largest built-in procedural mesh (~256 faces) so every
# built-in keeps shadow_mesh() == mesh (byte-identical shadows); only heavy
# imports (the ~10k-face FBX that froze the bake) are ever proxied.
SHADOW_PROXY_THRESHOLD = 1000

# LOD1/2/3 target ~50%/25%/12% of the source face count.
DEFAULT_LOD_RATIOS = (0.5, 0.25, 0.12)

# world-bound multiples at which an entity should step UP to a coarser LOD;
# extended/truncated to however many extra LOD levels a mesh actually has.
# e.g. a mesh whose LOD0 bounding radius is `b`: LOD1 kicks in beyond
# distance 8*b, LOD2 beyond 20*b, LOD3 beyond 45*b.
LOD_DISTANCE_FACTORS = (8.0, 20.0, 45.0)

# switch back DOWN to a finer LOD only once distance drops below
# up_threshold * (1 - HYSTERESIS) -- the dead band that stops an entity
# sitting near a boundary from flipping LODs (and thrashing GPU vertex-
# buffer caches, which key off id() of the mesh's array -- see gl_renderer.py
# / wgpu_renderer.py's geometry cache) every frame.
LOD_HYSTERESIS = 0.2


# ---------------------------------------------------------------------------
# decimation
# ---------------------------------------------------------------------------
def _tri_to_face_map(mesh: Mesh) -> np.ndarray:
    """Index into `mesh.faces` (M,) for every row of `mesh.tri_faces` (T,) --
    a quad contributes two consecutive triangles mapping to the same face id
    (mirrors the identical construction in raytrace.ShadowTracer.refresh)."""
    faces = mesh.faces
    is_tri = (faces[:, 3] == faces[:, 2]) | (faces[:, 3] == faces[:, 0])
    counts = np.where(is_tri, 1, 2)
    return np.repeat(np.arange(len(faces)), counts)


def _weld_and_rebuild(mesh: Mesh, grid_res: int):
    """One vertex-clustering pass at a given grid resolution (cells along the
    longest bbox axis). Returns (out_tri (M2,3) int, out_verts (V2,3) float,
    out_src (M2,) int -- representative index into `mesh.faces`), or None if
    this resolution welds everything down to degenerate triangles."""
    aabb_min, aabb_max = mesh.aabb_min, mesh.aabb_max
    extent = np.maximum(aabb_max - aabb_min, 1e-9)
    cell = float(extent.max()) / max(int(grid_res), 1)
    cell = max(cell, 1e-9)

    # exact-integer grid coordinates -- clustering key, no float snap fuzz
    grid_idx = np.round((mesh.vertices - aabb_min) / cell).astype(np.int64)
    keys, inverse = np.unique(grid_idx, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    tri_to_face = _tri_to_face_map(mesh)
    tri = inverse[mesh.tri_faces]  # (T, 3) welded vertex-cluster ids
    nondeg = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    if not nondeg.any():
        return None
    tri = tri[nondeg]
    src = tri_to_face[nondeg]

    # dedupe triangles that welded onto the same 3 clusters (order-invariant
    # key), keeping the FIRST occurrence -- its original winding (hence
    # normal direction) and its source face id for color-carrying.
    sorted_tri = np.sort(tri, axis=1)
    _, first = np.unique(sorted_tri, axis=0, return_index=True)
    out_tri = tri[first]
    out_src = src[first]
    if len(out_tri) == 0:
        return None

    # compact to only the clusters actually referenced by a surviving face
    used, remap = np.unique(out_tri, return_inverse=True)
    out_tri = remap.reshape(out_tri.shape).astype(np.int64)
    cluster_pos = np.clip(aabb_min + keys.astype(np.float64) * cell, aabb_min, aabb_max)
    out_verts = cluster_pos[used]
    return out_tri, out_verts, out_src


def decimate(mesh: Mesh, target_faces: int, min_faces: int = 4) -> Mesh:
    """A lower-poly `Mesh` built by vertex-clustering `mesh` toward
    `target_faces` triangles. Searches grid resolution (binary search on face
    count, which is monotonically non-decreasing in resolution) for the
    closest match found in a bounded number of tries -- vertex clustering
    doesn't hit an exact target count, this gets close.

    Falls back to returning `mesh` unchanged if no resolution in the search
    range produces a non-degenerate result (pathological input only -- every
    fixture and real mesh this engine has seen decimates fine).
    """
    orig_faces = len(mesh.faces)
    target_faces = max(int(min_faces), min(int(target_faces), max(orig_faces - 1, int(min_faces))))
    if orig_faces <= target_faces:
        return mesh

    lo, hi = 2, 512
    best = None  # (diff, result)
    for _ in range(14):
        mid = (lo + hi) // 2
        result = _weld_and_rebuild(mesh, mid)
        n = 0 if result is None else len(result[0])
        diff = abs(n - target_faces) if n > 0 else 10 ** 9
        if best is None or diff < best[0]:
            best = (diff, result)
        if n == target_faces or lo >= hi:
            break
        if n < target_faces:
            lo = mid + 1
        else:
            hi = mid - 1

    result = best[1] if best is not None else None
    if result is None:
        return mesh  # pathological mesh -- never seen in practice, stay safe

    out_tri, out_verts, out_src = result
    out = Mesh(out_verts, out_tri.tolist(),
              face_colors=mesh.face_colors[out_src],
              face_roughness=mesh.face_roughness[out_src],
              face_metallic=mesh.face_metallic[out_src],
              face_emissive=mesh.face_emissive[out_src],
              face_opacity=mesh.face_opacity[out_src])
    # bolted-on metadata, not a Mesh constructor param -- see module docstring
    out.lod_source_faces = out_src.astype(np.int64)
    return out


def generate_lods(mesh: Mesh, ratios=DEFAULT_LOD_RATIOS,
                  threshold: int = LOD_FACE_THRESHOLD) -> list[Mesh]:
    """[LOD0, LOD1, ...] at decreasing face counts; LOD0 IS `mesh` itself
    (same object, not a copy). Meshes at or below `threshold` faces return
    `[mesh]` only -- every built-in primitive and any small import never
    pays decimation cost and never carries LOD data, so `Entity.render_mesh`
    (see scene.py) is a no-op for them: byte-identical rendering to before
    this feature existed.
    """
    n = len(mesh.faces)
    if n <= threshold:
        return [mesh]
    lods = [mesh]
    prev = n
    for ratio in ratios:
        target = max(4, int(round(n * ratio)))
        if target >= prev:
            continue  # ratios must be strictly decreasing face counts
        lod = decimate(mesh, target)
        if 0 < len(lod.faces) < prev:
            lods.append(lod)
            prev = len(lod.faces)
    return lods


# ---------------------------------------------------------------------------
# runtime selection
# ---------------------------------------------------------------------------
def _world_bound(entity) -> float:
    """Approximate world-space bounding radius of `entity.mesh` (LOD0) --
    local bound scaled by the largest scale component. Non-uniform scale
    only gets an approximation (a true world AABB would need the rotated
    OBB), which is fine for a coarse LOD-distance threshold."""
    s = entity.transform.scale
    factor = max(abs(s.x), abs(s.y), abs(s.z), 1e-6)
    return float(entity.mesh.bound) * factor


def update_entity_lod(entity, camera) -> None:
    """Update `entity.lod_index` for one entity given the camera position,
    with hysteresis: stepping up (coarser) requires crossing
    `LOD_DISTANCE_FACTORS[i] * bound`, stepping back down requires dropping
    below that same threshold shrunk by `LOD_HYSTERESIS` -- so an entity
    hovering near a boundary doesn't flip every frame (see module docstring
    / gl_renderer.py's geometry cache, which keys off mesh id())."""
    levels = entity.lod_meshes
    if not levels:
        entity.lod_index = 0
        return
    total = 1 + len(levels)
    bound = _world_bound(entity)
    p = entity.transform.position
    cp = camera.position
    dx, dy, dz = cp.x - p.x, cp.y - p.y, cp.z - p.z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    idx = int(min(max(entity.lod_index, 0), total - 1))
    factors = LOD_DISTANCE_FACTORS[: total - 1]
    while idx < total - 1 and dist > bound * factors[idx]:
        idx += 1
    while idx > 0 and dist < bound * factors[idx - 1] * (1.0 - LOD_HYSTERESIS):
        idx -= 1
    entity.lod_index = idx


def shadow_gather_map(entity) -> np.ndarray:
    """(N0,) int array mapping every LOD0 face of `entity.mesh` to the
    nearest face (by local-space centroid distance) of `entity.shadow_mesh()`
    -- `arange(N0)` (identity) when the entity has no coarser LOD (shadow_
    mesh IS mesh). Used by `gpu_geometry._lod_gather` to gather shadow/GI
    results computed at `shadow_mesh` granularity onto LOD0 (see
    scene.py's `Entity.shadow_mesh` / renderer.py's `_directional_base`).

    Cached on the shadow mesh object (`_lod0_gather` attribute, mirroring
    `lod_source_faces`'s bolt-on pattern) since the mapping only depends on
    the fixed LOD0/coarse-mesh pair, never on anything per-frame -- computed
    once per high-poly entity, not once per frame. Chunked brute-force
    nearest-neighbor (no scipy dependency): the coarse side is already a
    few hundred to a couple thousand faces even for a 10k-face import, so
    this is milliseconds despite being O(N0 x Nc).
    """
    shadow = entity.shadow_mesh()
    if shadow is entity.mesh:
        return np.arange(len(entity.mesh.faces))
    cached = getattr(shadow, "_lod0_gather", None)
    if cached is not None:
        return cached
    src = entity.mesh.vertices[entity.mesh.faces].mean(axis=1)  # (N0, 3) local
    dst = shadow.vertices[shadow.faces].mean(axis=1)            # (Nc, 3) local
    out = np.empty(len(src), dtype=np.int64)
    chunk = 1024
    for i in range(0, len(src), chunk):
        d = src[i:i + chunk, None, :] - dst[None, :, :]
        out[i:i + chunk] = np.einsum("ijk,ijk->ij", d, d).argmin(axis=1)
    shadow._lod0_gather = out
    return out


def update_scene_lods(scene, camera) -> None:
    """Update every entity's `lod_index` for this frame. Called once by
    engine/core.py's `run()` before dispatching to whichever renderer
    (CPU/GL/wgpu) is active, so all three backends draw the SAME selected
    LOD per entity this frame -- see renderer.py / gl_renderer.py /
    wgpu_renderer.py's shared `Entity.render_mesh()`."""
    for e in scene.entities:
        if e.mesh is not None:
            update_entity_lod(e, camera)
