"""Shared, dependency-free geometry helpers for the GPU renderer backends.

Both `gl_renderer.py` (moderngl) and `wgpu_renderer.py` (wgpu-py) need the
exact same non-indexed "vertex soup" construction and world-space face
data -- this module is the one source of truth for that, kept free of any
GPU-library import (no `moderngl`, no `wgpu`) so that requesting one GPU
backend never fails because the *other* backend's optional dependency is
missing. Before this was split out, `wgpu_renderer.py` imported these
functions straight from `gl_renderer.py`, which does `import moderngl` at
module load -- meaning `api="dx12"` would fail with a moderngl import error
on a machine that has wgpu but not moderngl installed. See engine/core.py's
`_init_display` fallback chain for how backend failures are handled.
"""
from __future__ import annotations

import numpy as np


def _build_geometry(mesh):
    """Non-indexed (T*3, ...) vertex soup in LOCAL space + per-triangle face id."""
    faces = mesh.faces  # (M, 4) int32, padded per Mesh._build
    is_tri = (faces[:, 3] == faces[:, 2]) | (faces[:, 3] == faces[:, 0])
    tri1 = faces[:, (0, 1, 2)]
    face_id1 = np.arange(len(faces), dtype=np.int64)
    tri2 = faces[~is_tri][:, (0, 2, 3)]
    face_id2 = face_id1[~is_tri]
    tri_idx = np.concatenate([tri1, tri2], axis=0)
    face_id_tri = np.concatenate([face_id1, face_id2], axis=0)

    pos = mesh.vertices[tri_idx].reshape(-1, 3).astype(np.float32)
    nrm = np.repeat(mesh.normals[face_id_tri].astype(np.float32), 3, axis=0)
    fid = np.repeat(face_id_tri.astype(np.float32), 3)
    return pos, nrm, fid, face_id_tri, int(len(faces))


def _build_color(mesh, face_id_tri) -> np.ndarray:
    return np.repeat((mesh.face_colors[face_id_tri] / 255.0).astype(np.float32), 3, axis=0)


def _build_pbr(mesh, face_id_tri):
    """Per-vertex (roughness, metallic) and emissive (0..1), same repeat-per-
    triangle-vertex flow as `_build_color` -- these are per-face data baked
    by `Material.evaluate_pbr` (see engine/materials.py), so a flat repeat
    across a face's vertices is exact, not an approximation."""
    rm = np.stack([mesh.face_roughness[face_id_tri],
                   mesh.face_metallic[face_id_tri]], axis=1).astype(np.float32)
    emissive = (mesh.face_emissive[face_id_tri] / 255.0).astype(np.float32)
    return (np.repeat(rm, 3, axis=0), np.repeat(emissive, 3, axis=0))


def _build_opacity(mesh, face_id_tri) -> np.ndarray:
    """Per-vertex face opacity (0..1), same per-face-vertex repeat flow as
    `_build_color`/`_build_pbr` -- baked by `Material.apply()` into
    `mesh.face_opacity` (all-1.0 for opaque materials, see engine/mesh.py).
    Shape (T*3, 1) so it can bind to a 1-float vertex attribute directly."""
    op = mesh.face_opacity[face_id_tri].astype(np.float32)
    return np.repeat(op, 3)[:, None]


def _lod_gather(entity, render_mesh, values: np.ndarray) -> np.ndarray:
    """Map a per-LOD0-face array (shape (entity.mesh face count, ...)) onto
    `render_mesh`'s own faces via its `lod_source_faces` map (see
    engine/lod.py) -- identity when `render_mesh` IS `entity.mesh` (no LOD
    active, or the mesh has no LOD data at all -- every built-in asset).
    Used for shadow/GI values, which are always computed at LOD0 face
    granularity (ray-traced occlusion/bounce must never depend on which LOD
    the camera happens to be rasterizing -- see raytrace.py's world-version
    cache) then gathered onto whichever LOD is actually being drawn.
    """
    if render_mesh is entity.mesh:
        return values
    src = getattr(render_mesh, "lod_source_faces", None)
    return values if src is None else values[src]


def _entity_world_faces(entity, mesh=None):
    """World-space per-face centroids + normals, mirroring _entity_geometry.
    `mesh` defaults to `entity.mesh` (LOD0) -- pass a specific mesh (e.g. an
    entity's currently-selected render LOD) to get that mesh's own world
    geometry instead."""
    mesh = mesh if mesh is not None else entity.mesh
    model = entity.transform.matrix()
    verts_world = mesh.vertices @ model[:3, :3].T + model[:3, 3]
    try:
        normal_mat = np.linalg.inv(model[:3, :3]).T
    except np.linalg.LinAlgError:
        normal_mat = np.eye(3)
    normals_world = mesh.normals @ normal_mat.T
    normals_world /= np.maximum(np.linalg.norm(normals_world, axis=1, keepdims=True), 1e-12)
    centroids_world = verts_world[mesh.faces].mean(axis=1)
    return centroids_world, normals_world


def _scene_environment(scene):
    for e in scene.entities:
        if e.environment is not None and e.visible:
            return e.environment
    return None
