"""Polygon meshes (tris + quads) and procedural primitives.

Faces are 3- or 4-sided polygons with counter-clockwise winding viewed from
outside; quads must be planar and convex. Internally faces are stored padded
to 4 indices (triangles repeat their last index), so the whole rendering
pipeline is vectorized over one (M, 4) array; the ray tracer uses a
triangulated copy (`tri_faces`). Per-face lighting shades each quad as one
face — walls and floors get one clean shade per panel instead of a diagonal
tri seam.
"""
from __future__ import annotations

import math

import numpy as np


class Mesh:
    def __init__(self, vertices, faces, base_color=(200, 200, 200), face_colors=None,
                 face_uvs=None, face_roughness=None, face_metallic=None,
                 face_emissive=None):
        self.vertices = np.asarray(vertices, dtype=np.float64)   # (N, 3)
        self._polys = [tuple(int(i) for i in f) for f in faces]
        if face_colors is None:
            face_colors = np.tile(np.asarray(base_color, dtype=np.float64),
                                  (len(self._polys), 1))
        self.face_colors = np.asarray(face_colors, dtype=np.float64)  # (M, 3)
        # PBR per-face params, parallel to face_colors. Defaults (roughness=1,
        # metallic=0, emissive=0) are the backward-compat contract: a mesh
        # with default params must shade numerically identically to the old
        # lambert-only pipeline (see renderer.py's deferred pass).
        m = len(self._polys)
        self.face_roughness = (np.asarray(face_roughness, dtype=np.float64)
                               if face_roughness is not None else np.ones(m))
        self.face_metallic = (np.asarray(face_metallic, dtype=np.float64)
                              if face_metallic is not None else np.zeros(m))
        self.face_emissive = (np.asarray(face_emissive, dtype=np.float64)
                              if face_emissive is not None else np.zeros((m, 3)))
        # explicit per-face UV (e.g. from an FBX LayerElementUV) -- kept
        # separate from the box-projection fallback so re-`_build()`s (winding
        # flips) don't silently discard imported UVs
        self._user_face_uvs = (np.asarray(face_uvs, dtype=np.float64)
                               if face_uvs is not None else None)
        self._build()

    def _build(self) -> None:
        padded, tris = [], []
        for f in self._polys:
            if len(f) == 4 and f[3] != f[2] and f[3] != f[0]:
                padded.append(f)
                tris.append((f[0], f[1], f[2]))
                tris.append((f[0], f[2], f[3]))
            else:
                t = f[:3]
                padded.append((t[0], t[1], t[2], t[2]))
                tris.append(t)
        self.faces = np.asarray(padded, dtype=np.int32)      # (M, 4), tris padded
        self.tri_faces = np.asarray(tris, dtype=np.int32)    # (T, 3) for ray tracing
        self.normals = self._face_normals()
        self.aabb_min = self.vertices.min(axis=0)
        self.aabb_max = self.vertices.max(axis=0)
        self.bound = float(np.linalg.norm(self.vertices, axis=1).max())
        if self._user_face_uvs is not None and len(self._user_face_uvs) == len(self.faces):
            self.face_uvs = self._user_face_uvs               # (M, 2)
        else:
            self.face_uvs = box_project_uv(self.vertices, self.faces, self.normals,
                                           self.aabb_min, self.aabb_max)

    def _face_normals(self) -> np.ndarray:
        tri = self.vertices[self.faces[:, :3]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return n

    def orient_outward(self) -> "Mesh":
        """Flip faces whose normals point toward the origin.

        Only valid for convex meshes centered at the origin.
        """
        centroids = self.vertices[self.faces].mean(axis=1)
        flip = np.einsum("ij,ij->i", self.normals, centroids) < 0.0
        self._polys = [tuple(reversed(f)) if flipped else f
                       for f, flipped in zip(self._polys, flip)]
        self._build()
        return self


_UV_AXIS_PAIRS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}  # dominant normal axis -> (u axis, v axis)


def box_project_uv(vertices, faces, normals, aabb_min, aabb_max) -> np.ndarray:
    """Trivial auto-unwrap: per-face planar UV from the face's dominant
    normal axis, scaled 0..1 across the mesh's own bounding box on the two
    non-dominant axes. Default UV for any mesh without real (imported) UVs.
    """
    centroids = vertices[faces].mean(axis=1)
    extent = np.maximum(aabb_max - aabb_min, 1e-9)
    dominant = np.argmax(np.abs(normals), axis=1)  # 0=x, 1=y, 2=z
    uv = np.zeros((len(centroids), 2), dtype=np.float64)
    for axis, (ua, va) in _UV_AXIS_PAIRS.items():
        m = dominant == axis
        uv[m, 0] = (centroids[m, ua] - aabb_min[ua]) / extent[ua]
        uv[m, 1] = (centroids[m, va] - aabb_min[va]) / extent[va]
    return uv


def box(width: float = 1.0, height: float = 1.0, depth: float = 1.0,
        color=(200, 200, 200)) -> Mesh:
    """Axis-aligned box centered at the origin — 6 quad faces."""
    hx, hy, hz = width * 0.5, height * 0.5, depth * 0.5
    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    f = [(4, 5, 6, 7),   # +z
         (1, 0, 3, 2),   # -z
         (0, 4, 7, 3),   # -x
         (5, 1, 2, 6),   # +x
         (7, 6, 2, 3),   # +y
         (0, 1, 5, 4)]   # -y
    return Mesh(v, f, base_color=color).orient_outward()


def cube(size: float = 1.0, color=(200, 200, 200)) -> Mesh:
    return box(size, size, size, color)


def cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 12,
             color=(200, 200, 200)) -> Mesh:
    """Upright cylinder centered at the origin: quad sides, triangle-fan caps."""
    hy = height * 0.5
    verts = []
    for y in (-hy, hy):
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            verts.append((math.cos(a) * radius, y, math.sin(a) * radius))
    bottom_center = len(verts)
    verts.append((0.0, -hy, 0.0))
    top_center = len(verts)
    verts.append((0.0, hy, 0.0))

    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        b0, b1 = i, j
        t0, t1 = segments + i, segments + j
        faces.append((b0, b1, t1, t0))                            # side quad
        faces += [(bottom_center, b1, b0), (top_center, t0, t1)]  # cap tris
    return Mesh(verts, faces, base_color=color).orient_outward()


def cone(radius: float = 0.5, length: float = 1.0, segments: int = 12,
         color=(200, 200, 200)) -> Mesh:
    """Cone along Z: apex at +length/2, open base ring at -length/2.

    Built as a spotlight housing — the wide end faces -Z, the direction
    spotlights aim, so rotating the entity aims the housing and beam together.
    """
    hz = length * 0.5
    verts = []
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        verts.append((math.cos(a) * radius, math.sin(a) * radius, -hz))
    apex = len(verts)
    verts.append((0.0, 0.0, hz))
    base_center = len(verts)
    verts.append((0.0, 0.0, -hz))

    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [(i, j, apex), (base_center, j, i)]
    return Mesh(verts, faces, base_color=color).orient_outward()


def icosphere(radius: float = 1.0, subdivisions: int = 2, color=(200, 200, 200)) -> Mesh:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    v = [(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
         (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
         (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, dtype=np.float64) for p in v]
    faces = list(f)

    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in midpoint_cache:
            verts.append((verts[a] + verts[b]) * 0.5)
            midpoint_cache[key] = len(verts) - 1
        return midpoint_cache[key]

    for _ in range(subdivisions):
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    arr = np.array(verts)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True) * radius
    return Mesh(arr, faces, base_color=color).orient_outward()


def torus(ring_radius: float = 1.0, tube_radius: float = 0.35,
          ring_segments: int = 24, tube_segments: int = 14,
          color=(200, 200, 200)) -> Mesh:
    verts = []
    for i in range(ring_segments):
        u = 2.0 * math.pi * i / ring_segments
        cu, su = math.cos(u), math.sin(u)
        for j in range(tube_segments):
            v = 2.0 * math.pi * j / tube_segments
            r = ring_radius + tube_radius * math.cos(v)
            verts.append((r * cu, tube_radius * math.sin(v), r * su))

    faces = []
    for i in range(ring_segments):
        for j in range(tube_segments):
            a = i * tube_segments + j
            b = ((i + 1) % ring_segments) * tube_segments + j
            c = ((i + 1) % ring_segments) * tube_segments + (j + 1) % tube_segments
            d = i * tube_segments + (j + 1) % tube_segments
            faces.append((a, b, c, d))

    mesh = Mesh(verts, faces, base_color=color)
    # Fix winding once using the analytic outward direction of face 0: from the
    # tube's center circle toward the face centroid. Topology is uniform, so one
    # test decides the whole mesh.
    centroid = mesh.vertices[mesh.faces[0]].mean(axis=0)
    ring_point = np.array([centroid[0], 0.0, centroid[2]])
    ring_point *= ring_radius / max(np.linalg.norm(ring_point), 1e-12)
    if float(np.dot(mesh.normals[0], centroid - ring_point)) < 0.0:
        mesh._polys = [tuple(reversed(f)) for f in mesh._polys]
        mesh._build()
    return mesh


def checkerboard(squares: int = 24, square_size: float = 2.0,
                 color_a=(95, 98, 104), color_b=(60, 62, 68)) -> Mesh:
    """Flat ground on y=0, centered at the origin — one quad per square."""
    half = squares * square_size * 0.5
    verts, faces, colors = [], [], []
    for i in range(squares):
        for j in range(squares):
            x0, z0 = i * square_size - half, j * square_size - half
            x1, z1 = x0 + square_size, z0 + square_size
            base = len(verts)
            verts += [(x0, 0.0, z0), (x1, 0.0, z0), (x1, 0.0, z1), (x0, 0.0, z1)]
            faces.append((base, base + 3, base + 2, base + 1))  # +y winding
            colors.append(color_a if (i + j) % 2 == 0 else color_b)
    return Mesh(verts, faces, face_colors=colors)
