"""Polygon meshes (tris + quads) and procedural primitives.

Faces are 3- or 4-sided polygons with counter-clockwise winding viewed from
outside; quads must be planar and convex. Internally faces are stored padded
to 4 indices (triangles repeat their last index), so the whole rendering
pipeline is vectorized over one (M, 4) array; the ray tracer uses a
triangulated copy (`tri_faces`). Per-face lighting shades each quad as one
face — walls and floors get one clean shade per panel instead of a diagonal
tri seam.

UVs come in two parallel forms: `face_uvs` (M, 2), one value per face, is
what every current shading/material consumer reads (unchanged by the corner
form below); `corner_uvs` (M, 4, 2), parallel to `faces` with the same
padded-triangle convention, carries a UV per vertex-corner so a texture can
eventually be interpolated ACROSS a face instead of sampled once per polygon
-- the foundation for per-pixel texturing, not yet consumed by any renderer.
See `Mesh._build_uvs` for exactly how the two are populated/reconciled.
"""
from __future__ import annotations

import math

import numpy as np


class Mesh:
    def __init__(self, vertices, faces, base_color=(200, 200, 200), face_colors=None,
                 face_uvs=None, corner_uvs=None, face_roughness=None, face_metallic=None,
                 face_emissive=None, face_opacity=None):
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
        # face_opacity (M,) -- 1.0 = fully opaque, the backward-compat default.
        # Only meaningful when the mesh's material is blend_mode="translucent"
        # (see MaterialGraph.apply / evaluate_pbr); an opaque material never
        # writes this array away from its all-1.0 default.
        self.face_opacity = (np.asarray(face_opacity, dtype=np.float64)
                             if face_opacity is not None else np.ones(m))
        # explicit per-face UV (e.g. from an FBX LayerElementUV) -- kept
        # separate from the box-projection fallback so re-`_build()`s (winding
        # flips) don't silently discard imported UVs
        self._user_face_uvs = (np.asarray(face_uvs, dtype=np.float64)
                               if face_uvs is not None else None)
        # explicit per-CORNER UV (M, 4, 2), parallel to `faces` including the
        # padded-triangle convention (4th corner repeats the 3rd, same as
        # `faces` itself -- see `_build`). Real per-vertex detail, e.g. from
        # an FBX LayerElementUV, as opposed to `face_uvs`'s single value per
        # face. See `_build_uvs` for how the two combine.
        self._user_corner_uvs = (np.asarray(corner_uvs, dtype=np.float64)
                                 if corner_uvs is not None else None)
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
        self._build_uvs()

    def _build_uvs(self) -> None:
        """Populate `face_uvs` (M, 2) and `corner_uvs` (M, 4, 2), in this
        precedence:

        1. explicit `face_uvs` supplied (a re-`_build()` after `orient_
           outward`, or a legacy single-UV-per-face npz/asset): kept EXACTLY
           as given -- the byte-identical backward-compat contract every
           existing consumer (materials.py's `_evaluate_common` etc.) relies
           on. `corner_uvs` becomes that one value broadcast to all 4
           corners UNLESS real corner data was ALSO supplied -- honest,
           since a lone per-face value carries no actual per-corner detail
           to expose, and it keeps a caller's explicit face value from ever
           being silently replaced by a recomputed one.
        2. explicit `corner_uvs` supplied (real per-vertex detail, e.g. FBX)
           with NO accompanying `face_uvs`: `face_uvs` is DERIVED as the
           mean across the 4 (padded) corners -- genuinely one source of
           truth for a caller that only has corner data.
        3. neither supplied (every built-in primitive): `face_uvs` calls the
           ORIGINAL `box_project_uv` unchanged -- deliberately NOT `corner_
           uvs.mean(axis=1)` -- because float division does not commute with
           averaging: projecting a face's centroid and averaging 4
           independently-projected corners differ by ~1 ULP for non-trivial
           geometry (confirmed on cylinder/cone/icosphere/torus/checker-
           board; only the axis-aligned box's clean fractions happened to
           survive `np.array_equal`). This run's byte-identical gate is
           exactly this path, so bit-for-bit preservation of the pre-
           existing formula wins over always-literally-averaging. `corner_
           uvs` still comes from the same per-axis dominant-normal formula
           (`_UV_AXIS_PAIRS`) via `box_project_uv_corners`, just evaluated
           at each raw corner position instead of the precomputed centroid
           -- one shared formula, two granularities.
        """
        m = len(self.faces)
        if self._user_face_uvs is not None and len(self._user_face_uvs) == m:
            self.face_uvs = self._user_face_uvs                          # (M, 2)
            if self._user_corner_uvs is not None and len(self._user_corner_uvs) == m:
                self.corner_uvs = self._user_corner_uvs                  # (M, 4, 2)
            else:
                self.corner_uvs = np.repeat(self.face_uvs[:, None, :], 4, axis=1)
        elif self._user_corner_uvs is not None and len(self._user_corner_uvs) == m:
            self.corner_uvs = self._user_corner_uvs                      # (M, 4, 2)
            self.face_uvs = self.corner_uvs.mean(axis=1)                 # (M, 2), derived
        else:
            self.face_uvs = box_project_uv(self.vertices, self.faces, self.normals,
                                           self.aabb_min, self.aabb_max)
            self.corner_uvs = box_project_uv_corners(self.vertices, self.faces, self.normals,
                                                      self.aabb_min, self.aabb_max)

    def _face_normals(self) -> np.ndarray:
        tri = self.vertices[self.faces[:, :3]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return n

    def orient_outward(self) -> "Mesh":
        """Flip faces whose normals point toward the origin.

        Only valid for convex meshes centered at the origin. No current
        caller passes explicit corner UVs into a mesh that also calls this
        (the procedural builders below always call it on bare geometry), but
        if one ever did, a flipped face's corner order must be permuted the
        same way `merge_meshes` permutes a negative-determinant part's
        corner UVs -- otherwise `_user_corner_uvs` would stay indexed to the
        PRE-flip vertex order after `_build()` reassigns `self.faces`.
        """
        centroids = self.vertices[self.faces].mean(axis=1)
        flip = np.einsum("ij,ij->i", self.normals, centroids) < 0.0
        if self._user_corner_uvs is not None and len(self._user_corner_uvs) == len(self.faces):
            is_tri = self.faces[:, 2] == self.faces[:, 3]
            quad_rev = self._user_corner_uvs[:, [0, 3, 2, 1], :]
            tri_rev = self._user_corner_uvs[:, [0, 2, 1, 1], :]
            reordered = np.where(is_tri[:, None, None], tri_rev, quad_rev)
            self._user_corner_uvs = np.where(flip[:, None, None], reordered,
                                             self._user_corner_uvs)
        self._polys = [tuple(reversed(f)) if flipped else f
                       for f, flipped in zip(self._polys, flip)]
        self._build()
        return self


def merge_meshes(parts) -> "Mesh | None":
    """Merge posed mesh parts into one composite Mesh.

    `parts` is a list of (Mesh, 4x4 world matrix) pairs, matrix in the same
    row-vector convention as everywhere else in the engine (`v @ M[:3,:3].T
    + M[:3,3]`, see e.g. renderer.py's `_world_face_geometry`) -- pass
    identity for "no additional pose". Returns None for an empty list.

    Lives here (not engine/blueprint.py) because it's a general mesh
    operation like the primitive builders above it, independently testable
    and reusable outside the blueprint feature; engine/blueprint.py stays
    focused on script compiling. Used by BlueprintAsset.instantiate
    (engine/assets.py) to fold a blueprint's posed components into the
    single entity the engine expects -- see that method's docstring for why
    one entity, not N.

    Every part Mesh already carries fully-materialized per-face arrays
    (face_colors/face_uvs/corner_uvs/face_roughness/face_metallic/
    face_emissive/face_opacity) -- Mesh.__init__ always fills in its
    backward-compat defaults at construction time, so there's never a "None"
    array to worry about here; concatenating each part's already-realized
    arrays in the same order as its (offset) faces keeps everything aligned
    by construction. Both face_uvs AND corner_uvs are taken as-is (not
    recomputed) and passed to the composite Mesh together, so each part
    keeps its own box-projection/import UV values instead of ones computed
    over the composite's combined bounding box -- and passing face_uvs
    explicitly (not just corner_uvs) matters: `Mesh._build_uvs` only
    DERIVES face_uvs from corner_uvs when no face_uvs was also supplied
    (see its docstring), and a derived value is ~1-ULP different from the
    original per-part box-projected one, which would break the byte-
    identical single-component-at-identity contract this feature is tested
    against (bp_component_checks.py).

    lod_meshes are intentionally NOT merged -- the composite always gets
    lod_meshes=[] and relies on Entity.shadow_mesh()'s on-demand coarse-
    proxy decimation (see scene.py) for high-poly shadow/GI occlusion.
    Don't "fix" this by merging per-component LODs: the levels wouldn't
    correspond face-for-face across differently posed parts.

    A part matrix with a negative determinant (an odd number of negative
    scale axes) mirrors the geometry, which flips the mathematical winding
    of every face; such a part's faces are reversed here (both the padded-
    quad and padded-triangle cases, distinctly) so the merged mesh's
    normals still point outward everywhere -- Mesh._build() derives normals
    from winding + transformed positions, not from a transformed normal
    array, so this is the only correction negative scale needs.
    """
    if not parts:
        return None
    verts_out, faces_out = [], []
    colors_out, uvs_out, corner_uvs_out = [], [], []
    rough_out, metal_out, emis_out, opac_out = [], [], [], []
    vcount = 0
    for part_mesh, matrix in parts:
        verts_world = part_mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        faces = part_mesh.faces.astype(np.int64) + vcount
        corner_uv = part_mesh.corner_uvs
        if np.linalg.det(matrix[:3, :3]) < 0.0:
            is_tri = faces[:, 2] == faces[:, 3]           # padding convention (see _build)
            quad_rev = faces[:, [0, 3, 2, 1]]              # reverse a real quad's cycle
            tri_rev = faces[:, [0, 2, 1, 1]]                # reverse + re-pad a triangle
            faces = np.where(is_tri[:, None], tri_rev, quad_rev)
            # corner UVs must follow the exact same column permutation so
            # each UV stays attached to the same (now winding-reversed)
            # vertex -- same is_tri split as the face-index permutation above
            quad_uv_rev = corner_uv[:, [0, 3, 2, 1], :]
            tri_uv_rev = corner_uv[:, [0, 2, 1, 1], :]
            corner_uv = np.where(is_tri[:, None, None], tri_uv_rev, quad_uv_rev)
        verts_out.append(verts_world)
        faces_out.append(faces)
        colors_out.append(part_mesh.face_colors)
        uvs_out.append(part_mesh.face_uvs)
        corner_uvs_out.append(corner_uv)
        rough_out.append(part_mesh.face_roughness)
        metal_out.append(part_mesh.face_metallic)
        emis_out.append(part_mesh.face_emissive)
        opac_out.append(part_mesh.face_opacity)
        vcount += len(part_mesh.vertices)

    vertices = np.concatenate(verts_out, axis=0)
    faces = np.concatenate(faces_out, axis=0)
    return Mesh(vertices, [tuple(int(i) for i in f) for f in faces],
               face_colors=np.concatenate(colors_out, axis=0),
               face_uvs=np.concatenate(uvs_out, axis=0),
               corner_uvs=np.concatenate(corner_uvs_out, axis=0),
               face_roughness=np.concatenate(rough_out, axis=0),
               face_metallic=np.concatenate(metal_out, axis=0),
               face_emissive=np.concatenate(emis_out, axis=0),
               face_opacity=np.concatenate(opac_out, axis=0))


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


def box_project_uv_corners(vertices, faces, normals, aabb_min, aabb_max) -> np.ndarray:
    """Per-CORNER form of `box_project_uv`: the identical dominant-axis
    planar projection (same `_UV_AXIS_PAIRS` table, same per-axis affine
    formula), but evaluated at each of a face's 4 (padded) corner positions
    individually instead of the face centroid -- so the UV genuinely VARIES
    across a face, the entire point of this run. Returns (M, 4, 2).

    Deliberately NOT the thing `box_project_uv` is built from (i.e. that
    function is not reimplemented as `.mean(axis=1)` of this one's output):
    float division does not commute with averaging, so mean-then-project and
    project-then-mean differ by ~1 ULP for non-trivial coordinates -- see
    `Mesh._build_uvs`'s docstring for the empirical proof and why `box_
    project_uv`'s bit-for-bit output is a hard backward-compat requirement
    this function must not disturb.
    """
    corner_pos = vertices[faces]  # (M, 4, 3) -- includes the padded-triangle dup
    extent = np.maximum(aabb_max - aabb_min, 1e-9)
    dominant = np.argmax(np.abs(normals), axis=1)  # 0=x, 1=y, 2=z, one per face
    uv = np.zeros((len(faces), 4, 2), dtype=np.float64)
    for axis, (ua, va) in _UV_AXIS_PAIRS.items():
        m = dominant == axis
        uv[m, :, 0] = (corner_pos[m, :, ua] - aabb_min[ua]) / extent[ua]
        uv[m, :, 1] = (corner_pos[m, :, va] - aabb_min[va]) / extent[va]
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
