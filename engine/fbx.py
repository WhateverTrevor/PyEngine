"""Minimal binary FBX importer/exporter (pure Python).

Import: parses the documented Kaydara binary node format (FBX 7.x, including
the 64-bit variant used from version 7.5), extracts every Geometry node's
vertices and polygons, honors the file's up axis and unit scale, and merges
the result into one mesh. Triangles and quads are kept as-is; larger n-gons
are fan-split. Materials, transforms, and animation are ignored — this pulls
geometry only.

`import_fbx()` turns an .fbx file into a self-contained engine asset: the
geometry is saved to assets/models/<name>.npz and a matching asset .json is
written, so the model appears in the content browser like any other asset.

Export: `export_fbx()` writes a spec-correct binary FBX 7.4 file (32-bit node
headers) mirroring the layout this module's own importer reads back —
GlobalSettings (UpAxis=1/Y-up, UnitScaleFactor=1 so vertices round-trip
exactly at *100/‍/100 cm<->m), one Geometry + Model, one Material per unique
face color with a ByPolygon/IndexToDirect LayerElementMaterial, and the
Connections graph the importer's `_material_colors`/`_connections` resolve
through. There is no Blender available in this sandbox to confirm import,
so compatibility rests on the binary structure being spec-correct (correct
header/footer magic, node record shape, standard Geometry/Model/Material/
Connections layout) rather than on an external round-trip; this module's own
`extract_geometry()` re-importing the written file losslessly is the
acid test actually exercised here (see tests/fbx_checks.py).
"""
from __future__ import annotations

import json
import os
import struct
import zlib

import numpy as np

_MAGIC = b"Kaydara FBX Binary  \x00"

_ARRAY_TYPES = {
    b"f": ("<f4", 4), b"d": ("<f8", 8), b"i": ("<i4", 4),
    b"l": ("<i8", 8), b"b": ("<i1", 1),
}
_SCALAR_TYPES = {b"Y": ("<h", 2), b"C": ("<b", 1), b"I": ("<i", 4),
                 b"F": ("<f", 4), b"D": ("<d", 8), b"L": ("<q", 8)}


class FbxNode:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name = name
        self.props = props
        self.children = children

    def find(self, name: str):
        return [c for c in self.children if c.name == name]

    def first(self, name: str):
        for c in self.children:
            if c.name == name:
                return c
        return None


def _read_property(data: bytes, pos: int):
    kind = data[pos:pos + 1]
    pos += 1
    if kind in _SCALAR_TYPES:
        fmt, size = _SCALAR_TYPES[kind]
        return struct.unpack_from(fmt, data, pos)[0], pos + size
    if kind in _ARRAY_TYPES:
        dtype, itemsize = _ARRAY_TYPES[kind]
        length, encoding, comp_len = struct.unpack_from("<III", data, pos)
        pos += 12
        if encoding == 1:
            raw = zlib.decompress(data[pos:pos + comp_len])
            pos += comp_len
        else:
            raw = data[pos:pos + length * itemsize]
            pos += length * itemsize
        return np.frombuffer(raw, dtype=dtype, count=length), pos
    if kind in (b"S", b"R"):
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        raw = data[pos:pos + length]
        return (raw.decode("utf-8", "replace") if kind == b"S" else raw), pos + length
    raise ValueError(f"unknown FBX property type {kind!r}")


def _read_node(data: bytes, pos: int, big: bool):
    if big:
        end, num_props, _prop_len = struct.unpack_from("<QQQ", data, pos)
        pos += 24
    else:
        end, num_props, _prop_len = struct.unpack_from("<III", data, pos)
        pos += 12
    name_len = data[pos]
    pos += 1
    if end == 0 and num_props == 0 and name_len == 0:
        return None, pos  # null sentinel
    name = data[pos:pos + name_len].decode("ascii", "replace")
    pos += name_len
    props = []
    for _ in range(num_props):
        value, pos = _read_property(data, pos)
        props.append(value)
    children = []
    while pos < end:
        child, pos = _read_node(data, pos, big)
        if child is None:
            break
        children.append(child)
    return FbxNode(name, props, children), max(pos, end)


def parse_fbx(path: str) -> tuple[list[FbxNode], int]:
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(_MAGIC):
        if data[:1] == b";" or b"FBXHeaderExtension" in data[:2048]:
            raise ValueError("ASCII FBX is not supported — re-export as binary FBX")
        raise ValueError("not an FBX file")
    version = struct.unpack_from("<I", data, 23)[0]
    big = version >= 7500  # 7.5+ switched node headers to 64-bit
    pos = 27
    roots = []
    while pos < len(data):
        node, pos = _read_node(data, pos, big)
        if node is None:
            break
        roots.append(node)
    return roots, version


def _global_settings(roots):
    up_axis, unit_scale = 1, 1.0
    for r in roots:
        if r.name != "GlobalSettings":
            continue
        p70 = r.first("Properties70")
        if p70 is None:
            continue
        for p in p70.find("P"):
            if not p.props:
                continue
            if p.props[0] == "UpAxis" and len(p.props) >= 5:
                up_axis = int(p.props[4])
            elif p.props[0] == "UnitScaleFactor" and len(p.props) >= 5:
                unit_scale = float(p.props[4])
    return up_axis, unit_scale


_DEFAULT_DIFFUSE = (0.66, 0.67, 0.69)


def _material_colors(roots) -> dict[int, tuple]:
    """Material node id -> linear diffuse color (0..1)."""
    colors = {}
    for root in roots:
        if root.name != "Objects":
            continue
        for mat in root.find("Material"):
            if not mat.props:
                continue
            mat_id = int(mat.props[0])
            color = _DEFAULT_DIFFUSE
            p70 = mat.first("Properties70")
            if p70 is not None:
                for p in p70.find("P"):
                    if p.props and p.props[0] in ("DiffuseColor", "Diffuse") \
                            and len(p.props) >= 7:
                        color = (float(p.props[4]), float(p.props[5]),
                                 float(p.props[6]))
                        break
            colors[mat_id] = color
    return colors


def _connections(roots) -> list[tuple[int, int]]:
    """(child_id, parent_id) object-object connections."""
    out = []
    for root in roots:
        if root.name != "Connections":
            continue
        for c in root.find("C"):
            if len(c.props) >= 3 and c.props[0] == "OO":
                out.append((int(c.props[1]), int(c.props[2])))
    return out


def _geometry_materials(geo) -> tuple[list[int] | None, str]:
    """Per-polygon material indices for a Geometry node, or None."""
    layer = geo.first("LayerElementMaterial")
    if layer is None:
        return None, "AllSame"
    mapping = "AllSame"
    mnode = layer.first("MappingInformationType")
    if mnode is not None and mnode.props:
        mapping = str(mnode.props[0])
    arr = layer.first("Materials")
    if arr is None or not arr.props or not len(arr.props[0]):
        return None, mapping
    return [int(v) for v in np.asarray(arr.props[0])], mapping


def extract_geometry(path: str):
    """All geometry in the file, merged.

    Returns (vertices (N,3), polygon index tuples, per-face colors (M,3) 0..1).
    Face colors come from each geometry's material layer, resolved through the
    FBX connection graph (geometry -> model <- materials, in connection order).
    """
    roots, _version = parse_fbx(path)
    up_axis, unit_scale = _global_settings(roots)
    mat_colors = _material_colors(roots)
    connections = _connections(roots)

    # geometry id -> model id, model id -> [material ids in connection order]
    geo_ids = set()
    model_ids = set()
    for root in roots:
        if root.name != "Objects":
            continue
        for geo in root.find("Geometry"):
            if geo.props:
                geo_ids.add(int(geo.props[0]))
        for mdl in root.find("Model"):
            if mdl.props:
                model_ids.add(int(mdl.props[0]))
    geo_to_model: dict[int, int] = {}
    model_mats: dict[int, list[int]] = {}
    for child, parent in connections:
        if child in geo_ids and parent in model_ids:
            geo_to_model[child] = parent
        elif child in mat_colors and parent in model_ids:
            model_mats.setdefault(parent, []).append(child)

    all_verts, all_faces, all_colors = [], [], []
    offset = 0
    for root in roots:
        if root.name != "Objects":
            continue
        for geo in root.find("Geometry"):
            vnode, inode = geo.first("Vertices"), geo.first("PolygonVertexIndex")
            if vnode is None or inode is None or not len(vnode.props):
                continue
            verts = np.asarray(vnode.props[0], dtype=np.float64).reshape(-1, 3)
            idx = np.asarray(inode.props[0], dtype=np.int64)

            gid = int(geo.props[0]) if geo.props else -1
            palette = [mat_colors.get(mid, _DEFAULT_DIFFUSE)
                       for mid in model_mats.get(geo_to_model.get(gid, -1), [])]
            mat_idx, _mapping = _geometry_materials(geo)

            def poly_color(poly_i: int) -> tuple:
                if not palette:
                    return _DEFAULT_DIFFUSE
                if mat_idx is None:
                    return palette[0]
                i = mat_idx[poly_i] if poly_i < len(mat_idx) else mat_idx[-1]
                return palette[i] if 0 <= i < len(palette) else palette[0]

            poly: list[int] = []
            poly_i = 0
            for raw in idx:
                if raw < 0:
                    poly.append(int(~raw))
                    color = poly_color(poly_i)
                    if len(poly) == 3 or len(poly) == 4:
                        all_faces.append(tuple(i + offset for i in poly))
                        all_colors.append(color)
                    elif len(poly) > 4:  # fan-split n-gons
                        for k in range(1, len(poly) - 1):
                            all_faces.append((poly[0] + offset, poly[k] + offset,
                                              poly[k + 1] + offset))
                            all_colors.append(color)
                    poly = []
                    poly_i += 1
                else:
                    poly.append(int(raw))
            all_verts.append(verts)
            offset += len(verts)

    if not all_verts:
        raise ValueError("no polygon geometry found in FBX")
    vertices = np.concatenate(all_verts)
    vertices *= unit_scale * 0.01  # FBX native units are centimeters
    if up_axis == 2:  # Z-up -> engine Y-up
        vertices = vertices[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
    return vertices, all_faces, np.asarray(all_colors, dtype=np.float64)


def import_fbx(path: str, assets_dir: str, color=(168, 170, 176),
               max_bound: float = 4.0) -> str:
    """Convert an .fbx into a content-browser asset. Returns the asset name."""
    vertices, faces, face_colors = extract_geometry(path)

    # center on the ground and normalize outlandish scales
    vertices = vertices - vertices.mean(axis=0)
    vertices[:, 1] -= vertices[:, 1].min()
    bound = float(np.abs(vertices).max())
    if bound > max_bound and bound > 0:
        vertices *= max_bound / bound

    stem = os.path.splitext(os.path.basename(path))[0]
    name = stem.replace("_", " ").strip().title() or "Imported Model"
    models_dir = os.path.join(assets_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    padded = [f if len(f) == 4 else (f[0], f[1], f[2], f[2]) for f in faces]
    np.savez_compressed(os.path.join(models_dir, f"{stem}.npz"),
                        vertices=vertices.astype(np.float32),
                        faces=np.asarray(padded, dtype=np.int32),
                        face_colors=np.clip(face_colors * 255.0, 0, 255
                                            ).astype(np.uint8))

    asset = {
        "name": name,
        "category": "models",
        "mesh": {"primitive": "model", "path": f"models/{stem}.npz",
                 "color": list(color)},
    }
    with open(os.path.join(assets_dir, f"{stem}.json"), "w", encoding="utf-8") as fh:
        json.dump(asset, fh, indent=2)
    return name


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

_FBX_VERSION = 7400  # last version with 32-bit node headers (< 7500)

# Community-documented binary-FBX footer constants (Autodesk FBX SDK output).
# Not independently verifiable here (no Blender in this sandbox) — included
# so the file's tail matches the spec shape a real FBX writer produces.
_FOOTER_ID = bytes([0xfa, 0xbc, 0xab, 0x09, 0xd0, 0xc8, 0xd4, 0x66,
                    0xb1, 0x76, 0xfb, 0x83, 0x1c, 0xf7, 0x26, 0x7e])
_FOOTER_EXT = bytes([0xf8, 0x5a, 0x8c, 0x6a, 0xde, 0xf5, 0xd9, 0x7e,
                     0xec, 0xe9, 0x0c, 0xe3, 0x75, 0x8f, 0x29, 0x0b])


def _encode_prop(v) -> bytes:
    if isinstance(v, str):
        raw = v.encode("utf-8")
        return b"S" + struct.pack("<I", len(raw)) + raw
    if isinstance(v, bool):
        return b"C" + struct.pack("<b", 1 if v else 0)
    if isinstance(v, int):
        return b"L" + struct.pack("<q", v)
    if isinstance(v, float):
        return b"D" + struct.pack("<d", v)
    if isinstance(v, np.ndarray):
        if v.dtype == np.float64:
            code, dtype = b"d", "<f8"
        elif v.dtype == np.float32:
            code, dtype = b"f", "<f4"
        elif v.dtype == np.int64:
            code, dtype = b"l", "<i8"
        else:
            v = v.astype(np.int32)
            code, dtype = b"i", "<i4"
        raw = np.ascontiguousarray(v, dtype=dtype).tobytes()
        return code + struct.pack("<III", len(v), 0, len(raw)) + raw
    raise TypeError(f"unsupported FBX property type for export: {type(v)!r}")


def _encode_node(node: tuple, start: int) -> bytes:
    """(name, props, children) -> binary node bytes, 32-bit headers.

    `start` is this node's absolute file offset; end offsets are absolute,
    per the format, so nested nodes need the running position threaded
    through — mirrors `_read_node`'s layout in reverse.
    """
    name, props, children = node
    name_b = name.encode("ascii")
    prop_data = b"".join(_encode_prop(p) for p in props)
    pos = start + 13 + len(name_b) + len(prop_data)
    child_data = b""
    if children:
        for child in children:
            cb = _encode_node(child, pos)
            child_data += cb
            pos += len(cb)
        child_data += b"\x00" * 13  # null sentinel terminates the child list
        pos += 13
    header = struct.pack("<IIIB", pos, len(props), len(prop_data), len(name_b))
    return header + name_b + prop_data + child_data


def _unique_palette(face_colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(palette (P,3) 0..255, per-face index into palette)."""
    rounded = np.round(face_colors).astype(np.int64)
    palette_list: list[tuple[int, int, int]] = []
    lookup: dict[tuple[int, int, int], int] = {}
    indices = np.empty(len(rounded), dtype=np.int32)
    for i, row in enumerate(rounded):
        key = (int(row[0]), int(row[1]), int(row[2]))
        idx = lookup.get(key)
        if idx is None:
            idx = len(palette_list)
            lookup[key] = idx
            palette_list.append(key)
        indices[i] = idx
    return np.asarray(palette_list, dtype=np.float64), indices


def export_fbx(mesh, path: str, name: str = "Mesh") -> None:
    """Write `mesh` (an `engine.mesh.Mesh`) to `path` as binary FBX 7.4.

    Geometry is stored in centimeters with UpAxis=Y (matching the engine),
    so `extract_geometry()` re-imports it with vertices unchanged (see the
    UnitScaleFactor/UpAxis round-trip note in this module's docstring).
    """
    verts_cm = (mesh.vertices * 100.0).reshape(-1)
    palette, face_idx = _unique_palette(mesh.face_colors)

    pvi: list[int] = []
    normals_flat: list[float] = []
    for face_row, normal in zip(mesh.faces, mesh.normals):
        poly = (face_row[:3] if face_row[3] == face_row[2] else face_row)
        poly = [int(i) for i in poly]
        for i in poly[:-1]:
            pvi.append(i)
        pvi.append(~poly[-1])
        normals_flat.extend(float(c) for c in normal)

    GID, MODEL_ID = 1000, 2000
    mat_ids = [3000 + i for i in range(len(palette))]

    header_ext = ("FBXHeaderExtension", [], [
        ("FBXHeaderVersion", [1003], []),
        ("FBXVersion", [_FBX_VERSION], []),
        ("Creator", ["PyEngine FBX Exporter"], []),
    ])
    global_settings = ("GlobalSettings", [], [
        ("Version", [1000], []),
        ("Properties70", [], [
            ("P", ["UpAxis", "int", "Integer", "", 1], []),
            ("P", ["UnitScaleFactor", "double", "Number", "", 1.0], []),
        ]),
    ])
    definitions = ("Definitions", [], [
        ("Version", [100], []),
        ("Count", [2 + len(palette)], []),
    ])

    material_layer = None
    if len(palette):
        material_layer = ("LayerElementMaterial", [0], [
            ("Version", [101], []),
            ("Name", [""], []),
            ("MappingInformationType", ["ByPolygon"], []),
            ("ReferenceInformationType", ["IndexToDirect"], []),
            ("Materials", [face_idx.astype(np.int32)], []),
        ])
    normal_layer = ("LayerElementNormal", [0], [
        ("Version", [101], []),
        ("Name", [""], []),
        ("MappingInformationType", ["ByPolygon"], []),
        ("ReferenceInformationType", ["Direct"], []),
        ("Normals", [np.asarray(normals_flat, dtype=np.float64)], []),
    ])
    layer_elements = [("LayerElement", [], [
        ("Type", ["LayerElementNormal"], []),
        ("TypedIndex", [0], []),
    ])]
    if material_layer is not None:
        layer_elements.append(("LayerElement", [], [
            ("Type", ["LayerElementMaterial"], []),
            ("TypedIndex", [0], []),
        ]))
    layer = ("Layer", [0], [("Version", [100], [])] + layer_elements)

    geometry_children = [
        ("GeometryVersion", [124], []),
        ("Vertices", [verts_cm], []),
        ("PolygonVertexIndex", [np.asarray(pvi, dtype=np.int32)], []),
        normal_layer,
    ]
    if material_layer is not None:
        geometry_children.append(material_layer)
    geometry_children.append(layer)
    geometry = ("Geometry", [GID, f"Geometry::{name}", "Mesh"], geometry_children)

    model = ("Model", [MODEL_ID, f"Model::{name}", "Mesh"], [
        ("Version", [232], []),
        ("Shading", [True], []),
        ("Culling", ["CullingOff"], []),
    ])

    materials = []
    for mat_id, color in zip(mat_ids, palette):
        r, g, b = (color / 255.0).tolist()
        materials.append(("Material", [mat_id, f"Material::{name}_{mat_id}", ""], [
            ("Version", [102], []),
            ("ShadingModel", ["Lambert"], []),
            ("MultiLayer", [0], []),
            ("Properties70", [], [
                ("P", ["DiffuseColor", "Color", "", "A", r, g, b], []),
            ]),
        ]))

    objects = ("Objects", [], [geometry, model] + materials)
    connections = ("Connections", [], [
        ("C", ["OO", GID, MODEL_ID], []),
    ] + [("C", ["OO", mat_id, MODEL_ID], []) for mat_id in mat_ids])

    header = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", _FBX_VERSION)
    pos = len(header)
    body = b""
    for node in (header_ext, global_settings, definitions, objects, connections):
        chunk = _encode_node(node, pos)
        body += chunk
        pos += len(chunk)
    body += b"\x00" * 13  # top-level null sentinel; our parser stops here

    footer = _FOOTER_ID
    pad = (-(len(footer)) - 4) % 16  # pad so [footer_id|pad|version] lands on 16
    footer += b"\x00" * pad
    footer += struct.pack("<I", _FBX_VERSION)
    footer += b"\x00" * 120
    footer += _FOOTER_EXT
    footer += b"\x00" * 4

    with open(path, "wb") as fh:
        fh.write(header + body + footer)


def export_asset_fbx(asset_def, path: str) -> None:
    """Export a content-browser asset's mesh to `path` as binary FBX.

    Raises ValueError if the asset has no mesh (lights-only, fog volumes).
    Procedural primitives (cube, cylinder, ...) are instantiated first so
    their actual vertices/faces are written, same as imported .npz models.
    """
    if not has_mesh(asset_def):
        raise ValueError(f"asset '{asset_def.name}' has no mesh to export")
    entity = asset_def.instantiate()
    export_fbx(entity.mesh, path, name=asset_def.name.replace(" ", "_"))


def has_mesh(asset_def) -> bool:
    """Whether an AssetDef's definition contains mesh geometry to export."""
    return "mesh" in asset_def.data
