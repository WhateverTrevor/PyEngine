"""Minimal binary FBX importer (pure Python).

Parses the documented Kaydara binary node format (FBX 7.x, including the
64-bit variant used from version 7.5), extracts every Geometry node's
vertices and polygons, honors the file's up axis and unit scale, and merges
the result into one mesh. Triangles and quads are kept as-is; larger n-gons
are fan-split. Materials, transforms, and animation are ignored — this pulls
geometry only.

`import_fbx()` turns an .fbx file into a self-contained engine asset: the
geometry is saved to assets/models/<name>.npz and a matching asset .json is
written, so the model appears in the content browser like any other asset.
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


def extract_geometry(path: str) -> tuple[np.ndarray, list[tuple]]:
    """All geometry in the file, merged: (vertices (N,3), polygon index tuples)."""
    roots, _version = parse_fbx(path)
    up_axis, unit_scale = _global_settings(roots)

    all_verts, all_faces = [], []
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
            poly: list[int] = []
            for raw in idx:
                if raw < 0:
                    poly.append(int(~raw))
                    if len(poly) == 3 or len(poly) == 4:
                        all_faces.append(tuple(i + offset for i in poly))
                    elif len(poly) > 4:  # fan-split n-gons
                        for k in range(1, len(poly) - 1):
                            all_faces.append((poly[0] + offset, poly[k] + offset,
                                              poly[k + 1] + offset))
                    poly = []
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
    return vertices, all_faces


def import_fbx(path: str, assets_dir: str, color=(168, 170, 176),
               max_bound: float = 4.0) -> str:
    """Convert an .fbx into a content-browser asset. Returns the asset name."""
    vertices, faces = extract_geometry(path)

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
                        faces=np.asarray(padded, dtype=np.int32))

    asset = {
        "name": name,
        "category": "models",
        "mesh": {"primitive": "model", "path": f"models/{stem}.npz",
                 "color": list(color)},
    }
    with open(os.path.join(assets_dir, f"{stem}.json"), "w", encoding="utf-8") as fh:
        json.dump(asset, fh, indent=2)
    return name
