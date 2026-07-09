"""Node-based materials.

A MaterialGraph is a small dataflow graph — Color/Position/Normal/Noise/
Checker/Gradient sources flowing through Mix/Multiply into the Output node —
evaluated per face with numpy (all values are (M, 3) float in 0..1) and baked
onto a mesh's per-face colors. Graphs serialize to plain dicts, so they live
inside scene files, and evaluation is deterministic.
"""
from __future__ import annotations

import numpy as np

# node type -> (input port names, {param: default})
NODE_DEFS = {
    "output":   ((("color",)), {}),
    "color":    ((), {"r": 0.8, "g": 0.8, "b": 0.8}),
    "position": ((), {}),
    "normal":   ((), {}),
    "checker":  (("a", "b"), {"scale": 1.0}),
    "noise":    ((), {"scale": 1.0, "seed": 0.0}),
    "gradient": (("a", "b"), {"axis": 1.0}),
    "mix":      (("a", "b", "fac"), {}),
    "multiply": (("a", "b"), {}),
}

# slider ranges for the editor UI
PARAM_RANGES = {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0),
                "scale": (0.05, 8.0), "seed": (0.0, 100.0), "axis": (0.0, 2.0)}


def _hash_noise(cells: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random 0..1 per integer cell (M, 3) -> (M,)."""
    h = (cells[:, 0] * 374761393 + cells[:, 1] * 668265263
         + cells[:, 2] * 1442695041) & 0x7FFFFFFFFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0x7FFFFFFFFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFF) / float(0xFFFFF)


class MaterialGraph:
    def __init__(self):
        self.nodes: dict[int, dict] = {}   # id -> {"type", "pos": [x, y], "params": {}}
        self.links: list[list] = []        # [src_id, dst_id, input_name]
        self.next_id = 1
        self.add("output", (560.0, 180.0))

    # ---- editing ----
    def add(self, node_type: str, pos) -> int:
        inputs, params = NODE_DEFS[node_type]
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = {"type": node_type, "pos": [float(pos[0]), float(pos[1])],
                           "params": dict(params)}
        return nid

    def remove(self, nid: int) -> None:
        if nid in self.nodes and self.nodes[nid]["type"] != "output":
            del self.nodes[nid]
            self.links = [l for l in self.links if l[0] != nid and l[1] != nid]

    def upstream(self, nid: int) -> set[int]:
        seen: set[int] = set()
        stack = [nid]
        while stack:
            cur = stack.pop()
            for src, dst, _name in self.links:
                if dst == cur and src not in seen:
                    seen.add(src)
                    stack.append(src)
        return seen

    def connect(self, src: int, dst: int, input_name: str) -> bool:
        if src == dst or dst in self.upstream(src):  # would create a cycle
            return False
        self.links = [l for l in self.links
                      if not (l[1] == dst and l[2] == input_name)]
        self.links.append([src, dst, input_name])
        return True

    def disconnect(self, dst: int, input_name: str) -> None:
        self.links = [l for l in self.links
                      if not (l[1] == dst and l[2] == input_name)]

    def link_into(self, dst: int, input_name: str):
        for src, d, name in self.links:
            if d == dst and name == input_name:
                return src
        return None

    def output_id(self) -> int:
        for nid, n in self.nodes.items():
            if n["type"] == "output":
                return nid
        return self.add("output", (560.0, 180.0))

    # ---- evaluation ----
    def evaluate(self, mesh) -> np.ndarray:
        """Bake the graph to per-face colors (M, 3) uint8-range floats."""
        centroids = mesh.vertices[mesh.faces].mean(axis=1)
        extent = np.maximum(mesh.aabb_max - mesh.aabb_min, 1e-9)
        pos01 = np.clip((centroids - mesh.aabb_min) / extent, 0.0, 1.0)
        m = len(centroids)
        memo: dict[int, np.ndarray] = {}

        def const(v):
            return np.full((m, 3), v, dtype=np.float64)

        def ev(nid: int, depth: int = 0) -> np.ndarray:
            if depth > 32 or nid not in self.nodes:
                return const(0.5)
            if nid in memo:
                return memo[nid]
            node = self.nodes[nid]
            kind = node["type"]
            p = node["params"]

            def inp(name, default):
                src = self.link_into(nid, name)
                return ev(src, depth + 1) if src is not None else default

            if kind == "color":
                out = np.tile([p["r"], p["g"], p["b"]], (m, 1))
            elif kind == "position":
                out = pos01.copy()
            elif kind == "normal":
                out = mesh.normals * 0.5 + 0.5
            elif kind == "checker":
                scale = max(p["scale"], 1e-6)
                parity = np.floor(centroids / scale).sum(axis=1).astype(np.int64) % 2
                a = inp("a", const(0.1))
                b = inp("b", const(0.9))
                out = np.where(parity[:, None] == 0, a, b)
            elif kind == "noise":
                scale = max(p["scale"], 1e-6)
                cells = np.floor(centroids / scale).astype(np.int64) + int(p["seed"])
                out = np.repeat(_hash_noise(cells)[:, None], 3, axis=1)
            elif kind == "gradient":
                axis = int(round(np.clip(p["axis"], 0, 2)))
                f = pos01[:, axis][:, None]
                a = inp("a", const(0.0))
                b = inp("b", const(1.0))
                out = a * (1.0 - f) + b * f
            elif kind == "mix":
                fac = inp("fac", const(0.5)).mean(axis=1, keepdims=True)
                out = inp("a", const(0.0)) * (1.0 - fac) + inp("b", const(1.0)) * fac
            elif kind == "multiply":
                out = inp("a", const(1.0)) * inp("b", const(1.0))
            elif kind == "output":
                out = inp("color", mesh.face_colors / 255.0)
            else:
                out = const(0.5)
            memo[nid] = out
            return out

        return np.clip(ev(self.output_id()), 0.0, 1.0) * 255.0

    def apply(self, mesh) -> None:
        mesh.face_colors = self.evaluate(mesh)

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {"nodes": [{"id": nid, **{k: v for k, v in n.items()}}
                          for nid, n in self.nodes.items()],
                "links": [list(l) for l in self.links]}

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialGraph":
        g = cls.__new__(cls)
        g.nodes = {}
        g.links = [list(l) for l in data.get("links", [])]
        g.next_id = 1
        for n in data.get("nodes", []):
            nid = int(n["id"])
            g.nodes[nid] = {"type": n["type"], "pos": list(n["pos"]),
                            "params": dict(n.get("params", {}))}
            g.next_id = max(g.next_id, nid + 1)
        return g
