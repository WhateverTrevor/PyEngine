"""Node-based materials.

A MaterialGraph is a small dataflow graph -- Unreal Material Editor's core
vocabulary (Color/Position/Normal/Checker/Noise/Gradient/HDRI sources through
Add/Multiply/Power/Clamp/OneMinus/Mix/Lerp into an Output node), evaluated
with numpy and baked either onto a mesh's per-face colors or, for a sky
material, onto an equirectangular HDR image. Graphs serialize to plain dicts,
so they live inside scene files, and evaluation is deterministic.

Two evaluation *contexts* share one node graph:

- **face** (`evaluate`): one sample per mesh face, `position`/`normal` return
  the face's own (0..1-mapped position)/(normal*0.5+0.5), `hdri` samples
  along the face normal. Output is clamped to 0..255 (standard albedo).
- **sky** (`evaluate_sky`): one sample per equirect-grid direction (the
  owning entity's HDRI resolution, capped 1024x512), `position`/`normal`
  both return `direction*0.5+0.5`, `hdri` samples along that direction.
  Output is linear HDR radiance, clamped at >=0 only (no upper bound).

We match Unreal's *core* node vocabulary (the handful of ops that show up in
almost every graph), not its full node library.
"""
from __future__ import annotations

import numpy as np

from .environment import sample_equirect

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
    "add":      (("a", "b"), {}),
    "power":    (("a",), {"exp": 2.0}),
    "clamp":    (("a",), {"min": 0.0, "max": 1.0}),
    "one_minus": (("a",), {}),
    "lerp":     (("a", "b", "fac"), {}),
    "hdri":     ((), {}),
}

# slider ranges for the editor UI
PARAM_RANGES = {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0),
                "scale": (0.05, 8.0), "seed": (0.0, 100.0), "axis": (0.0, 2.0),
                "exp": (0.1, 8.0), "min": (0.0, 1.0), "max": (0.0, 1.0)}


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
    def _evaluate_common(self, m: int, sample_pts: np.ndarray, pos01: np.ndarray,
                         normal_out: np.ndarray, hdri_fn, output_default: np.ndarray
                         ) -> np.ndarray:
        """Shared node walk for both contexts (all arrays are (m, 3) float64).

        `sample_pts` is the "world position" procedural nodes (checker/noise)
        key off; `pos01` is what `position`/`gradient` read (0..1-mapped);
        `normal_out` is what the `normal` node returns; `hdri_fn()` computes
        the `hdri` node lazily (only called if the graph actually uses it).
        """
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
                out = normal_out
            elif kind == "hdri":
                out = hdri_fn()
            elif kind == "checker":
                scale = max(p["scale"], 1e-6)
                parity = np.floor(sample_pts / scale).sum(axis=1).astype(np.int64) % 2
                a = inp("a", const(0.1))
                b = inp("b", const(0.9))
                out = np.where(parity[:, None] == 0, a, b)
            elif kind == "noise":
                scale = max(p["scale"], 1e-6)
                cells = np.floor(sample_pts / scale).astype(np.int64) + int(p["seed"])
                out = np.repeat(_hash_noise(cells)[:, None], 3, axis=1)
            elif kind == "gradient":
                axis = int(round(np.clip(p["axis"], 0, 2)))
                f = pos01[:, axis][:, None]
                a = inp("a", const(0.0))
                b = inp("b", const(1.0))
                out = a * (1.0 - f) + b * f
            elif kind == "mix" or kind == "lerp":
                fac = inp("fac", const(0.5)).mean(axis=1, keepdims=True)
                out = inp("a", const(0.0)) * (1.0 - fac) + inp("b", const(1.0)) * fac
            elif kind == "multiply":
                out = inp("a", const(1.0)) * inp("b", const(1.0))
            elif kind == "add":
                out = inp("a", const(0.0)) + inp("b", const(0.0))
            elif kind == "power":
                out = np.power(np.maximum(inp("a", const(0.5)), 0.0),
                               max(p["exp"], 1e-6))
            elif kind == "clamp":
                lo, hi = p["min"], p["max"]
                out = np.clip(inp("a", const(0.5)), min(lo, hi), max(lo, hi))
            elif kind == "one_minus":
                out = 1.0 - inp("a", const(0.5))
            elif kind == "output":
                out = inp("color", output_default)
            else:
                out = const(0.5)
            memo[nid] = out
            return out

        return ev(self.output_id())

    def evaluate(self, mesh, source_image: np.ndarray | None = None) -> np.ndarray:
        """Bake the graph to per-face colors (M, 3) uint8-range floats -- face
        context. `source_image` is the equirect HDR array an `hdri` node
        samples along each face's normal (the owning entity's environment
        source, if any); with none, `hdri` falls back to neutral gray."""
        centroids = mesh.vertices[mesh.faces].mean(axis=1)
        extent = np.maximum(mesh.aabb_max - mesh.aabb_min, 1e-9)
        pos01 = np.clip((centroids - mesh.aabb_min) / extent, 0.0, 1.0)
        m = len(centroids)
        normal_out = mesh.normals * 0.5 + 0.5

        def hdri_fn():
            if source_image is None:
                return np.full((m, 3), 0.5)
            return sample_equirect(source_image, mesh.normals)

        out = self._evaluate_common(m, centroids, pos01, normal_out, hdri_fn,
                                    mesh.face_colors / 255.0)
        return np.clip(out, 0.0, 1.0) * 255.0

    def evaluate_sky(self, source_image: np.ndarray, max_w: int = 1024,
                     max_h: int = 512) -> np.ndarray:
        """Bake the graph to an equirect radiance image (h, w, 3) float32 --
        sky context. Resolution is the source HDRI's own, capped to
        `max_w`x`max_h`. Output is linear HDR: clamped at >=0 only."""
        sh, sw = source_image.shape[:2]
        h, w = max(1, min(sh, max_h)), max(1, min(sw, max_w))
        theta = (np.arange(h, dtype=np.float64) + 0.5) / h * np.pi
        phi = (np.arange(w, dtype=np.float64) + 0.5) / w * 2.0 * np.pi
        st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
        dirs = np.stack([st * np.cos(phi)[None, :],
                         np.broadcast_to(ct, (h, w)),
                         st * np.sin(phi)[None, :]], axis=-1).reshape(-1, 3)
        pos01 = dirs * 0.5 + 0.5
        m = len(dirs)

        def hdri_fn():
            return sample_equirect(source_image, dirs)

        out = self._evaluate_common(m, dirs, pos01, pos01, hdri_fn,
                                    np.zeros((m, 3)))
        return np.maximum(out, 0.0).astype(np.float32).reshape(h, w, 3)

    def apply(self, entity, draft: bool = False) -> None:
        """Bake onto whichever context `entity` supports: face colors for a
        mesh entity, or a sky-sphere's environment image + ambient cube for
        an entity that only carries an Environment. `draft` bakes the sky
        context at a quarter resolution (256x128 cap) -- cheap enough to run
        every frame while dragging a slider; call again with draft=False on
        release for the full-resolution result."""
        if entity.mesh is not None:
            source = entity.environment.source if entity.environment is not None else None
            entity.mesh.face_colors = self.evaluate(entity.mesh, source)
        elif entity.environment is not None:
            max_w, max_h = (256, 128) if draft else (1024, 512)
            image = self.evaluate_sky(entity.environment.source, max_w, max_h)
            entity.environment.set_image(image)

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
