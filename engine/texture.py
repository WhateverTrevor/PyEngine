"""Texture assets: image files loaded/cached as float arrays for material
node sampling. pygame.image loads .png/.jpg/.jpeg/.bmp natively -- no new
dependency is needed for any of the supported formats.

Two entry points:

- `import_texture` copies a source image into assets/textures/ and writes a
  self-contained `{"texture": {...}}` asset JSON, mirroring
  `fbx.import_fbx()` / `environment.import_hdri()`'s pattern so it appears
  in the content browser like any other asset.
- `load_texture_rel` resolves a path relative to the current texture root
  (set once via `set_texture_root`, called from `AssetLibrary.__init__` --
  the one place that knows the assets directory) and returns a cached
  (H, W, 4) float64 0..1 RGBA array, or None if the path is empty or the
  file is missing -- `TextureSample` treats None as "no texture assigned"
  and falls back to neutral gray.

Sampling is nearest-neighbor (no filtering, no mipmaps) -- this engine bakes
materials to per-face colors, one sample per face centroid, so there is no
per-pixel footprint to filter against.
"""
from __future__ import annotations

import os

import numpy as np

TEXTURE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

_cache: dict[str, np.ndarray] = {}
_texture_root: str | None = None


def set_texture_root(assets_dir: str) -> None:
    """Point texture-relative-path lookups at an assets/ directory."""
    global _texture_root
    _texture_root = assets_dir


def clear_cache() -> None:
    """Drop all cached images -- tests use this between temp asset dirs."""
    _cache.clear()


def load_texture(path: str) -> np.ndarray:
    """Load (or fetch from cache) an image file as (H, W, 4) float64 0..1."""
    key = os.path.abspath(path)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    import pygame

    surf = pygame.image.load(path)
    # pygame surfarray axes are (width, height, 3) -- transpose to (H, W, 3)
    # so row 0 is the image's top row, and v=0 (top of a UV) maps there.
    rgb = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).astype(np.float64) / 255.0
    try:
        alpha = np.transpose(pygame.surfarray.array_alpha(surf), (1, 0)).astype(np.float64) / 255.0
    except Exception:
        alpha = np.ones(rgb.shape[:2], dtype=np.float64)
    image = np.concatenate([rgb, alpha[..., None]], axis=-1)
    _cache[key] = image
    return image


def load_texture_rel(rel_path: str) -> np.ndarray | None:
    """Resolve `rel_path` against the current texture root, or None."""
    if not rel_path or _texture_root is None:
        return None
    path = os.path.join(_texture_root, rel_path)
    if not os.path.isfile(path):
        return None
    return load_texture(path)


def sample_texture(image: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor sample at (M, 2) UVs, tiled via wraparound modulo.

    Returns (rgb (M, 3) 0..1, alpha (M,) 0..1).
    """
    h, w = image.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    xi = np.clip((u * w).astype(np.int64), 0, w - 1)
    yi = np.clip(((1.0 - v) * h).astype(np.int64), 0, h - 1)  # v=0 -> top row
    px = image[yi, xi]
    return px[:, :3], px[:, 3]


def import_texture(src_path: str, assets_dir: str) -> str:
    """Copy an image into assets/textures/ and write a texture asset JSON.

    Returns the new asset's display name so the caller can reload the
    library and grab a thumbnail (same contract as `fbx.import_fbx` and
    `environment.import_hdri`).
    """
    import json
    import shutil

    ext = os.path.splitext(src_path)[1].lower()
    if ext not in TEXTURE_EXTS:
        raise ValueError(f"unsupported texture type: {ext}")
    load_texture(src_path)  # raises if pygame can't parse it

    stem = os.path.splitext(os.path.basename(src_path))[0]
    tex_dir = os.path.join(assets_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    dest_name = os.path.basename(src_path)
    dest = os.path.join(tex_dir, dest_name)
    if os.path.abspath(dest) != os.path.abspath(src_path):
        shutil.copyfile(src_path, dest)

    base_name = stem.replace("_", " ").replace("-", " ").title() or "Texture"
    name, n = base_name, 2
    dest_json = os.path.join(assets_dir, f"{name}.json")
    while os.path.exists(dest_json):
        name = f"{base_name} {n}"
        dest_json = os.path.join(assets_dir, f"{name}.json")
        n += 1

    asset = {"name": name, "category": "textures",
             "texture": {"path": f"textures/{dest_name}"}}
    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump(asset, f, indent=2)
    return name
