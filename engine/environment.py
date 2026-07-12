"""HDRI environment maps: Radiance .hdr (RGBE) I/O and image-based lighting.

An Environment wraps an equirectangular HDR image. The deferred renderer
samples it per pixel for the sky background, and diffuse environment lighting
comes from an "ambient cube": the map convolved down to six cosine-weighted
axis colors at load time, evaluated per face normal at runtime — cheap,
smooth, image-based ambient light.
"""
from __future__ import annotations

import os

import numpy as np


def load_hdr(path: str) -> np.ndarray:
    """Read a Radiance RGBE .hdr file -> linear float32 (H, W, 3) radiance."""
    with open(path, "rb") as fh:
        magic = fh.readline()
        if not magic.startswith(b"#?"):
            raise ValueError(f"{os.path.basename(path)}: not a Radiance .hdr file")
        while True:
            line = fh.readline()
            if line in (b"\n", b"\r\n", b""):
                break
        dims = fh.readline().split()
        if len(dims) != 4 or dims[0] != b"-Y" or dims[2] != b"+X":
            raise ValueError("unsupported .hdr orientation (need -Y H +X W)")
        height, width = int(dims[1]), int(dims[3])

        data = fh.read()
    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    pos = 0
    for y in range(height):
        if (width < 8 or width > 0x7FFF or pos + 4 > len(data)
                or data[pos] != 2 or data[pos + 1] != 2
                or (data[pos + 2] << 8 | data[pos + 3]) != width):
            # flat (uncompressed) scanline
            row = np.frombuffer(data, dtype=np.uint8, count=width * 4, offset=pos)
            rgbe[y] = row.reshape(width, 4)
            pos += width * 4
            continue
        pos += 4
        for c in range(4):  # adaptive RLE per channel plane
            x = 0
            while x < width:
                count = data[pos]
                pos += 1
                if count > 128:  # run
                    rgbe[y, x:x + count - 128, c] = data[pos]
                    pos += 1
                    x += count - 128
                else:            # literal
                    rgbe[y, x:x + count, c] = np.frombuffer(
                        data, dtype=np.uint8, count=count, offset=pos)
                    pos += count
                    x += count

    mantissa = rgbe[..., :3].astype(np.float32)
    exponent = rgbe[..., 3].astype(np.int32)
    scale = np.where(exponent > 0,
                     np.ldexp(1.0, exponent - 136), 0.0).astype(np.float32)
    return mantissa * scale[..., None]


def save_hdr(path: str, image: np.ndarray) -> None:
    """Write linear float (H, W, 3) radiance as a flat (non-RLE) .hdr file."""
    img = np.maximum(np.asarray(image, dtype=np.float32), 0.0)
    h, w = img.shape[:2]
    v = img.max(axis=-1)
    exponent = np.where(v >= 1e-32, np.ceil(np.log2(np.maximum(v, 1e-32))) + 1, 0)
    scale = np.where(v >= 1e-32, np.ldexp(1.0, (-exponent + 8).astype(np.int32)), 0.0)
    rgbe = np.zeros((h, w, 4), dtype=np.uint8)
    rgbe[..., :3] = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(v >= 1e-32, exponent + 128, 0).astype(np.uint8)
    with open(path, "wb") as fh:
        fh.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        fh.write(f"-Y {h} +X {w}\n".encode())
        fh.write(rgbe.tobytes())


def import_hdri(src_path: str, assets_dir: str) -> str:
    """Copy an .hdr file into assets/hdri/ and write a sky-sphere asset JSON.

    Mirrors `fbx.import_fbx`'s pattern: validate it parses, copy it into the
    asset library's hdri/ folder, generate a self-contained
    `{"environment": {"hdri": ...}}` asset (same shape as sky_sphere.json),
    and return the new asset's display name so the caller can reload the
    library and grab a thumbnail.
    """
    import json
    import shutil

    load_hdr(src_path)  # raises if the file doesn't parse as a Radiance .hdr
    stem = os.path.splitext(os.path.basename(src_path))[0]
    hdri_dir = os.path.join(assets_dir, "hdri")
    os.makedirs(hdri_dir, exist_ok=True)
    dest_name = os.path.basename(src_path)
    dest = os.path.join(hdri_dir, dest_name)
    if os.path.abspath(dest) != os.path.abspath(src_path):
        shutil.copyfile(src_path, dest)
    name = stem.replace("_", " ").replace("-", " ").title()
    asset = {"name": name, "category": "environment",
             "environment": {"hdri": f"hdri/{dest_name}", "strength": 1.0}}
    with open(os.path.join(assets_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(asset, f, indent=2)
    return name


_CUBE_AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                       [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32)


def sample_equirect(image: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Radiance along unit direction(s) from an equirect image: (N, 3) -> (N, 3)."""
    h, w = image.shape[:2]
    theta = np.arccos(np.clip(dirs[:, 1], -1.0, 1.0))
    phi = np.arctan2(dirs[:, 2], dirs[:, 0]) % (2.0 * np.pi)
    y = np.clip((theta / np.pi * h).astype(np.int32), 0, h - 1)
    x = np.clip((phi / (2.0 * np.pi) * w).astype(np.int32), 0, w - 1)
    return image[y, x]


class Environment:
    """Equirectangular HDR environment: sky sampling + diffuse ambient.

    `source` is the originally loaded HDR image and never changes; `image` is
    what actually renders and starts out equal to it, but a sky MaterialGraph
    (see materials.py) can re-bake `image` (and, via `set_image`, the ambient
    cube derived from it) while leaving `source` alone as the thing `hdri`
    material nodes sample from.
    """

    def __init__(self, image: np.ndarray, strength: float = 1.0):
        self.image = np.asarray(image, dtype=np.float32)
        self.source = self.image.copy()
        self.strength = strength
        self._build_ambient_cube()

    def set_image(self, image: np.ndarray) -> None:
        """Swap the rendered equirect image and rebuild the ambient cube."""
        self.image = np.asarray(image, dtype=np.float32)
        self._build_ambient_cube()

    def _build_ambient_cube(self) -> None:
        img = self.image[::4, ::4]  # irradiance is low-frequency; downsample
        h, w = img.shape[:2]
        theta = (np.arange(h, dtype=np.float32) + 0.5) / h * np.pi
        phi = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 * np.pi
        st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
        dirs = np.stack([st * np.cos(phi)[None, :],
                         np.broadcast_to(ct, (h, w)),
                         st * np.sin(phi)[None, :]], axis=-1)
        weight = st  # equirect solid-angle correction
        self.ambient_cube = np.empty((6, 3), dtype=np.float32)
        flat_dirs = dirs.reshape(-1, 3)
        flat_img = img.reshape(-1, 3)
        flat_w = np.broadcast_to(weight, (h, w)).reshape(-1)
        for i, axis in enumerate(_CUBE_AXES):
            cw = np.maximum(flat_dirs @ axis, 0.0) * flat_w
            total = cw.sum()
            self.ambient_cube[i] = ((flat_img * cw[:, None]).sum(axis=0)
                                    / max(total, 1e-9))

    def sample(self, dirs: np.ndarray) -> np.ndarray:
        """Radiance along unit direction(s): (N, 3) -> (N, 3) float32."""
        return sample_equirect(self.image, dirs)

    def ambient(self, normals: np.ndarray) -> np.ndarray:
        """Diffuse environment light for unit normals: (M, 3) -> (M, 3)."""
        pos = np.maximum(normals, 0.0)
        neg = np.maximum(-normals, 0.0)
        w = np.stack([pos[:, 0], neg[:, 0], pos[:, 1],
                      neg[:, 1], pos[:, 2], neg[:, 2]], axis=-1)
        w /= np.maximum(w.sum(axis=-1, keepdims=True), 1e-9)
        return (w @ self.ambient_cube) * self.strength
