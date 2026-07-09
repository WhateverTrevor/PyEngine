"""Real-time software 3D renderer.

Pipeline, per frame:
  1. Transform every mesh's vertices to world and camera space (numpy matmuls).
  2. Light each face: ambient + directional + every point/spot light with
     distance attenuation, cone falloff, and ray-traced soft shadow factors
     from the ShadowTracer (colored lights accumulate per-channel).
  3. Backface-cull using camera-space face normals (vectorized).
  4. Faces fully in front of the near plane take a fast vectorized projection
     path; faces straddling it are clipped (Sutherland-Hodgman) individually.
  5. Blend distance fog, depth-sort all faces from every mesh together
     (painter's algorithm), and fill them with pygame's polygon rasterizer.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from .lighting import SpotLight

_COORD_LIMIT = 20000.0  # keep projected coords in a range pygame handles safely


def _clip_near(points: np.ndarray, near: float) -> list[np.ndarray]:
    """Clip a camera-space polygon against the plane z = -near."""
    out: list[np.ndarray] = []
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        da, db = -a[2] - near, -b[2] - near   # >= 0 means in front
        if da >= 0.0:
            out.append(a)
        if (da >= 0.0) != (db >= 0.0):
            t = da / (da - db)
            out.append(a + (b - a) * t)
    return out


def _gather_lights(scene):
    """Collect enabled point/spot lights with world position and direction."""
    lights = []
    for e in scene.entities:
        light = e.light
        if light is None or not light.enabled or light.intensity <= 1e-4:
            continue
        m = e.transform.matrix()
        off = e.light_offset
        pos = m[:3, :3] @ np.array([off.x, off.y, off.z]) + m[:3, 3]
        spot_dir = cos_in = cos_out = None
        if isinstance(light, SpotLight):
            d = m[:3, :3] @ np.array([0.0, 0.0, -1.0])
            spot_dir = d / max(np.linalg.norm(d), 1e-12)
            cos_in = math.cos(math.radians(light.inner))
            cos_out = math.cos(math.radians(light.outer))
        lights.append((e, light, pos, spot_dir, cos_in, cos_out))
    return lights


class Renderer:
    def __init__(self):
        self.wireframe = False
        self.stats = {"triangles": 0, "shadow_lights": 0}
        self._sky_cache = None  # (key, surface)

    def render(self, surface: pygame.Surface, scene, camera, tracer=None) -> None:
        w, h = surface.get_size()
        self._draw_background(surface, scene)

        view = camera.view_matrix()
        near, far = camera.near, camera.far
        k = 0.5 * h / math.tan(math.radians(camera.fov) * 0.5)  # px per unit at z=-1
        cx, cy = 0.5 * w, 0.5 * h

        dl = scene.light
        dl_dir = dl.direction.to_array()
        to_light = -dl_dir / max(np.linalg.norm(dl_dir), 1e-12)
        dl_color = np.asarray(dl.color, dtype=np.float64) / 255.0 * dl.intensity
        ambient = dl.ambient
        fog = scene.fog
        fog_color = np.asarray(fog.color, dtype=np.float64) if fog else None

        lights = _gather_lights(scene)
        self.stats["shadow_lights"] = sum(1 for l in lights if l[1].cast_shadows)

        polys: list[tuple[float, list, list]] = []  # (depth, color, screen points)

        for entity in scene.entities:
            mesh = entity.mesh
            if mesh is None or not entity.visible:
                continue

            model = entity.transform.matrix()
            verts_world = mesh.vertices @ model[:3, :3].T + model[:3, 3]
            verts_cam = verts_world @ view[:3, :3].T + view[:3, 3]

            try:
                normal_mat = np.linalg.inv(model[:3, :3]).T
            except np.linalg.LinAlgError:
                continue
            normals_world = mesh.normals @ normal_mat.T
            normals_world /= np.maximum(
                np.linalg.norm(normals_world, axis=1, keepdims=True), 1e-12)
            normals_cam = normals_world @ view[:3, :3].T

            tri = verts_cam[mesh.faces]           # (M, 3, 3)
            centroids_cam = tri.mean(axis=1)
            centroids_world = verts_world[mesh.faces].mean(axis=1)
            depth = -centroids_cam[:, 2]

            front = np.einsum("ij,ij->i", normals_cam, centroids_cam) < 0.0
            front &= depth < far

            # --- lighting: ambient + directional + point/spot lights ---
            lambert_dl = np.clip(normals_world @ to_light, 0.0, 1.0)
            lum = ambient + dl_color[None, :] * ((1.0 - ambient) * lambert_dl)[:, None]

            for light_entity, light, pos, spot_dir, cos_in, cos_out in lights:
                delta = pos[None, :] - centroids_world
                dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
                atten = np.clip(1.0 - dist / light.range, 0.0, 1.0) ** 2
                lambert = np.clip(
                    np.einsum("ij,ij->i", normals_world, delta) / dist, 0.0, 1.0)
                strength = light.intensity * atten * lambert
                if spot_dir is not None:
                    cos_ang = (-delta / dist[:, None]) @ spot_dir
                    cone = np.clip((cos_ang - cos_out) / max(cos_in - cos_out, 1e-6),
                                   0.0, 1.0) ** 2
                    strength *= cone
                active = strength > 1e-3
                if not active.any():
                    continue
                if tracer is not None and light.cast_shadows:
                    shadow = tracer.shadow_factors(
                        entity, light, pos, centroids_world, normals_world, active)
                    strength = strength * shadow
                lum += (np.asarray(light.color, dtype=np.float64) / 255.0)[None, :] \
                    * strength[:, None]

            colors = mesh.face_colors * lum
            if fog is not None:
                f = np.clip((depth - fog.start) / (fog.end - fog.start), 0.0, 1.0)[:, None]
                colors = colors * (1.0 - f) + fog_color * f
            colors = np.clip(colors, 0.0, 255.0)

            # --- projection ---
            in_front = tri[:, :, 2] < -near
            fast = front & in_front.all(axis=1)
            crossing = front & ~fast & in_front.any(axis=1)

            if fast.any():
                t = tri[fast]
                inv_z = 1.0 / -t[:, :, 2]
                pts = np.stack([cx + k * t[:, :, 0] * inv_z,
                                cy - k * t[:, :, 1] * inv_z], axis=-1)
                np.clip(pts, -_COORD_LIMIT, _COORD_LIMIT, out=pts)
                polys += zip(depth[fast].tolist(),
                             colors[fast].astype(np.uint8).tolist(),
                             np.rint(pts).astype(np.int32).tolist())

            for i in np.nonzero(crossing)[0]:
                clipped = _clip_near(tri[i], near)
                if len(clipped) < 3:
                    continue
                pts = []
                for p in clipped:
                    inv_z = 1.0 / max(-p[2], 1e-9)
                    x = min(max(cx + k * p[0] * inv_z, -_COORD_LIMIT), _COORD_LIMIT)
                    y = min(max(cy - k * p[1] * inv_z, -_COORD_LIMIT), _COORD_LIMIT)
                    pts.append((int(x), int(y)))
                polys.append((float(depth[i]), colors[i].astype(np.uint8).tolist(), pts))

        polys.sort(key=lambda p: p[0], reverse=True)
        self.stats["triangles"] = len(polys)

        draw = pygame.draw.polygon
        if self.wireframe:
            for _, color, pts in polys:
                draw(surface, color, pts, 1)
        else:
            for _, color, pts in polys:
                draw(surface, color, pts)

    def _draw_background(self, surface: pygame.Surface, scene) -> None:
        if scene.sky is None:
            surface.fill(scene.background)
            return
        key = (surface.get_size(), scene.sky)
        if self._sky_cache is None or self._sky_cache[0] != key:
            top = np.asarray(scene.sky[0], dtype=np.float64)
            horizon = np.asarray(scene.sky[1], dtype=np.float64)
            strip = pygame.Surface((1, 128))
            for i in range(128):
                f = i / 127.0
                c = top * (1.0 - f) + horizon * f
                strip.set_at((0, i), tuple(int(v) for v in c))
            self._sky_cache = (key, pygame.transform.smoothscale(strip, key[0]))
        surface.blit(self._sky_cache[1], (0, 0))
