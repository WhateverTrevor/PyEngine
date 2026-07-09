"""Real-time software 3D renderer with two shading paths.

Per-pixel (default): a deferred pass. Triangles are depth-sorted and filled
into a low-resolution *face-ID buffer* (pygame's C rasterizer), then numpy
reconstructs every pixel's world position by intersecting its camera ray with
the face's plane, and lights each pixel individually — smooth distance
falloff, smooth spotlight penumbras, IES angular profiles, per-pixel fog.
Ray-traced shadow factors stay per-face (the tracer's granularity) and
modulate the per-pixel light. The result is upscaled to the window.

Flat (F2): the classic one-color-per-face path — faster, chunkier lighting.

Shared pipeline per frame: transform to world/camera space (numpy matmuls),
backface-cull, near-plane clip (Sutherland-Hodgman), painter's depth sort.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from .lighting import SpotLight, ies_curve
from .math3d import rotation_x, rotation_y

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


class _LightInfo:
    __slots__ = ("entity", "light", "pos", "colorf", "axis", "cos_in", "cos_out",
                 "curve")

    def __init__(self, entity, light, pos, colorf, axis, cos_in, cos_out, curve):
        self.entity = entity
        self.light = light
        self.pos = pos
        self.colorf = colorf
        self.axis = axis
        self.cos_in = cos_in
        self.cos_out = cos_out
        self.curve = curve


def _gather_lights(scene) -> list[_LightInfo]:
    """Collect enabled point/spot lights with world position, axis, IES curve."""
    lights = []
    for e in scene.entities:
        light = e.light
        if light is None or not light.enabled or light.intensity <= 1e-4:
            continue
        m = e.transform.matrix()
        off = e.light_offset
        pos = m[:3, :3] @ np.array([off.x, off.y, off.z]) + m[:3, 3]
        cos_in = cos_out = None
        if isinstance(light, SpotLight):
            axis = m[:3, :3] @ np.array([0.0, 0.0, -1.0])
            cos_in = math.cos(math.radians(light.inner))
            cos_out = math.cos(math.radians(light.outer))
        else:
            axis = m[:3, :3] @ np.array([0.0, -1.0, 0.0])
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        colorf = np.asarray(light.color, dtype=np.float64) / 255.0
        lights.append(_LightInfo(e, light, pos, colorf, axis, cos_in, cos_out,
                                 ies_curve(light.ies)))
    return lights


def _face_light_strength(info: _LightInfo, normals: np.ndarray,
                         centroids: np.ndarray) -> np.ndarray:
    """Per-face light strength (intensity*atten*lambert*cone*ies), no shadow."""
    light = info.light
    delta = info.pos[None, :] - centroids
    dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
    atten = np.clip(1.0 - dist / light.range, 0.0, 1.0) ** 2
    lambert = np.clip(np.einsum("ij,ij->i", normals, delta) / dist, 0.0, 1.0)
    strength = light.intensity * atten * lambert
    needs_angle = info.cos_in is not None or info.curve is not None
    if needs_angle:
        to_face = -delta / dist[:, None]
        cos_ang = to_face @ info.axis
        if info.cos_in is not None:
            cone = np.clip((cos_ang - info.cos_out)
                           / max(info.cos_in - info.cos_out, 1e-6), 0.0, 1.0) ** 2
            strength = strength * cone
        if info.curve is not None:
            ang = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
            strength = strength * info.curve[ang.astype(np.int32)]
    return strength


class Renderer:
    def __init__(self):
        self.wireframe = False
        self.per_pixel = True
        self.render_scale = 3  # internal resolution = window / render_scale
        self.stats = {"triangles": 0, "shadow_lights": 0, "mode": ""}
        self._sky_cache = None
        self._defer_cache = None

    def render(self, surface: pygame.Surface, scene, camera, tracer=None) -> None:
        if self.per_pixel and not self.wireframe:
            self.stats["mode"] = f"per-pixel 1/{self.render_scale}"
            self._render_deferred(surface, scene, camera, tracer)
        else:
            self.stats["mode"] = "wire" if self.wireframe else "flat"
            self._render_flat(surface, scene, camera, tracer)

    # ------------------------------------------------------------------
    # shared geometry: camera-space transform, cull, clip, project
    # ------------------------------------------------------------------
    def _entity_geometry(self, entity, view, k, cx, cy, near, far):
        """Returns per-entity geometry dict, or None if nothing to draw."""
        mesh = entity.mesh
        model = entity.transform.matrix()
        verts_world = mesh.vertices @ model[:3, :3].T + model[:3, 3]
        verts_cam = verts_world @ view[:3, :3].T + view[:3, 3]

        try:
            normal_mat = np.linalg.inv(model[:3, :3]).T
        except np.linalg.LinAlgError:
            return None
        normals_world = mesh.normals @ normal_mat.T
        normals_world /= np.maximum(
            np.linalg.norm(normals_world, axis=1, keepdims=True), 1e-12)
        normals_cam = normals_world @ view[:3, :3].T

        tri = verts_cam[mesh.faces]           # (M, 3, 3)
        centroids_cam = tri.mean(axis=1)
        depth = -centroids_cam[:, 2]

        front = np.einsum("ij,ij->i", normals_cam, centroids_cam) < 0.0
        front &= depth < far

        in_front = tri[:, :, 2] < -near
        fast = front & in_front.all(axis=1)
        crossing = front & ~fast & in_front.any(axis=1)

        fast_pts = None
        fast_idx = np.nonzero(fast)[0]
        if len(fast_idx) > 0:
            t = tri[fast_idx]
            inv_z = 1.0 / -t[:, :, 2]
            pts = np.stack([cx + k * t[:, :, 0] * inv_z,
                            cy - k * t[:, :, 1] * inv_z], axis=-1)
            np.clip(pts, -_COORD_LIMIT, _COORD_LIMIT, out=pts)
            fast_pts = np.rint(pts).astype(np.int32).tolist()

        clipped = []  # (face_index, pts)
        for i in np.nonzero(crossing)[0]:
            poly = _clip_near(tri[i], near)
            if len(poly) < 3:
                continue
            pts = []
            for p in poly:
                inv_z = 1.0 / max(-p[2], 1e-9)
                x = min(max(cx + k * p[0] * inv_z, -_COORD_LIMIT), _COORD_LIMIT)
                y = min(max(cy - k * p[1] * inv_z, -_COORD_LIMIT), _COORD_LIMIT)
                pts.append((int(x), int(y)))
            clipped.append((int(i), pts))

        return {"normals": normals_world, "centroids": verts_world[mesh.faces].mean(axis=1),
                "depth": depth, "fast_idx": fast_idx, "fast_pts": fast_pts,
                "clipped": clipped}

    def _directional_base(self, scene, normals):
        dl = scene.light
        dl_dir = dl.direction.to_array()
        to_light = -dl_dir / max(np.linalg.norm(dl_dir), 1e-12)
        dl_color = np.asarray(dl.color, dtype=np.float64) / 255.0 * dl.intensity
        lambert = np.clip(normals @ to_light, 0.0, 1.0)
        return dl.ambient + dl_color[None, :] * ((1.0 - dl.ambient) * lambert)[:, None]

    # ------------------------------------------------------------------
    # flat path: one color per face
    # ------------------------------------------------------------------
    def _render_flat(self, surface, scene, camera, tracer) -> None:
        w, h = surface.get_size()
        self._draw_background(surface, scene)
        view = camera.view_matrix()
        k = 0.5 * h / math.tan(math.radians(camera.fov) * 0.5)
        lights = _gather_lights(scene)
        self.stats["shadow_lights"] = sum(1 for l in lights if l.light.cast_shadows)
        fog = scene.fog
        fog_color = np.asarray(fog.color, dtype=np.float64) if fog else None

        polys = []
        for entity in scene.entities:
            if entity.mesh is None or not entity.visible:
                continue
            geo = self._entity_geometry(entity, view, k, w * 0.5, h * 0.5,
                                        camera.near, camera.far)
            if geo is None:
                continue
            normals, centroids, depth = geo["normals"], geo["centroids"], geo["depth"]

            lum = self._directional_base(scene, normals)
            for info in lights:
                strength = _face_light_strength(info, normals, centroids)
                active = strength > 1e-3
                if not active.any():
                    continue
                if tracer is not None and info.light.cast_shadows:
                    strength = strength * tracer.shadow_factors(
                        entity, info.light, info.pos, centroids, normals, active)
                lum += info.colorf[None, :] * strength[:, None]

            colors = entity.mesh.face_colors * lum
            if fog is not None:
                f = np.clip((depth - fog.start) / (fog.end - fog.start), 0.0, 1.0)[:, None]
                colors = colors * (1.0 - f) + fog_color * f
            colors = np.clip(colors, 0.0, 255.0)

            if geo["fast_pts"] is not None:
                idx = geo["fast_idx"]
                polys += zip(depth[idx].tolist(),
                             colors[idx].astype(np.uint8).tolist(), geo["fast_pts"])
            for i, pts in geo["clipped"]:
                polys.append((float(depth[i]), colors[i].astype(np.uint8).tolist(), pts))

        polys.sort(key=lambda p: p[0], reverse=True)
        self.stats["triangles"] = len(polys)
        draw = pygame.draw.polygon
        width = 1 if self.wireframe else 0
        for _, color, pts in polys:
            draw(surface, color, pts, width)

    # ------------------------------------------------------------------
    # deferred path: per-pixel lighting via a face-ID buffer
    # ------------------------------------------------------------------
    def _render_deferred(self, surface, scene, camera, tracer) -> None:
        w, h = surface.get_size()
        rs = max(1, int(self.render_scale))
        rw, rh = max(2, w // rs), max(2, h // rs)
        view = camera.view_matrix()
        near, far = camera.near, camera.far
        k = 0.5 * rh / math.tan(math.radians(camera.fov) * 0.5)
        cx, cy = rw * 0.5, rh * 0.5

        cache = self._defer_cache
        if cache is None or cache["size"] != (rw, rh):
            xs = ((np.arange(rw, dtype=np.float32) + 0.5) - cx) / k
            ys = -(((np.arange(rh, dtype=np.float32) + 0.5) - cy) / k)
            grid = np.empty((rw, rh, 3), dtype=np.float32)
            grid[..., 0] = xs[:, None]
            grid[..., 1] = ys[None, :]
            grid[..., 2] = -1.0
            cache = {"size": (rw, rh), "grid": grid,
                     "surf": pygame.Surface((rw, rh)), "sky": None, "skykey": None}
            self._defer_cache = cache
        small = cache["surf"]

        skykey = (scene.sky, scene.background)
        if cache["skykey"] != skykey:
            sky = np.empty((rw, rh, 3), dtype=np.float32)
            if scene.sky is not None:
                top = np.asarray(scene.sky[0], dtype=np.float32)
                horizon = np.asarray(scene.sky[1], dtype=np.float32)
                f = (np.arange(rh, dtype=np.float32) / max(rh - 1, 1))[:, None]
                sky[:] = (top[None, :] * (1.0 - f) + horizon[None, :] * f)[None, :, :]
            else:
                sky[:] = np.asarray(scene.background, dtype=np.float32)
            cache["sky"] = sky.astype(np.uint8)
            cache["skykey"] = skykey

        lights = _gather_lights(scene)
        self.stats["shadow_lights"] = sum(1 for l in lights if l.light.cast_shadows)
        fog = scene.fog

        # --- collect geometry + per-face attributes across all entities ---
        f_normals, f_centroids, f_albedo, f_base = [], [], [], []
        f_shadow = [[] for _ in lights]
        polys = []  # (depth, global_face_id, points)
        offset = 0
        for entity in scene.entities:
            if entity.mesh is None or not entity.visible:
                continue
            geo = self._entity_geometry(entity, view, k, cx, cy, near, far)
            if geo is None:
                continue
            normals, centroids, depth = geo["normals"], geo["centroids"], geo["depth"]
            m_faces = len(depth)

            f_normals.append(normals)
            f_centroids.append(centroids)
            f_albedo.append(entity.mesh.face_colors)
            f_base.append(self._directional_base(scene, normals))
            for li, info in enumerate(lights):
                if tracer is not None and info.light.cast_shadows:
                    strength = _face_light_strength(info, normals, centroids)
                    f_shadow[li].append(tracer.shadow_factors(
                        entity, info.light, info.pos, centroids, normals,
                        strength > 1e-3))
                else:
                    f_shadow[li].append(np.ones(m_faces))

            if geo["fast_pts"] is not None:
                idx = geo["fast_idx"]
                polys += zip(depth[idx].tolist(), (idx + offset).tolist(),
                             geo["fast_pts"])
            for i, pts in geo["clipped"]:
                polys.append((float(depth[i]), i + offset, pts))
            offset += m_faces

        polys.sort(key=lambda p: p[0], reverse=True)
        self.stats["triangles"] = len(polys)

        small.fill((0, 0, 0))  # id 0 = sky
        draw = pygame.draw.polygon
        for _, fid, pts in polys:
            c = fid + 1
            draw(small, ((c >> 16) & 255, (c >> 8) & 255, c & 255), pts)

        if offset == 0:
            pygame.surfarray.blit_array(small, cache["sky"].astype(np.uint8))
            pygame.transform.scale(small, (w, h), surface)
            return

        normals = np.concatenate(f_normals).astype(np.float32)
        centroids = np.concatenate(f_centroids).astype(np.float32)
        albedo = np.concatenate(f_albedo).astype(np.float32)
        base = np.concatenate(f_base).astype(np.float32)
        shadows = [np.concatenate(s).astype(np.float32) for s in f_shadow]

        # --- per-pixel pass, run only on visible (non-sky) pixels ---
        img = pygame.surfarray.array3d(small).astype(np.int32)
        ids = ((img[..., 0] << 16) | (img[..., 1] << 8) | img[..., 2]).reshape(-1)
        vis = np.flatnonzero(ids > 0)
        frame = cache["sky"].copy()  # uint8 (rw, rh, 3)
        if len(vis) == 0:
            pygame.surfarray.blit_array(small, frame)
            pygame.transform.scale(small, (w, h), surface)
            return
        fid = ids[vis] - 1

        n = normals[fid]                       # (V, 3)
        p0 = centroids[fid]
        cam = camera.position.to_array().astype(np.float32)

        # camera rays for visible pixels (rotated grid cached per view angle)
        dirs_key = (camera.yaw, camera.pitch)
        if cache.get("dirs_key") != dirs_key:
            rot = (rotation_y(camera.yaw) @ rotation_x(camera.pitch))[:3, :3]
            cache["dirs"] = (cache["grid"].reshape(-1, 3)
                             @ rot.T.astype(np.float32))
            cache["dirs_key"] = dirs_key
        dirs = cache["dirs"][vis]

        denom = np.einsum("ij,ij->i", dirs, n)
        tnum = np.einsum("ij,ij->i", p0 - cam[None, :], n)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = tnum / denom
        t = np.clip(np.nan_to_num(t, nan=far, posinf=far, neginf=far),
                    near, far).astype(np.float32)
        pos = cam[None, :] + dirs * t[:, None]

        lum = base[fid].copy()
        for li, info in enumerate(lights):
            light = info.light
            delta_all = info.pos.astype(np.float32)[None, :] - pos
            dist2 = np.einsum("ij,ij->i", delta_all, delta_all)
            sel = np.flatnonzero(dist2 < light.range * light.range)
            if len(sel) == 0:
                continue
            delta = delta_all[sel]
            dist = np.maximum(np.sqrt(dist2[sel]), 1e-6)
            atten = 1.0 - dist / light.range
            atten *= atten
            lambert = np.clip(
                np.einsum("ij,ij->i", n[sel], delta) / dist, 0.0, 1.0)
            strength = (light.intensity * atten) * lambert
            if info.cos_in is not None or info.curve is not None:
                to_px = -delta / dist[:, None]
                cos_ang = to_px @ info.axis.astype(np.float32)
                if info.cos_in is not None:
                    cone = np.clip((cos_ang - info.cos_out)
                                   / max(info.cos_in - info.cos_out, 1e-6),
                                   0.0, 1.0)
                    strength *= cone * cone
                if info.curve is not None:
                    ang = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
                    strength *= info.curve[ang.astype(np.int32)]
            strength *= shadows[li][fid[sel]]
            lum[sel] += info.colorf.astype(np.float32)[None, :] * strength[:, None]

        out = albedo[fid] * lum
        if fog is not None:
            f = np.clip((t - fog.start) / (fog.end - fog.start), 0.0, 1.0)[:, None]
            out = out * (1.0 - f) + np.asarray(fog.color, dtype=np.float32) * f
        np.clip(out, 0.0, 255.0, out=out)
        frame.reshape(-1, 3)[vis] = out.astype(np.uint8)

        pygame.surfarray.blit_array(small, frame)
        pygame.transform.scale(small, (w, h), surface)

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
