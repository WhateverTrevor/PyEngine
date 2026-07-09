"""Perspective camera with yaw/pitch orientation."""
from __future__ import annotations

import math

import numpy as np

from .math3d import Vec3, rotation_x, rotation_y, translation


class Camera:
    def __init__(self, position: Vec3 | None = None, yaw: float = 0.0, pitch: float = 0.0,
                 fov: float = 70.0, near: float = 0.1, far: float = 400.0):
        self.position = position or Vec3()
        self.yaw = yaw       # radians, rotation around +Y; 0 looks down -Z
        self.pitch = pitch   # radians, positive looks up
        self.fov = fov       # vertical field of view, degrees
        self.near = near
        self.far = far

    def view_matrix(self) -> np.ndarray:
        return rotation_x(-self.pitch) @ rotation_y(-self.yaw) @ translation(-self.position)

    def forward(self) -> Vec3:
        cp = math.cos(self.pitch)
        return Vec3(-math.sin(self.yaw) * cp, math.sin(self.pitch), -math.cos(self.yaw) * cp)

    def right(self) -> Vec3:
        return Vec3(math.cos(self.yaw), 0.0, -math.sin(self.yaw))

    def project(self, point: Vec3, width: int, height: int):
        """World point -> (screen_x, screen_y, depth), or None if behind camera."""
        v = self.view_matrix()
        p = v[:3, :3] @ point.to_array() + v[:3, 3]
        if p[2] >= -self.near:
            return None
        k = 0.5 * height / math.tan(math.radians(self.fov) * 0.5)
        inv_z = 1.0 / -p[2]
        return (0.5 * width + k * p[0] * inv_z,
                0.5 * height - k * p[1] * inv_z,
                -p[2])

    def mouse_ray(self, mx: float, my: float, width: int, height: int) -> np.ndarray:
        """World-space direction of the ray through a screen pixel (unit vector)."""
        k = 0.5 * height / math.tan(math.radians(self.fov) * 0.5)
        d_cam = np.array([(mx - 0.5 * width) / k, -(my - 0.5 * height) / k, -1.0])
        rot = (rotation_y(self.yaw) @ rotation_x(self.pitch))[:3, :3]
        d = rot @ d_cam
        return d / np.linalg.norm(d)
