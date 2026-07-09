"""Minimal 3D math for the engine: Vec3 and 4x4 matrix builders (numpy-backed)."""
from __future__ import annotations

import math

import numpy as np


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, o: "Vec3") -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: "Vec3") -> "Vec3":
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        l = self.length()
        return self * (1.0 / l) if l > 1e-12 else Vec3()

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def translation(v: Vec3) -> np.ndarray:
    m = identity()
    m[:3, 3] = (v.x, v.y, v.z)
    return m


def scaling(v: Vec3) -> np.ndarray:
    m = identity()
    m[0, 0], m[1, 1], m[2, 2] = v.x, v.y, v.z
    return m


def rotation_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotation_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotation_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m
