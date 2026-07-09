"""Scene lighting and atmospherics.

Point and spot lights support ray-traced soft shadows (see raytrace.py):
`radius` is the light's physical size — bigger radius means softer penumbras —
and `shadow_samples` controls how many shadow rays are cast per lit face.
"""
from __future__ import annotations

from dataclasses import dataclass

from .math3d import Vec3


class DirectionalLight:
    """Sun/moon-style light. `direction` is the direction the light travels."""

    def __init__(self, direction: Vec3, ambient: float = 0.3,
                 color=(255, 255, 255), intensity: float = 1.0):
        self.direction = direction.normalized()
        self.ambient = ambient
        self.color = color
        self.intensity = intensity


class PointLight:
    """Omnidirectional light attached to an entity (see Entity.light)."""

    def __init__(self, color=(255, 255, 255), intensity: float = 1.0,
                 range: float = 15.0, radius: float = 0.25,
                 cast_shadows: bool = True, shadow_samples: int = 8,
                 shadow_interval: int = 1, enabled: bool = True):
        self.color = color
        self.intensity = intensity
        self.range = range              # distance where light fades to zero
        self.radius = radius            # physical size -> penumbra softness
        self.cast_shadows = cast_shadows
        self.shadow_samples = shadow_samples
        self.shadow_interval = shadow_interval  # frames between shadow updates
        self.enabled = enabled


class SpotLight(PointLight):
    """Cone light (flashlight). Aims along the owning entity's -Z axis."""

    def __init__(self, inner: float = 14.0, outer: float = 28.0, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner  # full-brightness cone half-angle, degrees
        self.outer = outer  # falloff-to-zero half-angle, degrees


@dataclass
class Fog:
    color: tuple[int, int, int]
    start: float   # camera distance where fog begins
    end: float     # camera distance where fog is fully opaque
