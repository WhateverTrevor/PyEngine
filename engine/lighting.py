"""Scene lighting and atmospherics.

Point and spot lights support ray-traced soft shadows (see raytrace.py):
`radius` is the light's physical size — bigger radius means softer penumbras —
and `shadow_samples` controls how many shadow rays are cast per lit face.

Lights also carry an IES profile: an angular intensity curve (like real
photometric IES files) sampled against the angle between the light's axis and
the direction to the surface. Profiles: uniform, spot_soft, downlight, batwing.
A light's axis is its owning entity's -Z for spotlights and -Y (down) for
point lights, rotated by the entity's transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math3d import Vec3

# IES-style angular intensity curves: control points (angle_deg, multiplier),
# expanded to a 181-entry lookup table (1 degree resolution).
_IES_POINTS = {
    "uniform":   [(0, 1.0), (180, 1.0)],
    "spot_soft": [(0, 1.0), (20, 0.85), (40, 0.45), (60, 0.12), (80, 0.02), (180, 0.0)],
    "downlight": [(0, 1.0), (30, 0.95), (60, 0.55), (85, 0.08), (90, 0.0), (180, 0.0)],
    "batwing":   [(0, 0.45), (20, 0.7), (40, 1.0), (55, 0.85), (75, 0.25), (95, 0.0),
                  (180, 0.0)],
}

IES_PROFILES = list(_IES_POINTS)

_ANGLES = np.arange(181, dtype=np.float32)
_IES_CURVES = {
    name: np.interp(_ANGLES, [p[0] for p in pts], [p[1] for p in pts]).astype(np.float32)
    for name, pts in _IES_POINTS.items()
}


def ies_curve(name: str):
    """181-entry per-degree multiplier table, or None for uniform."""
    if name in (None, "uniform"):
        return None
    return _IES_CURVES.get(name)


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
                 shadow_interval: int = 1, enabled: bool = True,
                 ies: str = "uniform"):
        self.color = color
        self.intensity = intensity
        self.range = range              # throw: distance where light fades to zero
        self.radius = radius            # physical size -> shadow penumbra softness
        self.cast_shadows = cast_shadows
        self.shadow_samples = shadow_samples
        self.shadow_interval = shadow_interval  # frames between shadow updates
        self.enabled = enabled
        self.ies = ies                  # angular profile name (see IES_PROFILES)


class SpotLight(PointLight):
    """Cone light. Aims along the owning entity's -Z axis.

    `inner`/`outer` are the cone half-angles in degrees: full brightness
    inside `inner`, falling to zero at `outer` — the gap between them is the
    cone's penumbra. `range` is the total throw.
    """

    def __init__(self, inner: float = 14.0, outer: float = 28.0, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.outer = outer


@dataclass
class Fog:
    color: tuple[int, int, int]
    start: float   # camera distance where fog begins
    end: float     # camera distance where fog is fully opaque
