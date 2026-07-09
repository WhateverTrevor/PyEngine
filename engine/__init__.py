"""PyEngine — a compact real-time 3D game engine in pure Python (pygame + numpy)."""
from . import behaviors
from .assets import AssetLibrary, load_scene, save_scene
from .camera import Camera
from .core import Engine
from .lighting import IES_PROFILES, DirectionalLight, Fog, PointLight, SpotLight
from .math3d import Vec3
from .mesh import Mesh, box, checkerboard, cone, cube, cylinder, icosphere, torus
from .raytrace import ShadowTracer, pick_entity
from .scene import Behavior, Entity, Scene, Transform

__all__ = [
    "Engine", "Scene", "Entity", "Behavior", "Transform", "Camera",
    "Mesh", "cube", "box", "cone", "cylinder", "icosphere", "torus", "checkerboard",
    "DirectionalLight", "PointLight", "SpotLight", "Fog", "Vec3", "IES_PROFILES",
    "AssetLibrary", "save_scene", "load_scene", "ShadowTracer", "pick_entity",
    "behaviors",
]
