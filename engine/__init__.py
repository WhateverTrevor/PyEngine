"""PyEngine — a compact real-time 3D game engine in pure Python (pygame + numpy)."""
from . import behaviors
from .assets import AssetLibrary, load_scene, save_scene
from .camera import Camera
from .core import Engine
from .environment import Environment, import_hdri, load_hdr, save_hdr
from .fbx import import_fbx
from .lighting import (IES_PROFILES, DirectionalLight, Fog, FogVolume, PointLight,
                       SpotLight, SunDisc)
from .materials import NODE_DEFS, PARAM_RANGES, MaterialGraph
from .math3d import Vec3
from .mesh import Mesh, box, checkerboard, cone, cube, cylinder, icosphere, torus
from .raytrace import GITracer, ShadowTracer, pick_entity
from .scene import Behavior, Entity, Scene, Transform

__all__ = [
    "Engine", "Scene", "Entity", "Behavior", "Transform", "Camera",
    "Mesh", "cube", "box", "cone", "cylinder", "icosphere", "torus", "checkerboard",
    "DirectionalLight", "PointLight", "SpotLight", "Fog", "Vec3", "IES_PROFILES",
    "SunDisc", "FogVolume",
    "Environment", "load_hdr", "save_hdr", "import_hdri", "import_fbx",
    "MaterialGraph", "NODE_DEFS", "PARAM_RANGES",
    "AssetLibrary", "save_scene", "load_scene", "ShadowTracer", "GITracer", "pick_entity",
    "behaviors",
]
