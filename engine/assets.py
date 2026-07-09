"""Self-contained JSON assets and scene save/load.

An asset is one .json file describing everything the object needs — mesh,
light, behaviors — so it can be dropped into any scene:

    {
      "name": "Torch", "category": "lights",
      "mesh": {"primitive": "cylinder", "radius": 0.1, "height": 1.5,
               "color": [84, 62, 40]},
      "light": {"type": "point", "color": [255, 150, 60], "intensity": 1.7,
                "range": 11, "radius": 0.3, "offset": [0, 0.95, 0]},
      "behaviors": [{"type": "Flicker", "amount": 0.35, "speed": 9}]
    }

Scenes serialize as a list of (asset name, transform) plus the scene's
lighting/atmosphere, so anything placed in the editor round-trips to disk.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import behaviors as behaviors_mod
from . import mesh as mesh_mod
from .camera import Camera
from .environment import Environment, load_hdr
from .lighting import DirectionalLight, Fog, PointLight, SpotLight
from .materials import MaterialGraph
from .math3d import Vec3
from .scene import Entity, Scene

_MESH_FACTORIES = {
    "cube": mesh_mod.cube,
    "box": mesh_mod.box,
    "cylinder": mesh_mod.cylinder,
    "cone": mesh_mod.cone,
    "icosphere": mesh_mod.icosphere,
    "torus": mesh_mod.torus,
    "checkerboard": mesh_mod.checkerboard,
}

# light attributes the editor can change and scenes persist per entity
_LIGHT_PROPS = ("intensity", "color", "range", "radius", "inner", "outer",
                "ies", "enabled", "cast_shadows")


def _tupled(spec: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in spec.items()}


class AssetDef:
    def __init__(self, data: dict, path: str):
        self.name = data["name"]
        self.category = data.get("category", "misc")
        self.data = data
        self.path = path

    def instantiate(self, name: str | None = None) -> Entity:
        d = self.data
        entity = Entity(name or self.name)
        entity.asset_name = self.name
        entity.casts_shadow = d.get("casts_shadow", True)
        entity.collidable = d.get("collidable", True)
        if "rotation" in d:
            entity.transform.rotation = Vec3(*d["rotation"])

        if "mesh" in d:
            spec = _tupled(dict(d["mesh"]))
            primitive = spec.pop("primitive")
            if primitive == "model":  # imported geometry (e.g. FBX -> .npz)
                data = np.load(os.path.join(os.path.dirname(self.path),
                                            spec.pop("path")))
                face_colors = (data["face_colors"].astype(np.float64)
                               if "face_colors" in data else None)
                entity.mesh = mesh_mod.Mesh(
                    data["vertices"], [tuple(f) for f in data["faces"]],
                    base_color=spec.get("color", (170, 170, 175)),
                    face_colors=face_colors)
            else:
                entity.mesh = _MESH_FACTORIES[primitive](**spec)

        if "environment" in d:
            spec = dict(d["environment"])
            hdr_path = os.path.join(os.path.dirname(self.path), spec["hdri"])
            entity.environment = Environment(load_hdr(hdr_path),
                                             strength=spec.get("strength", 1.0))

        if "light" in d:
            spec = _tupled(dict(d["light"]))
            kind = spec.pop("type", "point")
            offset = spec.pop("offset", None)
            entity.light = SpotLight(**spec) if kind == "spot" else PointLight(**spec)
            if offset:
                entity.light_offset = Vec3(*offset)

        for b in d.get("behaviors", []):
            spec = dict(b)
            cls = getattr(behaviors_mod, spec.pop("type"))
            kwargs = {k: Vec3(*v) if isinstance(v, list) and len(v) == 3 else v
                      for k, v in spec.items()}
            entity.add_behavior(cls(**kwargs))

        return entity


class AssetLibrary:
    def __init__(self, directory: str):
        self.directory = directory
        self.assets: list[AssetDef] = []
        self.by_name: dict[str, AssetDef] = {}
        self.reload()

    def reload(self) -> None:
        self.assets.clear()
        self.by_name.clear()
        if not os.path.isdir(self.directory):
            return
        for fn in sorted(os.listdir(self.directory)):
            if not fn.lower().endswith(".json"):
                continue
            with open(os.path.join(self.directory, fn), encoding="utf-8") as f:
                asset = AssetDef(json.load(f), os.path.join(self.directory, fn))
            self.assets.append(asset)
            self.by_name[asset.name] = asset
        self.assets.sort(key=lambda a: (a.category, a.name))

    def instantiate(self, name: str, entity_name: str | None = None) -> Entity:
        return self.by_name[name].instantiate(entity_name)


def _vec(v: Vec3) -> list[float]:
    return [round(v.x, 4), round(v.y, 4), round(v.z, 4)]


def _entity_dict(e: Entity) -> dict:
    d = {"asset": e.asset_name, "name": e.name,
         "position": _vec(e.transform.position),
         "rotation": _vec(e.transform.rotation),
         "scale": _vec(e.transform.scale)}
    if e.light is not None:
        light = {k: getattr(e.light, k) for k in _LIGHT_PROPS if hasattr(e.light, k)}
        light["color"] = list(e.light.color)
        # a Flicker overwrites intensity every frame; persist its base value
        for b in e.behaviors:
            if isinstance(b, behaviors_mod.Flicker) and hasattr(b, "base"):
                light["intensity"] = b.base
        d["light"] = light
    if e.material is not None:
        d["material"] = e.material.to_dict()
    return d


def save_scene(scene: Scene, camera: Camera, path: str) -> None:
    dl = scene.light
    data = {
        "camera": {"position": _vec(camera.position),
                   "yaw": round(camera.yaw, 4), "pitch": round(camera.pitch, 4)},
        "directional_light": {"direction": _vec(dl.direction), "ambient": dl.ambient,
                              "color": list(dl.color), "intensity": dl.intensity},
        "fog": ({"color": list(scene.fog.color), "start": scene.fog.start,
                 "end": scene.fog.end} if scene.fog else None),
        "sky": ([list(scene.sky[0]), list(scene.sky[1])] if scene.sky else None),
        "background": list(scene.background),
        "enable_shadows": scene.enable_shadows,
        "entities": [_entity_dict(e) for e in scene.entities
                     if e.asset_name is not None],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_scene(path: str, library: AssetLibrary,
               camera: Camera | None = None) -> Scene:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    dl = data.get("directional_light", {})
    scene = Scene(
        light=DirectionalLight(Vec3(*dl.get("direction", [-0.5, -1.0, -0.3])),
                               ambient=dl.get("ambient", 0.3),
                               color=tuple(dl.get("color", [255, 255, 255])),
                               intensity=dl.get("intensity", 1.0)),
        fog=(Fog(tuple(data["fog"]["color"]), data["fog"]["start"], data["fog"]["end"])
             if data.get("fog") else None),
        sky=(tuple(tuple(c) for c in data["sky"]) if data.get("sky") else None),
        background=tuple(data.get("background", [12, 14, 20])),
        enable_shadows=data.get("enable_shadows", True),
    )
    for spec in data.get("entities", []):
        entity = library.instantiate(spec["asset"], spec.get("name"))
        t = entity.transform
        t.position = Vec3(*spec.get("position", [0, 0, 0]))
        t.rotation = Vec3(*spec.get("rotation", [0, 0, 0]))
        t.scale = Vec3(*spec.get("scale", [1, 1, 1]))
        if "light" in spec and entity.light is not None:
            for key, value in spec["light"].items():
                if key in _LIGHT_PROPS:
                    setattr(entity.light, key,
                            tuple(value) if key == "color" else value)
        if "material" in spec and entity.mesh is not None:
            entity.material = MaterialGraph.from_dict(spec["material"])
            entity.material.apply(entity.mesh)
        scene.add(entity)

    if camera is not None and "camera" in data:
        c = data["camera"]
        camera.position = Vec3(*c.get("position", [0, 4, 15]))
        camera.yaw = c.get("yaw", 0.0)
        camera.pitch = c.get("pitch", 0.0)
    return scene
