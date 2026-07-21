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
from .lighting import DirectionalLight, Fog, FogVolume, PointLight, SpotLight, SunDisc
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

# SunDisc / FogVolume attributes the editor can change and scenes persist
_SUN_PROPS = ("disc_size", "disc_softness", "glow", "enabled",
             "shadow_softness", "shadow_depth", "shadow_samples")
_FOG_VOLUME_PROPS = ("density", "color", "height_falloff", "enabled")


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
                face_uvs = (data["face_uvs"].astype(np.float64)
                           if "face_uvs" in data else None)
                # PBR arrays are optional in the npz -- absent for plain
                # FBX-imported geometry (diffuse-only import, no PBR source);
                # Mesh supplies the backward-compat defaults when omitted.
                face_roughness = (data["face_roughness"].astype(np.float64)
                                  if "face_roughness" in data else None)
                face_metallic = (data["face_metallic"].astype(np.float64)
                                 if "face_metallic" in data else None)
                face_emissive = (data["face_emissive"].astype(np.float64)
                                 if "face_emissive" in data else None)
                entity.mesh = mesh_mod.Mesh(
                    data["vertices"], [tuple(f) for f in data["faces"]],
                    base_color=spec.get("color", (170, 170, 175)),
                    face_colors=face_colors, face_uvs=face_uvs,
                    face_roughness=face_roughness, face_metallic=face_metallic,
                    face_emissive=face_emissive)
                # distance-based LOD (see engine/lod.py import_fbx's
                # generate_lods=True path) -- absent for every built-in and
                # any import below the decimation threshold, so this is a
                # no-op for them (entity.lod_meshes stays [] -> render_mesh()
                # always returns entity.mesh, byte-identical to before).
                n_lods = int(data["lod_levels"]) if "lod_levels" in data else 0
                for i in range(1, n_lods + 1):
                    lm = mesh_mod.Mesh(
                        data[f"lod{i}_vertices"],
                        [tuple(f) for f in data[f"lod{i}_faces"]],
                        face_colors=data[f"lod{i}_face_colors"].astype(np.float64),
                        face_roughness=data[f"lod{i}_face_roughness"].astype(np.float64),
                        face_metallic=data[f"lod{i}_face_metallic"].astype(np.float64),
                        face_emissive=data[f"lod{i}_face_emissive"].astype(np.float64),
                        face_opacity=data[f"lod{i}_face_opacity"].astype(np.float64))
                    lm.lod_source_faces = data[f"lod{i}_source_faces"].astype(np.int64)
                    entity.lod_meshes.append(lm)
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

        if "sun" in d:
            entity.sun = SunDisc(**dict(d["sun"]))

        if "fog_volume" in d:
            entity.fog_volume = FogVolume(**_tupled(dict(d["fog_volume"])))

        for b in d.get("behaviors", []):
            spec = dict(b)
            cls = getattr(behaviors_mod, spec.pop("type"))
            kwargs = {k: Vec3(*v) if isinstance(v, list) and len(v) == 3 else v
                      for k, v in spec.items()}
            entity.add_behavior(cls(**kwargs))

        return entity


class MaterialAsset:
    """A reusable material graph saved to assets/materials/<name>.json --
    dragged from the content browser onto a mesh entity's Details panel to
    assign it (see MaterialEditorUI's "Save as Asset" button and the
    Details material slot in editor.py)."""

    def __init__(self, name: str, graph_dict: dict, path: str):
        self.name = name
        self.graph_dict = graph_dict
        self.path = path

    def graph(self) -> "MaterialGraph":
        """A fresh, independent copy -- assigning never aliases the asset's
        own graph, so editing an assigned entity's material doesn't mutate
        (or get mutated by) the library asset or other entities using it."""
        return MaterialGraph.from_dict(self.graph_dict)


class AssetLibrary:
    """Owns the on-disk assets/*.json plus the content-browser folder tree.

    The folder tree is a separate manifest (folders.json) rather than a field
    on each asset JSON: assets stay self-contained/portable, and folders are
    purely an editor organization concern layered on top. `folders` maps
    folder id (str) -> {"name", "parent"} (parent is a folder id or None for
    root); `folder_of` maps asset name -> folder id, absent/None == root.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self.assets: list[AssetDef] = []
        self.by_name: dict[str, AssetDef] = {}
        self.materials: list[MaterialAsset] = []
        self.material_by_name: dict[str, MaterialAsset] = {}
        self.folders: dict[str, dict] = {}
        self.folder_of: dict[str, str] = {}
        self._next_folder_id = 1
        from . import texture as texture_mod
        texture_mod.set_texture_root(directory)  # material TextureSample nodes resolve here
        self.reload()

    def _materials_dir(self) -> str:
        return os.path.join(self.directory, "materials")

    def reload(self) -> None:
        self.assets.clear()
        self.by_name.clear()
        if os.path.isdir(self.directory):
            for fn in sorted(os.listdir(self.directory)):
                if not fn.lower().endswith(".json") or fn == "folders.json":
                    continue
                with open(os.path.join(self.directory, fn), encoding="utf-8") as f:
                    asset = AssetDef(json.load(f), os.path.join(self.directory, fn))
                self.assets.append(asset)
                self.by_name[asset.name] = asset
        self.assets.sort(key=lambda a: (a.category, a.name))
        self.materials.clear()
        self.material_by_name.clear()
        mdir = self._materials_dir()
        if os.path.isdir(mdir):
            for fn in sorted(os.listdir(mdir)):
                if not fn.lower().endswith(".json"):
                    continue
                with open(os.path.join(mdir, fn), encoding="utf-8") as f:
                    data = json.load(f)
                mat = MaterialAsset(data["name"], data["graph"], os.path.join(mdir, fn))
                self.materials.append(mat)
                self.material_by_name[mat.name] = mat
        self.materials.sort(key=lambda m: m.name)
        self._load_folders()

    def save_material(self, name: str, graph: "MaterialGraph") -> MaterialAsset:
        """Save (or overwrite) `name` as a reusable material asset."""
        os.makedirs(self._materials_dir(), exist_ok=True)
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip() or "material"
        path = os.path.join(self._materials_dir(), safe.replace(" ", "_").lower() + ".json")
        graph_dict = graph.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": name, "graph": graph_dict}, f, indent=2)
        mat = MaterialAsset(name, graph_dict, path)
        self.material_by_name[name] = mat
        if mat not in self.materials:
            self.materials = [m for m in self.materials if m.name != name] + [mat]
            self.materials.sort(key=lambda m: m.name)
        return mat

    def instantiate(self, name: str, entity_name: str | None = None) -> Entity:
        return self.by_name[name].instantiate(entity_name)

    # ---- folder tree ----
    def _folders_path(self) -> str:
        return os.path.join(self.directory, "folders.json")

    def _load_folders(self) -> None:
        self.folders = {}
        self.folder_of = {}
        self._next_folder_id = 1
        path = self._folders_path()
        assignments = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for fid, spec in data.get("folders", {}).items():
                    self.folders[fid] = {"name": spec.get("name", "Folder"),
                                         "parent": spec.get("parent")}
                self._next_folder_id = int(data.get("next_id", 1))
                assignments = data.get("assignments", {})
            except (OSError, ValueError):
                self.folders = {}
        # keep only assignments that point at real folders and real assets --
        # deleting a folder manually (or an asset going away) shouldn't crash
        # the browser, it should just fall the asset back to root.
        self.folder_of = {name: fid for name, fid in assignments.items()
                          if fid in self.folders and name in self.by_name}

    def save_folders(self) -> None:
        data = {"folders": self.folders, "assignments": self.folder_of,
                "next_id": self._next_folder_id}
        try:
            with open(self._folders_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def create_folder(self, name: str, parent: str | None = None) -> str:
        fid = str(self._next_folder_id)
        self._next_folder_id += 1
        self.folders[fid] = {"name": name, "parent": parent}
        return fid

    def rename_folder(self, folder_id: str, name: str) -> None:
        if folder_id in self.folders and name:
            self.folders[folder_id]["name"] = name

    def folder_children(self, parent: str | None) -> list[str]:
        return sorted((fid for fid, f in self.folders.items() if f["parent"] == parent),
                     key=lambda fid: self.folders[fid]["name"].lower())

    def set_asset_folder(self, asset_name: str, folder_id: str | None) -> None:
        if folder_id is None or folder_id not in self.folders:
            self.folder_of.pop(asset_name, None)
        else:
            self.folder_of[asset_name] = folder_id

    def assets_in(self, folder_id: str | None) -> list[AssetDef]:
        return [a for a in self.assets if self.folder_of.get(a.name) == folder_id]


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
        if e.material_asset:
            d["material_asset"] = e.material_asset
    if e.sun is not None:
        d["sun"] = {k: getattr(e.sun, k) for k in _SUN_PROPS}
    if e.fog_volume is not None:
        fv = e.fog_volume
        d["fog_volume"] = {"density": fv.density, "color": list(fv.color),
                           "height_falloff": fv.height_falloff, "enabled": fv.enabled}
    return d


def save_scene(scene: Scene, camera: Camera, path: str) -> None:
    dl = scene.light
    data = {
        "camera": {"position": _vec(camera.position),
                   "yaw": round(camera.yaw, 4), "pitch": round(camera.pitch, 4)},
        "directional_light": {"direction": _vec(dl.direction), "ambient": dl.ambient,
                              "color": list(dl.color), "intensity": dl.intensity},
        "fog": ({"color": list(scene.fog.color), "start": scene.fog.start,
                 "end": scene.fog.end, "height_falloff": scene.fog.height_falloff,
                 "sun_scatter": scene.fog.sun_scatter} if scene.fog else None),
        "sky": ([list(scene.sky[0]), list(scene.sky[1])] if scene.sky else None),
        "background": list(scene.background),
        "enable_shadows": scene.enable_shadows,
        "gi": dict(scene.gi),
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
    fog_data = data.get("fog")
    scene = Scene(
        light=DirectionalLight(Vec3(*dl.get("direction", [-0.5, -1.0, -0.3])),
                               ambient=dl.get("ambient", 0.3),
                               color=tuple(dl.get("color", [255, 255, 255])),
                               intensity=dl.get("intensity", 1.0)),
        fog=(Fog(tuple(fog_data["color"]), fog_data["start"], fog_data["end"],
                height_falloff=fog_data.get("height_falloff", 0.0),
                sun_scatter=fog_data.get("sun_scatter", 0.0))
             if fog_data else None),
        sky=(tuple(tuple(c) for c in data["sky"]) if data.get("sky") else None),
        background=tuple(data.get("background", [12, 14, 20])),
        enable_shadows=data.get("enable_shadows", True),
    )
    scene.gi = dict(data.get("gi", scene.gi))
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
        if "sun" in spec and entity.sun is not None:
            for key, value in spec["sun"].items():
                if key in _SUN_PROPS:
                    setattr(entity.sun, key, value)
        if "fog_volume" in spec and entity.fog_volume is not None:
            fv, fvd = entity.fog_volume, spec["fog_volume"]
            fv.density = fvd.get("density", fv.density)
            fv.color = tuple(fvd.get("color", fv.color))
            fv.height_falloff = fvd.get("height_falloff", fv.height_falloff)
            fv.enabled = fvd.get("enabled", fv.enabled)
        if "material" in spec and (entity.mesh is not None or entity.environment is not None):
            entity.material = MaterialGraph.from_dict(spec["material"])
            entity.material_asset = spec.get("material_asset")
            entity.material.apply(entity)
        scene.add(entity)

    if camera is not None and "camera" in data:
        c = data["camera"]
        camera.position = Vec3(*c.get("position", [0, 4, 15]))
        camera.yaw = c.get("yaw", 0.0)
        camera.pitch = c.get("pitch", 0.0)
    return scene
