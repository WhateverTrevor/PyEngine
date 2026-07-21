"""Scene graph: entities, transforms, and attachable behaviors."""
from __future__ import annotations

from .lighting import DirectionalLight, Fog, PointLight
from .math3d import Vec3, rotation_x, rotation_y, rotation_z, scaling, translation
from .mesh import Mesh


class Behavior:
    """Attachable per-entity logic. Subclass and override update()."""

    _started = False

    def start(self, entity: "Entity", engine) -> None:
        pass

    def update(self, entity: "Entity", dt: float, engine) -> None:
        pass


class Transform:
    def __init__(self, position: Vec3 | None = None, rotation: Vec3 | None = None,
                 scale: Vec3 | None = None):
        self.position = position or Vec3()
        self.rotation = rotation or Vec3()   # Euler radians: x=pitch, y=yaw, z=roll
        self.scale = scale or Vec3(1.0, 1.0, 1.0)
        self._key = None
        self._mat = None

    def matrix(self):
        """4x4 model matrix, memoized (callers must not mutate the result)."""
        p, r, s = self.position, self.rotation, self.scale
        key = (p.x, p.y, p.z, r.x, r.y, r.z, s.x, s.y, s.z)
        if key != self._key:
            self._mat = (translation(p)
                         @ rotation_y(r.y)
                         @ rotation_x(r.x)
                         @ rotation_z(r.z)
                         @ scaling(s))
            self._key = key
        return self._mat


class Entity:
    def __init__(self, name: str = "entity", mesh: Mesh | None = None,
                 position: Vec3 | None = None, rotation: Vec3 | None = None,
                 scale: Vec3 | None = None, light: PointLight | None = None):
        self.name = name
        self.mesh = mesh
        self.transform = Transform(position, rotation, scale)
        self.behaviors: list[Behavior] = []
        self.visible = True
        self.light = light               # PointLight/SpotLight carried by this entity
        self.light_offset = Vec3()       # light position in local space
        self.casts_shadow = True         # participates as a shadow occluder
        self.collidable = True           # blocks the player (see FlyController)
        self.environment = None          # Environment (HDRI sky + ambient) carrier
        self.material = None             # MaterialGraph baked onto this mesh
        self.material_asset = None       # name of the MaterialAsset this material was assigned from, or None
        self.sun = None                  # SunDisc: sky-disc + shadow tuning (see lighting.py)
        self.fog_volume = None           # FogVolume: local volumetric fog box (see lighting.py)
        self.asset_name: str | None = None  # set when spawned from an asset file
        # distance-based LOD (see engine/lod.py): `mesh` is always LOD0 --
        # collision, ray-traced shadows/GI, and gizmo/AABB math all keep
        # reading `.mesh` unchanged. `lod_meshes` are LOD1, LOD2, ... in
        # decreasing detail (empty for every built-in asset and any mesh at
        # or below lod.LOD_FACE_THRESHOLD); `lod_index` is the currently
        # selected level, updated once/frame by lod.update_scene_lods.
        self.lod_meshes: list[Mesh] = []
        self.lod_index = 0

    def add_behavior(self, behavior: Behavior) -> "Entity":
        self.behaviors.append(behavior)
        return self

    def render_mesh(self) -> Mesh | None:
        """The mesh the rasterizer should draw this frame: `mesh` (LOD0)
        unless LOD data exists and `lod_index` selects a coarser level. Ray-
        traced shadows/GI never call this -- they always read `.mesh`
        directly (see raytrace.py), which is the whole point: occlusion
        stays pinned to the full mesh regardless of camera-distance LOD."""
        if not self.lod_meshes or self.lod_index <= 0:
            return self.mesh
        return self.lod_meshes[min(self.lod_index, len(self.lod_meshes)) - 1]


class Scene:
    def __init__(self, light: DirectionalLight | None = None, fog: Fog | None = None,
                 background=(12, 14, 20), sky: tuple | None = None,
                 enable_shadows: bool = True):
        self.entities: list[Entity] = []
        self.light = light or DirectionalLight(Vec3(-0.5, -1.0, -0.3))
        self.fog = fog
        self.background = background
        self.sky = sky  # (top_color, horizon_color) vertical gradient, or None
        self.enable_shadows = enable_shadows
        # one-bounce global illumination (see raytrace.GITracer); persisted with
        # the scene, controlled from the Sun entity's Details panel
        self.gi = {"enabled": False, "intensity": 1.0, "samples": 16}

    def add(self, entity: Entity) -> Entity:
        self.entities.append(entity)
        return entity

    def remove(self, entity: Entity) -> None:
        if entity in self.entities:
            self.entities.remove(entity)

    def update(self, dt: float, engine) -> None:
        for entity in tuple(self.entities):  # copy: behaviors may add/remove
            for behavior in entity.behaviors:
                if not behavior._started:
                    behavior._started = True
                    behavior.start(entity, engine)
                behavior.update(entity, dt, engine)
