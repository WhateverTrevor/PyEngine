"""Headless smoke test: asset instantiation, placement math, scene round-trip."""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.gettempdir()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, REPO)

import numpy as np

import engine
from editor import base_height, build_starter_scene

lib = engine.AssetLibrary(os.path.join(REPO, "assets"))
print("assets:", [a.name for a in lib.assets])
# presence check, not an exact count — new assets shouldn't break the suite
for required in ("Sky Sphere", "Torch", "Spotlight", "Crate", "Stone Floor",
                 "Wall Segment", "Ghost", "Sun", "Fog Volume"):
    assert required in lib.by_name, f"missing asset: {required}"

# every asset instantiates, and placement height puts the mesh base at y=0
for a in lib.assets:
    e = a.instantiate()
    h = base_height(e)
    if e.mesh is not None:
        assert abs(float(e.mesh.vertices[:, 1].min()) + h) < 1e-9, a.name
print("instantiate + base_height OK")

# starter scene builds; picking hits the floor from above
scene = build_starter_scene(engine, lib)
origin = np.array([0.0, 5.0, 0.0])
down = np.array([0.0, -1.0, 0.0])
ent, t = engine.pick_entity(scene, origin, down)
assert ent is not None and ent.asset_name == "Stone Floor" and abs(t - 5.0) < 1e-6
print("pick_entity OK:", ent.name, "t=", round(t, 3))

# mouse_ray points at what's in the screen center
cam = engine.Camera(position=engine.Vec3(0, 2, 8))
ray = cam.mouse_ray(640, 360, 1280, 720)
assert np.allclose(ray, [0, 0, -1], atol=1e-9)
print("mouse_ray OK")

# save -> load round-trip preserves entities and transforms
n_assets = sum(1 for e in scene.entities if e.asset_name)
path = os.path.join(TMP, "roundtrip.json")
engine.save_scene(scene, cam, path)
scene2 = engine.load_scene(path, lib, engine.Camera())
n2 = sum(1 for e in scene2.entities if e.asset_name)
assert n2 == n_assets, (n_assets, n2)
g1 = next(e for e in scene.entities if e.asset_name == "Ghost")
g2 = next(e for e in scene2.entities if e.asset_name == "Ghost")
assert abs(g1.transform.position.x - g2.transform.position.x) < 1e-3
assert not g2.casts_shadow  # asset flags survive the round-trip
print(f"scene round-trip OK ({n_assets} entities)")

# quad meshes: cube is 6 quad faces, triangulated to 12 for ray tracing
c = engine.cube(1.0)
assert c.faces.shape == (6, 4) and c.tri_faces.shape == (12, 3)
board = engine.checkerboard(4, 1.0)
assert board.faces.shape == (16, 4)          # one quad per square
assert np.allclose(board.normals[:, 1], 1.0)  # all +y
print("quad meshes OK (cube 6 quads, checkerboard 1 quad/square)")

# shadow tracer: a box between light and floor should darken faces under it
scene3 = engine.Scene()
floor = engine.Entity("floor", mesh=engine.checkerboard(16, 0.5))
floor.casts_shadow = False
blocker = engine.Entity("blocker", mesh=engine.cube(1.5), position=engine.Vec3(0, 2.0, 0))
lamp = engine.Entity("lamp", light=engine.PointLight(range=20, radius=0.3, shadow_samples=8),
                     position=engine.Vec3(0, 5.0, 0))
for e in (floor, blocker, lamp):
    scene3.add(e)
tracer = engine.ShadowTracer()
tracer.refresh(scene3)
centroids = floor.mesh.vertices[floor.mesh.faces].mean(axis=1)
normals = floor.mesh.normals
active = np.ones(len(centroids), bool)
f = tracer.shadow_factors(floor, lamp.light, np.array([0.0, 5.0, 0.0]),
                          centroids, normals, active)
center = np.linalg.norm(centroids[:, [0, 2]], axis=1)
under, far = f[center < 0.45], f[center > 2.5]
assert under.mean() < 0.2, under.mean()   # hard shadow under the box
assert far.mean() > 0.9, far.mean()       # lit far away
fractional = ((f > 0.1) & (f < 0.9)).sum()
assert fractional >= 4, fractional       # soft edge: partially-lit faces exist
print(f"ray-traced shadows OK (under={under.mean():.2f}, "
      f"penumbra faces={fractional}, lit={far.mean():.2f})")

# collision: a sphere can't pass through a wall, and slides along it
from engine.behaviors import resolve_collisions
scene4 = engine.Scene()
wall = engine.Entity("wall", mesh=engine.box(4.0, 3.0, 0.4),
                     position=engine.Vec3(0, 1.5, 0))
scene4.add(wall)
inside = resolve_collisions(scene4, engine.Vec3(0.0, 1.5, 0.05), 0.45)
assert abs(inside.z) >= 0.2 + 0.45 - 1e-6, inside  # pushed out of the slab
near_wall = resolve_collisions(scene4, engine.Vec3(0.3, 1.5, 0.5), 0.45)
assert near_wall.z >= 0.2 + 0.45 - 1e-6, near_wall
assert abs(near_wall.x - 0.3) < 1e-6                # slides: x preserved
free = resolve_collisions(scene4, engine.Vec3(5.0, 1.5, 5.0), 0.45)
assert abs(free.x - 5.0) < 1e-9                     # untouched away from wall
ghost_scene = engine.Scene()
g = lib.instantiate("Ghost")
g.transform.position = engine.Vec3(0, 1, 0)
ghost_scene.add(g)
through = resolve_collisions(ghost_scene, engine.Vec3(0, 1, 0.1), 0.45)
assert abs(through.z - 0.1) < 1e-9                  # ghosts are walk-through
print("collision OK (blocked, slid, ghost pass-through)")

# light edits persist: change intensity/ies/outer, save, reload
t1 = next(e for e in scene.entities if e.asset_name == "Torch")
t1.light.intensity = 3.3
for b in t1.behaviors:
    if isinstance(b, engine.behaviors.Flicker):
        b.base = 3.3
t1.light.ies = "batwing"
t1.light.color = (10, 250, 30)
engine.save_scene(scene, cam, path)
scene5 = engine.load_scene(path, lib, engine.Camera())
t2 = next(e for e in scene5.entities if e.transform.position.x == t1.transform.position.x
          and e.asset_name == "Torch")
assert abs(t2.light.intensity - 3.3) < 1e-6
assert t2.light.ies == "batwing" and t2.light.color == (10, 250, 30)
print("light overrides round-trip OK")

# spotlight asset: cone mesh + spot with penumbra/throw, per-pixel render runs
spot = lib.instantiate("Spotlight")
assert isinstance(spot.light, engine.SpotLight)
assert spot.light.outer > spot.light.inner and spot.light.ies == "spot_soft"
scene3.add(spot)
spot.transform.position = engine.Vec3(2, 3, 2)
import pygame
from engine.renderer import Renderer
r = Renderer()
r.per_pixel = True
r.render_scale = 3
tracer.refresh(scene3)
target = pygame.Surface((300, 200))
r.render(target, scene3, engine.Camera(position=engine.Vec3(0, 3, 10)), tracer)
assert r.stats["mode"].startswith("per-pixel") and r.stats["triangles"] > 0
r.per_pixel = False
r.render(target, scene3, engine.Camera(position=engine.Vec3(0, 3, 10)), tracer)
assert r.stats["mode"] == "flat"
print("spotlight asset + per-pixel/flat render paths OK")

# environment: HDRI loads, ambient cube favors the sky, sky-sphere asset works
env_img = engine.load_hdr(os.path.join(REPO, "assets", "hdri", "night_sky.hdr"))
assert env_img.shape == (256, 512, 3) and env_img.max() > 1.0  # true HDR values
sky_ent = lib.instantiate("Sky Sphere")
assert sky_ent.environment is not None
amb_up = sky_ent.environment.ambient(np.array([[0.0, 1.0, 0.0]]))[0]
amb_dn = sky_ent.environment.ambient(np.array([[0.0, -1.0, 0.0]]))[0]
assert amb_up.sum() > amb_dn.sum() * 3       # sky is brighter than the ground
moon_dir = np.array([[np.sin(0.9) * np.cos(1.9), np.cos(0.9), np.sin(0.9) * np.sin(1.9)]])
assert sky_ent.environment.sample(moon_dir)[0].max() > 1.0  # moon is HDR-bright
scene3.add(sky_ent)
r.per_pixel = True
r.render(target, scene3, engine.Camera(position=engine.Vec3(0, 3, 10)), tracer)
print("HDRI environment OK (load, ambient cube, sky sampling)")

# FBX: write a real binary FBX cube, import it as an asset, instantiate it
import struct


def _enc_prop(v):
    if isinstance(v, np.ndarray) and v.dtype == np.float64:
        return b"d" + struct.pack("<III", len(v), 0, len(v) * 8) + v.tobytes()
    if isinstance(v, np.ndarray) and v.dtype == np.int32:
        return b"i" + struct.pack("<III", len(v), 0, len(v) * 4) + v.tobytes()
    if isinstance(v, str):
        raw = v.encode()
        return b"S" + struct.pack("<I", len(raw)) + raw
    if isinstance(v, float):
        return b"D" + struct.pack("<d", v)
    if isinstance(v, int):
        return b"L" + struct.pack("<q", v)
    raise TypeError(v)


def _build_node(node, start):
    name, props, children = node
    prop_data = b"".join(_enc_prop(p) for p in props)
    pos = start + 13 + len(name) + len(prop_data)
    child_data = b""
    if children:
        for ch in children:
            cb = _build_node(ch, pos)
            child_data += cb
            pos += len(cb)
        child_data += b"\x00" * 13
        pos += 13
    header = struct.pack("<IIIB", pos, len(props), len(prop_data), len(name))
    return header + name.encode() + prop_data + child_data


cube_verts = (np.array([[-50, -50, -50], [50, -50, -50], [50, 50, -50], [-50, 50, -50],
                        [-50, -50, 50], [50, -50, 50], [50, 50, 50], [-50, 50, 50]],
                       dtype=np.float64) + 50.0).reshape(-1)  # cm, off-center on purpose
quads = [(4, 5, 6, 7), (1, 0, 3, 2), (0, 4, 7, 3), (5, 1, 2, 6), (7, 6, 2, 3), (0, 1, 5, 4)]
pvi = []
for q in quads:
    pvi += [q[0], q[1], q[2], -(q[3] + 1)]

GID, MODEL_ID, MAT_RED, MAT_BLUE = 1001, 2001, 3001, 3002


def _material(mid, mname, rgb):
    p = ("P", ["DiffuseColor", "Color", "", "A", rgb[0], rgb[1], rgb[2]], [])
    return ("Material", [mid, f"Material::{mname}", ""], [("Properties70", [], [p])])


geometry = ("Geometry", [GID, "Geometry::cube", "Mesh"], [
    ("Vertices", [cube_verts], []),
    ("PolygonVertexIndex", [np.array(pvi, dtype=np.int32)], []),
    ("LayerElementMaterial", [], [
        ("MappingInformationType", ["ByPolygon"], []),
        ("Materials", [np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)], []),
    ]),
])
objects = ("Objects", [], [geometry,
                           ("Model", [MODEL_ID, "Model::cube", "Mesh"], []),
                           _material(MAT_RED, "red", (1.0, 0.1, 0.1)),
                           _material(MAT_BLUE, "blue", (0.1, 0.2, 1.0))])
connections = ("Connections", [], [
    ("C", ["OO", GID, MODEL_ID], []),
    ("C", ["OO", MAT_RED, MODEL_ID], []),
    ("C", ["OO", MAT_BLUE, MODEL_ID], []),
])

fbx_path = os.path.join(TMP, "test_cube.fbx")
header = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400)
body = _build_node(objects, len(header))
body += _build_node(connections, len(header) + len(body))
with open(fbx_path, "wb") as fh:
    fh.write(header + body + b"\x00" * 13)

name = engine.import_fbx(fbx_path, lib.directory)
lib.reload()
imported = lib.instantiate(name)
assert imported.mesh is not None
assert len(imported.mesh.faces) == 6 and imported.mesh.tri_faces.shape == (12, 3)
assert imported.mesh.bound < 4.1                       # cm -> m unit scaling
assert abs(imported.mesh.vertices[:, 1].min()) < 1e-5  # grounded at y=0
fc = imported.mesh.face_colors
assert np.allclose(fc[0], [255, 25.5, 25.5], atol=1.0)   # red material faces
assert np.allclose(fc[3], [25.5, 51.0, 255], atol=1.0)   # blue material faces
scene3.add(imported)
r.render(target, scene3, engine.Camera(position=engine.Vec3(0, 3, 10)), tracer)
print(f"FBX import OK ('{name}': 6 quads, 2 materials, unit-scaled, grounded)")

# material graph: checker of red/grey, cycle rejection, serialization, baking
g = engine.MaterialGraph()
checker = g.add("checker", (100, 100))
red = g.add("color", (10, 60))
g.nodes[red]["params"].update(r=1.0, g=0.15, b=0.15)
assert g.connect(red, checker, "a")
assert g.connect(checker, g.output_id(), "base_color")
assert not g.connect(g.output_id(), checker, "b")   # cycle rejected
board2 = engine.checkerboard(4, 1.0)
baked = g.evaluate(board2)
assert len(np.unique(baked.round(1), axis=0)) >= 2  # alternating colors
g2 = engine.MaterialGraph.from_dict(g.to_dict())
assert np.allclose(g2.evaluate(board2), baked)      # serialization round-trip
crate_e = next(e for e in scene.entities if e.asset_name == "Crate")
crate_e.material = g
g.apply(crate_e)  # apply() takes the entity since the sky-material change
engine.save_scene(scene, cam, path)
scene6 = engine.load_scene(path, lib, engine.Camera())
crate2 = next(e for e in scene6.entities if e.asset_name == "Crate")
assert crate2.material is not None
assert np.allclose(crate2.mesh.face_colors, crate_e.mesh.face_colors)
print("material graph OK (eval, cycle guard, serialize, scene round-trip)")

# clean up the imported test asset so it doesn't pollute the library
os.remove(os.path.join(lib.directory, "test_cube.json"))
os.remove(os.path.join(lib.directory, "models", "test_cube.npz"))
lib.reload()
print("ALL SMOKE TESTS PASSED")
