"""Judge checks: distance-based LOD (level of detail).

Covers engine/lod.py's vertex-clustering decimation + generate_lods, .npz
storage round-trip, the import dialog's "Generate LODs" toggle, runtime
distance selection with hysteresis, and renderer/raytrace LOD-pinning parity
(rasterization draws the selected LOD; ray-traced shadow/GI occlusion is
pinned to `Entity.shadow_mesh()` -- LOD0 for ordinary meshes, but the
COARSEST available LOD for high-poly ones, see checks 10-13 below and
scene.py/raytrace.py/gpu_geometry.py). High-poly fixtures are built
in-memory (a subdivided icosphere) -- this suite never touches the user's
real assets/gat.json, assets/models/gat.npz, or assets/folders.json.
"""
import os
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np

import engine
from engine import lod
from engine.mesh import icosphere

TMP = tempfile.gettempdir()

# ----------------------------------------------------------------------------
# fixtures: high-poly meshes built in-memory, never touching assets/gat.*
# ----------------------------------------------------------------------------
hp_mesh = icosphere(radius=1.0, subdivisions=3, color=(180, 120, 90))  # 1280 faces
assert len(hp_mesh.faces) == 1280, len(hp_mesh.faces)

small_mesh = engine.cube(1.0)  # 6 faces, well under LOD_FACE_THRESHOLD

# ========================================================================
# 1. decimate() reduces face count toward the target and preserves bbox
# ========================================================================
target = 320  # ~25%
lod1 = lod.decimate(hp_mesh, target)
assert len(lod1.faces) < len(hp_mesh.faces), \
    f"decimate must reduce face count: {len(lod1.faces)} vs {len(hp_mesh.faces)}"
assert len(lod1.faces) < target * 2.5, \
    f"decimate landed too far from target {target}: got {len(lod1.faces)}"
assert np.all(lod1.aabb_min >= hp_mesh.aabb_min - 1e-6), "decimated bbox min must stay inside the original"
assert np.all(lod1.aabb_max <= hp_mesh.aabb_max + 1e-6), "decimated bbox max must stay inside the original"
print(f"1. decimate OK: {len(hp_mesh.faces)} -> {len(lod1.faces)} faces "
     f"(target {target}), bbox contained within original")

# ========================================================================
# 2. generate_lods: decreasing face counts, LOD0 is the same object
# ========================================================================
lods = lod.generate_lods(hp_mesh)
assert lods[0] is hp_mesh, "LOD0 must be the original mesh object, unchanged"
counts = [len(m.faces) for m in lods]
assert counts == sorted(counts, reverse=True), f"LOD face counts must strictly decrease: {counts}"
assert len(counts) > 1, "a 1280-face mesh must produce at least one extra LOD level"
print(f"2. generate_lods OK: face counts {counts}")

# ========================================================================
# 3. colors carried: every output face's color is one of the source mesh's
#    original face colors (representative-source carry, not synthesized);
#    lod_source_faces indices are valid back-references into LOD0
# ========================================================================
src_colors = {tuple(c) for c in hp_mesh.face_colors.tolist()}
for i, m in enumerate(lods[1:], start=1):
    bad = [tuple(c) for c in m.face_colors.tolist() if tuple(c) not in src_colors]
    assert not bad, f"LOD{i} face_colors not sourced from the original mesh: {bad[:3]}"
    assert hasattr(m, "lod_source_faces"), f"LOD{i} must carry a lod_source_faces map"
    assert len(m.lod_source_faces) == len(m.faces)
    assert m.lod_source_faces.min() >= 0 and m.lod_source_faces.max() < len(hp_mesh.faces)
print("3. face_colors carried from representative source faces OK; "
     "lod_source_faces indices valid")

# ========================================================================
# 4. small-mesh passthrough: below-threshold meshes are untouched
# ========================================================================
small_lods = lod.generate_lods(small_mesh)
assert small_lods == [small_mesh], "a 6-face mesh must return [mesh] only"
assert lod.LOD_FACE_THRESHOLD >= len(small_mesh.faces)
print("4. small-mesh passthrough OK: 6-face cube returns [mesh] unchanged")

# ========================================================================
# 5. .npz round-trip through the real import_fbx()/AssetDef.instantiate()
#    path -- extract_geometry (the binary FBX parser) is monkeypatched to
#    hand back the high-poly fixture directly, so this exercises the exact
#    production write/read code without building a giant binary FBX file.
#    A fresh temp assets dir is used throughout -- the real assets/ dir
#    (and gat.*) is never touched.
# ========================================================================
import unittest.mock as um

from engine import fbx as fbx_mod
from engine.assets import AssetLibrary

tmp_assets = tempfile.mkdtemp(prefix="pyengine_lod_assets_")
fake_path = os.path.join(TMP, "fake_highpoly.fbx")


def _fake_extract(path):
    m = icosphere(radius=1.0, subdivisions=3, color=(90, 140, 200))
    faces = [tuple(int(i) for i in f) for f in m.faces.tolist()]
    colors = m.face_colors / 255.0  # extract_geometry's contract: 0..1
    return m.vertices.copy(), faces, colors, None


with um.patch.object(fbx_mod, "extract_geometry", _fake_extract):
    asset_name = fbx_mod.import_fbx(fake_path, tmp_assets, generate_lods=True)

npz_path = os.path.join(tmp_assets, "models", "fake_highpoly.npz")
data = np.load(npz_path)
assert "lod_levels" in data, "a 1280-face import must write lod_levels"
n_levels = int(data["lod_levels"])
assert n_levels >= 1, n_levels
for i in range(1, n_levels + 1):
    for key in ("vertices", "faces", "face_colors", "face_roughness",
               "face_metallic", "face_emissive", "face_opacity", "source_faces"):
        assert f"lod{i}_{key}" in data, f"missing lod{i}_{key} in npz"
assert int(data["lod{}_faces".format(1)].shape[0]) < int(data["faces"].shape[0])
print(f"5a. import_fbx(generate_lods=True) wrote {n_levels} LOD levels to the npz")

lib = AssetLibrary(tmp_assets)
entity = lib.instantiate(asset_name)
assert len(entity.lod_meshes) == n_levels, \
    f"AssetDef.instantiate must load every lod{{i}}_* level: {len(entity.lod_meshes)} vs {n_levels}"
lod_counts = [len(entity.mesh.faces)] + [len(m.faces) for m in entity.lod_meshes]
assert lod_counts == sorted(lod_counts, reverse=True), lod_counts
for m in entity.lod_meshes:
    assert hasattr(m, "lod_source_faces") and len(m.lod_source_faces) == len(m.faces)
print(f"5b. AssetDef.instantiate round-trip OK: face counts {lod_counts}")

# generate_lods=False must skip LOD generation entirely (no lod keys at all)
fake_path2 = os.path.join(TMP, "fake_highpoly_nolod.fbx")
with um.patch.object(fbx_mod, "extract_geometry", _fake_extract):
    asset_name2 = fbx_mod.import_fbx(fake_path2, tmp_assets, generate_lods=False)
data2 = np.load(os.path.join(tmp_assets, "models", "fake_highpoly_nolod.npz"))
assert "lod_levels" not in data2, "generate_lods=False must write no lod keys"
lib.reload()
entity2 = lib.instantiate(asset_name2)
assert entity2.lod_meshes == [], "generate_lods=False must leave lod_meshes empty"
print("5c. generate_lods=False OK: npz has no lod keys, entity.lod_meshes == []")

# a small (below-threshold) mesh must be unaffected by generate_lods=True --
# the npz comes out with no lod keys at all, same as generate_lods=False
def _fake_extract_small(path):
    m = engine.cube(1.0)
    faces = [tuple(int(i) for i in f) for f in m.faces.tolist()]
    colors = m.face_colors / 255.0
    return m.vertices.copy(), faces, colors, None


fake_small_path = os.path.join(TMP, "fake_small.fbx")
with um.patch.object(fbx_mod, "extract_geometry", _fake_extract_small):
    small_asset_name = fbx_mod.import_fbx(fake_small_path, tmp_assets, generate_lods=True)
small_data = np.load(os.path.join(tmp_assets, "models", "fake_small.npz"))
assert "lod_levels" not in small_data, \
    "a below-threshold mesh must write no lod keys even with generate_lods=True"
lib.reload()
small_entity = lib.instantiate(small_asset_name)
assert small_entity.lod_meshes == []
print("5d. small-mesh import OK: generate_lods=True is inert below the face threshold")

# ========================================================================
# 6. runtime distance selection WITH hysteresis: close -> LOD0, far -> a
#    higher LOD, and an entity dithering across a naive boundary distance
#    does NOT flip lod_index every call (the whole point of hysteresis --
#    see engine/lod.py's LOD_HYSTERESIS / gl_renderer.py's geometry cache,
#    which would otherwise thrash rebuilding GPU buffers every frame)
# ========================================================================
sel_mesh = icosphere(radius=1.0, subdivisions=3)  # bound == 1.0
sel_lods = lod.generate_lods(sel_mesh)
assert len(sel_lods) > 1

sel_entity = engine.Entity("selector", mesh=sel_mesh)
sel_entity.lod_meshes = sel_lods[1:]


class _FakeCamera:
    def __init__(self, pos):
        self.position = pos


bound = sel_mesh.bound
up0 = lod.LOD_DISTANCE_FACTORS[0]  # boundary between LOD0 and LOD1

lod.update_entity_lod(sel_entity, _FakeCamera(engine.Vec3(bound * 1.0, 0, 0)))
assert sel_entity.lod_index == 0, "close camera must select LOD0"

lod.update_entity_lod(sel_entity, _FakeCamera(engine.Vec3(bound * (up0 * 3.0), 0, 0)))
assert sel_entity.lod_index > 0, "far camera must select a coarser LOD"
print(f"6a. distance selection OK: LOD0 up close, LOD{sel_entity.lod_index} far away "
     f"(bound={bound:.3f})")

# dither the distance just above/below the naive (no-hysteresis) boundary --
# a real per-frame camera wobble -- and count how many times lod_index
# actually changes across 40 such frames
sel_entity.lod_index = 0
near_d = bound * (up0 * 0.9)   # just below the naive boundary
far_d = bound * (up0 * 1.1)    # just above it
changes = 0
prev = sel_entity.lod_index
for i in range(40):
    d = far_d if (i % 2 == 0) else near_d
    lod.update_entity_lod(sel_entity, _FakeCamera(engine.Vec3(d, 0, 0)))
    if sel_entity.lod_index != prev:
        changes += 1
        prev = sel_entity.lod_index
assert changes <= 1, f"hysteresis must stop boundary dithering from flipping lod_index repeatedly: {changes} changes"
print(f"6b. hysteresis OK: {changes} lod_index change(s) across 40 frames dithering "
     f"across the naive LOD0/LOD1 boundary (would be ~40 without hysteresis)")

# a big jump straight to very far away must land on the COARSEST level, not
# creep up one step per call
sel_entity.lod_index = 0
lod.update_entity_lod(sel_entity, _FakeCamera(engine.Vec3(bound * 1000.0, 0, 0)))
assert sel_entity.lod_index == len(sel_entity.lod_meshes), \
    f"a huge distance jump must select the coarsest LOD in one call: {sel_entity.lod_index}"
print(f"6c. large jump OK: selects the coarsest LOD ({sel_entity.lod_index}) in one call")

# ========================================================================
# 7. rasterizer draws fewer triangles for a distant high-poly entity than a
#    close one -- the actual FPS payoff this feature exists for
# ========================================================================
from engine.camera import Camera
from engine.renderer import Renderer

tri_scene = engine.Scene(enable_shadows=False)
tri_entity = engine.Entity("hp", mesh=hp_mesh)
tri_entity.lod_meshes = lod.generate_lods(hp_mesh)[1:]
tri_scene.add(tri_entity)

renderer = Renderer()
import pygame
pygame.init()
tri_surf = pygame.Surface((320, 240))

close_cam = Camera(position=engine.Vec3(0, 0, hp_mesh.bound * 3.0))
lod.update_scene_lods(tri_scene, close_cam)
assert tri_entity.lod_index == 0
renderer.render(tri_surf, tri_scene, close_cam)
close_tris = renderer.stats["triangles"]

far_cam = Camera(position=engine.Vec3(0, 0, hp_mesh.bound * lod.LOD_DISTANCE_FACTORS[-1] * 3.0))
lod.update_scene_lods(tri_scene, far_cam)
assert tri_entity.lod_index > 0
renderer.render(tri_surf, tri_scene, far_cam)
far_tris = renderer.stats["triangles"]

assert far_tris < close_tris, \
    f"a distant high-poly entity must draw fewer triangles than up close: far={far_tris} close={close_tris}"
print(f"7. triangle count drop OK: close={close_tris} tris (LOD0) -> "
     f"far={far_tris} tris (LOD{tri_entity.lod_index})")

# ========================================================================
# 8. shadows use a FIXED occlusion mesh (shadow_mesh(): LOD0, or the
#    coarsest LOD for high-poly meshes -- see checks 10-13) regardless of
#    which LOD the camera would select for rasterization -- the caster's
#    occluder role never reads entity.lod_index. Two INDEPENDENT
#    ShadowTracer instances (so nothing is cache-shared) trace the same
#    fixed receiver against the same fixed caster at two different
#    lod_index values; the shadow result must be identical.
# ========================================================================
from engine.raytrace import ShadowTracer
from engine.renderer import _gather_lights, _world_face_geometry

shadow_scene = engine.Scene(enable_shadows=True)
ground = engine.Entity("ground", mesh=engine.checkerboard(6, 1.0))
ground.casts_shadow = False
shadow_scene.add(ground)

caster = engine.Entity("caster", mesh=hp_mesh,
                       position=engine.Vec3(0, hp_mesh.bound + 0.5, 0))
caster.lod_meshes = lod.generate_lods(hp_mesh)[1:]
shadow_scene.add(caster)

lamp = engine.Entity("lamp", light=engine.PointLight(
    intensity=4.0, range=25.0, radius=0.2, cast_shadows=True),
    position=engine.Vec3(0, hp_mesh.bound + 6.0, 0))
shadow_scene.add(lamp)

gcentroids, gnormals = _world_face_geometry(ground)
info = _gather_lights(shadow_scene)[0]
active = np.ones(len(gcentroids), dtype=bool)

caster.lod_index = 0
tracer_near = ShadowTracer()
tracer_near.refresh(shadow_scene)
shadow_near = tracer_near.shadow_factors(ground, info.light, info.pos, gcentroids, gnormals, active)

caster.lod_index = len(caster.lod_meshes)  # coarsest -- simulates a far camera
tracer_far = ShadowTracer()
tracer_far.refresh(shadow_scene)
shadow_far = tracer_far.shadow_factors(ground, info.light, info.pos, gcentroids, gnormals, active)

assert np.allclose(shadow_near, shadow_far), \
    "ground shadow must be identical regardless of the caster's lod_index (occlusion always uses LOD0)"
assert shadow_near.min() < 0.9, "sanity: the caster must actually be casting SOME shadow onto the ground"
print(f"8. shadow LOD0-pinning OK: ground shadow identical at caster.lod_index=0 vs "
     f"{len(caster.lod_meshes)} (min factor {shadow_near.min():.3f})")

# ========================================================================
# 9. starter-scene parity: every built-in asset has no LOD data, so
#    render_mesh() must be a structural no-op for the whole scene (proves
#    the rasterizer path this feature added is never even exercised for
#    it) -- and it still renders to a sane, non-black frame.
# ========================================================================
sys.path.insert(0, WT)
from editor import build_starter_scene

starter_lib = engine.AssetLibrary(os.path.join(WT, "assets"))
starter_scene = build_starter_scene(engine, starter_lib)
for e in starter_scene.entities:
    if e.mesh is not None:
        assert e.render_mesh() is e.mesh, \
            f"built-in entity '{e.name}' must have no LOD data (render_mesh() must equal mesh)"
        assert e.lod_meshes == []

starter_cam = Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
lod.update_scene_lods(starter_scene, starter_cam)
for e in starter_scene.entities:
    assert e.lod_index == 0
starter_surf = pygame.Surface((320, 240))
renderer.render(starter_surf, starter_scene, starter_cam)
mean_px = pygame.surfarray.array3d(starter_surf).astype(np.float64).mean()
assert mean_px > 5.0, f"starter scene must not render as a black frame: mean {mean_px:.2f}"
print(f"9. starter-scene parity OK: every entity's render_mesh() is its own mesh (no-op), "
     f"renders fine (mean pixel {mean_px:.1f})")

# ========================================================================
# 10. Entity.shadow_mesh(): identity for no-LOD entities, COARSEST LOD for
#     entities with LOD data -- independent of lod_index/camera distance
#     (that's the whole point: occlusion cost must not depend on which LOD
#     the camera would pick for rasterization).
# ========================================================================
plain = engine.Entity("plain", mesh=small_mesh)
assert plain.shadow_mesh() is plain.mesh, \
    "a no-LOD entity's shadow_mesh() must be its mesh, unchanged"

hp_entity = engine.Entity("hp_shadow", mesh=hp_mesh)
hp_lods = lod.generate_lods(hp_mesh)
hp_entity.lod_meshes = hp_lods[1:]
assert hp_entity.shadow_mesh() is hp_lods[-1], \
    "a high-poly entity's shadow_mesh() must be the COARSEST available LOD"
hp_entity.lod_index = 0
assert hp_entity.shadow_mesh() is hp_lods[-1], "shadow_mesh() must not depend on lod_index"
hp_entity.lod_index = len(hp_entity.lod_meshes)
assert hp_entity.shadow_mesh() is hp_lods[-1], "shadow_mesh() must not depend on lod_index"
print(f"10. Entity.shadow_mesh() OK: no-LOD -> mesh identity; high-poly -> coarsest LOD "
     f"({len(hp_mesh.faces)} -> {len(hp_lods[-1].faces)} faces), independent of lod_index")

# ========================================================================
# 10b. On-demand shadow proxy for a HIGH-POLY mesh with NO precomputed LODs
#      (the exact case that froze the bake: the user's Gat was imported
#      before LODs existed, so lod_meshes is empty and shadow_mesh() would
#      otherwise fall back to the full 10k-face mesh). shadow_mesh() must
#      decimate on demand above SHADOW_PROXY_THRESHOLD and cache the proxy;
#      meshes between LOD_FACE_THRESHOLD and SHADOW_PROXY_THRESHOLD (e.g. the
#      256-face Stone Floor) must stay identity so built-ins are unchanged.
# ========================================================================
big = engine.icosphere(0.8, 3)   # 1280 faces, no lod_meshes -> must proxy
assert len(big.faces) > lod.SHADOW_PROXY_THRESHOLD
big_e = engine.Entity("big_nolod", mesh=big)
assert not big_e.lod_meshes, "fixture must have no precomputed LODs"
proxy = big_e.shadow_mesh()
assert proxy is not big, "high-poly no-LOD mesh must get an on-demand shadow proxy"
assert len(proxy.faces) < len(big.faces), "the proxy must be coarser than the full mesh"
assert big_e.shadow_mesh() is proxy, "the on-demand proxy must be cached (same object)"

mid = engine.icosphere(0.8, 2)   # 320 faces: above LOD threshold, below proxy threshold
assert lod.LOD_FACE_THRESHOLD < len(mid.faces) <= lod.SHADOW_PROXY_THRESHOLD
mid_e = engine.Entity("mid_nolod", mesh=mid)
assert mid_e.shadow_mesh() is mid, \
    "a mesh below SHADOW_PROXY_THRESHOLD must keep shadow_mesh() == mesh (built-in compat)"
print(f"10b. on-demand shadow proxy OK: {len(big.faces)}-face no-LOD mesh -> "
     f"{len(proxy.faces)}-face proxy (cached); {len(mid.faces)}-face mesh stays identity")

# ========================================================================
# 11. occluder coarsening: ShadowTracer's world-space triangle soup for a
#     high-poly shadow caster is built from the COARSE proxy (fewer
#     triangles than LOD0), while a caster with no LOD data still
#     contributes its full mesh unchanged -- the backward-compat contract
#     every built-in relies on. (ShadowTracer already imported by check 8.)
# ========================================================================
coarse_scene = engine.Scene(enable_shadows=True)
coarse_ground = engine.Entity("ground_c", mesh=engine.checkerboard(6, 1.0))
coarse_ground.casts_shadow = False
coarse_scene.add(coarse_ground)
coarse_caster = engine.Entity("hp_caster", mesh=hp_mesh,
                              position=engine.Vec3(0, hp_mesh.bound + 0.5, 0))
coarse_caster.lod_meshes = lod.generate_lods(hp_mesh)[1:]
coarse_scene.add(coarse_caster)

coarse_tracer = ShadowTracer()
coarse_tracer.refresh(coarse_scene)
occ_tris = len(coarse_tracer._occ[0])
assert occ_tris < len(hp_mesh.tri_faces), \
    (f"occluder soup for a high-poly caster must use the coarse LOD, not LOD0: "
     f"{occ_tris} tris vs {len(hp_mesh.tri_faces)} at LOD0")
print(f"11a. occluder coarsening OK: {len(hp_mesh.tri_faces)} LOD0 tris -> "
     f"{occ_tris} occluder tris (coarsest LOD)")

lowpoly_scene = engine.Scene(enable_shadows=True)
lowpoly_ground = engine.Entity("ground_lp", mesh=engine.checkerboard(6, 1.0))
lowpoly_ground.casts_shadow = False
lowpoly_scene.add(lowpoly_ground)
lowpoly_caster = engine.Entity("cube_caster", mesh=small_mesh,
                               position=engine.Vec3(0, 2.0, 0))
lowpoly_scene.add(lowpoly_caster)
lowpoly_tracer = ShadowTracer()
lowpoly_tracer.refresh(lowpoly_scene)
assert len(lowpoly_tracer._occ[0]) == len(small_mesh.tri_faces), \
    "a caster with no LOD data must contribute its full (unchanged) mesh to the occluder soup"
print(f"11b. low-poly backward-compat OK: occluder soup uses the full "
     f"{len(small_mesh.tri_faces)}-tri mesh unchanged (no coarsening applied)")

# ========================================================================
# 12. THE FIX: a high-poly shadow-casting mesh (subdivided icosphere, built
#     in-memory -- never the user's real assets/gat.*) bakes its first
#     shadow+GI pass in a few seconds, not the minutes the old LOD0-pinned
#     occluder soup cost (task diagnosis: a 10,448-face import took 295s
#     pre-fix vs 0.76s for a 594-face scene). Also proves the coarse
#     occluder still actually produces shadows -- coarsening must not
#     silently disable them.
# ========================================================================
import time

bake_scene = engine.Scene(enable_shadows=True)
bake_ground = engine.Entity("ground_bake", mesh=engine.checkerboard(6, 1.0))
bake_ground.casts_shadow = False
bake_scene.add(bake_ground)

bake_caster = engine.Entity("bake_caster", mesh=hp_mesh,
                            position=engine.Vec3(0, hp_mesh.bound + 0.5, 0))
bake_caster.lod_meshes = lod.generate_lods(hp_mesh)[1:]
bake_scene.add(bake_caster)

bake_lamp = engine.Entity("bake_lamp", light=engine.PointLight(
    intensity=4.0, range=25.0, radius=0.2, cast_shadows=True),
    position=engine.Vec3(0, hp_mesh.bound + 6.0, 0))
bake_scene.add(bake_lamp)
bake_scene.gi = {"enabled": True, "intensity": 1.0, "samples": 16}

bake_cam = Camera(position=engine.Vec3(0, hp_mesh.bound + 3.0, hp_mesh.bound * 4.0))
bake_surf = pygame.Surface((320, 240))
bake_tracer = ShadowTracer()

t0 = time.perf_counter()
bake_tracer.refresh(bake_scene)
renderer.render(bake_surf, bake_scene, bake_cam, tracer=bake_tracer)
elapsed = time.perf_counter() - t0
BOUND_S = 10.0
assert elapsed < BOUND_S, \
    f"high-poly shadow/GI bake took {elapsed:.2f}s, must be under {BOUND_S}s (was minutes pre-fix)"

bg_centroids, bg_normals = _world_face_geometry(bake_ground)
bg_active = np.ones(len(bg_centroids), dtype=bool)
binfo = _gather_lights(bake_scene)[0]
shadow_vals = bake_tracer.shadow_factors(bake_ground, binfo.light, binfo.pos,
                                         bg_centroids, bg_normals, bg_active)
assert shadow_vals.min() < 0.9, \
    "the coarse occluder must still cast a real shadow onto the ground, not just pass rays through"
print(f"12. shadow/GI bake time OK: {elapsed:.2f}s for a {len(hp_mesh.faces)}-face self-shadowing "
     f"+ GI scene (bound {BOUND_S}s; was minutes pre-fix with LOD0-pinned occlusion); "
     f"coarse occluder still shadows the ground (min factor {shadow_vals.min():.3f})")

# ========================================================================
# 13. starter-scene shadow/GI parity: every built-in has no LOD data, so
#     entity.shadow_mesh() is entity.mesh for the whole scene -- the new
#     coarsening mechanism is a structural no-op here -- and a full render
#     with ray-traced shadows/GI actually turned on (unlike check 9, which
#     renders with tracer=None) still produces a sane, non-black frame.
# ========================================================================
for e in starter_scene.entities:
    if e.mesh is not None:
        assert e.shadow_mesh() is e.mesh, \
            f"built-in entity '{e.name}' must take the shadow_mesh() == mesh identity path"

starter_scene.gi = {"enabled": True, "intensity": 1.0, "samples": 16}
starter_tracer = ShadowTracer()
starter_tracer.refresh(starter_scene)
starter_surf2 = pygame.Surface((320, 240))
renderer.render(starter_surf2, starter_scene, starter_cam, tracer=starter_tracer)
mean_px2 = pygame.surfarray.array3d(starter_surf2).astype(np.float64).mean()
assert mean_px2 > 5.0, f"starter scene with shadows/GI must not render black: mean {mean_px2:.2f}"
print(f"13. starter-scene shadow/GI parity OK: every built-in takes the shadow_mesh() == mesh "
     f"identity path; full shadow+GI render is sane (mean pixel {mean_px2:.1f})")

print("ALL LOD CHECKS PASSED")
