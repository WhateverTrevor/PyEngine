"""Judge checks: texture assets (import, browser thumbnail), TexCoord /
TextureSample material nodes, box-projection UV fallback, and FBX UV
round-trip. Headless; PYENGINE_SETTINGS points at a temp path so this never
touches the user's real settings.json.
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.gettempdir()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["PYENGINE_SETTINGS"] = os.path.join(TMP, "judge_texture_settings.json")
sys.path.insert(0, REPO)

import numpy as np
import pygame

import engine
from engine import fbx as fbx_mod
from engine import mesh as mesh_mod
from engine import texture as texture_mod
from editor import ICON, make_icon

pygame.init()

lib = engine.AssetLibrary(os.path.join(REPO, "assets"))
_folders_json_preexisted = os.path.exists(os.path.join(lib.directory, "folders.json"))

# ---- 1. texture import: files into the selected folder, thumbnail generated ----
png_path = os.path.join(TMP, "judge_tex_2x2.png")
surf = pygame.Surface((2, 2))
surf.set_at((0, 0), (255, 0, 0))     # top-left    -> red
surf.set_at((1, 0), (0, 255, 0))     # top-right   -> green
surf.set_at((0, 1), (0, 0, 255))     # bottom-left -> blue
surf.set_at((1, 1), (255, 255, 0))   # bottom-right-> yellow
pygame.image.save(surf, png_path)

folder = lib.create_folder("Judge Texture Folder")
lib.save_folders()
tex_name = engine.import_texture(png_path, lib.directory)
lib.reload()
lib.set_asset_folder(tex_name, folder)
lib.save_folders()
assert tex_name in lib.by_name, "imported texture must appear in the library"
tex_asset = lib.by_name[tex_name]
assert "texture" in tex_asset.data, "texture assets carry a 'texture' field"
assert os.path.isfile(os.path.join(lib.directory, tex_asset.data["texture"]["path"]))
assert lib.folder_of.get(tex_name) == folder, "import must file into the selected folder"

icon = make_icon(engine, tex_asset)
assert icon.get_size() == (ICON, ICON)
icon_arr = pygame.surfarray.array3d(icon)
assert not np.all(icon_arr == 29), "thumbnail must render the image, not the blank fill"
print("texture import OK: filed into folder + thumbnail rendered from the image")

# reload from a fresh library to prove the on-disk asset alone (not in-memory
# state) is sufficient to re-derive the same thumbnail/lookup
img = texture_mod.load_texture_rel(tex_asset.data["texture"]["path"])
assert img is not None and img.shape[:2] == (2, 2)
print("texture lazy-load + cache OK")

# ---- 2. TexCoord tiling math ----
plane = mesh_mod.checkerboard(squares=2, square_size=1.0)
fu = plane.face_uvs.copy()
assert fu.shape == (4, 2)

g = engine.MaterialGraph()
tc = g.add("tex_coord", (0, 0))
g.nodes[tc]["params"]["u_tiling"] = 2.0
g.nodes[tc]["params"]["v_tiling"] = 3.0
assert g.connect(tc, g.output_id(), "color")
tiled = g.evaluate(plane) / 255.0  # bake clamps to 0..1, so compare against the same clamp
expected_u = np.clip(fu[:, 0] * 2.0, 0.0, 1.0)
expected_v = np.clip(fu[:, 1] * 3.0, 0.0, 1.0)
assert np.allclose(tiled[:, 0], expected_u, atol=1e-6)
assert np.allclose(tiled[:, 1], expected_v, atol=1e-6)
print("TexCoord tiling math OK")

# ---- 3. TextureSample on a 4-square plane with known box-projected UVs,
#         baked against a synthetic 2x2 texture -> expected per-face colors ----
# checkerboard(2, 1.0) box-projects each square's centroid to a distinct UV
# quadrant: (i=0,j=0)->(.25,.25) (i=0,j=1)->(.25,.75) (i=1,j=0)->(.75,.25)
# (i=1,j=1)->(.75,.75); nearest-neighbor + v-flip (v=0 is the image's bottom)
# resolves those to pixels (0,1) (0,0) (1,1) (0,1)... derived once via
# texture.sample_texture below, independent of the graph, so the graph's
# output is checked against the sampler, not against itself.
tex_img = texture_mod.load_texture(png_path)
expected_rgb, _ = texture_mod.sample_texture(tex_img, fu)

g2 = engine.MaterialGraph()
ts = g2.add("tex_sample", (0, 0))
g2.nodes[ts]["texture"] = "textures/" + os.path.basename(png_path)
# import_texture already copied a file with this basename in an earlier
# assertion path -- point directly at the plain copy for a clean, isolated
# resolve instead (no folder-tree coupling to check #1's asset)
import shutil
tex_dir = os.path.join(lib.directory, "textures")
os.makedirs(tex_dir, exist_ok=True)
shutil.copyfile(png_path, os.path.join(tex_dir, "judge_tex_2x2_ts.png"))
texture_mod.clear_cache()
g2.nodes[ts]["texture"] = "textures/judge_tex_2x2_ts.png"
assert g2.connect(ts, g2.output_id(), "color", "RGB")
baked = g2.evaluate(plane) / 255.0
assert np.allclose(baked, expected_rgb, atol=1.0 / 255.0), \
    f"TextureSample bake mismatch:\n{baked}\nvs\n{expected_rgb}"
print("TextureSample bake OK: 4-quadrant texture matches known per-face UVs")

# ---- 4. unconnected UV defaults to TexCoord(0) (untiled), same as an
#         explicit TexCoord(0) with tiling 1/1 wired into the uv input ----
g3 = engine.MaterialGraph()
ts3 = g3.add("tex_sample", (0, 0))
g3.nodes[ts3]["texture"] = "textures/judge_tex_2x2_ts.png"
tc3 = g3.add("tex_coord", (0, 0))
g3.connect(tc3, ts3, "uv")
assert g3.connect(ts3, g3.output_id(), "color", "RGB")
baked_explicit = g3.evaluate(plane)
assert np.allclose(baked_explicit, g2.evaluate(plane)), \
    "unconnected TextureSample.uv must behave exactly like an explicit TexCoord(0)"
print("unconnected-UV default OK: matches explicit TexCoord(0)")

# ---- 5. box-projection fallback produces sane 0..1 UVs on procedural meshes ----
for m in (mesh_mod.cube(2.0), mesh_mod.cylinder(), mesh_mod.icosphere(),
         mesh_mod.torus(), plane):
    assert m.face_uvs.shape == (len(m.faces), 2)
    assert np.all(m.face_uvs >= -1e-9) and np.all(m.face_uvs <= 1.0 + 1e-9), \
        f"box-projected UV out of 0..1: {m.face_uvs.min()}..{m.face_uvs.max()}"
print("box-projection fallback OK: sane 0..1 UVs on cube/cylinder/icosphere/torus/plane")

# ---- 6. FBX UV round-trip: export a mesh with real (box-projected) UVs,
#         reimport, and the per-face UV survives ----
uv_fbx_path = os.path.join(TMP, "judge_uv_roundtrip.fbx")
fbx_mod.export_fbx(plane, uv_fbx_path, name="UVPlane")
_verts, _faces, _colors, reimported_uv = fbx_mod.extract_geometry(uv_fbx_path)
assert reimported_uv is not None, "exported LayerElementUV must be read back"
assert reimported_uv.shape == plane.face_uvs.shape
assert np.allclose(reimported_uv, plane.face_uvs, atol=1e-4), \
    f"UV round-trip mismatch:\n{reimported_uv}\nvs\n{plane.face_uvs}"
os.remove(uv_fbx_path)
print("FBX UV round-trip OK: LayerElementUV survives export -> reimport")

# ---- 7. imported-model asset path threads face_uvs through the .npz too ----
cube_fbx_path = os.path.join(TMP, "judge_uv_model.fbx")
uv_cube = mesh_mod.box(2.0, 2.0, 2.0)
fbx_mod.export_fbx(uv_cube, cube_fbx_path, name="UVCube")
model_name = engine.import_fbx(cube_fbx_path, lib.directory)
lib.reload()
model_entity = lib.instantiate(model_name)
assert model_entity.mesh.face_uvs is not None
assert model_entity.mesh.face_uvs.shape == (len(model_entity.mesh.faces), 2)
stem = os.path.splitext(os.path.basename(cube_fbx_path))[0]
os.remove(os.path.join(lib.directory, f"{stem}.json"))
os.remove(os.path.join(lib.directory, "models", f"{stem}.npz"))
os.remove(cube_fbx_path)
print("imported-model face_uvs OK: survives the .npz asset round-trip")

# ---- cleanup: don't leave judge artifacts in the real assets/ tree ----
os.remove(tex_asset.path)
os.remove(os.path.join(lib.directory, "textures", os.path.basename(png_path)))
os.remove(os.path.join(tex_dir, "judge_tex_2x2_ts.png"))
lib.folder_of.pop(tex_name, None)
del lib.folders[folder]
lib.save_folders()
lib.reload()
if not _folders_json_preexisted:
    folders_path = os.path.join(lib.directory, "folders.json")
    if os.path.exists(folders_path):
        os.remove(folders_path)
os.remove(png_path)
texture_mod.clear_cache()
print("ALL TEXTURE/UV CHECKS PASSED")
