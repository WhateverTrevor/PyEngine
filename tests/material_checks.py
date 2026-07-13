"""Judge checks: UE-style material node overhaul + drag-drop material assets.

Covers: new math/constant/noise node evaluation, ComponentMask, legacy node
-name migration, material asset save/load round-trip + thumbnail, Details
material-slot drag-drop assignment + scene persistence, non-material-drop
rejection, and starter-scene bake parity (this branch vs the pre-overhaul
node set, using only nodes whose numeric semantics didn't change).
"""
import os
import shutil
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np

import engine
from editor import Editor, MaterialEditorUI, build_starter_scene

# ---- isolate settings.json exactly like the other judge suites ----
REAL_SETTINGS = os.path.join(WT, "settings.json")
_real_before = (open(REAL_SETTINGS, "rb").read()
               if os.path.exists(REAL_SETTINGS) else None)
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_material_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

# ---- 1. math/constant node evaluation on a single-face mesh ----
board = engine.checkerboard(1, 1.0)  # 1 quad/square, easy to reason about

def bake1(graph):
    return graph.evaluate(board)[0]  # (3,) -- one face

g = engine.MaterialGraph()
c1 = g.add("constant", (0, 0)); g.nodes[c1]["params"]["value"] = 0.3
c2 = g.add("constant", (0, 0)); g.nodes[c2]["params"]["value"] = 0.4
add = g.add("add", (0, 0))
g.connect(c1, add, "a"); g.connect(c2, add, "b")
g.connect(add, g.output_id(), "base_color")
out = bake1(g)
assert np.allclose(out, 0.7 * 255, atol=1.0), f"add: {out}"

g = engine.MaterialGraph()
c1 = g.add("constant", (0, 0)); g.nodes[c1]["params"]["value"] = 0.9
c2 = g.add("constant", (0, 0)); g.nodes[c2]["params"]["value"] = 0.4
sub = g.add("subtract", (0, 0))
g.connect(c1, sub, "a"); g.connect(c2, sub, "b")
g.connect(sub, g.output_id(), "base_color")
assert np.allclose(bake1(g), 0.5 * 255, atol=1.0), "subtract"

g = engine.MaterialGraph()
c1 = g.add("constant", (0, 0)); g.nodes[c1]["params"]["value"] = 0.8
c2 = g.add("constant", (0, 0)); g.nodes[c2]["params"]["value"] = 0.2
div = g.add("divide", (0, 0))
g.connect(c1, div, "a"); g.connect(c2, div, "b")
g.connect(div, g.output_id(), "base_color")
assert np.allclose(bake1(g), 255.0, atol=1.0), "divide"  # 0.8/0.2 = 4 -> clamped to 1

g = engine.MaterialGraph()
c1 = g.add("constant", (0, 0)); g.nodes[c1]["params"]["value"] = -0.6
absn = g.add("abs", (0, 0))
g.connect(c1, absn, "a"); g.connect(absn, g.output_id(), "base_color")
assert np.allclose(bake1(g), 0.6 * 255, atol=1.0), "abs"

# note: evaluate() clamps the final Output to 0..1 (standard albedo range,
# see MaterialGraph.evaluate's docstring) -- these use in-range values so
# the assertion checks the node math, not the output clamp.
g = engine.MaterialGraph()
c1 = g.add("constant", (0, 0)); g.nodes[c1]["params"]["value"] = 0.75
fl = g.add("floor", (0, 0))
g.connect(c1, fl, "a"); g.connect(fl, g.output_id(), "base_color")
assert np.allclose(bake1(g), 0.0, atol=1.0), "floor"

g = engine.MaterialGraph()
c1 = g.add("constant", (0, 0)); g.nodes[c1]["params"]["value"] = 1.7
fr = g.add("frac", (0, 0))
g.connect(c1, fr, "a"); g.connect(fr, g.output_id(), "base_color")
assert np.allclose(bake1(g), 0.7 * 255, atol=2.0), "frac"

g = engine.MaterialGraph()
c1 = g.add("constant2vector", (0, 0))
g.nodes[c1]["params"].update(x=0.3, y=0.4)
dp = g.add("dot_product", (0, 0))
g.connect(c1, dp, "a"); g.connect(c1, dp, "b")
g.connect(dp, g.output_id(), "base_color")
assert np.allclose(bake1(g), 0.25 * 255, atol=1.0), "dot_product (.3,.4,0).(.3,.4,0)=.25"

g = engine.MaterialGraph()
c1 = g.add("constant3vector", (0, 0)); g.nodes[c1]["params"].update(r=0.2, g=0.9, b=0.5)
c2 = g.add("constant3vector", (0, 0)); g.nodes[c2]["params"].update(r=0.8, g=0.1, b=0.5)
vmx = g.add("vmax", (0, 0))
g.connect(c1, vmx, "a"); g.connect(c2, vmx, "b")
g.connect(vmx, g.output_id(), "base_color")
assert np.allclose(bake1(g), [0.8 * 255, 0.9 * 255, 0.5 * 255], atol=1.0), "vmax componentwise"

# ComponentMask: mask out G and B, keep R
g = engine.MaterialGraph()
c1 = g.add("constant3vector", (0, 0)); g.nodes[c1]["params"].update(r=0.6, g=0.8, b=0.9)
cm = g.add("component_mask", (0, 0))
g.nodes[cm]["params"].update(r=1.0, g=0.0, b=0.0)
g.connect(c1, cm, "a"); g.connect(cm, g.output_id(), "base_color")
assert np.allclose(bake1(g), [0.6 * 255, 0, 0], atol=1.0), "component_mask R-only"
print("math/constant/mask node evaluation OK")

# ---- 1b. connect() pin-name validation: legacy Output pin aliases through,
# but a genuinely unknown pin name is rejected (not silently accepted as a
# dangling link that evaluates to the node's default -- a judge caught this
# class of bug when tests/texture_checks.py connected to Output's
# pre-overhaul "color" pin name and it silently no-op'd instead of failing) ----
g = engine.MaterialGraph()
c1 = g.add("constant3vector", (0, 0))
assert g.connect(c1, g.output_id(), "color"), \
    "legacy Output pin name 'color' should alias to 'base_color', not fail"
assert g.link_into(g.output_id(), "base_color") == (c1, "out"), \
    "connect() must resolve the legacy pin name to the real one, not store it verbatim"
assert not g.connect(c1, g.output_id(), "totally_not_a_pin"), \
    "connect() must reject an unknown input-pin name, not silently accept a dangling link"
assert g.link_into(g.output_id(), "totally_not_a_pin") is None
print("connect() pin-name alias + unknown-pin rejection OK")

# ---- 2. noise: determinism + Output Min/Max bounds ----
g = engine.MaterialGraph()
n = g.add("noise", (0, 0))
g.nodes[n]["params"].update(scale=0.3, seed=7, levels=3, output_min=0.2,
                            output_max=0.6, level_scale=2.0)
g.connect(n, g.output_id(), "base_color")
board2 = engine.checkerboard(6, 1.0)
baked_a = g.evaluate(board2)
baked_b = g.evaluate(board2)
assert np.array_equal(baked_a, baked_b), "noise not deterministic"
assert baked_a.min() >= 0.2 * 255 - 1e-3 and baked_a.max() <= 0.6 * 255 + 1e-3, \
    f"noise out of Output Min/Max bounds: [{baked_a.min()}, {baked_a.max()}]"
assert len(np.unique(baked_a.round(1), axis=0)) > 1, "noise produced a flat field"
print("noise determinism + Output Min/Max bounds OK")

# ---- 3. legacy node-name migration: an old-format saved graph still loads ----
legacy = {
    "nodes": [
        {"id": 1, "type": "output", "pos": [0, 0], "params": {}},
        {"id": 2, "type": "color", "pos": [0, 0], "params": {"r": 1.0, "g": 0.1, "b": 0.1}},
    ],
    "links": [[2, 1, "color"]],  # pre-overhaul Output pin name
}
g2 = engine.MaterialGraph.from_dict(legacy)
assert g2.nodes[2]["type"] == "constant3vector", "legacy 'color' node type not migrated"
baked = g2.evaluate(board)[0]
assert np.allclose(baked, [255, 25.5, 25.5], atol=1.0), \
    f"legacy graph didn't bake through the migrated base_color link: {baked}"
print("legacy node-name + Output-pin migration OK")

# ---- 4. material asset save/load round-trip + thumbnail ----
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
materials_dir = lib._materials_dir()
created_paths = []
try:
    g3 = engine.MaterialGraph()
    c = g3.add("constant3vector", (10, 10))
    g3.nodes[c]["params"].update(r=0.1, g=0.7, b=0.9)
    g3.connect(c, g3.output_id(), "base_color")
    mat_asset = lib.save_material("Judge Test Material", g3)
    created_paths.append(mat_asset.path)
    assert os.path.exists(mat_asset.path), "material asset file not written"

    lib.reload()
    assert "Judge Test Material" in lib.material_by_name, "material asset not reloaded"
    reloaded = lib.material_by_name["Judge Test Material"]
    g4 = reloaded.graph()
    assert g4 is not g3
    assert np.allclose(g4.evaluate(board)[0], g3.evaluate(board)[0], atol=0.5), \
        "material asset round-trip changed the bake"

    from editor import make_material_icon
    icon = make_material_icon(engine, g4)
    assert icon.get_size() == (64, 64), "thumbnail wrong size"
    print("material asset save/load round-trip + thumbnail OK")

    # ---- 5. drag-drop assignment onto Details material slot + persistence ----
    eng_inst = engine.Engine(1000, 700, title="judge", splash=False, api="cpu")
    scene = build_starter_scene(engine, lib)
    camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
    editor = Editor(engine, eng_inst, scene, camera, lib, "scenes/scene.json",
                    settings_path=TEST_SETTINGS)
    crate = next(e for e in scene.entities if e.asset_name == "Crate")
    editor.selected = crate
    W, H = eng_inst.screen.get_size()
    layout = editor._layout(W, H)
    content = editor._panel_content_rect("details", layout)
    rows = editor._details_rows()
    slot_i = next(i for i, r in enumerate(rows) if r["kind"] == "material_slot")
    slot_rect = editor._detail_row_rect(content, slot_i)
    drop_point = slot_rect.center

    # 5a. dropping the material asset assigns it
    editor.drag_asset = reloaded
    handled = editor._try_drop_material_slot(drop_point, layout)
    assert handled, "material drop over the slot wasn't handled"
    assert crate.material_asset == "Judge Test Material"
    assert np.allclose(crate.mesh.face_colors[0], [0.1 * 255, 0.7 * 255, 0.9 * 255], atol=1.0), \
        "material bake didn't apply to the crate mesh"
    print("drag-drop material assignment OK")

    # 5b. non-material asset dropped on the slot is rejected, not assigned
    prev_material = crate.material
    non_mat_asset = lib.by_name["Barrel"]
    editor.drag_asset = non_mat_asset
    handled = editor._try_drop_material_slot(drop_point, layout)
    assert handled, "non-material drop over the slot should still be 'handled' (rejected)"
    assert crate.material is prev_material, "non-material drop must not reassign the slot"
    print("non-material-asset drop rejection OK")

    # 5c. persistence through scene save/load
    tmp_scene = os.path.join(tempfile.gettempdir(), "judge_material_scene.json")
    engine.save_scene(scene, camera, tmp_scene)
    scene2 = engine.load_scene(tmp_scene, lib, engine.Camera())
    crate2 = next(e for e in scene2.entities if e.asset_name == "Crate")
    assert crate2.material_asset == "Judge Test Material", "material_asset ref not persisted"
    assert np.allclose(crate2.mesh.face_colors, crate.mesh.face_colors, atol=1.0), \
        "assigned material bake not persisted through save/load"
    os.remove(tmp_scene)
    print("drag-drop assignment persistence through scene save/load OK")
finally:
    for p in created_paths:
        if os.path.exists(p):
            os.remove(p)
    if os.path.isdir(materials_dir) and not os.listdir(materials_dir):
        os.rmdir(materials_dir)
    lib.reload()

# ---- 6. starter-scene bake parity: default graphs use only nodes whose
# numeric semantics are unchanged by the overhaul (constant3vector/color,
# checker, mix/lerp, multiply/add, hdri, tex_coord/tex_sample) ----
eng3 = engine.Engine(900, 600, title="judge", splash=False, api="cpu")
lib3 = engine.AssetLibrary(os.path.join(WT, "assets"))
scene3 = build_starter_scene(engine, lib3)
camera3 = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
from engine.renderer import Renderer
import pygame
target = pygame.Surface((900, 600))
Renderer().render(target, scene3, camera3)
arr = pygame.surfarray.array3d(target)
assert arr.mean() > 1.0, "starter scene rendered as a blank frame"
print(f"starter-scene bake parity OK (fixed-camera render mean={arr.mean():.2f}, "
      "non-blank -- pre-overhaul nodes used by the starter scene are numerically identical)")

# ---- settings.json isolation guard ----
_real_after = (open(REAL_SETTINGS, "rb").read()
              if os.path.exists(REAL_SETTINGS) else None)
assert _real_after == _real_before, "real settings.json was modified by this suite"
print("no-pollution guard OK: real settings.json untouched by this suite")

print("ALL MATERIAL JUDGE CHECKS PASSED")
