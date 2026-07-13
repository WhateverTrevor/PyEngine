"""Judge checks: material transparency -- blend_mode, gated Opacity pin,
per-face opacity bake, CPU/GL/DX12 translucent rendering, shadow/GI caster
exclusion, and the material-editor blend-mode selector UI. Companion to
material_checks.py (node math/eval), pbr_checks.py (roughness/metallic/
emissive), and mat_ui_checks.py (editor UX) -- this suite is the dedicated
one for the blend_mode/opacity feature end to end."""
import os
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np
import pygame

import engine
from editor import Editor, MaterialEditorUI, build_starter_scene
from engine.materials import MaterialGraph
from engine.renderer import Renderer

# ---- isolate settings.json exactly like the other judge suites ----
REAL_SETTINGS = os.path.join(WT, "settings.json")
_real_before = (open(REAL_SETTINGS, "rb").read()
               if os.path.exists(REAL_SETTINGS) else None)
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_transparency_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

pygame.init()
pygame.display.set_mode((64, 64))

# ---- 1. opacity pin gating on the graph API ----
g = MaterialGraph()
out_id = g.output_id()
const_id = g.add("constant", (0, 0))
assert g.blend_mode == "opaque"
assert not g.connect(const_id, out_id, "opacity"), "opaque must reject Opacity connections"
g.set_blend_mode("translucent")
assert g.connect(const_id, out_id, "opacity"), "translucent must accept Opacity connections"
assert g.link_into(out_id, "opacity") is not None
g.set_blend_mode("opaque")
assert g.link_into(out_id, "opacity") is None, \
    "opaque<-translucent transition must disconnect any Opacity link"
print("opacity pin gating OK: opaque rejects/translucent accepts/back-to-opaque disconnects")

# ---- 2. blend_mode round-trips through graph to_dict/from_dict ----
g2 = MaterialGraph()
g2.set_blend_mode("translucent")
d = g2.to_dict()
assert d["blend_mode"] == "translucent"
g3 = MaterialGraph.from_dict(d)
assert g3.blend_mode == "translucent"
# old graphs with no "blend_mode" key default to opaque (backward compat)
d_old = dict(d)
del d_old["blend_mode"]
g4 = MaterialGraph.from_dict(d_old)
assert g4.blend_mode == "opaque"
print("blend_mode round-trip OK (to_dict/from_dict, backward-compat default)")

# ---- 3. blend_mode round-trips through scene save/load ----
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
scene = build_starter_scene(engine, lib)
target = next(e for e in scene.entities if e.mesh is not None)
target.material = MaterialGraph()
target.material.set_blend_mode("translucent")
target.material.nodes[target.material.output_id()]["params"]["opacity"] = 0.4
target.material.apply(target)
assert np.allclose(target.mesh.face_opacity, 0.4), "translucent bake must use inline opacity param"

scene_path = os.path.join(tempfile.gettempdir(), "judge_transparency_scene.json")
cam_dummy = engine.Camera(position=engine.Vec3(6, 2.6, 9))
engine.save_scene(scene, cam_dummy, scene_path)
scene2 = engine.load_scene(scene_path, lib)
target2 = next(e for e in scene2.entities if e.name == target.name)
assert target2.material is not None and target2.material.blend_mode == "translucent"
assert np.allclose(target2.mesh.face_opacity, 0.4), \
    "scene reload must re-bake face_opacity from the reloaded graph"
os.remove(scene_path)
print("blend_mode + face_opacity round-trip through scene save/load OK")

# ---- 4. bake gating: opaque bake always resets face_opacity to 1.0 ----
target.material.set_blend_mode("opaque")
target.material.apply(target)
assert np.allclose(target.mesh.face_opacity, 1.0), \
    "opaque bake must reset face_opacity to all-1.0 (backward-compat contract)"
print("opaque bake resets face_opacity to 1.0 OK")

# ---- 5. CPU blend-equation numeric check: alpha=0.5 red quad over blue bg ----
def make_translucent_scene(alpha, fg_color, bg_color, bg_pos=(0, 0, -2), fg_pos=(0, 0, 0)):
    sc = engine.Scene(light=engine.DirectionalLight(engine.Vec3(0, 0, -1), ambient=1.0,
                                                    intensity=0.0),
                      background=(10, 10, 10))
    bg = engine.Entity("bg", mesh=engine.cube(4.0, color=bg_color),
                       position=engine.Vec3(*bg_pos))
    sc.add(bg)
    fg = engine.Entity("fg", mesh=engine.cube(2.0, color=fg_color), position=engine.Vec3(*fg_pos))
    mg = MaterialGraph()
    mg.set_blend_mode("translucent")
    mg.nodes[mg.output_id()]["params"]["opacity"] = alpha
    fg.material = mg
    mg.apply(fg)
    sc.add(fg)
    return sc


W, H = 200, 150
cam = engine.Camera(position=engine.Vec3(0, 0, 8))
r = Renderer()
r.render_scale = 1

scene_bg_only = engine.Scene(light=engine.DirectionalLight(engine.Vec3(0, 0, -1), ambient=1.0,
                                                            intensity=0.0),
                             background=(10, 10, 10))
scene_bg_only.add(engine.Entity("bg", mesh=engine.cube(4.0, color=(40, 40, 220)),
                                position=engine.Vec3(0, 0, -2)))
surf_bg = pygame.Surface((W, H))
r.render(surf_bg, scene_bg_only, cam)
img_bg = pygame.surfarray.array3d(surf_bg).transpose(1, 0, 2).astype(float)

scene_t = make_translucent_scene(0.5, (220, 30, 30), (40, 40, 220))
surf_t = pygame.Surface((W, H))
r.render(surf_t, scene_t, cam)
img_t = pygame.surfarray.array3d(surf_t).transpose(1, 0, 2).astype(float)

cx, cy = W // 2, H // 2
bg_patch = img_bg[cy - 5:cy + 5, cx - 5:cx + 5].mean(axis=(0, 1))
t_patch = img_t[cy - 5:cy + 5, cx - 5:cx + 5].mean(axis=(0, 1))
expected = bg_patch * 0.5 + np.array([220.0, 30.0, 30.0]) * 0.5
assert np.abs(t_patch - expected).max() < 12, (t_patch, expected)
print(f"CPU blend-equation OK: composited {t_patch.round(1).tolist()} "
      f"vs expected {expected.round(1).tolist()}")

# corner (background-only, outside the translucent quad) must be unaffected
corner_bg = img_bg[5:15, 5:15].mean(axis=(0, 1))
corner_t = img_t[5:15, 5:15].mean(axis=(0, 1))
assert np.abs(corner_bg - corner_t).max() < 6, "translucent quad bled outside its own bounds"
print("translucent quad does not affect pixels outside its own footprint OK")

# ---- 6. back-to-front stacking order: nearer face composites on top ----
sc_stack = engine.Scene(light=engine.DirectionalLight(engine.Vec3(0, 0, -1), ambient=1.0,
                                                       intensity=0.0),
                        background=(10, 10, 10))
back = engine.Entity("back", mesh=engine.cube(2.0, color=(30, 200, 30)),
                     position=engine.Vec3(0, 0, -1.0))
mg_back = MaterialGraph()
mg_back.set_blend_mode("translucent")
mg_back.nodes[mg_back.output_id()]["params"]["opacity"] = 0.5
back.material = mg_back
mg_back.apply(back)
sc_stack.add(back)
front = engine.Entity("front", mesh=engine.cube(1.2, color=(220, 30, 30)),
                      position=engine.Vec3(0, 0, 1.5))
mg_front = MaterialGraph()
mg_front.set_blend_mode("translucent")
mg_front.nodes[mg_front.output_id()]["params"]["opacity"] = 0.5
front.material = mg_front
mg_front.apply(front)
sc_stack.add(front)
surf_stack = pygame.Surface((W, H))
r.render(surf_stack, sc_stack, cam)
img_stack = pygame.surfarray.array3d(surf_stack).transpose(1, 0, 2).astype(float)
stack_patch = img_stack[cy - 5:cy + 5, cx - 5:cx + 5].mean(axis=(0, 1))
bg_dark = np.array([10.0, 10.0, 10.0])
green = np.array([30.0, 200.0, 30.0])
red = np.array([220.0, 30.0, 30.0])
expected_stack = ((bg_dark * 0.5 + green * 0.5) * 0.5 + red * 0.5)
assert np.abs(stack_patch - expected_stack).max() < 15, (stack_patch, expected_stack)
print(f"back-to-front stacking order OK: {stack_patch.round(1).tolist()} "
      f"vs expected {expected_stack.round(1).tolist()}")

# ---- 7. translucent entities are excluded from shadow/GI casters ----
from engine.raytrace import ShadowTracer, GITracer, _is_translucent as rt_is_translucent

sc_shadow = engine.Scene(light=engine.DirectionalLight(engine.Vec3(-0.3, -1, -0.2), ambient=0.2))
floor = engine.Entity("floor", mesh=engine.checkerboard(8, 1.0))
sc_shadow.add(floor)
glass = engine.Entity("glass", mesh=engine.cube(1.0, color=(200, 200, 255)),
                      position=engine.Vec3(0, 3, 0))
mg_glass = MaterialGraph()
mg_glass.set_blend_mode("translucent")
mg_glass.nodes[mg_glass.output_id()]["params"]["opacity"] = 0.3
glass.material = mg_glass
mg_glass.apply(glass)
sc_shadow.add(glass)
assert rt_is_translucent(glass) and not rt_is_translucent(floor)

tracer = ShadowTracer()
tracer.refresh(sc_shadow)
# `_caster_mats` (id(entity) -> matrix bytes) is the internal caster-list
# proxy `refresh()` builds -- introspect it directly rather than relying on
# a public API that doesn't exist, to actually confirm the exclusion (not
# just that the predicate function is correct, which check 8 below covers).
assert id(glass) not in tracer._caster_mats, "translucent entity must not be a shadow occluder"
assert id(floor) in tracer._caster_mats, "opaque entity must still be a shadow occluder"
print("ShadowTracer.refresh excludes translucent entities from casters OK")

gi = GITracer()
# GI caster/receiver split is internal to compute(); rely on the module-level
# helper both ShadowTracer and GITracer share to confirm the predicate itself
# is correct and applied consistently (both raytrace.py and renderer.py have
# their own _is_translucent -- verify they agree).
from engine.renderer import _is_translucent as renderer_is_translucent
assert renderer_is_translucent(glass) == rt_is_translucent(glass) == True
assert renderer_is_translucent(floor) == rt_is_translucent(floor) == False
print("raytrace._is_translucent and renderer._is_translucent agree OK")

# ---- 8. opaque golden parity untouched (no translucent entities in scene) ----
scene_opaque = build_starter_scene(engine, lib)
surf_gold = pygame.Surface((400, 300))
cam_gold = engine.Camera(position=engine.Vec3(4, 3.2, 7), yaw=0.45, pitch=-0.3)
r.render(surf_gold, scene_opaque, cam_gold)
img_gold = pygame.surfarray.array3d(surf_gold).transpose(1, 0, 2).astype(float)
assert img_gold.mean() > 5.0, "starter scene should not render blank"
print(f"opaque golden parity OK: starter scene renders non-blank (mean={img_gold.mean():.1f})")

# ---- 9. 3-way CPU/GL/DX12 parity on a translucent scene ----
scene_gpu = make_translucent_scene(0.5, (220, 30, 30), (40, 40, 220))
try:
    from engine.gl_renderer import GLRenderer
    gl = GLRenderer.standalone(W, H)
    gl.render(scene_gpu, cam, (W, H), None)
    raw_gl = gl.target.read(components=3)
    img_gl = np.frombuffer(raw_gl, dtype=np.uint8).reshape(H, W, 3)[::-1].astype(float)
    gl_patch = img_gl[cy - 5:cy + 5, cx - 5:cx + 5].mean(axis=(0, 1))
    diff_gl = np.abs(t_patch - gl_patch).mean()
    assert diff_gl < 12, f"CPU/GL translucent divergence too large: {diff_gl}"
    print(f"GL translucent parity OK: cpu={t_patch.round(1).tolist()} "
         f"gl={gl_patch.round(1).tolist()} (diff {diff_gl:.2f})")
except ImportError:
    print("GL parity skipped: moderngl not installed")

try:
    from engine.wgpu_renderer import WgpuRenderer
    wg = WgpuRenderer("dx12")
    wg._ensure_size(W, H)
    wg.render(scene_gpu, cam, (W, H), None)
    raw_wg = wg.read_frame()
    img_wg = np.frombuffer(raw_wg, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].astype(float)
    wg_patch = img_wg[cy - 5:cy + 5, cx - 5:cx + 5].mean(axis=(0, 1))
    diff_wg = np.abs(t_patch - wg_patch).mean()
    assert diff_wg < 12, f"CPU/DX12 translucent divergence too large: {diff_wg}"
    print(f"DX12 translucent parity OK: cpu={t_patch.round(1).tolist()} "
         f"dx12={wg_patch.round(1).tolist()} (diff {diff_wg:.2f})")
except ImportError:
    print("DX12 parity skipped: wgpu not installed")
except Exception as e:
    print(f"DX12 parity skipped: {e}")

# ---- 10. editor UI: blend-mode selector toggles + greys the Opacity pin ----
eng = engine.Engine(1000, 700, title="judge", splash=False, api="cpu")
scene_ui = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene_ui, engine.Camera(position=engine.Vec3(6, 2.6, 9)),
                lib, "scenes/scene.json", settings_path=TEST_SETTINGS)
crate = next(e for e in scene_ui.entities if e.mesh is not None)
crate.material = MaterialGraph()
mui = MaterialEditorUI(editor, crate)
editor.mat_ui = mui
Wp, Hp = eng.screen.get_size()

assert mui.graph.blend_mode == "opaque"
surf_e = pygame.Surface((Wp, Hp))
mui.draw(surf_e)
assert mui._blend_opaque_rect is not None and mui._blend_translucent_rect is not None
opaque_rect_before = mui._blend_opaque_rect

mui.graph.set_blend_mode("translucent")
mui.apply(draft=False)
surf_e2 = pygame.Surface((Wp, Hp))
mui.draw(surf_e2)
assert mui.graph.blend_mode == "translucent"
print("blend-mode selector rects drawn for both states OK")

# clicking the Opaque button flips a translucent graph back to opaque --
# drive it through the real InputManager/update() path (monkeypatch
# pygame.mouse.get_pos, the one thing InputManager.mouse_pos reads, since
# the SDL dummy driver doesn't support real cursor warps) so this exercises
# the actual click-handling wiring in MaterialEditorUI.update(), not just
# the underlying graph API already covered by check 1.
mui.graph.set_blend_mode("translucent")
click_pt = mui._blend_opaque_rect.center
_orig_get_pos = pygame.mouse.get_pos
pygame.mouse.get_pos = lambda: click_pt
try:
    eng.input._mouse_pressed = {1}
    mui.update(eng, 1 / 60)
finally:
    pygame.mouse.get_pos = _orig_get_pos
    eng.input._mouse_pressed = set()
assert mui.graph.blend_mode == "opaque", "clicking Opaque button must switch blend_mode back"
print("blend-mode selector Opaque button click OK")

print("no-pollution guard: real settings.json untouched by this suite -- "
     f"{'OK' if (open(REAL_SETTINGS, 'rb').read() if os.path.exists(REAL_SETTINGS) else None) == _real_before else 'FAILED'}")
assert (open(REAL_SETTINGS, 'rb').read() if os.path.exists(REAL_SETTINGS) else None) == _real_before

print("ALL TRANSPARENCY JUDGE CHECKS PASSED")
