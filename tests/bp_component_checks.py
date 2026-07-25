"""Judge checks: Blueprint POSED MESHES (run 2a of the blueprint feature --
see engine/assets.py's BlueprintAsset and engine/mesh.py's merge_meshes for
the schema/design notes; run 1 shipped the script half, this run adds
posed-mesh components + composite instantiation + the Components UI).

Covers:
  - engine/mesh.py `merge_meshes`: the byte-identical single-component-at-
    identity gate, multi-component posing verified via ACTUAL transformed
    vertex math (not just counts), per-face array alignment when parts
    carry different explicit arrays, negative-scale winding flip (normals
    stay outward), zero-part -> None.
  - BlueprintAsset.instantiate: missing-component-asset skip (never
    raises), zero-component blueprint -> mesh=None, Entity.blueprint_name
    set / asset_name left None.
  - Scene save/load round-trip for a blueprint instance (position/rotation
    survive, mesh regenerated from the (possibly-edited) blueprint).
  - Content-browser blueprint tile: plain click (no movement) opens
    ScriptEditorUI without placing; press + drag-beyond-MARQUEE_THRESHOLD
    + release-over-viewport places an instance without opening the editor
    -- driven through the REAL click router (same _route_panel_click code
    path a mouse click takes, real pygame MOUSEBUTTONDOWN/UP events).
  - Ctrl+D duplicate and Del delete of a blueprint instance (both used to
    silently no-op on blueprint instances before this run, since they
    filtered on `asset_name is not None`).
  - New/Open Scene (`_replace_scene_content`) no longer misclassifies a
    placed blueprint instance as an editor-owned pseudo-entity (it used to
    key off `asset_name is None`, which is also true for a blueprint
    instance -- see Entity.is_placed()).
  - ScriptEditorUI's Components tab, driven through the REAL event path
    (per the project's hard rule for UI/interaction tests -- a ctx-menu
    click crash slipped past handler-only tests before): tab switch,
    Add-Component picker (+ immediate disk persistence), editing a
    Position field by typing digits and pressing Enter (+ persistence),
    remove-component, and that a freshly re-instantiated entity reflects
    the edited pose (documented: an ALREADY-PLACED instance in the scene
    does NOT retroactively update -- matches every other asset type here).

Isolation: same idiom as tests/blueprint_checks.py -- an isolated TEMP
COPY of the whole assets/ tree (never opens the real assets/blueprints,
assets/folders.json, assets/gat.json, assets/models/gat.npz for writing)
plus an isolated settings.json, with no-pollution guards at the end.
"""
import os
import shutil
import sys
import tempfile
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np
import pygame

import engine
from editor import Editor, EditorBehavior, ScriptEditorUI, build_starter_scene

REAL_SETTINGS = os.path.join(WT, "settings.json")
_real_settings_before = (open(REAL_SETTINGS, "rb").read()
                         if os.path.exists(REAL_SETTINGS) else None)
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_bp_component_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

# ---- isolated temp copy of assets/ -- see module docstring ----
TMP_ASSETS_ROOT = tempfile.mkdtemp(prefix="pyengine_bp_component_checks_")
ASSETS_DIR = os.path.join(TMP_ASSETS_ROOT, "assets")
shutil.copytree(os.path.join(WT, "assets"), ASSETS_DIR)
assert ASSETS_DIR != os.path.join(WT, "assets"), "must never point at the real assets dir"

eng = engine.Engine(1280, 800, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(ASSETS_DIR)
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
               settings_path=TEST_SETTINGS)
fly = engine.behaviors.FlyController(camera, look_buttons=(3,),
                                     look_guard=lambda p: not editor.over_ui(p))
editor.fly = fly
scene.add(engine.Entity("__camera").add_behavior(fly))
scene.add(engine.Entity("__editor").add_behavior(EditorBehavior(editor)))
eng.esc_handler = editor.handle_escape
W, H = eng.screen.get_size()

MESH_ASSETS = [a for a in lib.assets if "mesh" in a.data]
assert len(MESH_ASSETS) >= 2, "fixture assets/ needs at least 2 mesh assets for this suite"
A1, A2 = MESH_ASSETS[0], MESH_ASSETS[1]


class FakeKeys:
    """pygame.key.get_pressed() stand-in -- held-key state isn't exercised
    here (only single KEYDOWN presses), this just needs to never IndexError
    on pygame's huge SDLK_* values for special keys."""

    def __init__(self, held=()):
        self._held = set(held)

    def __getitem__(self, key):
        return key in self._held


_keys_patch = um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys())
_keys_patch.start()


def step(events):
    eng.input.process(events)
    editor.update(eng, 1 / 60)
    eng.input.consume_edges()


def mouse_down(pos):
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)])


def mouse_move(pos, held=True):
    pressed = (True, False, False) if held else (False, False, False)
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=pressed):
        step([])


def mouse_up(pos):
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
        step([pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos)])


def click(pos):
    """A plain press-then-release at `pos` -- same idiom as
    tests/blueprint_checks.py / tests/docktab_checks.py's `click`."""
    mouse_down(pos)
    mouse_up(pos)


def key(k, unicode="") -> None:
    step([pygame.event.Event(pygame.KEYDOWN, key=k, unicode=unicode, mod=0)])


def type_text(s: str) -> None:
    for ch in s:
        key(pygame.K_a, unicode=ch)  # physical key doesn't matter, only e.unicode


try:
    # ========================================================================
    # 1. merge_meshes: byte-identical single-component-at-identity gate
    # ========================================================================
    from engine.assets import BlueprintAsset

    bp1 = BlueprintAsset({"name": "BP_Identity", "components": [
        {"asset_name": A1.name, "position": [0, 0, 0], "rotation": [0, 0, 0],
         "scale": [1, 1, 1]}]}, path="<mem>")
    ent1 = bp1.instantiate(lib)
    assert ent1.blueprint_name == "BP_Identity" and ent1.asset_name is None
    ref = lib.instantiate(A1.name).mesh
    m = ent1.mesh
    for attr in ("vertices", "faces", "face_colors", "face_uvs", "face_roughness",
                "face_metallic", "face_emissive", "face_opacity"):
        assert np.array_equal(getattr(m, attr), getattr(ref, attr)), \
            f"single-component identity merge must be byte-identical on {attr}"
    print("1. byte-identical single-component-at-identity gate OK")

    # ========================================================================
    # 2. multi-component posing: verify ACTUAL transformed vertex positions
    # ========================================================================
    bp2 = BlueprintAsset({"name": "BP_Posed", "components": [
        {"asset_name": A1.name, "position": [5, 0, 0], "rotation": [0, 0, 0],
         "scale": [1, 1, 1]},
        {"asset_name": A2.name, "position": [0, 3, 0], "rotation": [0, 1.5708, 0],
         "scale": [2, 2, 2]},
    ]}, path="<mem>")
    ent2 = bp2.instantiate(lib)
    ref1, ref2 = lib.instantiate(A1.name).mesh, lib.instantiate(A2.name).mesh
    n1 = len(ref1.vertices)
    assert len(ent2.mesh.vertices) == n1 + len(ref2.vertices)
    assert np.allclose(ent2.mesh.vertices[:n1], ref1.vertices + np.array([5, 0, 0]))
    c, s = np.cos(1.5708), np.sin(1.5708)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    expected2 = (ref2.vertices * 2.0) @ ry.T + np.array([0, 3, 0])
    assert np.allclose(ent2.mesh.vertices[n1:], expected2, atol=1e-4)
    assert len(ent2.mesh.faces) == len(ref1.faces) + len(ref2.faces)
    assert ent2.mesh.faces[len(ref1.faces):].min() >= n1, "face indices must be offset"
    print("2. multi-component posing OK (verified transformed vertex positions + face offset)")

    # ========================================================================
    # 3. per-face array alignment when parts carry different explicit arrays
    # ========================================================================
    m_a = engine.cube(1.0, color=(10, 20, 30))
    m_b = engine.cube(1.0, color=(40, 50, 60))
    m_b.face_roughness = np.full(len(m_b.face_roughness), 0.25)
    merged = engine.merge_meshes([(m_a, np.eye(4)), (m_b, np.eye(4))])
    na = len(m_a.faces)
    assert np.allclose(merged.face_roughness[:na], 1.0), "part A keeps its default roughness"
    assert np.allclose(merged.face_roughness[na:], 0.25), "part B's explicit roughness lost"
    assert np.array_equal(merged.face_colors[:na], m_a.face_colors)
    assert np.array_equal(merged.face_colors[na:], m_b.face_colors)
    print("3. per-face array alignment (differing carried arrays) OK")

    # ========================================================================
    # 4. negative-scale winding flip: normals stay outward after a mirror
    # ========================================================================
    mat_mirror = np.eye(4)
    mat_mirror[0, 0] = -1.0
    merged_mirror = engine.merge_meshes([(engine.box(1, 1, 1), mat_mirror)])
    centroids = merged_mirror.vertices[merged_mirror.faces].mean(axis=1)
    dots = np.einsum("ij,ij->i", merged_mirror.normals, centroids)
    assert np.all(dots > 0), f"mirrored box has inward-facing normals: {dots}"
    print("4. negative-scale winding flip OK (normals still outward after mirror)")

    # ========================================================================
    # 5. missing-component-asset skip (never raises) + zero-component -> None
    # ========================================================================
    bp3 = BlueprintAsset({"name": "BP_Missing", "components": [
        {"asset_name": "DoesNotExist12345", "position": [0, 0, 0],
         "rotation": [0, 0, 0], "scale": [1, 1, 1]},
        {"asset_name": A1.name, "position": [1, 1, 1], "rotation": [0, 0, 0],
         "scale": [1, 1, 1]},
    ]}, path="<mem>")
    ent3 = bp3.instantiate(lib)
    assert ent3.mesh is not None and len(ent3.mesh.vertices) == n1
    bp4 = BlueprintAsset({"name": "BP_Empty", "components": []}, path="<mem>")
    ent4 = bp4.instantiate(lib)
    assert ent4.mesh is None
    print("5. missing-asset component skip + zero-component blueprint OK")

    # ========================================================================
    # 6. content browser: blueprint tile plain-click opens ScriptEditorUI,
    #    press+drag-beyond-threshold+release-over-viewport places an
    #    instance -- through the REAL click router (_route_panel_click)
    # ========================================================================
    bp_place = lib.new_blueprint("PlaceableBP")
    bp_place.components = [{"asset_name": A1.name, "position": [0, 0, 0],
                            "rotation": [0, 0, 0], "scale": [1, 1, 1]}]
    bp_place.save()
    editor.bp_icons[bp_place.name] = engine and __import__("editor").make_blueprint_icon()
    folder = lib.create_folder("M2Check", None)
    lib.set_asset_folder(bp_place.name, folder)
    lib.save_folders()
    editor.selected_folder = folder

    layout = editor._layout(W, H)
    content = editor._panel_content_rect("browser", layout)
    blay = editor._browser_layout(content)
    tiles = editor._tiles_in(folder)
    idx = tiles.index(bp_place)
    grid = blay["grid"]
    x0 = grid.x + 10 - editor.browser_scroll + idx * (84 + 8)  # TILE_W+8
    tile_pos = (x0 + 20, grid.y + 6 + 20)

    n_before = len(scene.entities)
    click(tile_pos)
    assert editor.script_ui is not None and editor.script_ui.blueprint is bp_place, \
        "plain click on a blueprint tile must open ScriptEditorUI"
    assert len(scene.entities) == n_before, "plain click must NOT place an entity"
    editor.script_ui.close()
    editor.script_ui = None
    print("6a. plain click (no movement) opens ScriptEditorUI, no placement OK")

    viewport = layout["viewport"]
    vp_pos = (viewport.centerx, viewport.centery)
    mouse_down(tile_pos)
    mouse_move((tile_pos[0] + 30, tile_pos[1] + 5))  # beyond MARQUEE_THRESHOLD (4px)
    mouse_move(vp_pos)
    mouse_up(vp_pos)
    assert editor.script_ui is None, "a real drag-place must NOT open the script editor"
    placed = [e for e in scene.entities if e.blueprint_name == "PlaceableBP"]
    assert len(placed) == 1, f"drag-place must add exactly one instance, got {len(placed)}"
    placed = placed[0]
    assert placed.mesh is not None and placed.asset_name is None
    print("6b. press+drag-beyond-threshold+release-over-viewport places one entity OK")

    # ========================================================================
    # 7. Ctrl+D duplicate and Del delete of a blueprint instance
    # ========================================================================
    editor.selected = placed
    editor.selection = [placed]
    n_before = len(scene.entities)
    editor._duplicate_selected()
    assert len(scene.entities) == n_before + 1, "Ctrl+D must duplicate a blueprint instance"
    dup = editor.selected
    assert dup is not placed and dup.blueprint_name == "PlaceableBP"
    print("7a. Ctrl+D duplicate of a blueprint instance OK")

    editor.selection = [dup]
    editor.selected = dup
    n_before = len(scene.entities)
    editor._delete_selected()
    assert len(scene.entities) == n_before - 1 and dup not in scene.entities
    print("7b. Del delete of a blueprint instance OK")

    # ========================================================================
    # 8. scene save/load round-trip for a blueprint instance
    # ========================================================================
    placed.transform.position = engine.Vec3(3.5, 1.0, -2.0)
    placed.transform.rotation = engine.Vec3(0.1, 0.5, 0.0)
    scene_path = os.path.join(TMP_ASSETS_ROOT, "test_scene.json")
    engine.save_scene(scene, camera, scene_path)
    import json
    with open(scene_path) as f:
        raw = json.load(f)
    ent_specs = [e for e in raw["entities"] if e.get("blueprint") == "PlaceableBP"]
    assert len(ent_specs) == 1 and ent_specs[0]["asset"] is None, \
        "save_scene must include the blueprint instance with asset=None"

    reloaded_scene = engine.load_scene(scene_path, lib, None)
    reloaded = [e for e in reloaded_scene.entities if e.blueprint_name == "PlaceableBP"]
    assert len(reloaded) == 1, "load_scene must round-trip the blueprint instance"
    r = reloaded[0]
    assert abs(r.transform.position.x - 3.5) < 1e-3
    assert abs(r.transform.position.z - (-2.0)) < 1e-3
    assert abs(r.transform.rotation.y - 0.5) < 1e-3
    assert r.mesh is not None and len(r.mesh.vertices) == len(placed.mesh.vertices)
    print("8. scene save/load round-trip OK (position/rotation + mesh)")

    # ========================================================================
    # 9. New/Open Scene must NOT treat a blueprint instance as editor-owned
    # ========================================================================
    assert placed in scene.entities
    editor._replace_scene_content(engine.Scene())
    assert placed not in scene.entities, \
        "_replace_scene_content wrongly carried a blueprint instance over as editor-owned"
    assert any(e.name == "__camera" for e in scene.entities)
    assert any(e.name == "__editor" for e in scene.entities)
    print("9. _replace_scene_content drops blueprint instances, keeps editor-owned ones OK")

    # ========================================================================
    # 10. ScriptEditorUI Components tab, through the REAL event path
    # ========================================================================
    bp_ui = lib.new_blueprint("ComposeBP")
    sui = ScriptEditorUI(editor, bp_ui)
    editor.script_ui = sui
    w, h = eng.screen.get_size()

    assert sui.tab == "script"
    ctab = sui._components_tab_rect(w, h)
    click((ctab.centerx, ctab.centery))
    assert sui.tab == "components"
    editor.draw(eng)  # must not crash with an empty components list
    print("10a. tab switch to Components (empty list draws clean) OK")

    add_btn = sui._add_component_btn_rect(w, h)
    click((add_btn.centerx, add_btn.centery))
    assert sui.add_picker_open is True
    items = sui._add_picker_item_rects(w, h)
    target_row = next(r for asset, r in items if asset.name == A1.name)
    click((target_row.centerx, target_row.centery))
    assert sui.add_picker_open is False
    assert len(bp_ui.components) == 1 and bp_ui.components[0]["asset_name"] == A1.name
    lib.reload()
    assert lib.blueprint_by_name["ComposeBP"].components == bp_ui.components, \
        "Add Component must persist via BlueprintAsset.save()"
    print("10b. Add Component via picker OK, persisted to disk")

    click((add_btn.centerx, add_btn.centery))
    items = sui._add_picker_item_rects(w, h)
    target_row = next(r for asset, r in items if asset.name == A2.name)
    click((target_row.centerx, target_row.centery))
    assert len(bp_ui.components) == 2

    cards = sui._visible_component_cards(w, h)
    idx1, card1 = cards[1]
    assert idx1 == 1
    row_rect = sui._component_field_row_rect(card1, 0)  # Position row
    x_field = Editor._transform_field_rects(row_rect)[0]
    click((x_field.centerx, x_field.centery))
    assert sui.editing_comp_field == (1, "Position", 0)
    sui.comp_edit_buffer = ""
    type_text("4.5")
    key(pygame.K_RETURN, unicode="\r")
    assert sui.editing_comp_field is None, "Enter must commit the field edit"
    assert bp_ui.components[1]["position"][0] == 4.5, bp_ui.components[1]
    lib.reload()
    assert lib.blueprint_by_name["ComposeBP"].components[1]["position"][0] == 4.5
    print("10c. editing a component's Position X field OK, persisted")

    ent = bp_ui.instantiate(lib)
    ref_a2 = lib.instantiate(A2.name).mesh
    n_a1 = len(lib.instantiate(A1.name).mesh.vertices)
    assert np.allclose(ent.mesh.vertices[n_a1:], ref_a2.vertices + np.array([4.5, 0, 0]),
                       atol=1e-4), "re-instantiate() must reflect the Components UI edit"
    print("10d. re-instantiate reflects Components UI edits OK (re-place picks up edits)")

    cards = sui._visible_component_cards(w, h)
    idx0, card0 = cards[0]
    rm = sui._component_remove_btn_rect(card0)
    click((rm.centerx, rm.centery))
    assert len(bp_ui.components) == 1 and bp_ui.components[0]["asset_name"] == A2.name
    print("10e. remove component OK")

    stab = sui._script_tab_rect(w, h)
    click((stab.centerx, stab.centery))
    assert sui.tab == "script"
    editor.draw(eng)
    click((ctab.centerx, ctab.centery))
    assert sui.tab == "components"
    editor.draw(eng)
    print("10f. tab round-trip + draw() clean on both tabs OK")

    sui.close()
    assert editor.script_ui is None
    print("10. ScriptEditorUI Components tab (real event path) OK")

    # ========================================================================
    # 11. screenshot: content browser + a placed blueprint instance in the
    #     viewport (visual sanity check)
    # ========================================================================
    editor.selected_folder = None
    OUT = os.path.join(tempfile.gettempdir(), "judge_bp_component.png")
    eng.run(scene, camera, max_frames=15, screenshot_path=OUT, overlay=editor.draw)
    print(f"11. screenshot saved: {OUT}")

    # ========================================================================
    # no-pollution guards
    # ========================================================================
    _real_settings_after = (open(REAL_SETTINGS, "rb").read()
                            if os.path.exists(REAL_SETTINGS) else None)
    assert _real_settings_after == _real_settings_before, (
        "bp_component_checks touched the real settings.json -- an Editor() in "
        "this suite is missing settings_path=TEST_SETTINGS")
    real_assets_dir = os.path.join(WT, "assets")
    assert not os.path.exists(os.path.join(real_assets_dir, "blueprints",
                                           "composebp.json")), \
        "a blueprint leaked into the REAL assets/blueprints dir"
    assert lib.directory == ASSETS_DIR and lib.directory != real_assets_dir
    print("no-pollution guards OK: real settings.json and real assets/ untouched")

    print("ALL BP COMPONENT CHECKS PASSED")

finally:
    _keys_patch.stop()
    shutil.rmtree(TMP_ASSETS_ROOT, ignore_errors=True)
