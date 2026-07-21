"""Judge checks: Unreal-style import-options dialog.

Covers the modal opened by the content browser's Import button after a file
is chosen (`_open_import_dialog`): type detection from extension, editable
Name + cycled target-folder, mesh-only Uniform Scale / Fit-to-~1-unit /
Up-Axis controls (baked into the imported vertices, Unreal-style "bake
import transform"), Cancel-imports-nothing, and post-import browser
navigation (selected_folder + selected_asset land on the new tile so it's
never "hidden in another folder" again). Also exercises the dialog-free
`_do_import` path tests and any future Explorer drag-drop can use directly.

Per the project's hard rule for UI/interaction tests, the dialog's field
editing and every-control click routing drive the REAL event path
(eng.input.process + editor.update), patching pygame.mouse.get_pos/
get_pressed and pygame.key.get_pressed at the OS boundary -- the idiom
tests/marquee_checks.py and tests/docktab_checks.py established.

Imports land in the real assets/ dir (same convention as browser_checks.py
and smoke_test.py -- AssetLibrary always points there), NOT a temp copy;
this suite backs up folders.json, restores it in `finally`, and deletes
every asset .json/.npz/.hdr/texture file it creates. Only the source FBX/
HDR/PNG fixtures fed to the importer live in a temp dir.
"""
import os
import struct
import sys
import tempfile
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np
import pygame

import engine
from editor import Editor, EditorBehavior, build_starter_scene

OUT = os.path.join(tempfile.gettempdir(), "judge_import_dialog.png")
ASSETS_DIR = os.path.join(WT, "assets")
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_import_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(ASSETS_DIR)
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
W, H = eng.screen.get_size()


def new_editor():
    return Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
                 settings_path=TEST_SETTINGS)


class FakeKeys:
    def __init__(self, held=()):
        self._held = set(held)
    def __getitem__(self, key):
        return key in self._held


def step(ed, events):
    eng.input.process(events)
    ed.update(eng, 1 / 60)
    eng.input.consume_edges()


def click(ed, pos):
    """A plain press-then-release at `pos`, driven through the real event
    path (mirrors docktab_checks.py's `click`, itself mirroring
    marquee_checks.py's `press` generator, collapsed to the no-drag case)."""
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step(ed, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)])
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
        step(ed, [])


def type_text(ed, s):
    """Real KEYDOWN-with-unicode events, one per character -- the same
    InputManager.process path a live keystroke takes (e.unicode.isprintable()
    accumulates into text_typed, drained by take_text())."""
    events = [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode=ch, mod=0)
             for ch in s]
    step(ed, events)


def press_key(ed, key):
    step(ed, [pygame.event.Event(pygame.KEYDOWN, key=key, unicode="", mod=0)])


# ----------------------------------------------------------------------------
# FBX fixture builder (same minimal binary-FBX construction browser_checks.py
# and smoke_test.py use) -- builds a box with given half-extents in FILE-SPACE
# CENTIMETERS (no GlobalSettings node, so up_axis defaults to 1/Y and
# extract_geometry applies no auto axis-swap -- only cm->m, *0.01).
# ----------------------------------------------------------------------------
def _enc_prop(v):
    if isinstance(v, np.ndarray) and v.dtype == np.float64:
        return b"d" + struct.pack("<III", len(v), 0, len(v) * 8) + v.tobytes()
    if isinstance(v, np.ndarray) and v.dtype == np.int32:
        return b"i" + struct.pack("<III", len(v), 0, len(v) * 4) + v.tobytes()
    if isinstance(v, str):
        raw = v.encode()
        return b"S" + struct.pack("<I", len(raw)) + raw
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


def build_box_fbx(path, hx, hy, hz, gid=8001, model_id=8002):
    verts = (np.array(
        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
         [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
        dtype=np.float64)).reshape(-1)
    quads = [(4, 5, 6, 7), (1, 0, 3, 2), (0, 4, 7, 3), (5, 1, 2, 6), (7, 6, 2, 3), (0, 1, 5, 4)]
    pvi = []
    for q in quads:
        pvi += [q[0], q[1], q[2], -(q[3] + 1)]
    geometry = ("Geometry", [gid, "Geometry::box", "Mesh"], [
        ("Vertices", [verts], []),
        ("PolygonVertexIndex", [np.array(pvi, dtype=np.int32)], []),
    ])
    objects = ("Objects", [], [geometry, ("Model", [model_id, "Model::box", "Mesh"], [])])
    connections = ("Connections", [], [("C", ["OO", gid, model_id], [])])
    header = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400)
    body = _build_node(objects, len(header))
    body += _build_node(connections, len(header) + len(body))
    with open(path, "wb") as fh:
        fh.write(header + body + b"\x00" * 13)


TMP = tempfile.gettempdir()
# Gat-like: 12cm world extent on every axis (12cm cube -> the exact "invisible
# speck" bug this dialog fixes) -- used for open/detect/scale/fit/cancel/click
tinycube_fbx = os.path.join(TMP, "judge_import_tinycube.fbx")
build_box_fbx(tinycube_fbx, 6, 6, 6)
# asymmetric box (distinct per-axis extents) to make the up-axis Y/Z swap
# numerically detectable -- file-space half-extents X=1cm Y=2cm Z=3cm
axisbox_fbx = os.path.join(TMP, "judge_import_axisbox.fbx")
build_box_fbx(axisbox_fbx, 1, 2, 3, gid=8011, model_id=8012)

hdr_fixture = os.path.join(TMP, "judge_import_sky.hdr")
engine.save_hdr(hdr_fixture, np.full((8, 16, 3), 0.35, dtype=np.float32))

tex_fixture = os.path.join(TMP, "judge_import_swatch.png")
_tex_surf = pygame.Surface((4, 4))
_tex_surf.fill((200, 120, 60))
pygame.image.save(_tex_surf, tex_fixture)

created_asset_names = []  # tracked for cleanup in `finally`

# ----------------------------------------------------------------------------
# backup folders.json exactly like browser_checks.py (never discard a real
# user manifest -- restore it verbatim whether or not this suite passes)
# ----------------------------------------------------------------------------
folders_manifest = os.path.join(ASSETS_DIR, "folders.json")
had_manifest = os.path.exists(folders_manifest)
_manifest_backup = open(folders_manifest, "rb").read() if had_manifest else None

try:
    ed = new_editor()

    # ========================================================================
    # 1. opening the dialog shows it -- does NOT import immediately
    # ========================================================================
    before = set(lib.by_name)
    ed._open_import_dialog(tinycube_fbx)
    assert ed.import_dialog is not None, "Import button must open the options dialog"
    assert set(lib.by_name) == before, "opening the dialog must not import anything yet"
    print("1. dialog-open-defers-import OK")

    # ========================================================================
    # 2. type detection per extension
    # ========================================================================
    ed.import_dialog = None
    cases = [(tinycube_fbx, "mesh", "Static Mesh"),
            (hdr_fixture, "hdri", "HDRI Environment"),
            (tex_fixture, "texture", "Texture")]
    for fpath, exp_kind, exp_label in cases:
        ed._open_import_dialog(fpath)
        assert ed.import_dialog["kind"] == exp_kind, (fpath, ed.import_dialog)
        assert ed.import_dialog["label"] == exp_label, (fpath, ed.import_dialog)
        ed.import_dialog = None
    print("2. type detection OK: fbx->Static Mesh, hdr->HDRI Environment, png->Texture")

    # ========================================================================
    # 3. target folder defaults to selected_folder; a real-event name edit +
    #    scale edit + clicking Import produces a correctly-scaled, correctly-
    #    filed asset, and navigates the browser to it
    # ========================================================================
    weapons = lib.create_folder("Weapons", None)
    lib.save_folders()
    ed.selected_folder = weapons
    ed._open_import_dialog(tinycube_fbx)
    assert ed.import_dialog["folder"] == weapons, "target folder must default to selected_folder"
    rect = ed._import_rect(W, H)

    name_r = ed._import_name_rect(rect)
    click(ed, (name_r.centerx, name_r.centery))
    assert ed.import_field == "name"
    for _ in range(len(ed.import_dialog["name"]) + 4):  # clear the filename default via real Backspace
        press_key(ed, pygame.K_BACKSPACE)
    assert ed.import_dialog["name"] == ""
    type_text(ed, "Import Dialog Cube")
    assert ed.import_dialog["name"] == "Import Dialog Cube", ed.import_dialog["name"]

    scale_r = ed._import_scale_rect(rect)
    click(ed, (scale_r.centerx, scale_r.centery))
    assert ed.import_field == "scale"
    press_key(ed, pygame.K_BACKSPACE)  # clear the default "1"
    type_text(ed, "2.5")
    assert ed.import_dialog["scale_text"] == "2.5", ed.import_dialog["scale_text"]

    ok_r = ed._import_ok_rect(rect)
    click(ed, (ok_r.centerx, ok_r.centery))
    assert ed.import_dialog is None, "Import must close the dialog"
    assert "import failed" not in ed.status[0], ed.status
    assert "Import Dialog Cube" in lib.by_name, lib.by_name.keys()
    created_asset_names.append("Import Dialog Cube")
    assert lib.folder_of.get("Import Dialog Cube") == weapons, \
        "must be filed into the target folder chosen in the dialog"
    assert ed.selected_folder == weapons and ed.selected_asset is lib.by_name["Import Dialog Cube"], \
        "Import must navigate the browser to the new asset (selected_folder + selected_asset)"

    orig_extent = 1.0 / engine.fbx_fit_scale(tinycube_fbx, up_axis="y")  # ~0.12 (12cm cube)
    entity = lib.by_name["Import Dialog Cube"].instantiate()
    extent = entity.mesh.vertices.max(axis=0) - entity.mesh.vertices.min(axis=0)
    expected = orig_extent * 2.5
    assert abs(float(extent.max()) - expected) < 0.01, (extent, expected)
    print(f"3. real-event name+scale edit -> Import OK: bbox largest dim "
         f"{float(extent.max()):.4f} ~= {expected:.4f} (scale 2.5x baked in)")

    # ========================================================================
    # 4. Fit-to-~1-unit button computes and applies a scale that yields a
    #    ~1-unit mesh
    # ========================================================================
    ed._open_import_dialog(tinycube_fbx)
    ed.import_dialog["name"] = "Fit Cube"
    fit_r = ed._import_fit_btn_rect(rect)
    click(ed, (fit_r.centerx, fit_r.centery))
    fit_scale = float(ed.import_dialog["scale_text"])
    assert abs(fit_scale - (1.0 / orig_extent)) < 0.05, fit_scale
    click(ed, (ok_r.centerx, ok_r.centery))
    assert "import failed" not in ed.status[0], ed.status
    created_asset_names.append("Fit Cube")
    fit_entity = lib.by_name["Fit Cube"].instantiate()
    fit_extent = fit_entity.mesh.vertices.max(axis=0) - fit_entity.mesh.vertices.min(axis=0)
    assert abs(float(fit_extent.max()) - 1.0) < 0.02, fit_extent
    print(f"4. Fit-to-~1-unit OK: computed scale {fit_scale:.3f}, "
         f"resulting bbox largest dim {float(fit_extent.max()):.4f}")

    # ========================================================================
    # 5. Up Axis Z swaps the mesh orientation as expected (Y<->Z, sign flip
    #    on the axis that was negated -- extent is unaffected by sign)
    # ========================================================================
    ed._open_import_dialog(axisbox_fbx)
    ed.import_dialog["name"] = "Axis Y Baseline"
    click(ed, (ok_r.centerx, ok_r.centery))
    assert "import failed" not in ed.status[0], ed.status
    created_asset_names.append("Axis Y Baseline")
    baseline_ent = lib.by_name["Axis Y Baseline"].instantiate()
    base_extent = baseline_ent.mesh.vertices.max(axis=0) - baseline_ent.mesh.vertices.min(axis=0)

    ed._open_import_dialog(axisbox_fbx)
    ed.import_dialog["name"] = "Axis Z Swapped"
    y_btn, z_btn = ed._import_axis_rects(rect)
    click(ed, (z_btn.centerx, z_btn.centery))
    assert ed.import_dialog["up_axis"] == "z"
    click(ed, (ok_r.centerx, ok_r.centery))
    assert "import failed" not in ed.status[0], ed.status
    created_asset_names.append("Axis Z Swapped")
    swapped_ent = lib.by_name["Axis Z Swapped"].instantiate()
    swap_extent = swapped_ent.mesh.vertices.max(axis=0) - swapped_ent.mesh.vertices.min(axis=0)
    expected_swap = np.array([base_extent[0], base_extent[2], base_extent[1]])
    assert np.allclose(swap_extent, expected_swap, atol=1e-4), (swap_extent, expected_swap)
    assert not np.allclose(swap_extent, base_extent, atol=1e-4), \
        "Z up-axis must actually change the mesh vs the Y baseline"
    print(f"5. Up Axis Z OK: baseline extent {base_extent}, "
         f"Z-swapped extent {swap_extent} (Y<->Z permuted)")

    # ========================================================================
    # 6. Cancel imports nothing
    # ========================================================================
    before_cancel = set(lib.by_name)
    ed._open_import_dialog(tinycube_fbx)
    cancel_r = ed._import_cancel_rect(rect)
    click(ed, (cancel_r.centerx, cancel_r.centery))
    assert ed.import_dialog is None, "Cancel must close the dialog"
    assert set(lib.by_name) == before_cancel, "Cancel must not import anything"
    print("6. Cancel-imports-nothing OK")

    # the X close button must behave the same as Cancel
    ed._open_import_dialog(tinycube_fbx)
    close_r = ed._import_close_rect(rect)
    click(ed, (close_r.centerx, close_r.centery))
    assert ed.import_dialog is None
    assert set(lib.by_name) == before_cancel
    print("6b. X close button OK: also cancels without importing")

    # ========================================================================
    # 7. texture import path still works; scale/up-axis are inert (kind !=
    #    "mesh", so clicking those rects must not even focus a field)
    # ========================================================================
    ed._open_import_dialog(tex_fixture)
    assert ed.import_dialog["kind"] == "texture"
    scale_r = ed._import_scale_rect(rect)
    click(ed, (scale_r.centerx, scale_r.centery))
    assert ed.import_field is None, "scale field must be inert for a non-mesh import"
    y_btn, z_btn = ed._import_axis_rects(rect)
    click(ed, (z_btn.centerx, z_btn.centery))
    assert ed.import_dialog["up_axis"] == "y", "up-axis buttons must be inert for a texture import"
    ed.import_dialog["name"] = "Import Dialog Swatch"
    click(ed, (ok_r.centerx, ok_r.centery))
    assert "import failed" not in ed.status[0], ed.status
    assert "Import Dialog Swatch" in lib.by_name
    created_asset_names.append("Import Dialog Swatch")
    print("7. texture import OK: name+folder applied, scale/up-axis inert, no crash")

    # ========================================================================
    # 8. a real-event click on every dialog control doesn't crash (the
    #    ctx-menu crash history makes this mandatory, not optional)
    # ========================================================================
    control_rects = []
    ed._open_import_dialog(tinycube_fbx)
    r = ed._import_rect(W, H)
    control_rects.append(ed._import_name_rect(r))
    prev_btn, _label, next_btn = ed._import_folder_rects(r)
    control_rects += [prev_btn, next_btn]
    control_rects.append(ed._import_scale_rect(r))
    control_rects.append(ed._import_fit_btn_rect(r))
    yb, zb = ed._import_axis_rects(r)
    control_rects += [yb, zb]
    for cr in control_rects:
        if ed.import_dialog is None:  # a prior click in this loop shouldn't close it
            ed._open_import_dialog(tinycube_fbx)
            r = ed._import_rect(W, H)
        click(ed, (cr.centerx, cr.centery))
    assert ed.import_dialog is not None, "none of these controls should close the dialog"
    click(ed, (ed._import_cancel_rect(r).centerx, ed._import_cancel_rect(r).centery))
    assert ed.import_dialog is None
    print("8. every-control real-event click OK: no crash, Cancel still closes cleanly")

    # ========================================================================
    # 9. _do_import: dialog-free path (tests / future Explorer drag-drop)
    # ========================================================================
    before_direct = set(lib.by_name)
    ed._do_import(tinycube_fbx, name="Direct Import Cube", folder=None, scale=3.0, up_axis="y")
    assert "import failed" not in ed.status[0], ed.status
    assert "Direct Import Cube" in lib.by_name
    created_asset_names.append("Direct Import Cube")
    assert lib.folder_of.get("Direct Import Cube") is None  # folder=None -> root
    assert ed.selected_asset is lib.by_name["Direct Import Cube"]
    direct_ent = lib.by_name["Direct Import Cube"].instantiate()
    direct_extent = direct_ent.mesh.vertices.max(axis=0) - direct_ent.mesh.vertices.min(axis=0)
    assert abs(float(direct_extent.max()) - orig_extent * 3.0) < 0.01, direct_extent
    print("9. _do_import dialog-free path OK: scale baked, filed, browser navigated")

    # screenshot: dialog open over the editor, for visual review
    ed._open_import_dialog(tinycube_fbx)
    fly = engine.behaviors.FlyController(camera, look_buttons=(3,),
                                         look_guard=lambda p: not ed.over_ui(p))
    ed.fly = fly
    scene.add(engine.Entity("__camera").add_behavior(fly))
    scene.add(engine.Entity("__editor").add_behavior(EditorBehavior(ed)))
    eng.esc_handler = ed.handle_escape
    eng.run(scene, camera, max_frames=10, screenshot_path=OUT, overlay=ed.draw)
    print("screenshot saved:", OUT)

finally:
    # remove every asset this suite created (json + any model/texture/hdri
    # payload file it wrote), then restore the real folders.json verbatim
    for nm in created_asset_names:
        a = lib.by_name.get(nm)
        if a is None:
            continue
        if os.path.exists(a.path):
            os.remove(a.path)
        mesh_rel = a.data.get("mesh", {}).get("path")
        if mesh_rel:
            p = os.path.join(os.path.dirname(a.path), mesh_rel)
            if os.path.exists(p):
                os.remove(p)
        env_rel = a.data.get("environment", {}).get("hdri")
        if env_rel:
            p = os.path.join(os.path.dirname(a.path), env_rel)
            if os.path.exists(p):
                os.remove(p)
        tex_rel = a.data.get("texture", {}).get("path") if "texture" in a.data else None
        if tex_rel:
            p = os.path.join(os.path.dirname(a.path), tex_rel)
            if os.path.exists(p):
                os.remove(p)
    lib.reload()
    if os.path.exists(folders_manifest):
        os.remove(folders_manifest)
    if _manifest_backup is not None:
        with open(folders_manifest, "wb") as f:
            f.write(_manifest_backup)
    for p in (tinycube_fbx, axisbox_fbx, hdr_fixture, tex_fixture):
        if os.path.exists(p):
            os.remove(p)
