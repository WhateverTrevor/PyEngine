"""Judge checks: content browser folder tree -- model logic, persistence,
rect/hit-test consistency, and import-into-folder routing."""
import os
import struct
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np

import engine
from editor import _NO_RENAME, Editor, EditorBehavior, build_starter_scene

OUT = os.path.join(tempfile.gettempdir(), "judge_browser_folders.png")
ASSETS_DIR = os.path.join(WT, "assets")


def fresh_lib():
    return engine.AssetLibrary(ASSETS_DIR)


eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = fresh_lib()
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json")
W, H = eng.screen.get_size()

# make sure a stray manifest from a previous failed run doesn't poison this one
folders_manifest = os.path.join(ASSETS_DIR, "folders.json")
had_manifest = os.path.exists(folders_manifest)
if had_manifest:
    os.remove(folders_manifest)
    lib.reload()

try:
    # 1. folder model: create (nested), rename, select
    assert lib.folders == {} and lib.folder_of == {}
    root_children0 = lib.folder_children(None)
    assert root_children0 == []
    weapons = lib.create_folder("Weapons", None)
    lib.save_folders()
    assert lib.folder_children(None) == [weapons]
    melee = lib.create_folder("Melee", weapons)
    lib.save_folders()
    assert lib.folder_children(weapons) == [melee]
    assert lib.folder_children(None) == [weapons]  # nested child doesn't leak to root
    lib.rename_folder(melee, "Blunt")
    assert lib.folders[melee]["name"] == "Blunt"
    print("folder model OK: create nested, folder_children scoping, rename")

    # 2. editor-level new-folder / rename flow (dedupe default names, F2 rename)
    editor.selected_folder = None
    editor._new_folder()
    fid1 = editor.selected_folder
    assert editor.lib.folders[fid1]["name"] == "New Folder"
    assert editor.renaming_folder == fid1  # new folder enters rename mode immediately
    editor._commit_rename()
    assert editor.renaming_folder is _NO_RENAME
    editor.selected_folder = None
    editor._new_folder()
    fid2 = editor.selected_folder
    assert editor.lib.folders[fid2]["name"] == "New Folder 2", editor.lib.folders[fid2]
    editor._commit_rename()
    editor._begin_rename(fid2)
    editor.rename_buffer = "Ammo"
    editor._commit_rename()
    assert editor.lib.folders[fid2]["name"] == "Ammo"
    assert editor.renaming_folder is _NO_RENAME
    # cancel leaves the prior name untouched
    editor._begin_rename(fid2)
    editor.rename_buffer = "discarded"
    editor._cancel_rename()
    assert editor.lib.folders[fid2]["name"] == "Ammo"
    print("editor new-folder/rename OK: default-name dedupe, commit, cancel")

    # 3. asset-to-folder assignment round-trips through save/reload
    some_asset = lib.assets[0].name
    lib.set_asset_folder(some_asset, weapons)
    lib.save_folders()
    lib2 = fresh_lib()
    assert lib2.folder_of.get(some_asset) == weapons
    assert lib2.folders[weapons]["name"] == "Weapons"
    assert some_asset in {a.name for a in lib2.assets_in(weapons)}
    assert some_asset not in {a.name for a in lib2.assets_in(None)}
    print("asset<->folder persistence OK: round-tripped through a fresh AssetLibrary")

    # 3b. built-in/starter assets default to root (unfiled) unless assigned
    other_asset = lib.assets[1].name
    assert lib2.folder_of.get(other_asset) is None
    assert other_asset in {a.name for a in lib2.assets_in(None)}
    print("starter-asset default OK: unassigned assets land in root")

    # 4. self-healing: a hand-edited manifest pointing at a folder/asset that
    # no longer exists doesn't crash, it just falls back to root
    import json
    with open(folders_manifest, encoding="utf-8") as f:
        raw = json.load(f)
    raw["assignments"]["__does_not_exist__"] = weapons
    raw["assignments"][some_asset] = "999"  # dangling folder id
    with open(folders_manifest, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    lib3 = fresh_lib()
    assert "__does_not_exist__" not in lib3.folder_of
    assert lib3.folder_of.get(some_asset) is None  # dangling folder id -> root
    print("self-healing OK: dangling assignments pruned on load, no crash")

    # restore a clean assignment for the rest of the script
    lib.set_asset_folder(some_asset, weapons)
    lib.save_folders()

    # 5. tree rows: flattened DFS, root first, depth-indented children
    editor.lib.reload()
    rows = editor._folder_tree_rows()
    assert rows[0] == (None, 0, "Assets")
    by_id = {fid: (depth, name) for fid, depth, name in rows if fid is not None}
    assert by_id[weapons][0] == 1 and by_id[weapons][1] == "Weapons"
    assert by_id[melee][0] == 2  # nested one level deeper than its parent
    print("folder tree rows OK: root-first DFS with correct depth")

    # 6. browser layout rects: topbar/tree/grid partition the content rect,
    # and hit-testing a tree row uses the exact rect drawing would use
    # (grow the bottom dock so every folder row created so far is visible,
    # rather than testing the tree's own scroll-clipping here)
    editor.dock_frac["bottom"] = 0.35
    layout = editor._layout(W, H)
    content = editor._panel_content_rect("browser", layout)
    blay = editor._browser_layout(content)
    assert blay["topbar"].y == content.y
    assert blay["tree"].y == blay["topbar"].bottom == blay["grid"].y
    assert blay["tree"].right == blay["grid"].x
    assert blay["tree"].width + blay["grid"].width == content.width
    # clicking inside the "Assets" root row (row 0) selects root
    row0 = editor._tree_row_rect(blay["tree"], 0)
    mp = (row0.centerx, row0.centery)
    idx, fid = editor._tree_row_at(mp, blay["tree"])
    assert idx == 0 and fid is None
    # clicking the Weapons row selects it via the same _route_panel_click path
    editor.selected_folder = None
    w_idx = next(i for i, (f, _d, _n) in enumerate(editor._folder_tree_rows()) if f == weapons)
    wrow = editor._tree_row_rect(blay["tree"], w_idx)
    editor._route_panel_click("browser", (wrow.centerx, wrow.centery), content)
    assert editor.selected_folder == weapons
    print("browser rect/hit-test OK: topbar/tree/grid partition + row click routing agree")

    # 7. grid filters to the selected folder
    editor.selected_folder = weapons
    tile_names = {a.name for a in editor.lib.assets_in(editor.selected_folder)}
    assert some_asset in tile_names
    assert other_asset not in tile_names
    editor.selected_folder = None
    tile_names_root = {a.name for a in editor.lib.assets_in(editor.selected_folder)}
    assert some_asset not in tile_names_root and other_asset in tile_names_root
    print("grid filtering OK: selecting a folder shows only its own assets")

    # 8. topbar buttons: New Folder creates+selects+renames; clicking Import
    # routes through the folder-aware path (exercised for real in step 9)
    nfb = editor._new_folder_btn_rect(blay["topbar"])
    before_folders = set(editor.lib.folders)
    editor._route_panel_click("browser", (nfb.centerx, nfb.centery), content)
    new_ids = set(editor.lib.folders) - before_folders
    assert len(new_ids) == 1
    assert editor.renaming_folder == next(iter(new_ids))
    editor._commit_rename()
    print("New Folder button OK: click creates a folder and enters rename")

    # 9. import routing: call the handler directly with fixture files instead
    # of opening a real dialog (headless-safe) -- both FBX and HDRI extensions
    editor.selected_folder = weapons
    icons_before = set(editor.icons)

    # 9a. FBX fixture (minimal binary FBX, same construction tests/smoke_test.py uses)
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

    cube_verts = (np.array(
        [[-50, -50, -50], [50, -50, -50], [50, 50, -50], [-50, 50, -50],
         [-50, -50, 50], [50, -50, 50], [50, 50, 50], [-50, 50, 50]],
        dtype=np.float64) + 50.0).reshape(-1)
    quads = [(4, 5, 6, 7), (1, 0, 3, 2), (0, 4, 7, 3), (5, 1, 2, 6), (7, 6, 2, 3), (0, 1, 5, 4)]
    pvi = []
    for q in quads:
        pvi += [q[0], q[1], q[2], -(q[3] + 1)]
    GID, MODEL_ID = 5001, 5002
    geometry = ("Geometry", [GID, "Geometry::browsercube", "Mesh"], [
        ("Vertices", [cube_verts], []),
        ("PolygonVertexIndex", [np.array(pvi, dtype=np.int32)], []),
    ])
    objects = ("Objects", [], [geometry, ("Model", [MODEL_ID, "Model::browsercube", "Mesh"], [])])
    connections = ("Connections", [], [("C", ["OO", GID, MODEL_ID], [])])
    fbx_path = os.path.join(tempfile.gettempdir(), "judge_browser_test_cube.fbx")
    header = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400)
    body = _build_node(objects, len(header))
    body += _build_node(connections, len(header) + len(body))
    with open(fbx_path, "wb") as fh:
        fh.write(header + body + b"\x00" * 13)

    editor._import_path_to_folder(fbx_path)
    assert "import failed" not in editor.status[0], editor.status
    fbx_name = editor.status[0].split("'")[1]
    assert editor.lib.folder_of.get(fbx_name) == weapons, "FBX import wasn't filed into the selected folder"
    assert fbx_name in editor.icons and fbx_name not in icons_before
    print(f"import routing OK (FBX): '{fbx_name}' filed into the selected folder")

    # 9b. HDRI fixture
    tmp_hdr = os.path.join(tempfile.gettempdir(), "judge_browser_test_sky.hdr")
    engine.save_hdr(tmp_hdr, np.full((16, 32, 3), 0.4, dtype=np.float32))
    editor.selected_folder = melee
    editor._import_path_to_folder(tmp_hdr)
    assert "import failed" not in editor.status[0], editor.status
    hdr_name = editor.status[0].split("'")[1]
    assert editor.lib.folder_of.get(hdr_name) == melee
    print(f"import routing OK (HDRI): '{hdr_name}' filed into the selected folder")

    # 9c. unsupported extension is rejected cleanly, no exception, no folder side effect
    bogus = os.path.join(tempfile.gettempdir(), "judge_browser_bogus.txt")
    with open(bogus, "w", encoding="utf-8") as f:
        f.write("not an asset")
    before = dict(editor.lib.folder_of)
    editor._import_path_to_folder(bogus)
    assert "unsupported" in editor.status[0]
    assert editor.lib.folder_of == before
    print("import routing OK: unsupported extension rejected without side effects")

    # clean up the imported fixture assets so the repo tree stays clean
    for name in (fbx_name, hdr_name):
        json_path = next((a.path for a in editor.lib.assets if a.name == name), None)
        if json_path and os.path.exists(json_path):
            os.remove(json_path)
    for a in list(editor.lib.assets):
        if a.name == fbx_name:
            model_rel = a.data.get("mesh", {}).get("path")
            if model_rel:
                model_abs = os.path.join(os.path.dirname(a.path), model_rel)
                if os.path.exists(model_abs):
                    os.remove(model_abs)
    hdr_copy = os.path.join(ASSETS_DIR, "hdri", "judge_browser_test_sky.hdr")
    if os.path.exists(hdr_copy):
        os.remove(hdr_copy)
    for p in (fbx_path, tmp_hdr, bogus):
        if os.path.exists(p):
            os.remove(p)
    lib.reload()

    # 10. F2 rename hotkey + Esc-cancels-rename, exercised through Editor.update()
    editor.selected_folder = weapons
    editor.renaming_folder = _NO_RENAME
    import pygame
    pygame.event.get()  # drain any queued events first
    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F2, unicode="", mod=0)
    eng.input.process([ev])
    editor.update(eng, eng.fixed_dt)
    eng.input.consume_edges()
    assert editor.renaming_folder == weapons, "F2 should start renaming the selected folder"
    assert editor.handle_escape() is True
    assert editor.renaming_folder is _NO_RENAME, "Esc should cancel an in-flight rename"
    assert editor.lib.folders[weapons]["name"] == "Weapons", "cancel must not rename"
    print("F2/Esc rename hotkeys OK: F2 starts rename, Esc cancels without renaming")

    # 10b. root-vs-not-renaming collision: root's folder id (None) must never
    # be confused with "not renaming" (a prior bug used None for both, which
    # made the "Assets" root row permanently render as an empty rename box)
    editor.selected_folder = None
    editor.renaming_folder = _NO_RENAME
    assert editor.renaming_folder is not None  # sentinel, not the root id
    root_row_text = next(name for fid, _depth, name in editor._folder_tree_rows()
                         if fid is None)
    assert root_row_text == "Assets"
    is_root_being_renamed = (editor.renaming_folder is not _NO_RENAME
                             and editor.renaming_folder == editor.selected_folder)
    assert is_root_being_renamed is False, \
        "root selected + not renaming must not read as 'renaming root'"
    # F2 on the root folder is a clean no-op (root has no name to edit)
    pygame.event.get()
    ev2 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F2, unicode="", mod=0)
    eng.input.process([ev2])
    editor.update(eng, eng.fixed_dt)
    eng.input.consume_edges()
    assert editor.renaming_folder is _NO_RENAME, "F2 on root must not start a rename"
    editor._begin_rename(None)  # direct call too -- must no-op, not crash
    assert editor.renaming_folder is _NO_RENAME
    print("root/no-rename collision OK: root row shows its name, F2 on root no-ops")

    # screenshot: folder tree with a selection + a couple of folders visible
    editor.selected_folder = weapons
    editor.renaming_folder = _NO_RENAME
    fly = engine.behaviors.FlyController(camera, look_buttons=(3,),
                                         look_guard=lambda p: not editor.over_ui(p))
    editor.fly = fly
    scene.add(engine.Entity("__camera").add_behavior(fly))
    scene.add(engine.Entity("__editor").add_behavior(EditorBehavior(editor)))
    eng.esc_handler = editor.handle_escape
    eng.run(scene, camera, max_frames=20, screenshot_path=OUT, overlay=editor.draw)
    print("screenshot saved")

finally:
    # always leave the assets dir clean, even on assertion failure
    if os.path.exists(folders_manifest):
        os.remove(folders_manifest)
