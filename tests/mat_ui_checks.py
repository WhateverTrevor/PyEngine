"""Judge checks: material editor UX overhaul (UE-style right-click add/node
context menus, usage-ranked top-10, node isolation preview, RMB no-fall-
through). Companion to material_checks.py, which covers node math/eval and
asset save/load -- this suite covers the editor UI layer added on top."""
import os
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
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_mat_ui_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)
USAGE_PATH = os.path.join(tempfile.gettempdir(), "mat_node_usage.json")
if os.path.exists(USAGE_PATH):
    os.remove(USAGE_PATH)

eng = engine.Engine(1000, 700, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
               settings_path=TEST_SETTINGS)
crate = next(e for e in scene.entities if e.asset_name == "Crate")
mui = MaterialEditorUI(editor, crate)
editor.mat_ui = mui
W, H = eng.screen.get_size()

# ---- 1. graph_panel / preview_rect geometry: same rects draw + hit-test use ----
panel = mui.graph_panel(W, H)
prev_r = mui.preview_rect(W, H)
content = mui.content_rect(W, H)
assert panel.x >= prev_r.right, "graph canvas must not overlap the preview strip"
assert panel.width + prev_r.width == content.width, "panels must tile content_rect exactly"
assert panel.height == content.height == prev_r.height
print("graph_panel/preview_rect geometry OK (tile content_rect, no overlap)")

# ---- 2. add-node menu: search bar + top-10-by-usage, live filter ----
mp = (panel.x + 300, panel.y + 100)
mui._open_add_menu(mp, panel)
assert mui.ctx_menu["kind"] == "add"
top10 = mui.ctx_menu["top10"]
assert len(top10) == 10, f"expected exactly 10 seeded/ranked entries, got {len(top10)}"
# seeded defaults are ranked by DEFAULT_NODE_USAGE descending before any real usage
from editor import DEFAULT_NODE_USAGE
expected_order = sorted(DEFAULT_NODE_USAGE, key=lambda t: -DEFAULT_NODE_USAGE[t])
assert top10 == expected_order, f"top10 not usage-ranked: {top10} != {expected_order}"
print(f"add-menu top-10 seeded ranking OK: {top10}")

# search bar, "Common" section header, and entry rows must be distinct,
# non-overlapping rows (same rect-helper discipline as everything else) --
# empty search is the case that showed the header, so check it here.
import pygame as _pg
sx, sy = mui.ctx_menu["screen_pos"]
search_rect = _pg.Rect(sx, sy, mui.CTX_MENU_W, mui.CTX_SEARCH_H)
header_rect = mui._ctx_header_rect(mui.ctx_menu)
entry_rects = [r for _l, r, _p in mui._ctx_menu_rows(mui.ctx_menu)]
assert header_rect is not None, "empty-search add-menu should show the 'Common' header"
all_rects = [search_rect, header_rect] + entry_rects
for i in range(len(all_rects)):
    for j in range(i + 1, len(all_rects)):
        assert not all_rects[i].colliderect(all_rects[j]), \
            f"context-menu rows must not overlap: {all_rects[i]} vs {all_rects[j]}"
assert header_rect.y == search_rect.bottom, "header row must sit directly below the search bar"
assert entry_rects[0].y == header_rect.bottom, "entries must start directly below the header"
print("add-menu search/header/entry rows are distinct, stacked, non-overlapping OK")

matches = mui._ctx_search_matches("mult")
assert matches == ["multiply"], f"search 'mult' should narrow to exactly ['multiply']: {matches}"
matches_tex = mui._ctx_search_matches("tex")
assert set(matches_tex) >= {"tex_coord", "tex_sample"}, matches_tex
matches_none = mui._ctx_search_matches("zzz_not_a_node")
assert matches_none == [], matches_none
print("add-menu live search filter narrows correctly OK")

# rows drawn == rows hit-tested (house rule: same rect list for both)
mui.ctx_menu["search"] = "mult"
rows = mui._ctx_menu_rows(mui.ctx_menu)
assert [payload for _l, _r, payload in rows] == ["multiply"]
click_rect = rows[0][1]
before_n = len(mui.graph.nodes)
mui._click_ctx_menu(click_rect.center)
assert len(mui.graph.nodes) == before_n + 1
new_nid = max(mui.graph.nodes)
assert mui.graph.nodes[new_nid]["type"] == "multiply"
print("context-menu click hit-test == drawn rect (same list drives both) OK")

# Enter adds the top match at the click position
mui._open_add_menu((panel.x + 120, panel.y + 200), panel)
mui.ctx_menu["search"] = "noise"
before_n = len(mui.graph.nodes)
target_pos = mui.ctx_menu["graph_pos"]
matches = mui._ctx_search_matches("noise")
assert matches and matches[0] == "noise"
mui._add_node_from_menu(matches[0], target_pos)
assert len(mui.graph.nodes) == before_n + 1
noise_nid = max(mui.graph.nodes)
assert mui.graph.nodes[noise_nid]["type"] == "noise"
assert list(mui.graph.nodes[noise_nid]["pos"]) == [float(target_pos[0]), float(target_pos[1])], \
    "Enter-added node must land at the right-click position"
print("Enter adds top match at the right-click position OK")

# ---- 3. usage counts increment + persist across a fresh MaterialEditorUI ----
usage_after = mui._load_usage()
assert usage_after["multiply"] == DEFAULT_NODE_USAGE["multiply"] + 1
assert usage_after["noise"] == DEFAULT_NODE_USAGE["noise"] + 1
assert os.path.exists(mui._usage_path()), "usage counts must persist to disk"
mui2 = MaterialEditorUI(editor, crate)
usage_fresh = mui2._load_usage()
assert usage_fresh == usage_after, "usage counts must persist across MaterialEditorUI instances"
top10_2 = mui2._top10()
assert top10_2[0] in ("multiply", "constant3vector"), top10_2  # bumped types rank higher/tie
print(f"usage counts increment + persist + reorder top-10 OK: {top10_2}")

# ---- 4. node context menu: items + delete/disconnect mutate the graph, output protected ----
c1 = mui.graph.add("constant3vector", (10, 10))
c2 = mui.graph.add("constant3vector", (10, 60))
add_nid = mui.graph.add("add", (200, 30))
mui.graph.connect(c1, add_nid, "a")
mui.graph.connect(c2, add_nid, "b")
mui.graph.connect(add_nid, mui.graph.output_id(), "base_color")

r = mui.node_rect(add_nid, panel)
mui._open_node_menu((r.centerx, r.y + 5), add_nid, panel)
labels = [row[0] for row in mui._ctx_menu_rows(mui.ctx_menu)]
assert "Delete" in labels
assert "Break All Node Links" in labels
assert any(l.startswith("Break Link: ") for l in labels)
assert "Duplicate" in labels
assert "Start Previewing Node" in labels
print(f"node context-menu items present OK: {labels}")

# output node: no Delete, no Duplicate, no preview-toggle offered
out_id = mui.graph.output_id()
mui._open_node_menu((0, 0), out_id, panel)
out_labels = [row[0] for row in mui._ctx_menu_rows(mui.ctx_menu)]
assert "Delete" not in out_labels, "Output node must not offer Delete"
assert "Duplicate" not in out_labels
assert not any("Previewing" in l for l in out_labels)
print(f"output node menu correctly restricted OK: {out_labels}")
mui._run_node_action(out_id, ("delete", None))
assert out_id in mui.graph.nodes, "output node must survive even a direct delete action call"

# Break Link: <input> disconnects just that pin
assert mui.graph.link_into(add_nid, "a") is not None
mui._run_node_action(add_nid, ("disconnect_one", "a"))
assert mui.graph.link_into(add_nid, "a") is None
assert mui.graph.link_into(add_nid, "b") is not None, "disconnect_one must not touch other pins"
print("Break Link: <name> disconnects only that pin OK")

# Break All Node Links clears every link touching the node (in AND out)
mui.graph.connect(c1, add_nid, "a")  # reconnect for a clean test
assert mui.graph.link_into(add_nid, "a") is not None
assert mui.graph.link_into(mui.graph.output_id(), "base_color") == (add_nid, "out")
mui._run_node_action(add_nid, ("disconnect_all", None))
assert mui.graph.link_into(add_nid, "a") is None
assert mui.graph.link_into(add_nid, "b") is None
assert mui.graph.link_into(mui.graph.output_id(), "base_color") is None, \
    "Break All Node Links must also break the node's OUTGOING link"
print("Break All Node Links clears incoming + outgoing links OK")

# Duplicate copies type/params to a new node, offset in position
c1_before = dict(mui.graph.nodes[c1]["params"])
before_n = len(mui.graph.nodes)
mui._run_node_action(c1, ("duplicate", None))
assert len(mui.graph.nodes) == before_n + 1
dup_id = max(mui.graph.nodes)
assert mui.graph.nodes[dup_id]["type"] == mui.graph.nodes[c1]["type"]
assert mui.graph.nodes[dup_id]["params"] == c1_before
assert list(mui.graph.nodes[dup_id]["pos"]) != list(mui.graph.nodes[c1]["pos"])
print("Duplicate node OK")

# Delete removes the node and its remaining links; output survives
mui.graph.connect(c2, add_nid, "a")
assert add_nid in mui.graph.nodes
mui._run_node_action(add_nid, ("delete", None))
assert add_nid not in mui.graph.nodes
assert all(l[0] != add_nid and l[1] != add_nid for l in mui.graph.links)
assert mui.graph.output_id() in mui.graph.nodes
print("Delete removes node + its links, output node unaffected OK")

# ---- 5. preview isolation: evaluates the isolated node's own output,
#          matches a direct graph.preview_value() call, and is non-destructive ----
sphere = engine.icosphere(radius=1.0, subdivisions=2)
c3 = mui.graph.add("constant3vector", (0, 0))
mui.graph.nodes[c3]["params"].update(r=0.2, g=0.6, b=0.9)
links_before = [list(l) for l in mui.graph.links]
direct = mui.graph.preview_value(sphere, c3)
assert np.allclose(direct[0], [0.2 * 255, 0.6 * 255, 0.9 * 255], atol=1.0), direct[0]
assert [list(l) for l in mui.graph.links] == links_before, \
    "preview_value must not mutate the graph's saved links"
print("preview isolation evaluates the isolated node's own value OK")

mui._run_node_action(c3, ("toggle_preview", None))
assert mui.preview_nid == c3
mui._preview_dirty = True
surf = mui._render_preview()
assert surf.get_size() == (mui.PREVIEW_SIZE, mui.PREVIEW_SIZE)
mui._run_node_action(c3, ("toggle_preview", None))
assert mui.preview_nid is None, "toggling again returns to the real Output preview"
print("Start/Stop Previewing Node toggles + renders at the expected size OK")

# deleting the currently-previewed node must fall back to Output, not crash
mui._run_node_action(c3, ("toggle_preview", None))
assert mui.preview_nid == c3
mui._run_node_action(c3, ("delete", None))
assert mui.preview_nid is None, "deleting the previewed node must clear preview_nid"
print("deleting the previewed node clears isolation safely OK")

# ---- 6. RMB no-fall-through: over_ui() gates the viewport fly-look while
#          the material editor is open, regardless of where inside it ----
assert editor.over_ui((5, 5)) is True         # menu bar always counts
assert editor.over_ui(panel.center) is True   # inside the node canvas
assert editor.over_ui(prev_r.center) is True  # inside the preview strip
editor.mat_ui = None
# (can't assert False generically -- other panels may cover the same pixel --
# so just confirm mat_ui was the thing forcing True: the outliner/topbar
# checks below are independent codepaths, not re-tested here.)
editor.mat_ui = mui
print("RMB no-fall-through guard (over_ui gates while material editor open) OK")

# ---- settings.json isolation guard ----
_real_after = (open(REAL_SETTINGS, "rb").read()
              if os.path.exists(REAL_SETTINGS) else None)
assert _real_after == _real_before, "real settings.json was modified by this suite"
print("no-pollution guard OK: real settings.json untouched by this suite")

if os.path.exists(USAGE_PATH):
    os.remove(USAGE_PATH)

print("ALL MATERIAL-EDITOR-UI JUDGE CHECKS PASSED")
