"""Judge checks: dock-preview ghost + tabbed panels.

Covers the group-based dock model (`dock_order[side]` is now a list of
{"ids": [...], "active": pid} groups, not a flat pid list): old-format
settings migration, degenerate (single-panel-group) layout staying
byte-identical to the pre-change flat-list math, drop-onto-title tab-
joining, tab click-to-switch, drag-a-tab-out un-tabbing to floating,
drop-on-the-edge-band still side-stacking (no tabbing), the drag preview's
no-drift guarantee (the ghost rect during a drag equals the rect the panel
actually gets after the drop, for both the band and tab-join cases),
settings round-trip of groups+active tab, and Reset Layout restoring the
factory (ungrouped) layout.

Per the project's hard rule for UI/interaction tests, the two genuinely
interactive checks (tab click switches active; a press on a tab strip must
not start a marquee) drive the REAL event path (eng.input.process +
editor.update), patching pygame.mouse.get_pos/get_pressed and
pygame.key.get_pressed at the OS boundary -- the idiom proven in
tests/marquee_checks.py. The drag-MECHANICS checks (drop math, preview
simulation, migration, settings round-trip) call the same internal
_begin_panel_drag/_finish_panel_drag/_layout API tests/window_checks.py
already calls directly (section 12 there is the precedent) -- these aren't
raw pygame state, they're the editor's own documented drag primitives.
"""
import os
import sys
import tempfile
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import pygame

import engine
from editor import (MENU_H, MIN_DOCK_W, MIN_PANEL_W, PANEL_TITLE_H, SPLITTER_PX,
                    Editor, EditorBehavior, build_starter_scene)

REAL_SETTINGS = os.path.join(WT, "settings.json")
_real_before = (open(REAL_SETTINGS, "rb").read()
               if os.path.exists(REAL_SETTINGS) else None)
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_docktab_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
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
    path (mirrors marquee_checks.py's `press` generator, collapsed to the
    no-drag case since these checks only need a tap)."""
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step(ed, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)])
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
        step(ed, [])


# ============================================================================
# 1. old-format settings migrate to groups
# ============================================================================
legacy = {"left": [], "right": ["outliner", "details"], "bottom": ["browser"]}
data = {"dock_order": legacy, "floating": []}
ed1 = new_editor()
ed1._apply_layout_settings(data)
assert ed1.dock_order["right"] == [
    {"ids": ["outliner"], "active": "outliner"},
    {"ids": ["details"], "active": "details"},
], ed1.dock_order["right"]
# "console" isn't in this legacy fixture (it predates that panel) --
# _apply_layout_settings auto-places it as a fresh solo group on "bottom"
# (its preferred fallback side, see _PANEL_ALLOWED_SIDES) rather than
# resetting the whole migrated layout just because one new id is missing
assert ed1.dock_order["bottom"] == [{"ids": ["browser"], "active": "browser"},
                                    {"ids": ["console"], "active": "console"}]
assert ed1.dock_order["left"] == []
print("old-format settings migrate to solo groups OK (+ console auto-placed)")

# ============================================================================
# 2. degenerate single-panel groups render byte-identical layout rects to
#    the pre-change math, for the factory layout
# ============================================================================
def pre_tabs_layout(ed, w, h, flat_dock_order):
    """Independent reimplementation of the pre-groups `_layout` algorithm
    (flat {side: [pid, ...]} instead of groups) -- the no-drift proof that
    today's degenerate-group rendering hasn't silently changed anything."""
    menu = pygame.Rect(0, 0, w, MENU_H)
    tb_w = ed._side_toolbar_w()  # side toolbar claims the window's left edge
                                  # (see editor.py's _layout) -- must match here too
    left_ids = [p for p in flat_dock_order["left"] if ed.panel_visible.get(p, True)]
    right_ids = [p for p in flat_dock_order["right"] if ed.panel_visible.get(p, True)]
    bottom_ids = [p for p in flat_dock_order["bottom"] if ed.panel_visible.get(p, True)]
    left_w = 0
    if left_ids:
        left_w = MIN_DOCK_W if ed._all_minimized(left_ids) else ed._dock_side_w("left", w)
    right_w = 0
    if right_ids:
        right_w = MIN_DOCK_W if ed._all_minimized(right_ids) else ed._dock_side_w("right", w)
    if left_w + right_w > max(0, w - tb_w - MIN_PANEL_W):
        scale = max(0, w - tb_w - MIN_PANEL_W) / max(1, left_w + right_w)
        left_w, right_w = int(left_w * scale), int(right_w * scale)
    bottom_h = 0
    if bottom_ids:
        bottom_h = PANEL_TITLE_H if ed._all_minimized(bottom_ids) else ed._dock_bottom_h(h)
    top = MENU_H
    stack_bottom = h - bottom_h
    panels = {}

    def stack(ids, x, width):
        n = len(ids)
        if n == 0:
            return
        avail = max(0, stack_bottom - top)
        minimized = [p for p in ids if ed.panel_minimized.get(p, False)]
        normal = [p for p in ids if not ed.panel_minimized.get(p, False)]
        remain = max(0, avail - len(minimized) * PANEL_TITLE_H)
        n_normal = len(normal)
        share = remain // n_normal if n_normal else 0
        y = top
        for pid in ids:
            if pid in minimized:
                hh = PANEL_TITLE_H
            else:
                idx = normal.index(pid)
                hh = share if idx < n_normal - 1 else remain - share * (n_normal - 1)
            panels[pid] = pygame.Rect(x, y, width, max(0, hh))
            y += hh

    stack(left_ids, tb_w, left_w)
    stack(right_ids, w - right_w, right_w)
    if bottom_ids:
        bw_total = max(0, w - tb_w - left_w - right_w)
        n = len(bottom_ids)
        share = bw_total // n
        x = tb_w + left_w
        for i, pid in enumerate(bottom_ids):
            ww = share if i < n - 1 else bw_total - share * (n - 1)
            hh = PANEL_TITLE_H if ed.panel_minimized.get(pid, False) else bottom_h
            panels[pid] = pygame.Rect(x, stack_bottom, max(0, ww), hh)
            x += ww
    for pid in ("outliner", "details", "browser", "console"):
        if not ed.panel_visible.get(pid, True) or pid in panels:
            continue
        r = ed._float_rect_for(pid, w, h)
        if ed.panel_minimized.get(pid, False):
            r = pygame.Rect(r.x, r.y, r.width, PANEL_TITLE_H)
        panels[pid] = r
    viewport = pygame.Rect(tb_w + left_w, top, max(0, w - tb_w - left_w - right_w),
                           max(0, stack_bottom - top))
    return {"viewport": viewport, "panels": panels,
            "left_w": left_w, "right_w": right_w, "bottom_h": bottom_h}


ed2 = new_editor()
factory_flat = {"left": [], "right": ["outliner", "details"],
                "bottom": ["browser", "console"]}
old = pre_tabs_layout(ed2, W, H, factory_flat)
new = ed2._layout(W, H)
assert new["viewport"] == old["viewport"]
assert new["left_w"] == old["left_w"] and new["right_w"] == old["right_w"]
assert new["bottom_h"] == old["bottom_h"]
for pid in ("outliner", "details", "browser", "console"):
    assert new["panels"][pid] == old["panels"][pid], (pid, new["panels"][pid], old["panels"][pid])
print("degenerate factory layout OK: byte-identical to the pre-groups flat-list math")

# ============================================================================
# 3. drop-onto-title tabs two panels: strip appears, one visible, correct active
# ============================================================================
ed3 = new_editor()
lay = ed3._layout(W, H)
outliner_rect = lay["panels"]["outliner"]
details_rect = lay["panels"]["details"]
ed3._begin_panel_drag("details", (details_rect.x + 5, details_rect.y + 5), details_rect)
drop_pos = (outliner_rect.x + 10, outliner_rect.y + 5)  # over outliner's title bar
ed3._finish_panel_drag(drop_pos, W, H)

side, group = ed3._group_for_pid("details")
assert side == "right"
assert group is ed3._group_for_pid("outliner")[1], "outliner+details must share ONE group object"
assert group["ids"] == ["outliner", "details"], group["ids"]
assert group["active"] == "details", "the just-dropped tab becomes active"
lay3 = ed3._layout(W, H)
assert "details" in lay3["panels"] and "outliner" not in lay3["panels"], (
    "only the active tab gets a panels rect -- the inactive one must not "
    "also render/hit-test as a separate panel")
tabs3 = ed3._panel_tab_strip("details", lay3["panels"]["details"])
assert tabs3 is not None and set(tabs3.keys()) == {"outliner", "details"}
print("drop-onto-title tab-join OK: shared group, one visible (details, the "
      "dropped tab), strip shows both")

# ============================================================================
# 4. clicking a tab switches active (real events)
# ============================================================================
outliner_tab_rect = tabs3["outliner"]
click(ed3, outliner_tab_rect.center)
side4, group4 = ed3._group_for_pid("outliner")
assert group4["active"] == "outliner", "clicking the outliner tab must switch active"
assert group4["ids"] == ["outliner", "details"], (
    "a plain switch-click must not reorder the tab strip", group4["ids"])
print("tab click switches active OK (real events), no reordering")

# ============================================================================
# 5. dragging a tab out un-tabs it back to floating
# ============================================================================
lay5 = ed3._layout(W, H)
active_rect = lay5["panels"]["outliner"]  # outliner is active again after #4
tabs5 = ed3._panel_tab_strip("outliner", active_rect)
details_tab_rect = tabs5["details"]
ed3._begin_panel_drag("details", details_tab_rect.center, active_rect)
viewport_mid = (lay5["viewport"].centerx, lay5["viewport"].centery)
ed3._finish_panel_drag(viewport_mid, W, H)
assert "details" in ed3.floating, ed3.floating
assert ed3._group_for_pid("details") is None
side5, group5 = ed3._group_for_pid("outliner")
assert group5["ids"] == ["outliner"], "outliner's group is solo/degenerate again"
print("drag-tab-out OK: un-tabbed back to floating, source group degenerate again")

# ============================================================================
# 6. drop-on-edge-band still side-stacks (no tabbing)
# ============================================================================
ed6 = new_editor()
ed6._dock_panel("details", "float")
lay6 = ed6._layout(W, H)
drect6 = lay6["panels"]["details"]
ed6._begin_panel_drag("details", (drect6.x, drect6.y), drect6)
# drop near the BOTTOM of the right dock's band, well clear of outliner's
# title/tab-strip rect at the top, so this must resolve to "band" not "tab"
right_zone = ed6._dock_zone_rect("right", W, H, lay6)
band_pos = (right_zone.centerx, right_zone.bottom - 10)
target6 = ed6._panel_drag_target("details", band_pos, W, H, lay6)
assert target6["kind"] == "band", target6
ed6._finish_panel_drag(band_pos, W, H)
assert ed6.dock_order["right"] == [
    {"ids": ["outliner"], "active": "outliner"},
    {"ids": ["details"], "active": "details"},
], ed6.dock_order["right"]
print("drop-on-edge-band OK: still side-stacks as a separate solo group, no tabbing")

# ============================================================================
# 7. no-drift: the preview rect during a drag equals the rect the panel
#    actually gets after the drop -- band case AND tab case
# ============================================================================
ed7 = new_editor()
ed7._dock_panel("details", "float")
lay7 = ed7._layout(W, H)
drect7 = lay7["panels"]["details"]
ed7._begin_panel_drag("details", (drect7.x, drect7.y), drect7)
right_zone7 = ed7._dock_zone_rect("right", W, H, lay7)
band_pos7 = (right_zone7.centerx, right_zone7.bottom - 10)
target7 = ed7._panel_drag_target("details", band_pos7, W, H, lay7)
assert target7["kind"] == "band"
preview7 = ed7._simulate_drop("details", target7, W, H)
ed7._finish_panel_drag(band_pos7, W, H)
actual7 = ed7._layout(W, H)["panels"]["details"]
assert preview7 == actual7, (preview7, actual7)
print(f"no-drift OK (band case): preview {tuple(preview7)} == actual {tuple(actual7)}")

ed7b = new_editor()
ed7b._dock_panel("details", "float")
lay7b = ed7b._layout(W, H)
drect7b = lay7b["panels"]["details"]
ed7b._begin_panel_drag("details", (drect7b.x, drect7b.y), drect7b)
outliner_rect7b = lay7b["panels"]["outliner"]
tab_pos7b = (outliner_rect7b.x + 10, outliner_rect7b.y + 5)
target7b = ed7b._panel_drag_target("details", tab_pos7b, W, H, lay7b)
assert target7b["kind"] == "tab", target7b
preview7b = ed7b._simulate_drop("details", target7b, W, H)
ed7b._finish_panel_drag(tab_pos7b, W, H)
actual7b = ed7b._layout(W, H)["panels"]["details"]
assert preview7b == actual7b, (preview7b, actual7b)
print(f"no-drift OK (tab case): preview {tuple(preview7b)} == actual {tuple(actual7b)}")

# ============================================================================
# 8. marquee doesn't start on tab strips
# ============================================================================
ed8 = new_editor()
lay8 = ed8._layout(W, H)
o_rect = lay8["panels"]["outliner"]
d_rect = lay8["panels"]["details"]
ed8._begin_panel_drag("details", (d_rect.x + 5, d_rect.y + 5), d_rect)
ed8._finish_panel_drag((o_rect.x + 10, o_rect.y + 5), W, H)  # tab-join, as in #3
lay8b = ed8._layout(W, H)
active_rect8 = lay8b["panels"]["details"]
tabs8 = ed8._panel_tab_strip("details", active_rect8)
assert tabs8 is not None
inactive_tab_rect8 = tabs8["outliner"]
ed8.marquee = None
click(ed8, inactive_tab_rect8.center)
assert ed8.marquee is None, "a press on a tab-strip header must never start a marquee"
print("marquee does not start on a tab-strip press OK (real events)")

# ============================================================================
# 9. settings round-trip of groups + active tab
# ============================================================================
ed9 = new_editor()
lay9 = ed9._layout(W, H)
o9, d9 = lay9["panels"]["outliner"], lay9["panels"]["details"]
ed9._begin_panel_drag("details", (d9.x + 5, d9.y + 5), d9)
ed9._finish_panel_drag((o9.x + 10, o9.y + 5), W, H)
side9, group9 = ed9._group_for_pid("outliner")
assert group9["ids"] == ["outliner", "details"] and group9["active"] == "details"
data9 = ed9._settings_dict()
assert data9["dock_order"]["right"] == [{"ids": ["outliner", "details"], "active": "details"}], \
    data9["dock_order"]["right"]
ed9b = new_editor()
ed9b._apply_layout_settings(data9)
side9b, group9b = ed9b._group_for_pid("outliner")
assert group9b["ids"] == ["outliner", "details"] and group9b["active"] == "details"
print("settings round-trip of groups + active tab OK")

# ============================================================================
# 10. Reset Layout restores the factory (ungrouped) layout
# ============================================================================
ed9._reset_layout()
assert ed9.dock_order["right"] == [
    {"ids": ["outliner"], "active": "outliner"},
    {"ids": ["details"], "active": "details"},
], ed9.dock_order["right"]
assert ed9.dock_order["bottom"] == [{"ids": ["browser"], "active": "browser"},
                                    {"ids": ["console"], "active": "console"}]
assert ed9.dock_order["left"] == []
assert not ed9.floating
print("Reset Layout OK: restores the factory ungrouped layout")

# screenshot: a real tabbed slot rendered (visual sanity check)
ed_shot = new_editor()
lay_s = ed_shot._layout(W, H)
os_, ds_ = lay_s["panels"]["outliner"], lay_s["panels"]["details"]
ed_shot._begin_panel_drag("details", (ds_.x + 5, ds_.y + 5), ds_)
ed_shot._finish_panel_drag((os_.x + 10, os_.y + 5), W, H)
fly = engine.behaviors.FlyController(camera, look_buttons=(3,),
                                     look_guard=lambda p: not ed_shot.over_ui(p))
ed_shot.fly = fly
scene.add(engine.Entity("__camera_docktab").add_behavior(fly))
scene.add(engine.Entity("__editor_docktab").add_behavior(EditorBehavior(ed_shot)))
eng.esc_handler = ed_shot.handle_escape
OUT = os.path.join(tempfile.gettempdir(), "judge_docktab.png")
eng.run(scene, camera, max_frames=10, screenshot_path=OUT, overlay=ed_shot.draw)
print(f"screenshot saved: {OUT}")

# no-pollution guard
_real_after = (open(REAL_SETTINGS, "rb").read()
              if os.path.exists(REAL_SETTINGS) else None)
assert _real_after == _real_before, (
    "docktab_checks touched the real settings.json -- an Editor() in this "
    "suite is missing settings_path=TEST_SETTINGS")
print("no-pollution guard OK: real settings.json untouched by this suite")

print("ALL DOCKTAB CHECKS PASSED")
