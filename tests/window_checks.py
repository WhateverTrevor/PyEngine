"""Judge checks: panel minimize/close/reset + material-editor conflict zone."""
import os
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import numpy as np

import engine
from editor import (Editor, EditorBehavior, MaterialEditorUI, PANEL_TITLE_H,
                    build_starter_scene)

OUT = os.path.join(tempfile.gettempdir(), "judge_winmgmt.png")

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json")
W, H = eng.screen.get_size()

# 1. minimize a docked panel: it collapses to its title bar, sibling grows
lay0 = editor._layout(W, H)
h_out0 = lay0["panels"]["outliner"].height
editor.panel_minimized["details"] = True
lay1 = editor._layout(W, H)
assert lay1["panels"]["details"].height == PANEL_TITLE_H
assert lay1["panels"]["outliner"].height > h_out0 + 100
print(f"minimize OK: details -> {PANEL_TITLE_H}px, outliner {h_out0} -> "
      f"{lay1['panels']['outliner'].height}")

# 2. all side panels minimized -> dock narrows, viewport widens
vw0 = lay1["viewport"].width
editor.panel_minimized["outliner"] = True
lay2 = editor._layout(W, H)
assert lay2["viewport"].width > vw0 + 50, (vw0, lay2["viewport"].width)
print(f"collapsed dock OK: viewport {vw0} -> {lay2['viewport'].width}")

# 3. close via flag; viewport reaches window bottom when browser closed
editor.panel_visible["browser"] = False
lay3 = editor._layout(W, H)
assert lay3["viewport"].bottom == H
print("close OK: viewport reaches window bottom with browser closed")

# 4. reset layout restores everything
editor.floating.append("details")
editor._reset_layout() if hasattr(editor, "_reset_layout") else editor.reset_layout()
lay4 = editor._layout(W, H)
assert all(editor.panel_visible.values())
assert not any(editor.panel_minimized.values())
assert not editor.floating
assert lay4["panels"]["browser"].height > PANEL_TITLE_H
print("reset layout OK: visibility, minimize, floating all restored")

# 5. settings round-trip includes minimized state
editor.panel_minimized["outliner"] = True
data = editor._settings_dict()
editor2 = Editor(engine, eng, scene, camera, lib, "scenes/scene.json")
editor2._apply_layout_settings(data)
assert editor2.panel_minimized["outliner"] is True
print("settings round-trip OK: minimized state persists")

# 6. conflict zone: material editor still bakes param edits (draft path merged)
crate = next(e for e in scene.entities if e.asset_name == "Crate")
editor.selected = crate
ui = MaterialEditorUI(editor, crate)
g = ui.graph
c = g.add("color", (40, 60))
g.nodes[c]["params"].update(r=1.0, g=0.05, b=0.05)
g.connect(c, g.output_id(), "color")
ui.apply()
fc = crate.mesh.face_colors
assert fc[:, 0].mean() > fc[:, 1].mean() * 3, "material bake broken post-merge"
print("material editor OK post-merge: bake applied "
      f"(R {fc[:, 0].mean():.0f} vs G {fc[:, 1].mean():.0f})")

# screenshot: details minimized (docked), Window menu open showing registry
editor.panel_minimized["details"] = True
editor.panel_minimized["outliner"] = False
editor.open_menu = "Window"
flash = engine.Entity("flashlight", light=engine.SpotLight(enabled=False))
scene.add(flash)
editor.flashlight = flash
fly = engine.behaviors.FlyController(camera, look_buttons=(3,),
                                     look_guard=lambda p: not editor.over_ui(p))
editor.fly = fly
scene.add(engine.Entity("__camera").add_behavior(fly))
scene.add(engine.Entity("__editor").add_behavior(EditorBehavior(editor)))
eng.esc_handler = editor.handle_escape
eng.run(scene, camera, max_frames=30, screenshot_path=OUT, overlay=editor.draw)
print("screenshot saved")
