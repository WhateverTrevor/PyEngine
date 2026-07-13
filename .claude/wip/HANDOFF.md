# Material Editor UX Overhaul (run A of 2) -- wip/mat-editor-ux

## Task
UE-style material editor UX: right-click context menu (add-node search + top10,
node actions), left preview panel with node isolation, incremental UE-ish
node styling. Run B (material types, opacity/transparency) is NOT started.

## DONE
- engine/materials.py: added `MaterialGraph.preview_value(mesh, nid, source_image)`
  -- non-destructively reroutes a node's output into Output.BaseColor, bakes,
  restores the real wiring in a finally block. Verified via scratch script
  (see below) that it returns deterministic values and doesn't mutate graph.links.
- editor.py `MaterialEditorUI`:
  - Removed the top-bar `+type` palette buttons and `_palette_rects`/`PALETTE`.
  - Added `ADDABLE_TYPES` (same node set as before), `NODE_DISPLAY` (UE-ish
    display names), `NODE_CATEGORY` + `CATEGORY_COLOR` (title-bar tinting:
    output/constant/texture/math).
  - Right-click (mouse button 3) inside `graph_panel()` opens a context menu:
    empty canvas -> `_open_add_menu` (search bar + top-10-by-usage, live
    filter matches anywhere in display name, Enter adds top match, Esc
    closes); on a node -> `_open_node_menu` (Delete [guarded: not for output],
    Break All Node Links, Break Link: <name> per connected input, Duplicate,
    Start/Stop Previewing Node).
  - Usage tracking: `_usage_path()` = adjacent to `editor.settings_path`
    (`mat_node_usage.json`), reuses existing `load_settings`/`save_settings`
    helpers (same PYENGINE_SETTINGS-relative isolation tests already rely on).
    Seeded via `DEFAULT_NODE_USAGE` so top-10 is useful pre-history.
  - Left preview panel: `preview_rect()`/`graph_panel()` split `content_rect`
    (preview strip width `PREVIEW_W=150`, rest is the node canvas -- all node
    hit-testing/dragging/wires now key off `graph_panel()`, not
    `content_rect()` directly). Renders a small icosphere baked with the
    current graph (or, while a node is being isolated via
    `self.preview_nid`, `graph.preview_value(sphere, nid)`); only re-bakes
    when `self._preview_dirty` (set by `apply()` and preview-toggle actions).
    "Stop Previewing" button in the panel when isolating.
  - Node body: title-bar strip tinted by category; isolated node gets an
    accent-colored 2px outline.
  - RMB fall-through guard: `Editor.over_ui()` already returns True whenever
    `mat_ui is not None` (pre-existing), which already gates FlyController's
    `look_guard` -- verified no change needed there, but confirmed by reading
    behaviors.py/editor.py wiring.
  - `Editor.handle_escape()`: now closes an open ctx_menu first (one Esc
    level) before closing the whole material editor.
- Verified via a throwaway script
  (scratchpad `smoke_matui.py`, not in repo) exercising: graph_panel/preview_rect
  non-overlap, add-menu top10 + search filter + add-from-menu + usage bump,
  node context-menu labels (Delete/Duplicate/Start Previewing present, Delete
  absent for output when tested against out_id), preview isolation via
  preview_value producing deterministic output and not mutating graph state,
  duplicate, delete, output-node delete guard.
- Headless smoke: `py editor.py --frames 60 --headless --screenshot ...`
  runs clean at ~22.7 FPS (matches CPU baseline ballpark; PYENGINE_SETTINGS
  pointed at a temp file, not the real settings.json).

## DONE (milestone 3)
- `tests/mat_ui_checks.py` written and passing: graph_panel/preview_rect
  geometry (tile content_rect, no overlap), add-menu top-10 seeded ranking,
  live search filter narrowing, click-hit-test using the same row list as
  draw, Enter-adds-at-click-position, usage counts increment + persist
  across a fresh `MaterialEditorUI` + reorder top-10, node context-menu item
  set (Delete/Break All/Break Link: <name>/Duplicate/Start Previewing),
  output-node restrictions (no Delete/Duplicate/preview-toggle, and a direct
  `("delete", None)` action call is a no-op on the output node), Break Link:
  <name> touches only that pin, Break All Node Links clears BOTH incoming
  and outgoing links, Duplicate copies type+params at an offset position,
  Delete removes the node + its links without touching Output, preview
  isolation matches a direct `graph.preview_value()` call and does not
  mutate `graph.links`, deleting the currently-previewed node safely clears
  `preview_nid` back to None, RMB no-fall-through via `over_ui()`.
- Ran ALL 11 suites individually, all PASSED: smoke_test, material_checks,
  mat_ui_checks (new), window_checks, toolbar_checks, browser_checks,
  texture_checks, env_checks, pbr_checks, gl_checks, wgpu_checks.
- FPS spot-check: `--headless` always forces the CPU renderer regardless of
  `--api` (see editor.py main()), so a real dx12 number isn't obtainable in
  this headless session -- editor CPU headless: 150f in 6.06s = 24.7 FPS
  (`--api dx12` flag, forced to cpu) and 150f in 6.11s = 24.6 FPS (`--api
  cpu` explicit) -- consistent with each other and with the smoke_test
  baseline (~22.7 FPS), no regression. demo.py headless: 90f in 1.67s =
  54.0 FPS. Since the preview panel only renders/re-bakes when
  `editor.mat_ui is not None` (the material editor is open), it adds zero
  per-frame cost whenever the material editor is closed -- structurally
  guaranteed, not just measured. A live-window dx12 FPS comparison against
  the ~49 reference should be re-verified by the supervisor with a real
  window if that number matters for the merge decision.

## NEXT
- Run B (not started, do not start): material types + opacity/transparency.
- Possible follow-up polish (not requested, noted only): draggable/resizable
  preview-panel width; a "New Material Node" quick-duplicate-with-rewire.

## Known issues / decisions to weigh
- Preview panel width is fixed-ish (`min(150, content.width-200)`) rather
  than draggable/resizable -- kept simple per "incremental restyle" scope.
- "Break Link: <name>" items are generated per currently-connected input
  pin (no items shown if nothing connected) -- judged "logical, not
  exhaustive" per the task wording.
- Node-type search matches against `NODE_DISPLAY` name OR raw type string
  (so "tex" matches TextureSample and TexCoord, "mult" matches Multiply).

## Temp artifacts
- scratchpad smoke script: `C:\Users\tseit\AppData\Local\Temp\claude\D--ClaudeCode-PyEngine\fff00280-50d8-4931-b697-241c3b13c260\scratchpad\smoke_matui.py`
  (not part of the repo, safe to ignore/regenerate)
