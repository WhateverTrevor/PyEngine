# Task: Window-management for the PyEngine editor

Branch: `wip/window-management` (merged with `origin/main` — the sun/GI/
volumetric-fog branch — at 2f2cda4).

## DONE

- Panel minimize/close buttons on Outliner, Details, Content Browser, and
  the Material Editor title bars: `Editor._panel_title_buttons(rect)` is the
  single rect helper used both for drawing (`_draw_title_buttons`) and
  hit-testing (in `update()`'s panel-click block). MaterialEditorUI has its
  own close/minimize rects (`outer_rect`, `minimize`/`close` locals in
  `update()`/`draw()`), collapsing via `self.minimized`.
- `Editor.panel_minimized` dict (outliner/details/browser) added alongside
  `panel_visible`. `_layout()` now: minimized docked panels occupy just
  `PANEL_TITLE_H` in the stack, remaining height redistributes to the
  non-minimized panels on that side (`stack()` helper rewritten); if every
  panel on a side dock is minimized the whole dock's width collapses to
  `MIN_DOCK_W` (150px, new constant) and the viewport widens; floating
  minimized panels draw/hit-test as title-bar-only rects
  (`_float_rect_for` clamped rect gets its height overridden in `_layout`,
  the underlying `float_rect` storage is untouched so restore recovers the
  full size). `_full_panel_size()` ensures dragging a minimized panel
  doesn't bake the collapsed height into `float_rect` on redock/refloat.
  Panel content is skipped both in `_draw_panel` (early return when
  minimized) and in the click router (`elif not self.panel_minimized...`).
- Window menu is the full registry: Outliner/Details/Content Browser
  (existing toggles, checkmark via new `_menu_checked()` helper), a new
  **Material Editor** entry (`_toggle_material_editor`: opens for the
  current mesh selection, status "select a mesh entity first" if none/
  invalid, checkmark reflects `self.mat_ui is not None`), Settings...,
  Reset Layout.
- `_reset_layout()` now also resets `panel_minimized` to all-False.
- Persistence: `panel_minimized` added to `_settings_dict()` and
  `_apply_layout_settings()`.
- README `#### Dockable panels` section rewritten to document the buttons,
  minimize semantics, the full Window menu, and what Reset Layout clears.
- Merge conflict in `MaterialEditorUI.update()` (main added `draft=True/
  False` param baking during slider drags) resolved by keeping both: the
  minimized-guard structure from this branch, wrapped around main's
  draft-bake calls (`self.apply(draft=True)` while dragging a param,
  `self.apply(draft=False)` on release).

## Verification evidence

- Throwaway script (not in repo):
  `C:\Users\tseit\AppData\Local\Temp\claude\D--ClaudeCode-Spotidownload\c5fe6e82-b615-4ec1-a768-de59192570bd\scratchpad\verify_window_mgmt.py`
  — 26/26 PASS against the merged tree (state-manipulation logic checks:
  dock-height redistribution, fully-minimized side-dock width, floating
  minimized title-bar-only rect, Reset Layout clearing everything,
  settings.json round-trip, Window-menu Material Editor entry/checkmark/
  status message).
- 3 headless screenshots viewed and confirmed visually: (a) Details
  minimized docked right, title-bar strip only, Outliner visibly taller;
  (b) Content Browser closed, viewport reaches window bottom, then
  restored via `_toggle_panel` (menu-equivalent) round-trip; (c) a floating
  Details panel minimized, showing only its title bar with [–][x] over the
  viewport. Screenshots were written outside the worktree and deleted
  after viewing — not part of the diff.
- `py editor.py --frames 150 --headless` on the merged tree: 20.5 FPS avg
  (was 29.2 FPS pre-merge — the drop is from origin/main's new sun/GI/
  volumetric-fog rendering, not from this branch's panel changes; verified
  by running the same command before merging, at 27–29 FPS).
- `py demo.py --frames 90 --headless`: 47.6 FPS avg.
- `git status --short` clean in the worktree after each verification pass
  (settings.json test artifacts removed).

## Known limitations / decisions

- `MIN_DOCK_W = 150` (fully-minimized side-dock width) is a judgment call,
  not specified numerically in the task — wide enough for the title text
  + both buttons, narrower than the normal 260px dock.
- Dragging a minimized panel keeps it minimized through the drag (per
  spec: "Dragging a minimized panel's title bar still moves/docks it" —
  did not require un-minimizing on drag).
- FPS regression noted above is pre-existing from the other branch's GI/
  fog work, not introduced here.

## Next

Task complete and verified. Ready for supervisor review.
