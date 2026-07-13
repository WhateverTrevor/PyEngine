# Task: fix-ctx-crash — DONE, awaiting supervisor review

Branch `wip/fix-ctx-crash` (from main @ bb12c31), pushed (commit 28514f5).
Fixed the crash reported by the supervisor: right-click to open the
material editor's add-node context menu, then left-click the first entry
row -> `TypeError: 'NoneType' object is not subscriptable`.

## Root cause (two compounding defects, both in editor.py MaterialEditorUI)

1. `_ctx_search_matches` ignored its `search` argument entirely and read
   `self.ctx_menu["top10"]` directly.
2. `_click_ctx_menu` set `self.ctx_menu = None` at the top, BEFORE calling
   `_ctx_menu_rows(menu)` (which in turn calls `_ctx_search_matches`). On
   the empty-search top10 path this hit defect 1 with `self.ctx_menu`
   already `None` -> the crash.

## DONE

- `_ctx_search_matches(self, menu)`: now takes the menu dict and reads only
  `menu["search"]` / `menu["top10"]`, never `self.ctx_menu`. Updated both
  call sites (`_update_ctx_menu`, `_ctx_menu_rows`).
- Added `_ctx_menu_total_rect(menu)`: single source of truth for the full
  menu rect (search bar + header + rows); `_draw_ctx_menu` now derives its
  `total_h` from it too, so draw and hit-test can never disagree.
- `_click_ctx_menu` rewritten: hit-test against `_ctx_menu_total_rect`
  first — a click truly outside closes the menu with no action run. Inside
  the rect, walk `_ctx_menu_rows`, run the matched entry's action if any,
  THEN set `self.ctx_menu = None` (closing only after the local `menu`
  capture is done being read).
- `tests/mat_ui_checks.py` section 8: event-driven regression suite that
  drives the REAL input path (`eng.input.process` + `editor.update`, not
  direct handler calls) — this exact gap (all prior tests called
  `_click_ctx_menu`/`_ctx_menu_rows` directly) is why the suite missed this
  crash originally. Covers: right-click opens add menu -> left-click an
  entry adds the node at the click position with no exception and closes
  the menu; click truly outside the menu rect closes without adding a
  node; typing "noise" via real KEYDOWN events + real Enter keydown adds
  the top search match at the right-click graph position.
- Also fixed 3 pre-existing direct calls to `_ctx_search_matches` in the
  same test file that passed a raw string (old signature) — updated to
  pass the menu dict (`mui.ctx_menu`).

## Verification (all real output, on this branch)

- Repro script (`ctx_crash.py`, scratchpad) via the real event path: no
  crash, node added, confirmed against the fix.
- Confirmed the new regression test actually pins the bug: stashed just
  `editor.py` (keeping the new test), reran — got
  `AttributeError: 'dict' object has no attribute 'lower'` in the old
  `_ctx_search_matches`, i.e. old code + new test fails as expected.
- Full twelve-suite battery run individually, all PASSED: smoke_test,
  browser_checks, toolbar_checks, window_checks, texture_checks,
  material_checks, mat_ui_checks (incl. new event-driven section 8),
  pbr_checks, transparency_checks, gl_checks, wgpu_checks, env_checks.
- `py editor.py --frames 120 --headless --screenshot ...`: 23.9 FPS (user's
  real settings.json, which overrides resolution to 5120x1369/pixel_scale
  3, was moved aside during this run and restored after — matches
  CLAUDE.md's ~23 CPU baseline). Screenshot looks correct (starter horror
  scene renders, all panels intact).
- `py demo.py --frames 90 --headless`: 54.2 FPS (within the ~44-52 CPU
  baseline range).
- `git status --short`: only `editor.py` and `tests/mat_ui_checks.py`
  modified. `mat_node_usage.json` untracked stray was already present
  before this task started (pre-existing artifact, not created by this
  work) and left untouched.

## NEXT

Nothing further planned for this task — ready for supervisor review and
squash-merge. Not committed to main by this agent (per hard rule).

## Known issues / decisions for reviewer

- None outstanding; no known regressions.

## Temp artifacts

- Repro script: scratchpad `ctx_crash.py` (path in session scratchpad dir).
- Verify screenshot: scratchpad `editor_verify.png`.
