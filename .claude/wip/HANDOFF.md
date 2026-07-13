# Task: fix-text-dup — DONE, awaiting supervisor review

Branch `wip/fix-text-dup` (from main @ acdd40f), pushed. Fixed the typed-
text duplication bug (typing "vector" produced "vvvveeeeccccttttoooorrrr"
at ~15 FPS): the fixed 60Hz timestep runs update 1-N times per rendered
frame; the three text consumers in editor.py appended
`InputManager.text_typed` once per UPDATE STEP instead of once per FRAME.

## DONE

- `engine/input.py`: added `InputManager.take_text()` — returns the
  accumulated string and clears it immediately (consume-once, same
  convention as `pressed()`/`consume_edges()`). `text_typed` accumulation
  in `process()` and the `consume_edges()` clear kept as backstop.
- `editor.py`: switched all three text consumers to `take_text()` —
  `_update_edit_field` (~953, Details transform fields), `_update_rename`
  (~1695, folder-rename buffer), `_update_ctx_menu` (~3129, material-editor
  node search bar). Audited repo for other `text_typed` readers — none
  found (only `tests/toolbar_checks.py`'s FakeInput, updated to match).
- `tests/mat_ui_checks.py`: added section 7, regression coverage for all
  three fields — simulates a KEYDOWN batch then runs each consumer's
  update path 3x in one frame-span (no consume_edges between, matching the
  N-steps-per-frame scenario), asserts each buffer receives the text
  exactly once. Passes.
- `tests/toolbar_checks.py`: FakeInput gained `take_text()` to match the
  new interface (existing typed-char-filter case still exercises it).

## Verification (all real output, on this branch)

- Full twelve-suite battery run individually, all PASSED: smoke, browser,
  toolbar, window, texture, material, mat_ui (incl. new consume-once
  cases), pbr, transparency, gl, wgpu, env.
- `git status --short` clean — no strays, no test-asset pollution, real
  settings.json untouched (each suite's own no-pollution guard also
  passed).
- FPS (real settings.json temporarily moved aside per CLAUDE.md gotcha,
  restored after): editor cpu headless 22.6 FPS (ref ~23), demo cpu
  headless 51.4 FPS (ref ~44-52) — both in line, no regression.

## NEXT

Nothing further planned for this task — ready for supervisor review and
squash-merge. Not committed to main by this agent (per hard rule).

## Known issues / decisions for reviewer

- `take_text()` assumes only one text field is active at a time (true
  today — editing_field / renaming_folder / ctx_menu search are mutually
  exclusive UI states); documented in the docstring.
- No other `text_typed` readers exist outside editor.py and the test
  FakeInput shims.
