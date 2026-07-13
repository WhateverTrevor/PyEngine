# Task: default renderer to DX12 (branch wip/dx12-default)

User request: "I don't ever want CPU rendering. We should default to DX12
and only have CPU rendering as an option."

## DONE

- `engine/core.py`: `Engine.__init__` default `api` changed from `"auto"` to
  `"dx12"`; `_init_display`'s `"auto"` mapping changed from `"gl"` to
  `"dx12"` (so an explicit `api="auto"` caller also gets dx12-first);
  docstring updated. Existing fallback chain (dx12/vulkan -> gl -> cpu on
  failure) untouched and still exercised by `_init_display`.
- `editor.py`: CLI default `api_mode` changed from `"auto"` to `"dx12"`;
  `--api` help text and module docstring updated; Settings dialog API-button
  ordering (`_settings_api_buttons`) reordered from
  `("auto","cpu","gl","dx12","vulkan")` to `("dx12","vulkan","gl","auto","cpu")`
  so CPU reads as the opt-in tail choice. `--headless` still forces `"cpu"`
  unconditionally (untouched).
- `demo.py`: same CLI default change (`"auto"` -> `"dx12"`), docstring/help
  text updated. `--headless` still forces `"cpu"`.
- `README.md`: GPU-rendering section and quickstart line rewritten to state
  dx12 is the default and cpu is opt-in only.
- Ten-suite battery, all green (see run output this session): smoke_test,
  browser_checks, toolbar_checks, window_checks, texture_checks,
  material_checks, pbr_checks, gl_checks, wgpu_checks, env_checks. None of
  these needed changes -- each test script hardcodes
  `SDL_VIDEODRIVER=dummy` + explicit `api="cpu"` on its own Engine() calls,
  unaffected by the default change.
- No-flag launch evidence (used `--settings-path` pointing at a scratch dir
  so the real settings.json was never touched, per the task's hard rule):
  - `py editor.py --frames 90 --settings-path <scratch>/nosettings.json
    --screenshot <scratch>/x.png` -> printed `Forcing backend: D3D12 (4)`
    (wgpu's own adapter log) and ran at 49.4 FPS avg -- confirms DX12 is
    the real default with no `--api` flag and no settings.json present.
  - `py demo.py --frames 300` (no flags at all) -> `Forcing backend: D3D12
    (4)`, 98.2 FPS avg -- matches the ~85-98 dx12 demo reference.
  - `py editor.py --frames 60 --settings-path <scratch>/cpu_settings.json
    ...` with that file containing `{"api": "cpu"}` -> no D3D12 log line,
    23.0 FPS avg (matches the cpu editor reference ~23) -- confirms a
    user's saved cpu preference is still respected, not overridden by the
    new default.
  - All scratch settings/screenshot files were deleted after verification;
    `git status` is clean.

## NEXT

Nothing outstanding for this task. Ready for supervisor review/squash-merge.

## Known issues / decisions for the reviewer

- Kept `"auto"` as a selectable option (Settings dialog + `--api auto`) for
  back-compat with anything that still passes it explicitly; it now behaves
  identically to `"dx12"` (tries dx12 first, same fallback chain). Could be
  removed/collapsed into `"dx12"` if the reviewer prefers a cleaner surface,
  but leaving it avoids breaking old `--api auto` callers or saved
  `settings.json` files with `"api": "auto"`.
- Settings dialog button order changed (dx12, vulkan, gl, auto, cpu) purely
  as a visual ordering; the underlying tuple order has no other consumers
  besides that button-layout function and `settings_api in (...)`
  membership checks (order-independent), confirmed by grep.

## Temp artifacts

None left in the repo. Verification screenshots/settings files were written
to and cleaned up from the scratchpad
(`C:\Users\tseit\AppData\Local\Temp\claude\...\scratchpad`), never under
`assets/` or `scenes/`.
