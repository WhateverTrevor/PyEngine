# No task in flight

Docking previews + tabbed panels was judged and squash-merged to main on
2026-07-14. Panels show a ghost preview of their actual post-drop rect
while dragging (replaying the real drop against scratch state — no
drift), and can be tabbed by dropping onto a docked panel's title; the
dock model is now {side: [groups]} (ids + active tab) with transparent
migration from old flat settings.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. UI/interaction tests MUST drive the real
event path; held modifiers need pygame.key.get_pressed + mouse.get_pos +
mouse.get_pressed patched at the OS boundary. DX12 is the default
renderer (CPU opt-in). The full battery is EIGHTEEN suites. FPS in this
working dir is intermittently 2-10x slow AND high-variance (demo dx12
swung 61-102 on identical code) — only same-environment A/B comparisons
are valid; never trust a single sample against the CLAUDE.md references.

Known nuance (deliberate, spec-literal): minimize applies to the active
tab, so switching tabs inside a collapsed strip can re-expand it if the
newly-active tab isn't itself minimized. Revisit if the user complains.

Backlog: wgpu directional sun-shadow attenuation; per-pixel texturing;
flat-mode translucency; triangle-precise floor snap; rotation matching
in mesh snap; marquee for non-mesh marker glyphs; floating-to-floating
tabbing; material assets in browser folders; folder deletion.
