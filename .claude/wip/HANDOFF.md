# No task in flight

Multi-selection foundation (run 2a) was judged and squash-merged to main
on 2026-07-14. The editor is now multi-select: `self.selection` (ordered
list) with `self.selected` a property returning the ACTIVE (last-clicked)
element; invariant `selected is None iff selection == []`. Shift+click
extends (viewport + outliner), gizmo anchors at the selection mean
("Median Point"), translate moves all, batch delete/duplicate/alt-drag/
floor-snap/focus operate on the whole set. Rotate/scale on a multi-
selection still operate on the ACTIVE element only — that is what run 2b
replaces.

When a task IS in flight, this file holds its resume state per the
checkpoint protocol in `CLAUDE.md` and `.claude/agents/engine-coder.md`.

Next queued (run 2b, final of this slate): Blender-style PIVOT MODES.
Add a pivot-mode selector (viewport toolbar / header) with Blender's
modes — Median Point (current behavior, the mean anchor), Individual
Origins (each selected entity rotates/scales about its own origin),
Active Element (rotate/scale about the active/last-selected entity's
origin), Bounding Box Center. Make rotate AND scale honor the chosen
pivot for multi-selections (translate is pivot-independent). Also fix the
2a-flagged gap: Alt+rotate-drag with multi-select currently only rotates
the active duplicate — under a pivot mode it should rotate all copies
about the pivot. Persist the pivot mode in settings.json.

IMPORTANT: test suites isolate settings via PYENGINE_SETTINGS; never
write the real settings.json. UI/interaction tests MUST drive the real
event path (pygame event injection). Held modifier keys aren't reachable
by synthetic SDL events — patch pygame.key.get_pressed AND
pygame.mouse.get_pos at the OS boundary (editor.update reads live
inp.mouse_pos, not the event's .pos). DX12 is the default renderer.

Backlog: wgpu directional sun-shadow attenuation; per-pixel texturing;
flat-mode translucency; triangle-precise floor snap; rotation matching in
mesh snap; box/marquee multi-select; material assets in browser folders;
folder deletion.
