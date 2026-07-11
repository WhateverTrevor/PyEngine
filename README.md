# PyEngine

A compact real-time 3D game engine and world editor written in pure Python,
using **numpy** for the vectorized transform/lighting/ray-tracing pipeline and
**pygame** (SDL) as the window/input backend. The default renderer is pure
Python (no OpenGL); two optional GPU backends mirror the same lighting model
for a large FPS win — OpenGL 3.3 via `moderngl`, or DirectX 12 / Vulkan via
`wgpu-py` — see [GPU rendering](#gpu-rendering).

| Editor | Demo |
|---|---|
| ![editor](docs/editor.png) | ![demo](docs/screenshot.png) |

## Run it

```
py -m pip install pygame numpy
py -m pip install moderngl   # optional: enables the OpenGL renderer
py -m pip install wgpu       # optional: enables the DirectX 12 / Vulkan renderer
py editor.py     # world editor with the survival-horror starter scene
py demo.py       # bright playground demo
py editor.py --api dx12          # force a backend: cpu / gl / dx12 / vulkan (default: auto)
py editor.py --gpu   /   --cpu   # aliases for --api gl / --api cpu
```

### Editor controls

A **menu bar** (File / Edit / Window / Help) runs across the top of the
window. Click a title to open its dropdown, click an item to run it, click
anywhere else to close it — **Help > Controls** opens an in-editor overlay
with this same list.

| Input | Action |
|---|---|
| **RMB (hold)** | mouse look + fly: WASD move, Q/E or Space/Ctrl down/up, Shift = fast, wheel = fly speed. Unreal-style: these movement keys only act while RMB is held |
| **LMB** | select in viewport/outliner; drag assets from the content browser into the world; drag the transform gizmo; drag sliders in the Details panel; drag a panel's title bar to move/dock/float it |
| **W / E / R** | transform gizmo mode: translate / rotate / scale (only while not looking) |
| **, / .** | rotate selection 15° around Y |
| **- / =** | scale selection down / up 10% |
| **F** | focus camera on selection (only while not looking) |
| **Ctrl+D / Del** | duplicate / delete selection |
| **Ctrl+S** | save scene |
| **L** | toggle flashlight |
| **C** | toggle player collision (on by default — walls block you) |
| **M** | open/close the material editor for the selected mesh |
| **F1 / F2** | wireframe / switch per-pixel <-> flat lighting (F2 is a no-op on any GPU renderer — flat vs. per-pixel is a software-only concept; F1 is also a no-op specifically on the dx12/vulkan backend — see [GPU rendering](#gpu-rendering)) |
| **H** | toggle HUD |
| **Esc** | close an open menu/dialog, else deselect, else quit |

Selecting an entity shows a **transform gizmo** in the mode set by W/E/R:
translate (drag the red/green/blue arrows), rotate (drag the projected axis
rings), scale (drag axis handles, or the center square for uniform scale).

**File**: New Scene, Open Scene..., Save, Save As..., Import FBX..., Exit.
**Edit**: Duplicate, Delete, Focus Selection — the same code path the
hotkeys use. **Window**: show/hide the Outliner, Details, and Content
Browser panels (checkmarked when visible), open **Settings...**, or
**Reset Layout** to restore the default arrangement.

#### Dockable panels

The Outliner, Details, and Content Browser are panels with a draggable
18px title bar. Drop one within 48px of the left or right edge to dock it
there (260px wide, panels docked to the same side split the vertical space
between them evenly); the Content Browser docks to the bottom edge instead
(118px tall, full width between any side docks). Drop anywhere else and the
panel floats at that position, keeping its current size. Floating panels
draw on top of the viewport and the clicked one comes to front. The
material editor is floating-only, with the same draggable title bar.
Layout, visibility, and floating positions persist to `settings.json`
(per-user, gitignored) and reload on the next launch.

#### Settings (Window > Settings)

A floating panel with resolution presets (1280x720, 1440x810, 1600x900,
1920x1080 — applied immediately via `Engine.set_resolution`), a pixel-scale
slider (1-6, lower = sharper per-pixel lighting, slower — hidden unless the
software renderer is the one actually active, since it only affects that
renderer's per-pixel pass), a max-FPS slider (30-240), and a **Graphics API**
row: `AUTO / CPU / GL / DX12 / VULKAN`. Next to it, an "Active: ..." label
shows which backend is *actually* running this session (read from the live
engine), which can differ from the saved preference if a GPU backend wasn't
available and the engine fell back. The API choice takes effect on the *next
launch* (live switching mid-session is out of scope); everything else
persists and applies immediately. CLI flags
(`--width`/`--height`/`--pixel-scale`/`--api`/`--gpu`/`--cpu`) still win over
the saved values.

The **+ Import FBX** button in the content browser (or **File > Import
FBX...**) opens a file picker and converts a binary FBX model into a
regular asset (saved to `assets/models/`, appears in the browser
immediately). Material diffuse colors are extracted and baked as per-face
colors — multi-material models keep their coloring.

![material editor](docs/material_editor.png)

**Material editor** (select a mesh entity, press **M** or click *material*
in the Details panel): a node-based graph — Color/Position/Normal/Checker/
Noise/Gradient sources wired through Mix/Multiply into the Output node.
Drag from an output port to an input port to connect; click a wired input to
unplug and re-route; drag the sliders inside nodes to tune parameters. Every
change re-bakes the mesh's per-face colors instantly, so the viewport behind
the panel is a live preview. Graphs are saved with the scene.

Select any light (viewport or outliner) and the **Details panel** exposes it:
brightness, RGB color, throw (range), shadow softness, spotlight cone inner
angle and penumbra, IES profile, enabled, and shadow casting. Edits apply
live and persist through Ctrl+S.

The mouse is only captured while a look button is held — release and the
cursor is free (demo uses LMB *or* RMB; the editor reserves LMB for selection
and panel/UI interaction, so looking is RMB-only).

## Architecture

```
engine/
  math3d.py       Vec3 + 4x4 matrix builders
  mesh.py         quad/tri polygon meshes + primitives (box, cylinder, cone, ...)
  camera.py       perspective camera, world<->screen projection, picking rays
  lighting.py     DirectionalLight, PointLight, SpotLight, IES profiles, Fog
  environment.py  Radiance .hdr (RGBE) I/O + HDRI sky sampling & ambient cube
  fbx.py          minimal binary FBX parser (geometry + materials) -> assets
  materials.py    node-based material graphs, baked to per-face colors
  raytrace.py     ray-traced soft shadows + scene picking (Moller-Trumbore)
  scene.py        Scene / Entity / Transform / Behavior (component system)
  behaviors.py    Spin, Bob, Orbit, Flicker, FlyController (+collision), ...
  input.py        per-frame keyboard/mouse state, hold-to-capture mouse
  renderer.py     software renderer: deferred per-pixel + flat shading paths,
                  painter's sort
  gl_renderer.py  optional GPU renderer (OpenGL 3.3 core via moderngl):
                  same lighting model in GLSL, ray-traced shadow factors
                  from raytrace.py uploaded as a per-face texture
  core.py         Engine: window, splash, fixed-timestep loop, HUD, benchmarks,
                  set_resolution, GPU context lifecycle + software fallback
  assets.py       self-contained JSON assets + scene save/load
editor.py         menu bar, dockable outliner/details/content-browser panels,
                  settings dialog, gizmo, material editor, FBX import
assets/*.json     the asset library (drag these into the world)
assets/hdri/      HDR environment maps (.hdr Radiance files)
assets/models/    imported model geometry (.npz)
scenes/           saved scenes
settings.json     per-user editor window/panel-layout state (gitignored)
demo.py           bright playground demo
```

### Quad meshes

Faces are polygons — quads where the shape allows (box sides, floor squares,
cylinder walls, torus), triangles elsewhere. Per-face lighting shades each
quad as one clean panel with no diagonal seam, and face counts drop by
nearly half. Quads are triangulated internally only for the ray tracer.

### HDRI environment (Sky Sphere)

Drag the **Sky Sphere** asset into a scene and the renderer switches to
image-based sky and ambient: sky pixels sample the equirectangular HDR map
along their camera rays (stars, moon glow), and diffuse environment light
comes from an ambient cube — the HDRI convolved to six cosine-weighted axis
colors at load, evaluated per face normal, so upward faces catch bluish
moonlight while undersides stay dark. `engine/environment.py` reads real
Radiance `.hdr` files (RLE and flat): drop your own HDRI into `assets/hdri/`,
point an asset's `"environment": {"hdri": ...}` at it, done. The bundled
`night_sky.hdr` is procedurally generated (see the repo history) with true
HDR values — the moon is ~5x brighter than white.

### Per-pixel deferred lighting

The default shading path lights every pixel individually. Depth-sorted
triangles are filled into a low-resolution *face-ID buffer* (pygame's C
rasterizer), then numpy reconstructs each pixel's world position by
intersecting its camera ray with the face's plane and evaluates every light
per pixel: smooth distance falloff, smooth spotlight cones with adjustable
penumbra, IES angular profiles, per-pixel fog. The frame is upscaled to the
window — the chunky internal resolution (`--pixel-scale`, default 1/4) is
both the performance budget and a deliberate PS1-horror aesthetic. F2 falls
back to classic flat per-face shading.

Lights carry an **IES profile** — an angular intensity curve like real
photometric IES files (`uniform`, `spot_soft`, `downlight`, `batwing`),
sampled against the angle from the light's axis per pixel.

### Collision

The player is a sphere tested against every collidable entity's oriented
bounding box, resolved in the entity's local space so rotated walls work and
the player slides along surfaces instead of sticking. `"collidable": false`
in an asset opts out (the Ghost — you walk right through it).

### Ray-traced soft shadows

Lights are physical spheres (`radius`), not points. For every face a light
reaches, the tracer casts `shadow_samples` rays from the face toward points
distributed across the light's sphere and intersects them against all
shadow-casting geometry (vectorized Moller-Trumbore, rays x triangles in
chunks). The unblocked fraction is the shadow factor — fully blocked faces go
dark, partially blocked faces land in the penumbra, so shadow edges are soft.
Shadow granularity is per *face*; the per-pixel path modulates its smooth
per-pixel light with these per-face factors.

What keeps it real-time:

- **Caching** — factors are cached per (receiver, light) and reused until the
  receiver, the light, or any shadow caster actually moves. A fully static
  scene traces once, then shadows are free; light *flicker* changes intensity,
  not geometry, so it never invalidates the cache.
- **Amortization** — moving lights retrace every `shadow_interval` frames
  (the flashlight uses 2), and receivers that move while everything else is
  static reuse their factors for up to 3 frames.
- **Culling** — only faces a light actually reaches get rays, and only
  occluders within the light's range are tested.

Tuning: fewer `shadow_samples` = faster + harder shadows; `cast_shadows:
false` on a light skips tracing entirely; `casts_shadow: false` on an entity
(the ghost, the floor) removes it from the occluder set — a moving caster
invalidates the whole cache, so keep animated things out of it when you can.

### Self-contained assets

One JSON file per asset in `assets/` — mesh, light, and behaviors together,
so the object works dropped into any scene:

```json
{
  "name": "Torch", "category": "lights",
  "mesh": {"primitive": "cylinder", "radius": 0.1, "height": 1.5, "color": [84, 62, 40]},
  "light": {"type": "point", "color": [255, 150, 60], "intensity": 2.2,
            "range": 12, "radius": 0.3, "shadow_samples": 8, "offset": [0, 0.95, 0]},
  "behaviors": [{"type": "Flicker", "amount": 0.35, "speed": 9}]
}
```

The content browser renders a live 3D thumbnail of each asset at startup.
Drop a new `.json` in `assets/` and restart the editor to see it. Scenes
serialize as asset name + transform per entity, plus the scene's lighting,
fog, and sky — everything you place round-trips through `Ctrl+S`.

### Renderer and loop

Per frame the renderer transforms every mesh to world/camera space in numpy
matmuls, accumulates lighting per face (ambient + directional + every
point/spot light with distance/cone attenuation, colored per channel, times
its ray-traced shadow factor), backface-culls, clips against the near plane,
blends distance fog, depth-sorts all faces from every mesh together (painter's
algorithm), and fills polygons with pygame's C rasterizer. `Engine.run()`
updates behaviors on a fixed 60 Hz timestep, decoupled from render rate.
Known trade-off: painter's sorting is per-face, so interpenetrating geometry
can occasionally sort wrong — the classic software-rendering compromise.

Measured on this machine at 1440x810: the starter horror scene (HDRI sky, 6
shadow-casting lights, flashlight on) runs ~27-32 FPS with per-pixel
lighting at 1/4 internal resolution; the bright demo scene ~68 FPS at
1280x720 since the quad-mesh switch.

### GPU rendering

Three renderer backends exist, chosen by `Engine(api=...)` — `"auto"` (the
default for both apps), `"cpu"`, `"gl"`, `"dx12"`, or `"vulkan"`. `--api` on
either app forces one from the command line (`--gpu`/`--cpu` remain as
aliases for `--api gl` / `--api cpu`); the editor's Settings dialog offers
the same five choices as a **Graphics API** row (see
[Settings](#settings-window--settings)), applied on the next launch.
`--headless` always forces `"cpu"`, since the SDL dummy driver used for
headless/CI runs has no GL surface or wgpu-presentable window to attach to.

**Fallback chain:** `"auto"` tries `"gl"` first (OpenGL/moderngl remains the
default GPU path when available). Any failure along a GPU path — missing
dependency, no suitable driver/adapter, context/device creation failure —
prints one warning line and falls back one step at a time: requested api ->
`"gl"` -> `"cpu"`. E.g. requesting `"dx12"` on a machine without `wgpu`
installed prints one warning and tries `"gl"`; if that also fails (no
`moderngl`, or no GL 3.3 driver), a second warning and the engine ends up on
`"cpu"`. The rest of the engine behaves exactly as if that were the
requested api all along.

#### OpenGL (`gl`)

`moderngl` is an **optional** dependency (`py -m pip install moderngl`).
When present, opens an OpenGL 3.3 core window instead of a plain software
surface and renders every frame through `engine/gl_renderer.py` — a GLSL
port of the same lighting model as `renderer.py` (ambient/ambient-cube,
directional light, up to 16 point/spot lights with distance falloff,
spotlight cones, and IES profiles), running per-fragment with a real depth
buffer instead of the software renderer's painter's-sort + face-ID buffer.
Measured on this machine: the demo scene goes from ~56 FPS (software,
default per-pixel path) to ~115 FPS GPU (both capped at `max_fps=120`, so
GPU is closer to the cap than the raw ratio suggests); scenes bottlenecked
on the software renderer's per-pixel pass see a much larger win.

**What stays on the CPU even in GL mode:**

- **Shadows** — `raytrace.ShadowTracer` still ray-casts per face against the
  triangle soup every frame (same caching/amortization as the software
  path); the GPU renderer uploads the resulting per-face factors as a small
  texture and samples it by a global face id in the fragment shader. Shadow
  quality/cost is therefore identical across all three renderers — GPU mode
  gets you faster *shading*, not faster *shadow tracing*.
- **Mouse picking** (`raytrace.pick_entity`) and **collision**
  (`behaviors.FlyController`'s sphere-vs-OBB test) — both are ray/geometry
  math against the same triangle soup, independent of how pixels get shaded.

**UI in GL mode:** the editor/HUD keep drawing with pygame exactly as in
software mode — `Engine.screen` becomes a transparent overlay surface that
gets composited over the 3D frame each tick as an alpha-blended fullscreen
quad, so none of the editor/HUD code needed to change.

**Known limitations:**

- Max **16 dynamic point/spot lights** per frame (extras are silently
  dropped, sorted by scene order); the software renderer has no such cap.
  Same cap applies to the dx12/vulkan backend below.
- Fog distance is the true Euclidean camera-to-fragment distance (matching
  the software renderer's *per-pixel* path); the software renderer's *flat*
  path instead fogs by view-space linear depth. The two software paths
  already disagree with each other here, so the GPU renderers matching the
  more precise of the two seemed the right call.
- Resizing the GL window (Settings dialog) recreates GL-side size-dependent
  objects (viewport, UI overlay texture) but keeps the existing GL context;
  this path is exercised by `set_resolution` but wasn't soak-tested under
  heavy interactive resizing.
- The GL path was verified against an NVIDIA RTX 5070 Ti; other vendors'
  GL 3.3 core drivers should work (nothing vendor-specific is used) but
  weren't tested.

#### DirectX 12 / Vulkan (`dx12` / `vulkan`)

`wgpu` (the [wgpu-py](https://github.com/pygfx/wgpu-py) WebGPU bindings) is
an **optional** dependency (`py -m pip install wgpu`), independent of
`moderngl` — one can be installed without the other and each GPU backend
still works. `engine/wgpu_renderer.py` implements the same lighting model a
third time in WGSL (ambient/ambient-cube, directional light, up to 16
point/spot lights, IES profiles, fog), reusing the exact same triangle-soup
and light-gathering code the other two renderers use (`engine/gpu_geometry.py`,
`engine/renderer.py`) so all three stay in sync from one source of truth.
DirectX 11 is **not** offered as a choice: wgpu-native removed its D3D11
backend, so DX12 is the supported Direct3D path on Windows.

**"Offscreen + readback"**, in plain terms: unlike the GL backend (which
renders straight into the window and composites the UI on the GPU), the
wgpu backend renders each frame into its own off-window color+depth texture,
then copies that texture back to normal CPU memory and blits it into the
actual window surface, after which the editor/HUD draw on top exactly as in
software mode. This is simpler to implement correctly than the GL
compositing path, but it pays for a full frame's worth of GPU->CPU transfer
every tick — it will not out-run the GL backend's in-framebuffer compositing
at high resolutions, though it's still dramatically faster than the software
renderer for shading-bound scenes.

**Known limitations specific to this backend:**

- **No wireframe.** wgpu's line-fill polygon mode needs a device feature
  that isn't guaranteed to exist on every adapter/vendor, so it isn't
  requested (for portability); `F1` is a documented no-op on dx12/vulkan.
- Same 16-light cap and CPU-side shadow/picking/collision notes as the GL
  backend above apply here too.
- Verified on this machine against an NVIDIA RTX 5070 Ti; other vendors'
  DX12/Vulkan drivers should work (nothing vendor-specific is used) but
  weren't tested.

## Extending it

- **New asset**: drop a JSON in `assets/` (see above) — it appears in the
  content browser on next launch.
- **New behavior**: subclass `Behavior` in `engine/behaviors.py`, reference it
  by class name from asset JSON.
- **New primitive**: add a `Mesh` factory in `engine/mesh.py` and register it
  in `engine/assets.py`.
- **Load models**: an OBJ loader is ~20 lines — parse `v`/`f` lines into the
  arrays `Mesh` takes, then register a `"model"` factory.
