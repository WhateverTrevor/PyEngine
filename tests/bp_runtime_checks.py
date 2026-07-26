"""Judge checks: Blueprint RUNTIME (run 2b of 2, the final run of the
blueprint feature -- see engine/blueprint.py's compile_blueprint_with_class
/instantiate_behavior/BlueprintBehaviorProxy and engine/assets.py's
BlueprintAsset.instantiate for the runtime-attach design; run 1 shipped
compile/bug-check, run 2a shipped posed-mesh components, this run attaches
and runs the compiled Behavior).

Covers:
  - M1 infinite-loop guard: compile_blueprint("while True: pass") returns a
    distinct "timeout" stage instead of hanging, bounded wall-clock time
    (the headline test -- asserts the call actually RETURNS, in roughly
    EXEC_TIMEOUT_SECONDS, not "eventually"); the pre-existing syntax/exec/
    validate stages are unchanged (same dict shape/messages as before this
    run, already exhaustively covered in blueprint_checks.py -- this suite
    only re-confirms the stage names from the new shared _compile_impl path
    didn't shift).
  - M2 attach + per-entity isolation: a blueprint entity placed through the
    REAL content-browser drag-place path (same _route_panel_click code a
    mouse drag takes -- see blueprint_checks.py/bp_component_checks.py's
    precedent for this idiom) gets its compiled Behavior attached and
    actually running on the real fixed-timestep tick (eng.run's own
    scene.update cadence, not a bypassed handler); a raising update()/
    start() logs EXACTLY ONE console warning and disables the behavior
    instead of spamming every frame; a non-compiling blueprint still places
    as a static mesh (mesh != None, behaviors == [], no crash).
  - Shadow-cache decision (see engine/blueprint.py's BlueprintBehaviorProxy
    docstring): a blueprint that moves its own entity every frame while
    casts_shadow stays True logs one warning pointing at the documented
    fix; a blueprint whose start() sets entity.casts_shadow = False itself
    (the documented fix, mirroring the built-in Ghost/Bob asset in
    assets/ghost.json) does not.

Isolation: same idiom as tests/blueprint_checks.py / bp_component_checks.py
-- an isolated TEMP COPY of the whole assets/ tree (never opens the real
assets/blueprints, assets/folders.json, assets/gat.json, assets/models/
gat.npz for writing) plus an isolated settings.json, no-pollution guards
at the end.
"""
import os
import shutil
import sys
import tempfile
import time
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import pygame

import engine
from engine import blueprint as blueprint_mod
from engine import console_log
from editor import Editor, EditorBehavior, build_starter_scene

REAL_SETTINGS = os.path.join(WT, "settings.json")
_real_settings_before = (open(REAL_SETTINGS, "rb").read()
                         if os.path.exists(REAL_SETTINGS) else None)
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_bp_runtime_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

# ---- isolated temp copy of assets/ -- see module docstring ----
TMP_ASSETS_ROOT = tempfile.mkdtemp(prefix="pyengine_bp_runtime_checks_")
ASSETS_DIR = os.path.join(TMP_ASSETS_ROOT, "assets")
shutil.copytree(os.path.join(WT, "assets"), ASSETS_DIR)
assert ASSETS_DIR != os.path.join(WT, "assets"), "must never point at the real assets dir"

eng = engine.Engine(1280, 800, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(ASSETS_DIR)
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
editor = Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
               settings_path=TEST_SETTINGS)
fly = engine.behaviors.FlyController(camera, look_buttons=(3,),
                                     look_guard=lambda p: not editor.over_ui(p))
editor.fly = fly
scene.add(engine.Entity("__camera").add_behavior(fly))
scene.add(engine.Entity("__editor").add_behavior(EditorBehavior(editor)))
eng.esc_handler = editor.handle_escape
W, H = eng.screen.get_size()

CRATE = "Crate"
assert CRATE in lib.by_name, "fixture assets/ needs the built-in Crate asset for this suite"


class FakeKeys:
    """pygame.key.get_pressed() stand-in -- same idiom as blueprint_checks.py
    / bp_component_checks.py."""

    def __init__(self, held=()):
        self._held = set(held)

    def __getitem__(self, key):
        return key in self._held


_keys_patch = um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys())
_keys_patch.start()


def step(events):
    eng.input.process(events)
    editor.update(eng, 1 / 60)
    eng.input.consume_edges()


def mouse_down(pos):
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)])


def mouse_move(pos, held=True):
    pressed = (True, False, False) if held else (False, False, False)
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=pressed):
        step([])


def mouse_up(pos):
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
        step([pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos)])


def new_blueprint(name, script):
    """A single-Crate-component blueprint with `script`, filed into a
    dedicated folder so drag_place() can find its tile deterministically."""
    bp = lib.new_blueprint(name)
    bp.components = [{"asset_name": CRATE, "position": [0.0, 0.0, 0.0],
                      "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}]
    bp.script = script
    bp.save()
    editor.bp_icons[bp.name] = __import__("editor").make_blueprint_icon()
    return bp


FOLDER = lib.create_folder("RuntimeCheck", None)


def drag_place(bp):
    """Place `bp` via the REAL content-browser drag path -- press on its
    tile, drag past MARQUEE_THRESHOLD, release over the viewport -- same
    _route_panel_click code a real mouse drag takes (see
    bp_component_checks.py's precedent for this idiom). Returns the newly
    placed Entity."""
    lib.set_asset_folder(bp.name, FOLDER)
    lib.save_folders()
    editor.selected_folder = FOLDER
    layout = editor._layout(W, H)
    content = editor._panel_content_rect("browser", layout)
    blay = editor._browser_layout(content)
    tiles = editor._tiles_in(FOLDER)
    idx = tiles.index(bp)
    grid = blay["grid"]
    x0 = grid.x + 10 - editor.browser_scroll + idx * (84 + 8)  # TILE_W+8
    tile_pos = (x0 + 20, grid.y + 6 + 20)
    viewport = layout["viewport"]
    vp_pos = (viewport.centerx + idx * 3, viewport.centery)  # nudge so instances don't overlap

    mouse_down(tile_pos)
    mouse_move((tile_pos[0] + 30, tile_pos[1] + 5))  # beyond MARQUEE_THRESHOLD (4px)
    mouse_move(vp_pos)
    mouse_up(vp_pos)
    assert editor.script_ui is None, "a real drag-place must not open the script editor"
    placed = [e for e in scene.entities if e.blueprint_name == bp.name]
    assert len(placed) == 1, f"drag-place must add exactly one instance of {bp.name}"
    return placed[0]


def tick(n: int) -> None:
    """Advance the simulation `n` fixed-timestep ticks by calling
    `scene.update` directly -- the exact same call `Engine.run`'s own loop
    makes each iteration (`scene.update(self.fixed_dt, self)` in
    engine/core.py), so a blueprint's Behavior runs through the real
    per-frame entry point, not a bypassed handler. Deliberately NOT
    `eng.run(..., max_frames=n)`: that method calls `pygame.quit()` when
    it finishes (see engine/core.py, end of `run()`) -- fine for the
    single screenshot run every blueprint suite does once at the very end,
    but fatal to call repeatedly mid-suite (confirmed the hard way: it
    tears down the font module, so the next new_blueprint()'s icon render
    crashes, and the CPU renderer path taken while pygame is mid-teardown
    is drastically slower to boot)."""
    for _ in range(n):
        scene.update(1 / 60, eng)


MOVER_SCRIPT = """
from engine.scene import Behavior

class Mover(Behavior):
    def update(self, entity, dt, engine):
        entity.transform.position.x += 0.05
"""

SPIN_SCRIPT = """
from engine.scene import Behavior

class Spin(Behavior):
    def update(self, entity, dt, engine):
        entity.transform.rotation.y += 0.02
"""

MOVER_SAFE_SCRIPT = """
from engine.scene import Behavior

class MoverSafe(Behavior):
    def start(self, entity, engine):
        # the documented fix (see BlueprintBehaviorProxy's docstring):
        # an entity that moves every frame should opt out of shadow
        # casting itself, same as the built-in Ghost asset's Bob animation.
        entity.casts_shadow = False

    def update(self, entity, dt, engine):
        entity.transform.position.x += 0.05
"""

RAISER_UPDATE_SCRIPT = """
from engine.scene import Behavior

class RaiserUpdate(Behavior):
    def update(self, entity, dt, engine):
        entity.transform.position.y = 42.0
        raise ValueError("boom-update")
"""

RAISER_START_SCRIPT = """
from engine.scene import Behavior

class RaiserStart(Behavior):
    def start(self, entity, engine):
        raise RuntimeError("boom-start")

    def update(self, entity, dt, engine):
        entity.transform.position.y = 99.0
"""

NONCOMPILING_SCRIPT = "this is not python !!!"

INFINITE_LOOP_SCRIPT = "while True:\n    pass\n"


try:
    # ========================================================================
    # 1. M1 headline test: infinite loop at module level does not hang --
    #    compile_blueprint returns within a bounded wall-clock window
    # ========================================================================
    t0 = time.perf_counter()
    r = engine.compile_blueprint(INFINITE_LOOP_SCRIPT, "Hang")
    elapsed = time.perf_counter() - t0
    assert r["ok"] is False and r["stage"] == "timeout", r
    assert isinstance(r["message"], str) and r["message"], r
    assert r["line"] is None and r["col"] is None, r
    # bounded: not instant (it really waited out the timeout), and not
    # "eventually" either -- generous slack for a slow/loaded CI machine,
    # but nowhere near "hung".
    timeout_const = blueprint_mod.EXEC_TIMEOUT_SECONDS
    assert timeout_const - 0.5 <= elapsed <= timeout_const + 10.0, (
        f"timeout guard took {elapsed:.2f}s, expected ~{timeout_const}s bounded")
    print(f"1. M1 headline: infinite loop -> stage=timeout in {elapsed:.2f}s "
         f"(EXEC_TIMEOUT_SECONDS={timeout_const}) OK, call actually returned")

    assert blueprint_mod.runaway_thread_count() == 1, (
        "the hung script's thread should be tracked as a live runaway")

    # ========================================================================
    # 2. the three pre-existing stages are unchanged (shape + stage names)
    # ========================================================================
    r_syn = engine.compile_blueprint("class Bad(Behavior)\n    pass\n", "Syn")
    assert set(r_syn) == {"ok", "stage", "message", "line", "col"}, r_syn
    assert r_syn["ok"] is False and r_syn["stage"] == "syntax", r_syn
    assert isinstance(r_syn["message"], str) and r_syn["message"], r_syn
    assert r_syn["line"] is not None

    r_exec = engine.compile_blueprint(
        "from engine.scene import Behavior\nclass Ok(Behavior):\n    pass\n"
        "\nundefined_name_xyz.go()\n", "Exec")
    assert r_exec["stage"] == "exec" and "NameError" in r_exec["message"]

    r_val = engine.compile_blueprint("x = 1\n", "Val")
    assert r_val["stage"] == "validate" and "Behavior" in r_val["message"]

    r_ok = engine.compile_blueprint(
        "from engine.scene import Behavior\nclass Fine(Behavior):\n    pass\n", "Ok")
    assert r_ok == {"ok": True, "class_name": "Fine"}
    print("2. syntax/exec/validate/ok stages unchanged OK:",
         [r_syn["stage"], r_exec["stage"], r_val["stage"], "ok"])

    # ========================================================================
    # 3. compile_blueprint_with_class hands back the actual class object,
    #    same dict either way
    # ========================================================================
    result, cls = engine.compile_blueprint_with_class(
        "from engine.scene import Behavior\nclass Grabbed(Behavior):\n    pass\n", "Cls")
    assert result == {"ok": True, "class_name": "Grabbed"}
    assert isinstance(cls, type) and issubclass(cls, engine.Behavior)
    assert cls.__name__ == "Grabbed"
    result_bad, cls_bad = engine.compile_blueprint_with_class("x = (", "ClsBad")
    assert result_bad["ok"] is False and cls_bad is None
    print("3. compile_blueprint_with_class returns the real class object OK")

    # ========================================================================
    # 4. Behavior attach + real per-frame mutation, placed through the REAL
    #    drag-place UI path, run through the REAL fixed-timestep tick
    # ========================================================================
    console_log.reset()
    bp_mover = new_blueprint("MoverBP", MOVER_SCRIPT)
    mover_ent = drag_place(bp_mover)
    assert len(mover_ent.behaviors) == 1, "compiling script must attach exactly one Behavior"
    assert isinstance(mover_ent.behaviors[0], blueprint_mod.BlueprintBehaviorProxy)
    x0 = mover_ent.transform.position.x
    tick(5)
    assert mover_ent.transform.position.x > x0 + 4 * 0.05 - 1e-6, (
        "Mover's update() must have actually run and mutated the placed entity "
        f"({mover_ent.transform.position.x} vs start {x0})")
    attach_logs = [e for e in console_log.get_log().entries
                  if "attached Behavior" in e["text"] and "MoverBP" in e["text"]]
    assert len(attach_logs) == 1, "placement must log the attach exactly once"
    print(f"4. real drag-place + real fixed-timestep tick mutated the entity OK "
         f"(x: {x0:.3f} -> {mover_ent.transform.position.x:.3f})")

    # ========================================================================
    # 5. raising update() logs ONCE and disables -- no 60Hz spam
    # ========================================================================
    console_log.reset()
    bp_raiser = new_blueprint("RaiserUpdateBP", RAISER_UPDATE_SCRIPT)
    raiser_ent = drag_place(bp_raiser)
    tick(10)  # many ticks -- must not produce 10 log entries
    assert raiser_ent.transform.position.y == 42.0, (
        "the one successful partial update() before the raise must have applied")
    warn_entries = [e for e in console_log.get_log().entries if e["level"] == "warn"]
    update_warns = [e for e in warn_entries if "update() raised" in e["text"]]
    assert len(update_warns) == 1, (
        f"raising update() must log exactly once, got {len(update_warns)}: {update_warns}")
    assert "RaiserUpdateBP" in update_warns[0]["text"] and "ValueError" in update_warns[0]["text"]
    assert raiser_ent.behaviors[0].disabled is True
    print("5. raising update() logs exactly once and disables (no 60Hz spam) OK")

    # ========================================================================
    # 6. raising start() disables before update() ever runs
    # ========================================================================
    console_log.reset()
    bp_raiser_start = new_blueprint("RaiserStartBP", RAISER_START_SCRIPT)
    raiser_start_ent = drag_place(bp_raiser_start)
    y0 = raiser_start_ent.transform.position.y
    tick(10)
    assert raiser_start_ent.transform.position.y == y0, (
        "update() (which sets y=99) must never have run after start() raised")
    start_warns = [e for e in console_log.get_log().entries
                   if e["level"] == "warn" and "start() raised" in e["text"]]
    assert len(start_warns) == 1, start_warns
    assert "RaiserStartBP" in start_warns[0]["text"] and "RuntimeError" in start_warns[0]["text"]
    assert raiser_start_ent.behaviors[0].disabled is True
    print("6. raising start() disables before update() ever runs, logs once OK")

    # ========================================================================
    # 7. non-compiling blueprint still places as static geometry
    # ========================================================================
    console_log.reset()
    bp_bad = new_blueprint("NonCompilingBP", NONCOMPILING_SCRIPT)
    bad_ent = drag_place(bp_bad)
    assert bad_ent.mesh is not None, "must still place as a static posed mesh"
    assert bad_ent.behaviors == [], "no Behavior may attach for a non-compiling script"
    placement_warns = [e for e in console_log.get_log().entries
                       if e["level"] == "warn" and "static mesh only" in e["text"]]
    assert len(placement_warns) == 1, placement_warns
    print("7. non-compiling blueprint places fine as static mesh (no crash) OK")

    # ========================================================================
    # 8. shadow-cache decision: moving while casts_shadow=True warns once;
    #    the documented fix (script sets casts_shadow=False itself) does not
    # ========================================================================
    console_log.reset()
    mover2_ent = drag_place(new_blueprint("MoverBP2", MOVER_SCRIPT))
    assert mover2_ent.casts_shadow is True, "default casts_shadow must stay True (no auto-flip)"
    tick(3)
    shadow_warns = [e for e in console_log.get_log().entries
                    if e["level"] == "warn" and "casts_shadow" in e["text"]]
    assert len(shadow_warns) == 1, (
        f"a moving entity with casts_shadow=True must warn exactly once, got {shadow_warns}")
    assert "MoverBP2" in shadow_warns[0]["text"]
    tick(5)  # more ticks -- still exactly one warning, not one per frame
    shadow_warns_after = [e for e in console_log.get_log().entries
                          if e["level"] == "warn" and "casts_shadow" in e["text"]]
    assert len(shadow_warns_after) == 1, "the shadow warning itself must not repeat every frame"
    print("8a. moving blueprint entity w/ casts_shadow=True warns exactly once OK")

    console_log.reset()
    safe_ent = drag_place(new_blueprint("MoverSafeBP", MOVER_SAFE_SCRIPT))
    safe_x0 = safe_ent.transform.position.x
    tick(5)
    assert safe_ent.casts_shadow is False, "start() setting casts_shadow=False must take effect"
    assert safe_ent.transform.position.x > safe_x0 + 4 * 0.05 - 1e-6, (
        "the entity must still be moving despite casts_shadow=False")
    safe_shadow_warns = [e for e in console_log.get_log().entries
                         if e["level"] == "warn" and "casts_shadow" in e["text"]]
    assert safe_shadow_warns == [], (
        "a script that opts out via casts_shadow=False must not trigger the warning")
    print("8b. the documented fix (script sets casts_shadow=False) suppresses "
         "the warning OK")

    # 8c. ROTATION counts too (supervisor-added after review). ShadowTracer
    # keys its caster cache on the entity's full transform matrix, not its
    # position (`e.transform.matrix().tobytes()` in engine/raytrace.py), so a
    # blueprint that only spins its entity invalidates the bake exactly as a
    # translating one does. Watching position alone would leave that author
    # paying the cost with no warning.
    # Instantiated directly rather than through drag_place(): that helper
    # lays tiles out in a single row (x0 + idx * TILE_W), so by this point
    # in the suite the next tile falls outside the visible grid and the
    # press lands on nothing. The drag-place path is already covered by
    # checks 4 and 6; what 8c exercises is the proxy's transform comparison,
    # which runs under scene.update with no editor involvement.
    console_log.reset()
    spin_bp = new_blueprint("SpinBP", SPIN_SCRIPT)
    spin_ent = spin_bp.instantiate(lib)
    scene.add(spin_ent)
    assert spin_ent.casts_shadow is True, "default casts_shadow must stay True"
    spin_rot0 = spin_ent.transform.rotation.y
    spin_pos0 = (spin_ent.transform.position.x, spin_ent.transform.position.y,
                spin_ent.transform.position.z)
    tick(5)
    assert spin_ent.transform.rotation.y != spin_rot0, "the spin script should rotate"
    spin_pos1 = (spin_ent.transform.position.x, spin_ent.transform.position.y,
                spin_ent.transform.position.z)
    assert spin_pos1 == spin_pos0, (
        "this check is only meaningful if the entity never TRANSLATES -- "
        "otherwise it would pass on the old position-only comparison too")
    spin_warns = [e for e in console_log.get_log().entries
                 if e["level"] == "warn" and "casts_shadow" in e["text"]
                 and "SpinBP" in e["text"]]
    assert len(spin_warns) == 1, (
        f"a rotation-only blueprint must warn exactly once, got {spin_warns}")
    # Take it back out of the scene: it has proved its point, and leaving a
    # perpetually-rotating shadow caster in place makes check 9's render pay
    # a whole-scene shadow/GI rebake on every one of its frames -- measured
    # 71s -> 534s for that 15-frame run, i.e. exactly the cost this warning
    # exists to tell authors about.
    scene.remove(spin_ent)
    print("8c. rotation-only blueprint also warns (transform, not just position) OK")

    # ========================================================================
    # 9. screenshot: viewport with several placed blueprint runtime instances.
    #    mover_ent/mover2_ent are still actively moving with casts_shadow=True
    #    (by design, from sections 4/8a) -- exactly the thrash the shadow-
    #    cache decision warns about (see shadow_cache_bench numbers in the
    #    task report: ~5x slower per frame). That's already been asserted;
    #    stop the thrash here purely so this suite's own screenshot step
    #    doesn't pay for it repeatedly on every test run.
    # ========================================================================
    mover_ent.casts_shadow = False
    mover2_ent.casts_shadow = False
    editor.selected_folder = None
    OUT = os.path.join(tempfile.gettempdir(), "judge_bp_runtime.png")
    eng.run(scene, camera, max_frames=15, screenshot_path=OUT, overlay=editor.draw)
    print(f"9. screenshot saved: {OUT}")

    # ========================================================================
    # 10. runaway-thread cap (supervisor-added after review).
    #
    # A timed-out script is abandoned, NOT stopped -- CPython cannot kill a
    # thread -- and each survivor contends for the GIL, which this pure-
    # Python renderer pays for directly: measured on the starter scene,
    # 8.8 FPS clean -> 0.66 with one runaway alive (13x) -> 0.31 with two
    # (29x), accumulating one per Compile press without bound. The cap
    # stops that growth. Deliberately the LAST check in this suite: it
    # leaks a second runaway thread on purpose, so everything above it
    # (including the screenshot render) runs at the cheaper 1-thread cost.
    # ========================================================================
    assert blueprint_mod.MAX_RUNAWAY_THREADS == 2, (
        "this check assumes a cap of 2; update it if the cap changes")
    before_cap = blueprint_mod.runaway_thread_count()
    assert before_cap == 1, f"expected 1 runaway from check 1, got {before_cap}"

    # a legitimate script must still compile while below the cap -- a cap
    # that locks the user out of recompiling their CORRECTED script would
    # break the whole repair workflow (this is why the cap isn't 1).
    r_ok = engine.compile_blueprint(engine.DEFAULT_BLUEPRINT_SCRIPT, "StillWorks")
    assert r_ok["ok"] is True, ("a valid script must still compile while a "
                                "runaway is alive but below the cap", r_ok)

    r_second = engine.compile_blueprint(INFINITE_LOOP_SCRIPT, "Hang2")
    assert r_second["stage"] == "timeout", r_second
    assert blueprint_mod.runaway_thread_count() == 2, "second runaway should be tracked"

    t0 = time.perf_counter()
    r_blocked = engine.compile_blueprint(INFINITE_LOOP_SCRIPT, "Hang3")
    blocked_elapsed = time.perf_counter() - t0
    assert r_blocked["ok"] is False and r_blocked["stage"] == "timeout", r_blocked
    assert "restart" in r_blocked["message"].lower(), (
        "a blocked compile must tell the user a restart is needed", r_blocked)
    assert blocked_elapsed < blueprint_mod.EXEC_TIMEOUT_SECONDS, (
        f"a blocked compile must return immediately rather than waiting out "
        f"the timeout again, took {blocked_elapsed:.2f}s")
    assert blueprint_mod.runaway_thread_count() == 2, (
        "a blocked compile must NOT leak another runaway thread")
    print(f"10. runaway threads capped at {blueprint_mod.MAX_RUNAWAY_THREADS}: "
         f"valid script still compiles, over-cap loop refused in "
         f"{blocked_elapsed:.3f}s with no extra thread OK")

    # ========================================================================
    # no-pollution guards
    # ========================================================================
    _real_settings_after = (open(REAL_SETTINGS, "rb").read()
                            if os.path.exists(REAL_SETTINGS) else None)
    assert _real_settings_after == _real_settings_before, (
        "bp_runtime_checks touched the real settings.json -- an Editor() in "
        "this suite is missing settings_path=TEST_SETTINGS")
    real_assets_dir = os.path.join(WT, "assets")
    assert not os.path.exists(os.path.join(real_assets_dir, "blueprints", "moverbp.json")), \
        "a blueprint leaked into the REAL assets/blueprints dir"
    assert lib.directory == ASSETS_DIR and lib.directory != real_assets_dir
    print("no-pollution guards OK: real settings.json and real assets/ untouched")

    print("ALL BP RUNTIME CHECKS PASSED")

finally:
    _keys_patch.stop()
    shutil.rmtree(TMP_ASSETS_ROOT, ignore_errors=True)
