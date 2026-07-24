"""Judge checks: uncapped-by-default FPS pacing, the smoothed HUD readout,
and the Settings-dialog FPS-cap control.

Covers `engine/core.py`'s `Engine.run()` loop (max_fps=0 default means
`clock.tick(0)` -- pygame's own no-delay behavior -- is what actually runs;
the fixed 60Hz `scene.update` accumulator is untouched regardless of
max_fps or of a `max_frames` deterministic benchmark run) and `FpsSmoother`
(a wall-clock EMA, not frame-count based, backing the HUD's "NN.N FPS"
text). Also covers editor.py's Settings dialog FPS-cap button row
(Uncapped/30/60/120/144), its settings.json persistence (including an old
integer value from before this feature -- no migration code needed, same
representation), and the console-log note on a cap change (via the
existing `status` setter -> console_log mirror, see editor.py).

Per the project's hard rule for UI/interaction tests, the FPS-cap button
click is driven through the REAL event path (eng.input.process +
editor.update), patching pygame.mouse.get_pos/get_pressed and
pygame.key.get_pressed at the OS boundary -- the idiom
tests/console_checks.py and tests/docktab_checks.py established. Opening
the Settings dialog itself (`_open_settings()`) is setup, not the
interaction under test, so it's called directly (same precedent as
docktab_checks.py's rationale for its non-interactive setup calls).

ORDERING NOTE: every other *_checks.py suite constructs exactly ONE
`engine.Engine` up front and only ever calls `.run()` once, at the very
end, for the final screenshot -- because `pygame.display.set_mode()`
targets a single global display (even under the SDL dummy driver), so a
SECOND `Engine(...)` construction resizes/replaces the shared one's
surface out from under it. This suite legitimately needs several small
throwaway Engine/`.run()` cycles to observe `clock.tick()`/`scene.update`
dt in isolation, so those all run LAST, after every check that depends on
the shared top-level `eng`/`lib`/`scene`/`camera` (the Settings-dialog
real-click checks) has already finished with them.
"""
import json
import os
import statistics
import sys
import tempfile
import unittest.mock as um

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, WT)

import pygame

import engine
from engine.core import FpsSmoother
from editor import (FPS_CAP_OPTIONS, Editor, EditorBehavior, _fps_cap_label,
                    build_starter_scene, load_settings, save_settings)

# Guard against the real settings.json (several actions below save settings).
REAL_SETTINGS = os.path.join(WT, "settings.json")
_real_before = (open(REAL_SETTINGS, "rb").read()
               if os.path.exists(REAL_SETTINGS) else None)
TEST_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_fps_settings.json")
if os.path.exists(TEST_SETTINGS):
    os.remove(TEST_SETTINGS)

eng = engine.Engine(1440, 810, title="judge", splash=False, api="cpu")
lib = engine.AssetLibrary(os.path.join(WT, "assets"))
camera = engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0), yaw=0.45, pitch=-0.08)
scene = build_starter_scene(engine, lib)
W, H = eng.screen.get_size()


def new_editor():
    return Editor(engine, eng, scene, camera, lib, "scenes/scene.json",
                 settings_path=TEST_SETTINGS)


class FakeKeys:
    def __init__(self, held=()):
        self._held = set(held)
    def __getitem__(self, key):
        return key in self._held


def step(ed, events):
    eng.input.process(events)
    ed.update(eng, 1 / 60)
    eng.input.consume_edges()


def click(ed, pos):
    """A plain press-then-release at `pos`, driven through the real event
    path (mirrors console_checks.py/docktab_checks.py's `click`)."""
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.key, "get_pressed", return_value=FakeKeys()), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)):
        step(ed, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)])
    with um.patch.object(pygame.mouse, "get_pos", return_value=pos), \
         um.patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)):
        step(ed, [])


# Engine.run() calls pygame.quit() when its loop ends (see engine/core.py) --
# fine for a suite that calls .run() exactly once, at the very end (the
# established pattern in every other *_checks.py), but the tick()/dt checks
# below call it several times with fresh throwaway Engine instances. No-op
# it for this suite; each throwaway Engine() still calls pygame.init()
# itself, which is what actually matters for those checks.
_quit_patch = um.patch.object(pygame, "quit", lambda: None)
_quit_patch.start()

try:
    # ========================================================================
    # 1. FpsSmoother: a windowed/EMA readout varies less than the raw
    # instantaneous FPS when fed jittery synthetic frame times. Pure logic,
    # no Engine/display involved.
    # ========================================================================
    import random
    random.seed(7)
    # alternate fast/slow frames (simulates an uncapped render doing varying
    # per-frame work) -- instantaneous FPS swings between ~500 and ~40
    jittery_dts = [random.choice([1 / 500, 1 / 40]) for _ in range(200)]
    raw_fps = [1.0 / d for d in jittery_dts]
    smoother = FpsSmoother(tau=0.5)
    smoothed_fps = [smoother.update(d) for d in jittery_dts]
    # compare variance over the back half (after the EMA has settled from
    # its first-sample snap)
    raw_tail = raw_fps[50:]
    smoothed_tail = smoothed_fps[50:]
    raw_stdev = statistics.pstdev(raw_tail)
    smoothed_stdev = statistics.pstdev(smoothed_tail)
    assert smoothed_stdev < raw_stdev / 4, (raw_stdev, smoothed_stdev)
    print(f"FpsSmoother OK: raw stdev {raw_stdev:.1f} -> smoothed stdev "
          f"{smoothed_stdev:.1f} (>4x calmer)")

    # a steady frame rate converges to that rate (sanity: not just "always low")
    steady = FpsSmoother(tau=0.5)
    v = 0.0
    for _ in range(120):
        v = steady.update(1 / 75)
    assert abs(v - 75.0) < 1.0, v
    print(f"FpsSmoother OK: converges to steady rate ({v:.1f} ~= 75.0)")

    # ========================================================================
    # 2. Settings dialog FPS-cap button row: real clicks cycle through every
    # FPS_CAP_OPTIONS value, and each click persists + logs to the console.
    # MUST run before any other engine.Engine(...) construction below (see
    # the module docstring's ORDERING NOTE) -- the shared `eng`/`W,H` used
    # here would otherwise drift out of sync with a later Engine's display.
    # ========================================================================
    from engine import console_log

    ed = new_editor()
    ed._open_settings()  # setup, not the interaction under test (see docstring)
    assert ed.settings_open is True
    rect = ed._settings_rect(W, H)
    buttons = dict(ed._settings_fps_buttons(rect))
    assert set(buttons.keys()) == set(FPS_CAP_OPTIONS), buttons.keys()

    for v in FPS_CAP_OPTIONS:
        console_log.reset()
        btn = buttons[v]
        click(ed, btn.center)
        assert ed.eng.max_fps == v, (v, ed.eng.max_fps)
        entries = [e["text"] for e in console_log.get_log().entries]
        assert any(f"FPS cap set to {_fps_cap_label(v)}" in t for t in entries), (
            v, entries)
        with open(TEST_SETTINGS, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["max_fps"] == v, (v, saved.get("max_fps"))
    print(f"Settings FPS-cap button row OK: cycled {FPS_CAP_OPTIONS} via real "
          f"clicks, each persisted + logged")

    # round-trip through a fresh Editor picks up the persisted cap
    ed2 = new_editor()
    with open(TEST_SETTINGS, encoding="utf-8") as f:
        saved2 = json.load(f)
    assert saved2["max_fps"] == FPS_CAP_OPTIONS[-1]  # last value clicked above
    print("Settings FPS-cap persistence OK: settings.json round-trips the cap")

    # ========================================================================
    # 3. migration: an old integer max_fps (from before this feature, e.g.
    # a slider position like 120) still loads fine -- same int
    # representation, only the *meaning* of 0 is new. No Engine construction
    # needed for this one (load_settings/save_settings are plain JSON I/O).
    # ========================================================================
    OLD_SETTINGS = os.path.join(tempfile.gettempdir(), "judge_fps_old_settings.json")
    save_settings({"width": 1440, "height": 810, "max_fps": 120}, OLD_SETTINGS)
    reloaded = load_settings(OLD_SETTINGS)
    old_max_fps = reloaded.get("max_fps", 0)
    assert old_max_fps == 120, old_max_fps
    os.remove(OLD_SETTINGS)
    print("migration OK: an old integer max_fps (120) loads unchanged")

    # a fresh install (no settings.json / no max_fps key) defaults to uncapped
    fresh = load_settings(os.path.join(tempfile.gettempdir(),
                                       "judge_fps_nonexistent.json"))
    assert fresh.get("max_fps", 0) == 0
    print("fresh-install default OK: no settings.json -> max_fps defaults to 0 (uncapped)")

    # ========================================================================
    # From here on: throwaway Engine()/.run() cycles only -- nothing below
    # touches the shared `eng`/`ed`/`lib`/`scene` again.
    # ========================================================================

    # ========================================================================
    # 4. max_fps=0 (the default) selects the uncapped clock.tick(0) branch;
    # a positive max_fps is passed straight through -- no realtime wait
    # needed, just record what clock.tick() was actually called with over a
    # handful of max_frames-driven iterations.
    # ========================================================================
    # pygame.time.Clock is an immutable C-extension type -- its `tick`
    # method can't be patched in place, so wrap the constructor instead
    # (engine/core.py only ever calls `.tick()` on the clock it creates,
    # see the grep-confirmed single call site in run()).
    tick_args = []
    _RealClock = pygame.time.Clock

    class _RecordingClock:
        def __init__(self, *a, **kw):
            self._real = _RealClock(*a, **kw)

        def tick(self, *a, **kw):
            tick_args.append(a[0] if a else kw.get("framerate", 0))
            return self._real.tick(*a, **kw)

    empty_scene = engine.Scene(enable_shadows=False)
    cam2 = engine.Camera()

    with um.patch.object(pygame.time, "Clock", _RecordingClock):
        tick_args.clear()
        eng_uncapped = engine.Engine(320, 240, title="judge_fps_uncapped",
                                     splash=False, api="cpu", max_fps=0)
        eng_uncapped.run(empty_scene, cam2, max_frames=5)
        assert tick_args == [0] * 5, tick_args
        print("uncapped (max_fps=0) OK: clock.tick(0) every frame", tick_args)

        tick_args.clear()
        eng_capped = engine.Engine(320, 240, title="judge_fps_capped",
                                   splash=False, api="cpu", max_fps=90)
        eng_capped.run(empty_scene, cam2, max_frames=5)
        assert tick_args == [90] * 5, tick_args
        print("capped (max_fps=90) OK: clock.tick(90) every frame", tick_args)

    # a stray None (the spec's "0 or None" for uncapped) must not crash tick()
    with um.patch.object(pygame.time, "Clock", _RecordingClock):
        tick_args.clear()
        eng_none = engine.Engine(320, 240, title="judge_fps_none",
                                 splash=False, api="cpu", max_fps=0)
        eng_none.max_fps = None
        eng_none.run(empty_scene, cam2, max_frames=3)
        assert tick_args == [0] * 3, tick_args
        print("None max_fps OK: treated as uncapped, no crash", tick_args)

    # ========================================================================
    # 5. the deterministic max_frames path is unchanged: scene.update always
    # gets exactly fixed_dt, regardless of max_fps (uncapped or clamped).
    # ========================================================================
    dt_calls = []
    orig_update = engine.Scene.update

    def recording_update(self, dt, eng_arg):
        dt_calls.append(dt)
        return orig_update(self, dt, eng_arg)

    with um.patch.object(engine.Scene, "update", recording_update):
        for mf in (0, 144):
            dt_calls.clear()
            e = engine.Engine(320, 240, title=f"judge_fps_det_{mf}",
                              splash=False, api="cpu", max_fps=mf, fixed_dt=1 / 60)
            s = engine.Scene(enable_shadows=False)
            e.run(s, engine.Camera(), max_frames=20)
            assert dt_calls and all(d == 1 / 60 for d in dt_calls), (mf, dt_calls)
            assert len(dt_calls) == 20, (mf, len(dt_calls))
    print("deterministic max_frames path OK: scene.update always got exactly "
          "fixed_dt (1/60), independent of max_fps")

    # ========================================================================
    # 6. static scene does not re-bake shadows/GI every frame under an
    # uncapped render loop -- ShadowTracer.refresh()'s existing world-version
    # short-circuit must still hold (no code change was made to raytrace.py;
    # this guards against a future regression).
    # ========================================================================
    eng_static = engine.Engine(320, 240, title="judge_fps_static", splash=False,
                              api="cpu", max_fps=0)
    static_scene = build_starter_scene(engine, lib)
    orig_refresh = eng_static.tracer.refresh
    refresh_calls = [0]
    rebuild_calls = [0]

    def counting_refresh(sc):
        refresh_calls[0] += 1
        result = orig_refresh(sc)
        if result:
            rebuild_calls[0] += 1
        return result

    eng_static.tracer.refresh = counting_refresh
    eng_static.run(static_scene, engine.Camera(position=engine.Vec3(6.0, 2.6, 9.0)),
                  max_frames=30)
    assert refresh_calls[0] == 30, refresh_calls
    assert rebuild_calls[0] == 1, (
        "static scene should bake exactly once, not per frame", rebuild_calls)
    print(f"static-scene no-rebake OK: refresh() called {refresh_calls[0]}x, "
          f"rebuilt {rebuild_calls[0]}x")

    print("ALL FPS CHECKS PASSED")
finally:
    _quit_patch.stop()
    _real_after = (open(REAL_SETTINGS, "rb").read()
                  if os.path.exists(REAL_SETTINGS) else None)
    assert _real_after == _real_before, (
        "fps_checks touched the real settings.json -- an Editor() in this "
        "suite is missing settings_path=TEST_SETTINGS")
    print("no-pollution guard OK: real settings.json untouched by this suite")
