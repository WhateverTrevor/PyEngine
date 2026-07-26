"""Blueprint script compile + in-engine bug check + runtime attach.

A Blueprint asset (`engine.assets.BlueprintAsset`) pairs posed-mesh
components ({"asset_name", "position", "rotation", "scale"} per entry,
merged into one composite entity by `BlueprintAsset.instantiate` -- see
engine/mesh.py's `merge_meshes` and assets.py's BlueprintAsset docstring)
with a Python script that defines a `Behavior` subclass (see
engine/scene.py). `compile_blueprint` is the
headline feature: it runs the script IN THIS PROCESS -- that's the intended
design, this is the user's own engine -- but every stage is exception-
isolated so a broken script can never crash the editor, and (this run) an
infinite loop at module level can no longer HANG it either. It's a plain
function (no pygame/editor dependency) so tests and editor.py's
ScriptEditorUI both call it the same way.

Four stages, each reported distinctly:
  1. `compile(source, "<blueprint:NAME>", "exec")` -- catches SyntaxError
     (and the rarer ValueError, e.g. an embedded NUL byte) with line/col.
  2. `exec` in a fresh namespace, run on a worker thread with a wall-clock
     timeout (`EXEC_TIMEOUT_SECONDS`) -- catches anything raised at
     definition time (NameError, ImportError, ZeroDivisionError, ...),
     including BaseException subclasses like SystemExit so a careless
     top-level `exit()` can't tear down the editor process. If the thread
     is still running after the timeout, this is reported as its own
     `"timeout"` stage (see `_exec_with_timeout`'s docstring for the honest
     limit of what a timeout can and can't do to that thread).
  3. Validate: the namespace must contain a Behavior subclass (other than
     Behavior itself) for the blueprint to be usable.

`compile_blueprint` returns only the JSON-safe result dict (persisted as
BlueprintAsset.compile_result). `compile_blueprint_with_class` runs the
exact same pipeline but also hands back the compiled Behavior subclass
itself, for callers that need to actually instantiate it (BlueprintAsset.
instantiate -- see its docstring in engine/assets.py, and
`instantiate_behavior`/`BlueprintBehaviorProxy` below, which turn that
class into a running, error-isolated per-entity Behavior).
"""
from __future__ import annotations

import threading
import traceback

from .scene import Behavior

DEFAULT_BLUEPRINT_SCRIPT = '''"""New blueprint behavior."""
from engine.scene import Behavior


class NewBehavior(Behavior):
    def update(self, entity, dt, engine):
        pass
'''

# Wall-clock ceiling for a blueprint script's module-level exec. Blueprint
# scripts are expected to do nothing heavier than define a class (import
# statements resolve modules already imported elsewhere in the process, so
# they're near-free); a few seconds is generous headroom for a legitimate
# script doing some one-off setup work (building a lookup table, etc.)
# while still catching a `while True: pass` hang within a human-noticeable,
# not "did the editor freeze?", pause.
EXEC_TIMEOUT_SECONDS = 2.0

# How many abandoned runaway threads may be alive before we refuse to start
# another. A timed-out script is NOT stopped (see `_exec_with_timeout`), and
# because this engine renders in pure Python, each surviving busy-loop thread
# contends for the GIL and costs real frame rate -- measured on the starter
# scene: 8.8 FPS clean, 0.66 FPS with one runaway alive (13x), 0.31 FPS with
# two (29x), and it accumulates without bound. Hitting Compile repeatedly on
# a script with an infinite loop is the obvious way a user reaches that, so
# past this cap we return the timeout verdict WITHOUT spawning another
# thread: the editor stays at a known, bounded level of degradation instead
# of sliding toward unusable. Recovery is a restart -- real termination
# needs OS-process isolation (subprocess + kill), a bigger feature.
#
# Why not 1: the cap blocks the NEXT exec unconditionally, and we cannot
# know whether that script loops without running it. A cap of 1 therefore
# locks the user out of compiling ANY script -- including the corrected
# one -- after a single accidental `while True:`, which breaks the actual
# repair workflow. A small allowance keeps fixing-and-recompiling possible
# while still bounding the damage.
MAX_RUNAWAY_THREADS = 2

_runaway_threads: list[threading.Thread] = []


def runaway_thread_count() -> int:
    """Number of abandoned blueprint-exec threads still alive (see
    MAX_RUNAWAY_THREADS). Prunes threads that have since finished, so a
    script that was merely slow rather than non-terminating stops counting
    against the cap once it completes. Exposed for the editor/status UI and
    for tests."""
    _runaway_threads[:] = [t for t in _runaway_threads if t.is_alive()]
    return len(_runaway_threads)


def _exec_with_timeout(code, namespace: dict, timeout: float):
    """Run `exec(code, namespace)` on a worker thread with a wall-clock
    timeout, so a module-level infinite loop (`while True: pass`) can no
    longer hang the calling thread (the editor's main thread, or a test).

    Returns one of:
        ("ok", None)      -- exec finished normally within `timeout`
        ("error", exc)     -- exec raised `exc` within `timeout`
        ("timeout", None)  -- exec was still running when `timeout` elapsed

    HONESTY NOTE (read before "fixing" this to be more thorough): CPython
    cannot forcibly terminate a running thread -- there is no safe,
    supported way to kill another thread's execution mid-bytecode. This
    function does NOT sandbox or stop the runaway code. All it does is stop
    WAITING for it: `thread.join(timeout)` returns control to the caller
    regardless of whether `runner` ever finishes, so the editor/caller stays
    responsive and gets a definite answer. The thread itself (daemon=True,
    so it can't block process exit) keeps running -- competing for the GIL,
    burning a CPU core -- until it either finishes on its own or the whole
    process exits. A genuinely non-terminating script (true `while True:
    pass`) will therefore leak one live thread for the rest of the process
    lifetime. That's the accepted trade named in the module docstring and
    the task brief: responsiveness + honest reporting, not real isolation.
    Real forced termination would need OS-process isolation (subprocess +
    kill), which is a bigger feature and out of scope here.

    Because those abandoned threads cost measurable frame rate and would
    otherwise accumulate one per Compile press, they are tracked and capped
    (see MAX_RUNAWAY_THREADS): at the cap this returns ("blocked", None)
    without starting another one.
    """
    if runaway_thread_count() >= MAX_RUNAWAY_THREADS:
        return "blocked", None

    result: dict = {"status": None, "exc": None}

    def runner():
        try:
            exec(code, namespace)
            result["status"] = "ok"
        except BaseException as ex:  # noqa: BLE001 -- mirrors the non-threaded
            result["status"] = "error"  # exec()'s own BaseException catch below
            result["exc"] = ex

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        _runaway_threads.append(thread)
        return "timeout", None
    return result["status"], result["exc"]


def _compile_impl(source: str, blueprint_name: str) -> tuple[dict, type | None]:
    """Shared implementation behind `compile_blueprint` and
    `compile_blueprint_with_class` -- see module docstring for the four
    stages. Returns (result_dict, class_or_None); class is None whenever
    result_dict["ok"] is False.
    """
    filename = f"<blueprint:{blueprint_name}>"
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as ex:
        return ({"ok": False, "stage": "syntax", "message": ex.msg or str(ex),
                "line": ex.lineno, "col": ex.offset}, None)
    except BaseException as ex:  # e.g. ValueError: source has a NUL byte
        return ({"ok": False, "stage": "syntax", "message": f"{type(ex).__name__}: {ex}",
                "line": None, "col": None}, None)

    # Behavior is seeded into the namespace as a convenience (Unreal
    # blueprints don't require an import either) -- a script can still
    # `from engine.scene import Behavior` itself, both work.
    namespace = {"__name__": f"blueprint_{blueprint_name}", "Behavior": Behavior}
    status, exc = _exec_with_timeout(code, namespace, EXEC_TIMEOUT_SECONDS)
    if status == "blocked":
        return ({"ok": False, "stage": "timeout",
                "message": f"not run: {runaway_thread_count()} earlier runaway "
                           f"script thread is still burning CPU and cannot be "
                           f"stopped -- RESTART THE EDITOR to recover full "
                           f"performance, then fix the infinite loop before "
                           f"compiling again",
                "line": None, "col": None}, None)
    if status == "timeout":
        return ({"ok": False, "stage": "timeout",
                "message": f"script did not finish within {EXEC_TIMEOUT_SECONDS:g}s "
                           f"(likely an infinite loop at module level) -- the editor "
                           f"stayed responsive, but the runaway thread was abandoned, "
                           f"not stopped, and will SLOW THE EDITOR until you restart "
                           f"it (measured ~13x on the starter scene's CPU renderer; "
                           f"less on a GPU backend, but never free)",
                "line": None, "col": None}, None)
    if status == "error":
        line = None
        for frame in traceback.extract_tb(exc.__traceback__):
            if frame.filename == filename:
                line = frame.lineno  # last match = innermost frame in the script
        return ({"ok": False, "stage": "exec", "message": f"{type(exc).__name__}: {exc}",
                "line": line, "col": None}, None)

    found = None
    for value in namespace.values():
        if isinstance(value, type) and value is not Behavior and issubclass(value, Behavior):
            found = value
            break
    if found is None:
        return ({"ok": False, "stage": "validate",
                "message": "no Behavior subclass found -- define a class that "
                           "subclasses Behavior (engine.scene.Behavior)",
                "line": None, "col": None}, None)
    return ({"ok": True, "class_name": found.__name__}, found)


def compile_blueprint(source: str, blueprint_name: str = "Blueprint") -> dict:
    """Compile + exec `source`, looking for a Behavior subclass.

    Returns a JSON-serializable dict (safe to store as
    BlueprintAsset.compile_result):
        {"ok": True, "class_name": "..."}
    or
        {"ok": False, "stage": "syntax" | "exec" | "timeout" | "validate",
         "message": "...", "line": int | None, "col": int | None}
    Never raises.
    """
    result, _cls = _compile_impl(source, blueprint_name)
    return result


def compile_blueprint_with_class(source: str, blueprint_name: str = "Blueprint"):
    """Like `compile_blueprint`, but also returns the compiled Behavior
    subclass itself (not just its name in the result dict), for callers
    that need to actually instantiate it -- e.g. `instantiate_behavior`
    below. `class` can't be part of the returned dict: it isn't
    JSON-serializable, and BlueprintAsset.compile_result must stay exactly
    that (persisted to disk).

    Returns (result_dict, class_or_None); class is None whenever
    result_dict["ok"] is False. result_dict is identical to what
    `compile_blueprint(source, blueprint_name)` would return for the same
    input -- safe to persist as BlueprintAsset.compile_result either way.
    """
    return _compile_impl(source, blueprint_name)


class BlueprintBehaviorProxy(Behavior):
    """Wraps one compiled, user-authored Behavior instance so a bug in
    blueprint script code can never crash the engine loop or spam the
    console at 60Hz. Attached by `instantiate_behavior` below; NOT used for
    any built-in behavior (FlyController/SunController/Flicker/... keep
    running through engine/scene.py's Scene.update loop completely
    unwrapped -- see that module's docstring for why a blanket try/except
    there would be the wrong fix).

    start()/update() are isolated independently: a script whose start()
    raises still gets its update() attempted, and vice versa, matching
    Scene.update's own start()-then-update() sequencing per behavior.
    First exception from either one logs ONCE (blueprint name + stage +
    exception) via engine.console_log.log_warn, then disables this proxy
    entirely (both start() and update() become no-ops from then on) --
    repeating the same failure every frame at 60Hz would drown the console.
    """

    def __init__(self, inner: Behavior, blueprint_name: str):
        self.inner = inner
        self.blueprint_name = blueprint_name
        self.disabled = False
        self._warned_shadow = False

    def _fail(self, stage: str, ex: Exception) -> None:
        from . import console_log
        console_log.log_warn(
            f"blueprint '{self.blueprint_name}': {stage}() raised "
            f"{type(ex).__name__}: {ex} -- disabling this behavior")
        self.disabled = True

    def start(self, entity, engine) -> None:
        if self.disabled:
            return
        try:
            self.inner.start(entity, engine)
        except Exception as ex:  # noqa: BLE001 -- deliberate: isolate user script bugs
            self._fail("start", ex)

    @staticmethod
    def _pose_key(entity):
        """The same thing ShadowTracer keys its caster cache on: the entity's
        FULL transform, not just its position. ShadowTracer.update compares
        `e.transform.matrix().tobytes()` (see engine/raytrace.py), so a
        blueprint that only ROTATES or SCALES its entity invalidates the bake
        exactly as a translating one does -- watching position alone would
        miss those and leave the author with the cost and no warning. Built
        from the Vec3 components rather than the 4x4 matrix to keep this
        per-frame check allocation-cheap."""
        t = entity.transform
        return (t.position.x, t.position.y, t.position.z,
               t.rotation.x, t.rotation.y, t.rotation.z,
               t.scale.x, t.scale.y, t.scale.z)

    def update(self, entity, dt: float, engine) -> None:
        if self.disabled:
            return
        before = self._pose_key(entity)
        try:
            self.inner.update(entity, dt, engine)
        except Exception as ex:  # noqa: BLE001 -- deliberate: isolate user script bugs
            self._fail("update", ex)
            return
        # SHADOW-CACHE TRAP: any moving shadow caster bumps ShadowTracer's
        # whole-scene world_version every frame (see engine/raytrace.py's
        # module docstring), which re-triggers a full lighting bake
        # continuously -- exactly what the built-in Ghost asset avoids by
        # declaring `"casts_shadow": false` for its Bob-animated entity
        # (assets/ghost.json). A blueprint script moving its own entity is
        # the same situation, just author-controlled instead of asset-JSON-
        # controlled, so this can't be auto-corrected without possibly
        # removing a shadow the author wants -- it's flagged once instead,
        # the same "make it visible, don't silently eat the cost" policy as
        # the exception path above.
        if not self._warned_shadow and entity.casts_shadow:
            if self._pose_key(entity) != before:
                self._warned_shadow = True
                from . import console_log
                console_log.log_warn(
                    f"blueprint '{self.blueprint_name}': entity transform changed "
                    f"every frame while casts_shadow=True -- this re-triggers a whole-scene "
                    f"shadow/GI bake every frame it keeps moving (see "
                    f"engine/raytrace.py's cache-invalidation rule). If this "
                    f"movement is intentional, set entity.casts_shadow = False "
                    f"in the blueprint script (same fix as the built-in Ghost "
                    f"asset's Bob animation, assets/ghost.json).")


def instantiate_behavior(script: str, blueprint_name: str) -> Behavior | None:
    """Compile `script` and return a fresh, error-isolated Behavior ready to
    attach to a newly-instantiated blueprint entity, or None if the script
    is blank, doesn't compile, times out, or has no Behavior subclass --
    all of which must still let the entity place fine as a static posed
    mesh (see BlueprintAsset.instantiate). Always re-compiles from source
    rather than trusting a persisted `compile_result`: that dict is
    JSON-only (no class object survives a JSON round-trip) and only
    reflects whatever the editor's Compile button last ran, which can go
    stale relative to the current script text (see ScriptEditorUI's save
    policy) -- instantiation needs a result for THIS exact source.
    """
    if not script.strip():
        return None
    result, cls = compile_blueprint_with_class(script, blueprint_name)
    if not result.get("ok") or cls is None:
        from . import console_log
        loc = f" (line {result['line']})" if result.get("line") else ""
        console_log.log_warn(
            f"blueprint '{blueprint_name}': placed as a static mesh only -- "
            f"{result['stage']} error{loc}: {result['message']}")
        return None
    try:
        instance = cls()
    except Exception as ex:  # noqa: BLE001 -- constructor is still user code
        from . import console_log
        console_log.log_warn(
            f"blueprint '{blueprint_name}': Behavior '{cls.__name__}' raised "
            f"{type(ex).__name__} in __init__, placed as a static mesh only: {ex}")
        return None
    return BlueprintBehaviorProxy(instance, blueprint_name)
