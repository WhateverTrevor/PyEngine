"""Blueprint script compile + in-engine bug check.

A Blueprint asset (`engine.assets.BlueprintAsset`) pairs posed-mesh
components (run 2 -- empty this run) with a Python script that defines a
`Behavior` subclass (see engine/scene.py). `compile_blueprint` is the
headline feature: it runs the script IN THIS PROCESS -- that's the intended
design, this is the user's own engine -- but every stage is exception-
isolated so a broken script can never crash or hang the editor. It's a
plain function (no pygame/editor dependency) so tests and editor.py's
ScriptEditorUI both call it the same way.

Three stages, each reported distinctly:
  1. `compile(source, "<blueprint:NAME>", "exec")` -- catches SyntaxError
     (and the rarer ValueError, e.g. an embedded NUL byte) with line/col.
  2. `exec` in a fresh namespace -- catches anything raised at definition
     time (NameError, ImportError, ZeroDivisionError, ...), including
     BaseException subclasses like SystemExit so a careless top-level
     `exit()` can't tear down the editor process.
  3. Validate: the namespace must contain a Behavior subclass (other than
     Behavior itself) for the blueprint to be usable.

Known limitation (documented, not solved this run): a script with an
infinite loop at module level (e.g. `while True: pass`) will hang the
`exec` call -- there's no worker-thread/timeout sandbox here. That's a
bigger feature (safe forced-termination of arbitrary Python code isn't
possible without OS-process isolation) and out of scope for "never crashes"
via try/except.
"""
from __future__ import annotations

import traceback

from .scene import Behavior

DEFAULT_BLUEPRINT_SCRIPT = '''"""New blueprint behavior."""
from engine.scene import Behavior


class NewBehavior(Behavior):
    def update(self, entity, dt, engine):
        pass
'''


def compile_blueprint(source: str, blueprint_name: str = "Blueprint") -> dict:
    """Compile + exec `source`, looking for a Behavior subclass.

    Returns a JSON-serializable dict (safe to store as
    BlueprintAsset.compile_result):
        {"ok": True, "class_name": "..."}
    or
        {"ok": False, "stage": "syntax" | "exec" | "validate",
         "message": "...", "line": int | None, "col": int | None}
    Never raises.
    """
    filename = f"<blueprint:{blueprint_name}>"
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as ex:
        return {"ok": False, "stage": "syntax", "message": ex.msg or str(ex),
                "line": ex.lineno, "col": ex.offset}
    except BaseException as ex:  # e.g. ValueError: source has a NUL byte
        return {"ok": False, "stage": "syntax", "message": f"{type(ex).__name__}: {ex}",
                "line": None, "col": None}

    # Behavior is seeded into the namespace as a convenience (Unreal
    # blueprints don't require an import either) -- a script can still
    # `from engine.scene import Behavior` itself, both work.
    namespace = {"__name__": f"blueprint_{blueprint_name}", "Behavior": Behavior}
    try:
        exec(code, namespace)
    except BaseException as ex:
        line = None
        for frame in traceback.extract_tb(ex.__traceback__):
            if frame.filename == filename:
                line = frame.lineno  # last match = innermost frame in the script
        return {"ok": False, "stage": "exec", "message": f"{type(ex).__name__}: {ex}",
                "line": line, "col": None}

    found = None
    for value in namespace.values():
        if isinstance(value, type) and value is not Behavior and issubclass(value, Behavior):
            found = value
            break
    if found is None:
        return {"ok": False, "stage": "validate",
                "message": "no Behavior subclass found -- define a class that "
                           "subclasses Behavior (engine.scene.Behavior)",
                "line": None, "col": None}
    return {"ok": True, "class_name": found.__name__}
