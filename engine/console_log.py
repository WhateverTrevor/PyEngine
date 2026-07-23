"""Lightweight in-engine message log -- a bounded ring buffer the editor's
Console panel renders, so long-running work (imports, blueprint compiles,
the shadow/GI bake) is visible WHILE it's happening instead of only after,
via the pre-existing transient status strip (which stays working -- see
Editor.status in editor.py, now a property that mirrors every message here
too).

A single module-level singleton (`_log`) backs the log_info/log_warn/
log_error/get_log helpers below, so call sites (engine.core, editor.py's
import/compile paths, ...) can log without an editor/engine reference
threaded through every function. No pygame/numpy dependency -- this module
is safe to import at any time, including from engine/core.py's hot loop.
"""
from __future__ import annotations

import time
from collections import deque

LEVELS = ("info", "warn", "error")
MAX_ENTRIES = 500  # ring buffer bound -- long sessions never grow unbounded


class ConsoleLog:
    def __init__(self, maxlen: int = MAX_ENTRIES):
        self.entries = deque(maxlen=maxlen)

    def log(self, level: str, text: str) -> None:
        if level not in LEVELS:
            level = "info"
        self.entries.append({"time": time.time(), "level": level, "text": text})

    def info(self, text: str) -> None:
        self.log("info", text)

    def warn(self, text: str) -> None:
        self.log("warn", text)

    def error(self, text: str) -> None:
        self.log("error", text)

    def clear(self) -> None:
        self.entries.clear()


_log = ConsoleLog()  # module-level singleton -- see module docstring


def log(level: str, text: str) -> None:
    """Dynamic-level variant of log_info/log_warn/log_error, for call sites
    that compute the level (e.g. Editor.status's setter, which infers
    error/info from the message text)."""
    _log.log(level, text)


def log_info(text: str) -> None:
    _log.log("info", text)


def log_warn(text: str) -> None:
    _log.log("warn", text)


def log_error(text: str) -> None:
    _log.log("error", text)


def get_log() -> ConsoleLog:
    """The singleton instance -- the console panel reads `get_log().entries`."""
    return _log


def reset() -> None:
    """Clear the singleton -- tests use this so one test's entries can't
    leak into the next test's assertions within the same process."""
    _log.clear()
