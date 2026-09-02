"""Optional compiled-kernel loader for the Case-A model.

The workspace keeps Python wheels in ``.codex_deps`` instead of modifying the
machine-wide interpreter.  Resolve that directory once here so every Case-A
entry point uses the compiled kernels without requiring a manual PYTHONPATH.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _import_njit():
    try:
        from numba import njit as compiled_njit

        return compiled_njit
    except ImportError:
        pass

    for parent in Path(__file__).resolve().parents:
        local_dependencies = parent / ".codex_deps"
        if not local_dependencies.is_dir():
            continue
        dependency_path = str(local_dependencies)
        if dependency_path not in sys.path:
            sys.path.insert(0, dependency_path)
        try:
            from numba import njit as compiled_njit

            return compiled_njit
        except ImportError:
            break

    def portable_njit(*_args, **_kwargs):
        def decorate(function):
            return function

        return decorate

    return portable_njit


njit = _import_njit()


__all__ = ["njit"]
