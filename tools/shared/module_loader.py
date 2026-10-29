# [MODULE DOC] Provides import_module, a helper that loads a Python source file as a module by
# explicit file path, without requiring it to be on the import path. No import-time side
# effects.
from importlib.abc import Loader
from types import ModuleType
from typing import cast


def import_module(name: str, path: str) -> ModuleType:
    # [FUNCTION DOC] Load and execute the Python file at path as a module named name and
    # return it. The module is not registered in the module table. Raises AssertionError if
    # no import spec can be built for the path. Executing the file runs its top-level code
    # (side effects of that code apply).
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise AssertionError(f"Failed to load spec for {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    cast(Loader, spec.loader).exec_module(module)
    return module
