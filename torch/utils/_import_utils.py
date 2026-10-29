# [MODULE DOC] Helpers for optional dependencies: checking whether a module exists without
# importing it, and cached access to the dill package with its global pickler side effect
# undone. No import-time side effects.
import functools
import importlib.util
from types import ModuleType


def _check_module_exists(name: str) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if a module spec can be found for the given name,
    # without importing it; returns False if the lookup raises ImportError. Note that the
    # docstring has a stray asterisk typo ("without**").
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    try:
        spec = importlib.util.find_spec(name)
        return spec is not None
    except ImportError:
        return False


@functools.lru_cache
def dill_available() -> bool:
    # [FUNCTION DOC] Returns whether the dill module is installed (cached, no import).
    return _check_module_exists("dill")


@functools.lru_cache
def import_dill() -> ModuleType | None:
    # [FUNCTION DOC] Returns the dill module, or None if it is not installed. On first import
    # it disables dill's extension of the standard pickler dispatch table, so the standard
    # library pickler behavior is not changed globally. Cached after the first call.
    if not dill_available():
        return None

    import dill

    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    return dill
