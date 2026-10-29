# [MODULE DOC] HELPER: Provides lazy_deprecated_import, a factory for module-level attribute
# hooks that warn about moved (deprecated) modules and forward lookups to the new location.
# Defines the warning message template. No import-time side effects beyond imports.
import importlib
import warnings
from collections.abc import Callable


_MESSAGE_TEMPLATE = (
    r"Usage of '{old_location}' is deprecated; please use '{new_location}' instead."
)


def lazy_deprecated_import(
    all: list[str],
    old_module: str,
    new_module: str,
) -> Callable:
    # [FUNCTION DOC] HELPER: Builds a module-level attribute hook for a deprecated module
    # location. Args: all is the list of attribute names that may be forwarded, old_module and
    # new_module are module names used in the warning text. Returns a callable taking an
    # attribute name that, for a listed name, emits a RuntimeWarning and returns that attribute
    # from new_module (imported on demand); for any other name it raises an AttributeError.
    r"""Import utility to lazily import deprecated packages / modules / functional.

    The old_module and new_module are also used in the deprecation warning defined
    by the `_MESSAGE_TEMPLATE`.

    Args:
        all: The list of the functions that are imported. Generally, the module's
            __all__ list of the module.
        old_module: Old module location
        new_module: New module location / Migrated location

    Returns:
        Callable to assign to the `__getattr__`

    Usage:

        # In the `torch/nn/quantized/functional.py`
        from torch.nn.utils._deprecation_utils import lazy_deprecated_import
        _MIGRATED_TO = "torch.ao.nn.quantized.functional"
        __getattr__ = lazy_deprecated_import(
            all=__all__,
            old_module=__name__,
            new_module=_MIGRATED_TO)
    """
    warning_message = _MESSAGE_TEMPLATE.format(
        old_location=old_module, new_location=new_module
    )

    def getattr_dunder(name: str) -> None:
        # [FUNCTION DOC] HELPER: Attribute hook returned by lazy_deprecated_import. If the name
        # is in the allowed list, warns (RuntimeWarning, stack level 2), imports the new module
        # and returns the attribute; otherwise raises AttributeError.
        # KNOWN ISSUE: the return annotation says None but the hook returns the looked-up
        # attribute.
        if name in all:
            # We are using the "RuntimeWarning" to make sure it is not
            # ignored by default.
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
            package = importlib.import_module(new_module)
            return getattr(package, name)
        raise AttributeError(f"Module {new_module!r} has no attribute {name!r}.")

    return getattr_dunder
