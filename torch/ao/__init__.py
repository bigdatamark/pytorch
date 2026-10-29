# [PACKAGE DOC] torch.ao (architecture optimization): lazily exposes the subpackages nn, ns, pruning and quantization via a module-level attribute hook to avoid cyclic imports; unknown names raise AttributeError.
# torch.ao is a package with a lot of interdependencies.
# We will use lazy import to avoid cyclic dependencies here.

from typing import TYPE_CHECKING as _TYPE_CHECKING


if _TYPE_CHECKING:
    from types import ModuleType

    from torch.ao import (
        nn as nn,
        ns as ns,
        pruning as pruning,
        quantization as quantization,
    )


__all__ = [
    "nn",
    "ns",
    "pruning",
    "quantization",
]


def __getattr__(name: str) -> "ModuleType":
    # [FUNCTION DOC] Lazy submodule loader: if name is one of the exported subpackages, import and return it relative to this package; otherwise raise AttributeError.
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
