# [PACKAGE DOC] torch.ao.nn.intrinsic: fused float modules (conv/linear/bn/relu combinations) re-exported from .modules; also imports the private _FusedModule base. Unknown names that appear in __all__ are imported lazily as submodules.
import types

from .modules import *  # noqa: F403
from .modules.fused import _FusedModule


# # Subpackages
# from . import qat
# from . import quantized

__all__ = [
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
    "BNReLU2d",
    "BNReLU3d",
    "LinearBn1d",
    "LinearLeakyReLU",
    "LinearTanh",
    "ConvAdd2d",
    "ConvAddReLU2d",
]


# We are exposing all subpackages to the end-user.
# Because of possible inter-dependency, we want to avoid
# the cyclic imports, thus implementing lazy version
# as per https://peps.python.org/pep-0562/
def __getattr__(name: str) -> types.ModuleType:
    # [FUNCTION DOC] Lazy loader: if the name is in __all__, import it as a submodule of this package and return it; otherwise raise AttributeError.
    # KNOWN ISSUE: every name in __all__ is a fused class already imported at top, so this hook is effectively only reached for missing submodule lookups and would fail for those names.
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
