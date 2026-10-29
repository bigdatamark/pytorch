# mypy: allow-untyped-defs
# [PACKAGE DOC] Exposes torch.backends.quantized with module attributes 'engine' (get/set the active quantized engine by name) and 'supported_engines' (read-only list of names).
# Import-time side effect: replaces this module in the module registry with a wrapper module type implementing those properties.
import sys
import types

import torch


# This function should correspond to the enums present in c10/core/QEngine.h
def _get_qengine_id(qengine: str) -> int:
    # [FUNCTION DOC] HELPER: Maps an engine name (none/empty/None, fbgemm, qnnpack, onednn, x86) to its integer id matching the C++ enum; raises RuntimeError for any other name.
    if qengine == "none" or qengine == "" or qengine is None:
        ret = 0
    elif qengine == "fbgemm":
        ret = 1
    elif qengine == "qnnpack":
        ret = 2
    elif qengine == "onednn":
        ret = 3
    elif qengine == "x86":
        ret = 4
    else:
        ret = -1
        raise RuntimeError(f"{qengine} is not a valid value for quantized engine")
    return ret


# This function should correspond to the enums present in c10/core/QEngine.h
def _get_qengine_str(qengine: int) -> str:
    # [FUNCTION DOC] HELPER: Maps an engine integer id to its name, returning the placeholder '*undefined' for unknown ids.
    all_engines = {0: "none", 1: "fbgemm", 2: "qnnpack", 3: "onednn", 4: "x86"}
    return all_engines.get(qengine, "*undefined")


class _QEngineProp:
    # [CLASS DOC] HELPER: Descriptor for the 'engine' attribute; reading returns the current engine name, assigning a name sets the engine in the C++ backend (invalid names raise RuntimeError).
    def __get__(self, obj, objtype) -> str:
        # [FUNCTION DOC] HELPER: Returns the current quantized engine as a string.
        return _get_qengine_str(torch._C._get_qengine())

    def __set__(self, obj, val: str) -> None:
        # [FUNCTION DOC] HELPER: Converts the given name to an id and sets it as the active quantized engine (side effect on global state).
        torch._C._set_qengine(_get_qengine_id(val))


class _SupportedQEnginesProp:
    # [CLASS DOC] HELPER: Read-only descriptor listing the names of quantized engines supported in this build; assignment raises RuntimeError.
    def __get__(self, obj, objtype) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns a list of engine names for all engines the backend supports.
        qengines = torch._C._supported_qengines()
        return [_get_qengine_str(qe) for qe in qengines]

    def __set__(self, obj, val) -> None:
        # [FUNCTION DOC] HELPER: Always raises RuntimeError because the attribute is read-only.
        raise RuntimeError("Assignment not supported")


class QuantizedEngine(types.ModuleType):
    # [CLASS DOC] Module subclass that replaces this module; keeps the original in m, forwards missing attribute lookups to it, and defines the engine and supported_engines properties.
    def __init__(self, m, name):
        # [FUNCTION DOC] Creates the replacement module with the given name and keeps the original module in the attribute m.
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        # [FUNCTION DOC] Fallback attribute lookup: returns the named attribute from the wrapped original module.
        return self.m.__getattribute__(attr)

    engine = _QEngineProp()
    supported_engines = _SupportedQEnginesProp()


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = QuantizedEngine(sys.modules[__name__], __name__)
engine: str
supported_engines: list[str]
