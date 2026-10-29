# mypy: allow-untyped-defs
# [PACKAGE DOC] Exposes torch.backends.xnnpack with a read-only 'enabled' attribute reporting whether XNNPACK is enabled in this build.
# Import-time side effect: replaces this module in the module registry with a wrapper module type that forwards unknown attribute lookups to the original module.
import sys
import types

import torch


class _XNNPACKEnabled:
    # [CLASS DOC] HELPER: Descriptor for the 'enabled' attribute; reads query the C++ backend and any assignment raises RuntimeError.
    def __get__(self, obj, objtype):
        # [FUNCTION DOC] HELPER: Returns whether XNNPACK is enabled, queried from the C++ backend (arguments unused).
        return torch._C._is_xnnpack_enabled()

    def __set__(self, obj, val):
        # [FUNCTION DOC] HELPER: Always raises RuntimeError because the attribute is read-only.
        raise RuntimeError("Assignment not supported")


class XNNPACKEngine(types.ModuleType):
    # [CLASS DOC] Module subclass that replaces this module; stores the original module in m, falls back to it for missing attributes, and exposes the read-only 'enabled' property.
    def __init__(self, m, name):
        # [FUNCTION DOC] Creates the replacement module with the given name and keeps the original module in the attribute m.
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        # [FUNCTION DOC] Fallback attribute lookup: returns the named attribute from the wrapped original module (raising AttributeError if absent).
        return self.m.__getattribute__(attr)

    enabled = _XNNPACKEnabled()


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = XNNPACKEngine(sys.modules[__name__], __name__)
