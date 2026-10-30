# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Fixture FX tracer subclass that treats every module as a leaf module, used by
# package tests that trace modules without descending into submodules.

from torch.fx import Tracer


class TestAllLeafModulesTracer(Tracer):
    # [CLASS DOC] HELPER: FX Tracer that overrides the leaf-module decision so that no module is
    # ever traced into.
    def is_leaf_module(self, m, qualname):
        # [FUNCTION DOC] HELPER: Ignores both arguments and always returns True, making every
        # module a leaf for tracing.
        return True
