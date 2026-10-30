# Owner(s): ["module: inductor"]
# [MODULE DOC] Checks that each Inductor backend codegen overrides class implements every
# operation declared in the ops handler interface and defines no undeclared public operation.
# Backends covered: Triton kernel, C++ scalar, C++ vectorized, Halide and Metal (Metal test
# skipped). No import-time side effects besides imports; run_tests is invoked only when run as a
# script.
import unittest

from torch._inductor.codegen.cpp import CppOverrides, CppVecOverrides
from torch._inductor.codegen.halide import HalideOverrides
from torch._inductor.codegen.mps import MetalOverrides
from torch._inductor.codegen.triton import TritonKernelOverrides
from torch._inductor.ops_handler import list_ops, OP_NAMES, OpsHandler
from torch._inductor.test_case import TestCase


class TestOpCompleteness(TestCase):
    # [CLASS DOC] Test case with one check per backend overrides class. All checks delegate to
    # the shared completeness helper; the class holds no state.
    def verify_ops_handler_completeness(self, handler):
        # [FUNCTION DOC] HELPER: Asserts that the given overrides class is in sync with the ops
        # handler interface. For every name in the set of declared operation names, the
        # attribute on the handler must not be the very same object as the default one on the
        # base interface class, otherwise the test fails with a message naming the missing
        # operation. Then any operation found on the handler that is not in the declared set
        # raises an AssertionError listing the extras. No return value.
        for op in OP_NAMES:
            self.assertIsNot(
                getattr(handler, op),
                getattr(OpsHandler, op),
                msg=lambda msg: f"{msg}\n{handler} must implement {op}",
            )
        extra_ops = list_ops(handler) - OP_NAMES
        if extra_ops:
            raise AssertionError(
                f"{handler} has an extra ops: {extra_ops}, add them to OpHandler class or prefix with `_`"
            )

    def test_triton_overrides(self):
        # [FUNCTION DOC] Verifies the Triton kernel overrides class implements all declared
        # operations.
        self.verify_ops_handler_completeness(TritonKernelOverrides)

    def test_cpp_overrides(self):
        # [FUNCTION DOC] Verifies the scalar C++ overrides class implements all declared
        # operations.
        self.verify_ops_handler_completeness(CppOverrides)

    def test_cpp_vec_overrides(self):
        # [FUNCTION DOC] Verifies the vectorized C++ overrides class implements all declared
        # operations.
        self.verify_ops_handler_completeness(CppVecOverrides)

    def test_halide_overrides(self):
        # [FUNCTION DOC] Verifies the Halide overrides class implements all declared operations.
        self.verify_ops_handler_completeness(HalideOverrides)

    @unittest.skip("MPS backend not yet finished")
    def test_metal_overrides(self):
        # [FUNCTION DOC] Verifies the Metal overrides class implements all declared operations.
        # Currently skipped unconditionally because the MPS backend is not finished.
        self.verify_ops_handler_completeness(MetalOverrides)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
