# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor test checking the Triton code generated for the grid_sampler_2d operator.
# Verifies on CUDA that compiled results match eager and that the generated kernels compute
# coordinate indices in 32-bit integers. Import time: only imports; the test is skipped when CUDA
# and Triton are unavailable. Run as a script, it calls the inductor run_tests with the sympy
# argument.
import unittest

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class GridSamplerCodegenTests(InductorTestCase):
    # [CLASS DOC] Inductor test case holding the single grid sampler code generation test.
    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    def test_grid_sampler_2d_uses_32bit_coordinate_indices(self):
        # [FUNCTION DOC] Compiles a function calling the 2D grid sampler (bilinear mode, zero
        # padding, align_corners off) with inductor on random CUDA float inputs (input 2x3x16x16,
        # grid 2x16x16x2 in the range minus one to one). Checks the output is close to eager
        # within tight tolerances, then asserts the generated source contains a cast to the 32-bit
        # integer type and never mentions the 64-bit integer type. Skipped without CUDA and
        # Triton.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Eager reference body: calls the aten 2D grid sampler on the
            # two inputs with interpolation mode 0, padding mode 0 and align_corners False.
            # Returns the sampled tensor.
            return torch.ops.aten.grid_sampler_2d(a, b, 0, 0, False)

        a = torch.randn([2, 3, 16, 16], dtype=torch.float32, device="cuda")
        b = torch.rand([2, 16, 16, 2], dtype=torch.float32, device="cuda") * 2 - 1

        expected = fn(a, b)
        actual, codes = run_and_get_code(torch.compile(fn, backend="inductor"), a, b)

        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        source_code = "\n".join(codes)
        self.assertIn(".to(tl.int32)", source_code)
        self.assertNotIn("tl.int64", source_code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests("sympy")
