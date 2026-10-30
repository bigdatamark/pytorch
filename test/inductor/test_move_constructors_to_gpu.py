# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the inductor pass that moves CPU tensor constructors (arange, ones, ...)
# onto the GPU when all of their consumers live on the GPU. Each test compiles a function and
# inspects the generated code for CPU fused kernels. The tests only run on Linux with a GPU and
# triton available; the main block does nothing otherwise.

import functools
import unittest

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    HAS_MULTIGPU,
)


requires_multigpu = functools.partial(
    unittest.skipIf, not HAS_MULTIGPU, f"requires multiple {GPU_TYPE} devices"
)


aten = torch.ops.aten


class TestMoveConstructorsToGpu(TestCase):
    # [CLASS DOC] Test case holding the constructor-movement checks; every test builds a small
    # function, compiles it with inductor and asserts whether CPU kernels remain in the output.
    def _check_fn(self, func, expect_cpu, *args):
        # [FUNCTION DOC] HELPER: Runs func eagerly and compiled on the given args, asserts equal
        # results, and that exactly one code string was generated. If expect_cpu is true the code
        # must contain a cpp_fused kernel, otherwise it must not. Raises on any mismatch.
        out_eager = func(*args)

        out_compiled, code = run_and_get_code(torch.compile(func), *args)
        self.assertEqual(out_eager, out_compiled)

        if len(code) != 1:
            raise AssertionError
        if expect_cpu:
            FileCheck().check("cpp_fused").run(code[0])
        else:
            FileCheck().check_not("cpp_fused").run(code[0])

    def test_simple(self):
        # [FUNCTION DOC] Indexing a GPU tensor with a CPU arange: the arange should be moved to
        # the GPU, so no CPU fused kernel is expected.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the input indexed along dim 0 by a CPU
            # arange of its length.
            return x[torch.arange(x.shape[0])]

        inp = torch.rand(32, 77, 512, device=GPU_TYPE)

        self._check_fn(foo, False, inp)

    def test_output_failure(self):
        # [FUNCTION DOC] When the arange is also returned as an output, it cannot be moved to the
        # GPU, so a CPU fused kernel is still expected.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the arange index tensor and the input indexed by it.
            tmp1 = torch.arange(x.shape[0])
            return tmp1, x[tmp1]

        inp = torch.rand(32, 77, 512, device=GPU_TYPE)

        self._check_fn(foo, True, inp)

    def test_non_convertable_op_failure(self):
        # [FUNCTION DOC] When the constructor result feeds an op that cannot be converted
        # (addition with a CPU input), the constructor stays on CPU; a CPU kernel is expected.
        def foo(x):
            # [FUNCTION DOC] HELPER: Adds a CPU arange to the input and also returns a ones tensor
            # created on the GPU device.
            y = torch.arange(x.shape[0])
            return x + y, torch.ones([4], device=GPU_TYPE)

        inp = torch.rand([100])

        self._check_fn(foo, True, inp)

    def test_multiple_constructors(self):
        # [FUNCTION DOC] Several arange constructors feeding indexing plus an add, on a CPU input:
        # the constructors remain on CPU and a CPU fused kernel is expected.
        def foo(x):
            # [FUNCTION DOC] HELPER: Indexes the input with two aranges (one reshaped to a row)
            # and returns both results and their sum.
            tmp1 = torch.arange(x.shape[0])
            o1 = x[tmp1]
            tmp2 = torch.arange(x.shape[1]).view([1, x.shape[1]])
            o2 = x[tmp2]
            return o1, o2, o1 + o2

        inp = torch.rand([200, 200])
        self._check_fn(foo, True, inp)

    def test_sets_equiv(self):
        # [FUNCTION DOC] Checks the result does not depend on the order in which two equivalent
        # constructors are defined: both compiled variants must produce code without a triton
        # kernel (the work is constant-folded) for a GPU-moved input.
        @torch.compile()
        def foo(x):
            # [FUNCTION DOC] HELPER: Compiled function creating a ones and an arange constructor
            # (ones first), indexing with their sum and returning the arange minus a constant.
            c1 = torch.ones([4], dtype=torch.long)
            c2 = torch.arange(-1, 3)
            return x[c1 + c2], c2 - 4 * 2

        inp = torch.rand([4]).to(GPU_TYPE)
        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

        @torch.compile()
        def foo(x):
            # [FUNCTION DOC] HELPER: Same as the previous variant but with the two constructors
            # created in swapped order.
            c2 = torch.arange(-1, 3)
            c1 = torch.ones([4], dtype=torch.long)
            return x[c1 + c2], c2 - 4 * 2

        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

    @requires_multigpu()
    @unittest.skip("https://github.com/pytorch/pytorch/issues/139520")
    def test_multi_gpu(self):
        # [FUNCTION DOC] Multi-GPU case: constructors targeting different GPU devices are not
        # supported yet, so a CPU kernel is expected. Skipped unconditionally (issue 139520) and
        # requires multiple GPUs.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns an indexed input plus ones tensors created on GPU 0
            # and GPU 1.
            return (
                x[torch.arange(x.shape[0])],
                torch.ones([4], device=f"{GPU_TYPE}:0"),
                torch.ones([4], device=f"{GPU_TYPE}:1"),
            )

        # nyi, multi-gpu
        inp = torch.rand([100], device=GPU_TYPE)
        self._check_fn(foo, True, inp)

    def test_no_gpu(self):
        # [FUNCTION DOC] With only CPU inputs, nothing is moved: a CPU kernel is expected.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the input indexed by a CPU arange.
            return x[torch.arange(x.shape[0])]

        inp = torch.rand([100])
        self._check_fn(foo, True, inp)

    def test_random_constructor_not_moved(self):
        # [FUNCTION DOC] Random constructors (randn and rand) must not be moved to the GPU: with
        # fallback_random enabled and the same seed, compiled and eager results must be equal.
        # Resets dynamo for each random function and patches the inductor config inside the loop.
        from torch._inductor import config

        for random_fn in [torch.randn, torch.rand]:
            torch._dynamo.reset()

            def foo(x, fn=random_fn):
                # [FUNCTION DOC] HELPER: Builds random values with the bound random function and a
                # CPU index tensor, moves both to the input device and returns the input after
                # adding the values at the indices along dim 0.
                values = fn(2, x.size(1)).to(x.device)
                indices = torch.tensor([0, 2], dtype=torch.long).to(x.device)
                return torch.index_add(x, 0, indices, values)

            inp = torch.randn(4, 8, device=GPU_TYPE)

            with config.patch(fallback_random=True):
                torch.manual_seed(0)
                out_eager = foo(inp)

                torch.manual_seed(0)
                out_compiled = torch.compile(foo)(inp)

            self.assertEqual(out_eager, out_compiled)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU_AND_TRITON:
        run_tests()
