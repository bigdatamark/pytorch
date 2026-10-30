# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for the pad_mm FX pass (padding matmul operands to aligned
# sizes). Contains one test class, PadMMTest, whose tests are mostly GPU-only. The test
# runner is invoked only when a GPU with Triton is available. No import-time side effects.
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.pad_mm import (
    can_pad,
    get_alignment_size,
    get_pad_cache,
    get_padded_length,
    should_pad_mm_bf16,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, is_big_gpu, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


class PadMMTest(TestCase):
    # [CLASS DOC] Tests for the pad_mm pass of Inductor: dynamic-shape padding of mm, bmm
    # and addmm, cache exclusion behavior, bf16 padding, kernel naming and stride
    # preservation. Skipped without a big GPU.
    def setUp(self):
        # [FUNCTION DOC] Runs the base setUp, then skips the test (via skipTest) when the
        # device is not a big GPU, since the max-autotune tests need one.
        super().setUp()
        if not is_big_gpu():
            return self.skipTest("Need a big GPU to run max_autotune=True")

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_mm_dyn_m(self):
        # [FUNCTION DOC] Tests padding of a mm whose left operand is a narrowed view with a
        # dynamic row count (dimension 0), with a fixed weight of non-aligned inner size.
        # Marks the chosen input dimensions dynamic, patches the module flag that skips
        # do_bench time comparison in pad_mm so padding is always judged profitable, and
        # compares eager and compiled results for equality. Also checks the generated code
        # for the padded K size computed from the alignment helper, via FileCheck.
        M = 40
        K1 = 581
        K2 = 49
        N = 30

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a fixed strided weight tensor and applying
            # a narrow on the input columns before a mm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the weight tensor with column-major strides
                # on the GPU device and stores it as a plain attribute.
                super().__init__()
                self.w = rand_strided(
                    (K2, N), (1, K2), device=GPU_TYPE, dtype=torch.float32
                )

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Narrows the input along dimension 1 to the weight's
                # inner size and returns the matrix product with the weight.
                a1 = torch.narrow(a, 1, 0, K2)
                return torch.mm(a1, self.w)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M, K1), (K1, 1), device=GPU_TYPE, dtype=torch.float32)
        aligned_k = get_padded_length(K2, get_alignment_size(a)) + K2
        torch._dynamo.mark_dynamic(a, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_cat_pad_mm_dyn_m(self):
        # [FUNCTION DOC] Tests padding when the mm input is a concatenation of two inputs
        # along dimension 0 followed by a narrow along columns, with both inputs dynamic in
        # dimension 0. Marks the chosen input dimensions dynamic, patches the module flag
        # that skips do_bench time comparison in pad_mm so padding is always judged
        # profitable, and compares eager and compiled results for equality. Also checks the
        # generated code for the padded K size.
        M1 = 128
        M2 = 40
        K1 = 129
        K2 = 111
        N = 100

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a fixed weight that concatenates two inputs
            # along rows, narrows the columns, and multiplies by the weight.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the column-major weight tensor on the GPU
                # device and stores it as an attribute.
                super().__init__()
                self.w = rand_strided(
                    (K2, N), (1, K2), device=GPU_TYPE, dtype=torch.float32
                )

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Concatenates the two inputs on dimension 0, narrows
                # dimension 1 to the weight's inner size, and returns the mm with the
                # weight.
                c = torch.cat([a, b], dim=0)
                a1 = torch.narrow(c, 1, 0, K2)
                return torch.mm(a1, self.w)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M1, K1), (K1, 1), device=GPU_TYPE, dtype=torch.float32)
        b = rand_strided((M2, K1), (K1, 1), device=GPU_TYPE, dtype=torch.float32)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        aligned_k = get_padded_length(K2, get_alignment_size(a)) + K2
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_mm_dyn_n(self):
        # [FUNCTION DOC] Tests padding of a plain mm where the second operand has a dynamic
        # column count (dimension 1). Marks the chosen input dimensions dynamic, patches the
        # module flag that skips do_bench time comparison in pad_mm so padding is always
        # judged profitable, and compares eager and compiled results for equality. Also
        # checks the generated code for the padded K size.
        M = 20
        K = 81
        N = 30

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a plain matrix multiplication of its two
            # inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base module initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
                return torch.mm(a, b)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M, K), (K, 1), device=GPU_TYPE, dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device=GPU_TYPE, dtype=torch.float32)
        aligned_k = get_padded_length(K, get_alignment_size(a)) + K
        torch._dynamo.mark_dynamic(b, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_mm_dyn_k(self):
        # [FUNCTION DOC] Tests a plain mm where the shared K dimension is dynamic in both
        # operands (dimension 1 of the first, dimension 0 of the second). The expected
        # padding is on M, computed from the first operand. Marks the chosen input
        # dimensions dynamic, patches the module flag that skips do_bench time comparison in
        # pad_mm so padding is always judged profitable, and compares eager and compiled
        # results for equality. Also checks the generated code for the padded M size.
        M = 21
        K = 80
        N = 30

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a plain matrix multiplication of its two
            # inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base module initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
                return torch.mm(a, b)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M, K), (K, 1), device=GPU_TYPE, dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device=GPU_TYPE, dtype=torch.float32)
        # TODO: Getting the alignment right requires pattern matcher to
        # run on newly added nodes
        aligned_m = get_padded_length(M, get_alignment_size(a)) + M
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"M = {aligned_m}").run(code)
        self.assertEqual(res1, res2)

    def test_pad_mm_dyn_mnk(self):
        # [FUNCTION DOC] Smoke test with all four mm dimensions marked dynamic and no forced
        # padding config; only checks that the compiled result equals eager with the
        # benchmark-skip flag patched. Does not inspect generated code.
        M = 20
        K = 81
        N = 30

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a plain matrix multiplication of its two
            # inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base module initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
                return torch.mm(a, b)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M, K), (K, 1), device=GPU_TYPE, dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device=GPU_TYPE, dtype=torch.float32)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(b, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (_,) = run_and_get_code(compiled_fn, a, b)
        self.assertEqual(res1, res2)

    @inductor_config.patch(force_shape_pad=True)
    def test_zero_dim(self):
        # [FUNCTION DOC] Checks that a compiled addmm with a zero-row operand (forced shape
        # padding) gives the same result as eager.
        def addmm(x, a, b):
            # [FUNCTION DOC] HELPER: Eager reference function returning the addmm of bias,
            # first and second matrix.
            return torch.addmm(x, a, b)

        x = torch.randn(100).to(GPU_TYPE)
        a = torch.randn(0, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)
        self.assertEqual(torch.compile(addmm)(x, a, b), addmm(x, a, b))

    @fresh_cache()
    @inductor_config.patch(shape_padding=True)
    @unittest.skipIf(
        GPU_TYPE != "cuda",
        "CUDA eager ignores addmm input shape when beta=0",
    )
    def test_addmm_beta_zero_mismatched_bias_skips_padding(self):
        # [FUNCTION DOC] Checks that padding is skipped, without error, for addmm with beta
        # zero and a bias whose shape does not match the output (the test only runs on
        # CUDA). Patches the compute-bound check to always say true and the benchmark getter
        # to return a fake timer, compiles with fullgraph, and compares with eager.
        def addmm(bias, x, weight):
            # [FUNCTION DOC] HELPER: Eager function computing addmm with beta zero and alpha
            # 0.1.
            return torch.addmm(bias, x, weight, beta=0.0, alpha=0.1)

        def bench(fn):
            # [FUNCTION DOC] HELPER: Fake benchmark function: runs the given callable once
            # and returns a constant time of 1.0.
            fn()
            return 1.0

        bias = torch.zeros(8, device=GPU_TYPE)
        x = torch.randn(2, 8, device=GPU_TYPE)
        weight = torch.randn(8, 13, device=GPU_TYPE)

        with (
            unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound",
                return_value=True,
            ),
            unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_do_bench",
                return_value=bench,
            ),
        ):
            actual = torch.compile(addmm, fullgraph=True)(bias, x, weight)
        self.assertEqual(actual, addmm(bias, x, weight))

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_bmm_dyn_b(self):
        # [FUNCTION DOC] Tests padding of a batched mm where the batch dimension (dimension
        # 0) of both inputs is dynamic. Marks the chosen input dimensions dynamic, patches
        # the module flag that skips do_bench time comparison in pad_mm so padding is always
        # judged profitable, and compares eager and compiled results for equality. Also
        # checks the generated code for the padded K size.
        B = 10
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a batched matrix multiplication of its two
            # inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base module initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the batched matrix product of the two
                # inputs.
                return torch.bmm(a, b)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(B, M, K, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(B, K, N, device=GPU_TYPE, dtype=torch.float32)
        aligned_k = get_padded_length(K, get_alignment_size(a)) + K
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_bmm_dyn_k(self):
        # [FUNCTION DOC] Tests padding of a batched mm where the K dimension is dynamic in
        # both inputs (dimension 2 of the first, dimension 1 of the second). The expected
        # padding is on N. Marks the chosen input dimensions dynamic, patches the module
        # flag that skips do_bench time comparison in pad_mm so padding is always judged
        # profitable, and compares eager and compiled results for equality. Also checks the
        # generated code for the padded N size.
        B = 10
        M = 128
        K = 40
        N = 41

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a batched matrix multiplication of its two
            # inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base module initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the batched matrix product of the two
                # inputs.
                return torch.bmm(a, b)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(B, M, K, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(B, K, N, device=GPU_TYPE, dtype=torch.float32)
        aligned_n = get_padded_length(N, get_alignment_size(b)) + N
        torch._dynamo.mark_dynamic(a, 2)
        torch._dynamo.mark_dynamic(b, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"N = {aligned_n}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_bmm_dyn_bm(self):
        # [FUNCTION DOC] Tests padding of a batched mm where batch and M are dynamic in the
        # first input and batch is dynamic in the second. The expected padding is on N.
        # Marks the chosen input dimensions dynamic, patches the module flag that skips
        # do_bench time comparison in pad_mm so padding is always judged profitable, and
        # compares eager and compiled results for equality. Also checks the generated code
        # for the padded N size.
        B = 10
        M = 128
        K = 40
        N = 41

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a batched matrix multiplication of its two
            # inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base module initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the batched matrix product of the two
                # inputs.
                return torch.bmm(a, b)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(B, M, K, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(B, K, N, device=GPU_TYPE, dtype=torch.float32)
        aligned_n = get_padded_length(N, get_alignment_size(b)) + N
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"N = {aligned_n}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_addmm_dyn_m(self):
        # [FUNCTION DOC] Tests padding of an addmm where M is dynamic in the bias and first
        # matrix. Marks the chosen input dimensions dynamic, patches the module flag that
        # skips do_bench time comparison in pad_mm so padding is always judged profitable,
        # and compares eager and compiled results for equality. Also checks the generated
        # code for the padded K size.
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an addmm of its three inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base module initializer.
                super().__init__()

            def forward(self, a, b, c):
                # [FUNCTION DOC] HELPER: Returns the addmm of bias, first and second matrix.
                return torch.addmm(a, b, c)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(M, N, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float32)
        c = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float32)
        aligned_k = get_padded_length(K, get_alignment_size(b)) + K
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b, c)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b, c)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_addmm_dyn_mn(self):
        # [FUNCTION DOC] Tests an addmm with M and N dynamic. Same setup as the dynamic-M
        # test, but the expectation is that no padding happens: the generated code must
        # still show the original K size. Also compares compiled and eager results with the
        # benchmark-skip flag patched.
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an addmm of its three inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base module initializer.
                super().__init__()

            def forward(self, a, b, c):
                # [FUNCTION DOC] HELPER: Returns the addmm of bias, first and second matrix.
                return torch.addmm(a, b, c)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(M, N, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float32)
        c = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float32)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(c, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b, c)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b, c)
            # no padding
            FileCheck().check(f"K = {K}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(force_shape_pad=True)
    def test_pad_single_cat(self):
        # [FUNCTION DOC] Checks that a compiled matrix product of two small square inputs,
        # with forced padding, matches the eager result.
        @torch.compile()
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
            return x @ y

        inps = [torch.rand([5, 5], device=GPU_TYPE) for _ in range(2)]
        out = foo(*inps)
        self.assertEqual(out, inps[0] @ inps[1])

    @inductor_config.patch(force_shape_pad=True)
    @fresh_cache()
    def test_pad_addmm_2d_bias(self):
        # [FUNCTION DOC] Tests that padded addmm gives eager-equal results for 2D biases of
        # shapes combining sizes 1 and 4 by 1 and 6, and for 1D biases of size 1 and 6,
        # using a fresh cache and forced padding. Compares against the aten addmm op.
        @torch.compile()
        def foo(input, x, y):
            # [FUNCTION DOC] HELPER: Returns the aten addmm of bias, first and second
            # matrix.
            return torch.ops.aten.addmm(input, x, y)

        for a in [1, 4]:
            for b in [1, 6]:
                inps = (
                    torch.rand([a, b], device=GPU_TYPE),
                    torch.rand([4, 5], device=GPU_TYPE),
                    torch.rand([5, 6], device=GPU_TYPE),
                )
                out = foo(*inps)
                out_eager = torch.ops.aten.addmm(*inps)
                self.assertEqual(out, out_eager)

        for a in [1, 6]:
            inps = (
                torch.rand([a], device=GPU_TYPE),
                torch.rand([4, 5], device=GPU_TYPE),
                torch.rand([5, 6], device=GPU_TYPE),
            )
            out = foo(*inps)
            out_eager = torch.ops.aten.addmm(*inps)
            self.assertEqual(out, out_eager)

    @inductor_config.patch(force_shape_pad=True)
    def test_pad_batch(self):
        # [FUNCTION DOC] Tests that bmm on float16 inputs gets padded. Verifies the
        # alignment size is 8 and that the common padding criteria hold (raising
        # AssertionError otherwise), compiles, and checks generated code has exactly two
        # kernel runs and a padded output allocation. Raises AssertionError if the result is
        # not close to eager.
        m = 6
        n = 9
        k = 11
        batch_size = 3
        mat1 = torch.ones((batch_size, m, k), device=GPU_TYPE, dtype=torch.float16)
        mat2 = torch.ones((batch_size, k, n), device=GPU_TYPE, dtype=torch.float16)
        expected_alignment = get_alignment_size(mat1)

        if expected_alignment != 8:
            raise AssertionError("Alignment for float16 should be 8")
        if not can_pad(mat1, mat2, torch.ops.aten.bmm):
            raise AssertionError("This should pass the common padding criteria")

        @torch.compile()
        def bmm(mat1, mat2):
            # [FUNCTION DOC] HELPER: Returns the batched matrix product of the two inputs.
            return torch.bmm(mat1, mat2)

        res2, (code,) = run_and_get_code(bmm, mat1, mat2)
        bmm_expected_result = torch.bmm(mat1, mat2)
        # in call code, expect to see a single pad per input, and then we should see padded allocation for output
        FileCheck().check("del async_compile").check_count(
            ".run(", 2, exactly=True
        ).check(f"empty_strided_{GPU_TYPE}((3, 8, 16)").run(code)

        if not torch.allclose(res2, bmm_expected_result):
            raise AssertionError("BMM results are not identical")

    @fresh_cache()
    def test_exclude_padding(self):
        # [FUNCTION DOC] Tests the exclude-pad cache keys: compiles a mm on 61 by 61 inputs
        # with the compute-bound check and benchmarker patched, and expects two entries in
        # the local pad cache, both with exclude_pad false. A second compiled function adds
        # one to the first input; it should add exactly one new entry (three total) with
        # exclude_pad true for one of them, reusing the original base timing.
        def bench(fn):
            # [FUNCTION DOC] HELPER: Fake benchmark function: runs the callable once and
            # returns a constant time of 1.0.
            fn()
            return 1.0

        @torch.compile()
        def mm(a, b):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
            return a @ b

        # Size must require padding to 4 elements; the compute-bound heuristic is
        # patched below so the cache-key assertions do not depend on GPU model.
        size = [61, 61]
        with (
            unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound",
                return_value=True,
            ),
            unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_do_bench",
                return_value=bench,
            ),
        ):
            mm(torch.rand(size, device=GPU_TYPE), torch.rand(size, device=GPU_TYPE))
            local_cache = get_pad_cache().get_local_cache()
            self.assertEqual(len(local_cache), 2)
            FileCheck().check_count("exclude_pad:False", 2, exactly=True).run(
                repr(local_cache)
            )

            @torch.compile()
            def mm(a, b):
                # [FUNCTION DOC] HELPER: Returns the matrix product of the first input plus
                # one with the second input.
                return (a + 1) @ b

            mm(torch.rand(size, device=GPU_TYPE), torch.rand(size, device=GPU_TYPE))
            local_cache = get_pad_cache().get_local_cache()
            # reuse original base timing
            self.assertEqual(len(local_cache), 3)

            FileCheck().check_count("exclude_pad:False", 3, exactly=True).run(
                repr(local_cache)
            )
            FileCheck().check_count("exclude_pad:True", 1, exactly=True).run(
                repr(local_cache)
            )

    @fresh_cache()
    @inductor_config.patch(max_pointwise_cat_inputs=2)
    def test_exclude_cat_padding(self):
        # [FUNCTION DOC] Tests that a concatenation feeding a mm is treated as excluded from
        # padding at the right times. With the maximum pointwise cat inputs limited to 2, a
        # three-chunk concatenation gives two cache entries with exclude_pad false, and a
        # two-chunk concatenation raises that count to three.
        @torch.compile()
        def mm(inps, b):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the concatenation of the
            # input list with the second argument.
            return torch.cat(inps) @ b

        inp = torch.rand([2046, 2046], device=GPU_TYPE)
        inp2 = torch.rand([2046, 2046], device=GPU_TYPE)

        inps = inp.chunk(3)
        mm(inps, inp2)
        FileCheck().check_count("exclude_pad:False", 2, exactly=True).run(
            repr(get_pad_cache().get_local_cache())
        )

        inps = inp.chunk(2)
        mm(inps, inp2)
        FileCheck().check_count("exclude_pad:False", 3, exactly=True).run(
            repr(get_pad_cache().get_local_cache())
        )

    @unittest.skipIf(
        (not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (9, 0))
        and (not torch.xpu.is_available()),
        "No perf regression on H100+ with BF16",
    )
    @fresh_cache()
    @inductor_config.patch(
        post_grad_fusion_options={"pad_aten_mm_pass": {"k_threshold_to_pad": 8388608}}
    )
    def test_pad_mm_bf16(self):
        # [FUNCTION DOC] Tests bf16 mm padding with a very large K on pre-Hopper CUDA or XPU
        # only, with a K threshold config of 8388608. Verifies alignment size 8, can_pad and
        # should_pad_mm_bf16 (raising AssertionError if any fails), compiles, checks
        # generated code for two kernel runs and a padded output allocation, and raises
        # AssertionError if the result is not close to eager.
        m = 2
        n = 13
        k = 15691904
        mat1 = torch.ones((m, k), device=GPU_TYPE, dtype=torch.bfloat16)
        mat2 = torch.ones((k, n), device=GPU_TYPE, dtype=torch.bfloat16)
        expected_alignment = get_alignment_size(mat1)

        if expected_alignment != 8:
            raise AssertionError("Alignment for bfloat16 should be 8")
        if not can_pad(mat1, mat2, torch.ops.aten.mm):
            raise AssertionError("This should pass the common padding criteria")
        if not should_pad_mm_bf16(mat1.dtype, m, n, k):
            raise AssertionError(
                "This should pass the should_pad_mm_bf16 padding criteria"
            )

        @torch.compile()
        def mm(mat1, mat2):
            # [FUNCTION DOC] HELPER: Returns the mm of the two inputs.
            return torch.mm(mat1, mat2)

        res2, (code,) = run_and_get_code(mm, mat1, mat2)
        mm_expected_result = torch.mm(mat1, mat2)
        # in call code, expect to see a single pad per input, and then we should see padded allocation for output
        FileCheck().check("del async_compile").check_count(
            ".run(", 2, exactly=True
        ).check(f"empty_strided_{GPU_TYPE}((8, 16)").run(code)

        if not torch.allclose(res2, mm_expected_result):
            raise AssertionError("MM results are not identical")

    @fresh_cache()
    @inductor_config.patch(
        {
            "triton.unique_kernel_names": "original_aten",
            "max_autotune_gemm_backends": "TRITON",
            # Force padding so it is applied regardless of the device-dependent
            # is_mm_compute_bound checks in should_pad_common.
            "force_shape_pad": True,
        }
    )
    def test_original_aten_preserved_pad_mm(self):
        # [FUNCTION DOC] Checks that with kernel names derived from the original aten op,
        # forced padding, and the Triton-only GEMM backend, the padded mm kernel template is
        # still named after mm. Compiles in max-autotune mode with benchmark times skipped,
        # expects one pattern match count (except on ROCm), drops code from decompose-K
        # variants, and checks the first remaining code for the fused mm template
        # definition.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
            return x @ y

        args = [
            torch.randn(2**4, 2**8 - 1, device=GPU_TYPE, dtype=torch.float16),
            torch.randn(2**8 - 1, 2**4, device=GPU_TYPE, dtype=torch.float16),
        ]

        counters.clear()

        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            opt_fn = torch.compile(fn, mode="max-autotune")
            ret, code = run_and_get_code(opt_fn, *args)
        # xref: https://github.com/pytorch/pytorch/pull/172780
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)

        code = [c for c in code if "decompose_k" not in c]
        # The mm kernel should use a template (because we set max_autotune_gemm_backends = TRITON).
        # Its name should contain `mm` because `mm` was the original aten op where the mm came from.
        FileCheck().check("def triton_tem_fused_mm").run(code[0])

    def test_no_autocast_in_pad_bmm_joint_graph_pass(self):
        # [FUNCTION DOC] Regression test for an issue where autocast in the pad bmm joint
        # graph pass changed bmm output dtypes and produced NaN gradients. Builds a masked
        # multi-head attention model, compiles it with custom joint pre and post passes that
        # record bmm output dtypes, runs forward under autocast with backward, then checks
        # no bmm dtype changed and neither input gradient contains NaN. Runs on the GPU
        # device in bfloat16 with fixed seed after a compiler reset.
        # Track bmm dtypes before and after joint graph passes
        bmm_dtypes_pre = {}
        bmm_dtypes_post = {}

        def make_bmm_dtype_tracker(dtype_dict):
            # [FUNCTION DOC] HELPER: Returns a graph pass function that records, into the
            # given mapping, the output dtype of every bmm node (keyed by node name) when
            # the node has a value with a dtype in its meta, and returns the graph
            # unchanged.
            def track_bmm_dtype(graph):
                # [FUNCTION DOC] HELPER: Graph pass: loops over nodes, stores dtype of each
                # bmm node's fake value in the enclosing mapping, returns the same graph.
                for node in graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.bmm.default
                    ):
                        # Store the output dtype
                        if hasattr(node.meta.get("val", None), "dtype"):
                            dtype_dict[str(node)] = node.meta["val"].dtype
                return graph

            return track_bmm_dtype

        class MaskedMHA(torch.nn.Module):
            # [CLASS DOC] HELPER: Attention module with separate video and text qkv linear
            # projections and an output projection. Stores kv head count, query head count
            # and head dimension.
            def __init__(self, H_q, H_kv, D):
                # [FUNCTION DOC] HELPER: Creates two linear layers projecting to query plus
                # key plus value heads and an output linear layer, and saves the head counts
                # and head dimension.
                super().__init__()
                self.H_kv = H_kv
                num_heads_total = H_q + 2 * H_kv
                self.qkv_proj_vid = torch.nn.Linear(H_q * D, num_heads_total * D)
                self.qkv_proj_txt = torch.nn.Linear(H_q * D, num_heads_total * D)
                self.out_proj = torch.nn.Linear(H_q * D, H_q * D)
                self.H_q = H_q
                self.D = D

            def forward(self, x_vid, x_txt, attn_mask):
                # [FUNCTION DOC] HELPER: Projects video and text inputs, splits each into
                # query, key and value heads, concatenates video and text parts along the
                # sequence-head dimension, runs scaled dot product attention with the mask
                # and grouped-query support, and returns the output with sequence and head
                # dimensions swapped back.
                qkv_vid = self.qkv_proj_vid(x_vid)
                qkv_txt = self.qkv_proj_txt(x_txt)
                qkv_vid = qkv_vid.reshape((*qkv_vid.shape[:-1], -1, self.D))
                qkv_txt = qkv_txt.reshape((*qkv_txt.shape[:-1], -1, self.D))

                q_vid = qkv_vid[..., : self.H_q, :]
                k_vid = qkv_vid[..., self.H_q : self.H_q + self.H_kv, :]
                v_vid = qkv_vid[..., self.H_q + self.H_kv :, :]

                q_txt = qkv_txt[..., : self.H_q, :]
                k_txt = qkv_txt[..., self.H_q : self.H_q + self.H_kv, :]
                v_txt = qkv_txt[..., self.H_q + self.H_kv :, :]

                q = torch.cat([q_vid, q_txt], dim=-3)
                k = torch.cat([k_vid, k_txt], dim=-3)
                v = torch.cat([v_vid, v_txt], dim=-3)

                out = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(-2, -3),
                    k.transpose(-2, -3),
                    v.transpose(-2, -3),
                    attn_mask=attn_mask,
                    enable_gqa=True,
                )
                out = out.transpose(-2, -3)

                return out

        def test_masked_mha(B, H, S, D, device, dtype):
            # [FUNCTION DOC] HELPER: Runs the compiled masked attention model under autocast
            # on random inputs of the given sizes (video length fixed at 300), performs
            # backward on the mean of a random-target difference, synchronizes, asserts that
            # every recorded bmm output dtype is equal before and after the joint passes,
            # and that both input gradients have no NaN.
            S_vid = 300
            S_txt = S - S_vid
            x1 = torch.randn(B, S_vid, H * D, requires_grad=True, device=device)
            x2 = torch.randn(B, S_txt, H * D, requires_grad=True, device=device)
            attn_mask = torch.ones(B, 1, S, S, dtype=torch.bool, device=device)

            H_kv = H // 4
            mha = MaskedMHA(H, H_kv, D)
            mha = mha.to(device)

            with torch._inductor.config.patch(
                joint_custom_pre_pass=make_bmm_dtype_tracker(bmm_dtypes_pre),
                joint_custom_post_pass=make_bmm_dtype_tracker(bmm_dtypes_post),
            ):
                mha = torch.compile(mha, fullgraph=True, backend="inductor")
                with torch.autocast(
                    device_type=GPU_TYPE, dtype=dtype, cache_enabled=False
                ):
                    out_vid = mha(x1, x2, attn_mask)
                    target_vid = torch.randn_like(out_vid)

                    loss_vid = (out_vid - target_vid).mean()
                    loss = loss_vid
                loss.backward()

            torch.accelerator.synchronize()

            # Check if any bmm operations had dtype changes
            for node_name_pre, node_name_post in zip(
                bmm_dtypes_pre, bmm_dtypes_post, strict=True
            ):
                pre_dtype = bmm_dtypes_pre[node_name_pre]
                post_dtype = bmm_dtypes_post[node_name_post]
                # Assert no bmm output dtype changes
                self.assertEqual(pre_dtype, post_dtype)

            # Based on issue https://github.com/pytorch/pytorch/issues/159469,
            # if autocast was applied in pad_bmm causing bmm's output dtype to be changed from fp32 to bf16,
            # gradient will have NaNs in this test case.
            self.assertFalse(torch.any(x1.grad.isnan()).item())
            self.assertFalse(torch.any(x2.grad.isnan()).item())

        B, H, S, D = 2, 32, 549, 128
        device = GPU_TYPE
        dtype = torch.bfloat16
        torch.compiler.reset()
        torch.manual_seed(42)
        test_masked_mha(B, H, S, D, device, dtype)

    @inductor_config.patch(force_shape_pad=True, strict_output_strides=True)
    def test_pad_mm_output_strides_preserved(self):
        # [FUNCTION DOC] Regression test: with forced shape padding and strict output
        # strides, the compiled mm output must have the same values and strides as eager
        # even though padding makes the output a sliced view of a wider result.
        """Regression test: pad_mm creates views with padded strides.
        User-visible output strides must match eager execution."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
            return torch.mm(x, y)

        # N=2 gets padded to 4, so the mm output is (3, 4) sliced to (3, 2),
        # creating a view with stride (4, 1) instead of the expected (2, 1).
        x = torch.randn(3, 5, device=GPU_TYPE)
        y = torch.randn(5, 2, device=GPU_TYPE)
        expected = fn(x, y)
        compiled = torch.compile(fn)(x, y)
        self.assertEqual(compiled, expected)
        self.assertEqual(compiled.stride(), expected.stride())


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON:
        run_tests()
