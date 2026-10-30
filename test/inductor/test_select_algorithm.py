# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor's algorithm selection and max-autotune machinery
# (select_algorithm module): choice callers, matmul/conv template autotuning, template
# kernels, kernel-input handling, caches and benchmark cleanup. Most tests compile small GPU
# functions under a decorator that enables max-autotune and verification, then check the
# autotune counter. Classes are mostly gated on a big GPU and Triton. The file defines an aten
# alias and a decorator helper; there are no import-time side effects beyond imports.
import contextlib
import functools
import unittest.mock
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
import torch.nn.functional as F
from torch._dynamo.testing import expectedFailureDynamicWrapper
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.autotune_process import (
    ExternKernelGPUBenchmarkRequest,
    TensorMeta,
    TritonBenchmarkRequest,
)
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import FixedLayout, ShapeAsConstantBuffer
from torch._inductor.kernel_inputs import KernelInputs
from torch._inductor.runtime.triton_compat import Config as TritonConfig
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternalTritonTemplateKernel,
    ExternKernelChoice,
    PartialRender,
    TritonTemplate,
    TritonTemplateKernel,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_big_gpu, run_and_get_code, run_and_get_kernels
from torch._inductor.virtualized import V
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    IS_LINUX,
    MI200_ARCH,
    skipIfRocmArch,
    skipIfXpu,
    TEST_WITH_ROCM,
    TEST_XPU,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    IS_H100,
    requires_gpu,
    requires_triton,
)


aten = torch.ops.aten


def patches(fn):
    # [FUNCTION DOC] HELPER: Decorator factory used by the autotune tests. Wraps the given
    # test function with config patches (dynamo verbose, inductor debug with max-autotune and
    # epilogue fusion enabled), a tight verification tolerance for autotuned choices, a lookup
    # replacement that bypasses the autotune cache, and disabled TF32 for cudnn. Returns a
    # wrapper that clears counters, seeds the random generator with 12345, raises
    # AssertionError if fp32 matmul precision is tf32, then calls the patched test.
    def skip_cache(self, choices, name, key, benchmark, hint_override=None):
        # [FUNCTION DOC] HELPER: Replacement for the selector cache lookup. Ignores the cache:
        # returns an empty mapping when no benchmark function is given, otherwise runs the
        # benchmark on the choices and returns its result.
        if benchmark is None:
            return {}
        return benchmark(choices)

    for patcher in [
        dynamo_config.patch(verbose=True),
        inductor_config.patch(debug=True, max_autotune=True, epilogue_fusion=True),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
        torch.backends.cudnn.flags(allow_tf32=False),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper executed in place of the test: clears dynamo
        # counters, seeds the random generator, raises AssertionError when TF32 matmul
        # precision is active, and returns the result of the patched test function called with
        # the same arguments.
        counters.clear()
        torch.manual_seed(12345)
        if torch.backends.cuda.matmul.fp32_precision == "tf32":
            raise AssertionError("correctness testing is allergic to tf32")
        return fn(*args, **kwargs)

    return wrapped


class TestAlgorithmSelectorChoiceTypes(TestCase):
    # [CLASS DOC] Tests that the algorithm selector prefers and classifies extern kernel
    # choice callers correctly, and that realizing a symbolic integer input keeps it as a
    # shape constant. Needs no GPU.
    def _extern_kernel_caller(self, name):
        # [FUNCTION DOC] HELPER: Looks up a registered extern kernel choice by name (failing
        # the test if it is missing) and returns it bound with no input nodes and no layout,
        # giving a choice caller.
        choice = ExternKernelChoice.lookup(name)
        if choice is None:
            self.fail(f"missing extern kernel choice: {name}")
        return choice.bind(input_nodes=[], layout=None)

    def test_pick_deterministic_choice_prefers_extern_kernel_caller(self):
        # [FUNCTION DOC] Checks that the deterministic-choice picker, given a plain choice
        # caller followed by the extern kernel caller for mm, returns the extern kernel caller
        # rather than the first one.
        first_choice = select_algorithm.ChoiceCaller("not_extern", [], None, "")
        extern_choice = self._extern_kernel_caller("mm")

        picked = select_algorithm.AlgorithmSelectorCache().pick_deterministic_choice(
            [first_choice, extern_choice]
        )

        self.assertIs(picked, extern_choice)

    def test_classify_kernel_operation_uses_extern_kernel_caller_name(self):
        # [FUNCTION DOC] Checks the kernel-operation classifier derives its category from the
        # extern kernel caller name: the mm and addmm callers classify as mm and the
        # convolution caller as conv, even though the passed name is unclassified. Runs each
        # case as a subtest.
        cases = (
            ("mm", "mm"),
            ("addmm", "mm"),
            ("convolution", "conv"),
        )
        for choice_name, expected in cases:
            with self.subTest(choice_name=choice_name):
                extern_choice = self._extern_kernel_caller(choice_name)

                self.assertEqual(
                    select_algorithm._classify_kernel_operation(
                        "unclassified_name", [extern_choice], []
                    ),
                    expected,
                )

    def test_realize_inputs_preserves_shape_constant(self):
        # [FUNCTION DOC] Checks that realizing a symbolic integer constant returns a shape-as-
        # constant buffer whose expression equals the original integer 2048.
        import sympy

        out = select_algorithm.realize_inputs(sympy.Integer(2048))

        self.assertIsInstance(out, ShapeAsConstantBuffer)
        self.assertEqual(out.expr, sympy.Integer(2048))


class TestSelectAlgorithm(TestCase):
    # [CLASS DOC] Main max-autotune test case: each test compiles a small function under the
    # autotune patches and checks how many times the algorithm autotuner ran (skipped on ROCm
    # where autotuning is not guaranteed). Fixture setup skips the test without a big GPU and
    # clears preprocessing functions; teardown resets the choices handler and the
    # preprocessing functions.
    def setUp(self):
        # [FUNCTION DOC] Runs the parent setup; skips the test when no big GPU is available
        # (autotuning needs one); otherwise clears registered preprocessing functions so tests
        # start clean.
        super().setUp()
        if not is_big_gpu():
            return self.skipTest("Need a big GPU to run max_autotune=True")
        # Clear preprocessing functions to ensure clean state
        select_algorithm.clear_preprocessing_fns()

    def tearDown(self):
        # [FUNCTION DOC] Runs the parent teardown, then resets the global choices handler to
        # none and clears registered preprocessing functions.
        super().tearDown()
        V.choices_handler = None
        select_algorithm.clear_preprocessing_fns()

    @patches
    def test_linear_relu(self):
        # [FUNCTION DOC] Compiles relu applied to a linear layer on GPU inputs and, off ROCm,
        # expects exactly one autotune run.
        @torch.compile
        def foo(input, weight, bias):
            # [FUNCTION DOC] HELPER: Compiled function applying relu to a linear layer built
            # from input, weight and bias.
            return F.relu(F.linear(input, weight, bias))

        foo(
            torch.randn(64, 32, device=GPU_TYPE),
            torch.randn(16, 32, device=GPU_TYPE),
            torch.randn(1, 16, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

        # It would be nice to assert this got fused into a single kernel, but that
        # only happens if we select a triton template (and not aten).

    @patches
    def test_addmm(self):
        # [FUNCTION DOC] Compiles an addmm on GPU inputs and, off ROCm, expects exactly one
        # autotune run.
        @torch.compile
        def foo(input, weight, bias):
            # [FUNCTION DOC] HELPER: Compiled function returning bias plus the product of
            # input and weight via addmm.
            return torch.addmm(bias, input, weight)

        inps = (
            torch.randn(20, 33, device=GPU_TYPE),
            torch.randn(33, 16, device=GPU_TYPE),
            torch.randn(20, 16, device=GPU_TYPE),
        )

        foo(*inps)
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_preprocessing_single_choice(self):
        # [FUNCTION DOC] Registers a preprocessing function that keeps only the first choice
        # (and records that it was called), then compiles an addmm. Off ROCm, expects the
        # autotuner counter to stay at zero because a single choice skips autotuning, and that
        # the preprocessing function was invoked.
        # pass a list to the preprocessing function to assert that it was
        # actually called
        func_called = [False]

        # Register a preprocessing function that returns only the first choice
        # This in turn will lead to autotuning being skipped as it's a single
        # choice, and the counter itself will not be bumped
        def return_first_choice_only(choices):
            # [FUNCTION DOC] HELPER: Preprocessing function marking itself as called and
            # returning only the first choice, or an empty list if there are none.
            func_called[0] = True
            return choices[:1] if choices else []

        select_algorithm.add_preprocessing_fn(return_first_choice_only)

        @torch.compile
        def foo(input, weight, bias):
            # [FUNCTION DOC] HELPER: Compiled function returning bias plus the product of
            # input and weight via addmm.
            return torch.addmm(bias, input, weight)

        inps = (
            torch.randn(20, 33, device=GPU_TYPE),
            torch.randn(33, 16, device=GPU_TYPE),
            torch.randn(20, 16, device=GPU_TYPE),
        )

        foo(*inps)
        # Since we only have one choice, autotuning should be skipped
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)
            # The preprocessing function should have been called
            self.assertTrue(func_called[0])

    @patch.object(select_algorithm, "VERIFY", dict(atol=5e-2, rtol=5e-2))
    @patches
    def test_addmm_fp16(self):
        # [FUNCTION DOC] Compiles an addmm with half-precision inputs (transposed weight,
        # uninitialized bias), using relaxed verification tolerances, and off ROCm expects one
        # autotune run.
        @torch.compile
        def foo(input, weight, bias):
            # [FUNCTION DOC] HELPER: Compiled function returning bias plus the product of
            # input and weight via addmm.
            return torch.addmm(bias, input, weight)

        inps = (
            torch.randn(2, 320, device=GPU_TYPE, dtype=torch.half),
            torch.randn(320, 320, device=GPU_TYPE, dtype=torch.half).t(),
            torch.empty(320, device=GPU_TYPE, dtype=torch.half),
        )

        foo(*inps)
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm(self):
        # [FUNCTION DOC] Compiles a plain matrix multiply of 8 by 32 and 32 by 8 inputs and,
        # off ROCm, expects one autotune run.
        @torch.compile
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Compiled function returning the matrix product of its
            # inputs.
            return torch.mm(a, b)

        foo(
            torch.randn(8, 32, device=GPU_TYPE),
            torch.randn(32, 8, device=GPU_TYPE),
        )
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test__int_mm(self):
        # [FUNCTION DOC] Compiles an integer matrix multiply on int8 inputs and, off ROCm,
        # expects one autotune run.
        @torch.compile
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Compiled function returning the integer matrix product of
            # two int8 matrices.
            return torch._int_mm(a, b)

        foo(
            torch.randint(-10, 10, (64, 32), device=GPU_TYPE, dtype=torch.int8),
            torch.randint(-10, 10, (32, 64), device=GPU_TYPE, dtype=torch.int8),
        )
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_skip(self):
        # [FUNCTION DOC] Compiles a matrix multiply of float64 inputs and, off ROCm, expects
        # the autotune counter to stay at zero because the Triton dot does not support
        # float64.
        @torch.compile
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Compiled function returning the matrix product of its
            # inputs.
            return torch.mm(a, b)

        foo(
            torch.randn(8, 32, device=GPU_TYPE, dtype=torch.float64),
            torch.randn(32, 8, device=GPU_TYPE, dtype=torch.float64),
        )
        # float64 not supported by tl.dot()
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)

    @patches
    def test_bmm(self):
        # [FUNCTION DOC] Compiles a batched matrix multiply of shapes 2x8x32 and 2x32x8 and,
        # off ROCm, expects one autotune run.
        @torch.compile
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Compiled function returning the batched matrix product of
            # its inputs.
            return torch.bmm(a, b)

        foo(
            torch.randn(2, 8, 32, device=GPU_TYPE),
            torch.randn(2, 32, 8, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_bmm_small_m(self):
        # [FUNCTION DOC] Regression test: compiles a batched matrix multiply whose M dimension
        # (2) is smaller than any autotune block size, to ensure the bmm template guards its
        # contiguity hints so no out-of-bounds vector loads occur. Off ROCm expects one
        # autotune run.
        # Verify BMM works when M < BLOCK_M. The triton_bmm template's
        # tl.max_contiguous/tl.multiple_of hints must be guarded by
        # M >= BLOCK_M to avoid out-of-bounds vectorized loads (see #179267).
        @torch.compile
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Compiled function returning the batched matrix product of
            # its inputs.
            return torch.bmm(a, b)

        # M=2 is smaller than any BLOCK_M autotuning config (typically >= 16)
        foo(
            torch.randn(4, 2, 64, device=GPU_TYPE),
            torch.randn(4, 64, 32, device=GPU_TYPE),
        )
        if not torch.version.hip:
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_not_even_k(self):
        # [FUNCTION DOC] Compiles a matrix multiply whose K dimension (22) is not a multiple
        # of any block size and, off ROCm, expects one autotune run.
        @torch.compile
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Compiled function returning the matrix product of its
            # inputs.
            return torch.mm(a, b)

        foo(
            torch.randn(11, 22, device=GPU_TYPE),
            torch.randn(22, 33, device=GPU_TYPE),
        )
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_baddbmm(self):
        # [FUNCTION DOC] Compiles a batched add-matmul with a broadcast bias and, off ROCm,
        # expects one autotune run.
        @torch.compile
        def foo(a, b, c):
            # [FUNCTION DOC] HELPER: Compiled function returning the bias plus the batched
            # matrix product of the first two inputs.
            return torch.baddbmm(c, a, b)

        foo(
            torch.randn(2, 8, 32, device=GPU_TYPE),
            torch.randn(2, 32, 8, device=GPU_TYPE),
            torch.randn(2, 1, 8, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @skipIfXpu(msg="https://github.com/pytorch/pytorch/issues/184490")
    @patches
    def test_mm_plus_mm(self):
        # [FUNCTION DOC] Compiles a sum of two 32 by 32 matrix products (the fused mm-plus-mm
        # pattern) and, off ROCm, expects one autotune run. Skipped on XPU because of a
        # tracked issue.
        @torch.compile
        def foo(a, b, c, d):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of two matrix
            # products.
            return (a @ b) + (c @ d)

        foo(
            torch.randn(32, 32, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    # TODO: fix accuracy failure of the triton template on XPU.
    # and enable this test case.
    @patches
    def test_mm_plus_mm2(self):
        # [FUNCTION DOC] Same mm-plus-mm pattern with 512 by 512 inputs; off ROCm expects one
        # autotune run. The preceding comment notes the Triton template has an accuracy
        # failure on XPU that should be fixed.
        @torch.compile
        def foo(a, b, c, d):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of two matrix
            # products.
            return (a @ b) + (c @ d)

        foo(
            torch.randn(512, 512, device=GPU_TYPE),
            torch.randn(512, 512, device=GPU_TYPE),
            torch.randn(512, 512, device=GPU_TYPE),
            torch.randn(512, 512, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_plus_mm3(self):
        # [FUNCTION DOC] Same mm-plus-mm pattern with tall-and-skinny shapes (512 by 32 times
        # 32 by 8, twice); off ROCm expects one autotune run.
        @torch.compile
        def foo(a, b, c, d):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of two matrix
            # products.
            return (a @ b) + (c @ d)

        foo(
            torch.randn(512, 32, device=GPU_TYPE),
            torch.randn(32, 8, device=GPU_TYPE),
            torch.randn(512, 32, device=GPU_TYPE),
            torch.randn(32, 8, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_dup_args(self):
        # [FUNCTION DOC] Compiles a matrix multiply of a tensor with itself (duplicate
        # arguments) and, off ROCm, expects one autotune run.
        @torch.compile
        def foo(a):
            # [FUNCTION DOC] HELPER: Compiled function returning the matrix product of its
            # input with itself.
            return torch.mm(a, a)

        foo(torch.randn(32, 32, device=GPU_TYPE))
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_dup_args_view(self):
        # [FUNCTION DOC] Compiles a matrix multiply between two views (top half and transposed
        # bottom half) of the same tensor, and off ROCm expects one autotune run.
        @torch.compile
        def foo(a):
            # [FUNCTION DOC] HELPER: Compiled function slicing the first and last 32 rows of
            # the input and multiplying the first by the transpose of the second.
            q = a[:32, :]
            k = a[32:, :]
            return torch.mm(q, k.transpose(0, 1))

        foo(torch.randn(64, 64, device=GPU_TYPE))
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @unittest.skipIf(IS_H100, "Fails on H100, see #143412")
    @expectedFailureDynamicWrapper
    @patches
    def test_convolution1(self):
        # [FUNCTION DOC] Compiles a strided, padded 2D convolution (stride 2 by 3, padding 4
        # by 5) of input plus one, with bias, and off ROCm expects one autotune run. Skipped
        # on H100 (known failure) and marked as an expected failure under the dynamic wrapper.
        @torch.compile
        def foo(x, w, b):
            # [FUNCTION DOC] HELPER: Compiled function running the aten convolution on input
            # plus one with stride 2 by 3, padding 4 by 5, no dilation or grouping.
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(2, 3),
                padding=(4, 5),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        foo(
            torch.randn(2, 33, 34, 41, device=GPU_TYPE),
            torch.randn(34, 33, 3, 3, device=GPU_TYPE),
            torch.randn(34, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @torch._inductor.config.patch(max_autotune_conv_backends="ATEN,TRITON")
    def test_non_1x1_convolution_caps_template_warps(self):
        # [FUNCTION DOC] Runs the strided non-1x1 convolution with the conv backends set to
        # aten and Triton, using a custom choices handler that offers a single Triton config
        # with 8 warps, and a patched template-choice appender that records the unroll flag
        # and warp count for each choice of the 2D convolution template. Checks the compiled
        # result matches eager, that exactly one choice was recorded with unroll false and 4
        # warps (warps capped from 8), and off ROCm one autotune run.
        @torch.compile
        def foo(x, w, b):
            # [FUNCTION DOC] HELPER: Compiled function running the aten convolution on input
            # plus one with stride 2 by 3, padding 4 by 5.
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(2, 3),
                padding=(4, 5),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        class EightWarpConvChoice(InductorChoices):
            # [CLASS DOC] HELPER: Choices handler that overrides the convolution config
            # generator to offer a single Triton config (large block sizes, one stage, eight
            # warps).
            def get_conv_configs(self, device_type: str | None = "cuda"):
                # [FUNCTION DOC] HELPER: Returns a config-generating function, ignoring the
                # device type argument.
                def configs(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: Generator yielding exactly one Triton config with
                    # block sizes M 1024, N 16, K 16, one stage and eight warps; ignores its
                    # arguments.
                    yield TritonConfig(
                        {"BLOCK_M": 1024, "BLOCK_N": 16, "BLOCK_K": 16},
                        num_stages=1,
                        num_warps=8,
                    )

                return configs

        recorded_conv_choices: list[tuple[bool, int]] = []
        maybe_append_choice = TritonTemplate.maybe_append_choice

        def record_conv_choice(template, choices, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper around the template choice appender: when the
            # template is the 2D convolution template, records the pair of unroll flag and
            # warp count from the keyword arguments; always forwards to the original appender
            # and returns its result.
            if template.name == "convolution2d":
                recorded_conv_choices.append((kwargs["UNROLL"], kwargs["num_warps"]))
            return maybe_append_choice(template, choices, **kwargs)

        x = torch.randn(2, 33, 34, 41, device=GPU_TYPE)
        w = torch.randn(34, 33, 3, 3, device=GPU_TYPE)
        b = torch.randn(34, device=GPU_TYPE)

        with (
            V.set_choices_handler(EightWarpConvChoice()),
            patch.object(TritonTemplate, "maybe_append_choice", record_conv_choice),
        ):
            result_compile = foo(x, w, b)
        result_eager = aten.convolution(
            x + 1,
            w,
            b,
            stride=(2, 3),
            padding=(4, 5),
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=1,
        )

        torch.testing.assert_close(result_compile, result_eager)
        self.assertEqual(recorded_conv_choices, [(False, 4)])
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_dropout(self):
        # [FUNCTION DOC] Compiles a matmul followed by multiplication with an inductor random
        # tensor (dropout-like), with relaxed verification tolerances on XPU and ROCm, and
        # checks the generated code. On CUDA (not ROCm) the code must contain no vectorized
        # four-way random helper and exactly one use of the Triton rand call. Off ROCm expects
        # one autotune run.
        @torch.compile
        def fn(x1, x2, seed):
            # [FUNCTION DOC] HELPER: Compiled function multiplying the matrix product of the
            # second and first inputs by random values generated from the seed input.
            mm_4 = torch.ops.aten.mm.default(x2, x1)
            rnd = torch.ops.prims.inductor_random.default(mm_4.shape, seed, "rand")
            return mm_4 * rnd

        # Triton MFMA matmul on AMD GPUs and Intel XPUs produces
        # ~1 ULP fp16 rounding differences vs the aten reference used by the
        # autotuner's internal VERIFY path. Max abs diff 0.0625 (= 2^-4),
        # max rel diff 2^-10 (fp16 mantissa LSB). Benign; expand tolerance
        # to match the existing XPU pattern.
        if GPU_TYPE == "xpu" or torch.version.hip:
            patcher = patch.object(
                select_algorithm, "VERIFY", dict(atol=1e-3, rtol=1e-3)
            )
            fn = patcher(fn)

        # sizes picked so triton autotuning wins
        _, (code,) = run_and_get_code(
            fn,
            torch.randn(512, 1024, dtype=torch.float16, device=GPU_TYPE),
            torch.randn(384, 512, dtype=torch.float16, device=GPU_TYPE),
            torch.tensor(12345, device=GPU_TYPE),
        )
        if GPU_TYPE == "cuda" and not torch.version.hip:
            self.assertEqual(code.count("triton_helpers.rand4x"), 0)
            self.assertEqual(code.count("tl.rand("), 1)
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @torch._inductor.config.patch(conv_1x1_as_mm=False)
    def test_convolution2(self):
        # [FUNCTION DOC] Compiles a 1x1 convolution with the setting that converts 1x1
        # convolutions to matmul disabled, and off ROCm expects one autotune run.
        @torch.compile
        def foo(x, w, b):
            # [FUNCTION DOC] HELPER: Compiled function running the aten convolution with unit
            # stride and no padding.
            return aten.convolution(
                x,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        foo(
            torch.randn(1, 33, 16, 16, device=GPU_TYPE),
            torch.randn(34, 33, 1, 1, device=GPU_TYPE),
            torch.randn(34, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @torch._inductor.config.patch(conv_1x1_as_mm=True)
    def test_convolution_as_mm(self):
        # [FUNCTION DOC] Compiles a 1x1 convolution of input plus one with the setting that
        # converts 1x1 convolutions to matmul enabled, and off ROCm expects one autotune run.
        @torch.compile
        def foo(x, w, b):
            # [FUNCTION DOC] HELPER: Compiled function running the aten convolution on input
            # plus one with unit stride and no padding.
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        foo(
            torch.randn(2, 33, 16, 16, device=GPU_TYPE),
            torch.randn(34, 33, 1, 1, device=GPU_TYPE),
            torch.randn(34, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @torch._inductor.config.patch(
        {"conv_1x1_as_mm": True, "max_autotune_gemm_backends": "TRITON"}
    )
    def test_convolution_as_mm_triton_only(self):
        # [FUNCTION DOC] Regression test for a 1x1 convolution lowered to matmul with only the
        # Triton gemm backend: a choices handler keeps only the first template config. The
        # channels-last prologue must not be fused into the matmul template, so the compiled
        # result must match eager (it would differ if fused) and no autotuning must occur off
        # ROCm.
        # To convert the 1x1 conv to matmul, x is converted to a channels last
        # tensor and the channels dimension is permuted to be innermost. This
        # prologue should not be fused with the matmul since the prologue writes
        # discontiguously, whilst the mm template currently only supports reading
        # the input contiguously.
        #
        # Before the change associated with this PR, fusion would occur because the actual kernel
        # input nodes (which don't include views e.g. permute) would be passed to the
        # `TritonTemplateCaller` rather than the input nodes that include views.
        # For example after x is converted to channels last, its layout is shape @ stride
        # [2, 33, 16, 16] @ [8432, 1, 528, 33], or [2, 33, 256] @ [8432, 1, 33], and the
        # prologue writes this value discontiguously.
        # After the permute, the mm template fixes the layout to [512, 33] @ [33, 1] and
        # reads the input contiguously. If the kernel input node for x is passed to the
        # `TritonTemplateCaller`, then the scheduler will fuse the prologue since the
        # write is compatible with the read. If however the viewed input is passed
        # to `TritonTemplateCaller`, then the write won't be compatible with the read,
        # and the prologue won't be fused.
        def foo(x, w, b):
            # [FUNCTION DOC] HELPER: Function running the aten convolution on input plus one
            # with unit stride and no padding; run eagerly and compiled.
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        x = torch.randn(2, 33, 16, 16, device=GPU_TYPE)
        w = torch.randn(34, 33, 1, 1, device=GPU_TYPE)
        b = torch.randn(34, device=GPU_TYPE)

        class SingleMMConfigChoice(InductorChoices):
            # [CLASS DOC] HELPER: Choices handler that returns only the first of the template
            # configs produced by the parent implementation.
            def get_template_configs(
                self,
                kernel_inputs: KernelInputs,
                templates: list[KernelTemplate | ExternKernelChoice],
                op_name: str,
                kwarg_overrides: dict[str, dict[str, Any]] | None = None,
            ):
                # [FUNCTION DOC] HELPER: Calls the parent config producer with the same
                # arguments and returns only its first result.
                return super().get_template_configs(
                    kernel_inputs, templates, op_name, kwarg_overrides
                )[:1]

        with V.set_choices_handler(SingleMMConfigChoice()):
            result_compile = torch.compile(foo)(x, w, b)
        result_eager = foo(x, w, b)

        # If the prologue has been fused this should fail
        torch.testing.assert_close(result_compile, result_eager)

        # There should not be any autotuning
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)

    @patches
    @torch._inductor.config.patch(conv_1x1_as_mm=False)
    def test_convolution2_group(self):
        # [FUNCTION DOC] Compiles a grouped convolution (32 groups, padding 1) with the
        # 1x1-to-matmul setting disabled, and off ROCm expects one autotune run.
        @torch.compile
        def foo(x, w, b):
            # [FUNCTION DOC] HELPER: Compiled function running the aten convolution with 32
            # groups and padding 1.
            return aten.convolution(
                x,
                w,
                b,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=32,  # group is not 1
            )

        foo(
            torch.randn(1, 32, 16, 16, device=GPU_TYPE),
            torch.randn(32, 1, 3, 3, device=GPU_TYPE),
            torch.randn(32, device=GPU_TYPE),
        )
        # Autotuning checks correctness of each version
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_convolution_depthwise_conv1d(self):
        # [FUNCTION DOC] Compiles a depthwise 1D convolution with bias (128 groups, padding 4)
        # and, off ROCm, expects one autotune run.
        @torch.compile
        def foo(x, w, b):
            # [FUNCTION DOC] HELPER: Compiled function running the aten 1D convolution with
            # 128 groups.
            return aten.convolution(
                x,
                w,
                b,
                stride=(1,),
                padding=(4,),
                dilation=(1,),
                transposed=False,
                output_padding=(0,),
                groups=128,
            )

        foo(
            torch.randn(2, 128, 202, device=GPU_TYPE),
            torch.randn(128, 1, 9, device=GPU_TYPE),
            torch.randn(128, device=GPU_TYPE),
        )
        if not torch.version.hip:
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_convolution_depthwise_conv1d_no_bias(self):
        # [FUNCTION DOC] Compiles a depthwise 1D convolution without bias (64 groups, stride
        # 2) and, off ROCm, expects one autotune run.
        @torch.compile
        def foo(x, w):
            # [FUNCTION DOC] HELPER: Compiled function running the aten 1D convolution with no
            # bias and 64 groups.
            return aten.convolution(
                x,
                w,
                None,
                stride=(2,),
                padding=(1,),
                dilation=(1,),
                transposed=False,
                output_padding=(0,),
                groups=64,
            )

        foo(
            torch.randn(4, 64, 100, device=GPU_TYPE),
            torch.randn(64, 1, 3, device=GPU_TYPE),
        )
        if not torch.version.hip:
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    def test_TritonTemplateCaller_str(self):
        # [FUNCTION DOC] Builds a Triton benchmark request with a dummy module path and
        # otherwise empty fields, creates a Triton template caller with the extra string, and
        # checks that converting it to text does not raise and equals the class name followed
        # by the module path and the extra string, comma separated, in parentheses.
        """
        Make sure str(TritonTemplateCaller) does not raise exceptions.
        """
        module_path = "abc.py"
        bmreq = TritonBenchmarkRequest(
            module_path=module_path,
            module_cache_key=None,
            kernel_name=None,
            extra_args=None,
            num_stages=None,
            num_warps=None,
            num_consumer_groups=None,
            num_buffers_warp_spec=None,
            input_tensor_meta=None,
            output_tensor_meta=None,
        )
        caller = select_algorithm.TritonTemplateCaller(
            None, None, None, None, "extra", bmreq
        )
        caller_str = str(caller)
        self.assertEqual(caller_str, f"TritonTemplateCaller({module_path}, extra)")


class TestSelectAlgorithmCleanup(TestCase):
    # [CLASS DOC] Tests that autotune benchmark cleanup releases only per-site state and
    # closes temporary benchmark artifacts.
    def test_benchmark_only_clears_matching_precompile_cache_entry(self):
        # [FUNCTION DOC] Sets up a selector cache with a precompile cache holding two entries,
        # patches its benchmark-function maker, runs a benchmark with an empty choice list and
        # the precompile key for one entry, then asserts the matching entry was removed and
        # the other kept.
        """
        Autotune cleanup should release only the closure for the active site.
        Clearing the whole precompile cache regresses compile-time reuse for
        unrelated autotune sites.
        """
        cache = select_algorithm.AlgorithmSelectorCache()
        cache.precompile_cache = {"keep": dict, "drop": dict}

        with patch.object(cache, "make_benchmark_fn", return_value=lambda _choices: {}):
            cache.benchmark([], [], unittest.mock.Mock(), None, precompile_key="drop")

        self.assertIn("keep", cache.precompile_cache)
        self.assertNotIn("drop", cache.precompile_cache)

    def test_release_benchmark_artifacts_closes_and_clears_state(self):
        # [FUNCTION DOC] Builds a caching autotuner without running its constructor, fills it
        # with a fake launcher (bound to a fake kernel), a fake compile result, a failure-
        # reason entry, a cached launcher and a debug-call marker, then calls its benchmark-
        # artifact release. Asserts the fake kernels' close methods ran in the order launcher
        # then compile result, that the launcher list, compile-result list and failure reasons
        # are emptied, and that the cached launcher and debug call are cleared to none.
        """
        Benchmark-only autotuners own temporary launchers and compile results.
        Cleanup must close their module owners and clear the references that
        otherwise keep Triton ``CompiledKernel`` objects alive.
        """
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        closed = []

        class FakeKernel:
            # [CLASS DOC] HELPER: Fake kernel object with a name; its close method appends the
            # name to the enclosing list of closed names.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Stores the given name on the instance.
                self.name = name

            def close(self):
                # [FUNCTION DOC] HELPER: Records this kernel's name by appending it to the
                # enclosing closed list.
                closed.append(self.name)

        class FakeCompileResult:
            # [CLASS DOC] HELPER: Fake compile result exposing a single fake kernel attribute
            # built from a name.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Creates the fake kernel for this compile result from
                # the given name.
                self.kernel = FakeKernel(name)

        def fake_launcher():
            # [FUNCTION DOC] HELPER: Placeholder launcher that does nothing and returns none;
            # the test sets its owner attribute to a fake kernel.
            return None

        launcher_kernel = FakeKernel("launcher")
        fake_launcher.__self__ = launcher_kernel  # type: ignore[attr-defined]

        autotuner = object.__new__(CachingAutotuner)
        autotuner.launchers = [fake_launcher]
        autotuner.compile_results = [FakeCompileResult("compile_result")]
        autotuner.benchmark_failure_reasons = {fake_launcher: "failed"}
        autotuner._cached_launcher = fake_launcher
        autotuner._debug_call = object()

        autotuner.release_benchmark_artifacts()

        self.assertEqual(closed, ["launcher", "compile_result"])
        self.assertEqual(autotuner.launchers, [])
        self.assertEqual(autotuner.compile_results, [])
        self.assertEqual(autotuner.benchmark_failure_reasons, {})
        self.assertIsNone(autotuner._cached_launcher)
        self.assertIsNone(autotuner._debug_call)


class TestExternKernelCaller(TestCase):
    # [CLASS DOC] Tests for extern kernel choice callers: tensor-meta failure handling, hash-
    # key based deduplication of identical autotune sites, benchmark timing, and benchmark
    # request variants. Most need a GPU.
    @requires_gpu()
    @patches
    @torch._inductor.config.patch(max_autotune_gemm_backends="ATEN")
    def test_extern_kernel_tensor_meta_failure(self):
        # [FUNCTION DOC] Compiles a matrix multiply with only the aten gemm backend while the
        # tensor-meta construction is patched to raise a value error. Asserts the selector
        # logger emitted a warning at warning level containing the text about failing to
        # construct input/output tensor meta for an extern choice, and that the compiled
        # result still matches eager within tolerance. Requires a GPU.
        """
        Test that when TensorMeta.from_irnodes fails during ExternKernelCaller
        initialization, a warning is logged with the correct message.
        """
        from unittest.mock import patch

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the matrix product of its two inputs.
            return torch.mm(a, b)

        a = torch.randn(64, 64, device=GPU_TYPE)
        b = torch.randn(64, 64, device=GPU_TYPE)

        with patch.object(
            TensorMeta, "from_irnodes", side_effect=ValueError("Mocked failure")
        ):
            with self.assertLogs(
                "torch._inductor.select_algorithm", level="WARNING"
            ) as log_context:
                compiled_fn = torch.compile(fn)
                result = compiled_fn(a, b)

        self.assertTrue(
            any(
                "Constructing input/output tensor meta failed for Extern Choice"
                in message
                for message in log_context.output
            ),
            lambda msg: f"{msg}\nExpected warning message not found in logs: {log_context.output}",
        )

        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    @skipIfRocmArch(MI200_ARCH)
    @patches
    def test_extern_kernel_caller_hash_key_deduplication(self):
        # [FUNCTION DOC] Compiles a function that adds two identical-shaped (64 by 64) matrix
        # products and checks the result against eager within tolerance, and that, off ROCm,
        # autotuning ran only once because the second site hits the cache. Skipped on MI200
        # ROCm.
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Computes two matrix products and returns their sum.
            # Two identical matmuls with same shapes
            result1 = torch.mm(a, b)
            result2 = torch.mm(c, d)
            return result1 + result2

        # Use identical shapes for all tensors
        shape = (64, 64)
        a = torch.randn(*shape, device=GPU_TYPE)
        b = torch.randn(*shape, device=GPU_TYPE)
        c = torch.randn(*shape, device=GPU_TYPE)
        d = torch.randn(*shape, device=GPU_TYPE)

        compiled_fn = torch.compile(fn)
        result = compiled_fn(a, b, c, d)

        # Verify correctness
        expected = torch.mm(a, b) + torch.mm(c, d)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

        # Only autotune once, cache hit
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @skipIfRocmArch(MI200_ARCH)
    @patches
    def test_extern_kernel_benchmark_valid_timing(self):
        # [FUNCTION DOC] Compiles a 256 by 256 matrix multiply in max-autotune mode without
        # cuda graphs, checks the result against eager and, off ROCm, that autotune ran once.
        # Skipped on MI200 ROCm.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the matrix product of its two inputs.
            return torch.mm(a, b)

        # Use larger matrices to ensure measurable timing
        a = torch.randn(256, 256, device=GPU_TYPE)
        b = torch.randn(256, 256, device=GPU_TYPE)

        compiled_fn = torch.compile(fn, mode="max-autotune-no-cudagraphs")
        result = compiled_fn(a, b)

        # Verify correctness
        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @requires_gpu()
    def test_extern_kernel_benchmark_request_variations(self):
        # [FUNCTION DOC] Exercises the extern-kernel GPU benchmark request in four
        # configurations, all with 64 by 64 float32 tensor metadata for the mm kernel: out-
        # variant with no output tensor and no args; no out-variant with no output and no
        # args; no out-variant with real args and an output tensor (checking the output equals
        # the true matrix product); and the same with the profiling-based do-bench config
        # enabled. Each benchmark call must return a float timing that is non-negative.
        # Requires a GPU.
        """
        Test that ExternKernelBenchmarkRequest.benchmark behaves correctly across
        different configurations:
        - With has_out_variant=True
        - When out is None (tensors created from metadata)
        - With profile_bandwidth_with_do_bench_using_profiling enabled
        - When len(args) is 0
        """

        input_meta = [
            TensorMeta(
                device=torch.device(GPU_TYPE),
                dtype=torch.float32,
                sizes=(64, 64),
                strides=(64, 1),
                offset=0,
            ),
            TensorMeta(
                device=torch.device(GPU_TYPE),
                dtype=torch.float32,
                sizes=(64, 64),
                strides=(64, 1),
                offset=0,
            ),
        ]
        output_meta = TensorMeta(
            device=torch.device(GPU_TYPE),
            dtype=torch.float32,
            sizes=(64, 64),
            strides=(64, 1),
            offset=0,
        )

        # Test 1: has_out_variant=True with out=None and len(args)==0
        # This should call super().benchmark() which creates tensors from metadata
        request_out_variant = ExternKernelGPUBenchmarkRequest(
            kernel_name="mm",
            input_tensor_meta=input_meta,
            output_tensor_meta=output_meta,
            extra_args=[],
            callable_path="extern_kernels.mm",
            has_out_variant=True,
        )
        timing = request_out_variant.benchmark(out=None)
        self.assertIsInstance(timing, float)
        self.assertGreaterEqual(timing, 0.0)

        # Test 2: has_out_variant=False with out=None and len(args)==0
        # When has_out_variant=False but len(args)==0, it should still call super().benchmark()
        request_no_out_variant = ExternKernelGPUBenchmarkRequest(
            kernel_name="mm",
            input_tensor_meta=input_meta,
            output_tensor_meta=output_meta,
            extra_args=[],
            callable_path="extern_kernels.mm",
            has_out_variant=False,
        )
        timing = request_no_out_variant.benchmark(out=None)
        self.assertIsInstance(timing, float)
        self.assertGreaterEqual(timing, 0.0)

        # Test 3: has_out_variant=False with args provided
        # This should execute the non-out-variant path: call algo(*args) and copy result
        a = torch.randn(64, 64, device=GPU_TYPE)
        b = torch.randn(64, 64, device=GPU_TYPE)
        out = torch.empty(64, 64, device=GPU_TYPE)

        request_with_args = ExternKernelGPUBenchmarkRequest(
            kernel_name="mm",
            input_tensor_meta=input_meta,
            output_tensor_meta=output_meta,
            extra_args=[],
            callable_path="extern_kernels.mm",
            has_out_variant=False,
        )
        timing = request_with_args.benchmark(a, b, out=out)
        self.assertIsInstance(timing, float)
        self.assertGreaterEqual(timing, 0.0)
        # Verify that the output was copied correctly
        expected = torch.mm(a, b)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

        # Test 4: profile_bandwidth_with_do_bench_using_profiling enabled
        # with has_out_variant=False and len(args) > 0
        with config.patch(profile_bandwidth_with_do_bench_using_profiling=True):
            a = torch.randn(64, 64, device=GPU_TYPE)
            b = torch.randn(64, 64, device=GPU_TYPE)
            out = torch.empty(64, 64, device=GPU_TYPE)

            request_profiling = ExternKernelGPUBenchmarkRequest(
                kernel_name="mm",
                input_tensor_meta=input_meta,
                output_tensor_meta=output_meta,
                extra_args=[],
                callable_path="extern_kernels.mm",
                has_out_variant=False,
            )
            timing = request_profiling.benchmark(a, b, out=out)
            self.assertIsInstance(timing, float)
            self.assertGreaterEqual(timing, 0.0)


@contextlib.contextmanager
def patch_lowering(lowering_overrides) -> Callable[[], None]:
    # [FUNCTION DOC] HELPER: Context manager that temporarily overrides inductor lowerings.
    # Takes a mapping from op to a tuple of (lowering function, broadcast flag, type promotion
    # kind, convert-input-to-bool flag). Inside a patched copy of the global lowering table
    # (restored on exit) it registers each override into that table, then yields to the body.
    # KNOWN ISSUE: the return annotation claims a callable, but this is a generator-based
    # context manager that yields no value.
    import torch._inductor.lowering as inductor_lowering

    with unittest.mock.patch.dict(inductor_lowering.lowerings):
        for fn, (
            decomp_fn,
            broadcast,
            type_promotion_kind,
            convert_input_to_bool,
        ) in lowering_overrides.items():
            inductor_lowering._register_lowering(
                fn,
                decomp_fn,
                broadcast=broadcast,
                type_promotion_kind=type_promotion_kind,
                convert_input_to_bool=convert_input_to_bool,
                lowering_dict=inductor_lowering.lowerings,
            )

        yield


class TestDtypeViewAutotuning(TestCase):
    # [CLASS DOC] Test case for autotuning inputs created from dtype-view IR nodes.
    @requires_gpu()
    @unittest.skipIf(
        not hasattr(torch, "float4_e2m1fn_x2"),
        "float4_e2m1fn_x2 dtype not available",
    )
    @patches
    def test_benchmark_example_value_preserves_dtype_view(self):
        # [FUNCTION DOC] Compiles a trivial add while the add lowering is replaced by a
        # function that, inside the compilation context, builds a uint8 buffer and a 4-bit
        # float reinterpret view over it, calls the selector cache's example-value builder on
        # the view, and records the IR dtype, example dtype and example shape before
        # delegating to the original lowering. Asserts the lowering ran, that the example
        # keeps the 4-bit float dtype and has the expected shape. Resets dynamo first.
        # Requires GPU and a build that has the 4-bit dtype.
        """
        Verify that benchmark_example_value preserves the dtype when the IR
        node is a dtype view (e.g. uint8 storage viewed as float4_e2m1fn_x2).
        """
        from torch._inductor import ir

        m, k = 256, 2048
        device = torch.device(GPU_TYPE)

        # We need a V.graph context for benchmark_example_value.
        # Compile a trivial function to set up V.graph, then call the
        # function under test within the compilation callback.
        captured_results = {}

        orig_lowering = torch._inductor.lowering.lowerings[aten.add.Tensor]

        def patched_lowering(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement add lowering: builds the dtype-viewed IR
            # node, calls the example-value builder, stores dtype and shape results in the
            # enclosing dictionary, and returns the result of the original lowering.
            # We're inside compilation — V.graph is valid.
            # Construct a dtype-viewed IR node and test benchmark_example_value.
            base_layout = ir.FixedLayout(
                device=device, dtype=torch.uint8, size=[m, k], stride=[k, 1]
            )
            base_buf = ir.Buffer(name="test_buf", layout=base_layout)

            fp4_layout = ir.FixedLayout(
                device=device,
                dtype=torch.float4_e2m1fn_x2,
                size=[m, k],
                stride=[k, 1],
            )
            fp4_view = ir.ReinterpretView(data=base_buf, layout=fp4_layout)

            example = select_algorithm.AlgorithmSelectorCache.benchmark_example_value(
                fp4_view
            )
            captured_results["ir_dtype"] = fp4_view.get_dtype()
            captured_results["example_dtype"] = example.dtype
            captured_results["example_shape"] = tuple(example.shape)

            return orig_lowering(*args, **kwargs)

        torch._dynamo.reset()
        with patch.dict(
            torch._inductor.lowering.lowerings,
            {aten.add.Tensor: patched_lowering},
        ):
            compiled_fn = torch.compile(lambda x: x + 1)
            compiled_fn(torch.randn(4, device=device))

        self.assertIn("ir_dtype", captured_results, "Patched lowering was not called")
        self.assertEqual(
            captured_results["example_dtype"],
            torch.float4_e2m1fn_x2,
            lambda msg: f"{msg}\nbenchmark_example_value should preserve float4_e2m1fn_x2 dtype "
            f"after unwrapping the view, but got {captured_results['example_dtype']}",
        )
        self.assertEqual(captured_results["example_shape"], (m, k))


class TestGetInputsStorageSizeCheck(TestCase):
    # [CLASS DOC] Regression test class for the storage-size check in the selector cache input
    # builder.
    def test_get_inputs_realloc_with_undersized_base(self):
        # [FUNCTION DOC] Regression test: with a mocked graph handler and a patched example-
        # value builder that returns a bfloat16 base tensor too small in elements (200) for
        # the padded strides (340 needed) but large in bytes, calls the input builder and
        # asserts it reallocates instead of failing in the strided view, producing a single
        # extern input tensor with the node's shape. Runs on CPU.
        """
        Regression test: get_inputs compared storage size in bytes against
        needed size in elements.  For multi-byte dtypes the byte count
        exceeds the element count, so the check could miss cases where
        reallocation is required and torch.as_strided would crash.

        We mock benchmark_example_value to return a bf16 tensor whose
        storage is too small (in elements) for the IR node's padded
        strides, then call get_inputs with a mocked V.graph context.
        Without the fix the as_strided inside get_inputs raises
        RuntimeError; with the fix the reallocation branch fires.
        """
        from torch._inductor import ir

        device = torch.device("cpu")
        dtype = torch.bfloat16
        # Padded strides [100, 8, 1] with size [4, 5, 8] need 340
        # elements.  We give the base tensor only 200 elements (400
        # bytes), so 200 < 340 <= 400.
        node_size = [4, 5, 8]
        padded_strides = [100, 8, 1]
        base_elements = 200

        mock_graph = unittest.mock.MagicMock()
        mock_graph.sizevars.optimization_hints_with_override = (
            lambda exprs, hint_override=None: tuple(int(e) for e in exprs)
        )
        mock_graph.sizevars.optimization_hint_with_override = (
            lambda expr, hint_override=None: int(expr)
        )
        mock_graph.sizevars.optimization_hint = (
            lambda expr, fallback=None: int(expr) if expr is not None else fallback
        )
        mock_graph.buffer_layout_constraints = {}
        mock_graph.get_allocation_size = lambda node: node.get_size()

        with V.set_graph_handler(mock_graph):
            layout = ir.FixedLayout(
                device=device,
                dtype=dtype,
                size=node_size,
                stride=padded_strides,
            )
            buf = ir.Buffer(name="test_padded_buf", layout=layout)
            out_layout = ir.FixedLayout(
                device=device,
                dtype=dtype,
                size=node_size,
                stride=padded_strides,
            )

            def undersized_example(node, hint_override=None):
                # [FUNCTION DOC] HELPER: Replacement example-value builder: for a layout
                # argument returns a strided tensor sized exactly to the required storage; for
                # any other node returns a flat random tensor with the undersized element
                # count.
                if isinstance(node, ir.Layout):
                    sizes = tuple(int(s) for s in node.size)
                    strides = tuple(int(s) for s in node.stride)
                    needed = torch._prims_common.compute_required_storage_length(
                        sizes, strides, 0
                    )
                    t = torch.randn(needed, device=device, dtype=dtype)
                    return t.as_strided(sizes, strides, 0)
                return torch.randn(base_elements, device=device, dtype=dtype)

            with unittest.mock.patch.object(
                select_algorithm.AlgorithmSelectorCache,
                "benchmark_example_value",
                staticmethod(undersized_example),
            ):
                autotune_args = select_algorithm.AlgorithmSelectorCache.get_inputs(
                    choices=[],
                    input_nodes=[buf],
                    layout=out_layout,
                    input_gen_fns=None,
                )

            extern_inputs = autotune_args.extern.input_tensors
            self.assertEqual(len(extern_inputs), 1)
            self.assertEqual(extern_inputs[0].shape, torch.Size(node_size))


class TestGetInputsAddmmBias(TestCase):
    # [CLASS DOC] Tests the selector input builder's addmm handling: the aten choice
    # benchmarks the original 1D bias while the Triton template benchmarks the bias expanded
    # to rows by columns, so the builder recovers the former from the first row of the latter.
    # Class constants N (8) and K (4) are the output columns and inner size.
    """
    Aten addmm benchmarks the original 1D bias while the Triton template
    benchmarks the bias expanded to [M, N], so get_inputs recovers the former
    from row 0 of the latter.  These drive that path directly: an M whose hint
    is 0 has no row 0 to recover from.
    """

    N = 8
    K = 4

    def _get_inputs(self, m):
        # [FUNCTION DOC] HELPER: Builds a mocked graph handler and four CPU float32 IR buffers
        # (expanded 2D bias with zero row stride, 1D bias, mat1 of m by K, mat2 of K by N),
        # makes an automatic-spec extern addmm choice using the 1D bias, and returns the
        # result of the selector cache's input builder for those nodes and an m by N output
        # layout, with no custom generators. The m argument sets the row count.
        from torch._inductor import ir

        device = torch.device("cpu")
        dtype = torch.float32

        mock_graph = unittest.mock.MagicMock()
        mock_graph.sizevars.optimization_hints_with_override = (
            lambda exprs, hint_override=None: tuple(int(e) for e in exprs)
        )
        mock_graph.sizevars.optimization_hint_with_override = (
            lambda expr, hint_override=None: int(expr)
        )
        mock_graph.sizevars.optimization_hint = (
            lambda expr, fallback=None: int(expr) if expr is not None else fallback
        )
        mock_graph.buffer_layout_constraints = {}
        mock_graph.get_allocation_size = lambda node: node.get_size()

        def buf(name, size, stride):
            # [FUNCTION DOC] HELPER: Creates an IR buffer with the given name and a fixed
            # layout of the given size and stride on the enclosing device and dtype.
            return ir.Buffer(
                name=name,
                layout=ir.FixedLayout(
                    device=device, dtype=dtype, size=size, stride=stride
                ),
            )

        with V.set_graph_handler(mock_graph):
            # The expanded bias and the 1D bias are distinct buffers, which is
            # what makes get_inputs reconcile the two.
            bias_2d = buf("bias_expanded", [m, self.N], [0, 1])
            bias_1d = buf("bias", [self.N], [1])
            mat1 = buf("mat1", [m, self.K], [self.K, 1])
            mat2 = buf("mat2", [self.K, self.N], [self.N, 1])

            extern_choice = unittest.mock.create_autospec(
                select_algorithm.ExternKernelCaller, instance=True
            )
            extern_choice.name = "addmm"
            extern_choice.input_nodes = [bias_1d, mat1, mat2]

            return select_algorithm.AlgorithmSelectorCache.get_inputs(
                choices=[extern_choice],
                input_nodes=[bias_2d, mat1, mat2],
                layout=ir.FixedLayout(
                    device=device, dtype=dtype, size=[m, self.N], stride=[self.N, 1]
                ),
                input_gen_fns=None,
            )

    def test_addmm_1d_bias(self):
        # [FUNCTION DOC] With 4 rows, checks the Triton bias input has shape 4 by N, the
        # extern bias has shape N, and the extern bias equals the first row of the Triton
        # bias.
        args = self._get_inputs(m=4)
        triton_bias = args.triton.input_tensors[0]
        extern_bias = args.extern.input_tensors[0]
        self.assertEqual(triton_bias.shape, torch.Size([4, self.N]))
        self.assertEqual(extern_bias.shape, torch.Size([self.N]))
        # Both choices must benchmark the same bias values.
        self.assertEqual(extern_bias, triton_bias[0])

    def test_addmm_1d_bias_zero_rows(self):
        # [FUNCTION DOC] With zero rows, checks the Triton bias input has shape 0 by N and the
        # extern bias still has shape N (no first row is needed to recover it).
        args = self._get_inputs(m=0)
        self.assertEqual(args.triton.input_tensors[0].shape, torch.Size([0, self.N]))
        self.assertEqual(args.extern.input_tensors[0].shape, torch.Size([self.N]))


class TestTemplateRender(TestCase):
    # [CLASS DOC] Tests for Triton template rendering features: hook finalization in
    # subclasses, index dtype selection from subgraph nodes, external-template fusion, and
    # atomic-add epilogue fusion.
    @staticmethod
    def _template_kernel_has_atomic_add(code):
        # [FUNCTION DOC] HELPER: Static helper taking generated code text; splits it at jit
        # decorators and returns true if any chunk is a template kernel (its definition name
        # starts with the template prefix) whose portion before the first triple quote
        # contains the atomic add call.
        template_kernels = code.split("@triton.jit\n")
        return any(
            kernel.startswith("def triton_tem")
            and "tl.atomic_add" in kernel.split("'''", 1)[0]
            for kernel in template_kernels
        )

    @requires_gpu()
    @requires_triton()
    @config.patch(cuda_backend="triton")
    def test_finalized_subclass_hooks(self):
        # [FUNCTION DOC] Defines a Triton template kernel subclass that registers a custom
        # hook adding an identifier comment, a template subclass using it, and an add lowering
        # override that builds one choice with that template and autotune-selects it. Compiles
        # an add of zero vectors under that lowering and checks exactly one kernel is
        # produced, that it contains the hook text (so all hooks were finalized) and that its
        # metadata contains the custom triton meta. Requires GPU and Triton.
        """
        Tests that all registered triton template hooks have been finalized,
        especially in the case that the hooks are finalized manually by the
        caller i.e. by calling template.finalize_hook(hook_name)
        """
        hook_identifier = "# CUSTOM_HOOK"

        class ExtensionTritonTemplateKernel(TritonTemplateKernel):
            # [CLASS DOC] HELPER: Template kernel subclass that registers an extra template
            # environment function called custom_hook.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Runs the parent constructor with all arguments, then
                # registers the custom hook method as an extra template environment function.
                super().__init__(*args, **kwargs)
                self._register_extra_template_env_fns(
                    self.custom_hook,
                )

            def custom_hook(self) -> str:
                # [FUNCTION DOC] HELPER: Registers a hook under the custom hook placeholder
                # text whose body returns the hook identifier string, and returns the
                # registration result.
                """
                Custom hook that just returns a test string for validation
                """

                def hook() -> str:
                    # [FUNCTION DOC] HELPER: Returns the hook identifier string from the
                    # enclosing test.
                    return hook_identifier

                return self._register_hook("<CUSTOM_HOOK>", hook)

            def inductor_meta_common(self):
                # [FUNCTION DOC] HELPER: Returns the parent implementation's common inductor
                # metadata unchanged.
                return super().inductor_meta_common()

        class ExtensionTritonTemplate(TritonTemplate):
            # [CLASS DOC] HELPER: Triton template subclass whose kernel type is the extension
            # kernel class above.
            kernel_type = ExtensionTritonTemplateKernel

        add_template = ExtensionTritonTemplate(
            name="add",
            grid=lambda *args, **kwargs: (1, 1, 1),
            source=(
                r"""
{{def_kernel("A", "B")}}
    {{custom_hook()}}
    xoffset = tl.program_id(0)
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(A + xindex)
    tmp1 = tl.load(B + xindex)
    tmp2 = tmp0 + tmp1
    {{store_output(("xindex",), "tmp2", mask="xmask", val_shape=("XBLOCK",))}}
    """
            ),
        )

        XBLOCK = 32
        custom_triton_meta = {"foo": "bar"}

        def add_override(a, b, alpha=None):
            # [FUNCTION DOC] HELPER: Lowering override for add: builds a fixed layout from the
            # first input, appends one choice from the extension template (one stage, two
            # warps, the block size and the custom meta), runs autotune selection among the
            # choices and returns the selected node. The alpha argument is ignored.
            layout = FixedLayout(a.get_device(), a.get_dtype(), a.get_size())
            choices = []
            add_template.maybe_append_choice(
                choices,
                input_nodes=(a, b),
                layout=layout,
                num_stages=1,
                num_warps=2,
                XBLOCK=XBLOCK,
                triton_meta=custom_triton_meta,
            )
            node, _ = autotune_select_algorithm("add", choices, [a, b], layout)
            return node

        with patch_lowering(
            {
                torch.ops.aten.add.Tensor: (
                    add_override,
                    True,
                    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
                    False,
                )
            }
        ):

            @torch.compile
            def add(a, b):
                # [FUNCTION DOC] HELPER: Compiled function adding its two inputs.
                return a + b

            a = torch.zeros((XBLOCK,), device=GPU_TYPE)
            b = torch.zeros((XBLOCK,), device=GPU_TYPE)

            _result, kernels = run_and_get_kernels(add, a, b)
            if len(kernels) != 1:
                raise AssertionError
            if hook_identifier not in kernels[0]:
                raise AssertionError
            FileCheck().check("triton_meta=").check(str(custom_triton_meta)).run(
                kernels[0]
            )

    def test_subgraph_nodes_participate_in_template_index_dtype(self):
        # [FUNCTION DOC] Covers implicit template buffers that are not explicit arguments:
        # with a fake graph handler that resolves one buffer of 2**31+1 elements by name and a
        # fake subgraph reading it, checks that the subgraph-index-dtype node collector
        # returns a one-tuple holding that buffer.
        """Covers implicit template buffers that are not explicit arguments."""
        import sympy

        from torch._inductor import ir
        from torch._inductor.dependencies import MemoryDep, ReadWrites
        from torch._inductor.select_algorithm import template_subgraph_index_dtype_nodes
        from torch._inductor.virtualized import V
        from torch.utils._ordered_set import OrderedSet

        large_capture = ir.Buffer(
            name="large_capture",
            layout=FixedLayout(torch.device("cpu"), torch.float32, [2**31 + 1]),
        )

        class FakeSubgraph:
            # [CLASS DOC] HELPER: Fake subgraph whose read-writes record has a single memory
            # read of the large buffer and no writes or index expressions.
            def get_read_writes(self):
                # [FUNCTION DOC] HELPER: Returns a read-writes record with one read dependency
                # on the large capture buffer at offset zero, empty writes and empty index
                # expressions.
                return ReadWrites(
                    reads=OrderedSet(
                        [MemoryDep("large_capture", sympy.Integer(0), (), ())]
                    ),
                    writes=OrderedSet(),
                    index_exprs=OrderedSet(),
                )

        class FakeGraph:
            # [CLASS DOC] HELPER: Fake graph handler that can look up the large capture buffer
            # by name.
            def try_get_buffer(self, name):
                # [FUNCTION DOC] HELPER: Returns the large capture buffer when the name
                # matches it, otherwise none.
                return large_capture if name == "large_capture" else None

        with V.set_graph_handler(FakeGraph()):
            self.assertEqual(
                template_subgraph_index_dtype_nodes([FakeSubgraph()]),
                (large_capture,),
            )

    @unittest.skipIf(
        TEST_WITH_ROCM or TEST_XPU, "https://github.com/pytorch/pytorch/issues/179959"
    )
    @requires_gpu()
    @requires_triton()
    @config.patch(cuda_backend="triton")
    def test_external_template_prologue_epilogue_fusion(self):
        # [FUNCTION DOC] Tests prologue fusion, epilogue fusion and extra inputs through the
        # external template kernel render path. Registers a mock external add template buffer
        # as the add lowering, compiles relu of (a plus sigmoid of b) times a bias, and checks
        # results match eager, the render hook was called, the mock kernel appears in the
        # code, and the maximum op (epilogue) and sigmoid (prologue) were fused into it.
        # Skipped on ROCm and XPU (tracked issue); requires GPU and Triton.
        """
        Tests prologue fusion, epilogue fusion, and extra inputs through the
        ExternalTritonTemplateKernel render()-based path.

        Compiled function: relu(template_add(a, sigmoid(b))) * bias
          - Prologue: sigmoid(b) fused into template as <LOAD_INPUT_B>
          - Epilogue: relu(...) * bias fused into template as <STORE_OUTPUT_0>
          - Extra inputs: bias is read by the epilogue but is not among the
            template's original inputs, exercising kernel._extra_inputs
        """
        import torch._inductor.ir as ir
        from torch._inductor.ir import OrderedSet
        from torch._inductor.utils import Placeholder, run_and_get_code

        XBLOCK = 128
        render_called = [False]

        # Template source with placeholders filled in by _render()
        _MOCK_ADD_KERNEL_TEMPLATE = (
            "import triton\n"
            "import triton.language as tl\n"
            "import torch\n"
            "from torch._inductor.runtime import triton_helpers\n"
            "\n"
            "@triton.jit\n"
            "def _mock_inner_add(A, B, {out_param}{extra_sig},"
            " numel, XBLOCK: tl.constexpr):\n"
            "    xoffset = tl.program_id(0) * XBLOCK\n"
            "    xindex = xoffset + tl.arange(0, XBLOCK)\n"
            "    xmask = xindex < numel\n"
            "    a = tl.load(A + xindex, mask=xmask)\n"
            "{prologue_load_b}"
            "    _kernel_val_0 = a + b\n"
            "    x_epilogue0_0 = xindex\n"
            "    _tile_mask_0 = xmask\n"
            "    <STORE_OUTPUT_0>\n"
            "\n"
            "def {kernel_name}(A, B, {out_param}{extra_sig}, numel):\n"
            "    grid = ((numel + {xblock} - 1) // {xblock},)\n"
            "    _mock_inner_add[grid]("
            "A, B, {out_param}{extra_sig}, numel, XBLOCK={xblock})\n"
            "    return {out_param}\n"
        )

        class _MockExternalTemplateBuffer(ir.TemplateBuffer):
            # [CLASS DOC] HELPER: Template buffer standing in for an external add template:
            # its kernel-render callable creates an external template kernel and a render
            # function delegating to the buffer's render method. Allows prologue fusion on the
            # second input and exposes the output as epilogue-fusable.
            def __init__(self, layout, inputs):
                # [FUNCTION DOC] HELPER: Builds a kernel-and-render factory closure, calls the
                # parent constructor with layout, inputs, the factory and named inputs A and
                # B, then allows prologue fusion for input B and marks the buffer's own name
                # as an epilogue-fusable output named result.
                tb_self = self

                def _make_kernel_render(out_node, hint_override=None):
                    # [FUNCTION DOC] HELPER: Creates an external template kernel for the
                    # enclosing buffer and returns it with a render function; the output node
                    # and hint override arguments are unused.
                    kernel = ExternalTritonTemplateKernel(tb_self)

                    def render():
                        # [FUNCTION DOC] HELPER: Renders the buffer's template source for the
                        # enclosing kernel.
                        return tb_self._render(kernel)

                    return kernel, render

                super().__init__(
                    layout,
                    inputs,
                    _make_kernel_render,
                    named_inputs={"A": inputs[0], "B": inputs[1]},
                )
                # Allow prologue fusion on input B (sigmoid(b) can be
                # absorbed so the template reads b directly)
                self.allowed_prologue_inps = OrderedSet([inputs[1].get_name()])
                self.epilogue_fusable_outputs = {self.name: "result"}

            def _render(self, kernel):
                # [FUNCTION DOC] HELPER: Marks the render-called flag, sets up fusion hooks on
                # the kernel, then builds the kernel source text: uses the fused prologue
                # source buffer for input B if present (loading it through a prologue
                # placeholder) else loads B directly; takes the output parameter and buffer
                # from the extra store targets if any; adds extra input parameters and
                # buffers, the element count and signature text; fills the mock source
                # template with the kernel name placeholder and block size; sets the kernel's
                # call arguments and empty preamble; returns a partial render of the source
                # with the kernel's render hooks.
                render_called[0] = True

                # Set up all fusion hooks in one call
                kernel._setup_fusion_hooks()

                # --- Prologue handling for B ---
                b_arg = self.inputs[1].get_name()
                prologue_load_b = "    b = tl.load(B + xindex, mask=xmask)\n"
                if kernel._prologue_source_buffers.get("B") is not None:
                    b_arg = kernel._prologue_source_buffers["B"]
                    prologue_load_b = (
                        "    _prologue_B_xindex = xindex\n"
                        "    _prologue_B_xmask = xmask\n"
                        "    <LOAD_INPUT_B>\n"
                        "    b = _prologue_B_result\n"
                    )

                call_args = [self.inputs[0].get_name(), b_arg]

                # --- Epilogue handling ---
                out_param = "result"
                out_arg = self.name
                for buf, param in kernel._extra_store_targets.items():
                    out_param = param
                    out_arg = buf
                    break

                call_args.append(out_arg)

                # --- Extra inputs ---
                extra_params = []
                for buf_name, param_name in kernel._extra_inputs.items():
                    extra_params.append(param_name)
                    call_args.append(buf_name)

                numel = self.get_size()[0]
                call_args.append(str(numel))

                extra_sig = ", " + ", ".join(extra_params) if extra_params else ""

                source = _MOCK_ADD_KERNEL_TEMPLATE.format(
                    out_param=out_param,
                    extra_sig=extra_sig,
                    prologue_load_b=prologue_load_b,
                    kernel_name=str(Placeholder.KERNEL_NAME),
                    xblock=XBLOCK,
                )

                kernel._call_preamble = []
                kernel._call_args = call_args

                return PartialRender(source, kernel.render_hooks)

        def add_override(a, b, alpha=None):
            # [FUNCTION DOC] HELPER: Lowering override for add: builds a fixed layout,
            # realizes both inputs with stride-one requirement, and returns a tensor box
            # wrapping a mock external template buffer. The alpha argument is ignored.
            layout = FixedLayout(a.get_device(), a.get_dtype(), a.get_size())
            a = ir.ExternKernel.require_stride1(ir.ExternKernel.realize_input(a))
            b = ir.ExternKernel.require_stride1(ir.ExternKernel.realize_input(b))
            return ir.TensorBox.create(_MockExternalTemplateBuffer(layout, [a, b]))

        # (override_fn, decompose, type_promotion, convert_input_to_bool)
        with patch_lowering(
            {
                torch.ops.aten.add.Tensor: (
                    add_override,
                    True,
                    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
                    False,
                )
            }
        ):

            @torch.compile
            def f(a, b, bias):
                # [FUNCTION DOC] HELPER: Compiled function returning relu of (a plus sigmoid
                # of b) multiplied by bias; multiplication is used for bias so the add
                # lowering override is not triggered again.
                # Use * for bias so it doesn't trigger add_override again
                return torch.relu(a + torch.sigmoid(b)) * bias

            a = torch.randn(32, device=GPU_TYPE)
            b = torch.randn(32, device=GPU_TYPE)
            bias = torch.randn(32, device=GPU_TYPE)

            result, (code,) = run_and_get_code(f, a, b, bias)
            expected = torch.relu(a + torch.sigmoid(b)) * bias
            torch.testing.assert_close(result, expected)

            # Verify render() was called (new protocol)
            self.assertTrue(render_called[0])
            # Verify template kernel was used
            self.assertIn("_mock_inner_add", code)
            # Verify epilogue fusion: relu fused via hook
            self.assertIn("maximum", code)
            # Verify prologue fusion: sigmoid fused via hook
            self.assertIn("tl.sigmoid", code)

    @inductor_config.patch(
        {
            "epilogue_fusion_first": True,
            "epilogue_fusion_with_atomic_add": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    @requires_gpu()
    @requires_triton()
    @patches
    def test_template_atomic_add_epilogue_fusion(self):
        # [FUNCTION DOC] With epilogue fusion first and atomic-add epilogue fusion enabled and
        # the Triton gemm backend, compiles a matrix multiply followed by an in-place indexed
        # add into an output, in max-autotune mode. Checks the result matches eager within
        # tolerance and that a template kernel contains an atomic add. Requires GPU and
        # Triton.
        from torch._inductor.utils import run_and_get_code

        def fn(a, b, index, out):
            # [FUNCTION DOC] HELPER: Matrix-multiplies a and b and adds the rows into out at
            # the given indices in place, returning out.
            mm = torch.matmul(a, b)
            return out.index_add_(0, index, mm)

        a = torch.randn(64, 64, device=GPU_TYPE)
        b = torch.randn(64, 64, device=GPU_TYPE)
        index = torch.arange(64, device=GPU_TYPE) // 2
        out = torch.randn(32, 64, device=GPU_TYPE)

        expected = fn(a, b, index, out.clone())
        opt_fn = torch.compile(fn, mode="max-autotune-no-cudagraphs")
        actual, (code,) = run_and_get_code(opt_fn, a, b, index, out.clone())

        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertTrue(self._template_kernel_has_atomic_add(code))

    @inductor_config.patch(
        {
            "epilogue_fusion_first": True,
            "epilogue_fusion_with_atomic_add": False,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    @requires_gpu()
    @requires_triton()
    @patches
    def test_template_atomic_add_epilogue_fusion_disabled(self):
        # [FUNCTION DOC] Same as the atomic-add fusion test but with atomic-add epilogue
        # fusion disabled: checks the result matches eager, that the generated code still
        # contains an atomic add (from the index add) but no template kernel contains one.
        from torch._inductor.utils import run_and_get_code

        def fn(a, b, index, out):
            # [FUNCTION DOC] HELPER: Matrix-multiplies a and b and adds the rows into out at
            # the given indices in place, returning out.
            mm = torch.matmul(a, b)
            return out.index_add_(0, index, mm)

        a = torch.randn(64, 64, device=GPU_TYPE)
        b = torch.randn(64, 64, device=GPU_TYPE)
        index = torch.arange(64, device=GPU_TYPE) // 2
        out = torch.randn(32, 64, device=GPU_TYPE)

        expected = fn(a, b, index, out.clone())
        opt_fn = torch.compile(fn, mode="max-autotune-no-cudagraphs")
        actual, (code,) = run_and_get_code(opt_fn, a, b, index, out.clone())

        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertIn("tl.atomic_add", code)
        self.assertFalse(self._template_kernel_has_atomic_add(code))


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU and is_big_gpu():
        run_tests()
