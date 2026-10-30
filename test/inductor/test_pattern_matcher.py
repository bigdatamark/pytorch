# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Inductor pattern matcher and the FX graph passes built on it (post-
# grad addmm/mm fusion and unfusion, mixed-dtype mm, cat/split rewrites, scaled softmax,
# mutation-region tracking, register_replacement and register_graph_pattern behavior, pattern
# pretty printing). Defines the opaque test class OpaqueScaleFactor (registered as a hoisted
# constant custom class at import time) and the TestPatternMatcher suite, plus further suites
# later in the file. Most tests compile small functions with torch.compile and check counters or
# generated code. Running the file as a script calls run_tests.
import copy
import os
import types
import unittest
from collections.abc import Callable

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.fx_passes.post_grad
import torch._inductor.pattern_matcher as pattern_matcher
import torch.nn.functional as F
from torch._dynamo.utils import count_calls, counters, detect_fake_mode
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._higher_order_ops.out_dtype import out_dtype
from torch._inductor.fx_passes import joint_graph
from torch._inductor.fx_passes.replace_random import replace_random_passes
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    fwd_only,
    gen_pattern,
    gen_pattern_and_search_gm,
    GetAttr,
    is_mutation_op,
    KeywordArg,
    Match,
    PatternMatcherPass,
    PatternPrettyPrinter,
    register_graph_pattern,
    register_replacement,
    ReplacementPatternEntry,
    stable_topological_sort,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch._library.opaque_object import (
    CustomClassBase,
    get_opaque_type_name,
    register_custom_class,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater, xfailIfSM89
from torch.testing._internal.common_device_type import skipCUDAIf
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, IS_BIG_GPU
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.utils import _pytree as pytree


aten = torch.ops.aten


class OpaqueScaleFactor(CustomClassBase):
    # [CLASS DOC] HELPER: Test-only opaque value class holding one scalar, used as a constant
    # opaque object in compiled graphs. Equality and hash are by the wrapped value; the FX repr
    # hook returns a constructor string plus the namespace needed to rebuild it. Registered
    # right after the definition as a constant, hoisted custom class.
    def __init__(self, val):
        # [FUNCTION DOC] HELPER: Stores the wrapped value in the val attribute.
        self.val = val

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal only to another OpaqueScaleFactor holding an equal val;
        # returns False for any other type.
        return isinstance(other, OpaqueScaleFactor) and self.val == other.val

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Hashes by the wrapped val so equal instances hash alike.
        return hash(self.val)

    def __fx_repr__(self):
        # [FUNCTION DOC] HELPER: Returns a pair: a repr string that reconstructs the instance
        # from val, and a mapping from the class name to the class so generated FX code can
        # resolve it.
        return (
            f"OpaqueScaleFactor({self.val!r})",
            {"OpaqueScaleFactor": OpaqueScaleFactor},
        )


register_custom_class(OpaqueScaleFactor, typ="constant", hoist=True)


@instantiate_parametrized_tests
class TestPatternMatcher(TestCase):
    # [CLASS DOC] Main pattern matcher test suite, run on the GPU device type (GPU_TYPE). Tests
    # compile functions and check pattern_matcher counters, generated code (via FileCheck) and
    # numerics against eager. Decorated with instantiate_parametrized_tests so parametrize
    # decorators expand into per-case tests. Holds shared helpers common,
    # _test_fused_int_mm_mul_impl and _test_mixed_impl.
    device_type = GPU_TYPE

    def common(
        self,
        fn,
        args,
        expected_matches,
        expected_nodes,
        additional_check=lambda code: None,
        reference_in_float=False,
    ):
        # [FUNCTION DOC] HELPER: Shared checker. Clears counters, seeds the RNG, optionally
        # upcasts tensor args to float32 for the eager reference, compiles fn with torch.compile
        # and runs it capturing generated code, then asserts closeness to the reference (dtype
        # checked only when no float reference is used). Asserts the pattern_matcher_count and
        # pattern_matcher_nodes inductor counters equal expected_matches and expected_nodes,
        # then calls additional_check with the code (a single code string if only one was
        # produced) and clears counters again.
        counters.clear()
        torch.manual_seed(42)
        if reference_in_float:
            ref_inputs = pytree.tree_map_only(
                torch.Tensor, lambda x: x.to(torch.float32), args
            )
        else:
            ref_inputs = args
        expected = fn(*ref_inputs)
        torch.manual_seed(42)
        actual, codes = run_and_get_code(torch.compile(fn), *args)
        if len(codes) == 1:
            codes = codes[0]
        torch.testing.assert_close(actual, expected, check_dtype=not reference_in_float)

        self.assertEqual(
            counters["inductor"]["pattern_matcher_count"], expected_matches
        )
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], expected_nodes)
        additional_check(codes)
        counters.clear()

    @inductor_config.patch(max_autotune_gemm=True)
    def test_mm_plus_mm(self):
        # [FUNCTION DOC] Checks mm+mm fusion with max autotune gemm enabled. For shapes where
        # the output dims of both mms agree, the mm_plus_mm pattern fuses (expected 1 match, 3
        # nodes). For the unfusable shapes (issue 100670) only the add-of-mm pattern matches (1
        # match, 2 nodes).
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Adds the results of two matrix multiplications.
            return torch.add(torch.mm(a, b), torch.mm(c, d))

        # when m1 == n1 and m2 == n2, mm_plus_mm can be matched to fused op
        fusible_args_list = [
            (
                torch.randn(16, 16, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
            ),
            (
                torch.randn(1, 4, device=GPU_TYPE),
                torch.randn(4, 2, device=GPU_TYPE),
                torch.randn(1, 5, device=GPU_TYPE),
                torch.randn(5, 2, device=GPU_TYPE),
            ),
        ]
        for args in fusible_args_list:
            self.common(fn, args, 1, 3)

        # if not fusible, it can only match add(mm())
        unfusible_args_list = [
            # https://github.com/pytorch/pytorch/issues/100670.
            (
                torch.randn(1, 4, device=GPU_TYPE),
                torch.randn(4, 2, device=GPU_TYPE),
                torch.randn(1, 2, device=GPU_TYPE),
                torch.randn(2, 1, device=GPU_TYPE),
            ),
            (
                torch.randn(1, 2, device=GPU_TYPE),
                torch.randn(2, 1, device=GPU_TYPE),
                torch.randn(1, 4, device=GPU_TYPE),
                torch.randn(4, 2, device=GPU_TYPE),
            ),
        ]
        for args in unfusible_args_list:
            self.common(fn, args, 1, 2)

    def _test_fused_int_mm_mul_impl(self, fn, args, fused_int_mm_mul_expected=True):
        # [FUNCTION DOC] HELPER: Compiles fn in max-autotune mode with fresh dynamo
        # state and cleared counters, and checks whether the generated code contains the
        # fused int mm template kernel according to fused_int_mm_mul_expected. When a
        # fusion is expected, compares the non-inf entries of the compiled result
        # against eager and checks dtype too.
        torch._dynamo.reset()
        counters.clear()
        ref = fn(*args)
        test, (code,) = run_and_get_code(torch.compile(fn, mode="max-autotune"), *args)
        self.assertEqual("triton_tem_fused__int" in code, fused_int_mm_mul_expected)
        if fused_int_mm_mul_expected:
            indices = ~ref.isinf()
            torch.testing.assert_close(
                ref[indices], test[indices]
            )  # also checks that dtype is correct

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_epilogue_fusion": False,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_fused_int_mm_mul(self):
        # [FUNCTION DOC] Checks that an int8 matmul (through out_dtype to int32) multiplied by a
        # scale tensor is fused into a Triton template, for float16, bfloat16 and float32 scale
        # inputs, with and without a trailing cast to bfloat16. Requires SM80, a big GPU and
        # Triton gemm backend settings.
        def fn1(a, b, c):
            # [FUNCTION DOC] HELPER: Int8 mm accumulated to int32 then multiplied by c.
            return out_dtype(torch.ops.aten.mm.default, torch.int32, a, b) * c

        def fn2(a, b, c):
            # [FUNCTION DOC] HELPER: Same as the first variant but the product is
            # converted to bfloat16.
            return (out_dtype(torch.ops.aten.mm.default, torch.int32, a, b) * c).to(
                torch.bfloat16
            )

        args_list = [
            (
                torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=GPU_TYPE),
                torch.randint(-128, 127, (32, 8), dtype=torch.int8, device=GPU_TYPE),
                torch.randn((32, 1), dtype=torch.float16, device=GPU_TYPE) * 0 + 0.5,
            ),
            (
                torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=GPU_TYPE),
                torch.randint(-128, 127, (32, 8), dtype=torch.int8, device=GPU_TYPE),
                torch.randn((1, 8), dtype=torch.bfloat16, device=GPU_TYPE),
            ),
            (
                torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=GPU_TYPE),
                torch.randint(-128, 127, (32, 8), dtype=torch.int8, device=GPU_TYPE),
                torch.randn((1, 8), dtype=torch.float32, device=GPU_TYPE),
            ),
        ]

        for args in args_list:
            self._test_fused_int_mm_mul_impl(fn1, args, True)
            self._test_fused_int_mm_mul_impl(fn2, args, True)

    def test_duplicate_search(self):
        # [FUNCTION DOC] Registers two independent replacements (add one to subtract one, add
        # two to subtract two) in one pass and applies the pass through a custom post grad pass
        # in a custom backend. A compiled function containing both patterns must be rewritten
        # and match the manually rewritten function, and the pass must report two replacements.
        from collections.abc import Callable, Iterable

        import torch
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )

        def pattern1(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Pattern: x plus one.
            return x + 1

        def replacement1(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Replacement: x minus one.
            return x - 1

        def pattern2(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Pattern: x plus two.
            return x + 2

        def replacement2(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Replacement: x minus two.
            return x - 2

        patterns = PatternMatcherPass()
        inputs = [torch.empty(4, 5, dtype=torch.float32, device=GPU_TYPE)]
        register_replacement(pattern1, replacement1, inputs, fwd_only, patterns)
        register_replacement(pattern2, replacement2, inputs, fwd_only, patterns)

        count = 0

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that stores the number of replacements
            # applied by the patterns into the enclosing count variable.
            nonlocal count
            count = patterns.apply(graph)

        def custom_backend(
            graph: torch.fx.GraphModule, example_inputs: Iterable[torch.Tensor]
        ) -> Callable:
            # [FUNCTION DOC] HELPER: Backend that copies the inductor config, installs the
            # custom post grad pass and compiles with compile_fx; returns the compiled callable.
            from torch._inductor import config

            current_config = config.shallow_copy_dict()
            from torch._inductor.compile_fx import compile_fx

            current_config["post_grad_custom_post_pass"] = custom_pass
            return compile_fx(graph, example_inputs, config_patches=current_config)

        @torch.compile(backend=custom_backend)
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Compiled function: relu of x plus one, plus two.
            y = x + 1
            y2 = y.relu() + 2
            return y2

        def f_replaced(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference for the expected rewritten computation: relu of x
            # minus one, minus two.
            y = x - 1
            y2 = y.relu() - 2
            return y2

        inp = torch.rand(3, 5, device=GPU_TYPE)
        self.assertEqual(f(inp), f_replaced(inp))
        self.assertEqual(count, 2)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_epilogue_fusion": False,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    @inductor_config.patch(force_fuse_int_mm_with_mul=True)
    @inductor_config.patch("test_configs.runtime_triton_dtype_assert", True)
    def test_fused_int_mm_mul_epilogue(self):
        # [FUNCTION DOC] Like test_fused_int_mm_mul but with a multiply-by-half and relu
        # epilogue after the scaled int mm, with force_fuse_int_mm_with_mul and runtime triton
        # dtype asserts enabled. Expects the fused template in all cases.
        def fn1(a, b, c):
            # [FUNCTION DOC] HELPER: Int8 mm to int32, times c, times 0.5, then relu.
            return (
                (out_dtype(torch.ops.aten.mm.default, torch.int32, a, b) * c) * 0.5
            ).relu()

        def fn2(a, b, c):
            # [FUNCTION DOC] HELPER: Same as the first variant but with a cast to bfloat16
            # between the scale multiply and the 0.5 multiply.
            return (
                (out_dtype(torch.ops.aten.mm.default, torch.int32, a, b) * c).to(
                    torch.bfloat16
                )
                * 0.5
            ).relu()

        args_list = [
            (
                torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=GPU_TYPE),
                torch.randint(-128, 127, (32, 8), dtype=torch.int8, device=GPU_TYPE),
                torch.randn((32, 1), dtype=torch.float16, device=GPU_TYPE) * 0 + 0.5,
            ),
            (
                torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=GPU_TYPE),
                torch.randint(-128, 127, (32, 8), dtype=torch.int8, device=GPU_TYPE),
                torch.randn((1, 8), dtype=torch.bfloat16, device=GPU_TYPE),
            ),
            (
                torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=GPU_TYPE),
                torch.randint(-128, 127, (32, 8), dtype=torch.int8, device=GPU_TYPE),
                torch.randn((1, 8), dtype=torch.float32, device=GPU_TYPE),
            ),
        ]

        for args in args_list:
            self._test_fused_int_mm_mul_impl(fn1, args, True)
            self._test_fused_int_mm_mul_impl(fn2, args, True)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_epilogue_fusion": False,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_fused_int_mm_mul_gating(self):
        # [FUNCTION DOC] Gating test: a scale tensor of shape eight (a one dimensional bias-like
        # scale) still allows the fused int mm mul template to be generated.
        def fn1(a, b, c):
            # [FUNCTION DOC] HELPER: Int8 mm accumulated to int32 then multiplied by c.
            return out_dtype(torch.ops.aten.mm.default, torch.int32, a, b) * c

        args1 = (
            torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=GPU_TYPE),
            torch.randint(-128, 127, (32, 8), dtype=torch.int8, device=GPU_TYPE),
            torch.randn((8), dtype=torch.float32, device=GPU_TYPE),
        )
        self._test_fused_int_mm_mul_impl(fn1, args1, True)

    def _test_mixed_impl(
        self,
        fn,
        args,
        mixed_mm_expected,
        fallback_mixed_mm_expected,
        rtol=None,
        atol=None,
    ):
        # [FUNCTION DOC] HELPER: Compiles fn with fresh dynamo state, compares against eager
        # with optional rtol and atol, then inspects the generated code. If mixed_mm_expected,
        # the kernel must show the k index loop with a dtype cast before the dot; otherwise (and
        # when no extern mm is used) no cast may appear before the dot. If
        # fallback_mixed_mm_expected, the code must call a run entry point of either a triton
        # template or the extern mm kernel.
        torch._dynamo.reset()
        counters.clear()
        ref = fn(*args)
        test, (code,) = run_and_get_code(torch.compile(fn), *args)
        torch.testing.assert_close(ref, test, rtol=rtol, atol=atol)

        if mixed_mm_expected:
            FileCheck().check("k_idx").check(".to(").check("tl.dot").run(code)
        else:
            if "extern_kernels.mm" not in code:
                FileCheck().check("k_idx").check_not(".to(").check("tl.dot").run(code)

        if fallback_mixed_mm_expected:
            extern_mm = "extern_kernels.mm" in code
            FileCheck().check("def call").check(".run").check(
                "triton_tem" if not extern_mm else "extern_kernels.mm"
            ).run(code)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_fusion": False,
            "benchmark_epilogue_fusion": False,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_mixed_mm(self):
        # [FUNCTION DOC] Checks that a matmul whose right operand is cast to the left operand's
        # dtype is turned into a mixed-dtype mm template, for several dtype combinations (int8,
        # uint8, bfloat16 with float32), under max autotune gemm.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: mm of a with b cast to a's dtype.
            return torch.mm(a, b.to(a.dtype))

        args_list = [
            (
                torch.randn(8, 8, device=GPU_TYPE),
                torch.randint(-128, 127, (8, 8), dtype=torch.int8, device=GPU_TYPE),
            ),
            (
                torch.randn(8, 2, device=GPU_TYPE, dtype=torch.bfloat16),
                torch.randint(-128, 127, (2, 8), dtype=torch.int8, device=GPU_TYPE),
            ),
            (
                torch.randn(8, 5, device=GPU_TYPE, dtype=torch.float16),
                torch.randint(0, 255, (5, 2), dtype=torch.uint8, device=GPU_TYPE),
            ),
            (
                torch.randn(8, 8, device=GPU_TYPE, dtype=torch.float32),
                torch.randn(8, 8, device=GPU_TYPE, dtype=torch.bfloat16),
            ),
        ]

        for args in args_list:
            self._test_mixed_impl(fn, args, True, False)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_fusion": False,
            "benchmark_epilogue_fusion": False,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    @parametrize("dtype_left", (torch.float16, torch.float32, torch.bfloat16))
    @parametrize("dtype_right", (torch.int8, torch.uint8))
    def test_mixed_mm_exhaustive(self, dtype_left, dtype_right):
        # [FUNCTION DOC] Parametrized over left dtype (float16, float32, bfloat16) and right
        # dtype (int8, uint8): builds 256x256 operands in the integer range of the right dtype
        # and checks the mixed mm template is used, with loose tolerance (rtol 0.16).
        def fn(a, b):
            # [FUNCTION DOC] HELPER: mm of a with b cast to a's dtype.
            return torch.mm(a, b.to(a.dtype))

        dtype_ranges = {torch.uint8: (0, 255), torch.int8: (-128, 127)}
        low, high = dtype_ranges[dtype_right]
        args = (
            torch.randn(256, 256, dtype=dtype_left, device=GPU_TYPE),
            torch.randint(low, high, (256, 256), dtype=dtype_right, device=GPU_TYPE),
        )
        self._test_mixed_impl(fn, args, True, False, rtol=0.16, atol=1e-4)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_fusion": False,
            "benchmark_epilogue_fusion": False,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_mixed_mm_bad_cases(self):
        # [FUNCTION DOC] Mixed mm with transposed and strided (every other column)
        # integer right operands in float16 and bfloat16 left operands; still expects
        # the mixed mm template path.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: mm of a with b cast to a's dtype.
            return torch.mm(a, b.to(a.dtype))

        args_list = [
            (
                torch.randn(8, 8, device=GPU_TYPE, dtype=torch.float16),
                torch.randint(-128, 127, (4, 8), dtype=torch.int8, device=GPU_TYPE).t()[
                    :, ::2
                ],
            ),
            (
                torch.randn(8, 8, device=GPU_TYPE, dtype=torch.bfloat16),
                torch.randint(0, 255, (4, 8), dtype=torch.uint8, device=GPU_TYPE).t()[
                    :, ::2
                ],
            ),
        ]

        for args in args_list:
            self._test_mixed_impl(fn, args, True, False)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @inductor_config.patch(
        {
            "benchmark_fusion": False,
            "benchmark_epilogue_fusion": False,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_gemm": True,
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_mixed_mm_epi_works(self):
        # [FUNCTION DOC] Mixed mm followed by a multiply and add of vectors (epilogue); checks
        # the template path is still used for int8 and uint8 right operands across dtypes.
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: mm of a with b cast to a's dtype, scaled by c and
            # offset by d.
            return torch.mm(a, b.to(a.dtype)) * c + d

        args_list = [
            (
                torch.randn(8, 8, device=GPU_TYPE),
                torch.randint(-128, 127, (8, 8), dtype=torch.int8, device=GPU_TYPE),
                torch.randn(8, device=GPU_TYPE),
                torch.randn(8, device=GPU_TYPE),
            ),
            (
                torch.randn(8, 2, device=GPU_TYPE, dtype=torch.bfloat16),
                torch.randint(-128, 127, (2, 8), dtype=torch.int8, device=GPU_TYPE),
                torch.randn(8, device=GPU_TYPE, dtype=torch.bfloat16),
                torch.randn(8, device=GPU_TYPE, dtype=torch.bfloat16),
            ),
            (
                torch.randn(8, 5, device=GPU_TYPE, dtype=torch.float16),
                torch.randint(0, 255, (5, 2), dtype=torch.uint8, device=GPU_TYPE),
                torch.randn(2, device=GPU_TYPE, dtype=torch.float16),
                torch.randn(2, device=GPU_TYPE, dtype=torch.float16),
            ),
        ]

        for args in args_list:
            self._test_mixed_impl(fn, args, True, False)

    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_mixed_mm_gating(self):
        # [FUNCTION DOC] Without max autotune the mixed mm template is not generated (a fallback
        # path is expected); with max autotune and Triton backend settings patched in, the mixed
        # mm template is expected.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: mm of a with b cast to a's dtype.
            return torch.mm(a, b.to(a.dtype))

        args = (
            torch.randn(8, 8, device=GPU_TYPE),
            torch.randint(-128, 127, (8, 8), dtype=torch.int8, device=GPU_TYPE),
        )
        # will no max autotune, will not generate fused template
        self._test_mixed_impl(fn, args, False, True)

        with inductor_config.patch(
            {
                "benchmark_fusion": False,
                "benchmark_epilogue_fusion": False,
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune_gemm": True,
            }
        ):
            self._test_mixed_impl(fn, args, True, False)

    def test_mixed_mm_cpu(self):
        # [FUNCTION DOC] CPU variant: mixed mm on CPU tensors must give correct results and
        # generate neither the mixed mm template nor the fallback expectation.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: mm of a with b cast to a's dtype.
            return torch.mm(a, b.to(a.dtype))

        args = (
            torch.randn(8, 8),
            torch.randint(-128, 127, (8, 8), dtype=torch.int8),
        )
        self._test_mixed_impl(fn, args, False, False)

    @parametrize(
        "case",
        [
            ((4, 8), GPU_TYPE),
            ("dynamic", GPU_TYPE),
        ],
    )
    def test_unsuccessful_partial_reuse(self, case):
        # [FUNCTION DOC] Parametrized by a static shape (4, 8) or dynamic shape: computes a
        # partial amax along dim zero (keepdim) and a full amax; since the full reduction cannot
        # reuse the partial one in these cases, partial_reduction_reuse stays at zero and the
        # output matches eager.
        shape, device = case

        def test_fn(x):
            # [FUNCTION DOC] HELPER: Returns a partial amax over dim zero with keepdim and a
            # full amax of x.
            partial = torch.amax(x, [0], True)
            full = torch.amax(x)
            return partial, full

        if shape == "dynamic":
            x = torch.rand([2048, 64], device=GPU_TYPE)
            torch._dynamo.mark_dynamic(x, 0)
        else:
            x = torch.randn(*shape, device=device)

        compiled_fn = torch.compile(test_fn)

        self.assertEqual(compiled_fn(x), test_fn(x))
        self.assertEqual(counters["inductor"]["partial_reduction_reuse"], 0)

    @parametrize(
        "case",
        [
            ((2048, 2048), (torch.amax, torch.amax)),
            ((1024, 1024), (torch.amin, torch.min)),
            ((4096, 512), (torch.amax, torch.max)),
        ],
    )
    def test_successful_partial_reuse(self, case):
        # [FUNCTION DOC] Parametrized over shape and a pair of (partial reduction, full
        # reduction) functions such as amax/amax, amin/min, amax/max: the full reduction is
        # rewritten to reuse the partial one, so partial_reduction_reuse is one and results
        # match eager.
        shape, (partial_fn, full_fn) = case

        def test_fn(x):
            # [FUNCTION DOC] HELPER: Applies the partial reduction function over dim zero with
            # keepdim and the full reduction function to x, returning both.
            partial = partial_fn(x, [0], True)
            full = full_fn(x)
            return partial, full

        x = torch.randn(*shape, device=GPU_TYPE)

        compiled_fn = torch.compile(test_fn)

        self.assertEqual(compiled_fn(x), test_fn(x))
        self.assertEqual(counters["inductor"]["partial_reduction_reuse"], 1)

    def test_addmm(self):
        # [FUNCTION DOC] Checks addmm fusion of add(a, mm(b, c)) and the reversed order on
        # several operand shapes: broadcastable bias shapes fuse (2 matches, 4 nodes) while a
        # mismatching batch shape or a python scalar bias do not (0, 0). Resets dynamo and
        # counters per case and compares with eager.
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Returns add of a with mm(b, c) and mm(b, c) plus a,
            # as a pair.
            return torch.add(a, torch.mm(b, c)), torch.mm(b, c) + a

        args_list = [
            (
                torch.randn(16, 16, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
                True,
            ),
            (
                torch.randn(8, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
                torch.randn(16, 8, device=GPU_TYPE),
                True,
            ),
            (
                torch.randn(16, 16, device=GPU_TYPE),
                torch.randn(1, 16, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
                False,
            ),
            (
                torch.randn(1, 16, 16, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
                False,
            ),
            (
                4,
                torch.randn(16, 16, device=GPU_TYPE),
                torch.randn(16, 16, device=GPU_TYPE),
                False,
            ),
        ]
        for a, b, c, should_fuse in args_list:
            torch._dynamo.reset()
            counters.clear()
            args = (a, b, c)
            e1, e2 = fn(*args)
            a1, a2 = torch.compile(fn)(*args)
            torch.testing.assert_close(a1, e1)
            torch.testing.assert_close(a2, e2)
            count, nodes = (2, 4) if should_fuse else (0, 0)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], count)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], nodes)

    def test_addmm_symbolic_scalar(self):
        # [FUNCTION DOC] With dynamic shapes, an mm plus a symbolic size scalar bias must not be
        # fused: output matches eager and pattern_matcher_count stays zero.
        def fn(m1, m2):
            # [FUNCTION DOC] HELPER: Adds the symbolic first dimension size of m1 to the product
            # of m1 and m2, in both operand orders.
            bias = m1.size(0)
            return torch.add(bias, torch.mm(m1, m2)), torch.mm(m1, m2) + bias

        m1 = torch.randn(16, 16, device=GPU_TYPE)
        m2 = torch.randn(16, 16, device=GPU_TYPE)

        counters.clear()
        expect = fn(m1, m2)
        actual = torch.compile(fn, dynamic=True)(m1, m2)
        self.assertEqual(expect, actual)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)

    def test_addmm_broadcasting_bias(self):
        # [FUNCTION DOC] A linear layer with a bias of shape (1, 4) and a 3-D input must still
        # compile and match eager results (broadcasting bias must not break addmm handling).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the functional linear with fixed random weight
            # and a (1, 4) bias on the GPU.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the functional linear, a random 4x4 weight and a
                # random (1, 4) bias on the GPU device.
                super().__init__()
                self.linear = torch.nn.functional.linear
                self.linear_weight = torch.randn(4, 4).to(GPU_TYPE)
                self.bias = torch.randn(1, 4).to(GPU_TYPE)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear to x with the stored weight and bias.
                x = self.linear(x, self.linear_weight, self.bias)
                return x

        input_tensor = torch.randn(1, 3, 4).to(GPU_TYPE)

        func = Model().to(GPU_TYPE)

        res1 = func(input_tensor)
        jit_func = torch.compile(func)
        res2 = jit_func(input_tensor)

        self.assertEqual(res1, res2)

    @inductor_config.patch(
        {
            "max_autotune_gemm_backends": "ATEN",
        }
    )
    def test_bmm_to_mm(self):
        # [FUNCTION DOC] With the ATEN gemm backend, a batch-1 bmm is converted to mm (checked
        # on GPU through the generated code containing the extern mm and no extern bmm), while a
        # batch-3 bmm stays an extern bmm. Results are compared against eager in both cases. On
        # non-GPU builds the first check is only constructed, not run.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Batched matmul of a and b.
            return torch.bmm(a, b)

        a = torch.randn(1, 16, 8, device=GPU_TYPE)
        b = torch.randn(1, 8, 32, device=GPU_TYPE)

        result, (code,) = run_and_get_code(torch.compile(fn), a, b)

        expected = fn(a, b)
        torch.testing.assert_close(result, expected)

        # The mm kernel should use ATen (because we set max_autotune_gemm_backends = ATEN).
        # Its name should contain `aten.bmm` since this is the original aten op where the bmm came from.
        if HAS_GPU:
            FileCheck().check("extern_kernels.mm(").check_not(
                "extern_kernels.bmm("
            ).run(code)
        else:
            FileCheck().check("extern_kernels.bmm(")

        a_multi = torch.randn(3, 16, 8, device=GPU_TYPE)
        b_multi = torch.randn(3, 8, 32, device=GPU_TYPE)

        result_multi, (code_multi,) = run_and_get_code(
            torch.compile(fn), a_multi, b_multi
        )

        expected_multi = fn(a_multi, b_multi)
        torch.testing.assert_close(result_multi, expected_multi)

        FileCheck().check("extern_kernels.bmm(").run(code_multi)

    def test_cat_mm(self):
        # [FUNCTION DOC] Concatenation of three mm results along dim one is compiled; asserts
        # the output equals eager and that the generated call function contains no triton run
        # (all work is done by one extern call).
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Concatenates mm of a,b; b,c; a,c along dim one.
            return torch.cat(
                [
                    torch.mm(a, b),
                    torch.mm(b, c),
                    torch.mm(a, c),
                ],
                1,
            )

        args = [
            torch.randn(16, 16, device=GPU_TYPE),
            torch.randn(16, 16, device=GPU_TYPE),
            torch.randn(16, 16, device=GPU_TYPE),
        ]
        out, code = run_and_get_code(torch.compile(fn), *args)
        self.assertEqual(out, fn(*args))
        FileCheck().check("call").check_not(".run").run(code[0])

    def test_cat_addmm(self):
        # [FUNCTION DOC] Same as test_cat_mm but with addmm operands rotated across
        # three inputs; checks equality with eager and that no kernel run appears in the
        # call function.
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Concatenates three addmm results with rotated operands
            # along dim one.
            return torch.cat(
                [
                    torch.addmm(a, b, c),
                    torch.addmm(b, c, a),
                    torch.addmm(c, a, b),
                ],
                1,
            )

        args = [
            torch.randn(16, 16, device=GPU_TYPE),
            torch.randn(16, 16, device=GPU_TYPE),
            torch.randn(16, 16, device=GPU_TYPE),
        ]
        out, code = run_and_get_code(torch.compile(fn), *args)
        self.assertEqual(out, fn(*args))
        FileCheck().check("call").check_not(".run").run(code[0])

    def test_cat_slice_cat_cuda(self):
        # [FUNCTION DOC] Cat followed by slice-slice and a second cat is optimized (1 match, 3
        # nodes) for a slice end of 19; for another shape compiled directly the counters are
        # checked only when dynamic shapes are not assumed by default; with a negative slice end
        # the optimized path is not used but a match of 1 and 3 nodes is still expected by the
        # helper. Aliased as the xpu test below for the cpp wrapper test.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Cat of a and b along dim one, slices of it with end 19,
            # then cat of the full cat and the slice.
            cat_1 = torch.ops.aten.cat.default([a, b], 1)
            slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
            slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
            return torch.ops.aten.cat.default([cat_1, slice_2], 1)

        args = [
            torch.randn(2, 32, device=GPU_TYPE),
            torch.randn(2, 16, device=GPU_TYPE),
        ]
        self.common(fn, args, 1, 3)

        args = [
            torch.randn(2, 8, device=GPU_TYPE),
            torch.randn(2, 16, device=GPU_TYPE),
        ]
        torch._dynamo.reset()
        counters.clear()
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        # We don't recompile for dynamic-shape cases.
        if dynamo_config.assume_static_by_default:
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 3)

        # Verify we fallback to non-optimal path for negative `end`.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Same structure as the first variant but the inner slice end
            # is negative one.
            cat_1 = torch.ops.aten.cat.default([a, b], 1)
            slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
            slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, -1)
            return torch.ops.aten.cat.default([cat_1, slice_2], 1)

        args = [
            torch.randn(2, 8, device=GPU_TYPE),
            torch.randn(2, 16, device=GPU_TYPE),
        ]
        self.common(fn, args, 1, 3)

    # called in test_gpu_cpp_wrapper
    test_cat_slice_cat_xpu = test_cat_slice_cat_cuda

    def test_pointless_view_pair(self):
        # [FUNCTION DOC] Pointless pair of view ops: two successive views are traced with
        # make_fx and joint_graph_passes must remove both (zero calls remain). If the
        # intermediate view is also returned the pair is kept. A view with a negative one size
        # is also collapsed.
        def f(x):
            # [FUNCTION DOC] HELPER: Two successive views reshaping to a 3-D shape then
            # to a 2-D one.
            x = aten.view.default(x, [3, 5, 7])
            x = aten.view.default(x, [15, 7])
            return x

        x = torch.randn(15, 7, device=GPU_TYPE)
        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 0)

        def f(x):
            # [FUNCTION DOC] HELPER: Same pair of views, but returns both the intermediate and
            # final views.
            x1 = aten.view.default(x, [3, 5, 7])
            x2 = aten.view.default(x1, [15, 7])
            return x1, x2

        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

        # handle negative 1 in size argument of view
        def f(x):
            # [FUNCTION DOC] HELPER: Two views where the second uses a negative one size.
            x = aten.view.default(x, [3, 5, 7])
            x = aten.view.default(x, [-1, 7])
            return x

        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 0)

    def test_pointless_view_pair_dynamic_shapes(self):
        # [FUNCTION DOC] With dynamic and unbacked shapes, a flatten then reshape back to the
        # original symbolic dims is removed; the output equals the input and the
        # removed_pointless_view_pair counter is one.
        def f(x):
            # [FUNCTION DOC] HELPER: Flattens x with a view then views it back to its original
            # two symbolic dims.
            s1, s2 = x.shape
            x = aten.view.default(x, [-1])
            x = aten.view.default(x, [s1, s2])
            return x

        x = torch.randn(15, 7, device=GPU_TYPE)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        out = torch.compile(f, dynamic=True)(x)
        self.assertTrue(torch.equal(x, out))

        self.assertEqual(counters["inductor"]["removed_pointless_view_pair"], 1)

    def test_pointless_permute_pair(self):
        # [FUNCTION DOC] Two successive transposing permutes are removed by joint graph passes
        # when the intermediate is unused elsewhere (zero calls left), and kept when the
        # intermediate is also returned.
        def f(x):
            # [FUNCTION DOC] HELPER: Applies the same two dim swap permute twice.
            x = aten.permute.default(x, [1, 0])
            x = aten.permute.default(x, [1, 0])
            return x

        x = torch.randn(15, 7, device=GPU_TYPE)
        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 0)

        def f(x):
            # [FUNCTION DOC] HELPER: Same double permute but returns the intermediate and the
            # final result.
            x1 = aten.permute.default(x, [1, 0])
            x2 = aten.permute.default(x1, [1, 0])
            return x1, x2

        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

    def test_pointless_permute_pair_3d(self):
        # [FUNCTION DOC] Three dimensional version of the pointless permute pair test
        # using a permutation swapping the first two dims; removed when the intermediate
        # is unused, kept otherwise.
        def f(x):
            # [FUNCTION DOC] HELPER: Applies the same 3-D permute twice.
            x = aten.permute.default(x, [1, 0, 2])
            x = aten.permute.default(x, [1, 0, 2])
            return x

        x = torch.randn(3, 5, 7, device=GPU_TYPE)
        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 0)

        def f(x):
            # [FUNCTION DOC] HELPER: Same double permute returning both intermediate and
            # final value.
            x1 = aten.permute.default(x, [1, 0, 2])
            x2 = aten.permute.default(x1, [1, 0, 2])
            return x1, x2

        gm = make_fx(f)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

    @parametrize(
        "input_dtype, intermediate_dtype, emulate_precision_casts, expected_calls",
        [
            (torch.float32, torch.float16, False, 2),
            (torch.float32, torch.bfloat16, False, 2),
            (torch.float32, torch.float64, False, 1),
            (torch.float32, torch.float16, True, 2),
            (torch.float16, torch.float32, True, 1),
        ],
    )
    def test_pointless_convert(
        self, input_dtype, intermediate_dtype, emulate_precision_casts, expected_calls
    ):
        # [FUNCTION DOC] Parametrized over input and intermediate dtype, emulate_precision_casts
        # and the expected number of remaining calls: a round trip dtype conversion pair is
        # removed only when the intermediate type preserves the input values (and not when
        # precision casts are emulated for narrowing). A float to int32 to float round trip is
        # never removed.
        def fn(x):
            # [FUNCTION DOC] HELPER: Converts x to the intermediate dtype then back to
            # the input dtype.
            x = torch.ops.prims.convert_element_type.default(x, intermediate_dtype)
            x = torch.ops.prims.convert_element_type.default(x, input_dtype)
            return x

        x = torch.randn(8, device=GPU_TYPE, dtype=input_dtype)
        gm = make_fx(fn)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        with inductor_config.patch(emulate_precision_casts=emulate_precision_casts):
            joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), expected_calls)

        def fn_int(x):
            # [FUNCTION DOC] HELPER: Converts float32 to int32 and back to float32.
            x = torch.ops.prims.convert_element_type.default(x, torch.int32)
            x = torch.ops.prims.convert_element_type.default(x, torch.float32)
            return x

        x = torch.randn(8, device=GPU_TYPE, dtype=torch.float32)
        gm = make_fx(fn_int)(x)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

    # Constant folding was explicitly turned off due to issue #108388
    # Turn it back on for test
    @inductor_config.patch(joint_graph_constant_folding=True)
    def test_pointless_cumsum(self):
        # [FUNCTION DOC] With joint graph constant folding enabled, cumsum of constant tensors
        # created with full is folded away: for each of eleven variants the compiled code has no
        # cumsum, results equal eager, and at least one pattern match is counted.
        def fn1():
            # [FUNCTION DOC] HELPER: Cumsum over dim one of ones converted to int64,
            # times those ones.
            ones = torch.full(
                [1, 128], 1, layout=torch.strided, dtype=torch.float32
            ).to(torch.int64)
            return torch.cumsum(ones, 1) * ones

        def fn2():
            # [FUNCTION DOC] HELPER: Cumsum over dim one of ones of shape 55 by 10
            # converted to int64.
            ones = torch.full(
                [55, 10], 1, layout=torch.strided, dtype=torch.float32
            ).to(torch.int64)
            return torch.cumsum(ones, 1)

        def fn3():
            # [FUNCTION DOC] HELPER: Cumsum over dim zero of a constant int64 tensor of twos.
            twos = torch.full([5, 4, 3], 2, dtype=torch.int64)
            return torch.cumsum(twos, 0)

        def fn4():
            # [FUNCTION DOC] HELPER: Cumsum of a float tensor filled with 0.1.
            x = torch.full([100], 0.1, dtype=torch.float32)
            return torch.cumsum(x, 0)

        def fn5():
            # [FUNCTION DOC] HELPER: Cumsum over a boolean tensor obtained from a filled tensor.
            t1 = torch.full([2, 4], 1)
            t2 = t1.to(dtype=torch.bool)
            return torch.cumsum(t2, 1)

        def fn6():
            # [FUNCTION DOC] HELPER: Cumsum of an int32 tensor filled with True.
            x = torch.full([10, 10], True, dtype=torch.int32)
            return torch.cumsum(x, 1)

        def fn7():
            # [FUNCTION DOC] HELPER: Cumsum of a boolean tensor with output dtype bfloat16.
            ones = torch.full([2, 4, 4], True, dtype=torch.bool)
            return torch.cumsum(ones, 1, dtype=torch.bfloat16)

        def fn8():
            # [FUNCTION DOC] HELPER: Cumsum of an int32 tensor with output dtype float64.
            x = torch.full([10, 10], 2, dtype=torch.int32)
            return torch.cumsum(x, 1, dtype=torch.float64)

        def fn9():
            # [FUNCTION DOC] HELPER: Cumsum of float32 0.1 values with output dtype float64.
            x = torch.full([100], 0.1, dtype=torch.float32)
            return torch.cumsum(x, 0, dtype=torch.float64)

        def fn10():
            # [FUNCTION DOC] HELPER: Cumsum of float16 ones with output dtype float32
            # over 5000 elements.
            x = torch.full([5000], 1.0, dtype=torch.float16)
            return torch.cumsum(x, 0, dtype=torch.float32)

        def fn11():
            # [FUNCTION DOC] HELPER: Cumsum of float32 2.5 values with output dtype int64.
            x = torch.full([10], 2.5, dtype=torch.float32)
            return torch.cumsum(x, 0, dtype=torch.int64)

        for fn in (fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn11):
            result, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True))
            self.assertNotIn("aten.cumsum", code)
            self.assertEqual(result, fn())
            self.assertGreaterEqual(counters["inductor"]["pattern_matcher_count"], 1)
            counters.clear()

    def test_reciprocal_sqrt_to_rsqrt(self):
        # [FUNCTION DOC] reciprocal of sqrt must be compiled into a single rsqrt: the
        # code contains rsqrt, no plain sqrt call, matches eager and at least one
        # pattern match is counted.
        # reciprocal(sqrt(x)) should fuse into a single rsqrt in the kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Reciprocal of the square root of x.
            return torch.reciprocal(torch.sqrt(x))

        x = torch.rand(64) + 1.0
        result, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        # A standalone sqrt (".sqrt(") must not survive; only ".rsqrt(" should.
        self.assertIn("rsqrt", code)
        self.assertNotIn(".sqrt(", code)
        self.assertEqual(result, fn(x))
        self.assertGreaterEqual(counters["inductor"]["pattern_matcher_count"], 1)

    def test_splitwithsizes_cat(self):
        # [FUNCTION DOC] Checks the split_with_sizes followed by cat rewrite: the good case (all
        # splits concatenated in order on the same dim) matches once with 4 nodes; cases where
        # not all getitems feed the cat, the cat dim differs, or the order is permuted (issue
        # 99686) do not match.
        # Good case
        def fn(a):
            # [FUNCTION DOC] HELPER: Splits into sizes 8 and 24 then concatenates the pieces in
            # order and squares.
            split_with_sizes = torch.ops.aten.split_with_sizes.default(a, [8, 24], 1)
            getitem = split_with_sizes[0]
            getitem_1 = split_with_sizes[1]
            cat = torch.ops.aten.cat.default([getitem, getitem_1], 1)
            return cat**2

        args = [
            torch.randn(2, 32, device=GPU_TYPE),
        ]
        self.common(fn, args, 1, 4)

        # Not all getitems are passed to cat
        def fn(a):
            # [FUNCTION DOC] HELPER: Three-way split where only the first two pieces are
            # concatenated and the third is added separately.
            split_with_sizes = torch.ops.aten.split_with_sizes.default(a, [8, 8, 16], 1)
            getitem = split_with_sizes[0]
            getitem_1 = split_with_sizes[1]
            getitem_2 = split_with_sizes[2]
            cat = torch.ops.aten.cat.default([getitem, getitem_1], 1)
            return cat**2 + getitem_2

        args = [
            torch.randn(2, 32, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # Different dimensions  (TODO this case should be handled by replacing with a reshape)
        def fn(a):
            # [FUNCTION DOC] HELPER: Split into four pieces then concatenation along a
            # different dim.
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                a, [8, 8, 8, 8], 1
            )
            cat = torch.ops.aten.cat.default(split_with_sizes, 0)
            return cat**2

        args = [
            torch.randn(2, 32, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # https://github.com/pytorch/pytorch/issues/99686.
        def fn(a):
            # [FUNCTION DOC] HELPER: Splits into three and concatenates the pieces in a
            # permuted order.
            x = torch.ops.aten.split_with_sizes.default(a, [3, 2, 3], dim=1)
            cat = torch.ops.aten.cat.default([x[1], x[0], x[2]], dim=1)
            return cat

        args = [
            torch.randn(1, 8, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

    def test_cat_splitwithsizes(self):
        # [FUNCTION DOC] Checks cat followed by split_with_sizes elimination across five
        # cases: the good case matches (1 match, 2 nodes); no match when cat has other
        # users, when dims differ, when the number of splits differs, or when split
        # sizes differ from cat input sizes.
        # good case
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Cat of three inputs on dim one, split by matching sizes,
            # each piece squared.
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                cat, [2, 3, 5], 1
            )
            return [s**2 for s in split_with_sizes]

        args = [
            torch.randn(2, 2, device=GPU_TYPE),
            torch.randn(2, 3, device=GPU_TYPE),
            torch.randn(2, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 1, 2)

        # cat node has other users
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Same as the good case but also returns the cube of the cat
            # result, giving it another user.
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                cat, [2, 3, 5], 1
            )
            return [s**2 for s in split_with_sizes] + [cat**3]

        args = [
            torch.randn(2, 2, device=GPU_TYPE),
            torch.randn(2, 3, device=GPU_TYPE),
            torch.randn(2, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # cat and split dims are different
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Cat on dim one but split on dim zero.
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                cat, [2, 3, 5], 0
            )
            return [s**2 for s in split_with_sizes]

        args = [
            torch.randn(10, 2, device=GPU_TYPE),
            torch.randn(10, 3, device=GPU_TYPE),
            torch.randn(10, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # cat and split lengths are different
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Cat of three inputs, split into only two pieces.
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = torch.ops.aten.split_with_sizes.default(cat, [5, 5], 1)
            return [s**2 for s in split_with_sizes]

        args = [
            torch.randn(2, 2, device=GPU_TYPE),
            torch.randn(2, 3, device=GPU_TYPE),
            torch.randn(2, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

        # cat input sizes and split sizes are different
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Cat of three inputs, split with the same count but a
            # different ordering of sizes.
            cat = torch.ops.aten.cat.default([a, b, c], 1)
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                cat, [2, 5, 3], 1
            )
            return [s**2 for s in split_with_sizes]

        args = [
            torch.randn(2, 2, device=GPU_TYPE),
            torch.randn(2, 3, device=GPU_TYPE),
            torch.randn(2, 5, device=GPU_TYPE),
        ]
        self.common(fn, args, 0, 0)

    def test_symint_pattern_matching(self):
        # [FUNCTION DOC] Registers a replacement with a custom post grad pass for dynamic
        # shapes: replaces add with an expression involving a symbolic size. Checks the
        # extra_check hook ran, compiled output equals eager, and the saved graph contains a
        # symbolic size node with two users.
        import torch._inductor.config as config
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )

        saved_graph = None

        class _CustomPass(PatternMatcherPass):
            # [CLASS DOC] HELPER: PatternMatcherPass subclass used as the post grad custom pass;
            # applies the patterns to the graph and saves the graph in an enclosing variable.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor with no arguments.
                super().__init__()

            def __call__(self, g: torch.fx.graph.Graph):
                # [FUNCTION DOC] HELPER: Applies all registered patterns to graph g and stores g
                # into the enclosing saved_graph variable.
                self.apply(g)
                nonlocal saved_graph
                saved_graph = g

        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            # define pattern match as custom post grad opt pass
            post_grad_custom_pre_pass=None,
            post_grad_custom_post_pass=_CustomPass(),
        ):

            def add(x, y):
                # [FUNCTION DOC] HELPER: Pattern function: sum of the two inputs.
                return x + y

            # testing that
            def sym_minus(x, y):
                # [FUNCTION DOC] HELPER: Replacement function that computes the sum through
                # subtraction of negated terms plus a symbolic size of y, intended to trace to
                # equivalent math with symbolic sizes.
                return (x - (-y.size(0))) - (y * -1) - y.size(0)

            device = "cpu"
            my_args = [
                torch.empty([8, 1], device=device),
                torch.empty([10], device=device),
            ]

            invoked = False

            def extra_check(match):
                # [FUNCTION DOC] HELPER: Match filter that records it was invoked in an
                # enclosing flag and always accepts.
                nonlocal invoked
                invoked = True
                return True

            register_replacement(
                add,
                sym_minus,
                my_args,
                fwd_only,
                [config.post_grad_custom_post_pass],
                extra_check=extra_check,
            )

            @torch.compile(dynamic=True)
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Compiled function with dynamic shapes returning
                # x plus y.
                return x + y

            x = torch.rand([8, 1])
            y = torch.rand([10])

            self.assertEqual(foo(x, y), x + y)

            self.assertTrue(invoked)
            # we trace out the y.sym_size in replacement
            FileCheck().check("sym_size_int").check_same("num_users=2").check_same(
                "target=torch.ops.aten.sym_size"
            ).run(str(saved_graph))

    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_match_with_mutation(self):
        # [FUNCTION DOC] A graph pattern on add(x, sin(x)) registered in a private
        # PatternMatcherPass is swapped into the pre grad patterns. It must match for the plain
        # function and not match when a mutation (copy_, out= write, manual_seed, no_grad or
        # autocast regions) lies between the sin and the add. Results are checked against eager.
        counter = 0
        test_pass = PatternMatcherPass(pass_name="test")

        @register_graph_pattern(
            CallFunction(
                torch.add, KeywordArg("x"), CallFunction(torch.sin, KeywordArg("x"))
            ),
            pass_dict=test_pass,
        )
        def _test(match, x):
            # [FUNCTION DOC] HELPER: Handler that increments the enclosing match counter and
            # does not rewrite anything.
            nonlocal counter
            counter += 1

        def fn0(x, y):
            # [FUNCTION DOC] HELPER: sin then add with no side effects; expected to match.
            a = torch.sin(x)
            b = torch.add(x, a)
            return b

        def fn1(x, y):
            # [FUNCTION DOC] HELPER: Mutates x in place with copy_ between the sin and the add;
            # expected not to match.
            a = torch.sin(x)
            x.copy_(y)
            b = torch.add(x, a)
            return b

        def fn2(x, y):
            # [FUNCTION DOC] HELPER: Adds under a no_grad context; expected not to match.
            a = torch.sin(x)
            with torch.no_grad():
                b = torch.add(x, a)
            return b

        def fn3(x, y):
            # [FUNCTION DOC] HELPER: Adds under an autocast context for the GPU;
            # expected not to match.
            a = torch.sin(x)
            with torch.autocast(GPU_TYPE):
                b = torch.add(x, a)
            return b

        def fn4(x, y):
            # [FUNCTION DOC] HELPER: Seeds the global RNG between sin and add; expected
            # not to match.
            a = torch.sin(x)
            torch.manual_seed(1234)
            b = torch.add(x, a)
            return b

        def fn5(x, y):
            # [FUNCTION DOC] HELPER: Writes into x through an out= argument between sin and add;
            # expected not to match.
            a = torch.sin(x)
            torch.add(y, 1, out=x)
            b = torch.add(x, a)
            return b

        args = [
            torch.randn(5, 5, device=GPU_TYPE),
            torch.randn(5, 5, device=GPU_TYPE),
        ]

        with (
            unittest.mock.patch(
                "torch._inductor.fx_passes.pre_grad.config.pre_grad_fusion_options",
                {"test": {}},
            ),
            unittest.mock.patch(
                "torch._inductor.fx_passes.pre_grad.PRE_GRAD_FUSIONS",
                [],
            ),
            unittest.mock.patch(
                "torch._inductor.fx_passes.pre_grad.PRE_GRAD_PATTERNS",
                {"test": test_pass},
            ),
        ):
            for fn in (fn0, fn1, fn2, fn3, fn4, fn5):
                counter = 0
                expected = fn(*copy.deepcopy(args))
                actual = torch.compile(fn)(*copy.deepcopy(args))
                # should not match
                self.assertEqual(counter, int(fn is fn0))
                torch.testing.assert_close(actual, expected)

    def test_mutation_region_ids_update_after_replacement_adds_mutation(self):
        # [FUNCTION DOC] A graph pattern replaces sin with sin plus an in-place copy,
        # introducing a mutation. After the pass, mutation_region_id metadata must be
        # recomputed: sin has region zero, copy_ one and the following cos one, and a later pass
        # whose pattern spans the sin to cos edge must find no match.
        insert_mutation_pass = PatternMatcherPass()

        @register_graph_pattern(
            CallFunction(aten.sin.default, KeywordArg("x")),
            pass_dict=insert_mutation_pass,
        )
        def insert_mutation(match: Match, x):
            # [FUNCTION DOC] HELPER: Replacement handler that rewrites a match using
            # replace_by_example with a function doing sin and an in-place self copy, without
            # running functional passes.
            def repl(a):
                # [FUNCTION DOC] HELPER: Replacement body: computes sin of a, copies a onto
                # itself in place, returns the sin result.
                y = torch.sin(a)
                a.copy_(a)
                return y

            match.replace_by_example(repl, [x], run_functional_passes=False)

        later_pass = PatternMatcherPass()
        later_matches = 0

        @register_graph_pattern(
            CallFunction(
                aten.cos.default,
                CallFunction(aten.sin.default, KeywordArg("x")),
            ),
            pass_dict=later_pass,
        )
        def match_across_mutation(match: Match, x):
            # [FUNCTION DOC] HELPER: Handler for the later pass that counts matches in an
            # enclosing variable.
            nonlocal later_matches
            later_matches += 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Cos of sin of x.
            y = torch.sin(x)
            return torch.cos(y)

        gm = make_fx(fn)(torch.randn(4))
        graph = gm.graph

        self.assertEqual(insert_mutation_pass.apply(graph), 1)
        self.assertEqual(later_pass.apply(graph), 0)
        self.assertEqual(later_matches, 0)

        nodes_by_target = {
            node.target: node for node in graph.nodes if node.op == "call_function"
        }
        self.assertEqual(
            nodes_by_target[aten.sin.default].meta["mutation_region_id"], 0
        )
        self.assertEqual(
            nodes_by_target[aten.copy_.default].meta["mutation_region_id"], 1
        )
        self.assertEqual(
            nodes_by_target[aten.cos.default].meta["mutation_region_id"], 1
        )

    def test_pure_replacement_skips_mutation_region_refresh(self):
        # [FUNCTION DOC] A register_replacement of sin by cos (pure) must not call the mutation
        # region refresh function: that function is patched to raise on use, and the pass must
        # still apply exactly one replacement.
        pure_replacement_pass = PatternMatcherPass()

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern: sin of x.
            return torch.sin(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement: cos of x.
            return torch.cos(x)

        register_replacement(
            pattern,
            replacement,
            [torch.randn(4)],
            fwd_only,
            pure_replacement_pass,
        )

        gm = make_fx(pattern)(torch.randn(4))
        with unittest.mock.patch.object(
            pattern_matcher,
            "_mutation_region_refresh",
            side_effect=AssertionError("pure replacements should not refresh"),
        ):
            self.assertEqual(pure_replacement_pass.apply(gm.graph), 1)

    def test_pure_graph_pattern_skips_mutation_region_refresh(self):
        # [FUNCTION DOC] A pure graph pattern handler (does nothing) must not trigger
        # recomputation of mutation region ids: compute_mutation_region_ids is called once
        # beforehand, then patched to raise, and the pass still reports one match.
        pure_graph_pass = PatternMatcherPass()

        @register_graph_pattern(
            CallFunction(aten.sin.default, KeywordArg("x")),
            pass_dict=pure_graph_pass,
        )
        def pure_graph_pattern(match: Match, x):
            # [FUNCTION DOC] HELPER: Handler that does nothing.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Sin of x.
            return torch.sin(x)

        gm = make_fx(fn)(torch.randn(4))
        pattern_matcher.compute_mutation_region_ids(gm.graph)
        with unittest.mock.patch.object(
            pattern_matcher,
            "compute_mutation_region_ids",
            side_effect=AssertionError("pure graph patterns should not refresh"),
        ):
            self.assertEqual(pure_graph_pass.apply(gm.graph), 1)

    def test_mutating_register_replacement_refreshes_mutation_regions(self):
        # [FUNCTION DOC] Like the mutation-introducing graph pattern test but via
        # register_replacement with a mutating replacement; mutation regions must be
        # refreshed afterward (sin zero, copy_ one, cos one) and the later pass must not
        # match across the mutation.
        insert_mutation_pass = PatternMatcherPass()

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern: sin of x.
            return torch.sin(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement: sin of x plus in-place self copy; returns the
            # sin value.
            y = torch.sin(x)
            x.copy_(x)
            return y

        register_replacement(
            pattern,
            replacement,
            [torch.randn(4)],
            fwd_only,
            insert_mutation_pass,
        )

        later_pass = PatternMatcherPass()
        later_matches = 0

        @register_graph_pattern(
            CallFunction(
                aten.cos.default,
                CallFunction(aten.sin.default, KeywordArg("x")),
            ),
            pass_dict=later_pass,
        )
        def match_across_mutation(match: Match, x):
            # [FUNCTION DOC] HELPER: Handler for the later pass that counts matches in an
            # enclosing variable.
            nonlocal later_matches
            later_matches += 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Cos of sin of x.
            y = torch.sin(x)
            return torch.cos(y)

        gm = make_fx(fn)(torch.randn(4))
        graph = gm.graph

        self.assertEqual(insert_mutation_pass.apply(graph), 1)
        self.assertEqual(later_pass.apply(graph), 0)
        self.assertEqual(later_matches, 0)

        nodes_by_target = {
            node.target: node for node in graph.nodes if node.op == "call_function"
        }
        self.assertEqual(
            nodes_by_target[aten.sin.default].meta["mutation_region_id"], 0
        )
        self.assertEqual(
            nodes_by_target[aten.copy_.default].meta["mutation_region_id"], 1
        )
        self.assertEqual(
            nodes_by_target[aten.cos.default].meta["mutation_region_id"], 1
        )

    def test_graph_pattern_mutation_before_match_refreshes_full_graph(self):
        # [FUNCTION DOC] A handler inserts an in-place copy_ before a matched node. After the
        # pass the whole graph's mutation region ids are refreshed: sin region zero, copy_ one,
        # cos, neg and relu one, and a later deep pattern spanning the mutation finds no match.
        insert_mutation_pass = PatternMatcherPass()

        @register_graph_pattern(
            CallFunction(
                aten.relu.default,
                CallFunction(aten.neg.default, KeywordArg("z")),
            ),
            pass_dict=insert_mutation_pass,
        )
        def insert_mutation_before_match(match: Match, z):
            # [FUNCTION DOC] HELPER: Handler that walks to the sin input via the argument chain
            # of z and inserts a copy_ call before z in the match graph.
            sin_node = z.args[0]
            x = sin_node.args[0]
            with match.graph.inserting_before(z):
                match.graph.call_function(aten.copy_.default, (x, x))

        later_pass = PatternMatcherPass()
        later_matches = 0

        @register_graph_pattern(
            CallFunction(
                aten.relu.default,
                CallFunction(
                    aten.neg.default,
                    CallFunction(
                        aten.cos.default,
                        CallFunction(aten.sin.default, KeywordArg("x")),
                    ),
                ),
            ),
            pass_dict=later_pass,
        )
        def match_across_mutation(match: Match, x):
            # [FUNCTION DOC] HELPER: Handler for the later pass that counts matches in an
            # enclosing variable.
            nonlocal later_matches
            later_matches += 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Relu of neg of cos of sin of x.
            return torch.relu(torch.neg(torch.cos(torch.sin(x))))

        gm = make_fx(fn)(torch.randn(4))
        graph = gm.graph

        self.assertEqual(insert_mutation_pass.apply(graph), 1)
        self.assertEqual(later_pass.apply(graph), 0)
        self.assertEqual(later_matches, 0)

        nodes_by_target = {
            node.target: node for node in graph.nodes if node.op == "call_function"
        }
        self.assertEqual(
            nodes_by_target[aten.sin.default].meta["mutation_region_id"], 0
        )
        self.assertEqual(
            nodes_by_target[aten.copy_.default].meta["mutation_region_id"], 1
        )
        self.assertEqual(
            nodes_by_target[aten.cos.default].meta["mutation_region_id"], 1
        )
        self.assertEqual(
            nodes_by_target[aten.neg.default].meta["mutation_region_id"], 1
        )
        self.assertEqual(
            nodes_by_target[aten.relu.default].meta["mutation_region_id"], 1
        )

    def test_remove_pointless_clones(self):
        # [FUNCTION DOC] A clone after mm is removed: the compiled code returns buf0 directly
        # and does not use the CPP async compile path.
        @torch.compile(fullgraph=True)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Compiled function returning a clone of mm of a and b.
            return torch.mm(a, b).clone()

        _, (code) = run_and_get_code(fn, torch.randn(8, 8), torch.randn(8, 8))
        # clone would create a buf1
        self.assertIn("return (buf0, )", code[0])
        self.assertNotIn("async_compile.cpp", code[0])

    def test_unfuse_bias_addmm(self):
        # [FUNCTION DOC] Unfuse behavior of bias addmm: a plain addmm stays an extern addmm;
        # addmm followed by gelu, or followed by unsqueeze and gelu (the view path), is unfused
        # so no extern addmm remains.
        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

        @torch.compile()
        def fn(inp, a, b):
            # [FUNCTION DOC] HELPER: Compiled addmm of inp with a and b.
            return torch.ops.aten.addmm(inp, a, b)

        _, (code) = run_and_get_code(fn, args[0], args[1], args[2])
        FileCheck().check("extern_kernels.addmm(").run(code[0])

        @torch.compile()
        def fn2(inp, a, b):
            # [FUNCTION DOC] HELPER: Compiled gelu of an addmm.
            return torch.nn.functional.gelu(torch.ops.aten.addmm(inp, a, b))

        _, (code) = run_and_get_code(fn2, args[0], args[1], args[2])
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

        @torch.compile()
        def fn2(inp, a, b):
            # [FUNCTION DOC] HELPER: Compiled gelu of an addmm result after unsqueeze.
            return torch.nn.functional.gelu(
                torch.ops.aten.addmm(inp, a, b).unsqueeze(0)
            )

        # hit the view path
        _, (code) = run_and_get_code(fn2, args[0], args[1], args[2])
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    def test_unfuse_broadcast_bias_baddbmm(self):
        # [FUNCTION DOC] Broadcast bias baddbmm: plain baddbmm stays extern baddbmm; with gelu
        # epilogue, or an explicitly expanded bias with zero stride and relu, it is unfused into
        # extern bmm. Results are compared against eager baddbmm.
        args = [
            torch.randn(4, 1, 8, device=GPU_TYPE),
            torch.randn(4, 6, 5, device=GPU_TYPE),
            torch.randn(4, 5, 8, device=GPU_TYPE),
        ]

        @torch.compile()
        def fn(inp, a, b):
            # [FUNCTION DOC] HELPER: Compiled baddbmm.
            return torch.ops.aten.baddbmm(inp, a, b)

        actual, (code) = run_and_get_code(fn, args[0], args[1], args[2])
        self.assertEqual(actual, torch.baddbmm(*args))
        FileCheck().check("extern_kernels.baddbmm(").run(code[0])

        @torch.compile()
        def fn2(inp, a, b):
            # [FUNCTION DOC] HELPER: Compiled gelu of baddbmm.
            return torch.nn.functional.gelu(torch.ops.aten.baddbmm(inp, a, b))

        actual, (code) = run_and_get_code(fn2, args[0], args[1], args[2])
        self.assertEqual(actual, torch.nn.functional.gelu(torch.baddbmm(*args)))
        FileCheck().check_not("extern_kernels.baddbmm(").check(
            "extern_kernels.bmm("
        ).run(code[0])

        @torch.compile()
        def fn3(inp, a, b):
            # [FUNCTION DOC] HELPER: Expands the bias to 4 by 6 by 8 then baddbmm and relu.
            inp = inp.expand(4, 6, 8)
            return torch.ops.aten.baddbmm(inp, a, b).relu()

        actual, (code) = run_and_get_code(fn3, args[0], args[1], args[2])
        expanded_bias = args[0].expand(4, 6, 8)
        self.assertEqual(expanded_bias.stride(1), 0)
        self.assertEqual(actual, torch.baddbmm(expanded_bias, args[1], args[2]).relu())
        FileCheck().check_not("extern_kernels.baddbmm(").check(
            "extern_kernels.bmm("
        ).run(code[0])

    def test_unfuse_broadcast_bias_baddbmm_alpha_beta(self):
        # [FUNCTION DOC] baddbmm with alpha 0.8 and beta 0.2 and a broadcast bias followed by
        # relu is unfused into extern bmm (no extern baddbmm) and matches eager.
        args = [
            torch.randn(4, 1, 8, device=GPU_TYPE),
            torch.randn(4, 6, 5, device=GPU_TYPE),
            torch.randn(4, 5, 8, device=GPU_TYPE),
        ]

        @torch.compile()
        def fn(inp, a, b):
            # [FUNCTION DOC] HELPER: Relu of baddbmm with alpha and beta set.
            return torch.nn.functional.relu(
                torch.ops.aten.baddbmm(inp, a, b, alpha=0.8, beta=0.2)
            )

        actual, (code) = run_and_get_code(fn, args[0], args[1], args[2])
        expected = torch.nn.functional.relu(
            torch.baddbmm(args[0], args[1], args[2], alpha=0.8, beta=0.2)
        )
        self.assertEqual(actual, expected)
        FileCheck().check_not("extern_kernels.baddbmm(").check(
            "extern_kernels.bmm("
        ).run(code[0])

    def test_preserve_accumulator_addmm_with_pointwise(self):
        # [FUNCTION DOC] When the bias has a pointwise op (sin) applied before an addmm and a
        # gelu consumer, the addmm is kept as an extern addmm; result is compared with eager.
        args = [
            torch.randn(10, 20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

        @torch.compile()
        def fn(inp, a, b):
            # [FUNCTION DOC] HELPER: Applies sin to inp then gelu of addmm with it.
            inp = torch.sin(inp)
            return torch.nn.functional.gelu(torch.ops.aten.addmm(inp, a, b))

        actual, (code) = run_and_get_code(fn, args[0], args[1], args[2])
        self.assertEqual(
            actual, torch.nn.functional.gelu(torch.addmm(torch.sin(args[0]), *args[1:]))
        )
        FileCheck().check("extern_kernels.addmm(").run(code[0])

    def test_unfuse_same_shape_leaf_bias_addmm(self):
        # [FUNCTION DOC] A module that adds a same-shape parameter bias after mm then relu:
        # compiled output equals eager and the add is not turned into an extern addmm.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 10 by 20 parameter bias on the GPU.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the random bias parameter.
                super().__init__()
                self.bias = torch.nn.Parameter(torch.randn(10, 20, device=GPU_TYPE))

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns relu of mm of a and b plus the bias.
                return (torch.mm(a, b) + self.bias).relu()

        mod = Mod()
        args = [
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

        actual, (code) = run_and_get_code(torch.compile(mod), args[0], args[1])
        self.assertEqual(actual, mod(*args))
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    def test_addmm_fusion_extra_check_without_beta_kwarg(self):
        # [FUNCTION DOC] Builds an FX graph by hand with mm followed by add (both operand
        # orders) with fake tensors on an mps device and asserts that the post grad function
        # should_prefer_unfused_addmm returns True for a match namespace with positional args
        # and an inp keyword, even without a beta keyword.
        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        mat1 = graph.placeholder("mat1")
        mat2 = graph.placeholder("mat2")
        mm = graph.call_function(torch.ops.aten.mm.default, (mat1, mat2))
        mm_plus_inp = graph.call_function(torch.ops.aten.add.Tensor, (mm, inp))
        graph.call_function(torch.ops.aten.relu.default, (mm_plus_inp,))
        inp_plus_mm = graph.call_function(torch.ops.aten.add.Tensor, (inp, mm))
        graph.call_function(torch.ops.aten.relu.default, (inp_plus_mm,))

        mps_device = torch.device("mps")
        with torch._subclasses.FakeTensorMode():
            inp.meta["val"] = torch.empty(10, 20, device=mps_device)
            mat1.meta["val"] = torch.empty(10, 15, device=mps_device)
            mat2.meta["val"] = torch.empty(15, 20, device=mps_device)
            mm_plus_inp.meta["val"] = torch.empty(10, 20, device=mps_device)
            inp_plus_mm.meta["val"] = torch.empty(10, 20, device=mps_device)

        for output in (mm_plus_inp, inp_plus_mm):
            match = types.SimpleNamespace(
                args=[mat1, mat2],
                kwargs={"inp": inp},
                output_node=lambda output=output: output,
            )
            self.assertTrue(
                torch._inductor.fx_passes.post_grad.should_prefer_unfused_addmm(match)
            )

    def test_unfuse_expanded_bias_addmm(self):
        # [FUNCTION DOC] Bias expanded from a vector to the output shape (zero stride on dim
        # zero) followed by relu: addmm is unfused, so no extern addmm appears; the result
        # equals eager.
        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

        @torch.compile()
        def fn(bias, a, b):
            # [FUNCTION DOC] HELPER: Unsqueezes and expands bias to 10 by 20, then relu
            # of addmm.
            bias = bias.unsqueeze(0).expand(10, 20)
            return torch.ops.aten.addmm(bias, a, b).relu()

        actual, (code) = run_and_get_code(fn, args[0], args[1], args[2])
        bias = args[0].unsqueeze(0).expand(10, 20)
        self.assertEqual(bias.stride(0), 0)
        self.assertEqual(actual, torch.addmm(bias, args[1], args[2]).relu())
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    @parametrize("inplace", [False, True])
    def test_accumulator_addmm_loop_does_not_delay_pointwise(self, inplace):
        # [FUNCTION DOC] Parametrized by inplace: a loop of addmm (in-place or functional)
        # followed by cos on an accumulator must keep each addmm as an extern call (three of
        # them, no extern mm) and give eager-equal results.
        def fn(x, ws):
            # [FUNCTION DOC] HELPER: Starts from zeros like x and, for each weight, accumulates
            # addmm of the buffer with the weight used as both matrices, then applies cos.
            buf = torch.zeros_like(x)
            for w in ws:
                if inplace:
                    buf.addmm_(w, w)
                else:
                    buf = torch.addmm(buf, w, w)
                buf = torch.cos(buf)
            return buf

        args = [
            torch.randn(32, 32, device=GPU_TYPE),
            [torch.randn(32, 32, device=GPU_TYPE) for _ in range(3)],
        ]

        actual, (code) = run_and_get_code(torch.compile(fn), args[0], args[1])
        self.assertEqual(actual, fn(*args))
        self.assertEqual(code[0].count("extern_kernels.addmm("), 3)
        self.assertNotIn("extern_kernels.mm(", code[0])

    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @inductor_config.patch(
        {
            "fx_graph_remote_cache": False,
            "keep_addmm_fused_for_half_dtypes": True,
        }
    )
    def test_unfuse_bias_addmm_half_dtypes(self, dtype):
        # [FUNCTION DOC] Parametrized over bfloat16 and float16 with
        # keep_addmm_fused_for_half_dtypes on: addmm with a pointwise consumer is kept fused as
        # an extern addmm to avoid extra truncation.
        args = [
            torch.randn(20, device=GPU_TYPE, dtype=dtype),
            torch.randn(10, 15, device=GPU_TYPE, dtype=dtype),
            torch.randn(15, 20, device=GPU_TYPE, dtype=dtype),
        ]

        # addmm with pointwise consumer should not be unfused for half dtypes
        # to avoid precision loss from extra truncation at the mm output
        @torch.compile()
        def fn(inp, a, b):
            # [FUNCTION DOC] HELPER: Compiled gelu of addmm.
            return torch.nn.functional.gelu(torch.ops.aten.addmm(inp, a, b))

        _, (code) = run_and_get_code(fn, args[0], args[1], args[2])
        FileCheck().check("extern_kernels.addmm(").run(code[0])

    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @inductor_config.patch(
        {
            "fx_graph_remote_cache": False,
            "keep_addmm_fused_for_half_dtypes": False,
        }
    )
    def test_unfuse_bias_addmm_half_dtypes_when_flag_disabled(self, dtype):
        # [FUNCTION DOC] Same as the half dtype test but with keep_addmm_fused_for_half_dtypes
        # off: the addmm is unfused and no extern addmm remains.
        args = [
            torch.randn(20, device=GPU_TYPE, dtype=dtype),
            torch.randn(10, 15, device=GPU_TYPE, dtype=dtype),
            torch.randn(15, 20, device=GPU_TYPE, dtype=dtype),
        ]

        @torch.compile()
        def fn(inp, a, b):
            # [FUNCTION DOC] HELPER: Compiled gelu of addmm.
            return torch.nn.functional.gelu(torch.ops.aten.addmm(inp, a, b))

        _, (code) = run_and_get_code(fn, args[0], args[1], args[2])
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @inductor_config.patch(
        {
            "fx_graph_remote_cache": False,
            "keep_addmm_fused_for_half_dtypes": True,
        }
    )
    @unittest.skipIf(
        GPU_TYPE != "xpu",
        "narrowing-cast unfuse is XPU-only; CUDA/ROCm keep addmm fused",
    )
    def test_unfuse_bias_addmm_half_dtypes_narrowing_cast(self, dtype):
        # [FUNCTION DOC] Skipped unless the GPU is xpu. For a float32 bias narrowed to half
        # dtype before addmm, with keep fused on, the addmm is still unfused so the pointwise
        # kernel loads the float32 bias; asserts no extern addmm.
        # When bias is fp32 and cast to a half dtype (e.g. AMP), unfusing
        # lets the Triton pointwise kernel load the fp32 bias directly,
        # preserving precision instead of truncating before fused addmm.
        bias_fp32 = torch.randn(20, device=GPU_TYPE, dtype=torch.float32)
        args = [
            torch.randn(10, 15, device=GPU_TYPE, dtype=dtype),
            torch.randn(15, 20, device=GPU_TYPE, dtype=dtype),
        ]

        @torch.compile()
        def fn(bias, a, b):
            # [FUNCTION DOC] HELPER: Casts bias to the half dtype then gelu of addmm.
            bias_half = bias.to(dtype)
            return torch.nn.functional.gelu(torch.ops.aten.addmm(bias_half, a, b))

        _, (code) = run_and_get_code(fn, bias_fp32, args[0], args[1])
        # Should be unfused (mm, not addmm) because bias is a narrowing cast
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @inductor_config.patch(
        {
            "fx_graph_remote_cache": False,
            "keep_addmm_fused_for_half_dtypes": True,
        }
    )
    def test_unfuse_bias_addmm_half_dtypes_narrowing_cast_numerics(self, dtype):
        # [FUNCTION DOC] Numerics check of unfusing narrowing-cast biases: builds a stack of
        # eight residual linear layers, computes a float64 reference, runs eager and compiled
        # under autocast with the half dtype and requires the compiled RMSE to be within 3x of
        # the eager RMSE (plus a small epsilon).
        # Verify that unfusing a narrowing-cast bias produces results whose
        # RMSE vs fp64 stays within 3x of eager's RMSE (the torchbench
        # accuracy check threshold for half dtypes).
        torch.manual_seed(42)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Eight 256 by 256 linear layers with biases near 1.0 and scaled
            # down weights, applied with residual connections.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the module list of linear layers and overwrites
                # each bias and weight data in place.
                super().__init__()
                self.linears = torch.nn.ModuleList(
                    [torch.nn.Linear(256, 256) for _ in range(8)]
                )
                for l in self.linears:
                    l.bias.data = torch.randn(256, device=GPU_TYPE) * 0.01 + 1.0
                    l.weight.data *= 0.1

            def forward(self, x):
                # [FUNCTION DOC] HELPER: For each layer sets x to the layer output plus x,
                # returns the result.
                for l in self.linears:
                    x = l(x) + x  # residual add is pointwise use
                return x

        model = Model().to(GPU_TYPE)
        x = torch.randn(32, 256, device=GPU_TYPE)

        def rmse(a, b):
            # [FUNCTION DOC] HELPER: Root mean square error between two tensors computed in
            # float32, returned as a python float.
            return torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item()

        # fp64 ground truth
        model_fp64 = Model().double().to(GPU_TYPE)
        model_fp64.load_state_dict(
            {k: v.double() for k, v in model.state_dict().items()}
        )
        with torch.no_grad():
            fp64_ref = model_fp64(x.double())

        # Eager under AMP
        with torch.no_grad(), torch.autocast(GPU_TYPE, dtype=dtype):
            eager_out = model(x)

        # Compiled under AMP
        torch._dynamo.reset()
        compiled = torch.compile(model)
        with torch.no_grad(), torch.autocast(GPU_TYPE, dtype=dtype):
            compiled_out = compiled(x)

        eager_err = rmse(fp64_ref, eager_out)
        compiled_err = rmse(fp64_ref, compiled_out)
        # compiled should not be significantly worse than eager
        self.assertLessEqual(compiled_err, 3.0 * eager_err + 1e-4)

    def test_addmm_alpha_beta_with_pointwise(self):
        # [FUNCTION DOC] Issue 167313: addmm with alpha 0.8 and beta 0.2 followed by relu must
        # match eager and be unfused (no extern addmm); the same check is done for default alpha
        # and beta.
        # Test that addmm with alpha/beta != 1 is unfused correctly with pointwise ops
        # See https://github.com/pytorch/pytorch/issues/167313
        x = torch.rand(2, device=GPU_TYPE)
        a = torch.rand(2, 3, device=GPU_TYPE)
        b = torch.rand(3, 2, device=GPU_TYPE)

        def f(x, a, b):
            # [FUNCTION DOC] HELPER: Relu of addmm with scaling arguments.
            return torch.nn.functional.relu(torch.addmm(x, a, b, alpha=0.8, beta=0.2))

        fc = torch.compile(f)

        expected = f(x, a, b)
        actual = fc(x, a, b)

        # The compiled version should produce the same result as eager
        torch.testing.assert_close(actual, expected)

        # Verify that addmm is unfused (should not use extern_kernels.addmm)
        # The pattern should be replaced with beta * x + alpha * (a @ b)
        _, (code) = run_and_get_code(fc, x, a, b)
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

        # Test with alpha=1, beta=1 (default) - should also unfuse
        def f_default(x, a, b):
            # [FUNCTION DOC] HELPER: Relu of default addmm.
            return torch.nn.functional.relu(torch.addmm(x, a, b))

        fc_default = torch.compile(f_default)
        expected_default = f_default(x, a, b)
        actual_default = fc_default(x, a, b)

        torch.testing.assert_close(actual_default, expected_default)

        # Should unfuse and not use extern_kernels.addmm
        _, (code) = run_and_get_code(fc_default, x, a, b)
        FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    def test_serialized_patterns_up_to_date(self):
        # [FUNCTION DOC] Verifies that serialized precompiled patterns are in sync with their
        # source: after loading patterns (with the generation env var removed) it regenerates
        # each pattern from its search function with example inputs remapped into a fresh fake
        # mode, compares pretty printed forms and requires pattern equality. A mismatch message
        # tells the developer to rerun the pattern generation script.
        import torch.utils._pytree as pytree
        from torch._inductor.fx_passes import joint_graph
        from torch._inductor.pattern_matcher import _known_precompiled_patterns

        # Ensure the patterns are loaded
        os.environ.pop("PYTORCH_GEN_PATTERNS", None)
        joint_graph.lazy_init()

        with torch._subclasses.FakeTensorMode() as mode:
            for precompiled in _known_precompiled_patterns:
                # Because the example_inputs were saved as fake tensors in a
                # different FakeTensorMode we need to update them to our
                # FakeTensorMode().
                def remap_fake_tensor(x):
                    # [FUNCTION DOC] HELPER: Maps a real tensor to a fake tensor in the current
                    # fake mode; other values are returned unchanged.
                    if isinstance(x, torch.Tensor):
                        return torch._subclasses.FakeTensor.from_tensor(x, mode)
                    return x

                example_inputs = pytree.tree_map(
                    remap_fake_tensor, precompiled.example_inputs
                )

                pattern = gen_pattern(
                    precompiled.search_fn,
                    example_inputs,
                    precompiled.trace_fn,
                    precompiled.scalar_workaround,
                )
                pattern_pp = PatternPrettyPrinter.run(pattern)

                self.assertEqual(
                    pattern_pp,
                    PatternPrettyPrinter.run(precompiled.search_fn_pattern),
                    msg=lambda msg: f"{msg}\nFound mismatched pattern {precompiled.search_fn.__name__}. Run torchgen/fuse/gen_patterns.py",
                )

                # Since we've already checked that the serialized patterns match
                # lets verify the serializer by ensuring the generated patterns
                # also match (since search_fn_pattern is the serialized version
                # of search_fn).
                self.assertTrue(pattern.pattern_eq(precompiled.search_fn_pattern))

    @xfailIfSM89
    @inductor_config.patch(
        {
            "triton.unique_kernel_names": "original_aten",
            "fx_graph_remote_cache": False,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_original_aten_preserved_split_addmm(self):
        # [FUNCTION DOC] Marked as expected failure on SM89. With the TRITON gemm backend and
        # original_aten kernel names, a bias addmm with a sin epilogue is decomposed into mm and
        # add (one pattern match) and the generated template kernel name contains addmm; no
        # extern addmm is called.
        # bias addmm -> elementwise should be decomposed into
        # mm -> add -> elementwise.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Sin of addmm with bias z.
            return torch.addmm(z, x, y).sin()

        args = [
            torch.randn(16, 24, device=GPU_TYPE),
            torch.randn(24, 32, device=GPU_TYPE),
            torch.randn(32, device=GPU_TYPE),
        ]

        counters.clear()

        opt_fn = torch.compile(fn, mode="max-autotune")
        ret, code = run_and_get_code(opt_fn, *args)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)

        # The mm kernel should use a template (because we set max_autotune_gemm_backends = TRITON).
        # Its name should contain `addmm` because `addmm` was the original aten op where the mm came from.
        FileCheck().check_not("extern_kernels.addmm(").check(
            "def triton_tem_fused_addmm"
        ).run(code[0])

    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_match_equivalent_function_invocations1(self):
        # [FUNCTION DOC] A graph pattern matching addmm with optional beta and alpha keywords is
        # registered in a test pass appended to post grad patterns and must match addmm calls
        # written without keywords, with beta, and with beta and alpha (one match each); the
        # replacement removes extern addmm.
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

        def f0(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm with default scaling.
            return torch.ops.aten.addmm(inp, a, b)

        def f1(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm with beta given.
            return torch.ops.aten.addmm(inp, a, b, beta=1.0)

        def f2(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm with beta and alpha given.
            return torch.ops.aten.addmm(inp, a, b, beta=1.0, alpha=1.0)

        # This graph pattern should successfully match all of the above functions
        @register_graph_pattern(
            CallFunction(
                torch.ops.aten.addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            pass_dict=test_pass,
        )
        def addmm_replacement(match: Match, inp, mat1, mat2, beta, alpha):
            # [FUNCTION DOC] HELPER: Handler that counts the match and replaces it, under the
            # fake mode, with a matmul scaled by alpha plus the input scaled by beta.
            nonlocal counter
            counter += 1

            def repl(inp, x1, x2):
                # [FUNCTION DOC] HELPER: Computes the matmul times alpha plus inp times beta.
                return (x1 @ x2) * alpha + inp * beta

            with V.fake_mode:
                match.replace_by_example(repl, [inp, mat1, mat2])

        with unittest.mock.patch(
            "torch._inductor.fx_passes.post_grad.pass_patterns",
            torch._inductor.fx_passes.post_grad.pass_patterns + [test_pass],
        ):
            for fn in (f0, f1, f2):
                counter = 0
                expected = fn(*copy.deepcopy(args))
                opt_fn = torch.compile(fn)
                actual, (code) = run_and_get_code(opt_fn, args[0], args[1], args[2])
                # pattern should match
                self.assertEqual(counter, 1)
                self.assertEqual(actual, expected)
                # addmm should be replaced
                FileCheck().check_not("extern_kernels.addmm(").run(code[0])

    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_replace_by_example_in_pre_grad(self):
        # [FUNCTION DOC] Same graph pattern as the previous test but applied in the pre grad
        # stage through the pre_grad_custom_pass option with early timing; the handler runs once
        # and output matches eager.
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

        def fn(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm call.
            return torch.ops.aten.addmm(inp, a, b)

        # This graph pattern should successfully match all of the above functions
        @register_graph_pattern(
            CallFunction(
                torch.ops.aten.addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            pass_dict=test_pass,
        )
        def addmm_replacement(match: Match, inp, mat1, mat2, beta, alpha):
            # [FUNCTION DOC] HELPER: Handler that counts the match and replaces it with a matmul
            # scaled by alpha plus the input scaled by beta.
            nonlocal counter
            counter += 1

            def repl(inp, x1, x2):
                # [FUNCTION DOC] HELPER: Computes the matmul times alpha plus inp times beta.
                return (x1 @ x2) * alpha + inp * beta

            match.replace_by_example(repl, [inp, mat1, mat2])

        with inductor_config.patch(
            pre_grad_custom_pass=lambda *args: test_pass.apply(*args),
            pre_grad_pass_timing="early",
        ):
            counter = 0
            expected = fn(*copy.deepcopy(args))
            opt_fn = torch.compile(fn)
            actual, (code) = run_and_get_code(opt_fn, args[0], args[1], args[2])
            # pattern should match
            self.assertEqual(counter, 1)
            self.assertEqual(actual, expected)

    def test_addmm_dtype_mismatch(self):
        # [FUNCTION DOC] A float16 linear without bias followed by adding a float32
        # tensor must not be turned into addmm (dtype mismatch); output equals eager and
        # the code has no addmm.
        a = torch.nn.Linear(1024, 1024, bias=False).to(GPU_TYPE)
        a = a.to(dtype=torch.float16)

        w = torch.randn(1024, 1024, device=GPU_TYPE)

        def func():
            # [FUNCTION DOC] HELPER: Runs the linear on float16 ones then adds the float32 w.
            x = torch.ones(1024, 1024, device=GPU_TYPE, dtype=torch.float16)
            x = a(x)
            x = x + w
            return x

        actual, (code) = run_and_get_code(torch.compile(func))
        self.assertEqual(actual, func())
        FileCheck().check_not("addmm").run(code[0])

    def test_replace_mul_zero(self):
        # [FUNCTION DOC] Checks the multiply-by-zero replacement: for integer inputs the
        # multiplication by zero is folded away so no kernel runs; for floats it must
        # stay (so nan or inf times zero gives nan) and a kernel run appears. Results
        # compared with eager.
        def test(x, y):
            # [FUNCTION DOC] HELPER: x plus y times zero.
            return x + (y * 0)

        # For integer dtypes, x * 0 is uniformly 0, so the multiply is folded
        # away entirely and no kernel needs to run.
        xi = torch.randint(0, 10, [256], device=GPU_TYPE)
        yi = torch.randint(0, 10, [256], device=GPU_TYPE)
        out, code = run_and_get_code(torch.compile(test), xi, yi)
        FileCheck().check_not(".run").run(code[0])
        self.assertEqual(out, test(xi, yi))

        # For floating point dtypes, x * 0 must NOT be folded to 0: nan * 0 == nan
        # and (+/-inf) * 0 == nan, so the multiply has to run to match eager.
        xf = torch.rand([8], device=GPU_TYPE)
        yf = torch.tensor(
            [float("nan"), float("inf"), float("-inf"), 0.0, 1.0, -1.0, 2.0, -3.0],
            device=GPU_TYPE,
        )
        out, code = run_and_get_code(torch.compile(test), xf, yf)
        FileCheck().check(".run").run(code[0])
        self.assertEqual(out, test(xf, yf))

    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_match_equivalent_function_invocations2(self):
        # [FUNCTION DOC] Pattern with no beta or alpha keywords registered in a test pass:
        # matches the plain addmm call only (one match for each function individually since the
        # counter is reset per function).
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

        def f0(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm with default scaling.
            return torch.ops.aten.addmm(inp, a, b)

        def f1(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm with beta given.
            return torch.ops.aten.addmm(inp, a, b, beta=1.0)

        def f2(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm with beta and alpha given.
            return torch.ops.aten.addmm(inp, a, b, beta=1.0, alpha=1.0)

        # This graph pattern should only match f0
        @register_graph_pattern(
            CallFunction(torch.ops.aten.addmm, Arg(), Arg(), Arg()),
            pass_dict=test_pass,
        )
        def addmm_replacement(match: Match, inp, mat1, mat2):
            # [FUNCTION DOC] HELPER: Handler that counts the match and replaces it with matmul
            # plus input under fake mode.
            nonlocal counter
            counter += 1

            def repl(inp, x1, x2):
                # [FUNCTION DOC] HELPER: Computes matmul plus inp.
                return x1 @ x2 + inp

            with V.fake_mode:
                match.replace_by_example(repl, [inp, mat1, mat2])

        with unittest.mock.patch(
            "torch._inductor.fx_passes.post_grad.pass_patterns",
            torch._inductor.fx_passes.post_grad.pass_patterns + [test_pass],
        ):
            for fn in (f0, f1, f2):
                counter = 0
                expected = fn(*copy.deepcopy(args))
                actual = torch.compile(fn)(*copy.deepcopy(args))
                self.assertEqual(counter, 1)
                torch.testing.assert_close(actual, expected)

    def test_input_output_same(self):
        # [FUNCTION DOC] register_replacement where the pattern returns one of its
        # inputs as an output: add replaced by mul pair. With a custom post grad pass
        # that applies the patterns, sorts and removes dead code, the compiled code has
        # neither add nor mul remaining.
        def pattern(x, y):
            # [FUNCTION DOC] HELPER: Pattern: returns the sum and the first input.
            out1 = torch.add(x, y)
            return out1, x

        def replace(x, y):
            # [FUNCTION DOC] HELPER: Replacement: returns the product and the product times y.
            out1 = torch.mul(x, y)
            out2 = torch.mul(out1, y)
            return out1, out2

        my_patterns = PatternMatcherPass()
        inputs = (torch.ones(3, 3), torch.ones(3, 3))
        register_replacement(pattern, replace, inputs, fwd_only, my_patterns)

        def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
            # [FUNCTION DOC] HELPER: Post grad pass that applies the patterns, stable
            # topologically sorts the graph, removes dead code and returns the graph.
            _ = my_patterns.apply(graph)
            stable_topological_sort(graph)
            graph.eliminate_dead_code()
            return graph

        @torch.compile(
            options={
                "post_grad_custom_post_pass": custom_pass,
            }
        )
        def f(x, y):
            # [FUNCTION DOC] HELPER: Add, then subtract x.
            res = torch.add(x, y)
            sub = torch.sub(res, x)
            return sub

        test, (code,) = run_and_get_code(f, *(torch.ones(3, 3), torch.ones(3, 3)))

        self.assertTrue("aten.add.default" not in code)
        self.assertTrue("aten.mul.default" not in code)

    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_match_equivalent_function_invocations3(self):
        # [FUNCTION DOC] Pattern with only a beta keyword registered in a test pass; counter is
        # checked to be one for each function individually.
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

        def f0(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm with default scaling.
            return torch.ops.aten.addmm(inp, a, b)

        def f1(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm with beta given.
            return torch.ops.aten.addmm(inp, a, b, beta=1.0)

        def f2(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm with beta and alpha given.
            return torch.ops.aten.addmm(inp, a, b, beta=1.0, alpha=1.0)

        # This graph pattern should only match f1
        @register_graph_pattern(
            CallFunction(
                torch.ops.aten.addmm, Arg(), Arg(), Arg(), beta=KeywordArg("beta")
            ),
            pass_dict=test_pass,
        )
        def addmm_replacement(match: Match, inp, mat1, mat2, beta):
            # [FUNCTION DOC] HELPER: Handler that counts the match and replaces it with matmul
            # plus input under fake mode.
            nonlocal counter
            counter += 1

            def repl(inp, x1, x2):
                # [FUNCTION DOC] HELPER: Computes matmul plus inp.
                return x1 @ x2 + inp

            with V.fake_mode:
                match.replace_by_example(repl, [inp, mat1, mat2])

        with unittest.mock.patch(
            "torch._inductor.fx_passes.post_grad.pass_patterns",
            torch._inductor.fx_passes.post_grad.pass_patterns + [test_pass],
        ):
            for fn in (f0, f1, f2):
                counter = 0
                expected = fn(*copy.deepcopy(args))
                actual = torch.compile(fn)(*copy.deepcopy(args))
                self.assertEqual(counter, 1)
                torch.testing.assert_close(actual, expected)

    def test_stable_topological_sort(self):
        # [FUNCTION DOC] Checks stable_topological_sort keeps already sorted graphs
        # unchanged, keeps placeholder order when they are reversed relative to
        # declaration, and moves an out-of-order node to after its dependencies while
        # keeping the relative order of others.
        def fn1(a, b):
            # [FUNCTION DOC] HELPER: Adds its two arguments.
            return a + b

        graph = torch.fx.Graph()
        a = graph.placeholder("x")
        b = graph.placeholder("y")
        c = graph.call_function(fn1, (a, b))
        stable_topological_sort(graph)
        self.assertEqual(list(graph.nodes), [a, b, c])

        graph = torch.fx.Graph()
        b = graph.placeholder("y")
        a = graph.placeholder("x")
        c = graph.call_function(fn1, (a, b))
        stable_topological_sort(graph)
        self.assertEqual(list(graph.nodes), [b, a, c])

        graph = torch.fx.Graph()
        a = graph.placeholder("x")
        b = graph.placeholder("y")
        c = graph.call_function(fn1, (b, a))
        c.append(a)
        stable_topological_sort(graph)
        self.assertEqual(list(graph.nodes), [b, a, c])

    def test_scaled_softmax(self):
        # [FUNCTION DOC] Scaled softmax pattern: multiply or divide by a scalar, broadcastable
        # tensor scale, or tensor with a different leading dim before a softmax over dim zero is
        # matched (1 match, 3 nodes; 4 nodes for bfloat16 with a float reference). A scale
        # varying along the softmax dim is not matched.
        def mul_softmax(a, b):
            # [FUNCTION DOC] HELPER: Softmax over dim zero of a times b.
            return F.softmax(a * b, dim=0)

        def div_softmax(x, inv_scale):
            # [FUNCTION DOC] HELPER: Softmax over dim zero of x divided by inv_scale.
            return F.softmax(x / inv_scale, dim=0)

        x = torch.randn(10, 10)
        scale = 1e6
        inv_scale = 1 / scale
        self.common(mul_softmax, (x, scale), 1, 3)
        self.common(mul_softmax, (scale, x), 1, 3)
        self.common(div_softmax, (x, inv_scale), 1, 3)

        scale = torch.randn(10) * 1e6
        inv_scale = 1 / scale
        self.common(mul_softmax, (x, scale), 1, 3)
        self.common(mul_softmax, (scale, x), 1, 3)
        self.common(div_softmax, (x, inv_scale), 1, 3)

        scale = torch.randn(1, 10) * 1e6
        inv_scale = 1 / scale
        self.common(mul_softmax, (x, scale), 1, 3)
        self.common(mul_softmax, (scale, x), 1, 3)
        self.common(div_softmax, (x, inv_scale), 1, 3)

        # Test matching with type promotion
        x = torch.randn(10, 10, dtype=torch.bfloat16)
        scale = torch.randn(10, dtype=torch.bfloat16) * 1e6
        inv_scale = 1 / scale
        self.common(mul_softmax, (x, scale), 1, 4, reference_in_float=True)
        self.common(mul_softmax, (scale, x), 1, 4, reference_in_float=True)
        self.common(div_softmax, (x, inv_scale), 1, 4, reference_in_float=True)

        # No match if scale changes in softmax dim
        scale = torch.randn(10, 10)
        self.common(mul_softmax, (x, scale), 0, 0)
        self.common(mul_softmax, (scale, x), 0, 0)
        self.common(div_softmax, (x, scale), 0, 0)

    @inductor_config.patch(fx_graph_cache=False)
    def test_scaled_softmax_nonfinite_matches_eager(self):
        # [FUNCTION DOC] With non-finite intermediates (large scales giving nan), the compiled
        # scaled softmax rewrite must give the same non-finite results as eager, while still
        # being matched at least once.
        def check(fn, args):
            # [FUNCTION DOC] HELPER: Resets dynamo and counters, runs eager and compiled,
            # compares the first output for closeness, asserts the last outputs are all nan in
            # both, and asserts at least one match.
            torch._dynamo.reset()
            counters.clear()
            expected = fn(*args)
            actual, _ = run_and_get_code(torch.compile(fn), *args)
            torch.testing.assert_close(actual[0], expected[0])
            self.assertTrue(torch.isnan(expected[-1]).all().item())
            self.assertTrue(torch.isnan(actual[-1]).all().item())
            self.assertGreaterEqual(counters["inductor"]["pattern_matcher_count"], 1)

        def mul_softmax(x, scale):
            # [FUNCTION DOC] HELPER: Multiplies x by scale three times and returns x and its
            # softmax over dim zero.
            for _ in range(3):
                x = x * scale
            return x, F.softmax(x, dim=0)

        def div_softmax(x, inv_scale):
            # [FUNCTION DOC] HELPER: Divides x by the scale and returns x and its softmax.
            x = x / inv_scale
            return x, F.softmax(x, dim=0)

        def duplicated_mul_softmax(x, scale):
            # [FUNCTION DOC] HELPER: Two separate chains of three scale multiplications; returns
            # the first chain and the softmax of the second.
            y = x
            for _ in range(3):
                y = y * scale
            z = x
            for _ in range(3):
                z = z * scale
            return y, F.softmax(z, dim=0)

        torch.manual_seed(100)
        check(mul_softmax, (torch.randn((1, 10)), torch.tensor([-1.7e14])))
        check(duplicated_mul_softmax, (torch.randn((1, 10)), torch.tensor([-1.7e14])))
        check(
            div_softmax,
            (
                torch.tensor([[1.0, -1.0, 2.0, -2.0]]),
                torch.tensor([1e-45]),
            ),
        )

    @inductor_config.patch(fx_graph_cache=False)
    def test_scaled_amax_sub_nonfinite_matches_eager(self):
        # [FUNCTION DOC] For a scaled value subtracted by its amax over dim zero, large scale
        # overflowing to infinity must give nan for the infinite row and negative infinity for
        # the other, matching eager, with at least one pattern match counted.
        def fn(x, scale):
            # [FUNCTION DOC] HELPER: Multiplies x by scale and subtracts the amax over dim zero.
            y = x * scale
            return y - torch.amax(y, dim=0, keepdim=True)

        torch._dynamo.reset()
        counters.clear()
        x = torch.tensor([[10.0], [0.0]])
        scale = torch.tensor([1e38])
        expected = fn(x, scale)
        actual, _ = run_and_get_code(torch.compile(fn), x, scale)
        self.assertTrue(torch.isnan(expected[0, 0]).item())
        self.assertTrue(torch.isnan(actual[0, 0]).item())
        self.assertEqual(actual[1, 0].item(), float("-inf"))
        self.assertGreaterEqual(counters["inductor"]["pattern_matcher_count"], 1)

    def test_mutation_op_matching(self):
        # [FUNCTION DOC] Checks is_mutation_op on FX nodes: true for a grad mode switch, in-
        # place copy_ and relu_, manual_seed, set_, autocast enter and exit, c10d out variants,
        # resize_storage_bytes_, fsdp split copy and copy_ ops; false for rshift scalar, the
        # functional all_gather_into_tensor, and a custom op declared with no mutated args.
        def check(type, func_name, args, kwargs, expect=True):
            # [FUNCTION DOC] HELPER: Builds a one-node graph via the named graph method kind
            # (raises an error if it is not function or method call), runs is_mutation_op on the
            # node and asserts the result matches expect.
            if type not in ["call_function", "call_method"]:
                raise AssertionError
            graph = torch.fx.Graph()
            getattr(graph, type)(func_name, args, kwargs)
            res = is_mutation_op(next(iter(graph.nodes)))
            if expect:
                self.assertTrue(res)
            else:
                self.assertFalse(res)

        t = torch.randn(1)
        check("call_function", torch._C._set_grad_enabled, (False,), {})
        check("call_method", "copy_", (t, t), {})
        check("call_method", "relu_", (t,), {})
        check("call_function", torch.manual_seed, (0,), {})
        check("call_function", torch.ops.aten.set_.source_Tensor, (t, t), {})
        check(
            "call_function",
            torch.amp.autocast_mode._enter_autocast,
            (GPU_TYPE, None, True, None),
            {},
        )
        check("call_function", torch.amp.autocast_mode._exit_autocast, (None,), {})
        check(
            "call_function",
            torch.ops._c10d_functional.all_gather_into_tensor_out,
            (t, 2, "0"),
            {"out": t},
        )
        check("call_function", torch.ops.inductor.resize_storage_bytes_, (t, 0), {})
        check(
            "call_function",
            torch.ops.inductor.resize_storage_bytes_.default,
            (t, 0),
            {},
        )
        check(
            "call_function",
            torch.ops.fsdp.split_with_sizes_copy,
            (t, [64, 128, 8, 8]),
            {"dim": 1, "out": [t, t, t, t]},
        )
        check("call_function", torch.ops.fsdp.copy_, (t, t), {})
        check(
            "call_function", torch.ops.aten.__rshift__.Scalar, (t, 2), {}, expect=False
        )
        check(
            "call_function",
            torch.ops._c10d_functional.all_gather_into_tensor,
            (t, 2, "0"),
            {},
            expect=False,
        )

        @torch.library.custom_op("vllm::fused_rms_norm_quant_static", mutates_args=[])
        def fused_rms_norm_quant_static(out: torch.Tensor, input: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Custom op declared as non-mutating; its body does nothing.
            pass

        check(
            "call_function",
            torch.ops.vllm.fused_rms_norm_quant_static,
            (t, t),
            {},
            expect=False,
        )

    def test_multioutput_register_replacement(self):
        # [FUNCTION DOC] Registers a multi-output replacement in a fake vllm library: a pattern
        # of rms_norm followed by int8 quant (both with auto functionalized mutation) is
        # replaced by a fused op. After applying through a custom post grad pass in a custom
        # backend, the compiled code no longer contains the separate quant op.
        @torch.library.custom_op(
            "vllm::fused_rms_norm_quant_static", mutates_args=["result", "scale"]
        )
        def fused_rms_norm_quant_static(
            result: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
            azp: torch.Tensor,
            epsilon: float,
        ) -> None:
            # [FUNCTION DOC] HELPER: Custom op with mutated result and scale arguments that
            # computes bogus values and fills the scale with 0.5; prints a marker string.
            print("vllm::fused_rms_norm_quant_static")
            result_rms = torch.mul(input, weight) + epsilon
            _result = torch.mul(result_rms, scale).to(torch.int8)
            scale.fill_(0.5)

        @torch.library.custom_op("vllm::rms_norm", mutates_args=["result"])
        def rms_norm(
            result: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            epsilon: float,
        ) -> None:
            # [FUNCTION DOC] HELPER: Custom op that mutates result; body computes bogus data.
            # bogus implementation doesn't matter
            _result = torch.mul(input, weight) + epsilon

        @torch.library.custom_op(
            "vllm::static_scaled_int8_quant", mutates_args=["result", "scale"]
        )
        def static_scaled_int8_quant(
            result: torch.Tensor,
            input: torch.Tensor,
            scale: torch.Tensor,
            azp: torch.Tensor | None = None,
        ) -> None:
            # [FUNCTION DOC] HELPER: Custom op that mutates result and scale; fills scale with
            # 0.5, bogus data.
            # bogus implementation doesn't matter
            _result = torch.mul(input, scale).to(torch.int8)
            scale.fill_(0.5)

        def rms_pattern_static(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Pattern function: auto functionalized rms_norm followed by
            # auto functionalized quant op; returns the quant result and scale outputs.
            at1 = auto_functionalized(
                torch.ops.vllm.rms_norm.default,
                result=result_rms,
                input=input,
                weight=weight,
                epsilon=1e-6,
            )
            at2 = auto_functionalized(
                torch.ops.vllm.static_scaled_int8_quant.default,
                result=result,
                input=at1[1],
                scale=scale,
                azp=None,
            )

            return at2[1], at2[2]

        def rms_replacement_static(
            result: torch.Tensor,
            result_rms: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Replacement function: single auto functionalized fused op;
            # returns result and scale outputs.
            at = auto_functionalized(
                torch.ops.vllm.fused_rms_norm_quant_static.default,
                result=result,
                input=input,
                weight=weight,
                epsilon=1e-6,
                scale=scale,
                azp=None,
            )
            return at[1], at[2]

        def empty_bf16(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Empty bfloat16 tensor of the given shape.
            return torch.empty(*args, **kwargs, dtype=torch.bfloat16)

        def empty_int8(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Empty int8 tensor of the given shape.
            return torch.empty(*args, **kwargs, dtype=torch.int8)

        my_patterns = PatternMatcherPass()
        inputs = [
            empty_int8(5, 4),
            empty_bf16(5, 4),
            empty_bf16(5, 4),
            empty_bf16(5, 1),
            torch.empty(1, 1),
        ]
        register_replacement(
            rms_pattern_static, rms_replacement_static, inputs, fwd_only, my_patterns
        )

        def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
            # [FUNCTION DOC] HELPER: Post grad pass that applies the patterns and removes dead
            # code, returns the graph.
            _count = my_patterns.apply(graph)
            # print(f"Count: {_count}")
            graph.eliminate_dead_code()
            # graph.print_tabular()
            return graph

        def custom_backend(
            graph: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
        ) -> Callable:
            # [FUNCTION DOC] HELPER: Backend that copies the inductor config, sets the custom
            # post grad pass, and calls compile_fx with those patches.
            from torch._inductor import config

            current_config = config.shallow_copy_dict()
            from torch._inductor.compile_fx import compile_fx

            current_config["post_grad_custom_post_pass"] = custom_pass
            return compile_fx(graph, example_inputs, config_patches=current_config)

        @torch.compile(backend=custom_backend)
        def my_func_static(x, w, epsilon):
            # [FUNCTION DOC] HELPER: Compiled function creating output buffers, casting inputs
            # to bfloat16, and calling the pattern function to get quant result and scale.
            quant_result = torch.empty_like(x, dtype=torch.int8)
            result_rms = torch.empty_like(x, dtype=torch.bfloat16)
            scale = torch.ones((1, 1))

            x = x.to(torch.bfloat16)
            w = w.to(torch.bfloat16)

            quant_result, scale = rms_pattern_static(
                result=quant_result,
                result_rms=result_rms,
                input=x,
                weight=w,
                scale=scale,
            )

            return quant_result, scale

        inputs = [torch.empty((5, 4)), torch.empty((5, 1)), 1e-6]
        # print(my_func_static(*inputs))
        test, (code,) = run_and_get_code(my_func_static, *inputs)
        self.assertTrue("static_scaled_int8_quant" not in code)

    def test_nested_replacement_args_do_not_percolate_tags(self):
        # [FUNCTION DOC] ReplacementPatternEntry.replace_with_graph with a nested arguments
        # tuple must not copy the recompute and ac_graph_id meta tags from the replaced node
        # onto its inputs; the new graph still contains a mul and getitem.
        class DummyMatch:
            # [CLASS DOC] HELPER: Minimal stand-in for a match object exposing nodes and
            # output nodes.
            def __init__(self, outputs):
                # [FUNCTION DOC] HELPER: Stores the output nodes as both nodes and outputs.
                self.nodes = outputs
                self._outputs = outputs

            def output_nodes(self):
                # [FUNCTION DOC] HELPER: Returns the stored output nodes.
                return self._outputs

            def erase_nodes(self):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        sin = graph.call_function(torch.ops.aten.sin.default, (x,))
        old = graph.call_function(torch.ops.aten.add.Tensor, (sin, y))
        graph.output(old)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        def replacement(args):
            # [FUNCTION DOC] HELPER: Unpacks a pair argument and multiplies the two elements.
            a, b = args
            return torch.mul(a, b)

        replacement_gm = torch.fx.symbolic_trace(replacement)

        old.meta["recompute"] = torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
        old.meta["ac_graph_id"] = 1

        ReplacementPatternEntry.replace_with_graph(
            DummyMatch([old]),
            gm.graph,
            replacement_gm,
            args=[(sin, y)],
        )

        self.assertFalse(
            "recompute" in sin.meta,
            "recompute tag should not percolate past nested replacement inputs",
        )
        self.assertFalse(
            "ac_graph_id" in sin.meta,
            "ac_graph_id should not percolate past nested replacement inputs",
        )
        graph_str = str(gm.graph)
        self.assertIn("torch.mul", graph_str)
        self.assertIn("operator.getitem", graph_str)

    def test_fwd_only_generate_original_aten_meta(self):
        # [FUNCTION DOC] fwd_only traced graph carries original_aten metadata on the
        # sigmoid node.
        def f(x):
            # [FUNCTION DOC] HELPER: Sigmoid via the aten op.
            return torch.ops.aten.sigmoid(x)

        sample_input = torch.randn(3, 5, device=GPU_TYPE)
        gm_with_meta = fwd_only(f, args=[sample_input])
        sigmoid_nodes = gm_with_meta.graph.find_nodes(
            op="call_function", target=torch.ops.aten.sigmoid.default
        )
        self.assertEqual(len(sigmoid_nodes), 1)
        self.assertTrue("original_aten" in sigmoid_nodes[0].meta)

    def test_fwd_only_uses_get_decomp_fn(self):
        # [FUNCTION DOC] fwd_only uses the table from get_decomp_fn: by default gelu decomposes
        # into erf-based ops, with an empty table it stays as gelu.
        """fwd_only traces the pattern graph using the table from get_decomp_fn."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Gelu of x.
            return F.gelu(x)

        x = torch.randn(4, 4, device=GPU_TYPE)

        # Default: gelu decomposes into erf/mul/add primitives.
        gm = fwd_only(fn, args=[x])
        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertNotIn(aten.gelu.default, targets)
        self.assertIn(aten.erf.default, targets)

        # Empty decomp table: gelu stays intact as aten.gelu.default.
        gm_nodec = fwd_only(fn, args=[x], get_decomp_fn=dict)
        targets_nodec = {
            n.target for n in gm_nodec.graph.nodes if n.op == "call_function"
        }
        self.assertIn(aten.gelu.default, targets_nodec)
        self.assertNotIn(aten.erf.default, targets_nodec)

    def test_register_replacement_get_decomp_fn(self):
        # [FUNCTION DOC] A pattern registered with get_decomp_fn empty (dict) matches a graph
        # traced without decompositions (one replacement) and does not match one traced with
        # default decompositions.
        """A pattern registered with get_decomp_fn matches graphs traced with
        the same decomposition table, and does not match graphs traced with a
        different one."""

        def gelu_pattern(x):
            # [FUNCTION DOC] HELPER: Pattern: gelu of x.
            return F.gelu(x)

        def gelu_double(x):
            # [FUNCTION DOC] HELPER: Replacement: gelu times 2.
            return F.gelu(x) * 2.0

        x = torch.randn(4, 4, device=GPU_TYPE)

        # Pattern traced without decompositions: stored as aten.gelu.default.
        my_patterns = PatternMatcherPass()
        register_replacement(
            gelu_pattern,
            gelu_double,
            [x],
            fwd_only,
            my_patterns,
            get_decomp_fn=dict,
        )

        # Graph where gelu is also not decomposed: pattern should match once.
        gm_nodec = make_fx(gelu_pattern, {})(x)
        self.assertEqual(my_patterns.apply(gm_nodec.graph), 1)

        # Graph where gelu is decomposed (default decomps): no match.
        gm_decomposed = fwd_only(gelu_pattern, args=[x])
        self.assertEqual(my_patterns.apply(gm_decomposed.graph), 0)

    @inductor_config.patch(is_predispatch=True)
    def test_remove_noop_pass_with_remove_passes(self):
        # [FUNCTION DOC] With predispatch on, compiling with remove_pre_grad_passes set to
        # remove_noop or unset gives matching results; at the pre grad pass level, the graph
        # with remove_noop removed has at least as many view ops as the default.
        def fn_with_noop(x):
            # [FUNCTION DOC] HELPER: Views x to its own shape then adds one.
            batch_size, dim = x.shape
            y = x.view(batch_size, dim)
            return y + 1

        def count_view_ops(graph_module):
            # [FUNCTION DOC] HELPER: Counts nodes of call_function type whose target is view or
            # reshape in a graph module.
            count = 0
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target in [
                    torch.ops.aten.view.default,
                    torch.ops.aten.reshape.default,
                ]:
                    count += 1
            return count

        device = GPU_TYPE if HAS_GPU else "cpu"
        input_tensor = torch.randn(8, 16, device=device)

        with inductor_config.patch(remove_pre_grad_passes=None):
            compiled_fn_default = torch.compile(fn_with_noop, fullgraph=True)
            result_default = compiled_fn_default(input_tensor)

        with inductor_config.patch(remove_pre_grad_passes="remove_noop"):
            compiled_fn_skip_noop = torch.compile(fn_with_noop, fullgraph=True)
            result_skip_noop = compiled_fn_skip_noop(input_tensor)

        expected = fn_with_noop(input_tensor)
        torch.testing.assert_close(result_default, expected)
        torch.testing.assert_close(result_skip_noop, expected)

        from torch._inductor.fx_passes.pre_grad import pre_grad_passes
        from torch.fx.experimental.proxy_tensor import make_fx

        with inductor_config.patch(
            is_predispatch=True, pattern_matcher=True, remove_pre_grad_passes=None
        ):
            gm_default = make_fx(fn_with_noop)(input_tensor)
            gm_default_processed = pre_grad_passes(
                gm_default, [input_tensor], add_passes=None, remove_passes=None
            )
            view_count_default = count_view_ops(gm_default_processed)

        with inductor_config.patch(
            is_predispatch=True,
            pattern_matcher=True,
            remove_pre_grad_passes="remove_noop",
        ):
            gm_skip_noop = make_fx(fn_with_noop)(input_tensor)
            gm_skip_noop_processed = pre_grad_passes(
                gm_skip_noop,
                [input_tensor],
                add_passes=None,
                remove_passes="remove_noop",
            )
            view_count_skip_noop = count_view_ops(gm_skip_noop_processed)

        self.assertGreaterEqual(
            view_count_skip_noop,
            view_count_default,
            lambda msg: f"{msg}\nExpected view count with remove_noop disabled ({view_count_skip_noop}) "
            f"to be >= view count with remove_noop enabled ({view_count_default})",
        )

    def test_bound_method_pattern_matcher(self):
        # [FUNCTION DOC] A pattern defined via bound methods of a helper class with an exponent
        # attribute (sum of powers replaced by power of sum) is registered through
        # register_replacement. A compiled function with such sums gets exactly one replacement
        # applied; outputs match the manually rewritten function.
        class ReluSumPattern:
            # [CLASS DOC] HELPER: Holds an exponent and provides bound methods used as pattern,
            # replacement, example inputs and registration.
            def __init__(self, e: float):
                # [FUNCTION DOC] HELPER: Stores the exponent.
                self.e = e

            def pattern(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
                # [FUNCTION DOC] HELPER: Pattern: sum of x, y and z each raised to the exponent.
                return x.pow(self.e) + y.pow(self.e) + z.pow(self.e)

            def replacement(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
                # [FUNCTION DOC] HELPER: Replacement: sum of x, y and z raised to the exponent.
                return (x + y + z).pow(self.e)

            def inputs(self):
                # [FUNCTION DOC] HELPER: Returns three empty 5 by 5 example tensors.
                return [
                    torch.empty(5, 5),  # x
                    torch.empty(5, 5),  # y
                    torch.empty(5, 5),  # z
                ]

            def register(self, pm: PatternMatcherPass):
                # [FUNCTION DOC] HELPER: Registers the bound pattern and replacement methods
                # with the example inputs and fwd_only tracing into the given pass.
                register_replacement(
                    self.pattern, self.replacement, self.inputs(), fwd_only, pm
                )

        my_patterns = PatternMatcherPass()
        ReluSumPattern(4).register(my_patterns)

        count = 0

        def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
            # [FUNCTION DOC] HELPER: Post grad pass that stores the number of applied
            # replacements in an enclosing variable, removes dead code and returns the graph.
            nonlocal count
            count = my_patterns.apply(graph)
            graph.eliminate_dead_code()
            return graph

        def custom_backend(graph: torch.fx.GraphModule, example_inputs):
            # [FUNCTION DOC] HELPER: Backend that sets the custom post pass and disables auto
            # functionalized v2 in a copy of the inductor config, then calls compile_fx.
            from torch._inductor import config

            current_config = config.shallow_copy_dict()
            from torch._inductor.compile_fx import compile_fx

            current_config["post_grad_custom_post_pass"] = custom_pass
            current_config["enable_auto_functionalized_v2"] = False
            return compile_fx(graph, example_inputs, config_patches=current_config)

        @torch.compile(fullgraph=True, backend=custom_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes sums of powers of x, y and z for exponents two,
            # three and four plus five.
            y = x.relu()
            z = y.tanh()
            z2 = x.pow(2) + y.pow(2) + z.pow(2)
            z3 = x.pow(3) + y.pow(3) + z2.pow(3)
            z4 = x.pow(4) + y.pow(4) + z3.pow(4)
            return z4 + 5

        def fn_replaced(x):
            # [FUNCTION DOC] HELPER: Same as the compiled function but with the final exponent
            # four sum written in its replaced form.
            y = x.relu()
            z = y.tanh()
            z2 = x.pow(2) + y.pow(2) + z.pow(2)
            z3 = x.pow(3) + y.pow(3) + z2.pow(3)
            z4 = (x + y + z3).pow(4)
            return z4 + 5

        x = [torch.ones((5, 4))]
        fn_result = fn(*x)
        fn_replaced_result = fn_replaced(*x)
        self.assertEqual(count, 1)
        self.assertEqual(fn_result, fn_replaced_result)

    def test_empty_like_pattern_matching(self):
        # [FUNCTION DOC] A pattern using an uninitialized empty tensor times two added to x is
        # replaced by x times three. The compiled function using empty_like instead of empty
        # must be matched (count of one) and return x times three.
        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern: x plus empty tensor of x's shape times two.
            y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            z = y * 2
            return x + z

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement: x times three.
            return x * 3

        my_patterns = PatternMatcherPass()
        inputs = [torch.randn(4, 4, device=GPU_TYPE)]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        count = 0

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that stores and returns the number of
            # applied replacements.
            nonlocal count
            count = my_patterns.apply(graph)
            return count

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement: x times three (redefinition of the
            # earlier one).
            return x * 3

        my_patterns = PatternMatcherPass()
        inputs = [torch.randn(4, 4, device=GPU_TYPE)]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        def fn(x):
            # [FUNCTION DOC] HELPER: x plus empty_like of x times two.
            y = torch.empty_like(x)
            z = y * 2
            return x + z

        x = torch.randn(4, 4, device=GPU_TYPE)

        compiled_fn = torch.compile(
            fn, fullgraph=True, options={"post_grad_custom_post_pass": custom_pass}
        )

        result = compiled_fn(x)
        self.assertEqual(result, x * 3)
        self.assertEqual(count, 1)

    def test_get_attr_tensor_constant_pattern_matching(self):
        # [FUNCTION DOC] A generated pattern containing a constant tensor holding the
        # minimum finite value of the dtype matches the corresponding graph but not one
        # with a different constant.
        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern: maximum of x plus one and a constant tensor of the
            # dtype's minimum.
            y = x + 1
            fill = torch.tensor(torch.finfo(y.dtype).min)
            return torch.maximum(y, fill)

        def different_constant(x):
            # [FUNCTION DOC] HELPER: Same as the pattern but with a constant of zero.
            y = x + 1
            fill = torch.tensor(0.0)
            return torch.maximum(y, fill)

        x = torch.randn(4, 4)
        generated_pattern = gen_pattern(pattern, [x], fwd_only)

        gm = fwd_only(pattern, [x])
        maximum = next(
            n for n in gm.graph.nodes if n.target == torch.ops.aten.maximum.default
        )
        self.assertTrue(generated_pattern.match(maximum))

        other_gm = fwd_only(different_constant, [x])
        other_maximum = next(
            n
            for n in other_gm.graph.nodes
            if n.target == torch.ops.aten.maximum.default
        )
        self.assertFalse(generated_pattern.match(other_maximum))

    def test_pretty_print_get_attr_tensor_constant_under_fake_mode(self):
        # [FUNCTION DOC] Under fake mode and tracing context, pretty printing a generated
        # pattern with a get_attr tensor constant yields text containing the tensor constructor
        # with a float32 dtype.
        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern: maximum of x plus one and a constant tensor of the
            # dtype's minimum.
            y = x + 1
            fill = torch.tensor(torch.finfo(y.dtype).min)
            return torch.maximum(y, fill)

        x = torch.randn(2, 2)
        with torch._guards.tracing(None), torch._subclasses.FakeTensorMode() as mode:
            fx = mode.from_tensor(x)
            generated_pattern, _ = gen_pattern_and_search_gm(pattern, [fx], fwd_only)
            pattern_str = PatternPrettyPrinter.run(generated_pattern)

        self.assertIn("GetAttr(torch.tensor(", pattern_str)
        self.assertIn("dtype=torch.float32", pattern_str)

    def test_pretty_print_get_attr_nonfinite_tensor_constant_executes(self):
        # [FUNCTION DOC] Pretty printing a GetAttr pattern with inf, negative inf and nan (and
        # complex counterparts) produces text that can be executed to rebuild an equal pattern;
        # text must contain the non-finite literal spellings.
        get_attr_pattern = GetAttr(
            torch.tensor([float("inf"), -float("inf"), float("nan")])
        )
        pattern_str = PatternPrettyPrinter.run(get_attr_pattern)
        namespace = {"GetAttr": GetAttr, "torch": torch}

        exec(pattern_str, namespace)

        self.assertIsInstance(namespace["output"], GetAttr)
        self.assertTrue(get_attr_pattern.pattern_eq(namespace["output"]))
        self.assertIn("float('inf')", pattern_str)
        self.assertIn("float('nan')", pattern_str)

        complex_get_attr_pattern = GetAttr(
            torch.tensor([complex(float("nan"), 1.0), complex(2.0, float("inf"))])
        )
        complex_pattern_str = PatternPrettyPrinter.run(complex_get_attr_pattern)
        namespace = {"GetAttr": GetAttr, "torch": torch}

        exec(complex_pattern_str, namespace)

        self.assertIsInstance(namespace["output"], GetAttr)
        self.assertTrue(complex_get_attr_pattern.pattern_eq(namespace["output"]))
        self.assertIn("complex(float('nan'), 1.0)", complex_pattern_str)
        self.assertIn("complex(2.0, float('inf'))", complex_pattern_str)

    def test_pretty_print_get_attr_tensor_constant_requires_contiguous(self):
        # [FUNCTION DOC] Pretty printing a GetAttr pattern built from a non-contiguous tensor
        # slice raises NotImplementedError mentioning a non-contiguous get_attr tensor.
        get_attr_pattern = GetAttr(torch.arange(6)[1:5:2])

        with self.assertRaisesRegex(
            NotImplementedError, "non-contiguous get_attr tensor"
        ):
            PatternPrettyPrinter.run(get_attr_pattern)

    def test_get_attr_nonfinite_tensor_constant_pattern_matching(self):
        # [FUNCTION DOC] GetAttr pattern matching with non-finite tensor constants: a pattern
        # holding inf, negative inf and nan matches a get_attr node whose constant is an equal
        # clone (nan compared as equal) and does not match when a value differs; same for a
        # complex tensor with a nan real part.
        def make_attr_node(value):
            # [FUNCTION DOC] HELPER: Builds a graph with one get_attr node reading a module
            # attribute holding value, wraps it in a GraphModule and returns the get_attr node.
            graph = torch.fx.Graph()
            attr = graph.get_attr("constant")
            graph.output(attr)
            mod = torch.nn.Module()
            mod.constant = value
            gm = torch.fx.GraphModule(mod, graph)
            return next(n for n in gm.graph.nodes if n.op == "get_attr")

        expected = torch.tensor([float("inf"), -float("inf"), float("nan")])

        self.assertTrue(GetAttr(expected).match(make_attr_node(expected.clone())))
        self.assertFalse(
            GetAttr(expected).match(
                make_attr_node(torch.tensor([float("inf"), -float("inf"), 0.0]))
            )
        )

        complex_expected = torch.tensor([complex(float("nan"), 1.0)])
        self.assertTrue(
            GetAttr(complex_expected).match(make_attr_node(complex_expected.clone()))
        )
        self.assertFalse(
            GetAttr(complex_expected).match(
                make_attr_node(torch.tensor([complex(float("nan"), 2.0)]))
            )
        )

    def test_get_attr_tensor_constant_metadata_must_match(self):
        # [FUNCTION DOC] GetAttr matching requires matching tensor metadata, not only values: a
        # tensor with the same size, stride and storage offset matches, while a contiguous copy,
        # a different storage offset, and a tensor with a different dtype or requires_grad flag
        # do not.
        def make_attr_node(value):
            # [FUNCTION DOC] HELPER: Builds a graph with one get_attr node reading a module
            # attribute holding value, wraps it in a GraphModule and returns the get_attr node.
            graph = torch.fx.Graph()
            attr = graph.get_attr("constant")
            graph.output(attr)
            mod = torch.nn.Module()
            mod.constant = value
            gm = torch.fx.GraphModule(mod, graph)
            return next(n for n in gm.graph.nodes if n.op == "get_attr")

        expected = torch.arange(6)[1:5:2]
        same_metadata = torch.as_strided(torch.tensor([0, 1, 0, 3]), (2,), (2,), 1)
        same_values_contiguous = expected.clone()
        same_values_different_offset = torch.as_strided(
            torch.tensor([1, 0, 3]), (2,), (2,), 0
        )
        same_values_requires_grad = torch.tensor([1.0, 3.0], requires_grad=True)

        self.assertTrue(GetAttr(expected).match(make_attr_node(same_metadata)))
        self.assertFalse(
            GetAttr(expected).match(make_attr_node(same_values_contiguous))
        )
        self.assertFalse(
            GetAttr(expected).match(make_attr_node(same_values_different_offset))
        )
        self.assertFalse(
            GetAttr(expected.float()).match(make_attr_node(same_values_requires_grad))
        )

    def test_get_attr_fake_tensor_constant_requires_value(self):
        # [FUNCTION DOC] GetAttr with fake tensor constants: a fake tensor created
        # without a known real value never matches a real get_attr constant, while a
        # fake tensor made constant from a real tensor (make_constant) matches the
        # get_attr node holding the real tensor.
        def make_attr_node(value):
            # [FUNCTION DOC] HELPER: Builds a graph with one get_attr node reading a module
            # attribute holding value, wraps it in a GraphModule and returns the get_attr node.
            graph = torch.fx.Graph()
            attr = graph.get_attr("constant")
            graph.output(attr)
            mod = torch.nn.Module()
            mod.constant = value
            gm = torch.fx.GraphModule(mod, graph)
            return next(n for n in gm.graph.nodes if n.op == "get_attr")

        expected = torch.tensor([1.0, 2.0])
        unknown_source = torch.empty_like(expected)
        with torch._subclasses.FakeTensorMode() as mode:
            unknown_fake = mode.from_tensor(unknown_source)
            known_fake = mode.fake_tensor_converter.from_real_tensor(
                mode, expected, make_constant=True
            )

        self.assertFalse(GetAttr(expected).match(make_attr_node(unknown_fake)))
        self.assertTrue(GetAttr(known_fake).match(make_attr_node(expected)))

    def test_register_replacement_with_get_attr_tensor_constant(self):
        # [FUNCTION DOC] register_replacement with a pattern containing a constant tensor (dtype
        # minimum) in the pattern: applying the pass on a traced graph with the same constant
        # gives one replacement; after dead code elimination the module output equals the
        # replacement and no get_attr node remains.
        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern: maximum of x plus one and a constant tensor of the
            # dtype's minimum.
            y = x + 1
            fill = torch.tensor(torch.finfo(y.dtype).min)
            return torch.maximum(y, fill)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement: x plus two.
            return x + 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Same computation as the pattern, used to build the
            # graph to rewrite.
            y = x + 1
            fill = torch.tensor(torch.finfo(y.dtype).min)
            return torch.maximum(y, fill)

        my_patterns = PatternMatcherPass()
        x = torch.randn(4, 4)
        register_replacement(pattern, replacement, [x], fwd_only, my_patterns)

        gm = fwd_only(fn, [x])
        self.assertEqual(my_patterns.apply(gm), 1)
        gm.graph.eliminate_dead_code()
        gm.recompile()

        self.assertEqual(gm(x), replacement(x))
        self.assertFalse(any(n.op == "get_attr" for n in gm.graph.nodes))

    def test_register_replacement_single_tensor_input(self):
        # [FUNCTION DOC] register_replacement must reject a single tensor given as example
        # inputs: raises TypeError with a message saying example_inputs must be a list or tuple
        # and naming the type received.
        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern: x plus one.
            return x + 1

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement: x minus one.
            return x - 1

        my_patterns = PatternMatcherPass()

        # Single tensor should fail fast instead of reaching tracing logic.
        example_input = torch.randn(4, 4, device=GPU_TYPE, requires_grad=True)
        with self.assertRaisesRegex(
            TypeError,
            f"example_inputs must be a list or tuple, got {type(example_input)}",
        ):
            register_replacement(
                pattern, replacement, example_input, fwd_only, my_patterns
            )

    def _inject_test_metadata(self, graph):
        # [FUNCTION DOC] HELPER: Fills metadata on every call_function node of graph: a
        # stack_trace string derived from the node name, an nn_module_stack dictionary, a
        # numeric debug handle of 42 and a custom dictionary. The graph is mutated in place; the
        # original docstring says it injects identifiable metadata for tests.
        """Inject identifiable metadata on all call_function nodes for testing."""
        for node in graph.nodes:
            if node.op == "call_function":
                node.meta["stack_trace"] = f"trace_for_{node.name}"
                node.meta["nn_module_stack"] = {"test": ("m", "M")}
                node.meta["_numeric_debug_handle"] = 42
                node.meta["custom"] = {"test_key": "test_value"}

    def test_replace_random_pass_preserves_stack_trace_on_rng_prims(self):
        # [FUNCTION DOC] Traces a function with a randn call in fake tracing mode, sets
        # stack_trace and source_fn_stack metadata on the randn node, then runs
        # replace_random_passes under the fake mode. Each of the generated seeds, lookup_seed
        # and random prim nodes must carry the same stack_trace and source_fn_stack, and have
        # value and tensor_meta metadata with the expected shape and dtype (seeds: size 1 int64,
        # lookup: scalar int64, random: size 4 float32).
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds a random normal tensor of x's shape on its
            # device to x.
            return x + torch.randn(x.shape, device=x.device)

        gm = make_fx(fn, tracing_mode="fake")(torch.randn(4))

        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        self.assertIsNotNone(fake_mode)

        stack_trace = '  File "test_replace_random.py", line 1, in fn\n'
        source_fn_stack = [("randn", torch.randn)]

        randn_node = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten.randn.default
        )
        randn_node.meta["stack_trace"] = stack_trace
        randn_node.meta["source_fn_stack"] = source_fn_stack

        with V.set_fake_mode(fake_mode):
            replace_random_passes(gm)
        gm.graph.lint()

        for target in (
            torch.ops.prims.inductor_seeds.default,
            torch.ops.prims.inductor_lookup_seed.default,
            torch.ops.prims.inductor_random.default,
        ):
            node = next(node for node in gm.graph.nodes if node.target == target)
            self.assertEqual(
                node.meta.get("stack_trace"),
                stack_trace,
                msg=lambda msg: f"{msg}\n{target}: {node.meta}",
            )
            self.assertEqual(
                node.meta.get("source_fn_stack"),
                source_fn_stack,
                msg=lambda msg: f"{msg}\n{target}: {node.meta}",
            )

        expected_metadata = {
            torch.ops.prims.inductor_seeds.default: (torch.Size([1]), torch.int64),
            torch.ops.prims.inductor_lookup_seed.default: (torch.Size([]), torch.int64),
            torch.ops.prims.inductor_random.default: (torch.Size([4]), torch.float32),
        }
        for target, (shape, dtype) in expected_metadata.items():
            node = next(node for node in gm.graph.nodes if node.target == target)
            self.assertEqual(
                node.meta["val"].shape,
                shape,
                msg=lambda msg: f"{msg}\n{target}: {node.meta}",
            )
            self.assertEqual(
                node.meta["val"].dtype,
                dtype,
                msg=lambda msg: f"{msg}\n{target}: {node.meta}",
            )
            self.assertEqual(
                node.meta["tensor_meta"].shape,
                shape,
                msg=lambda msg: f"{msg}\n{target}: {node.meta}",
            )
            self.assertEqual(
                node.meta["tensor_meta"].dtype,
                dtype,
                msg=lambda msg: f"{msg}\n{target}: {node.meta}",
            )

    def test_metadata_propagation_register_replacement(self):
        # [FUNCTION DOC] Compiles an addition with a custom post grad pass that injects metadata
        # then applies a register_replacement that turns add into mul. Only checks that
        # compilation and execution succeed (metadata transfer runs inside replacement); there
        # are no assertions on the metadata.
        """Verify metadata from matched nodes transfers to replacement nodes."""

        def pattern(x, y):
            # [FUNCTION DOC] HELPER: Pattern: x plus y.
            return x + y

        def replacement(x, y):
            # [FUNCTION DOC] HELPER: Replacement: x times y.
            return x * y

        my_patterns = PatternMatcherPass()
        inputs = [
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        ]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that injects metadata on nodes then applies
            # the patterns; returns nothing.
            self._inject_test_metadata(graph)
            # _transfer_meta runs inside replace_with_graph for
            # each old->new pair to propagate metadata fields
            my_patterns.apply(graph)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: x plus y.
            return x + y

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        compiled_fn = torch.compile(
            fn, options={"post_grad_custom_post_pass": custom_pass}
        )
        compiled_fn(x, y)

    def test_metadata_propagation_register_replacement_multinode(self):
        # [FUNCTION DOC] Same as the single node variant but with a two-node pattern (add then
        # multiply by two) replaced by add then multiply by three; checks only that compilation
        # runs without error.
        """Verify metadata propagation for multi-node patterns."""

        def pattern(x, y):
            # [FUNCTION DOC] HELPER: Pattern: sum of x and y times two.
            tmp = x + y
            return tmp * 2

        def replacement(x, y):
            # [FUNCTION DOC] HELPER: Replacement: sum of x and y times three.
            return (x + y) * 3

        my_patterns = PatternMatcherPass()
        inputs = [
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        ]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that injects metadata on nodes then applies
            # the patterns.
            self._inject_test_metadata(graph)
            my_patterns.apply(graph)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sum of x and y times two.
            tmp = x + y
            return tmp * 2

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        compiled_fn = torch.compile(
            fn, options={"post_grad_custom_post_pass": custom_pass}
        )
        compiled_fn(x, y)

    def test_metadata_propagation_graph_pattern_replace_by_example(self):
        # [FUNCTION DOC] Registers a graph pattern on add that replaces it with a multiply via
        # replace_by_example under the fake mode; custom post grad pass injects metadata first.
        # Checks only that compilation and execution succeed.
        """Verify metadata propagation for replace_by_example (single-node match)."""

        test_pass = PatternMatcherPass()

        @register_graph_pattern(
            CallFunction(aten.add.Tensor, KeywordArg("x"), KeywordArg("y")),
            pass_dict=test_pass,
        )
        def add_to_mul(match: Match, x, y):
            # [FUNCTION DOC] HELPER: Handler that replaces the matched add with a
            # multiply using replace_by_example.
            def repl(a, b):
                # [FUNCTION DOC] HELPER: Replacement body: product of a and b.
                return a * b

            with V.fake_mode:
                match.replace_by_example(repl, [x, y])

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that injects metadata then applies
            # the test pass.
            self._inject_test_metadata(graph)
            test_pass.apply(graph)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: x plus y.
            return x + y

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        compiled_fn = torch.compile(
            fn, options={"post_grad_custom_post_pass": custom_pass}
        )
        compiled_fn(x, y)

    def test_replace_by_example_returns_inserted_nodes(self):
        # [FUNCTION DOC] replace_by_example returns the nodes it inserted: a graph pattern
        # replaces add with a product plus b and records targets of call_function nodes
        # returned; asserts the compiled result is x times y plus y and the recorded targets are
        # mul followed by add.
        test_pass = PatternMatcherPass()
        replacement_targets = []

        @register_graph_pattern(
            CallFunction(aten.add.Tensor, KeywordArg("x"), KeywordArg("y")),
            pass_dict=test_pass,
        )
        def add_to_mul_add(match: Match, x, y):
            # [FUNCTION DOC] HELPER: Handler that replaces the matched add with a function
            # computing a product plus b and appends the call_function targets of the returned
            # nodes to the enclosing list.
            def repl(a, b):
                # [FUNCTION DOC] HELPER: Replacement body: product of a and b plus b.
                prod = a * b
                return prod + b

            with V.fake_mode:
                replacement_nodes = match.replace_by_example(repl, [x, y])

            replacement_targets.extend(
                node.target for node in replacement_nodes if node.op == "call_function"
            )

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that applies the test pass.
            test_pass.apply(graph)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: x plus y.
            return x + y

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        compiled_fn = torch.compile(
            fn, options={"post_grad_custom_post_pass": custom_pass}
        )
        self.assertEqual(compiled_fn(x, y), x * y + y)
        self.assertEqual(replacement_targets, [aten.mul.Tensor, aten.add.Tensor])

    def test_metadata_propagation_replace_by_example_multinode(self):
        # [FUNCTION DOC] Same as the single node variant for a two-node match (add followed by
        # multiply replaced by subtract followed by multiply); metadata is injected first; only
        # verifies compilation and execution complete.
        """Verify metadata propagation for replace_by_example with a multi-node
        match.  The output node should inherit metadata from the matched output
        node via replace_with_graph's replace() inner function."""

        test_pass = PatternMatcherPass()

        @register_graph_pattern(
            CallFunction(
                aten.mul.Tensor,
                CallFunction(aten.add.Tensor, KeywordArg("x"), KeywordArg("y")),
                KeywordArg("z"),
            ),
            pass_dict=test_pass,
        )
        def add_mul_to_sub(match: Match, x, y, z):
            # [FUNCTION DOC] HELPER: Handler that replaces the matched add-then-mul with a
            # subtract-then-mul using replace_by_example.
            def repl(a, b, c):
                # [FUNCTION DOC] HELPER: Replacement body: difference of a and b times c.
                return (a - b) * c

            with V.fake_mode:
                match.replace_by_example(repl, [x, y, z])

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that injects metadata then applies
            # the test pass.
            self._inject_test_metadata(graph)
            test_pass.apply(graph)

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Sum of x and y, times z.
            return (x + y) * z

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)
        z = torch.randn(4, 4, device=GPU_TYPE)

        compiled_fn = torch.compile(
            fn, options={"post_grad_custom_post_pass": custom_pass}
        )
        compiled_fn(x, y, z)

    def test_metadata_propagation_lowering_pattern(self):
        # [FUNCTION DOC] Manually exercises the metadata transfer used by lowering pattern
        # entries: a handler creates a multiply node before the matched add, copies meta with
        # the private transfer function, redirects uses and erases the match. Asserts the
        # handler ran exactly once during compilation.
        """Verify metadata propagation for LoweringPatternEntry.apply.

        LoweringPatternEntry uses _transfer_meta to copy _COPY_META_FIELDS
        and stack_trace from the matched node to the replacement.
        """
        from torch._inductor.pattern_matcher import _transfer_meta

        test_pass = PatternMatcherPass()

        counter = 0

        @register_graph_pattern(
            CallFunction(aten.add.Tensor, KeywordArg("x"), KeywordArg("y")),
            pass_dict=test_pass,
        )
        def manual_lowering(match: Match, x, y):
            # [FUNCTION DOC] HELPER: Handler that inserts a multiply before the matched output
            # node, transfers metadata from the old node to the new node, replaces all uses,
            # erases the matched nodes and increments the enclosing counter.
            nonlocal counter
            # Manually exercise the LoweringPatternEntry code path:
            # create a replacement node, propagate meta, and replace
            node = match.output_node()
            graph = match.graph
            with graph.inserting_before(node):
                replacement = graph.call_function(aten.mul.Tensor, (x, y))
                _transfer_meta(replacement.meta, node)
                node.replace_all_uses_with(replacement)
            match.erase_nodes()
            counter += 1

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that injects metadata then applies
            # the test pass.
            self._inject_test_metadata(graph)
            test_pass.apply(graph)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: x plus y.
            return x + y

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        compiled_fn = torch.compile(
            fn, options={"post_grad_custom_post_pass": custom_pass}
        )
        compiled_fn(x, y)
        self.assertEqual(counter, 1)


@inductor_config.patch(fx_graph_cache=False)
class TestPatternMatcherLogging(LoggingTestCase):
    # [CLASS DOC] Logging tests for the pattern matcher (based on LoggingTestCase), run on the
    # GPU device type with the FX graph cache turned off for the whole class. Tests set the
    # pattern match debug environment variable to enable detailed log records and then check the
    # emitted records and counters.
    device_type = GPU_TYPE

    @make_logging_test()
    def test_pattern_match_debug_output(self, records):
        # [FUNCTION DOC] Records logs while an add-to-mul replacement is applied with the debug
        # variable naming add; asserts the compiled result equals x times y and that the
        # Specific pattern match record message contains a match description with args x and y
        # and the add call text.
        def pattern(x, y):
            # [FUNCTION DOC] HELPER: Pattern: x plus y.
            return x + y

        def replacement(x, y):
            # [FUNCTION DOC] HELPER: Replacement: x times y.
            return x * y

        my_patterns = PatternMatcherPass()
        inputs = [
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        ]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that applies the patterns and
            # returns the count.
            return my_patterns.apply(graph)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: x plus y.
            return x + y

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        with unittest.mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_PATTERN_MATCH_DEBUG": "add"}
        ):
            compiled_fn = torch.compile(
                fn, options={"post_grad_custom_post_pass": custom_pass}
            )
            result = compiled_fn(x, y)
            self.assertEqual(result, x * y)

        specific_record = self.getRecord(records, "Specific pattern match")
        self.assertIn(
            "Match(..., [], {'x': arg0_1, 'y': arg1_1})", specific_record.getMessage()
        )
        self.assertIn("add(arg0_1, arg1_1)", specific_record.getMessage())

    @make_logging_test()
    def test_failed_match_constant_args_format_string(self, records):
        # [FUNCTION DOC] Registers an add-one pattern, compiles a function adding two with the
        # debug variable set to add; the result is unchanged and the Specific pattern match
        # record message reports a constant args mismatch of 2 against 1 together with the
        # printed pattern.
        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern: x plus one.
            return x + 1

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement: x times two.
            return x * 2

        my_patterns = PatternMatcherPass()
        inputs = [
            torch.randn(4, 4, device=GPU_TYPE),
        ]
        register_replacement(pattern, replacement, inputs, fwd_only, my_patterns)

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that applies the patterns and
            # returns the count.
            return my_patterns.apply(graph)

        def fn(x):
            # [FUNCTION DOC] HELPER: x plus two.
            return x + 2

        x = torch.randn(4, 4, device=GPU_TYPE)

        with unittest.mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_PATTERN_MATCH_DEBUG": "add"}
        ):
            compiled_fn = torch.compile(
                fn, options={"post_grad_custom_post_pass": custom_pass}
            )
            result = compiled_fn(x)
            self.assertEqual(result, x + 2)

        specific_record = self.getRecord(records, "Specific pattern match")
        self.assertIn(
            "add(arg0_1, 2) constant_args: add 2!=1 CallFunction(aten.add.Tensor, KeywordArg('x'), 1, _users=0)",
            specific_record.getMessage(),
        )

    @make_logging_test()
    def test_pattern_match_debug_multiple_nodes(self, records):
        # [FUNCTION DOC] Two patterns (add and sub) are registered; with the debug
        # variable listing add and sub, log records for specific pattern match of both
        # add and sub must exist.
        def pattern_add(x, y):
            # [FUNCTION DOC] HELPER: Pattern for add: x plus y.
            return x + y

        def replacement_add(x, y):
            # [FUNCTION DOC] HELPER: Replacement for add: x times y.
            return x * y

        def pattern_sub(x, y):
            # [FUNCTION DOC] HELPER: Pattern for sub: x minus y.
            return x - y

        def replacement_sub(x, y):
            # [FUNCTION DOC] HELPER: Replacement for sub: x times y.
            return x * y

        my_patterns = PatternMatcherPass()
        inputs = [
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        ]
        register_replacement(
            pattern_add, replacement_add, inputs, fwd_only, my_patterns
        )
        register_replacement(
            pattern_sub, replacement_sub, inputs, fwd_only, my_patterns
        )

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that applies the patterns and
            # returns the count.
            return my_patterns.apply(graph)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sum of (x plus y) and (x minus y).
            return (x + y) + (x - y)

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        # Debug both "add" and "sub" nodes
        with unittest.mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_PATTERN_MATCH_DEBUG": "add,sub"}
        ):
            compiled_fn = torch.compile(
                fn, options={"post_grad_custom_post_pass": custom_pass}
            )
            _ = compiled_fn(x, y)

        self.assertTrue(self.hasRecord(records, "Specific pattern match: add"))
        self.assertTrue(self.hasRecord(records, "Specific pattern match: sub"))

    @make_logging_test()
    def test_pattern_match_debug_all_nodes(self, records):
        # [FUNCTION DOC] Same as the multiple nodes test, but with the debug variable set to
        # all; records for both add and sub must exist.
        def pattern_add(x, y):
            # [FUNCTION DOC] HELPER: Pattern for add: x plus y.
            return x + y

        def replacement_add(x, y):
            # [FUNCTION DOC] HELPER: Replacement for add: x times y.
            return x * y

        def pattern_sub(x, y):
            # [FUNCTION DOC] HELPER: Pattern for sub: x minus y.
            return x - y

        def replacement_sub(x, y):
            # [FUNCTION DOC] HELPER: Replacement for sub: x times y.
            return x * y

        my_patterns = PatternMatcherPass()
        inputs = [
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        ]
        register_replacement(
            pattern_add, replacement_add, inputs, fwd_only, my_patterns
        )
        register_replacement(
            pattern_sub, replacement_sub, inputs, fwd_only, my_patterns
        )

        def custom_pass(graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Post grad pass that applies the patterns and
            # returns the count.
            return my_patterns.apply(graph)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sum of (x plus y) and (x minus y).
            return (x + y) + (x - y)

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        with unittest.mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_PATTERN_MATCH_DEBUG": "all"}
        ):
            compiled_fn = torch.compile(
                fn, options={"post_grad_custom_post_pass": custom_pass}
            )
            _ = compiled_fn(x, y)
        self.assertTrue(self.hasRecord(records, "Specific pattern match: add"))
        self.assertTrue(self.hasRecord(records, "Specific pattern match: sub"))

    def test_gumbel_max_trick(self):
        # [FUNCTION DOC] Statistical test of the Gumbel max trick rewrite for sampling: compiles
        # a sampling function (temperature scaled softmax divided by exponential noise then
        # argmax), draws one million samples from logits with a known distribution, and requires
        # each category frequency within 10 percent relative error of expected; also asserts the
        # apply_gumbel_max_trick counter equals one. Can be flaky by nature (random sampling).
        counters.clear()

        @torch.compile
        def sample(logits, temperature):
            # [FUNCTION DOC] HELPER: Compiled sampler: divides logits by max of temperature and
            # a small floor, applies softmax, divides probabilities by exponential noise and
            # returns the argmax index per row as int.
            logits = logits / max(temperature, 1e-5)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            q = torch.empty_like(logits).exponential_(1)
            return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

        N = 10
        temperature = 0.8
        row = (
            torch.arange(1, N + 1, dtype=torch.float, device=GPU_TYPE).log()
            * temperature
        )
        expected_distribution = []
        tot_val = N * (N + 1) / 2
        for i in range(1, N + 1):
            expected_distribution.append(float(i) / tot_val)

        # Item 0 expect to appear M / (1 + 2 +...+ N) times. Make M large enough
        # so the test is less flaky.
        # If this is still flaky, either recduce N or increase M
        M = 1000000
        logits = row[None, :].repeat(M, 1)
        output = sample(logits, temperature=temperature)
        stat = (torch.bincount(output.flatten()) / M).tolist()

        for expected, actual in zip(expected_distribution, stat):
            tol = 0.1
            ratio = actual / expected

            self.assertTrue(
                abs(ratio - 1) < tol, lambda msg: f"{msg}\n{expected} v.s. {actual}"
            )

        self.assertTrue(counters["inductor"]["apply_gumbel_max_trick"] == 1)

    def test_per_pattern_counter(self):
        # [FUNCTION DOC] With the pattern match debug variable set to 1, compiling a bmm records
        # a per-pattern counter dictionary; asserts it is a nonempty dict and that the key for
        # the bmm call function pattern has the value one.
        """Test that per-pattern counters track individual pattern matches"""
        with unittest.mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_PATTERN_MATCH_DEBUG": "1"}
        ):
            counters.clear()

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Batched matmul of x and y.
                return torch.bmm(x, y)

            x = torch.randn(4, 10, 10, device=GPU_TYPE)
            y = torch.randn(4, 10, 10, device=GPU_TYPE)

            compiled = torch.compile(fn)
            compiled(x, y)

            counter_key = "inductor_pattern_matcher_per_pattern"
            per_pattern = counters.get(counter_key, None)

            self.assertIsInstance(per_pattern, dict)
            self.assertGreater(len(per_pattern), 0)
            self.assertIn("CallFunction_aten.bmm.default", per_pattern)
            self.assertEqual(per_pattern["CallFunction_aten.bmm.default"], 1)

    def test_per_pattern_counter_accumulation(self):
        # [FUNCTION DOC] Per-pattern counters accumulate across compilations: counts
        # after compiling a bmm function, then a second function without clearing
        # counters, must equal the first count plus a count measured by recompiling the
        # second function on cleared counters.
        """Test that per-pattern counters accumulate across compilations"""
        with unittest.mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_PATTERN_MATCH_DEBUG": "1"}
        ):
            counter_key = "inductor_pattern_matcher_per_pattern"

            counters.clear()

            x = torch.randn(2, 10, 10, device=GPU_TYPE)
            y = torch.randn(2, 10, 10, device=GPU_TYPE)

            def fn1(a, b):
                # [FUNCTION DOC] HELPER: Batched matmul.
                return torch.bmm(a, b)

            compiled1 = torch.compile(fn1)
            compiled1(x, y)
            count1 = sum(counters.get(counter_key, {}).values())

            # Compile second function without clearing counters
            def fn2(a, b):
                # [FUNCTION DOC] HELPER: Batched matmul times two.
                return torch.bmm(a, b) * 2

            compiled2 = torch.compile(fn2)
            compiled2(x, y)
            accumulated_count = sum(counters.get(counter_key, {}).values())

            # Verify accumulation
            counters.clear()
            torch._dynamo.reset()
            compiled2 = torch.compile(fn2)
            compiled2(x, y)
            count2 = sum(counters.get(counter_key, {}).values())

            self.assertEqual(accumulated_count, count1 + count2)

    def test_list_tensor_pattern_replacement(self):
        # [FUNCTION DOC] Registers a replacement for a custom op taking and returning a list of
        # tensors (defined in a scoped library with a fake implementation) that returns its
        # input list unchanged. For list lengths one and two, a custom backend applying the pass
        # must report one replacement, the result must equal the inputs, and the op must be gone
        # from the graph.
        with torch.library._scoped_library("_test_pm_list", "FRAGMENT") as lib:
            lib.define("list_op(Tensor[] xs) -> Tensor[]")
            lib.impl(
                "list_op",
                lambda xs: [x + 1 for x in xs],
                "CompositeExplicitAutograd",
            )

            @torch.library.register_fake("_test_pm_list::list_op", lib=lib)
            def list_op_fake(xs):
                # [FUNCTION DOC] HELPER: Fake implementation returning the input list.
                return xs

            def src_pattern(xs: list[torch.Tensor]):
                # [FUNCTION DOC] HELPER: Pattern: calls the custom list op on xs.
                return torch.ops._test_pm_list.list_op.default(xs)

            def target_pattern(xs: list[torch.Tensor]):
                # [FUNCTION DOC] HELPER: Replacement: returns xs.
                return xs

            class Backend:
                # [CLASS DOC] HELPER: Torch compile backend that registers the list pattern at
                # construction, applies it to the graph on call, records the count and graph,
                # lints, recompiles and returns the forward.
                def __init__(self, example_inputs) -> None:
                    # [FUNCTION DOC] HELPER: Creates the pattern pass, zeroes the count, clears
                    # the graph and registers the replacement with a list of the example inputs.
                    self.pm = PatternMatcherPass()
                    self.count = 0
                    self.graph = None
                    register_replacement(
                        src_pattern,
                        target_pattern,
                        [example_inputs],
                        fwd_only,
                        self.pm,
                    )

                def __call__(self, gm, example_inputs):
                    # [FUNCTION DOC] HELPER: Applies the pass to the graph, stores the
                    # replacement count, lints and recompiles the module, saves the graph and
                    # returns the module forward function.
                    self.count = self.pm.apply(gm.graph)
                    gm.graph.lint()
                    gm.recompile()
                    self.graph = gm.graph
                    return gm.forward

            def run_case(length):
                # [FUNCTION DOC] HELPER: Resets dynamo, builds the backend for a list of the
                # given length of empty tensors, compiles the custom list op call, runs it on
                # random inputs and asserts one replacement, equal outputs and that the op is
                # absent from the recorded graph (fails the test if no graph was recorded).
                torch._dynamo.reset()
                backend = Backend([torch.empty(4, 5) for _ in range(length)])

                def fn(xs):
                    # [FUNCTION DOC] HELPER: Calls the custom list op on xs.
                    return torch.ops._test_pm_list.list_op.default(xs)

                xs = [torch.randn(4, 5) for _ in range(length)]
                result = torch.compile(fn, backend=backend)(xs)

                self.assertEqual(backend.count, 1)
                self.assertEqual(len(result), length)
                for actual, expected in zip(result, xs):
                    self.assertEqual(actual, expected)

                if backend.graph is None:
                    self.fail("Expected the custom backend to record a graph")
                op_targets = [
                    n.target for n in backend.graph.nodes if n.op == "call_function"
                ]
                self.assertNotIn(torch.ops._test_pm_list.list_op.default, op_targets)

            run_case(1)
            run_case(2)

    def test_scalar_workaround_arg_survives_specific_match(self):
        # [FUNCTION DOC] Checks a replacement with a scalar workaround on the dim argument: a
        # softmax written as amax, subtract, exp, sum and divide over dim one is replaced by
        # softmax even though the pattern was traced with dim negative one. The extra check
        # records the matched dim; asserts one replacement, recorded dim equals one, and results
        # equal softmax.
        def src_pattern(x, dim):
            # [FUNCTION DOC] HELPER: Pattern: explicit softmax decomposition over the given dim.
            xmax = x.amax(dim=dim, keepdim=True)
            xsub = x - xmax
            xexp = xsub.exp()
            xsum = xexp.sum(dim=dim, keepdim=True)
            return xexp / xsum

        def target_pattern(x, dim):
            # [FUNCTION DOC] HELPER: Replacement: softmax over the given dim.
            return torch.softmax(x, dim=dim)

        def fn(x):
            # [FUNCTION DOC] HELPER: Explicit softmax decomposition over dim one.
            xmax = x.amax(dim=1, keepdim=True)
            xsub = x - xmax
            xexp = xsub.exp()
            xsum = xexp.sum(dim=1, keepdim=True)
            return xexp / xsum

        patterns = PatternMatcherPass()
        dims = []

        def extra_check(match):
            # [FUNCTION DOC] HELPER: Match check that appends the matched dim keyword to the
            # enclosing list and accepts.
            dims.append(match.kwargs["dim"])
            return True

        register_replacement(
            src_pattern,
            target_pattern,
            [torch.empty(4, 8)],
            fwd_only,
            patterns,
            extra_check=extra_check,
            scalar_workaround={"dim": -1},
        )

        x = torch.randn(4, 8)
        gm = fwd_only(fn, [x])
        count = patterns.apply(gm.graph)
        gm.graph.lint()
        gm.recompile()

        self.assertEqual(count, 1)
        self.assertEqual(dims, [1])
        self.assertEqual(gm(x), torch.softmax(x, dim=1))

    def test_opaque_obj_custom_op(self):
        # [FUNCTION DOC] Replacement involving an opaque object argument: in a scoped library
        # defines an original op and a replacement op taking a tensor and the OpaqueScaleFactor
        # (with fake implementations), registers the pattern, runs a compiled function with a
        # custom backend; asserts the result is input plus 4 (replacement op multiplication
        # swapped for add), one replacement, and the graph holds the replacement op but no
        # original op.
        with torch.library._scoped_library("_test_pm", "FRAGMENT") as lib:
            lib.define(
                f"original_op(Tensor x, {get_opaque_type_name(OpaqueScaleFactor)} s) -> Tensor"
            )
            lib.impl("original_op", lambda x, s: x * s.val, "CompositeExplicitAutograd")

            @torch.library.register_fake("_test_pm::original_op", lib=lib)
            def _orig_fake(x, s):
                # [FUNCTION DOC] HELPER: Fake implementation for the original op returning an
                # empty tensor like x.
                return torch.empty_like(x)

            lib.define(
                f"replacement_op(Tensor x, {get_opaque_type_name(OpaqueScaleFactor)} s) -> Tensor"
            )
            lib.impl(
                "replacement_op", lambda x, s: x + s.val, "CompositeExplicitAutograd"
            )

            @torch.library.register_fake("_test_pm::replacement_op", lib=lib)
            def _repl_fake(x, s):
                # [FUNCTION DOC] HELPER: Fake implementation for the replacement op returning an
                # empty tensor like x.
                return torch.empty_like(x)

            def pattern(x, factor):
                # [FUNCTION DOC] HELPER: Pattern: calls the original op.
                return torch.ops._test_pm.original_op(x, factor)

            def replacement(x, factor):
                # [FUNCTION DOC] HELPER: Replacement: calls the replacement op.
                return torch.ops._test_pm.replacement_op(x, factor)

            patterns = PatternMatcherPass()
            register_replacement(
                pattern,
                replacement,
                [torch.randn(4, 4), OpaqueScaleFactor(2.0)],
                fwd_only,
                patterns,
            )

            count = 0
            post_pass_graph = None

            def custom_pass(graph):
                # [FUNCTION DOC] HELPER: Post grad pass that applies the patterns, stores the
                # count and graph in enclosing variables and returns the graph.
                nonlocal count, post_pass_graph
                count = patterns.apply(graph)
                post_pass_graph = graph
                return graph

            def custom_backend(graph, example_inputs):
                # [FUNCTION DOC] HELPER: Backend copying the inductor config, setting the custom
                # post grad pass and compiling the graph with compile_fx.
                from torch._inductor.compile_fx import compile_fx

                current_config = inductor_config.get_config_copy()
                current_config["post_grad_custom_post_pass"] = custom_pass
                return compile_fx(graph, example_inputs, config_patches=current_config)

            @torch.compile(backend=custom_backend)
            def f(x, s):
                # [FUNCTION DOC] HELPER: Compiled function calling the original op.
                return torch.ops._test_pm.original_op(x, s)

            inp = torch.randn(4, 4)
            result = f(inp, OpaqueScaleFactor(4.0))
            self.assertEqual(result, inp + 4.0)
            self.assertEqual(count, 1)
            op_targets = [
                n.target for n in post_pass_graph.nodes if n.op == "call_function"
            ]
            self.assertNotIn(torch.ops._test_pm.original_op.default, op_targets)
            self.assertIn(torch.ops._test_pm.replacement_op.default, op_targets)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
