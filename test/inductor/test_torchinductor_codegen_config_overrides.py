# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that individual inductor config overrides change generated code as
# intended (forced pointwise cat, block pointer CSE, disabling Welford reductions, fusion size
# thresholds). Each test compiles a small function, compares against eager, and counts
# substrings in the generated source. Import-time side effects: requires the filelock module
# (import error if missing). The main block runs the inductor test runner only when a CPU or
# GPU is available.
import importlib
from collections.abc import Callable
from typing import Any
from unittest import skipIf

import torch
import torch.utils._pytree as pytree
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)


importlib.import_module("filelock")


@instantiate_parametrized_tests
class CodegenInductorTest(InductorTestCase):
    # [CLASS DOC] Inductor test case whose tests patch config values, compile a function
    # with the inductor backend and inspect the generated code text. Has two shared
    # helpers: run_and_compare (compile, run, compare to eager, return code) and count_code
    # (count a substring over all generated programs). Parametrized tests are expanded by
    # the class decorator.
    def run_and_compare(
        self,
        func: Callable[..., Any],
        *args,
        compile_kwargs: dict | None = None,
        config_patches: dict | None = None,
        atol: float | None = 1e-05,
        rtol: float | None = 1e-08,
    ):
        # [FUNCTION DOC] Compiles func with the inductor backend (compile_kwargs forwarded to
        # torch.compile) under an inductor config patch built from config_patches, runs it on
        # args via run_and_get_code, then flattens both the eager and compiled outputs with
        # pytree and asserts each pair of leaves is close using the given atol and rtol (note
        # the default rtol is very small). Returns the compiled result and the list of
        # generated source strings. None for either dict argument means empty.
        # KNOWN ISSUE: zip stops at the shorter output list, so a missing output in the
        # compiled result would not be reported.
        """
        Runs the module through Inductor, comparing to eager reference.
        """
        if compile_kwargs is None:
            compile_kwargs = {}
        if config_patches is None:
            config_patches = {}

        def flatten_tensors(tensors):
            # [FUNCTION DOC] HELPER: Flattens a nested structure of tensors with pytree and
            # returns only the list of leaves, discarding the tree spec.
            flat, spec = pytree.tree_flatten(tensors)
            return flat

        with config.patch(config_patches):
            compiled = torch.compile(func, backend="inductor", **compile_kwargs)
            result, code = run_and_get_code(compiled, *args)

        # Check numerical accuracy
        ref_tensors = flatten_tensors(func(*args))
        actual_tensors = flatten_tensors(result)
        for ref, actual in zip(ref_tensors, actual_tensors):
            self.assertTrue(torch.allclose(ref, actual, atol=atol, rtol=rtol))

        return result, code

    def count_code(self, substr: str, code: list[str], expected: int | None):
        # [FUNCTION DOC] HELPER: Adds up how many times substr occurs in each generated
        # program string. If expected is not None the total must equal it (assertEqual); with
        # None nothing is checked. Returns nothing.
        count = sum(prog.count(substr) for prog in code)
        if expected is not None:
            self.assertEqual(count, expected)

    @parametrize("force_pointwise_cat", [False, True])
    def test_force_pointwise_cat(self, force_pointwise_cat: bool):
        # [FUNCTION DOC] Parametrized over force_pointwise_cat. Compiles a cat of two shifted
        # 1024-element CPU tensors with that config flag, then counts reinterpret-tensor calls
        # in the generated code (the cpp wrapper uses a different call name). Expects zero
        # when pointwise cat is forced and two otherwise.
        def func(a, b):
            # [FUNCTION DOC] HELPER: Concatenates a plus one with b plus two along dim 0.
            return torch.cat([a + 1, b + 2], dim=0)

        a = torch.randn(1024, device=torch.device("cpu"))
        b = torch.randn(1024, device=torch.device("cpu"))
        config_patches = {
            "force_pointwise_cat": force_pointwise_cat,
        }
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
        )

        reinterpret_call = (
            "= reinterpret_tensor_wrapper("
            if config.cpp_wrapper
            else "= reinterpret_tensor("
        )
        if force_pointwise_cat:
            self.count_code(reinterpret_call, code, 0)
        else:
            self.count_code(reinterpret_call, code, 2)

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Triton is not available for MPS")
    def test_cse_make_block_ptr_reduction(self):
        # [FUNCTION DOC] GPU only (skipped for MPS). Compiles a sum-reduction over a 512 by
        # 4096 pair of GPU tensors with block pointers, reduction tiling, nd tiling, up to
        # three tiles and split reductions all configured, using atol 1e-4. Expects exactly
        # two make_block_ptr from input pointers and two block pointer loads, i.e. the shared
        # operand loads are deduplicated.
        def func(a, b):
            # [FUNCTION DOC] HELPER: Computes the elementwise product and sum of a
            # and b, adds them, and reduces over dim 0 so that both operands are
            # reused by two expressions.
            tmp0 = a * b
            tmp1 = a + b
            c = tmp0 + tmp1
            return c.sum(dim=0)

        config_patches = {
            "triton.use_block_ptr": True,
            "triton.tile_reductions": True,
            "triton.prefer_nd_tiling": True,
            "triton.max_tiles": 3,
            "split_reductions": False,
        }
        a = torch.randn((512, 4096), device=torch.device(GPU_TYPE))
        b = torch.randn((512, 4096), device=torch.device(GPU_TYPE))
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
            atol=1e-4,
        )
        self.count_code("= tl.make_block_ptr(in_ptr", code, 2)
        self.count_code("= tl.load(block_ptr", code, 2)

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Triton is not available for MPS")
    @parametrize("disable_welford_reduction", [True, False])
    def test_disable_welford_reduction(self, disable_welford_reduction: bool):
        # [FUNCTION DOC] GPU only (skipped for MPS). Parametrized over the mtia
        # disable_welford_reduction config flag. Compiles var_mean over a 4 by 65536
        # GPU tensor (wide enough that codegen would prefer Welford) with loose
        # tolerances, then counts Welford helper uses in the generated code: zero when
        # disabled, positive otherwise.
        def func(x):
            # [FUNCTION DOC] HELPER: Returns the variance and mean of x over dim 1.
            return torch.var_mean(x, dim=1)

        # Use a reduction larger than the CUDA two-step variance threshold to
        # force codegen to prefer Welford reduction, in order to test
        # effectiveness of config flag disable_welford_reduction.
        # This test should run fine on GPU as the configuration is not specific to MTIA backend.
        x = torch.randn((4, 65536), device=torch.device(GPU_TYPE))
        config_patches = {
            "mtia.disable_welford_reduction": disable_welford_reduction,
        }
        _, code = self.run_and_compare(
            func,
            x,
            config_patches=config_patches,
            atol=1e-2,
            rtol=1e-4,
        )

        welford_count = sum(prog.count("triton_helpers.welford") for prog in code)
        if disable_welford_reduction:
            self.assertEqual(welford_count, 0)
        else:
            self.assertGreater(welford_count, 0)

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Triton is not available for MPS")
    def test_kernel_fusion_thresholds(self):
        # [FUNCTION DOC] GPU only (skipped for MPS). Sets max_fusion_size and the realize read
        # and opcount thresholds to one and disables inplace buffers, compiles a chain of
        # pointwise ops over 1024-element GPU tensors and expects exactly three Triton jit
        # kernels in the generated code (fusion is effectively prevented).
        def func(a, b):
            # [FUNCTION DOC] HELPER: Chain of three scalar additions to a, then adds b;
            # returns the first, third and fourth intermediates so they must be materialized.
            tmp0 = a + 1
            tmp1 = tmp0 + 2
            tmp2 = tmp1 + 3
            tmp3 = tmp2 + b
            return tmp0, tmp2, tmp3

        a = torch.randn(1024, device=torch.device(GPU_TYPE))
        b = torch.randn(1024, device=torch.device(GPU_TYPE))
        config_patches = {
            "max_fusion_size": 1,
            "realize_reads_threshold": 1,
            "realize_opcount_threshold": 1,
            "inplace_buffers": False,
        }
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
        )
        self.count_code("@triton.jit", code, 3)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or HAS_CPU:
        run_tests(needs="filelock")
