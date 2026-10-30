# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor's multi-kernel feature, where several variants of a generated
# kernel (for example persistent and non-persistent reductions, or kernels specialized to size
# hints) are compiled and one is picked by benchmarking at run time. Defines a small transformer-
# like module, regex helpers that inspect generated wrapper code, a factory that turns a test
# into a cpp-wrapper variant, and the test class, whose config patch enables multi kernel
# (default taken from the environment) and kernel benchmarking. The main block runs the tests
# only with a GPU.

import os
import re
import unittest

import torch
from torch import nn
from torch._dynamo.testing import reset_rng_state
from torch._inductor import config, test_operators
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.nn import functional as F
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    IS_BIG_GPU,
    requires_triton,
)


class TransformerSnippet(nn.Module):
    # [CLASS DOC] HELPER: Small module with two layer norms over 64 features. Its forward applies
    # dropout to the inputs, normalizes the second one, adds them and normalizes the sum. State:
    # two layer norm submodules.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the two layer norm submodules of size 64.
        super().__init__()
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)

    def forward(self, x1, x2):
        # [FUNCTION DOC] HELPER: Applies dropout with probability 0.1 to x1, layer norm one then
        # dropout to x2, and returns layer norm two of their sum.
        x1 = F.dropout(x1, 0.1)
        x2 = F.dropout(self.ln1(x2), 0.1)

        return self.ln2(x1 + x2)

    def example_inputs(self):
        # [FUNCTION DOC] HELPER: Returns a tuple of two random 2 by 64 tensors on the GPU device.
        return (torch.randn(2, 64).to(GPU_TYPE), torch.randn(2, 64).to(GPU_TYPE))


def _contains_multi_kernel_code(wrapper_code: str):
    # [FUNCTION DOC] HELPER: Returns true when the wrapper source text contains an assignment from
    # the async compile multi kernel constructor (the plain, not the size-hint, variant).
    return (
        re.search(r"multi_kernel_[^ ]* = async_compile.multi_kernel[(]", wrapper_code)
        is not None
    )


def _contains_size_hint_multi_kernel_code(wrapper_code: str):
    # [FUNCTION DOC] HELPER: Returns true when the wrapper source text contains an assignment from
    # the async compile size-hint multi kernel constructor.
    return (
        re.search(
            r"multi_kernel_[^ ]* = async_compile.size_hint_multi_kernel[(]",
            wrapper_code,
        )
        is not None
    )


def make_cpp_wrapper_test(orig_test, **extra_args):
    # [FUNCTION DOC] HELPER: Wraps a test function into a new test function that enables the
    # cpp_wrapper config and compile-time autotuning, clears the code cache first so kernels get
    # recompiled with cpp wrapper, and then calls the original test with the extra keyword
    # arguments. Returns the new function. Kept as a free function because of a static method
    # problem on old Python versions.
    """
    Wrap an existing test into a new test with cpp-wrapper enabled.

    Make this as a free function rather than staticmethod in MultiKernelTest.
    Otherwise we get 'TypeError: 'staticmethod' object is not callable'
    error in py3.8. (py3.10 works)
    """

    @config.patch("cpp_wrapper", True)
    @config.patch("triton.autotune_at_compile_time", True)
    def fn(self):
        # [FUNCTION DOC] HELPER: The produced test body: clears the python code cache, then
        # calls the wrapped test with the saved keyword arguments. Config patches are
        # applied by decorators.
        # The same kernel may have been compiled by previous tests with
        # cpp_wrapper disabled. Clear the cache so we go ahead to re-compile
        # the kernel with cpp_wrapper enabled.
        from torch._inductor import codecache

        codecache.PyCodeCache.cache_clear()
        return orig_test(self, **extra_args)

    return fn


@config.patch(
    {
        "triton.multi_kernel": int(os.environ.get("TORCHINDUCTOR_MULTI_KERNEL", "1")),
        "benchmark_kernel": True,
        "multi_kernel_hints": [64, 256, 4096],
    }
)
@instantiate_parametrized_tests
class MultiKernelTest(TestCase):
    # [CLASS DOC] Test case for multi-kernel behavior. The class-level config patch turns on the
    # multi kernel option (value from an environment variable, default one), kernel benchmarking
    # and a list of size hints. Includes softmax, gemm, layernorm, transformer, batch norm and
    # scratch-buffer reduction cases plus cpp-wrapper variants assigned as class attributes.
    def test_softmax(self, expect_multi_kernel=True):
        # [FUNCTION DOC] Compiles softmax over a 2 by 1024 GPU tensor, compares with eager and
        # checks the last generated wrapper code contains (or, if expect_multi_kernel is false,
        # does not contain) a multi kernel. Also reused by other tests.
        x = torch.rand(2, 1024).to(GPU_TYPE)
        ref = torch.softmax(x, -1)
        compiled_fn = torch.compile(torch.softmax)
        act, wrapper_code = run_and_get_code(compiled_fn, x, -1)

        # wrapper_code will contains 2 entries if cpp_wrapper=True.
        # One for the first pass and one for the second pass.
        # We mainly care about the wrapper for the final pass here.
        wrapper_code = wrapper_code[-1]
        self.assertEqual(ref, act)
        if expect_multi_kernel:
            self.assertTrue(_contains_multi_kernel_code(wrapper_code))
        else:
            self.assertFalse(_contains_multi_kernel_code(wrapper_code))

    @requires_triton()
    # TODO: bobrenjc93 to fix multi-kernel for ROCM
    @skipIfRocm
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    @skipIfXpu(msg="driver issue, torch-xpu-ops: 2295")
    def test_triton_gemm(self):
        # [FUNCTION DOC] Compiles a matmul of 4096 square tensors with the first dim dynamic under
        # max autotune using only triton gemm templates. Checks the result matches eager and that
        # the final wrapper code uses a size-hint multi kernel. Skipped without triton, on ROCm,
        # XPU or a small GPU.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Matrix product of x and y.
            return x @ y

        compiled_fn = torch.compile(
            fn,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )
        x = torch.randn(4096, 4096, device=GPU_TYPE)
        y = torch.randn(4096, 4096, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)
        act, wrapper_code = run_and_get_code(compiled_fn, x, y)
        ref = fn(x, y)

        # wrapper_code will contains 2 entries if cpp_wrapper=True.
        # One for the first pass and one for the second pass.
        # We mainly care about the wrapper for the final pass here.
        wrapper_code = wrapper_code[-1]
        self.assertEqual(ref, act)
        self.assertTrue(_contains_size_hint_multi_kernel_code(wrapper_code))

    @skipIfXpu(msg="driver issue, torch-xpu-ops: 2295")
    @requires_triton()
    # TODO: bobrenjc93 to fix multi-kernel for ROCM
    @skipIfRocm
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_triton_relu_fused_gemm(self):
        # [FUNCTION DOC] Same as the triton gemm test but with a relu fused after the matmul.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Matrix product of x and y followed by relu.
            return (x @ y).relu()

        compiled_fn = torch.compile(
            fn,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )
        x = torch.randn(4096, 4096, device=GPU_TYPE)
        y = torch.randn(4096, 4096, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)
        act, wrapper_code = run_and_get_code(compiled_fn, x, y)
        ref = fn(x, y)

        # wrapper_code will contains 2 entries if cpp_wrapper=True.
        # One for the first pass and one for the second pass.
        # We mainly care about the wrapper for the final pass here.
        wrapper_code = wrapper_code[-1]
        self.assertEqual(ref, act)
        self.assertTrue(_contains_size_hint_multi_kernel_code(wrapper_code))

    @parametrize("force_kernel", (0, 1))
    @unittest.mock.patch.dict(
        os.environ, {"TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE": "1"}
    )
    def test_softmax_force_non_persistent_reduction(self, force_kernel):
        # [FUNCTION DOC] With the multi kernel cache disabled by an environment patch, forces the
        # benchmark to report one sub-kernel as fastest (index given by force_kernel) by patching
        # the benchmark method, wraps the run method to record the picked kernel, compiles
        # softmax plus a constant and asserts the picked kernel index equals force_kernel.
        # Parametrized over zero/one.
        """
        Force a specific sub-kernel being picked by mocking the benchmark result.
        """
        x = torch.rand(2, 1024).to(GPU_TYPE)
        mock_latency = [0.2, 0.2]
        mock_latency[force_kernel] = 0.1  # this make sure force_kernel will be picked

        def f(x):
            # [FUNCTION DOC] HELPER: Softmax over the last dim plus the force_kernel integer.
            return torch.softmax(x, -1) + force_kernel

        orig_run = MultiKernelCall.run
        picked_kernel = None

        def mock_run(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement run method of the multi kernel call: calls the
            # original run, records which sub-kernel was picked in the enclosing variable and
            # returns the output.
            out = orig_run(self, *args, **kwargs)
            nonlocal picked_kernel
            picked_kernel = self.picked_kernel
            return out

        with (
            unittest.mock.patch.object(MultiKernelCall, "run", mock_run),
            unittest.mock.patch.object(
                MultiKernelCall,
                "benchmark_sub_kernels",
                lambda *args, **kwargs: mock_latency,
            ),
        ):
            torch.compile(f)(x)
        self.assertEqual(picked_kernel, force_kernel)

    @config.patch("warn_mix_layout", True)
    def test_softmax_warn_mixed_layout(self):
        # [FUNCTION DOC] Runs the softmax test with the warn-on-mixed-layout config enabled.
        self.test_softmax()

    test_softmax_cpp_wrapper = make_cpp_wrapper_test(
        test_softmax, expect_multi_kernel=True
    )

    def test_layernorm(self):
        # [FUNCTION DOC] Compares a compiled layer norm of size 1024 against eager on a 2 by 1024
        # input with tolerance 1e-4.
        ln = nn.LayerNorm(1024).to(GPU_TYPE)
        x = torch.rand(2, 1024).to(GPU_TYPE)
        ref = ln(x)
        act = torch.compile(ln)(x)
        self.assertEqual(ref, act, atol=1e-4, rtol=1e-4)

    def test_inplace_update(self):
        # [FUNCTION DOC] Compiles a function multiplying a row sum by a matrix product
        # (which makes inductor emit an in-place kernel for the multiplication) and checks
        # it matches eager.
        """
        Inductor generate inplace kernel for mul.
        """

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the row sum of x with kept dim times the product of
            # y with itself.
            return x.sum(dim=-1, keepdims=True) * (y @ y)

        x = torch.rand(1024, 1024).to(GPU_TYPE)
        y = torch.rand(1024, 1024).to(GPU_TYPE)
        ref = f(x, y)
        act = torch.compile(f)(x, y)
        self.assertEqual(ref, act)

    def test_transformer_snippet(self):
        # [FUNCTION DOC] Runs the transformer snippet model eagerly and compiled with the same
        # random state. Results are compared (tolerance 1e-4) only when the fallback_random config
        # is on, since inductor's random numbers differ from eager.
        model = TransformerSnippet().to(GPU_TYPE)
        x = model.example_inputs()

        def f(*x):
            # [FUNCTION DOC] HELPER: Calls the model on the unpacked arguments and
            # returns the output.
            y = model(*x)
            return y

        reset_rng_state()
        ref = f(*x)

        opt_f = torch.compile(f)
        reset_rng_state()
        act = opt_f(*x)

        # don't compare tensor if using inductor random number generator.
        # inductor random number implementation is different to eager.
        # We should fallback to eager if we want to test accuracy.
        if config.fallback_random:
            self.assertEqual(ref, act, atol=1e-4, rtol=1e-4)

    def test_transformer_snippet_with_fallback_random(self):
        # [FUNCTION DOC] Runs the transformer snippet test with fallback_random enabled so random
        # numbers come from eager and results can be compared.
        """
        Same as test_transformer_snippet but fallback the random number
        generator to eager so we can check accuracy.
        """
        with config.patch("fallback_random", True):
            self.test_transformer_snippet()

    def test_batchnorm_training(self):
        # [FUNCTION DOC] Compiles a function running batch norm training forward and
        # backward on a 2x3x8x8 input, where running stats tensors are passed twice (input
        # and output) to a generated kernel, and asserts the last of two wrapper code
        # outputs contains a multi kernel.
        """
        For training, batchnorm will tracking running mean/variance during forward pass.
        The kernel generated by inductor currently will pass in those tensors twice as arguments:
        once for input and once for output. They are ruled out as in-out argument because
        they are considered as graph inputs.

        Multi-kernel previously assumes that we never pass the same argument multi times
        for a kernel. No matter if we change inductor behavior to assure that, it's better
        to make multi-kernel being able to handle those cases.
        """
        bn = nn.BatchNorm2d(3).to(GPU_TYPE)

        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Runs batch norm on x, sums it and runs backward.
            bn(x).sum().backward()

        _, (wrapper_code, _) = run_and_get_code(
            f, torch.randn(2, 3, 8, 8, device=GPU_TYPE)
        )
        self.assertTrue(_contains_multi_kernel_code(wrapper_code))

    def test_pass_same_arg_multi_times(self):
        # [FUNCTION DOC] Compiles an update of running-stats style: the sum over dim one of x is
        # mixed into y in place with weights 0.9 and 0.1. Checks the in-place updated y matches
        # the eager result on a clone.
        """
        A super simple example that simulate how BatchNorm update the running
        stats.

        Inductor currently pass the same tensor multiple times for the generated
        kernel: once for input and once for output.

        Here is a paster for the generated kernel (without multi-kernel enabled):
        https://gist.github.com/shunting314/f0b446b4b9a28f4940e31dcd3e809cf9
        """

        def f(x, y):
            # [FUNCTION DOC] HELPER: Sums x over dim 1 and overwrites y in place with 0.9 times y
            # plus 0.1 times that sum.
            x = x.sum(dim=1, keepdim=False)
            y.copy_(y * 0.9 + x * 0.1)

        x = torch.randn(8, 16, device=GPU_TYPE)
        y = torch.randn(8, device=GPU_TYPE)
        y_ref = y.clone()

        ref = f(x, y_ref)  # noqa: F841
        act = torch.compile(f)(x, y)  # noqa: F841
        self.assertEqual(y_ref, y)

    def test_reduction_scratch_buffer(self, force_multi_kernel=1):
        # [FUNCTION DOC] Compiles a function with two row sums separated by an explicitly realized
        # buffer under a config patch of the multi kernel option set to force_multi_kernel
        # (default one), and compares with eager. The realized buffer is a scratch argument for
        # non-persistent reductions only, so kernel variants have different argument lists. Also
        # reused with other force values.
        """
        The explicitly realized buffer in the test function will be passed in
        as a scratch buffer for the non-persistent reduction kernel but
        can be skipped for the persistent reduction kernel.

        This causes different argument lists for non-persistent reduction kernel and
        persistent reduction kernel.

        Check documentation around torch._inductor.config.triton.multi_kernel about
        how to interpret the force_multi_kernel argument.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Adds the keepdim row sum to x, forces realization of the
            # result, then repeats the row sum addition and returns it.
            x = x.sum(dim=-1, keepdim=True) + x
            x = test_operators.realize(x)
            x = x.sum(dim=-1, keepdim=True) + x
            return x

        x = torch.rand(16, 16, device=GPU_TYPE)
        ref = f(x)
        with config.patch("triton.multi_kernel", force_multi_kernel):
            act = torch.compile(f)(x)
        self.assertEqual(ref, act)

    def test_split_scan(self, force_multi_kernel=1):
        # [FUNCTION DOC] Compares compiled and eager cumulative sum of a flattened 10x3x352x352
        # nonnegative tensor with the multi kernel option patched to force_multi_kernel.
        def f(x):
            # [FUNCTION DOC] HELPER: Flattens x and returns its cumulative sum over dim 0.
            x = x.view(-1)
            return torch.cumsum(x, 0)

        x = make_tensor(10, 3, 352, 352, low=0, dtype=torch.float32, device=GPU_TYPE)
        expect = f(x)
        with config.patch("triton.multi_kernel", force_multi_kernel):
            actual = torch.compile(f)(x)
        self.assertEqual(expect, actual)

    def test_sort_disables_multi_kernel(self, force_multi_kernel=1):
        # [FUNCTION DOC] Sort needs a persistent kernel so multi kernel cannot apply; checks that
        # compiling a sort of a 32 by 32 tensor with the multi kernel option patched on still
        # gives the eager result.
        """
        Sort currently requires a persistent kernel, so multi-kernel is not
        possible. Make sure this falls back gracefully.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the values of x sorted along the last dim.
            return x.sort(-1).values

        x = torch.rand(32, 32, device=GPU_TYPE)
        expect = f(x)
        with config.patch("triton.multi_kernel", force_multi_kernel):
            actual = torch.compile(f)(x)
        self.assertEqual(expect, actual)

    # Use benchmarking to pick the faster kernel
    test_reduction_scratch_buffer_cpp_wrapper = make_cpp_wrapper_test(
        test_reduction_scratch_buffer, force_multi_kernel=1
    )
    # force pick persistent reduction. This can be a good test since this persistent
    # reduction uses less call arguments than the corresponding non-persistent
    # reduction.
    test_reduction_scratch_buffer_cpp_wrapper_persistent_reduction = (
        make_cpp_wrapper_test(test_reduction_scratch_buffer, force_multi_kernel=2)
    )
    # force pick non-persistent reduction
    test_reduction_scratch_buffer_cpp_wrapper_non_persistent_reduction = (
        make_cpp_wrapper_test(test_reduction_scratch_buffer, force_multi_kernel=3)
    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
