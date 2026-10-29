# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor freezing (optimize-for-inference) on CPU and GPU: constant
# folding, conv+batchnorm folding, concatenated matmul fusion, layout conversion of conv
# weights, and parameter deallocation. Defines a base TestCase that turns freezing on via
# config, small conv/batchnorm modules, and a template test class that is copied into concrete
# CPU and GPU classes with copy_tests, then deleted. Import-time side effects: adds the test
# directory to sys.path, imports functorch and filelock, and on ROCm sets the
# force_layout_optimization config and the MIOpen NHWC environment variable.
import contextlib
import copy
import functools
import importlib
import itertools
import os
import sys
import unittest
import weakref

import torch
from torch import nn
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import override_lowering, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater, tf32_on_and_off
from torch.testing._internal.common_utils import IS_FBCODE, TEST_WITH_SLOW_GRADCHECK


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
)
from torch.testing._internal.common_utils import TEST_WITH_ROCM


importlib.import_module("functorch")
importlib.import_module("filelock")

from torch.testing._internal.common_utils import IS_MACOS, skipIfRocm
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)


aten = torch.ops.aten
prims = torch.ops.prims


class TestCase(InductorTestCase):
    # [CLASS DOC] Base test case that patches the Inductor config for the whole class
    # (debug on, cpp min_chunk_size 1, no pointwise autotune, no implicit fallbacks,
    # freezing on, and frozen parameters discarded). The patch stack is stored on the
    # class and closed at teardown.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the parent setup, then enters a config patch
        # context on an ExitStack kept in the class attribute _stack so every test in the
        # class runs frozen.
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                    "freezing": True,
                    "freezing_discard_parameters": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: closes the saved ExitStack, restoring the config, then
        # calls the parent teardown.
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        # [FUNCTION DOC] Per-test setup: resets torch._dynamo caches, then calls the
        # parent setup.
        torch._dynamo.reset()
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] Per-test teardown: calls the parent teardown, then
        # resets torch._dynamo.
        super().tearDown()
        torch._dynamo.reset()


class ConvBN(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module with a Conv2d followed by a BatchNorm2d (eps 0.001, float
    # dtype); the standard target for conv-batchnorm folding tests.
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        # [FUNCTION DOC] HELPER: Builds a conv layer with the given channels, bias flag
        # and extra keyword arguments (kernel size, stride, ...) plus a float batchnorm
        # over out_channels.
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the convolution and then the batchnorm to the input.
        return self.bn(self.conv(x))


class ConvBNHardswish(torch.nn.Module):
    # [CLASS DOC] HELPER: Module of Conv2d, BatchNorm2d and an in-place Hardswish activation,
    # used to test folding when an activation follows the batchnorm.
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        # [FUNCTION DOC] HELPER: Builds the conv, a float batchnorm (eps 0.001) and an in-place
        # Hardswish; arguments mirror ConvBN.
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, dtype=torch.float)
        self.hardswish = nn.Hardswish(inplace=True)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs conv, then batchnorm, then hardswish and returns
        # the result.
        return self.hardswish(self.bn(self.conv(x)))


class ConvFunctionalBN(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a Conv2d followed by the functional batch_norm using
    # externally supplied running statistics, weight and bias that are stored as plain
    # attributes rather than as batchnorm module state.
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        kernel_size=3,
        stride=2,
        running_mean=None,
        running_var=None,
        weight=None,
        bn_bias=None,
    ):
        # [FUNCTION DOC] HELPER: Builds the conv from channels, bias flag, kernel size and
        # stride and stores the given running mean, running variance, scale (weight) and
        # shift (bn_bias).
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride
        )
        self.running_mean = running_mean
        self.running_var = running_var
        self.weight = weight
        self.bias = bn_bias

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Convolves the input and applies functional batch norm
        # in inference mode (training flag false, momentum 0.1, eps 1e-5) with the
        # stored statistics.
        return torch.nn.functional.batch_norm(
            self.conv(x),
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.1,
            1e-5,
        )


class ConvMultiBN(torch.nn.Module):
    # [CLASS DOC] HELPER: Module in which one Conv2d output feeds two different BatchNorm2d
    # layers (eps 0.001 and 0.1) whose results are summed; the conv is shared by both.
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        # [FUNCTION DOC] HELPER: Builds one conv and two float batchnorm layers with
        # different eps.
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, dtype=torch.float)
        self.bn2 = torch.nn.BatchNorm2d(out_channels, eps=0.1, dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the conv twice on the input, through each batchnorm
        # separately, and returns the sum of both results.
        tmp = self.bn(self.conv(x))
        tmp2 = self.bn2(self.conv(x))
        return tmp + tmp2


class ConvMultiFunctionalBN(torch.nn.Module):
    # [CLASS DOC] HELPER: Functional-batchnorm variant of the shared-conv module; two functional
    # batch_norm calls use one conv, with separate running means but the same variance, weight
    # and bias.
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        kernel_size=3,
        stride=2,
        running_mean=None,
        running_var=None,
        weight=None,
        bn_bias=None,
        running_mean2=None,
    ):
        # [FUNCTION DOC] HELPER: Builds the conv and stores running mean, running variance,
        # weight, bias and a second running mean as plain attributes.
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride
        )
        self.running_mean = running_mean
        self.running_var = running_var
        self.weight = weight
        self.bias = bn_bias
        self.running_mean2 = running_mean2

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the conv twice, each followed by inference-mode
        # functional batch norm (the second one using running_mean2), and returns the sum.
        tmp = torch.nn.functional.batch_norm(
            self.conv(x),
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.1,
            1e-5,
        )
        tmp2 = torch.nn.functional.batch_norm(
            self.conv(x),
            self.running_mean2,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.1,
            1e-5,
        )
        return tmp + tmp2


class OptimizeForInferenceTemplate(TestCase):
    # [CLASS DOC] Template of freezing tests that rely on self.device. It is not run directly:
    # it is copied into CPU and GPU test classes via copy_tests at the bottom of the file and
    # then deleted from the module namespace.
    def test_mutation(self):
        # [FUNCTION DOC] Verifies a module that mutates its own parameter in forward still gives
        # the same results as eager when compiled under no_grad, across two successive calls.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 10x10 zero parameter that is incremented by 10
            # in place on every forward, and returned.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the zero-initialised 10x10 parameter.
                super().__init__()
                self.mutated_param = torch.nn.Parameter(torch.zeros([10, 10]))

            def forward(self):
                # [FUNCTION DOC] HELPER: Adds 10 to the parameter in place and returns
                # the parameter.
                self.mutated_param.add_(10)
                return self.mutated_param

        with torch.no_grad():
            mod = Mod().to(self.device)
            out_eager = mod()
            out_eager2 = mod()

            mod = Mod().to(self.device)

            @torch.compile
            def foo(mod):
                # [FUNCTION DOC] HELPER: Compiled wrapper that calls the given module with
                # no arguments.
                return mod()

            out_comp = foo(mod)
            out_comp2 = foo(mod)

            self.assertEqual(out_eager, out_comp)
            self.assertEqual(out_eager2, out_comp2)

    def test_aliased_param_return(self):
        # [FUNCTION DOC] Verifies that returning both a parameter and a view (slice) of it from
        # a frozen compiled module matches eager output under no_grad.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns a slice of its parameter and
            # the parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a 10x10 zero parameter.
                super().__init__()
                self.aliased_param = torch.nn.Parameter(torch.zeros([10, 10]))

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the parameter minus its first row, and
                # the parameter.
                return self.aliased_param[1:], self.aliased_param

        mod = Mod().to(self.device).eval()

        @torch.compile()
        def foo(mod):
        # [FUNCTION DOC] HELPER: Compiled wrapper that calls the given module with no arguments.
            return mod()

        with torch.no_grad():
            mod_eager = mod()
            self.assertEqual(foo(mod), mod_eager)

    def test_autocast(self):
        # [FUNCTION DOC] Compiles a Linear layer under autocast and no_grad with half inputs and
        # checks the output equals eager and the generated code has no Triton kernel. Raises
        # SkipTest on CPU (marked as an MLKDNN bug).
        if self.device == "cpu":
            raise unittest.SkipTest("MLKDNN Bug")

        mod = torch.nn.Linear(10, 10).to(self.device).eval()
        inp = torch.rand([10, 10]).to(self.device).to(torch.half)

        @torch.compile()
        def foo(mod, inp):
        # [FUNCTION DOC] HELPER: Compiled wrapper that calls the module on the input.
            return mod(inp)

        with torch.no_grad():
            with torch.autocast(self.device):
                out_eager = mod(inp)
                out_compiled, code = run_and_get_code(foo, mod, inp)

                FileCheck().check_not("@triton.jit").run(code[0])
                self.assertEqual(out_eager, out_compiled)

    @torch._inductor.config.patch(
        {"cpp.enable_concat_linear": True, "shape_padding": False}
    )
    def test_mm_concat(self):
        # [FUNCTION DOC] Checks that sibling matmuls on the same input are concatenated into a
        # single matmul under freezing (cpp.enable_concat_linear on, shape padding off). Covers
        # three modules: three matmuls, two matmuls, and three addmm. On CPU the expected matmul
        # call name depends on MKLDNN/ACL/MKL availability. Also reruns with weights of
        # different output widths, which must still fuse into one matmul. Compares output with
        # eager each time.
        class MM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three 10x10 parameters; forward multiplies the
            # input by each and returns the three results as a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers three random 10x10 parameters t1, t2 and t3.
                super().__init__()

                self.t1 = torch.nn.Parameter(torch.rand(10, 10))
                self.t2 = torch.nn.Parameter(torch.rand(10, 10))
                self.t3 = torch.nn.Parameter(torch.rand(10, 10))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the three matrix products of the input with t1,
                # t2, t3.
                return x @ self.t1, x @ self.t2, x @ self.t3

        class MM2(torch.nn.Module):
            # [CLASS DOC] HELPER: Like MM but with only two parameters, t1 and t2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers two random 10x10 parameters.
                super().__init__()

                self.t1 = torch.nn.Parameter(torch.rand(10, 10))
                self.t2 = torch.nn.Parameter(torch.rand(10, 10))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the two matrix products of the input with t1
                # and t2.
                return x @ self.t1, x @ self.t2

        class AddMM(MM):
            # [CLASS DOC] HELPER: Subclass of MM that adds three bias parameters and computes
            # addmm with each bias and weight pair, returning a list of three results.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Runs the parent init, then registers three random
                # bias vectors.
                super().__init__()

                self.b1 = torch.nn.Parameter(torch.rand([10]))
                self.b2 = torch.nn.Parameter(torch.rand([10]))
                self.b3 = torch.nn.Parameter(torch.rand([10]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a list of addmm results pairing b1,b2,b3
                # with t1,t2,t3.
                return [
                    aten.addmm(b, x, p)
                    for b, p in [
                        (self.b1, self.t1),
                        (self.b2, self.t2),
                        (self.b3, self.t3),
                    ]
                ]

        for mod_fn in [
            lambda: MM().to(self.device),
            lambda: MM2().to(self.device),
            lambda: AddMM().to(self.device),
        ]:
            mod = mod_fn()
            inp = torch.rand([10, 10]).to(self.device)

            @torch.compile()
            def foo(mod, inp):
            # [FUNCTION DOC] HELPER: Compiled wrapper that calls the module on the input.
                return mod(inp)

            kernel_invoke = "kernel_cpp_0" if self.device == "cpu" else "triton.jit"
            mm_invoke = "mm("
            # https://github.com/pytorch/pytorch/blob/e754611d190b323e53c5d17db0dc39a96687513c/torch/_inductor/fx_passes/mkldnn_fusion.py#L1263
            mkldnn_weight_pack_init = (
                torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()
            )
            if self.device == "cpu" and mkldnn_weight_pack_init:
                if torch.ops.mkldnn._is_mkldnn_acl_supported():
                    # for aarch64 with acl supported, use mkldnn weight prepack
                    # https://github.com/pytorch/pytorch/blob/e754611d190b323e53c5d17db0dc39a96687513c/torch/_inductor/fx_passes/mkldnn_fusion.py#L1176-L1184
                    mm_invoke = "mkldnn._linear_pointwise.default("
                elif torch._C.has_mkl:
                    mm_invoke = "mkl_linear.default("

            with torch.no_grad():
                out_eager = mod(inp)
                out, code = run_and_get_code(foo, mod, inp)
                FileCheck().check_not(kernel_invoke).check_count(
                    mm_invoke, count=1, exactly=True
                ).run(code[0])
                self.assertEqual(out_eager, out)

            mod2 = mod_fn()
            mod2.t1 = torch.nn.Parameter(torch.rand([10, 15], device=self.device))
            mod2.t2 = torch.nn.Parameter(torch.rand([10, 20], device=self.device))

            if hasattr(mod2, "b1"):
                mod2.b1 = torch.nn.Parameter(torch.rand([15], device=self.device))
                mod2.b2 = torch.nn.Parameter(torch.rand([20], device=self.device))

            # fused: weights share same dim 0 (in_features), different dim 1 is OK
            with torch.no_grad():
                out_eager = mod2(inp)
                out, code = run_and_get_code(foo, mod2, inp)
                FileCheck().check_not(kernel_invoke).check_count(
                    mm_invoke, count=1, exactly=True
                ).run(code[0])
                self.assertEqual(out_eager, out)

    def test_static_indices_cudagraph(self):
        # [FUNCTION DOC] Returns immediately unless the device is cuda. Otherwise compares eager
        # and a reduce-overhead (CUDA graphs) compiled function on two copies of a two-layer
        # Linear model, where the function mutates its input and parameters in place. Checks
        # outputs and the mutated inputs match.
        if self.device != "cuda":
            return

        mod1 = torch.nn.Sequential(
            torch.nn.Linear(2, 2).to(self.device), torch.nn.Linear(2, 2).to(self.device)
        )
        mod2 = copy.deepcopy(mod1)

        def fn(x, y, mod):
        # [FUNCTION DOC] HELPER: Adds 1 to x in place, adds 2 to the first layer bias and 3 to
        # the second layer weight in place, then returns the model output on x plus y.
            x.add_(1)
            getattr(mod, "0").bias.add_(2)
            getattr(mod, "1").weight.add_(3)
            return mod(x) + y

        x1 = torch.randn(2, 2, device=self.device)
        y1 = torch.randn(2, 2, device=self.device)
        x2 = x1.clone()
        y2 = y1.clone()

        opt_fn = torch.compile(fn, mode="reduce-overhead")

        with torch.no_grad():
            ref = fn(x1, y1, mod1)
            res = opt_fn(x2, y2, mod2)
        self.assertEqual(ref, res)
        self.assertEqual(x1, x2)
        self.assertEqual(y1, y2)

    def test_rng_op(self):
        # [FUNCTION DOC] Checks a compiled function producing random numbers is not constant
        # folded: two calls under no_grad must give different results.
        @torch.compile()
        def foo():
        # [FUNCTION DOC] HELPER: Returns a 4x4 random tensor on the test device plus one.
            return torch.rand([4, 4], device=self.device) + 1

        with torch.no_grad():
            o1 = foo()
            o2 = foo()
            self.assertNotEqual(o1, o2)

    def test_symint_not_folded(self):
        # [FUNCTION DOC] Checks dynamic-shape symbolic ints are not folded as constants: a
        # function returning cos and a zeros tensor sized from the input shape is compiled with
        # dynamic shapes and compared with eager on two inputs of different sizes.
        def fn(a):
        # [FUNCTION DOC] HELPER: Returns the cosine of the input and a zeros tensor built from
        # the first two dimensions of the input.
            return a.cos(), torch.zeros(a.shape[0], a.shape[1])

        fn_opt = torch.compile(fn, backend="inductor", dynamic=True)
        inp = torch.randn(2, 4, 6).to(self.device)
        torch._dynamo.mark_dynamic(inp, 0)
        torch._dynamo.mark_dynamic(inp, 1)

        with torch.no_grad():
            self.assertEqual(fn(inp), fn_opt(inp))
            inp2 = torch.randn(3, 5, 6).to(self.device)
            torch._dynamo.mark_dynamic(inp2, 0)
            torch._dynamo.mark_dynamic(inp2, 1)
            self.assertEqual(fn(inp2), fn_opt(inp2))

    @requires_gpu()
    def test_conv_multiple_uses(self):
        # [FUNCTION DOC] GPU-only test (requires_gpu): a model using the same conv on two
        # inputs, one followed by batchnorm, must compile under no_grad and match eager output.
        from torch import nn

        class ToyModel(nn.Module):
        # [CLASS DOC] HELPER: Model with a 1-channel conv and batchnorm whose weight is
        # randomised; forward adds conv of x to batchnorm of conv of y.
            def __init__(self, *args, **kwargs) -> None:
            # [FUNCTION DOC] HELPER: Builds the conv and batchnorm and fills the batchnorm
            # weight with normal random values.
                super().__init__(*args, **kwargs)
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.bn1 = nn.BatchNorm2d(1)
                self.bn1.weight.data.normal_()

            def forward(self, x, y):
            # [FUNCTION DOC] HELPER: Returns conv of x plus batchnorm of conv of y.
                return self.conv1(x) + self.bn1(self.conv1(y))

        model = ToyModel()
        model.eval().to(GPU_TYPE)

        a = torch.rand(64, 1, 32, 32).to(GPU_TYPE)
        b = torch.rand(64, 1, 32, 32).to(GPU_TYPE)

        output = model(a, b)

        with torch.no_grad():
            output2 = torch.compile(model)(a, b)

        self.assertEqual(output, output2)

    def test_unfolded_bn(self):
        # [FUNCTION DOC] Checks an eval-mode batchnorm compiled under no_grad (plus 10)
        # matches the result of the earlier compile done with grad enabled, i.e. batchnorm
        # decomposes properly.
        x = torch.rand([3, 32, 15, 15]).to(self.device)

        mod = torch.nn.BatchNorm2d(32, eps=0.001).eval().to(self.device)

        @torch.compile()
        def foo(mod, x):
        # [FUNCTION DOC] HELPER: Compiled wrapper returning the module output plus 10.
            return mod(x) + 10

        out_compiled_no_inference = foo(mod, x)

        # would error if not decomposed
        with torch.no_grad():
            out_compiled = foo(mod, x)

            self.assertEqual(out_compiled_no_inference, out_compiled)

    @torch._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_bn(self):
        # [FUNCTION DOC] Loops over bias on/off and float16, bfloat16, float32 (skipping
        # unsupported combinations: half on CPU, bfloat16 on GPU before SM80). Compiles a
        # conv-batchnorm module in eval mode, checks outputs against eager with loose tolerance
        # and that the binary_folding counter equals 4. On cuda also pattern-checks the
        # generated code for one frozen constant.
        for use_bias, dtype in itertools.product(
            [True, False], [torch.float16, torch.bfloat16, torch.float32]
        ):
            if self.device == "cpu" and dtype == torch.float16:
                continue

            if self.device == GPU_TYPE and dtype == torch.bfloat16 and not SM80OrLater:
                continue

            mod = (
                ConvBN(3, 32, bias=use_bias, kernel_size=3, stride=2)
                .eval()
                .to(self.device)
                .to(dtype)
            )

            x = torch.rand(3, 3, 32, 32).to(self.device).to(dtype)

            torch._dynamo.reset()
            counters.clear()

            @torch.compile()
            def foo(mod, x):
            # [FUNCTION DOC] HELPER: Compiled wrapper that calls the module on the input.
                return mod(x)

            # TODO - bias is separate kernel right now, we should only unfuse it
            # from conv if it can be fused

            with torch.no_grad():
                out_eager = mod(x)
                out_optimized_for_infernece, code = run_and_get_code(foo, mod, x)

            # we unfuse the conv bias, but it should only have one constant in the kernel
            if self.device == "cuda":
                FileCheck().check_not(".run(").check("conv").check(".run(").check_same(
                    "frozen_param"
                ).check_not("frozen_param").check_next("return").run(code[0])

            self.assertEqual(
                out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2
            )
            self.assertEqual(counters["inductor"]["binary_folding"], 4)

    @torch._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_bn_hardswish(self):
        # [FUNCTION DOC] Same as the conv-batchnorm folding test but with a trailing Hardswish
        # in the module: loops over bias and dtype, checks eager equality and a binary_folding
        # counter of 4, and on cuda checks the frozen constant pattern in the generated code.
        for use_bias, dtype in itertools.product(
            [True, False], [torch.float16, torch.bfloat16, torch.float32]
        ):
            if self.device == "cpu" and dtype == torch.float16:
                continue

            if self.device == GPU_TYPE and dtype == torch.bfloat16 and not SM80OrLater:
                continue

            mod = (
                ConvBNHardswish(3, 32, bias=use_bias, kernel_size=3, stride=2)
                .eval()
                .to(self.device)
                .to(dtype)
            )

            x = torch.rand(3, 3, 32, 32).to(self.device).to(dtype)

            torch._dynamo.reset()
            counters.clear()

            @torch.compile()
            def foo(mod, x):
            # [FUNCTION DOC] HELPER: Compiled wrapper that calls the module on the input.
                return mod(x)

            # TODO - bias is separate kernel right now, we should only unfuse it
            # from conv if it can be fused

            with torch.no_grad():
                out_eager = mod(x)
                out_optimized_for_infernece, code = run_and_get_code(foo, mod, x)

            # we unfuse the conv bias, but it should only have one constant in the kernel
            if self.device == "cuda":
                FileCheck().check_not(".run(").check("conv").check(".run(").check_same(
                    "frozen_param"
                ).check_not("frozen_param").check_next("return").run(code[0])

            self.assertEqual(
                out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2
            )
            self.assertEqual(counters["inductor"]["binary_folding"], 4)

    @torch._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_bn_with_module_sharing(self):
        # [FUNCTION DOC] Warms the batchnorm running statistics with ten forward passes,
        # switches to eval, and compiles a function that calls the same module twice (module
        # sharing); checks the compiled result equals eager.
        mod = (
            ConvBN(32, 32, bias=True, kernel_size=3, stride=2)
            .to(self.device)
            .to(torch.float32)
        )

        # Update the default parameters of BN module
        for _ in range(10):
            mod(torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32))

        mod.eval()
        x = torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32)

        def foo(mod, x):
        # [FUNCTION DOC] HELPER: Calls the module once, discards the result, then returns a
        # second call.
            mod(x)
            return mod(x)

        with torch.no_grad():
            out_eager = foo(mod, x)
            out_optimized_for_infernece, _ = run_and_get_code(
                torch.compile(foo), mod, x
            )

        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)

    @torch._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_functional_bn_with_module_sharing(self):
        # [FUNCTION DOC] Same as the module-sharing test but with the functional batchnorm
        # module whose statistics come from the input mean and variance; the compiled result of
        # two calls must match eager.
        x = torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32)
        running_mean = torch.mean(x, dim=(0, 2, 3)).to(self.device)
        running_var = torch.var(x, dim=(0, 2, 3)).to(self.device)

        mod = (
            ConvFunctionalBN(
                32,
                32,
                bias=True,
                kernel_size=3,
                stride=2,
                running_mean=running_mean,
                running_var=running_var,
                weight=torch.ones(32).to(self.device),
                bn_bias=torch.zeros(32).to(self.device),
            )
            .eval()
            .to(self.device)
            .to(torch.float32)
        )

        def foo(mod, x):
        # [FUNCTION DOC] HELPER: Calls the module once, discards the result, then returns a
        # second call.
            mod(x)
            return mod(x)

        with torch.no_grad():
            out_eager = foo(mod, x)
            out_optimized_for_infernece, _ = run_and_get_code(
                torch.compile(foo), mod, x
            )

        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)

    @torch._inductor.config.patch(layout_optimization=False)
    def test_conv_bn_with_multi_bn_share_conv(self):
        # [FUNCTION DOC] Checks a conv shared by two batchnorm layers (ConvMultiBN) is handled
        # correctly: after warming statistics and eval, the compiled output matches eager.
        mod = (
            ConvMultiBN(32, 32, bias=True, kernel_size=3, stride=2)
            .to(self.device)
            .to(torch.float32)
        )

        # Update the default parameters of BN module
        for _ in range(10):
            mod(torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32))

        mod.eval()
        x = torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32)

        def foo(mod, x):
        # [FUNCTION DOC] HELPER: Returns the module output for the input.
            return mod(x)

        with torch.no_grad():
            out_eager = foo(mod, x)
            out_optimized_for_infernece, _ = run_and_get_code(
                torch.compile(foo), mod, x
            )

        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)

    @torch._inductor.config.patch(layout_optimization=False)
    def test_conv_functional_bn_with_multi_bn_share_conv(self):
        # [FUNCTION DOC] Checks the functional-batchnorm shared-conv module
        # (ConvMultiFunctionalBN) with running statistics computed from the input; compiled
        # output must match eager.
        x = torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32)
        running_mean = torch.mean(x, dim=(0, 2, 3)).to(self.device)
        running_var = torch.var(x, dim=(0, 2, 3)).to(self.device)
        running_mean2 = torch.mean(x, dim=(0, 2, 3)).to(self.device)

        mod = (
            ConvMultiFunctionalBN(
                32,
                32,
                bias=True,
                kernel_size=3,
                stride=2,
                running_mean=running_mean,
                running_var=running_var,
                weight=torch.ones(32).to(self.device),
                bn_bias=torch.zeros(32).to(self.device),
                running_mean2=running_mean2,
            )
            .eval()
            .to(self.device)
            .to(torch.float32)
        )

        def foo(mod, x):
        # [FUNCTION DOC] HELPER: Returns the module output for the input.
            return mod(x)

        with torch.no_grad():
            out_eager = foo(mod, x)
            out_optimized_for_infernece, _ = run_and_get_code(
                torch.compile(foo), mod, x
            )
        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)

    @torch._inductor.config.patch(layout_optimization=False)
    def test_dont_change_dtype_folding(self):
        # [FUNCTION DOC] Checks conv folding does not change dtypes: a bias-free conv in half
        # (GPU) or bfloat16 (otherwise) times a float32 constant must give the same result
        # compiled and eager.
        dtype = torch.float16 if self.device == GPU_TYPE else torch.bfloat16

        mod = (
            torch.nn.Conv2d(3, 32, bias=None, kernel_size=3, stride=2)
            .eval()
            .to(self.device)
            .to(dtype)
        )
        x = torch.rand(3, 3, 32, 32).to(self.device).to(dtype)

        def foo(mod, x):
        # [FUNCTION DOC] HELPER: Returns the module output multiplied by a one-element float
        # tensor 2.0.
            return mod(x) * torch.full([1], 2.0, device=self.device)

        foo_c = torch.compile(foo)

        with torch.no_grad():
            out_eager = foo(mod, x)
            out_compiled = foo_c(mod, x)
            self.assertEqual(out_eager, out_compiled)

    def test_param_deallocated(self):
        # [FUNCTION DOC] Checks that after compiling under no_grad with freezing, the
        # original module parameter is freed: a weak reference to it is None. Raises
        # SkipTest on CPU (not implemented).
        # TODO: cpu path keeps an extra copy of graph around somewhere,
        # memory not as important for cpu
        if self.device == "cpu":
            raise unittest.SkipTest("NYI CPU")

        class Mod(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with a 10x10 zero parameter; forward returns the parameter
        # plus 10 plus the input.
            def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Registers the zero 10x10 parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros([10, 10]))

            def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns the parameter plus 10, plus the input.
                return (self.param + 10) + x

        mod = Mod().eval().to(self.device)
        inp = torch.rand([10], device=self.device)

        with torch.no_grad():
            eager = mod(inp)

        weight_ref = weakref.ref(mod.param)

        @torch.compile()
        def foo(mod, inp):
        # [FUNCTION DOC] HELPER: Compiled wrapper that calls the module on the input.
            return mod(inp)

        with torch.no_grad():
            compiled = foo(mod, inp)

        self.assertEqual(eager, compiled)
        self.assertTrue(weight_ref() is None)

    @torch._inductor.config.patch(layout_optimization=True)
    def test_conv_with_as_strided(self):
        # [FUNCTION DOC] With layout optimization on, tests a grouped 1x1 conv followed by
        # padding, two as_strided views and a contiguous clone (as_strided arguments depend on
        # input strides); for groups 1 and 2, compiled output must match eager.
        class Model(nn.Module):
        # [CLASS DOC] HELPER: Module with a 256-to-384 1x1 conv with the given group count;
        # forward pads the conv output, applies two as_strided views and clones to
        # contiguous format.
            def __init__(self, groups):
            # [FUNCTION DOC] HELPER: Builds the grouped bias-free conv.
                super().__init__()
                self.kv = torch.nn.Conv2d(
                    256,
                    384,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    bias=False,
                    groups=groups,
                )

            def forward(self, x):
            # [FUNCTION DOC] HELPER: Runs conv, constant padding of 2 on each side, two
            # as_strided calls with fixed sizes and strides, and returns the contiguous clone.
                convolution = self.kv(x)
                constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                    convolution, [2, 2, 2, 2], 0.0
                )
                # as_strided inputs are depend on input's size and stide.
                as_strided = torch.ops.aten.as_strided.default(
                    constant_pad_nd, [8, 384, 2, 20, 12], [153600, 400, 160, 1, 20]
                )
                as_strided_1 = torch.ops.aten.as_strided.default(
                    as_strided, [8, 384, 2, 2, 12, 12], [153600, 400, 160, 8, 20, 1]
                )
                clone = torch.ops.aten.clone.default(
                    as_strided_1, memory_format=torch.contiguous_format
                )
                return clone

        @torch.compile()
        def foo(mod, inp):
        # [FUNCTION DOC] HELPER: Compiled wrapper that calls the module on the input.
            return mod(inp)

        with torch.no_grad():
            x = torch.randn(8, 256, 16, 16).to(self.device)
            for groups in [1, 2]:
                mod = Model(groups).to(self.device).eval()
                mod_eager = mod(x)
                self.assertEqual(foo(mod, x), mod_eager)

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180128")
    @unittest.skipIf(IS_MACOS, "https://github.com/pytorch/pytorch/issues/106557")
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(
        TEST_WITH_SLOW_GRADCHECK,
        "Failing in slow gradcheck on cuda12.8, see https://github.com/pytorch/pytorch/pull/156731 for example",
    )
    def test_cpp_wrapper(self):
        # [FUNCTION DOC] Compiles a conv-batchnorm module with the cpp_wrapper option and checks
        # that two calls under no_grad both equal the eager output. Skipped on ROCm, macOS,
        # fbcode, and slow gradcheck builds.
        mod = ConvBN(3, 32, kernel_size=3, stride=2).eval().to(self.device)

        x = torch.rand(3, 3, 32, 32).to(self.device)

        @torch.compile(options={"cpp_wrapper": True})
        def foo(mod, x):
        # [FUNCTION DOC] HELPER: Compiled wrapper (cpp_wrapper option) that calls the module.
            return mod(x)

        out_eager = mod(x)

        with torch.no_grad():
            self.assertEqual(foo(mod, x), out_eager)
            self.assertEqual(foo(mod, x), out_eager)

    @tf32_on_and_off(0.001)
    def test_conv_layout_convert_with_view(self):
        # [FUNCTION DOC] Checks a batchnorm, conv, flatten model (layout conversion with a view
        # after conv) compiled under no_grad matches eager, with tf32 toggled on and off.
        class Model(torch.nn.Module):
        # [CLASS DOC] HELPER: Model of batchnorm over 3 channels, a 3-to-128 conv with padding 1
        # and no bias, and a flatten from dimension 1.
            def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Builds the batchnorm and conv.
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies batchnorm, conv, then flattens all but the
            # batch dimension.
                x = self.bn(x)
                x = self.conv(x)
                return torch.flatten(x, 1)

        mod = Model().to(self.device).eval()

        @torch.compile()
        def foo(mod, inp):
        # [FUNCTION DOC] HELPER: Compiled wrapper that calls the module on the input.
            return mod(inp)

        with torch.no_grad():
            x = torch.rand(2, 3, 5, 5).to(self.device)
            mod_eager = mod(x)
            self.assertEqual(foo(mod, x), mod_eager)

    @torch._inductor.config.patch(layout_optimization=True)
    def test_conv_weight_layout_convert(self):
        # [FUNCTION DOC] With layout optimization on, compiles with a custom inner compile hook
        # and asserts every aten convolution seen has a channels_last constant weight, with node
        # metadata value also channels_last. On cuda also requires exactly one convolution seen.
        class Model(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with one 3-to-128 conv (padding 1, no bias) and a static
        # method producing example inputs.
            def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Builds the conv.
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns the conv of the input.
                return self.conv(x)

            @staticmethod
            def get_example_inputs():
            # [FUNCTION DOC] HELPER: Static method returning a one-element tuple with a random
            # 2x3x5x5 tensor moved to the device of the enclosing test (self is captured by
            # closure, not a parameter).
                return (torch.rand(2, 3, 5, 5).to(self.device),)

        from torch._inductor.compile_fx import compile_fx, compile_fx_inner

        nconv = 0

        def my_inner_compile(gm, example_inputs, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Inner-compile hook: runs the real inner compile, then for each
        # aten convolution in the graph checks (via the test assertions) that the weight
        # constant and its metadata value are channels_last, and adds to the nonlocal counter
        # nconv. Returns the compiled result.
            out = compile_fx_inner(gm, example_inputs, *args, **kwargs)

            nonlocal nconv
            convs = [n for n in gm.graph.nodes if n.target == aten.convolution.default]
            nconv += len(convs)
            for conv in convs:
                weight_node = conv.args[1]
                weight_const_tensor = getattr(gm, weight_node.target)
                self.assertTrue(
                    weight_const_tensor.is_contiguous(memory_format=torch.channels_last)
                )
                self.assertTrue(
                    weight_node.meta["val"].is_contiguous(
                        memory_format=torch.channels_last
                    )
                )

            return out

        mod = torch.compile(
            Model().eval().to(self.device),
            backend=functools.partial(compile_fx, inner_compile=my_inner_compile),
        )
        inp = mod.get_example_inputs()
        with torch.no_grad():
            mod(*inp)

        # Only check the assertion for CUDA.
        # For CPU, we may get torch.ops.mkldnn._convolution_pointwise.default
        # in the joint graph rather than torch.ops.aten.convolution.default.
        # Currently we only handle aten.convolution.default in layout
        # optimization. That's why the count may be 0 here for CPU.
        if self.device == "cuda":
            self.assertTrue(nconv == 1)

    def test_unequal_bias_horizontal_addmm_fusion(self):
        # [FUNCTION DOC] Checks horizontal fusion of three linear layers with different bias
        # shapes (vector and one-row matrix) on the same input: compiled results equal eager.
        device = self.device

        class Model(torch.nn.Module):
        # [CLASS DOC] HELPER: Model holding three constant 3x3 weights and three biases (zeros,
        # a 1x3 row of -1, and 1,2,3) as plain tensors; applies three linear layers.
            def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the constant weight and bias tensors on the
            # test device.
                super().__init__()
                self.w1 = torch.tensor(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=device
                )
                self.b1 = torch.zeros(3, device=device)
                self.w2 = torch.tensor(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device
                )
                self.b2 = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
                self.w3 = torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
                )
                self.b3 = torch.tensor([1.0, 2.0, 3.0], device=device)

            def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns a tuple of three linear outputs for the input.
                out1 = torch.nn.functional.linear(x, self.w1, self.b1)
                out2 = torch.nn.functional.linear(x, self.w2, self.b2)
                out3 = torch.nn.functional.linear(x, self.w3, self.b3)
                return (out1, out2, out3)

        func = Model().to(device).eval()
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)

        with torch.no_grad():
            out_eager = func(x.clone())

            func1 = torch.compile(func)
            out_compiled = func1(x.clone())
            self.assertEqual(out_eager, out_compiled)

    @torch._inductor.config.patch(
        layout_optimization=True, force_layout_optimization=True
    )
    def test_as_strided_input_layout_with_symbolic_strides(self):
        # [FUNCTION DOC] With layout and forced layout optimization on and dynamic shapes,
        # checks that in the optimized graph the stride-order enforcement node comes before the
        # as_strided node and any node used as a stride argument precedes the enforcement node.
        # Also checks results.
        from torch._inductor.compile_fx import compile_fx, compile_fx_inner

        class Model(torch.nn.Module):
        # [CLASS DOC] HELPER: Model that transposes dimensions 1 and 2 and rebuilds a view with
        # as_strided using the transposed size and stride.
            def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns the as_strided view of the transposed input.
                y = x.transpose(1, 2)
                return torch.as_strided(y, y.size(), y.stride())

        graph_modules = []

        def my_inner_compile(gm, example_inputs, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Inner-compile hook that records the graph module in a list and
        # then delegates to the real inner compile, returning its result.
            graph_modules.append(gm)
            return compile_fx_inner(gm, example_inputs, *args, **kwargs)

        mod = Model().eval().to(self.device)
        inp = torch.randn(2, 3, 4, device=self.device)
        torch._dynamo.mark_dynamic(inp, 1)
        torch._dynamo.mark_dynamic(inp, 2)

        with torch.no_grad():
            expected = mod(inp)
            opt_mod = torch.compile(
                mod,
                backend=functools.partial(compile_fx, inner_compile=my_inner_compile),
                dynamic=True,
            )
            actual = opt_mod(inp)
            self.assertEqual(expected, actual)

        self.assertTrue(graph_modules)
        gm = graph_modules[0]
        nodes = list(gm.graph.nodes)
        force_stride_node = next(
            n for n in nodes if n.target == prims.inductor_force_stride_order.default
        )
        as_strided_node = next(n for n in nodes if n.target == aten.as_strided.default)

        self.assertLess(nodes.index(force_stride_node), nodes.index(as_strided_node))
        for stride_arg in force_stride_node.args[1]:
            if isinstance(stride_arg, torch.fx.Node):
                self.assertLess(nodes.index(stride_arg), nodes.index(force_stride_node))

    @tf32_on_and_off(0.001)
    @torch._inductor.config.patch(layout_optimization=True)
    def test_redundant_clone_for_layout_convert(self):
        # [FUNCTION DOC] Counts, via an overridden lowering of the force-stride-order primitive,
        # how many calls get an input already of the requested stride versus a different stride.
        # On GPU with a conv plus an unrelated add it expects one of each; outputs must be
        # contiguous and equal eager. CPU returns early after the output equality checks.
        class Model(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with a 3-to-128 conv; forward returns the conv output and
        # the input plus one; has a static method producing example inputs.
            def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Builds the conv.
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns a tuple of the conv output and the input plus 1.
                y = x + 1
                return self.conv(x), y

            @staticmethod
            def get_example_inputs():
            # [FUNCTION DOC] HELPER: Static method returning a one-element tuple with a random
            # 2x3x5x5 tensor moved to the device of the enclosing test (self is captured by
            # closure, not a parameter).
                return (torch.rand(2, 3, 5, 5).to(self.device),)

        mod = Model().eval().to(self.device)
        inp = mod.get_example_inputs()
        with torch.no_grad():
            expected_outputs = mod(*inp)

        num_same_stride = 0
        num_diff_stride = 0

        def debug_inductor_force_stride_order(orig_fn, input_tensor, stride):
        # [FUNCTION DOC] HELPER: Replacement lowering: realizes the input, increments the
        # same-stride or different-stride nonlocal counter depending on whether its stride
        # equals the requested stride, then calls the original lowering and returns its result.
            nonlocal num_same_stride, num_diff_stride
            input_tensor.realize()
            if tuple(input_tensor.get_stride()) == tuple(stride):
                num_same_stride += 1
            else:
                num_diff_stride += 1
            return orig_fn(input_tensor, stride)

        with override_lowering(
            prims.inductor_force_stride_order.default, debug_inductor_force_stride_order
        ):
            opt_mod = torch.compile(mod)
            with torch.no_grad():
                actual_outputs = opt_mod(*inp)

        self.assertEqual(len(actual_outputs), len(expected_outputs))
        self.assertEqual(2, len(actual_outputs))
        for actual, expected in zip(actual_outputs, expected_outputs):
            self.assertEqual(expected, actual)

        if self.device == "cpu":
            # CPU use different convolution implementation, skip the checks below
            return

        self.assertTrue(
            actual_outputs[0].is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(
            actual_outputs[1].is_contiguous(memory_format=torch.contiguous_format)
        )

        # we don't change the stride of y returned by forward. So there will
        # be no extra copy
        self.assertTrue(
            num_same_stride == 1,
            lambda msg: f"{msg}\nnum_same_stride is {num_same_stride}",
        )
        # we changed the stride of self.conv(x) returned by forward. So there
        # may be an extra copy
        self.assertTrue(
            num_diff_stride == 1,
            lambda msg: f"{msg}\nnum_diff_stride is {num_diff_stride}",
        )


if TEST_WITH_ROCM:
    torch._inductor.config.force_layout_optimization = 1
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"

if HAS_CPU and not torch.backends.mps.is_available():

    class FreezingCpuTests(TestCase):
        # [CLASS DOC] CPU instantiation of the template tests: uses check_model and device cpu,
        # with the CPU autocast context; created only when a CPU is available and MPS is not.
        common = check_model
        device = "cpu"
        autocast = torch.cpu.amp.autocast

    copy_tests(OptimizeForInferenceTemplate, FreezingCpuTests, "cpu")

if HAS_GPU:

    class FreezingGpuTests(TestCase):
        # [CLASS DOC] GPU instantiation of the template tests: uses check_model_gpu and the GPU
        # device type; created only when a GPU is available.
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(OptimizeForInferenceTemplate, FreezingGpuTests, GPU_TYPE)


del OptimizeForInferenceTemplate


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
