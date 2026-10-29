# Owner(s): ["oncall: cpu inductor"]
# [MODULE DOC] Regression and repro tests for the Inductor CPU backend (C++/OpenMP
# codegen). Defines a large CPUReproTests case (compile vs eager checks for vectorization,
# conv/linear packing, layout, dtype, reductions, scheduling, etc.), small module helpers and test
# utilities. Importing this module sets ROCm layout env settings when running on ROCm and imports
# the sibling test_torchinductor module (exiting quietly when that module raises SkipTest as a
# script). It is run through the standard PyTorch test runner entry point at the bottom.
import contextlib
import copy
import ctypes
import functools
import itertools
import math
import os
import platform
import sys
import unittest
from collections.abc import Callable
from unittest.mock import patch

import torch
from torch import nn
from torch._C import FileCheck
from torch._dynamo.testing import CompileCounterWithBackend, rand_strided
from torch._dynamo.utils import same
from torch._inductor import config, cpu_vec_isa, metrics, test_operators
from torch._inductor.codegen.cpp import CppOverrides, CppVecOverrides
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.exc import InductorError
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import fresh_cache, timed
from torch._prims_common import is_float_dtype
from torch.autograd.functional import vjp
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    get_gcc_major_version,
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_CI,
    IS_CPU_EXT_SVE_SUPPORTED,
    IS_FBCODE,
    IS_MACOS,
    MI200_ARCH,
    parametrize,
    requires_mkl,
    skipIfNoLapack,
    skipIfRocm,
    skipIfRocmArch,
    slowTest,
    TEST_CUDA,
    TEST_WITH_ROCM,
    xfailIf,
    xfailIfS390X,
)
from torch.utils._python_dispatch import TorchDispatchMode


if TEST_WITH_ROCM:
    config.force_layout_optimization = 1
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


vec_dtypes = test_torchinductor.vec_dtypes
_lowp_fp_dtypes = (
    torch.bfloat16,
    torch.float16,
)
run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
TestCase = test_torchinductor.TestCase
aten = torch.ops.aten
check_model = test_torchinductor.check_model
skip_if_cpp_wrapper = test_torchinductor.skip_if_cpp_wrapper

requires_vectorization = unittest.skipUnless(
    cpu_vec_isa.valid_vec_isa_list() and os.getenv("ATEN_CPU_CAPABILITY") != "default",
    "Does not support vectorization",
)


def _can_check_vec_metrics():
    # [FUNCTION DOC] HELPER: Return true when vectorized-kernel metrics can be asserted: a valid
    # vector ISA exists, the ATen CPU capability is not forced to default, and the configured cpp
    # simdlen is not 1.
    return (
        cpu_vec_isa.valid_vec_isa_list()
        and os.getenv("ATEN_CPU_CAPABILITY") != "default"
        and config.cpp.simdlen != 1
    )


def check_metrics_vec_kernel_count(num_expected_vec_kernels):
    # [FUNCTION DOC] HELPER: Assert the count of generated C++ vectorized kernels recorded in
    # metrics equals the expected number. Does nothing when vector metrics cannot be checked;
    # raises AssertionError with both counts on mismatch.
    if _can_check_vec_metrics():
        if metrics.generated_cpp_vec_kernel_count != num_expected_vec_kernels:
            raise AssertionError(
                f"Expected {num_expected_vec_kernels} vec kernels, got {metrics.generated_cpp_vec_kernel_count}"
            )


def simd_lengths_to_test():
    # [FUNCTION DOC] HELPER: Return a short list of SIMD lengths to parametrize over: None
    # (default), 1 (scalar) and, if any vector ISA is valid, the bit width of the first valid ISA.
    """Returns a minimal list of simd lengths to cover common cases"""
    simdlens = [None, 1]
    valid_isa_list = cpu_vec_isa.valid_vec_isa_list()
    if valid_isa_list:
        simdlens.append(valid_isa_list[0].bit_width())
    return simdlens


@contextlib.contextmanager
def set_num_threads(num_threads):
    # [FUNCTION DOC] HELPER: Context manager that sets the torch intra-op thread count for the
    # body and restores the previous count afterwards. Note the restore is not in a finally block,
    # so an exception in the body leaves the changed thread count in place.
    orig_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    yield
    torch.set_num_threads(orig_num_threads)


class LstmModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Thin torch Module wrapping an LSTM layer; the constructor forwards input
    # size, hidden size, layer count, bias, bidirectional and batch_first settings to the LSTM.
    # Used as a test model.
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        bidirectional=False,
        batch_first=False,
    ):
        # [FUNCTION DOC] HELPER: Build the wrapped LSTM from the given sizes and flags; stores it
        # as the lstm attribute.
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(self, x, h=None):
        # [FUNCTION DOC] HELPER: Run the LSTM on input x with optional initial state h and return
        # the pair of output sequence and final state.
        x, h = self.lstm(x, h)
        return x, h


@instantiate_parametrized_tests
class CPUReproTests(TestCase):
    # [CLASS DOC] Test case collecting CPU-specific Inductor regression repros. Each test compiles
    # a small function or module and compares against eager (mostly via the common helper, an
    # alias of check_model) or inspects generated C++ code and metrics. Parametrized tests are
    # expanded by the decorator.
    common = check_model

    @skipIfNoLapack
    def test_torch_linalg_qr_tuple_slice(self):
        # [FUNCTION DOC] Compile a function returning a slice of the qr decomposition tuple and
        # check the compiled result is a one-element tuple matching eager. Skipped when LAPACK is
        # unavailable.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the first element of the qr decomposition tuple.
            return torch.linalg.qr(x)[:1]

        x = torch.randn(4, 4)
        compiled = torch.compile(fn, backend="inductor")

        expected = fn(x)
        actual = compiled(x)

        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 1)
        torch.testing.assert_close(actual[0], expected[0])

    @parametrize("p", ("fro", "nuc"))
    @skipIfNoLapack
    def test_linalg_cond_string_dynamic_shapes(self, p):
        # [FUNCTION DOC] Compile a function computing the condition number with string norm p (fro
        # or nuc) plus a count of positive values, with dynamic shapes and full graph; compare to
        # eager. Needs LAPACK.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compute the condition number with the enclosing norm string
            # and count how many are positive; returns both.
            condition_number = torch.linalg.cond(x, p=p)
            observer = (
                condition_number > torch.tensor(0.0, dtype=condition_number.dtype)
            ).count_nonzero()
            return condition_number, observer

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        expected = fn(x)
        compiled = torch.compile(fn, backend="inductor", dynamic=True, fullgraph=True)
        actual = compiled(x)
        self.assertEqual(actual, expected)

    def _check_conv_stride_constraints(self, formats):
        # [FUNCTION DOC] HELPER: For each memory format in formats, compile a conv2d graph via
        # make_fx and compile_fx_inner, run it under a dispatch mode that checks the convolution
        # input and weight are contiguous in the expected format (channels_last when layout
        # optimization is forced or mkldnn is usable, else the given format), and assert a
        # convolution was actually seen.
        for fmt in formats:
            # TorchDispatch doesn't work in our cuda invocation for some reason
            m = torch.nn.Conv2d(5, 6, [3, 3])

            def fn(inp, weight):
                # [FUNCTION DOC] HELPER: Traced function performing a conv2d with the parameters
                # of the enclosing module m and returning the result in a one-tuple.
                return (
                    F.conv2d(
                        inp, weight, None, m.stride, m.padding, m.dilation, m.groups
                    ),
                )

            inp = torch.randn([2, 5, 16, 16])
            inps = [inp, m.weight.to(memory_format=fmt)]
            fn_fx = make_fx(fn)(*inps)
            fn_compiled = compile_fx_inner(fn_fx, inps)
            test_self = self
            conv_seen = False

            class RecordFunctions(TorchDispatchMode):
                # [CLASS DOC] HELPER: Dispatch mode that, for aten convolution calls, asserts
                # input and weight strides match the expected memory format and flags that a
                # convolution was seen; always forwards the call.
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    # [FUNCTION DOC] Intercept a dispatch call; on convolution default verify
                    # memory format of args 0 and 1 against the enclosing expected format, set the
                    # conv_seen flag of the enclosing scope, then run the op and return its
                    # result.
                    kwargs = kwargs if kwargs else {}
                    if func == torch.ops.aten.convolution.default:
                        expected_fmt = fmt
                        if config.force_layout_optimization or (
                            torch.backends.mkldnn.enabled
                            and torch.backends.mkldnn.is_available()
                        ):
                            expected_fmt = torch.channels_last
                        test_self.assertTrue(
                            args[0].is_contiguous(memory_format=expected_fmt),
                            lambda msg: f"{msg}\ninput stride {args[0].stride()} is not {expected_fmt}",
                        )
                        test_self.assertTrue(
                            args[1].is_contiguous(memory_format=expected_fmt),
                            lambda msg: f"{msg}\nweight stride {args[1].stride()} is not {expected_fmt}",
                        )
                        nonlocal conv_seen
                        conv_seen = True

                    return func(*args, **kwargs)

            with RecordFunctions():
                fn_compiled(inps)

            self.assertTrue(conv_seen)

    @torch._inductor.config.patch({"layout_optimization": True})
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_stride_constraints(self):
        # [FUNCTION DOC] Run the conv stride check for contiguous and channels_last weights with
        # layout optimization on and CUDA reported unavailable.
        self._check_conv_stride_constraints(
            [torch.contiguous_format, torch.channels_last]
        )

    @torch._inductor.config.patch(
        {"layout_optimization": True, "force_layout_optimization": True}
    )
    @patch("torch.backends.mkldnn.is_available", lambda: False)
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_stride_constraints_forced_without_mkldnn(self):
        # [FUNCTION DOC] Run the conv stride check for contiguous format with forced layout
        # optimization while mkldnn and CUDA are reported unavailable.
        self._check_conv_stride_constraints([torch.contiguous_format])

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_bias_flattened_view(self):
        # [FUNCTION DOC] Check a conv2d whose bias is a flattened view of the input compiles (full
        # graph) and matches eager on a 1x1x1x1 tensor, with CUDA reported unavailable.
        def fn(x):
            # [FUNCTION DOC] HELPER: Flatten the boolean-converted input as float and use it as
            # the bias of a 1x1 convolution of x with itself.
            bias = x.to(torch.bool).float().flatten()
            return F.conv2d(x, x, bias, stride=1, padding=0)

        x = torch.ones(1, 1, 1, 1)
        expected = fn(x)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)

        self.assertEqual(compiled(x), expected)

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_bn_mixed_dtype(self):
        # [FUNCTION DOC] Check a bfloat16 conv followed by a float32 batch norm (eval mode, no
        # grad) compiles and matches eager via the common helper.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a bfloat16 bias-free conv and a default-dtype batch
            # norm applied in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the conv and batch norm submodules.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3,
                    16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    dtype=torch.bfloat16,
                )
                self.bn = torch.nn.BatchNorm2d(
                    16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply conv then batch norm and return the result.
                x = self.conv(x)
                x = self.bn(x)
                return x

        v = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)
        mod = Model().eval()
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    @skipIfNoLapack
    @skip_if_cpp_wrapper("conj issue, to be fixed")
    def test_complex_cholesky_mh_view_fallback(self):
        # [FUNCTION DOC] Compile a function building a Hermitian positive matrix from a complex
        # input and taking an upper Cholesky factor; compare loss and input gradient to eager
        # after backward. Skipped without LAPACK or under the cpp wrapper.
        torch.manual_seed(0)

        n = 8

        def fn(inp: torch.Tensor):
            # [FUNCTION DOC] HELPER: Build a hermitian matrix as identity plus half of the input
            # times its conjugate transpose, take its upper Cholesky factor and return the sum of
            # absolute values.
            I0 = torch.eye(n, dtype=inp.dtype, device=inp.device)
            I = I0.unsqueeze(0).expand(inp.shape[0], n, n).contiguous()
            hermitian = I + 0.5 * (inp @ inp.mH)
            chol = torch.linalg.cholesky(hermitian, upper=True)
            return chol.abs().sum()

        base = torch.randn(4, n, n, dtype=torch.complex64)

        def run(compiled_fn):
            # [FUNCTION DOC] HELPER: Run the given function on a fresh grad-requiring copy of the
            # base input, call backward on the result and return detached loss and gradient.
            inp = base.clone().detach().requires_grad_(True)
            loss = compiled_fn(inp)
            loss.backward()
            return loss.detach(), inp.grad.detach()

        expected_loss, expected_grad = run(fn)

        compiled = torch.compile(fn, backend="inductor")
        actual_loss, actual_grad = run(compiled)

        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_nn_fold(self):
        # [FUNCTION DOC] Regression test for fold (issue 147848): compare the Fold module against
        # eager for two size configs, each with one thread and default threads (looser tolerance
        # when multithreaded).
        # Fix https://github.com/pytorch/pytorch/issues/147848

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a Fold layer with the given output size, kernel
            # size and stride.
            def __init__(self, output_size, kernel_size, stride) -> None:
                # [FUNCTION DOC] HELPER: Create the Fold submodule.
                super().__init__()
                self.fold = torch.nn.Fold(
                    output_size=output_size, kernel_size=kernel_size, stride=stride
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply fold to the input.
                x = self.fold(x)
                return x

        output_sizes = [(64, 64), (64, 64)]
        kernel_sizes = [(32, 32), (32, 32)]
        strides = [(1, 1), (2, 2)]
        input_sizes = [(1, 32 * 32, 1089), (1, 64 * 64, 289)]

        for idx in range(len(output_sizes)):
            output_size = output_sizes[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            input_size = input_sizes[idx]

            for num_threads in [1, None]:
                torch._dynamo.reset()
                metrics.reset()
                v = torch.randn(*input_size)
                mod = Model(output_size, kernel_size, stride).eval()
                with (
                    contextlib.nullcontext()
                    if (num_threads != 1)
                    else set_num_threads(1)
                ):
                    with torch.no_grad():
                        # Fold accumulates overlapping patches, so parallel CPU
                        # codegen can differ slightly from eager accumulation order.
                        tol_kwargs = (
                            {"atol": 1e-4, "rtol": 1e-4} if num_threads is None else {}
                        )
                        self.common(
                            mod,
                            (v,),
                            **tol_kwargs,
                        )

    @requires_vectorization
    def test_nn_fold_permuted_input(self):
        # [FUNCTION DOC] Regression test (issue 191837) for fold applied to a permuted and
        # reshaped input, compiled with two threads. Checks generated C++ contains the transpose
        # and atomic vector add helpers and that the result matches eager. Requires vectorization.
        # Fix https://github.com/pytorch/pytorch/issues/191837
        def fn(x):
            # [FUNCTION DOC] HELPER: Permute and reshape the input into fold column layout, then
            # fold to a 4x4 output.
            x = x.permute(0, 1, 4, 3, 2)
            x = x.reshape(1, 192 * 3 * 3, 2 * 2)
            return F.fold(x, output_size=(4, 4), kernel_size=3, padding=1, stride=2)

        v = torch.randn(1, 6, 4, 9, 32)
        with set_num_threads(2):
            torch._dynamo.reset()
            opt_fn = torch.compile(fn, backend="inductor")
            actual, code = run_and_get_cpp_code(opt_fn, v)
        FileCheck().check("transpose_mxn").run(code)
        FileCheck().check("atomic_add_vec").run(code)
        self.assertEqual(fn(v), actual, atol=1e-4, rtol=1e-4)

    def test_conv_max_pool_where_backward_mutation_dep(self):
        # [FUNCTION DOC] Repro for issue 185509: conv, then max pool, then where on a mask;
        # compares output and all input gradients between eager and compiled.
        # Repro for https://github.com/pytorch/pytorch/issues/185509
        def fn(x, weight, bias, mask):
            # [FUNCTION DOC] HELPER: Convolution with padding, stride-1 max pool, then select conv
            # or pool by mask.
            conv = F.conv2d(x, weight, bias, padding=1)
            pool = F.max_pool2d(conv, kernel_size=3, stride=1, padding=1)
            return torch.where(mask, conv, pool)

        def clone_inputs(inputs):
            # [FUNCTION DOC] HELPER: Return detached copies of the inputs that require grad.
            return [x.detach().clone().requires_grad_(True) for x in inputs]

        def run(f, inputs, mask):
            # [FUNCTION DOC] HELPER: Run function on inputs and mask, backpropagate the sum,
            # return detached output and list of detached input gradients.
            out = f(*inputs, mask)
            out.sum().backward()
            return out.detach(), [x.grad.detach().clone() for x in inputs]

        torch.manual_seed(0)
        inputs = [
            torch.randn(2, 3, 6, 7, requires_grad=True),
            torch.randn(4, 3, 3, 3, requires_grad=True),
            torch.zeros(4, requires_grad=True),
        ]
        mask = (torch.rand(2, 4, 6, 7) > 0.5).detach()

        expected_out, expected_grads = run(fn, clone_inputs(inputs), mask)

        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, backend="inductor")
        actual_out, actual_grads = run(compiled_fn, clone_inputs(inputs), mask)

        torch.testing.assert_close(actual_out, expected_out)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_packed(self):
        # [FUNCTION DOC] Check packed mkldnn conv2d across training and eval mode and scalar vs
        # tuple padding. Skipped when mkldnn is unavailable; CUDA reported unavailable.
        options = itertools.product([[3, 56, 56]], [True, False], [0, (0,)])
        for x_shape, mode_train, padding in options:
            mod = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, 3, padding=padding)
            ).train(mode=mode_train)
            v = torch.randn(x_shape, dtype=torch.float32)

            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_autocast(self):
        # [FUNCTION DOC] Check conv2d under CPU autocast matches eager on a float32 input.
        v = torch.randn(1, 3, 28, 18, dtype=torch.float32)
        mod = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 3)).eval()
        with torch.no_grad(), torch.cpu.amp.autocast():
            self.common(
                mod,
                (v,),
            )

    def test_conv1d_strided_weight_torch_compile(self):
        # [FUNCTION DOC] Compile a conv1d with a transposed (strided) weight with dynamic shapes
        # and full graph; compare output and both input and weight gradients with eager.
        def fn(x, w):
            # [FUNCTION DOC] HELPER: Transpose the weight, run conv1d and return a clone of the
            # result.
            wt = w.transpose(2, 1)
            y = F.conv1d(x, wt)
            return y.clone()

        x_eager = torch.randn(2, 3, 5, requires_grad=True)
        w_eager = torch.randn(4, 2, 3, requires_grad=True)

        out_eager = fn(x_eager, w_eager)
        grad = torch.randn_like(out_eager)
        out_eager_val = out_eager.detach()
        out_eager.backward(grad)
        grad_x_eager = x_eager.grad.detach().clone()
        grad_w_eager = w_eager.grad.detach().clone()

        x_comp = x_eager.detach().requires_grad_(True)
        w_comp = w_eager.detach().requires_grad_(True)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        out_comp = compiled(x_comp, w_comp)
        out_comp_val = out_comp.detach()
        out_comp.backward(grad)

        torch.testing.assert_close(out_comp_val, out_eager_val)
        torch.testing.assert_close(x_comp.grad, grad_x_eager)
        torch.testing.assert_close(w_comp.grad, grad_w_eager)

    def test_compile_dynamic_grad_conv_transpose2d(self):
        # [FUNCTION DOC] Regression test for issue 181175: gradient of a batch norm, transposed
        # conv, max pool and softmax pipeline compiled with dynamic shapes; compares to eager.
        # Regression test for https://github.com/pytorch/pytorch/issues/181175
        batch_norm = torch.nn.BatchNorm2d(5).eval()
        conv_transpose = torch.nn.ConvTranspose2d(5, 2, 3)
        max_pool = torch.nn.MaxPool2d(2)

        torch.manual_seed(0)
        x = torch.randn(3, 5, 14, 15)

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply batch norm, transposed conv, max pool, then mean of the
            # softmax over dim 0.
            y = batch_norm(x)
            y = conv_transpose(y)
            y = max_pool(y)
            return torch.softmax(y, dim=0).mean()

        expected = torch.func.grad(fn)(x)

        torch._dynamo.reset()
        compiled = torch.compile(torch.func.grad(fn), backend="inductor", dynamic=True)
        result = compiled(x)
        torch.testing.assert_close(result, expected)

    @config.patch(freezing=True)
    @requires_mkl
    @patch("torch.cuda.is_available", lambda: False)
    def test_mkl_linear(self):
        # [FUNCTION DOC] Check linear layers executed through MKL with freezing, with and without
        # bias, float32 only.
        dtypes = [torch.float32]
        options = itertools.product([[2, 3, 10]], [2], [True, False], dtypes)
        for input_shape, out_dim, bias, dtype in options:
            mod = torch.nn.Sequential(
                torch.nn.Linear(input_shape[-1], out_dim, bias=bias)
            ).eval()

            v = torch.randn(input_shape)
            with torch.no_grad():
                self.common(
                    mod.to(dtype),
                    (v.to(dtype),),
                )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_unsupported_conv_transpose(self):
        # [FUNCTION DOC] Verify an unsupported transposed conv (output padding not smaller than
        # stride/dilation) raises a RuntimeError on compiled call; the expected message differs
        # under the cpp wrapper.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a transposed conv with invalid output padding
            # followed by tanh.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the transposed conv submodule.
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    3, 6, 3, stride=1, padding=1, output_padding=1
                )

            def forward(self, input_tensor):
                # [FUNCTION DOC] HELPER: Apply transposed conv and tanh.
                x = self.conv_transpose(input_tensor)
                output = torch.tanh(x)
                return output

        input = torch.randn(1, 3, 28, 28)
        m = Model().eval()

        with torch.no_grad():
            compiled_m = torch.compile(m)
            # The cpp_wrapper C-shim can't utilize the Python error API, so error
            # messages are printed to stderr directly, and the intercepted RuntimeError
            # is significantly less verbose.
            msg = (
                r"aoti_torch_cpu_convolution\(.*\) API call failed"
                if config.cpp_wrapper
                else "output padding must be smaller than either stride or dilation"
            )
            with self.assertRaisesRegex(RuntimeError, msg):
                compiled_m(input)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_used_from_multiple_places(self):
        # [FUNCTION DOC] Check a conv module used twice in one forward (conv, relu, conv) matches
        # eager under no grad.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reusing one conv layer twice with a relu in between.
            def __init__(self, conv_in_channel, conv_out_channel) -> None:
                # [FUNCTION DOC] HELPER: Create the shared conv layer.
                super().__init__()
                self.conv = torch.nn.Conv2d(conv_in_channel, conv_out_channel, (3, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the conv, relu, then the same conv again.
                res = self.conv(x)
                res = F.relu(res)
                res = self.conv(res)
                return res

        with torch.no_grad():
            mod = M(3, 3).eval()
            x = torch.randn(1, 3, 224, 224)
            self.common(
                mod,
                (x,),
            )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_linear_used_from_multiple_places(self):
        # [FUNCTION DOC] Check a linear layer reused twice matches eager for each supported
        # low-precision dtype (bfloat16 and float16 depending on mkldnn support).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reusing one linear layer twice with a relu in between.
            def __init__(self, in_channel, out_channel) -> None:
                # [FUNCTION DOC] HELPER: Create the shared linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(in_channel, out_channel)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply linear, relu, then linear again.
                res = self.linear(x)
                res = F.relu(res)
                res = self.linear(res)
                return res

        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            with torch.no_grad():
                m = M(224, 224).to(dtype).eval()
                m_opt = torch.compile(m)
                x = torch.randn(224, 224, dtype=dtype)
                m_opt(x)
                self.assertEqual(m(x), m_opt(x))

    @config.patch(implicit_fallbacks=True)
    def test_multihead_attention_cpu(self):
        # [FUNCTION DOC] Check the native multi head attention op with implicit fallbacks enabled
        # via the common helper.
        def fn(
            q,
            k,
            v,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask,
            need_weights,
        ):
            # [FUNCTION DOC] HELPER: Forward all arguments to the native multi head attention op
            # and return its result.
            return torch._native_multi_head_attention(
                q,
                k,
                v,
                embed_dim,
                num_heads,
                qkv_weight,
                qkv_bias,
                proj_weight,
                proj_bias,
                mask,
                need_weights,
            )

        B = 1
        T = 3
        embed_dim = 6
        num_heads = 2
        q = torch.randn([B, T, embed_dim])
        k = torch.randn([B, T, embed_dim])
        v = torch.randn([B, T, embed_dim])
        qkv_weight = torch.randn([3 * embed_dim, embed_dim])
        qkv_bias = torch.randn([3 * embed_dim])
        proj_weight = torch.randn([3 * embed_dim, embed_dim])
        proj_bias = torch.randn([3 * embed_dim])
        mask = None
        need_weights = False

        inps = [
            q,
            k,
            v,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask,
            need_weights,
        ]
        self.common(fn, inps)

    @config.patch(freezing=True)
    def test_module_buffer_mutation(self):
        # [FUNCTION DOC] Check a module reading a row of a buffer in a loop and concatenating
        # results works with freezing enabled.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer; forward concatenates input-plus-buffer-row
            # for three rows (input and two clones).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the module with a random buffer.
                super().__init__()
                self.foo = torch.nn.Buffer(torch.rand((3, 10)))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Add each of three input copies to successive buffer rows
                # and concatenate on dim 1.
                lx = [x, x.clone(), x.clone()]
                y = []
                for i in range(3):
                    y.append(lx[i] + self.foo[i])
                return torch.cat(y, 1)

        with torch.no_grad():
            example_inputs = (torch.rand(1, 10),)
            self.common(Model(), example_inputs)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_linear_packed(self):
        # [FUNCTION DOC] Check packed mkldnn linear over several input ranks, zero-size dims, bias
        # on/off and low precision dtypes against eager.
        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        options = itertools.product(
            [[2, 3, 10], [2, 10], [10], [2, 0]], [3, 0], [True, False], dtypes
        )
        for input_shape, out_dim, bias, dtype in options:
            mod = torch.nn.Sequential(
                torch.nn.Linear(input_shape[-1], out_dim, bias=bias)
            ).eval()

            v = torch.randn(input_shape)
            with torch.no_grad():
                self.common(
                    mod.to(dtype),
                    (v.to(dtype),),
                )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_transpose2d_packed_cpu(self):
        # [FUNCTION DOC] Check packed transposed conv2d for batched and unbatched input and
        # scalar/tuple padding.
        options = itertools.product([[1, 3, 28, 28], [3, 28, 28]], [0, (0,)])
        for x_shape, padding in options:
            mod = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(3, 64, 3, 3, padding=padding)
            ).eval()
            v = torch.randn(x_shape, dtype=torch.float32)
            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )

    @torch._dynamo.config.patch(
        {"dynamic_shapes": True, "assume_static_by_default": False}
    )
    def test_full_boolean_dynamic_shape(self):
        # [FUNCTION DOC] Check a full-tensor creation whose boolean fill value depends on a
        # dynamic integer argument, for argument values on both sides of the threshold; result and
        # its increment must match eager.
        def fn(n):
            # [FUNCTION DOC] HELPER: Create a 1024-element tensor filled with the comparison
            # result and return it with that tensor plus one.
            x = torch.full((1024,), n >= 1024)
            return x, x + 1

        self.common(fn, (1024,))
        self.common(fn, (1023,))

    @config.patch(freezing=True)
    @unittest.skipIf(not torch._C._has_mkldnn, "MKLDNN is not enabled")
    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_conv_in_channel_1_dynamic_shapes(self):
        # [FUNCTION DOC] Check a conv followed by relu where the input has a single channel, under
        # dynamic shapes and freezing, with autocast off and (when bf16 is supported) on.
        # Reproducer from maml_omniglot.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one conv and a relu.
            def __init__(self, in_channel, out_channel) -> None:
                # [FUNCTION DOC] HELPER: Create the conv layer.
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channel, out_channel, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply conv then relu.
                res = self.conv(x)
                res = F.relu(res)
                return res

        # test the case where the channels dim of the input is 1
        # Reproducer from the maml_omniglot model in Torchbench
        in_channel = 1
        out_channel = 3
        amp_enabled_configs = [False]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            # When amp is enabled here, the input to Conv is a FlexibleLayout.
            # While it's disabled, the input is a FixedLayout.
            amp_enabled_configs.append(True)
        for amp_enabled in amp_enabled_configs:
            mod = M(in_channel, out_channel).eval()
            v = torch.randn(5, in_channel, 15, 15)
            with torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
                self.common(
                    mod,
                    (v,),
                )

    @unittest.skipIf(not torch._C._has_mkldnn, "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    @torch._dynamo.config.patch(allow_rnn=True)
    @config.patch(freezing=True)
    def _test_lstm_packed(
        self,
        unbatched,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        empty_state,
        batch_first,
        batch_size,
        seq_len,
        change_input_sizes=False,
    ):
        # [FUNCTION DOC] HELPER: Shared body for the packed LSTM tests. For each supported dtype
        # (float, plus bf16 and fp16 when mkldnn supports them) builds batched or unbatched
        # inputs, optionally with initial state, compiles an LstmModule, and asserts: the
        # generated code uses the mkldnn rnn layer, flat weights of a deep copy are not functional
        # tensors, output equals eager, and the pattern matcher count equals layers times
        # directions plus two. If change_input_sizes is set, also reruns with a longer sequence to
        # exercise recompilation.
        from torch._dynamo.utils import counters

        dtypes = [torch.float]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            counters.clear()
            num_directions = 2 if bidirectional else 1

            seq_len_var = seq_len + 3
            if unbatched:
                v = torch.randn(seq_len, input_size)
                v_var = torch.randn(seq_len_var, input_size)
                h = torch.randn(num_layers * num_directions, hidden_size)
                c = torch.randn(num_layers * num_directions, hidden_size)
            else:
                if batch_first:
                    v = torch.randn(batch_size, seq_len, input_size)
                    v_var = torch.randn(batch_size, seq_len_var, input_size)
                else:
                    v = torch.randn(seq_len, batch_size, input_size)
                    v_var = torch.randn(seq_len_var, batch_size, input_size)
                h = torch.randn(num_layers * num_directions, batch_size, hidden_size)
                c = torch.randn(num_layers * num_directions, batch_size, hidden_size)

            mod = LstmModule(
                input_size,
                hidden_size,
                num_layers,
                bias,
                bidirectional,
                batch_first,
            ).eval()
            maybe_autocast = (
                torch.cpu.amp.autocast()
                if dtype == torch.bfloat16
                else contextlib.nullcontext()
            )

            with torch.no_grad(), maybe_autocast:
                inps = [v]
                if not empty_state:
                    inps.append((h, c))

                fn_opt = torch.compile(mod, backend="inductor")
                _, code = run_and_get_cpp_code(fn_opt, *inps)

                # Check that _flat_weights are not functional_tensor, otherwise
                # deepcopy will fail during recompilation.
                fn_opt_copy = copy.deepcopy(fn_opt)
                _flat_weights = fn_opt_copy.lstm._flat_weights
                for _flat_weight in _flat_weights:
                    self.assertFalse(torch._is_functional_tensor(_flat_weight))

                self.assertTrue("aten.mkldnn_rnn_layer" in code)
                self.assertEqual(fn_opt(*inps), mod(*inps))
                self.assertEqual(
                    counters["inductor"]["pattern_matcher_count"],
                    num_layers * num_directions
                    + 2,  # num of mkldnn_rnn_layer call + 2 view call on the concatenated hy, cy.
                )

                # Change input sizes
                if change_input_sizes:
                    inps_var = [v_var]
                    self.assertEqual(fn_opt(*inps_var), mod(*inps_var))

    @parametrize(
        "unbatched, input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, batch_size, seq_len",
        itertools.product(
            *[
                [True, False],
                [1, 7],
                [7],
                [1, 7],
                [False, True],
                [False, True],
                [False, True],
                [True, False],
                [1, 7],
                [1, 7],
            ]
        ),
    )
    @unittest.skipIf(
        IS_ARM64 and not IS_CPU_EXT_SVE_SUPPORTED,
        "flaky on AArch64 (no SVE)",
    )
    def test_lstm_packed(
        self,
        unbatched,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        empty_state,
        batch_first,
        batch_size,
        seq_len,
    ):
        # [FUNCTION DOC] Packed LSTM test over the product of batching, sizes, direction, bias,
        # state and batch_first options; delegates to the shared helper. Skipped on ARM64 without
        # SVE (flaky).
        self._test_lstm_packed(
            unbatched,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            bias,
            empty_state,
            batch_first,
            batch_size,
            seq_len,
        )

    _test_lstm_packed_change_input_sizes_cpu_params = list(
        itertools.product(
            *[
                [False],
                [2],
                [5],
                [3],
                [True],
                [True],
                [False],
                [False],
                [2],
                [3],
            ]
        )
    )

    @parametrize(
        "unbatched, input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, batch_size, seq_len",
        _test_lstm_packed_change_input_sizes_cpu_params,
    )
    @unittest.skipIf(
        IS_ARM64 and not IS_CPU_EXT_SVE_SUPPORTED,
        "flaky on AArch64 (no SVE)",
    )
    def test_lstm_packed_change_input_sizes_cpu(
        self,
        unbatched,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        empty_state,
        batch_first,
        batch_size,
        seq_len,
    ):
        # [FUNCTION DOC] Packed LSTM test with a fixed config that additionally changes the
        # sequence length after the first run; delegates to the shared helper with input size
        # changes enabled. Skipped on ARM64 without SVE.
        self._test_lstm_packed(
            unbatched,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            bias,
            empty_state,
            batch_first,
            batch_size,
            seq_len,
            change_input_sizes=True,
        )

    def test_set_source_Tensor(self):
        # [FUNCTION DOC] Check compile of a masked conv model (mask zeroes part of the weight in
        # place) emits a shallow copy of tensor data, and that the result matches eager. Then
        # checks, with the cpp wrapper, that the kernel source marker appears in the code only
        # when separate kernel building is enabled.
        class MaskedConv2d(torch.nn.Conv2d):
            # [CLASS DOC] HELPER: Conv2d subclass that holds a fixed mask buffer (ones in the top
            # half rows and left half of the middle row) and multiplies its weight data by the
            # mask in place before each forward.
            def __init__(
                self,
                *,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                padding: int = 0,
            ) -> None:
                # [FUNCTION DOC] HELPER: Build the conv, then create the mask from the weight
                # shape and register it as buffer.
                super().__init__(
                    in_channels, out_channels, kernel_size, padding=padding
                )
                mask = torch.zeros_like(self.weight)

                mask[:, :, : kernel_size // 2, :] = 1
                mask[:, :, kernel_size // 2, : kernel_size // 2] = 1
                self.register_buffer("mask", mask)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Apply the mask to the weight data under no-grad, then run
                # the parent forward.
                with torch.no_grad():
                    self.weight.data *= self.mask
                return super().forward(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model permuting a channels-last image to channels-first and
            # applying one masked 7x7 conv with 64 outputs.
            def __init__(
                self, num_channels: int, num_colors: int, H: int, W: int
            ) -> None:
                # [FUNCTION DOC] HELPER: Store the sizes and build a sequential of one masked
                # conv.
                super().__init__()
                self.num_channels = num_channels
                self.num_colors = num_colors
                self.H = H
                self.W = W
                kernel_size = 7
                padding = (kernel_size - 1) // 2
                # 1 7x7 Mask
                layers = [
                    MaskedConv2d(
                        in_channels=self.num_channels,
                        out_channels=64,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                ]
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Permute input to channel-first and run the model.
                x = x.permute(0, 3, 1, 2)
                return self.model(x)

        model = M(H=32, W=32, num_channels=4, num_colors=2)
        fn_opt = torch.compile(model, backend="inductor")
        v = (torch.rand(10, 32, 32, 4) > 0.5).to(torch.float32)
        inp = v.clone()
        result, code = run_and_get_cpp_code(fn_opt, inp)
        self.assertIn(
            "shallow_copy_data_",
            code,
        )
        expected = model(inp)
        self.assertEqual(expected, result)

        # test cpp_wrapper_build_separate
        with config.patch(cpp_wrapper=True, cpp_wrapper_build_separate=True):
            result, code = run_and_get_cpp_code(fn_opt, inp)
            self.assertIn("kernel_src", code)
            self.assertEqual(expected, result)

        with config.patch(cpp_wrapper=True, cpp_wrapper_build_separate=False):
            result, code = run_and_get_cpp_code(fn_opt, inp)
            self.assertNotIn("kernel_src", code)
            self.assertEqual(expected, result)

    @torch._dynamo.config.patch("graph_break_on_nn_param_ctor", False)
    def test_set_source_tensor_with_view_source(self):
        # [FUNCTION DOC] Check compile (full graph) of a function that makes Parameters from views
        # and results of computations, with an in-place scaling between; compare to eager under
        # no-grad.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sigmoid-flatten the input, build three Parameters from it, a
            # concatenation and a sum with zeros, with in-place multiplication of the flattened
            # tensor in between; returns the three.
            x = torch.sigmoid(x).view(x.size(0), -1)
            param1 = nn.Parameter(x)
            with torch.no_grad():
                x.mul_(1.4386868137611386)
            y = torch.cat([x, x], dim=1)
            param2 = nn.Parameter(y)
            z = torch.zeros_like(x) + x
            param3 = nn.Parameter(z)
            return param1, param2, param3

        inp = torch.randn(2, 3, 4)
        with torch.no_grad():
            expected = fn(inp)
        opt_fn = torch.compile(fn, fullgraph=True)
        with torch.no_grad():
            result = opt_fn(inp)
        self.assertEqual(expected, result)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    @torch._dynamo.config.patch(allow_rnn=True)
    def test_pack_padded_sequence_lstm(self):
        # [FUNCTION DOC] Check an LSTM fed a packed padded sequence is compiled without the mkldnn
        # lstm op (unsupported case) and still matches eager, with dynamic shapes. Raises
        # AssertionError if helper data is inconsistent with batch size or max length.
        embedding_dim = 12
        hidden_dim = 10
        batch_size = 24
        num_layers = 1
        bidirectional = True
        num_direc = 2
        max_lens = 96

        sent = torch.randn(batch_size, max_lens, embedding_dim)
        hid_0 = torch.rand(num_layers * num_direc, batch_size, hidden_dim)
        hid_1 = torch.randn(num_layers * num_direc, batch_size, hidden_dim)

        sent_lens = torch.Tensor(
            [1, 2, 3, 4, 5, 1, 3, 2, 96, 5, 3, 1, 1, 2, 1, 2, 3, 6, 1, 2, 4, 6, 2, 1]
        )

        if sent_lens.shape[0] != batch_size:
            raise AssertionError(
                f"Expected sent_lens.shape[0] == {batch_size}, got {sent_lens.shape[0]}"
            )
        if sent_lens.max().item() != max_lens:
            raise AssertionError(
                f"Expected sent_lens.max().item() == {max_lens}, got {sent_lens.max().item()}"
            )

        hidden_0 = hid_0.clone().requires_grad_(False)
        hidden_1 = hid_1.clone().requires_grad_(False)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(
            sent, sent_lens, batch_first=True, enforce_sorted=False
        )

        mod = LstmModule(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bias=True,
            bidirectional=bidirectional,
            batch_first=True,
        ).eval()

        with torch.no_grad():
            inps = [embeds, (hidden_0, hidden_1)]
            fn_opt = torch.compile(mod, backend="inductor")
            _, code = run_and_get_cpp_code(fn_opt, *inps)
            # This case is unsupported
            self.assertFalse("torch.ops.mkldnn._lstm" in code)
            self.assertEqual(fn_opt(*inps), mod(*inps))

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_transpose2d_has_output_size_input(self):
        # [FUNCTION DOC] Regression test (issue 100344) for a transposed conv called with an
        # explicit output size argument; compare to eager.
        # https://github.com/pytorch/pytorch/issues/100344.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a transposed conv whose forward passes an output
            # size of 10 by 10.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the transposed conv layer.
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply transposed conv with an explicit output size.
                return self.conv_transpose(x, output_size=(10, 10))

        mod = M().eval()
        v = torch.randn(1, 3, 10, 10, dtype=torch.float32)
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_pad_with_nan_value(self):
        # [FUNCTION DOC] Regression test (issue 100988): padding with a NaN fill value matches
        # eager.
        # https://github.com/pytorch/pytorch/issues/100988.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module padding by one on every side with NaN.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Pad the input with NaN.
                x = F.pad(x, (1, 1, 1, 1), value=float("nan"))
                return x

        mod = Model().eval()
        v = torch.randn(1, 3, 10, 10, dtype=torch.float32)
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_masked_fill_with_inf_or_nan_value(self):
        # [FUNCTION DOC] Check masked_fill with positive infinity, negative infinity and NaN
        # values matches eager.
        def fn(value, mask):
            # [FUNCTION DOC] HELPER: Return three masked_fill results for the three special
            # values.
            y1 = torch.masked_fill(value, mask, float("inf"))
            y2 = torch.masked_fill(value, mask, float("-inf"))
            y3 = torch.masked_fill(value, mask, float("nan"))
            return y1, y2, y3

        value = torch.randn((2, 17))
        mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8).to(torch.bool)
        with torch.no_grad():
            self.common(
                fn,
                (value, mask),
            )

    def test_relu_with_inf_value(self):
        # [FUNCTION DOC] Regression test (issue 117544): sinh followed by relu on large values
        # that overflow to infinity.
        # https://github.com/pytorch/pytorch/issues/117544.

        def fn(out):
            # [FUNCTION DOC] HELPER: Apply sinh then relu.
            out = torch.sinh(input=out)
            out = torch.relu(input=out)
            return out

        x = torch.Tensor([-572373.5000, 755109.1250, 330995.5625])
        with torch.no_grad():
            self.common(
                fn,
                (x,),
            )

    def test_acosh_with_negative_large_input(self):
        # [FUNCTION DOC] Regression test (issue 118267): acosh on large negative and positive
        # inputs, run in float32, bfloat16 and double with dynamo and metrics reset each time.
        # https://github.com/pytorch/pytorch/issues/118267.

        def fn(input):
            # [FUNCTION DOC] HELPER: Apply acosh.
            out = torch.acosh(input)
            return out

        x = torch.Tensor(
            [
                [
                    -8493.9854,
                    431654.1250,
                    71741.5859,
                    608234.5000,
                    -103814.7500,
                    -699397.0000,
                    -910685.8125,
                    -832737.1875,
                    875343.5000,
                ]
            ]
        ).repeat(3, 9)

        for dtype in [torch.float32, torch.bfloat16, torch.double]:
            with torch.no_grad():
                torch._dynamo.reset()
                metrics.reset()
                _x = x.to(dtype)
                self.common(
                    fn,
                    (_x,),
                )

    @requires_vectorization
    def test_cosh_near_overflow(self):
        # [FUNCTION DOC] Regression test (issue 183765): cosh near overflow does not give infinity
        # in float32 or double and matches eager. Requires vectorization.
        # https://github.com/pytorch/pytorch/issues/183765
        def fn(x):
            # [FUNCTION DOC] HELPER: Apply cosh.
            return torch.cosh(x)

        x = torch.tensor([88.5, 88.85, 89.0, 89.4, -88.85, -89.0]).repeat(3, 6)
        for dtype in [torch.float32, torch.double]:
            with torch.no_grad():
                torch._dynamo.reset()
                _x = x.to(dtype)
                self.assertFalse(torch.cosh(_x).isinf().any())
                self.common(fn, (_x,))

    @requires_vectorization
    def test_sinh_near_overflow(self):
        # [FUNCTION DOC] Regression test (issue 183763): sinh near overflow does not give infinity
        # in float32 or double and matches eager. Requires vectorization.
        # https://github.com/pytorch/pytorch/issues/183763
        def fn(x):
            # [FUNCTION DOC] HELPER: Apply sinh.
            return torch.sinh(x)

        x = torch.tensor([88.5, 88.85, 89.0, -89.0, -89.2, 0.0]).repeat(3, 6)
        for dtype in [torch.float32, torch.double]:
            with torch.no_grad():
                torch._dynamo.reset()
                _x = x.to(dtype)
                self.assertFalse(torch.sinh(_x).isinf().any())
                self.common(fn, (_x,))

    @requires_vectorization
    def test_acosh_near_overflow(self):
        # [FUNCTION DOC] Regression test (issue 183768): acosh near overflow does not give
        # infinity in float32 or double and matches eager. Requires vectorization.
        # https://github.com/pytorch/pytorch/issues/183768
        def fn(x):
            # [FUNCTION DOC] HELPER: Apply acosh.
            return torch.acosh(x)

        x = torch.tensor([2.0, 1e10, 5e22, 7e21, 9e25, 1.0]).repeat(3, 6)
        for dtype in [torch.float32, torch.double]:
            with torch.no_grad():
                torch._dynamo.reset()
                _x = x.to(dtype)
                self.assertFalse(torch.acosh(_x).isinf().any())
                self.common(fn, (_x,))

    @requires_vectorization
    def test_asinh_with_corner_inputs(self):
        # [FUNCTION DOC] Regression test (issue 142345): asinh on corner inputs for four dtypes
        # and each valid SIMD bit width; also expects exactly one vectorized kernel. Requires
        # vectorization.
        # https://github.com/pytorch/pytorch/issues/142345

        def fn(input):
            # [FUNCTION DOC] HELPER: Apply asinh.
            out = torch.asinh(input)
            return out

        x = torch.tensor([0, 0, 0, -10000.1]).repeat(3, 4)

        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()]
        for dtype in [torch.float32, torch.bfloat16, torch.float16, torch.double]:
            for simdlen in bit_widths:
                with torch.no_grad(), config.patch({"cpp.simdlen": simdlen}):
                    torch._dynamo.reset()
                    metrics.reset()
                    _x = x.to(dtype)
                    self.common(fn, (_x,))
                    check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_asinh_large_float32_inplace(self):
        # [FUNCTION DOC] Regression test (issue 152299): in-place asinh on very large float32
        # values for each valid SIMD bit width, checking one vectorized kernel. Requires
        # vectorization.
        # https://github.com/pytorch/pytorch/issues/152299

        def fn(input):
            # [FUNCTION DOC] HELPER: In-place asinh on the input; returns it.
            return input.asinh_()

        x = torch.tensor(
            [
                [
                    -1e30,
                    1e30,
                    -5e28,
                    5e28,
                    -7.5e29,
                    7.5e29,
                    -2e30,
                    2e30,
                    0.0,
                ]
            ],
            dtype=torch.float32,
        ).repeat(3, 1)

        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()]
        for simdlen in bit_widths:
            with torch.no_grad(), config.patch({"cpp.simdlen": simdlen}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x.clone(),))
                check_metrics_vec_kernel_count(1)

    @config.patch(fallback_random=True)
    def test_rrelu_fallback_random(self):
        # [FUNCTION DOC] Check a training-mode RReLU compiled with fallback random matches eager
        # when both are run after seeding the generator identically.
        rrelu = nn.RReLU().train()
        x = torch.randn(1, 1, 100, 100)

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply the enclosing RReLU module.
            return rrelu(x)

        compiled = torch.compile(fn, backend="inductor")

        with torch.no_grad():
            torch.manual_seed(0)
            expected = fn(x)
            torch.manual_seed(0)
            actual = compiled(x)

        self.assertEqual(actual, expected)

    @config.patch(fallback_random=True)
    def test_rrelu_with_noise_functional_fallback_random(self):
        # [FUNCTION DOC] Check the functional rrelu-with-noise op under fallback random, in
        # training (seeded, matching eager) and in evaluation mode with a constant noise tensor.
        x = torch.randn(64, 64)

        def fn(x):
            # [FUNCTION DOC] HELPER: Training-mode functional rrelu with noise buffer of the same
            # shape, slopes 1/8 to 1/3.
            noise = torch.empty_like(x)
            return torch.ops.aten.rrelu_with_noise_functional.default(
                x, noise, 0.125, 1.0 / 3.0, True
            )

        compiled = torch.compile(fn, backend="inductor")

        with torch.no_grad():
            torch.manual_seed(123)
            expected = fn(x)
            torch.manual_seed(123)
            actual = compiled(x)

        self.assertEqual(actual, expected)

        def fn_eval(x):
            # [FUNCTION DOC] HELPER: Evaluation-mode functional rrelu with a noise tensor filled
            # with 7.
            noise = torch.full_like(x, 7)
            return torch.ops.aten.rrelu_with_noise_functional.default(
                x, noise, 0.125, 1.0 / 3.0, False
            )

        compiled_eval = torch.compile(fn_eval, backend="inductor")

        with torch.no_grad():
            expected = fn_eval(x)
            actual = compiled_eval(x)

        self.assertEqual(actual, expected)

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/179957")
    @config.patch(fallback_random=True)
    def test_require_stride_order_non_owning(self):
        # [FUNCTION DOC] Regression test: compile a function concatenating two channels-last
        # tensors and also running a conv on one of them. Checks result equals eager and the
        # generated code allocates the conv inputs with channels-last strides (using the cpp
        # wrapper formatted arrays when that is on). Skipped on ROCm.
        def test_concat_with_conv():
            # [FUNCTION DOC] HELPER: Build two channels-last tensors, concatenate them on dim 1
            # and conv the first with channels-last weights; returns the concat result and the
            # conv result. Takes no inputs.
            x1 = torch.randn(2, 3, 4, 4).to(memory_format=torch.channels_last)
            x2 = torch.randn(2, 5, 4, 4).to(memory_format=torch.channels_last)

            # First do the concatenation
            cat_result = torch.cat([x1, x2], dim=1)

            # Then use x1 (which was an input to the cat) in a conv
            conv_weight = torch.randn(4, 3, 3, 3).to(memory_format=torch.channels_last)
            x1_conv = torch.nn.functional.conv2d(x1, conv_weight, padding=1)

            return cat_result, x1_conv

        torch.manual_seed(1)
        f_c = torch.compile(test_concat_with_conv)
        out_result, code = run_and_get_cpp_code(f_c)

        torch.manual_seed(1)
        self.assertEqual(out_result, test_concat_with_conv())

        # both inputs to conv should be channels last
        if config.cpp_wrapper:
            cpp_int_array_str = test_torchinductor.cpp_int_array_str
            FileCheck().check(cpp_int_array_str([2, 3, 4, 4])).check(
                cpp_int_array_str([128, 1, 32, 8])
            ).check(cpp_int_array_str([4, 3, 3, 3])).check(
                cpp_int_array_str([27, 1, 9, 3])
            ).check("aoti_torch_empty_strided").run(code)
        else:
            FileCheck().check("(2, 3, 4, 4), (128, 1, 32, 8)").check(
                "empty_strided_cpu((4, 3, 3, 3), (27, 1, 9, 3)"
            ).run(code)

    @config.patch(implicit_fallbacks=True)
    def test_repeat_interleave(self):
        # [FUNCTION DOC] Check repeat_interleave with a repeat count and fixed output size under
        # implicit fallbacks.
        def fn(y):
            # [FUNCTION DOC] HELPER: Repeat each element twice with output size 8.
            return torch.repeat_interleave(y, 2, output_size=8)

        a = torch.tensor([[1, 2], [3, 4]])
        self.common(
            fn,
            (a,),
        )

    def test_codegen_int_array_var_cache(self):
        # [FUNCTION DOC] Regression test for the C++ wrapper int array variable helper: bound
        # methods have transient ids, so the cache must be keyed on the owning object. Using a
        # mocked graph, calls the codegen twice with two bound methods of one buffer and asserts
        # both return the same variable name.
        """
        Test for the bug in codegen_int_array_var where bound method ids could collide,
        leading to undefined 'int_array_XXX' identifiers in generated C++ code.
        """
        from unittest.mock import MagicMock

        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
        from torch._inductor.virtualized import V

        class MockBuffer:
            # [CLASS DOC] HELPER: Minimal line sink recording written lines in a list.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create an empty list of lines.
                self.lines = []

            def writeline(self, line):
                # [FUNCTION DOC] HELPER: Append a line to the recorded list.
                self.lines.append(line)

        # Mock V.graph to avoid needing to compile a real model
        mock_graph = MagicMock()
        mock_graph.cpp_wrapper = True
        mock_graph.aot_mode = False
        mock_graph.is_const_graph = False
        mock_graph.device_types = []

        with V.set_graph_handler(mock_graph):
            wrapper = CppWrapperCpu()
            buffer = MockBuffer()

            # Bound methods in Python are instantiated on access.
            # Storing them ensures they have different ids.
            w1 = buffer.writeline
            w2 = buffer.writeline

            # The bug: without the fix, codegen_int_array_var uses id(writeline)
            # as the cache key, which leads to cache misses for bound methods.
            var1 = wrapper.codegen_int_array_var("{1, 2, 3}", w1)
            var2 = wrapper.codegen_int_array_var("{1, 2, 3}", w2)

            # They should return the same variable name because it's the same buffer
            self.assertEqual(
                var1,
                var2,
                "codegen_int_array_var should cache based on the bound method's __self__, "
                "not the transient bound method id itself.",
            )

    def test_inplace_squeeze_needed(self):
        # [FUNCTION DOC] Check an unbatched input through linear, layer norm and relu matches
        # eager with loose tolerance (parallel reduction order is not deterministic).
        mod = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.LayerNorm(10),
            torch.nn.ReLU(),
        ).eval()

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply the enclosing sequential model.
            return mod(x)

        v = torch.randn(10)
        # TODO: OMP parallel reduction order is not deterministic.
        # Hence, the accuracy might vary up and down. For short term,
        # we increase the tolerance and will fix it later by using
        # aten parallel.
        self.common(fn, (v,), atol=5e-1, rtol=5e-1)

    def test_parallel_reduction_vectorization(self):
        # [FUNCTION DOC] Regression test (issue 151523): parallel reduction vectorization with a
        # conv, optional hardshrink and a matrix-vector product, with and without masked tail
        # vectorization; loose tolerance.
        # Fix issue: https://github.com/pytorch/pytorch/issues/151523
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a conv, optionally a hardshrink, flatten of
            # batch, and a matvec with the given weight on the first row; the flag chooses whether
            # hardshrink is skipped.
            def __init__(self, enable_masked_tail_vec):
                # [FUNCTION DOC] HELPER: Build the conv and store the flag.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=(1, 7),
                    stride=(2, 1),
                    padding=0,
                )
                self.enable_masked_tail_vec = enable_masked_tail_vec

            def forward(self, x, weight):
                # [FUNCTION DOC] HELPER: Conv, optional hardshrink when the flag is off, flatten
                # per batch, matvec with weight.
                x = self.conv(x)
                if not self.enable_masked_tail_vec:
                    x = F.hardshrink(x, lambd=0)
                x = x.view(x.size(0), -1)
                x = torch.mv(weight, x[0])
                return x

        for enable_masked_tail_vec in [True, False]:
            mod = Model(enable_masked_tail_vec).eval()
            x = torch.randn(2, 3, 127, 255)
            weight = torch.randn(10, 254976)
            # Use same criterion as test_inplace_squeeze_needed
            # for parallel reduction.
            self.common(mod, (x, weight), atol=5e-1, rtol=5e-1)

    def test_cat_mul(self):
        # [FUNCTION DOC] Regression test (issue 93365): concatenation followed by an elementwise
        # square of the result, returning both.
        # https://github.com/pytorch/pytorch/issues/93365
        def fn(p0, p1):
            # [FUNCTION DOC] HELPER: Concatenate on dim 0 and multiply the result by itself;
            # returns both.
            y1 = torch.cat([p0, p1], dim=0)
            y2 = torch.mul(y1, y1)
            return y1, y2

        p0 = torch.randn(3, 4)
        p1 = torch.randn(3, 4)
        self.common(fn, (p0, p1))

    def test_pow_cos(self):
        # [FUNCTION DOC] Regression test (issue 98149): fifth power of a uint8 value followed by
        # cosine.
        # https://github.com/pytorch/pytorch/issues/98149
        def fn(x):
            # [FUNCTION DOC] HELPER: Raise to the fifth power then cosine.
            t = x.pow(5)
            return torch.cos(t)

        x = torch.tensor([4], dtype=torch.uint8)
        self.common(fn, (x,))

    def test_reduce_with_masked(self):
        # [FUNCTION DOC] Regression test (issue 96484): negative pad followed by add and a minimum
        # reduction.
        # https://github.com/pytorch/pytorch/issues/96484
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Negative pad of a, add b, then min along dim 0 values.
            a = torch.nn.functional.pad(a, (0, -1))
            c = a + b
            return c.min(0).values

        a = torch.randn([2])
        b = torch.randn([2])
        self.common(fn, (a, b))

    def test_scalar_sign_with_min(self):
        # [FUNCTION DOC] Regression test (issue 101340): sign of tanh combined with elementwise
        # minimum.
        # https://github.com/pytorch/pytorch/issues/101340
        def fn(a):
            # [FUNCTION DOC] HELPER: Tanh, then sign of that, then elementwise minimum of the two.
            t1 = torch.tanh(a)
            t2 = torch.sign(t1)
            return torch.min(t1, t2)

        a = torch.randn(1, 3)
        self.common(fn, (a,))

    def test_tanh_atan2(self):
        # [FUNCTION DOC] Regression test (issue 148241): tanhshrink followed by atan2 of a value
        # with itself matches eager.
        # https://github.com/pytorch/pytorch/issues/148241
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying tanhshrink then atan2 of the result with itself.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the tanhshrink submodule.
                super().__init__()
                self.shrink = nn.Tanhshrink()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply tanhshrink then atan2 with itself.
                x = self.shrink(x)
                x = torch.atan2(x, x)
                return x

        x = torch.randn(1, 3, 64, 64)
        self.common(Model(), (x,))

    @unittest.skipIf(
        os.getenv("ATEN_CPU_CAPABILITY") == "default",
        "Failing in periodic nogpu_NO_AVX2 after added in #152542",
    )
    @config.patch("cpp.use_decompose_tanh", "1")
    def test_tanh_atan2_use_decompose_tanh(self):
        # [FUNCTION DOC] Same model as the plain tanh atan2 test but with the decomposed-tanh cpp
        # option enabled; the test expects the comparison to raise AssertionError (known accuracy
        # mismatch). Skipped when the ATen CPU capability is forced to default.
        # https://github.com/pytorch/pytorch/issues/148241
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying tanhshrink then atan2 of the result with itself.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the tanhshrink submodule.
                super().__init__()
                self.shrink = nn.Tanhshrink()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply tanhshrink then atan2 with itself.
                x = self.shrink(x)
                x = torch.atan2(x, x)
                return x

        x = torch.randn(1, 3, 64, 64)
        with self.assertRaises(AssertionError):
            self.common(Model(), (x,))

    def test_index_propagation_issue_102065(self):
        # [FUNCTION DOC] Regression test (issue 102065): index propagation for an arange-based
        # pairwise squared difference matrix.
        def fn(x):
            # [FUNCTION DOC] HELPER: Build an arange of the input's element count and return
            # squared pairwise differences.
            x = torch.arange(x.numel())
            return (x.unsqueeze(0) - x.unsqueeze(1)) ** 2

        self.common(
            fn,
            (torch.randn(8),),
        )

    def test_low_fp_index_expr_issue_147279(self):
        # [FUNCTION DOC] Regression test (issue 147279): sum of a float16 arange over dim 0, with
        # arange bounds given as runtime arguments.
        # https://github.com/pytorch/pytorch/issues/147279
        def fn(start, end, dtype, dim):
            # [FUNCTION DOC] HELPER: Sum an arange of the given start, end and dtype along the
            # given dims.
            return torch.sum(
                torch.arange(start=start, end=end, dtype=dtype),
                dim=dim,
            )

        self.common(
            fn,
            (300, 400, torch.float16, (0,)),
        )

    def test_index_put(self):
        # [FUNCTION DOC] Regression test (issue 138908): in-place accumulate through an indexed
        # read of the same tensor with a negative-wrapped index; compiled and eager results of the
        # mutated tensor match.
        # https://github.com/pytorch/pytorch/issues/138908
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add ten to x and then add y at that index into y in place;
            # mutates y.
            x = x + 10
            y[x] += y[x]

        x = torch.randint(-10, -9, (1, 2), dtype=torch.int64)
        y = torch.randn((2, 32), dtype=torch.float32)
        x_clone = x.clone()
        y_clone = y.clone()
        with torch.no_grad():
            fn(x, y)
            torch.compile(fn)(x_clone, y_clone)
            self.assertEqual(y, y_clone, atol=1e-3, rtol=1e-3)

    def test_index_put2(self):
        # [FUNCTION DOC] Regression test (issue 138908): in-place add of an indexed slice into
        # another indexed slice using index tensors; compared to eager on the mutated tensor.
        # https://github.com/pytorch/pytorch/issues/138908
        def fn(y, index0, index1):
            # [FUNCTION DOC] HELPER: Add y gathered by index0 into y at index1 in place.
            y[index1] += y[index0]

        y = torch.randn((2, 32), dtype=torch.float32)
        index0 = torch.tensor([[0, 1]])
        index1 = torch.tensor([[1, 0]])
        y_clone = y.clone()
        index0_clone = index0.clone()
        index1_clone = index1.clone()
        with torch.no_grad():
            fn(y, index0, index1)
            torch.compile(fn)(y_clone, index0_clone, index1_clone)
            self.assertEqual(y, y_clone, atol=1e-3, rtol=1e-3)

    def test_index_put_bool_accumulate_codegen(self):
        # [FUNCTION DOC] Regression test (issue 113692): boolean index_put with accumulate must be
        # code-generated and not use the fallback. Runs with a values tensor and with computed
        # values (with dynamic threads), under fresh cache and a patch that raises AssertionError
        # if the fallback is called.
        # https://github.com/pytorch/pytorch/issues/113692
        def fn(x, index, values):
            # [FUNCTION DOC] HELPER: Clone x, index_put with accumulate using given values and
            # return the clone.
            x = x.clone()
            x.index_put_((index,), values, accumulate=True)
            return x

        def fn_computed_values(x, index):
            # [FUNCTION DOC] HELPER: Clone x, index_put with accumulate using a boolean computed
            # from index and return it.
            x = x.clone()
            x.index_put_((index,), index > 0, accumulate=True)
            return x

        x = torch.tensor([False, False, True, False], dtype=torch.bool)
        index = torch.tensor([0, 1, 1, 3], dtype=torch.int64)
        values = torch.tensor([True, True, False, True], dtype=torch.bool)

        def forbid_index_put_fallback():
            # [FUNCTION DOC] HELPER: Return a patch context object replacing the index_put
            # fallback lowering with a mock that raises AssertionError when called.
            return patch(
                "torch._inductor.lowering.index_put_fallback",
                side_effect=AssertionError("unexpected CPU index_put fallback"),
            )

        with fresh_cache(), forbid_index_put_fallback():
            expected = fn(x, index, values)
            actual = torch.compile(fn, fullgraph=True)(x, index, values)
        self.assertEqual(actual, expected)

        with (
            fresh_cache(),
            config.patch({"cpp.dynamic_threads": True}),
            forbid_index_put_fallback(),
        ):
            expected = fn_computed_values(x, index)
            actual = torch.compile(fn_computed_values, fullgraph=True)(x, index)
        self.assertEqual(actual, expected)

    def test_index_add(self):
        # [FUNCTION DOC] Regression test (issue 138908): index_add of a computed source, with
        # int32 indices; compares eager and compiled twice including a second run with zero
        # indices on fresh clones.
        # https://github.com/pytorch/pytorch/issues/138908
        def fn(x, y, scale_y, index):
            # [FUNCTION DOC] HELPER: Compute values from x at index plus scaled y, index_add them
            # into x in place, return x.
            values = x[index] + y * scale_y
            out = x.index_add_(dim=0, source=values, index=index)
            return out

        inp = (
            torch.randn(10, 10),
            torch.randn(5, 10),
            torch.randn(10),
            torch.randperm(10, device="cpu")[:5].to(torch.int32),
        )
        inp_clones = []
        for i in range(3):
            inp_clones.append(
                [
                    inp[0].clone(),
                    inp[1].clone(),
                    inp[2].clone(),
                    inp[3].clone()
                    if i == 0
                    else torch.zeros(10, device="cpu")[:5].to(torch.int32),
                ]
            )
        inp_clone, inp_clone2, inp_clone3 = inp_clones
        with torch.no_grad():
            cfn = torch.compile(fn)
            ref = fn(*inp)
            res = cfn(*inp_clone)
            self.assertEqual(ref, res, atol=1e-3, rtol=1e-3)
            ref = fn(*inp_clone2)
            res = cfn(*inp_clone3)
            self.assertEqual(ref, res, atol=1e-3, rtol=1e-3)

    def test_randperm_full_advanced_indexing_issue_158457(self):
        # [FUNCTION DOC] Regression test (issue 158457): advanced indexing with a full random
        # permutation compiles with full graph; checks output shape and that the first column is a
        # permutation of the original.
        def full_index(x):
            # [FUNCTION DOC] HELPER: Index rows with a random permutation of the row count.
            perm = torch.randperm(x.size(0))
            return x[perm]

        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        full_actual = torch.compile(full_index, backend="inductor", fullgraph=True)(x)
        self.assertEqual(full_actual.shape, x.shape)
        self.assertEqual(torch.sort(full_actual[:, 0]).values, x[:, 0])

    def test_randperm_sliced_advanced_indexing_issue_158457(self):
        # [FUNCTION DOC] Regression test (issue 158457): advanced indexing with a sliced random
        # permutation. Checks the shape, row structure and that selected rows are distinct and
        # drawn from the input.
        def sliced_index(x):
            # [FUNCTION DOC] HELPER: Index rows with the first two entries of a random
            # permutation.
            perm = torch.randperm(x.size(0))[:2]
            return x[perm]

        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        sliced_actual = torch.compile(sliced_index, backend="inductor", fullgraph=True)(
            x
        )
        self.assertEqual(sliced_actual.shape, (2, x.size(1)))
        self.assertEqual(sliced_actual[:, 1:], sliced_actual[:, :1] + x[0, 1:])
        self.assertTrue((sliced_actual[:, :1] == x[:, 0]).any(dim=1).all().item())
        self.assertEqual(
            torch.unique(sliced_actual[:, 0]).numel(), sliced_actual.size(0)
        )

    def test_ModularIndexing_range_issue_103133(self):
        # [FUNCTION DOC] Regression test (issue 103133): modular indexing range handling in an
        # einsum, pad, view and slice-assign pattern on strided empty inputs.
        def fn(q, k):
            # [FUNCTION DOC] HELPER: Einsum, pad the last dim, reshape, and assign a slice into a
            # zero tensor; returns it.
            einsum = torch.einsum("bcxd,bcyd->bcxy", (q, k))
            constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                einsum, [0, 0, 0, 1], 0.0
            )
            view = torch.ops.aten.view.default(constant_pad_nd, [12, 1, 512, 513])
            y = view.new_zeros((12, 2, 256, 513))
            y[:, :-1, :, 256:] = view[:, :, :256, :257]
            return y

        self.common(
            fn,
            (
                torch.empty_strided((12, 1, 512, 64), (64, 196608, 768, 1)),
                torch.empty_strided((12, 1, 512, 64), (64, 196608, 768, 1)),
            ),
        )

    @patch("torch.cuda.is_available", lambda: False)
    def test_max_reduction_lowp_fp(self):
        # [FUNCTION DOC] Check max reduction along dim 1 with keepdim for bf16 and fp16 inputs,
        # converted to float.
        def fn(x):
            # [FUNCTION DOC] HELPER: Max along dim 1 keeping dims, take values, convert to float.
            return torch.ops.aten.max(x, 1, keepdim=True)[0].float()

        for dtype in _lowp_fp_dtypes:
            self.common(
                fn,
                (torch.randn(1, 32, 4, 4).to(dtype),),
            )

    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_transpose_lowp_fp(self):
        # [FUNCTION DOC] Check conversion of float32 input to channels last and then to each
        # low-precision dtype.
        for dtype in _lowp_fp_dtypes:

            def fn(x):
                # [FUNCTION DOC] HELPER: Convert to channels-last then to the loop's enclosing
                # dtype.
                return x.to(memory_format=torch.channels_last).to(dtype)

            self.common(
                fn,
                (torch.randn(2, 3, 4, 4),),
            )

    def test_load_inf_bf16(self):
        # [FUNCTION DOC] Check where with positive and negative infinity scalars as the other
        # branch.
        def fn1(x):
            # [FUNCTION DOC] HELPER: Select x where positive else positive infinity.
            return torch.where(x > 0, x, math.inf)

        def fn2(x):
            # [FUNCTION DOC] HELPER: Select x where positive else negative infinity.
            return torch.where(x > 0, x, -math.inf)

        for fn in [fn1, fn2]:
            self.common(
                fn,
                (torch.randn(1, 3, 16, 16),),
            )

    @patch("torch.cuda.is_available", lambda: False)
    def test_fp32_load_with_to_lowp_fp(self):
        # [FUNCTION DOC] Check (from llama) a module that casts a stored float32 cache to the
        # input dtype and assigns a slice, for bf16 and fp16, compared with an eager copy.
        # From llama model.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a zero cache buffer attribute; forward casts the
            # cache to x and writes xk into a slice, returning the cache.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the zero cache tensor.
                super().__init__()
                self.cache_k = torch.zeros(8, 4, 2, 2)

            def forward(self, x, xk):
                # [FUNCTION DOC] HELPER: Cast the cache to the input, write xk into a slice of it
                # and return it.
                bsz, seqlen, _ = x.shape
                self.cache_k = self.cache_k.to(x)
                self.cache_k[:bsz, 1 : 1 + seqlen] = xk
                return self.cache_k

        for dtype in _lowp_fp_dtypes:
            ref_model = Model().eval()
            opt_model = torch.compile()(Model().eval())
            x = torch.randn(4, 2, 2).to(dtype)
            xk = torch.randn(4, 2, 2, 2).to(dtype)
            self.assertEqual(opt_model(x, xk), ref_model(x, xk))

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_sigmoid_with_reduction(self):
        # [FUNCTION DOC] Check sigmoid followed by a mean over the last two dims with keepdim
        # under default simdlen. Requires vectorization.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sigmoid, then mean over last two dims keeping dims.
            x = torch.ops.aten.sigmoid.default(x)
            return torch.ops.aten.mean.dim(x, [-1, -2], True)

        x = torch.randn((1, 8, 8, 8))
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))

    def test_slice_scatter_default_end_value(self):
        # [FUNCTION DOC] Check (from Longformer) slice-scatter of a key slice into a zeros tensor
        # with a default end value. Raises AssertionError inside if the sequence length is not a
        # multiple of twice the overlap.
        # From HF AllenaiLongformerBase.
        def fn(query, key, window_overlap):
            # [FUNCTION DOC] HELPER: Build zeros of chunked shape and assign part of key into a
            # slice of it; returns it.
            batch_size, seq_len, num_heads, head_dim = query.size()
            if seq_len % (window_overlap * 2) != 0:
                raise AssertionError(
                    f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
                )

            chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
            diagonal_chunked_attention_scores = key
            diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
                (
                    batch_size * num_heads,
                    chunks_count + 1,
                    window_overlap,
                    window_overlap * 2 + 1,
                )
            )
            diagonal_attention_scores[:, :3, :, window_overlap:] = (
                diagonal_chunked_attention_scores[
                    :, :, :window_overlap, : window_overlap + 1
                ]
            )
            return diagonal_attention_scores

        self.common(
            fn,
            (
                torch.randn(1, 1024, 12, 64),
                torch.randn(12, 3, 512, 513),
                256,
            ),
        )

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_to_uint8_rounding_method(self):
        # [FUNCTION DOC] Check conversion to uint8 of values 4.4 to 5.5 vectorizes (one vec
        # kernel) and rounds like eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to uint8.
            return x.to(torch.uint8)

        numerical_testsuit = [4.4, 4.5, 4.6, 5.5]
        for numerical_number in numerical_testsuit:
            x = torch.ones(17) * numerical_number
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def _test_decomposed_dequant_relu_quant_helper(self, dtype):
        # [FUNCTION DOC] HELPER: Shared body for decomposed dequantize, relu and quantize tests
        # with uint8 or int8 dtype. Runs all combinations of dequantize and quantize on/off, and
        # checks results equal eager and exactly one vectorized kernel. Raises AssertionError for
        # other dtypes.
        def fn(
            x, scale, zero_point, use_dequant, use_quant, quant_min, quant_max, dtype
        ):
            # [FUNCTION DOC] HELPER: Optionally dequantize manually, apply relu, optionally
            # quantize manually with rounding and clamping, returning the result.
            # For quantized_decomposed.dequantize_per_tensor
            # Refer to torch/ao/quantization/fx/_decomposed.py
            if use_dequant:
                x = (x.to(torch.float32) - zero_point) * scale

            x = torch.relu(x)

            # For quantized_decomposed.quantize_per_tensor
            # Refer to torch/ao/quantization/fx/_decomposed.py
            if use_quant:
                inv_scale = 1.0 / scale
                x = torch.clamp(
                    torch.round(x * inv_scale) + zero_point, quant_min, quant_max
                ).to(dtype)
            return x

        if dtype not in [torch.uint8, torch.int8]:
            raise AssertionError(f"Unexpected dtype: {dtype}")
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        use_dequant_list = [False, True]
        use_quant_list = [False, True]
        for use_dequant, use_quant in itertools.product(
            use_dequant_list, use_quant_list
        ):
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            )
            if use_dequant:
                x = x.to(dtype)
            zero_point = 100
            scale = 0.01
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        use_dequant,
                        use_quant,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                )
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_decomposed_dequant_relu_quant_uint8(self):
        # [FUNCTION DOC] Run the decomposed dequant relu quant helper with uint8.
        self._test_decomposed_dequant_relu_quant_helper(torch.uint8)

    @requires_vectorization
    def test_decomposed_dequant_relu_quant_int8(self):
        # [FUNCTION DOC] Run the decomposed dequant relu quant helper with int8.
        self._test_decomposed_dequant_relu_quant_helper(torch.int8)

    def _test_dequant_quant_lowering_helper(
        self,
        dtype,
        input_dtype=torch.float32,
        dequant_out_dtype=None,
    ):
        # [FUNCTION DOC] HELPER: Shared body for dequant, relu, quant lowering tests using the
        # quantized_decomposed ops. Accepts uint8, int8 or float8 dtypes, an input dtype and
        # optional dequantize output dtype; loops over dequant/quant/tensor-overload variants,
        # checks eager equality and one vector kernel, and for float8 additionally checks the code
        # has exactly two vector load sites (main and tail).
        def fn(
            x,
            scale,
            zero_point,
            use_dequant,
            use_quant,
            quant_min,
            quant_max,
            dtype,
            dequant_out_dtype,
        ):
            # [FUNCTION DOC] HELPER: Optionally dequantize with the decomposed op, relu,
            # optionally quantize; returns result.
            if use_dequant:
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x,
                    scale,
                    zero_point,
                    quant_min,
                    quant_max,
                    dtype,
                    out_dtype=dequant_out_dtype,
                )

            x = torch.relu(x)

            if use_quant:
                x = torch.ops.quantized_decomposed.quantize_per_tensor(
                    x, scale, zero_point, quant_min, quant_max, dtype
                )
            return x

        use_dequant_list = [False, True]
        use_quant_list = [False, True]
        use_tensor_overload_list = [False, True]

        if dtype not in [
            torch.uint8,
            torch.int8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            raise AssertionError(f"Unexpected dtype: {dtype}")
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127
        if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            quant_min = int(torch.finfo(dtype).min)
            quant_max = int(torch.finfo(dtype).max)
            use_tensor_overload_list = [
                False,
            ]

        for (
            use_dequant,
            use_quant,
            use_tensor_overload,
        ) in itertools.product(
            use_dequant_list,
            use_quant_list,
            use_tensor_overload_list,
        ):
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=input_dtype) * 100,
                quant_min,
                quant_max,
            )
            if use_dequant:
                x = x.to(dtype)
            zero_point = 100
            scale = 0.01
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)
                scale = torch.tensor(scale)
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                inputs = (
                    x,
                    scale,
                    zero_point,
                    use_dequant,
                    use_quant,
                    quant_min,
                    quant_max,
                    dtype,
                    dequant_out_dtype,
                )
                self.common(fn, inputs)
                check_metrics_vec_kernel_count(1)

                # Check that both main and tail loops are vectorized
                if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    compiled_fn = torch.compile(fn)
                    _, code = run_and_get_cpp_code(compiled_fn, *inputs)
                    FileCheck().check_count("loadu", 2, exactly=True).run(code)

    @requires_vectorization
    def test_dequant_quant_lowering_uint8(self):
        # [FUNCTION DOC] Dequant relu quant lowering for uint8 with default, bf16 input and bf16
        # dequant output variants.
        self._test_dequant_quant_lowering_helper(torch.uint8)
        self._test_dequant_quant_lowering_helper(
            torch.uint8, input_dtype=torch.bfloat16
        )
        self._test_dequant_quant_lowering_helper(
            torch.uint8, dequant_out_dtype=torch.bfloat16
        )

    @requires_vectorization
    def test_dequant_quant_lowering_int8(self):
        # [FUNCTION DOC] Dequant relu quant lowering for int8 with default, bf16 input and bf16
        # dequant output variants.
        self._test_dequant_quant_lowering_helper(torch.int8)
        self._test_dequant_quant_lowering_helper(torch.int8, input_dtype=torch.bfloat16)
        self._test_dequant_quant_lowering_helper(
            torch.int8, dequant_out_dtype=torch.bfloat16
        )

    @requires_vectorization
    def test_dequant_quant_lowering_fp8_e4m3(self):
        # [FUNCTION DOC] Dequant relu quant lowering for float8 e4m3.
        self._test_dequant_quant_lowering_helper(torch.float8_e4m3fn)

    @requires_vectorization
    def test_dequant_quant_lowering_fp8_e5m2(self):
        # [FUNCTION DOC] Dequant relu quant lowering for float8 e5m2.
        self._test_dequant_quant_lowering_helper(torch.float8_e5m2)

    def _test_dequant_maxpool2d_lowering_helper(self, dtype):
        # [FUNCTION DOC] HELPER: Shared body checking dequantize followed by max pool on
        # channels-last data for uint8 or int8, with scalar and tensor scale/zero point; expects
        # one vector kernel. Raises AssertionError for other dtypes.
        def fn(x, scale, zero_point, quant_min, quant_max, dtype):
            # [FUNCTION DOC] HELPER: Dequantize then 2x2 max pool with stride 2 and padding 1;
            # return the values only.
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale, zero_point, quant_min, quant_max, dtype
            )
            max_pool2d_with_indices_default = (
                torch.ops.aten.max_pool2d_with_indices.default(
                    x, [2, 2], [2, 2], [1, 1]
                )[0]
            )
            return max_pool2d_with_indices_default

        if dtype not in [torch.uint8, torch.int8]:
            raise AssertionError(f"Unexpected dtype: {dtype}")
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        use_tensor_overload_list = [False, True]
        for use_tensor_overload in use_tensor_overload_list:
            x = (
                torch.clamp(
                    torch.randn((3, 16, 8, 8), dtype=torch.float32) * 100,
                    quant_min,
                    quant_max,
                )
                .to(dtype)
                .contiguous(memory_format=torch.channels_last)
            )
            zero_point = 100
            scale = 0.01
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)
                scale = torch.tensor(scale)
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x, scale, zero_point, quant_min, quant_max, dtype))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_dequant_maxpool2d_lowering_uint8(self):
        # [FUNCTION DOC] Dequant max pool lowering for uint8.
        self._test_dequant_maxpool2d_lowering_helper(torch.uint8)

    @requires_vectorization
    def test_dequant_maxpool2d_lowering_int8(self):
        # [FUNCTION DOC] Dequant max pool lowering for int8.
        self._test_dequant_maxpool2d_lowering_helper(torch.int8)

    def _test_tile2d_load_decomposed_dequant_add_relu_quant_helper(self, dtype):
        # [FUNCTION DOC] HELPER: Shared body for tiled 2D load tests of two optionally dequantized
        # channels-last inputs added, relu'd and optionally quantized; checks eager equality and
        # two vector kernels for uint8 or int8. Raises AssertionError for other dtypes.
        def fn(
            x,
            scale,
            zero_point,
            x2,
            scale2,
            zero_point2,
            output_scale,
            output_zero_point,
            use_dequant,
            use_dequant2,
            use_quant,
            quant_min,
            quant_max,
            dtype,
        ):
            # [FUNCTION DOC] HELPER: Optionally dequantize both inputs, add, relu, optionally
            # quantize, return contiguous.
            if use_dequant:
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, scale, zero_point, quant_min, quant_max, dtype
                )
            if use_dequant2:
                x2 = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x2, scale2, zero_point2, quant_min, quant_max, dtype
                )
            temp = x + x2
            y = torch.relu(temp)

            if use_quant:
                y = torch.ops.quantized_decomposed.quantize_per_tensor(
                    y, output_scale, output_zero_point, quant_min, quant_max, dtype
                )
            return y.contiguous()

        if dtype not in [torch.uint8, torch.int8]:
            raise AssertionError(f"Unexpected dtype: {dtype}")
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        use_dequant_list = [False, True]
        use_dequant_list2 = [False, True]
        use_quant_list = [False, True]

        for use_dequant, use_dequant2, use_quant in itertools.product(
            use_dequant_list, use_dequant_list2, use_quant_list
        ):
            x = torch.clamp(
                torch.randn((1, 1024, 14, 14), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            ).contiguous(memory_format=torch.channels_last)
            x2 = torch.clamp(
                torch.randn((1, 1024, 14, 14), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            ).contiguous(memory_format=torch.channels_last)
            if use_dequant:
                x = x.to(dtype).contiguous(memory_format=torch.channels_last)
            if use_dequant2:
                x2 = x2.to(dtype).contiguous(memory_format=torch.channels_last)
            zero_point = 1
            scale = 0.01
            zero_point2 = 2
            scale2 = 0.02
            output_zero_point = 3
            output_scale = 0.03
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        x2,
                        scale2,
                        zero_point2,
                        output_scale,
                        output_zero_point,
                        use_dequant,
                        use_dequant2,
                        use_quant,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                )
                check_metrics_vec_kernel_count(2)

    @requires_vectorization
    def test_tile2d_load_decomposed_dequant_add_relu_quant_uint8(self):
        # [FUNCTION DOC] Tile 2D load decomposed dequant add relu quant test for uint8.
        self._test_tile2d_load_decomposed_dequant_add_relu_quant_helper(torch.uint8)

    @requires_vectorization
    def test_tile2d_load_decomposed_dequant_add_relu_quant_int8(self):
        # [FUNCTION DOC] Tile 2D load decomposed dequant add relu quant test for int8.
        self._test_tile2d_load_decomposed_dequant_add_relu_quant_helper(torch.int8)

    @requires_vectorization
    def _test_per_tensor_fake_quant_helper(self, dtype):
        # [FUNCTION DOC] HELPER: Shared body for per-tensor fake quantization (quantize then
        # dequantize) for uint8 or int8 with scalar and tensor parameters; expects exactly one
        # vector kernel. Requires vectorization.
        def fn(input, scales, zero_points, quant_min, quant_max, dtype):
            # [FUNCTION DOC] HELPER: Quantize per tensor then dequantize per tensor.
            input = torch.ops.quantized_decomposed.quantize_per_tensor(
                input, scales, zero_points, quant_min, quant_max, dtype
            )
            input = torch.ops.quantized_decomposed.dequantize_per_tensor(
                input, scales, zero_points, quant_min, quant_max, dtype
            )
            return input

        use_tensor_overload_list = [False, True]
        for use_tensor_overload in use_tensor_overload_list:
            if dtype not in [torch.uint8, torch.int8]:
                raise AssertionError(f"Unexpected dtype: {dtype}")
            quant_min = 0 if dtype == torch.uint8 else -128
            quant_max = 255 if dtype == torch.uint8 else 127
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            )
            zero_point = 100
            scale = 0.01
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)
                scale = torch.tensor(scale)
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x, scale, zero_point, quant_min, quant_max, dtype))
                if metrics.generated_cpp_vec_kernel_count != 1:
                    raise AssertionError(
                        f"Expected 1 vec kernel, got {metrics.generated_cpp_vec_kernel_count}"
                    )

    @requires_vectorization
    def test_per_tensor_fake_quant_uint8(self):
        # [FUNCTION DOC] Per tensor fake quant test for uint8.
        self._test_per_tensor_fake_quant_helper(torch.uint8)

    @requires_vectorization
    def test_per_tensor_fake_quant_int8(self):
        # [FUNCTION DOC] Per tensor fake quant test for int8.
        self._test_per_tensor_fake_quant_helper(torch.int8)

    def _test_per_channel_fake_quant_helper(
        self, dtype, input_dtype=torch.float32, output_dtype=None
    ):
        # [FUNCTION DOC] HELPER: Shared body for per-channel fake quantization on a 3-channel
        # image along axis 1 with chosen input and output dtypes; checks eager equality and one
        # vector kernel. Raises AssertionError for dtypes other than uint8 and int8.
        def fn(
            input, scales, zero_points, axis, quant_min, quant_max, dtype, output_dtype
        ):
            # [FUNCTION DOC] HELPER: Quantize per channel then dequantize per channel with the
            # output dtype.
            input = torch.ops.quantized_decomposed.quantize_per_channel(
                input, scales, zero_points, axis, quant_min, quant_max, dtype
            )
            input = torch.ops.quantized_decomposed.dequantize_per_channel(
                input,
                scales,
                zero_points,
                axis,
                quant_min,
                quant_max,
                dtype,
                out_dtype=output_dtype,
            )
            return input

        if dtype not in [torch.uint8, torch.int8]:
            raise AssertionError(f"Unexpected dtype: {dtype}")
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127
        x = torch.clamp(
            torch.randn((1, 3, 224, 224), dtype=torch.float32) * 100,
            quant_min,
            quant_max,
        )
        if input_dtype != torch.float32:
            x = x.to(dtype=input_dtype)
        scales = torch.ones((3,))
        zero_points = torch.zeros((3,))
        axis = 1
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(
                fn,
                (
                    x,
                    scales,
                    zero_points,
                    axis,
                    quant_min,
                    quant_max,
                    dtype,
                    output_dtype,
                ),
            )
            check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_per_channel_fake_quant_uint8(self):
        # [FUNCTION DOC] Per channel fake quant test for uint8.
        self._test_per_channel_fake_quant_helper(torch.uint8)

    @requires_vectorization
    def test_per_channel_fake_quant_module_uint8(self):
        # [FUNCTION DOC] Per channel fake quant through a module holding double scales and int64
        # zero points; expects eager equality and exactly one vector kernel (raises AssertionError
        # otherwise).
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing per-channel parameters as plain attributes and
            # applying quantize then dequantize per channel.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Store scales, zero points, axis, range and dtype
                # attributes.
                super().__init__()
                self.scales = torch.ones((3,)).to(torch.float64)
                self.zero_points = torch.zeros((3,)).to(torch.int64)
                self.axis = 1
                self.quant_min = 0
                self.quant_max = 255
                self.dtype = torch.uint8

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Quantize then dequantize per channel using stored
                # attributes.
                input = torch.ops.quantized_decomposed.quantize_per_channel(
                    input,
                    self.scales,
                    self.zero_points,
                    self.axis,
                    self.quant_min,
                    self.quant_max,
                    self.dtype,
                )
                input = torch.ops.quantized_decomposed.dequantize_per_channel(
                    input,
                    self.scales,
                    self.zero_points,
                    self.axis,
                    self.quant_min,
                    self.quant_max,
                    self.dtype,
                )
                return input

        m = Mod().eval()
        x = torch.clamp(
            torch.randn((1, 3, 224, 224), dtype=torch.float32) * 100,
            0,
            255,
        )
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(m, (x,))
            if metrics.generated_cpp_vec_kernel_count != 1:
                raise AssertionError(
                    f"Expected 1 vec kernel, got {metrics.generated_cpp_vec_kernel_count}"
                )

    @requires_vectorization
    def test_per_channel_fake_quant_int8(self):
        # [FUNCTION DOC] Per channel fake quant test for int8.
        self._test_per_channel_fake_quant_helper(torch.int8)

    @requires_vectorization
    def test_per_channel_fake_quant_uint8_bf16_input(self):
        # [FUNCTION DOC] Per channel fake quant uint8 with bf16 input, with and without bf16
        # output dtype.
        self._test_per_channel_fake_quant_helper(
            torch.uint8, input_dtype=torch.bfloat16
        )
        self._test_per_channel_fake_quant_helper(
            torch.uint8, input_dtype=torch.bfloat16, output_dtype=torch.bfloat16
        )

    @requires_vectorization
    def test_per_channel_fake_quant_int8_bf16_input(self):
        # [FUNCTION DOC] Per channel fake quant int8 with bf16 input, with and without bf16 output
        # dtype.
        self._test_per_channel_fake_quant_helper(torch.int8, input_dtype=torch.bfloat16)
        self._test_per_channel_fake_quant_helper(
            torch.int8, input_dtype=torch.bfloat16, output_dtype=torch.bfloat16
        )

    def _test_non_contiguous_load_buf_quant_helper(self, dtype):
        # [FUNCTION DOC] HELPER: Shared body for a channel-shuffle-like pattern: two channels-last
        # integer tensors are concatenated, dequantized, regrouped, requantized, dequantized,
        # transposed and flattened back. Checks eager equality and two vector kernels for uint8 or
        # int8. Raises AssertionError otherwise.
        def fn(
            x1,
            x2,
            groups,
            quant_min,
            quant_max,
            dtype,
        ):
            # [FUNCTION DOC] HELPER: Concatenate on dim 1, dequantize, view as groups, quantize,
            # dequantize, transpose group and channel axes, and view back to the original rank.
            x = torch.cat((x1, x2), dim=1)
            batchsize, num_channels, height, width = x.size()
            channels_per_group = num_channels // groups
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, 1.0, 0, quant_min, quant_max, dtype
            )
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, 1.0, 0, quant_min, quant_max, dtype
            )
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, 1.0, 0, quant_min, quant_max, dtype
            )
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, num_channels, height, width)
            return x

        if dtype not in [torch.uint8, torch.int8]:
            raise AssertionError(f"Unexpected dtype: {dtype}")
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        x = torch.randint(0, 8, (1, 116, 28, 28), dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )
        x2 = torch.randint(0, 8, (1, 116, 28, 28), dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(
                fn,
                (
                    x,
                    x2,
                    2,
                    quant_min,
                    quant_max,
                    dtype,
                ),
            )
            check_metrics_vec_kernel_count(2)

    @requires_vectorization
    def test_non_contiguous_load_buf_quant_uint8(self):
        # [FUNCTION DOC] Non-contiguous load buffer quant test for uint8.
        self._test_non_contiguous_load_buf_quant_helper(torch.uint8)

    @requires_vectorization
    def test_non_contiguous_load_buf_quant_int8(self):
        # [FUNCTION DOC] Non-contiguous load buffer quant test for int8.
        self._test_non_contiguous_load_buf_quant_helper(torch.int8)

    def _test_tile2d_store_channel_shuffle_cl_quant_output_helper(self, dtype):
        # [FUNCTION DOC] HELPER: Shared body checking a channel shuffle with quantized
        # channels-last output in 2D tiled store form for uint8 or int8; expects two vector
        # kernels. Raises AssertionError for other dtypes.
        def channel_shuffle(
            x, groups, output_scale, output_zero_point, quant_min, quant_max, dtype
        ):
            # [FUNCTION DOC] HELPER: Group-wise channel shuffle via view and transpose, quantize
            # per tensor, and return the result in channels-last layout.
            batchsize, num_channels, height, width = x.size()
            channels_per_group = num_channels // groups
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, -1, height, width)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, output_scale, output_zero_point, quant_min, quant_max, dtype
            )
            return x.contiguous(memory_format=torch.channels_last)

        if dtype not in [torch.uint8, torch.int8]:
            raise AssertionError(f"Unexpected dtype: {dtype}")
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            x = torch.randn(64, 58, 28, 28)
            output_zero_point = 3
            output_scale = 0.03
            self.common(
                channel_shuffle,
                (x, 2, output_scale, output_zero_point, quant_min, quant_max, dtype),
            )
            check_metrics_vec_kernel_count(2)

    @requires_vectorization
    def test_tile2d_store_channel_shuffle_cl_quant_output_uint8(self):
        # [FUNCTION DOC] Channel shuffle channels-last quant output test for uint8.
        self._test_tile2d_store_channel_shuffle_cl_quant_output_helper(torch.uint8)

    @requires_vectorization
    def test_tile2d_store_channel_shuffle_cl_quant_output_int8(self):
        # [FUNCTION DOC] Channel shuffle channels-last quant output test for int8.
        self._test_tile2d_store_channel_shuffle_cl_quant_output_helper(torch.int8)

    @requires_vectorization
    def test_to_channels_last_fp8(self):
        # [FUNCTION DOC] Check converting float8 (both formats) tensors to channels last matches
        # eager with two vector kernels. Requires vectorization.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to channels-last.
            return x.to(memory_format=torch.channels_last)

        for dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            torch._dynamo.reset()
            metrics.reset()
            self.common(
                fn,
                (torch.randn(20, 16, 48, 48).to(dtype=dtype),),
            )
            check_metrics_vec_kernel_count(2)

    def _test_dequant_relu_quant_dequant_relu_quant_lowering_helper(self, dtype):
        # [FUNCTION DOC] HELPER: Shared body for two chained dequant, relu, quant stages in uint8
        # or int8, with scalar or tensor scale and zero point; expects one vector kernel and loose
        # tolerance. Raises AssertionError for other dtypes.
        def fn(
            x,
            scale,
            zero_point,
            scale2,
            zero_point2,
            scale3,
            zero_point3,
            quant_min,
            quant_max,
            dtype,
        ):
            # [FUNCTION DOC] HELPER: Dequantize, relu, quantize, dequantize, relu, quantize using
            # three parameter sets.
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale, zero_point, quant_min, quant_max, dtype
            )
            x = torch.relu(x)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, scale2, zero_point2, quant_min, quant_max, dtype
            )
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale2, zero_point2, quant_min, quant_max, dtype
            )
            x = torch.relu(x)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, scale3, zero_point3, quant_min, quant_max, dtype
            )
            return x

        if dtype not in [torch.uint8, torch.int8]:
            raise AssertionError(f"Unexpected dtype: {dtype}")
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        for use_tensor_overload in [True, False]:
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            ).to(dtype)
            zero_point_list = [100, 101, 102]
            scale_list = [0.01, 0.02, 0.03]
            if use_tensor_overload:
                for i in range(len(zero_point_list)):
                    zero_point_list[i] = torch.tensor(
                        zero_point_list[i], dtype=torch.int64
                    )
                    scale_list[i] = torch.tensor(scale_list[i])
            zero_point, zero_point2, zero_point3 = zero_point_list
            scale, scale2, scale3 = scale_list
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        scale2,
                        zero_point2,
                        scale3,
                        zero_point3,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                    rtol=1e-2,
                    atol=1e-2,
                )
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_dequant_relu_quant_dequant_relu_quant_lowering_uint8(self):
        # [FUNCTION DOC] Chained dequant relu quant lowering test for uint8.
        self._test_dequant_relu_quant_dequant_relu_quant_lowering_helper(torch.uint8)

    @requires_vectorization
    def test_dequant_relu_quant_dequant_relu_quant_lowering_int8(self):
        # [FUNCTION DOC] Chained dequant relu quant lowering test for int8.
        self._test_dequant_relu_quant_dequant_relu_quant_lowering_helper(torch.int8)

    def test_inplace_add_alpha(self):
        # [FUNCTION DOC] Check in-place add with a scaling alpha through a traced graph compiled
        # with the inner compile function; compiled result in place on a tensor equals the eager
        # result.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: In-place add of y into x scaled by 0.55; returns x in a
            # tuple.
            aten.add_.Tensor(x, y, alpha=0.55)
            return (x,)

        x1 = torch.zeros(10)
        x2 = torch.zeros(10)
        x3 = torch.zeros(10)
        y = torch.randn(10)
        fn_fx = make_fx(fn)(x1, y)
        fn_compiled = compile_fx_inner(fn_fx, [x1, y])
        fn(x2, y)
        fn_compiled([x3, y])
        if not same(x2, x3):
            raise AssertionError("x2 and x3 are not the same")

    @parametrize("dtype", (torch.int8, torch.int16, torch.int32, torch.int64))
    def test_signed_int_overflow_wraps(self, dtype):
        # [FUNCTION DOC] Check integer overflow wraps for add, subtract and multiply at dtype
        # extremes across int8 to int64 and several SIMD lengths, comparing with eager on 67
        # elements.
        def add(x, y):
            # [FUNCTION DOC] HELPER: Add then convert to int64.
            return (x + y).to(torch.int64)

        def sub(x, y):
            # [FUNCTION DOC] HELPER: Subtract then convert to int64.
            return (x - y).to(torch.int64)

        def mul(x, y):
            # [FUNCTION DOC] HELPER: Multiply then convert to int64.
            return (x * y).to(torch.int64)

        info = torch.iinfo(dtype)
        n = 67
        two = torch.full((n,), 2, dtype=dtype)
        for fn, v in ((add, info.max), (sub, info.min), (mul, info.max)):
            x = torch.full((n,), v, dtype=dtype)
            for simdlen in simd_lengths_to_test():
                with config.patch({"cpp.simdlen": simdlen}):
                    torch._dynamo.reset()
                    metrics.reset()
                    self.common(fn, (x, two))

    def test_int_div(self):
        # [FUNCTION DOC] Check mixed integer shape arithmetic (half of size plus one) used to size
        # a ones tensor, and the size returned as an output.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Build ones of half the second dim size plus one, add y in
            # place, return it and the size.
            s3 = x.size(1)
            a = torch.ones((1 + s3) // 2)
            a += y
            return a, s3

        p0 = torch.randint(5, (1, 8))
        p1 = torch.randn(1)
        self.common(fn, (p0, p1))

    def test_no_op_squeeze(self):
        # [FUNCTION DOC] Check that squeeze of a dimension of size greater than one is a no-op and
        # matches eager.
        @torch.compile(backend="inductor")
        def forward(arg0_1):
            # [FUNCTION DOC] HELPER: Squeeze dim 1 of the input.
            return torch.ops.aten.squeeze.dim(arg0_1, 1)

        x = torch.randn((10, 20))
        self.common(forward, (x,))

    def test_parallel_num_threads(self):
        # [FUNCTION DOC] Check a compiled add gives correct results when run with one and with
        # four threads.
        @torch.compile(backend="inductor")
        def fn(x1, x2):
            # [FUNCTION DOC] HELPER: Add the two inputs.
            return x1 + x2

        x1 = torch.randn((10, 20))
        x2 = torch.randn((10, 20))
        with set_num_threads(1):
            if not same(x1 + x2, fn(x1, x2)):
                raise AssertionError("Results differ with 1 thread")
        with set_num_threads(4):
            if not same(x1 + x2, fn(x1, x2)):
                raise AssertionError("Results differ with 4 threads")

    @patch("torch.cuda.is_available", lambda: False)
    def test_timed_cpu_only(self):
        # [FUNCTION DOC] Check the timing utility works with a CPU-only function and CUDA reported
        # unavailable.
        timed(lambda: torch.randn(10), ())

    def test_vec_sve_armv9_arch_flags(self):
        # [FUNCTION DOC] Check SVE vector ISA flags: for 128 and 256 bit vectors, with SVE2 and
        # bf16 capabilities the architecture flags include sve2, bf16, i8mm and the vector bits
        # option; with SVE only they equal the base flags. Then with the platform mocked as
        # aarch64 it checks the first valid ISA is SVE with the matching bit width (for SVE2 off
        # and on) and NEON when no capabilities exist. Clears the valid ISA list cache before and
        # after (finally).
        for vector_bits in (128, 256):
            isa = cpu_vec_isa.VecSVE(vector_bits)

            # Armv9-A + SVE2 path should switch to the Armv9 flag set with bf16/i8mm
            with patch(
                "torch.cpu.get_capabilities",
                return_value={
                    "bf16": True,
                    "sve": True,
                    "sve2": True,
                    "sve_max_length": vector_bits,
                },
            ):
                isa._armv9a_supported = None
                flags = isa.build_arch_flags()
                self.assertIn("+sve2", flags)
                self.assertIn("+bf16", flags)
                self.assertIn("+i8mm", flags)
                self.assertIn(f"-msve-vector-bits={vector_bits}", flags)

            # SVE-only path should stick to the base flags
            with patch(
                "torch.cpu.get_capabilities",
                return_value={
                    "bf16": True,
                    "sve": True,
                    "sve2": False,
                    "sve_max_length": vector_bits,
                },
            ):
                isa._armv9a_supported = None
                flags = isa.build_arch_flags()
                self.assertEqual(flags, isa._arch_flags)

        try:
            for sve2 in (False, True):
                for vector_bits in (128, 256):
                    cpu_vec_isa.valid_vec_isa_list.cache_clear()
                    with (
                        patch("sys.platform", "linux"),
                        patch("platform.machine", return_value="aarch64"),
                        patch(
                            "torch.cpu.get_capabilities",
                            return_value={
                                "bf16": True,
                                "sve": True,
                                "sve2": sve2,
                                "sve_max_length": vector_bits,
                            },
                        ),
                    ):
                        selected_isa = cpu_vec_isa.valid_vec_isa_list()[0]
                        self.assertIsInstance(selected_isa, cpu_vec_isa.VecSVE)
                        self.assertEqual(selected_isa.bit_width(), vector_bits)

            cpu_vec_isa.valid_vec_isa_list.cache_clear()
            with (
                patch("sys.platform", "linux"),
                patch("platform.machine", return_value="aarch64"),
                patch("torch.cpu.get_capabilities", return_value={}),
            ):
                self.assertIsInstance(
                    cpu_vec_isa.valid_vec_isa_list()[0], cpu_vec_isa.VecNEON
                )
        finally:
            cpu_vec_isa.valid_vec_isa_list.cache_clear()

    @requires_vectorization
    def test_vec_dynamic_shapes(self):
        # [FUNCTION DOC] Check softmax over the last dim with dynamic default simdlen matches
        # eager. Requires vectorization.
        def fn(x):
            # [FUNCTION DOC] HELPER: Softmax over the last dim.
            return torch.softmax(x, -1)

        value = torch.randn((2, 10))
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (value,))

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(
        not cpu_vec_isa.valid_vec_isa_list()
        or "avx2" in [str(vec_isa) for vec_isa in cpu_vec_isa.valid_vec_isa_list()]
        or "asimd" in [str(vec_isa) for vec_isa in cpu_vec_isa.valid_vec_isa_list()],
        "Does not support vectorization or not s390x/ppc64le machine",
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_auto_zvec_vsx_simd(self):
        # [FUNCTION DOC] Check ISA selection on s390x/ppc64le (256-bit zvector/vsx): invalid
        # simdlen values give no ISA, 256 selects it, and the ATEN_CPU_CAPABILITY variable (avx2,
        # avx512, zvector, vsx select the ISA, default disables) is honored. Restores the
        # environment variable in a finally block. Skipped under fbcode or on machines lacking
        # such a ISA.
        vec_zvec_vsx = cpu_vec_isa.valid_vec_isa_list()[0]
        self.assertTrue(vec_zvec_vsx.bit_width() == 256)

        with config.patch({"cpp.simdlen": 0}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 1}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 257}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 256}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertTrue(isa == vec_zvec_vsx)

        pre_var = os.getenv("ATEN_CPU_CAPABILITY")
        if pre_var:
            os.environ.pop("ATEN_CPU_CAPABILITY")

        try:
            with config.patch({"cpp.simdlen": None}):
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "avx512"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "default"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertFalse(isa)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "zvector"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "vsx"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

        finally:
            if pre_var:
                os.environ["ATEN_CPU_CAPABILITY"] = pre_var
            elif os.getenv("ATEN_CPU_CAPABILITY"):
                os.environ.pop("ATEN_CPU_CAPABILITY")

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(
        platform.machine() != "x86_64" or not cpu_vec_isa.valid_vec_isa_list(),
        "Does not support vectorization or not x86_64 machine",
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_auto_simd(self):
        # [FUNCTION DOC] Check x86_64 ISA tables and selection: bit widths and element counts of
        # AMX, AVX512 VNNI, AVX512 and AVX2, invalid simdlens give no ISA, simdlen 512 and 256
        # pick the best valid ISA, and the ATEN_CPU_CAPABILITY variable restricts selection (avx2,
        # avx512, default, zvector, vsx). Restores the environment variable in a finally block.
        # Skipped under fbcode or off x86_64.
        vec_amx = cpu_vec_isa.supported_vec_isa_list[0]
        vec_avx512_vnni = cpu_vec_isa.supported_vec_isa_list[1]
        vec_avx512 = cpu_vec_isa.supported_vec_isa_list[2]
        vec_avx2 = cpu_vec_isa.supported_vec_isa_list[3]
        self.assertTrue(vec_amx.bit_width() == 512)
        self.assertTrue(vec_amx.nelements() == 16)
        self.assertTrue(vec_amx.nelements(torch.bfloat16) == 32)
        self.assertTrue(vec_avx512_vnni.bit_width() == 512)
        self.assertTrue(vec_avx512_vnni.nelements(torch.int8) == 64)
        self.assertTrue(vec_avx512_vnni.nelements(torch.uint8) == 64)
        self.assertTrue(vec_avx512.bit_width() == 512)
        self.assertTrue(vec_avx2.bit_width() == 256)
        self.assertTrue(vec_avx512.nelements() == 16)
        self.assertTrue(vec_avx2.nelements() == 8)
        self.assertTrue(vec_avx512.nelements(torch.bfloat16) == 32)
        self.assertTrue(vec_avx2.nelements(torch.bfloat16) == 16)

        with config.patch({"cpp.simdlen": 0}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 1}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 257}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 513}):
            isa_list = cpu_vec_isa.valid_vec_isa_list()
            if vec_avx512 in isa_list:
                self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 512}):
            isa_list = cpu_vec_isa.valid_vec_isa_list()
            isa = cpu_vec_isa.pick_vec_isa()
            if vec_amx in isa_list:
                self.assertTrue(isa == vec_amx)
            elif vec_avx512_vnni in isa_list:
                self.assertTrue(isa == vec_avx512_vnni)
            elif vec_avx512 in isa_list:
                self.assertTrue(isa == vec_avx512)

        with config.patch({"cpp.simdlen": 256}):
            isa_list = cpu_vec_isa.valid_vec_isa_list()
            if vec_avx2 in isa_list:
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_avx2)

        pre_var = os.getenv("ATEN_CPU_CAPABILITY")
        if pre_var:
            os.environ.pop("ATEN_CPU_CAPABILITY")

        try:
            with config.patch({"cpp.simdlen": None}):
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_amx)
                elif vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512_vnni)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512)
                else:
                    self.assertTrue(isa == vec_avx2)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx2)
                if vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx2)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx2)
                elif vec_avx2 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx2)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "avx512"
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_amx)
                elif vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512_vnni)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512)
                else:
                    self.assertTrue(isa == vec_avx2)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "default"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertFalse(isa)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "zvector"
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_amx)
                elif vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512_vnni)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512)
                else:
                    self.assertTrue(isa == vec_avx2)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "vsx"
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_amx)
                elif vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512_vnni)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512)
                else:
                    self.assertTrue(isa == vec_avx2)

        finally:
            if pre_var:
                os.environ["ATEN_CPU_CAPABILITY"] = pre_var
            elif os.getenv("ATEN_CPU_CAPABILITY"):
                os.environ.pop("ATEN_CPU_CAPABILITY")

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_masked_fill_softmax(self):
        # [FUNCTION DOC] Check masked fill followed by softmax for each vectorizable dtype, with
        # cpp wrapper on and off; raises AssertionError if no vector kernel was generated.
        # Requires vectorization.
        def fn(value, mask):
            # [FUNCTION DOC] HELPER: Convert the mask to boolean, fill masked positions with -33,
            # softmax on the last dim.
            mask = mask.to(torch.bool)
            x = torch.masked_fill(value, mask, -33.0)
            return torch.softmax(x, -1)

        for dtype in vec_dtypes:
            value = torch.randn((2, 17), dtype=dtype)
            mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8)
            with config.patch({"cpp.simdlen": None}):
                for cpp_wrapper_flag in [True, False]:
                    with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                        torch._dynamo.reset()
                        metrics.reset()
                        self.common(fn, (value, mask))
                        if metrics.generated_cpp_vec_kernel_count < 1:
                            raise AssertionError(
                                f"Expected >= 1 vec kernels, got {metrics.generated_cpp_vec_kernel_count}"
                            )

    def test_channels_last_view_as_complex(self):
        # [FUNCTION DOC] Regression test (issue 122448): view_as_complex of a stacked pair of
        # float tensors compiles with full graph and matches eager.
        # https://github.com/pytorch/pytorch/issues/122448#issuecomment-2046169554

        def reduce_example(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Stack the float-converted inputs on the last dim and view as
            # complex.
            """Applies the rotary embedding to the query and key tensors."""
            x_out = torch.view_as_complex(torch.stack([x.float(), y.float()], dim=-1))
            return x_out

        args = [torch.randn(1, 1, 1, 128), torch.randn(1, 1, 1, 128)]
        expected = reduce_example(*args)
        actual = torch.compile(reduce_example, fullgraph=True)(*args)
        self.assertEqual(expected, actual)

    def test_load_same_bool_tensor_twice(self):
        # [FUNCTION DOC] Check loading the same boolean mask twice inside one kernel compiles and
        # runs.
        @torch.compile(backend="inductor")
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Two masked_fill calls on a with mask b; returns both.
            x = torch.masked_fill(a, b, -33.0)
            y = torch.masked_fill(a, b, -33.0)
            return x, y

        value = torch.randn((2, 17))
        mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8).to(torch.bool)
        fn(value, mask)

    def test_cpu_vec_cosim(self):
        # [FUNCTION DOC] Co-simulation check: every static method override in the scalar C++
        # overrides class must also exist in the vectorized overrides class, except for a fixed
        # allow list of ops (special functions, polynomials, random, logical and bitwise ops,
        # etc.). Fails listing the unexpected ones.
        cpp_vec_op_list = []
        cpp_op_list = []

        for k, v in CppVecOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                cpp_vec_op_list.append(k)
        for k, v in CppOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                cpp_op_list.append(k)

        diff = [
            "airy_ai",
            "bessel_j0",
            "bessel_j1",
            "bessel_y0",
            "bessel_y1",
            "modified_bessel_i0",
            "modified_bessel_i1",
            "modified_bessel_k0",
            "modified_bessel_k1",
            "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1",
            "spherical_bessel_j0",
            "i1",
            "i1e",
            "ndtr",
            "ndtri",
            "log_ndtr",
            "erfcx",
            "gammainc",
            "gammaincc",
            "igamma",
            "igammac",
            "polygamma",
            "zeta",
            "shifted_chebyshev_polynomial_u",
            "chebyshev_polynomial_u",
            "chebyshev_polynomial_t",
            "shifted_chebyshev_polynomial_w",
            "chebyshev_polynomial_w",
            "shifted_chebyshev_polynomial_t",
            "chebyshev_polynomial_v",
            "shifted_chebyshev_polynomial_v",
            "hermite_polynomial_he",
            "laguerre_polynomial_l",
            "hermite_polynomial_h",
            "legendre_polynomial_p",
            "constant",
            "index_expr",
            "signbit",
            "isinf",
            "frexp",
            "mod",
            "masked",
            "randn",
            "isnan",
            "rand",
            "randint64",
            "logical_and",
            "logical_not",
            "logical_or",
            "logical_xor",
            "bitwise_and",
            "bitwise_left_shift",
            "bitwise_not",
            "bitwise_right_shift",
            "bitwise_or",
            "bitwise_xor",
            "to_dtype_bitcast",
        ]
        union = {*cpp_vec_op_list, *diff}
        self.assertTrue(
            set(cpp_op_list).issubset(union),
            lambda msg: f"{msg}\nunexpected: {set(cpp_op_list) - union}",
        )

    def test_atomic_add_lowp_fp(self):
        # [FUNCTION DOC] Check atomic add codegen for gather backward in bf16 and fp16: forward
        # result and the input gradient from a backward pass with identically seeded gradient
        # match eager.
        def fn(test_args):
            # [FUNCTION DOC] HELPER: Call gather with the keyword argument mapping given.
            res = torch.gather(**test_args)
            return res

        for dtype in _lowp_fp_dtypes:
            input_tensor_for_ref = torch.tensor(
                [[3.0, -5.0]], dtype=dtype, requires_grad=True
            )
            input_tensor_for_opt = torch.tensor(
                [[3.0, -5.0]], dtype=dtype, requires_grad=True
            )

            test_args_for_ref = {
                "input": input_tensor_for_ref,
                "dim": 1,
                "index": torch.tensor([[1]]),
            }
            test_args_for_opt = {
                "input": input_tensor_for_opt,
                "dim": 1,
                "index": torch.tensor([[1]]),
            }

            opt_fn = torch.compile(fn)

            ref_fwd = fn(test_args_for_ref)
            res_fwd = opt_fn(test_args_for_opt)
            self.assertEqual(res_fwd, ref_fwd)

            torch.manual_seed(1)
            bwd_tensor_for_ref = torch.randn(ref_fwd.shape, dtype=dtype)
            torch.manual_seed(1)
            bwd_tensor_for_opt = torch.randn(res_fwd.shape, dtype=dtype)
            self.assertEqual(bwd_tensor_for_ref, bwd_tensor_for_opt)

            ref_fwd.backward(bwd_tensor_for_ref)
            res_fwd.backward(bwd_tensor_for_opt)

            ref_grad = test_args_for_ref["input"].grad
            res_grad = test_args_for_opt["input"].grad
            self.assertEqual(ref_grad, res_grad)

    def test_meta_device(self):
        # [FUNCTION DOC] Check an empty tensor on the meta device compiles with full graph and has
        # the expected shape.
        @torch.compile(fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Create a meta float16 empty tensor, then sine plus one.
            x = torch.ops.aten.empty.memory_format(
                [1024, 128, 128],
                dtype=torch.float16,
                device="meta",
                pin_memory=False,
            )
            return x.sin() + 1

        self.assertEqual(fn().shape, [1024, 128, 128])

    def test_decomposed_fake_quant_per_channel(self):
        # [FUNCTION DOC] Compare the fake quant per channel affine op, its decomposed counterpart
        # in eager and the compiled decomposed version in forward results and input gradients
        # (including zero gradient for a clipped out-of-range element), and expect two vector
        # kernels.
        def fq(input, scales, zero_points, axis, quant_min, quant_max):
            # [FUNCTION DOC] HELPER: Eager fake quantize per channel affine op.
            res = torch.fake_quantize_per_channel_affine(
                input, scales, zero_points, axis, quant_min, quant_max
            )
            return res

        def qdq(input, scales, zero_points, axis, quant_min, quant_max):
            # [FUNCTION DOC] HELPER: Decomposed fake quant per channel op.
            res = torch.ops.quantized_decomposed.fake_quant_per_channel(
                input, scales, zero_points, axis, quant_min, quant_max
            )
            return res

        def run_eager_aten_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        ):
            # [FUNCTION DOC] HELPER: Clear the grad, run the ATen fake quant, backpropagate the
            # sum, return result and grad.
            input.grad = None
            res = fq(input, scales, zero_points, axis, quant_min, quant_max)
            res.sum().backward()
            return res, input.grad

        def run_eager_decomposed_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        ):
            # [FUNCTION DOC] HELPER: Same as the ATen runner but using the decomposed fake quant
            # op in eager.
            input.grad = None
            res = qdq(input, scales, zero_points, axis, quant_min, quant_max)
            res.sum().backward()
            return res, input.grad

        def run_compile_decomposed_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        ):
            # [FUNCTION DOC] HELPER: Same runner but compiles the decomposed fake quant function
            # first.
            input.grad = None
            compiled_qdq = torch.compile(qdq)
            res = compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max)
            res.sum().backward()
            return res, input.grad

        input = torch.randn(2, 3, 224, 224)
        input[1, 2, 3, 4] = 257
        input.requires_grad_()
        scales = torch.ones((3,))
        zero_points = torch.zeros((3,))
        axis = 1
        quant_min = -128
        quant_max = 127

        aten_input = copy.deepcopy(input)
        compiler_input = copy.deepcopy(input)

        res_aten_eager, input_grad_aten_eager = run_eager_aten_fake_quant(
            aten_input, scales, zero_points, axis, quant_min, quant_max
        )
        res_decomp_eager, input_grad_decomp_eager = run_eager_decomposed_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        )
        res, input_grad = run_compile_decomposed_fake_quant(
            compiler_input, scales, zero_points, axis, quant_min, quant_max
        )

        self.assertEqual(res_aten_eager, res)
        self.assertEqual(res_decomp_eager, res)
        self.assertEqual(input_grad_aten_eager, input_grad)
        self.assertEqual(input_grad_decomp_eager, input_grad)
        self.assertEqual(input_grad[1, 2, 3, 4], torch.tensor(0.0))
        # For forward and backward kernel
        check_metrics_vec_kernel_count(2)

    @requires_vectorization
    def test_ops_masked_with_bool_input(self):
        # [FUNCTION DOC] Check the masked op with a boolean input via constant padding of a
        # boolean tensor, compiled vs eager, with one vector kernel. Requires vectorization.
        x = torch.zeros(129, dtype=torch.bool)
        size = [2, 3]
        res_aten_eager = torch.constant_pad_nd(x, size)
        cfn = torch.compile(torch.constant_pad_nd)
        res = cfn(x, size)
        self.assertEqual(res_aten_eager, res)
        check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_frexp(self):
        # [FUNCTION DOC] Check frexp followed by multiply of mantissa and exponent matches eager
        # with one vector kernel. Requires vectorization.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiply the mantissa and exponent from frexp.
            x_frac, x_exp = torch.frexp(x)  # x_frac: int32, x_exp: float32
            x = x_frac * x_exp
            return x

        x = torch.randn(64, 1)
        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_bitwise_right_shift(self):
        # [FUNCTION DOC] Check right shift of negative int64 values by 31 in compiled equals
        # eager.
        x = torch.randint(-1, 0, (1, 1, 1), device="cpu", dtype=torch.int64)
        bit_num = 31
        res_aten_eager = torch.bitwise_right_shift(x, bit_num)
        cfn = torch.compile(torch.bitwise_right_shift)
        res = cfn(x, bit_num)
        self.assertEqual(res_aten_eager, res)

    def test_bitwise_shift_corner_inputs(self):
        # [FUNCTION DOC] Regression test (issues 143555 and 143566): left and right shift by 64
        # bits of an int64 scalar tensor compiled equals eager.
        # Fix https://github.com/pytorch/pytorch/issues/143555
        # and https://github.com/pytorch/pytorch/issues/143566
        bitwise_fns = (
            torch.bitwise_left_shift,
            torch.bitwise_right_shift,
        )
        for bitwise_fn in bitwise_fns:
            torch._dynamo.reset()
            metrics.reset()
            x = torch.tensor(1000, dtype=torch.int64)
            bit_num = torch.tensor(64, dtype=torch.int64)
            res_aten_eager = bitwise_fn(x, bit_num)
            cfn = torch.compile(bitwise_fn)
            res = cfn(x, bit_num)
            self.assertEqual(res_aten_eager, res)

    def test_view_dtype(self):
        # [FUNCTION DOC] Check view of float32 as int32 followed by right shift equals eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Reinterpret as int32 then shift right by two.
            return x.view(torch.int32) >> 2

        input = torch.ones(16, 16)
        res_aten_eager = f(input)
        cfn = torch.compile(f)
        res = cfn(input)
        self.assertEqual(res_aten_eager, res)

    @patch("torch.cuda.is_available", lambda: False)
    def test_scatter_using_atomic_add(self):
        # [FUNCTION DOC] Check scatter with add reduction: with the scatter-reduce fallback off,
        # code uses atomic add; with it on, code uses the scatter_reduce fallback op
        # (wrapper-specific name). When OpenMP is the parallel backend, more checks follow.
        def fn(a, dim, index, b):
            # [FUNCTION DOC] HELPER: Scatter with add reduction.
            return aten.scatter(a, dim, index, b, reduce="add")

        inps = (
            torch.randn(5, 29, 13),
            2,
            torch.tensor([[[3, 5, 7, 9]]]),
            torch.randn(1, 1, 10),
        )

        def _internal_check(
            _fn,
            _inps,
            _target_code_check=None,
            _target_code_check_not=None,
        ):
            # [FUNCTION DOC] HELPER: Compile a function, run it, check the generated code contains
            # (or does not contain, and has output) the given strings, and check the compiled
            # result equals eager. Note it uses the enclosing inputs when running the code capture
            # rather than the passed inputs.
            # KNOWN ISSUE: the code capture uses the enclosing inps variable instead of the
            # inputs parameter, so the passed inputs are ignored by the capture.
            torch._dynamo.reset()
            metrics.reset()
            _fn_opt = torch.compile()(_fn)
            _, code = run_and_get_cpp_code(_fn_opt, *inps)
            if _target_code_check:
                FileCheck().check(_target_code_check).run(code)
            if _target_code_check_not:
                FileCheck().check_not(_target_code_check_not).run(code)
                # Verify that the output isn't empty
                FileCheck().check("Output code:").run(code)

            self.assertEqual(
                _fn(*_inps),
                _fn_opt(*_inps),
            )

        with config.patch({"cpp.fallback_scatter_reduce_sum": False}):
            _internal_check(fn, inps, "atomic_add")

        scatter_reduce_func = (
            "aoti_torch_cpu_scatter_reduce_"
            if config.cpp_wrapper
            else "aten.scatter_reduce_"
        )
        with config.patch({"cpp.fallback_scatter_reduce_sum": True}):
            _internal_check(fn, inps, scatter_reduce_func)

        if "ATen parallel backend: OpenMP" in torch.__config__.parallel_info():
            with set_num_threads(1):
                # When running with a single thread, we expect the aten.scatter will go
                # into the cpp backend codegen instead of a fallback to aten.scatter_reduce_.
                # Avoid the inductor cache so we don't serve an entry compiled above.
                with config.patch(
                    {"fx_graph_cache": False, "fx_graph_remote_cache": False}
                ):
                    _internal_check(
                        fn, inps, _target_code_check_not=scatter_reduce_func
                    )

            with config.patch({"cpp.dynamic_threads": True}), set_num_threads(1):
                _internal_check(fn, inps, scatter_reduce_func)

    @patch("torch.cuda.is_available", lambda: False)
    @requires_vectorization
    @torch._inductor.config.patch({"cpp.fallback_scatter_reduce_sum": False})
    def test_scatter_using_atomic_add_vec(self):
        # [FUNCTION DOC] Check scatter with add reduction is lowered to vectorized atomic add (two
        # vector kernels) with the scatter-reduce fallback disabled, in both default threading and
        # a single-thread run with graph caches turned off. CUDA reported unavailable; requires
        # vectorization.
        def fn(a, dim, index, b):
            # [FUNCTION DOC] HELPER: Scatter with add reduction.
            return aten.scatter(a, dim, index, b, reduce="add")

        inps = (
            torch.zeros(1, 1, 25),
            2,
            torch.tensor([[[3, 5, 7, 9] * 5]]),
            torch.ones(1, 1, 25),
        )
        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, inps)
        if metrics.generated_cpp_vec_kernel_count != 2:
            raise AssertionError(
                f"Expected 2 vec kernels, got {metrics.generated_cpp_vec_kernel_count}"
            )

        with (
            set_num_threads(1),
            config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False}),
        ):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, inps)
            if metrics.generated_cpp_vec_kernel_count != 2:
                raise AssertionError(
                    f"Expected 2 vec kernels, got {metrics.generated_cpp_vec_kernel_count}"
                )

    @requires_vectorization
    def test_index_put_accumulate_atomic_add_vec_scalar_index(self):
        # [FUNCTION DOC] Check index_put with accumulation using broadcast index tensors on a
        # small output generates a vectorized atomic add (code contains the helper name), matches
        # eager and produces a vector kernel.
        def fn(values, idx0, idx1):
            # [FUNCTION DOC] HELPER: Accumulate values into a zeros tensor at the two
            # last-dimension index tensors.
            out = torch.zeros(1, 32, 1, 1)
            return aten.index_put(out, [None, None, idx0, idx1], values, True)

        inps = (
            torch.ones(1, 32, 10, 20),
            torch.zeros(10, 1, dtype=torch.long),
            torch.zeros(20, dtype=torch.long),
        )

        with set_num_threads(2):
            torch._dynamo.reset()
            metrics.reset()
            opt_fn = torch.compile(fn, backend="inductor")
            actual, code = run_and_get_cpp_code(opt_fn, *inps)
        FileCheck().check("atomic_add_vec").run(code)
        self.assertEqual(fn(*inps), actual)
        self.assertGreater(metrics.generated_cpp_vec_kernel_count, 0)

    def test_large_mean(self):
        # [FUNCTION DOC] Check mean of a very large float tensor matches eager both with default
        # threads and one thread.
        size = (30000, 100000)
        t = torch.rand(size, dtype=torch.float)
        op = torch.mean
        expected = op(t)
        actual = torch.compile(op)(t)
        self.assertEqual(expected, actual)
        with set_num_threads(1):
            expected = op(t)
            actual = torch.compile(op)(t)
            self.assertEqual(expected, actual)

    def test_outer_mean_large_size(self):
        # [FUNCTION DOC] Check mean over an outer product of a flattened tensor and ones, for a
        # large size, within loose tolerance.
        def fn(x):
            # [FUNCTION DOC] HELPER: Flatten, outer product with ones, mean along dim 1.
            x = x.flatten()
            x_one = torch.ones_like(x)
            x = torch.outer(x, x_one)
            return torch.mean(x, dim=1)

        x = torch.randn(2, 2, 64, 64)
        expected = fn(x)
        actual = torch.compile(fn)(x)
        self.assertEqual(expected, actual, atol=1e-4, rtol=1e-4)

    def test_two_step_variance(self):
        # [FUNCTION DOC] Check layer norm variance accuracy over inputs with extreme mean/std
        # combos (including 1e10); compiled output must be close to eager with relaxed tolerance
        # for extreme cases.
        M = 64
        N = 1024

        class L(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a layer norm of the given normalized shape and
            # epsilon.
            def __init__(self, normalized_shape=N, eps=1e-5):
                # [FUNCTION DOC] HELPER: Create the layer norm.
                super().__init__()
                self.layernorm = torch.nn.LayerNorm(normalized_shape, eps=eps)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the layer norm.
                return self.layernorm(x)

        mod = L().eval()
        for mean, std in [
            (0, 1e10),
            (1e10, 10),
            (1e10, 1e10),
            (0, 1),
            (0, 0.5),
            (0.5, 1),
            (0, 2),
        ]:
            x = torch.randn(M, N)
            row_means = x.mean(dim=1, keepdim=True)
            row_stds = x.std(dim=1, keepdim=True, unbiased=True)
            x_norm = (x - row_means) / (row_stds + 1e-5)
            x = x_norm * std + mean
            input = (x,)
            output_eager = mod(*input)
            with torch.no_grad():
                m = torch.compile(mod)
                output_compiled = m(*input)
            if mean == 1e10 or std == 1e10:
                self.assertTrue(
                    torch.allclose(output_eager, output_compiled, atol=1, rtol=1e-4)
                )
            else:
                self.assertTrue(
                    torch.allclose(output_eager, output_compiled, atol=1e-4, rtol=1e-4)
                )

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_new_vec_op_cpu_only(self):
        # [FUNCTION DOC] Check vectorized log1p, expm1 and erf chain on inputs containing NaN for
        # each vectorizable dtype, with cpp wrapper on and off, expecting one vector kernel.
        # Skipped under fbcode.
        def fn(x):
            # [FUNCTION DOC] HELPER: Apply erf, expm1 then log1p.
            return torch.log1p(torch.expm1(torch.erf(x)))

        for dtype in vec_dtypes:
            torch.manual_seed(0)
            x = torch.randn((2, 9), dtype=dtype)
            x[0, 0] = torch.nan
            x[1, -1] = torch.nan

            with config.patch({"cpp.simdlen": None}):
                for cpp_wrapper_flag in [True, False]:
                    with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                        torch._dynamo.reset()
                        metrics.reset()
                        self.common(fn, (x,))
                        check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_expm1_small_values(self):
        # [FUNCTION DOC] Check vectorized expm1 on tiny values keeps precision (tight tolerance)
        # and yields one vector kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Apply expm1.
            return torch.expm1(x)

        x = torch.linspace(-1e-6, 1e-6, 1024)
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,), atol=1e-9, rtol=1e-6)
            check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_cpu_only_for_all_available_isa(self):
        # [FUNCTION DOC] Check erf, cos, sin chain with NaN inputs under every valid ISA bit width
        # (and default), each giving one vector kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Apply erf, cos, then sin.
            return torch.sin(torch.cos(torch.erf(x)))

        x = torch.randn((2, 9))
        x[0, 0] = torch.nan
        x[1, -1] = torch.nan

        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()] + [
            None
        ]
        for item in bit_widths:
            with config.patch({"cpp.simdlen": item}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    @slowTest
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch("cpp.enable_tiling_heuristics", False)
    def test__adaptive_avg_pool2d(self):
        # [FUNCTION DOC] Slow test: adaptive average pool 2D in channels-last across input and
        # output sizes (16, 65), valid SIMD widths and vectorizable dtypes with tiling heuristics
        # off; expects one vector kernel.
        def wrap_fn(oh, ow):
            # [FUNCTION DOC] HELPER: Build a function that adaptive-average-pools to the given
            # output size.
            def fn(x):
                # [FUNCTION DOC] HELPER: Adaptive average pool to the captured output height and
                # width.
                return torch._adaptive_avg_pool2d(x, (oh, ow))

            return fn

        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()]
        ih = [16, 65]
        iw = ih
        oh = ih
        ow = ih
        for _ih, _iw, _oh, _ow, _simd_len, dtype in itertools.product(
            ih, iw, oh, ow, bit_widths, vec_dtypes
        ):
            x = torch.randn(2, 3, _ih, _iw, dtype=dtype).to(
                memory_format=torch.channels_last
            )
            _fn = wrap_fn(_oh, _ow)
            with config.patch({"cpp.simdlen": _simd_len}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(_fn, (x,))
                check_metrics_vec_kernel_count(1)

    def test_adaptive_avg_pool2d_dynamic_input_output_sizes(self):
        # [FUNCTION DOC] Check adaptive average pooling with dynamic input and output sizes
        # compiles once (frame count 1) for several output sizes, still works for new shapes, and
        # works separately for a large window.
        def fn(x, out_h, out_w):
            # [FUNCTION DOC] HELPER: Adaptive average pool to the given height and width.
            return torch._adaptive_avg_pool2d(x, [out_h, out_w])

        cnt = CompileCounterWithBackend("inductor")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True, dynamic=True)

        x = torch.randn(2, 3, 19, 11)
        for out_h, out_w in [(20, 13), (21, 14), (22, 15)]:
            self.assertEqual(opt_fn(x, out_h, out_w), fn(x, out_h, out_w))
        self.assertEqual(cnt.frame_count, 1)

        for shape, output_size in [
            ((2, 3, 23, 15), (24, 17)),
            ((2, 3, 17, 9), (18, 11)),
        ]:
            x = torch.randn(shape)
            out_h, out_w = output_size
            self.assertEqual(opt_fn(x, out_h, out_w), fn(x, out_h, out_w))

        large_window_cnt = CompileCounterWithBackend("inductor")
        large_window_opt_fn = torch.compile(
            fn, backend=large_window_cnt, fullgraph=True, dynamic=True
        )
        x = torch.randn(2, 3, 21, 21)
        self.assertEqual(large_window_opt_fn(x, 4, 4), fn(x, 4, 4))
        self.assertEqual(large_window_cnt.frame_count, 1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_logical(self):
        # [FUNCTION DOC] Check vectorized logical and, not, or, xor, each wrapped by selecting 1
        # or 0, for each vectorizable dtype, with one vector kernel.
        def wrap_fn1(op: Callable):
            # [FUNCTION DOC] HELPER: Build a unary function mapping the logical op result to 1.0
            # or 0.0.
            def fn(x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Apply the logical op then select 1.0 or 0.0.
                return torch.where(op(x), 1.0, 0.0)

            return fn

        def wrap_fn2(op: Callable):
            # [FUNCTION DOC] HELPER: Build a binary function mapping the logical op result to 1.0
            # or 0.0.
            def fn(x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Apply the binary logical op then select 1.0 or 0.0.
                return torch.where(op(x, y), 1.0, 0.0)

            return fn

        for dtype in vec_dtypes:
            x = torch.randn(64, dtype=dtype)
            y = torch.randn(64, dtype=dtype)
            logical_fns = [
                torch.logical_and,
                torch.logical_not,
                torch.logical_or,
                torch.logical_xor,
            ]
            for logical_fn in logical_fns:
                torch._dynamo.reset()
                metrics.reset()
                if logical_fn == torch.logical_not:
                    _fn = wrap_fn1(logical_fn)
                    _args = (x,)
                else:
                    _fn = wrap_fn2(logical_fn)
                    _args = (x, y)
                self.common(_fn, _args)
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_vec_bitwise(self):
        # [FUNCTION DOC] Check vectorized bitwise and, not, or, xor and shifts for bool and
        # integer dtypes, skipping shifts for bool (unsupported in eager); each gives one vector
        # kernel.
        for dtype in [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
        ]:
            x = torch.randn(64, dtype=torch.float32)
            y = torch.randn(64, dtype=torch.float32)
            if dtype == torch.bool:
                x = x > 0
                y = y > 0
            else:
                x = x.to(dtype)
                y = y.to(dtype)
            bitwise_fns = [
                torch.bitwise_and,
                torch.bitwise_not,
                torch.bitwise_or,
                torch.bitwise_xor,
                torch.bitwise_left_shift,
                torch.bitwise_right_shift,
            ]
            for bitwise_fn in bitwise_fns:
                if (
                    bitwise_fn
                    in [
                        torch.bitwise_left_shift,
                        torch.bitwise_right_shift,
                    ]
                    and dtype == torch.bool
                ):
                    # Eager doesn't support bool
                    # https://pytorch.org/docs/stable/generated/torch.bitwise_left_shift.html
                    continue
                torch._dynamo.reset()
                metrics.reset()
                if bitwise_fn == torch.bitwise_not:
                    _args = (x,)
                else:
                    _args = (x, y)
                self.common(bitwise_fn, _args)
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_vec_randn(self):
        # [FUNCTION DOC] Check vectorized random generation (normal, uniform, integer) for float
        # and int dtypes: results are deterministic with a manual seed, one vector kernel is
        # produced, and the vector result equals the scalar result (simdlen 0). Graph caches are
        # disabled.
        funcs = [torch.randn, torch.rand, torch.randint]
        float_dtypes = [
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ]
        int_dtypes = [
            torch.int8,
            torch.uint8,
            torch.int32,
            torch.int64,
        ]
        dtypes = float_dtypes + int_dtypes
        for rand_func, dtype in itertools.product(funcs, dtypes):
            if (rand_func == torch.randint and dtype not in int_dtypes) or (
                rand_func != torch.randint and dtype not in float_dtypes
            ):
                # Skip the invalid combination
                continue
            with config.patch(
                {"fx_graph_cache": False, "fx_graph_remote_cache": False}
            ):
                torch._dynamo.reset()
                metrics.reset()

                def func(seed):
                    # [FUNCTION DOC] HELPER: Seed the generator, then draw 64 values using the
                    # current random function and dtype.
                    torch.manual_seed(seed)
                    if rand_func == torch.randint:
                        return rand_func(0, 64, (64,), dtype=dtype)
                    else:
                        return rand_func(64, dtype=dtype)

                cfn = torch.compile(func)
                # Check the result is deterministic with mauanl seed
                self.assertEqual(cfn(2024), cfn(2024))
                check_metrics_vec_kernel_count(1)
                res_vec = cfn(2024)

                torch._dynamo.reset()
                metrics.reset()
                with config.patch({"cpp.simdlen": 0}):
                    res_scalar = torch.compile(func)(2024)
                    # Check the same result between scalar and vec
                    self.assertEqual(res_vec, res_scalar)

    @requires_vectorization
    def test_bitwise_logical_op_bool(self):
        # [FUNCTION DOC] Check bitwise and logical and/or/xor on boolean comparison results of
        # int64 and uint8 inputs matches eager.
        bitwise_fns = [
            torch.bitwise_and,
            torch.bitwise_or,
            torch.bitwise_xor,
            torch.logical_and,
            torch.logical_or,
            torch.logical_xor,
        ]

        for bitwise_fn in bitwise_fns:

            def fn(a, b):
                # [FUNCTION DOC] HELPER: Apply the loop's bitwise function to booleans from
                # comparing both inputs to one.
                c = bitwise_fn((a > 1), (b > 1))
                return c

            a = torch.ones((64), dtype=torch.int64)
            b = torch.ones((64), dtype=torch.uint8)

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (a, b))

    def test_torch_logit(self):
        # [FUNCTION DOC] Regression test (issue 145379): logit with a tensor epsilon as the second
        # positional argument.
        # fix https://github.com/pytorch/pytorch/issues/145379
        def fn(*args):
            # [FUNCTION DOC] HELPER: Logit of the first argument with the second as epsilon.
            return torch.logit(args[0], args[1])

        input = torch.tensor(0.3, dtype=torch.float64)
        eps = torch.tensor(0.9, dtype=torch.float64)
        self.common(fn, (input, eps))

    @slowTest
    # Pure CPU vec codegen; running it on a GPU CI runner leaves the GPU idle
    # for the ~1h this test takes.
    @unittest.skipIf(IS_CI and TEST_CUDA, "CPU-only test, skip on GPU runners")
    @requires_vectorization
    def test_vec_compare_op_cpu_only(self):
        # [FUNCTION DOC] Slow, CPU-only test (skipped on GPU CI runners): a long chain of
        # comparison ops and selects for each vectorizable dtype; expects one vector kernel and
        # that all generated kernels are vector.
        def fn(x):
            # [FUNCTION DOC] HELPER: Chain of equality, inequality and ordering comparisons, each
            # followed by a select that negates or shifts the value.
            y1 = torch.eq(x, 1.0)
            x = torch.where(y1, x, -x)
            y2 = torch.ne(x, 0.0)
            x = torch.where(y2, x, -x)
            y3 = torch.lt(x, 5.0)
            x = torch.where(y3, x, x - 1.0)
            y4 = torch.gt(x, -2.0)
            x = torch.where(y4, x, x + 1.0)
            y5 = torch.le(x, 8.0)
            x = torch.where(y5, x, x - 1.0)
            y6 = torch.ge(x, -3.0)
            x = torch.where(y6, x, x + 1.0)
            y7 = x == 1.0
            x = torch.where(y7, x, -x)
            y8 = x != 0.0
            x = torch.where(y8, x, -x)
            y9 = x < 5.0
            x = torch.where(y9, x, x - 1.0)
            y10 = x > -2.0
            x = torch.where(y10, x, x + 1.0)
            y11 = x <= 8.0
            x = torch.where(y11, x, x - 1.0)
            y12 = x >= -3.0
            x = torch.where(y12, x, x + 1.0)
            return x

        for dtype in vec_dtypes:
            x = torch.randn((2, 9), dtype=dtype)

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)
                if (
                    metrics.generated_kernel_count
                    - metrics.generated_cpp_vec_kernel_count
                ) != 0:
                    raise AssertionError(
                        f"Expected 0, got {metrics.generated_kernel_count - metrics.generated_cpp_vec_kernel_count}"
                    )

    @requires_vectorization
    def test_vec_remainder(self):
        # [FUNCTION DOC] Check vectorized remainder for ints and floats, avoiding zero divisors
        # for integers; one vector kernel expected.
        for dtype in [
            torch.int8,
            torch.uint8,
            torch.int32,
            torch.int64,
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float64,
        ]:
            if is_float_dtype(dtype):
                x = torch.randn(64, dtype=dtype)
                y = torch.randn(64, dtype=dtype)
            else:
                lower = 1 if dtype == torch.uint8 else -100
                x = torch.randint(lower, 100, (64,), dtype=dtype)
                y = torch.randint(lower, 100, (64,), dtype=dtype)
                y = torch.where(
                    y == torch.zeros_like(y),
                    torch.ones_like(y),
                    y,
                )

            torch._dynamo.reset()
            metrics.reset()
            _args = (x, y)
            self.common(torch.remainder, _args)
            check_metrics_vec_kernel_count(1)

    def test_skip_cpp_codegen(self):
        # [FUNCTION DOC] Check the option that disables C++ codegen: with the cpp wrapper it must
        # raise InductorError; otherwise code has no generated kernel, results match eager,
        # constants propagate through a fallback slice, and a module taking a min on an empty
        # dimension then random_like works.
        with config.patch({"disable_cpp_codegen": True}):
            inps = torch.ones([20]), torch.rand([20])

            def f(x, y):
                # [FUNCTION DOC] HELPER: Add x, y and a constant tensor.
                return x + y + torch.tensor(1)

            f_opt = torch.compile(f)
            if config.cpp_wrapper:
                # cpp_wrapper on CPU doesn't work without CPP codegen
                with self.assertRaises(InductorError):
                    f_opt(*inps)
                return

            _, code = run_and_get_cpp_code(f_opt, inps[0], inps[1])
            FileCheck().check_not("void kernel").run(code)

            self.assertEqual(
                f(*inps),
                f_opt(*inps),
            )

            # constant needs to be propagated on fallback
            def f(x):
                # [FUNCTION DOC] HELPER: Slice x starting at a tensor-valued index then double.
                return x[torch.tensor(1) :] * 2

            f_opt = torch.compile()(f)
            _, code = run_and_get_cpp_code(f_opt, inps[0])
            FileCheck().check_not("void kernel").run(code)
            self.assertEqual(f_opt(inps[0]), f(inps[0]))

            class Model(torch.nn.Module):
                # [CLASS DOC] HELPER: Module taking min along dim 1 and returning random values
                # shaped like it.
                def __init__(
                    self,
                ):
                    # [FUNCTION DOC] HELPER: Create the module (no state).
                    super().__init__()

                def forward(self, v1: torch.Tensor):
                    # [FUNCTION DOC] HELPER: Min along dim 1, then random_like of those values.
                    vx = v1.min(dim=1).values
                    v2 = torch.randn_like(vx)
                    return v2

            model = Model()
            x = torch.rand(10, 3, 0)
            model_f = torch.compile()(model)

            self.assertEqual(model(x), model_f(x))

    def test_redundant_to_node_elimination_lowp_fp(self):
        # [FUNCTION DOC] Check a low-precision add and mean avoids redundant dtype-conversion
        # nodes: compiled equals eager and one vector kernel, with trace enabled and not.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add then mean.
            res = x + y
            res = torch.mean(res)
            return res

        for dtype in _lowp_fp_dtypes:
            x = torch.randn((2, 9), dtype=dtype)
            y = torch.randn((2, 9), dtype=dtype)

            for torch_compile_debug in [True, False]:
                with config.patch(
                    {"trace.enabled": torch_compile_debug, "cpp.simdlen": None}
                ):
                    torch._dynamo.reset()
                    metrics.reset()
                    self.common(fn, (x, y))
                    check_metrics_vec_kernel_count(1)

    def test_do_not_insert_to_dtype_for_memory_copy_only_kernel(self):
        # [FUNCTION DOC] Check a pure memory-copy kernel (clone of bf16) inserts no dtype
        # conversions and is vectorized.
        def fn(x):
            # [FUNCTION DOC] HELPER: Clone the input.
            res = x.clone()
            return res

        x = torch.randn((100, 100), dtype=torch.bfloat16)

        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        if metrics.cpp_to_dtype_count != 0:
            raise AssertionError(
                f"Expected cpp_to_dtype_count == 0, got {metrics.cpp_to_dtype_count}"
            )
        check_metrics_vec_kernel_count(1)

    def test_insert_to_dtype_count(self):
        # [FUNCTION DOC] Check a relu on bf16 inserts exactly two dtype conversions (to and from
        # float) and one vector kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Apply relu.
            res = x.relu()
            return res

        x = torch.randn((100, 100), dtype=torch.bfloat16)

        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        if metrics.cpp_to_dtype_count != 2:
            raise AssertionError(
                f"Expected cpp_to_dtype_count == 2, got {metrics.cpp_to_dtype_count}"
            )
        check_metrics_vec_kernel_count(1)

    def test_emulate_precision_casts_explicit_lowp_round_trip(self):
        # [FUNCTION DOC] Regression test (issue 185337): an explicit float32 to float16 to float32
        # round trip must keep the intermediate rounding when precision cast emulation is on.
        # An explicit fp32->fp16->fp32 round-trip must keep its intermediate
        # rounding under emulate_precision_casts. The CPU codegen used to collapse
        # it via the reverse lowp-fp->fp32 CSE cache (populated at to_dtype time),
        # silently dropping the fp16 rounding. See issue #185337.
        def fn(x):
            # [FUNCTION DOC] HELPER: Round-trip the input through float16 and return it with its
            # row sums.
            y = x.to(torch.float16).to(torch.float32)
            return y, y.sum(dim=1)

        x = torch.arange(20, dtype=torch.float32).reshape(5, 4).t() / 7.0
        with config.patch({"emulate_precision_casts": True}):
            torch._dynamo.reset()
            self.common(fn, (x,))

    def test_memory_copy_with_fusion(self):
        # [FUNCTION DOC] Check relu whose result is copied back into the input in place: two dtype
        # conversions and one vector kernel for bf16.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compute relu, copy it into x in place, return it in a tuple.
            res = x.relu()
            x.copy_(res)
            return (res,)

        x = torch.randn((100, 100), dtype=torch.bfloat16)

        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        if metrics.cpp_to_dtype_count != 2:
            raise AssertionError(
                f"Expected cpp_to_dtype_count == 2, got {metrics.cpp_to_dtype_count}"
            )
        check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_maxpool2d_cpu_only(self):
        # [FUNCTION DOC] Check 3x3 stride 2 max pool on channels-last input for each vectorizable
        # dtype yields one vector kernel and equals eager.
        for dtype in vec_dtypes:
            input = torch.randn(26, 32, 112, 112, dtype=dtype).to(
                memory_format=torch.channels_last
            )
            maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def func(x):
                # [FUNCTION DOC] HELPER: Apply the enclosing max pool module.
                return maxpool(x)

            with patch.object(config.cpp, "simdlen", None):
                torch._dynamo.reset()
                metrics.reset()
                self.common(func, (input,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_maxpool2d_with_pre_loop_collapse_cpu_only(self):
        # [FUNCTION DOC] Check max pool with ceil mode on the sum of two channels-last tensors,
        # tiling heuristics off: two vector kernels expected.
        x1 = torch.randn(2, 3, 20, 20).to(memory_format=torch.channels_last)
        x2 = torch.randn(2, 3, 20, 20).to(memory_format=torch.channels_last)
        maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        def func(x1, x2):
            # [FUNCTION DOC] HELPER: Add the inputs then max pool.
            y = x1 + x2
            return maxpool(y)

        with patch.object(config.cpp, "simdlen", None):
            torch._dynamo.reset()
            metrics.reset()
            self.common(func, (x1, x2))
            check_metrics_vec_kernel_count(2)

    def test_randint_symint_input(self):
        # [FUNCTION DOC] Regression test (issue 122405): random integers whose range comes from a
        # symbolic shape, called with two differently sized inputs under full graph.
        # https://github.com/pytorch/pytorch/issues/122405
        @torch.compile(fullgraph=True)
        def get_traj_idx(lengths: torch.Tensor, num_slices: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Random integers below the first dim of lengths, with the
            # given number of slices.
            return torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)

        lengths = torch.zeros(10, dtype=torch.long)
        get_traj_idx(lengths, num_slices=4)
        lengths = torch.zeros(11, dtype=torch.long)
        get_traj_idx(lengths, num_slices=4)

    def test_store_reduction(self):
        # [FUNCTION DOC] Regression test (issue 157683): amax over dim 0 for matching integer and
        # float dtypes.
        # fix https://github.com/pytorch/pytorch/issues/157683
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Amax of each input along dim 0, returned as a pair.
            r1 = x.amax(dim=0)
            r2 = y.amax(dim=0)
            return r1, r2

        device = "cpu"
        for int_dypte, float_dtype in zip(
            [torch.int64, torch.int32, torch.int16, torch.int8],
            [torch.float64, torch.float32, torch.float16, torch.bfloat16],
        ):
            x = torch.randint(
                low=0, high=100, size=(16, 24, 59), dtype=int_dypte, device=device
            )
            y = torch.randn(16, 24, 59, dtype=float_dtype, device=device)
            self.common(
                fn,
                (
                    x,
                    y,
                ),
            )

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_sign_cpu_only(self):
        # [FUNCTION DOC] Check sign on inputs containing NaN for each vectorizable dtype with one
        # vector kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sign of the input.
            return torch.sign(x)

        for dtype in vec_dtypes:
            x = torch.randn((2, 9), dtype=dtype)
            x[0, 0] = torch.nan
            x[1, -1] = torch.nan

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_reduction_cpu_only(self):
        # [FUNCTION DOC] Check argmax along the last dim for each vectorizable dtype yields one
        # vector kernel (raises AssertionError otherwise).
        def fn(x):
            # [FUNCTION DOC] HELPER: Argmax along the last dim.
            return torch.argmax(x, -1)

        for dtype in vec_dtypes:
            x = torch.randn((10, 10), dtype=dtype)

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                if metrics.generated_cpp_vec_kernel_count != 1:
                    raise AssertionError(
                        f"Expected 1 vec kernel, got {metrics.generated_cpp_vec_kernel_count}"
                    )

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_outer_loop_fusion(self):
        # [FUNCTION DOC] Check outer loop fusion of a max reduction and subtraction: metrics
        # report one fused group with two inner kernels. Graph caches disabled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Subtract the max along the last dim.
            max = torch.amax(x, dim=-1, keepdim=True)
            return x - max

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                2,
            )

    def test_outer_loop_fusion_buffer_remove(self):
        # [FUNCTION DOC] Regression test (issue 144186): outer loop fusion buffer removal for sum
        # then softmax.
        # https://github.com/pytorch/pytorch/issues/144186
        def fn(x):
            # [FUNCTION DOC] HELPER: Sum over the last dim then softmax.
            x = x.sum(dim=-1)
            x = torch.softmax(x, -1)
            return x

        x = torch.randn(8, 8, 2)
        metrics.reset()
        self.common(fn, (x,))

    def test_softmax_with_zero_dim(self):
        # [FUNCTION DOC] Check softmax on a zero-dimensional bfloat16 tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Softmax over dim 0.
            x = torch.softmax(x, 0)
            return x

        x = torch.rand([], dtype=torch.bfloat16)
        metrics.reset()
        self.common(fn, (x,))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_softmax_with_unbacked_zero_dim(self):
        # [FUNCTION DOC] Check softmax and log softmax with a data-dependent slice length
        # (including zero) under full graph with scalar capture, compared to eager including NaNs.
        def softmax_fn(x, n):
            # [FUNCTION DOC] HELPER: Softmax over a slice of x with length taken from n.
            return torch.softmax(x[:, : n.item()], -1)

        def log_softmax_fn(x, n):
            # [FUNCTION DOC] HELPER: Log softmax over a slice of x with length taken from n.
            return torch.log_softmax(x[:, : n.item()], -1)

        x = torch.randn(3, 5)
        for fn in (softmax_fn, log_softmax_fn):
            compiled_fn = torch.compile(fn, fullgraph=True)
            for length in (0, 2, 5):
                n = torch.tensor(length)
                actual = compiled_fn(x, n)
                expected = fn(x, n)
                torch.testing.assert_close(actual, expected, equal_nan=True)

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_local_buffer_in_outer_loop_fusion(self):
        # [FUNCTION DOC] Check local buffer use in outer loop fusion for subtraction of softmax:
        # one fused group of three inner kernels with one local buffer, and three global buffer
        # allocations in generated code.
        def fn(x):
            # [FUNCTION DOC] HELPER: Subtract softmax from the input.
            max = torch.nn.functional.softmax(x, dim=-1)
            return x - max

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                3,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                1,
            )
            # Check the number of global buffer allocation
            torch._dynamo.reset()
            metrics.reset()
            _, code = run_and_get_cpp_code(
                torch.compile(fn, backend="inductor"),
                x,
            )
            self.assertEqual(
                code.count(
                    "aoti_torch_empty_strided("
                    if config.cpp_wrapper
                    else "empty_strided_cpu("
                ),
                3,
            )

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_two_local_buffers_in_outer_loop_fusion(self):
        # [FUNCTION DOC] Check two local buffers in an outer loop fusion of a softmax, sums, a
        # broadcast, exp and subtractions: fusion count one, five inner kernels, two local
        # buffers; tolerances differ with vectorization availability.
        def fn(x):
            # [FUNCTION DOC] HELPER: Softmax, row sum broadcast to 256 columns, exp, subtract
            # second row sum, and subtract from a slice of the input.
            softmax = torch.nn.functional.softmax(x, dim=-1)
            sum = torch.sum(softmax, dim=-1)
            sum_broadcast = torch.broadcast_to(
                sum.unsqueeze(-1), [*(sum.size()[0:3]), 256]
            )
            sum_exp = torch.exp(sum_broadcast)
            sum2 = torch.sum(sum_exp, dim=-1)
            sub = sum_exp - sum2.unsqueeze(-1)
            return x[:, :, :, 0:256] - sub

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            # Outer-loop fusion changes fp32 reduction order enough to exceed
            # the default tolerance on numerically sensitive inputs.
            atol = 1e-5
            rtol = 2e-6
            if (
                not cpu_vec_isa.valid_vec_isa_list()
                or os.getenv("ATEN_CPU_CAPABILITY") == "default"
            ):
                atol = 1e-5
                rtol = 1e-5
            self.common(fn, (x,), atol=atol, rtol=rtol)
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                5,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                2,
            )

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_share_local_buffers_in_outer_loop_fusion(self):
        # [FUNCTION DOC] Check local buffer sharing in an outer loop fusion of two chained
        # softmaxes: five inner kernels and one local buffer shared by two global buffers.
        def fn(x):
            # [FUNCTION DOC] HELPER: Two chained softmaxes, subtracting from the input.
            max = torch.nn.functional.softmax(x, dim=-1)
            max = torch.nn.functional.softmax(max, dim=-1)
            return x - max

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                5,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                1,  # 2 global bufs share 1 local buf
            )

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_two_local_buffers_in_outer_loop_fusion_case2(self):
        # [FUNCTION DOC] Check two local buffers in outer loop fusion for a chain of exp and sum
        # steps where an earlier exp result is used again later, so it cannot share a local
        # buffer: one fused group with four inner kernels and two local buffers.
        # exp and exp2 should be replaced by local buffer
        # since exp will be used after exp2, exp2 can't share the same
        # local buffer as exp
        def fn(x):
            # [FUNCTION DOC] HELPER: Chain of amax, exp, sums and subtractions that reuses the
            # first exp at the end.
            a_max = torch.amax(x, -1, keepdim=True)
            exp = torch.exp(x - a_max)
            sum = torch.sum(exp, -1, keepdim=True)
            exp2 = torch.exp(exp - sum)
            sum2 = torch.sum(exp2, -1, keepdim=True)
            sub = exp2 - sum2
            sub2 = exp - sub
            return sub2

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                4,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                2,
            )

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_local_buffer_with_line_reuse(self):
        # [FUNCTION DOC] Check a global in-place buffer replaced by a local buffer in outer loop
        # fusion after a matmul, with softmax-like arithmetic: one fused group and one local
        # buffer.
        # Test Global buffer which is inplace buffer and replaced by local buffer
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Matmul, max of x, exp of a scaled difference, sum, then
            # subtract the sum from exp.
            z = torch.matmul(x, y)
            a_max = torch.amax(x, -1, keepdim=True)
            # Previous is a inplace buffer and now is a local buffer
            exp = torch.exp((z - a_max) / z)
            sum = torch.sum(exp, -1, keepdim=True)
            return exp - sum

        inputs = [torch.rand(4, 32), torch.rand(32, 32)]

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, inputs)
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                1,
            )

    @requires_vectorization
    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_outer_loop_local_buffer_with_tiled_outer_dim(self):
        # [FUNCTION DOC] Check outer loop local buffers when the outer dimension is tiled: a chain
        # of softplus, gammaln, erf, xlogy, logaddexp and stacking, compared to eager with tight
        # tolerances. Requires vectorization; caches disabled.
        def fn(l_c, y, alpha, beta):
            # [FUNCTION DOC] HELPER: Compute special-function terms from a slice of l_c, y, alpha
            # and beta and stack three results on a new last dim.
            x = l_c[..., 16]
            positive = F.softplus(x) + 0.0001
            rates = F.softplus(y) + 0.0001

            gammaln_term = torch.special.gammaln(positive + alpha.view(1, -1))
            erf_term = torch.special.erf(x * beta.view(1, -1))

            xlogy_term = torch.special.xlogy(positive, rates)
            logadd = torch.logaddexp(gammaln_term, xlogy_term)
            centered = logadd - logadd.mean(dim=-1, keepdim=True)

            return torch.stack([centered, erf_term, centered * erf_term], dim=-1)

        torch.manual_seed(20592561)
        inputs = (
            torch.randn([4, 8, 64], dtype=torch.float32) * 0.1,
            torch.randn([4, 8], dtype=torch.float32) * 0.1,
            torch.randn([8], dtype=torch.float32) * 0.1,
            torch.zeros([8], dtype=torch.float32),
        )

        with torch.no_grad():
            self.common(fn, inputs, atol=1e-6, rtol=1e-6)

    @requires_vectorization
    def test_argmin(self):
        # [FUNCTION DOC] Check argmin along the last dim for vectorizable dtypes yields one vector
        # kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Argmin along the last dim.
            return torch.argmin(x, -1)

        for dtype in vec_dtypes:
            x = torch.randn((10, 10), dtype=dtype)
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            if metrics.generated_cpp_vec_kernel_count != 1:
                raise AssertionError(
                    f"Expected 1 vec kernel, got {metrics.generated_cpp_vec_kernel_count}"
                )

    @requires_vectorization
    def test_argmax_argmin_with_nan_value(self):
        # [FUNCTION DOC] Check whole-tensor argmax and argmin on inputs containing NaN values
        # (from log of negatives) produce one vector kernel each.
        def fn(x):
            # [FUNCTION DOC] HELPER: Argmax over all elements.
            return torch.argmax(x)

        def fn2(x):
            # [FUNCTION DOC] HELPER: Argmin over all elements.
            return torch.argmin(x)

        inputs = [
            torch.Tensor([-755832.1250, 100]),
            torch.Tensor([-755832.1250, 100, 200]),
            torch.Tensor([100, -755832.1250]),
            torch.Tensor([100, 200, -755832.1250]),
        ]

        for x in inputs:
            x = x.repeat(16, 16)
            x = torch.log1p(x)

            # Test argmax
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            if metrics.generated_cpp_vec_kernel_count != 1:
                raise AssertionError(
                    f"Expected generated_cpp_vec_kernel_count == 1, got {metrics.generated_cpp_vec_kernel_count}"
                )

            # Test argmin
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn2, (x,))
            if metrics.generated_cpp_vec_kernel_count != 1:
                raise AssertionError(
                    f"Expected generated_cpp_vec_kernel_count == 1, got {metrics.generated_cpp_vec_kernel_count}"
                )

    def test_argmax_argmin_transpose_logical_index_cpu(self):
        # [FUNCTION DOC] Check argmax of a transposed tensor after an in-place sine yields the
        # expected flat index 31 and matches eager; also checks argmin and argmax of a transposed
        # matrix with planted extremes.
        def issue_fn(x):
            # [FUNCTION DOC] HELPER: Clone, apply sine in place, then argmax of the transpose.
            x = x.clone()
            x.sin_()
            return x.t().argmax()

        x = torch.tensor(
            [
                [-1.5256, -0.7502, -0.6540, -1.6095, -0.1002, -0.6092, -0.9798],
                [-1.6091, -0.7121, 0.3037, -0.7773, -0.2515, -0.2223, 1.6871],
                [0.2284, 0.4676, -0.6970, -1.1608, 0.6995, 0.1991, 0.8657],
                [0.2444, -0.6629, 0.8073, 1.1017, -0.1759, -2.2456, -1.4465],
                [0.0612, -0.6177, -0.7981, -0.1316, -0.7984, 0.3357, 0.2753],
            ]
        )
        self.assertEqual(issue_fn(x).item(), 31)
        self.common(issue_fn, (x,))

        def argmin_argmax_fn(x):
            # [FUNCTION DOC] HELPER: Return argmin and argmax of the transposed input.
            return (x.t().argmin(), x.t().argmax())

        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        x[4, 0] = -100
        x[1, 3] = 100
        self.common(argmin_argmax_fn, (x,))

    @requires_vectorization
    def test_argmax_argmin_cpptile2d_2d_input(self):
        # [FUNCTION DOC] Check max and min along dim 1 of the sum of a transposed and a plain
        # square matrix, for sizes 8, 32 and 35, matching eager. Requires vectorization.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Max along dim 1 of the sum.
            return (a + b).max(dim=1)

        def fn_min(a, b):
            # [FUNCTION DOC] HELPER: Min along dim 1 of the sum.
            return (a + b).min(dim=1)

        torch.manual_seed(0)
        for sz in [8, 32, 35]:
            for f in [fn, fn_min]:
                torch._dynamo.reset()
                a = torch.randn(sz, sz).transpose(0, 1)
                b = torch.randn(sz, sz)
                self.common(f, (a, b))

    @requires_vectorization
    def test_argmax_argmin_cpptile2d_3d_input(self):
        # [FUNCTION DOC] Check max and min along the last dim of the sum of a permuted and plain
        # 3D tensor.
        def fn_3d(a, b):
            # [FUNCTION DOC] HELPER: Max along dim 2 of the sum.
            return (a + b).max(dim=2)

        def fn_3d_min(a, b):
            # [FUNCTION DOC] HELPER: Min along dim 2 of the sum.
            return (a + b).min(dim=2)

        for f in [fn_3d, fn_3d_min]:
            torch._dynamo.reset()
            a = torch.randn(4, 16, 16).permute(0, 2, 1)
            b = torch.randn(4, 16, 16)
            self.common(f, (a, b))

    # Currently, we enabled AVX2 and AVX512 for vectorization. If the platform is not
    # supported, the vectorization will not work and skip this test case. For ARM or
    # other platforms support, we just need to add the ISA info to the supported_vector_isa
    # and include proper aten vectorization head file.
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_kernel_cpu_only(self):
        # [FUNCTION DOC] Check vectorization over a long chain of elementwise ops: with simdlen 1
        # no vector kernel is generated, with default simdlen one is, per vectorizable dtype; then
        # a permuted input gives two vector kernels and an odd-width input one. Skipped under
        # fbcode.
        def fn(x1, x2):
            # [FUNCTION DOC] HELPER: Long chain of unary and binary elementwise ops on x1 and x2
            # ending with sign plus x2.
            # Current, there are some limitations as follows.
            #   rsqrt:
            #     assert [both a fallback and a decomp for same kernel: aten.rsqrt.default]
            #   round:
            #     couldn't find symbolic meta function/decomposition
            #   fmod/logical_and/logic_or:
            #     vec kernel has not support to_type
            x = torch.abs(x1)
            x = torch.sin(x)
            x = torch.neg(x)
            x = torch.square(x)
            x = torch.sigmoid(x)
            x = torch.relu(x)
            x = torch.cos(x)
            x = torch.exp(x)
            x = torch.sqrt(x)
            x = torch.add(x, x1)
            x = torch.sub(x, x2)
            x = torch.mul(x, x1)
            x = torch.div(x, x1)
            x = torch.pow(x, 10)
            x = torch.log(x)
            x = torch.floor(x)
            x = torch.ceil(x)
            x = torch.trunc(x)
            x = torch.lgamma(x)
            x = torch.fmod(x, x2)
            x = torch.sign(x)
            res = x + x2
            return res

        for dtype in vec_dtypes:
            torch.manual_seed(0)
            x1 = torch.randn((5, 20), dtype=dtype)
            x2 = torch.randn((5, 20), dtype=dtype)

            with config.patch({"cpp.simdlen": 1}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x1, x2))
                if metrics.generated_cpp_vec_kernel_count != 0:
                    raise AssertionError(
                        f"Expected generated_cpp_vec_kernel_count == 0, got {metrics.generated_cpp_vec_kernel_count}"
                    )

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x1, x2))
                check_metrics_vec_kernel_count(1)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            x1 = torch.randn(10, 20).permute(1, 0)
            x2 = torch.randn((20, 10))
            self.common(fn, (x1, x2))
            check_metrics_vec_kernel_count(2)

            torch._dynamo.reset()
            metrics.reset()
            x1 = torch.randn((10, 7))
            x2 = torch.randn((10, 7))
            self.common(fn, (x1, x2))
            check_metrics_vec_kernel_count(1)

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(
        sys.platform not in ["linux", "win32"],
        "cpp kernel profile only support linux now",
    )
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch({"cpp.enable_kernel_profile": True})
    @config.patch({"cpp.descriptive_names": "original_aten"})
    def test_cpp_kernel_profile(self):
        # [FUNCTION DOC] Check the C++ kernel profiling option: a compiled add run under the
        # profiler yields at least one event whose name contains the fused add kernel name; raises
        # AssertionError otherwise. Skipped under fbcode and on platforms other than linux or
        # windows.
        from torch.profiler import profile

        @torch.compile(backend="inductor", fullgraph=True)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Add the inputs.
            return a + b

        a = torch.rand((100,))
        b = torch.rand((100,))
        with profile() as prof:
            fn(a, b)

        kernel_profile_events = []
        for e in prof.profiler.function_events:
            if "cpp_fused_add_0" in e.name:
                kernel_profile_events.append(e.name)
        if len(kernel_profile_events) <= 0:
            raise AssertionError(
                f"Expected kernel_profile_events to be non-empty, got {len(kernel_profile_events)}"
            )

    @xfailIfS390X
    @requires_vectorization
    def test_channel_shuffle_cl_output(self):
        # [FUNCTION DOC] Check channel shuffle producing channels-last output (shapes from
        # shufflenet) for each SIMD length, expecting two vector kernels. Expected to fail on
        # s390x.
        """code and shape extracted from shufflenet_v2_x1_0"""

        def channel_shuffle(x, groups):
            # [FUNCTION DOC] HELPER: Group-wise channel shuffle via view, transpose, view and
            # channels-last conversion.
            batchsize, num_channels, height, width = x.size()
            channels_per_group = num_channels // groups
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, -1, height, width)
            return x.contiguous(memory_format=torch.channels_last)

        for simdlen in simd_lengths_to_test():
            with config.patch({"cpp.simdlen": simdlen}):
                torch._dynamo.reset()
                metrics.reset()
                x = torch.randn(64, 58, 28, 28)
                self.common(channel_shuffle, (x, 2))
                check_metrics_vec_kernel_count(2)

    @slowTest
    @requires_vectorization
    def test_transpose_with_norm(self):
        # [FUNCTION DOC] Slow test: a TIMM gmlp sub-module (linear, gelu, chunk, layer norm,
        # transposed projection, product, linear) in eval and train mode across SIMD lengths,
        # expecting eight vector kernels.
        """a sub-module from TIMM gmlp_s16_224"""

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with linear, gelu, layer norm, projection and output
            # linear layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the layers.
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features=256, out_features=1536, bias=True
                )
                self.act = torch.nn.GELU()
                self.norm = torch.nn.LayerNorm(768)
                self.proj = torch.nn.Linear(196, 196)
                self.fc = torch.nn.Linear(in_features=768, out_features=256, bias=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Linear, gelu, split in halves, normalize and project one
                # half transposed, multiply with the other, and apply the final linear.
                x = self.linear(x)
                x = self.act(x)
                u, v = x.chunk(2, dim=-1)
                v = self.norm(v)
                v = self.proj(v.transpose(-1, -2))
                y = u * v.transpose(-1, -2)
                return self.fc(y)

        x = torch.randn(128, 196, 256)
        for simdlen in simd_lengths_to_test():
            with config.patch({"cpp.simdlen": simdlen}):
                for eval_mode in [True, False]:
                    torch._dynamo.reset()
                    metrics.reset()
                    m = Model().eval() if eval_mode else Model()
                    self.common(m, (x,))
                    check_metrics_vec_kernel_count(8)

    @requires_vectorization
    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_transpose_copy(self):
        # [FUNCTION DOC] Check transposed copy of square matrices of sizes around the tile
        # boundaries, in float and bf16 for each SIMD length, with tiling heuristics off; expects
        # two vector kernels.
        def fn(a):
            # [FUNCTION DOC] HELPER: Transposed contiguous copy.
            return a.t().contiguous()

        for simdlen in simd_lengths_to_test():
            with config.patch({"cpp.simdlen": simdlen}):
                for dtype in (torch.float, torch.bfloat16):
                    for shape in (
                        (7, 7),
                        (8, 8),
                        (9, 9),
                        (16, 16),
                        (17, 17),
                        (32, 32),
                        (33, 33),
                    ):
                        torch._dynamo.reset()
                        metrics.reset()
                        x = torch.randn(shape, dtype=dtype)
                        self.common(fn, (x,))
                        check_metrics_vec_kernel_count(2)

    @torch._dynamo.config.patch(specialize_int=False)
    def test_slice_scatter_issue122291(self):
        # [FUNCTION DOC] Regression test (issue 122291): slice_scatter with mismatched source
        # shape and unspecialized ints must raise an InductorError mentioning a shape error in the
        # scatter op.
        @torch.compile(fullgraph=True)
        def fn(t, t_src, dim, start, end, step):
            # [FUNCTION DOC] HELPER: Slice scatter of the source into t along a dim with start,
            # end and step.
            return t.slice_scatter(t_src, dim, start, end, step)

        shape = ((16, 16), (16, 2), 1, 4, 10, 1)
        input_tensor = torch.zeros(shape[0], requires_grad=False, device="cpu")
        src_tensor = torch.ones(shape[1], requires_grad=False, device="cpu")
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError, r".*shape error in scatter op"
        ):
            fn(input_tensor, src_tensor, shape[2], shape[3], shape[4], shape[5])

    def test_horizontal_fusion(self):
        # [FUNCTION DOC] Check horizontal fusion limits: with maximum fusion size 0, 1, 2 and 3
        # three index_select operations produce 3, 3, 2 and 1 kernels respectively, and results
        # match eager.
        def fn(a, b, c, idx):
            # [FUNCTION DOC] HELPER: Index select along dim 0 on each of three inputs with a
            # shared index; returns all three.
            _a = torch.index_select(a, dim=0, index=idx)
            _b = torch.index_select(b, dim=0, index=idx)
            _c = torch.index_select(c, dim=0, index=idx)
            return _a, _b, _c

        with config.patch({"cpp.max_horizontal_fusion_size": 0}):
            metrics.reset()
            torch._dynamo.reset()
            a = torch.randn(size=(4, 16), dtype=torch.bfloat16)
            b = torch.randn(size=(4, 16), dtype=torch.bfloat16)
            c = torch.randn(size=(4, 16), dtype=torch.bfloat16)
            idx = torch.zeros(size=[4], dtype=torch.int64)
            opt_fn = torch.compile(fn, backend="inductor")
            opt_fn(a, b, c, idx)
            self.assertEqual(metrics.generated_kernel_count, 3)
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))

        with config.patch({"cpp.max_horizontal_fusion_size": 1}):
            metrics.reset()
            torch._dynamo.reset()
            a = torch.randn(size=(4, 32), dtype=torch.bfloat16)
            b = torch.randn(size=(4, 32), dtype=torch.bfloat16)
            c = torch.randn(size=(4, 32), dtype=torch.bfloat16)
            idx = torch.zeros(size=[4], dtype=torch.int64)
            opt_fn = torch.compile(fn, backend="inductor")
            opt_fn(a, b, c, idx)
            self.assertEqual(metrics.generated_kernel_count, 3)
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))

        with config.patch({"cpp.max_horizontal_fusion_size": 2}):
            metrics.reset()
            torch._dynamo.reset()
            a = torch.randn(size=(4, 64), dtype=torch.bfloat16)
            b = torch.randn(size=(4, 64), dtype=torch.bfloat16)
            c = torch.randn(size=(4, 64), dtype=torch.bfloat16)
            idx = torch.zeros(size=[4], dtype=torch.int64)
            opt_fn = torch.compile(fn, backend="inductor")
            opt_fn(a, b, c, idx)
            print(metrics.generated_kernel_count)
            self.assertEqual(metrics.generated_kernel_count, 2)
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))

        with config.patch({"cpp.max_horizontal_fusion_size": 3}):
            metrics.reset()
            torch._dynamo.reset()
            a = torch.randn(size=(4, 128), dtype=torch.bfloat16)
            b = torch.randn(size=(4, 128), dtype=torch.bfloat16)
            c = torch.randn(size=(4, 128), dtype=torch.bfloat16)
            idx = torch.zeros(size=[4], dtype=torch.int64)
            opt_fn = torch.compile(fn, backend="inductor")
            opt_fn(a, b, c, idx)
            self.assertEqual(metrics.generated_kernel_count, 1)
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))

    def test_lowp_fp_neg_abs(self):
        # [FUNCTION DOC] Check negation then absolute value in low precision creates no dtype
        # conversions and one vector kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Negate then absolute value.
            return x.neg().abs()

        for dtype in _lowp_fp_dtypes:
            metrics.reset()
            x = torch.randn(100, 100).to(dtype)
            opt_fn = torch.compile(fn, backend="inductor")
            self.assertTrue(same(fn(x), opt_fn(x)))
            if metrics.cpp_to_dtype_count != 0:
                raise AssertionError(
                    f"Expected cpp_to_dtype_count == 0, got {metrics.cpp_to_dtype_count}"
                )
            check_metrics_vec_kernel_count(1)

    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_transpose_non_contiguous(self):
        # [FUNCTION DOC] Regression test (issue 94269): a chain of strided views, clones, splits
        # and permutes extracted from timm halo attention on a channels-last input matches eager.
        # Tiling heuristics off.
        def fn(a):
            # [FUNCTION DOC] HELPER: Chain of as_strided, clone, view, permute, split, expand and
            # clone operations.
            # From part of timm HaloAttn:
            # (https://github.com/rwightman/pytorch-image-models/blob/main/timm/layers/halo_attn.py#L97).
            # Fixed https://github.com/pytorch/pytorch/issues/94269 accuracy issue.
            as_strided = torch.ops.aten.as_strided.default(
                a, [1, 384, 2, 20, 12], [153600, 1, 61440, 384, 7680]
            )
            as_strided_1 = torch.ops.aten.as_strided.default(
                as_strided,
                [1, 384, 2, 2, 12, 12],
                [153600, 1, 61440, 3072, 7680, 384],
            )
            clone_1 = torch.ops.aten.clone.default(
                as_strided_1, memory_format=torch.contiguous_format
            )
            _unsafe_view_1 = torch.ops.aten._unsafe_view.default(
                clone_1, [8, 48, 4, 144]
            )
            permute_2 = torch.ops.aten.permute.default(_unsafe_view_1, [0, 2, 3, 1])
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                permute_2, [16, 32], -1
            )
            getitem = split_with_sizes[0]
            _getitem_1 = split_with_sizes[1]
            permute_3 = torch.ops.aten.permute.default(getitem, [0, 1, 3, 2])
            expand_1 = torch.ops.aten.expand.default(permute_3, [8, 4, 16, 144])
            clone_3 = torch.ops.aten.clone.default(
                expand_1, memory_format=torch.contiguous_format
            )
            return clone_3

        metrics.reset()
        x = torch.randn(1, 384, 20, 20).to(memory_format=torch.channels_last)
        self.common(fn, (x,))

    def test_issue_148058(self):
        # [FUNCTION DOC] Regression test (issue 148058): compile gumbel softmax, selection and
        # scatter; only checks it runs.
        # Fix issue https://github.com/pytorch/pytorch/issues/148058
        def fn(x):
            # [FUNCTION DOC] HELPER: Gumbel softmax with hard sampling, zero out values at or
            # below half, scatter ones.
            x = F.gumbel_softmax(x, tau=1.0, hard=True)
            x = torch.where(x > 0.5, x, torch.zeros_like(x))
            x = torch.scatter(
                x,
                dim=1,
                index=torch.ones(1, 2, dtype=torch.long),
                src=torch.ones_like(x),
            )
            return x

        metrics.reset()
        x = torch.randn(1, 2)
        # Only test for functionality since the output of gumbel_softmax has randomness
        torch.compile(fn, backend="inductor")(x)

    def test_non_contiguous_index_with_constant_stride(self):
        # [FUNCTION DOC] Check a non-contiguous index with constant stride (strided slices, stack,
        # flatten) matches eager and generates two fused kernels (three with the cpp wrapper).
        def fn(x):
            # [FUNCTION DOC] HELPER: Stack negated odd-indexed and even-indexed slices of the last
            # dim, then flatten.
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            return x.flatten(-2)

        metrics.reset()
        x = torch.randn(1, 32, 16, 68)
        opt_fn = torch.compile(fn, backend="inductor")
        _, code = run_and_get_cpp_code(opt_fn, x)
        self.assertTrue(same(fn(x), opt_fn(x)))
        # declare, def, and use (declare and def are the same in non-cpp_wrapper mode)
        FileCheck().check_count(
            "cpp_fused", 3 if config.cpp_wrapper else 2, exactly=True
        ).run(code)

    @requires_vectorization
    def test_vertical_reduction_tail_store_uses_tail_width(self):
        # [FUNCTION DOC] Check the tail store of a vertical reduction uses the tail width: code
        # must contain the masked reduction helper and a store with width two, a scalar store
        # loop, and not a 16-wide store. Matches eager. Requires vectorization.
        def fn(x, y, z, residual, weight, bias):
            # [FUNCTION DOC] HELPER: Leaky relu of x times y plus z, permute, add residual, layer
            # norm over 14 elements.
            x = F.leaky_relu(x * y + z)
            x = x.permute(1, 0).unsqueeze(1)
            return F.layer_norm(x + residual, (14,), weight, bias)

        torch.manual_seed(0)
        x = torch.randn(14, 2)
        y = torch.randn(14, 2)
        z = torch.randn(14, 2)
        residual = torch.randn(2, 2, 14)
        weight = torch.randn(14)
        bias = torch.randn(14)
        opt_fn = torch.compile(fn, backend="inductor")

        actual, code = run_and_get_cpp_code(opt_fn, x, y, z, residual, weight, bias)

        self.assertTrue(same(fn(x, y, z, residual, weight, bias), actual))
        self.assertIn("sum_masked_reduce", code)
        self.assertIn(
            "tmp_acc0_vec.store(tmpbuf.data(), static_cast<int64_t>(2L));",
            code,
        )
        self.assertIn(
            "out_ptr0[static_cast<int64_t>(x1 + 2L*x0 + 2L*x0_inner)] = tmpbuf[x0_inner];",
            code,
        )
        self.assertNotIn(
            ".store(tmpbuf.data(), static_cast<int64_t>(16));",
            code,
        )

    def test_invalid_index_of_empty_tensor(self):
        # [FUNCTION DOC] Check indexing an empty tensor with a list index raises RuntimeError when
        # compiled.
        def fn(a):
            # [FUNCTION DOC] HELPER: Index with a one-element list.
            b = a[[0]]
            return b

        a = torch.tensor([])
        with self.assertRaises(RuntimeError):
            torch.compile(fn)(a)

    @torch.no_grad()
    @torch._inductor.config.patch(freezing=True)
    def test_issue122380(self):
        # [FUNCTION DOC] Regression test (issue 122380): unbind, stack and tanh under no-grad with
        # freezing matches eager.
        def func(x):
            # [FUNCTION DOC] HELPER: Unbind, restack on dim 1, tanh.
            t1 = torch.unbind(x)
            t2 = torch.stack(t1, dim=1)
            t3 = torch.tanh(t2)
            return t3

        x = torch.randn(2, 3, 4)
        self.assertEqual(torch.compile(func)(x), func(x))

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_ir_node_str(self):
        # [FUNCTION DOC] Check IR nodes can be converted to strings: wraps the graph lowering
        # run_node method to record the string of each result and asserts more than three were
        # recorded. Caches disabled.
        @torch.compile
        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Return sine and softmax of cosine.
            return x.sin(), torch.nn.Softmax(dim=1)(x.cos())

        def run_node_alt(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Call the original run_node, record the string of the result
            # and return it.
            rv = run_node(*args, **kwargs)
            strings.append(str(rv))
            return rv

        strings = []
        run_node = GraphLowering.run_node
        with patch.object(GraphLowering, "run_node", run_node_alt):
            fn(torch.randn([8, 128]))
        self.assertGreater(len(strings), 3)

    def test_vertical_sum_cpu_only(self):
        # [FUNCTION DOC] Check vertical sum along dim 0 (2D) and dim 1 (3D) each yield one vector
        # kernel.
        def fn1(a):
            # [FUNCTION DOC] HELPER: Sum over dim 0.
            return a.sum(dim=0)

        def fn2(a):
            # [FUNCTION DOC] HELPER: Sum over dim 1.
            return a.sum(dim=1)

        metrics.reset()
        x = torch.randn(100, 100)
        self.common(fn1, (x,))
        check_metrics_vec_kernel_count(1)

        metrics.reset()
        x = torch.randn(100, 100, 100)
        self.common(fn2, (x,))
        check_metrics_vec_kernel_count(1)

    def test_transpose_vertical_sum_cpu_only(self):
        # [FUNCTION DOC] Check sum over dim 1 of the product of a plain and a transposed 3D tensor
        # gives two vector kernels.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Multiply and sum along dim 1.
            c = a * b
            return c.sum(dim=1)

        metrics.reset()
        x = torch.randn(100, 50, 50)
        y = torch.randn(100, 50, 50).transpose(1, 2)
        self.common(fn, (x, y))
        check_metrics_vec_kernel_count(2)

    def test_transpose_mxn_16_16_bf16_fp16(self):
        # [FUNCTION DOC] Check the 16 by 16 transpose path for bf16 and fp16 in the same pattern:
        # two vector kernels.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Multiply and sum along dim 1.
            c = a * b
            return c.sum(dim=1)

        for dtype in [torch.bfloat16, torch.float16]:
            metrics.reset()
            x = torch.randn(100, 50, 50).to(dtype)
            y = torch.randn(100, 50, 50).to(dtype).transpose(1, 2)
            self.common(fn, (x, y))
            check_metrics_vec_kernel_count(2)

    def test_transpose_mxn_32_32_bf16_fp16(self):
        # [FUNCTION DOC] Check the 32 by 32 transpose path for bf16 and fp16 on large permuted
        # tensors: two vector kernels.
        def fn(a):
            # [FUNCTION DOC] HELPER: Permute last two dims and make contiguous.
            return a.permute(0, 2, 1).contiguous()

        for dtype in [torch.bfloat16, torch.float16]:
            metrics.reset()
            x = torch.randn(2, 9216, 9216).to(dtype)
            self.common(fn, (x,))
            check_metrics_vec_kernel_count(2)

    def test_transpose_sum2d_cpu_only(self):
        # [FUNCTION DOC] Check a full sum of the product of a matrix and a transposed matrix gives
        # two vector kernels.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Multiply then sum all.
            c = a * b
            return c.sum()

        metrics.reset()
        x = torch.randn(50, 50)
        y = torch.randn(50, 50).transpose(0, 1)
        self.common(fn, (x, y))
        check_metrics_vec_kernel_count(2)

    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_transpose_sum_outer(self):
        # [FUNCTION DOC] Regression test (issue 98573): transposed sum along dim 1 with tiling
        # heuristics off.
        # https://github.com/pytorch/pytorch/issues/98573
        def fn(a):
            # [FUNCTION DOC] HELPER: Transpose dims 2 and 3, sum along dim 1, make contiguous.
            return a.transpose(2, 3).sum(dim=1).contiguous()

        metrics.reset()
        x = torch.randn(10, 50, 50, 50)
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_to_dtype_bool_float(self):
        # [FUNCTION DOC] Regression test (issue 100800): float to bool conversion in a where
        # condition.
        # https://github.com/pytorch/pytorch/issues/100800
        def f(a):
            # [FUNCTION DOC] HELPER: Select zeros or twice ones based on a bool conversion of
            # ones.
            return torch.where(
                torch.ones_like(a).to(torch.bool),
                torch.zeros_like(a),
                torch.ones_like(a) * 2,
            )

        self.common(f, (torch.ones(16),))

    def test_to_dtype_float_bool(self):
        # [FUNCTION DOC] Regression test (issue 100466): bool to float conversion of a comparison
        # multiplying the input.
        # https://github.com/pytorch/pytorch/issues/100466
        def f(a):
            # [FUNCTION DOC] HELPER: Multiply by a float tensor made from a comparison
            # (non-negative mask).
            a = a * torch.tensor(a >= 0, dtype=torch.float32)
            return a

        x = torch.rand(16)
        self.common(f, (x,))

    def test_constant_store(self):
        # [FUNCTION DOC] Regression test (issue 104515): assigning negative infinity into an
        # advanced-indexed location.
        # https://github.com/pytorch/pytorch/issues/104515
        def f(a):
            # [FUNCTION DOC] HELPER: Write negative infinity at row 0 and columns 3 and 3; returns
            # a (mutated).
            a[0, [3, 3]] = -float("inf")
            return a

        x = torch.rand(4, 5)
        self.common(f, (x,))

    def test_broadcast_scalar_cpp_tile_2d_kernel(self):
        # [FUNCTION DOC] Check broadcast scalar handling in a 2D tile kernel using two
        # convolutions and a max pool (detectron2-like) with dynamic spatial dims, comparing
        # against eager including gradients setup.
        # Based on detectron2_maskrcnn backbone (conv2d -> max_pool2d)
        s0 = 12
        s1 = 21

        data = torch.randn(
            [1, 256, 8 * s0, 8 * s1],
        )
        weight_one = torch.randn([256, 256, 1, 1], requires_grad=True)
        weight_two = torch.randn((256, 256, 3, 3), requires_grad=True)
        bias_one = torch.randn([256], requires_grad=True)
        bias_two = torch.randn([256], requires_grad=True)

        @torch.compile
        def fn(data, weight_one, weight_two, bias_one, bias_two):
            # [FUNCTION DOC] HELPER: Two 1x1 and 3x3 convolutions with padding of one and a stride
            # 2 max pool on the first.
            conv_result_one = torch.ops.aten.convolution.default(
                data,
                weight_one,
                bias_one,
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                1,
            )

            conv_result_two = torch.ops.aten.convolution.default(
                data,
                weight_two,
                bias_two,
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                1,
            )

            max_pool_result = torch.nn.functional.max_pool2d(
                conv_result_one,
                [1, 1],
                [2, 2],
                [0, 0],
                [1, 1],
                False,
            )
            return conv_result_one, conv_result_two, max_pool_result

        torch._dynamo.mark_dynamic(data, 2)
        torch._dynamo.mark_dynamic(data, 3)
        self.common(fn, (data, weight_one, weight_two, bias_one, bias_two))

    def test_to_channels_last_lowp_fp(self):
        # [FUNCTION DOC] Check conversion to channels last for low-precision dtypes.
        def f(a):
            # [FUNCTION DOC] HELPER: Convert to channels last.
            return a.to(memory_format=torch.channels_last)

        for dtype in _lowp_fp_dtypes:
            x = torch.rand(2, 3, 14, 14).to(dtype)
            self.common(f, (x,))

    def test_broadcast_mul_lowp_fp(self):
        # [FUNCTION DOC] Check multiplication by a broadcast tensor for low-precision dtypes.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Multiply the inputs.
            return a * b

        for dtype in _lowp_fp_dtypes:
            a = torch.randn(2, 16, 16).to(dtype)
            b = torch.randn(2, 1, 1).to(dtype)
            self.common(f, (a, b))

    def test_linear_buffer_reuse(self):
        # [FUNCTION DOC] Check a module with linear, tanh, linear compiles with buffer reuse:
        # generated code must not contain an as_strided call and results match eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two linear layers separated by tanh.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the layers.
                super().__init__()
                self.linear1 = torch.nn.Linear(16, 16)
                self.tanh = torch.nn.Tanh()
                self.linear2 = torch.nn.Linear(16, 16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Linear, tanh, linear.
                x = self.linear1(x)
                x = self.tanh(x)
                x = self.linear2(x)
                return x

        mod = M().eval()
        v = torch.randn(1, 16)

        with torch.no_grad():

            def compile_fx_wrapper(model_, example_inputs_):
                # [FUNCTION DOC] HELPER: Backend function delegating to compile_fx.
                return compile_fx(model_, example_inputs_)

            def run(*ex, **kwargs):
                # [FUNCTION DOC] HELPER: Call the enclosing module with the given arguments.
                return mod(*ex, **kwargs)

            run = torch.compile(run, backend=compile_fx_wrapper)
            _, code = run_and_get_cpp_code(run, v)
            self.assertFalse("= as_strided(" in code)
            self.assertEqual(run(*v), mod(*v))

    def test_invalid_dropout_args(self):
        # [FUNCTION DOC] Check dropout on an integer tensor raises RuntimeError in both eager and
        # compiled modes.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module doubling input, dropout with probability one half, relu.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Double, dropout, relu.
                x = x * 2
                x = torch.nn.functional.dropout(x, p=0.5)
                x = torch.relu(x)
                return x

        example_inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])

        func = MyModel()
        jit_func = torch.compile(func)
        self.assertRaises(RuntimeError, lambda: func(example_inputs))
        self.assertRaises(RuntimeError, lambda: jit_func(example_inputs))

    def test_nn_param_assign(self):
        # [FUNCTION DOC] Regression test (issue 99569): a module reassigning conv parameters
        # inside forward compiles with full graph and matches eager in inference.
        # https://github.com/pytorch/pytorch/issues/99569
        class Model2(nn.Module):
            # [CLASS DOC] HELPER: Module with conv and batch norm; forward assigns new Parameters
            # to conv weight and bias, sets conv to eval mode, then applies conv, batch norm and
            # relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create conv, batch norm and the replacement tensors.
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
                self.batchnorm = nn.BatchNorm2d(num_features=5)
                self.conv_weight = torch.randn(5, 3, 3, 3)
                self.conv_bias = torch.randn(5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Assign parameters, conv, batch norm, relu.
                self.conv.weight = nn.Parameter(self.conv_weight)
                self.conv.bias = nn.Parameter(self.conv_bias, requires_grad=False)
                self.conv.eval()
                x = self.conv(x)
                x = self.batchnorm(x)
                x = F.relu(x)
                return x

        input_tensor = torch.randn(1, 3, 10, 10)
        func = Model2().to("cpu")

        with torch.no_grad():
            func.train(False)
            v1 = func(input_tensor)
            jit_func = torch.compile(func, fullgraph=True)
            v2 = jit_func(input_tensor)
            self.assertEqual(v1, v2)

    def test_nn_param_assign_wrapped(self):
        # [FUNCTION DOC] Same as the parameter-assign test but the module is called through a
        # functools-wrapped function.
        class Model2(nn.Module):
            # [CLASS DOC] HELPER: Module with conv and batch norm; forward assigns new Parameters
            # to conv weight and bias, sets conv to eval mode, then applies conv, batch norm and
            # relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create conv, batch norm and the replacement tensors.
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
                self.batchnorm = nn.BatchNorm2d(num_features=5)
                self.conv_weight = torch.randn(5, 3, 3, 3)
                self.conv_bias = torch.randn(5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Assign parameters, conv, batch norm, relu.
                self.conv.weight = nn.Parameter(self.conv_weight)
                self.conv.bias = nn.Parameter(self.conv_bias, requires_grad=False)
                self.conv.eval()
                x = self.conv(x)
                x = self.batchnorm(x)
                x = F.relu(x)
                return x

        input_tensor = torch.randn(1, 3, 10, 10)
        func = Model2().to("cpu")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Forward to the wrapped module.
            return func(*args, **kwargs)

        with torch.no_grad():
            func.train(False)
            v1 = func(input_tensor)
            jit_func = torch.compile(wrapper, fullgraph=True)
            v2 = jit_func(input_tensor)
            self.assertEqual(v1, v2)

    @config.patch(inplace_buffers=True)
    def test_in_out_buffer(self):
        # [FUNCTION DOC] Check a matmul scaled by a constant uses an in-out pointer with inplace
        # buffers enabled and matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Matmul with transposed y divided by 8.
            z = torch.matmul(x, y.transpose(-1, -2)) / 8.0
            return z

        inps = [torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)]
        fn_opt = torch.compile(fn, backend="inductor")
        _, code = run_and_get_cpp_code(fn_opt, *inps)
        self.assertTrue("in_out_ptr" in code)
        self.assertEqual(fn_opt(*inps), fn(*inps))

    def test_eliminate_meaningless_copy(self):
        # [FUNCTION DOC] Check that a permute, clone, view then batched matmul pattern with
        # strided inputs gets its redundant copy eliminated: matches eager and generates exactly
        # one kernel.
        def fn(x1, x2):
            # [FUNCTION DOC] HELPER: Permute, contiguous clone, reshape, batched matmul with x1,
            # plus a permute of the view.
            permute = torch.ops.aten.permute.default(x2, [0, 2, 1, 3])
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            view = torch.ops.aten.view.default(clone, [1024, -1, 32])
            bmm = torch.ops.aten.bmm.default(view, x1)
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            return (bmm, permute)

        metrics.reset()
        self.common(
            fn,
            [
                rand_strided(
                    (1024, 32, 128), (4096, 1, 32), device="cpu", dtype=torch.float32
                ),
                rand_strided(
                    (64, 128, 16, 32),
                    (65536, 512, 32, 1),
                    device="cpu",
                    dtype=torch.float32,
                ),
            ],
        )
        self.assertEqual(metrics.generated_kernel_count, 1)

    def test_relu_permute_reshape_reinterpret_view(self):
        # [FUNCTION DOC] Check relu followed by permute and reshape on channels-last input gives
        # one kernel and no transpose helper in code, and equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Relu, permute to channels last order, reshape to batch by
            # spatial by channel.
            n, c, h, w = x.shape
            return torch.relu(x).permute(0, 2, 3, 1).reshape(n, h * w, c)

        x = torch.randn(2, 32, 4, 4).to(memory_format=torch.channels_last)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated kernel
            self.assertEqual(metrics.generated_kernel_count, 1)
            # check that there is no transpose
            FileCheck().check_count("transpose_mxn", 0, exactly=True).run(code)

    def test_attention_size_mismatch(self):
        # [FUNCTION DOC] Check an attention module with an inverse-scale parameter, using
        # transposed key, matches eager under no-grad (size mismatch regression).
        class Attention(torch.nn.Module):
            # [CLASS DOC] HELPER: Single-head-capable attention module with query, key, value
            # linear layers and a fixed inverse scale parameter.
            def __init__(self, hidden_size, num_heads):
                # [FUNCTION DOC] HELPER: Create the three linear layers, head size and inverse
                # scale parameter.
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_size = hidden_size // num_heads
                self.query = torch.nn.Linear(hidden_size, hidden_size)
                self.key = torch.nn.Linear(hidden_size, hidden_size)
                self.value = torch.nn.Linear(hidden_size, hidden_size)
                self.inv_scale = torch.nn.Parameter(
                    torch.Tensor([1 / self.head_size**0.5]), requires_grad=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Project to query, key, value, split into heads, scaled
                # softmax attention and weighted sum.
                query = self.query(x)
                key = self.key(x)
                value = self.value(x)
                (batch_size, seq_len, hidden_size) = query.size()
                query = query.view(
                    batch_size, seq_len, self.num_heads, self.head_size
                ).permute(0, 2, 1, 3)
                key = key.view(
                    batch_size, seq_len, self.num_heads, self.head_size
                ).permute(0, 2, 3, 1)
                value = value.view(
                    batch_size, seq_len, self.num_heads, self.head_size
                ).permute(0, 2, 1, 3)
                attention_weights = (
                    torch.matmul(query, key).mul(self.inv_scale).softmax(dim=-1)
                )
                output = torch.matmul(attention_weights, value)
                return output

        torch.manual_seed(123)
        hidden_size = 16
        num_heads = 1
        seq_len = 4
        batch_size = 1
        x = torch.randn(batch_size, seq_len, hidden_size)

        func = Attention(hidden_size, num_heads).to("cpu")

        with torch.no_grad():
            res1 = func(x)
            jit_func = torch.compile(func)
            res2 = jit_func(x)
        self.assertEqual(res1, res2)

    def test_sdpa_closure_mask_recompile(self):
        # [FUNCTION DOC] Regression test: scaled dot product attention with a mask built by
        # composing closure functions (causal and padding masks), first a prefill step then three
        # decode steps with growing key/value caches that force recompilation; outputs and caches
        # match eager.
        # Regression test for closure-composed attention masks on CPU Inductor.
        def causal_fn(b, h, q, kv):
            # [FUNCTION DOC] HELPER: Causal mask predicate: key index not greater than query
            # index.
            return kv <= q

        def padding_fn(padding_mask):
            # [FUNCTION DOC] HELPER: Build a mask predicate indexing a padding mask by batch and
            # key position.
            def inner(b, h, q, kv):
                # [FUNCTION DOC] HELPER: Return the padding value for the given batch and key
                # indices.
                return padding_mask[b, kv]

            return inner

        def and_masks(f1, f2):
            # [FUNCTION DOC] HELPER: Combine two mask predicates with elementwise and.
            def combined(b, h, q, kv):
                # [FUNCTION DOC] HELPER: Evaluate both predicates and combine.
                return f1(b, h, q, kv) & f2(b, h, q, kv)

            return combined

        def make_mask_closure(batch_size, q_len, kv_len, q_offset, padding_mask):
            # [FUNCTION DOC] HELPER: Build the broadcast attention mask tensor for given batch,
            # query/key lengths, query offset and padding mask by evaluating the combined
            # predicate on index grids.
            fn = and_masks(causal_fn, padding_fn(padding_mask))
            b = torch.arange(batch_size)[:, None, None, None]
            h = torch.arange(1)[None, :, None, None]
            q = (torch.arange(q_len) + q_offset)[None, None, :, None]
            kv = torch.arange(kv_len)[None, None, None, :]
            return fn(b, h, q, kv).expand(batch_size, 1, q_len, kv_len)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a bias-free projection that runs scaled dot product
            # attention using the closure mask and optional past key/value caches; returns output,
            # keys and values.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the projection layer.
                super().__init__()
                self.proj = torch.nn.Linear(32, 32, bias=False)

            def forward(self, x, past_k, past_v, padding_mask, q_offset):
                # [FUNCTION DOC] HELPER: Project queries, build keys and values (concatenated with
                # past caches if given), build the mask and run attention; returns reshaped output
                # plus keys and values.
                B, S, _ = x.shape
                q = self.proj(x).view(B, S, 4, 8).transpose(1, 2)
                k = x.view(B, S, 4, 8).transpose(1, 2)
                v = k.clone()
                if past_k is not None:
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)

                mask = make_mask_closure(B, S, k.shape[2], q_offset, padding_mask)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                return out.transpose(1, 2).reshape(B, S, 32), k, v

        torch._dynamo.reset()
        torch.manual_seed(0)

        eager_model = Model().eval()
        compiled_model = Model().eval()
        compiled_model.load_state_dict(eager_model.state_dict())
        compiled = torch.compile(compiled_model, backend="inductor")

        pad_mask = torch.ones(1, 8, dtype=torch.bool)
        pad_mask[0, :2] = False

        # Prefill: seq_len=8
        x = torch.randn(1, 8, 32)
        with torch.no_grad():
            ref_out, ref_k, ref_v = eager_model(x, None, None, pad_mask, q_offset=0)
            out, k, v = compiled(x, None, None, pad_mask, q_offset=0)
        torch.testing.assert_close(out, ref_out)
        torch.testing.assert_close(k, ref_k)
        torch.testing.assert_close(v, ref_v)

        # Decode: seq_len=1 and kv grows, which exercises recompilation.
        for _ in range(3):
            pad_mask = torch.cat([pad_mask, torch.ones(1, 1, dtype=torch.bool)], dim=1)
            x = torch.randn(1, 1, 32)
            with torch.no_grad():
                ref_out, ref_k, ref_v = eager_model(
                    x,
                    ref_k,
                    ref_v,
                    pad_mask,
                    q_offset=ref_k.shape[2],
                )
                out, k, v = compiled(
                    x,
                    k,
                    v,
                    pad_mask,
                    q_offset=k.shape[2],
                )
            torch.testing.assert_close(out, ref_out)
            torch.testing.assert_close(k, ref_k)
            torch.testing.assert_close(v, ref_v)

    def test_issue_181624_cpu_fusion_with_constant_index_expr(self):
        # [FUNCTION DOC] Regression test (issue 181624): hardswish, unfold, scaled dot product
        # attention, roll, layer norm and nan_to_num fuse correctly with constant index
        # expressions; matches eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module chaining the listed operations.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create hardswish and layer norm submodules.
                super().__init__()
                self.hardswish = torch.nn.Hardswish()
                self.layernorm = torch.nn.LayerNorm([2])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply hardswish, unfold on dim 1, self-attention, roll,
                # layer norm, nan_to_num.
                out = self.hardswish(x)
                out = out.unfold(1, 2, 1)
                out = F.scaled_dot_product_attention(out, out, out)
                out = torch.roll(out, 1, 2)
                out = self.layernorm(out)
                return torch.nan_to_num(out)

        torch.manual_seed(0)
        model = Model().eval()
        x = torch.randn([2, 6])

        with torch.no_grad():
            expected = model(x)
            actual = torch.compile(model, backend="inductor")(x)

        torch.testing.assert_close(actual, expected)

    def test_scalar_mul_bfloat16(self):
        # [FUNCTION DOC] Check multiplying bf16 by a float scalar gives one vector kernel and
        # matches eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Multiply tensor by a scalar via the aten op.
            return torch.ops.aten.mul.Tensor(x, 1.7015043497085571)

        metrics.reset()
        x = torch.randn(4, 5, dtype=torch.bfloat16)
        self.common(f, (x,))
        check_metrics_vec_kernel_count(1)

    def test_bf16_zeros(self):
        # [FUNCTION DOC] Check a function with no inputs creating bf16 zeros compiles.
        def fn():
            # [FUNCTION DOC] HELPER: Return a bf16 zeros tensor.
            x = torch.zeros(1, 1, 32, dtype=torch.bfloat16)
            return x

        self.common(fn, ())

    def test_select_tiliing_with_index_expr(self):
        # [FUNCTION DOC] Check tiling selection with index expressions: view, permute, multiply
        # and constant pad on large tensors matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: View and permute x, multiply with y, then pad one row at the
            # front of dim 2.
            x = torch.ops.aten.view.default(x, [8, 8, 8, 3136])
            x = torch.ops.aten.permute.default(x, [0, 1, 3, 2])
            y = torch.ops.aten.mul.Tensor(y, x)
            return torch.ops.aten.constant_pad_nd.default(y, [0, 0, 1, 0, 0, 0], 0.0)

        x = torch.randn(8, 64, 56, 56)
        y = torch.randn(8, 8, 3136, 8)
        self.common(fn, (x, y))

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch(freezing=True)
    def test_linear_with_no_default_contiguous_input(self):
        # [FUNCTION DOC] Check a linear on an input that is contiguous but with zero strides
        # (built with as_strided), frozen, for float32 and supported low precision types.
        dtypes = [
            torch.float32,
        ]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        mod = torch.nn.Sequential(torch.nn.Linear(16, 16)).eval()
        temp = torch.randn(1, 16, 1, 1)
        v = torch.as_strided(temp, [1, 16], [0, 1], 0)
        self.assertTrue(v.is_contiguous())
        for dtype in dtypes:
            with torch.no_grad():
                self.common(
                    mod.to(dtype),
                    (v.to(dtype),),
                )

    @patch("torch.cuda.is_available", lambda: False)
    @config.patch(freezing=True)
    def test_linear_with_reshape(self):
        # [FUNCTION DOC] Check a bias-free linear followed by a view with freezing generates zero
        # kernels (handled by the packed linear alone).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a bias-free linear and a reshape to 4x4x4.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Linear then reshape.
                x = self.linear(x)
                return x.view(4, 4, 4)

        mod = M().eval()
        v = torch.randn(4, 16)
        with torch.no_grad():
            torch._dynamo.reset()
            metrics.reset()
            self.common(
                mod,
                (v,),
            )
            if metrics.generated_kernel_count != 0:
                raise AssertionError(
                    f"Expected generated_kernel_count == 0, got {metrics.generated_kernel_count}"
                )

    @config.patch(implicit_fallbacks=True)
    def test_aten_normal_dtype(self):
        # [FUNCTION DOC] Check the dtype of torch.normal output for float64, float16 and default
        # dtype, under both decomposition-partition and inductor backends.
        for dtype in [torch.float64, torch.float16, None]:

            def fn():
                # [FUNCTION DOC] HELPER: Draw a 10x10 normal sample with the loop dtype.
                return torch.normal(2, 3, (10, 10), dtype=dtype, device="cpu")

            self.assertEqual(
                torch.compile(fn, backend="aot_eager_decomp_partition")().dtype,
                dtype if dtype else torch.float32,
            )
            self.assertEqual(
                torch.compile(fn, backend="inductor")().dtype,
                dtype if dtype else torch.float32,
            )

    def test_group_norm_vec(self):
        # [FUNCTION DOC] Check group norm vectorization for each vector dtype, memory format and
        # static/dynamic shapes: three vector kernels, and for channels-last no non-contiguous
        # load arrays in code.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a group norm of 3 groups over 90 channels.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the group norm.
                super().__init__()
                self.group_norm = torch.nn.GroupNorm(3, 90)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply group norm.
                return self.group_norm(x)

        options = itertools.product(
            vec_dtypes, [torch.contiguous_format, torch.channels_last], [True, False]
        )
        for dtype, fmt, dynamic in options:
            torch._dynamo.reset()
            metrics.reset()
            mod = M().eval()
            x = torch.randn((2, 90, 6, 6), dtype=dtype).to(memory_format=fmt)
            with torch.no_grad():
                expected = mod(x)
                compiled_m = torch.compile(mod, dynamic=dynamic)
                actual, code = run_and_get_cpp_code(compiled_m, x)
                self.assertEqual(expected, actual)
                # 3 generated kernels (first one for var_mean, last two for result)
                check_metrics_vec_kernel_count(3)

                # check loop split optimization
                if fmt == torch.channels_last:
                    # check that there are no non_contiguous loads
                    FileCheck().check_count(
                        "__at_align__ std::array", 0, exactly=True
                    ).run(code)

    @requires_vectorization
    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    def test_group_norm_sum_conv1d_tail_reduction_store(self):
        # [FUNCTION DOC] Regression test (issue 181618): group norm, sum, and two conv1d layers;
        # checks the code uses the masked reduce helper and stores the correct tail width in the
        # vector accumulator.
        # https://github.com/pytorch/pytorch/issues/181618
        torch.manual_seed(0)
        m_norm = nn.GroupNorm(1, 9).eval()
        m_conv1 = nn.Conv1d(9, 1, 3)
        m_conv2 = nn.Conv1d(1, 7, 3)
        x = torch.randn(8, 9, 14, 12)

        def model():
            # [FUNCTION DOC] HELPER: Group norm, sum over last dim, then two conv1d layers.
            out = m_norm(x)
            out = out.sum(dim=3)
            out = m_conv1(out)
            return m_conv2(out)

        expected = model()
        actual, code = run_and_get_cpp_code(torch.compile(model, backend="inductor"))

        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-3)
        self.assertIn("sum_masked_reduce", code)
        vec_width = cpu_vec_isa.pick_vec_isa().nelements()
        tail_width = 9 if vec_width > 9 else 9 % vec_width or vec_width
        self.assertIn(
            f"tmp_acc0_vec.store(tmpbuf.data(), static_cast<int64_t>({tail_width}L));",
            code,
        )

    @unittest.skipIf(
        os.getenv("ATEN_CPU_CAPABILITY") == "default",
        "Failing in periodic nogpu_NO_AVX2, see #150059 for example",
    )
    def test_group_norm_large_input(self):
        # [FUNCTION DOC] Check group norm on large inputs in two memory formats: three vector
        # kernels, no outer loop fusion and one parallel reduction. Skipped when capability is
        # forced to default.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a group norm of 2 groups over 64 channels.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the group norm.
                super().__init__()
                self.group_norm = torch.nn.GroupNorm(2, 64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply group norm.
                return self.group_norm(x)

        for fmt in [torch.contiguous_format, torch.channels_last]:
            torch._dynamo.reset()
            metrics.reset()
            mod = M().eval()
            x = torch.randn(2, 64, 168, 168).to(memory_format=fmt)
            with torch.no_grad():
                expected = mod(x)
                compiled_m = torch.compile(mod)
                actual = compiled_m(x)
                self.assertEqual(expected, actual)
                # 3 generated kernels (first one for var_mean, last two for result)
                check_metrics_vec_kernel_count(3)
                # check that there is no outer loop fusion.
                self.assertEqual(
                    len(metrics.cpp_outer_loop_fused_inner_counts),
                    0,
                )
                # check for parallel reduction.
                self.assertEqual(metrics.parallel_reduction_count, 1)

    @unittest.skipIf(
        os.getenv("ATEN_CPU_CAPABILITY") == "default",
        "Failing in periodic nogpu_NO_AVX2, see #150059 for example",
    )
    def test_group_norm_large_size(self):
        # [FUNCTION DOC] Regression test (issue 141541): group norm whose reduction size exceeds
        # the cascade summation chunk of 4096, with and without vectorization and dynamic shapes.
        # https://github.com/pytorch/pytorch/issues/141541
        # We are using the chunk size of 4096 for cascade summation,
        # the reduction size of this test case exceeded it.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 32 group, 32 channel group norm.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the group norm.
                super().__init__()
                self.gn = torch.nn.GroupNorm(32, 32)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply group norm.
                return self.gn(x)

        for simdlen, dynamic in itertools.product([None, 0], [True, False]):
            with config.patch({"cpp.simdlen": simdlen}):
                torch._dynamo.reset()
                metrics.reset()
                mod = M().eval()
                x = torch.randn(1, 32, 128, 128, 128)
                with torch.no_grad():
                    expected = mod(x)
                    compiled_m = torch.compile(mod, dynamic=dynamic)
                    actual = compiled_m(x)
                    self.assertEqual(expected, actual)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @config.patch(emulate_precision_casts=True)
    def test_group_norm_backward_symint_divisible_channels(self):
        # [FUNCTION DOC] Check backward of a group norm with symbolic sizes where channels divide
        # evenly, followed by max and sigmoid, with precision cast emulation; compares output and
        # all gradients with eager.
        def fn(x, weight, bias):
            # [FUNCTION DOC] HELPER: Group norm with one group, max over dim 0 values, sigmoid.
            y = torch.nn.functional.group_norm(x, 1, weight=weight, bias=bias)
            return torch.sigmoid(y.max(dim=0).values)

        torch._dynamo.reset()
        metrics.reset()

        shape = (2, 33, 4, 5)
        x_ref = torch.rand(shape, dtype=torch.float32, requires_grad=True)
        weight_ref = torch.rand((33,), dtype=torch.float32, requires_grad=True)
        bias_ref = torch.rand((33,), dtype=torch.float32, requires_grad=True)

        x_cmp = x_ref.clone().detach().requires_grad_(True)
        weight_cmp = weight_ref.clone().detach().requires_grad_(True)
        bias_cmp = bias_ref.clone().detach().requires_grad_(True)

        eager_out = fn(x_ref, weight_ref, bias_ref)
        eager_out.sum().backward()

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        compiled_out = compiled(x_cmp, weight_cmp, bias_cmp)
        compiled_out.sum().backward()

        torch.testing.assert_close(compiled_out, eager_out)
        torch.testing.assert_close(x_cmp.grad, x_ref.grad)
        torch.testing.assert_close(weight_cmp.grad, weight_ref.grad)
        torch.testing.assert_close(bias_cmp.grad, bias_ref.grad)

    def test_group_norm_torch_func_grad_dynamic(self):
        # [FUNCTION DOC] Check gradient transform of a pipeline of average pool, layer norm, batch
        # norm and group norm with dynamic shapes matches eager.
        avg_pool = nn.AvgPool2d(2)
        layer_norm = nn.LayerNorm([5])
        batch_norm = nn.BatchNorm2d(12).eval()
        group_norm = nn.GroupNorm(12, 12).eval()

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply pool, layer norm, batch norm, group norm, then mean of
            # absolute values.
            x = avg_pool(x)
            x = layer_norm(x)
            x = batch_norm(x)
            x = group_norm(x)
            return x.abs().mean()

        torch.manual_seed(0)
        x = torch.randn([2, 12, 10, 10])

        expected = torch.func.grad(fn)(x)

        torch._dynamo.reset()
        compiled = torch.compile(torch.func.grad(fn), backend="inductor", dynamic=True)
        actual = compiled(x)

        self.assertEqual(actual, expected)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_backward_dynamic_item_from_size1_tensor(self):
        # [FUNCTION DOC] Check backward through an item taken from a size-one tensor (computed
        # from nonzero count) with dynamic output shapes captured; checks output scalar shape and
        # gradient equality.
        def fn(x, w):
            # [FUNCTION DOC] HELPER: Count nonzero elements, convert to a scalar, return the
            # weighted sum divided by it.
            num = x.nonzero().numel()
            num = x.new_tensor([num]).item()
            y = w * x
            return y.sum() / num

        torch._dynamo.reset()
        metrics.reset()

        x = torch.tensor([0.0, 1.0, -1.0, 0.5])
        w_ref = torch.tensor(0.0, requires_grad=True)
        w_cmp = w_ref.detach().clone().requires_grad_(True)

        eager_out = fn(x, w_ref)
        eager_out.backward()

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        compiled_out = compiled(x, w_cmp)
        compiled_out.backward()

        self.assertEqual(eager_out.shape, torch.Size([]))
        self.assertEqual(compiled_out.shape, eager_out.shape)
        torch.testing.assert_close(compiled_out, eager_out)
        torch.testing.assert_close(w_cmp.grad, w_ref.grad)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_lp_pool2d_symbolic_float_min_norm_type(self):
        # [FUNCTION DOC] Check lp pool 2D whose norm type is a symbolic float derived from the
        # data (clamped with min); compiled equals eager including NaNs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compute norm type from the mean of x, then apply lp pool with
            # kernel size one.
            norm_type = min(4.0, 0.5 + float(x.mean().abs().detach()) * 1.5)
            return F.lp_pool2d(x, norm_type=norm_type, kernel_size=1)

        generator = torch.Generator().manual_seed(0)
        x = torch.randn(1, 1, 4, 4, generator=generator)
        expected = fn(x)
        actual = torch.compile(fn, backend="inductor")(x)

        torch.testing.assert_close(actual, expected, equal_nan=True)

    @config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_cpp_backend_no_error(self):
        # [FUNCTION DOC] Regression test (issue 167205): precision cast emulation used to raise
        # TypeError on the CPP backend; now a power function with thresholding compiles and its
        # vector-Jacobian product returns a float16 gradient.
        """
        See https://github.com/pytorch/pytorch/issues/167205
        emulate_precision_casts threw TypeError on CPP backend.

        Before fix: TypeError: CppVecOverrides.to_dtype() got an unexpected
        keyword argument 'use_compute_types'

        After fix: Should compile and run without error.
        """

        def robust_power(base, exponent, threshold):
            # [FUNCTION DOC] HELPER: Where abs(base) is below threshold use the scaled base else
            # the base raised to exponent.
            threshold1 = threshold
            broadcasted_base = torch.abs(base)
            threshold_bc = threshold.expand_as(base)
            cond = broadcasted_base < threshold_bc
            return torch.where(cond, base / threshold1, base**exponent)

        device = torch.device("cpu")
        base = torch.randn(10, dtype=torch.float16, device=device)
        exponent = torch.tensor(2.0, dtype=torch.float16, device=device)
        threshold = torch.tensor(0.01, dtype=torch.float16, device=device)
        v = torch.ones_like(base)

        # Main test, this should not raise TypeError (before fix it would)
        compiled_fn = torch.compile(robust_power)
        y, (grad_b, grad_e, grad_t) = vjp(
            lambda b, e, t: compiled_fn(b, e, t), (base, exponent, threshold), v=v
        )

        # Sanity check that gradients were computed
        self.assertIsNotNone(grad_b)
        self.assertEqual(grad_b.dtype, torch.float16)

    @config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_preserves_fused_fp16_rounding(self):
        # [FUNCTION DOC] Check precision cast emulation preserves fp16 rounding of a fused where
        # plus add, for each SIMD length, comparing both outputs with eager.
        def fn(x, c):
            # [FUNCTION DOC] HELPER: Choose plus or minus half by a condition, add in fp16, return
            # it with its float sum.
            z = torch.where(c, torch.full_like(x, 0.5), torch.full_like(x, -0.5))
            y = z.to(torch.float16) + x.to(torch.float16)
            return y, y.float().sum()

        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
        c = (torch.arange(24) % 5 == 0).reshape(2, 3, 4)
        expected = fn(x, c)

        for simdlen in simd_lengths_to_test():
            with config.patch({"cpp.simdlen": simdlen}):
                torch._dynamo.reset()
                metrics.reset()
                actual = torch.compile(fn, backend="inductor", fullgraph=True)(x, c)
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[1], expected[1])

    @config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_explicit_fp16_cast_mul_overflow(self):
        # [FUNCTION DOC] Check precision cast emulation for many explicit float16/bfloat16 cast
        # patterns (multiplication overflow, casting chains, aliases such as half, type, type_as,
        # asarray and as_tensor, promoted consumers, cat, copy, scatter, reductions) against
        # eager; one case expects all NaN.
        def check(fn, *args, assert_all_nan=False):
            # [FUNCTION DOC] HELPER: Run eager and compiled versions of a function on given
            # arguments and compare, or check that both results are all NaN when requested.
            expected = fn(*args)

            torch._dynamo.reset()
            actual = torch.compile(fn, backend="inductor", fullgraph=True)(*args)
            if assert_all_nan:
                self.assertTrue(torch.isnan(expected).all())
                self.assertTrue(torch.isnan(actual).all())
            else:
                self.assertEqual(actual, expected)

        def fn(x):
            # [FUNCTION DOC] HELPER: Cast to float16, square, cast back to float32.
            y = x.to(torch.float16)
            return (y * y).to(torch.float32)

        def bfloat16_rounding(x):
            # [FUNCTION DOC] HELPER: Cast to bfloat16, square, cast back to float32.
            y = x.to(torch.bfloat16)
            return (y * y).to(torch.float32)

        def cast_chain(x):
            # [FUNCTION DOC] HELPER: Cast to float16 and back to float32.
            return x.to(torch.float16).to(torch.float32)

        def repeated_lowp_cast(x):
            # [FUNCTION DOC] HELPER: Cast to float16, back to float32, to float16 again, square,
            # convert to float.
            y = x.to(torch.float16)
            z = y.float().to(torch.float16)
            return (z * z).float()

        def repeated_bfloat16_cast(x):
            # [FUNCTION DOC] HELPER: Round trip through bfloat16 and float twice.
            return x.bfloat16().float().bfloat16().float()

        def cast_alias_half(x):
            # [FUNCTION DOC] HELPER: Cast with the half method, square, convert to float.
            y = x.half()
            return (y * y).float()

        def cast_alias_type(x):
            # [FUNCTION DOC] HELPER: Cast with the type method to float16, square, convert to
            # float.
            y = x.type(torch.float16)
            return (y * y).float()

        def cast_alias_type_as(x, h):
            # [FUNCTION DOC] HELPER: Cast with type_as to the dtype of h, square, convert to
            # float.
            y = x.type_as(h)
            return (y * y).float()

        def cast_alias_asarray(x):
            # [FUNCTION DOC] HELPER: Cast with asarray to float16, square, convert to float.
            y = torch.asarray(x, dtype=torch.float16)
            return (y * y).float()

        def cast_alias_as_tensor(x):
            # [FUNCTION DOC] HELPER: Cast with as_tensor to float16, square, convert to float.
            y = torch.as_tensor(x, dtype=torch.float16)
            return (y * y).float()

        def promoted_consumer(x):
            # [FUNCTION DOC] HELPER: Cast to float16 and add the original float32 input.
            y = x.to(torch.float16)
            return y + x

        def promoted_where_branch(x):
            # [FUNCTION DOC] HELPER: Select between the float16 cast and the original by sign of
            # the input.
            y = x.to(torch.float16)
            return torch.where(x > 0, y, x)

        def promoted_cat_input(x):
            # [FUNCTION DOC] HELPER: Concatenate the float16 cast with the original input.
            y = x.to(torch.float16)
            return torch.cat([y, x])

        def promoted_cat_dim_input(x):
            # [FUNCTION DOC] HELPER: Concatenate the float16 cast with the original along dim 1.
            y = x.to(torch.float16)
            return torch.cat([y, x], 1)

        def copy_from_explicit_lowp_cast(x):
            # [FUNCTION DOC] HELPER: Copy a float16 cast into an empty float32 tensor and return
            # it.
            y = x.to(torch.float16)
            out = torch.empty_like(x)
            out.copy_(y)
            return out

        def select_scatter_explicit_lowp_cast(x):
            # [FUNCTION DOC] HELPER: Scatter a float16 cast into a zeros 1x1 tensor at a selected
            # position.
            y = x.to(torch.float16)
            return torch.select_scatter(torch.zeros(1, 1), y, 0, 0)

        def slice_scatter_explicit_lowp_cast(x):
            # [FUNCTION DOC] HELPER: Scatter a float16 cast into a zeros tensor of length two by
            # slice.
            y = x.to(torch.float16)
            return torch.slice_scatter(torch.zeros(2), y, 0, 0, 1)

        def reduction_dtype_explicit_lowp_cast(x):
            # [FUNCTION DOC] HELPER: Return the float32-typed sum and the mean of the float16
            # cast.
            y = x.to(torch.float16)
            return torch.sum(y, dtype=torch.float32), torch.mean(y)

        def mixed_lowp_consumer(x, h):
            # [FUNCTION DOC] HELPER: Cast x to float16, add the float16 tensor h and convert to
            # float.
            y = x.to(torch.float16)
            return (y + h).float()

        def lowp_consumer_before_widening(x):
            # [FUNCTION DOC] HELPER: Cast to float16, square, multiply by zeros and convert to
            # float (overflow gives NaN).
            y = x.to(torch.float16)
            z = y * y
            return (z * torch.zeros_like(y)).float()

        for size in (1, 32):
            check(fn, torch.full((size,), 5000.0))

        check(bfloat16_rounding, torch.tensor([257.0]))
        check(cast_chain, torch.tensor([70000.0]))
        check(repeated_lowp_cast, torch.tensor([5001.0]))
        check(repeated_bfloat16_cast, torch.tensor([2.0039]))

        x_alias = torch.tensor([5000.0])
        h = torch.tensor([1.0], dtype=torch.float16)
        check(cast_alias_half, x_alias)
        check(cast_alias_type, x_alias)
        check(cast_alias_type_as, x_alias, h)
        check(cast_alias_asarray, x_alias)
        check(cast_alias_as_tensor, x_alias)

        x = torch.tensor([70000.0])
        check(promoted_consumer, x)
        check(promoted_where_branch, x)
        check(promoted_cat_input, x)
        check(promoted_cat_dim_input, x.reshape(1, 1))
        check(copy_from_explicit_lowp_cast, x)
        check(select_scatter_explicit_lowp_cast, x)
        check(slice_scatter_explicit_lowp_cast, x)
        check(reduction_dtype_explicit_lowp_cast, x)
        check(mixed_lowp_consumer, torch.tensor([65504.0]), h)
        check(lowp_consumer_before_widening, x, assert_all_nan=True)

    @config.patch(emulate_precision_casts=True)
    def test_predicate_fp16_cast_does_not_round_unrelated_data_path(self):
        # [FUNCTION DOC] Check that a float16 cast used only in a predicate does not change
        # rounding of the unrelated data path: results with and without the cast in the predicate
        # are equal.
        def predicate_cast(x, h, one):
            # [FUNCTION DOC] HELPER: Predicate computed from a float16 cast, select between equal
            # values of a float16 sum.
            c = x.to(torch.float16) > 0
            y = h + one
            return torch.where(c, y, y).to(torch.float32)

        def predicate_no_cast(x, h, one):
            # [FUNCTION DOC] HELPER: Same as above with the predicate computed without a cast.
            c = x > 0
            y = h + one
            return torch.where(c, y, y).to(torch.float32)

        x = torch.tensor([1.0], dtype=torch.float32)
        h = torch.tensor([65504.0], dtype=torch.float16)
        one = torch.tensor([1.0], dtype=torch.float16)

        torch._dynamo.reset()
        actual_with_predicate_cast = torch.compile(
            predicate_cast, backend="inductor", fullgraph=True
        )(x, h, one)

        torch._dynamo.reset()
        actual_without_predicate_cast = torch.compile(
            predicate_no_cast, backend="inductor", fullgraph=True
        )(x, h, one)

        self.assertEqual(actual_with_predicate_cast, actual_without_predicate_cast)

    def test_int_div_vec(self):
        # [FUNCTION DOC] Check vectorized integer division with no rounding, trunc and floor modes
        # for four int dtypes.
        def fn(x, y, mode):
            # [FUNCTION DOC] HELPER: Divide with the given rounding mode.
            return torch.div(x, y, rounding_mode=mode)

        for dtype in [
            torch.int8,
            torch.uint8,
            torch.int32,
            torch.int64,
        ]:
            x = torch.randint(1, 100, (32, 32), dtype=dtype)
            y = torch.randint(1, 100, (32, 32), dtype=dtype)
            for mode in [None, "trunc", "floor"]:
                with torch.no_grad():
                    metrics.reset()
                    self.common(fn, (x, y, mode))
                    check_metrics_vec_kernel_count(1)

    def test_uint8_add(self):
        # [FUNCTION DOC] Regression test (issue 113016): uint8 add, negate and convert to int32.
        # https://github.com/pytorch/pytorch/issues/113016
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add, negate, convert to int32.
            return torch.add(x, y).neg().to(torch.int32)

        x = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        y = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        self.common(fn, (x, y))

    def test_uint8_sub(self):
        # [FUNCTION DOC] Regression test (issue 113016): uint8 subtract, negate and convert to
        # int32.
        # https://github.com/pytorch/pytorch/issues/113016
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Subtract, negate, convert to int32.
            return torch.sub(x, y).neg().to(torch.int32)

        x = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        y = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        self.common(fn, (x, y))

    @xfailIf(IS_ARM64)  # see https://github.com/pytorch/pytorch/issues/168972
    def test_float32_to_uint8(self):
        # [FUNCTION DOC] Regression test (issue 156788): negative floats converted to uint8 match
        # eager. Expected to fail on ARM64 (issue 168972).
        # https://github.com/pytorch/pytorch/issues/156788
        @torch.compile
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to uint8.
            return x.to(torch.uint8)

        x = torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float32, device="cpu")
        self.assertEqual(
            x.to(torch.uint8),
            fn(x),
            msg=lambda msg: f"{msg}\nExpected {x.to(torch.uint8)} but got {fn(x)}",
        )

    def test_non_contiguous_reduction_store(self):
        # [FUNCTION DOC] Regression test (issue 113018): max over a dim followed by a conv with
        # non-contiguous reduction store.
        # https://github.com/pytorch/pytorch/issues/113018
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a conv to the max values over dim 3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the conv layer.
                super().__init__()
                self.conv = torch.nn.Conv2d(39, 1, kernel_size=(1, 17), stride=(2, 2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Max over dim 3 values then conv.
                return self.conv(x.max(3).values)

        m = M()
        x = torch.randn(1, 39, 1, 18, 17)
        self.common(m, (x,))

    def test_embedding_vec(self):
        # [FUNCTION DOC] Check embedding lookup plus an add yields one vector kernel.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an embedding that adds its output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the embedding.
                super().__init__()
                self.emb = torch.nn.Embedding(64, 128)

            def forward(self, idx, x):
                # [FUNCTION DOC] HELPER: Embedding of idx plus x.
                return self.emb(idx) + x

        idx = torch.randint(0, 64, (4, 32))
        x = torch.randn(4, 32, 128)
        m = M().eval()
        with torch.no_grad():
            metrics.reset()
            self.common(m, (idx, x))
            check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_embedding_vec_bf16(self):
        # [FUNCTION DOC] Check embedding lookup with bf16 data yields one vector kernel, and
        # generated code uses vector loads without low precision conversion helpers (direct load
        # and store).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the embedding of idx (x unused).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the embedding.
                super().__init__()
                self.emb = torch.nn.Embedding(64, 128)

            def forward(self, idx, x):
                # [FUNCTION DOC] HELPER: Embedding of idx.
                return self.emb(idx)

        idx = torch.randint(0, 64, (4, 32))
        x = torch.randn(4, 32, 128).to(torch.bfloat16)
        m = M().eval()
        with torch.no_grad():
            metrics.reset()
            self.common(m, (idx, x))
            check_metrics_vec_kernel_count(1)

        # we are doing direct load/store, make sure we do not generate
        # redundant type casts
        m_opt = torch.compile(m)
        _, code = run_and_get_cpp_code(m_opt, idx, x)
        self.assertTrue("Vectorized" in code)
        self.assertTrue("cvt_lowp_fp_to_fp32" not in code)
        self.assertTrue("cvt_fp32_to_lowp_fp" not in code)

    @xfailIfS390X
    def test_concat_inner_vec(self):
        # [FUNCTION DOC] Check relu of concatenation along dim 1 gives three vector kernels.
        # Expected to fail on s390x.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Relu of concat on dim 1.
            return F.relu(torch.cat([x, y], dim=1))

        x = torch.randn(32, 35)
        y = torch.randn(32, 120)
        metrics.reset()
        self.common(fn, (x, y))
        check_metrics_vec_kernel_count(3)

    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_expr_vec_non_contiguous(self):
        # [FUNCTION DOC] Check vectorization of non-contiguous expressions in a
        # pad/reshape/expand/permute/softmax pattern from sebotnet: four vector kernels and no
        # integer vector load through a temp buffer.
        def fn(x):
            # [FUNCTION DOC] HELPER: Pad, reshape, slice, expand, permute, copy, view and softmax
            # on the last dim.
            # the pattern from sebotnet33ts_256
            y = torch.nn.functional.pad(x, (0, 31)).reshape(-1, 33, 63)
            y = y[:, :32, 31:].reshape(4, 32, 1, 32, 32).expand(-1, -1, 32, -1, -1)
            y = y.permute(0, 3, 1, 4, 2).clone(memory_format=torch.contiguous_format)
            y = y.view(4, 1024, 1024)
            return y.softmax(dim=-1)

        x = torch.randn(128, 2048)
        opt_fn = torch.compile(fn)
        metrics.reset()
        _, code = run_and_get_cpp_code(opt_fn, x)
        self.assertTrue(same(fn(x), opt_fn(x)))
        # 4 kernels for max, exp, sum and div
        check_metrics_vec_kernel_count(4)
        FileCheck().check_count(
            "Vectorized<int>::loadu(tmpbuf.data())", 0, exactly=True
        ).run(code)

    def test_vec_contiguous_ModularIndexing(self):
        # [FUNCTION DOC] Regression test (issue 114488): contiguous modular indexing in a swin
        # patch-merge pattern followed by layer norm; four vector kernels and no float vector load
        # through a temp buffer.
        # https://github.com/pytorch/pytorch/issues/114488
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that merges 2x2 patches via reshape, permute and flatten
            # then layer norm.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Create the layer norm over four times dim.
                super().__init__()
                self.norm = torch.nn.LayerNorm(dim * 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Rearrange patches and apply layer norm.
                # the pattern from swin_base_patch4_window7_224
                B, H, W, C = x.shape
                x = (
                    x.reshape(B, H // 2, 2, W // 2, 2, C)
                    .permute(0, 1, 3, 4, 2, 5)
                    .flatten(3)
                )
                x = self.norm(x)
                return x

        x = torch.randn(1, 56, 56, 128)
        m = M(128)
        opt_m = torch.compile(m)
        with torch.no_grad():
            metrics.reset()
            _, code = run_and_get_cpp_code(opt_m, x)
            self.assertTrue(same(m(x), opt_m(x)))
            # Two kernels: one for reduction, one pointwises
            check_metrics_vec_kernel_count(4)
            FileCheck().check_count(
                "Vectorized<float>::loadu(tmpbuf.data())", 0, exactly=True
            ).run(code)

    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("shape", ("15,3,13", "4,2048,4096"))
    def test_fp8_cast(self, dtype: torch.dtype, shape: str):
        # [FUNCTION DOC] Parametrized by dtype and shape: casts to float8 e4m3 and e5m2 and back,
        # comparing with eager.
        def fp8_cast(x):
            # [FUNCTION DOC] HELPER: Round trip through both float8 types, returning both results.
            y0 = x.to(dtype=torch.float8_e4m3fn).to(dtype)
            y1 = x.to(dtype=torch.float8_e5m2).to(dtype)
            return y0, y1

        shape = [int(dim) for dim in shape.split(",")]
        x = torch.rand(*shape, device="cpu", dtype=dtype)
        self.common(fp8_cast, (x,))

    def test_logical_op_store_to_lowp_data_dtype(self):
        # [FUNCTION DOC] Regression test (issues 117624 and 117627): logical or and xor writing
        # into low-precision output tensors via the out argument, for bf16 and fp16, under
        # no-grad.
        # https://github.com/pytorch/pytorch/issues/117624
        # https://github.com/pytorch/pytorch/issues/117627
        def fn(out1, out2, input, other):
            # [FUNCTION DOC] HELPER: Compute logical or and logical xor of input and other into
            # the given outputs.
            o1 = torch.logical_or(out=out1, input=input, other=other)
            o2 = torch.logical_xor(out=out2, input=input, other=other)
            return o1, o2

        x = torch.rand([3, 3, 2, 8, 9, 2], dtype=torch.float)
        y = torch.rand([3, 3, 2, 8, 9, 2], dtype=torch.float)
        for dtype in _lowp_fp_dtypes:
            o1 = torch.rand([3, 3, 2, 8, 9, 2], dtype=dtype)
            o2 = torch.rand([3, 3, 2, 8, 9, 2], dtype=dtype)
            with torch.no_grad():
                self.common(fn, (o1, o2, x, y))

    def test_constant_bool_vec(self):
        # [FUNCTION DOC] Check a constant boolean mask selecting between input and a scalar gives
        # one vector kernel, and with a tail-length input both main and tail loops use vector mask
        # construction (two uses).
        def fn(x):
            # [FUNCTION DOC] HELPER: Select x where a zero bool mask holds, else -1.
            mask = torch.zeros(1, dtype=torch.bool)
            return torch.where(mask, x, -1.0)

        x = torch.rand(1000)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.rand(31)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VecMask<float,1>::from", 2, exactly=True
                ).run(code)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_symbolic_shape_scalar_value_reduction(self):
        # [FUNCTION DOC] Check a function adding a scalar sum of ones (size from a dynamic
        # integer) to a tensor under dynamic shapes gives two vector kernels.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add the sum of a ones tensor of size x to y.
            return y + torch.ones(x).sum()

        with torch.no_grad():
            metrics.reset()
            y = torch.randn(100)
            self.common(fn, (100, y))
            check_metrics_vec_kernel_count(2)

    def test_int32_pointwise_vec(self):
        # [FUNCTION DOC] Check int32 elementwise multiply is vectorized (one kernel), and for a
        # non-multiple size both main and tail loops use vector loads.
        def fn(x):
            # [FUNCTION DOC] HELPER: Square elementwise.
            return x * x

        x = torch.randint(0, 100, (32, 32), dtype=torch.int32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randint(0, 100, (37, 37), dtype=torch.int32)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::Vectorized<int32_t>::loadu", 2, exactly=True
                ).run(code)

    def test_int32_reduction_vec(self):
        # [FUNCTION DOC] Check int32 sum along dim 1 is vectorized, including vector loads in both
        # main and tail loops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sum along dim 1.
            return x.sum(dim=1)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randint(0, 100, (37, 37), dtype=torch.int32)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::Vectorized<int32_t>::loadu", 2, exactly=True
                ).run(code)

    def test_uint32_pointwise_vec(self):
        # [FUNCTION DOC] Check uint32 multiply currently generates no vector kernel (vectorized
        # uint32 load pending).
        def fn(x):
            # [FUNCTION DOC] HELPER: Square elementwise.
            return x * x

        x = torch.randint(0, 100, (32, 32), dtype=torch.uint32)
        metrics.reset()
        self.common(fn, (x,))
        # TODO(jgong5): change to 1 with vectorized uint32 load
        if metrics.generated_cpp_vec_kernel_count != 0:
            raise AssertionError(
                f"Expected generated_cpp_vec_kernel_count == 0, got {metrics.generated_cpp_vec_kernel_count}"
            )

    def test_uint32_reduction_vec(self):
        # [FUNCTION DOC] Check uint32 sum along dim 1 currently generates no vector kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sum along dim 1.
            return x.sum(dim=1)

        x = torch.randint(0, 100, (32, 32), dtype=torch.uint32)
        metrics.reset()
        self.common(fn, (x,))
        # TODO(jgong5): change to 1 with vectorized uint32/uint64 load
        if metrics.generated_cpp_vec_kernel_count != 0:
            raise AssertionError(
                f"Expected generated_cpp_vec_kernel_count == 0, got {metrics.generated_cpp_vec_kernel_count}"
            )

    def test_int64_pointwise_vec(self):
        # [FUNCTION DOC] Check int64 elementwise multiply is vectorized with a two-register vector
        # type in main and tail.
        def fn(x):
            # [FUNCTION DOC] HELPER: Square elementwise.
            return x * x

        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randint(0, 100, (37, 37), dtype=torch.int64)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VectorizedN<int64_t,2>::loadu", 2, exactly=True
                ).run(code)

    def test_int64_reduction_vec(self):
        # [FUNCTION DOC] Check int64 sum along dim 1 is vectorized with main and tail vector
        # loads.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sum along dim 1.
            return x.sum(dim=1)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randint(0, 100, (37, 37), dtype=torch.int64)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VectorizedN<int64_t,2>::loadu", 2, exactly=True
                ).run(code)

    @requires_vectorization
    def test_indirect_assert_scalar_mask_tail_vec_no_crash(self):
        # [FUNCTION DOC] Regression test (issue 178136): indirect index assertion with scalar mask
        # in a vector tail loop does not crash; loop tail vectorization is enabled and result
        # matches eager.
        # https://github.com/pytorch/pytorch/issues/178136
        def fn(positions, cache):
            # [FUNCTION DOC] HELPER: Gather rows from cache by positions, clone the first and
            # overwrite strided columns with those of the second and third.
            x = cache[positions]
            y = x[0].clone()
            y[..., 1::3] = x[1, ..., 1::3]
            y[..., 2::3] = x[2, ..., 2::3]
            return y

        positions = torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.int64)
        cache = torch.arange(3, dtype=torch.float32).reshape(1, 3)

        with config.patch({"cpp.enable_loop_tail_vec": True}):
            expected = fn(positions, cache)
            compiled_fn = torch.compile(fn, fullgraph=True)
            actual = compiled_fn(positions, cache)

        torch.testing.assert_close(actual, expected)

    def test_uint64_pointwise_vec(self):
        # [FUNCTION DOC] Check uint64 multiply currently generates no vector kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Square elementwise.
            return x * x

        x = torch.randint(0, 100, (32, 32), dtype=torch.uint64)
        metrics.reset()
        self.common(fn, (x,))
        # TODO(jgong5): change to 1 with vectorized uint64 load
        if metrics.generated_cpp_vec_kernel_count != 0:
            raise AssertionError(
                f"Expected generated_cpp_vec_kernel_count == 0, got {metrics.generated_cpp_vec_kernel_count}"
            )

    def test_uint64_reduction_vec(self):
        # [FUNCTION DOC] Check uint64 sum along dim 1 currently generates no vector kernel.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sum along dim 1.
            return x.sum(dim=1)

        x = torch.randint(0, 100, (32, 32), dtype=torch.uint64)
        metrics.reset()
        self.common(fn, (x,))
        # TODO(jgong5): change to 1 with vectorized uint64 load
        if metrics.generated_cpp_vec_kernel_count != 0:
            raise AssertionError(
                f"Expected generated_cpp_vec_kernel_count == 0, got {metrics.generated_cpp_vec_kernel_count}"
            )

    def test_convert_int8_to_half_vec(self):
        # [FUNCTION DOC] Check vectorized conversion from int8 and uint8 to bf16 and half for each
        # valid SIMD width.
        src_dtypes = [torch.int8, torch.uint8]
        dst_dtypes = [torch.bfloat16, torch.half]
        _simd_lens = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()]
        for src_dtype, dst_dtype, _simd_len in itertools.product(
            src_dtypes, dst_dtypes, _simd_lens
        ):

            def fn(x):
                # [FUNCTION DOC] HELPER: Convert to the loop's destination dtype.
                return x.to(dst_dtype)

            low = 0 if src_dtype == torch.uint8 else -100

            x = torch.randint(low, 100, (32, 32), dtype=src_dtype)
            with config.patch({"cpp.simdlen": _simd_len}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    def test_convert_int32_to_int64_vec(self):
        # [FUNCTION DOC] Check vectorized int32 to int64 conversion.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to int64.
            return x.to(torch.int64)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_convert_int64_to_int32_vec(self):
        # [FUNCTION DOC] Check vectorized int64 to int32 conversion.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to int32.
            return x.to(torch.int32)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_convert_fp32_to_int64_vec(self):
        # [FUNCTION DOC] Check vectorized float32 to int64 conversion.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to int64.
            return x.to(torch.int64)

        x = torch.rand(32, 32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_convert_int64_to_fp32_vec(self):
        # [FUNCTION DOC] Check vectorized int64 to float32 conversion.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to float32.
            return x.to(torch.float32)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_double_pointwise_vec(self):
        # [FUNCTION DOC] Check double elementwise multiply is vectorized, with main and tail
        # vector loads.
        def fn(x):
            # [FUNCTION DOC] HELPER: Square elementwise.
            return x * x

        x = torch.randn((32, 32), dtype=torch.double)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randn((37, 37), dtype=torch.double)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VectorizedN<double,2>::loadu", 2, exactly=True
                ).run(code)

    def test_double_reduction_vec(self):
        # [FUNCTION DOC] Check double sum along dim 1 is vectorized, with main and tail vector
        # loads.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sum along dim 1.
            return x.sum(dim=1)

        x = torch.randn((32, 32), dtype=torch.double)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randn((37, 37), dtype=torch.double)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VectorizedN<double,2>::loadu", 2, exactly=True
                ).run(code)

    @unittest.skipIf(
        get_gcc_major_version() == 13,
        "Fails under GCC 13 due to vector codegen (passes with GCC 11)",
    )
    def test_convert_fp32_to_double_vec(self):
        # [FUNCTION DOC] Check vectorized float32 to double conversion in main and tail loops.
        # Skipped under GCC 13.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to double.
            return x.to(torch.double)

        x = torch.randn(32, 32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randn(37, 37)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::convert<double,2,float,1>", 2, exactly=True
                ).run(code)

    def test_convert_double_to_fp32_vec(self):
        # [FUNCTION DOC] Check vectorized double to float32 conversion in main and tail loops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to float32.
            return x.to(torch.float32)

        x = torch.randn((32, 32), dtype=torch.double)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randn((37, 37), dtype=torch.double)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::convert<float,1,double,2>", 2, exactly=True
                ).run(code)

    def test_no_redundant_to_dtypes_between_fused_scheduler_node(self):
        # [FUNCTION DOC] Regression test (issue 115260): redundant dtype conversions between fused
        # scheduler nodes for a half precision concat, max, multiply and tan model.
        # https://github.com/pytorch/pytorch/issues/115260
        p0 = torch.tensor([1.0879], dtype=torch.float16)

        class Model1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating four inputs on dim 2, taking a max with a
            # fixed half tensor, multiplying, and returning the product and its tangent.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Empty constructor.
                super().__init__()

            def forward(self, *args):
                # [FUNCTION DOC] HELPER: Concat args 3, 2, 1, 0, multiply by max of arg 4 and the
                # outer constant, return product and tan.
                cat = torch.cat((args[3], args[2], args[1], args[0]), dim=2)
                max_1 = torch.max(args[4], p0)
                mul = torch.mul(cat, max_1)
                tan = torch.tan(mul)
                return (mul, tan)

        metrics.reset()
        m = Model1()
        self.common(
            m,
            (
                torch.randn((17, 5, 1, 7)).half(),
                torch.randn((17, 5, 1, 7)).half(),
                torch.randn((17, 5, 11, 7)).half(),
                torch.randn((17, 5, 1, 7)).half(),
                torch.tensor(4.39, dtype=torch.float16),
            ),
        )

    def test_masked_load_int64_vec(self):
        # [FUNCTION DOC] Regression test (issue 120377): masked load of int64 via padding is
        # vectorized (one kernel).
        # https://github.com/pytorch/pytorch/issues/120377
        def fn(x):
            # [FUNCTION DOC] HELPER: Pad 13 zeros at the end.
            return torch.nn.functional.pad(x, (0, 13))

        x = torch.randint(0, 100, (819,), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_masked_handle_scalar_var(self):
        # [FUNCTION DOC] Reduced reproducer of issue 173626: builds a shifted-window attention
        # mask with slices, views, permutes and masked fills (no input); expects one vector
        # kernel.
        def fn():
            # [FUNCTION DOC] HELPER: Build the mask image by labeling nine regions,
            # window-partition it, take pairwise differences, and replace nonzero with -100 and
            # zero with 0.
            # Simplified reproducer of https://github.com/pytorch/pytorch/issues/173626
            window_size = 7
            shift_size = 3
            height, width = 14, 14
            img_mask = torch.zeros((1, height, width, 1), dtype=torch.float32)
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            count = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = count
                    count += 1

            masked_window = img_mask.view(
                1,
                height // window_size,
                window_size,
                width // window_size,
                window_size,
                1,
            )
            masked_window = (
                masked_window.permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .view(-1, window_size * window_size)
            )
            attn_mask = masked_window.unsqueeze(1) - masked_window.unsqueeze(2)
            res = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
            return res

        metrics.reset()
        self.common(fn, ())
        check_metrics_vec_kernel_count(1)

    def test_highp_to_lowp_cse_var_cache_with_store(self):
        # [FUNCTION DOC] Regression test (issue 128263): CSE variable cache when high precision to
        # low precision conversion is combined with a store, using a realized half precision
        # intermediate.
        # Fix issue: https://github.com/pytorch/pytorch/issues/128263
        input = torch.randn(5, 128, dtype=torch.float32)
        input2 = torch.randint(0, 10, (5, 128), dtype=torch.int8)
        input3 = torch.randn(128, 128, dtype=torch.float32)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module converting int8 input, realizing a half tensor, returning
            # matmul with half x3 and float product of the intermediate with x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Empty constructor.
                super().__init__()

            def forward(self, x, x2, x3):
                # [FUNCTION DOC] HELPER: Convert x2 to int32 then half, realize it, matmul with
                # half x3, and multiply the float version of the intermediate by x; returns both.
                x2 = x2.to(torch.int32)
                temp = test_operators.realize(x2.to(torch.float16))
                temp2 = temp.to(torch.float32)
                temp2 = temp2 * x
                return torch.mm(temp, x3.to(torch.float16)), temp2

        metrics.reset()
        m = Model()
        self.common(
            m,
            (input, input2, input3),
        )

    def test_reduction_float_to_int64(self):
        # [FUNCTION DOC] Regression test (issue 124821): max reduction along dim 0 of int64 is
        # vectorized.
        # https://github.com/pytorch/pytorch/issues/124821
        def fn(x):
            # [FUNCTION DOC] HELPER: Max along dim 0 values.
            return x.max(0).values

        x = torch.randint(0, 100, (22, 51), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    @config.patch({"cpp.dynamic_threads": True})
    def test_reduction_with_dynamic_threads(self):
        # [FUNCTION DOC] Check reductions with dynamic OpenMP threads enabled give the same sums
        # as eager.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return sums of both inputs.
            return a.sum(), b.sum()

        self.common(
            fn,
            (torch.randn(1000), torch.rand(1000)),
        )

    def test_max_autotune_bmm_omp_dynamic(self):
        # [FUNCTION DOC] Check batched matmul with max autotune does not break when OpenMP dynamic
        # threading is enabled globally: sets the OpenMP dynamic flag, compiles and compares with
        # eager under four threads, and restores the flag in a finally block. Skipped when OpenMP
        # symbols are not found.
        try:
            omp = ctypes.CDLL("", ctypes.RTLD_GLOBAL)
        except OSError:
            self.skipTest("Could not access OpenMP symbols from process")

        if not hasattr(omp, "omp_set_dynamic") or not hasattr(omp, "omp_get_dynamic"):
            self.skipTest("omp_set_dynamic/omp_get_dynamic not found in OpenMP library")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Batched matmul.
            return torch.bmm(x, y)

        B, M, K, N = 1, 32, 32, 32
        x = torch.randn(B, M, K)
        y = torch.randn(B, K, N)

        with config.patch({"max_autotune": True}):
            compiled_fn = torch.compile(fn)
            original_dynamic = omp.omp_get_dynamic()
            omp.omp_set_dynamic(1)
            try:
                with set_num_threads(4):
                    actual = compiled_fn(x, y)
                    expected = fn(x, y)
                    torch.testing.assert_close(actual, expected)
            finally:
                omp.omp_set_dynamic(original_dynamic)

    @patch("torch.cuda.is_available", lambda: False)
    @config.patch(freezing=True)
    def test_linear_float64(self):
        # [FUNCTION DOC] Check float64 mm and addmm with frozen parameters match eager. CUDA
        # reported unavailable.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two float64 weight matrices and a bias, computing mm
            # and addmm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the float64 parameters.
                super().__init__()
                self.weight1 = torch.nn.Parameter(
                    torch.randn(10, 10, dtype=torch.float64)
                )
                self.weight2 = torch.nn.Parameter(
                    torch.randn(10, 10, dtype=torch.float64)
                )
                self.bias = torch.nn.Parameter(torch.randn(10, dtype=torch.float64))

            def forward(self, x1):
                # [FUNCTION DOC] HELPER: Matmul with weight1 and addmm with bias and weight2;
                # returns both.
                v1 = torch.mm(x1, self.weight1)
                v2 = torch.addmm(self.bias, x1, self.weight2)
                return (v1, v2)

        mod = M().eval()
        v = torch.randn(10, 10, dtype=torch.float64)
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_fused_attention_conv(self):
        # [FUNCTION DOC] Regression test (issue 121174): scaled dot product attention where query,
        # key, value come from permuted 1x1 convolutions.
        # https://github.com/pytorch/pytorch/issues/121174.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three 1x1 convs feeding scaled dot product
            # attention.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the three convs.
                super().__init__()
                self.q_conv = torch.nn.Conv2d(4, 4, 1)
                self.k_conv = torch.nn.Conv2d(4, 4, 1)
                self.v_conv = torch.nn.Conv2d(4, 4, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Convolve, permute and run attention without dropout or
                # causality.
                q = self.q_conv(x)
                k = self.k_conv(x)
                v = self.v_conv(x)
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=False
                )

        fn = Model()
        x = torch.randn(1, 4, 2, 2)
        self.common(fn, (x,))

    @parametrize("is_inference", (True, False))
    def test_disabled_amp(self, is_inference):
        # [FUNCTION DOC] Parametrized by inference flag: bf16 attention with dropout and a dense
        # layer under CPU autocast with random fallback and the math attention backend; compared
        # to eager with the same seed.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with scaled dot product attention (dropout 0.2), reshape
            # and dense layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the dense layer sized for 12 heads of 64.
                super().__init__()
                self.all_head_size = 12 * 64
                self.dense = nn.Linear(self.all_head_size, self.all_head_size)

            def forward(self, q, k, v):
                # [FUNCTION DOC] HELPER: Attention, permute heads, make contiguous, merge head
                # dims, then dense.
                context_layer = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.2
                )
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = context_layer.size()[:-2] + (
                    self.all_head_size,
                )
                context_layer = context_layer.view(new_context_layer_shape)
                return self.dense(context_layer)

        mod = M().to(torch.bfloat16).eval()

        q = torch.randn((4, 12, 512, 64), dtype=torch.bfloat16) / 10.0
        k = torch.randn((4, 12, 512, 64), dtype=torch.bfloat16) / 10.0
        v = torch.randn((4, 12, 512, 64), dtype=torch.bfloat16) / 10.0
        inputs = (
            q,
            k,
            v,
        )
        compiler_mode = torch.compile(mod)
        from torch.nn.attention import sdpa_kernel, SDPBackend

        context = contextlib.nullcontext if not is_inference else torch.no_grad
        with (
            config.patch({"fallback_random": True}),
            torch.cpu.amp.autocast(),
            context(),
            sdpa_kernel(SDPBackend.MATH),
        ):
            torch.manual_seed(0)
            eager = mod(*inputs)
            torch.manual_seed(0)
            self.assertEqual(compiler_mode(*inputs), eager, atol=1e-4, rtol=1.6e-2)

    def test_fused_node(self):
        # [FUNCTION DOC] Regression test (issue 138550): fused scheduler node correctness on a
        # large bf16/fp32 graph of attention backward style operations (casts, views, bmm, fma,
        # permutes); compared to eager under autocast with loose tolerance.
        # https://github.com/pytorch/pytorch/issues/138550.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module carrying a long captured graph of aten ops.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Empty constructor.
                super().__init__()

            def forward(
                self,
                clone_50,
                gt_scalar,
                div_tensor,
                convert_element_type_default_7,
                convert_element_type_default_13,
                convert_element_type_default_14,
            ):
                # [FUNCTION DOC] HELPER: Replays the captured graph: casts and reshapes, batched
                # matmuls, scaling, a fused multiply add for softmax backward style math, and
                # returns two reshaped tensors.
                convert_element_type_default_4 = (
                    torch.ops.prims.convert_element_type.default(
                        clone_50, torch.float32
                    )
                )
                clone_50 = None
                view_default_6 = torch.ops.aten.view.default(
                    convert_element_type_default_4, [336, 512, 64]
                )
                convert_element_type_default_4 = None
                convert_element_type_default_5 = (
                    torch.ops.prims.convert_element_type.default(
                        view_default_6, torch.bfloat16
                    )
                )
                view_default_6 = None
                mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor)
                mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112)
                mul_tensor = None
                expand_default_2 = torch.ops.aten.expand.default(
                    mul_tensor_1, [28, 12, 512, 512]
                )
                mul_tensor_1 = None
                view_default_3 = torch.ops.aten.view.default(
                    expand_default_2, [336, 512, 512]
                )
                expand_default_2 = None
                permute_default_4 = torch.ops.aten.permute.default(
                    view_default_3, [0, 2, 1]
                )
                view_default_3 = None
                convert_element_type_default_6 = (
                    torch.ops.prims.convert_element_type.default(
                        permute_default_4, torch.bfloat16
                    )
                )
                permute_default_4 = None
                bmm_default_2 = torch.ops.aten.bmm.default(
                    convert_element_type_default_6, convert_element_type_default_5
                )
                convert_element_type_default_6 = None
                convert_element_type_default_10 = (
                    torch.ops.prims.convert_element_type.default(
                        bmm_default_2, torch.float32
                    )
                )
                bmm_default_2 = None
                view_default_7 = torch.ops.aten.view.default(
                    convert_element_type_default_10, [28, 12, 512, 64]
                )
                convert_element_type_default_10 = None
                convert_element_type_default_18 = (
                    torch.ops.prims.convert_element_type.default(
                        view_default_7, torch.bfloat16
                    )
                )
                view_default_7 = None
                permute_default_9 = torch.ops.aten.permute.default(
                    convert_element_type_default_18, [0, 2, 1, 3]
                )
                convert_element_type_default_18 = None
                bmm_default_3 = torch.ops.aten.bmm.default(
                    convert_element_type_default_5, convert_element_type_default_7
                )
                convert_element_type_default_5 = convert_element_type_default_7 = None
                convert_element_type_default_9 = (
                    torch.ops.prims.convert_element_type.default(
                        bmm_default_3, torch.float32
                    )
                )
                bmm_default_3 = None
                view_default_8 = torch.ops.aten.view.default(
                    convert_element_type_default_9, [28, 12, 512, 512]
                )
                convert_element_type_default_9 = None
                convert_element_type_default_11 = (
                    torch.ops.prims.convert_element_type.default(
                        gt_scalar, torch.float32
                    )
                )
                gt_scalar = None
                mul_tensor_2 = torch.ops.aten.mul.Tensor(
                    convert_element_type_default_11, 1.1111111111111112
                )
                convert_element_type_default_11 = None
                mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default_8, mul_tensor_2)
                view_default_8 = mul_tensor_2 = None
                mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, div_tensor)
                mul_tensor_3 = None
                sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(
                    mul_tensor_4, [-1], True
                )
                neg_default = torch.ops.aten.neg.default(div_tensor)
                div_tensor = None
                fma_default = torch.ops.prims.fma.default(
                    neg_default, sum_dim_int_list_1, mul_tensor_4
                )
                neg_default = sum_dim_int_list_1 = mul_tensor_4 = None
                view_default_9 = torch.ops.aten.view.default(
                    fma_default, [336, 512, 512]
                )
                fma_default = None
                convert_element_type_default_12 = (
                    torch.ops.prims.convert_element_type.default(
                        view_default_9, torch.bfloat16
                    )
                )
                view_default_9 = None
                bmm_default_4 = torch.ops.aten.bmm.default(
                    convert_element_type_default_13, convert_element_type_default_12
                )
                convert_element_type_default_13 = None
                convert_element_type_default_17 = (
                    torch.ops.prims.convert_element_type.default(
                        bmm_default_4, torch.float32
                    )
                )
                bmm_default_4 = None
                view_default_10 = torch.ops.aten.view.default(
                    convert_element_type_default_17, [28, 12, 64, 512]
                )
                convert_element_type_default_17 = None
                mul_scalar_2 = torch.ops.aten.mul.Scalar(
                    view_default_10, 0.3535533905932738
                )
                view_default_10 = None
                permute_default_8 = torch.ops.aten.permute.default(
                    mul_scalar_2, [0, 1, 3, 2]
                )
                mul_scalar_2 = None
                convert_element_type_default_19 = (
                    torch.ops.prims.convert_element_type.default(
                        permute_default_8, torch.bfloat16
                    )
                )
                permute_default_8 = None
                _permute_default_10 = torch.ops.aten.permute.default(
                    convert_element_type_default_19, [0, 2, 1, 3]
                )
                convert_element_type_default_19 = None
                bmm_default_5 = torch.ops.aten.bmm.default(
                    convert_element_type_default_12, convert_element_type_default_14
                )
                convert_element_type_default_12 = convert_element_type_default_14 = None
                convert_element_type_default_16 = (
                    torch.ops.prims.convert_element_type.default(
                        bmm_default_5, torch.float32
                    )
                )
                bmm_default_5 = None
                view_default_11 = torch.ops.aten.view.default(
                    convert_element_type_default_16, [28, 12, 512, 64]
                )
                convert_element_type_default_16 = None
                mul_scalar_3 = torch.ops.aten.mul.Scalar(
                    view_default_11, 0.3535533905932738
                )
                view_default_11 = None
                convert_element_type_default_20 = (
                    torch.ops.prims.convert_element_type.default(
                        mul_scalar_3, torch.bfloat16
                    )
                )
                mul_scalar_3 = None
                permute_default_11 = torch.ops.aten.permute.default(
                    convert_element_type_default_20, [0, 2, 1, 3]
                )
                convert_element_type_default_20 = None
                clone_52 = torch.ops.aten.clone.default(
                    permute_default_11, memory_format=torch.contiguous_format
                )
                permute_default_11 = None
                view_283 = torch.ops.aten.view.default(clone_52, [28, 512, 768])
                clone_52 = None
                clone_53 = torch.ops.aten.clone.default(
                    permute_default_9, memory_format=torch.contiguous_format
                )
                permute_default_9 = None
                view_284 = torch.ops.aten.view.default(clone_53, [28, 512, 768])
                clone_53 = None
                view_285 = torch.ops.aten.view.default(view_284, [14336, 768])
                view_284 = None
                return view_283, view_285

        clone_50 = torch.randn((28, 12, 512, 64), dtype=torch.bfloat16) / 10
        gt_scalar = torch.randint(0, 2, (28, 12, 512, 512), dtype=torch.bool)
        div_tensor = torch.randn((28, 12, 512, 512), dtype=torch.float) / 10
        convert_element_type_default_7 = (
            torch.randn((336, 64, 512), dtype=torch.bfloat16) / 10
        )
        convert_element_type_default_13 = (
            torch.randn((336, 64, 512), dtype=torch.bfloat16) / 10
        )
        convert_element_type_default_14 = (
            torch.randn((336, 512, 64), dtype=torch.bfloat16) / 10
        )
        inputs = (
            clone_50,
            gt_scalar,
            div_tensor,
            convert_element_type_default_7,
            convert_element_type_default_13,
            convert_element_type_default_14,
        )

        with torch.cpu.amp.autocast():
            mod = M().to(torch.bfloat16).eval()
            self.common(mod, inputs, atol=1e-3, rtol=1e-3)

    @requires_vectorization
    def test_vec_indirect_load_cse_cache(self):
        # [FUNCTION DOC] Regression test (issue 123502): vectorized indirect load with the CSE
        # cache, using a captured graph with scatter add, power, comparisons, and where. Requires
        # vectorization.
        # https://github.com/pytorch/pytorch/issues/123502
        from math import inf

        def fn(arg0_1):
            # [FUNCTION DOC] HELPER: Captured graph of aten ops (full, select, scatter add, pow,
            # eq, where, etc.).
            full_default = torch.ops.aten.full.default([209985], 1)
            select = torch.ops.aten.select.int(arg0_1, 0, 0)
            select_1 = torch.ops.aten.select.int(arg0_1, 0, 1)
            view = torch.ops.aten.reshape.default(select_1, [-1])
            expand = torch.ops.aten.expand.default(view, [209985])
            full_default_1 = torch.ops.aten.full.default([10000], 0)
            scatter_add = torch.ops.aten.scatter_add.default(
                full_default_1, 0, expand, full_default
            )
            pow_1 = torch.ops.aten.pow.Tensor_Scalar(scatter_add, -0.5)
            eq = torch.ops.aten.eq.Scalar(pow_1, inf)
            full_default_2 = torch.ops.aten.full.default([], 0.0)
            where = torch.ops.aten.where.self(eq, full_default_2, pow_1)
            index = torch.ops.aten.index.Tensor(where, [select])
            index_1 = torch.ops.aten.index.Tensor(where, [select_1])
            mul_1 = torch.ops.aten.mul.Tensor(index, index_1)
            return (mul_1,)

        x = torch.zeros(2, 209985).to(torch.int64)
        opt_fn = torch.compile(fn, backend="inductor")
        _, code = run_and_get_cpp_code(opt_fn, x)
        FileCheck().check_count(
            "return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(),",
            8,
            exactly=True,
        ).run(code)

    def test_load_half(self):
        # [FUNCTION DOC] Check that with vectorization off, an in-place copy of half precision
        # tensors generates code with no float casts.
        def fn(arg0_1, arg0_2):
            # [FUNCTION DOC] HELPER: Copy the second input into the first in place.
            return arg0_1.copy_(arg0_2)

        with config.patch({"cpp.simdlen": 0}):
            x1 = torch.randn(2, 10).to(torch.half)
            x2 = torch.randn(2, 10).to(torch.half)
            opt_fn = torch.compile(fn, backend="inductor")
            _, code = run_and_get_cpp_code(opt_fn, x1, x2)
            FileCheck().check_count(
                "static_cast<float>",
                0,
                exactly=True,
            ).run(code)

    @requires_vectorization
    def test_repeated_exp(self):
        # [FUNCTION DOC] Check a sigmoid reused by an add and a sum computes the exponential only
        # once in the code.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sigmoid; return it plus one and its sum over the last dim.
            y = x.sigmoid()
            return y + 1, y.sum(-1)

        x = torch.randn(1000, 1000)
        opt_fn = torch.compile(fn)
        _, code = run_and_get_cpp_code(opt_fn, x)
        FileCheck().check_count(
            ".exp()",
            1,
            exactly=True,
        ).run(code)

    def test_convert_fp32_int64_oob_vec(self):
        # [FUNCTION DOC] Regression test (issue 129863): int64 minimum value converted through
        # float32 back to int64 (out of range) matches eager for every SIMD length, one vector
        # kernel.
        # https://github.com/pytorch/pytorch/issues/129863
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to float32 then to int64.
            float32 = x.to(torch.float32)
            return float32.to(torch.int64)

        x = torch.full((32,), -9223372036854775808, dtype=torch.int64)

        for simdlen in simd_lengths_to_test():
            with config.patch({"cpp.simdlen": simdlen}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_consistent_remove_buffers(self):
        # [FUNCTION DOC] Check consistent buffer removal for a bf16 kernel with both vector and
        # scalar paths and inplace buffers disabled: one vector kernel and exactly two occurrences
        # of the add expression.
        def fn(x):
            # [FUNCTION DOC] HELPER: Add x to itself, realize it, add x to that.
            z = x + x
            z1 = test_operators.realize(z)
            return x + z1

        # The shape makes sure we generate both vec and scalar kernels
        x = torch.randn((65,), dtype=torch.bfloat16)
        with config.patch(inplace_buffers=False):
            metrics.reset()
            self.common(fn, (x,))
            check_metrics_vec_kernel_count(1)
            _, code = run_and_get_cpp_code(torch.compile(fn), x)
            FileCheck().check_count(
                "tmp1 + tmp2",
                2,
                exactly=True,
            ).run(code)

    @requires_vectorization
    def test_bool_reduction_vec(self):
        # [FUNCTION DOC] Check vectorized boolean reductions (any, min, max) over all-false,
        # one-true and all-true inputs; expects three vector kernels per operation.
        for op in (
            torch.any,
            torch.min,
            torch.max,
        ):

            def fn(x1, x2, x3):
                # [FUNCTION DOC] HELPER: Apply the loop's reduction operator to each of the three
                # inputs.
                return op(x1), op(x2), op(x3)

            c = [False] * 63
            input1 = torch.Tensor(c).to(torch.bool)
            c[10] = True
            input2 = torch.Tensor(c).to(torch.bool)
            input3 = torch.Tensor([True] * 63).to(torch.bool)
            metrics.reset()
            self.common(
                fn,
                (
                    input1,
                    input2,
                    input3,
                ),
            )
            n_veckernel = 6 if op is torch.masked.mean else 3
            check_metrics_vec_kernel_count(n_veckernel)

    @requires_vectorization
    def test_full_bits_lowp(self):
        # [FUNCTION DOC] Check that low-precision kernels use full-width (wide vector) registers:
        # for bf16 and fp16 and four function variants (atanh sum, widen/narrow, mixed dtype,
        # small shape) it asserts the generated code contains (or for the small shape does not
        # contain) wide vector types or wide conversions.
        def check_use_full_bits(func, shapes, dtype, mixed, check_vecn):
            # [FUNCTION DOC] HELPER: Compile a function on random inputs (optionally mixing half
            # and bf16 for the first input), and assert presence or absence of wide vector markers
            # in the generated code.
            example_inputs = [torch.randn(shape, dtype=dtype) for shape in shapes]
            if mixed:
                example_inputs[0] = example_inputs[0].to(
                    dtype=torch.half if dtype == torch.bfloat16 else torch.bfloat16
                )
            f_opt = torch.compile()(func)
            _, code = run_and_get_cpp_code(f_opt, *example_inputs)
            if check_vecn:
                self.assertTrue(
                    "at::vec::VectorizedN" in code or "at::vec::convert<float,2" in code
                )
            else:
                self.assertFalse(
                    "at::vec::VectorizedN" in code or "at::vec::convert<float,2" in code
                )

        funcs = []

        def func0(arg0, arg1):
            # [FUNCTION DOC] HELPER: Sum over dims 2 and 3 of atanh of the first plus the second
            # input.
            return torch.ops.aten.sum(
                torch.ops.aten.add(torch.ops.aten.atanh(arg0), arg1), (2, 3)
            )

        funcs.append(func0)

        def func1(arg0):
            # [FUNCTION DOC] HELPER: Add atanh of the input to its float conversion and convert
            # back to the input dtype.
            v = torch.ops.prims.convert_element_type.default(arg0, torch.float)
            v = torch.ops.aten.add(torch.ops.aten.atanh(arg0), v)
            return torch.ops.prims.convert_element_type.default(v, arg0.dtype)

        funcs.append(func1)

        def func2(arg0, arg1):
            # [FUNCTION DOC] HELPER: Atanh of first plus second, converted to the second input
            # dtype.
            v = torch.ops.aten.atanh(arg0)
            v = torch.ops.aten.add(v, arg1)
            return torch.ops.prims.convert_element_type.default(v, arg1.dtype)

        funcs.append(func2)

        # test small shapes
        funcs.append(func2)
        small_size = cpu_vec_isa.pick_vec_isa().nelements(dtype=torch.bfloat16) // 2

        example_shapes = [
            [(10, 32, 20, 20), (10, 32, 20, 20)],
            [(10, 32, 20, 20)],
            [(10, 32, 20, 20), (10, 32, 20, 20)],
            # test small shapes
            [(small_size), (small_size)],
        ]
        mixed_types = [False, False, True, False]
        check_vecns = [True, True, True, False]

        for dtype in [torch.bfloat16, torch.float16]:
            for func, shapes, mixed, check_vecn in zip(
                funcs, example_shapes, mixed_types, check_vecns
            ):
                check_use_full_bits(func, shapes, dtype, mixed, check_vecn)

    @requires_vectorization
    def test_full_bits_fp8_e4m3fn(self):
        # [FUNCTION DOC] Check wide vector conversions between float and float8 e4m3 are generated
        # when fp8 is mixed with bf16 in the same kernel (dequantize times bf16, and float times
        # bf16 quantized to fp8).
        """
        Test VecConvert<float,2,Float8_e4m3fn,1> and VecConvert<Float8_e4m3fn,1,float,2>
        are emitted when fp8 and bf16 tensors are mixed in the same kernel.

        A pure fp8 cast alone uses a 16-element loop (convert<float,1,...>).
        Only when fp8 is combined with bf16 does the loop widen to 32 elements,
        triggering convert<float,2,Float8_e4m3fn,1> and
        convert<Float8_e4m3fn,1,float,2>.

        Sub-cases:
          func0 - fp8 dequant * bf16: tests convert<float,2,Float8_e4m3fn,1>
          func1 - float * bf16 → fp8: tests convert<Float8_e4m3fn,1,float,2>
        """
        fp8_dtype = torch.float8_e4m3fn
        deq_scale = 0.15

        # func0: fp8 dequant * bf16
        # Tests convert<float,2,Float8_e4m3fn,1>
        def func0(arg0_fp8, arg1_bf16):
            # [FUNCTION DOC] HELPER: Dequantize an fp8 tensor with the scale, multiply with the
            # float version of a bf16.
            x = arg0_fp8.to(torch.float) * deq_scale
            y = arg1_bf16.to(torch.float)
            return x * y

        # func1: float * bf16 -> fp8
        # Tests convert<Float8_e4m3fn,1,float,2>
        def func1(arg0_float, arg1_bf16):
            # [FUNCTION DOC] HELPER: Multiply float input with the float version of a bf16,
            # convert to fp8.
            y = arg1_bf16.to(torch.float)
            z = arg0_float * y
            return z.to(fp8_dtype)

        large_shape = (10, 32, 20, 20)

        # func0: verify fp8 dequant wide path
        torch._dynamo.reset()
        x_fp8 = torch.randn(large_shape).to(fp8_dtype)
        x_bf16 = torch.randn(large_shape, dtype=torch.bfloat16)
        _, code0 = run_and_get_cpp_code(torch.compile()(func0), x_fp8, x_bf16)
        self.assertIn(
            "at::vec::convert<float,2,at::Float8_e4m3fn,1>",
            code0,
            "Expected convert<float,2,at::Float8_e4m3fn,1> in generated code for func0",
        )

        # func1: verify fp8 quant wide path
        torch._dynamo.reset()
        x_float = torch.randn(large_shape)
        x_bf16 = torch.randn(large_shape, dtype=torch.bfloat16)
        _, code1 = run_and_get_cpp_code(torch.compile()(func1), x_float, x_bf16)
        self.assertIn(
            "at::vec::convert<at::Float8_e4m3fn,1,float,2>",
            code1,
            "Expected convert<at::Float8_e4m3fn,1,float,2> in generated code for func1",
        )

    @requires_vectorization
    def test_bool_to_float8_e4m3fn(self):
        # [FUNCTION DOC] Regression test (issue 178095): converting a boolean tensor to float8
        # e4m3 succeeds.
        """
        Test that bool to float8_e4m3fn cast succeeds.
        Issue: https://github.com/pytorch/pytorch/issues/178095
        """

        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to float8 e4m3.
            return x.to(dtype=torch.float8_e4m3fn)

        x = torch.ones(64, dtype=torch.bool)
        self.common(fn, (x,))

    @requires_vectorization
    def test_bool_to_float8_e5m2(self):
        # [FUNCTION DOC] Check converting a boolean tensor to float8 e5m2 succeeds.
        """
        Test that bool to float8_e5m2 cast succeeds.
        """

        def fn(x):
            # [FUNCTION DOC] HELPER: Convert to float8 e5m2.
            return x.to(dtype=torch.float8_e5m2)

        x = torch.ones(64, dtype=torch.bool)
        self.common(fn, (x,))

    @config.patch("cpp.simdlen", 256)
    @requires_vectorization
    def test_avx2_bool_constant_pad_nd(self):
        # [FUNCTION DOC] Check constant padding of an empty boolean tensor with simdlen 256
        # matches eager (AVX2 repro).
        # NOTE: I tried using (0, 12, 12) and removing the cpp.simdlen=256 override, but
        # that didn't repro the issue.
        result = torch.testing.make_tensor(
            (0, 6, 6), dtype=torch.bool, device=torch.device("cpu")
        )

        def fn(arg):
            # [FUNCTION DOC] HELPER: Constant pad by one on every side.
            return torch.constant_pad_nd(arg, (1, 1, 1, 1, 1, 1))

        self.common(fn, (result,))

    @config.patch(unroll_reductions_threshold=9999)
    @requires_vectorization
    def test_unrolled_bool_prod_vectorized(self):
        # [FUNCTION DOC] Check the product reduction along dim 1 of a boolean tensor with planted
        # zeros, with unrolled reductions allowed by a high threshold and vectorization, matches
        # eager.
        result = torch.zeros((37, 37, 37), dtype=torch.bool)
        dim_select = [0, 1]
        result.narrow(dim_select[0], 0, 1).narrow(dim_select[1], 1, 1).zero_()
        result.narrow(dim_select[0], 2, 1).narrow(dim_select[1], 3, 1).zero_()
        result.narrow(dim_select[0], 4, 1).narrow(dim_select[1], 3, 1).zero_()

        def fn(arg):
            # [FUNCTION DOC] HELPER: Product along dim 1 producing bool.
            return torch.prod(arg, 1, dtype=torch.bool)

        self.common(fn, (result,))

    @requires_vectorization
    @config.patch("cpp.min_chunk_size", 1)
    def test_for_loop_collapsed(self):
        # [FUNCTION DOC] Regression test (issue 122281): a transposed copy with minimum chunk size
        # 1 produces exactly one collapsed parallel for pragma and the right result.
        # https://github.com/pytorch/pytorch/issues/122281
        def fn(x):
            # [FUNCTION DOC] HELPER: Transpose and make contiguous.
            return x.transpose(1, 0).contiguous()

        x = torch.randn(199, 2)
        opt_fn = torch.compile(fn, backend="inductor")
        _, code = run_and_get_cpp_code(opt_fn, x)
        self.assertTrue(same(fn(x), opt_fn(x)))
        FileCheck().check_count("#pragma omp for collapse(2)", 1, exactly=True).run(
            code
        )

    @config.patch(freezing=True)
    def test_add_layernorm(self):
        # [FUNCTION DOC] Original PR 141766: a statically quantized linear layer followed by add
        # and layer norm, exported then compiled with freezing; code must contain a fused add plus
        # layer norm kernel, and at least five vector kernels when metrics can be checked.
        """
        Original PR: https://github.com/pytorch/pytorch/pull/141766
        """
        from torch.testing._internal.common_quantization import (
            _static_quantized_linear_module,
        )

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a quantized linear layer then layer norm over the
            # sum with the second input.
            def __init__(self, example_input):
                # [FUNCTION DOC] HELPER: Build the quantized dense layer from an example input and
                # a layer norm.
                super().__init__()
                self.dense = _static_quantized_linear_module(
                    N=768, K=768, bias=True, example_input=example_input
                )
                self.layernorm = torch.nn.LayerNorm(768, eps=1e-12)

            def forward(self, context_layer, hidden_states):
                # [FUNCTION DOC] HELPER: Dense on the first input, add the second, layer norm.
                attention_output = self.dense(context_layer)
                hidden_states = attention_output + hidden_states
                layer_output = self.layernorm(hidden_states)
                return layer_output

        example_batch = (torch.rand(1, 197, 768), torch.rand(1, 197, 768))
        model = Model(example_batch[0]).eval()
        model = torch.export.export(model, example_batch, strict=True).module()

        with torch.no_grad():
            expected = model(*example_batch)
            metrics.reset()
            actual, code = run_and_get_cpp_code(torch.compile(model), *example_batch)
            self.assertTrue(same(expected, actual))
            FileCheck().check("cpp_fused_add_native_layer_norm").run(code)
            if _can_check_vec_metrics():
                # The exact split of vectorized loops can vary by CPU backend.
                self.assertGreaterEqual(metrics.generated_cpp_vec_kernel_count, 5)

    def test_dropout(self):
        # [FUNCTION DOC] Check Dropout1d, Dropout2d and Dropout3d with p one half match eager when
        # the seed is reset in forward.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module with a dropout layer of the given spatial rank that
            # reseeds the generator on every forward call.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Create the dropout layer by name from the rank.
                super().__init__()
                self.dropout = eval(f"nn.Dropout{dim}d(p=0.5)")

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Seed with zero then apply dropout.
                torch.manual_seed(0)
                x = self.dropout(x)
                return x

        for dim in [1, 2, 3]:
            model = Model(dim)
            torch.manual_seed(0)
            shape = [1, 3] + [256] * dim
            x = torch.randn(*shape)
            output = model(x)
            c_model = torch.compile(model)
            c_output = c_model(x)
            self.assertTrue(torch.allclose(output, c_output))

    @requires_vectorization
    def test_bool_max(self):
        # [FUNCTION DOC] Check max along dim 1 of a boolean tensor matches eager. Requires
        # vectorization.
        torch.manual_seed(777)
        x = torch.randn(size=[128, 2501]).ge(0)

        def fn(x):
            # [FUNCTION DOC] HELPER: Max along dim 1 without keeping dims.
            return torch.max(x, 1, False)

        self.common(fn, (x,))

    def test_vector_norm_compile(self):
        # [FUNCTION DOC] Check the compiled vector norm with an empty dim list matches eager.
        x = torch.randn([16, 32], dtype=torch.float)
        ref = torch.linalg.vector_norm(x, ord=2, dim=[], keepdim=False, dtype=None)
        compiled_vector_norm = torch.compile(
            torch.linalg.vector_norm, backend="inductor"
        )
        res = compiled_vector_norm(x, ord=2, dim=[], keepdim=False, dtype=None)
        self.assertEqual(ref, res)

    def test_lp_pool_inf_norm_type_compile(self):
        # [FUNCTION DOC] Check lp pool 1D, 2D and 3D with positive and negative infinite norm
        # types in a compiled full graph against expected max and min values.
        x1 = torch.tensor([[[1.0, -3.0, 2.0, 4.0]]])
        x2 = torch.tensor([[[[1.0, -3.0, 2.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]]]])
        x3 = torch.tensor(
            [
                [
                    [
                        [[1.0, -3.0, 2.0], [4.0, -5.0, 6.0]],
                        [[-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]],
                    ]
                ]
            ]
        )

        def fn(x1, x2, x3):
            # [FUNCTION DOC] HELPER: Apply the six lp pool variants and return a tuple.
            return (
                F.lp_pool1d(x1, float("inf"), 2, 1),
                F.lp_pool1d(x1, -float("inf"), 2, 1),
                F.lp_pool2d(x2, float("inf"), 2, 1),
                F.lp_pool2d(x2, -float("inf"), 2, 1),
                F.lp_pool3d(x3, float("inf"), 2, 1),
                F.lp_pool3d(x3, -float("inf"), 2, 1),
            )

        expected = (
            torch.tensor([[[3.0, 3.0, 4.0]]]),
            torch.tensor([[[1.0, 2.0, 2.0]]]),
            torch.tensor([[[[5.0, 6.0], [8.0, 9.0]]]]),
            torch.tensor([[[[1.0, 2.0], [4.0, 5.0]]]]),
            torch.tensor([[[[[11.0, 12.0]]]]]),
            torch.tensor([[[[[1.0, 2.0]]]]]),
        )

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(x1, x2, x3)
        self.assertEqual(actual, expected)

    def test_fractional_max_pool2d_3d_input(self):
        # [FUNCTION DOC] Regression test (issue 156682): fractional max pool 2D on 3D inputs of
        # several shapes with explicit random samples compiles and matches eager.
        """Test for https://github.com/pytorch/pytorch/issues/156682 - 3D input causing assertion error"""

        # Test various 3D input shapes to ensure the compilation crash is fixed
        test_shapes = [
            (1, 8, 8),  # Original failing case
            (3, 16, 16),  # Different channel count
            (2, 12, 10),  # Non-square input
            (5, 20, 20),  # Larger input
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                torch.manual_seed(42)
                x = torch.randn(shape)

                # Generate explicit samples to ensure deterministic, correct results
                n_batch = 1 if x.dim() == 3 else x.size(0)
                torch.manual_seed(42)
                samples = torch.rand(
                    n_batch, x.size(-3), 2, dtype=x.dtype, device=x.device
                )

                def fn(x, samples):
                    # [FUNCTION DOC] HELPER: Fractional max pool with kernel size 3, output size
                    # 4x4 and the given samples.
                    return F.fractional_max_pool2d(
                        x, kernel_size=3, output_size=(4, 4), _random_samples=samples
                    )

                # Test that eager mode works
                expected = fn(x, samples)

                # Test that compiled mode works (was failing with AssertionError before fix)
                compiled_fn = torch.compile(fn, backend="inductor")
                result = compiled_fn(x, samples)

                # Verify correctness with explicit samples (should match exactly)
                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_outer_looop_fusion_with_local_buf(self):
        # [FUNCTION DOC] Check outer loop fusion with a local buffer for an einsum, exp, sum and
        # normalization pipeline.
        def fn(
            xs: torch.Tensor,
            Ls: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Negative scaled einsum, exp, sum over elements, and
            # normalization by the sum.
            arr = -torch.einsum("i...,i->i...", xs, Ls)
            temp = torch.exp(arr)
            Q = torch.einsum("i...->i", temp)
            ans = torch.einsum("i,i...->i...", 1 / Q, temp)
            return ans

        xs = torch.ones((5, 1, 32, 32), requires_grad=False)
        Ls = torch.ones((5), requires_grad=False)
        expected = fn(xs, Ls)
        compiled_func = torch.compile(fn, backend="inductor")
        result = compiled_func(xs, Ls)
        torch.testing.assert_close(result, expected)

    def test_special_float_pow(self):
        # [FUNCTION DOC] Check power with infinite and NaN exponents compiles and matches eager
        # for random values.
        def fn(exp: float) -> None:
            # [FUNCTION DOC] HELPER: Compare eager and compiled aten pow for random values with a
            # given float exponent; asserts close including NaNs.
            val = torch.randn(10)
            torch.testing.assert_close(
                aten.pow(val, exp), torch.compile(aten.pow)(val, exp), equal_nan=True
            )

        fn(-math.inf)
        fn(math.inf)
        fn(math.nan)

    def test_sin_atan_nan(self):
        # [FUNCTION DOC] Check sine of arctangent of NaN.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sine of arctangent.
            return torch.sin(torch.atan(x))

        x = torch.tensor([float("nan")])
        self.common(fn, (x,))

    def test_pdist_fallback_continuous(self):
        # [FUNCTION DOC] Regression test (issue 170939): pdist on a transposed non-contiguous
        # input compiles.
        # https://github.com/pytorch/pytorch/issues/170939
        def fn(x):
            # [FUNCTION DOC] HELPER: Transpose and compute pairwise distances.
            # Creating a non-contiguous tensor via permute
            x = x.permute(1, 0)
            return F.pdist(x)

        torch.compile(fn)(torch.randn(2, 2))

    @skipIfRocmArch(MI200_ARCH)
    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @requires_vectorization
    @config.patch(freezing=True)
    def test_upsample_layout(self):
        # [FUNCTION DOC] Check upsample layout: conv, bilinear upsample, conv, relu with freezing
        # in float32 and bf16 (autocast); the code has exactly two transpose helpers (or two
        # external convolutions on some ROCm cases) and results match eager with dtype dependent
        # tolerance.
        class UpsampleModel(nn.Module):
            # [CLASS DOC] HELPER: Module with conv, bilinear 2x upsample, second conv and relu.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the layers.
                super().__init__()
                self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.upsample = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )
                self.conv2 = nn.Conv2d(64, 16, kernel_size=3, padding=1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Conv, upsample, conv, relu.
                x = self.conv(x)
                x = self.upsample(x)
                x = self.conv2(x)
                x = torch.relu(x)
                return x

        mod = UpsampleModel()
        cmod = torch.compile(mod)
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(4, 64, 64, 64, dtype=dtype)
            with (
                torch.no_grad(),
                torch.amp.autocast(
                    device_type="cpu", dtype=dtype, enabled=dtype == torch.bfloat16
                ),
            ):
                ref_res = mod(x)
                res, code = run_and_get_cpp_code(cmod, x)
                # The 2 transpose_mxns are unrelated to upsample.
                # They are generated by the input of first conv and
                # the final output.

                # HIP: Some architectures (e.g. MI200) do not use the MKLDNN backend for bfloat16
                # and instead use extern_kernels.convolution that does not need a call to transpose_mxn.
                # We check for this by looking for the string "extern_kernels.convolution" in the code.
                if (
                    torch.version.hip
                    and dtype == torch.bfloat16
                    and "torch.ops.mkldnn._convolution" not in code
                ):
                    FileCheck().check_count(
                        "extern_kernels.convolution",
                        2,
                        exactly=True,
                    ).run(code)
                else:
                    FileCheck().check_count(
                        "transpose_mxn",
                        2,
                        exactly=True,
                    ).run(code)
                atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
                rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
                torch.testing.assert_close(ref_res, res, atol=atol, rtol=rtol)

    # https://github.com/pytorch/pytorch/issues/136640
    def test_inductor_dynamic_shapes_broadcasting(self) -> None:
        # [FUNCTION DOC] Regression test (issue 136640): dynamic shape broadcasting between two
        # differently sized views.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Multiply a view of x with a view of y, both with four
            # columns.
            x_view = x.view(-1, 4)
            y_view = y.view(-1, 4)
            return x_view * y_view

        x = torch.randn(4)
        y = torch.randn(8)
        out_ref = fn(x, y)
        out_test = torch.compile(fn, dynamic=True, backend="inductor")(x, y)
        self.assertEqual(out_ref, out_test)

    # https://github.com/pytorch/pytorch/issues/119162
    def test_inductor_rng_default_dtype(self) -> None:
        # [FUNCTION DOC] Regression test (issue 119162): random normal with an explicit bf16 dtype
        # while the default dtype is bf16; output must be bf16. Restores the default dtype in a
        # finally block.
        @torch.compile
        def fn():
            # [FUNCTION DOC] HELPER: Draw a 4x4 bf16 normal tensor.
            tmp = torch.randn(4, 4, dtype=torch.bfloat16)
            return tmp

        try:
            old = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            out = fn()
        finally:
            torch.set_default_dtype(old)
        # output dtype should be float32
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_inductor_no_recursionerror_on_for_loops(self):
        # [FUNCTION DOC] Check compiling a function with a very long Python loop (10000
        # multiplications) does not hit a recursion limit and returns the right value.
        def forward(x):
            # [FUNCTION DOC] HELPER: Multiply by one repeatedly in a long loop.
            for _ in range(10000):
                x = 1.0 * x
            return x

        self.assertTrue(
            same(
                torch.compile(forward, backend="inductor")(torch.tensor([1.0])),
                torch.tensor([1.0]),
            )
        )

    def test_mutation_transpose_reshape_ordering(self):
        # [FUNCTION DOC] Check an in-place add on a transposed tensor followed by reshape orders
        # mutation and reshape correctly: eager and compiled outputs agree.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add y in place to x then reshape to three columns.
            return x.add_(y).reshape(-1, 1, 3)

        torch.manual_seed(0)
        x1 = torch.ones([1, 2, 3]).transpose(1, 2)
        y1 = torch.randn(1, 3, 1)

        x2 = x1.clone()
        y2 = y1.clone()

        cfunc = torch.compile(fn, backend="inductor")

        out1 = fn(x1, y1)
        out2 = cfunc(x2, y2)

        self.assertTrue(torch.allclose(out1, out2, equal_nan=True))

    def test_indirect_index_transposed_tensor(self):
        # [FUNCTION DOC] Regression test (issue 178521): indirect indexing of a transposed tensor
        # must not use a temporary variable out of scope in the transpose helper.
        # https://github.com/pytorch/pytorch/issues/178521
        # transpose_mxn was referencing a tmp variable defined inside the inner
        # loop when the index used indirect (SymT.TMP) indexing.
        def f(buf, idx):
            # [FUNCTION DOC] HELPER: Index with an arange for the first dim and idx for the
            # second.
            return buf[torch.arange(buf.shape[0]), idx, :]

        buf = torch.randn(16, 2, 16).permute(2, 1, 0)
        idx = torch.randint(0, 2, (16,))
        expected = f(buf, idx)
        actual = torch.compile(f, backend="inductor")(buf, idx)
        self.assertEqual(actual, expected)

    def test_layernorm_nan_with_inf_cpu_float16(self):
        # [FUNCTION DOC] Regression test (issue 173885): layer norm in half precision on values
        # near the fp16 maximum must not turn into unexpected NaNs; compiled output equals eager
        # for fp16 and bf16.
        # https://github.com/pytorch/pytorch/issues/173885
        # LayerNorm with torch.compile on CPU produces NaN when input has Inf
        # values due to inf - inf = NaN in Welford variance calculation.
        for dtype in _lowp_fp_dtypes:
            torch._dynamo.reset()

            class Model(torch.nn.Module):
                # [CLASS DOC] HELPER: Module wrapping a layer norm over four features.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Create the layer norm.
                    super().__init__()
                    self.ln = torch.nn.LayerNorm(4)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Apply the layer norm.
                    return self.ln(x)

            mod = Model().to(dtype).eval()
            # Create input that overflows during Welford M2 accumulation
            x = torch.tensor(
                [[65504.0, 65504.0, -65504.0, -65504.0]], dtype=torch.float32
            ).to(dtype)

            with torch.no_grad():
                eager_out = mod(x)
                compiled_mod = torch.compile(mod, backend="inductor")
                compiled_out = compiled_mod(x)
                # Compiled output should match eager mode (whether Eager produces
                # inf or finite numbers, it shouldn't degrade into unexpected NaNs).
                self.assertEqual(eager_out, compiled_out)

    def test_cpu_realization_thresholds(self):
        # [FUNCTION DOC] Check realization thresholds on CPU: pointwise nodes with large inner
        # operation counts, many reads or multiple users are realized according to config
        # thresholds; on CUDA devices thresholds differ (large). Builds Pointwise and StorageBox
        # IR nodes directly and asserts the boolean answers of their threshold query methods, with
        # thresholds patched through config.
        from torch._inductor.ir import Pointwise, StorageBox
        from torch._inductor.virtualized import ops

        def inner_opcount_fn(index):
            # [FUNCTION DOC] HELPER: Inner function emitting a chain of 45 constant adds.
            value = ops.constant(0.0, torch.float32)
            for _ in range(45):
                value = ops.add(value, ops.constant(1.0, torch.float32))
            return value

        cpu_pointwise = Pointwise(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=inner_opcount_fn,
            ranges=[1],
        )
        self.assertFalse(cpu_pointwise.has_large_inner_fn())
        with config.patch(realize_opcount_threshold=1):
            self.assertTrue(cpu_pointwise.has_large_inner_fn())
        with config.patch(realize_opcount_threshold=30):
            self.assertTrue(cpu_pointwise.has_large_inner_fn())
        self.assertFalse(cpu_pointwise.has_large_inner_fn())

        cuda_pointwise = Pointwise(
            device=torch.device("cuda"),
            dtype=torch.float32,
            inner_fn=inner_opcount_fn,
            ranges=[1],
        )
        self.assertTrue(cuda_pointwise.has_large_inner_fn())

        def inner_tanh_fn(index):
            # [FUNCTION DOC] HELPER: Inner function applying tanh to a load.
            return ops.tanh(ops.load("in0", index[0]))

        cpu_tanh_storage = StorageBox(
            Pointwise(
                device=torch.device("cpu"),
                dtype=torch.float32,
                inner_fn=inner_tanh_fn,
                ranges=[10],
            )
        )
        self.assertFalse(cpu_tanh_storage.should_realize_on_reuse(1))
        self.assertTrue(cpu_tanh_storage.should_realize_on_reuse(2))

        def inner_multi_user_fn(index):
            # [FUNCTION DOC] HELPER: Inner function with 23 multiply and add steps on a loaded
            # value.
            value = ops.load("in0", index[0])
            for _ in range(23):
                value = ops.mul(value, ops.constant(1.0001, torch.float32))
                value = ops.add(value, ops.constant(0.1, torch.float32))
            return value

        cpu_multi_user_storage = StorageBox(
            Pointwise(
                device=torch.device("cpu"),
                dtype=torch.float32,
                inner_fn=inner_multi_user_fn,
                ranges=[10],
            )
        )
        self.assertEqual(cpu_multi_user_storage.data.inner_fn_opcount().num_ops, 49)
        self.assertFalse(cpu_multi_user_storage.has_large_inner_fn())
        self.assertFalse(cpu_multi_user_storage.should_realize_on_reuse(5))
        self.assertTrue(cpu_multi_user_storage.should_realize_on_reuse(6))
        self.assertFalse(
            cpu_multi_user_storage.should_realize_on_reuse(6, graph_reuse=False)
        )
        with config.patch(realize_opusers_threshold=6):
            self.assertFalse(cpu_multi_user_storage.should_realize_on_reuse(6))

        def inner_reads_fn(index):
            # [FUNCTION DOC] HELPER: Inner function summing loads from ten different inputs.
            value = ops.constant(0.0, torch.float32)
            for i in range(10):
                value = ops.add(value, ops.load(f"in{i}", index[0]))
            return value

        cpu_storage = StorageBox(
            Pointwise(
                device=torch.device("cpu"),
                dtype=torch.float32,
                inner_fn=inner_reads_fn,
                ranges=[10],
            )
        )
        self.assertFalse(cpu_storage.has_exceeded_max_reads())
        with config.patch(realize_acc_reads_threshold=1):
            self.assertTrue(cpu_storage.has_exceeded_max_reads())
        with config.patch(realize_acc_reads_threshold=8):
            self.assertTrue(cpu_storage.has_exceeded_max_reads())
        self.assertFalse(cpu_storage.has_exceeded_max_reads())

        cuda_storage = StorageBox(
            Pointwise(
                device=torch.device("cuda"),
                dtype=torch.float32,
                inner_fn=inner_reads_fn,
                ranges=[10],
            )
        )
        self.assertTrue(cuda_storage.has_exceeded_max_reads())


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests(needs="filelock")
