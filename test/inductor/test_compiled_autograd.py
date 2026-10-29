# Owner(s): ["module: inductor"]
# ruff: noqa: F841
# [MODULE DOC] Tests for compiled autograd (the mode that captures the whole backward pass
# into one FX graph and compiles it with dynamo/inductor). Defines compiler factories, hook
# helpers, a large TestCompiledAutograd class, tests for the class-wrapping helper, and an
# operator-database test of higher-order ops in backward. At import time it loads the
# autograd, custom op and higher-order op test modules and creates wrapped copies of their
# test classes that run every test under compiled autograd, using tables of known failures,
# skipped tests and per-test contexts. Test entry runs only when a CPU is available.
import contextlib
import dataclasses
import functools
import io
import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from importlib.machinery import SourceFileLoader
from pathlib import Path
from string import Template
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd, config
from torch._dynamo.backends.debugging import aot_eager
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.cpp_builder import is_msvc_cl
from torch._inductor.test_case import run_tests, TestCase
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.overrides import BaseTorchFunctionMode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    skipOps,
    xfail,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_S390X,
    IS_WINDOWS,
    noncontiguous_like,
    parametrize,
    scoped_load_inline,
    skipIfWindows,
    skipIfXpu,
)
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
    HAS_GPU_AND_TRITON,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.testing._internal.triton_utils import (
    requires_cuda_and_triton,
    requires_gpu_and_triton,
)
from torch.utils._python_dispatch import TorchDispatchMode


try:
    from torch.distributed.tensor import DeviceMesh, DTensor, Shard
    from torch.testing._internal.distributed.fake_pg import FakeStore

    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False


# note: these tests are not run on windows due to inductor_utils.HAS_CPU


def make_compiler_fn(
    fullgraph=True, dynamic=True, backend="inductor", gm_hook=lambda gm: None
):
    # [FUNCTION DOC] Builds and returns a graph-compiler callable to hand to the compiled
    # autograd enable context manager. Arguments: fullgraph and dynamic are forwarded to
    # torch.compile; backend must be one of inductor, aot_eager, eager or ca_eager (otherwise
    # an AssertionError is raised); gm_hook is called on every captured graph module before
    # compilation.
    if backend not in ["inductor", "aot_eager", "eager", "ca_eager"]:
        raise AssertionError(f"unexpected backend: {backend}")

    def _compiler_fn(gm):
        # [FUNCTION DOC] HELPER: Compiles one captured backward graph module. Runs the
        # gm_hook, then returns the module unchanged for the ca_eager backend, otherwise wraps
        # it with torch.compile (using a counting inner compiler unless the backend is plain
        # eager) with the factory's fullgraph and dynamic settings.
        """Same as torch.compile() but counts number of compiles"""
        gm_hook(gm)

        _backend = backend
        if backend == "ca_eager":
            return gm
        elif backend != "eager":

            def _inner_compiler(gm_, example_inputs_):
                # [FUNCTION DOC] HELPER: Backend used by the compiled autograd compiler:
                # increments the compiled_autograd compiles counter, then compiles with
                # inductor or aot_eager according to the enclosing backend choice. Returns the
                # compiled callable (None if the backend is neither, which cannot happen after
                # the factory check).
                counters["compiled_autograd"]["compiles"] += 1
                if backend == "inductor":
                    return inductor.compile(gm_, example_inputs_)
                elif backend == "aot_eager":
                    return aot_eager(gm_, example_inputs_)

            _backend = _inner_compiler

        return torch.compile(gm, backend=_backend, fullgraph=fullgraph, dynamic=dynamic)

    return _compiler_fn


compiler_fn = make_compiler_fn()


# TODO(jansel): hooks as lambdas creates recompiles in dynamo, we should fix that
def hook1(grad):
    # [FUNCTION DOC] HELPER: Tensor gradient hook that returns the incoming gradient
    # multiplied by two.
    return grad * 2


def hook2(grads):
    # [FUNCTION DOC] HELPER: Node pre-hook that takes a tuple of gradients and returns a
    # one-element tuple whose value is the first gradient plus one.
    return (grads[0] + 1,)


def hook3(gI, gO):
    # [FUNCTION DOC] HELPER: Node post-hook taking grad inputs and grad outputs; returns a
    # one-element tuple holding the sine of the first grad input plus the first grad output.
    return (torch.sin(gI[0]) + gO[0],)


def reset():
    # [FUNCTION DOC] HELPER: Restores global state between tests: turns off compiled autograd
    # verbose logging, clears the compiled_autograd config flag, resets the compiled autograd
    # module state, and clears the dynamo counters.
    torch._logging.set_logs(compiled_autograd_verbose=False)
    config.compiled_autograd = False
    compiled_autograd.reset()
    torch._dynamo.utils.counters.clear()


class BaseCustomOp(torch.autograd.Function):
    # [CLASS DOC] HELPER: Custom autograd Function base whose forward doubles its input;
    # subclasses must override backward, which here raises NotImplementedError. Used as a base
    # for custom-function tests.
    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: Forward of the base custom op; returns the input multiplied
        # by two without saving anything.
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Backward placeholder; always raises NotImplementedError
        # telling the subclass to override it.
        raise NotImplementedError("must override")


class TestCompiledAutograd(TestCase):
    # [CLASS DOC] Main test case for compiled autograd, running each scenario in eager mode
    # and again under the compiled autograd context, then comparing outputs and
    # capture/compile counters. setUp and tearDown install a config patch that disables
    # runtime overhead recording and reset global compiled autograd state around every test.
    # Holds helper methods for counting recompiles and running scripts in subprocesses.
    def setUp(self) -> None:
        # [FUNCTION DOC] Test fixture setup: opens an ExitStack holding a config patch that
        # turns off record_runtime_overhead, calls the parent setUp, then resets compiled
        # autograd global state.
        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(config.patch("record_runtime_overhead", False))
        super().setUp()
        reset()

    def tearDown(self) -> None:
        # [FUNCTION DOC] Test fixture teardown: closes the exit stack (undoing the config
        # patch), calls the parent tearDown and resets compiled autograd global state.
        self.exit_stack.close()
        super().tearDown()
        reset()

    def check_output_and_recompiles(
        self, fn, count=1, compiler_fn=compiler_fn, compile_fn=False
    ):
        # [FUNCTION DOC] HELPER: Core check used by most tests. Runs the generator-producing
        # function once in eager mode and once with compiled autograd enabled (optionally
        # under torch.compile), with multithreaded autograd disabled and the same random seed.
        # Asserts the yielded values are equal and that the capture and compile counters equal
        # the expected count (a single number, or a list of captures then compiles).
        if isinstance(count, list):
            captures, compiles = count
        else:
            captures, compiles = count, count
        with torch.autograd.set_multithreading_enabled(False):
            torch._dynamo.reset()
            counters["compiled_autograd"].clear()
            torch.manual_seed(123)
            expected = list(fn())
            torch.manual_seed(123)
            with (
                compiled_autograd._enable(compiler_fn),
                mock.patch(
                    "torch._functorch.aot_autograd.AOT_COUNTER",
                    new_callable=itertools.count,
                ),
            ):
                opt_fn = torch.compile(fn) if compile_fn else fn
                actual = list(opt_fn())
            self.assertEqual(expected, actual)
            self.assertEqual(counters["compiled_autograd"]["captures"], captures)
            self.assertEqual(counters["compiled_autograd"]["compiles"], compiles)

    def run_as_subprocess(self, script) -> bytes:
        # [FUNCTION DOC] HELPER: Runs the given script text in a fresh Python interpreter,
        # merging stderr into stdout, with the working directory set to this test directory.
        # Returns the captured output bytes; fails the test with the return code if the
        # subprocess exits non-zero.
        try:
            return subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            self.fail(f"Subprocess exited with return code: {e.returncode}")

    def test_hipify_not_loaded_with_import_torch(self):
        # [FUNCTION DOC] Verifies in a subprocess that importing torch alone does not load the
        # hipify module into the global namespace.
        script = """
import torch
assert globals().get("hipify", False) is False  # noqa: S101
"""
        self.run_as_subprocess(script)

    def test_hipify_not_loaded_with_import_cpp_extension(self):
        # [FUNCTION DOC] Verifies in a subprocess that importing the cpp_extension utility
        # module does not load the hipify module into the global namespace.
        script = """
import torch.utils.cpp_extension
assert globals().get("hipify", False) is False  # noqa: S101
"""
        self.run_as_subprocess(script)

    def test_dynamo_flaky_segfault(self):
        # [FUNCTION DOC] Regression test for a flaky dynamo segfault: a subprocess script runs
        # a linear-layer backward under compiled autograd with an eager-compiled graph, resets
        # dynamo, and repeats; the subprocess is launched three times to catch bad dynamo
        # state resets.
        script = """
import torch

def main():
    def compiler_fn(gm):
        return torch.compile(gm, backend="eager")

    def inner():
        x = torch.randn(1000, 3000)
        w = torch.randn(1000, 3000, requires_grad=True)
        def model(i):
            return torch.nn.functional.linear(i, w)
        out = model(x)
        loss = out.sum()
        with torch._dynamo.compiled_autograd._enable(compiler_fn):
            loss.backward()
        assert(w.grad is not None)  # noqa: S101

    inner()
    torch._dynamo.reset()
    inner()

main()
        """
        # Run it three times to catch bad dynamo state resets
        for _ in range(3):
            self.run_as_subprocess(script)

    def test_index_fill_value_grad_noncontiguous(self):
        # [FUNCTION DOC] Checks that the gradient of index_fill with respect to a scalar value
        # tensor works for non-contiguous inputs under compiled autograd with inductor and
        # dynamic output shape ops allowed. Expects the eager result 200, one capture, one
        # compile and no graph break caused by a unique op.
        def make_inputs():
            # [FUNCTION DOC] HELPER: Builds non-contiguous input, index tensor, scalar value
            # requiring grad, and a non-contiguous grad output; returns them as a tuple.
            inp = torch.arange(25, dtype=torch.float32).reshape(5, 5)
            inp = noncontiguous_like(inp.requires_grad_())
            index = torch.tensor([2, 1, 4, 2, 2])
            value = torch.tensor(-1.7658, requires_grad=True)
            grad_out = torch.arange(1, 26, dtype=torch.float32).reshape(5, 5)
            grad_out = noncontiguous_like(grad_out)
            return inp, index, value, grad_out

        def fn(inp, index, value, grad_out):
            # [FUNCTION DOC] HELPER: Computes index_fill along the last dim and returns only
            # the gradient with respect to the value, using autograd.grad with unused inputs
            # allowed.
            out = inp.index_fill(-1, index, value)
            return torch.autograd.grad(out, (inp, value), grad_out, allow_unused=True)[
                1
            ]

        expected = fn(*make_inputs())
        self.assertEqual(expected, torch.tensor(200.0))

        counters["compiled_autograd"].clear()
        with (
            config.patch(capture_dynamic_output_shape_ops=True),
            compiled_autograd._enable(make_compiler_fn(backend="inductor")),
            torch.autograd.set_multithreading_enabled(False),
        ):
            actual = torch.compile(fn, backend="inductor")(*make_inputs())

        self.assertEqual(actual, expected)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 1)
        self.assertFalse(any("_unique" in reason for reason in counters["graph_break"]))

    def gen_cache_miss_log_prefix(self):
        # [FUNCTION DOC] HELPER: Returns the expected prefix of the compiled autograd
        # cache-miss log message. On Windows with the MSVC compiler the prefix includes the
        # word struct; on Windows with another compiler the test fails; elsewhere the plain
        # prefix is returned.
        if IS_WINDOWS:
            if is_msvc_cl():
                return "Cache miss due to new autograd node: struct "
            else:
                self.fail(
                    "Compilers other than msvc have not yet been verified on Windows."
                )
                return ""
        else:
            return "Cache miss due to new autograd node: "

    def test_reset(self):
        # [FUNCTION DOC] Checks that the compiled autograd reset function restores clean
        # state: after enabling the flag, installing an autograd compiler and clearing the
        # compile counter, a reset leaves the enabled flag false, no prior compiler or dynamic
        # flag, and a fresh compile counter starting at zero. Raises AssertionError on any
        # mismatch.
        compiled_autograd.compiled_autograd_enabled = True
        torch._C._dynamo.compiled_autograd.set_autograd_compiler(lambda: None, True)
        # TODO: return prior verbose logger
        # torch._C._dynamo.compiled_autograd.set_verbose_logger(dummy)
        compiled_autograd.COMPILE_COUNTER = None

        # state should be clean after reset
        compiled_autograd.reset()

        if compiled_autograd.compiled_autograd_enabled is not False:
            raise AssertionError
        (
            prior_compiler,
            prior_dynamic,
        ) = torch._C._dynamo.compiled_autograd.set_autograd_compiler(None, False)
        if prior_compiler is not None:
            raise AssertionError
        if prior_dynamic is not False:
            raise AssertionError
        if not (
            compiled_autograd.COMPILE_COUNTER is not None
            and next(compiled_autograd.COMPILE_COUNTER) == 0
        ):
            raise AssertionError

    def test_basic(self):
        # [FUNCTION DOC] Basic test: a small two-layer linear and ReLU model backward,
        # yielding all weight and bias grads; checks equality with eager and default one
        # capture and one compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that builds a Sequential of linear and ReLU
            # layers, runs forward and backward on a random batch, and yields the weight and
            # bias gradients of both linear layers.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])
            result = model(x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        self.check_output_and_recompiles(fn)

    def test_cache_hit(self):
        # [FUNCTION DOC] Checks the compiled autograd cache is hit: three iterations each
        # create a fresh model and run backward, and the default expectation of one capture
        # and one compile holds.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that for three iterations builds a new model,
            # runs backward and yields the four parameter gradients.
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])
                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad

        self.check_output_and_recompiles(fn)

    def test_graph_break_custom_op(self):
        # [FUNCTION DOC] Checks that a user-defined custom op with registered autograd works
        # under compiled autograd without raising: defines a sine op, registers its backward,
        # and runs backward on its output.
        @torch.library.custom_op("mylib::sin", mutates_args={})
        def sin(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation returning the elementwise sine
            # of the input.
            return x.sin()

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup hook for the sine op; saves the input
            # tensor for backward.
            (x,) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward for the sine op; returns the incoming gradient
            # multiplied by the cosine of the saved input.
            (x,) = ctx.saved_tensors
            return grad * x.cos()

        sin.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        y = sin(x.clone()).sum()
        with compiled_autograd._enable(compiler_fn):
            y.backward()

    def test_tensor_grad_hook1(self):
        # [FUNCTION DOC] Tests a tensor gradient hook registered on a weight parameter, over
        # three iterations, with the default one capture and one compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that builds a small model, registers the
            # doubling hook on the first weight, runs backward and yields the first layer
            # grads, three times.
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])

                model[0].weight.register_hook(hook1)

                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook2(self):
        # [FUNCTION DOC] Tests a pre-hook registered on the grad function of the loss, over
        # three iterations, with one capture and one compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that registers the pre-hook helper on the
            # output grad function before backward and yields the first layer grads, three
            # times.
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.grad_fn.register_prehook(hook2)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook3(self):
        # [FUNCTION DOC] Tests a post-hook registered on the grad function of the loss, over
        # three iterations, with one capture and one compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that registers the post-hook helper on the
            # output grad function before backward and yields the first layer grads, three
            # times.
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.grad_fn.register_hook(hook3)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_reorder_acc_grad(self):
        # [FUNCTION DOC] Checks that compiled autograd reorders gradient accumulation
        # correctly for a two-convolution model: grads from an eager backward must equal those
        # from a compiled-autograd backward over a compiled model.
        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, bias=True),
            torch.nn.Conv2d(4, 4, 3, bias=True),
        )
        compiled_model = torch.compile(model)
        x = torch.randn([1, 4, 32, 32])

        model(x).sum().backward()
        ref_res = [
            model[0].weight.grad,
            model[0].bias.grad,
            model[1].weight.grad,
            model[1].bias.grad,
        ]

        model[0].weight.grad = None
        model[0].bias.grad = None
        model[1].weight.grad = None
        model[1].bias.grad = None
        with compiled_autograd._enable(compiler_fn):
            compiled_model(x).sum().backward(retain_graph=True)
        res = [
            model[0].weight.grad,
            model[0].bias.grad,
            model[1].weight.grad,
            model[1].bias.grad,
        ]

        self.assertEqual(res[0], ref_res[0])
        self.assertEqual(res[1], ref_res[1])
        self.assertEqual(res[2], ref_res[2])
        self.assertEqual(res[3], ref_res[3])

    def test_reorder_post_hook1(self):
        # [FUNCTION DOC] Checks post-hooks on gradient accumulator nodes that modify param
        # grad in place (dividing by four) give the same result in eager and compiled
        # autograd.
        def grad_div(param):
            # [FUNCTION DOC] HELPER: Divides the gradient of the given parameter by four,
            # replacing the stored grad.
            param.grad = param.grad / 4.0

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Two-layer bias-free linear module that on construction
            # registers a hook on each weight's gradient accumulator node which calls the grad
            # division helper. Keeps the hooks and accumulators alive in lists.
            def __init__(self, ioc):
                # [FUNCTION DOC] HELPER: Builds the two linear layers, then for each weight
                # obtains its accumulator node through an expanded view and registers a hook
                # that divides the grad, storing node and handle on the module.
                super().__init__()
                self.fc1 = torch.nn.Linear(ioc, ioc, bias=False)
                self.fc2 = torch.nn.Linear(ioc, ioc, bias=False)

                self.grad_acc_hooks = []
                self.grad_acc = []
                self.params = [self.fc1.weight, self.fc2.weight]
                for param in self.params:

                    def wrapper(param):
                        # [FUNCTION DOC] HELPER: Looks up the gradient accumulator node of one
                        # parameter and registers the hook on it, keeping both referenced on
                        # the module.
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def grad_acc_hook(*notneeded):
                            # [FUNCTION DOC] HELPER: Accumulator hook ignoring its arguments;
                            # divides the captured parameter's gradient.
                            grad_div(param)

                        self.grad_acc.append(grad_acc)
                        self.grad_acc_hooks.append(
                            grad_acc.register_hook(grad_acc_hook)
                        )

                    wrapper(param)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both linear layers and returns the sum of the
                # result.
                x = self.fc1(x)
                x = self.fc2(x)
                return x.sum()

        bs = 8
        ioc = 16
        model = Module(ioc)
        input = torch.randn([bs, ioc])

        # eager ref
        model(input).backward()
        ref_res = [model.fc1.weight.grad, model.fc2.weight.grad]

        # cag
        model.fc1.weight.grad = None
        model.fc2.weight.grad = None
        model_to_train = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            model_to_train(input).backward()
        res = [model_to_train.fc1.weight.grad, model_to_train.fc2.weight.grad]

        self.assertEqual(res[0], ref_res[0])
        self.assertEqual(res[1], ref_res[1])

    def test_reorder_post_hook2(self):
        # [FUNCTION DOC] Checks ordering of post-hooks on two chained activation nodes
        # (sigmoid then tanh) where one hook scales grad outputs and the other scales grad
        # inputs; eager and compiled grads on the input must match.
        x = torch.randn([1, 4, 32, 32], requires_grad=True)
        y = torch.sigmoid(x)
        z = torch.tanh(y)

        if not isinstance(z.grad_fn, torch.autograd.graph.Node):
            raise AssertionError
        if not isinstance(y.grad_fn, torch.autograd.graph.Node):
            raise AssertionError
        handle_z = z.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
        handle_y = y.grad_fn.register_hook(lambda gI, gO: (gI[0] * 2,))
        z.sum().backward(retain_graph=True)
        ref_res = x.grad

        x.grad = None
        with compiled_autograd._enable(compiler_fn):
            z.sum().backward(retain_graph=True)
        res = x.grad

        self.assertEqual(res, ref_res)

    def test_reorder_post_hook3(self):
        # [FUNCTION DOC] Checks a post-hook on a convolution node that doubles the weight
        # gradient entry of the grad inputs; eager and compiled input grads must match.
        conv = torch.nn.Conv2d(4, 4, 3, bias=False)
        x = torch.randn([1, 4, 32, 32])
        y = conv(x)

        if not isinstance(y.grad_fn, torch.autograd.graph.Node):
            raise AssertionError
        # this hook will mul 2.0 to the conv weight gradient
        handle_y = y.grad_fn.register_hook(lambda gI, gO: (gI[0], gI[1] * 2, gI[2]))
        y.sum().backward(retain_graph=True)
        ref_res = x.grad

        x.grad = None
        with compiled_autograd._enable(compiler_fn):
            y.sum().backward(retain_graph=True)
        res = x.grad

        self.assertEqual(res, ref_res)

    def test_reorder_all_bwd_hooks(self):
        # [FUNCTION DOC] Combines every backward hook kind on two convolution weights (tensor
        # hook, post-accumulate hook, accumulator pre-hook and post-hook) and checks eager and
        # compiled grads match.
        def tensor_hook(grad):
            # [FUNCTION DOC] HELPER: Tensor hook subtracting two from the gradient.
            return grad.sub(2.0)

        def acc_grad_node_pre_hook(grad_out):
            # [FUNCTION DOC] HELPER: Accumulator pre-hook returning a tuple with the first
            # grad output divided by five.
            return (grad_out[0].div(5.0),)

        def post_acc_grad_hook(tensor):
            # [FUNCTION DOC] HELPER: Post accumulate-grad hook adding three to the tensor's
            # grad in place.
            tensor.grad.add_(3.0)

        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Two-convolution module whose constructor installs all hook
            # kinds on each weight and its accumulator node, so backward exercises every hook
            # ordering.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates two bias-free convolutions, finds each
                # weight's accumulator node, and registers the tensor hook, post-accumulate
                # hook, pre-hook and a post-hook that halves the grad.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]
                self.conv1.weight.register_hook(tensor_hook)
                self.conv1.weight.register_post_accumulate_grad_hook(post_acc_grad_hook)
                self.acc_grad1.register_prehook(acc_grad_node_pre_hook)

                def acc_grad_node_post_hook1(grad_in, grad_out):
                    # [FUNCTION DOC] HELPER: Post-hook on the first accumulator node;
                    # multiplies the first convolution's weight grad by one half in place.
                    self.conv1.weight.grad.mul_(0.5)

                self.acc_grad1.register_hook(acc_grad_node_post_hook1)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]
                self.conv2.weight.register_hook(tensor_hook)
                self.conv2.weight.register_post_accumulate_grad_hook(post_acc_grad_hook)
                self.acc_grad2.register_prehook(acc_grad_node_pre_hook)

                def acc_grad_node_post_hook2(grad_in, grad_out):
                    # [FUNCTION DOC] HELPER: Post-hook on the second accumulator node;
                    # multiplies the second convolution's weight grad by one half in place.
                    self.conv2.weight.grad.mul_(0.5)

                self.acc_grad2.register_hook(acc_grad_node_post_hook2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both convolutions and returns the sum.
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_reorder_multi_post_hooks(self):
        # [FUNCTION DOC] Checks multiple post-hooks on the same accumulator nodes run in
        # registration order under compiled autograd, matching eager grads.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Two-convolution module registering two in-place post-hooks
            # (multiply then subtract) on each weight's accumulator node.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the convolutions and registers two post-hooks
                # per accumulator node.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]

                def acc_grad_node1_post_hook1(grad_in, grad_out):
                    # [FUNCTION DOC] HELPER: Post-hook multiplying the first weight grad by
                    # one half in place.
                    self.conv1.weight.grad.mul_(0.5)

                def acc_grad_node1_post_hook2(grad_in, grad_out):
                    # [FUNCTION DOC] HELPER: Post-hook subtracting 0.3 from the first weight
                    # grad in place.
                    self.conv1.weight.grad.sub_(0.3)

                self.acc_grad1.register_hook(acc_grad_node1_post_hook1)
                self.acc_grad1.register_hook(acc_grad_node1_post_hook2)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]

                def acc_grad_node2_post_hook1(grad_in, grad_out):
                    # [FUNCTION DOC] HELPER: Post-hook multiplying the second weight grad by
                    # 0.3 in place.
                    self.conv2.weight.grad.mul_(0.3)

                def acc_grad_node2_post_hook2(grad_in, grad_out):
                    # [FUNCTION DOC] HELPER: Post-hook subtracting one half from the second
                    # weight grad in place.
                    self.conv2.weight.grad.sub_(0.5)

                self.acc_grad2.register_hook(acc_grad_node2_post_hook1)
                self.acc_grad2.register_hook(acc_grad_node2_post_hook2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both convolutions and returns the sum.
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_reorder_multi_pre_hooks(self):
        # [FUNCTION DOC] Checks multiple pre-hooks on the same accumulator nodes apply in
        # registration order under compiled autograd, matching eager grads.
        def acc_grad_node_pre_hook1(grad_out):
            # [FUNCTION DOC] HELPER: Accumulator pre-hook returning a tuple with the first
            # grad divided by five.
            return (grad_out[0].div(5.0),)

        def acc_grad_node_pre_hook2(grad_out):
            # [FUNCTION DOC] HELPER: Accumulator pre-hook returning a tuple with 0.3
            # subtracted from the first grad.
            return (grad_out[0].sub(0.3),)

        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Two-convolution module registering two pre-hooks on each
            # weight's accumulator node.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the convolutions and registers both pre-hooks
                # on each accumulator node.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]
                self.acc_grad1.register_prehook(acc_grad_node_pre_hook1)
                self.acc_grad1.register_prehook(acc_grad_node_pre_hook2)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]
                self.acc_grad2.register_prehook(acc_grad_node_pre_hook1)
                self.acc_grad2.register_prehook(acc_grad_node_pre_hook2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both convolutions and returns the sum.
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_reorder_multi_tensor_pre_hooks(self):
        # [FUNCTION DOC] Checks multiple tensor hooks on the same weights run in registration
        # order under compiled autograd, matching eager grads.
        def tensor_hook1(grad):
            # [FUNCTION DOC] HELPER: Tensor hook subtracting two from the gradient.
            return grad.sub(2.0)

        def tensor_hook2(grad):
            # [FUNCTION DOC] HELPER: Tensor hook halving the gradient.
            return grad.mul(0.5)

        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Two-convolution module registering two tensor hooks on each
            # weight.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the convolutions and registers both tensor
                # hooks on each weight.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(4, 4, 3, bias=False)
                self.conv2 = torch.nn.Conv2d(4, 4, 3, bias=False)

                self.acc_grad1 = self.conv1.weight.view_as(
                    self.conv1.weight
                ).grad_fn.next_functions[0][0]
                self.conv1.weight.register_hook(tensor_hook1)
                self.conv1.weight.register_hook(tensor_hook2)

                self.acc_grad2 = self.conv2.weight.view_as(
                    self.conv2.weight
                ).grad_fn.next_functions[0][0]
                self.conv2.weight.register_hook(tensor_hook1)
                self.conv2.weight.register_hook(tensor_hook2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both convolutions and returns the sum.
                y = self.conv1(x)
                y = self.conv2(y)
                return y.sum()

        input = torch.randn([1, 4, 32, 32])

        # eager ref
        model = TestModel()
        model(input).backward()
        ref_results = [model.conv1.weight.grad, model.conv2.weight.grad]

        # cag
        model.conv1.weight.grad = None
        model.conv2.weight.grad = None
        compiled_model = torch.compile(model, backend="inductor")
        with compiled_autograd._enable(compiler_fn):
            compiled_model(input).backward()
        results = [compiled_model.conv1.weight.grad, compiled_model.conv2.weight.grad]

        self.assertEqual(results[0], ref_results[0])
        self.assertEqual(results[1], ref_results[1])

    def test_torch_compile(self):
        # [FUNCTION DOC] Tests a model wrapped in torch.compile with full graph, running three
        # forward/backward iterations under compiled autograd with grads zeroed between them.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that compiles a linear-sigmoid model, runs
            # backward three times yielding the first layer grads and zeroing grads after
            # each.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Sigmoid(),
            )
            opt_model = torch.compile(model, fullgraph=True)

            for _ in range(3):
                x = torch.randn([1, 4])

                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    @parametrize("api", ("compile", "optimize"))
    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_compile_api(self, api, backend):
        # [FUNCTION DOC] Parametrized over the compile entry point (torch.compile or the
        # dynamo optimize decorator) and three backends. Wraps a function that runs backward
        # over three input batches while the compiled_autograd config is on, then checks
        # results equal eager and exactly two captures were recorded.
        def wrap(fn, backend):
            # [FUNCTION DOC] HELPER: Wraps a function with torch.compile or the dynamo
            # optimize decorator, depending on the parametrized api, using the given backend;
            # returns None for an unknown api.
            if api == "compile":
                return torch.compile(fn, backend=backend)
            elif api == "optimize":
                return torch._dynamo.optimize(backend)(fn)

        def fn(model, inputs):
            # [FUNCTION DOC] HELPER: Runs forward and backward for each input on the model,
            # collects the first layer weight and bias grads after each, zeroes grads, and
            # returns the list.
            res = []
            for inp in inputs:
                result = model(inp).sum()
                result.backward()
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                model.zero_grad()
            return res

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        inputs = [
            torch.randn([1, 4]),
            torch.randn([2, 4]),
            torch.randn([3, 4]),
        ]

        expected = fn(model, inputs)
        with config.patch(compiled_autograd=True):
            compiled_fn = wrap(fn, backend)
        actual = compiled_fn(model, inputs)
        self.assertEqual(expected, actual)
        self.assertEqual(counters["compiled_autograd"]["captures"], 2)

    @parametrize("api", ("compile", "optimize"))
    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_compile_api_disable(self, api, backend):
        # [FUNCTION DOC] Like the previous compile-api test, but runs the compiled function
        # inside the compiled autograd disable context; asserts results match eager and that
        # no compiled autograd counters were recorded at all.
        def wrap(fn, backend):
            # [FUNCTION DOC] HELPER: Wraps a function with torch.compile or the dynamo
            # optimize decorator, depending on the parametrized api.
            if api == "compile":
                return torch.compile(fn, backend=backend)
            elif api == "optimize":
                return torch._dynamo.optimize(backend)(fn)

        def fn(model, inputs):
            # [FUNCTION DOC] HELPER: Runs forward and backward per input on the model,
            # collecting first layer grads and zeroing grads each time; returns the list.
            res = []
            for inp in inputs:
                result = model(inp).sum()
                result.backward()
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                model.zero_grad()
            return res

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        inputs = [
            torch.randn([1, 4]),
            torch.randn([2, 4]),
            torch.randn([3, 4]),
        ]

        expected = fn(model, inputs)
        with config.patch(compiled_autograd=True):
            compiled_fn = wrap(fn, backend)
        with torch._dynamo.compiled_autograd._disable():
            actual = compiled_fn(model, inputs)
        self.assertEqual(expected, actual)
        self.assertTrue("compiled_autograd" not in counters)

    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_optimize_assert(self, backend):
        # [FUNCTION DOC] Checks the dynamo optimize_assert wrapper built while the
        # compiled_autograd config is on does not fail on a missing rebuild context when a
        # function without backward inside is run; backward is then called outside, so zero
        # captures are expected and grads must match eager.
        # can be merged into the test above once we support
        # no graph break on .backward

        def fn(model, inp):
            # [FUNCTION DOC] HELPER: Forward-only function returning the sum of the model
            # output; backward is intentionally done outside the compiled region.
            # NOTE: not calling .backward in the compiled fn
            return model(inp).sum()

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        inp = torch.randn([1, 4])

        out = fn(model, inp)
        out.backward()
        expected = [p.grad for p in model.parameters()]
        model.zero_grad()
        with config.patch(compiled_autograd=True):
            compiled_fn = torch._dynamo.optimize_assert(backend)(fn)

        # should not error due to undefined `rebuild_ctx`
        out = compiled_fn(model, inp)
        out.backward()
        actual = [p.grad for p in model.parameters()]
        self.assertEqual(expected, actual)
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)

    @config.patch(compiled_autograd=True)
    def test_nested_context_manager(self):
        # [FUNCTION DOC] Checks the compiled autograd context manager nesting rule: entering
        # outer then inner and exiting inner then outer is fine, but exiting the outer one
        # first raises an AssertionError saying nested contexts must return before their
        # parent.
        def ctx():
            # [FUNCTION DOC] HELPER: Creates a new compiled autograd enable context using
            # torch.compile as the compiler.
            return compiled_autograd._enable(torch.compile)

        # ok
        outer = ctx()
        inner = ctx()
        outer.__enter__()
        inner.__enter__()
        inner.__exit__(None, None, None)
        outer.__exit__(None, None, None)

        # not ok
        outer = ctx()
        inner = ctx()
        outer.__enter__()
        inner.__enter__()
        with self.assertRaisesRegex(
            AssertionError,
            "Nested Compiled Autograd Contexts must return before their parent context",
        ):
            outer.__exit__(None, None, None)

    @config.patch(compiled_autograd=True)
    def test_nested_compile(self):
        # [FUNCTION DOC] Checks a custom library op whose implementation internally uses an
        # eager torch.compile, called from the backward of a custom autograd Function while
        # the whole function is compiled under compiled autograd; the test only requires that
        # it runs without error.
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            lib.define("square(Tensor x) -> Tensor")

            @torch.library.impl("testlib::square", "CPU")
            def square_impl(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: CPU implementation of the square test op; compiles a
                # nested squaring function with the eager backend and applies it to the input.
                # nested inference graph compile
                @torch.compile(backend="eager")
                def fn(x):
                    # [FUNCTION DOC] HELPER: Nested function squaring its input.
                    return x**2

                return fn(x)

            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function whose forward is the identity and
                # whose backward calls the test library square op on the gradient.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Identity forward.
                    return x

                @staticmethod
                def backward(ctx, x):
                    # [FUNCTION DOC] HELPER: Backward that applies the registered square op to
                    # the incoming gradient.
                    return torch.ops.testlib.square(x)

            x = torch.tensor([2.0, 3.0], requires_grad=True)

            @torch.compile
            def fn(x):
                # [FUNCTION DOC] HELPER: Compiled function applying the custom autograd
                # Function to its input.
                return MyFn.apply(x)

            fn(x).sum().backward()

    @config.patch(compiled_autograd=True)
    def test_no_nested_compiled_autograd(self):
        # [FUNCTION DOC] Checks that compiled autograd is disabled once inside a captured
        # graph: a custom Function backward runs an unrelated eager autograd call, and the
        # outer compiled backward with create_graph yields exactly one capture.
        # We disable CA before entering the CA graph
        # So re-entrants should be running with the eager autograd engine

        def unrelated_autograd_call():
            # [FUNCTION DOC] HELPER: Builds two random matrices requiring grad, multiplies
            # them, sums and runs backward; used to start a nested eager backward.
            x = torch.randn(20, 20, requires_grad=True)
            y = torch.randn(20, 20, requires_grad=True)
            loss = torch.matmul(x, y).sum()
            loss.backward()

        class MyFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function whose backward triggers a nested,
            # unrelated backward call and returns the gradient unchanged.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Identity forward.
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Runs the unrelated nested backward helper and returns
                # the incoming gradient.
                unrelated_autograd_call()
                return gO

        x = torch.randn(10, 10, requires_grad=True)
        loss = MyFn.apply(x).sum()

        torch.compile(lambda: loss.backward(create_graph=True))()
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_multiple_torch_compile(self):
        # [FUNCTION DOC] Checks that two functions compiled by torch.compile, one without and
        # one with the compiled_autograd config, only capture compiled autograd for the one
        # built under the config; a later compile of the first function outside the config
        # again captures nothing.
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        def fn():
            # [FUNCTION DOC] HELPER: Runs forward on the first model and calls backward on the
            # sum.
            result = model(x).sum()
            result.backward()

        model2 = torch.nn.Linear(4, 4)
        x2 = torch.randn([1, 4])

        def fn2():
            # [FUNCTION DOC] HELPER: Runs forward on the second model and calls backward on
            # the sum.
            result = model2(x2).sum()
            result.backward()

        no_ca1 = torch.compile(fn)
        no_ca1()
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        counters.clear()

        with config.patch(compiled_autograd=True):
            with_ca = torch.compile(fn2)
            with_ca()
            self.assertEqual(counters["compiled_autograd"]["captures"], 1)
            counters.clear()

        no_ca2 = torch.compile(fn)
        no_ca2()
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)

    def test_torch_compile_graph_break(self):
        # [FUNCTION DOC] Checks that a dynamo-disabled function containing backward, called
        # from a compiled function under the compiled_autograd config, still results in one
        # capture.
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        @torch._dynamo.disable()
        def fn():
            # [FUNCTION DOC] HELPER: Dynamo-disabled function that runs forward and backward
            # on the shared model.
            result = model(x).sum()
            result.backward()

        with config.patch(compiled_autograd=True):
            opt_fn = torch.compile(fn)
            opt_fn()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_graph_break2(self):
        # [FUNCTION DOC] Checks one capture when backward is called inside a dynamo-disabled
        # helper that is itself called from a compiled function under the compiled_autograd
        # config.
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        @torch._dynamo.disable()
        def inner_fn(loss):
            # [FUNCTION DOC] HELPER: Dynamo-disabled function that calls backward on the given
            # loss.
            loss.backward()

        def fn():
            # [FUNCTION DOC] HELPER: Computes the model loss and hands it to the disabled
            # helper.
            result = model(x).sum()
            inner_fn(result)

        with config.patch(compiled_autograd=True):
            opt_fn = torch.compile(fn)
            opt_fn()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_only_backward_call(self):
        # [FUNCTION DOC] Checks that compiling only a function that calls backward on a
        # precomputed loss under the compiled_autograd config produces one capture.
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        result = model(x).sum()
        with config.patch(compiled_autograd=True):
            opt_bwd = torch.compile(lambda: result.backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_dynamo_boxed(self):
        # [FUNCTION DOC] Checks the boxed-input convention: the forward graph compiled by
        # dynamo gets unboxed placeholders, while the compiled autograd backward graph
        # receives a list as its first placeholder. Runs matrix multiplications, calls
        # backward under a compiled autograd context with a checking compiler, and asserts all
        # three input grads are present.
        def get_placeholders(gm_):
            # [FUNCTION DOC] HELPER: Returns the list of placeholder nodes in the given graph
            # module.
            placeholders = []
            for node in gm_.graph.nodes:
                if node.op == "placeholder":
                    placeholders.append(node)
            return placeholders

        def eager_with_check(gm, is_bwd):
            # [FUNCTION DOC] HELPER: Compiler factory that compiles the given module with a
            # custom backend verifying whether the first placeholder example value is a list
            # (required when the is_bwd flag is set, forbidden otherwise); raises
            # AssertionError on violation.
            def inner_compiler(gm_, example_inputs_):
                # [FUNCTION DOC] HELPER: Custom backend that fetches placeholders, asserts the
                # boxed or unboxed expectation, and returns the graph module unchanged.
                placeholders = get_placeholders(gm_)
                if is_bwd:
                    # boxed inputs
                    if not isinstance(placeholders[0].meta["example_value"], list):
                        raise AssertionError
                else:
                    # not boxed inputs
                    if isinstance(placeholders[0].meta["example_value"], list):
                        raise AssertionError

                return gm_

            return torch.compile(gm, backend=inner_compiler)

        bwd_compiler_fn = functools.partial(eager_with_check, is_bwd=True)

        def fn(inputs):
            # [FUNCTION DOC] HELPER: Generator that unpacks three tensors, multiplies them as
            # matrices, sums, runs backward under the compiled autograd context with the
            # boxed-checking compiler, and yields the three input grads.
            args_0, args_1, args_2 = inputs
            out = torch.mm(args_0, args_1)
            out = torch.mm(out, args_2)
            loss = out.sum()
            with compiled_autograd._enable(bwd_compiler_fn):
                loss.backward()
            yield args_0.grad
            yield args_1.grad
            yield args_2.grad

        inputs = [
            torch.randn([1, 2], requires_grad=True),
            torch.randn([2, 3], requires_grad=True),
            torch.randn([3, 4], requires_grad=True),
        ]

        compiled_fn = eager_with_check(fn, is_bwd=False)
        grads = list(compiled_fn(inputs))
        self.assertEqual(len(grads), 3)
        self.assertNotEqual(grads[0], None)
        self.assertNotEqual(grads[1], None)
        self.assertNotEqual(grads[2], None)

    def test_inputs_aliasing_bytecode_attr_mutations(self):
        # [FUNCTION DOC] Checks bytecode produced when compiled autograd runs a captured
        # graph: after a hand-built capture of a gradient accumulation, a bytecode hook
        # inspects the instructions after the compiled graph call and verifies the inputs list
        # is not accessed there, only an alias taken earlier (at most one load, with a store
        # existing only if a load exists). Finally checks that the parameter grad equals twice
        # the active tensor. Relies on bytecode instruction names of the running interpreter.
        # Freeze compiled autograd graph
        compiler = torch._dynamo.compiled_autograd.AutogradCompilerInstance(compiler_fn)
        param = torch.ones(100)
        active = torch.ones(100) * 2
        inputs = [param, active]
        _, proxies, _, _ = compiler.begin_capture(
            inputs=inputs,
            sizes=[],
            scalars=[],
            origins=[[], [], []],
            accumulate_grad=False,
            check_nans=False,
        )
        param_proxy, activ_proxy = proxies
        buf = activ_proxy * 2
        if len(torch.ops.inductor.accumulate_grad_.default._schema.arguments) == 2:
            torch.ops.inductor.accumulate_grad_.default(param_proxy, buf)
        else:
            param_proxy.grad = torch.ops.inductor.accumulate_grad_.default(
                param_proxy, param_proxy.grad, buf
            )
        runtime_wrapper, compiled_fn = compiler.end_capture(buf)

        def bytecode_hook(code, out_code):
            # [FUNCTION DOC] HELPER: Bytecode hook given the original and transformed code.
            # Locates the load of the compiled function global and the call following it, then
            # asserts there are no references to the inputs variable after the call, the alias
            # loads number at most one and the alias stores equal whether a load exists.
            # Chooses the call opcode name by interpreter version.
            import dis
            import sys

            if sys.version_info < (3, 11):
                call_op = "CALL_FUNCTION"
            else:
                call_op = "CALL"

            insts = list(dis.get_instructions(out_code))
            # Find the CALL that invokes the compiled graph function
            # (not an earlier CALL from e.g. store_user_object_weakrefs).
            # The compiled fn is loaded via LOAD_GLOBAL __compiled_fn_*.
            load_graph_idx = next(
                i
                for i, inst in enumerate(insts)
                if inst.opname == "LOAD_GLOBAL"
                and isinstance(inst.argval, str)
                and inst.argval.startswith("__compiled_fn")
            )
            call_graph_idx = next(
                i
                for i, inst in enumerate(insts)
                if i > load_graph_idx and inst.opname == call_op
            )
            # If post-graph bytecode needs inputs[0], it must use an alias
            # captured before the graph call because the inputs list may be
            # stolen by compiled graph execution.
            pre_alias_stores = [
                inst
                for inst in insts[:call_graph_idx]
                if inst.opname == "STORE_FAST" and inst.argval == "inputs_ref_0"
            ]
            # post-graph should access inputs_ref_0 instead of inputs
            matches = [
                inst for inst in insts[call_graph_idx:] if inst.argval == "inputs"
            ]
            self.assertTrue(len(matches) == 0)
            post_alias_loads = [
                inst
                for inst in insts[call_graph_idx:]
                if inst.opname == "LOAD_FAST" and inst.argval == "inputs_ref_0"
            ]
            self.assertEqual(len(pre_alias_stores), int(bool(post_alias_loads)))
            self.assertLessEqual(len(post_alias_loads), 1)

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)
        try:
            runtime_wrapper(
                compiled_fn=compiled_fn,
                inputs=[param, active],
                sizes=(),
                scalars=(),
                hooks=[],
                packed_inputs=[],
            )
        finally:
            handle.remove()
        self.assertEqual(param.grad, active * 2)

    def test_inputs_aliasing_bytecode_stack_restore(self):
        # [FUNCTION DOC] Checks input-stealing bytecode with a tensor subclass: compiles a
        # traced function whose locals are marked as stolen, runs it with one plain tensor and
        # one logging tensor, and asserts through a bytecode hook that exactly one alias store
        # occurs before the graph call and one alias load after it, never the inputs variable.
        # Afterwards the inputs list must be empty because the graph stole it.
        logging.getLogger().setLevel(logging.WARNING)
        from torch.testing._internal.logging_tensor import LoggingTensor

        # Create a graph that allows inputs stealing
        def forward(inputs):
            # [FUNCTION DOC] HELPER: Function adding one to the first input, adding the second
            # input, moving the result to CPU and returning it in a tuple; traced
            # symbolically.
            add = inputs[0] + 1
            add_1 = add + inputs[1]  # handled in suffix for tensor subclass
            out = add_1.cpu()
            return (out,)

        gm = torch.fx.symbolic_trace(forward)
        torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
        compiled_fn = torch.compile(gm)

        inputs = [
            torch.ones(1000000, dtype=torch.float32),
            LoggingTensor(torch.ones(1)),
        ]
        match_done = False

        def bytecode_hook(code, out_code):
            # [FUNCTION DOC] HELPER: Bytecode hook that, only for the first checked frame
            # (tracked by an enclosing flag), asserts the alias is stored once before the
            # compiled graph call, never referenced by name after it, and loaded once after
            # it, then sets the done flag.
            import dis
            import sys

            nonlocal match_done

            # test is sensitive to what Dynamo traces. So as soon as the main
            # graph is tested, we skip the bytecode hook checks for future
            # frames.
            if not match_done:
                if sys.version_info < (3, 11):
                    call_op = "CALL_FUNCTION"
                else:
                    call_op = "CALL"

                insts = list(dis.get_instructions(out_code))
                # Find the CALL that invokes the compiled graph function
                # (not an earlier CALL from e.g. store_user_object_weakrefs).
                load_graph_idx = next(
                    i
                    for i, inst in enumerate(insts)
                    if inst.opname == "LOAD_GLOBAL"
                    and isinstance(inst.argval, str)
                    and inst.argval.startswith("__compiled_fn")
                )
                call_graph_idx = next(
                    i
                    for i, inst in enumerate(insts)
                    if i > load_graph_idx and inst.opname == call_op
                )
                # pre-graph should alias: inputs_ref_0 = inputs[0]
                matches = [
                    inst
                    for inst in insts[:call_graph_idx]
                    if inst.opname == "STORE_FAST" and inst.argval == "inputs_ref_0"
                ]
                self.assertTrue(len(matches) == 1)
                # post-graph should access inputs_ref_0 instead of inputs
                matches = [
                    inst for inst in insts[call_graph_idx:] if inst.argval == "inputs"
                ]
                self.assertTrue(len(matches) == 0)
                matches = [
                    inst
                    for inst in insts[call_graph_idx:]
                    if inst.opname == "LOAD_FAST" and inst.argval == "inputs_ref_0"
                ]
                self.assertTrue(len(matches) == 1)
                match_done = True

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)
        try:
            compiled_fn(inputs)
            self.assertTrue(len(inputs) == 0)
        finally:
            handle.remove()

    def test_implicit_add(self):
        # [FUNCTION DOC] Checks gradients accumulate correctly when one tensor is used several
        # times in the forward (implicit add of grads), over three iterations.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that evaluates a model using the same tensor in
            # three terms, runs backward, yields the loss and grad of that tensor, then clears
            # the grad, three times.
            y = torch.randn(1, 4, requires_grad=True)

            def model(x):
                # [FUNCTION DOC] HELPER: Model applying a sigmoid to a product plus sine and
                # cosine of the shared tensor.
                # y is used multiple times, gradients get added
                return torch.sigmoid(x * y + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.backward()
                yield result
                yield y.grad
                y.grad = None

        self.check_output_and_recompiles(fn)

    def test_output_nodes_all_leaves(self):
        # [FUNCTION DOC] Checks autograd.grad with all leaf tensors as requested outputs under
        # compiled autograd; neither leaf may get its grad attribute set.
        def fn():
            # [FUNCTION DOC] HELPER: Generator computing grads of the loss with respect to two
            # leaves via the functional grad API, asserting grad attributes stay unset, and
            # yielding both grads, three times.
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                # [FUNCTION DOC] HELPER: Model combining the input with the two leaves through
                # sigmoid, sine and cosine.
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                gy, gz = torch.autograd.grad(result, inputs=[y, z])
                if y.grad is not None:
                    raise AssertionError
                if z.grad is not None:
                    raise AssertionError
                yield gy
                yield gz

        self.check_output_and_recompiles(fn)

    def test_output_nodes_some_leaves(self):
        # [FUNCTION DOC] Checks autograd.grad requesting only some leaves, where another leaf
        # is behind a custom Function whose backward raises; compiled autograd must not run
        # the unreachable backward.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that computes the grad of only one leaf through
            # a graph that also contains the raising Function, asserts no grad attributes are
            # populated, and yields the grad, three times.
            class UnreachableBwd(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function with identity forward and a backward
                # that raises RuntimeError; used to prove a branch is never executed.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Identity forward.
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Backward that always raises RuntimeError.
                    raise RuntimeError

            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                # [FUNCTION DOC] HELPER: Model multiplying the output of the raising Function
                # by a second leaf and applying a sigmoid.
                return torch.sigmoid(UnreachableBwd.apply(y) * z)

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                gz = torch.autograd.grad(result, inputs=[z])
                if y.grad is not None:
                    raise AssertionError
                if z.grad is not None:
                    raise AssertionError
                yield gz

        self.check_output_and_recompiles(fn)

    def test_no_output_nodes_all_leaves(self):
        # [FUNCTION DOC] Checks plain backward on a graph with several leaves: backward
        # returns None, both leaf grads are set and match eager.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running backward, asserting it returns None and
            # that both leaf grads are populated, yielding both grads and clearing them, three
            # times.
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                # [FUNCTION DOC] HELPER: Model combining the input with the two leaves through
                # sigmoid, sine and cosine.
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])
                result = model(x).sum()
                out = result.backward()
                if out is not None:
                    raise AssertionError
                if y.grad is None:
                    raise AssertionError
                if z.grad is None:
                    raise AssertionError
                yield y.grad
                yield z.grad
                y.grad = None
                z.grad = None

        self.check_output_and_recompiles(fn)

    def test_no_output_nodes_some_leaves(self):
        # [FUNCTION DOC] Checks backward restricted with the inputs argument to a subset of
        # leaves; the third leaf sits behind a raising Function and must stay without a grad.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running backward with two chosen inputs,
            # asserting the returned value is None, the two grads are set and the excluded one
            # is unset, yielding both grads, three times.
            class UnreachableBwd(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function with identity forward and a backward
                # that raises RuntimeError.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Identity forward.
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Backward that always raises RuntimeError.
                    raise RuntimeError

            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)
            a = torch.randn(1, 4, requires_grad=True)

            def model(x):
                # [FUNCTION DOC] HELPER: Model multiplying the input by two leaves and the
                # output of the raising Function, then applying a sigmoid.
                return torch.sigmoid(x * y * z * UnreachableBwd.apply(a))

            for _ in range(3):
                x = torch.randn([1, 4])
                result = model(x).sum()
                out = result.backward(inputs=[y, z])
                if out is not None:
                    raise AssertionError
                if y.grad is None:
                    raise AssertionError
                if z.grad is None:
                    raise AssertionError
                if a.grad is not None:
                    raise AssertionError
                yield y.grad
                yield z.grad
                y.grad = None
                z.grad = None

        self.check_output_and_recompiles(fn)

    def test_no_output_nodes_different_leaves_will_recompile(self):
        # [FUNCTION DOC] Checks that backward limited to different input leaves between calls
        # forces a recompile: two compiled backward calls restrict to different inputs and two
        # captures and compiles are expected.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running two compiled restricted backward calls
            # on freshly computed losses, yielding the grad of the first requested leaf and
            # then the second.
            def fwd(x, y, z):
                # [FUNCTION DOC] HELPER: Multiplies the three tensors in sequence and returns
                # the sum.
                out = x * y  # MulBackward0
                out2 = out * z  # MulBackward0
                return out2.sum()  # SumBackward0

            x = torch.randn(5, requires_grad=True)
            y = torch.randn(5, requires_grad=True)
            z = torch.randn(5, requires_grad=True)
            loss = fwd(x, y, z)
            torch.compile(lambda: torch.autograd.backward(loss, inputs=[x]))()
            yield x.grad
            x.grad = None

            loss = fwd(x, y, z)
            torch.compile(lambda: torch.autograd.backward(loss, inputs=[y]))()
            yield y.grad

        # Guarded by TensorArg id, mismatch on last MulBackward0
        self.check_output_and_recompiles(fn, 2)

    def test_dynamic_shapes(self):
        # [FUNCTION DOC] Checks compiled autograd with a forward compiled with dynamic shapes
        # and varying batch sizes; defaults of one capture and one compile apply.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running a dynamic-shape compiled model backward
            # for batch sizes ten to ninety and yielding the four parameter grads each time
            # while zeroing grads.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for b in range(10, 100, 10):
                x = torch.randn([b, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    def test_dynamic_shapes_from_forward(self):
        # [FUNCTION DOC] Checks that dynamic shapes coming from the compiled forward are
        # reused by compiled autograd: runs a compiled model with batch sizes three through
        # six, expecting one capture and compile for the first two sizes and none for the next
        # two.
        class ToyModel(nn.Module):
            # [CLASS DOC] HELPER: Three-layer linear network with a mean squared error loss on
            # its output.
            def __init__(self, in_feat=10, hidden_feat=50, out_feat=5):
                # [FUNCTION DOC] HELPER: Creates three linear layers with the given feature
                # sizes and an MSE loss module.
                super().__init__()
                self.linear1 = nn.Linear(in_feat, hidden_feat)
                self.linear2 = nn.Linear(hidden_feat, hidden_feat)
                self.linear3 = nn.Linear(hidden_feat, out_feat)
                self.mse_loss = torch.nn.MSELoss()

            def forward(self, inputs, output):
                # [FUNCTION DOC] HELPER: Applies the three linear layers in sequence and
                # returns the MSE loss against the target.
                out1 = self.linear1(inputs)
                out2 = self.linear2(out1)
                out3 = self.linear3(out2)
                return self.mse_loss(out3, output)

        m = ToyModel()
        m = torch.compile(m)

        def run(i):
            # [FUNCTION DOC] HELPER: Clears dynamo counters, runs the compiled model on a
            # random batch of the given size and runs backward under a compiled autograd
            # context whose compiler uses automatic dynamic shapes.
            torch._dynamo.utils.counters.clear()
            inp = torch.randn(i, 10)
            target = torch.randn(i, 5)
            loss = m(inp, target)
            with compiled_autograd._enable(make_compiler_fn(dynamic=None)):
                loss.backward()

        counters = torch._dynamo.utils.counters
        run(3)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 1)
        run(4)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 1)
        run(5)
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 0)
        run(6)
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 0)

    def test_dynamic_shapes_eager_node(self):
        # [FUNCTION DOC] Checks a case where a node in the eager-only section (the sum
        # backward node) saves symbolic sizes that cannot be marked dynamic; default one
        # capture and compile still hold.
        # Here, we have no way of marking the symbolic sizes using in SumBackward as dynamic
        def fn():
            # [FUNCTION DOC] HELPER: Generator that runs a dynamic-shape compiled model on
            # varied batch sizes, reshapes the output, sums and runs backward, yielding the
            # four parameter grads and zeroing grads each time.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for b, s in zip([10, 20, 30], [2, 4, 8]):
                x = torch.randn([b, 4])
                result = opt_model(x)
                view = result.view(s, -1)
                # sum will save dynamic sizes
                loss = view.sum()
                loss.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    def test_dynamic_shapes_annotations(self):
        # [FUNCTION DOC] Checks that marking an input dimension dynamic on a compiled function
        # does not trigger a constraint violation under compiled autograd: two calls with
        # different sizes give a single capture.
        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function applying sine twice.
            return x.sin().sin()

        with torch._dynamo.compiled_autograd._enable(torch.compile):
            x = torch.randn(2, 3, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            out = f(x)
            out.sum().backward()

            x = torch.randn(4, 3, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            out = f(x)
            out.sum().backward()

        # mark_dynamic should not cause ConstraintViolationError
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_torch_compile_api_dynamic_shapes(self):
        # [FUNCTION DOC] Checks the torch.compile API with dynamic shapes applied to a
        # function that only calls backward, while a model runs eagerly with varying batch
        # sizes; eager and compiled results are computed and exactly one capture is expected.
        # Here, we have no way of marking the symbolic sizes using in SumBackward as dynamic
        def fn(call_backward):
            # [FUNCTION DOC] HELPER: Generator that builds a model, runs it on three batch
            # sizes, reshapes and sums the output, delegates the backward call to the supplied
            # callable and yields the four parameter grads, zeroing grads each time.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )

            for b, s in zip([10, 20, 30], [2, 4, 8]):
                x = torch.randn([b, 4])
                result = model(x)
                view = result.view(s, -1)
                # sum will save dynamic sizes
                loss = view.sum()
                call_backward(loss)
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        def call_backward(loss):
            # [FUNCTION DOC] HELPER: Calls backward on the given loss.
            loss.backward()

        eager_out = list(fn(call_backward))
        with config.patch(compiled_autograd=True):
            compiled_out = list(fn(torch.compile(call_backward, dynamic=True)))
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    def test_accumulate_without_zero(self):
        # [FUNCTION DOC] Checks grads accumulate across ten iterations without zeroing when
        # the forward model is compiled with dynamic shapes; two captures and compiles are
        # expected (first backward with no existing grads, later ones accumulating).
        def fn():
            # [FUNCTION DOC] HELPER: Generator running ten backward passes without zeroing and
            # yielding clones of the four parameter grads each time.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for _ in range(10):
                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad.clone()
                yield model[0].bias.grad.clone()
                yield model[2].weight.grad.clone()
                yield model[2].bias.grad.clone()

        self.check_output_and_recompiles(fn, count=2)

    def test_inplace_grad_update(self):
        # [FUNCTION DOC] Checks that existing grad tensors preset on parameters are updated in
        # place by compiled autograd (same tensor object afterwards), over ten iterations with
        # one capture.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that assigns fresh random grads to the first
            # layer parameters, runs backward, asserts the grad objects are unchanged
            # identity-wise and yields clones of them, ten times.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for _ in range(10):
                w_grad = torch.rand_like(model[0].weight)
                b_grad = torch.rand_like(model[0].bias)
                model[0].weight.grad = w_grad
                model[0].bias.grad = b_grad

                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                if model[0].weight.grad is not w_grad:
                    raise AssertionError
                if model[0].bias.grad is not b_grad:
                    raise AssertionError
                yield w_grad.clone()
                yield b_grad.clone()

        self.check_output_and_recompiles(fn, count=1)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_issue106555(self):
        # [FUNCTION DOC] Regression test for issue 106555 (GPU only): a model containing
        # compiled bias-sigmoid-multiply submodules run through reentrant gradient
        # checkpointing, repeated ten times with grads cleared. It only needs to run without
        # raising.
        DEVICE = torch.device(GPU_TYPE, 0)
        NUM_FEATURES = 256

        def bias_sigmoid_mul(x1, x2, bias):
            # [FUNCTION DOC] HELPER: Adds the bias to the second input, applies a sigmoid,
            # multiplies elementwise by the first input and returns the product.
            x2 = torch.sigmoid(x2 + bias)
            y = x1 * x2
            return y

        bias_sigmoid_mul_jit = torch.compile(bias_sigmoid_mul)

        class ModuleWithJit(nn.Module):
            # [CLASS DOC] HELPER: Module with two linear layers (one without bias) and a
            # separate bias parameter, whose forward combines their outputs with the compiled
            # bias-sigmoid-multiply function.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two linear layers and a zero-initialised
                # bias parameter of the feature size.
                super().__init__()
                self.linear_1 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=True)
                self.linear_2 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=False)
                self.linear_2_bias = nn.Parameter(torch.zeros(NUM_FEATURES))

            def forward(self, input_tensor):
                # [FUNCTION DOC] HELPER: Applies both linear layers to the input and returns
                # the compiled gated combination.
                x1 = self.linear_1(input_tensor)
                x2 = self.linear_2(input_tensor)
                output = bias_sigmoid_mul_jit(x1, x2, self.linear_2_bias)
                return output

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Wrapper holding two ModuleWithJit instances; forward
            # optionally runs the inner computation through reentrant checkpointing.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two submodules.
                super().__init__()
                self.module_with_jit_1 = ModuleWithJit()
                self.module_with_jit_2 = ModuleWithJit()

            def forward(self, x, gradient_checkpointing: bool):
                # [FUNCTION DOC] HELPER: Runs the inner computation, through reentrant
                # checkpointing when the flag is set, otherwise directly, and returns the
                # result.
                if gradient_checkpointing:
                    y = torch.utils.checkpoint.checkpoint(
                        self._forward, x, use_reentrant=True
                    )
                else:
                    y = self._forward(x)
                return y

            def _forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the first submodule output as a residual, then
                # applies the second submodule on a transposed view of the activations,
                # transposes back and adds as another residual.
                x = x + self.module_with_jit_1(x)
                x = x + self.module_with_jit_2(x.transpose(-2, -3)).transpose(-2, -3)
                return x

        device_interface = get_interface_for_device(GPU_TYPE)
        device_interface.set_device(device=DEVICE)
        torch.manual_seed(1234567890)
        model = Model()
        model.train()
        model.to(device=DEVICE)
        model_parameters = list(model.parameters())

        torch.manual_seed(1234567890)
        input_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(device=DEVICE)
        input_tensor.requires_grad = True
        target_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(
            dtype=input_tensor.dtype, device=DEVICE
        )

        for _ in range(10):
            for param in model_parameters:
                param.grad = None
            output_tensor = model(
                x=input_tensor.clone(),
                gradient_checkpointing=True,
            )
            loss = torch.mean(torch.abs(target_tensor - output_tensor))
            loss.backward()

    def test_keep_graph_simple(self):
        # [FUNCTION DOC] Checks retain_graph with compiled autograd: a first eager backward on
        # a squared scalar gives gradient four, then the compiled-vs-eager helper re-runs
        # backward on the retained graph.
        x = torch.tensor([2.0], requires_grad=True)
        y = x**2

        # First backward pass; keep the computation graph
        y.backward(retain_graph=True)
        self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4

        # Note - this will run under both the eager and compiled regime.
        def fn():
            # [FUNCTION DOC] HELPER: Resets the grad to zero, runs backward with retained
            # graph, asserts the grad equals four and returns it.
            # Reset the gradients
            x.grad = torch.tensor([0.0])
            # Second and Third backward pass; keep the computation graph
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            return x.grad

        self.check_output_and_recompiles(fn, count=1)

    def test_keep_graph_usage_after_compiled(self):
        # [FUNCTION DOC] Checks that a graph retained through eager backward can still be used
        # by backward under compiled autograd and then eagerly again, across five iterations.
        x = torch.tensor([2.0], requires_grad=True)
        y = x**2

        # First backward pass; keep the computation graph
        def eager_check():
            # [FUNCTION DOC] HELPER: Runs backward with retained graph, asserts the grad is
            # four, then resets the grad to zero.
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            x.grad = torch.tensor([0.0])

        eager_check()

        for _ in range(5):
            with compiled_autograd._enable(compiler_fn):
                eager_check()

            eager_check()

    def test_custom_fn_saved_tensors(self):
        # [FUNCTION DOC] Tests a custom Function saving its input for backward, with varying
        # input lengths; default one capture and compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator defining a sine custom Function that saves its
            # input and runs it on arange tensors of several sizes, yielding the input grad
            # each time.
            class MySin(torch.autograd.Function):
                # [CLASS DOC] HELPER: Custom Function computing the sine and saving the input
                # for backward.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Saves the input and returns its sine.
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Loads the saved input and returns the gradient
                    # times the cosine of the input.
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MySin.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_multiple_tensors(self):
        # [FUNCTION DOC] Tests a custom Function that saves two input tensors and returns two
        # outputs, with varying sizes.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running the two-input custom Function on arange
            # tensors of several sizes, summing the product of its outputs and yielding the
            # first input grad.
            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Custom Function with two inputs and two sine outputs,
                # saving both inputs.
                @staticmethod
                def forward(ctx, x, y):
                    # [FUNCTION DOC] HELPER: Saves both inputs and returns their sines as a
                    # tuple.
                    ctx.save_for_backward(x, y)
                    return torch.sin(x), torch.sin(y)

                @staticmethod
                def backward(ctx, gO_x, gO_y):
                    # [FUNCTION DOC] HELPER: Returns each gradient times the cosine of the
                    # matching saved input.
                    (x, y) = ctx.saved_tensors
                    return gO_x * torch.cos(x), gO_y * torch.cos(y)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                y = torch.arange(0.0, i, requires_grad=True)
                out1, out2 = MyFn.apply(x, y)
                loss = (out1 * out2).sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_multiple_tensors_dedup(self):
        # [FUNCTION DOC] Tests a custom Function that saves the same tensor twice, so the
        # saved tensors must be deduplicated.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running the duplicate-saving Function on arange
            # tensors of several sizes and yielding the input grad.
            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Custom Function saving its single input twice.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Saves the input twice and returns its sine.
                    ctx.save_for_backward(x, x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Returns the gradient times the cosine of both
                    # saved copies.
                    (x1, x2) = ctx.saved_tensors
                    return gO * torch.cos(x1) * torch.cos(x2)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MyFn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_shape_tensor(self):
        # [FUNCTION DOC] Tests a custom Function whose backward uses the shape of a saved
        # tensor, with varying sizes (so dynamic shape handling is required).
        def fn():
            # [FUNCTION DOC] HELPER: Generator running the shape-using Function on arange
            # tensors of several sizes and yielding the input grad.
            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Identity custom Function that saves its input.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Saves the input and returns it unchanged.
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Returns the gradient scaled by the first
                    # dimension size of the saved input.
                    (x,) = ctx.saved_tensors
                    return gO * x.shape[0]

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MyFn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_saved_attr(self):
        # [FUNCTION DOC] Tests a custom Function that stores the input shape as a plain
        # attribute on its context; the compiler is run without full graph so graph breaks are
        # tolerated.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running the attribute-storing Function on
            # arange tensors of several sizes and yielding the input grad.
            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Identity custom Function that stores the input shape as
                # a context attribute.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Stores the shape in the context and returns the
                    # input.
                    ctx.shape = x.shape
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Returns the gradient scaled by the first
                    # dimension of the stored shape.
                    x_shape = ctx.shape[0]
                    return gO * x_shape

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MyFn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_custom_fn_multiple_grads(self):
        # [FUNCTION DOC] Tests a custom Function with two inputs and two outputs, where
        # backward returns both incoming gradients.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running the pass-through two-output Function on
            # varying sizes, summing both outputs and yielding both input grads.
            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Custom Function returning the sum of its inputs and the
                # second input, with a backward returning the two gradients unchanged.
                @staticmethod
                def forward(ctx, x, y):
                    # [FUNCTION DOC] HELPER: Returns the sum of the inputs and the second
                    # input.
                    return x + y, y

                @staticmethod
                def backward(ctx, gO_1, gO_2):
                    # [FUNCTION DOC] HELPER: Returns both incoming gradients unchanged.
                    return gO_1, gO_2

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                y = torch.arange(0.0, i, requires_grad=True)
                out1, out2 = MyFn.apply(x, y)
                loss = (out1 + out2).sum()
                loss.backward()
                yield x.grad
                yield y.grad

        self.check_output_and_recompiles(fn)

    def test_custom_fn_non_variable_input(self):
        # [FUNCTION DOC] Tests a custom Function receiving a non-tensor integer argument among
        # its inputs; the test yields the inputs themselves.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running the three-input Function with an
            # integer in the middle on varying sizes and yielding the inputs after backward.
            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Custom Function scaling its three inputs by two, three
                # and four; backward returns the gradients unchanged.
                @staticmethod
                def forward(ctx, x, y, z):
                    # [FUNCTION DOC] HELPER: Returns the inputs scaled by two, three and four.
                    return x * 2, y * 3, z * 4

                @staticmethod
                def backward(ctx, gO_1, gO_2, gO_3):
                    # [FUNCTION DOC] HELPER: Returns the three gradients unchanged.
                    return gO_1, gO_2, gO_3

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                y = 1
                z = torch.arange(0.0, i, requires_grad=True)
                out1, out2, out3 = MyFn.apply(x, y, z)
                loss = (out1 + out2 + out3).sum()
                loss.backward()
                yield x
                yield y
                yield z

        self.check_output_and_recompiles(fn)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_logging_tensor_flaky(self) -> None:
        # [FUNCTION DOC] Regression test (GPU only) for flaky failures when a triton-using
        # compiled autograd test runs before a test with a logging tensor subclass: first runs
        # a trivial backward under compiled autograd on a GPU tensor, resets the root log
        # level to warning, then compiles a traced add function stealing its input list with a
        # plain and a logging tensor. It only needs to run without raising; it also prints the
        # readable graph.
        # when you first run some test using triton and then run test_inputs_aliasing_bytecode_stack_restore
        # resulting in:
        #   - pytest: `TypeError: unsupported operand type(s) for +: 'Tensor' and 'LoggingTensor'`
        #   - python: `TypeError: not all arguments converted during string formatting`

        # 1. some triton involving test
        def fn():
            # [FUNCTION DOC] HELPER: Generator-free function creating a half precision GPU
            # tensor requiring grad, passing it through an identity helper, summing and
            # running backward.
            def _fn(x):
                # [FUNCTION DOC] HELPER: Identity function.
                return x

            x = torch.arange(
                1, 10, requires_grad=True, dtype=torch.float16, device=GPU_TYPE
            )
            out = _fn(x)
            loss = out.sum()
            loss.backward()

        with compiled_autograd._enable(compiler_fn):
            fn()

        logging.getLogger().setLevel(
            logging.WARNING
        )  # triton setup overwrote it to INFO
        # 2. test_inputs_aliasing_bytecode_stack_restore
        from torch.testing._internal.logging_tensor import LoggingTensor

        def forward(inputs):
            # [FUNCTION DOC] HELPER: Adds one to the first input, adds the second input, moves
            # the result to CPU and returns it in a tuple.
            add = inputs[0] + 1
            add_1 = add + inputs[1]
            out = add_1.cpu()
            return (out,)

        gm = torch.fx.symbolic_trace(forward)
        print(gm.print_readable())
        torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
        compiled_fn = torch.compile(gm)

        inputs = [
            torch.ones(1000000, dtype=torch.float32),
            LoggingTensor(torch.ones(1)),
        ]

        compiled_fn(inputs)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_custom_fn_output_metadata(self):
        # [FUNCTION DOC] Checks (GPU only) that a custom Function applied to a reshaped
        # half-precision tensor produces no implicit dtype-cast copy ops in the captured
        # backward graph, using a compiler that raises AssertionError if a copy op appears;
        # one capture and compile expected.
        def my_compiler_fn(gm):
            # [FUNCTION DOC] HELPER: Compiler that raises AssertionError if the captured graph
            # contains an aten copy-with-cast op, then compiles the graph with inductor
            # through a counting backend using full graph and dynamic shapes.
            for node in gm.graph.nodes:
                if isinstance(node.target, torch._ops.OpOverload):
                    if node.target._name == "aten::_to_copy":
                        raise AssertionError(
                            "there should be no implicit copies (e.g. dtype casting)"
                        )

            def inner_compiler(gm_, example_inputs_):
                # [FUNCTION DOC] HELPER: Counting inductor backend: increments the compiles
                # counter and compiles with inductor.
                counters["compiled_autograd"]["compiles"] += 1
                return inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        def fn():
            # [FUNCTION DOC] HELPER: Generator that applies an identity custom Function to a
            # view of a half precision GPU tensor, runs backward and yields the dtype, device
            # and grad of the original tensor.
            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Identity custom Function passing the gradient through.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Identity forward.
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Returns the gradient unchanged.
                    return gO

            x = torch.arange(
                1, 10, requires_grad=True, dtype=torch.float16, device=GPU_TYPE
            )
            x_view = x.view(3, 3)
            out = MyFn.apply(x_view)
            loss = out.sum()
            loss.backward()
            yield x.dtype
            yield x.device
            yield x.grad

        self.check_output_and_recompiles(fn, count=1)

    def test_custom_fn_with_same_graph(self):
        # [FUNCTION DOC] Checks two structurally identical custom Function classes are not
        # conflated by the cache: alternating them gives two compiles (one per class).
        def fn():
            # [FUNCTION DOC] HELPER: Generator that applies two identical identity Function
            # classes alternately and yields the input grad each time.
            class MyFn1(torch.autograd.Function):
                # [CLASS DOC] HELPER: First identity custom Function.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Identity forward.
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Returns the gradient unchanged.
                    return gO

            # same as MyFn1, but different autograd function id
            # should not be using same graph as MyFn1
            class MyFn2(torch.autograd.Function):
                # [CLASS DOC] HELPER: Second identity custom Function, identical to the first
                # but a different class.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Identity forward.
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Returns the gradient unchanged.
                    return gO

            for myfn in [MyFn1, MyFn2, MyFn1, MyFn2]:
                x = torch.arange(0.0, 10, requires_grad=True)
                out = myfn.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=2
        )  # should compile once for MyFn1 and once for MyFn2

    def test_custom_fn_dynamically_defined_class(self):
        # [FUNCTION DOC] Checks custom Function classes created dynamically in a loop (each
        # capturing a different multiplier) compile separately: three compiles for three
        # multipliers.
        def fn():
            # [FUNCTION DOC] HELPER: Generator creating a new Function class for each
            # multiplier, running backward and yielding the input grad.
            def create_class(multiplier: int):
                # [FUNCTION DOC] HELPER: Factory returning a new custom Function class whose
                # forward and backward scale by the captured multiplier.
                class DynamicFn(torch.autograd.Function):
                    # [CLASS DOC] HELPER: Custom Function that multiplies by a captured
                    # integer in both forward and backward.
                    @staticmethod
                    def forward(ctx, x):
                        # [FUNCTION DOC] HELPER: Returns the input multiplied by the captured
                        # multiplier.
                        return x * multiplier

                    @staticmethod
                    def backward(ctx, gO):
                        # [FUNCTION DOC] HELPER: Returns the gradient multiplied by the
                        # captured multiplier.
                        return gO * multiplier

                return DynamicFn

            for multiplier in [10, 20, 30]:
                x = torch.arange(0.0, 10, requires_grad=True)
                out = create_class(multiplier).apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(fn, count=3)

    def test_custom_fn_bw_graph_break(self):
        # [FUNCTION DOC] Tests a custom Function whose backward contains print calls that
        # cause graph breaks; without full graph, expects one capture and two compiles.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running the printing-backward Function on
            # varying sizes and yielding the input grad.
            class MySin(torch.autograd.Function):
                # [CLASS DOC] HELPER: Sine custom Function whose backward prints twice,
                # forcing graph breaks.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Saves the input and returns its sine.
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Prints, loads the saved input, prints again and
                    # returns the gradient times the cosine.
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    print("graph break")
                    return gO * torch.cos(x)

            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MySin.apply(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_custom_fn_compiled_fw_graph_break(self):
        # [FUNCTION DOC] Tests a custom Function whose forward prints (graph break) and is
        # compiled via torch.compile of the apply entry point; expects one capture and four
        # unique dynamo graphs (three forward, one backward).
        def fn():
            # [FUNCTION DOC] HELPER: Generator running the compiled apply of the
            # printing-forward Function on varying sizes and yielding the input grad.
            class MySin(torch.autograd.Function):
                # [CLASS DOC] HELPER: Sine custom Function whose forward prints, causing a
                # graph break.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Prints, saves the input and returns its sine.
                    print("graph break")
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Loads the saved input and returns the gradient
                    # times the cosine.
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            opt_model = torch.compile(MySin.apply)
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = opt_model(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=1, compiler_fn=make_compiler_fn(fullgraph=False)
        )
        self.assertEqual(counters["stats"]["unique_graphs"], 4)  # 3 fw, 1 bw

    def test_custom_fn_compiled_fw_bw_graph_break(self):
        # [FUNCTION DOC] Tests a custom Function that prints in both forward and backward,
        # compiled through torch.compile; expects one capture, two compiles and five unique
        # dynamo graphs.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running the compiled apply of the doubly
            # printing Function on varying sizes and yielding the input grad.
            class MySin(torch.autograd.Function):
                # [CLASS DOC] HELPER: Sine custom Function that prints in both forward and
                # backward.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Prints, saves the input and returns its sine.
                    print("graph break")
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Prints, loads the saved input and returns the
                    # gradient times the cosine.
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            opt_model = torch.compile(MySin.apply)
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = opt_model(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        self.check_output_and_recompiles(
            fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
        )
        self.assertEqual(counters["stats"]["unique_graphs"], 5)

    def test_mismatch_fake_tensor_mode(self, dynamic_shape=False):
        # [FUNCTION DOC] Reproduces a failure when training nanogpt with compiled autograd and
        # lazy graph modules (a fake tensor mode mismatch). The dynamic_shape flag marks both
        # operand batch dimensions dynamic. Runs a compiled add and backward through the
        # standard check helper.
        """
        Repro the failure of training nanogpt with both compiled-autograd
        and _LazyGraphModule. Check https://github.com/pytorch/pytorch/pull/118981
        for more context.
        """
        B = 8
        x = torch.rand(B, 16)
        y = torch.rand(B, 16, requires_grad=True)

        if dynamic_shape:
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(y, 0)

        def f():
            # [FUNCTION DOC] HELPER: Clears the grad of the leaf, adds the two tensors, runs
            # backward on the sum and returns the sum and the leaf grad.
            y.grad = None
            out = x + y

            # make sure the backward call does not trigger any error when
            # compiling the backward graph
            out.sum().backward()
            return out, y.grad

        self.check_output_and_recompiles(f, compile_fn=True)

    def test_mismatch_fake_tensor_mode_dynamic_shape(self):
        # [FUNCTION DOC] Runs the fake tensor mode mismatch test with dynamic shapes enabled.
        self.test_mismatch_fake_tensor_mode(dynamic_shape=True)

    def test_accumulate_grad_accuracy(self):
        # [FUNCTION DOC] Checks accumulation accuracy for a two-layer bias-free model after
        # reseeding the random generator before backward; default one capture and compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that builds a tiny model, runs backward and
            # yields both weight grads.
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 1, bias=False),
                torch.nn.Linear(1, 2, bias=False),
            )
            x = torch.randn(2, 2)

            out = model(x)
            loss = out.sum()
            torch.manual_seed(0)
            loss.backward()

            yield model[0].weight.grad
            yield model[1].weight.grad

        self.check_output_and_recompiles(fn, 1)

    def test_accumulate_grad_capture_lifts_current_grad(self):
        # [FUNCTION DOC] Checks the captured graph for a leaf that already has a grad: the
        # single accumulate-grad call node must receive the existing grad as a graph node
        # argument (four arguments), and the final grad is three. Skips if the C++ interface
        # was not rebuilt with the new schema.
        from torch._dynamo.external_utils import call_accumulate_grad

        graphs = []

        def compiler_fn(gm):
            # [FUNCTION DOC] HELPER: Compiler that records the graph module in a list and
            # returns it unchanged.
            graphs.append(gm)
            return gm

        w = torch.ones(2, 2, requires_grad=True)
        w.grad = torch.ones_like(w)
        with compiled_autograd._enable(compiler_fn):
            (w * 2).sum().backward()

        self.assertEqual(w.grad, torch.full_like(w, 3))
        self.assertEqual(len(graphs), 1)
        acc_grad_nodes = [
            node
            for node in graphs[0].graph.nodes
            if node.op == "call_function" and node.target is call_accumulate_grad
        ]
        self.assertEqual(len(acc_grad_nodes), 1)
        if (
            len(acc_grad_nodes[0].args) == 3
            and len(torch.ops.inductor.accumulate_grad_.default._schema.arguments) == 2
        ):
            self.skipTest("requires rebuilt compiled-autograd C++ interface")
        self.assertEqual(len(acc_grad_nodes[0].args), 4)
        self.assertIsInstance(acc_grad_nodes[0].args[1], torch.fx.Node)

    def test_accumulate_grad_op_functional_contract(self):
        # [FUNCTION DOC] Checks the functional contract of the inductor accumulate-grad op:
        # traced with functionalization it contains the op and does not mutate the existing
        # variable grad; run eagerly it sets the leaf grad to a fresh tensor with new storage
        # equal to the sum; a missing grad copies a dense new grad (no aliasing) and a sparse
        # new grad gets copied indices and values. Skips if the op schema has only two
        # arguments.
        if len(torch.ops.inductor.accumulate_grad_.default._schema.arguments) == 2:
            self.skipTest("requires rebuilt accumulate_grad_ schema")

        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(x, variable_grad, new_grad):
            # [FUNCTION DOC] HELPER: Calls the inductor accumulate-grad op on the given leaf,
            # current grad and new grad and returns its result.
            return torch.ops.inductor.accumulate_grad_.default(
                x, variable_grad, new_grad
            )

        x = torch.randn(2, 2, requires_grad=True)
        variable_grad = torch.ones_like(x)
        new_grad = torch.ones_like(x) * 2
        gm = make_fx(torch.func.functionalize(fn))(x, variable_grad, new_grad)

        self.assertEqual(
            gm(x, variable_grad.clone(), new_grad), variable_grad + new_grad
        )
        self.assertIn("accumulate_grad_", gm.code)
        self.assertEqual(variable_grad, torch.ones_like(x))

        x.grad = None
        with torch.no_grad():
            accumulated_grad = torch.ops.inductor.accumulate_grad_.default(
                x, variable_grad, new_grad
            )
        self.assertEqual(accumulated_grad, variable_grad + new_grad)
        self.assertEqual(x.grad, accumulated_grad)
        self.assertEqual(
            x.grad.untyped_storage().data_ptr(),
            accumulated_grad.untyped_storage().data_ptr(),
        )
        self.assertEqual(variable_grad, torch.ones_like(x))
        self.assertNotEqual(
            accumulated_grad.untyped_storage().data_ptr(),
            variable_grad.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            accumulated_grad.untyped_storage().data_ptr(),
            new_grad.untyped_storage().data_ptr(),
        )

        dense_new_grad = torch.ones_like(x) * 4
        dense_expected = dense_new_grad.clone()
        x.grad = None
        with torch.no_grad():
            initialized_grad = torch.ops.inductor.accumulate_grad_.default(
                x, None, dense_new_grad
            )
        self.assertEqual(initialized_grad, dense_expected)
        self.assertEqual(x.grad, initialized_grad)
        self.assertEqual(
            x.grad.untyped_storage().data_ptr(),
            initialized_grad.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            initialized_grad.untyped_storage().data_ptr(),
            dense_new_grad.untyped_storage().data_ptr(),
        )
        dense_new_grad.add_(1)
        self.assertEqual(initialized_grad, dense_expected)

        sparse_new_grad = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
            torch.tensor([3.0, 4.0]),
            x.size(),
        )
        x.grad = None
        with torch.no_grad():
            initialized_sparse_grad = torch.ops.inductor.accumulate_grad_.default(
                x, None, sparse_new_grad
            )
        self.assertTrue(initialized_sparse_grad.is_sparse)
        self.assertEqual(x.grad, initialized_sparse_grad)
        self.assertEqual(initialized_sparse_grad.to_dense(), sparse_new_grad.to_dense())
        self.assertNotEqual(
            initialized_sparse_grad._indices().untyped_storage().data_ptr(),
            sparse_new_grad._indices().untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            initialized_sparse_grad._values().untyped_storage().data_ptr(),
            sparse_new_grad._values().untyped_storage().data_ptr(),
        )

    def test_trace_run_with_rng_state(self):
        # [FUNCTION DOC] Checks a tensor hook that calls the rng-state higher-order op during
        # backward is preserved through compiled autograd and AOTAutograd: a custom inductor
        # post-grad pass records whether that op appears in any graph, and the test asserts it
        # did, after the standard output and recompile check with full graph.
        rng_state = torch.get_rng_state()

        def g(x):
            # [FUNCTION DOC] HELPER: Returns the sum of squares of the input.
            return (x * x).sum()

        def f():
            # [FUNCTION DOC] HELPER: Creates a leaf, registers the rng hook, runs backward
            # through the compiled function and returns the loss and leaf grad.
            x = torch.ones((2, 2), requires_grad=True, device="cpu")

            def hook(grad):
                # [FUNCTION DOC] HELPER: Hook that returns a random tensor like the gradient
                # generated with the saved rng state.
                return torch._prims.rng_prims.run_with_rng_state(
                    rng_state, torch.ops.aten.rand_like.default, grad
                )

            x.register_hook(hook)
            out = torch.compile(g, fullgraph=True)(x)
            out.backward()
            return out, x.grad

        """
        Walkthrough of what happens with `run_with_rng_state`:
        1. The tensor hook calls `run_with_rng_state` during backward.
        2. Compiled Autograd captures the backward graph with the hook call.
        3. The Dynamo graph captured by Compiled Autograd looks like:
        ```
        ===== __compiled_fn_3 =====
        torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
            def forward(self, L_inputs_ : list):
                ...
                run_with_rng_state = torch.ops.higher_order.run_with_rng_state(
                    getitem_8,
                    torch.ops.aten.rand_like.default,
                    aot0_add,
                )
                ...
        ```
        4. We want to preserve this `run_with_rng_state` op when going through AOTAutograd. We do it by having special handling
        in `run_with_rng_state` op's py_functionalize_impl.
        """

        saw_run_with_rng_state = False

        def _run_with_rng_state_op_check(inductor_post_grad_graph):
            # [FUNCTION DOC] HELPER: Inductor post-grad pass that sets an enclosing flag when
            # the graph contains the rng-state higher-order op; does not modify the graph.
            # Checks that `run_with_rng_state` op exists in one of Compiled Autograd's Inductor post-grad graphs.
            nonlocal saw_run_with_rng_state
            op_set = {node.target for node in inductor_post_grad_graph.nodes}
            saw_run_with_rng_state = (
                saw_run_with_rng_state
                or torch.ops.higher_order.run_with_rng_state in op_set
            )

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=_run_with_rng_state_op_check
        ):
            self.check_output_and_recompiles(
                f, compiler_fn=make_compiler_fn(fullgraph=True), compile_fn=False
            )

        self.assertTrue(saw_run_with_rng_state)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_trace_auto_functionalized_v2(self):
        # [FUNCTION DOC] Runs the shared auto-functionalized tracing test with the second
        # version of auto functionalization enabled.
        self.trace_auto_functionalized_base()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_trace_auto_functionalized(self):
        # [FUNCTION DOC] Runs the shared auto-functionalized tracing test with the second
        # version of auto functionalization disabled.
        self.trace_auto_functionalized_base()

    def trace_auto_functionalized_base(self):
        # [FUNCTION DOC] HELPER: Shared body of the auto-functionalized tracing tests. Defines
        # a scoped test library with an in-place mutating op (CPU and Meta implementations)
        # and a composite wrapper, then compiles a checkpointed function with a selective
        # policy forcing recomputation of the auto-functionalized higher-order op so it
        # appears in the backward graph. Runs the standard output and recompile check with a
        # compiler that asserts the auto-functionalized op (v1 or v2 by config) is present in
        # each captured graph.
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            torch.library.define(
                "testlib::foo",
                "(Tensor(a!) x) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "testlib::foo_mutated",
                "(Tensor(a!) x) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("testlib::foo", "cpu", lib=lib)
            def foo(x):
                # [FUNCTION DOC] HELPER: CPU implementation of the mutating test op; adds five
                # to the input in place and returns it.
                x.add_(5)
                return x

            @torch.library.impl("testlib::foo", "Meta", lib=lib)
            def foo_meta(x):
                # [FUNCTION DOC] HELPER: Meta implementation of the mutating test op; returns
                # the input.
                return x

            @torch.library.impl(
                "testlib::foo_mutated", "CompositeImplicitAutograd", lib=lib
            )
            def foo_mutated(x):
                # [FUNCTION DOC] HELPER: Composite implementation of the wrapper op; calls the
                # mutating op on the input and returns its result.
                return torch.ops.testlib.foo(x)

            def _get_custom_policy(must_recompute_list=None):
                # [FUNCTION DOC] HELPER: Builds a selective-checkpoint policy function that
                # forces recomputation of ops in the given list (if any) and prefers
                # recomputation otherwise.
                def _custom_policy(ctx, func, *args, **kwargs):
                    # [FUNCTION DOC] HELPER: Checkpoint policy: returns the must-recompute
                    # decision when the function is in the captured list, else the
                    # prefer-recompute decision.
                    if must_recompute_list is not None and func in must_recompute_list:
                        return torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE
                    else:
                        return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

                return _custom_policy

            def context_fn():
                # [FUNCTION DOC] HELPER: Creates selective checkpoint contexts whose policy
                # forces recomputation of the auto-functionalized higher-order op.
                must_recompute_list = [
                    torch.ops.higher_order.auto_functionalized,
                ]
                return torch.utils.checkpoint.create_selective_checkpoint_contexts(
                    _get_custom_policy(
                        must_recompute_list=must_recompute_list,
                    ),
                )

            def g(x):
                # [FUNCTION DOC] HELPER: Squares the matrix input via matmul, calls the
                # mutating wrapper op for its side effect on it and returns another matmul of
                # the result.
                x = torch.matmul(x, x)
                torch.ops.testlib.foo_mutated(x)
                return torch.matmul(x, x)

            def g_cp(x):
                # [FUNCTION DOC] HELPER: Runs the matmul function under non-reentrant
                # checkpointing with the selective context function.
                return torch.utils.checkpoint.checkpoint(
                    g, x, use_reentrant=False, context_fn=context_fn
                )

            def f():
                # [FUNCTION DOC] HELPER: Runs the checkpointed function compiled with
                # aot_eager and full graph on a random leaf, runs backward and returns the
                # output and the leaf grad.
                inps = (torch.randn(4, 4, requires_grad=True),)
                output = torch.compile(g_cp, backend="aot_eager", fullgraph=True)(*inps)
                output.sum().backward()
                return output, inps[0].grad

            """
            Walkthrough of what happens with `auto_functionalized`:
            1. `auto_functionalized` op is inserted into the graph during AOTAutograd functionalization.
            We force the op to be recomputed (by using SAC), so it appears in the backward graph.
            2. The AOT backward graph looks like:
            ```
            ===== Backward graph 0 =====
            def forward(self, primals_1: "f32[4, 4][4, 1]cpu", tangents_1: "f32[4, 4][4, 1]cpu"):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = mm)
                ...
                return (add_1,)
            ```
            3. The Compiled Autograd graph looks like:
            ```
            ===== Compiled autograd graph =====
            def forward(self, inputs, sizes, scalars, hooks):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = aot0_mm)
                ...
                return []
            ```
            4. The Dynamo graph captured by Compiled Autograd looks like:
            ```
            ===== __compiled_fn_3 =====
            def forward(self, L_inputs_ : list):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = aot0_mm)
                ...
                return (new_grad,)
            ```
            5. The Compiled Autograd's AOT "forward-only" graph looks like:
            ```
            ===== Forward graph 1 =====
            def forward(self, arg0_1: "f32[][]cpu", arg1_1: "f32[4, 4][4, 1]cpu"):
                ...
                X = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = mm)
                ...
                return (clone_1,)
            ```
            6. The `auto_functionalized` op should then be lowered using the normal lowering path in Inductor.
            """

            compiler_fn = make_compiler_fn(fullgraph=True, backend="aot_eager")

            def make_compiler_fn_with_op_check():
                # [FUNCTION DOC] HELPER: Builds a compiler function that verifies each
                # captured graph contains the auto-functionalized op before delegating to the
                # aot_eager compiler.
                def _compiler_fn(gm):
                    # [FUNCTION DOC] HELPER: Picks the v2 or v1 auto-functionalized op
                    # according to the inductor config, asserts some graph node targets it
                    # (failing with a message that prints the graph), then compiles the module
                    # with the delegate compiler.
                    auto_functionalize_func = (
                        torch.ops.higher_order.auto_functionalized
                        if not torch._inductor.config.enable_auto_functionalized_v2
                        else torch.ops.higher_order.auto_functionalized_v2
                    )

                    # Checks that `auto_functionalized` op exists in Compiled Autograd's Dynamo graph.
                    self.assertTrue(
                        any(
                            node.target is auto_functionalize_func
                            for node in gm.graph.nodes
                        ),
                        lambda msg: f"{msg}\n{auto_functionalize_func} op not found in {gm.graph}",
                    )
                    return compiler_fn(gm)

                return _compiler_fn

            compiler_fn_with_op_check = make_compiler_fn_with_op_check()
            self.check_output_and_recompiles(
                f, compiler_fn=compiler_fn_with_op_check, compile_fn=False
            )

    @scoped_load_inline
    def test_autograd_cpp_node_non_traceable(self, load_inline):
        # [FUNCTION DOC] Builds a C++ extension whose autograd function declares itself
        # non-traceable and checks it runs through compiled autograd without raising, with
        # graph breaks allowed (one capture, two compiles).
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = false;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_non_traceable_autograd_cpp_node, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = load_inline(
            name="test_non_traceable_autograd_cpp_node",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            # [FUNCTION DOC] HELPER: Generator applying the extension op to a ones tensor,
            # running backward and yielding the grad.
            x = torch.ones(10, 10, requires_grad=True)
            out = module.custom_op_backed_by_autograd_fn(x)
            loss = out.sum()
            loss.backward()
            yield x.grad

        # should not raise
        self.check_output_and_recompiles(
            fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
        )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_basic(self, load_inline, is_traceable):
        # [FUNCTION DOC] Parametrized over traceability: builds a C++ identity autograd
        # function extension (with the is_traceable flag substituted into the source) and runs
        # it for several sizes. Traceable: one capture and compile; non-traceable: one
        # capture, two compiles with graph breaks.
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_basic_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name=f"test_autograd_cpp_node_basic_{is_traceable}",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            # [FUNCTION DOC] HELPER: Generator applying the extension op to ones tensors of
            # sizes ten, hundred, ten, twenty, ten, running backward and yielding the grad.
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn, 1)
        else:
            # compiles for 10 (static) and 100 (dynamic), each with a graph break
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_id(self, load_inline, is_traceable):
        # [FUNCTION DOC] Parametrized over traceability: builds an extension with two distinct
        # C++ autograd functions and checks that graph identity depends on the function.
        # Repeating one function reuses the compiled graph; alternating two traceable
        # functions gives two compiles. KNOWN ISSUE: in the non-traceable branch of the second
        # phase the code runs the single-function generator again instead of the two-function
        # one (the comment marks it unclear), so the different-function case is only checked
        # for the traceable variant.
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

struct CustomOpAutogradFunction2 : public torch::autograd::Function<CustomOpAutogradFunction2> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

torch::Tensor custom_op_backed_by_autograd_fn2(torch::Tensor x) {
  return CustomOpAutogradFunction2::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_id_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("custom_op_backed_by_autograd_fn2", custom_op_backed_by_autograd_fn2);
}
        """
        )

        module = load_inline(
            name=f"test_autograd_cpp_node_id_{is_traceable}",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions=[
                "custom_op_backed_by_autograd_fn",
                "custom_op_backed_by_autograd_fn2",
            ],
            verbose=True,
        )

        def same_autograd_fn():
            # [FUNCTION DOC] HELPER: Generator that runs the single-function scenario four
            # times (the first compiles, the rest reuse) and yields each grad.
            def fn():
                # [FUNCTION DOC] HELPER: Generator applying the first extension op to a ones
                # tensor, running backward and yielding the grad.
                x = torch.ones(10, 10, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

            yield from fn()  # compile
            yield from fn()  # reuse
            yield from fn()  # reuse
            yield from fn()  # reuse

        if is_traceable:
            self.check_output_and_recompiles(same_autograd_fn, 1)
        else:
            self.check_output_and_recompiles(
                same_autograd_fn,
                count=[1, 2],
                compiler_fn=make_compiler_fn(fullgraph=False),
            )

        def different_autograd_fn():
            # [FUNCTION DOC] HELPER: Generator that runs the scenario with the first then
            # second extension op, then each again, yielding each grad.
            def fn(op):
                # [FUNCTION DOC] HELPER: Generator applying the given op to a ones tensor,
                # running backward and yielding the grad.
                x = torch.ones(10, 10, requires_grad=True)
                out = op(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

            op1 = module.custom_op_backed_by_autograd_fn
            op2 = module.custom_op_backed_by_autograd_fn2
            yield from fn(op1)  # compile
            yield from fn(op2)  # compile
            yield from fn(op1)  # reuse
            yield from fn(op2)  # reuse

        if is_traceable:
            self.check_output_and_recompiles(different_autograd_fn, 2)
        else:
            # ????
            self.check_output_and_recompiles(
                same_autograd_fn,
                count=[1, 2],
                compiler_fn=make_compiler_fn(fullgraph=False),
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_basic(self, load_inline, is_traceable):
        # [FUNCTION DOC] Parametrized over traceability: builds an extension whose C++
        # function saves tensors, a fixed tensor, bool, int, list and dict in its context, and
        # checks the backward sees them. Traceable: one capture and compile; non-traceable:
        # one capture, two compiles.
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y,
      const torch::Tensor& fixed) {
    ctx->save_for_backward({x, y});
    ctx->saved_data["fixed_tensor"] = fixed;
    ctx->saved_data["bool"] = true;
    ctx->saved_data["int"] = 1;
    c10::List<std::string> list({"string"});
    ctx->saved_data["list"] = std::move(list);
    c10::Dict<std::string, double> dict;
    dict.insert("string", 1.0);
    ctx->saved_data["dict"] = std::move(dict);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor y = saved_variables[1];
    torch::Tensor fixed = ctx->saved_data["fixed_tensor"].toTensor();
    assert(ctx->saved_data["bool"].isBool());
    c10::SymInt i = ctx->saved_data["int"].toSymInt();
    c10::List<c10::IValue> list = ctx->saved_data["list"].toList();
    assert(list.size() == 1);
    assert(list.get(0).toStringRef() == "string");
    c10::Dict<c10::IValue, c10::IValue> dict = ctx->saved_data["dict"].toGenericDict();
    assert(dict.size() == 1);
    assert(dict.at("string") == 1.0);

    torch::autograd::variable_list grad_inputs(3);
    grad_inputs[0] = x + y + torch::sum(fixed) + i;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& fixed) {
  return CustomOpAutogradFunction::apply(x, y, fixed);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_basic_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name=f"test_autograd_cpp_node_saved_basic_{is_traceable}",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            # [FUNCTION DOC] HELPER: Generator applying the extension op to varying sized ones
            # tensors with random second tensors and a fixed tensor, running backward and
            # yielding the grad.
            fixed = torch.ones(2, 2)
            for i in [10, 100, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                y = torch.randn(i, i)
                out = module.custom_op_backed_by_autograd_fn(x, y, fixed)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn, 1)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_dynamic(self, load_inline, is_traceable):
        # [FUNCTION DOC] Parametrized over traceability: builds an extension saving a view of
        # the input as context data (a tensor whose size varies) and checks dynamic sizes are
        # handled. Traceable: one capture and compile; non-traceable: one capture, two
        # compiles.
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    ctx->save_for_backward({x});
    ctx->saved_data["dynamic"] = x.view(-1);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    torch::Tensor z = ctx->saved_data["dynamic"].toTensor();

    torch::autograd::variable_list grad_inputs(1);
    grad_inputs[0] = x + torch::sum(z);
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_dynamic_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name=f"test_autograd_cpp_node_saved_dynamic_{is_traceable}",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            # [FUNCTION DOC] HELPER: Generator applying the extension op to ones tensors of
            # varying sizes, running backward and yielding the grad.
            for i in [10, 30, 10, 20, 10]:
                x = torch.ones(i, i, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        # compiles for 10 (static) and 100 (dynamic)
        if is_traceable:
            self.check_output_and_recompiles(fn, 1)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_int(self, load_inline, is_traceable):
        # [FUNCTION DOC] Parametrized over traceability: builds an extension saving an integer
        # argument both as an int and a symbolic int in its context, called with several
        # integers. Traceable: single default capture and compile; non-traceable: one capture,
        # two compiles.
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      int64_t y) {
    ctx->save_for_backward({x});
    ctx->saved_data["int"] = y;
    ctx->saved_data["symint"] = c10::SymInt(y);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    c10::SymInt y = ctx->saved_data["int"].toSymInt();
    c10::SymInt ys = ctx->saved_data["symint"].toSymInt();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + y + ys;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, int64_t y) {
  return CustomOpAutogradFunction::apply(x, y);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_int_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name=f"test_autograd_cpp_node_saved_int_{is_traceable}",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            # [FUNCTION DOC] HELPER: Generator applying the extension op with each integer
            # argument, running backward and yielding the grad.
            for y in [1, 2, 3, 1]:
                x = torch.ones(10, 10, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x, y)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 2], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_saved_float(self, load_inline, is_traceable):
        # [FUNCTION DOC] Parametrized over traceability: builds an extension saving a float
        # argument as both float and symbolic float, called with several floats. Traceable
        # expects one capture and four compiles and three unique graphs; non-traceable expects
        # one capture, three compiles and two unique graphs.
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      double z) {
    ctx->save_for_backward({x});
    ctx->saved_data["float"] = z;
    ctx->saved_data["symfloat"] = c10::SymFloat(z);
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 1);
    torch::Tensor x = saved_variables[0];
    c10::SymFloat z = ctx->saved_data["float"].toSymFloat();
    c10::SymFloat zs = ctx->saved_data["symfloat"].toSymFloat();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + z + zs;
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x, double z) {
  return CustomOpAutogradFunction::apply(x, z);
}

TORCH_LIBRARY(test_autograd_cpp_node_saved_float_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """
        )

        module = load_inline(
            name=f"test_autograd_cpp_node_saved_float_{is_traceable}",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        def fn():
            # [FUNCTION DOC] HELPER: Generator applying the extension op with each float
            # argument, running backward and yielding the grad.
            for z in [1.1, 2.2, 3.3, 1.1]:
                x = torch.ones(10, 10, requires_grad=True)
                out = module.custom_op_backed_by_autograd_fn(x, z)
                loss = out.sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            # compiled autograd and dynamo both support symfloat, but not backend
            self.check_output_and_recompiles(fn, [1, 4])
            # 1 restart analysis due to specialize_float=False
            self.assertEqual(counters["stats"]["unique_graphs"], 3)
        else:
            self.check_output_and_recompiles(
                fn, count=[1, 3], compiler_fn=make_compiler_fn(fullgraph=False)
            )
            self.assertEqual(counters["stats"]["unique_graphs"], 2)

    @parametrize("is_traceable", (True, False))
    @scoped_load_inline
    def test_autograd_cpp_node_data_dependent(self, load_inline, is_traceable):
        # [FUNCTION DOC] Parametrized over traceability: builds an extension whose C++
        # function behaves differently per call (a counter selects: plain, extra saved data,
        # disabled grad materialization, then a reuse), checking data-dependent node structure
        # forces recompiles. Traceable: three captures and compiles; non-traceable: three
        # captures, six compiles.
        cpp_source = Template(
            """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = $is_traceable;
  static int iteration;

  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y) {
    ctx->save_for_backward({x, y});
    ctx->saved_data["bool"] = true;
    ctx->saved_data["int"] = 1;

    switch (iteration) {
        case 0: {
            break;
        }
        case 1: {
            // recompile
            ctx->saved_data["forces_recompile"] = iteration;
            break;
        }
        case 2: {
            // recompile
            ctx->set_materialize_grads(false);
            break;
        }
        case 3: {
            // reuse
            break;
        }
        default: {
            throw std::runtime_error("unexpected iteration");
        }
    }
    iteration++;
    return {x, y};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor y = saved_variables[1];
    c10::SymInt i = ctx->saved_data["int"].toSymInt();

    torch::autograd::variable_list grad_inputs(2);
    grad_inputs[0] = x + y + i;
    return grad_inputs;
  }
};

int CustomOpAutogradFunction::iteration = 0;

torch::autograd::variable_list custom_op_backed_by_autograd_fn(const torch::Tensor& x, const torch::Tensor& y) {
  return CustomOpAutogradFunction::apply(x, y);
}

void reset() {
    CustomOpAutogradFunction::iteration = 0;
}

TORCH_LIBRARY(test_autograd_cpp_node_data_dependent_$is_traceable, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    m.def("reset", reset);
}
        """
        )

        module = load_inline(
            name=f"test_autograd_cpp_node_data_dependent_{is_traceable}",
            cpp_sources=cpp_source.substitute(
                is_traceable="true" if is_traceable else "false"
            ),
            functions=["custom_op_backed_by_autograd_fn", "reset"],
            verbose=True,
        )

        def fn():
            # [FUNCTION DOC] HELPER: Generator that resets the extension counter, then four
            # times applies the two-output extension op, sums both outputs, runs backward and
            # yields the grad.
            module.reset()
            for i in [10, 10, 10, 10]:
                x = torch.ones(i, i, requires_grad=True)
                y = torch.randn(i, i)
                (
                    out1,
                    out2,
                ) = module.custom_op_backed_by_autograd_fn(x, y)
                loss = (out1 + out2).sum()
                loss.backward()
                yield x.grad

        if is_traceable:
            self.check_output_and_recompiles(fn, 3)
        else:
            self.check_output_and_recompiles(
                fn, count=[3, 6], compiler_fn=make_compiler_fn(fullgraph=False)
            )

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_free_activation_memory(self):
        # [FUNCTION DOC] GPU only. Runs in a subprocess a script that verifies activation
        # memory is freed by the time a captured op runs: a traced function stealing its input
        # list is compiled and, inside a test op, the allocated device memory must be below
        # the size of the input; afterwards the stolen list must be empty. Fails the test if
        # the subprocess fails.
        script = """
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.testing._internal.inductor_utils import GPU_TYPE

def main():
    device_interface = get_interface_for_device(GPU_TYPE)
    assert(device_interface.memory_allocated() == 0)  # noqa: S101

    # Use an op to check that the memory is freed by the time the op is executed
    def assertion_impl(to_clone):
        mem_allocated = device_interface.memory_allocated()
        assert mem_allocated < 4000000  # noqa: S101  # some activations should be freed
        return to_clone.clone()

    with torch.library._scoped_library("test_compiled_autograd", "FRAGMENT") as lib:
        lib.define(
            "assertion_op(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,)
        )
        lib.impl("assertion_op", assertion_impl, "CPU")
        lib.impl("assertion_op", lambda x: x.clone(), "Meta")

        # Create a graph that allows inputs stealing
        def forward(activations):
            add = activations[0] + 1
            out = add.cpu()
            cloned_out = torch.ops.test_compiled_autograd.assertion_op(out)
            return (cloned_out,)

        gm = torch.fx.symbolic_trace(forward)
        torch._dynamo.utils.set_locals_to_steal(gm, ["activations"])
        compiled_fn = torch.compile(gm)

        # allocate at least 4,000,000 bytes (1,000,000 * 4 bytes)
        activations = [torch.ones(1000000, dtype=torch.float32, device=GPU_TYPE)]
        assert device_interface.memory_allocated() > 4000000  # noqa: S101

        out = compiled_fn(activations)
        assert len(activations) == 0  # noqa: S101

main()
        """
        self.run_as_subprocess(script)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_free_activation_memory_subclass(self):
        # [FUNCTION DOC] GPU only. Subprocess script like the free-activation memory test, but
        # one input is a nested (jagged) tensor subclass; checks allocated memory within a
        # range, since subclass inputs are currently not freed in inductor, and that the
        # stolen list ends empty. Fails the test if the subprocess fails.
        # cover the case when aot inputs have subclasses, resulting in a different runtime wrapper

        script = """
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.testing._internal.inductor_utils import GPU_TYPE

def main():
    device_interface = get_interface_for_device(GPU_TYPE)
    assert device_interface.memory_allocated() == 0  # noqa: S101

    # Use an op to check that the memory is freed by the time the op is executed
    def assertion_impl(to_clone):
        mem_allocated = device_interface.memory_allocated()
        assert mem_allocated < 1200000  # noqa: S101  # some activations should be freed
        assert mem_allocated > 800000  # noqa: S101  # currently subclasses don't seem to be freed in inductor
        return to_clone.clone()

    with torch.library._scoped_library("test_compiled_autograd", "FRAGMENT") as lib:
        lib.define(
            "assertion_op(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,)
        )
        lib.impl("assertion_op", assertion_impl, "CPU")
        lib.impl("assertion_op", lambda x: x.clone(), "Meta")
        lib.impl("assertion_op", lambda x: x.clone(), "NestedTensor")

        def fn(inputs):
            _, y = inputs
            out = y.cpu()
            cloned_out = torch.ops.test_compiled_autograd.assertion_op(out)
            return cloned_out

        gm = torch.fx.symbolic_trace(fn)
        torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
        compiled_fn = torch.compile(gm)

        from torch.nested._internal.nested_tensor import jagged_from_list

        activations = [
            jagged_from_list(
                [
                    torch.ones((1, 100000), device=GPU_TYPE),  # 400,000 bytes
                    torch.ones((1, 100000), device=GPU_TYPE),  # 400,000 bytes
                ],
                None,
            )[
                0
            ],  # NestedTensor
            torch.ones((1, 100000), device=GPU_TYPE),  # 400,000 bytes
        ]
        # 1,200,000 bytes (3 * 4 * 100,000 bytes)
        assert device_interface.memory_allocated() > 1200000  # noqa: S101

        out = compiled_fn(activations)
        assert len(activations) == 0  # noqa: S101

main()
        """
        self.run_as_subprocess(script)

    def test_callback_graph_break_throws_error(self):
        # [FUNCTION DOC] Checks that a graph break inside the backward of a custom Function
        # that queued an engine final callback is rejected: under a compiled autograd context
        # without full graph, backward must raise an AssertionError saying callbacks are only
        # supported with full graph.
        called = [0]

        def callback_final():
            # [FUNCTION DOC] HELPER: Final callback; increments the first entry of an
            # enclosing counter list.
            called[0] += 1

        class MyFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity custom Function whose backward queues a final
            # callback with the execution engine and then forces a graph break.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Identity forward.
                return input

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Queues the final callback, triggers a dynamo graph
                # break and returns the gradient; marked once-differentiable.
                torch.autograd.Variable._execution_engine.queue_callback(callback_final)
                torch._dynamo.graph_break()
                return grad

        a = torch.rand((3, 3), requires_grad=True)
        with self.assertRaisesRegex(
            AssertionError,
            "only supported when Compiled Autograd is enabled with fullgraph=True",
        ):
            with compiled_autograd._enable(make_compiler_fn(fullgraph=False)):
                b = MyFunc.apply(a)
                b.sum().backward()

    @requires_cuda_and_triton
    def test_cudagraphs_cpu_division(self):
        # [FUNCTION DOC] CUDA only. Checks that cudagraphs are used for a compiled autograd
        # backward when a CPU division is involved: backward of a half-precision linear loss
        # runs with cudagraphs enabled while stderr is captured. With the C++ wrapper config
        # the skip message and one skip counter are expected; otherwise neither.
        from torch._dynamo.testing import reduce_to_scalar_loss

        model = torch.nn.Linear(10, 10, dtype=torch.float16).cuda()
        inputs = torch.randn(10, 10, dtype=torch.float16).cuda()
        out = model(inputs)
        loss = reduce_to_scalar_loss(out)

        stderr_msgs = io.StringIO()
        with (
            mock.patch("sys.stderr", stderr_msgs),
            compiled_autograd._enable(compiler_fn),
        ):
            torch._inductor.config.triton.cudagraphs = True
            loss.backward()
            torch._inductor.config.triton.cudagraphs = False

        if inductor_config.cpp_wrapper:
            self.assertIn("skipping cudagraphs", stderr_msgs.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
        else:
            self.assertNotIn("skipping cudagraphs", stderr_msgs.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    @parametrize("graph_partition", [False, True])
    def test_cudagraphs_cpu_graph(self, graph_partition):
        # [FUNCTION DOC] Parametrized over inductor graph partition. A CPU-only backward under
        # compiled autograd with the cudagraphs option enabled must record exactly one
        # cudagraph skip regardless of partitioning, since there is no GPU device. Restores
        # the cudagraphs option to false afterward.
        from torch._dynamo.testing import reduce_to_scalar_loss

        model = torch.nn.Linear(10, 10, dtype=torch.float16)
        inputs = torch.randn(10, 10, dtype=torch.float16)
        out = model(inputs)
        loss = reduce_to_scalar_loss(out)

        with (
            torch._inductor.config.patch(graph_partition=graph_partition),
            compiled_autograd._enable(compiler_fn),
        ):
            torch._inductor.config.triton.cudagraphs = True
            loss.backward()
            torch._inductor.config.triton.cudagraphs = False

        # CPU-only graphs skip cudagraphs regardless of graph_partition setting
        # (no GPU devices to use cudagraphs with)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

    @skipIfXpu(msg="cudagraphs not supported on xpu for now!")
    @requires_gpu_and_triton
    def test_cudagraphs_sdpa(self):
        # [FUNCTION DOC] GPU only (not XPU). Backward of scaled dot-product attention compiled
        # under the compiled_autograd config with cudagraphs enabled; expects one capture and
        # a skip count of two with the C++ wrapper, else zero.
        query = torch.rand(
            32, 8, 128, 64, dtype=torch.float16, device=GPU_TYPE, requires_grad=True
        )
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=GPU_TYPE)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=GPU_TYPE)
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        with (
            config.patch(compiled_autograd=True),
            inductor_config.patch("triton.cudagraphs", True),
        ):
            opt_bwd = torch.compile(lambda: out.sum().backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(
            counters["inductor"]["cudagraph_skips"],
            2 if inductor_config.cpp_wrapper else 0,
        )

    @requires_cuda_and_triton
    def test_cudagraphs_cpu_scalar_used_in_python_custom_op(self):
        # [FUNCTION DOC] CUDA only. A python custom Function saves a CPU tensor and a plain
        # Python scalar; checks compiled autograd with cudagraphs gives one capture and one
        # cudagraph skip (zero when graph partition is on, since it splits off CPU work).
        class MyFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function whose forward sums the input while saving a
            # CPU scalar tensor and a Python scalar attribute; backward scales the expanded
            # gradient by both.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Saves the input and a CPU tensor holding five, stores
                # a Python scalar attribute of five and returns the sum of the input.
                cpu_tensor = torch.tensor(5)
                ctx.save_for_backward(x, cpu_tensor)  # visible to c++/autograd
                ctx.cpu_scalar = 5  # opaque to c++/autograd
                return x.sum()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Expands the incoming gradient to the saved input
                # shape and multiplies by the saved CPU tensor and the stored scalar.
                x, cpu_tensor = ctx.saved_tensors
                expand = gO * torch.ones_like(x)
                return expand * cpu_tensor * ctx.cpu_scalar

        x = torch.randn(10, requires_grad=True, device="cuda")
        out = MyFn.apply(x)
        with (
            config.patch(compiled_autograd=True),
            inductor_config.patch("triton.cudagraphs", True),
        ):
            opt_bwd = torch.compile(lambda: out.backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        # Compiled autograd lifts custom autograd.Function bwd instead of tracing it.
        # Must skip since we do not know if the cpu scalar will be used only in ATen/prim ops.
        if inductor_config.graph_partition:
            # instead of skipping cudagraph, graph partition splits off cpu inputs/outputs and ops
            # and cudagraphify the remaining computation. So there is no cudagraph skip.
            expected_cudagraph_skips = 0
        else:
            expected_cudagraph_skips = 1

        self.assertEqual(
            counters["inductor"]["cudagraph_skips"], expected_cudagraph_skips
        )

    @scoped_load_inline
    @requires_cuda_and_triton
    def test_cudagraphs_cpu_scalar_used_in_cpp_custom_op(self, load_inline):
        # [FUNCTION DOC] CUDA only. Same idea as the Python version but the custom function is
        # a traceable C++ autograd function built with load_inline that saves a CPU tensor and
        # an integer in its context. Expects one capture and cudagraph skips: zero with graph
        # partition, two with the C++ wrapper, otherwise one.
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    const auto& cpu_tensor = torch::tensor(1);
    ctx->save_for_backward({x, cpu_tensor});
    ctx->saved_data["cpu_scalar"] = 1;
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();
    assert(saved_variables.size() == 2);
    torch::Tensor x = saved_variables[0];
    torch::Tensor cpu_tensor = saved_variables[1];
    int cpu_scalar = ctx->saved_data["cpu_scalar"].toInt();
    auto expand = grad_output[0] * torch::ones_like(x);
    torch::autograd::variable_list grad_inputs(1);
    grad_inputs[0] = expand * cpu_tensor * cpu_scalar;  // autograd engine asserts that tensors are on same device
    return grad_inputs;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_cudagraphs_cpu_scalar_used_in_cpp_custom_op, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = load_inline(
            name="test_cudagraphs_cpu_scalar_used_in_cpp_custom_op",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        x = torch.randn(2, 2, requires_grad=True, device="cuda")
        with (
            config.patch(compiled_autograd=True),
            inductor_config.patch("triton.cudagraphs", True),
        ):
            out = torch.ops.test_cudagraphs_cpu_scalar_used_in_cpp_custom_op.custom_op_backed_by_autograd_fn(
                x
            )
            opt_bwd = torch.compile(lambda: out.sum().backward())
            opt_bwd()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        # Compiled autograd's initial capture lifts custom C++ autograd::Function bwd instead of tracing
        # into it. We must skip since we do not know if the cpu scalar will be used only in ATen/prim ops.
        # In the future, we can consider having a cpu scalar movement pass sometime after we trace
        # into the custom C++ autograd::Function (like in AOTDispatcher)
        if inductor_config.graph_partition:
            # instead of skipping cudagraph, graph partition splits off cpu inputs/outputs and ops
            # and cudagraphify the remaining computation. So there is no cudagraph skip.
            expected_cudagraph_skips = 0
        elif inductor_config.cpp_wrapper:
            expected_cudagraph_skips = 2
        else:
            expected_cudagraph_skips = 1

        self.assertEqual(
            counters["inductor"]["cudagraph_skips"],
            expected_cudagraph_skips,
        )

    def test_logs(self):
        # [FUNCTION DOC] Checks the compiled autograd debug log: a backward on a small random
        # leaf under the logging context gives one capture and compile, the log mentions the
        # accumulate-grad node call, and no cache-miss line mentions the graph root node.
        # Raises AssertionError if the checks fail.
        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd"
        )
        with compiled_autograd._enable(compiler_fn), ctx():
            torch.randn(4, 4, requires_grad=True).sum().backward()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        self.assertEqual(counters["compiled_autograd"]["compiles"], 1)
        if "torch::autograd::AccumulateGrad (NodeCall" not in logs.getvalue():
            raise AssertionError
        if (
            self.gen_cache_miss_log_prefix() + "torch::autograd::GraphRoot"
            in logs.getvalue()
        ):
            raise AssertionError

    def test_logs_aot_bwd_reuse(self):
        # [FUNCTION DOC] Regression test that reusing one AOT backward graph three times in a
        # single backward under compiled autograd does not raise a redefined node name error.
        @torch.compile(backend="aot_eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (aot_eager) function returning the sum of its
            # input.
            return x.sum()

        with compiled_autograd._enable(compiler_fn):
            x = torch.randn(4, 4, requires_grad=True)
            y = torch.randn(4, 4, requires_grad=True)
            z = torch.randn(4, 4, requires_grad=True)
            # reuse the same AOT bwd graph 3 times
            out = fn(x) + fn(y) + fn(z)
            out.backward()
        # should not RuntimeError: Node redefined name aot0_expand!

    def test_verbose_logs_graph(self):
        # [FUNCTION DOC] Checks the verbose compiled autograd log lists the expected sequence
        # of node calls (graph root, activation, addmm, transposes, accumulate-grad and so on)
        # in order for a small two-layer model; a cursor scans log lines and all expected
        # entries must be found.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running backward on a small two-layer model and
            # yielding its four parameter grads.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])
            result = model(x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            self.check_output_and_recompiles(fn)

        expected_logs = [
            "torch::autograd::GraphRoot (NodeCall 0)",
            "ReluBackward0 (NodeCall 2)",
            "AddmmBackward0 (NodeCall 3)",
            "ReluBackward0 (NodeCall 5)",
            "TBackward0 (NodeCall 6)",
            "torch::autograd::AccumulateGrad (NodeCall 7)",
            "torch::autograd::AccumulateGrad (NodeCall 9)",
            "TBackward0 (NodeCall 10)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
            "SumBackward0 (NodeCall 1)",
            "ReluBackward0 (NodeCall 2)",
            "AddmmBackward0 (NodeCall 3)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
            "TBackward0 (NodeCall 4)",
            "torch::autograd::AccumulateGrad (NodeCall 5)",
            "ReluBackward0 (NodeCall 6)",
            "AddmmBackward0 (NodeCall 7)",
            "torch::autograd::AccumulateGrad (NodeCall 10)",
            "TBackward0 (NodeCall 8)",
            "torch::autograd::AccumulateGrad (NodeCall 9)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
        ]

        found = 0
        for line in logs.getvalue().split("\n"):
            if found == len(expected_logs):
                break
            if expected_logs[found] in line:
                found += 1

        self.assertEqual(found, len(expected_logs))

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_verbose_logs_aot_id(self, _):
        # [FUNCTION DOC] Checks the verbose log names the AOT compiled backward node with its
        # numbered call when the model forward is compiled (with the AOT counter patched to a
        # fresh counter). The extra mock argument is unused.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that compiles the model forward, runs backward
            # and yields the four parameter grads.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])

            @torch.compile
            def forward(model, x):
                # [FUNCTION DOC] HELPER: Compiled forward applying the model to the input.
                return model(x)

            result = forward(model, x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            self.check_output_and_recompiles(fn)

        expected_logs = [
            "code: CompiledFunctionBackward (NodeCall 2)",
        ]

        found = 0
        for line in logs.getvalue().split("\n"):
            if found == len(expected_logs):
                break
            if expected_logs[found] in line:
                found += 1

        self.assertEqual(found, len(expected_logs))

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_verbose_logs_aot_dispatcher_nodes(self, _):
        # [FUNCTION DOC] Checks the verbose log contains the expected AOT dispatcher node
        # names for a function with a graph break (two compiled backward regions), all
        # expected names being present. The extra mock argument is unused.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that runs backward through a compiled function
            # with a graph break and yields the input grad.
            @torch.compile
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function computing sine and cosine, graph
                # breaking, then returning sin of the first plus cos of the second.
                tmp1 = x.sin()
                tmp2 = x.cos()
                torch._dynamo.graph_break()
                return tmp1.sin() + tmp2.cos()

            x = torch.randn(4, requires_grad=True)
            out = f(x)
            out.sum().backward()
            yield x.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            self.check_output_and_recompiles(fn)

        expected_logs = [
            "CompiledFunctionBackward1",
            "aot1_sin_1",
            "aot1_neg",
            "aot0_tangents_2",
            "aot1_cos_1",
            "aot0_tangents_1",
            "CompiledFunctionBackward0",
            "aot0_sin_1",
            "aot0_neg",
            "aot0_mul",
            "aot0_cos_1",
            "aot0_mul_1",
            "aot0_add",
        ]

        self.assertEqual(
            sum(1 for e in expected_logs if e in logs.getvalue()), len(expected_logs)
        )

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_verbose_logs_aot_dispatcher_nodes_hop(self, _):
        # [FUNCTION DOC] Checks the verbose log for a tensor hook closing over a tensor and a
        # dataclass object (a higher-order wrapped hook): compiled autograd grads equal eager
        # grads, and all expected AOT node names appear in the log. The extra mock argument is
        # unused.
        @dataclasses.dataclass
        class CustomObj:
            # [CLASS DOC] HELPER: Simple data holder with a single tensor field.
            val: torch.Tensor

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Takes the sine, registers a hook adding the object tensor
            # and a closure tensor to the gradient, and returns the sine of that.
            y = x.sin()
            closure_var = y + 1
            y.register_hook(lambda grad: grad + obj.val + closure_var)
            z = y.sin()
            return z

        opt_fn = torch.compile(fn)

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=True)
        obj = CustomObj(torch.tensor(88))
        fn(x, obj).sum().backward()

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx(), compiled_autograd._enable(compiler_fn):
            opt_fn(y, obj).sum().backward()
        self.assertEqual(x.grad, y.grad)

        expected_logs = [
            "CompiledFunctionBackward0",
            "aot0_primals_2",
            "aot0_tangents_2",
            "aot0_tangents_1",
            "aot0_sin",
            "aot0_cos",
            "aot0_mul",
            "aot0_add_1",
            "aot0_trace_wrapped",
            "aot0_cos_1",
            "aot0_mul_1",
        ]

        self.assertEqual(
            sum(1 for e in expected_logs if e in logs.getvalue()), len(expected_logs)
        )

    def test_verbose_logs_cpp(self):
        # [FUNCTION DOC] Checks the verbose C++ log: after enabling verbose logging, the
        # cache-miss line for the graph root node with its key size and empty previous key
        # sizes appears exactly once over three varying-batch runs.
        torch._logging.set_logs(compiled_autograd_verbose=True)

        def fn():
            # [FUNCTION DOC] HELPER: Generator that zeroes grads, runs backward at batch sizes
            # ten to twelve and yields the four parameter grads.
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            for i in [10, 11, 12]:
                model.zero_grad()
                x = torch.randn([i, 4])
                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            self.check_output_and_recompiles(fn)

        patterns1 = [
            r".*"
            + self.gen_cache_miss_log_prefix()
            + r"torch::autograd::GraphRoot \(NodeCall 0\) with key size (\d+), previous key sizes=\[\]\n",
        ]

        all_logs = logs.getvalue()

        pattern1 = r"".join(patterns1)
        matches1 = re.findall(pattern1, all_logs)
        self.assertEqual(len(matches1), 1)
        if not isinstance(
            matches1[0], str
        ):  # for a single match: matches1=['match'], for multiple matches: matches1=[('match1', 'match2')]...
            raise AssertionError
        self.assertEqual(len(matches1), len(patterns1))

    @skipIfWindows(msg="node name demangling inconsistent on windows")
    def test_verbose_logs_dynamic_shapes(self):
        # [FUNCTION DOC] Skipped on Windows. Checks verbose logs under dynamic shapes: with
        # batch sizes varying across three calls, only one capture happens and the first call
        # logs the graph root cache miss with a key size of 39 and no previous key sizes.
        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )

        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
        )

        for i, j in zip([10, 11, 12], [10, 10, 11]):
            model.zero_grad()
            x = torch.randn([i, 4])
            y = torch.randn([j, 4])
            result = model(x).sum() + model(y).sum()
            with ctx(), compiled_autograd._enable(torch.compile(backend="eager")):
                result.backward()

        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

        actual_logs = logs.getvalue()
        expected_logs = [
            self.gen_cache_miss_log_prefix()
            + "torch::autograd::GraphRoot (NodeCall 0) with key size 39, previous key sizes=[]",
        ]
        for expected in expected_logs:
            self.assertTrue(expected in actual_logs)

    def test_verbose_logs_snapshot(self):
        # [FUNCTION DOC] Checks that the verbose level is snapshotted when the logging context
        # is entered: enabling verbose logs inside an already open context has no effect, so
        # no graph root cache-miss line is logged.
        def fn():
            # [FUNCTION DOC] HELPER: Runs backward on a small two-layer model (no yields
            # despite being a plain function).
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])
            result = model(x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        with ctx():
            with compiled_autograd._enable(compiler_fn):
                # unused, verbose level already snapshot with contextmanager
                torch._logging.set_logs(compiled_autograd_verbose=True)
                fn()

        unexpected_logs = [
            self.gen_cache_miss_log_prefix() + "torch::autograd::GraphRoot (NodeCall 0)"
        ]

        self.assertEqual(sum(1 for e in unexpected_logs if e in logs.getvalue()), 0)

    def test_tensor_subclass_basic(self):
        # [FUNCTION DOC] Checks compiled autograd with tensor subclass inputs: registers two
        # custom ops converting between a pair-subclass tensor and its parts with dispatch
        # rules, builds a subclass tensor from two params, runs a compiled function and
        # backward. Asserts the param grads are twice the params, one graph captured and its
        # printed code equals the expected text containing subclass construction.
        from torch.testing._internal.two_tensor import TwoTensor, TwoTensorMode

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("to_twotensor(Tensor a, Tensor b) -> Tensor")
            lib.define("from_twotensor(Tensor c) -> (Tensor, Tensor)")

            def to_twotensor_backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward of the to-pair op: splits the gradient into
                # its parts with the other op.
                return torch.ops.mylib.from_twotensor(grad)

            def from_twotensor_backward(ctx, grad_a, grad_b):
                # [FUNCTION DOC] HELPER: Backward of the from-pair op; should never be called
                # and raises AssertionError.
                raise AssertionError("shouldn't get hit")

            torch.library.register_autograd(
                "mylib::to_twotensor", to_twotensor_backward, lib=lib
            )
            torch.library.register_autograd(
                "mylib::from_twotensor", from_twotensor_backward, lib=lib
            )

            @torch.library.register_torch_dispatch(
                "mylib::to_twotensor", TwoTensorMode, lib=lib
            )
            def _(_0, _1, _2, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch rule for the to-pair op: asserts there are
                # no keyword arguments and returns a pair subclass holding clones of both
                # inputs.
                if kwargs:
                    raise AssertionError
                a, b = args
                return TwoTensor(a.clone(), b.clone())

            @torch.library.register_torch_dispatch(
                "mylib::from_twotensor", TwoTensor, lib=lib
            )
            def _(_0, _1, _2, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch rule for the from-pair op: asserts there are
                # no keyword arguments and returns clones of the two inner tensors.
                if kwargs:
                    raise AssertionError
                (c,) = args
                return c.a.clone(), c.b.clone()

            @torch.compile(backend="aot_eager", fullgraph=True)
            def fn(x):
                # [FUNCTION DOC] HELPER: Compiled (aot_eager, full graph) function returning
                # the square plus two.
                return x * x + 2

            param1 = torch.randn(4, 4, requires_grad=True)
            param2 = torch.randn(4, 4, requires_grad=True)
            with TwoTensorMode():
                x = torch.ops.mylib.to_twotensor(param1, param2)

            inner_compiler_fn = make_compiler_fn(fullgraph=True, backend="aot_eager")
            graphs = []

            def compiler_fn(gm):
                # [FUNCTION DOC] HELPER: Compiler that records each captured graph module in a
                # list and compiles it with the aot_eager compiler.
                graphs.append(gm)
                return inner_compiler_fn(gm)

            with (
                compiled_autograd._enable(compiler_fn),
                mock.patch(
                    "torch._functorch.aot_autograd.AOT_COUNTER",
                    new_callable=itertools.count,
                ),
            ):
                res = fn(x)
                res.sum().backward()

            self.assertEqual(param1.grad, 2 * param1)
            self.assertEqual(param2.grad, 2 * param2)
            self.assertEqual(len(graphs), 1)

            graph_code = normalize_gm(graphs[0].print_readable(print_output=False))
            # The graph should have make_subclass calls in it.
            self.assertExpectedInline(
                graph_code,
                """\
class CompiledAutograd0(torch.nn.Module):
    def forward(self, inputs, sizes, scalars, hooks, packed_data):
        getitem = inputs[0]
        getitem_1 = inputs[1]
        getitem_2 = inputs[2]
        getitem_3 = inputs[3]
        getitem_4 = inputs[4];  inputs = None
        getitem_5 = sizes[0]
        getitem_6 = sizes[1]
        getitem_7 = sizes[2]
        getitem_8 = sizes[3]
        getitem_21 = sizes[4]
        getitem_22 = sizes[5]
        getitem_23 = sizes[6]
        getitem_24 = sizes[7];  sizes = None
        unwrap_maybe_dynamic_int = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_5);  getitem_5 = None
        unwrap_maybe_dynamic_int_1 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_6);  getitem_6 = None
        unwrap_maybe_dynamic_int_2 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_7);  getitem_7 = None
        unwrap_maybe_dynamic_int_3 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_8);  getitem_8 = None
        unwrap_maybe_dynamic_int_16 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_21);  getitem_21 = None
        unwrap_maybe_dynamic_int_17 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_22);  getitem_22 = None
        unwrap_maybe_dynamic_int_18 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_23);  getitem_23 = None
        unwrap_maybe_dynamic_int_19 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_24);  getitem_24 = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [], True, 6)]);  getitem = None
        getitem_25 = validate_outputs[0];  validate_outputs = None

        sum_backward0 = torch__dynamo_compiled_autograd_ops_SumBackward0([getitem_25], [True], [unwrap_maybe_dynamic_int, unwrap_maybe_dynamic_int_1]);  getitem_25 = unwrap_maybe_dynamic_int = unwrap_maybe_dynamic_int_1 = None
        getitem_26 = sum_backward0[0];  sum_backward0 = None
        validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_26], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_2, unwrap_maybe_dynamic_int_3], True, 6)]);  getitem_26 = unwrap_maybe_dynamic_int_2 = unwrap_maybe_dynamic_int_3 = None
        getitem_27 = validate_outputs_1[0];  validate_outputs_1 = None

        getitem_28 = hooks[0];  getitem_28 = None
        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((getitem_1, getitem_2), [], [], (getitem_27,));  getitem_1 = getitem_2 = getitem_27 = None
        aot0_primals_1 = call_aot_bwd_prologue[0]
        aot0_primals_2 = call_aot_bwd_prologue[1]
        aot0_tangents_1 = call_aot_bwd_prologue[2]
        aot0_tangents_2 = call_aot_bwd_prologue[3];  call_aot_bwd_prologue = None

        aot0_mul_2 = torch.ops.aten.mul.Tensor(aot0_tangents_1, aot0_primals_1);  aot0_tangents_1 = aot0_primals_1 = None
        aot0_mul_3 = torch.ops.aten.mul.Tensor(aot0_tangents_2, aot0_primals_2);  aot0_tangents_2 = aot0_primals_2 = None
        aot0_add_2 = torch.ops.aten.add.Tensor(aot0_mul_2, aot0_mul_2);  aot0_mul_2 = None
        aot0_add_3 = torch.ops.aten.add.Tensor(aot0_mul_3, aot0_mul_3);  aot0_mul_3 = None

        make_subclass = torch__dynamo_compiled_autograd_make_subclass(aot0_add_2, aot0_add_3);  aot0_add_2 = aot0_add_3 = None

        getitem_33 = hooks[1];  hooks = None
        call_backward = torch__dynamo_external_utils_call_backward(getitem_33, (), make_subclass);  getitem_33 = make_subclass = None
        getitem_36 = call_backward[0]
        getitem_37 = call_backward[1];  call_backward = None
        validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_36, getitem_37], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_16, unwrap_maybe_dynamic_int_17], False, 6), ((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_18, unwrap_maybe_dynamic_int_19], False, 6)]);  getitem_36 = getitem_37 = unwrap_maybe_dynamic_int_16 = unwrap_maybe_dynamic_int_17 = unwrap_maybe_dynamic_int_18 = unwrap_maybe_dynamic_int_19 = None
        getitem_39 = validate_outputs_2[0]

        call_accumulate_grad_1 = torch__dynamo_external_utils_call_accumulate_grad(getitem_4, None, getitem_39, False);  getitem_4 = getitem_39 = call_accumulate_grad_1 = None

        getitem_40 = validate_outputs_2[1];  validate_outputs_2 = None

        call_accumulate_grad = torch__dynamo_external_utils_call_accumulate_grad(getitem_3, None, getitem_40, False);  getitem_3 = getitem_40 = call_accumulate_grad = None

        _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
        return []
""",
            )

    # https://github.com/pytorch/pytorch/issues/138920
    # Inductor has a joint graph pattern to remove pointless view pairs.
    # That will remove the no-op view pairs this test is checking. Disable
    # pattern matcher for this test.
    @inductor_config.patch(pattern_matcher=False)
    def test_compiled_autograd_does_not_specialize_on_bw_symints(self):
        # [FUNCTION DOC] Regression test for issue 138920 related behavior: with the inductor
        # pattern matcher disabled, checks that backward graphs do not specialize on backward
        # symbolic sizes. Runs two compiled forwards with different dict-of-tensor sizes (with
        # graph breaks) and backward under a recording compiler, then asserts the first two
        # reshape nodes of the second graph take symbolic nodes rather than constants.
        # Documented as fragile.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a view, a linear layer, a view-contiguous
            # reshuffle and a flatten.
            def __init__(self, a, b, c):
                # [FUNCTION DOC] HELPER: Stores the sizes and creates a linear layer from
                # input features to output features on CPU.
                super().__init__()
                self.a = a
                self.c = c
                self.b = b
                self.lin1 = torch.nn.Linear(b * a, b * c, device="cpu")

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Views the input to the feature width, applies the
                # linear layer, reshapes and flattens the result.
                x = x.view(-1, self.a * self.b)
                y = self.lin1(x)
                y = y.view(-1, self.c, self.b).contiguous()
                y = torch.flatten(y, start_dim=1)
                return y

        class Mod2(torch.nn.Module):
            # [CLASS DOC] HELPER: Wraps the previous module and feeds it a concatenation of
            # tensors from a dictionary entry.
            def __init__(self, a, b, c):
                # [FUNCTION DOC] HELPER: Builds the wrapped module with the given sizes.
                super().__init__()
                self.mod = Mod(a, b, c)

            def forward(self, s, tensor_dict):
                # [FUNCTION DOC] HELPER: Looks up the list of tensors for the key,
                # concatenates them and applies the inner module.
                args = tensor_dict[s]
                x = torch.cat(list(args))
                out = self.mod(x)
                return out

        class Mod3(torch.nn.Module):
            # [CLASS DOC] HELPER: Container applying several modules per key and summing a
            # concatenation of outputs.
            def __init__(self, mods):
                # [FUNCTION DOC] HELPER: Stores the list of modules.
                super().__init__()
                self.mods = mods

            def forward(self, strs, tensor_dict, x):
                # [FUNCTION DOC] HELPER: Prints (graph break), applies each module with its
                # key and the tensor dictionary, appends outputs after the input and returns
                # the column sum of the concatenation.
                outs = [x]
                for i, m in enumerate(self.mods):
                    s = strs[i]
                    print("graph break")
                    out = m(s, tensor_dict)
                    outs.append(out)
                return torch.cat(outs).sum(0)

        def gen_tensor_dict(sizes):
            # [FUNCTION DOC] HELPER: Builds a dictionary with four random tensors for the
            # first key and seven for the second, with the given leading sizes.
            tensor_dict = {
                "a": [torch.randn(sizes[0], 48, device="cpu") for _ in range(4)],
                "b": [torch.randn(sizes[1], 48, device="cpu") for _ in range(7)],
            }
            return tensor_dict

        mods = [
            Mod2(192, 1, 48),
            Mod2(336, 1, 48),
        ]
        m = Mod3(mods)

        strs = ["a", "b"]

        m = torch.compile(m)

        graphs = []

        def compiler_fn(gm):
            # [FUNCTION DOC] HELPER: Compiler that compiles the graph with a backend recording
            # the graph module, using full graph and dynamic shapes.
            def inner_compiler(gm_, example_inputs_):
                # [FUNCTION DOC] HELPER: Backend that appends the graph module to the list and
                # returns it unchanged.
                graphs.append(gm_)
                return gm_

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        x = torch.zeros(100, 48, device="cpu")
        tensor_dict = gen_tensor_dict([101, 102])
        out = m(strs, tensor_dict, x)

        with torch._dynamo.compiled_autograd._enable(compiler_fn) as ctx:
            out.sum().backward()

        x = torch.zeros(103, 48, device="cpu")
        tensor_dict = gen_tensor_dict([104, 105])
        out = m(strs, tensor_dict, x)

        with torch._dynamo.compiled_autograd._enable(compiler_fn) as ctx:
            out.sum().backward()

        # This test is a bit fragile (I failed to create a better repro).
        # The important bit is that the second CA graph has not specialized the value
        # of aot4_sym_size_int_ to a constant.
        # This happens via suppressing any dynamic shape guards that CA generates
        # when it runs make_fx.
        # Suppressing these guards is strictly better than the current state,
        # because we ignore all of these guards anyway in CA.
        # Once we stop using make_fx in CA, we won't have to worry about this specialization.
        view_nodes = graphs[1].graph.find_nodes(
            op="call_function", target=torch.ops.aten.reshape.default
        )
        # First 2 view nodes have a first argument that is a SymInt, not an int burned into the graph
        self.assertTrue(isinstance(view_nodes[0].args[1][0], torch.fx.Node))
        self.assertTrue(isinstance(view_nodes[1].args[1][0], torch.fx.Node))

    @requires_gpu_and_triton
    def test_flex_attention(self):
        # [FUNCTION DOC] GPU only. Checks compiled autograd with flex attention using a custom
        # score modification across varying sequence sizes with an aot_eager compiler; expects
        # two captures and compiles.
        def _squared(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: Score modification returning the score squared; needs the
            # joint graph for correctness.
            """Joint graph needed for correctness"""
            return score * score

        def fn():
            # [FUNCTION DOC] HELPER: Generator running flex attention forward and backward for
            # several shapes on bfloat16 GPU tensors and yielding the grad.
            @torch.compile(backend="aot_eager")
            def fwd_bwd(x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Compiled function running flex attention with the
                # squaring score modification, summing and calling backward.
                flex_attention(x, x, x, score_mod=_squared).sum().backward()

            for a, b in zip([12, 24, 12], [64, 128, 64]):
                v = torch.zeros(
                    1,
                    1,
                    a * b,
                    b,
                    dtype=torch.bfloat16,
                    device=GPU_TYPE,
                    requires_grad=True,
                )
                fwd_bwd(v)
                yield v.grad

        self.check_output_and_recompiles(
            fn, count=2, compiler_fn=make_compiler_fn(backend="aot_eager")
        )

    def test_saved_tensor_unpack_hook_ordering(self):
        # [FUNCTION DOC] Checks saved tensor hook ordering: with pack and unpack hooks
        # installed and compiled autograd enabled without full graph, packing happens once in
        # forward and unpacking has not happened when a hook registered on the loss fires;
        # after backward it has happened once.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Elementwise product of two tensors.
            return x * y

        pack_count = 0
        unpack_count = 0

        def pack_hook(x):
            # [FUNCTION DOC] HELPER: Pack hook; increments the enclosing pack counter and
            # returns the tensor.
            nonlocal pack_count
            pack_count += 1
            return x

        def unpack_hook(x):
            # [FUNCTION DOC] HELPER: Unpack hook; increments the enclosing unpack counter and
            # returns the tensor.
            nonlocal unpack_count
            unpack_count += 1
            return x

        def tensor_hook(_):
            # [FUNCTION DOC] HELPER: Tensor hook asserting that no saved tensor has been
            # unpacked yet.
            self.assertEqual(unpack_count, 0)

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with (
            torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook),
            compiled_autograd._enable(make_compiler_fn(fullgraph=False)),
        ):
            out_test = f(x, y)
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 0)
            loss = out_test.sum()
            loss.register_hook(
                tensor_hook
            )  # scheduled to fire before any saved activations
            loss.backward()
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 1)

    @parametrize("reentrant", (True, False))
    def test_checkpointing_simple(self, reentrant):
        # [FUNCTION DOC] Parametrized over reentrant versus non-reentrant checkpointing; runs
        # backward through a checkpointed sine-cosine function under compiled autograd and
        # compares with eager.
        def fn():
            # [FUNCTION DOC] HELPER: Generator applying checkpointing to a small function,
            # running backward and yielding results.
            def _fn(x):
                # [FUNCTION DOC] HELPER: Computes sine then cosine and returns the sum of
                # their product.
                y = x.sin()
                z = y.cos()
                return (y * z).sum()

            inp = torch.rand(10, 10, requires_grad=True)
            out = torch.utils.checkpoint.checkpoint(_fn, inp, use_reentrant=reentrant)
            out.backward()
            yield inp.grad

        if reentrant:
            self.check_output_and_recompiles(
                fn, count=[1, 3], compiler_fn=make_compiler_fn(fullgraph=False)
            )
        else:
            # dynamo issues, just run the CA graph directly for now
            def check(gm):
                # [FUNCTION DOC] HELPER: Graph hook for the non-reentrant case; prints the
                # captured graph readably, normalizes it and asserts it equals the expected
                # text showing unpack hook calls for the saved tensors of the checkpointed
                # ops.
                graph_code = normalize_gm(gm.print_readable(print_output=False))
                self.assertExpectedInline(
                    graph_code,
                    """\
class CompiledAutograd0(torch.nn.Module):
    def forward(self, inputs, sizes, scalars, hooks, packed_data):
        getitem = inputs[0]
        getitem_1 = inputs[1];  inputs = None
        getitem_2 = sizes[0]
        getitem_3 = sizes[1]
        getitem_4 = sizes[2]
        getitem_5 = sizes[3]
        getitem_6 = sizes[4]
        getitem_7 = sizes[5]
        getitem_8 = sizes[6]
        getitem_9 = sizes[7]
        getitem_10 = sizes[8]
        getitem_11 = sizes[9]
        getitem_12 = sizes[10]
        getitem_13 = sizes[11];  sizes = None
        unwrap_maybe_dynamic_int = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_2);  getitem_2 = None
        unwrap_maybe_dynamic_int_1 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_3);  getitem_3 = None
        unwrap_maybe_dynamic_int_2 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_4);  getitem_4 = None
        unwrap_maybe_dynamic_int_3 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_5);  getitem_5 = None
        unwrap_maybe_dynamic_int_4 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_6);  getitem_6 = None
        unwrap_maybe_dynamic_int_5 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_7);  getitem_7 = None
        unwrap_maybe_dynamic_int_6 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_8);  getitem_8 = None
        unwrap_maybe_dynamic_int_7 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_9);  getitem_9 = None
        unwrap_maybe_dynamic_int_8 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_10);  getitem_10 = None
        unwrap_maybe_dynamic_int_9 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_11);  getitem_11 = None
        unwrap_maybe_dynamic_int_10 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_12);  getitem_12 = None
        unwrap_maybe_dynamic_int_11 = torch__dynamo_external_utils_unwrap_maybe_dynamic_int(getitem_13);  getitem_13 = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [], False, 6)]);  getitem = None
        getitem_14 = validate_outputs[0];  validate_outputs = None

        sum_backward0 = torch__dynamo_compiled_autograd_ops_SumBackward0([getitem_14], [True], [unwrap_maybe_dynamic_int, unwrap_maybe_dynamic_int_1]);  getitem_14 = unwrap_maybe_dynamic_int = unwrap_maybe_dynamic_int_1 = None
        getitem_15 = sum_backward0[0];  sum_backward0 = None
        validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_15], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_2, unwrap_maybe_dynamic_int_3], False, 6)]);  getitem_15 = unwrap_maybe_dynamic_int_2 = unwrap_maybe_dynamic_int_3 = None
        getitem_16 = validate_outputs_1[0];  validate_outputs_1 = None

        getitem_17 = hooks[0]
        getitem_18 = packed_data[0]
        getitem_19 = hooks[1]
        getitem_20 = packed_data[1]
        call_hook = torch__dynamo_external_utils_call_hook(getitem_17, getitem_18, hook_type = 'unpack_hook');  getitem_17 = getitem_18 = None
        call_hook_1 = torch__dynamo_external_utils_call_hook(getitem_19, getitem_20, hook_type = 'unpack_hook');  getitem_19 = getitem_20 = None
        mul_backward0 = torch__dynamo_compiled_autograd_ops_MulBackward0([getitem_16], [True, True], call_hook, 6, call_hook_1, 6);  getitem_16 = call_hook = call_hook_1 = None
        getitem_21 = mul_backward0[0]
        getitem_22 = mul_backward0[1];  mul_backward0 = None
        validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_21, getitem_22], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_4, unwrap_maybe_dynamic_int_5], False, 6), ((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_6, unwrap_maybe_dynamic_int_7], False, 6)]);  getitem_21 = getitem_22 = unwrap_maybe_dynamic_int_4 = unwrap_maybe_dynamic_int_5 = unwrap_maybe_dynamic_int_6 = unwrap_maybe_dynamic_int_7 = None
        getitem_23 = validate_outputs_2[0]
        getitem_24 = validate_outputs_2[1];  validate_outputs_2 = None

        getitem_25 = hooks[2]
        getitem_26 = packed_data[2]
        call_hook_2 = torch__dynamo_external_utils_call_hook(getitem_25, getitem_26, hook_type = 'unpack_hook');  getitem_25 = getitem_26 = None
        cos_backward0 = torch__dynamo_compiled_autograd_ops_CosBackward0([getitem_24], [True], call_hook_2);  getitem_24 = call_hook_2 = None
        getitem_27 = cos_backward0[0];  cos_backward0 = None
        validate_outputs_3 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_27], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_8, unwrap_maybe_dynamic_int_9], False, 6)]);  getitem_27 = unwrap_maybe_dynamic_int_8 = unwrap_maybe_dynamic_int_9 = None
        getitem_28 = validate_outputs_3[0];  validate_outputs_3 = None
        add = torch.add(getitem_23, getitem_28);  getitem_23 = getitem_28 = None

        getitem_29 = hooks[3];  hooks = None
        getitem_30 = packed_data[3];  packed_data = None
        call_hook_3 = torch__dynamo_external_utils_call_hook(getitem_29, getitem_30, hook_type = 'unpack_hook');  getitem_29 = getitem_30 = None
        sin_backward0 = torch__dynamo_compiled_autograd_ops_SinBackward0([add], [True], call_hook_3);  add = call_hook_3 = None
        getitem_31 = sin_backward0[0];  sin_backward0 = None
        validate_outputs_4 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_31], [((None, None, device(type='cpu'), 6, 0, None), [unwrap_maybe_dynamic_int_10, unwrap_maybe_dynamic_int_11], False, 6)]);  getitem_31 = unwrap_maybe_dynamic_int_10 = unwrap_maybe_dynamic_int_11 = None
        getitem_32 = validate_outputs_4[0];  validate_outputs_4 = None

        call_accumulate_grad = torch__dynamo_external_utils_call_accumulate_grad(getitem_1, None, getitem_32, False);  getitem_1 = getitem_32 = call_accumulate_grad = None
        _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
        return []
""",
                )

            self.check_output_and_recompiles(
                fn,
                count=[1, 0],
                compiler_fn=make_compiler_fn(backend="ca_eager", gm_hook=check),
            )

    @requires_cuda_and_triton
    def test_cpu_offloading(self):
        # [FUNCTION DOC] CUDA only. Checks CPU offloading of saved tensors through pack and
        # unpack hooks that move tensors to CPU and back to GPU, over varying sizes. The hook
        # on the second captured graph asserts the expected graph text containing the unpack
        # hook, a call to the custom backward and a copy backward node.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that under saved-tensor offload hooks runs a
            # custom matmul-like Function on GPU tensors of varying sizes, runs backward and
            # yields the grad.
            def pack(x):
                # [FUNCTION DOC] HELPER: Pack hook moving the tensor to CPU.
                return x.cpu()

            def unpack(x):
                # [FUNCTION DOC] HELPER: Unpack hook moving the tensor back to the GPU.
                return x.cuda()

            class MyMatMul(torch.autograd.Function):
                # [CLASS DOC] HELPER: Custom Function that saves its input, forward returns a
                # matmul of the input with itself and backward multiplies the gradient by the
                # saved input.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Saves the input and returns the matmul of it with
                    # itself.
                    ctx.save_for_backward(x)
                    return torch.matmul(x, x)

                @staticmethod
                def backward(ctx, grad_out):
                    # [FUNCTION DOC] HELPER: Returns the gradient times the saved input.
                    (x,) = ctx.saved_tensors
                    return grad_out * x

            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                for i in [10, 100, 10, 20, 30]:
                    x = torch.randn(i, requires_grad=True).cuda()
                    MyMatMul.apply(x).sum().backward()
                    yield x.grad

        i = 0

        def check(gm):
            # [FUNCTION DOC] HELPER: Graph hook skipping the first captured graph and
            # asserting that the second equals the expected readable text (GPU variant with
            # copy-backward and a call to the custom backward).
            nonlocal i
            if i == 0:
                i += 1
                return

            graph_code = normalize_gm(gm.print_readable(print_output=False))
            self.assertExpectedInline(
                graph_code,
                """\
class CompiledAutograd1(torch.nn.Module):
    def forward(self, inputs, sizes, scalars, hooks, packed_data):
        getitem = inputs[0]
        getitem_1 = inputs[1];  inputs = None
        getitem_2 = sizes[0];  getitem_2 = None
        getitem_3 = sizes[1]
        getitem_4 = sizes[2];  sizes = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cuda', index=0), 6, 0, None), [], False)]);  getitem = None
        getitem_5 = validate_outputs[0];  validate_outputs = None

        sum_backward0 = torch__dynamo_compiled_autograd_ops_SumBackward0([getitem_5], [True], []);  getitem_5 = None
        getitem_6 = sum_backward0[0];  sum_backward0 = None
        validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_6], [((None, None, device(type='cuda', index=0), 6, 0, None), [], False)]);  getitem_6 = None
        getitem_7 = validate_outputs_1[0];  validate_outputs_1 = None

        getitem_8 = hooks[0]
        getitem_9 = packed_data[0];  packed_data = None
        getitem_10 = hooks[1];  hooks = None
        call_hook = torch__dynamo_external_utils_call_hook(getitem_8, getitem_9, hook_type = 'unpack_hook');  getitem_8 = getitem_9 = None
        call_backward = torch__dynamo_external_utils_call_backward(getitem_10, (call_hook,), getitem_7);  getitem_10 = call_hook = getitem_7 = None
        getitem_12 = call_backward[0];  call_backward = None
        validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_12], [((None, None, device(type='cuda', index=0), 6, 0, None), [getitem_3], False)]);  getitem_12 = getitem_3 = None
        getitem_13 = validate_outputs_2[0];  validate_outputs_2 = None

        to_copy_backward0 = torch__dynamo_compiled_autograd_ops_ToCopyBackward0([getitem_13], [True], (None, None, device(type='cpu'), 6, 0, None));  getitem_13 = None
        getitem_14 = to_copy_backward0[0];  to_copy_backward0 = None
        validate_outputs_3 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_14], [((None, None, device(type='cpu'), 6, 0, None), [getitem_4], False)]);  getitem_14 = getitem_4 = None
        getitem_15 = validate_outputs_3[0];  validate_outputs_3 = None

        accumulate_grad__default = torch.ops.inductor.accumulate_grad_.default(getitem_1, None, getitem_15);  getitem_1 = getitem_15 = accumulate_grad__default = None
        _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
        return []
""",
            )

        self.check_output_and_recompiles(
            fn, compiler_fn=make_compiler_fn(gm_hook=check)
        )

    @skipIfWindows(msg="temp dir not compatible")
    def test_disk_offloading(self):
        # [FUNCTION DOC] Skipped on Windows. Checks offloading saved tensors to disk with pack
        # and unpack hooks (torch save and load in a temporary directory); the load causes a
        # graph break, giving one capture and two compiles; the hook asserts the second
        # captured graph text.
        with tempfile.TemporaryDirectory() as d:

            def fn():
                # [FUNCTION DOC] HELPER: Generator with disk-offload saved-tensor hooks
                # running a custom matmul-like Function on tensors of varying sizes and
                # yielding the grad.
                pack_count = 0

                def pack(x):
                    # [FUNCTION DOC] HELPER: Pack hook saving the tensor to a file in the
                    # temporary directory named by an enclosing counter and returning the
                    # path.
                    nonlocal pack_count
                    path = f"{d}/{pack_count}.pt"
                    torch.save(x, path)
                    return path

                def unpack(path):
                    # [FUNCTION DOC] HELPER: Unpack hook loading the tensor from the stored
                    # path.
                    x = torch.load(path)
                    return x

                class MyMatMul(torch.autograd.Function):
                    # [CLASS DOC] HELPER: Custom Function saving its input, with a matmul
                    # forward and a backward multiplying the gradient by the saved input.
                    @staticmethod
                    def forward(ctx, x):
                        # [FUNCTION DOC] HELPER: Saves the input and returns the matmul of it
                        # with itself.
                        ctx.save_for_backward(x)
                        return torch.matmul(x, x)

                    @staticmethod
                    def backward(ctx, grad_out):
                        # [FUNCTION DOC] HELPER: Returns the gradient times the saved input.
                        (x,) = ctx.saved_tensors
                        return grad_out * x

                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    for i in [10, 100, 10, 20, 30]:
                        x = torch.randn(i, requires_grad=True)
                        MyMatMul.apply(x).sum().backward()
                        yield x.grad

            i = 0

            def check(gm):
                # [FUNCTION DOC] HELPER: Graph hook skipping the first captured graph and
                # asserting the second equals the expected readable text (CPU variant).
                nonlocal i
                if i == 0:
                    i += 1
                    return

                graph_code = normalize_gm(gm.print_readable(print_output=False))
                self.assertExpectedInline(
                    graph_code,
                    """\
class CompiledAutograd1(torch.nn.Module):
    def forward(self, inputs, sizes, scalars, hooks, packed_data):
        getitem = inputs[0]
        getitem_1 = inputs[1];  inputs = None
        getitem_2 = sizes[0];  getitem_2 = None
        getitem_3 = sizes[1];  sizes = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [], False)]);  getitem = None
        getitem_4 = validate_outputs[0];  validate_outputs = None

        sum_backward0 = torch__dynamo_compiled_autograd_ops_SumBackward0([getitem_4], [True], []);  getitem_4 = None
        getitem_5 = sum_backward0[0];  sum_backward0 = None
        validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_5], [((None, None, device(type='cpu'), 6, 0, None), [], False)]);  getitem_5 = None
        getitem_6 = validate_outputs_1[0];  validate_outputs_1 = None

        getitem_7 = hooks[0]
        getitem_8 = packed_data[0];  packed_data = None
        getitem_9 = hooks[1];  hooks = None
        call_hook = torch__dynamo_external_utils_call_hook(getitem_7, getitem_8, hook_type = 'unpack_hook');  getitem_7 = getitem_8 = None
        call_backward = torch__dynamo_external_utils_call_backward(getitem_9, (call_hook,), getitem_6);  getitem_9 = call_hook = getitem_6 = None
        getitem_11 = call_backward[0];  call_backward = None
        validate_outputs_2 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_11], [((None, None, device(type='cpu'), 6, 0, None), [getitem_3], False)]);  getitem_11 = getitem_3 = None
        getitem_12 = validate_outputs_2[0];  validate_outputs_2 = None

        accumulate_grad__default = torch.ops.inductor.accumulate_grad_.default(getitem_1, None, getitem_12);  getitem_1 = getitem_12 = accumulate_grad__default = None
        _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
        return []
""",
                )

            # 1 graph break on torch.load -> 2 dynamo graphs
            self.check_output_and_recompiles(
                fn,
                count=[1, 2],
                compiler_fn=make_compiler_fn(fullgraph=False, gm_hook=check),
            )

    @skipIfWindows(msg="node name demangling inconsistent on windows")
    def test_backward_hook_relative_ordering_partial(self):
        # [FUNCTION DOC] Skipped on Windows. Checks backward hook relative ordering where
        # compiled autograd matches eager: hooks (pre and post) on every reachable node plus a
        # tensor hook and post-accumulate hook on the weight record names into an ordered list
        # that is compared between eager and compiled runs.
        # test backward hooks for cases that CA matches eager

        def fn():
            # [FUNCTION DOC] HELPER: Generator that registers pre and post hooks appending
            # labelled entries to an order list on every autograd node reachable from the
            # loss, plus tensor and post-accumulate hooks on the weight, runs backward and
            # yields the recorded order as a tuple.
            order = []

            class MyModule(nn.Module):
                # [CLASS DOC] HELPER: Module with a single bias-free linear layer.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 10, bias=False)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the linear layer.
                    return self.linear(x)

            x = torch.randn(10, 10)
            module = MyModule()

            def make_pre_hook(id):
                # [FUNCTION DOC] HELPER: Returns a pre-hook that appends a pre-hook label with
                # the given id to the order list.
                return lambda _: order.append(f"pre_hook_{id}")

            def make_post_hook(id):
                # [FUNCTION DOC] HELPER: Returns a post-hook that appends a post-hook label
                # with the given id to the order list.
                return lambda _1, _2: order.append(f"post_hook_{id}")

            count = 0

            def register_hooks_on_all_nodes(nodes):
                # [FUNCTION DOC] HELPER: Recursively visits the next functions of the given
                # node pairs, skipping missing ones, incrementing a counter and registering
                # labelled pre and post hooks on each node.
                nonlocal count
                for node, _ in nodes:
                    if node is None:
                        continue
                    count += 1
                    id = f"{node.name()}_{count}"
                    node.register_prehook(make_pre_hook(id))
                    node.register_hook(make_post_hook(id))
                    register_hooks_on_all_nodes(node.next_functions)

            loss = module(x).sum()
            register_hooks_on_all_nodes(((loss.grad_fn, None),))

            def make_tensor_pre_hook(id):
                # [FUNCTION DOC] HELPER: Returns a tensor hook that appends a tensor pre-hook
                # label with the given id to the order list.
                return lambda _: order.append(f"tensor_pre_hook_{id}")

            def make_post_acc_grad_hook(id):
                # [FUNCTION DOC] HELPER: Returns a post-accumulate hook that appends a label
                # with the given id to the order list.
                return lambda _: order.append(f"post_acc_grad_hook_{id}")

            module.linear.weight.register_hook(make_tensor_pre_hook("weight"))

            module.linear.weight.register_post_accumulate_grad_hook(
                make_post_acc_grad_hook("weight")
            )

            loss.backward()
            yield tuple(order)

        self.check_output_and_recompiles(fn)

    def test_checkpointing_sac(self):
        # [FUNCTION DOC] Checks selective activation checkpointing with a policy forcing
        # recomputation of addmm in a four-layer model; expects one capture and five compiles
        # with graph breaks allowed.
        # circular import
        from torch.utils.checkpoint import (
            checkpoint,
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def fn():
            # [FUNCTION DOC] HELPER: Generator that checkpoints a four-layer network with a
            # selective context, runs backward and yields all eight parameter grads.
            class mlp(nn.Module):
                # [CLASS DOC] HELPER: Four stacked linear layers of equal width.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Creates the four linear layers.
                    super().__init__()
                    self.layer1 = nn.Linear(10, 10)
                    self.layer2 = nn.Linear(10, 10)
                    self.layer3 = nn.Linear(10, 10)
                    self.layer4 = nn.Linear(10, 10)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the four layers in sequence.
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    return x

            recompute_list = [torch.ops.aten.addmm.default]

            def recompute_policy(ctx, op, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Checkpoint policy returning must-recompute for addmm
                # and prefer-save for everything else.
                if op in recompute_list:
                    return CheckpointPolicy.MUST_RECOMPUTE
                else:
                    return CheckpointPolicy.PREFER_SAVE

            def context_fn():
                # [FUNCTION DOC] HELPER: Creates selective checkpoint contexts from the
                # recompute policy.
                return create_selective_checkpoint_contexts(recompute_policy)

            model = mlp()
            input = torch.randn(1, 10)

            out = checkpoint(model, input, use_reentrant=False, context_fn=context_fn)
            out.sum().backward()
            yield model.layer1.weight.grad
            yield model.layer1.bias.grad
            yield model.layer2.weight.grad
            yield model.layer2.bias.grad
            yield model.layer3.weight.grad
            yield model.layer3.bias.grad
            yield model.layer4.weight.grad
            yield model.layer4.bias.grad

        self.check_output_and_recompiles(
            fn, count=[1, 5], compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_dont_dce_side_effects(self):
        # [FUNCTION DOC] Checks that dead code elimination in the captured graph keeps a
        # backward with a side effect (a random number generation whose result is unused): the
        # captured graph text must still contain the randn op. Sets the inductor fallback
        # random config as a side effect (not restored).
        class SideEffectfulBackward(torch.autograd.Function):
            # [CLASS DOC] HELPER: Identity custom Function whose backward generates an unused
            # random tensor as a side effect.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Identity forward.
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Draws and discards a random tensor, then returns the
                # gradient.
                torch.randn(10, 10)
                return gO

        x = torch.randn(10, 10, requires_grad=True)

        # https://github.com/pytorch/pytorch/issues/147171
        torch._inductor.config.fallback_random = True

        @torch.compile(backend="aot_eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (aot_eager) function applying the side-effecting
            # Function and summing.
            return SideEffectfulBackward.apply(x).sum()

        gm = None

        def extract(ca_gm):
            # [FUNCTION DOC] HELPER: Compiler that records the captured graph module in an
            # enclosing variable and returns it.
            nonlocal gm
            gm = ca_gm
            return ca_gm

        with compiled_autograd._enable(extract):
            fn(x).backward()

        self.assertTrue("aten.randn" in str(gm))

    def test_aot_bwd_gm_runnable(self):
        # [FUNCTION DOC] Checks that the backward module saved in the lazy backward info of an
        # AOT compiled function is runnable because post grad passes have not been applied to
        # it: a custom post-grad pass collects two graphs (forward and backward), the saved
        # backward module's graph differs from the collected backward graph, and compiled
        # autograd can then run backward.
        # This test ensures that the bw_module saved in
        # CompiledFunction._lazy_backward_info is executable,
        # by ensuring post grad passes have not ran on it.

        post_grad_graphs = []

        def post_grad_pass(graph):
            # [FUNCTION DOC] HELPER: Post-grad pass recording the graph into an enclosing list
            # and returning it unchanged.
            nonlocal post_grad_graphs
            post_grad_graphs.append(graph)
            return graph

        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        # forces symints to be saved for backward
        # and forces aot compilation of the backward
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 1)

        @torch.compile
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of a matmul of the
            # two inputs.
            return torch.matmul(x, y).sum()

        with inductor_config.patch(post_grad_custom_post_pass=post_grad_pass):
            loss = fn(x, y)
            self.assertEqual(len(post_grad_graphs), 2)  # 1 fwd and 1 bwd

        self.assertTrue(loss.grad_fn.name(), "CompiledFunctionBackward")
        self.assertIsNot(
            post_grad_graphs[1],
            loss.grad_fn._forward_cls._lazy_backward_info.bw_module.graph,
        )

        with compiled_autograd._enable(lambda gm: gm):
            loss.backward()

    @unittest.skipIf(
        not HAS_DTENSOR,
        "DTensor/FakePG requires distributed build",
    )
    def test_dtensor_backward_symints_do_not_reuse_native_sharding_cache(self):
        # [FUNCTION DOC] Skipped if the distributed build lacks DTensor. Regression test:
        # backward symbolic sizes under compiled autograd must not reuse the native sharding
        # cache entry from an eager backward on a distributed tensor. Initialises a fake
        # process group, runs the scenario twice and always destroys the group afterwards.
        def run_case():
            # [FUNCTION DOC] HELPER: Resets dynamo, then under compiled autograd builds a
            # two-rank CPU mesh, creates sharded distributed tensors, runs an eager
            # forward/backward and then a compiled one with the same shapes.
            torch._dynamo.reset()
            with compiled_autograd._enable(compiler_fn):
                mesh = DeviceMesh("cpu", torch.arange(2))

                def fn(x, y):
                    # [FUNCTION DOC] HELPER: Takes the sine of the first input and mutates the
                    # second input in place by adding two; returns the sine.
                    out = x.sin()
                    y.add_(2)
                    return out

                opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

                x_eager = DTensor.from_local(
                    torch.randn(4), mesh, [Shard(0)], run_check=False
                ).requires_grad_(True)
                y_eager = DTensor.from_local(
                    torch.randn(4), mesh, [Shard(0)], run_check=False
                ).requires_grad_(False)

                x = x_eager.clone().detach().requires_grad_(True)
                y = y_eager.clone().detach().requires_grad_(False)

                # The first backward populates the native sharding cache.
                # The compiled-autograd backward must not reuse that entry when it
                # contains SymInts from a different tracing context.
                fn(x_eager.clone(), y_eager).sum().backward()
                opt_fn(x.clone(), y).sum().backward()

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        try:
            run_case()
            run_case()
        finally:
            dist.destroy_process_group()

    def test_anomaly_mode_already_nan(self):
        # [FUNCTION DOC] Checks that compiled autograd with anomaly detection on raises
        # AssertionError saying a gradient is already NaN and unsupported, when a leaf starts
        # with a NaN grad.
        def fn():
            # [FUNCTION DOC] HELPER: Under anomaly detection, creates a leaf with a NaN grad,
            # runs matmul and backward inside the compiled autograd context.
            with torch.autograd.detect_anomaly():
                a = torch.randn(5, 5, requires_grad=True)
                a.grad = torch.full((5, 5), float("nan"))
                b = torch.randn(5, 5)
                out = torch.matmul(a, b)
                loss = out.sum()
                with torch._dynamo.compiled_autograd._enable(lambda gm: gm):
                    loss.backward()

        with self.assertRaisesRegex(
            AssertionError, "already having NaN gradient. This is not supported."
        ):
            fn()

    def test_anomaly_mode_backward(self):
        # [FUNCTION DOC] Checks that, with anomaly detection, a backward producing NaN
        # gradients raises RuntimeError saying Compiled Autograd returned NaN gradients for
        # parameters.
        def fn():
            # [FUNCTION DOC] HELPER: Under anomaly detection runs backward through a custom
            # Function returning NaN gradients inside the compiled autograd context.
            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Identity custom Function whose backward returns a
                # NaN-filled tensor of the gradient's size.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Identity forward.
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Returns a NaN-filled tensor of the gradient's
                    # size.
                    return torch.full(gO.size(), float("nan"))

            with torch.autograd.detect_anomaly():
                a = torch.randn(5, 5, requires_grad=True)
                out = MyFn.apply(a)
                loss = out.sum()
                with torch._dynamo.compiled_autograd._enable(lambda gm: gm):
                    loss.backward()

        with self.assertRaisesRegex(
            RuntimeError, "Compiled Autograd returned NaN gradients for parameters"
        ):
            fn()

    def test_anomaly_mode_grad(self):
        # [FUNCTION DOC] Checks that, with anomaly detection, torch.autograd.grad producing
        # NaN raises RuntimeError saying Compiled Autograd returned NaN gradients for output
        # nodes.
        def fn():
            # [FUNCTION DOC] HELPER: Under anomaly detection computes the functional grad
            # through a NaN-producing custom Function inside the compiled autograd context.
            class MyFn(torch.autograd.Function):
                # [CLASS DOC] HELPER: Identity custom Function whose backward returns a
                # NaN-filled tensor of the gradient's size.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Identity forward.
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # [FUNCTION DOC] HELPER: Returns a NaN-filled tensor of the gradient's
                    # size.
                    return torch.full(gO.size(), float("nan"))

            with torch.autograd.detect_anomaly():
                a = torch.randn(5, 5, requires_grad=True)
                out = MyFn.apply(a)
                loss = out.sum()
                with torch._dynamo.compiled_autograd._enable(lambda gm: gm):
                    torch.autograd.grad(loss, inputs=a)

        with self.assertRaisesRegex(
            RuntimeError, "Compiled Autograd returned NaN gradients for output nodes"
        ):
            fn()

    def test_higher_order_gradients(self):
        # [FUNCTION DOC] Checks higher order gradients with different compilation backends:
        # with the plain eager backend all four derivatives (12, 12, 6, 0) are captured (four
        # captures, then a cache hit on repeat); with AOTAutograd in the forward or in
        # compiled autograd, creating a graph for the further derivative fails with a
        # RuntimeError about higher order gradients, which the helper tolerates leaving the
        # later derivatives None.
        def f(x):
            # [FUNCTION DOC] HELPER: Cubes the input.
            return x**3

        def fn(fwd_compiler, ca_compiler):
            # [FUNCTION DOC] HELPER: Computes up to four successive derivatives of the cubed
            # scalar under compiled autograd with the given forward compiler decorator and
            # compiled autograd compiler; tolerates only the expected higher-order
            # RuntimeError (re-raising other errors as AssertionError) and returns the four
            # values, some possibly None.
            torch.manual_seed(123)
            x = torch.tensor(2.0, requires_grad=True)
            first, second, third, fourth = None, None, None, None
            try:
                with compiled_autograd._enable(ca_compiler):
                    first = torch.autograd.grad(
                        fwd_compiler(f)(x), x, create_graph=True
                    )[0]
                    second = torch.autograd.grad(first, x, create_graph=True)[0]
                    third = torch.autograd.grad(second, x, create_graph=True)[0]
                    fourth = torch.autograd.grad(third, x, create_graph=True)[0]
            except RuntimeError as e:
                if "does not currently support higher order gradients" not in str(e):
                    raise AssertionError(
                        f"Expected error about higher order gradients, got: {e}"
                    ) from e
                return (first, second, third, fourth)

            return (first, second, third, fourth)

        def eager():
            # [FUNCTION DOC] HELPER: Returns a torch.compile decorator using the eager
            # backend.
            return torch.compile(backend="eager")

        def aot_eager():
            # [FUNCTION DOC] HELPER: Returns a torch.compile decorator using the aot_eager
            # backend.
            return torch.compile(backend="aot_eager")

        # Without AOTAutograd, no problem
        first, second, third, fourth = fn(eager(), eager())
        self.assertEqual(counters["compiled_autograd"]["captures"], 4)
        self.assertEqual(first, 12)  # 3x^2
        self.assertEqual(second, 12)  # 6x
        self.assertEqual(third, 6)  # 6
        self.assertEqual(fourth, 0)
        # and should cache hit
        counters.clear()
        _ = fn(eager(), eager())
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        torch._dynamo.reset()

        # With AOTAutograd, can't create_graph
        first, second, third, fourth = fn(aot_eager(), aot_eager())
        self.assertIsNone(second)

        first, second, third, fourth = fn(aot_eager(), eager())
        self.assertIsNone(second)

        first, second, third, fourth = fn(eager(), aot_eager())
        self.assertIsNone(third)

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "FakePG relies on distributed build",
    )
    def test_ddp_cpp_reducer_error(self):
        # [FUNCTION DOC] Skipped without distributed support. Checks that compiled autograd
        # with a model wrapped in the C++-reducer DDP raises RuntimeError about
        # incompatibility with the C++ reducer, suggesting the python reducer option. Uses a
        # fake process group and destroys it afterward.
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        try:
            model = torch.nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
            model = DDP(model)
            inputs = torch.randn(10, 10)
            loss = model(inputs).sum()
            with (
                compiled_autograd._enable(compiler_fn),
                self.assertRaisesRegex(
                    RuntimeError,
                    (
                        r"Compiled autograd is not compatible with C\+\+ DDP Reducer, "
                        r'please use torch._dynamo.config.optimize_ddp="python_reducer"'
                    ),
                ),
            ):
                loss.backward()

        finally:
            dist.destroy_process_group()

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "FakePG relies on distributed build",
    )
    @config.patch(optimize_ddp="python_reducer")
    def test_ddp_python_reducer(self):
        # [FUNCTION DOC] Skipped without distributed support. Checks DDP with the python
        # reducer config works under compiled autograd (no error, one capture), using a fake
        # process group destroyed afterward.
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        try:
            model = torch.nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
            model = DDP(model)
            inputs = torch.randn(10, 10)
            loss = model(inputs).sum()
            with compiled_autograd._enable(compiler_fn):
                # no error expected
                loss.backward()
            self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        finally:
            dist.destroy_process_group()

    # Case 1.1: Stealable dense new_grad
    # if (!GradMode::is_enabled() && !new_grad.is_sparse() &&
    #     !new_grad.is_sparse_csr() &&
    #     !(variable.is_sparse_csr() && new_grad.layout() == at::kStrided) &&
    #     at::caching::adjusted_use_count(new_grad) <= num_expected_refs &&
    #     (new_grad.is_mkldnn() || utils::obeys_layout_contract(new_grad, variable))) {
    @unittest.expectedFailure
    def test_accumulate_grad_polyfill_case_1_1(self):
        # [FUNCTION DOC] Expected failure. Polyfill of accumulate-grad case 1.1: a stealable
        # dense gradient should be moved into the leaf without copying (same storage as seen
        # in a tensor hook) and detached. Currently fails in compiled autograd, as the test is
        # marked expectedFailure.
        def fn():
            # [FUNCTION DOC] HELPER: Generator that applies a custom Function returning a
            # fresh ones-times-five gradient, records the storage identity inside a tensor
            # hook and asserts the final grad is defined, has the right content, does not
            # require grad, and shares storage with what the hook saw; yields the grad.
            class StealableDenseOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a freshly created
                # gradient of fives that does not require grad.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Returns a new ones tensor times five with no grad
                    # tracking.
                    return torch.ones_like(grad_output, requires_grad=False) * 5

            pre_hook_storage_id = None

            def check(grad):
                # [FUNCTION DOC] HELPER: Tensor hook recording the storage identity of the
                # gradient once; raises AssertionError if called twice.
                nonlocal pre_hook_storage_id
                if pre_hook_storage_id is not None:
                    raise AssertionError
                pre_hook_storage_id = id(grad.untyped_storage())

            var = torch.randn(2, 2, requires_grad=True)
            var.register_hook(check)
            output = StealableDenseOp.apply(var)
            output.backward(torch.ones_like(output))

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if not torch.equal(var.grad, torch.ones_like(var) * 5):
                raise AssertionError("Grad content should be as returned by backward")
            if var.grad.requires_grad is not False:
                raise AssertionError("Detached grad should not require grad")
            if id(var.grad.untyped_storage()) != pre_hook_storage_id:
                raise AssertionError("Should be stolen")
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 2],
        )

    # Case 1.2: Stealable sparse new_grad
    # } else if (!GradMode::is_enabled() && new_grad.is_sparse() &&
    #            new_grad._indices().is_contiguous() &&
    #            new_grad._values().is_contiguous() &&
    #            new_grad._indices().use_count() <= 1 &&
    #            new_grad._values().use_count() <= 1 &&
    #            new_grad.use_count() <= num_expected_refs) {
    @unittest.expectedFailure
    def test_accumulate_grad_polyfill_case_1_2(self):
        # [FUNCTION DOC] Expected failure. Polyfill of accumulate-grad case 1.2: a stealable
        # sparse gradient with contiguous indices and values should be moved without copying.
        # Currently fails under compiled autograd (marked expectedFailure).
        def fn():
            # [FUNCTION DOC] HELPER: Generator applying a custom Function returning a sparse
            # gradient, recording the storage identities of indices and values in a hook, and
            # asserting the final grad is sparse, has the expected dense content and shares
            # both storages; yields the grad.
            class StealableSparseOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a sparse COO
                # gradient with two diagonal entries of five.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Builds and returns a sparse tensor of fives on
                    # the diagonal with the gradient's size, without grad tracking.
                    size = grad_output.size()
                    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
                    values = torch.tensor([5.0, 5.0])
                    return torch.sparse_coo_tensor(
                        indices, values, size, requires_grad=False
                    )

            pre_hook_storages_id = None

            def check(grad):
                # [FUNCTION DOC] HELPER: Tensor hook recording the storage identities of the
                # indices and values once; raises AssertionError if called twice.
                nonlocal pre_hook_storages_id
                if pre_hook_storages_id is not None:
                    raise AssertionError
                pre_hook_storages_id = [
                    id(grad._indices().untyped_storage()),
                    id(grad._values().untyped_storage()),
                ]

            var = torch.randn(2, 2, requires_grad=True)
            var.register_hook(check)
            output = StealableSparseOp.apply(var)
            output.backward(torch.ones_like(output))

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if not var.grad.is_sparse:
                raise AssertionError("Grad should be sparse")
            expected_dense_grad = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
            if not torch.equal(var.grad.to_dense(), expected_dense_grad):
                raise AssertionError("Content should be equal after shallow copy")
            if var.grad.requires_grad is not False:
                raise AssertionError("Detached grad should not require grad")
            if id(var.grad._indices().untyped_storage()) != pre_hook_storages_id[0]:
                raise AssertionError("Should be stolen")
            if id(var.grad._values().untyped_storage()) != pre_hook_storages_id[1]:
                raise AssertionError("Should be stolen")
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 2],
        )

    # Case 1.3: Cloning sparse/nested new_grad
    # else {
    #   if (new_grad.is_sparse() || new_grad.is_sparse_csr() ||
    #       new_grad.is_nested()) {
    def test_accumulate_grad_polyfill_case_1_3(self):
        # [FUNCTION DOC] Polyfill of accumulate-grad case 1.3: with create_graph, a sparse
        # gradient requiring grad must be cloned rather than stolen. Checks the result is
        # sparse, equal in content, requires grad and uses new storage for indices and values;
        # counts one capture and one compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator running backward with create_graph through a
            # custom Function returning a sparse gradient that requires grad, asserting the
            # grad properties and that storages differ from those seen in the hook, then
            # yielding the grad.
            class CloneSparseGradOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a sparse COO
                # gradient requiring grad.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Builds a sparse gradient (values requiring grad)
                    # of the gradient's size.
                    size = grad_output.size()
                    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
                    values = torch.tensor(
                        [5.0, 5.0], requires_grad=True
                    )  # Requires grad
                    return torch.sparse_coo_tensor(
                        indices, values, size, requires_grad=True
                    )

            pre_hook_storages_id = None

            def check(grad):
                # [FUNCTION DOC] HELPER: Tensor hook recording the storage identities of the
                # indices and values once; raises AssertionError if called twice.
                nonlocal pre_hook_storages_id
                if pre_hook_storages_id is not None:
                    raise AssertionError
                pre_hook_storages_id = [
                    id(grad._indices().untyped_storage()),
                    id(grad._values().untyped_storage()),
                ]

            var = torch.randn(2, 2, requires_grad=True)
            var.register_hook(check)
            output = CloneSparseGradOp.apply(var)
            output.backward(
                torch.ones_like(output), create_graph=True
            )  # grad mode == create_graph

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if not var.grad.is_sparse:
                raise AssertionError("Grad should be sparse")
            expected_dense_grad = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
            if not torch.equal(var.grad.to_dense(), expected_dense_grad):
                raise AssertionError("Content should be equal after clone")
            if not var.grad.requires_grad:
                raise AssertionError("Grad should require grad for double backward")
            if id(var.grad._indices().untyped_storage()) == pre_hook_storages_id[0]:
                raise AssertionError("Should be copied")
            if id(var.grad._values().untyped_storage()) == pre_hook_storages_id[1]:
                raise AssertionError("Should be copied")
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 1],
        )

    # Case 1.5.1: Dense variable gradient layout contract
    # else { // Covers various deep copy scenarios not covered by specific stealable paths
    #   ...
    #   if (new_grad.is_mkldnn()) {
    #     ...
    #   } else {
    #       // Deep copies new_grad according to the "Gradient Layout Contract."
    #       update_grad(utils::clone_obey_contract(new_grad, variable));
    #   }
    # }
    def test_accumulate_grad_polyfill_case_1_5_1(self):
        # [FUNCTION DOC] Polyfill of accumulate-grad case 1.5.1: a dense gradient with extra
        # references (held by a hook) must be deep copied following the layout contract;
        # asserts the stored grad does not share storage with the hooked one.
        def fn():
            # [FUNCTION DOC] HELPER: Generator applying a custom Function returning a
            # tens-valued gradient, holding a reference in a hook, and asserting content and
            # copied storage; yields the grad.
            class NotStealableRefsOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a gradient of
                # tens.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Returns a ones-like gradient times ten without
                    # grad tracking.
                    return torch.ones_like(grad_output, requires_grad=False) * 10.0

            var = torch.randn(2, 2, requires_grad=True)
            grad_ref_holder = [None]

            def check(grad):
                # [FUNCTION DOC] HELPER: Tensor hook storing a reference to the gradient
                # (forcing a copy) and returning it.
                # forces a clone due to refcount
                grad_ref_holder[0] = grad
                return grad

            var.register_hook(check)
            output = NotStealableRefsOp.apply(var)
            output.backward(torch.ones_like(output))

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if not torch.equal(var.grad, torch.ones_like(var) * 10.0):
                raise AssertionError("Grad content should be as returned by backward")
            if grad_ref_holder[0].untyped_storage() is var.grad.untyped_storage():
                raise AssertionError("Should be copied")
            yield var.grad

        self.check_output_and_recompiles(fn)

    # Case 1.5.2: Non-dense variable gradient layout contract
    # else { // Covers various deep copy scenarios not covered by specific stealable paths
    #   ...
    #   if (new_grad.is_mkldnn()) {
    #     ...
    #   } else {
    #       // Deep copies new_grad according to the "Gradient Layout Contract."
    #       update_grad(utils::clone_obey_contract(new_grad, variable));
    #   }
    # }
    def test_accumulate_grad_polyfill_case_1_5_2(self):
        # [FUNCTION DOC] Polyfill of accumulate-grad case 1.5.2: a non-contiguous leaf
        # (strided slice) gets a contiguous cloned grad; asserts content seven, contiguity and
        # non-shared storage.
        def fn():
            # [FUNCTION DOC] HELPER: Generator creating a non-contiguous leaf, applying a
            # custom Function returning a sevens gradient while a hook holds a reference,
            # asserting contiguity, content and storage distinctness, then yielding the grad.
            class SimpleDenseGradOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a gradient of
                # sevens.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Returns a ones-like gradient times seven without
                    # grad tracking.
                    return torch.ones_like(grad_output, requires_grad=False) * 7.0

            # Create a non-contiguous variable
            base_tensor = torch.randn(4, 4)
            var = base_tensor[::2, ::2]
            if var.is_contiguous():
                raise AssertionError("Variable should be non-contiguous for this test")
            var.requires_grad_(True)

            grad_ref_holder = [None]

            def check(grad):
                # [FUNCTION DOC] HELPER: Tensor hook storing a reference to the gradient
                # (forcing a copy) and returning it.
                # forces a clone due to refcount
                grad_ref_holder[0] = grad
                return grad

            var.register_hook(check)
            output = SimpleDenseGradOp.apply(var)
            output.backward(torch.ones_like(output))

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            # The `clone_obey_contract` branch 2 (`new_grad.clone(at::MemoryFormat::Contiguous)`)
            # will make the resulting grad contiguous.
            if not var.grad.is_contiguous():
                raise AssertionError(
                    "Resulting grad should be contiguous due to branch 2 of clone_obey_contract"
                )
            if not torch.equal(var.grad, torch.ones_like(var) * 7.0):
                raise AssertionError("Grad content should be as returned by backward")
            if grad_ref_holder[0].untyped_storage() is var.grad.untyped_storage():
                raise AssertionError("Should be copied")
            yield var.grad

        self.check_output_and_recompiles(
            fn,
        )

    # Case 2.1: Sparse variable_grad + Dense new_grad
    # } else if (!GradMode::is_enabled()) {
    #   if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
    #       auto result = new_grad + variable_grad;
    def test_accumulate_grad_polyfill_case_2_1(self):
        # [FUNCTION DOC] Polyfill of accumulate-grad case 2.1: a sparse existing grad plus a
        # dense new grad must yield a dense out-of-place sum (new grad object). Uses an
        # identity compiler (known tracking issue) and expects one capture and no compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator presetting a sparse grad of ones on the
            # diagonal, backpropagating a dense gradient of threes and asserting the grad is
            # dense, equals the sum and is a new object; yields it.
            class SparseVarGradDenseNewGradOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a dense gradient
                # of threes.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Returns a ones-like tensor times three.
                    return torch.ones_like(grad_output) * 3.0

            var = torch.randn(2, 2, requires_grad=True)
            indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
            values = torch.tensor([1.0, 1.0])
            var.grad = torch.sparse_coo_tensor(
                indices, values, var.size(), requires_grad=False
            )
            initial_grad_ref = var.grad
            output = SparseVarGradDenseNewGradOp.apply(var)

            expected_sum = (torch.ones_like(var) * 3.0) + initial_grad_ref.to_dense()
            output.backward(torch.ones_like(output))

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if var.grad.is_sparse:
                raise AssertionError("Resulting grad should be dense")
            if not torch.equal(var.grad, expected_sum):
                raise AssertionError("Grad content should be the sum")
            if var.grad is initial_grad_ref:
                raise AssertionError("Grad object should be replaced (out-of-place)")
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=lambda gm: gm,  # https://github.com/pytorch/pytorch/issues/154161
            count=[1, 0],
        )

    # Case 2.3.1: Dense/Dense in-place addition
    # } else if (!GradMode::is_enabled()) {
    #   ...
    # } else {
    #   variable_grad += new_grad;
    def test_accumulate_grad_polyfill_case_2_3_1(self):
        # [FUNCTION DOC] Polyfill of accumulate-grad case 2.3.1: dense existing grad plus
        # dense new grad is added in place, so the same grad object holds the sum.
        def fn():
            # [FUNCTION DOC] HELPER: Generator presetting a dense grad of ones,
            # backpropagating a dense gradient of threes and asserting the sum is correct,
            # dense and the same object; yields it.
            class DenseVarGradDenseNewGradOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a dense gradient
                # of threes.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Returns a ones-like tensor times three.
                    return torch.ones_like(grad_output) * 3.0

            var = torch.randn(2, 2, requires_grad=True)
            var.grad = torch.ones_like(var) * 1.0
            initial_grad_ref = var.grad
            output = DenseVarGradDenseNewGradOp.apply(var)
            expected_sum = initial_grad_ref + (torch.ones_like(var) * 3.0)
            output.backward(torch.ones_like(output))

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if var.grad.is_sparse:
                raise AssertionError("Resulting grad should be dense")
            if not torch.equal(var.grad, expected_sum):
                raise AssertionError("Grad content should be the sum")
            if var.grad is not initial_grad_ref:
                raise AssertionError(
                    "Grad object should be modified in-place (same object)"
                )
            yield var.grad

        self.check_output_and_recompiles(fn)

    # Case 2.3.2: Sparse/Sparse in-place addition
    # } else if (!GradMode::is_enabled()) {
    #   ...
    # } else {
    #   variable_grad += new_grad;
    def test_accumulate_grad_polyfill_case_2_3_2(self):
        # [FUNCTION DOC] Polyfill of accumulate-grad case 2.3.2: sparse existing grad plus
        # sparse new grad is added in place and stays sparse. Uses an identity compiler (known
        # tracking issue), one capture, no compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator presetting a sparse grad, backpropagating a
            # sparse gradient of threes and asserting the grad stays sparse, holds the
            # dense-equivalent sum and is the same object; yields it.
            class SparseVarGradSparseNewGradOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a sparse gradient
                # with threes on the diagonal.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Builds a sparse COO tensor with threes on the
                    # diagonal sized like the gradient.
                    size = grad_output.size()
                    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
                    values = torch.tensor([3.0, 3.0])
                    return torch.sparse_coo_tensor(
                        indices, values, size, requires_grad=False
                    )

            var = torch.randn(2, 2, requires_grad=True)
            indices_v = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
            values_v = torch.tensor([1.0, 2.0])
            var.grad = torch.sparse_coo_tensor(
                indices_v, values_v, var.size(), requires_grad=False
            )
            initial_grad_ref = var.grad

            output = SparseVarGradSparseNewGradOp.apply(var)

            new_grad_for_sum = torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
                torch.tensor([3.0, 3.0]),
                var.size(),
            )
            expected_sum_dense = (
                initial_grad_ref.to_dense() + new_grad_for_sum.to_dense()
            )

            output.backward(torch.ones_like(output))

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if not var.grad.is_sparse:
                raise AssertionError("Resulting grad should remain sparse")
            if not torch.equal(var.grad.to_dense(), expected_sum_dense):
                raise AssertionError("Grad content should be the sum of sparse grads")
            if var.grad is not initial_grad_ref:
                raise AssertionError(
                    "Grad object should be modified in-place (same object)"
                )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=lambda gm: gm,  # https://github.com/pytorch/pytorch/issues/154161
            count=[1, 0],
        )

    # Case 2.3.3: Dense/Sparse in-place addition
    # } else if (!GradMode::is_enabled()) {
    #   ...
    # } else {
    #   variable_grad += new_grad;
    def test_accumulate_grad_polyfill_case_2_3_3(self):
        # [FUNCTION DOC] Polyfill of accumulate-grad case 2.3.3: dense existing grad plus
        # sparse new grad is added in place, remaining dense. One capture, one compile with
        # graph breaks allowed.
        def fn():
            # [FUNCTION DOC] HELPER: Generator presetting a dense grad of ones,
            # backpropagating a sparse gradient and asserting the dense in-place sum and
            # object identity; yields it.
            class DenseVarGradSparseNewGradOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a sparse gradient
                # with threes on the diagonal.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Builds a sparse COO tensor with threes on the
                    # diagonal sized like the gradient.
                    size = grad_output.size()
                    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
                    values = torch.tensor([3.0, 3.0])  # New sparse values
                    return torch.sparse_coo_tensor(
                        indices, values, size, requires_grad=False
                    )

            var = torch.randn(2, 2, requires_grad=True)
            var.grad = torch.ones_like(var) * 1.0  # Initial value
            initial_grad_ref = var.grad
            output = DenseVarGradSparseNewGradOp.apply(var)

            new_grad_for_sum = torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
                torch.tensor([3.0, 3.0]),
                var.size(),
            ).to_dense()
            expected_sum = initial_grad_ref + new_grad_for_sum

            output.backward(torch.ones_like(output))

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if var.grad.is_sparse:
                raise AssertionError("Resulting grad should be dense")
            if not torch.equal(var.grad, expected_sum):
                raise AssertionError("Grad content should be the sum")
            if var.grad is not initial_grad_ref:
                raise AssertionError(
                    "Grad object should be modified in-place (same object)"
                )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 1],
        )

    # Case 3.1: Sparse variable_grad + Dense new_grad (reorder into Dense + Sparse)
    # } else { // if GradMode::is_enabled()
    #   at::Tensor result;
    #   if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
    #     result = new_grad + variable_grad;
    #   }
    # }
    def test_accumulate_grad_polyfill_case_3_1(self):
        # [FUNCTION DOC] Polyfill of accumulate-grad case 3.1: with create_graph, a sparse
        # existing grad plus a dense new grad gives a dense out-of-place result that tracks
        # history. Identity compiler, one capture, no compile.
        def fn():
            # [FUNCTION DOC] HELPER: Generator presetting a sparse grad requiring grad,
            # running backward with create_graph on a dense gradient requiring grad and
            # asserting a dense, correct, new, history-tracking result; yields it.
            class SparseVarGradDenseNewGradDoubleBackwardOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a dense gradient
                # of threes that requires grad.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Returns a ones-like tensor requiring grad times
                    # three.
                    return torch.ones_like(grad_output, requires_grad=True) * 3.0

            var = torch.randn(2, 2, requires_grad=True)
            indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
            values = torch.tensor([1.0, 1.0], requires_grad=True)
            var.grad = torch.sparse_coo_tensor(
                indices, values, var.size(), requires_grad=True
            )
            initial_grad_ref = var.grad

            output = SparseVarGradDenseNewGradDoubleBackwardOp.apply(var)

            expected_sum = (
                torch.ones_like(var, requires_grad=True) * 3.0
            ) + initial_grad_ref.to_dense()

            output.backward(torch.ones_like(output), create_graph=True)

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if var.grad.is_sparse:
                raise AssertionError("Resulting grad should be dense")
            if not torch.equal(var.grad, expected_sum):
                raise AssertionError("Grad content should be the sum")
            if var.grad is initial_grad_ref:
                raise AssertionError("Grad object should be replaced (out-of-place)")
            if not var.grad.requires_grad:
                raise AssertionError(
                    "Resulting grad should track history for double backward"
                )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=lambda gm: gm,  # https://github.com/pytorch/pytorch/issues/154161
            count=[1, 0],
        )

    # Case 3.2: variable_grad.defined() & GradMode::is_enabled() - Double backward (dense variable_grad + dense new_grad)
    # } else { // if GradMode::is_enabled()
    #   at::Tensor result;
    #   ...
    #   } else {
    #     result = variable_grad + new_grad;
    #   }
    # }
    def test_accumulate_grad_polyfill_case_3_2(self):
        # [FUNCTION DOC] Polyfill of accumulate-grad case 3.2: with create_graph, a dense
        # existing grad plus a dense new grad gives a new tracked tensor. One capture and two
        # compiles with graph breaks allowed.
        def fn():
            # [FUNCTION DOC] HELPER: Generator presetting a dense grad, running backward with
            # create_graph and asserting the new grad is dense, correct, a different object
            # and requires grad; yields it.
            class DenseVarGradDenseNewGradDoubleBackwardOp(BaseCustomOp):
                # [CLASS DOC] HELPER: Custom Function whose backward returns a dense gradient
                # of threes that requires grad.
                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Returns a ones-like tensor requiring grad times
                    # three.
                    return torch.ones_like(grad_output, requires_grad=True) * 3.0

            var = torch.randn(2, 2, requires_grad=True)
            var.grad = torch.ones_like(var) * 1.0
            initial_grad_ref = var.grad

            output = DenseVarGradDenseNewGradDoubleBackwardOp.apply(var)

            expected_sum = initial_grad_ref + (
                torch.ones_like(var, requires_grad=True) * 3.0
            )

            output.backward(torch.ones_like(output), create_graph=True)

            if var.grad is None:
                raise AssertionError("Grad should be defined")
            if var.grad.is_sparse:
                raise AssertionError("Resulting grad should be dense")
            if not torch.equal(var.grad, expected_sum):
                raise AssertionError("Grad content should be the sum")
            if var.grad is initial_grad_ref:
                raise AssertionError("Grad object should be replaced (out-of-place)")
            if not var.grad.requires_grad:
                raise AssertionError(
                    "Resulting grad should track history for double backward"
                )
            yield var.grad

        self.check_output_and_recompiles(
            fn,
            compiler_fn=make_compiler_fn(fullgraph=False),
            count=[1, 2],
        )

    def test_torch_function_mode(self):
        # [FUNCTION DOC] Checks which torch function calls a logging function mode sees:
        # during the forward it records the multiplications and the sum; during the compiled
        # autograd backward it only sees the multithreading toggles around the backward call.
        # Compares the recorded names to an expected inline string.
        called_funcs = []

        class LoggingTorchFunctionMode(BaseTorchFunctionMode):
            # [CLASS DOC] HELPER: Torch function mode recording the name of every intercepted
            # function into an enclosing list before delegating to the base mode.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Appends the function name to the log list and returns
                # the result of the base implementation.
                called_funcs.append(str(func.__name__))
                return super().__torch_function__(func, types, args, kwargs)

        class MyLoss(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function reducing its input to a sum and, in
            # backward, returning a ones tensor scaled by the incoming gradient.
            @staticmethod
            def forward(ctx, out):
                # [FUNCTION DOC] HELPER: Saves the input and returns its sum.
                ctx.save_for_backward(out)
                return out.sum()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Loads the saved tensor and returns ones of its shape
                # times the gradient.
                (saved,) = ctx.saved_tensors
                return torch.ones_like(saved) * grad_output

        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2)
        z = torch.randn(2, 2)

        def fwd(x, y, z):
            # [FUNCTION DOC] HELPER: Multiplies three tensors and passes the product through
            # the sum Function.
            out = x * y * z
            loss = MyLoss.apply(out)
            return loss

        with LoggingTorchFunctionMode():
            called_funcs.append("Forward")
            loss = fwd(x, y, z)
            called_funcs.append("Backward")
            with torch._dynamo.compiled_autograd._enable(torch.compile):
                loss.backward()

        self.assertExpectedInline(
            "\n".join(called_funcs),
            """\
Forward
mul
mul
sum
Backward
_set_multithreading_enabled
backward
_set_multithreading_enabled""",
        )

    def test_torch_dispatch_mode(self):
        # [FUNCTION DOC] Checks which dispatcher calls a logging dispatch mode sees during
        # compiled autograd backward with an identity compiler; compares the recorded op names
        # (including tensor allocations and copies from the compiled autograd runtime) to an
        # expected inline string.
        called_funcs = []

        class LoggingTorchDispatchMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Torch dispatch mode recording the name of every intercepted
            # op into an enclosing list and running the op normally.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Appends the op name to the log list and returns the
                # result of calling the op with the given arguments.
                called_funcs.append(str(func.__name__))
                return func(*args, **kwargs)

        class MyLoss(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom Function reducing its input to a sum and, in
            # backward, returning a ones tensor scaled by the incoming gradient.
            @staticmethod
            def forward(ctx, out):
                # [FUNCTION DOC] HELPER: Saves the input and returns its sum.
                ctx.save_for_backward(out)
                return out.sum()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Loads the saved tensor and returns ones of its shape
                # times the gradient.
                (saved,) = ctx.saved_tensors
                return torch.ones_like(saved) * grad_output

        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2)
        z = torch.randn(2, 2)

        def fwd(x, y, z):
            # [FUNCTION DOC] HELPER: Multiplies three tensors and passes the product through
            # the sum Function.
            out = x * y * z
            loss = MyLoss.apply(out)
            return loss

        with LoggingTorchDispatchMode():
            called_funcs.append("Forward")
            loss = fwd(x, y, z)
            called_funcs.append("Backward")
            with torch._dynamo.compiled_autograd._enable(lambda gm: gm):
                loss.backward()

        self.assertExpectedInline(
            "\n".join(called_funcs),
            """\
Forward
mul.Tensor
mul.Tensor
sum.default
Backward
ones_like.default
empty.memory_format
empty.memory_format
empty.memory_format
empty.memory_format
empty.memory_format
empty.memory_format
ones_like.default
mul.Tensor
mul.Tensor
mul.Tensor
new_empty_strided.default
copy_.default""",
        )


def load_test_module(name):
    # [FUNCTION DOC] HELPER: Loads a sibling test module by dotted name from the parent
    # directory of this test directory (temporarily extending the module search path), using a
    # source file loader, and returns the loaded module.
    testdir = Path(__file__).absolute().parent.parent
    with mock.patch("sys.path", [*sys.path, str(testdir)]):
        return SourceFileLoader(
            name, str(testdir / f"{name.replace('.', '/')}.py")
        ).load_module()


def make_wrapped(fn, ctx_fns):
    # [FUNCTION DOC] HELPER: Wraps a test method so each call first resets dynamo, enters
    # every context produced by the given factory list in a fresh exit stack, runs the method
    # and closes the contexts even on exceptions.
    @functools.wraps(fn)
    def wrapped(self):
        # [FUNCTION DOC] HELPER: The wrapper: resets dynamo, creates and enters a context from
        # each factory, calls the original test with the instance and returns its result.
        torch._dynamo.reset()
        with contextlib.ExitStack() as stack:
            for ctx_fn in ctx_fns:
                stack.enter_context(ctx_fn())
            return fn(self)

    return wrapped


def lookup_backend(test_name):
    # [FUNCTION DOC] HELPER: Chooses the backend for a test by name: the next weaker backend
    # than the lowest xfail table that lists it, or the inductor default if unlisted; raises
    # AssertionError if the name is in the weakest backend xfail table.
    if test_name in xfail_by_backend["inductor"]:
        return "aot_eager"
    elif test_name in xfail_by_backend["aot_eager"]:
        return "eager"
    elif test_name in xfail_by_backend["eager"]:
        return "ca_eager"
    else:
        if test_name in xfail_by_backend["ca_eager"]:
            raise AssertionError(f"test {test_name} is in ca_eager xfail list")
        return "inductor"


def wrap_test_class(orig_cls):
    # [FUNCTION DOC] HELPER: Builds a subclass named after the original with the suffix
    # indicating compiled autograd. Copies the class namespace and, for each callable member
    # not in the skipped set, marks it an expected failure if it matches the unsupported
    # pattern, the weakest-backend xfail set or the divergence set; otherwise wraps test
    # methods to run inside the compiled autograd context with a backend-specific compiler
    # (full graph unless known to graph break), omitting inductor-backed tests when no GPU and
    # triton. Sets the file attribute and returns the new class.
    dct = orig_cls.__dict__.copy()
    for name in list(dct.keys()):
        fn = dct[name]
        if not callable(fn) or name in skipped_tests:
            continue
        elif (
            xfail_re.match(name)
            or name in xfail_by_backend["ca_eager"]
            or name in xfail_divergence_from_eager
        ):
            dct[name] = unittest.expectedFailure
        elif name.startswith("test_"):
            backend = lookup_backend(name)
            if not HAS_CUDA_AND_TRITON and backend == "inductor":
                continue
            compiler_fn = make_compiler_fn(
                backend=backend,
                fullgraph=name not in known_graph_breaks_tests,
            )
            ctx_fns = [
                functools.partial(compiled_autograd._enable, compiler_fn),
                test_contexts.get(name, contextlib.nullcontext),
            ]
            dct[name] = make_wrapped(fn, ctx_fns)

    cls = type(
        orig_cls.__name__ + "WithCompiledAutograd",
        (orig_cls,),
        dct,
    )
    cls.__file__ = __file__
    return cls


class WrapTestClassTests(TestCase):
    # [CLASS DOC] Unit tests for the class wrapping helper (inheritance, context re-creation,
    # cleanup on error).
    def test_wrap_preserves_inheritance_and_super(self):
        # [FUNCTION DOC] Checks the wrapped class is a subclass of the original and its
        # tearDown still calls the parent through super.
        class DummyTest(unittest.TestCase):
            # [CLASS DOC] HELPER: Minimal test case with a trivial run method and a tearDown
            # that sets a flag then calls the parent tearDown.
            def runTest(self):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def tearDown(self):
                # [FUNCTION DOC] HELPER: Sets a flag recording the call and delegates to the
                # parent tearDown.
                self.super_called = True
                super().tearDown()

        wrapped = wrap_test_class(DummyTest)
        self.assertTrue(issubclass(wrapped, DummyTest))
        test = wrapped("runTest")
        test.setUp()
        test.tearDown()
        self.assertTrue(getattr(test, "super_called", False))

    def test_wrap_recreates_contexts_for_repeated_runs(self):
        # [FUNCTION DOC] Checks the wrapper creates the contexts anew for every call: calling
        # a wrapped test twice records enter and exit twice and the test body ran twice.
        # Temporarily registers the test in the inductor xfail table and the context table,
        # and removes the entries afterward.
        class DummyTest(unittest.TestCase):
            # [CLASS DOC] HELPER: Test case with one test counting its invocations.
            def test_repeated_contexts(self):
                # [FUNCTION DOC] HELPER: Increments an instance call counter.
                self.calls = getattr(self, "calls", 0) + 1

        events = []

        @contextlib.contextmanager
        def ctx():
            # [FUNCTION DOC] HELPER: Context manager recording enter and exit events in an
            # enclosing list, exit even if the body raises.
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        # Use a CPU-capable backend so this test still exercises the wrapper on
        # builds where the inductor backend is skipped by HAS_CUDA_AND_TRITON.
        test_name = "test_repeated_contexts"
        xfail_by_backend["inductor"][test_name] = None
        test_contexts[test_name] = ctx
        try:
            wrapped = wrap_test_class(DummyTest)
        finally:
            del xfail_by_backend["inductor"][test_name]
            del test_contexts[test_name]

        test = wrapped(test_name)
        test.test_repeated_contexts()
        test.test_repeated_contexts()
        self.assertEqual(test.calls, 2)
        self.assertEqual(events, ["enter", "exit", "enter", "exit"])

    def test_wrap_closes_contexts_on_exceptions(self):
        # [FUNCTION DOC] Checks the wrapper still exits its contexts when the wrapped function
        # raises: expects the RuntimeError to propagate and the event order to be enter, body,
        # exit.
        events = []

        @contextlib.contextmanager
        def ctx():
            # [FUNCTION DOC] HELPER: Context manager recording enter and exit events in an
            # enclosing list, exit even if the body raises.
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        def fn(self):
            # [FUNCTION DOC] HELPER: Records a body event and raises RuntimeError with the
            # message boom.
            events.append("body")
            raise RuntimeError("boom")

        wrapped = make_wrapped(fn, [ctx])

        with self.assertRaisesRegex(RuntimeError, "boom"):
            wrapped(self)
        self.assertEqual(events, ["enter", "body", "exit"])


known_graph_breaks_tests = {
    "test_hook_none",  # uses assert in hook
    "test_post_accumulate_grad_hook_e2e",  # optim.Adam manually graph breaks
    "test_tensor_hooks_inplace",  # uses assert in hook
    "test_tensor_hooks_inplace_over_view",  # uses assert in hook
    "test_grad_fn_prehooks",  # uses assert in hook
    "test_grad_fn_prehooks_multiple_outputs",  # uses assert in hook
    "test_grad_fn_prehooks_remove_hooks",  # uses handle.remove() in hook
    "test_tensor_hooks_inplace_multiple_outputs",  # uses assert in hook
    "test_hooks",  # uses assert in hook
    "test_accumulate_grad_posthooks_can_observe_tensor_prehook",  # allclose
    "test_saved_tensors_hook_version_counter_not_shared",  # assertEqual
    "test_post_accumulate_grad_hook_returns_not_None",  # throws
    "test_custom_function_cycle",  # assertEqual
    "test_mark_non_differentiable_mixed",  # assertTrue
    "test_materialize_grads",  # assertEqual
    "test_return_leaf",  # assertEqual
    "test_save_none_for_backward",  # assertIsNone
    "test_saved_variables_deprecated",  # warnings.warn
    "test_autograd_node_isinstance",  # assertIsInstance
    "test_set_materialize_non_diff_grads",  # assertIsNone
    "test_backward_dict_grad_for_nontensor",  # torch/_custom_op/autograd.py in skip files
    "test_backward_dict_invalid_keys",  # torch/_custom_op/autograd.py in skip files
    "test_backward_dict_requires_keys_for_input_optional_tensors",  # torch/_custom_op/autograd.py in skip files
    "test_backward_dict_requires_keys_for_input_tensors",  # torch/_custom_op/autograd.py in skip files
    "test_backward_grads_are_tensor_or_none",  # torch/_custom_op/autograd.py in skip files
    "test_backward_impl_on_existing_op",  # torch/_custom_op/autograd.py in skip files
    "test_backward_returns_dict",  # torch/_custom_op/autograd.py in skip files
    "test_backward_tensorlist_input_requires_list_grads",  # torch/_custom_op/autograd.py in skip files
    "test_backward_tensorlist_input_requires_list_grads_none_or_Tensor",  # torch/_custom_op/autograd.py in skip files
    "test_backward_tensorlist_input_requires_list_grads_with_same_numel",  # torch/_custom_op/autograd.py in skip files
    "test_save_for_backward_inputs_are_namedtuple",  # torch/_custom_op/autograd.py in skip files
    "test_reentrant_with_leaf_variable_hook",  # reentrant .backward
    "test_reentrant_with_non_leaf_variable_hook",  # reentrant .backward
    "test_reentrant_child_error",  # reentrant .backward
    "test_deep_reentrant",  # reentrant .backward
    "test_reentrant_priority",  # reentrant .backward
    "test_simple_reentrant",  # reentrant .backward
    "test_checkpoint_detects_non_determinism",  # unpack hook in skip files
    "test_checkpoint_valid_reset_on_error",  # unpack hook in skip files
    "test_checkpointing_non_reentrant_autocast_cpu",  # unpack hook in skip files
    "test_checkpointing_non_reentrant_autocast_gpu",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_arbitrary_input_output",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_correct_grad",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_custom_function_works",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_dataparallel",  # _get_device_index in skip files
    "test_checkpointing_without_reentrant_detached_tensor_use_reentrant_True",  # reentrant .backward
    "test_checkpointing_without_reentrant_parameter_used_in_an_out",  # unpack hook in skip files
    "test_checkpointing_without_reentrant_with_context_fn",  # unpack hook in skip files
    "test_save_on_cpu_and_checkpoint",  # unpack hook in skip files
    "test_saved_tensor_hooks_custom_error_propagation",  # CustomError
    "test_access_saved_tensor_twice_without_recomputation_works",  # unpack hook in skip files
    "test_saved_tensor_hooks_extra_enter_during_bw_no_leak",  # ctx in skip files
    "test_saved_tensor_hooks_extra_exit_during_bw_no_crash",  # ctx in skip files
    "test_checkpointing",  # reentrant .backward
    "test_checkpointing_without_reentrant_input_requires_grad_False",  # reentrant .backward
    "test_checkpointing_without_reentrant_input_requires_grad_True",  # reentrant .backward
    "test_checkpointing_without_reentrant_memory_savings",  # reentrant .backward
    "test_dtensor_basic",  # torch._dynamo.exc.Unsupported: Failed to convert args/kwargs to proxy
    "test_dtensor_contiguous_dtensor_noncontiguous_local_as_tangent",  # subclass constructor
    "test_retain_grad",  # retains_grad_hooks
    "test_retain_grad_cycle",  # retains_grad_hooks
    "test_retain_grad_inplace",  # retains_grad_hooks
    "test_retain_grad_inplace_over_view",  # retains_grad_hooks
    "test_retains_grad_can_always_observe_tensor_prehook",  # retains_grad_hooks
    "test_retains_grad_inplace_multiple_outputs",  # retains_grad_hooks
    "test_hook_edge_case_when_called_with_grad",  # retains_grad_hooks
    "test_multi_grad_all_hooks",  # retains_grad_hooks
    "test_prehook_ordering",  # retains_grad_hooks
    "test_will_engine_execute_node",  # retains_grad_hooks
    "test_backward_to_node",  # retains_grad_hooks
    "test_backward_with_nonleaf_inputs",  # retains_grad_hook on non-leaf input
    "test_create_graph_and_full_backward_hook_cycle",  # _pack_with_none
    "test_full_backward_hook_double_backward",  # _pack_with_none
    "test_grad_mode_restored_reentrant",  # assertTrue
    "test_multi_grad_any_hooks",  # register_multi_grad_hook
    "test_saved_variable_packing_unpacking_did_not_save_original_with_hooks",  # register_hooks
    "test_graph_save_on_cpu",  # dynamo disabled
    "test_nested_checkpoint_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_kwargs_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_kwargs_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_non_tensor_inputs_and_outputs_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_non_tensor_inputs_and_outputs_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_reentrant_backwards_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_reentrant_backwards_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_same_graph_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_same_graph_early_stop_True",  # dynamo disable
    "test_nested_checkpoint_set_early_stop",  # dynamo disable
    "test_nested_checkpoint_set_early_stop_no_recompution_needed",  # TorchDispatchMode causes frame skip
    "test_nested_checkpoint_two_children_early_stop_False",  # dynamo disable
    "test_nested_checkpoint_two_children_early_stop_True",  # dynamo disable
    "test_custom_autograd_ac_early_stop",  # marked as skipped
    "test_checkpoint_compile_no_recompile",  # torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
    "test_checkpoint_device_context_fn",  # torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
    "test_dropout",  # dynamo disable
    "test_dropout_inductor",  # dynamo disable
    "test_function_with_kwargs",  # dynamo disable
    "test_module",  # dynamo disable
}

test_contexts = {
    "test_setitem_mask": functools.partial(
        config.patch, capture_dynamic_output_shape_ops=True
    ),
    "test_index_backward_does_not_save_tensor": functools.partial(
        config.patch, capture_dynamic_output_shape_ops=True
    ),
}

# These groups of tests aren't supported yet
xfail_re = re.compile(r"^test_(sparse|profiler|gradcheck)")

# Tests fail at different stages, we categorize them wrt to their backends
# We run only the last passing backend in this order:
# ca_eager -> eager -> aot_eager -> inductor
xfail_by_backend = {
    "ca_eager": {  # xfail
        "test_callback_propagates_errors_from_device_thread",  # fullgraph for queue_callback, but graph break for RuntimeError
        "test_reentrant_with_callbacks_both_depths",  # queue_callback
        "test_reentrant_with_callbacks_depth_0",  # queue_callback
        "test_reentrant_with_callbacks_depth_1",  # queue_callback
        "test_checkpoint_graph_execution_group",  # Attempted to call function marked as skipped
        "test_current_graph_task_execution_order",  # nodes are already freed by the time dynamo traces the lifted hook
        "test_autograd_inplace_views_cross_dtype",  # view_fn not supported by compiled autograd
        "test_post_accumulate_grad_hook_ordering",  # accuracy error
        "test_current_graph_task_id",  # autograd state already cleared once dynamo is called
        "test_custom_function_forward_mode_forward_is_no_op",  # forward AD
        "test_custom_function_forward_mode_inplace_checks",  # forward AD
        "test_custom_function_forward_mode_view_checks",  # forward AD
        "test_custom_function_forward_mode_wrong_formula",  # forward AD
        "test_node_post_hook_registered_during_unpack_hook",  # 'NoneType' object has no attribute 'register_hook'
        "test_custom_function_error",  # forward AD
        "test_custom_function_save_for_forward",  # forward AD
        "test_dont_materialize_grads",  # undefined grad
        "test_no_grad_copy",  # setting static member in lifted backward
        "test_no_grad_copy_sparse",  # setting static member in lifted backward
        "test_node_ordering_when_none_returned",  # torch._dynamo.exc.Unsupported: TypeError <built-in method clone
        "test_save_output_nr",  # output_nr grad passed as None
        # IndexError: list index out of range (NB: x.grad = y where both x and y are input tensors)
        "test_grad_nonleaf_register_hook",
        "test_backward_twice_without_saved_values",  # https://github.com/pytorch/pytorch/issues/129938
        # Category: Higher Order Gradients
        "test_default_saved_tensors_hooks_double_backward",  # wrong when pack hook returns non-leaf
        "test_saved_variable_packing_unpacking_saved_original_with_hooks",  # wrong when pack hook returns non-leaf
        "test_nested_anomaly_detect_nan",  # nested anomaly
        "test_select_sum",  # batched gradients
        "test_custom_autograd_no_early_free",  # batched gradients
        "test_grad_batched_grad",  # batched gradients
        "test_grad_dict_inputs_batched_grads",  # batched gradients
        # Uncategorized
        "test_lobpcg",  # NaNs
        "test_autograd_simple_views_python",  # gradient is None
        "test_function_returns_undefined_tensor",  # gradient is None
        "test_input_buffer_accum",  # add(sparse, dense)
        "test_return_duplicate",  # batched gradients
        "test_return_duplicate_inplace",  # batched gradients
        "test_naughty_autograd_function_stashing_ctx",  # error not raised
        "test_unrelated_inputs",  # batched gradients
        "test_nested_checkpoint_early_stop_False",  # unpack hook grad_fn semantics
        "test_nested_checkpoint_early_stop_True",  # unpack hook grad_fn semantics
        "test_nested_checkpoint_two_children_early_stop_False",  # unpack hook grad_fn semantics
        "test_nested_checkpoint_two_children_early_stop_True",  # unpack hook grad_fn semantics
        "test_dropout",  # functionalize_rng_ops not yet supported
        "test_dropout_inductor",  # functionalize_rng_ops not yet supported
        "test_function_with_kwargs",  # functionalize_rng_ops not yet supported
        "test_module",  # functionalize_rng_ops not yet supported
        "test_grad_dtype",  # AttributeError: args / Float did not match Double
    },
    "eager": {  # will be run without torch.compiling the CA graph
        "test_setup_context_when_forward_has_default_args",  # autograd.Function with class methods
        "test_accumulate_grad_tensor_reference",  # Out of bounds: frame_state_entry.stride[i] is None
        "test_custom_function_exception",  # torch.no_grad(), torch._dynamo.exc.Unsupported: missing: WITH_EXCEPT_START
        "test_to_sparse_backward",  # Out of bounds: frame_state_entry.stride[i] is None
        "test_custom_function_non_tensor_inputs_outputs",  # gradient batching rule not implemented for aten::sym_size.int
        "test_setitem",  # CopySlices accuracy error
        "test_checkpointing_without_reentrant_saved_object_identity",  # same as https://github.com/pytorch/pytorch/issues/136193
        "test_node_creation_hook_checkpoint_recompute",  # checkpoint unpack_hook is in dynamo MOD_SKIPLIST
        "test_dtensor_different_gradient_placement",  # Dynamo failed to run FX node with fake tensors
        "test_dtensor_noncontiguous_output",  # Dynamo failed to run FX node with fake tensors
        "test_dtensor_partial_placement_graph_output",  # Dynamo failed to run FX node with fake tensors
        "test_unwrap_async_collective_tensor_tangent",  # AttributeError: 'PlainTensorMeta' object has no attribute 'attrs'
        "test_graph_save_on_cpu",  # torch.save should no-op and be recorded in the graph
        "test_saving_variable_to_disk",  # torch.save should no-op and be recorded in the graph
        "test_nested_checkpoint_early_stop_False",  # AOT backward higher order gradients
        # Slow tests, these tests are close to CI timeout if we try to torch.compile them
        "test_checkpointing",
        "test_checkpointing_without_reentrant_memory_savings",
        "test_checkpointing_without_reentrant_input_requires_grad_True",
        "test_checkpointing_without_reentrant_input_requires_grad_False",
    },
    "aot_eager": {  # will be run with torch.compile(backend="eager")
        # Category: FakeTensor
        "test_wrapped_number_saved_tensors_hooks",  # Proxy tensor should carryover is_wrapped_number_ of its original
        "test_scalar_grad_mixed_device",  # Fake Tensors aren't propagating device properly for 0-dim grads
        "test_grad",  # AOT backward higher order gradients
        "test_grad_materialize_grads",  # AOT backward higher order gradients
    },
    "inductor": {},  # will be run with torch.compile(backend="aot_eager")
    # tests not present in this dict will be run with torch.compile(backend="inductor")
}

# These tests fail due to difference in semantics that we won't fix
xfail_divergence_from_eager = {
    "test_invalid_gradients",  # can't give autograd error due to inaccurate output metadata of lifted backward
    "test_autograd_node_isinstance",  # backward ctx is a fake cls and not directly a Node instance
    "test_backward_hook_relative_ordering",  # compiled autograd collects breadth first, and module backward hook not supported
    "test_checkpointing_without_reentrant_custom_function_works",  # ctx.saved_tensors are cached by CA
    "test_anomaly_mode_no_check_nan",  # different error messages
    "test_anomaly_grad_warnings",  # different error messages
    "test_anomaly_detect_nan",  # fake tensor errors on NaN
    "test_once_differentiable",  # different node name: CompiledFunctionBackward
    "test_function",  # different node name: CompiledFunctionBackward
    "test_inplace_on_view_backward",  # different node name: CompiledFunctionBackward
    "test_nested_anomaly_printstack_cleanup",  # anomaly NaN error message different
    "test_not_implemented_grad",  # Dynamo changes the types of exceptions
    "test_grad_call_compiled_backward_fn",  # different functorch error
    "test_vjp_call_compiled_backward_fn",  # different functorch error
    "test_vmap_call_compiled_backward_fn",  # different functorch error
    "test_accumulate_grad",  # always out of place add for compiled autograd
    "test_current_node",  # slightly different dispatched ops
}

skipped_tests = set()

if not HAS_CUDA_AND_TRITON:
    # Found Tesla M60 which is too old to be supported by the triton GPU compiler
    skipped_tests.add("test_type_conversions")

if IS_S390X:
    skipped_tests.add("test_deep_reentrant")

# clear_saved_tensors_on_access is incompatible with compiled autograd
skipped_tests.add("test_clear_saved_tensors_on_access")
skipped_tests.add("test_clear_saved_tensors_on_access_double_access_error")
skipped_tests.add("test_forward_traceback_preserves_exception_with_checkpoint")
skipped_tests.add("test_checkpoint_error_suggests_mark_dynamic")
skipped_tests.add("test_checkpoint_automatic_dynamic_graph_shadowing")
skipped_tests.add("test_checkpoint_automatic_dynamic_mark_dynamic_workaround")
skipped_tests.add("test_checkpoint_automatic_dynamic_lru_disabled_workaround")
# Compiled autograd does not support the higher-order gradients this test needs.
skipped_tests.add("test_batch_norm_errors_on_third_order_grad")

# Dynamo support for the curried checkpoint API is added in a later commit
skipped_tests.add(
    "test_checkpoint_curried_kwargs_do_not_collide_with_checkpoint_kwargs"
)
skipped_tests.add("test_checkpoint_zero_arg_function")
skipped_tests.add("test_checkpoint_curried_method")

# boxed_grads_call relies on eager C++ PyNode::apply, incompatible with compiled autograd
skipped_tests.add("test_custom_function_boxed_grads")
skipped_tests.add("test_custom_function_boxed_grads_multi_output")
skipped_tests.add("test_custom_function_boxed_grads_no_extra_refs")
skipped_tests.add("test_custom_function_boxed_grads_cleanup_on_error")
skipped_tests.add("test_custom_function_boxed_grads_chain")
skipped_tests.add("test_custom_function_boxed_grads_none_grads")
skipped_tests.add("test_custom_function_boxed_grads_materialize_grads")
skipped_tests.add("test_custom_function_boxed_grads_direct_apply")
skipped_tests.add("test_custom_function_boxed_grads_single_list_arg")

skipped_tests.add("test_pyobject_dispatch_normalizes_tensor_list_output")
skipped_tests.add("test_needs_input_grad_setter_roundtrip_num_inputs_2")
skipped_tests.add("test_needs_input_grad_setter_roundtrip_num_inputs_25")

# DTensor backward calls a skipped global-shape helper under compiled autograd.
skipped_tests.add("test_compile_dtensor_local_tensor_act_backward_passthrough")

test_autograd = load_test_module("test_autograd")
test_custom_ops = load_test_module("test_custom_ops")
test_higher_order_ops = load_test_module("dynamo/test_higher_order_ops")

# grad_dtype is not supported in compile (every eager test here is already
# skipIfTorchDynamo). Most of them also report the dtype they observed by
# setattr-ing on the Function class, which dynamo cannot trace once compiled
# autograd inlines the backward.
for name in dir(test_autograd.TestAutograd):
    if name.startswith("test_ctx_output_grad_dtype"):
        skipped_tests.add(name)

TestAutogradWithCompiledAutograd = wrap_test_class(test_autograd.TestAutograd)
TestNestedCheckpointWithCompiledAutograd = wrap_test_class(
    test_autograd.TestNestedCheckpoint
)
TestCustomOpWithCompiledAutograd = wrap_test_class(test_custom_ops.TestCustomOp)
HigherOrderOpTestsWithCompiledAutograd = wrap_test_class(
    test_higher_order_ops.HigherOrderOpTests
)
FuncTorchHigherOrderOpTestsWithCompiledAutograd = wrap_test_class(
    test_higher_order_ops.FuncTorchHigherOrderOpTests
)
ActivationCheckpointingTestsWithCompiledAutograd = wrap_test_class(
    test_higher_order_ops.ActivationCheckpointingTests
)

if torch.distributed.is_available() and HAS_GPU_AND_TRITON:
    test_dtensor = load_test_module("distributed/tensor/test_dtensor_compile")
    TestDTensorCompileWithCompiledAutograd = wrap_test_class(
        test_dtensor.TestDTensorCompile
    )

xfail_hops = {"local_map_hop"}
hop_test_hops_in_bwd_failures = {
    xfail("register_hook", "simple"),
}


class TestCompiledAutogradOpInfo(TestCase):
    # [CLASS DOC] Operator-database test case: instantiated per device and dtype. setUp and
    # tearDown reset compiled autograd global state.
    def setUp(self) -> None:
        # [FUNCTION DOC] Calls the parent setUp and resets compiled autograd global state.
        super().setUp()
        reset()

    def tearDown(self) -> None:
        # [FUNCTION DOC] Calls the parent tearDown and resets compiled autograd global state.
        super().tearDown()
        reset()

    @ops(
        list(filter(lambda op: op.name not in xfail_hops, hop_db)),
        allowed_dtypes=(torch.float,),
    )
    @skipOps(hop_test_hops_in_bwd_failures)
    def test_hops_in_bwd(self, device, dtype, op):
        # [FUNCTION DOC] For each higher-order-op database entry (excluding listed ones, float
        # dtype only, with listed known failures skipped), checks that calling the op inside
        # the backward of a custom Function gives the same result under eager and under
        # compiled autograd with an aot_eager compiler. Sample inputs are deep copied for the
        # compiled run and the random seed is reset before each run; each run must record
        # exactly one op output.
        def create_bwd_fn_closure(op_args, op_kwargs):
            # [FUNCTION DOC] HELPER: Builds a function applying a custom Function whose
            # backward calls the op on the given arguments and records its output in a list;
            # returns the function and the list.
            op_out_ref = []

            class Foo(torch.autograd.Function):
                # [CLASS DOC] HELPER: Identity custom Function whose backward invokes the
                # operator under test, stores the result in an enclosing list and returns the
                # gradient unchanged.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Identity forward.
                    return x

                @staticmethod
                def backward(ctx, grad):
                    # [FUNCTION DOC] HELPER: Calls the operator with the captured arguments,
                    # appends the output to the list and returns the gradient.
                    out = op.op(*op_args, **op_kwargs)
                    op_out_ref.append(out)
                    return grad

            def fn(x):
                # [FUNCTION DOC] HELPER: Applies the custom Function and returns the sum.
                return Foo.apply(x).sum()

            return fn, op_out_ref

        # Note: requires_grad=False because aot dispatch is already covered elsewhere
        for inp in op.sample_inputs(device, dtype, requires_grad=False):
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            eager_args = (*input, *inp.args)
            eager_kwargs = inp.kwargs
            compiled_args = deepcopy(eager_args)
            compiled_kwargs = deepcopy(eager_kwargs)

            # 1. Run eager
            torch.manual_seed(123)
            dummy = torch.randn(2, 2, dtype=dtype, device=device, requires_grad=True)
            fn, op_out_ref = create_bwd_fn_closure(eager_args, eager_kwargs)
            fn(dummy).backward()
            self.assertEqual(len(op_out_ref), 1)
            expected = op_out_ref[0]

            # 2. Run under CA
            torch.manual_seed(123)
            dummy = torch.randn(2, 2, dtype=dtype, device=device, requires_grad=True)
            fn, op_out_ref = create_bwd_fn_closure(compiled_args, compiled_kwargs)
            with compiled_autograd._enable(make_compiler_fn(backend="aot_eager")):
                fn(dummy).backward()
            self.assertEqual(len(op_out_ref), 1)
            actual = op_out_ref[0]

            self.assertEqual(expected, actual)


instantiate_device_type_tests(TestCompiledAutogradOpInfo, globals(), allow_xpu=True)
instantiate_parametrized_tests(TestCompiledAutograd)

if __name__ == "__main__":
    if HAS_CPU:
        run_tests(needs="filelock")
