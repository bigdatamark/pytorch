# Owner(s): ["module: ProxyTensor"]
# ruff: noqa: F841
# [MODULE DOC] Tests for make_fx and the proxy tensor tracing machinery in
# torch.fx.experimental.proxy_tensor. Provides generic trace tests run in real, fake and
# symbolic tracing modes, tests of fake-tensor and symbolic-shape tracing, decomposition and
# functionalization interplay, plus OpInfo-driven make_fx tests (instantiated per device) with
# large xfail/skip tables. Import-time side effects: tries to import torchvision (warns if
# missing), defines the aten alias and HAS_CUDA flag. Several tests compare generated graph code
# against inline golden strings.

from torch.testing._internal.common_utils import (
    skipIfTorchDynamo,
    TestCase,
    run_tests,
    xfailIfNoAcceleratorTriton,
)
import torch
import torch._dynamo
import unittest
import warnings
import operator
from collections.abc import Iterable
from torch.nn.utils import stateless
from torch.testing._internal.common_device_type import instantiate_device_type_tests, skipOps, skip, xfail
from torch.testing._internal.common_methods_invocations import op_db
from torch._subclasses.fake_tensor import DynamicOutputShapeException, DataDependentOutputException, FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._decomp import decomposition_table
from torch.fx.experimental.symbolic_shapes import (
    eval_guards, bind_symbols, fx_placeholder_vals, fx_placeholder_targets,
    guard_int, GuardOnDataDependentSymNode
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.common_device_type import ops
import torch.testing._internal.optests as optests
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import (
    DecompositionInterpreter,
    get_isolated_graphmodule,
    make_fx,
    ProxyTorchDispatchMode,
    PythonKeyTracer,
)
from torch.utils._pytree import tree_map
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch import nn
import torch._functorch.config
import re

import functools
import itertools
from pathlib import Path

aten = torch.ops.aten

HAS_CUDA = torch.cuda.is_available()


def strip_end(s, suffix):
    # [FUNCTION DOC] HELPER: Returns s with the trailing suffix removed when suffix is non-empty
    # and s ends with it; otherwise returns s unchanged.
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s


def show_guards(gm):
    # [FUNCTION DOC] HELPER: Renders the shape-environment guards of a traced graph module as
    # one string. Placeholder names come from fx_placeholder_targets with a trailing _1
    # stripped, guards are produced with simplification from the placeholder values, and are
    # joined with newlines.
    names = [strip_end(n, "_1") for n in fx_placeholder_targets(gm)]
    return "\n".join(
        gm.shape_env.produce_guards(fx_placeholder_vals(gm), names, _simplified=True, input_contexts=None)
    )


def process_failures():
    # [FUNCTION DOC] HELPER: Maintenance utility, not a test. Reads the local file
    # pytorch_failures (one pytest failure line per row), extracts the op name and failure
    # reason from each line, maps normalized op names back to op_db entries, and prints a
    # Python-style xfail dictionary to stdout. Raises if a line does not match the expected
    # pattern or an op is not found.
    """
    Takes file containing failures like

    FAILED test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive___getitem___cpu_float32 - RuntimeError: aten.size.default - couldn't find symbolic meta function/decomposition  # noqa: B950

    and processes them into a list of opinfo xfails
    """
    failures = [i.strip() for i in Path('pytest_failures').read_text().splitlines()]

    def process_failure_string(s, matcher):
        # [FUNCTION DOC] HELPER: Applies a regex search to s and returns the tuple of captured
        # groups. Fails with an attribute error if the pattern does not match.
        out = re.search(matcher, s)
        return out.groups()

    SYMBOLIC_TRACE_MATCH = r'exhaustive_(.*)_cpu.*: (.*)'
    failures = [process_failure_string(s, SYMBOLIC_TRACE_MATCH) for s in failures]

    def create_normalized_name(op):
        # [FUNCTION DOC] HELPER: Builds a lookup key for an OpInfo: its name, followed by a dot
        # and the variant test name when non-empty, with all dots replaced by underscores.
        if op.variant_test_name == '':
            s = op.name
        else:
            s = f"{op.name}.{op.variant_test_name}"
        return s.replace('.', '_')

    remap_opinfo = {create_normalized_name(op): (op.name, op.variant_test_name) for op in op_db}

    print("symbolic_tensor_failures = {")
    for failure, reason in failures:
        print(f"    xfail{remap_opinfo[failure]},  # {reason}")
    print("}")


USE_TORCHVISION = False
try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try "
                  "to install it with commands from pytorch.org, post-fixed with "
                  "`--no-deps` to avoid overwriting the pytorch installation",
                  UserWarning)


def _create_new_input(x):
    # [FUNCTION DOC] HELPER: Makes a fresh input resembling x for re-running a traced function.
    # Non tensors are returned as is; non-float tensors are shifted by one; float leaf tensors
    # become random tensors keeping requires_grad; other float tensors become plain random
    # tensors.
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype != torch.float:
        return x + 1
    if x.is_leaf:
        return torch.rand_like(x, requires_grad=x.requires_grad)
    else:
        return torch.rand_like(x)

"""
Delays a cos being executed on the unwraptensor until its used. Simulates a CommTensor used
"""
class UnwrapTensor(torch.Tensor):
    # [CLASS DOC] Tensor wrapper subclass that holds an inner tensor and applies a cosine to it
    # only when unwrapped inside a dispatch call, simulating a delayed communication tensor.
    # Stores the inner tensor in an attribute; torch function handling is disabled so only
    # dispatch is intercepted.
    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        # [FUNCTION DOC] Static constructor. Creates a wrapper subclass instance with the same
        # size, dtype, device, layout and requires_grad as the given tensor, stores the tensor
        # on it and returns it.
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            dtype=tensor.dtype,
            device=tensor.device,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
        )
        r._tensor = tensor
        return r

    def __repr__(self):
        # [FUNCTION DOC] Returns a string showing the class name around the repr of the wrapped
        # tensor.
        # TODO: consider all_gather the local tensors for better debugging
        return f"UnwrapTensor({self._tensor})"

    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Dispatch hook. Replaces every UnwrapTensor in args and kwargs (via
        # tree_map) with the cosine of its inner tensor, then calls the op on the unwrapped
        # arguments and returns the result.
        def unwrap(e):
            # [FUNCTION DOC] HELPER: Returns the cosine of the inner tensor if e is an
            # UnwrapTensor, else e.
            ret = e
            if isinstance(e, UnwrapTensor):
                ret = e._tensor.cos()

            return ret

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        return func(*args, **kwargs)

class TestGenericProxyTensor(TestCase):
    # [CLASS DOC] Base test case for make_fx tracing behavior; it relies on a tracing_mode
    # attribute provided by the real, fake and symbolic subclasses defined later. Checks pre-
    # dispatch tracing, isolated graph modules, decomposition tables, constants, mutation,
    # subclasses and many bug repros.
    # WARNING: if any of your inputs are index tensors, DO NOT use this
    # function
    def _test(self, f, inps):
        # [FUNCTION DOC] HELPER: Traces f with make_fx in the class tracing mode on the given
        # inputs, then builds fresh inputs, runs both the traced graph and the original
        # function, and asserts equal results. The inputs must not contain index tensors since
        # fresh inputs are random floats.
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(*inps)
        new_inps = tree_map(_create_new_input, inps)
        r1 = fx_f(*new_inps)
        r2 = f(*new_inps)
        self.assertEqual(r1, r2)

    def test_pre_dispatch_mode_stack(self):
        # [FUNCTION DOC] Checks that pre-dispatch tracing keeps matmul undecomposed in the graph
        # and that factory calls such as ones do not appear as traced nodes of the input;
        # compares the generated code to a golden string. Also runs the function under the
        # python dispatcher first to check caches are cleared by pre-dispatch tracing.
        def f(a):
            # [FUNCTION DOC] HELPER: Multiplies the input by a freshly created matrix of ones.
            b = torch.ones(4, 4)
            return torch.matmul(a, b)
        # We expect to see matmul in the trace - it should NOT be decomposed into mm.
        # Also, torch.ones() doesn't show up in the trace.
        # This is annoying but expected: ones() never dispatches to the Autograd dispatch key,
        # so our mode never sees it - it goes directly to the BackendSelect key.
        inp = torch.ones(4, 4)
        # Test that make_fx(pre_dispatch=True) clears caches properly.
        from torch._dispatch.python import enable_python_dispatcher
        with enable_python_dispatcher():
            out1 = f(inp)
        fx_g = make_fx(f, pre_dispatch=True)(inp)
        self.assertExpectedInline(fx_g.code.strip(), """\
def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 4], device = device(type='cpu'), pin_memory = False)
    matmul = torch.ops.aten.matmul.default(a_1, ones);  a_1 = ones = None
    return matmul""")

    def test_pre_dispatch_linear(self):
        # [FUNCTION DOC] Pre-dispatch tracing of a linear op: traced graph and eager function
        # give equal results.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Applies the functional linear op to the three arguments.
            return torch.nn.functional.linear(a, b, c)
        a = torch.ones(4, 4)
        b = torch.ones(4, 4)
        c = torch.ones(4)
        fx_g = make_fx(f, pre_dispatch=True)(a, b, c)
        out1 = f(a, b, c)
        out2 = fx_g(a, b, c)
        self.assertEqual(out1, out2)

    def test_pre_dispatch_no_grad(self):
        # [FUNCTION DOC] Pre-dispatch tracing of code that toggles grad mode off and on inside
        # the function; checks outputs match eager and that gradients from backward match for
        # both.
        def f(a):
            # [FUNCTION DOC] HELPER: Computes sin, then cos with grad disabled, re-enables grad
            # and returns the sum of sin and the sine of the cos result. Mutates global grad
            # mode and leaves it enabled.
            b = a.sin()
            torch.set_grad_enabled(False)
            c = b.cos()
            torch.set_grad_enabled(True)
            return b + c.sin()
        a1 = torch.randn(4, requires_grad=True)
        a2 = a1.detach().clone().requires_grad_(True)
        a_tmp = a1.detach().clone().requires_grad_(True)
        fx_g = make_fx(f, pre_dispatch=True)(a_tmp)
        out1 = f(a1)
        out2 = fx_g(a2)
        self.assertEqual(out1, out2)
        out1.sum().backward()
        out2.sum().backward()
        self.assertEqual(a1.grad, a2.grad)

    def test_make_fx_simple(self):
        # [FUNCTION DOC] Smoke test: traces a sine function and compares with eager through the
        # _test helper.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return torch.sin(x)
        self._test(f, (torch.randn(3),))

    def test_scalar_device(self, device='cpu'):
        # [FUNCTION DOC] Traces an addition of a tensor on the given device (default cpu) and a
        # scalar tensor.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b
        self._test(f, [torch.randn(3, device=device), torch.tensor(5)])

    def test_isolated_graphmodule(self):
        # [FUNCTION DOC] Tests get_isolated_graphmodule: nested capture must not leak into the
        # outer make_fx graph. Covers factory functions inside the isolated function, nested
        # make_fx calls, tracing of a produced graph module forward, interaction with logging
        # tensor mode (works) and with a logging tensor subclass (currently leaks a sigmoid into
        # the outer trace).
        def is_any_sum(gm):
            # [FUNCTION DOC] HELPER: Returns whether the graph of gm has a node targeting aten
            # sum.
            return any(node.target == torch.ops.aten.sum.default for node in gm.graph.nodes)

        def is_any_digamma(gm):
            # [FUNCTION DOC] HELPER: Returns whether the graph of gm has a node targeting aten
            # digamma.
            return any(node.target == torch.ops.aten.digamma.default for node in gm.graph.nodes)

        def is_any_sigmoid(gm):
            # [FUNCTION DOC] HELPER: Returns whether the graph of gm has a node targeting aten
            # sigmoid.
            return any(node.target == torch.ops.aten.sigmoid.default for node in gm.graph.nodes)

        def inner(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x.
            return torch.sum(x)

        def f(x):
            # [FUNCTION DOC] HELPER: Captures the sum function in an isolated graph module,
            # asserts it holds a sum node, and returns x plus random noise.
            gm = get_isolated_graphmodule(inner, (x,), {})
            self.assertTrue(is_any_sum(gm))
            return x + torch.randn(x.shape)

        # get_isolated_graphmodule uses make_fx internally that shouldn't be traced
        # by the outer make_fx call
        traced = make_fx(f)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))

        # When factory functions are used, they should not be traced
        # by the outer make_fx call
        def inner_with_factory():
            # [FUNCTION DOC] HELPER: Creates a scalar tensor, adds 2 in place, and returns the
            # sum of a 10 by 10 tensor filled with it; used to exercise factory functions in
            # isolation.
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((10, 10), val).sum()

        def f1(x):
            # [FUNCTION DOC] HELPER: Captures the factory helper in an isolated graph module,
            # asserts it contains a sum node, and returns the sigmoid of x.
            gm = get_isolated_graphmodule(inner_with_factory, (), {})
            self.assertTrue(is_any_sum(gm))
            return torch.sigmoid(x)

        def f2(x):
            # [FUNCTION DOC] HELPER: Isolates f1 in a graph module, asserts it has no sum but
            # has sigmoid, and returns the digamma of x.
            gm = get_isolated_graphmodule(f1, (x,), {})
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify nested make_fx calls don't make factory functions to be leaked
        # into the outer graph. Verify that `make_fx`` itself does not leak its execution.
        def f2(x):
            # [FUNCTION DOC] HELPER: Redefinition of f2 that runs make_fx on f1 nested, asserts
            # the result has no sum but has sigmoid, and returns the digamma of x.
            gm = make_fx(f1)(x)
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify that the `forward`` function of a graph module produced as a
        # side effect of an interior `make_fx` is still traced
        def f3(x):
            # [FUNCTION DOC] HELPER: Runs make_fx on f1, asserts no sum and a sigmoid in the
            # result, and returns the digamma of the output of calling that graph module, so its
            # forward is traced.
            gm = make_fx(f1)(x)
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            # `gm.forward`` is still traced
            return torch.digamma(gm(x))

        traced = make_fx(f3)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertTrue(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify interaction with non-ProxyTensor modes
        from torch.testing._internal.logging_tensor import LoggingTensorMode

        def f1_logging(x):
            # [FUNCTION DOC] HELPER: Like f1 but the isolated capture happens inside a logging
            # tensor mode.
            with LoggingTensorMode():
                gm = get_isolated_graphmodule(inner_with_factory, (), {})
            self.assertTrue(is_any_sum(gm))
            return torch.sigmoid(x)

        def f2_logging(x):
            # [FUNCTION DOC] HELPER: Isolates f1_logging under two nested logging modes, asserts
            # no sum but a sigmoid, and returns the digamma of x.
            with LoggingTensorMode(), LoggingTensorMode():
                gm = get_isolated_graphmodule(f1_logging, (x,), {})
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2_logging)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify interaction with another tensor subclass
        # This case currently doesn't work and should raise an error
        # See: https://github.com/pytorch/pytorch/pull/81764#issuecomment-1200472068
        from torch.testing._internal.logging_tensor import LoggingTensor

        def f1_logging_tensor(x):
            # [FUNCTION DOC] HELPER: Isolates the factory helper, asserts it holds a sum,
            # returns sigmoid of x.
            gm = get_isolated_graphmodule(inner_with_factory, (), {})
            self.assertTrue(is_any_sum(gm))
            return torch.sigmoid(x)

        def f2_logging_tensor(x):
            # [FUNCTION DOC] HELPER: Wraps x in a logging tensor, isolates f1_logging_tensor,
            # asserts no sum but a sigmoid in the isolated graph, and returns the digamma of x.
            x = LoggingTensor(x)
            gm = get_isolated_graphmodule(f1_logging_tensor, (x,), {})
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2_logging_tensor)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))  # this fails, sigmoid is traced with LoggingTensor
        self.assertTrue(is_any_digamma(traced))

    # See https://github.com/pytorch/pytorch/issues/97541
    def test_empty_like_doesnt_burn_in_defaults(self):
        # [FUNCTION DOC] Regression test for issue 97541: tracing empty_like must not bake in
        # default arguments; compares generated code to a golden string.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns an uninitialized tensor like x.
            return torch.empty_like(x)
        out = make_fx(f)(torch.randn(3))
        self.assertExpectedInline(out.code.strip(), """\
def forward(self, x_1):
    empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False);  x_1 = None
    return empty_like""")

    def test_proxy_tensor_mode_with_decomp_table_preserves_proxy(self):
        # [FUNCTION DOC] Checks that when a decomposition (for new_zeros) calls zeros, the proxy
        # mode is still active re-entrantly so that the zeros call shows up in the traced graph;
        # compares with golden code.
        def f(x):
            # [FUNCTION DOC] HELPER: Creates new zeros shaped like x, copies x into them and
            # returns the result.
            y = x.new_zeros(x.size())
            y.copy_(x)
            return y

        def _new_zeros_decomp(inp, size, dtype=None, layout=None, device=None, pin_memory=None):
            # [FUNCTION DOC] HELPER: Decomposition for new_zeros: returns a zeros tensor of the
            # given size with the dtype and device of the input; other arguments are ignored.
            return torch.zeros(size, dtype=inp.dtype, device=inp.device)

        factory_func_decomp = {torch.ops.aten.new_zeros.default: _new_zeros_decomp}

        # When new_zeros() decomposes into torch.zero(), we expect ProxyTensorMode
        # to still be (re-entrantly) enabled, so that the `torch.zero()` call
        # returns a ProxyTensor.
        out = make_fx(f, decomposition_table=factory_func_decomp)(torch.ones(2))
        self.assertExpectedInline(out.code, """\



def forward(self, x_1):
    zeros = torch.ops.aten.zeros.default([2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    copy_ = torch.ops.aten.copy_.default(zeros, x_1);  zeros = x_1 = None
    return copy_
    """)

    def test_proxy_tensor_mode_tracks_nested_decomp_tables(self):
        # [FUNCTION DOC] Verifies that entering enable_decompositions contexts on a
        # ProxyTorchDispatchMode swaps its decomposition table and restores the previous one on
        # exit, at two nesting levels.
        sin_table = {torch.ops.aten.sin.default: lambda x: x}
        cos_table = {torch.ops.aten.cos.default: lambda x: x}
        tan_table = {torch.ops.aten.tan.default: lambda x: x}

        mode = ProxyTorchDispatchMode(
            PythonKeyTracer(),
            tracing_mode="real",
            decomposition_table=sin_table,
        )

        self.assertIs(mode.decomposition_table, sin_table)
        with mode.enable_decompositions(cos_table):
            self.assertIs(mode.decomposition_table, cos_table)
            with mode.enable_decompositions(tan_table):
                self.assertIs(mode.decomposition_table, tan_table)
            self.assertIs(mode.decomposition_table, cos_table)
        self.assertIs(mode.decomposition_table, sin_table)

    def test_make_fx_reentrant_dispatch(self):
        # [FUNCTION DOC] Traces the norm op with a decomposition table and checks no square or
        # norm op targets remain in the graph, i.e. the decomposition was applied.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls the aten norm Scalar overload with p of 2.
            return torch.ops.aten.norm.Scalar(x, 2.0)

        def norm_decomp(x, p=2.0):
            # [FUNCTION DOC] HELPER: Decomposition of norm: raises RuntimeError if p is not 2,
            # otherwise returns the square root of the sum of squares of x.
            if p != 2.0:
                raise RuntimeError("can't handle with p != 2")
            return torch.sqrt(torch.sum(torch.square(x)))

        decomp = {torch.ops.aten.norm.Scalar: norm_decomp}

        traced = make_fx(f, decomposition_table=decomp, tracing_mode=self.tracing_mode)(torch.rand(3))

        for n in traced.graph.nodes:
            self.assertTrue("square" not in str(n.target))
            self.assertTrue("norm" not in str(n.target))

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_resnet18_backward_trace(self):
        # [FUNCTION DOC] Traces forward and backward of a torchvision resnet18 using
        # functional_call with parameters and buffers passed in; skipped without torchvision.
        # Compares with eager results.
        mod = torchvision.models.resnet18()

        # An old version of this test called the module directly.  This works
        # for tracing_mode == "real", but for fake tensors, we also have to
        # ensure that the parameters and buffers get wrapped in fake tensors
        # because free fake tensors are not supported.  Fortunately functional_call
        # does precisely this for us.
        def f(x, params, buffers):
            # [FUNCTION DOC] HELPER: Clears parameter grads, computes the summed output of the
            # module through functional_call, runs backward (mutating .grad) and returns the
            # list of parameter gradients.
            for p in params.values():
                p.grad = None
            loss = torch.func.functional_call(mod, {**params, **buffers}, (x,)).sum()
            # I could have done this with the functional API, but there is
            # plenty of exercising this; I want to show mutating API still
            # works
            loss.backward()
            return [p.grad for p in params.values()]

        inp = torch.randn(3, 3, 250, 250)
        self._test(f, [inp, dict(mod.named_parameters()), dict(mod.named_buffers())])

    def test_varargs(self):
        # [FUNCTION DOC] Traces a function taking variable positional arguments and returning
        # their sum.
        def f(*args):
            # [FUNCTION DOC] HELPER: Returns the Python sum of the positional arguments.
            return sum(args)

        self._test(f, [torch.randn(2), torch.randn(2)])

    def test_proxy_tensor(self):
        # [FUNCTION DOC] Traces functions computing gradients through autograd (grad call and
        # backward) and compares with eager.
        def f_grad(x):
            # [FUNCTION DOC] HELPER: Returns the autograd gradient of the summed double cosine
            # of x w.r.t. x.
            val = x.cos().cos().sum()
            return torch.autograd.grad(val, x)

        def f_backward(x):
            # [FUNCTION DOC] HELPER: Runs backward on the summed double cosine of x and returns
            # x.grad.
            val = x.cos().cos().sum()
            val.backward()
            return x.grad

        for f in [f_grad, f_backward]:
            self._test(f, [torch.randn(3, requires_grad=True)])

    def test_pickle_issue89626(self):
        # [FUNCTION DOC] Regression test for issue 89626: tracing a lambda and then pickling the
        # input tensor must succeed.
        import pickle
        x = torch.randn(2)
        make_fx(lambda x: x * 2, tracing_mode=self.tracing_mode)(x)
        pickle.dumps(x)

    def test_inplace_metadata(self):
        # [FUNCTION DOC] Traces a function that clones x and unsqueezes it in place; the
        # metadata change must yield a trailing dimension of one in eager and traced versions.
        def f(x):
            # [FUNCTION DOC] HELPER: Clones x, unsqueezes the last dim in place, raises
            # AssertionError if the last dim is not 1, and returns the result.
            x = x.clone()
            x.unsqueeze_(-1)
            if x.shape[-1] != 1:
                raise AssertionError(f"expected x.shape[-1] == 1, got {x.shape[-1]}")
            return x

        self._test(f, [torch.randn(5)])

    def test_mode_tracing_factory_function(self):
        # [FUNCTION DOC] Checks that a random factory call inside the traced function appears in
        # the graph as a randn node in the default mode.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus a random tensor of its shape.
            return x + torch.randn(x.shape)

        # default behavior should trace factory functions
        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3))
        self.assertTrue(
            any(
                node.target == aten.randn.default
                for node in traced.graph.nodes
            )
        )

    def test_pre_dispatch_functionalization(self):
        # [FUNCTION DOC] Pre-dispatch tracing with a FunctionalTensorMode (export on): matmul
        # stays undecomposed and the in place multiply is functionalized; compares generated
        # code to golden text.
        def f(x):
            # [FUNCTION DOC] HELPER: Under the functional mode, converts x to functional,
            # computes matmul, adds x, multiplies in place by 5 and returns the unwrapped inner
            # tensor.
            a = FunctionalTensorMode(pre_dispatch=True, export=True)
            with a:
                x_unwrapped = FunctionalTensor.to_functional(x)
                y = torch.matmul(x_unwrapped, x_unwrapped)
                y = y + x_unwrapped
                y.mul_(5)
                y_unwrapped = torch._from_functional_tensor(y.elem)
                return y_unwrapped

        from torch._dispatch.python import enable_python_dispatcher

        with enable_python_dispatcher():
            inp = torch.randn(4, 4)
            gm = make_fx(f, pre_dispatch=True)(inp)

        # TODO actually not decompose
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1):
    matmul = torch.ops.aten.matmul.default(x_1, x_1)
    add = torch.ops.aten.add.Tensor(matmul, x_1);  matmul = x_1 = None
    mul = torch.ops.aten.mul.Tensor(add, 5);  add = None
    return mul""")

    def test_pre_dispatch_functionalization_view_op(self):
        # [FUNCTION DOC] As above but with a transpose and a view in the function; checks the
        # golden graph retains the transpose and view ops.
        def f(x):
            # [FUNCTION DOC] HELPER: Under the functional mode, computes matmul, adds the
            # transposed input, views the result as 2 by 8 and returns the unwrapped tensor.
            a = FunctionalTensorMode(pre_dispatch=True, export=True)
            with a:
                x_unwrapped = FunctionalTensor.to_functional(x)
                y = torch.matmul(x_unwrapped, x_unwrapped)
                x_unwrapped = x_unwrapped.transpose(1, 0)
                y = y + x_unwrapped
                y = y.view(2, 8)
                y_unwrapped = torch._from_functional_tensor(y.elem)
                return y_unwrapped

        from torch._dispatch.python import enable_python_dispatcher

        with enable_python_dispatcher():
            inp = torch.randn(4, 4)
            gm = make_fx(f, pre_dispatch=True)(inp)

        # TODO actually not decompose
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1):
    matmul = torch.ops.aten.matmul.default(x_1, x_1)
    transpose = torch.ops.aten.transpose.int(x_1, 1, 0);  x_1 = None
    add = torch.ops.aten.add.Tensor(matmul, transpose);  matmul = transpose = None
    view = torch.ops.aten.view.default(add, [2, 8]);  add = None
    return view""")

    def test_val_metadata_mutation(self):
        # [FUNCTION DOC] Checks the recorded val metadata shapes on graph nodes of a traced
        # clone plus unsqueeze: shapes are (3), (3), (1 by 3).
        def f(x):
            # [FUNCTION DOC] HELPER: Clones x and unsqueezes it in place at dim 0, returning it.
            y = x.clone()
            y.unsqueeze_(0)
            return y

        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3, requires_grad=True))
        self.assertEqual([
            tuple(node.meta['val'].shape)
            for node in traced.graph.nodes
            if 'val' in node.meta
        ], [(3,), (3,), (1, 3)])

    def test_make_fx_overloads(self):
        # [FUNCTION DOC] Checks every call_function node in the traced graph targets an
        # OpOverload.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns cos of x plus a random tensor of its shape.
            return x.cos() + torch.randn(x.shape)

        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3))

        self.assertTrue(all(isinstance(node.target, torch._ops.OpOverload)
                            for node in traced.graph.nodes if node.op == 'call_function'))

    def test_tensor_constants(self):
        # [FUNCTION DOC] Traces a function building a constant tensor of infinity and filling a
        # 100 by 100 tensor with it; compares with eager.
        def f():
            # [FUNCTION DOC] HELPER: Returns a 100 by 100 tensor filled with an infinity scalar
            # tensor.
            val = torch.tensor(float('inf'))
            return torch.full((100, 100), val)

        self._test(f, [])

    def test_allclose(self):
        # [FUNCTION DOC] Tracing allclose on zero tensors needs a data-dependent value: expects
        # a DataDependentOutputException for fake or symbolic modes and a RuntimeError
        # mentioning data-dependence in real mode.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns allclose of a and b.
            return torch.allclose(a, b)

        def test_f():
            # [FUNCTION DOC] HELPER: Attempts to trace f on two zero tensors in the class
            # tracing mode.
            make_fx(f, tracing_mode=self.tracing_mode)(
                torch.zeros(3), torch.zeros(3)
            )

        if self.tracing_mode != "real":
            self.assertRaises(DataDependentOutputException, test_f)
        else:
            self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_constant_proxy_tensor_mut(self):
        # [FUNCTION DOC] Traces a function that mutates a constant tensor; the traced graph,
        # when run twice, must keep matching eager, i.e. shared constant state is not mutated.
        def f():
            # [FUNCTION DOC] HELPER: Makes a scalar tensor, adds 2 in place and returns a 100 by
            # 100 tensor filled with it.
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((100, 100), val)

        g = make_fx(f, tracing_mode=self.tracing_mode)()
        self.assertEqual(g(), f())
        # In case we mutated shared state in the g graph!
        self.assertEqual(g(), f())

    def test_constant_unbind(self):
        # [FUNCTION DOC] Traces unbinding a constant tensor and taking the item; graph output
        # must match eager.
        def f():
            # [FUNCTION DOC] HELPER: Unbinds a one-element constant tensor and returns its item.
            val = torch.tensor([2])
            r, = torch.unbind(val, 0)
            return r.item()

        g = make_fx(f, tracing_mode=self.tracing_mode)()
        self.assertEqual(g(), f())

    def test_constant_blowup(self):
        # [FUNCTION DOC] Checks that tracing a function that repeats a constant 1000 times and
        # reads a value from it raises a RuntimeError about data-dependence.
        def f():
            # [FUNCTION DOC] HELPER: Repeats a constant tensor 1000 times and returns whether
            # the item sum is 2.
            val = torch.tensor([2])
            blowup = val.repeat(1000)
            return bool(blowup.sum().item() == 2)

        def test_f():
            # [FUNCTION DOC] HELPER: Attempts to trace f with no inputs in the class tracing
            # mode.
            make_fx(f, tracing_mode=self.tracing_mode)()

        self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_constant_random(self):
        # [FUNCTION DOC] Checks that tracing a function that fills a constant with random normal
        # values then reads its item raises a RuntimeError about data-dependence.
        def f():
            # [FUNCTION DOC] HELPER: Fills a constant tensor with normal samples in place and
            # returns whether its item equals a fixed number.
            val = torch.tensor([2.0])
            val.normal_()
            return bool(val.item() == 2.1)

        def test_f():
            # [FUNCTION DOC] HELPER: Attempts to trace f with no inputs in the class tracing
            # mode.
            make_fx(f, tracing_mode=self.tracing_mode)()

        self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_decomposition_interpreter(self):
        # [FUNCTION DOC] Traces silu with no decomposition table and checks a silu node exists,
        # then replays the graph through DecompositionInterpreter using only the silu
        # decomposition into a new graph, checks no silu node remains in the new module, and
        # asserts both modules give equal outputs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the silu activation of x.
            return torch.nn.functional.silu(x)

        x = torch.rand((4, 4))
        fx_module = make_fx(fn, tracing_mode=self.tracing_mode, decomposition_table=None)(x)

        found_silu = False
        for n in fx_module.graph.nodes:
            if n.target == torch.ops.aten.silu or n.target == torch.ops.aten.silu.default:
                found_silu = True

        self.assertTrue(found_silu)

        new_graph = torch.fx.Graph()
        silu_decomp_table = {torch.ops.aten.silu.default: decomposition_table[torch.ops.aten.silu.default]}
        DecompositionInterpreter(
            fx_module,
            new_graph=new_graph,
            decomposition_table=silu_decomp_table,
        ).run(x)

        decomposed_module = torch.fx.GraphModule(fx_module, new_graph)

        for n in decomposed_module.graph.nodes:
            self.assertTrue(n.target != torch.ops.aten.silu)
            self.assertTrue(n.target != torch.ops.aten.silu.default)

        self.assertEqual(fx_module(x), decomposed_module(x))

    def test_make_fx_model_fwd_bwd(self):
        # [FUNCTION DOC] Traces forward plus backward of a linear-relu module through
        # functional_call and checks that each of the two returned parameters matches one of the
        # eager ones (order may differ).
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module with one 5 by 5 linear layer whose forward
            # applies the layer then relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module and its linear layer attribute.
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of the linear layer applied to x.
                return self.linear(x).relu()

        model = Foo()

        def f(x, params):
            # [FUNCTION DOC] HELPER: Runs the model with functional_call and the given
            # parameters, backpropagates the summed output (populating grads) and returns the
            # list of parameter values.
            out = torch.func.functional_call(model, params, x).sum()
            out.backward()
            return list(params.values())
        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(input, params)
        # fx may change the order of parameters in list, so using set() to compare
        self.assertTrue(
            torch.allclose(fx_f(input, params)[0], f(input, params)[0])
            or
            torch.allclose(fx_f(input, params)[0], f(input, params)[1])
        )
        self.assertTrue(
            torch.allclose(fx_f(input, params)[1], f(input, params)[0])
            or
            torch.allclose(fx_f(input, params)[1], f(input, params)[1])
        )

    def test_make_fx_model_double_param(self):
        # [FUNCTION DOC] Traces a module that applies the same layer norm twice and checks the
        # module weight is a real tensor inside forward and that the traced graph has exactly
        # two distinct call targets.
        class Emformer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single layer norm over the given input dimension
            # (default 256).
            def __init__(
                self,
                input_dim: int = 256,
            ) -> None:
                # [FUNCTION DOC] HELPER: Stores a layer norm module sized by input_dim.
                super().__init__()

                self.layer_norm = torch.nn.LayerNorm(input_dim)

            def forward(mod_self, x):
                # [FUNCTION DOC] HELPER: Asserts (through the enclosing test case) that the
                # layer norm weight is a tensor before and after the first application, then
                # applies the layer norm twice and returns it.
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                y = mod_self.layer_norm(x)
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                z = mod_self.layer_norm(y)
                return z


        gm = make_fx(Emformer())(torch.randn(16, 1, 256))
        ops = {n.target for n in gm.graph.nodes if n.op == 'call_function'}
        self.assertEqual(len(ops), 2)


    def test_make_fx_model_fwd_bwd_wgtupdate(self):
        # [FUNCTION DOC] Traces forward, backward and a manual weight update (parameter minus a
        # small multiple of its grad) of a linear-relu module; compares each result to eager
        # within a loose tolerance, allowing either parameter order.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module with one 5 by 5 linear layer whose forward
            # applies the layer then relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module and its linear layer attribute.
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of the linear layer applied to x.
                return self.linear(x).relu()

        model = Foo()

        def f(args, params, buffers):
            # [FUNCTION DOC] HELPER: Clears param grads, wraps non-iterable args in a list,
            # calls the model through functional_call with merged parameters and buffers,
            # backpropagates the summed output and returns each parameter minus a small multiple
            # of its gradient.
            for p in params.values():
                p.grad = None
            if not isinstance(args, Iterable):
                args = [args]
            params_and_buffers = {**params, **buffers}
            out = torch.func.functional_call(model, params_and_buffers, args)
            out.sum().backward()
            return [p - 1e-4 * p.grad for p in params.values()]

        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(input, params, buffers)
        # fx may change the order of parameters in list, so using set() to compare
        # also there is a numerical difference in results so changing atol from 1e-08 to 1e-03
        self.assertTrue(
            torch.allclose(fx_f(input, params, buffers)[0], f(input, params, buffers)[0], atol=1e-03)
            or
            torch.allclose(fx_f(input, params, buffers)[0], f(input, params, buffers)[1], atol=1e-03)
        )
        self.assertTrue(
            torch.allclose(fx_f(input, params, buffers)[1], f(input, params, buffers)[0], atol=1e-03)
            or
            torch.allclose(fx_f(input, params, buffers)[1], f(input, params, buffers)[1], atol=1e-03)
        )

    def test_trace_subclasses(self):
        # [FUNCTION DOC] Traces functions that wrap inputs in UnwrapTensor and multiply,
        # checking that the traced graph matches eager.
        def f1(x):
            # [FUNCTION DOC] HELPER: Wraps x in UnwrapTensor and returns the product with 2.
            x = UnwrapTensor(x)
            y = x * 2
            return y

        def f2(x):
            # [FUNCTION DOC] HELPER: Multiplies x by an UnwrapTensor wrapping x and returns the
            # result.
            wrapped = UnwrapTensor(x)
            y = x * wrapped
            return y

        inp = [torch.randn(5)]
        self._test(f1, inp)
        self._test(f2, inp)

    def test_partial_decomp(self):
        # [FUNCTION DOC] Checks a decomposition that returns NotImplemented for default
        # arguments leaves those addmm calls in the graph: the plain trace has two addmm nodes,
        # the decomposed trace has one, and both produce equal values.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Returns the sum of two addmm results, one default and one
            # with beta of 2.
            x = torch.addmm(a, b, c)
            y = torch.addmm(a, b, c, beta=2, alpha=1)
            return x + y
        inps = [torch.randn(5, 5), torch.randn(5, 5), torch.randn(5, 5)]
        fx_g = make_fx(f)(*inps)

        def addmm(a, b, c, beta=1, alpha=1):
            # [FUNCTION DOC] HELPER: Decomposition of addmm: returns NotImplemented when beta
            # and alpha are both 1, otherwise the explicit beta times a plus alpha times the
            # matrix product of b and c.
            if beta == 1 and alpha == 1:
                return NotImplemented
            return beta * a + alpha * (b @ c)

        decomposed_fx = make_fx(f, decomposition_table={aten.addmm.default: addmm})(*inps)

        self.assertEqual(fx_g(*inps), decomposed_fx(*inps))
        self.assertEqual(len([n for n in fx_g.graph.nodes if n.target == aten.addmm.default]), 2)
        self.assertEqual(len([n for n in decomposed_fx.graph.nodes if n.target == aten.addmm.default]), 1)

    def test_decomp_of_capture(self):
        # [FUNCTION DOC] Checks that a decomposition table is applied to ops on captured
        # constant tensors: after decomposing the transpose op into a cosine, no transpose node
        # remains in the traced graph.
        val = torch.randn(5)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the transpose of x added to the transpose of a
            # captured tensor.
            return x.t() + val.t()

        def nop(x):
            # [FUNCTION DOC] HELPER: Decomposition replacement that returns the cosine of x.
            return x.cos()

        traced = make_fx(f, decomposition_table={torch.ops.aten.t.default: nop})(torch.randn(5))
        self.assertEqual(len([n for n in traced.graph.nodes if n.target == torch.ops.aten.t.default]), 0)


    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_amp_cache(self):
        # [FUNCTION DOC] CUDA only. Traces a convolution twice under autocast and checks both
        # traces produce graphs with the same node count and the same node ops, i.e. the
        # autocast cache does not change the result of the second trace.
        layer = torch.nn.Conv2d(3, 3, 3).cuda()

        def f(x, w):
            # [FUNCTION DOC] HELPER: Applies conv2d of x with weight w using the stride of the
            # captured layer.
            return torch.nn.functional.conv2d(x, w, stride=layer.stride)

        inp = torch.randn(4, 3, 10, 10, device='cuda')
        with torch.autocast('cuda'):
            out_graph = make_fx(f)(inp, layer.weight).graph
            out_graph2 = make_fx(f)(inp, layer.weight).graph

        self.assertEqual(len(out_graph.nodes), len(out_graph2.nodes))
        for a, b in zip(out_graph.nodes, out_graph2.nodes):
            self.assertEqual(a.op, b.op)

    def test_strides(self):
        # [FUNCTION DOC] Checks stride and contiguity queries inside traced functions: a permute
        # to channels last order flips contiguity, and slicing a contiguous tensor yields non-
        # contiguous views.
        def f(x):
            # [FUNCTION DOC] HELPER: Asserts contiguity expectations before and after permuting
            # to channels last layout and returns the permuted tensor.
            self.assertTrue(x.is_contiguous())
            self.assertFalse(x.is_contiguous(memory_format=torch.channels_last))
            x = x.permute(0, 3, 1, 2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))
            return x
        make_fx(f)(torch.randn(2, 3, 4, 5))

        def f(x):
            # [FUNCTION DOC] HELPER: Redefinition. Asserts x is contiguous, that an index slice
            # and a stepped slice are not, and returns the cosine of x.
            self.assertTrue(x.is_contiguous())
            y = x[:, 1]
            self.assertFalse(y.is_contiguous())
            y = x[:, ::2]
            self.assertFalse(y.is_contiguous())
            return x.cos()

        make_fx(f)(torch.randn(2, 3, 4, 5))

    def test_pr_86917(self):
        # [FUNCTION DOC] Regression test for PR 86917: traces nll_loss_forward on a small batch
        # through _test.
        # Tests the issue brought up here https://github.com/pytorch/pytorch/pull/86917#issuecomment-1283155344
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns the nll_loss_forward of a and b with no weight,
            # reduction 1, ignore 10.
            return torch.ops.aten.nll_loss_forward(a, b, None, 1, 10)

        self._test(f, [torch.randn(1, 10), torch.zeros(1, dtype=torch.long)])

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_T244632748(self):
        # [FUNCTION DOC] CUDA only, expected to fail without triton. Exports a module adding a
        # dynamic-size dependent value to its input, rewires a symbolic size node into its uses,
        # removes dead code and compiles it with the inductor fx wrapper option; it only checks
        # that compilation does not raise.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds twice the first dimension size to
            # the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus twice its leading dimension size.
                return x + (x.shape[0] * 2)

        mod = TestModule()
        sample = torch.randn((5, 5)).to("cuda")
        dim0 = torch.export.Dim.DYNAMIC(max=100)
        dynamic_shapes = {"x": (dim0, torch.export.Dim.STATIC)}
        ep = torch.export.export(mod, (sample,), dynamic_shapes=dynamic_shapes)
        gm = ep.module()
        symint = list(gm.graph.nodes)[3].meta["val"]
        list(gm.graph.nodes)[3].replace_all_uses_with(symint)
        gm.graph.eliminate_dead_code()

        inductor_fx = torch._inductor.aot_compile(
            gm, (sample,), options={"fx_wrapper": True, "compile_threads": 1}
        )


class TestGenericProxyTensorReal(TestGenericProxyTensor):
    # [CLASS DOC] Runs every generic proxy tensor test with the real tracing mode.
    tracing_mode = "real"


class TestGenericProxyTensorFake(TestGenericProxyTensor):
    # [CLASS DOC] Runs every generic proxy tensor test with the fake tensor tracing mode.
    tracing_mode = "fake"


class TestGenericProxyTensorSymbolic(TestGenericProxyTensor):
    # [CLASS DOC] Runs every generic proxy tensor test with the symbolic shape tracing mode.
    tracing_mode = "symbolic"


del TestGenericProxyTensor


class TestRealProxyTensor(TestCase):
    # [CLASS DOC] Tests specific to real-tensor tracing mode.
    def test_error_on_data_dependent_ops(self):
        # [FUNCTION DOC] Smoke test: tracing a function that converts tensors to Python floats
        # and compares values must succeed when error on data dependent ops is turned off, with
        # and without pre-dispatch.
        def f():
            # [FUNCTION DOC] HELPER: Creates two scalar random tensors, asserts that products
            # commute, converts them to floats and returns nothing.
            x = torch.randn([])
            y = torch.randn([])
            if not torch.allclose(x * y, y * x):
                raise AssertionError("x * y should equal y * x")
            z = float(x)
            z2 = float(y)

        # Smoke tests
        make_fx(f, _error_on_data_dependent_ops=False)()
        make_fx(f, pre_dispatch=True, _error_on_data_dependent_ops=False)()

    def test_disable_torch_fn_metadata_mode(self):
        # [FUNCTION DOC] Checks the torch_fn node metadata is recorded when module stack
        # recording is on and is absent on every call_function node when the torch function
        # metadata mode is disabled.
        class MyModule(nn.Module):
            # [FUNCTION DOC] HELPER: Module returning the sum of sin and cos of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sine plus cosine of x.
                return torch.sin(x) + torch.cos(x)

        mod = MyModule()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the captured module on x.
            return mod(x)
        fn._orig_mod = mod

        # With TorchFunctionMetadataMode enabled (default), torch_fn should be present
        gm_with = make_fx(fn, record_module_stack=True)(torch.randn(3))
        torch_fn_present = any(
            "torch_fn" in n.meta for n in gm_with.graph.nodes if n.op == "call_function"
        )
        self.assertTrue(torch_fn_present, "torch_fn metadata should be present by default")

        # With TorchFunctionMetadataMode disabled, torch_fn should be absent
        gm_without = make_fx(fn, record_module_stack=True, _disable_torch_fn_metadata_mode=True)(torch.randn(3))
        torch_fn_absent = all(
            "torch_fn" not in n.meta for n in gm_without.graph.nodes if n.op == "call_function"
        )
        self.assertTrue(torch_fn_absent, "torch_fn metadata should be absent when mode is disabled")

class TestFakeProxyTensor(TestCase):
    # [CLASS DOC] Tests of make_fx in fake tracing mode and with fake tensor mode interaction.
    def test_issue82547(self):
        # [FUNCTION DOC] Regression test for issue 82547: tracing in fake mode with a real
        # parameter raises an error telling the user to convert all tensors, and with a tensor
        # subclass raises a TypeError about multiple dispatch failing.
        x = nn.Parameter(torch.randn(3, 3))

        def f():
            # [FUNCTION DOC] HELPER: Returns the transpose op applied to the captured tensor x.
            return torch.ops.aten.t.default(x)
        self.assertRaisesRegex(Exception, "Please convert all Tensors", lambda: make_fx(f, tracing_mode="fake")())

        class A(torch.Tensor):
            # [CLASS DOC] HELPER: Empty tensor subclass used to trigger the multiple dispatch
            # failure.
            pass

        x = A(torch.randn(3, 3))
        self.assertRaisesRegex(TypeError, "Multiple dispatch failed", lambda: make_fx(f, tracing_mode="fake")())

    def test_use_fake_and_tensor(self):
        # [FUNCTION DOC] Traces a function that creates a constant tensor in fake mode and
        # checks the traced graph run on real inputs equals eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y plus a constant two-element tensor.
            z = torch.tensor([2.0, 3.0])
            return x + y + z

        g = make_fx(f, tracing_mode="fake")(torch.randn(2), torch.randn(2))
        x, y = torch.randn(2), torch.randn(2)
        self.assertEqual(g(x, y), f(x, y))

    def test_free_fake(self):
        # [FUNCTION DOC] Creates a tensor under a FakeTensorMode context and then traces, in
        # real mode, a function capturing that tensor; it only checks that tracing does not
        # raise.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x added to a captured outer tensor y created later
            # in the test.
            return torch.add(x, y)

        with FakeTensorMode() as fake_mode:
            y = torch.randn(2)
            make_fx(f, tracing_mode="real")(torch.randn(2))

    def test_fused_adam(self):
        # [FUNCTION DOC] Regression test for issue 99356: traces the fused Adam op in fake mode
        # on lists of ten random tensors and checks the fused op and its getitem nodes all have
        # val metadata.
        # See https://github.com/pytorch/pytorch/issues/99356
        params = [torch.randn(10, 10) for _ in range(10)]
        grads = [torch.randn(10, 10) for _ in range(10)]
        exp_avgs = [torch.randn(10, 10) for _ in range(10)]
        exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        max_exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        state_steps = [torch.tensor(0) for _ in range(10)]

        def fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps):
            # [FUNCTION DOC] HELPER: Calls the fused Adam op with fixed hyperparameters, copies
            # the new values into each parameter in place and returns the parameter list.
            (new_params, _, _, _, _) = aten._fused_adam.default(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                lr=0.1,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.01,
                eps=1e-8,
                amsgrad=False,
                maximize=False,
            )

            for p, new_p in zip(params, new_params):
                p.copy_(new_p)

            return params

        gm = make_fx(fused_adam, tracing_mode='fake')(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )
        ensure_ops_have_val = [aten._fused_adam.default, operator.getitem]
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target in ensure_ops_have_val:
                self.assertIn('val', n.meta)

    def test_alias(self):
        # [FUNCTION DOC] Checks that tracing alias in fake mode produces a graph with just the
        # alias call and no detach, via a golden string.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the aten alias of x.
            return torch.ops.aten.alias(x)

        r = str(make_fx(f, tracing_mode="fake")(torch.randn(2)).code).strip()
        # NB: this should not have a detach call
        self.assertExpectedInline(r, """\
def forward(self, x_1):
    alias = torch.ops.aten.alias.default(x_1);  x_1 = None
    return alias""")

    def test_meta(self):
        # [FUNCTION DOC] Traces cos, var_mean and a multiply in fake mode and checks every non-
        # output node has val metadata.
        def f(x):
            # [FUNCTION DOC] HELPER: Takes cos of x, computes var_mean along dim 0, multiplies
            # by 2 and returns it.
            a = x.cos()
            b = torch.var_mean(a, dim=0)
            c = b * 2
            return c

        out = make_fx(f, tracing_mode="fake")(torch.randn(5, 5))
        for n in out.graph.nodes:
            if n.op == 'output':
                continue
            self.assertTrue('val' in n.meta)

    def test_fake_tensor_mode(self):
        # [FUNCTION DOC] Traces in real mode inside an existing FakeTensorMode and checks the
        # fake mode detected from node val metadata is that same mode.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns cos of a.
            d = a.cos()
            return d

        from torch._guards import detect_fake_mode

        existing_fake_mode = FakeTensorMode()
        with existing_fake_mode:
            out = make_fx(f, tracing_mode="real")(torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]))

        fake_mode = detect_fake_mode([node.meta.get('val', None) for node in out.graph.nodes])
        self.assertEqual(fake_mode, existing_fake_mode)

def _get_node(fx_g, cond):
    # [FUNCTION DOC] HELPER: Returns the first graph node of fx_g for which cond is true; raises
    # AssertionError if none matches.
    for n in fx_g.graph.nodes:
        if cond(n):
            return n
    raise AssertionError

def _get_free_symbols(shape_env):
    # [FUNCTION DOC] HELPER: Counts shape-environment symbols with a backed value that were not
    # replaced by another expression.
    vars = tuple(shape_env.backed_var_to_val.keys())
    return len([var for var in vars if var not in shape_env.replacements])

def _trace(f, *args):
    # [FUNCTION DOC] HELPER: Builds random tensors for each shape argument and returns f traced
    # with the symbolic tracing mode on them.
    inps = [torch.randn(arg) for arg in args]
    return make_fx(f, tracing_mode="symbolic")(*inps)

# TODO: Need to test the guards themselves specifically as well
class TestSymbolicTracing(TestCase):
    # [CLASS DOC] Tests of symbolic tracing with a shape environment: golden graphs, guards,
    # dynamic shape handling of many ops and bug repros. Provides the _test_dynamic helper.
    def _test_dynamic(self, fn, trace_inputs, test_inputs, assert_eq=True):
        # [FUNCTION DOC] HELPER: Traces fn symbolically with random inputs from trace_inputs
        # shapes, then for each shape list in test_inputs runs both the traced and original
        # function on random inputs and, if assert_eq is true, asserts equality. Returns the
        # traced graph module.
        """
        Tests fn traced with trace_inputs against test_inputs
        Also returns shape env
        """
        trace_inputs = [torch.randn(shape) for shape in trace_inputs]
        traced_f = make_fx(fn, tracing_mode="symbolic")(*trace_inputs)
        for input in test_inputs:
            input = [torch.randn(shape) for shape in input]
            rx, ry = traced_f(*input), fn(*input)
            if assert_eq:
                self.assertEqual(rx, ry)
        return traced_f


    def test_debug_interpreter(self):
        # [FUNCTION DOC] Defines a scoped custom op with differing cpu (transposed) and meta
        # (non transposed) strides, traces it symbolically and runs DebugInterpreter: a
        # mismatching input size raises an AssertionError about sizes, and the incorrect meta
        # raises one about strides.
        from torch.library import _scoped_library

        with _scoped_library("foo", "DEF") as foo:
            foo.define("foo(Tensor self) -> Tensor")

            # Operator where meta and cpu disagree on strides
            @torch.library.impl(foo, "foo", "CPU")
            def foo_cpu(x):
                # [FUNCTION DOC] HELPER: CPU kernel returning a clone transposed.
                return x.clone().T

            @torch.library.impl(foo, "foo", "Meta")
            def foo_meta(x):
                # [FUNCTION DOC] HELPER: Meta kernel returning a clone without transposing, so
                # strides disagree.
                return x.clone()

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the custom foo op on x.
                return torch.ops.foo.foo.default(x)

            gm = make_fx(f, tracing_mode="symbolic")(torch.randn(2, 2))
            from torch._functorch.compilers import DebugInterpreter

            interp = DebugInterpreter(gm)

            # input mismatch is caught (indicates guard problem)
            self.assertRaisesRegex(
                AssertionError, r"3 != 1",
                lambda: interp.run(torch.randn(3, 3).T),
            )

            # Catch the incorrect meta
            self.assertRaisesRegex(
                AssertionError, r"\(3, 1\) != \(1, 3\)",
                lambda: interp.run(torch.randn(3, 3))
            )

    def test_int_input(self):
        # [FUNCTION DOC] Symbolic tracing with an integer input used as a view size; golden
        # string check.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns x viewed with shape given by y.
            return x.view(y)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(3, 4), 12).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    view = torch.ops.aten.view.default(x_1, [y_1]);  x_1 = y_1 = None
    return view""")

    def test_resize_from_zero(self):
        # [FUNCTION DOC] Symbolic tracing of in place resize of an empty tensor to the size of
        # another; golden string check.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Resizes x in place to the first dimension size of y;
            # returns nothing.
            x.resize_(y.size(0))

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(0), torch.empty(2)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    resize_ = torch.ops.aten.resize_.default(x_1, [sym_size_int]);  x_1 = sym_size_int = resize_ = None
    return None""")

    def test_broadcast_shapes(self):
        # [FUNCTION DOC] Symbolic tracing of broadcast_shapes with sizes; golden string shows
        # the size nodes.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the broadcast of the shape of x and the first size
            # of y.
            return torch.functional.broadcast_shapes(x.size(), y.size()[0])

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(3, 1), torch.empty(5)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    return (sym_size_int, sym_size_int_1)""")

    def test_deduped_shape(self):
        # [FUNCTION DOC] Under a fake mode with a shared shape env, traces in real mode with
        # explicit symbolic sizes passed as inputs; golden string shows the shape inputs reused
        # (deduplicated) as symbols.
        def f(s0, s1, x, y):
            # [FUNCTION DOC] HELPER: Returns a pair of broadcast shapes and an empty tensor
            # sized by x leading dim.
            return torch.functional.broadcast_shapes(x.size(), y.size()[0]), torch.empty(x.shape[0])

        x = torch.empty(3, 1)
        y = torch.empty(5)
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        shape_env = ShapeEnv()

        with FakeTensorMode(shape_env=shape_env, static_shapes=False) as fake_mode:
            x = fake_mode.from_tensor(x)
            y = fake_mode.from_tensor(y)
            r = str(make_fx(f, tracing_mode="real")(x.shape[0], y.shape[0], x, y).code).strip()
            self.assertExpectedInline(r, """\
def forward(self, s0_1, s1_1, x_1, y_1):
    empty = torch.ops.aten.empty.memory_format([s0_1], device = device(type='cpu'), pin_memory = False)
    return ((s0_1, s1_1), empty)""")

    def test_non_deduped_shape(self):
        # [FUNCTION DOC] Same as the deduped case but without passing sizes explicitly: golden
        # string shows sym_size nodes computed from the inputs.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair of broadcast shapes and an empty tensor
            # sized by x leading dim.
            return torch.functional.broadcast_shapes(x.size(), y.size()[0]), torch.empty(x.shape[0])

        x = torch.empty(3, 1)
        y = torch.empty(5)
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        shape_env = ShapeEnv()

        with FakeTensorMode(shape_env=shape_env, static_shapes=False) as fake_mode:
            x = fake_mode.from_tensor(x)
            y = fake_mode.from_tensor(y)
            r = str(make_fx(f, tracing_mode="real")(x, y).code).strip()
            self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    empty = torch.ops.aten.empty.memory_format([sym_size_int], device = device(type='cpu'), pin_memory = False)
    return ((sym_size_int, sym_size_int_1), empty)""")

    def test_unary(self):
        # [FUNCTION DOC] Symbolic trace of cos with a guard on the leading dim; checks guards
        # evaluate true for a conforming shape, bound symbols print as expected, false for a too
        # large shape, and the guard text.
        def f(x):
            # [FUNCTION DOC] HELPER: Raises AssertionError if the first dim is 20 or more;
            # otherwise returns cos of x.
            if x.shape[0] >= 20:
                raise AssertionError(f"expected x.shape[0] < 20, got {x.shape[0]}")
            return x.cos()
        test_inputs = []
        test_inputs.append([(2, 5)])
        test_inputs.append([(6, 8)])
        gm = self._test_dynamic(f, [(3, 4)], test_inputs)
        self.assertTrue(eval_guards(gm, torch.randn(4, 5)))
        self.assertEqual(repr(bind_symbols(gm, torch.randn(4, 5))), "{s75: 4, s96: 5}")
        self.assertFalse(eval_guards(gm, torch.randn(25, 5)))
        self.assertExpectedInline(show_guards(gm), """L['x'].size()[0] <= 19""")

    def test_repeat_interleave(self):
        # [FUNCTION DOC] Symbolic trace of repeat_interleave with a count taken from another
        # tensor size; checks no guards are recorded in the shape environment.
        def f(src_tokens, beam_size_src):
            # [FUNCTION DOC] HELPER: Repeats rows of src_tokens by the size of the first dim of
            # beam_size_src.
            return src_tokens.repeat_interleave(beam_size_src.size(0), 0)

        prompt_size = 64
        vocab_size = 64
        batch_size = 4
        src_tokens = torch.randint(1, vocab_size, (batch_size, prompt_size))
        gm = make_fx(f, tracing_mode="symbolic")(src_tokens, torch.randn(5))
        self.assertEqual(len(gm.shape_env.guards), 0)

    def test_non_symint_size_spec(self):
        # [FUNCTION DOC] Smoke test: under symbolic tracing, calls the internal non-symbolic-
        # sizes query on a fake tensor with symbolic sizes; it only checks tracing succeeds.
        # this isn't really a proxy tensor test, but it's the most convenient
        # way to get a fake tensor with symbolic sizes
        def f(x):
            # [FUNCTION DOC] HELPER: Queries the non symint sizes of x and returns x plus 1.
            torch._C._non_sym_sizes(x)
            return x + 1

        x = torch.randn(2, 3)
        make_fx(f, tracing_mode="symbolic")(x)

    def test_scalar_placeholder_meta(self):
        # [FUNCTION DOC] Traces a function taking a tensor, an int, a float and a bool; checks
        # the placeholder nodes for the scalar arguments carry their concrete values in val
        # metadata.
        def f(x, n, alpha, flag):
            # [FUNCTION DOC] HELPER: Adds n to x and multiplies by alpha when flag is true.
            y = x + n
            return y * alpha if flag else y

        gm = make_fx(f)(torch.ones(2), 3, 2.0, True)
        placeholder_meta = {
            node.target: node.meta.get("val")
            for node in gm.graph.nodes
            if node.op == "placeholder"
        }
        self.assertEqual(placeholder_meta["n_1"], 3)
        self.assertEqual(placeholder_meta["alpha_1"], 2.0)
        self.assertEqual(placeholder_meta["flag_1"], True)

    def test_unbacked_bindings_on_multiple_token_grid_placeholders(self):
        # [FUNCTION DOC] Builds three fake tensors with an unbacked leading dimension (one
        # shared shape id) in one shape env, traces a function over them, and checks each of the
        # three placeholders records exactly one unbacked binding, matching the pending fresh
        # unbacked symbol in order, with path to the first size.
        from torch._dynamo.source import LocalSource
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )
        from torch.utils._pytree import keystr

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)

        def fakeify(t, name):
            # [FUNCTION DOC] HELPER: Converts t into a fake tensor of the test fake mode, with
            # local source name and a stateless symbolic context making dim 0 unbacked and dim 1
            # static, sharing a shape id.
            return fake_mode.from_tensor(
                t,
                source=LocalSource(name, is_input=True),
                symbolic_context=StatelessSymbolicContext(
                    dynamic_sizes=[DimDynamic.UNBACKED, DimDynamic.STATIC],
                    shape_ids={0: "token_grid_batch"},
                ),
            )

        token_ids = fakeify(torch.randint(8, (2, 4)), "token_ids")
        labels = fakeify(torch.randint(8, (2, 4)), "labels")
        position_ids = fakeify(torch.arange(4).expand(2, 4), "position_ids")
        expected_symbols = tuple(shape_env.pending_fresh_unbacked_symbols)

        def fn(token_ids, labels, position_ids):
            # [FUNCTION DOC] HELPER: Adds float versions of token ids and position ids, then
            # selects between that and its negation by whether labels are non negative; returns
            # the result.
            token_features = token_ids.to(torch.float32) + position_ids.to(
                torch.float32
            )
            return torch.where(labels >= 0, token_features, token_features.neg())

        with fake_mode:
            gm = make_fx(fn)(token_ids, labels, position_ids)

        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        self.assertEqual(len(placeholders), 3)
        for placeholder, expected_symbol in zip(
            placeholders, expected_symbols, strict=True
        ):
            bindings = placeholder.meta.get("unbacked_bindings")
            self.assertIsNotNone(bindings)
            self.assertEqual(len(bindings), 1)
            self.assertEqual(set(bindings), {expected_symbol})
            self.assertEqual(
                [keystr(path) for path in bindings.values()], [".size()[0]"]
            )

    # https://github.com/pytorch/pytorch/issues/108195
    def test_symbolic_repeat_interleave(self):
        # [FUNCTION DOC] Regression test for issue 108195: symbolic trace of repeat_interleave
        # with a tensor of repeats lowers to repeat_interleave and index_select; golden string
        # check.
        def f(y, x):
            # [FUNCTION DOC] HELPER: Returns y repeated along dim 1 by the counts in x.
            return y.repeat_interleave(x, dim=1)

        y = torch.tensor([[1, 2], [3, 4]])
        x = torch.tensor([2, 3])
        r = str(make_fx(f, tracing_mode="symbolic")(y, x).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, y_1, x_1):
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1);  x_1 = None
    index_select = torch.ops.aten.index_select.default(y_1, 1, repeat_interleave);  y_1 = repeat_interleave = None
    return index_select""")

    def test_mod_gcd_unbacked(self):
        # [FUNCTION DOC] Traces concatenation of two tensors whose lengths are unbacked values
        # times a stride, viewed as a matrix; golden string check that the reshape works with
        # unbacked divisibility.
        def f(_a, _b, _stride):
            # [FUNCTION DOC] HELPER: Reads three scalars, builds two random vectors sized a and
            # b times stride, concatenates them and views as (a plus b) rows of stride columns.
            a = _a.item()
            b = _b.item()
            stride = _stride.item()
            ta = torch.randn(a * stride)
            tb = torch.randn(b * stride)
            r = torch.cat([ta, tb])
            return r.view(a + b, stride)

        _a = torch.tensor(30)
        _b = torch.tensor(20)
        _stride = torch.tensor(10)
        r = str(make_fx(f, tracing_mode="symbolic")(_a, _b, _stride).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, _a_1, _b_1, _stride_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(_a_1);  _a_1 = None
    _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(_b_1);  _b_1 = None
    _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(_stride_1);  _stride_1 = None
    mul = _local_scalar_dense * _local_scalar_dense_2
    randn = torch.ops.aten.randn.default([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    mul_1 = _local_scalar_dense_1 * _local_scalar_dense_2
    randn_1 = torch.ops.aten.randn.default([mul_1], device = device(type='cpu'), pin_memory = False);  mul_1 = None
    cat = torch.ops.aten.cat.default([randn, randn_1]);  randn = randn_1 = None
    add = _local_scalar_dense + _local_scalar_dense_1;  _local_scalar_dense = _local_scalar_dense_1 = None
    view = torch.ops.aten.view.default(cat, [add, _local_scalar_dense_2]);  cat = add = _local_scalar_dense_2 = None
    return view""")

    def test_cumsum_unbacked(self):
        # [FUNCTION DOC] Symbolic trace of cumsum on a tensor whose middle size is an unbacked
        # item; golden string.
        def f(x):
            # [FUNCTION DOC] HELPER: Reads an item y from x, builds a random tensor of shape 3,
            # y, 3 and returns its cumulative sum along dim 0.
            y = x.item()
            z = torch.randn((3, y, 3))
            return z.cumsum(0)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor([5])).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    randn = torch.ops.aten.randn.default([3, _local_scalar_dense, 3], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    cumsum = torch.ops.aten.cumsum.default(randn, 0);  randn = None
    return cumsum"""
        )


    def test_repeat_interleave_unbacked_output_size(self):
        # [FUNCTION DOC] Symbolic trace of repeat_interleave with an unbacked output_size
        # obtained from item; golden string check.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Sums x to a scalar s and returns y repeated by x along dim
            # 0 with output size s.
            s = x.sum().item()
            return y.repeat_interleave(x, dim=0, output_size=s)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor([2, 3]), torch.randn(2)).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1, y_1):
    sum_1 = torch.ops.aten.sum.default(x_1)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(sum_1);  sum_1 = None
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1, output_size = _local_scalar_dense);  x_1 = _local_scalar_dense = None
    index_select = torch.ops.aten.index_select.default(y_1, 0, repeat_interleave);  y_1 = repeat_interleave = None
    return index_select"""
        )

    def test_arange_unbacked_output_size(self):
        # [FUNCTION DOC] Symbolic trace of arange with an unbacked end taken from a tensor
        # input; golden string.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns arange from 0 up to x.
            return torch.arange(0, x)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor(10)).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    arange = torch.ops.aten.arange.start(0, _local_scalar_dense, device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return arange"""
        )

    def test_adv_index_batch(self):
        # [FUNCTION DOC] Traces a beam-search style token buffer fill with a batched slice
        # assignment and repeat_interleave under symbolic mode; checks no guards are added to
        # the shape env.
        def f(src_tokens):
            # [FUNCTION DOC] HELPER: Builds a zero token buffer of batch by (source length plus
            # 64) in the dtype of the input, assigns the repeated source tokens into its leading
            # columns and returns it.
            bsz, src_len = src_tokens.size()[:2]
            start_step = src_tokens.shape[1]
            beam_size = 1
            generate_size = 64
            max_len = src_len + generate_size
            tokens = torch.zeros(bsz * beam_size, max_len).to(src_tokens).long().fill_(0)
            tokens[:, :start_step] = src_tokens.repeat_interleave(beam_size, 0)
            return tokens

        prompt_size = 64
        vocab_size = 64
        batch_size = 4
        src_tokens = torch.randint(1, vocab_size, (batch_size, prompt_size))
        gm = make_fx(f, tracing_mode="symbolic")(src_tokens)
        # Guards to rule out batch_size == sys.maxsize (wobbling between 2 and
        # 1 ok)
        self.assertEqual(len(gm.shape_env.guards), 0)

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_cpu_scalar_cuda(self):
        # [FUNCTION DOC] CUDA only. Symbolic trace of a CPU scalar tensor multiplied with a CUDA
        # matrix then matrix multiplied; golden string shows mul and mm (extracted from
        # wave2vec2).
        # Extracted from wave2vec2
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns the product of a and b matrix-multiplied with b.
            return (a * b) @ b

        r = str(
            make_fx(f, tracing_mode="symbolic")(
                torch.tensor(1.0), torch.randn(2, 2, device='cuda')
            ).code
        ).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1, b_1):
    mul = torch.ops.aten.mul.Tensor(a_1, b_1);  a_1 = None
    mm = torch.ops.aten.mm.default(mul, b_1);  mul = b_1 = None
    return mm""")

    def test_binary_broadcast(self):
        # [FUNCTION DOC] Traces a broadcasting multiply using _test_dynamic with two differently
        # broadcast test shapes and asserts the shape env recorded no guards, else raises
        # AssertionError.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns the product of a and b.
            c = a * b
            return c

        test_inputs = []
        test_inputs.append([(1, 5), (3, 1)])
        test_inputs.append([(1, 4), (4, 1)])
        shape_env = self._test_dynamic(f, [(1, 2), (3, 1)], test_inputs).shape_env
        if len(shape_env.guards) != 0:
            raise AssertionError(f"expected no guards, got {len(shape_env.guards)}")

    def test_multiply_shape(self):
        # [FUNCTION DOC] Symbolic trace of an empty tensor sized as twice the input leading
        # dimension; golden string.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns an uninitialized vector of length twice the first
            # dim of a.
            return torch.empty(a.shape[0] * 2)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0);  a_1 = None
    mul = sym_size_int * 2;  sym_size_int = None
    empty = torch.ops.aten.empty.memory_format([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    return empty""")

    def test_item(self):
        # [FUNCTION DOC] Symbolic trace of using item as a multiplier of the tensor; golden
        # string.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns a.item() times a.
            r = a.item()
            return r * a

        r = str(make_fx(f, tracing_mode="symbolic")(torch.randn(1)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1)
    mul = torch.ops.aten.mul.Tensor(a_1, _local_scalar_dense);  a_1 = _local_scalar_dense = None
    return mul""")

    def test_tensor_symfloat(self):
        # [FUNCTION DOC] Checks that building a tensor from a symbolic float (size squared)
        # infers float32 dtype; the trace specializes to a constant, whose value is asserted to
        # be 4.0.
        def f(a):
            # [FUNCTION DOC] HELPER: Creates a tensor from the squared first size of a, asserts
            # the dtype is float, raising AssertionError otherwise, and returns it.
            r = torch.tensor(a.size(0) ** 2.0)
            if r.dtype is not torch.float:
                raise AssertionError(f"expected dtype torch.float, got {r.dtype}")
            return r

        gm = make_fx(f, tracing_mode="symbolic")(torch.randn(2))
        r = str(gm.code).strip()
        # NB: this specializes, which is fine, the point is to make sure the
        # dtype inference is correct
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    return lift_fresh_copy""")
        self.assertEqual(gm._tensor_constant0, torch.tensor(4.0))

    def test_item_to_constructor(self):
        # [FUNCTION DOC] Symbolic trace of using an item as the size of an empty tensor; golden
        # string.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns an uninitialized vector whose length is the item of
            # a.
            r = a.item()
            return torch.empty(r)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.randint(5, (1,))).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, a_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1);  a_1 = None
    empty = torch.ops.aten.empty.memory_format([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return empty"""
        )


    def test_setitem_symint(self):
        # [FUNCTION DOC] Regression test from moco (issue 101939): in place assignment of a
        # symbolic size into an element; golden string shows scalar_tensor, select and copy_.
        # from moco
        # https://github.com/pytorch/pytorch/issues/101939
        def f(x):
            # [FUNCTION DOC] HELPER: Sets element 0 of x to the size of its first dim and
            # returns x.
            x[0] = x.size(0)
            return x

        r = str(make_fx(f, tracing_mode="symbolic")(torch.randn(10)).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    scalar_tensor = torch.ops.aten.scalar_tensor.default(sym_size_int, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'));  sym_size_int = None
    select = torch.ops.aten.select.int(x_1, 0, 0)
    copy_ = torch.ops.aten.copy_.default(select, scalar_tensor);  select = scalar_tensor = copy_ = None
    return x_1"""
        )

    def test_dynamic_pointwise_scalar(self):
        # [FUNCTION DOC] Symbolic trace of a masked in place multiply of a column by -1; golden
        # string shows the select, index, mul and index_put_ ops.
        def f(gravity, mask):
            # [FUNCTION DOC] HELPER: Multiplies the masked entries of column 0 of gravity by -1
            # in place.
            gravity[mask, 0] = gravity[mask, 0] * -1

        r = str(make_fx(f, tracing_mode="symbolic")(
            torch.randn((12, 4)),
            torch.randint(0, 2, (12,), dtype=torch.bool)
        ).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, gravity_1, mask_1):
    select = torch.ops.aten.select.int(gravity_1, 1, 0)
    index = torch.ops.aten.index.Tensor(select, [mask_1]);  select = None
    mul = torch.ops.aten.mul.Tensor(index, -1);  index = None
    select_1 = torch.ops.aten.select.int(gravity_1, 1, 0);  gravity_1 = None
    index_put_ = torch.ops.aten.index_put_.default(select_1, [mask_1], mul);  select_1 = mask_1 = mul = index_put_ = None
    return None""")

    def test_reflect_r_over_x(self):
        # [FUNCTION DOC] Symbolic trace of reflecting masked 3 by 3 matrices over x via boolean-
        # mask in place assignment; golden string pins the generated ops including a lifted
        # constant.
        def reflect_R_over_x(R):
            # [FUNCTION DOC] HELPER: Builds an identity matrix, negates its first entry, and
            # returns reflect times R times reflect.
            reflect = torch.eye(3, device=R.device)
            reflect[0, 0] = -1
            return reflect @ R @ reflect

        def f(crop_camera, mask):
            # [FUNCTION DOC] HELPER: Replaces masked rows of crop_camera in place by their
            # reflected version.
            crop_camera[mask] = reflect_R_over_x(crop_camera[mask])

        r = str(make_fx(f, tracing_mode="symbolic")(
            torch.randn((12, 3, 3)),
            torch.randint(0, 2, (12,), dtype=torch.bool)
        ).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, crop_camera_1, mask_1):
    index = torch.ops.aten.index.Tensor(crop_camera_1, [mask_1])
    eye = torch.ops.aten.eye.default(3, device = device(type='cpu'), pin_memory = False)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    select = torch.ops.aten.select.int(eye, 0, 0)
    select_1 = torch.ops.aten.select.int(select, 0, 0);  select = None
    copy_ = torch.ops.aten.copy_.default(select_1, lift_fresh_copy);  select_1 = lift_fresh_copy = copy_ = None
    sym_size_int = torch.ops.aten.sym_size.int(index, 0)
    expand = torch.ops.aten.expand.default(eye, [sym_size_int, 3, 3])
    view = torch.ops.aten.view.default(expand, [sym_size_int, 3, 3]);  expand = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(crop_camera_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(crop_camera_1, 2)
    expand_1 = torch.ops.aten.expand.default(index, [sym_size_int, sym_size_int_1, sym_size_int_2]);  index = None
    view_1 = torch.ops.aten.view.default(expand_1, [sym_size_int, sym_size_int_1, sym_size_int_2]);  expand_1 = sym_size_int_1 = sym_size_int_2 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [sym_size_int, 3, 3]);  bmm = None
    mul_11 = sym_size_int * 3
    view_3 = torch.ops.aten.view.default(view_2, [mul_11, 3]);  view_2 = mul_11 = None
    mm = torch.ops.aten.mm.default(view_3, eye);  view_3 = eye = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(mm, [sym_size_int, 3, 3]);  mm = sym_size_int = None
    index_put_ = torch.ops.aten.index_put_.default(crop_camera_1, [mask_1], _unsafe_view);  crop_camera_1 = mask_1 = _unsafe_view = index_put_ = None
    return None""")

    def test_unbacked_slice(self):
        # [FUNCTION DOC] Smoke test: symbolic trace of boolean mask selection followed by
        # slicing the last dim.
        def f(x, m):
            # [FUNCTION DOC] HELPER: Selects rows by mask m, then returns the first two entries
            # of the last dim.
            x = x[m]
            return x[slice(None, None, None), slice(None, None, None), slice(None, 2, None)]

        make_fx(f, tracing_mode="symbolic")(
            torch.randn((12, 3, 3)),
            torch.randint(0, 2, (12,), dtype=torch.bool)
        )

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_unbacked_batch_resnet(self):
        # [FUNCTION DOC] Needs torchvision. Traces resnet18 forward with a boolean-masked
        # (unbacked) batch, first specializing all parameter and input shapes, then asserting
        # the batch is at least one.
        mod = torchvision.models.resnet18()

        def f(x, mask, params, buffers):
            # [FUNCTION DOC] HELPER: Guards on every size of inputs, params and buffers, selects
            # the masked batch, asserts via a check that it is nonempty, clears param grads and
            # returns the summed module output.
            for p in itertools.chain([x, mask], params.values(), buffers.values()):
                for s in p.shape:
                    guard_int(s)
            x = x[mask]
            torch._check(x.shape[0] >= 1)
            for p in params.values():
                p.grad = None
            return torch.func.functional_call(mod, {**params, **buffers}, (x,)).sum()

        make_fx(f, tracing_mode="symbolic")(
            torch.randn(3, 3, 250, 250),
            torch.randint(0, 2, (3,), dtype=torch.bool),
            dict(mod.named_parameters()),
            dict(mod.named_buffers()),
        )

    def test_boolean_index(self):
        # [FUNCTION DOC] Symbolic trace of masked indexing of images and flipping those with
        # right hand mask, assigned back in place; golden string check.
        def f(images, handedness, valid):
            # [FUNCTION DOC] HELPER: Selects valid images and handedness, flips the images whose
            # handedness equals one along the last dim and writes them back in place; returns
            # nothing.
            images = images[valid]
            handedness = handedness[valid]
            right_hand_mask = handedness == 1
            images[right_hand_mask] = images[right_hand_mask].flip(-1)

        r = str(make_fx(f, tracing_mode="symbolic")(
            torch.randint(0, 256, (512, 1, 96, 96)),
            torch.randint(0, 1, (512,)),
            torch.randint(0, 2, (512,), dtype=torch.bool)
        ).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, images_1, handedness_1, valid_1):
    index = torch.ops.aten.index.Tensor(images_1, [valid_1]);  images_1 = None
    index_1 = torch.ops.aten.index.Tensor(handedness_1, [valid_1]);  handedness_1 = valid_1 = None
    eq = torch.ops.aten.eq.Scalar(index_1, 1);  index_1 = None
    index_2 = torch.ops.aten.index.Tensor(index, [eq])
    flip = torch.ops.aten.flip.default(index_2, [-1]);  index_2 = None
    index_put_ = torch.ops.aten.index_put_.default(index, [eq], flip);  index = eq = flip = index_put_ = None
    return None""")

    def test_neg_shape(self):
        # [FUNCTION DOC] Symbolic trace of a shape using a negated size plus ten; golden string
        # shows neg and add.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns an uninitialized vector of length ten minus the
            # first size of a.
            return torch.empty(-a.shape[0] + 10)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(2)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0);  a_1 = None
    neg = -sym_size_int;  sym_size_int = None
    add = neg + 10;  neg = None
    empty = torch.ops.aten.empty.memory_format([add], device = device(type='cpu'), pin_memory = False);  add = None
    return empty""")

    def test_unbacked_unification(self):
        # [FUNCTION DOC] Symbolic trace where a zeros tensor with unbacked length is added to a
        # tensor of length ten; golden string checks the two sizes get unified.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns zeros whose length is the item of x, added to y.
            z = torch.zeros(x.item())
            return z + y

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor(10), torch.randn(10)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    add = torch.ops.aten.add.Tensor(zeros, y_1);  zeros = y_1 = None
    return add""")

    def test_reshape_divisibility_unbacked(self):
        # [FUNCTION DOC] Smoke test: reshape of a transposed tensor with unbacked leading size
        # succeeds because the product of other dims is divisible.
        def f(x):
            # [FUNCTION DOC] HELPER: Reads item i0, builds zeros of shape i0, 4, 20, swaps dims
            # 1 and 2 and reshapes to rows of 80.
            i0 = x.item()
            r = torch.zeros(i0, 4, 20)
            r = r.transpose(2, 1)
            return r.reshape(-1, 80)
        make_fx(f, tracing_mode="symbolic")(torch.tensor(24))

    def test_view_divisibility_unbacked(self):
        # [FUNCTION DOC] Smoke test: view of an unbacked-leading tensor into three dims where 12
        # is the first.
        def f(x):
            # [FUNCTION DOC] HELPER: Reads item i0, builds zeros of shape i0 by 192 and views it
            # as 12, inferred, 192.
            i0 = x.item()
            r = torch.zeros(i0, 192)
            return r.view(12, -1, 192)
        make_fx(f, tracing_mode="symbolic")(torch.tensor(24))

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_view_divisibility_unbacked_relatively_prime(self):
        # [FUNCTION DOC] CUDA only. Regression for issue 123651: viewing a tensor with an
        # unbacked length bounded between 1 and 448 into rows of 447 must be accepted.
        # See https://github.com/pytorch/pytorch/issues/123651
        def f(x):
            # [FUNCTION DOC] HELPER: Reads item i0 from x, checks it lies in 1 to 448, and
            # returns zeros of length 256 times i0 viewed with rows of 447.
            i0 = x.item()
            # To trigger the original issue, the max bound has to
            # be chosen such that 448 / 447 < 2 (which it is.)
            torch._check(i0 > 0)
            torch._check(i0 <= 448)
            return torch.zeros(256 * i0).view(-1, 447)
        make_fx(f, tracing_mode="symbolic")(torch.tensor(256 * 447, device="cuda"))

    def test_unbacked_unify_guard(self):
        # [FUNCTION DOC] Symbolic trace where a check unifies an unbacked size with a backed
        # size, and a branch on the size equal to four resolves; golden string shows only the
        # else branch addition remains.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Creates zeros sized by the item of x, checks its size
            # equals that of y, then returns y times 2 if the size is 4 else y plus 2.
            z = torch.zeros(x.item())
            torch._check(z.size(0) == y.size(0))  # refines i0 = s0
            if z.size(0) == 4:
                return y * 2
            else:
                return y + 2

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor(10), torch.randn(10)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = zeros = None
    add = torch.ops.aten.add.Tensor(y_1, 2);  y_1 = None
    return add""")

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    @unittest.expectedFailure
    def test_unbacked_unify_guard_transitivity(self):
        # [FUNCTION DOC] CUDA only and marked expected failure. Traces with two unbacked sizes
        # unified through two checks and a size branch, inserts deferred runtime asserts and
        # recompiles; the golden check is commented out in the source.
        def f(x1, x2, y):
            # [FUNCTION DOC] HELPER: Creates zeros from two items, checks their sizes equal each
            # other and the size of y, then returns y times 2 if the size is 4 else y plus 2.
            z1 = torch.zeros(x1.item())
            z2 = torch.zeros(x2.item())
            torch._check(z1.size(0) == z2.size(0))  # refines i0 = i1
            torch._check(z2.size(0) == y.size(0))  # refines i0 = s0
            if z1.size(0) == 4:
                return y * 2
            else:
                return y + 2

        gm = make_fx(f, tracing_mode="symbolic")(
            torch.tensor(10, device="cuda"),
            torch.tensor(10, device="cuda"),
            torch.randn(10, device="cuda")
        )
        insert_deferred_runtime_asserts(gm, gm.shape_env, "test")
        gm.recompile()
        r = str(gm.code).strip()
        # self.assertExpectedInline(
        #     r, """"""
        # )

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_unbacked_unify_dependency_violation(self):
        # [FUNCTION DOC] CUDA only. Traces with unbacked items constrained by a divisibility
        # check and a sum equality; after inserting runtime asserts, input satisfying the
        # constraints gives the expected output and violating input raises a RuntimeError.
        def f(x1, x2, x3, y):
            # [FUNCTION DOC] HELPER: Checks the first item divided by nine equals one and that
            # it equals the sum of the other two items, then returns y times 2.
            z1 = x1.item()
            torch._check(z1 // 9 == 1)
            z2 = x2.item()
            z3 = x3.item()
            torch._check(z1 == z2 + z3)
            return y * 2
        # NB: inputs are done as CUDA to ensure they aren't queried to be
        # backed

        gm = make_fx(f, tracing_mode="symbolic")(
            torch.tensor(10, device="cuda"), torch.tensor(5, device="cuda"),
            torch.tensor(5, device="cuda"), torch.randn(1, device="cuda")
        )
        insert_deferred_runtime_asserts(gm, gm.shape_env, "test")
        gm.recompile()
        self.assertEqual(gm(
            torch.tensor(12, device="cuda"), torch.tensor(6, device="cuda"),
            torch.tensor(6, device="cuda"), torch.tensor([1.0], device="cuda")),
            torch.tensor([2.0], device="cuda")
        )
        with self.assertRaises(RuntimeError):
            gm(
                torch.tensor(20, device="cuda"), torch.tensor(10, device="cuda"),
                torch.tensor(10, device="cuda"), torch.tensor([1.0], device="cuda")
            )


    def test_split_unbacked_sizes(self):
        # [FUNCTION DOC] Symbolic trace of splitting a tensor by sizes taken as items of a
        # length tensor; golden string shows split_with_sizes with unbacked sizes.
        def f(lengths, values):
            # [FUNCTION DOC] HELPER: Builds a list of items from lengths and returns the split
            # of values by them.
            # tolist not directly supported atm
            sizes = [lengths[i].item() for i in range(lengths.size(0))]
            return torch.split(values, sizes)

        r = str(make_fx(f, tracing_mode="symbolic")(
            torch.tensor([2, 3, 4]),
            torch.randn(9)
        ).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, lengths_1, values_1):
    select = torch.ops.aten.select.int(lengths_1, 0, 0)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(select);  select = None
    select_1 = torch.ops.aten.select.int(lengths_1, 0, 1)
    _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(select_1);  select_1 = None
    select_2 = torch.ops.aten.select.int(lengths_1, 0, 2);  lengths_1 = None
    _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(select_2);  select_2 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(values_1, [_local_scalar_dense, _local_scalar_dense_1, _local_scalar_dense_2]);  values_1 = _local_scalar_dense = _local_scalar_dense_1 = _local_scalar_dense_2 = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1]
    getitem_2 = split_with_sizes[2];  split_with_sizes = None
    return (getitem, getitem_1, getitem_2)""")

    def test_invalidate_nonzero(self):
        # [FUNCTION DOC] Checks the nonzero result is cached for unchanged data (equal sizes),
        # then after an in place normal fill a new nonzero size comparison raises
        # GuardOnDataDependentSymNode, signalling the cache was invalidated.
        ok = False

        def f(a):
            # [FUNCTION DOC] HELPER: Calls nonzero repeatedly on a clone, asserts consistent
            # sizes, sets the outer flag ok, mutates the clone and expects a data-dependent
            # guard error on comparing sizes (fails the test otherwise).
            nonlocal ok
            b = a.clone()
            x = b.nonzero()
            x1 = b.nonzero()
            x2 = b.nonzero()
            if x1.shape[0] != x2.shape[0]:
                raise AssertionError("x1.shape[0] should equal x2.shape[0]")
            ok = True
            b.normal_()
            y = b.nonzero()
            try:
                bool(x1.shape[0] == y.shape[0])
                self.fail("didn't raise exception")
            except GuardOnDataDependentSymNode:
                pass

        make_fx(f, tracing_mode="symbolic")(torch.randn(4))

    @torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True)
    def test_invalidate_nonzero_propagate_real_tensors(self):
        # [FUNCTION DOC] Same scenario as the previous nonzero test but with the config option
        # propagating real tensors enabled: after in place mutation the new nonzero size is
        # simply asserted equal to the old one.
        def f(a):
            # [FUNCTION DOC] HELPER: Calls nonzero on a clone thrice, asserts equal sizes,
            # mutates with normal values and asserts the new nonzero size equals the old; raises
            # AssertionError otherwise.
            b = a.clone()
            x = b.nonzero()
            x1 = b.nonzero()
            x2 = b.nonzero()
            if x1.shape[0] != x2.shape[0]:
                raise AssertionError("x1.shape[0] should equal x2.shape[0]")
            b.normal_()
            y = b.nonzero()
            # Because you're not actually going to generate exactly zero with
            # normal_ lol
            if x1.shape[0] != y.shape[0]:
                raise AssertionError("x1.shape[0] should equal y.shape[0]")

        make_fx(f, tracing_mode="symbolic")(torch.randn(4))

    def test_sqrt_size(self):
        # [FUNCTION DOC] Symbolic trace of dividing by the square root of a size; golden string
        # shows sym_float and power ops.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns a divided by the square root of its last dim size.
            return a / a.size(-1) ** 0.5

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    sym_float = torch.sym_float(sym_size_int);  sym_size_int = None
    pow_1 = sym_float ** 0.5;  sym_float = None
    div = torch.ops.aten.div.Tensor(a_1, pow_1);  a_1 = pow_1 = None
    return div""")

    def test_make_fx_with_custom_tracer_preserving_nn_module_stack(self):
        # [FUNCTION DOC] Checks nn_module_stack metadata: absent by default for a nested module,
        # present with record_module_stack and a reparametrizing wrapper (stacks of depth 1 name
        # Foo, depth 2 name Bar, deeper depths fail), and absent again when tracing the wrapper
        # without the option.

        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner module whose forward adds one to its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the base module.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus 1.
                return x + 1

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module holding a Bar submodule; forward adds the
            # submodule output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module with a Bar submodule attribute.
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus the output of the Bar submodule on x.
                return x + self.bar(x)

        gm = make_fx(Foo())(torch.randn(4, 4))
        for node in gm.graph.nodes:
            self.assertTrue("nn_module_stack" not in node.meta)

        foo = Foo()

        def functional_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the Foo module with an empty parameter substitution
            # in effect and returns its output; carries an original-module attribute so the
            # tracer can find the module.
            with stateless._reparametrize_module(foo, {}):
                return foo(*args, **kwargs)

        functional_call._orig_mod = foo

        gm_with_stack = make_fx(functional_call, record_module_stack=True)(torch.randn(4, 4))
        found = False
        for node in gm_with_stack.graph.nodes:
            if "nn_module_stack" in node.meta:
                if len(node.meta["nn_module_stack"]) == 1:
                    self.assertTrue("custom_tracer_preserving_nn_module_stack.<locals>.Foo" in str(node.meta["nn_module_stack"]))
                    found = True
                elif len(node.meta["nn_module_stack"]) == 2:
                    self.assertTrue("preserving_nn_module_stack.<locals>.Bar" in str(node.meta["nn_module_stack"]))
                    found = True
                else:
                    # there can be at most 2 level
                    self.assertTrue(False)

        self.assertTrue(found)

        gm_without_stack = make_fx(functional_call)(torch.randn(4, 4))
        for node in gm_without_stack.graph.nodes:
            self.assertTrue("nn_module_stack" not in node.meta)

    def test_symint_to_tensor(self):
        # [FUNCTION DOC] Symbolic trace of dividing a tensor by its size; golden string with the
        # default table shows a plain div by the size node, and with the full decomposition
        # table shows sym_float and prims div.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns a divided by its first dim size.
            return a / a.shape[0]

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    div = torch.ops.aten.div.Tensor(a_1, sym_size_int);  a_1 = sym_size_int = None
    return div""")

        r = str(make_fx(f, tracing_mode="symbolic", decomposition_table=decomposition_table)(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    sym_float = torch.sym_float(sym_size_int);  sym_size_int = None
    div = torch.ops.prims.div.default(a_1, sym_float);  a_1 = sym_float = None
    return div""")

    def test_cat(self):
        # [FUNCTION DOC] Symbolic trace of concatenation and a branch on the product of the
        # result dims; checks the guard evaluates true for a satisfying shape, false for a small
        # one, and its printed form.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Multiplies a and b, concatenates the result with itself and
            # returns its cosine if the result has more than 20 elements per the two dims, else
            # the concatenation.
            val = torch.mul(a, b)
            out = torch.cat([val, val])
            if out.shape[0] * out.shape[1] > 20:
                out = out.cos()
            return out

        test_inputs = []
        test_inputs.append([(1, 5), (6, 1)])
        test_inputs.append([(1, 4), (3, 1)])
        gm = self._test_dynamic(f, [(1, 6), (8, 1)], test_inputs)
        self.assertTrue(eval_guards(gm, torch.randn(1, 10), torch.randn(6, 1)))
        self.assertFalse(eval_guards(gm, torch.randn(1, 2), torch.randn(4, 1)))
        self.assertExpectedInline(show_guards(gm), """2*L['b'].size()[0]*L['a'].size()[1] > 20""")

    def test_new_empty(self):
        # [FUNCTION DOC] Runs new_empty with sizes derived from another tensor under
        # _test_dynamic without comparing values (shapes only).
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns an uninitialized tensor like a with dims from b
            # first dim and twice its second.
            return a.new_empty(b.shape[0], b.shape[1] * 2)

        self._test_dynamic(f, [(2, 4), (4, 5)], [[(2, 3), (5, 7)], [(3, 7), (9, 3)]], assert_eq=False).shape_env

    def test_size_with_tensor(self):
        # [FUNCTION DOC] Smoke test: new_empty with a list mixing an int, shape entries and
        # tensor-derived sizes works in eager and under symbolic tracing. The source comment
        # doubts the test targets the originally intended error case.
        # I think I messed up writing this test case originally, I think
        # I'm supposed to hit an error case, but the code here works in both
        # eager and tracing
        def f(tensor):
            # [FUNCTION DOC] HELPER: Builds a batch shape list from an int, the leading dims of
            # the input and the entries of a constant int64 tensor, and returns a new empty
            # tensor of that shape.
            max_size = torch.tensor([800, 1216], dtype=torch.int64)
            batch_shape = [2] + list(tensor.shape[:-2]) + list(max_size)
            return tensor.new_empty(batch_shape)

        a = torch.randn(3, 800, 1199)
        f(a)
        make_fx(f, tracing_mode="symbolic")(a)

    def test_fake_tensor_as_size(self):
        # [FUNCTION DOC] Symbolic trace of zeros sized by a tensor input; golden string shows
        # the scalar extraction feeding zeros.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns zeros with shape given by the single value x.
            r = torch.zeros([x])
            return r

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.tensor(4))
        self.assertExpectedInline(fx_g.code.strip(), """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return zeros""")

    def test_expand(self):
        # [FUNCTION DOC] Runs a multiply-then-expand-to-own-shape function under dynamic tracing
        # for several 1D and 2D shapes.
        def f(a):
            # [FUNCTION DOC] HELPER: Squares a elementwise and expands it to the shape of a.
            b = torch.mul(a, a)
            c = b.expand(a.shape)
            return c

        self._test_dynamic(f, [(3,)], [[(3,)], [(4,)], [(2,)]])
        self._test_dynamic(f, [(5, 1)], [[(4, 1)], [(3, 1)], [(6, 1)]])

    def test_metadata(self):
        # [FUNCTION DOC] Checks that the symbolic size of a new_empty node equals the symbolic
        # expression of the add node of two sizes, i.e. metadata expressions are shared.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns an uninitialized vector sized by the sum of the
            # first dims of a and b.
            d = a.new_empty(a.shape[0] + b.shape[0])
            return d
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(5), torch.randn(4))
        meta_c = _get_node(fx_g, lambda x: x.target == aten.new_empty.default)
        meta_d = _get_node(fx_g, lambda x: x.target == operator.add)
        self.assertTrue(meta_c.meta['val'].shape[0].node.expr == meta_d.meta['val'].node.expr)

    def test_metadata_fresh(self):
        # [FUNCTION DOC] Checks that after a later constraint pins the first dim to 3, both the
        # output node and the placeholder metadata show the dimension as 3.
        def f(x):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless the first dim of x is 3;
            # returns cos of x.
            if x.shape[0] != 3:
                raise AssertionError(f"expected x.shape[0] == 3, got {x.shape[0]}")
            return x.cos()

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(3))
        meta_cos = _get_node(fx_g, lambda x: x.target == aten.cos.default)
        meta_inp = _get_node(fx_g, lambda x: x.op == 'placeholder')
        self.assertTrue(meta_cos.meta['val'].shape[0] == 3)
        # Checks if the input expr has been updated even though the constraint
        # happened afterwards
        self.assertTrue(meta_inp.meta['val'].shape[0] == 3)

    def test_elementwise_meta_with_sym_numbers(self):
        # [FUNCTION DOC] Checks metadata of an add node combining a size or float-cast size with
        # an offset: the result is scalar shaped, float32 for float offset, int64 for int offset
        # with int size, float32 when the size is cast to float.
        def f(x, offset, as_sym_float=False):
            # [FUNCTION DOC] HELPER: Takes the first dim size of x, optionally converts it to a
            # symbolic float, and returns its sum with offset.
            x0 = x.size()[0]
            if as_sym_float:
                x0 = torch.sym_float(x0)
            return torch.add(x0, offset)

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.rand(2, 3), 2.0, False)
        meta_add = _get_node(fx_g, lambda x: x.target == aten.add.Tensor)
        self.assertEqual(meta_add.meta['val'].shape, ())
        self.assertEqual(meta_add.meta['val'].dtype, torch.float32)

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.rand(2, 3), 2, False)
        meta_add = _get_node(fx_g, lambda x: x.target == aten.add.Tensor)
        self.assertEqual(meta_add.meta['val'].shape, ())
        self.assertEqual(meta_add.meta['val'].dtype, torch.int64)

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.rand(2, 3), 2, True)
        meta_add = _get_node(fx_g, lambda x: x.target == aten.add.Tensor)
        self.assertEqual(meta_add.meta['val'].shape, ())
        self.assertEqual(meta_add.meta['val'].dtype, torch.float32)

    def test_return_symint(self):
        # [FUNCTION DOC] Runs functions returning symbolic ints and the whole shape under
        # dynamic tracing.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first size, the cosine of x and the first size
            # divided by 5.
            return x.shape[0], x.cos(), x.shape[0] / 5
        self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])

        def f(x):
            # [FUNCTION DOC] HELPER: Redefinition returning the shape of x.
            return x.shape
        self._test_dynamic(f, [(5, 3)], [[(4, 6)]])

    def test_rmethod(self):
        # [FUNCTION DOC] Dynamic trace of a reflected add (integer size plus tensor).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first size of x plus x.
            return x.size(0) + x
        self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])

    def test_mega_guard(self):
        # [FUNCTION DOC] Traces with an equality constraint of one size being twice another;
        # golden strings list all produced guards including strides and storage offsets, and the
        # reduced list when static guards are ignored.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless the first dim of a is twice
            # that of b; returns cos of a.
            if a.shape[0] != b.shape[0] * 2:
                raise AssertionError("a.shape[0] should equal b.shape[0] * 2")
            return a.cos()
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(16), torch.randn(8))
        from torch._dynamo.source import LocalSource
        self.assertExpectedInline(
            str(fx_g.shape_env.produce_guards(fx_placeholder_vals(fx_g), [LocalSource("a"), LocalSource("b")], ignore_static=False)),
            """["L['a'].size()[0] == 2*L['b'].size()[0]", "L['a'].stride()[0] == 1", "L['a'].storage_offset() == 0", "L['b'].stride()[0] == 1", "L['b'].storage_offset() == 0", "2 <= L['b'].size()[0]"]"""
        )
        self.assertExpectedInline(
            str(fx_g.shape_env.produce_guards(fx_placeholder_vals(fx_g), [LocalSource("a"), LocalSource("b")], ignore_static=True)),
            """["L['a'].size()[0] == 2*L['b'].size()[0]", "2 <= L['b'].size()[0]"]"""
        )

    def test_guard_upperbound_range_refinement(self):
        # [FUNCTION DOC] Checks branch conditions of two lower thresholds on a size are merged
        # into the tightest guard. NOTE: the name says upper bound but the checked refinement is
        # a lower bound.
        def f(a):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless the first dim of a exceeds 12;
            # returns cos of a.
            if not (a.shape[0] > 5 and a.shape[0] > 12):
                raise AssertionError("a.shape[0] should be > 12")
            return a.cos()
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn(15))
        self.assertExpectedInline(show_guards(tensor), """13 <= L['a'].size()[0]""")

    def test_guard_lowerbound_range_refinement(self):
        # [FUNCTION DOC] Checks branch conditions of two upper thresholds on a size are merged
        # into the tightest guard. NOTE: the name says lower bound but the checked refinement is
        # an upper bound.
        def f(a):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless the first dim of a is below
            # 20; returns cos of a.
            if not (a.shape[0] < 20 and a.shape[0] < 30):
                raise AssertionError("a.shape[0] should be < 20")
            return a.cos()
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn(15))
        self.assertExpectedInline(show_guards(tensor), """L['a'].size()[0] <= 19""")

    def test_guard_upperbound_range_refinement_multivariate(self):
        # [FUNCTION DOC] As the single-variable lower-bound test, with an extra relation between
        # the second and first dims; golden guards show the refined bounds and the relation. The
        # name says upper bound.
        def f(a):
            # [FUNCTION DOC] HELPER: Requires first dim above 12 and second dim above the first,
            # raising AssertionError otherwise; returns cos of a.
            if not (a.shape[0] > 5 and a.shape[0] > 12):
                raise AssertionError("a.shape[0] should be > 12")
            if not (a.shape[1] > 5 and a.shape[1] > a.shape[0]):
                raise AssertionError("a.shape[1] should be > a.shape[0]")
            return a.cos()
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn((15, 20)))
        self.assertExpectedInline(show_guards(tensor), """\
L['a'].size()[1] > L['a'].size()[0]
13 <= L['a'].size()[0]
14 <= L['a'].size()[1]""")

    def test_guard_lowerbound_range_refinement_multivariate(self):
        # [FUNCTION DOC] As the single-variable upper-bound test, with an extra relation between
        # the second and first dims; golden guards show the relation and refined bounds. The
        # name says lower bound.
        def f(a):
            # [FUNCTION DOC] HELPER: Requires first dim below 20 and second dim below the first,
            # raising AssertionError otherwise; returns cos of a.
            if not (a.shape[0] < 20 and a.shape[0] < 30):
                raise AssertionError("a.shape[0] should be < 20")
            if not (a.shape[1] < 30 and a.shape[1] < a.shape[0]):
                raise AssertionError("a.shape[1] should be < a.shape[0]")
            return a.cos()
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn((15, 5)))
        self.assertExpectedInline(
            show_guards(tensor),
            """\
L['a'].size()[1] < L['a'].size()[0]
3 <= L['a'].size()[0] and L['a'].size()[0] <= 19
L['a'].size()[1] <= 18""")

    def test_sym_storage_offset(self):
        # [FUNCTION DOC] Traces an add of a sliced (nonzero storage offset) tensor and a plain
        # one under symbolic mode and checks the traced graph on fresh similar inputs equals
        # eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        inp = (torch.randn(8)[3:], torch.randn(5))
        fx_g = make_fx(f, tracing_mode="symbolic")(*inp)
        inp = (torch.randn(8)[3:], torch.randn(5))
        self.assertEqual(fx_g(*inp), f(*inp))

    def _assert_no_guards(self, fx_g, free_symbols):
        # [FUNCTION DOC] HELPER: Asserts the traced graph module has the expected number of free
        # (unreplaced) backed symbols and no non-trivial guards; raises AssertionError with
        # details otherwise.
        if _get_free_symbols(fx_g.shape_env) != free_symbols:
            raise AssertionError(f"free symbols mismatch: {fx_g.shape_env.backed_var_to_val}")
        if len(fx_g.shape_env.get_nontrivial_guards()) != 0:
            raise AssertionError(f"expected no guards: {fx_g.shape_env.format_guards()}")

    def test_guards_equal(self):
        # [FUNCTION DOC] Traces several functions over differently related input sizes (equal,
        # broadcast, concat sums, view to another tensor size) with numbers chosen to avoid duck
        # shaping, and uses _assert_no_guards to check the expected symbol counts and absence of
        # guards.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns the product of a and b.
            return a * b

        # NB: Numbers are carefully chosen to avoid duck shaping from applying

        fx_g = _trace(f, (5, 6), (5, 6))
        self._assert_no_guards(fx_g, 2)

        fx_g = _trace(f, (5, 6, 7), (5, 6, 7))
        self._assert_no_guards(fx_g, 3)

        fx_g = _trace(f, (5, 1), (1, 6))
        self._assert_no_guards(fx_g, 2)

        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Adds a and b, concatenates c and d, and returns their sum.
            a = a + b
            cat = torch.cat([c, d])
            return a + cat

        fx_g = _trace(f, 7, 7, 4, 3)
        self._assert_no_guards(fx_g, 2)

        def f(a, b, c, d, e):
            # [FUNCTION DOC] HELPER: Folds five tensors by repeatedly concatenating the running
            # value with one input and adding the next, returning the final tensor.
            vals = [a, b, c, d, e]
            x = a
            for idx in range(len(vals) - 1):
                x = torch.cat([x, vals[idx]]) + vals[idx + 1]
            return x

        fx_g = _trace(f, 2, 4, 8, 16, 32)
        self._assert_no_guards(fx_g, 1)

        def f(a, b):
            # [FUNCTION DOC] HELPER: Views a to the size of b first dim and adds the total sum
            # of b.
            a = a.view(b.shape[0])
            return a + b.sum()

        fx_g = _trace(f, (4, 2), 8)
        self._assert_no_guards(fx_g, 2)

        fx_g = _trace(f, (4, 2), (8, 5))
        self._assert_no_guards(fx_g, 3)

        fx_g = _trace(f, (2, 3, 4), 24)
        self._assert_no_guards(fx_g, 3)

    def test_nonidentity_transitive_guards(self):
        # [FUNCTION DOC] Traces a chain of concatenations and adds over five vectors whose sizes
        # double, and asserts that the shape guards printed are empty.
        def f(a, b, c, d, e):
            # [FUNCTION DOC] HELPER: Concatenates each of the first four inputs with itself and
            # adds each result to the next input, returning the sums in reverse order.
            vals = [a, b, c, d, e]
            cat_vals = []
            for idx in range(len(vals) - 1):
                cat_vals.append(torch.cat([vals[idx], vals[idx]]))
            final_vals = []
            for a, b in reversed(list(zip(cat_vals, vals[1:]))):
                final_vals.append(a + b)
            return final_vals

        fx_g = _trace(f, 2, 4, 8, 16, 32)
        self.assertExpectedInline(show_guards(fx_g), """""")

    @torch.fx.experimental._config.patch(translation_validation=True)
    def test_constant_specialization(self):
        # [FUNCTION DOC] With translation validation on, traces a function requiring a size of
        # ten and checks the printed guards are empty (the size is specialized as a constant).
        def f(t):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless the first dim of t is 10;
            # returns t.
            if t.shape[0] != 10:
                raise AssertionError(f"expected t.shape[0] == 10, got {t.shape[0]}")
            return t

        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn(10))
        self.assertExpectedInline(show_guards(tensor), """""")

    @skipIfTorchDynamo("make_fx cannot trace dynamo-optimized functions")
    def test_make_fx_second_order_grad(self):
        # [FUNCTION DOC] Regression test for issue 175477. Skipped under Dynamo. Traces a
        # function computing an energy and its force via double-differentiation: with the
        # default table no detach op remains in the graph, grads through the traced function
        # match eager, and an empty explicit decomposition table or pre-dispatch tracing keeps
        # detach nodes.
        # Regression test for https://github.com/pytorch/pytorch/issues/175477
        weight = torch.randn(4, 3, requires_grad=True)

        def fn(x, weight):
            # [FUNCTION DOC] HELPER: Detaches x, applies a linear transform and normalizes it,
            # sums squares as energy and returns the energy and the negative gradient of energy
            # with respect to x.
            x = x.detach().requires_grad_(True)
            h = x @ weight.T
            mean = h.mean(-1, keepdim=True)
            var = h.var(-1, keepdim=True, correction=0)
            h = (h - mean) / torch.sqrt(var + 1e-5)
            energy = (h ** 2).sum()
            force = -torch.autograd.grad(energy, x, create_graph=True)[0]
            return energy, force

        traced = make_fx(fn, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
            torch.randn(3, 3), weight.clone().requires_grad_(True)
        )
        self.assertFalse(
            any(
                n.op == "call_function" and n.target == torch.ops.aten.detach.default
                for n in traced.graph.nodes
            )
        )

        x = torch.randn(3, 3)
        target_energy = torch.randn(())
        target_force = torch.randn(3, 3)

        def loss_fn(e, f):
            # [FUNCTION DOC] HELPER: Returns the squared error of energy plus the summed squared
            # error of force against captured targets.
            return (e - target_energy).pow(2) + (f - target_force).pow(2).sum()

        w1 = weight.detach().clone().requires_grad_(True)
        e1, f1 = fn(x, w1)
        loss_fn(e1, f1).backward()

        w2 = weight.detach().clone().requires_grad_(True)
        e2, f2 = traced(x, w2)
        loss_fn(e2, f2).backward()

        self.assertEqual(w1.grad, w2.grad)

        traced_explicit = make_fx(
            fn,
            decomposition_table={},
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(torch.randn(3, 3), weight.clone().requires_grad_(True))
        self.assertTrue(
            any(
                n.op == "call_function" and n.target == torch.ops.aten.detach.default
                for n in traced_explicit.graph.nodes
            )
        )

        def detach_only(x):
            # [FUNCTION DOC] HELPER: Returns x detached and multiplied by 2.
            return x.detach() * 2

        traced_pre_dispatch = make_fx(detach_only, pre_dispatch=True)(
            torch.randn(2, requires_grad=True)
        )
        self.assertTrue(
            any(
                n.op == "call_function" and n.target == torch.ops.aten.detach.default
                for n in traced_pre_dispatch.graph.nodes
            )
        )


make_fx_failures = {
    # unknown
    xfail('allclose'),
    xfail('equal'),
    # empty
    skip('new_empty'),
    skip('empty_like'),
    skip('empty'),
    skip('empty_permuted'),
    # flaky
    skip('linalg.lstsq', 'grad_oriented'),
    skip('nn.functional.max_unpool1d', '', device_type='cpu'),
    skip('nn.functional.max_unpool2d', '', device_type='cpu'),
    skip('nn.functional.max_unpool3d', '', device_type='cpu'),
    skip('linalg.lstsq'),  # flaky, probably just a precision issue

    # data-dependent control flow
    skip('item'),
    xfail('cov'),
    xfail('nn.functional.gaussian_nll_loss'),
    xfail('corrcoef'),

    # Seems like it's creating a sparse tensor that isn't captured by tensor.is_sparse
    xfail('sparse.sampled_addmm'),
    xfail('sparse.mm', 'reduce'),

    # proxy tensor doesn't support sparse correctly right now
    skip('to_sparse'),
    # segfaults
    skip('block_diag'),

    # AssertionError: Tensor-likes are not close!
    skip('empty_strided', '', device_type='cpu'),
}

only_real_tensor_failures = {
    xfail('narrow'),
    xfail('tensor_split'),
}

only_fake_tensor_failures = {
    xfail('tensor_split'),
}

fake_tensor_failures = set()

symbolic_tensor_failures = {
    xfail('geqrf', ''),  # aten.geqrf.default - couldn't find symbolic meta function/decomposition
    xfail('histogram', ''),  # Could not run 'aten::histogram.bin_ct' with arguments from the 'Meta' backend. This c...
    xfail('histogramdd', ''),  # aten._histogramdd_bin_edges.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.ctc_loss'),  # aten._ctc_loss.Tensor - couldn't find symbolic meta function/decomposition

    xfail('max_pool2d_with_indices_backward', ''),  # Expected a value of type 'List[int]' for argument 'kernel_size' but...
}
symbolic_tensor_segfaults = {
    skip('nn.functional.batch_norm')  # Segfault??
}

symbolic_tensor_failures.update(symbolic_tensor_segfaults)

inplace_symbolic_tensor_failures = {
    # bugs
    xfail('float_power', ''),  # base given to float_power_ has dtype Float but the operation's result requires dtype Double
}

out_symbolic_tensor_failures = {
    # Cast error details: Unable to cast (...) to Tensor
    #
    # This happens because the test is set up to call the out variant using the `out` kwarg:
    #   torch._some_op(arg1, arg2, out=(out1, out2, out3))
    #
    # However, this only works on torch ops, not aten ops. For `_batch_norm_with_update`,
    # this fails because the op has no python bindings, so it doesn't support the `out` kwarg
    # way of calling its out variant.
    xfail('_batch_norm_with_update', ''),
    xfail('_native_batch_norm_legit', ''),
    xfail('angle', ''),
    xfail('argmax', ''),
    xfail('argmin', ''),
    xfail('gather', ''),
    xfail('scatter_add', ''),
    xfail('scatter', ''),
    xfail('take_along_dim', ''),

    # SymIntArrayRef expected to contain only concrete
    xfail('randn', ''),

    # RuntimeError: Cannot call numel() on tensor with symbolic sizes/strides
    xfail('index_reduce', 'prod'),
    xfail('index_reduce', 'mean'),
    xfail('index_reduce', 'amax'),
    xfail('index_reduce', 'amin'),
}

out_symbolic_tensor_segfaults = {
    skip('nanmean', ''),
}

out_symbolic_tensor_failures.update(out_symbolic_tensor_segfaults)

# Copies inputs to inplace operations to avoid inplace modifications
#   to leaves requiring gradient
def _get_safe_inplace(inplace_variant):
    # [FUNCTION DOC] HELPER: Wraps an in place op variant in a function that first clones its
    # first argument, so the original leaf requiring grad is not mutated. Returns the wrapper.
    @functools.wraps(inplace_variant)
    def _fn(t, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the wrapped in place variant on a clone of t.
        return inplace_variant(t.clone(), *args, **kwargs)

    return _fn

def _test_make_fx_helper(self, device, dtype, op, tracing_mode, inplace=False, out=False):
    # [FUNCTION DOC] HELPER: Shared body of OpInfo tests. Picks the op or its in place or out
    # variant, takes up to 100 samples (5 for out), skips broadcast-input samples for in place,
    # runs make_fx_check with the given tracing mode and randomized data, and skips the test if
    # the trace hits a dynamic output shape.
    fn = _get_safe_inplace(op.get_inplace()) if inplace else op.op
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

    # Limit ourselves to first 100 inputs so symbolic tracing tests don't take too long
    count = 100
    if out:
        count = 5
    for sample_input in itertools.islice(sample_inputs_itr, count):
        if inplace and sample_input.broadcasts_input:
            continue
        args = [sample_input.input] + list(sample_input.args)
        kwargs = sample_input.kwargs
        if out:
            expected = fn(*args, **kwargs)
            kwargs['out'] = expected

        try:
            optests.make_fx_check(fn, args, kwargs, tracing_mode, self.assertEqual,
                                  randomize_data=True)
        except DynamicOutputShapeException:
            self.skipTest("Dynamic output shape operation in trace")


def skipIfNameMatches(pattern):
    # [FUNCTION DOC] HELPER: Decorator factory returning a decorator that skips a test when the
    # name of the decorated function matches the given regular expression, by raising SkipTest
    # at call time.
    """
    Decorator to skip a test if its name matches the given pattern.
    """
    def decorator(test_func):
        # [FUNCTION DOC] HELPER: Wraps test_func in a function that skips when its name matches
        # the pattern.
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Raises SkipTest if the wrapped function name matches the
            # pattern; otherwise calls the wrapped function with the same arguments and returns
            # its result.
            if re.match(pattern, test_func.__name__):
                raise unittest.SkipTest(f"Test '{test_func.__name__}' skipped because its name matches the pattern '{pattern}'")
            return test_func(*args, **kwargs)
        return wrapper
    return decorator

# Auto functionalize shouldn't work with make_fx directly
filtered_hop_db = [op for op in hop_db if op.name != "auto_functionalize"]

@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "Cond requires dynamo")
class TestProxyTensorOpInfo(TestCase):
    # [CLASS DOC] OpInfo-driven make_fx tests, skipped unless dynamo is supported. Each test
    # runs the shared helper over float OpInfo ops in real, fake or symbolic mode, with xfail
    # and skip tables applied; the class is instantiated for cpu only.
    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(make_fx_failures.union(only_real_tensor_failures))
    def test_make_fx_exhaustive(self, device, dtype, op):
        # [FUNCTION DOC] OpInfo test tracing each op in real mode, with the common failures and
        # real-only failures skipped.
        _test_make_fx_helper(self, device, dtype, op, "real")

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(make_fx_failures.union(fake_tensor_failures, only_fake_tensor_failures))
    def test_make_fx_fake_exhaustive(self, device, dtype, op):
        # [FUNCTION DOC] OpInfo test tracing each op in fake mode, with common, fake and fake-
        # only failures skipped.
        _test_make_fx_helper(self, device, dtype, op, "fake")

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(make_fx_failures | fake_tensor_failures | symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive(self, device, dtype, op):
        # [FUNCTION DOC] OpInfo test tracing each op in symbolic mode, with common, fake and
        # symbolic failures skipped.
        _test_make_fx_helper(self, device, dtype, op, "symbolic")

    @ops(op_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(make_fx_failures | fake_tensor_failures | symbolic_tensor_failures | inplace_symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive_inplace(self, device, dtype, op):
        # [FUNCTION DOC] OpInfo test tracing the in place variant of each op symbolically; skips
        # ops without one.
        if not op.get_inplace():
            self.skipTest("No inplace variable for this op")
        _test_make_fx_helper(self, device, dtype, op, "symbolic", inplace=True)

    @ops(op_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps(make_fx_failures | fake_tensor_failures | symbolic_tensor_failures | out_symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive_out(self, device, dtype, op):
        # [FUNCTION DOC] OpInfo test tracing the out variant of each op symbolically; skips ops
        # without out support.
        if not op.supports_out:
            self.skipTest("Op doesn't support out")
        _test_make_fx_helper(self, device, dtype, op, "symbolic", out=True)


only_for = ("cpu")
instantiate_device_type_tests(TestProxyTensorOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
