# Owner(s): ["module: dynamo"]
# ruff: noqa: F841
# [MODULE DOC] Large grab-bag of Dynamo (torch.compile) regression tests. Defines the MiscTests
# TestCase plus many smaller test classes, helper modules, and module-level fixtures (pytree
# module table, a namedtuple, fake-quant configs). Import time work: builds the pytree module
# mapping (python, plus cxx/optree when traceable), the parametrize decorator over it, and the
# quantization config objects.
import abc
import builtins
import collections
import collections.abc
import copy
import dataclasses
import dis
import enum
import functools
import gc
import importlib
import itertools
import json
import logging
import math
import operator
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import threading
import traceback
import types
import typing
import unittest
import unittest.mock as mock
import warnings
import weakref
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils._pytree as python_pytree
import torch.utils.cpp_extension
from torch import Tensor
from torch._C import FileCheck
from torch._dynamo import allow_in_graph
from torch._dynamo.comptime import comptime
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch._dynamo.exc import Unsupported
from torch._dynamo.source import ConstantSource, GetItemSource, LocalSource
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    expectedFailureDynamic,
    same,
    skipIfNotPy311,
    unsupported,
)
from torch._dynamo.utils import call_size, counters, ifdynstaticdefault
from torch._dynamo.variables import builder
from torch._inductor.codecache import WritableTempFile
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.ao.quantization import MinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.fx.experimental.recording import NotEqualError, replay_shape_env_events
from torch.fx.experimental.symbolic_shapes import (
    _constrain_range_for_size,
    constrain_range,
    constrain_unify,
    ConstraintViolationError,
    expect_true,
    guard_or_false,
    guard_size_oblivious,
    ShapeEnv,
)
from torch.nn import functional as F
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    SM80OrLater,
    TEST_CUDA,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    PYTORCH_CUDA_MEMCHECK,
)
from torch.testing._internal.common_methods_invocations import (
    sample_inputs_take_along_dim,
)
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    recover_orig_fp32_precision,
    scoped_load_inline,
    set_default_dtype,
    skipCUDAMemoryLeakCheckIf,
    skipIfHpu,
    skipIfNNModuleInlined,
    skipIfWindows,
    subtest,
    TEST_HPU,
    TEST_XPU,
    wrapDeterministicFlagAPITest,
)
from torch.testing._internal.jit_utils import JitTestCase


pytree_modules = {
    "python": python_pytree,
}
if python_pytree._cxx_pytree_dynamo_traceable:
    import torch.utils._cxx_pytree as cxx_pytree

    pytree_modules["cxx"] = cxx_pytree
    pytree_modules["native_optree"] = cxx_pytree.optree
else:
    cxx_pytree = None

parametrize_pytree_module = parametrize(
    "pytree",
    [subtest(module, name=name) for name, module in pytree_modules.items()],
)

MyTuple = collections.namedtuple("MyTuple", ["a", "b", "ab"])
T = typing.TypeVar("T")


# Defined in CPython's Include/object.h
TPFLAGS_MAPPING = 1 << 6

GLOBAL_INT = 1


# Module-level function used as a torch._check message (UserFunctionVariable,
# as opposed to a nested closure defined inside the compiled region).
def global_check_message():
    # [FUNCTION DOC] HELPER: Returns the fixed string used as a torch._check failure message,
    # so tests can exercise a module-level function (rather than a nested closure) as the
    # message callable.
    return "global check message"


# Specializes a test to run only if translation validation is set.
def onlyIfTranslationValidation(fn: typing.Callable) -> typing.Callable:
    # [FUNCTION DOC] HELPER: Decorator factory for tests. Returns a wrapper around the given test
    # function that only runs it when translation validation is enabled.
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper produced by the decorator. Imports the fx validator
        # module and, if translation validation is enabled, forwards all arguments to the wrapped
        # test and returns its result. Otherwise raises unittest.SkipTest.
        import torch.fx.experimental.validator

        if torch.fx.experimental.validator.translation_validation_enabled():
            return fn(*args, **kwargs)
        raise unittest.SkipTest(f"only works when TV is True.")

    return wrapper


class MyPickledModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Small nn.Module used by pickling tests. Stores one extra value z and
    # computes the cube of the first input plus the second input plus z.
    def __init__(self, z):
        # [FUNCTION DOC] HELPER: Calls the Module base initializer and stores the given value as
        # attribute z.
        super().__init__()
        self.z = z

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Returns the cube of x added to y and to the stored z attribute.
        return x * x * x + y + self.z


# These are used for test_{cond/map}_with_quantization
default_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=torch.per_tensor_symmetric, dtype=torch.quint8
)
default_weight_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
)
uniform_qconfig_8bit = QConfig(
    activation=default_symmetric_fake_quant,
    weight=default_weight_symmetric_fake_quant.with_args,
)
qconfig_dict = {"object_type": [(torch.nn.Linear, uniform_qconfig_8bit)]}


def closure_adder(val):
    # [FUNCTION DOC] HELPER: Builds and returns a closure over val that computes the sine of its
    # tensor argument shifted by val.
    def inner(x):
        # [FUNCTION DOC] HELPER: Closure body: returns the sine of x plus the captured val.
        return torch.sin(x + val)

    return inner


class UserDefineSetAttr:
    # [CLASS DOC] HELPER: Fixture class with custom attribute hooks that store and read every
    # attribute under a name with a pfx_ prefix. The class flag setup must be true, or Dynamo
    # must be compiling, for the hooks to be callable; otherwise an assertion fails.
    setup = False

    def __setattr__(self, key, value):
        # [FUNCTION DOC] HELPER: Asserts that Dynamo is compiling or the class flag setup is set,
        # then stores the value under the prefixed key via the base class setter.
        assert torch.compiler.is_dynamo_compiling() or UserDefineSetAttr.setup  # noqa: S101
        super().__setattr__(f"pfx_{key}", value)

    def __getattr__(self, key, c=1):
        # [FUNCTION DOC] HELPER: Asserts the same compile-or-setup condition. Looks up the
        # prefixed key in the instance dict (KeyError if absent) when c is truthy, else returns
        # None. The c default exists only to force a guard on the function defaults.
        assert torch.compiler.is_dynamo_compiling() or UserDefineSetAttr.setup  # noqa: S101
        # c is added to force a guard on __defaults__ and checks the source for __getattr__
        if c:
            return self.__dict__[f"pfx_{key}"]
        else:
            return None


class MiscTests(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Main collection of miscellaneous Dynamo tests, built on the inductor TestCase.
    # Each method defines small functions or modules, compiles them with various backends and
    # options, and checks results, graph counts, guards, counters, or raised errors.
    def test_get_cache_entry(self):
        # [FUNCTION DOC] Checks the debug helper that lists cache entries of a code object: a
        # compiled function has entries, an uncompiled one has none, a non-code argument raises
        # TypeError mentioning the expected code object, and a skipped (graph break) code object
        # has zero entries.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        torch.compile(f, backend="eager")(torch.randn(5, 5, 5))
        entries = _debug_get_cache_entry_list(f)
        self.assertTrue(len(entries) > 0)

        def g(x):
            # [FUNCTION DOC] HELPER: Returns x plus 2; never compiled.
            return x + 2

        entries = _debug_get_cache_entry_list(g)
        self.assertTrue(len(entries) == 0)

        try:
            _debug_get_cache_entry_list(1)
        except TypeError as e:
            self.assertIn("expected a code object!", str(e))

        # test get cache entry on skipped code object
        def h(x):
            # [FUNCTION DOC] HELPER: Adds 1, forces a graph break, and adds 1 again; used to
            # check a skipped function has no cache entries.
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1

        torch.compile(h, backend="eager")(torch.randn(3, 3))

        entries = _debug_get_cache_entry_list(torch._dynamo.graph_break)
        self.assertEqual(len(entries), 0)

    @torch.testing._internal.common_utils.scoped_load_inline
    def test_pybind11_enum_conversion(self, load_inline):
        # [FUNCTION DOC] Builds a C++ extension inline with a pybind11 enum, then checks that
        # int, float, and list indexing conversions of an enum member work under fullgraph eager
        # compilation (values 0, 0.0 and 20).
        cpp_source = """
        #include <torch/extension.h>

        enum class E { A = 0, B = 1 };

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            py::enum_<E>(m, "E")
                .value("A", E::A)
                .value("B", E::B);
        }
        """
        mod = load_inline(name="pybind11_enum_test", cpp_sources=cpp_source)
        e = mod.E.A
        self.assertEqual(
            torch.compile(lambda x: int(x), backend="eager", fullgraph=True)(e), 0
        )
        self.assertEqual(
            torch.compile(lambda x: float(x), backend="eager", fullgraph=True)(e), 0.0
        )
        self.assertEqual(
            torch.compile(lambda x: [10, 20][x], backend="eager", fullgraph=True)(
                mod.E.B
            ),
            20,
        )

    def test_boolarg(self):
        # [FUNCTION DOC] Checks that a function with a boolean flag argument recompiles per
        # distinct flag value (True, False, None) with optimize_assert, reuses the cache for a
        # repeated True, and gives results matching eager; expects 3 frames.
        def boolarg(aa, bb, flag):
            # [FUNCTION DOC] HELPER: Returns difference of aa and bb if flag is truthy, otherwise
            # bb minus aa.
            if flag:
                return aa - bb
            else:
                return bb - aa

        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        correct1 = boolarg(a, b, True)
        correct2 = boolarg(a, b, False)
        correct3 = boolarg(a, b, None)
        counter = CompileCounter()
        opt_boolarg = torch._dynamo.optimize_assert(counter)(boolarg)
        val1 = opt_boolarg(a, b, True)
        val2 = opt_boolarg(a, b, False)
        val3 = opt_boolarg(a, b, None)
        val4 = opt_boolarg(a, b, True)
        self.assertTrue(same(val1, correct1))
        self.assertTrue(same(val2, correct2))
        self.assertTrue(same(val3, correct3))
        self.assertTrue(same(val4, correct1))
        self.assertEqual(counter.frame_count, 3)

    @unittest.skipIf(not TEST_CUDA, "cuda needed")
    def test_assume_32_bit_indexing(self):
        # [FUNCTION DOC] CUDA-only test. Compiles with inductor a function with concat, reshape,
        # slice and cumsum on unbacked-dimension inputs, and checks generated code uses int64
        # index arithmetic by default and does not when the assume_32bit_indexing inductor option
        # is on.
        @torch.compile(backend="inductor")
        def func(a, b):
            # [FUNCTION DOC] HELPER: Concatenates a and b along two dims, flattens, truncates
            # to the smaller length, adds and scales by 10, and returns the sum of the
            # cumulative sum.
            # Multiple concat operations
            x = torch.concat([a, b], dim=0)
            y = torch.concat([a, b], dim=1)

            # Reshape to create indexing patterns
            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)

            # Take the smaller one and expand
            min_size = min(x_flat.shape[0], y_flat.shape[0])
            x_trunc = x_flat[:min_size]
            y_trunc = y_flat[:min_size]

            # Combine and compute
            result = (x_trunc + y_trunc) * 10

            # Cumulative operations create complex indexing
            cumsum = result.cumsum(dim=0)

            return cumsum.sum()

        a = torch.rand(100, 30, device="cuda")
        b = torch.rand(100, 30, device="cuda")

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(a, 1)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        torch._dynamo.decorators.mark_unbacked(b, 1)

        source_code = run_and_get_code(func, a, b)[1]
        # Check that int64 indexing is used (either 1D [:] or 2D [:, None] form)
        self.assertTrue(
            "tl.arange(0, XBLOCK)[:].to(tl.int64)" in str(source_code)
            or "tl.arange(0, XBLOCK)[:, None].to(tl.int64)" in str(source_code)
        )
        # Check that 32-bit indexing is NOT used
        self.assertFalse(
            "tl.arange(0, XBLOCK)[:]\n" in str(source_code)
            and ".to(tl.int64)" not in str(source_code)
        )

        torch._dynamo.reset()

        with torch._inductor.config.patch(assume_32bit_indexing=True):
            source_code = run_and_get_code(func, a, b)[1]
            # Check that int64 indexing is NOT used when assume_32bit_indexing=True
            self.assertFalse(
                "tl.arange(0, XBLOCK)[:].to(tl.int64)" in str(source_code)
                or "tl.arange(0, XBLOCK)[:, None].to(tl.int64)" in str(source_code)
            )

    def test_dynamo_side_effect(self):
        # [FUNCTION DOC] Checks that an object created and mutated inside a compiled region (a
        # context manager holding a tensor dict) is correctly materialized on output, with side-
        # effect replay disabled and the warn policy.
        class GlobalContext:
            # [CLASS DOC] HELPER: Context manager holding a dict of tensors.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the empty tensor dictionary.
                self._tensors = {}

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self

            def __exit__(self, exc_type, exc, tb):
                # [FUNCTION DOC] HELPER: Does not suppress exceptions; returns False.
                return False

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward builds a context object, stores x plus 2
            # in it, and returns x plus 1 and the context.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes z and mutates the context's tensor dict,
                # returning both z and the context.
                with GlobalContext() as ctx:
                    z = x + 1
                    ctx._tensors["6"] = x + 2
                return z, ctx

        mod = Module()
        inp = torch.randn(4, 4)
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="warn"
        ):
            val = torch.compile(mod, backend="eager")(inp)
            # Verify new object is properly initialized
            self.assertIn("6", val[1]._tensors)
            self.assertEqual(val[1]._tensors["6"], inp + 2)

    def test_dynamo_inside_custom_op(self):
        # [FUNCTION DOC] Checks that torch.compile can be called inside the implementation of a
        # custom op defined in a scoped library, including with an outer compile calling that op.
        # Verifies results and that the inductor graphs recorded for outer and inner compile
        # match expected text.
        cnt = torch._dynamo.testing.InductorAndRecordGraphs()
        cnt1 = torch._dynamo.testing.InductorAndRecordGraphs()

        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x) -> Tensor")

            def inner(x):
                # [FUNCTION DOC] HELPER: Returns cosine of sine of x.
                return x.sin().cos()

            def foo_impl(x):
                # [FUNCTION DOC] HELPER: Custom op implementation that compiles inner with
                # fullgraph and dynamic shapes and calls it.
                return torch.compile(inner, fullgraph=True, dynamic=True, backend=cnt)(
                    x
                )

            m.impl("foo", foo_impl, "CompositeExplicitAutograd")

            @torch.compile(fullgraph=True, dynamic=True, backend=cnt1)
            def f(x):
                # [FUNCTION DOC] HELPER: Outer compiled function calling the custom op foo on x.
                return torch.ops.mylib.foo.default(x)

            x = torch.randn(3)
            res = f(x)
            res1 = f(x)
            res2 = f(x)
            expected = x.sin().cos()
            self.assertEqual(res, expected)
            self.assertEqual(res1, expected)
            self.assertEqual(res2, expected)
            self.assertTrue(len(cnt.inductor_graphs), 1)
            self.assertTrue(len(cnt1.inductor_graphs), 1)
            self.assertExpectedInline(
                str(cnt.inductor_graphs[0].graph).strip(),
                """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=0] = placeholder[target=arg1_1]
    %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%arg0_1,), kwargs = {})
    %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sin,), kwargs = {})
    return (cos,)""",
            )
            self.assertExpectedInline(
                str(cnt1.inductor_graphs[0].graph).strip(),
                """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=0] = placeholder[target=arg1_1]
    %foo : [num_users=1] = call_function[target=torch.ops.mylib.foo.default](args = (%arg0_1,), kwargs = {})
    return (foo,)""",
            )

    def test_compile_non_infra_inside_compile(self):
        # [FUNCTION DOC] Checks that a torch.compile call made inside a dispatch mode's
        # torch_dispatch handler is compiled once (one recorded graph) for the user call.
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode whose handler compiles the intercepted op with
            # fullgraph and returns the result.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Compiles the intercepted function with the recording
                # backend and calls it with the given arguments.
                out = torch.compile(func, backend=backend, fullgraph=True)(
                    *args, **kwargs
                )
                return out

        x = torch.randn(5)
        with YoloMode():
            out = torch.add(x, x)

        self.assertEqual(len(backend.graphs), 1)

    def test_compile_non_infra_empty(self):
        # [FUNCTION DOC] Checks a compiled add under a dispatch mode that handles ops itself:
        # result is the multiplication result (sum 5.0), no graphs are recorded, and full-graph
        # mode raises a RuntimeError about no compiled frames.
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode replacing every op by multiplication.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Returns multiplication of the first two args.
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            out = torch.compile(torch.add, backend=backend, fullgraph=False)(x, x)
        self.assertEqual(out.sum().item(), 5.0)
        self.assertEqual(len(backend.graphs), 0)

        with YoloMode():
            with self.assertRaisesRegex(RuntimeError, "found no compiled frames"):
                torch.compile(torch.add, backend=backend, fullgraph=True)(x, x)

    def test_dispatch_key_set_preserved_across_compile(self):
        # [FUNCTION DOC] Checks that compiled calls (cache hits and one that raises a user
        # ValueError) leave the thread-local dispatch key include and exclude sets unchanged.
        # compile_wrapper snapshots and restores the local dispatch key set
        # around every compiled call. Make sure cache-hit calls, and a call that
        # raises a user error, leave the include/exclude sets unchanged.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        cf = torch.compile(f, backend="eager")
        x = torch.randn(3)
        cf(x)  # warm up / compile

        include = torch._C._dispatch_tls_local_include_set()
        exclude = torch._C._dispatch_tls_local_exclude_set()
        for _ in range(5):
            cf(x)
        self.assertEqual(include, torch._C._dispatch_tls_local_include_set())
        self.assertEqual(exclude, torch._C._dispatch_tls_local_exclude_set())

        def raises(x):
            # [FUNCTION DOC] HELPER: Always raises ValueError.
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            torch.compile(raises, backend="eager")(x)
        self.assertEqual(include, torch._C._dispatch_tls_local_include_set())
        self.assertEqual(exclude, torch._C._dispatch_tls_local_exclude_set())

    def test_dispatch_key_set_stack_balanced_on_fullgraph_error(self):
        # [FUNCTION DOC] Checks the saved dispatch key set stack stays balanced when fullgraph
        # raises the no compiled frames error: repeated errors under a dispatch mode, then a
        # normal call, then a manual restore must raise an empty stack error.
        # compile_wrapper pushes the local dispatch key set onto a C++ stack and
        # must pop it even when the wrapper's own finally raises. The
        # fullgraph=True "found no compiled frames" error is raised from inside
        # that finally, so a naive restore-at-end-of-finally would be skipped and
        # leak a stack entry. Verify the save/restore stack stays balanced.
        from torch.utils._python_dispatch import TorchDispatchMode

        class YoloMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode replacing ops by multiplication.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Returns multiplication of the first two args.
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            for _ in range(3):
                with self.assertRaisesRegex(RuntimeError, "found no compiled frames"):
                    torch.compile(torch.add, backend="eager", fullgraph=True)(x, x)

        # A normal compiled call after the error loop must still work. This
        # guards against an over-restore / double-pop regression, which would
        # leave the stack empty and make the wrapper's own restore raise.
        cf = torch.compile(lambda t: t + 1, backend="eager")
        self.assertEqual(cf(x), x + 1)

        # If any push had leaked, the stack would be non-empty here. When
        # balanced it is empty, so a manual restore raises instead of silently
        # popping a stale entry.
        with self.assertRaisesRegex(RuntimeError, "empty stack"):
            torch._C._dynamo_restore_local_dispatch_key_set()

    def test_tracing_context_tls_defaults_present(self):
        # [FUNCTION DOC] Checks that tracing and compile context thread-local defaults exist on a
        # fresh thread (try_get returns None and get raises AssertionError), and that tracing
        # sets and restores the tracing context.
        # The _guards thread-local defaults tracing_context/compile_context to
        # None per thread, so the hot-path try_get() calls (once per compiled
        # call) hit a present attribute instead of a slow getattr-miss on a
        # thread that never ran compilation itself (e.g. a worker thread running
        # code compiled on another thread).
        import threading

        from torch._guards import _TLS, CompileContext, TracingContext

        result = {}

        def worker():
            # [FUNCTION DOC] HELPER: Thread body recording presence of the two thread-local
            # attributes, try_get results, and whether CompileContext.get raised AssertionError,
            # into the shared result dict.
            result["tc_present"] = "tracing_context" in _TLS.__dict__
            result["cc_present"] = "compile_context" in _TLS.__dict__
            result["tc"] = TracingContext.try_get()
            result["cc"] = CompileContext.try_get()
            # Off-context (compile_context defaulted to None), get() still raises
            # -- the default makes this the "not set" AssertionError rather than
            # an AttributeError, on every thread regardless of history.
            try:
                CompileContext.get()
                result["get_raised"] = None
            except AssertionError:
                result["get_raised"] = "AssertionError"

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertTrue(result["tc_present"])
        self.assertTrue(result["cc_present"])
        self.assertIsNone(result["tc"])
        self.assertIsNone(result["cc"])
        self.assertEqual(result["get_raised"], "AssertionError")

        # Exercise the set -> restore cycle against the new storage: try_get()
        # sees the context inside tracing() and returns to the None default after.
        from torch._guards import tracing

        self.assertIsNone(TracingContext.try_get())
        tc = TracingContext(None)
        with tracing(tc):
            self.assertIs(TracingContext.try_get(), tc)
        self.assertIsNone(TracingContext.try_get())

    def test_compile_non_infra_empty_with_disalloed_dispatch_mode(self):
        # [FUNCTION DOC] Checks that a dispatch mode whose should-skip hook returns False does
        # not hide the compile: fullgraph compile of add under it yields one recorded graph.
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that opts out of being skipped by Dynamo and
            # replaces ops by multiplication.
            @classmethod
            def _should_skip_dynamo(cls):
                # [FUNCTION DOC] HELPER: Returns False so Dynamo does not skip this mode.
                return False

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Returns multiplication of the first two args.
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            out = torch.compile(torch.add, backend=backend, fullgraph=True)(x, x)

        self.assertEqual(len(backend.graphs), 1)

    def test_compile_non_infra_multiple(self):
        # [FUNCTION DOC] Checks nested dispatch modes: only the inner compile triggered from the
        # handler of the topmost mode records a graph (backend2); the outer user compile and the
        # other mode's compile record none.
        from torch.utils._python_dispatch import TorchDispatchMode

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode2(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode compiling the op through a lambda with backend3.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Compiles a lambda calling the op and returns its result.
                out = torch.compile(
                    lambda x, y: func(x, y), backend=backend3, fullgraph=True
                )(*args, **kwargs)
                return out

        class YoloMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that runs the op via a plain helper then compiles
            # add with backend2.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Calls the op through the helper, then compiles add with
                # backend2 and returns the result.
                def random_fn(func, *args, **kwargs):
                    # [FUNCTION DOC] HELPER: Calls the given function with the given args.
                    return func(*args, **kwargs)

                random_fn(func, *args, **kwargs)
                out = torch.compile(torch.add, backend=backend2, fullgraph=True)(
                    args[0], args[1]
                )
                return out

        x = torch.ones(5)
        with YoloMode(), YoloMode2():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 0)
        self.assertEqual(len(backend.graphs), 0)

    def test_compile_non_infra_multiple_compile_internal(self):
        # [FUNCTION DOC] Like the multiple dispatch mode test, but both modes declare they ignore
        # compile internals, so all three backends each record one graph.
        from torch.utils._python_dispatch import TorchDispatchMode

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode2(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode compiling through backend3 and opting in to
            # ignoring compile internals.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Compiles a lambda calling the op with backend3.
                out = torch.compile(
                    lambda x, y: func(x, y), backend=backend3, fullgraph=True
                )(*args, **kwargs)
                return out

            @classmethod
            def ignore_compile_internals(cls):
                # [FUNCTION DOC] HELPER: Returns True.
                return True

        class YoloMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode compiling add with backend2 and ignoring
            # compile internals.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Compiles add with backend2 and returns the result.
                out = torch.compile(torch.add, backend=backend2, fullgraph=True)(
                    args[0], args[1]
                )
                return out

            @classmethod
            def ignore_compile_internals(cls):
                # [FUNCTION DOC] HELPER: Returns True.
                return True

        x = torch.ones(5)
        with YoloMode(), YoloMode2():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 1)
        self.assertEqual(len(backend.graphs), 1)

    def test_compile_non_infra_multiple_compile_internal_mixed(self):
        # [FUNCTION DOC] Mixed case: only one of two dispatch modes ignores compile internals.
        # Checks graph counts for one mode order (only backend2 records), then after reset for
        # the reverse order (backend2 and backend3 each record one).
        from torch.utils._python_dispatch import TorchDispatchMode

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode2(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode compiling via backend3 and ignoring
            # compile internals.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Compiles a lambda calling the op with backend3.
                out = torch.compile(
                    lambda x, y: func(x, y), backend=backend3, fullgraph=True
                )(*args, **kwargs)
                return out

            @classmethod
            def ignore_compile_internals(cls):
                # [FUNCTION DOC] HELPER: Returns True.
                return True

        class YoloMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode compiling add with backend2; does not ignore
            # compile internals.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Compiles add with backend2 and returns the result.
                out = torch.compile(torch.add, backend=backend2, fullgraph=True)(
                    args[0], args[1]
                )
                return out

        x = torch.ones(5)
        with YoloMode(), YoloMode2():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 0)
        self.assertEqual(len(backend.graphs), 0)

        torch._dynamo.reset()

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        with YoloMode2(), YoloMode():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 1)
        self.assertEqual(len(backend.graphs), 0)

    @torch._dynamo.config.patch(inline_torch_dispatch_torch_compile=False)
    def test_compile_non_infra_disabled_config(self):
        # [FUNCTION DOC] Checks that with the inline-dispatch-compile config turned off, a
        # compile call inside a dispatch mode handler is skipped, so the recording backend
        # captures zero graphs.
        """Test that setting inline_torch_dispatch_torch_compile=False reverts to old behavior."""
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        # When the config is False, torch.compile inside __torch_dispatch__ should
        # be skipped (old behavior) because we are inside a dispatch mode.
        class YoloMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that is not skipped by Dynamo and whose handler
            # compiles the intercepted op.
            @classmethod
            def _should_skip_dynamo(cls):
                # [FUNCTION DOC] HELPER: Returns False so the mode is not skipped.
                return False

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Compiles the intercepted function with the recording
                # backend and returns its result.
                out = torch.compile(func, backend=backend, fullgraph=False)(
                    *args, **kwargs
                )
                return out

        x = torch.randn(5)
        with YoloMode():
            out = torch.add(x, x)

        # With the config disabled, compile should be skipped, so 0 graphs captured
        self.assertEqual(len(backend.graphs), 0)

    @torch._dynamo.config.patch(accumulated_recompile_limit=1)
    def test_dynamo_disabled_in_custom_op_kernels(self):
        # [FUNCTION DOC] Checks that Dynamo is turned off inside custom op kernels. With a
        # recompile limit of one, a custom op whose body graph-breaks must produce no graph break
        # counters, for the default kernel, a kernel added via register_kernel, and a library-
        # level registered kernel.
        counters.clear()

        @torch.library.custom_op("mylib::foo9", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op body that forces a graph break and returns a clone
            # of x.
            torch._dynamo.graph_break()
            return x.clone()

        foo.register_fake(torch.clone)

        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Calls the custom op overload on x.
            return foo._opoverload(x)

        x = torch.randn(2)
        f(x)
        x = torch.randn(3)
        # Recompile hits the cache size limit, which will cause Dynamo to
        # recurse into the frames. The only frame is the implementation
        # of foo. If Dynamo was not turned off correctly, then
        # we'll see a graph break
        f(x)
        self.assertEqual(len(counters["graph_break"]), 0)

        counters.clear()

        called = 0

        # test register_kernel
        @foo.register_kernel("cpu")
        def _(x):
            # [FUNCTION DOC] HELPER: Replacement CPU kernel that counts calls in the enclosing
            # variable, forces a graph break, and returns a clone.
            nonlocal called
            called += 1
            torch._dynamo.graph_break()
            return x.clone()

        f(x)
        self.assertEqual(called, 1)
        self.assertEqual(len(counters["graph_break"]), 0)

        # test torch.library.register_kernel
        counters.clear()
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo2(Tensor x) -> Tensor")

            @torch.library.register_fake("mylib::foo2", lib=m)
            def _(x):
                # [FUNCTION DOC] HELPER: Fake implementation returning a clone of x.
                return x.clone()

            @torch.library.register_kernel("mylib::foo2", "cpu", lib=m)
            def _(x):
                # [FUNCTION DOC] HELPER: CPU kernel for the library op; forces a graph break and
                # returns a clone.
                torch._dynamo.graph_break()
                return x.clone()

            @torch.compile(backend="eager")
            def g(x):
                # [FUNCTION DOC] HELPER: Calls the library op foo2 on x.
                return torch.ops.mylib.foo2.default(x)

            x = torch.randn(2)
            g(x)  # compiles
            x = torch.randn(3)
            g(x)  # dynamo falls back on the outermost frame
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_invalid_args_builtin(self):
        # [FUNCTION DOC] Checks that calling isinstance with an invalid keyword argument inside a
        # compiled function raises TypeError.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Takes sine of x, then calls isinstance with an invalid
            # keyword; the call raises TypeError.
            x = x.sin()
            if isinstance(x, torch.Tensor, invalid=True):
                x = x.sin()
            return x

        with self.assertRaises(TypeError):
            fn(torch.randn(16))

    def test_scalar_device_movement(self):
        # [FUNCTION DOC] Checks that calling add with python scalars and an out tensor
        # under fullgraph gives the same result as eager. Skipped when shapes are not
        # static by default.
        if not torch._dynamo.config.assume_static_by_default:
            self.skipTest("Doesn't work with symints")

        def add_fn(a, b, out):
            # [FUNCTION DOC] HELPER: Adds a and b writing into out, and returns the result.
            res = torch.add(a, b, out=out)
            return res

        res = add_fn(2, 3, torch.tensor(0.0))
        add_fn = torch.compile(add_fn, backend="eager", fullgraph=True)
        res_compiled = add_fn(2, 3, torch.tensor(0.0))
        self.assertEqual(res, res_compiled)

    def test_callpacked(self):
        # [FUNCTION DOC] Checks that a function unpacking an argument sequence compiles once for
        # lists and once for tuples (two frames total) and results match eager.
        def call_packed(args):
            # [FUNCTION DOC] HELPER: Unpacks three tensors and returns a minus b times c.
            a, b, c = args
            return a - b * c

        counter = CompileCounter()
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        correct = call_packed([a, b, c])
        opt_call_packed = torch._dynamo.optimize_assert(counter)(call_packed)
        val1 = opt_call_packed([a, b, c])
        val2 = opt_call_packed((a, b, c))
        val3 = opt_call_packed([a, b, c])
        val4 = opt_call_packed((a, b, c))
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))
        self.assertTrue(same(val3, correct))
        self.assertTrue(same(val4, correct))
        self.assertEqual(counter.frame_count, 2)

    def test_raises(self):
        # [FUNCTION DOC] Checks that a compiled function that raises an exception of a given
        # class propagates it, after compiling a frame with three ops.
        def fn(a, b, c, cls):
            # [FUNCTION DOC] HELPER: Computes a tensor expression, then raises the given
            # exception class with the tensor text as message.
            x = a + b - c * 10
            raise cls(str(x))

        counter = CompileCounter()
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        opt_fn = torch.compile(fn, backend=counter)
        self.assertRaises(AssertionError, lambda: opt_fn(a, b, c, AssertionError))
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

    def test_module_not_callable(self):
        # [FUNCTION DOC] Checks that calling a module object as a function raises TypeError with
        # the message about the module not being callable.
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the torch.fft module as a function, which is invalid.
            return torch.fft(x)

        counter = CompileCounter()
        a = torch.randn(10, 10)
        opt_fn = torch.compile(fn, backend=counter)
        self.assertRaisesRegex(
            TypeError, "'module' object is not callable", lambda: opt_fn(a)
        )

    def test_inplace(self):
        # [FUNCTION DOC] Runs the standard test helper on a function mixing an in-place copy and
        # in-place subtraction, expecting 3 ops.
        def inplace1(a, b):
            # [FUNCTION DOC] HELPER: Allocates an empty tensor, copies a into it, subtracts b in
            # place, returns it.
            o = torch.empty((10, 10))
            o.copy_(a)
            o -= b
            return o

        torch._dynamo.testing.standard_test(self, inplace1, 2, expected_ops=3)

    def test_inplace_desugaring(self):
        # [FUNCTION DOC] Runs the standard test helper on a function with in-place add and
        # subtract on python int literals combined with a tensor, expecting 2 ops.
        def inplace_on_literals(y):
            # [FUNCTION DOC] HELPER: Does in-place add and subtract of tensor y on two int
            # variables and returns both.
            x0 = 1
            x0 += y
            x1 = 1
            x1 -= y
            return x0, x1

        torch._dynamo.testing.standard_test(
            self, inplace_on_literals, 1, expected_ops=2
        )

    def test_bool_literal_iand_unspecialized_float(self):
        # [FUNCTION DOC] Checks that an in-place bitwise and on a bool derived from an optional
        # float argument works, compiling with the eager backend for a zero and a non-zero value
        # of c.
        @torch.compile(backend="eager")
        def mul(a: torch.Tensor, b: torch.Tensor, c: float | int | None):
            # [FUNCTION DOC] HELPER: Computes a boolean that c is set and nonzero; returns
            # product plus c if so, else just the product.
            has_bias = c is not None
            has_bias &= c != 0.0

            if has_bias:
                return a * b + c
            return a * b

        a = torch.randn(10)
        b = torch.randn(10)

        self.assertEqual(mul(a, b, c=0.0), a * b)
        self.assertEqual(mul(a, b, c=0.5), a * b + 0.5)

    def test_unpack4(self):
        # [FUNCTION DOC] Runs the standard test helper on a function that slices, unpacks the
        # size into two ints, and builds an output tensor; expects 5 ops.
        def unpack4(a, b):
            # [FUNCTION DOC] HELPER: Slices both inputs to first five rows, unpacks size,
            # allocates output, copies ratio into it.
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.size()
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torch._dynamo.testing.standard_test(
            self,
            unpack4,
            2,
            expected_ops=5,
        )

    def test_unpack5(self):
        # [FUNCTION DOC] Same as the previous unpack test but unpacks the shape attribute instead
        # of size; expects 5 ops.
        def unpack5(a, b):
            # [FUNCTION DOC] HELPER: Slices inputs, unpacks shape, allocates output, copies the
            # ratio into it.
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.shape
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torch._dynamo.testing.standard_test(
            self,
            unpack5,
            2,
            expected_ops=5,
        )

    def test_matmul1(self):
        # [FUNCTION DOC] Runs the standard test helper on a matrix multiply operator expression;
        # expects a single op.
        def matmul_op1(a, b):
            # [FUNCTION DOC] HELPER: Returns matrix product of a and b.
            return a @ b

        # TODO(jansel): FX doesn't support this, should add upstream support
        torch._dynamo.testing.standard_test(self, matmul_op1, 2, expected_ops=1)

    def test_int_shape_binops(self):
        # [FUNCTION DOC] Checks int-on-the-left binary operators (subtract, add, multiply,
        # modulo, power, floor divide, pow, true divide) with a tensor shape element. Expects 1
        # op statically and 9 when dynamic.
        def fn(x):
            # [FUNCTION DOC] HELPER: Chains int and shape-derived arithmetic ops, then adds the
            # result to x.
            # Test reversal by putting int arg first.
            y = 15 - x.shape[0]
            y = 4 + y
            y = 5 * y
            y = 2 % y
            y = 3**y
            y = 10 // y
            y = pow(2, y)
            y = 10 / y
            return x + y

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 9)
        )

    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_pt2_compliant_ops_are_allowed(self):
        # [FUNCTION DOC] With only-pt2-compliant ops enforced, checks that a custom op tagged
        # compliant (defined in a scoped library) compiles under fullgraph both when called
        # through the namespace and through the overload.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::bar",
                "(Tensor x) -> Tensor",
                lib=lib,
                tags=(torch.Tag.pt2_compliant_tag,),
            )
            torch.library.impl(
                "mylib::bar", "CompositeImplicitAutograd", torch.sin, lib=lib
            )
            if torch.Tag.pt2_compliant_tag not in torch.ops.mylib.bar.default.tags:
                raise AssertionError("Expected pt2_compliant_tag in bar.default.tags")

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the library op bar on x.
                return torch.ops.mylib.bar(x)

            overload = torch.ops.mylib.bar.default

            def g(x):
                # [FUNCTION DOC] HELPER: Calls the saved overload on x; defined but never
                # compiled (the test compiles f twice).
                return overload(x)

            x = torch.randn(3)

            counts = torch._dynamo.testing.CompileCounter()
            optimized_f = torch.compile(f, backend=counts, fullgraph=True)
            _ = optimized_f(x)

            optimized_g = torch.compile(f, backend=counts, fullgraph=True)
            _ = optimized_g(x)

    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_non_pt2_compliant_ops_graph_break(self):
        # [FUNCTION DOC] With only-pt2-compliant ops enforced, checks that an untagged custom op
        # raises Unsupported mentioning not PT2 compliant, via both the namespace and overload
        # call forms (both compile f).
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define("mylib::bar2", "(Tensor x) -> Tensor", lib=lib)
            torch.library.impl(
                "mylib::bar2", "CompositeImplicitAutograd", torch.sin, lib=lib
            )
            if torch.Tag.pt2_compliant_tag in torch.ops.mylib.bar2.default.tags:
                raise AssertionError(
                    "Expected pt2_compliant_tag not in bar2.default.tags"
                )

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the library op bar2 on x.
                return torch.ops.mylib.bar2(x)

            overload = torch.ops.mylib.bar2.default

            def g(x):
                # [FUNCTION DOC] HELPER: Calls the saved overload on x; not actually compiled by
                # the test.
                return overload(x)

            x = torch.randn(3)

            counts = torch._dynamo.testing.CompileCounter()
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                optimized_f = torch.compile(f, backend=counts, fullgraph=True)
                y = optimized_f(x)

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                optimized_g = torch.compile(f, backend=counts, fullgraph=True)
                y = optimized_g(x)

    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_pt2_compliant_overload(self):
        # [FUNCTION DOC] Checks per-overload compliance: only the overload tagged compliant
        # compiles; calling the untagged int overload raises Unsupported about PT2 compliance,
        # and a bad argument list raises Unsupported mentioning failed to.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::bar3.tensor",
                "(Tensor x) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "mylib::bar3.int", "(Tensor x, int dim) -> Tensor", lib=lib
            )

            torch.library.impl(
                "mylib::bar3.tensor",
                "CompositeImplicitAutograd",
                torch.sin,
                lib=lib,
            )
            torch.library.impl(
                "mylib::bar3.int", "CompositeImplicitAutograd", torch.sum, lib=lib
            )

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the tensor overload of bar3.
                return torch.ops.mylib.bar3(x)

            def g(x):
                # [FUNCTION DOC] HELPER: Calls bar3 with a tensor and an int.
                return torch.ops.mylib.bar3(x, 1)

            def h(x):
                # [FUNCTION DOC] HELPER: Calls bar3 with three tensors (invalid signature).
                return torch.ops.mylib.bar3(x, x, x)

            x = torch.randn(3)

            counts = torch._dynamo.testing.CompileCounter()
            optimized_f = torch.compile(f, backend=counts, fullgraph=True)
            optimized_g = torch.compile(g, backend=counts, fullgraph=True)
            optimized_h = torch.compile(h, backend=counts, fullgraph=True)

            # No error: the overload is PT2 compliant
            optimized_f(x)

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                y = optimized_g(x)

            # graph break on incorrect parsing
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "failed to"):
                y = optimized_h(x)

    def test_user_defined_setattr1(self):
        # [FUNCTION DOC] Checks a compiled function reading and writing attributes on an object
        # with custom prefixing attribute hooks; stored keys end up as prefixed x and y.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # [FUNCTION DOC] HELPER: Sets attribute y to x plus 1.
            obj.y = obj.x + 1

        obj = UserDefineSetAttr()
        with patch.object(UserDefineSetAttr, "setup", True):
            obj.x = torch.randn(8)
        fn(obj)
        with patch.object(UserDefineSetAttr, "setup", True):
            self.assertEqual(obj.y, obj.x + 1)
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    def test_user_defined_setattr2(self):
        # [FUNCTION DOC] Checks that an object with custom attribute hooks created and mutated
        # inside a compiled function keeps prefixed keys and the right values on output.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates the object, stores x, stores x plus 1 as y, and
            # returns it.
            obj = UserDefineSetAttr()
            obj.x = x
            obj.y = obj.x + 1
            return obj

        x = torch.randn(8)
        obj = fn(x)
        with patch.object(UserDefineSetAttr, "setup", True):
            self.assertIs(obj.x, x)
            self.assertEqual(obj.y, x + 1)
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_repeat_cat(self):
        # [FUNCTION DOC] With scalar output capture enabled, checks that repeating a tensor by an
        # unbacked length and concatenating compiles with dynamic shapes and fullgraph.
        def f(x, n):
            # [FUNCTION DOC] HELPER: Reads a scalar item, repeats an empty tensor n times, and
            # concatenates it with itself.
            m = x.item()
            x = torch.empty(x).repeat(n)  # s0*u0
            return torch.cat([x, x], dim=0)

        fn = torch.compile(f, backend="eager", dynamic=True, fullgraph=True)
        fn(torch.tensor([5]), 5)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_cond_runtime_assert_generation(self):
        # [FUNCTION DOC] Checks that a runtime assertion from a check on an unbacked nonzero size
        # is generated inside a cond branch and fails at runtime for input sizes not divisible by
        # 4 (RuntimeError naming the failed expression).
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes nonzero, checks its first dim is divisible by 4,
            # and returns random values of that size.
            y = x.nonzero()  # unbacked binding u0
            torch._check(y.shape[0] % 4 == 0)

            return torch.randn(y.shape[0])

        @torch.compile(dynamic=True, backend="aot_eager")
        def foo(x):
            # [FUNCTION DOC] HELPER: Compiled function using cond with a divisibility predicate
            # on the input size and the same helper in both branches.
            b = torch.cond(
                pred=(x.shape[0] % 4 == 0),
                true_fn=lambda: fn(x),
                false_fn=lambda: fn(x),
            )

            return b

        foo(torch.randn(4, 4))
        with self.assertRaisesRegex(
            RuntimeError, "Runtime assertion failed for expression Eq(Mod(u1, 4), 0)*"
        ):
            foo(torch.randn(5, 5))

    def test_tensor_setattr_getset_descriptor(self):
        # [FUNCTION DOC] Checks that assigning to the real attribute (a getset descriptor) of a
        # complex tensor in compiled code matches eager.
        # Tensor attribute `real` has special getter/setter for complex dtype.
        def f(x):
            # [FUNCTION DOC] HELPER: Sets the real part to 10 and returns x plus 1.
            x.real = 10
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=False)
        x = torch.ones(5, dtype=torch.cfloat)

        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)

    def test_newly_constructed_tensor_attr_mutation(self):
        # [FUNCTION DOC] Checks that custom attributes (grad and an arbitrary one) set on a
        # tensor created inside a compiled function are visible on the output.
        def f(x):
            # [FUNCTION DOC] HELPER: Creates y from x, assigns grad and a custom attribute,
            # returns y.
            y = x + 10
            y.grad = x
            y.foo = 42
            return y

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)
        self.assertEqual(res.grad, ref.grad)
        self.assertEqual(res.foo, ref.foo)

    def test_input_tensor_custom_attr_mutation(self):
        # [FUNCTION DOC] Checks that setting a custom attribute on an input tensor inside a
        # compiled function is applied to the original input.
        def f(x, flag):
            # [FUNCTION DOC] HELPER: Sets a flag attribute on x and returns x plus 1.
            x.offloading_activation = flag
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x, True)
        self.assertEqual(res, torch.ones(5) + 1)
        self.assertTrue(x.offloading_activation)

    def test_intermediate_tensor_custom_attr_mutation(self):
        # [FUNCTION DOC] Checks that a custom attribute set on an intermediate tensor in compiled
        # code appears on the returned tensor.
        def f(x, flag):
            # [FUNCTION DOC] HELPER: Computes y as x plus 1, sets a flag attribute on it,
            # returns y.
            y = x + 1
            y.offloading_activation = flag
            return y

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x, True)
        self.assertEqual(res, torch.ones(5) + 1)
        self.assertTrue(res.offloading_activation)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
        "requires Hopper+ (SM >= 9.0) for TMA",
    )
    @unittest.skipIf(
        not torch.utils._triton.has_triton()
        or not hasattr(__import__("triton"), "set_allocator"),
        "requires triton with set_allocator support",
    )
    def test_triton_set_allocator_no_graph_break(self):
        # [FUNCTION DOC] Requires Hopper+ GPU and triton with allocator support. Checks that
        # setting the triton allocator inside a compiled function does not graph break, and that
        # it is replayed on a cache hit.
        """set_allocator inside torch.compile does not graph break and
        replays correctly at runtime (including cache hits)."""
        import triton
        import triton.language as tl
        from triton.runtime._allocation import NullAllocator

        # [FUNCTION DOC] HELPER: Triton kernel copying a tile of x_ptr into out_ptr using tensor
        # descriptors; each program handles one block of rows addressed by its program id.
        @triton.jit
        def tma_copy_kernel(
            x_ptr,
            out_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            pid = tl.program_id(0)
            desc = tl.make_tensor_descriptor(
                x_ptr,
                shape=[M, N],
                strides=[stride_m, stride_n],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            block = tl.load_tensor_descriptor(desc, [pid * BLOCK_M, 0])
            out_desc = tl.make_tensor_descriptor(
                out_ptr,
                shape=[M, N],
                strides=[stride_m, stride_n],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            tl.store_tensor_descriptor(out_desc, [pid * BLOCK_M, 0], block)

        M, N, BLOCK_M, BLOCK_N = 128, 64, 64, 64

        def run_kernel(x):
            # [FUNCTION DOC] HELPER: Allocates an output like x, launches the TMA copy kernel
            # with a grid of row blocks, and returns the output.
            out = torch.empty_like(x)
            tma_copy_kernel[(M // BLOCK_M,)](
                x,
                out,
                M,
                N,
                x.stride(0),
                x.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            return out

        x = torch.randn(M, N, device="cuda")

        from contextlib import contextmanager

        from triton.runtime._allocation import _allocator

        @contextmanager
        def triton_allocator(allocator):
            # [FUNCTION DOC] HELPER: Context manager that installs a given triton allocator and
            # restores the previous one on exit, even on error.
            prev = _allocator.get()
            triton.set_allocator(allocator)
            try:
                yield
            finally:
                triton.set_allocator(prev)

        def fn_with_set_allocator(x):
            # [FUNCTION DOC] HELPER: Installs a triton allocator that returns int8 CUDA tensors,
            # then runs the kernel on x.
            triton.set_allocator(
                lambda size, alignment, stream: torch.empty(
                    size, device="cuda", dtype=torch.int8
                )
            )
            return run_kernel(x)

        opt_fn = torch.compile(
            fn_with_set_allocator, backend="aot_eager", fullgraph=True
        )

        # set_allocator inside compiled region does NOT graph break
        with triton_allocator(NullAllocator()):
            out = opt_fn(x)
            self.assertEqual(out, x)

            # Verify set_allocator replays on cache hit (not just tracing)
            triton.set_allocator(NullAllocator())
            out2 = opt_fn(x)
            self.assertEqual(out2, x)

    def test_closure_recompiles(self):
        # [FUNCTION DOC] Checks that closures with equal constant captured values share a
        # compiled frame while different tensor captures recompile; expects 2 frames.
        cnt = CompileCounter()

        def fn(x, other_fn):
            # [FUNCTION DOC] HELPER: Applies other_fn to x plus 1 and subtracts 1.
            return other_fn(x + 1) - 1

        opt = torch.compile(fn, backend=cnt, fullgraph=True)

        x = torch.randn(8)
        for f in (
            closure_adder(5),
            closure_adder(5),
            closure_adder(torch.randn(8)),
            closure_adder(torch.randn(8)),
        ):
            self.assertEqual(opt(x, f), fn(x, f))

        self.assertEqual(cnt.frame_count, 2)

    def test_generate_trivial_abstract_impl(self):
        # [FUNCTION DOC] Checks that a custom op returning nothing, with a mutable optional
        # tensor and symint arguments, compiles under fullgraph with a generated trivial fake
        # implementation and returns None.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo_generate_trivial_abstract_impl",
                "(Tensor x, Tensor[] y, Tensor(a!)? z, SymInt w) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl(
                "mylib::foo_generate_trivial_abstract_impl", "cpu", lib=lib
            )
            @torch._dynamo.disable
            def foo_impl(x, y, z, w):
                # [FUNCTION DOC] HELPER: Eager implementation with Dynamo disabled; computes a
                # sum it discards and returns nothing.
                x + y[0] + w
                return

            def f(x, y, z, w):
                # [FUNCTION DOC] HELPER: Calls the custom op with the constant 2 in place of w.
                return torch.ops.mylib.foo_generate_trivial_abstract_impl(x, y, z, 2)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            w = torch.randn(3)
            args = (x, y, z, w)

            output = torch.compile(f, backend="eager", fullgraph=True)(*args)
            self.assertEqual(output, None)

    def test_shape_int_inplace_binops(self):
        # [FUNCTION DOC] Runs the standard test helper on a function applying many in-place
        # numeric operators to a shape element; expects 1 op statically and 6 dynamically.
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies in-place add, subtract, power, divide, multiply,
            # floor divide and modulo on the first dimension size, then adds it to x.
            p = x.shape[0]
            p += 2
            p -= 2
            p **= 2
            p /= 2
            p *= 2
            p //= 2
            p %= 2
            return x + p

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 6)
        )

    def test_int_shape_inplace_binops(self):
        # [FUNCTION DOC] Same idea with an int on the left of each in-place operator and the
        # shape element on the right; expects 1 op statically and 2 dynamically.
        def fn(x):
            # [FUNCTION DOC] HELPER: Resets an int to 2 and applies each in-place operator with
            # the shape element; returns x plus the last result.
            p = x.shape[0]
            # Test reversal by putting constant first
            y = 2
            y += p
            y = 2
            y -= p
            y = 2
            y **= p
            y = 2
            y /= p
            y = 2
            y *= p
            y = 2
            y //= p
            y = 2
            y %= p
            return x + y

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 2)
        )

    def test_int_int_comparisons(self):
        # [FUNCTION DOC] Standard test helper on constant int-versus-int comparisons; only the
        # equality branch is taken, 1 op.
        def fn(x):
            # [FUNCTION DOC] HELPER: Chains comparisons on constants and adds the selected int
            # to x.
            if 2 != 2:
                out = 1
            elif 2 < 1:
                out = 1
            elif 1 > 2:
                out = 1
            elif 1 >= 2:
                out = 1
            elif 2 <= 1:
                out = 1
            elif 2 == 2:
                out = 2
            else:
                out = 1
            return x + out

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_shape_int_comparisons(self):
        # [FUNCTION DOC] Standard test helper on comparisons of a shape element against constants
        # on the right side; 1 op.
        def fn(x):
            # [FUNCTION DOC] HELPER: Chains comparisons of the first dim size with constants and
            # adds the chosen int to x.
            a = x.shape[0]
            # Ensure support for constant on right side
            if a != 10:
                out = 1
            elif a < 2:
                out = 1
            elif a > 12:
                out = 1
            elif a >= 12:
                out = 1
            elif a <= 2:
                out = 1
            elif a == 10:
                out = 2
            else:
                out = 1
            return x + out

        # TODO: Test the guards maybe?
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_int_shape_comparisons(self):
        # [FUNCTION DOC] Standard test helper on comparisons with the constant on the left side
        # of a shape element; 1 op.
        def fn(x):
            # [FUNCTION DOC] HELPER: Chains comparisons of constants with the first dim size and
            # adds the chosen int to x.
            a = x.shape[0]
            # Ensure support for constant on left side
            if 10 != a:
                out = 1
            elif 12 < a:
                out = 1
            elif 2 > a:
                out = 1
            elif 2 >= a:
                out = 1
            elif 12 <= a:
                out = 1
            elif 10 == a:
                out = 2
            else:
                out = 1
            return x + out

        # TODO: Test the guards maybe?
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_param_shape_binops(self):
        # [FUNCTION DOC] Checks arithmetic with a parameter's shape element on the left, compiled
        # under fullgraph with one frame; op count 1 for static shapes and 9 for dynamic.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 15-element parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(15))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies a chain of arithmetic operators between the
                # parameter size and the input size and adds the result to x.
                # Test reversal by putting param shape arg first.
                p = self.param.shape[0]
                y = p - x.shape[0]
                y = p + y
                y = p * y
                y = p % y
                y = p**y
                y = p // y
                y = pow(p, y)
                y = p / y
                return x + y

        counts = torch._dynamo.testing.CompileCounter()
        mod = MyModule()
        optimized_mod = torch.compile(mod, backend=counts, fullgraph=True)

        x = torch.randn(3)
        ref = mod(x)
        res = optimized_mod(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(counts.op_count, """1""")
        else:
            self.assertExpectedInline(counts.op_count, """9""")

    def test_user_defined_binop(self):
        # [FUNCTION DOC] Checks that an int plus an object defining a reflected add compiles in
        # one frame; ops 1 static or 2 dynamic.
        class MyClass:
            # [CLASS DOC] HELPER: Holds a value and supports reflected addition.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores value.
                self.value = value

            def __radd__(self, other):
                # [FUNCTION DOC] HELPER: Returns the held value plus other.
                return self.value + other

        def fn(x, c):
            # [FUNCTION DOC] HELPER: Adds the first dim size plus c to x.
            y = x.shape[0] + c
            return x + y

        counts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=counts)

        x = torch.randn(3)
        c = MyClass(4)
        ref = fn(x, c)
        res = opt_fn(x, c)

        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(counts.op_count, """1""")
        else:
            self.assertExpectedInline(counts.op_count, """2""")

    def test_user_defined_iter(self):
        # [FUNCTION DOC] Checks iteration over a user-defined iterable inlined under fullgraph;
        # one frame, and mutating the underlying list length causes one recompile (two frames).
        class Mod:
            # [CLASS DOC] HELPER: Holds a list of two tensors and is iterable over it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the list of two random tensors.
                self.a = [torch.randn(2, 2), torch.randn(2, 2)]

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns an iterator over the list.
                return iter(self.a)

        def f(mod):
            # [FUNCTION DOC] HELPER: Collects x plus 1 for each element of mod.
            ret = []
            for x in mod:
                ret.append(x + 1)
            return ret

        mod = Mod()
        counts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=counts, fullgraph=True)
        ref = f(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)

        mod.a.append(torch.randn(2, 2))
        # `for x in mod` is inlined, where iter(m.a) creates a guard on the list length of m.a
        # Mutating length of mod.a causes a re-compilation.
        ref2 = f(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        self.assertTrue(same(ref2, res2))
        self.assertEqual(counts.frame_count, 2)

    def test_compare_shapes_eq(self):
        # [FUNCTION DOC] Checks shape equality comparisons through the standard test helper for
        # list conversions and for the shape object itself.
        def compare_shapes(a, b, to_list):
            # [FUNCTION DOC] HELPER: Compares the shapes of the unsqueezed inputs (as lists or
            # shape objects) and returns a plus 1 if equal else a plus 2.
            x = list(a.unsqueeze(-1).shape) if to_list else a.shape
            y = list(b.unsqueeze(-1).shape) if to_list else b.shape
            if x == y:
                return a + 1
            else:
                return a + 2

        # Test both ListVariable and ShapeVariable
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=True), 2
        )
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=False), 2
        )

    def test_compare_shapes_tuple_eq(self):
        # [FUNCTION DOC] Standard test helper on equality of tuple-converted shapes.
        def compare_shapes(a, b):
            # [FUNCTION DOC] HELPER: Compares tuple shapes of unsqueezed inputs and returns a
            # plus 1 if equal else a plus 2.
            x = tuple(a.unsqueeze(-1).shape)
            y = tuple(b.unsqueeze(-1).shape)
            if x == y:
                return a + 1
            else:
                return a + 2

        torch._dynamo.testing.standard_test(self, lambda a, b: compare_shapes(a, b), 2)

    def test_compare_shapes_tuple_neq(self):
        # [FUNCTION DOC] Standard test helper on inequality of tuple-converted shapes.
        def compare_shapes(a, b):
            # [FUNCTION DOC] HELPER: Returns a plus 1 if tuple shapes differ else a plus 2.
            x = tuple(a.unsqueeze(-1).shape)
            y = tuple(b.unsqueeze(-1).shape)
            if x != y:
                return a + 1
            else:
                return a + 2

        torch._dynamo.testing.standard_test(self, lambda a, b: compare_shapes(a, b), 2)

    def test_compare_shapes_neq(self):
        # [FUNCTION DOC] Checks shape inequality comparisons for list and shape-object forms.
        def compare_shapes(a, b, to_list):
            # [FUNCTION DOC] HELPER: Returns a plus 1 if the shapes differ else a plus 2.
            x = list(a.unsqueeze(-1).shape) if to_list else a.shape
            y = list(b.unsqueeze(-1).shape) if to_list else b.shape
            if x != y:
                return a + 1
            else:
                return a + 2

        # Test both ListVariable and ShapeVariable
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=True), 2
        )
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=False), 2
        )

    def test_compare_shapes_with_constant(self):
        # [FUNCTION DOC] Checks the guard failure reason when the first dim compared with a
        # constant changes: message reports a size mismatch at index 0 with expected 3, actual 4.
        def compare_shapes(a):
            # [FUNCTION DOC] HELPER: Multiplies a by 4 if the first dim is not 3, else by 3.
            x = a.shape
            if x[0] != 3:
                return a * 4
            return a * 3

        guard_failure = None

        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Records the failure object in the enclosing variable.
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(compare_shapes)
        opt_fn(torch.randn([3, 4]))
        opt_fn(torch.randn([4, 3]))
        self.assertIn(
            """tensor 'a' size mismatch at index 0. expected 3, actual 4""",
            guard_failure.reason,
        )

    def test_recompile_message_on_parameter(self):
        # [FUNCTION DOC] Checks that the recompile failure message for parameters mentions the
        # force-parameter-static-shapes config, for three parameters of different shapes.
        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Asserts the failure text mentions the config option.
            self.assertIn("torch._dynamo.config.force_parameter_static_shapes", failure)

        @torch._dynamo.optimize("eager", guard_fail_fn=guard_failures)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns cosine of x.
            return torch.cos(x)

        x1 = torch.nn.Parameter(torch.rand(32, 16))
        x2 = torch.nn.Parameter(torch.rand(8, 4, 3, 3))
        x3 = torch.nn.Parameter(torch.rand(8, 8, 3, 3))
        fn(x1)
        fn(x2)
        fn(x3)

    def test_builtin_abs(self):
        # [FUNCTION DOC] Checks the builtin absolute value on tensors, python ints and floats
        # under fullgraph.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns absolute value of x plus absolute value of y.
            return abs(x) + abs(y)

        sample = torch.randn(10, 10)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        for sample in [
            (torch.randn(10, 10), torch.randn(10, 10)),
            (-10, make_tensor(10, dtype=torch.int64, device="cpu")),
            (-0.1, torch.randn(10)),
        ]:
            expect = fn(*sample)
            actual = opt_fn(*sample)
            self.assertEqual(expect, actual)

    def test_builtin_eval_constant_expr(self):
        # [FUNCTION DOC] Checks that eval on constant string expressions is evaluated at trace
        # time under fullgraph with one frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Evaluates constant expression strings for a dict, an integer
            # sum and a division and combines them with x.
            values = eval("{'scale': 2, 'offset': (3, 4)}")
            return x * values["scale"] + eval(" 1+1\n") + eval("1/2")

        x = torch.randn(4)
        cnt = CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_builtin_eval_rejects_non_constant_expr(self):
        # [FUNCTION DOC] Checks that eval of a non-constant expression raises Unsupported with a
        # builtin-operator trace failure message.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds x to an eval of a length call expression string.
            return x + eval("len([1])")

        with self.assertRaisesRegex(Unsupported, "Failed to trace builtin operator"):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(1))

    def test_builtin_eval_propagates_syntax_error(self):
        # [FUNCTION DOC] Checks that a syntax error from eval is raised and catchable inside
        # compiled code.
        def fn(x):
            # [FUNCTION DOC] HELPER: Evals an invalid string, returning x plus 1 on SyntaxError
            # else x minus 1.
            try:
                eval("1+")
            except SyntaxError:
                return x + 1
            return x - 1

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_builtin_isinstance(self):
        # [FUNCTION DOC] Standard test helper for isinstance checks on tensors, ints, lists and
        # dicts; 1 op.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a list of six isinstance results.
            t = torch.arange(1, 3)
            a = isinstance(x, torch.Tensor)
            b = isinstance(t, torch.Tensor)
            c = isinstance(x, int)
            d = isinstance(3, int)
            e = isinstance([1, 2, 3], list)
            f = isinstance({"foo": 1, "bar": 2}, dict)
            res = [a, b, c, d, e, f]
            # Can't run yet due to other unimplemented instructions
            # res += [isinstance(torch.nn.LazyLinear(2, 3), torch.nn.Linear)]
            return res

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_os_environ_get(self):
        # [FUNCTION DOC] Checks that reading an environment variable inside a compiled function
        # is guarded: flipping the variable between runs causes a recompile (two frames).
        # Restores or deletes the variable afterwards.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if the environment variable equals 1, else
            # x minus 1.
            if os.environ.get("OS_ENVIRON_TEST") == "1":
                return x + 1
            else:
                return x - 1

        x = torch.ones(2, 3)
        try:
            original = os.environ.get("OS_ENVIRON_TEST", None)

            os.environ["OS_ENVIRON_TEST"] = "1"
            res1 = fn(x)
            self.assertEqual(res1, x + 1)
            self.assertEqual(cnts.frame_count, 1)
            os.environ["OS_ENVIRON_TEST"] = "0"
            res2 = fn(x)
            self.assertEqual(res2, x - 1)
            # Ensure re-compile if os.environ items updated
            self.assertEqual(cnts.frame_count, 2)
        finally:
            if original is None:
                del os.environ["OS_ENVIRON_TEST"]
            else:
                os.environ["OS_ENVIRON_TEST"] = original

    def test_os_environ_set_graph_break(self):
        # [FUNCTION DOC] Checks that writing to the environment mapping inside compiled code
        # causes a graph break (two frames) and the write takes effect. Restores the
        # variable afterwards.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=False)
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1, sets the environment variable to 0, then returns
            # sine of the result.
            x = x + 1
            os.environ["OS_ENVIRON_TEST"] = "0"
            return torch.sin(x)

        x = torch.ones(2, 3)
        try:
            original = os.environ.get("OS_ENVIRON_TEST", None)

            os.environ["OS_ENVIRON_TEST"] = "1"
            res1 = fn(x)
            self.assertEqual(res1, torch.sin(x + 1))
            self.assertEqual(os.environ["OS_ENVIRON_TEST"], "0")
            # Ensure we graph break on os.environ.__setitem__
            self.assertEqual(cnts.frame_count, 2)
        finally:
            if original is None:
                del os.environ["OS_ENVIRON_TEST"]
            else:
                os.environ["OS_ENVIRON_TEST"] = original

    def test_sys_modules(self):
        # [FUNCTION DOC] Checks that lookups in the module table (get, contains, index) are
        # traced under the standard helper with 3 ops, then exports the function and checks the
        # resulting dictionary-contains guards (excluding id-based ones) against an expected
        # text, in the order added.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Asserts on module-table membership, fetches operator and
            # builtins modules, and returns the sum of x and y and the negated absolute value
            # of x.
            mod_a = sys.modules.get("aaaaaaaa")
            assert mod_a is None  # noqa: S101
            assert "bbbbbbbb" not in sys.modules  # noqa: S101

            assert "operator" in sys.modules  # noqa: S101
            operator = sys.modules["operator"]
            builtins = sys.modules.get("builtins")
            operator2 = sys.modules.get("cccccccc", operator)

            return operator.add(x, y), operator2.neg(builtins.abs(x))

        torch._dynamo.testing.standard_test(self, fn, 2, expected_ops=3)

        x = torch.randn(10, 10)
        _, guards = torch._dynamo.export(fn, x, x)
        guard_code = []
        for guard in guards:
            if guard.code_list:
                guard_code += guard.code_list

        # Filter out id-matches that won't reproduce run to run
        guard_code = filter(
            lambda line: "id" not in line and "lookup_backend" not in line,
            guard_code,
        )
        guard_code_str = "\n".join(guard_code)

        # Make sure that the dict_contains are present in the order of added
        self.assertExpectedInline(
            guard_code_str,
            """\
L['x'].size()[1] == L['x'].size()[0]
L['x'].storage_offset() == 0
2 <= L['x'].size()[0]
utils_device.CURRENT_DEVICE == None
str(L['x'].dtype) == 'torch.float32'
str(L['x'].device) == 'cpu'
L['x'].requires_grad == False
L['x'].ndimension() == 2
hasattr(L['x'], '_dynamo_dynamic_indices') == False
hasattr(L['x'], '_dynamo_weak_dynamic_indices') == False
hasattr(L['x'], '_dynamo_unbacked_indices') == False
hasattr(L['x'], '_dynamo_strict_unbacked_indices') == False
hasattr(L['x'], '_dynamo_static_indices') == False
L['x'] is L['y']
not ___dict_contains('aaaaaaaa', G['sys'].modules)
not ___dict_contains('bbbbbbbb', G['sys'].modules)
___dict_contains('operator', G['sys'].modules)
not ___dict_contains('cccccccc', G['sys'].modules)""",
        )

    def test_fold(self):
        # [FUNCTION DOC] Standard test helper: math square root of a constant is folded; 1 op.
        def fn(a):
            # [FUNCTION DOC] HELPER: Adds the square root of 63 to a.
            return a + math.sqrt(63)

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_getattr_dict(self):
        # [FUNCTION DOC] Checks that a function reading the size of a masked-tensor function
        # table dict (imported inside the function) matches eager under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Imports the masked tensor function table and scales x by
            # its length.
            from torch.masked.maskedtensor._ops_refs import _MASKEDTENSOR_FUNCTION_TABLE

            return x * len(_MASKEDTENSOR_FUNCTION_TABLE)

        i = torch.randn(5)
        r1 = fn(i)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r2 = opt_fn(i)
        self.assertEqual(r1, r2)

    def test_tensor_hasattr(self):
        # [FUNCTION DOC] Checks that hasattr on a tensor is evaluated at trace time and guarded:
        # plain tensor takes one branch, a tensor with a custom attribute takes the other.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 2 if x has an attribute named test, else x
            # plus 1.
            if hasattr(x, "test"):
                return x + 2
            else:
                return x + 1

        self.assertEqual(torch.ones(2, 2) + 1, fn(torch.ones(2, 2)))

        inp = torch.ones(2, 2)
        inp.test = None
        self.assertEqual(torch.ones(2, 2) + 2, fn(inp))

    def test_tensor_call_obj_hasattr_view(self):
        # [FUNCTION DOC] Checks that fetching the view method by getattr with a default and
        # calling it compiles and matches eager.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Gets the view method of x via getattr and calls it with 10.
            output3 = getattr(x, "view", None)(10)
            return output3

        x = torch.randn(10)
        self.assertEqual(x.view(10), fn(x))

    def test_mro_type_tensor_no_source(self):
        # [FUNCTION DOC] Checks that walking the MRO of the type of a freshly created tensor (no
        # source) compiles under fullgraph.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Collects the class names along the MRO of the tensor type
            # and returns x, the type, and the names.
            z = []
            input_type = type(torch.ones(2, 2))
            for cls in input_type.__mro__:
                z.append(cls.__name__)

            return x, input_type, z

        inp = torch.ones(2, 2)
        fn(inp)

    def test_tensor_dynamic_method(self):
        # [FUNCTION DOC] Checks that a function attached as an attribute on a Parameter can be
        # called as a method from compiled code.
        def add_one(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        t = torch.nn.Parameter(torch.ones(1))
        t.add_one = add_one

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the attached method on the global parameter and
            # adds x.
            return t.add_one(t) + x

        result = fn(torch.ones(1))
        self.assertEqual(torch.ones(1) + 2, result)

    def test_known_tensor_methods_traced(self):
        # [FUNCTION DOC] Checks that known tensor methods are traced into the graph: one frame
        # and two ops.
        # Verify that known tensor methods (in all_tensor_attrs) are still
        # traced into the graph via the generic proxy path.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns cosine of the absolute value of x.
            return x.abs().cos()

        result = fn(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_tensor_subclass_method_traced(self):
        # [FUNCTION DOC] Checks that a method dynamically added to the tensor class is proxied as
        # a call_method node in the FX graph rather than graph-breaking; one frame.
        # Methods defined on the actual tensor class (including dynamically
        # added ones) should be proxied through the generic call_method path,
        # not graph-broken.  This validates that the guard uses the concrete
        # class_type rather than the static all_tensor_attrs dict.
        def _dynamo_test_method(self):
            # [FUNCTION DOC] HELPER: Method installed on the tensor class returning self plus 1.
            return self + 1

        with unittest.mock.patch.object(
            torch.Tensor, "_dynamo_test_method", _dynamo_test_method, create=True
        ):
            cnt = CompileCounterWithBackend("eager")

            @torch.compile(backend=cnt)
            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the injected tensor method and adds 1.
                y = x._dynamo_test_method()
                return y + 1

            result = fn(torch.randn(4))
            self.assertEqual(cnt.frame_count, 1)
            # Verify _dynamo_test_method appears as a call_method in the FX graph
            call_method_targets = [
                n.target for n in cnt.graphs[0].graph.nodes if n.op == "call_method"
            ]
            self.assertIn("_dynamo_test_method", call_method_targets)

    def test_unknown_tensor_method_graph_break(self):
        # [FUNCTION DOC] Checks that calling a nonexistent tensor method inside compiled code
        # raises AttributeError.
        # Truly unknown methods raise AttributeError during tracing at
        # LOAD_ATTR time (dynamic_getattr), ensuring dynamo does not
        # silently proxy them into the compiled graph.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls a nonexistent method on x.
            y = x._nonexistent_test_method_xyz()
            return y + 1

        with self.assertRaises(AttributeError):
            fn(torch.randn(4))

    def test_shape_unpack(self):
        # [FUNCTION DOC] Checks unpacking a tensor size into two names in compiled code
        # matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Unpacks the size of x and scales x by the second dim.
            a, b = x.size()
            return x * b

        i = torch.randn(5, 10)
        r1 = fn(i)
        opt_fn = torch.compile(fn, backend="eager")
        r2 = opt_fn(i)
        self.assertTrue(same(r1, r2))

    def test_typing_dict(self):
        # [FUNCTION DOC] Checks that a dict keyed by a typing TypeVar can be indexed
        # under fullgraph.
        def fn(d):
            # [FUNCTION DOC] HELPER: Looks up the TypeVar key in the dict.
            return d[T]

        d = {T: torch.randn(3)}
        r1 = fn(d)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r2 = opt_fn(d)
        self.assertEqual(r1, r2)

    def test_tensor__iter__(self):
        # [FUNCTION DOC] Standard test helper: explicit iterator over a tensor with in-place add
        # on each row; expects 20 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Gets an iterator from the tensor, adds 1.0 in place to each
            # element, and returns the last element.
            it = x.__iter__()
            for y in it:
                y.add_(1.0)
            return y

        torch._dynamo.testing.standard_test(
            self,
            fn,
            1,
            expected_ops=20,
        )

    def test_tensor_iter(self):
        # [FUNCTION DOC] Standard test helper: plain loop over a tensor with in-place add;
        # expects 20 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1.0 in place to each element of x and returns the
            # last one.
            for y in x:
                y.add_(1.0)
            return y

        torch._dynamo.testing.standard_test(
            self,
            fn,
            1,
            expected_ops=20,
        )

    def test_tensor_share_memory(self):
        # [FUNCTION DOC] Checks that a module creating a tensor and calling share_memory in
        # forward compiles (without fullgraph) and matches eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with fixed hidden size 64 and 2 layers.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores hidden size and layer count.
                super().__init__()
                self.hidden_size = 64
                self.num_layers = 2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds zeros tensors (one shared-memory), and returns x
                # plus the sums of both.
                batch_size = x.size(0)
                h = torch.zeros(
                    self.num_layers, batch_size, self.hidden_size
                ).share_memory_()
                c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
                return x + h.sum() + c.sum()

        model = Model()
        x = torch.randn(4, 10)
        expected = model(x)
        compiled_model = torch.compile(model, fullgraph=False, backend="eager")
        actual = compiled_model(x)
        self.assertEqual(expected, actual)

    def test_empty_list(self):
        # [FUNCTION DOC] Checks that emptiness checks on a list and on a tuple argument are
        # handled; the same result for both.
        def fn(x, ll):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if the container is empty and not None,
            # else implicitly returns None.
            if len(ll) == 0 and not ll and ll is not None:
                return x + 1

        i = torch.randn(5, 10)
        r1 = fn(i, [])
        opt_fn = torch.compile(fn, backend="eager")
        r2 = opt_fn(i, [])
        r3 = opt_fn(i, ())
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

    def test_min_max_over_iterable(self):
        # [FUNCTION DOC] Standard test helper for min and max over a list, iterator, tuple and
        # varargs of mixed int and shape items; 1 op static, 7 dynamic.
        def get_test_fn(func):
            # [FUNCTION DOC] HELPER: Builds a test function applying the given reducer in four
            # calling forms.
            def _fn(a, b, func=func):
                # [FUNCTION DOC] HELPER: Applies func to a list, iterator, tuple and unpacked
                # args from shape-derived ints and adds the total to a.
                # try all of list, iterator, tuple, vararg.
                lst = [a.shape[0] + 1, 8, a.shape[0]]
                x = func(lst)
                y = func(iter(lst))
                z = func(tuple(lst))
                w = func(*lst)
                return a + (x + y + z + w)

            return _fn

        torch._dynamo.testing.standard_test(
            self,
            get_test_fn(func=min),
            2,
            expected_ops=1,
            expected_ops_dynamic=ifdynstaticdefault(1, 7),
        )
        torch._dynamo.testing.standard_test(
            self,
            get_test_fn(func=max),
            2,
            expected_ops=1,
            expected_ops_dynamic=ifdynstaticdefault(1, 7),
        )

    def test_min_max_tuple_const(self):
        # [FUNCTION DOC] Checks min and max over constant tuples under fullgraph return
        # tuple results.
        # min/max on tuple constants should use ConstantVariable.create
        # to properly handle the tuple return type
        def fn():
            # [FUNCTION DOC] HELPER: Returns min and max of two constant tuples.
            a = min((1, 2), (3, 4))
            b = max((1, 2), (3, 4))
            return a, b

        opt_fn = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        result = opt_fn()
        self.assertEqual(result, ((1, 2), (3, 4)))

    def test_min_max_key_and_default(self):
        # [FUNCTION DOC] Checks min and max with key and default arguments on tensors under
        # fullgraph; one frame, results match eager.
        # Items are non-constant (tensors), so the fully-constant fold path is
        # skipped; the key maps each to a compile-time-constant (ndim), which is
        # what the key comparison requires. Comparison is on key(item) but the
        # original tensor is returned.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Uses max and min on tensors with a key by dimension count,
            # with defaults and on varargs, summing each result.
            tensors = [x, y, z]
            return (
                max(tensors, key=lambda t: t.ndim).sum(),
                min(tensors, key=lambda t: t.ndim).sum(),
                max(tensors, key=lambda t: t.ndim, default=x).sum(),
                max([], default=y).sum(),
                min([x + 1], default=z).sum(),
                max(x, z, key=lambda t: t.ndim).sum(),
                min(x, z, key=lambda t: t.ndim).sum(),
            )

        cnt = CompileCounter()
        opt = torch.compile(fn, backend=cnt, fullgraph=True)
        x, y, z = torch.randn(2), torch.randn(2, 2), torch.randn(2, 2, 2)
        self.assertEqual(opt(x, y, z), fn(x, y, z))
        self.assertEqual(cnt.frame_count, 1)

    def test_min_max_key_none(self):
        # [FUNCTION DOC] Checks key set to None behaves as no key for min and max.
        # key=None behaves like no key argument at all.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds x to max and min of a constant list with key None.
            return x + max([4, 1, 7], key=None) + min([4, 1, 7], key=None)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(3)
        self.assertEqual(opt(x), fn(x))

    def test_min_max_default_none_sentinel(self):
        # [FUNCTION DOC] Checks that default set to None is honored for empty inputs and not
        # confused with the omitted sentinel.
        # A real default=None must be honored for empty iterables and not be
        # confused with the internal "omitted" sentinel (which raises).
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns max and min of empty tuples with default None, max
            # with default 0, and x plus 1.
            return (
                max((), default=None),
                min((), default=None),
                max((), default=0),
                x + 1,
            )

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(3)
        self.assertEqual(opt(x), fn(x))
        self.assertEqual(opt(x)[:3], (None, None, 0))

    def test_min_max_default_varargs_type_error(self):
        # [FUNCTION DOC] Checks that passing default with multiple positional args raises
        # TypeError in eager and compiled code.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls max on two values with default.
            return max(a, b, default=0)

        opt = torch.compile(fn, backend="eager")
        with self.assertRaises(TypeError):
            fn(1, 2)
        with self.assertRaises(TypeError):
            opt(1, 2)

    def test_min_max_empty_no_default_value_error(self):
        # [FUNCTION DOC] Checks that max of an empty list with no default raises ValueError in
        # eager and compiled code.
        def fn(xs):
            # [FUNCTION DOC] HELPER: Returns max of the argument.
            return max(xs)

        opt = torch.compile(fn, backend="eager")
        with self.assertRaises(ValueError):
            fn([])
        with self.assertRaises(ValueError):
            opt([])

    def test_int_base_indexable(self):
        # [FUNCTION DOC] Checks int with a string and a non-constant base object supporting the
        # index protocol under fullgraph; one frame.
        # int(x, base) resolves base via __index__ (PyNumber_AsSsize_t), so a
        # non-constant __index__-able base must be accepted. Previously this
        # raised an "invalid call to builtin op handler" graph break because
        # call_int ignored the base argument entirely.
        class MyIndexable:
            # [CLASS DOC] HELPER: Object that converts to an integer through the index protocol.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value

            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value.
                return self.value

        def fn(x):
            # [FUNCTION DOC] HELPER: Parses strings and bytes with the indexable bases and adds
            # results to x.
            a = int("101", base=MyIndexable(2))
            b = int("101", MyIndexable(36))
            c = int(b"ff", base=MyIndexable(16))
            return x + a + b + c

        cnt = CompileCounter()
        opt = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.randn(3)
        self.assertEqual(opt(x), fn(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_int_base_out_of_range_value_error(self):
        # [FUNCTION DOC] Checks that an int base out of valid range raises ValueError in eager
        # and compiled code.
        class MyIndexable:
            # [CLASS DOC] HELPER: Object whose index conversion yields 37.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns 37.
                return 37

        def fn(xs):
            # [FUNCTION DOC] HELPER: Parses a string with the invalid base.
            return int("43", base=MyIndexable())

        opt = torch.compile(fn, backend="eager")
        with self.assertRaises(ValueError):
            fn([])
        with self.assertRaises(ValueError):
            opt([])

    def test_int_base_non_string_type_error(self):
        # [FUNCTION DOC] Checks that int with a base on a non-string argument raises TypeError in
        # eager and compiled code.
        def fn(n):
            # [FUNCTION DOC] HELPER: Calls int with base 16.
            return int(n, base=16)

        opt = torch.compile(fn, backend="eager")
        with self.assertRaises(TypeError):
            fn(5)
        with self.assertRaises(TypeError):
            opt(5)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_bound_shape_checks(self):
        # [FUNCTION DOC] With scalar output capture, checks that bound checks on an item value
        # through torch._check compile under fullgraph for slicing and indexing.
        def f1(x, y):
            # [FUNCTION DOC] HELPER: Reads a scalar, checks it is non-negative and below y's
            # size, slices y.
            b = x.item()
            torch._check(b >= 0)
            torch._check(b < y.shape[0])
            return y[:b]

        fn1 = torch.compile(f1, fullgraph=True, backend="eager")
        fn1(torch.tensor(4), torch.ones(10))

        def f2(x, index):
            # [FUNCTION DOC] HELPER: Reads a scalar index, bound-checks it against the row count,
            # indexes x.
            idx = index.item()
            torch._check(idx >= 0)
            torch._check(idx < x.size(0))
            return x[idx]

        A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        index = torch.tensor(1, dtype=torch.int64)
        fn2 = torch.compile(f2, fullgraph=True, backend="eager")
        fn2(A, index)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_arange_length_with_float32_dtype(self):
        # [FUNCTION DOC] With scalar capture, checks that an arange of an unbacked length in
        # float32 followed by a size comparison with that length compiles.
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Builds an arange with the item as length; adds 1 if size
            # equals it.
            y = x.item()
            r = torch.arange(y, dtype=torch.float32)

            if r.size(0) == y:
                return r + 1

            return r

        x = torch.tensor([300])
        r = f(x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check(self):
        # [FUNCTION DOC] With scalar capture, checks torch._check on an item value compiles once
        # for two different inputs.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks the item is non-negative and returns an arange up
            # to it.
            y = x.item()
            torch._check(y >= 0)
            return torch.arange(0, y)

        f(torch.tensor([3]))
        f(torch.tensor([4]))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check_symbolic_shape_rel(self):
        # [FUNCTION DOC] With scalar capture, checks that several symbolic relation checks on a
        # shape element compile once for two inputs.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Adds six checks on the first dim size and returns an arange
            # up to the item.
            y = x.item()
            torch._check(x.shape[0] == 1)
            torch._check(x.shape[0] != 2)
            torch._check(x.shape[0] >= 0)
            torch._check(x.shape[0] > 0)
            torch._check(x.shape[0] < 4)
            torch._check(x.shape[0] <= 3)
            return torch.arange(0, y)

        f(torch.tensor([3]))
        f(torch.tensor([4]))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    # Translation validation changes the exception type, don't run with it
    @torch.fx.experimental._config.patch(translation_validation=False)
    def test_torch_check_nonnegative(self):
        # [FUNCTION DOC] With scalar capture and translation validation disabled, checks that
        # branching on an unbacked value even after a non-negative check raises UserError.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks item non-negative then branches on it equal to zero,
            # asserting false there; otherwise an arange.
            y = x.item()
            torch._check(y >= 0)
            # Cannot conditional on unbacked SymInt
            if y == 0:
                assert False  # noqa: B011, S101
            else:
                return torch.arange(0, y)

        self.assertRaises(torch._dynamo.exc.UserError, lambda: f(torch.tensor([3])))

    def test_check_compiles_when_predicate_true_and_message_has_no_closure(self):
        # [FUNCTION DOC] Checks a check with a closure-free message callable compiles when the
        # predicate is true on a dynamic dimension.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with a constant message function
            # and returns x plus 1.
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_constant_and_message_has_no_closure(
        self,
    ):
        # [FUNCTION DOC] Same as the previous test but with a static input size.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with a constant message function
            # and returns x plus 1.
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_and_message_string(self):
        # [FUNCTION DOC] Checks a check with a plain string message compiles when true on a
        # dynamic dimension.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with a string message and returns x
            # plus 1.
            torch._check(x.shape[0] > 3, "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_raises_at_runtime_when_predicate_false_and_message_string(self):
        # [FUNCTION DOC] Checks that a false check with a string message raises RuntimeError with
        # that message at runtime.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with a string message and returns x
            # plus 1.
            torch._check(x.shape[0] > 3, "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

    def test_check_compiles_when_predicate_true_constant_and_message_None(self):
        # [FUNCTION DOC] Checks a true static check with no message compiles.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 and returns x plus 1.
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(4)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_and_message_None(self):
        # [FUNCTION DOC] Checks a true check with no message on a dynamic dimension compiles.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 and returns x plus 1.
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_and_message_has_global(self):
        # [FUNCTION DOC] Sets the module global integer to 1, then checks a true check whose
        # message function reads the global compiles.
        global GLOBAL_INT
        GLOBAL_INT = 1

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with a message formatting the
            # global and returns x plus 1.
            torch._check(x.shape[0] > 3, lambda: f"{GLOBAL_INT} is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_raises_at_runtime_when_predicate_false_and_message_has_global(self):
        # [FUNCTION DOC] Same global message setup, with a false predicate; checks RuntimeError
        # carrying the formatted global message.
        global GLOBAL_INT
        GLOBAL_INT = 1

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with a message formatting the
            # global and returns x plus 1.
            torch._check(x.shape[0] > 3, lambda: f"{GLOBAL_INT} is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            RuntimeError, f"{GLOBAL_INT} is not greater than 3"
        ):
            f(x)

    def test_check_compiles_when_predicate_true_and_message_module_level_fn(self):
        # [FUNCTION DOC] Checks a true check whose message is the module-level function compiles.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with the module-level message
            # function; returns x plus 1.
            torch._check(x.shape[0] > 3, global_check_message)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_raises_at_runtime_when_predicate_false_and_message_module_level_fn(
        self,
    ):
        # [FUNCTION DOC] Checks a false check with the module-level message function raises
        # RuntimeError with the message text.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with the module-level message
            # function; returns x plus 1.
            torch._check(x.shape[0] > 3, global_check_message)
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "global check message"):
            f(x)

    def test_check_compiles_when_predicate_true_and_message_non_str_constant(self):
        # [FUNCTION DOC] Checks a true check with a non-string constant message compiles.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with message 12345 and returns x
            # plus 1.
            torch._check(x.shape[0] > 3, 12345)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_raises_at_runtime_when_predicate_false_and_message_non_str_constant(
        self,
    ):
        # [FUNCTION DOC] Checks a false check with a non-string constant message raises
        # RuntimeError containing the number text.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with message 12345 and returns x
            # plus 1.
            torch._check(x.shape[0] > 3, 12345)
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "12345"):
            f(x)

    def test_check_graph_breaks_when_message_is_unsupported_type(self):
        # [FUNCTION DOC] Checks that a tensor as check message raises Unsupported saying the
        # message cannot be extracted.
        # A tensor message is neither a constant nor a function.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim with the tensor x as message.
            torch._check(x.shape[0] > 3, x)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't extract message from torch._check"
        ):
            f(x)

    def test_check_graph_breaks_when_message_is_bound_method(self):
        # [FUNCTION DOC] Checks that a bound method as check message is rejected with Unsupported
        # (cannot extract message) because the reconstructed function would drop the receiver.
        # A bound method is a function-VT subclass whose reconstructed function
        # drops the receiver, so it is rejected rather than mis-evaluated.
        class M:
            # [CLASS DOC] HELPER: Holder with a message method and a forward using it as
            # check message.
            def msg(self):
                # [FUNCTION DOC] HELPER: Returns a fixed message string.
                return "bound method message"

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Checks first dim above 3 with the bound method message;
                # returns x plus 1.
                torch._check(x.shape[0] > 3, self.msg)
                return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't extract message from torch._check"
        ):
            torch.compile(M().forward, backend="eager", fullgraph=True)(x)

    def test_check_raises_at_runtime_when_predicate_false_and_message_None(self):
        # [FUNCTION DOC] Checks a false check with no message raises RuntimeError at runtime
        # (dynamic input).
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 and returns x plus 1.
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, None):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_constant_and_message_None(
        self,
    ):
        # [FUNCTION DOC] Checks a false check with no message raises RuntimeError for a
        # static input.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 and returns x plus 1.
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, None):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_and_message_has_no_closure(
        self,
    ):
        # [FUNCTION DOC] Checks a false check with closure-free message callable raises
        # RuntimeError with that message (dynamic input).
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with constant message function;
            # returns x plus 1.
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_constant_and_message_has_no_closure(
        self,
    ):
        # [FUNCTION DOC] Same as above for a static input size.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with constant message function;
            # returns x plus 1.
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

    def test_check_assert_error_at_runtime_when_predicate_false_and_message_has_closure(
        self,
    ):
        # [FUNCTION DOC] Checks a false check whose message function reads a tensor size (closure
        # over input) raises Unsupported about extracting the message.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with a message built from x's size;
            # returns x plus 1.
            torch._check(x.shape[0] > 3, lambda: f"{x.shape[0]} is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't .* torch._check"
        ):
            f(x)

    def test_check_assert_error_at_runtime_when_predicate_true_and_message_has_closure(
        self,
    ):
        # [FUNCTION DOC] Same as above with a true predicate; still raises Unsupported because
        # the closure message is not supported.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks first dim above 3 with a message built from x's size;
            # returns x plus 1.
            torch._check(x.shape[0] > 3, lambda: f"{x.shape[0]} is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't .* torch._check"
        ):
            f(x)

    def test_check_with_constant_true(self):
        # [FUNCTION DOC] Checks check_with of ValueError with true predicate compiles.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Calls check_with and returns x plus 1.
            torch._check_with(ValueError, x.shape[0] > 3, lambda: "bad shape")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

    def test_check_with_constant_false_raises(self):
        # [FUNCTION DOC] Checks that a false check_with raises ValueError or RuntimeError.
        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Calls check_with with a failing size condition and returns x
            # plus 1.
            torch._check_with(ValueError, x.shape[0] > 3, lambda: "bad shape")
            return x + 1

        with self.assertRaises((ValueError, RuntimeError)):
            f(torch.randn(3))

    def test_check_with_try_except(self):
        # [FUNCTION DOC] Checks that a failing check_with inside try/except ValueError is handled
        # in compiled code under fullgraph.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Catches ValueError from check_with with a false predicate
            # and returns x plus 1.
            try:
                torch._check_with(ValueError, False, lambda: "expected")
            except ValueError:
                pass
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

    def test_check_value(self):
        # [FUNCTION DOC] Checks check_value compiles with a true predicate and raises ValueError
        # or RuntimeError when false.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Calls check_value with true size condition; returns x
            # plus 1.
            torch._check_value(x.shape[0] > 3, lambda: "bad value")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

        @torch.compile(backend="eager")
        def f_fail(x):
            # [FUNCTION DOC] HELPER: Same as above but invoked with a failing input.
            torch._check_value(x.shape[0] > 3, lambda: "bad value")
            return x + 1

        # RuntimeError because graph lowers to torch._check
        with self.assertRaises((ValueError, RuntimeError)):
            f_fail(torch.randn(3))

    def test_check_index(self):
        # [FUNCTION DOC] Checks check_index compiles when true and raises IndexError or
        # RuntimeError when false.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Calls check_index with size condition; returns x plus 1.
            torch._check_index(x.shape[0] > 3, lambda: "bad index")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

        @torch.compile(backend="eager")
        def f_fail(x):
            # [FUNCTION DOC] HELPER: Same call used for the failing input.
            torch._check_index(x.shape[0] > 3, lambda: "bad index")
            return x + 1

        with self.assertRaises((IndexError, RuntimeError)):
            f_fail(torch.randn(3))

    def test_check_type(self):
        # [FUNCTION DOC] Checks check_type compiles when true and raises TypeError or
        # RuntimeError when false.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Calls check_type with size condition; returns x plus 1.
            torch._check_type(x.shape[0] > 3, lambda: "bad type")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

        @torch.compile(backend="eager")
        def f_fail(x):
            # [FUNCTION DOC] HELPER: Same call used for the failing input.
            torch._check_type(x.shape[0] > 3, lambda: "bad type")
            return x + 1

        with self.assertRaises((TypeError, RuntimeError)):
            f_fail(torch.randn(3))

    def test_check_not_implemented(self):
        # [FUNCTION DOC] Checks check_not_implemented compiles when true and raises
        # NotImplementedError or RuntimeError when false.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Calls check_not_implemented with size condition; returns x
            # plus 1.
            torch._check_not_implemented(x.shape[0] > 3, lambda: "not impl")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

        @torch.compile(backend="eager")
        def f_fail(x):
            # [FUNCTION DOC] HELPER: Same call used for the failing input.
            torch._check_not_implemented(x.shape[0] > 3, lambda: "not impl")
            return x + 1

        with self.assertRaises((NotImplementedError, RuntimeError)):
            f_fail(torch.randn(3))

    def test_check_tensor_all(self):
        # [FUNCTION DOC] Checks check_tensor_all with an all-positive input compiles.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks all elements are positive and returns x plus 1.
            torch._check_tensor_all(x > 0, lambda: "all must be positive")
            return x + 1

        x = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(f(x), x + 1)

    def test_check_tensor_all_with(self):
        # [FUNCTION DOC] Checks check_tensor_all_with ValueError for an all-positive
        # input compiles.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Checks all elements are positive with the ValueError type;
            # returns x plus 1.
            torch._check_tensor_all_with(
                ValueError, x > 0, lambda: "all must be positive"
            )
            return x + 1

        x = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(f(x), x + 1)

    def test_check_tensor_predicate_raises_clear_error(self):
        # [FUNCTION DOC] Checks that passing a tensor predicate to check raises TypeError with a
        # message suggesting the tensor-all variant, without leaking FakeTensor in the message.
        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Checks a tensor predicate and returns log of x.
            torch._check(x > 0)
            return torch.log(x)

        with self.assertRaisesRegex(
            TypeError, r"cond must be a bool.*torch[.]_check_tensor_all"
        ) as cm:
            f(torch.rand(1))

        self.assertNotIn("FakeTensor", str(cm.exception))

    def test_check_with_closure_constant(self):
        # [FUNCTION DOC] Checks check_with accepts a message callable closing over a local
        # string constant.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Defines a local message, calls check_with and returns x
            # plus 1.
            msg = "shape check failed"
            torch._check_with(ValueError, x.shape[0] > 3, lambda: msg)
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

    def test_assert(self):
        # [FUNCTION DOC] Checks that an assert on an always-false shape comparison raises
        # AssertionError, and that an always-true assert compiles with 1 op.
        @torch.compile(backend="eager")
        def fn1(x):
            # [FUNCTION DOC] HELPER: Asserts the shape differs from itself, which always fails.
            assert x.shape != x.shape  # noqa: S101

        with self.assertRaises(AssertionError):
            a = torch.randn(10)
            fn1(a)

        def fn2(x):
            # [FUNCTION DOC] HELPER: Asserts shape equals itself, returns absolute value of x.
            assert x.shape == x.shape  # noqa: S101
            return x.abs()

        torch._dynamo.testing.standard_test(self, fn=fn2, nargs=1, expected_ops=1)

    # When we unspecialize float, we wobble this test by changing
    # the op count since previously we would just specialize and constant
    # fold floats into the graph, whereas when we unspecialize we will have
    # ops for item, add, and all other tensorified operations. Since this
    # test really isn't testing that, we purposely specialize floats here.
    @torch._dynamo.config.patch(specialize_float=True)
    def test_config_obj(self):
        # [FUNCTION DOC] With float specialization on, checks a config object with float and int
        # fields drives a loop; recompiles on field changes; final value 7 and 8 ops.
        class Cfg:
            # [CLASS DOC] HELPER: Config holder with a float val and an int count.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets val to 0.5 and count to 3.
                self.val = 0.5
                self.count = 3

        def fn(x, cfg):
            # [FUNCTION DOC] HELPER: Adds cfg.val to x, cfg.count times.
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        cfg1.val = 1.0
        cfg2 = Cfg()
        v = torch.zeros(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        v = opt_fn(v, cfg1)  # 3
        v = opt_fn(v, cfg2)  # 4.5
        cfg2.count = 1
        v = opt_fn(v, cfg2)  # 5
        cfg2.val = 2.0
        v = opt_fn(v, cfg2)  # 7
        self.assertEqual(v[0], 7)
        self.assertEqual(cnts.op_count, 8)

    def test_config_getattr_default(self):
        # [FUNCTION DOC] Checks reading an optional attribute with getattr default; setting it to
        # true then false changes the path and recompiles three frames total.
        class Cfg:
            # [CLASS DOC] HELPER: Config holder with val 0.5 and count 10.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets val and count.
                self.val = 0.5
                self.count = 10

        def fn(x, cfg):
            # [FUNCTION DOC] HELPER: Returns x plus 7 if the optional flag is set; else adds
            # cfg.val to x, cfg.count times.
            if getattr(cfg, "just_add_7", False):
                return x + 7
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        v = torch.zeros(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        cfg1.just_add_7 = True
        self.assertEqual(opt_fn(v, cfg1)[0], 7)
        self.assertEqual(opt_fn(v, cfg1)[0], 7)
        cfg1.just_add_7 = False
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(cnts.frame_count, 3)

    def test_size_input(self):
        # [FUNCTION DOC] Checks that unpacking a size, a tuple and a list recompiles once per
        # input type (three frames) with identical results.
        def fn(x, s):
            # [FUNCTION DOC] HELPER: Unpacks s into two numbers and adds their difference to x.
            a, b = s
            return x + (a - b)

        v = torch.zeros(10, 20)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v, v.size())[0, 0], -10)
        self.assertEqual(opt_fn(v, (10, 20))[0, 0], -10)
        self.assertEqual(opt_fn(v, [10, 20])[0, 0], -10)
        # One recompile per differing input type
        self.assertEqual(cnts.frame_count, 3)

    def test_cell_output1(self):
        # [FUNCTION DOC] Checks that a function assigning to an enclosing-scope variable compiles
        # with two ops and updates the variable.
        out = None

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Stores a plus 10 times b in the enclosing variable.
            nonlocal out
            out = a + b * 10

        v = torch.Tensor([100])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIsNone(opt_fn(v, v))
        self.assertEqual(out[0], 1100)
        self.assertEqual(cnts.op_count, 2)

    def test_cell_output2(self):
        # [FUNCTION DOC] Same as above with an unsupported call in the middle; checks the
        # enclosing variable holds the right value and 3 ops.
        out = None

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls the unsupported function, then stores a combination in
            # the enclosing variable.
            nonlocal out
            c = unsupported(a, b)
            out = a + b * 10 + c

        v = torch.Tensor([100])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIsNone(opt_fn(v, v))
        self.assertEqual(out[0], 1200)
        self.assertEqual(cnts.op_count, 3)

    def test_return_nested_function(self):
        # [FUNCTION DOC] Checks nested function return (continues in later part).
        out = None

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds values c and d and a nested function.
            nonlocal out
            c = a + b
            d = a + 1.0

            def fn2(f: int = 7, g: float = 9.0):
                # [FUNCTION DOC] HELPER: Nested function with defaults for f and g; stores a plus
                # b times 10 in the enclosing variable and returns c times f minus d times g.
                nonlocal out
                out = a + b * 10
                return c * f - d * g

            return fn2

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn_ret = torch.compile(opt_fn(v1, v2), backend=cnts)
        self.assertEqual(opt_fn_ret(1.5)[0], -459)
        self.assertEqual(out[0], 2100)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 7)

    def test_tensor_dict1(self):
        # [FUNCTION DOC] Checks indexing a dict of tensors passed as input under fullgraph: one
        # frame, two ops, value -200.
        def fn(inputs):
            # [FUNCTION DOC] HELPER: Returns entry a minus entry b times 1.5.
            return inputs["a"] - inputs["b"] * 1.5

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn({"a": v1, "b": v2})[0], -200)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_tensor_dict3(self):
        # [FUNCTION DOC] Checks key-set union of two dict inputs and conditional accumulation
        # over keys; one frame, 5 ops, matches eager.
        def fn(inputs_a, inputs_b):
            # [FUNCTION DOC] HELPER: Iterates over the union of keys, adding each present tensor
            # from either dict into a running total.
            total = torch.zeros(1)
            input_keys = inputs_a.keys() | inputs_b.keys()
            for k in input_keys:
                if k in inputs_a:
                    total += inputs_a[k]
                if k in inputs_b:
                    total += inputs_b[k]
            return total

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(
            opt_fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
            fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
        )
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_tensor_dict2(self):
        # [FUNCTION DOC] Checks three ways of iterating a dict (items, values, keys) each compile
        # under fullgraph; each sums to 300; 3 frames, 9 ops.
        def fn1(inputs):
            # [FUNCTION DOC] HELPER: Sums tensors by iterating over items.
            total = torch.zeros(1)
            for k, v in inputs.items():
                total += v
            return total

        def fn2(inputs):
            # [FUNCTION DOC] HELPER: Sums tensors by iterating over values.
            total = torch.zeros(1)
            for v in inputs.values():
                total += v
            return total

        def fn3(inputs):
            # [FUNCTION DOC] HELPER: Sums tensors by iterating over keys and indexing.
            total = torch.zeros(1)
            for k in inputs.keys():
                total += inputs[k]
            return total

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch.compile(fn1, backend=cnts, fullgraph=True)
        opt_fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        opt_fn3 = torch.compile(fn3, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn2({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn3({"a": v1, "b": v2})[0], 300)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 9)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_user_code_statically_known(self):
        # [FUNCTION DOC] With scalar capture on, checks the helper functions for statically known
        # truth and static values: before a bound check the comparison is not statically known,
        # after it is; python floats are static and a constant-checked item is static.
        from torch.fx.experimental.symbolic_shapes import (
            has_static_value,
            statically_known_true,
        )

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Raises if the size comparison is statically known true
            # too early; then checks the size is at least 10 and returns the statically
            # known results.
            # At this point, this isn't statically known, only the hint says so.
            if statically_known_true(x.shape[0] > 9):
                raise Exception()
            torch._check(x.shape[0] >= 10)
            # But now it is.
            return statically_known_true(x.shape[0] > 9), has_static_value(x.shape[0])

        x = torch.zeros(10)
        torch._dynamo.mark_dynamic(x, 0)
        self.assertEqual(f(x), (True, False))

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def g(x, y):
            # [FUNCTION DOC] HELPER: Reads an item, checks it equals 3, returns static-value
            # flags for a float constant and the item.
            n = x.item()
            torch._check(n == 3)
            return has_static_value(4.0), has_static_value(n)

        out = g(torch.tensor([3]), torch.zeros(1))
        self.assertEqual(out, (True, True))

    def test_dictcomp(self):
        # [FUNCTION DOC] Checks a dict comprehension over input items compiles in one frame with
        # 2 ops.
        def fn1(inputs):
            # [FUNCTION DOC] HELPER: Returns a dict with each value incremented.
            return {k: v + 1 for k, v in inputs.items()}

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch.compile(fn1, backend=cnts)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})["a"], 101)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})["b"], 201)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_listcomp(self):
        # [FUNCTION DOC] Checks a list comprehension feeding concat and sum compiles in one frame
        # with 4 ops.
        def fn2(inputs):
            # [FUNCTION DOC] HELPER: Sums the concatenation of all values plus 1.
            return torch.sum(torch.cat([v + 1 for k, v in inputs.items()], 0))

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn2 = torch.compile(fn2, backend=cnts)
        self.assertEqual(opt_fn2({"a": v1, "b": v2}), 302)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_is_floating_point(self):
        # [FUNCTION DOC] Standard test helper for the is-floating-point function used as a branch
        # condition; 3 ops. Returns the helper's result.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds 1.0, adds b if b is floating, then adds 2.0.
            x = a + 1.0
            if torch.is_floating_point(b):
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_floating_point2(self):
        # [FUNCTION DOC] Same as above using the tensor method form.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds 1.0, adds b if b is floating (method form), then
            # adds 2.0.
            x = a + 1.0
            if b.is_floating_point():
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor(self):
        # [FUNCTION DOC] Standard test helper for the is-tensor function as branch condition;
        # 3 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds 1.0, adds b if it is a tensor, then adds 2.0.
            x = a + 1.0
            if torch.is_tensor(b):
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor2(self):
        # [FUNCTION DOC] Checks is_tensor on a dict input versus a tensor input gives eager-equal
        # results for both.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 for tensors, else a tensor of ones of shape
            # two by three.
            if torch.is_tensor(x):
                return x + 1
            else:
                return torch.ones([2, 3])

        x1 = {"input": torch.rand(2, 3)}
        x2 = torch.rand(2, 3)
        ref1 = fn(x1)
        ref2 = fn(x2)
        opt_fn = torch.compile(fn, backend="eager")
        res1 = opt_fn(x1)
        res2 = opt_fn(x2)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_numel(self):
        # [FUNCTION DOC] Standard test helper for numel and nelement; 3 ops static, 4 dynamic.
        def fn(a):
            # [FUNCTION DOC] HELPER: Returns a tuple of two tensors built by adding
            # element counts.
            return (a + a.numel() + torch.numel(a), a + a.nelement())

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=3,
            expected_ops_dynamic=ifdynstaticdefault(3, 4),
        )

    def test_pair(self):
        # [FUNCTION DOC] Standard test helper for the nn utils pair and ntuple helpers; 5 ops
        # either way.
        def fn(a):
            # [FUNCTION DOC] HELPER: Builds zeros from the pair of the size, adds a and adds the
            # sum of ones from a triple.
            return (
                torch.zeros(torch.nn.modules.utils._pair(a.size()))
                + a
                + torch.ones(torch.nn.modules.utils._ntuple(3)(3)).sum()
            )

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=5,
            expected_ops_dynamic=5,
        )

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_tensor_item_capture(self):
        # [FUNCTION DOC] With scalar capture on, checks item of a sum is captured: one frame and
        # 4 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the item of the sum of a and b.
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
    def test_tensor_item_no_capture(self):
        # [FUNCTION DOC] With scalar capture off, checks item causes a split: one frame, 2 ops,
        # correct value.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the item of the sum of a and b.
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple1(self):
        # [FUNCTION DOC] Checks creation, field access, indexing, and rebuild of a namedtuple in
        # compiled code: one frame, 2 ops, last field 50.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds a namedtuple, then returns a new one from its fields.
            tmp = MyTuple(a, b, a + b)
            return MyTuple(tmp.a, tmp[1], tmp.ab + b)

        v1 = torch.Tensor([10])
        v2 = torch.Tensor([20])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2).ab, 50)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple2(self):
        # [FUNCTION DOC] Checks namedtuple input unpacking, hasattr, index, and fields length;
        # value 10, one frame, 4 ops.
        def fn(packed):
            # [FUNCTION DOC] HELPER: Unpacks the tuple, uses a field, indexes, counts fields
            # and sums.
            a, b, c = packed
            if hasattr(packed, "b"):
                b = packed.b + 1
            c = packed[2]
            d = len(packed._fields)
            return a + b + c + d

        v1 = torch.Tensor([1])
        v2 = torch.Tensor([2])
        v3 = torch.Tensor([3])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(MyTuple(v1, v2, v3))[0], 10)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_namedtuple3(self):
        # [FUNCTION DOC] Checks isinstance with a namedtuple class as a branch condition.
        def fn(x, packed):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if packed is the namedtuple type, else x
            # minus 1.
            if isinstance(packed, MyTuple):
                return x + 1
            else:
                return x - 1

        x = torch.rand([2, 3])
        packed = MyTuple(1, 2, 3)
        ref = fn(x, packed)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, packed)
        self.assertTrue(same(ref, res))

    def test_namedtuple_with_custom_getitem(self):
        # [FUNCTION DOC] Checks that a namedtuple subclass defining a custom item getter is
        # accepted under fullgraph and guards evaluate on the second call.
        @torch.compile(fullgraph=True, backend="eager")
        def f(my_tuple):
            # [FUNCTION DOC] HELPER: Returns the first field plus 1.
            return my_tuple.a + 1

        class MyTuple(typing.NamedTuple):
            # [CLASS DOC] HELPER: Typed namedtuple of two tensors that overrides item access.
            a: torch.Tensor
            b: torch.Tensor

            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: Builds a new tuple from the enclosing tensors indexed by
                # index (ignores self).
                return MyTuple(a[index], b[index])

        a = torch.randn(2)
        b = torch.randn(2)

        out = f(MyTuple(a, b))
        self.assertTrue(same(a + 1, out))

        # Test guard evaluation in the second call
        out = f(MyTuple(a, b))
        self.assertTrue(same(a + 1, out))

    def test_namedtuple_source_dynamic_attributes(self):
        # [FUNCTION DOC] Checks that a dynamic attribute added to an input namedtuple subclass
        # instance inside compiled code persists on output.
        class MyNamedTuple(typing.NamedTuple):
            # [CLASS DOC] HELPER: Typed namedtuple with fields a and b.
            a: torch.Tensor
            b: torch.Tensor

        class MyNamedTupleSubclass(MyNamedTuple):
            # [CLASS DOC] HELPER: Empty subclass allowing attribute assignment.
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def f(tup):
            # [FUNCTION DOC] HELPER: Sets attribute c to a new tensor and returns the tuple.
            c = torch.tensor(3.0)
            tup.c = c  # Add dynamic attribute
            return tup

        extended_tup = MyNamedTupleSubclass(a=torch.tensor([1.0]), b=torch.tensor(2.0))
        result = f(extended_tup)
        # Verify the tuple has the expected structure
        self.assertEqual(result.a, torch.tensor([1.0]))
        self.assertEqual(result.b, torch.tensor(2.0))
        self.assertTrue(hasattr(result, "c"))
        self.assertEqual(result.c, torch.tensor(3.0))

    def test_namedtuple_sourceless_dynamic_attributes(self):
        # [FUNCTION DOC] Same as above but the tuple is created inside the compiled function.
        class MyNamedTuple(typing.NamedTuple):
            # [CLASS DOC] HELPER: Typed namedtuple with fields a and b.
            a: torch.Tensor
            b: torch.Tensor

        class MyNamedTupleSubclass(MyNamedTuple):
            # [CLASS DOC] HELPER: Empty subclass allowing attribute assignment.
            pass

        @torch.compile(backend="eager")
        def f():
            # [FUNCTION DOC] HELPER: Creates the tuple, sets dynamic attribute c, returns
            # the tuple.
            # Create namedtuple inside function (sourceless)
            tup = MyNamedTupleSubclass(a=torch.tensor([1.0]), b=torch.tensor(2.0))
            # Add dynamic attribute
            tup.c = torch.tensor(3.0)
            return tup

        result = f()
        # Verify the tuple has the expected structure
        self.assertEqual(result.a, torch.tensor([1.0]))
        self.assertEqual(result.b, torch.tensor(2.0))
        # Verify the dynamic attribute is preserved
        self.assertTrue(hasattr(result, "c"))
        self.assertEqual(result.c, torch.tensor(3.0))

    def test_namedtuple___eq__(self):
        # [FUNCTION DOC] Checks equality between a namedtuple and a plain tuple of equal values
        # inside compiled code is true.
        class MyNamedTuple(typing.NamedTuple):
            # [CLASS DOC] HELPER: Typed namedtuple with two ints.
            a: int
            b: int

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns sine of x and the equality result.
            t1 = MyNamedTuple(a=1, b=2)
            t2 = (1, 2)
            return x.sin(), (t1 == t2)

        x = torch.randn(2)
        res = f(x)
        self.assertTrue(res[1])

    def test_structseq1(self):
        # [FUNCTION DOC] Checks constructing the max structseq return type from a tuple under
        # fullgraph matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds the max return type from x and y.
            return torch.return_types.max((x, y))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    def test_structseq2(self):
        # [FUNCTION DOC] Checks constructing the linalg qr structseq and converting to a tuple
        # matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds the qr return type from scaled and shifted inputs,
            # converts to tuple.
            return tuple(torch.return_types.linalg_qr((2 * x, y - 1)))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    def test_structseq_repr(self):
        # [FUNCTION DOC] Checks calling repr on a structseq raises Unsupported under fullgraph,
        # and without fullgraph compiles one frame.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Computes max along dim 0, takes repr, returns the values.
            result = torch.max(x, dim=0)
            s = repr(result)
            return result.values

        x = torch.randn(3, 2)

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, fullgraph=True, backend="eager")(x)

        opt_fn = torch._dynamo.optimize(cnts)(fn)
        result = opt_fn(x)
        self.assertEqual(cnts.frame_count, 1)

    def test_range_input(self):
        # [FUNCTION DOC] Standard test helper for a loop over a range passed as an argument;
        # 3 ops.
        def fn(a, rng):
            # [FUNCTION DOC] HELPER: Adds each element of rng to a.
            x = a
            for i in rng:
                x = x + i
            return x

        def fn1(a):
            # [FUNCTION DOC] HELPER: Calls fn with range of 3.
            return fn(a, rng=range(3))

        return torch._dynamo.testing.standard_test(
            self, fn=fn1, nargs=1, expected_ops=3
        )

    def test_range_with_shape(self):
        # [FUNCTION DOC] Standard test helper for a loop with a shape-dependent range; 9 ops.
        def fn(a):
            # [FUNCTION DOC] HELPER: Adds 1 in place for each element of the range up to the
            # first dim.
            for i in range(1, a.shape[0]):
                a += 1
            return a

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=9,
        )

    def test_range_iter_guards(self):
        # [FUNCTION DOC] Checks guards for range iterators when a list comprehension is the root
        # frame; outputs are ones of lengths 2 and 3.
        @torch.compile(backend="eager")
        def func():
            # [FUNCTION DOC] HELPER: Defines a non-recursively disabled helper and calls it with
            # 2 and 3.
            @torch._dynamo.disable(recursive=False)
            def run(n):
                # [FUNCTION DOC] HELPER: Builds a list of ones of length n and concatenates it.
                # For python <= 3.11, list comprehension is implemented by
                # desugaring to:
                # 1. creation of an iterator object
                # 2. calling a new `listcomp` function with (1)
                #
                # In this test we force Dynamo to trace through (2) as the root
                # frame, thereby ensuring we have the right guards for range
                # iterators.
                xs = [torch.ones(1) for i in range(n)]
                return torch.concat(xs)

            return run(2), run(3)

        res2, res3 = func()
        self.assertTrue(same(res2, torch.ones(2)))
        self.assertTrue(same(res3, torch.ones(3)))

    def test_range___iter__(self):
        # [FUNCTION DOC] Checks calling the iterator method of range and advancing it
        # under fullgraph.
        def func(x):
            # [FUNCTION DOC] HELPER: Adds the first element of the range iterator to x.
            it = range(3).__iter__()
            return x + next(it)

        opt_func = torch.compile(func, backend="eager", fullgraph=True)
        x = torch.randn(3)
        self.assertTrue(same(func(x), opt_func(x)))

    def test_range_iter_side_effects(self):
        # [FUNCTION DOC] Checks a range iterator passed as input is advanced as a side effect:
        # result ones and next value 2.
        @torch.compile(backend="eager", fullgraph=True)
        def run(x, it):
            # [FUNCTION DOC] HELPER: Takes the next value of the iterator and adds it to x.
            n = next(it)
            return x + n

        it = iter(range(1, 3))
        res = run(torch.zeros(1), it)
        self.assertTrue(same(res, torch.ones(1)))
        self.assertEqual(next(it), 2)

    def test_build_tuple_unpack(self):
        # [FUNCTION DOC] Standard test helper for packing and unpacking tuples into call
        # arguments, 2 ops each form.
        def fn1(a, b, c):
            # [FUNCTION DOC] HELPER: Returns a minus b divided by c.
            return a - b / c

        def fn2(a, b, c):
            # [FUNCTION DOC] HELPER: Builds tuples, unpacks them into a combined argument tuple
            # and calls fn1.
            tmp1 = (a,)
            tmp2 = (b, c)
            args = (*tmp1, *tmp2)
            return fn1(*args)

        def fn3(a, *args):
            # [FUNCTION DOC] HELPER: Forwards star args to fn1.
            return fn1(a, *args)

        torch._dynamo.testing.standard_test(self, fn=fn2, nargs=3, expected_ops=2)
        torch._dynamo.testing.standard_test(self, fn=fn3, nargs=3, expected_ops=2)

    def test_list_mul(self):
        # [FUNCTION DOC] Checks multiplication of a list by an int on both sides yields the right
        # list and no frame is recorded for it.
        def fn(count):
            # [FUNCTION DOC] HELPER: Multiplies a single-None list by count on both sides.
            head_mask = count * [None] * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(2), [None] * 4)
        # TODO: the captured frame here is a bit goofy, because we don't
        # output anything and none of the traced operations have side
        # effects.  Probably need better heuristic for bailing on
        # dynamo if there are no outputs
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")

    def test_list_slice_mul(self):
        # [FUNCTION DOC] Checks multiplication of a sliced list; 0 frames and 0 ops.
        def fn(count):
            # [FUNCTION DOC] HELPER: Slices a list and multiplies by count on both sides.
            a = [1, 2, 3]
            head_mask = count * a[1:] * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(2), [2, 3] * 4)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")

    def test_tuple_mul(self):
        # [FUNCTION DOC] Checks multiplication of a tuple by count on both sides; 0 frames and
        # 0 ops.
        def fn(count):
            # [FUNCTION DOC] HELPER: Multiplies a constant tuple by count both sides.
            head_mask = count * (2, 3) * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(2), (2, 3) * 4)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")

    def test_tuple_mul_with_shape(self):
        # [FUNCTION DOC] Standard test helper for tuple repetition with a shape element; 1 op.
        def fn(a):
            # [FUNCTION DOC] HELPER: Repeats a tuple containing the shape element and adds an
            # element of it to a.
            x = a.shape[0]
            y = 2 * (x, 3) * 2
            return a + y[4]

        # expect 3 ops post folding for dynamic case: size, index, add
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=1
        )

    def test_tuple_iadd_with_shape(self):
        # [FUNCTION DOC] Standard test helper for in-place tuple concatenation with shape-derived
        # tensors and constants; 4 ops.
        def fn(a):
            # [FUNCTION DOC] HELPER: Builds a tuple and extends it twice with tuples.
            output = (a + a.shape[0], a - a.shape[0])
            # tuple += tuple
            output += (a - a.shape[0], a + a.shape[0])
            # tuple += constant tuple
            output += (2, 3)
            return output

        # expect 4 add / subs for static
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=4, expected_ops_dynamic=4
        )

    def test_list_iadd_with_shape(self):
        # [FUNCTION DOC] Standard test helper for in-place list concatenation with a list and a
        # tuple; 6 ops.
        def fn(a):
            # [FUNCTION DOC] HELPER: Builds a list and extends it with a list and a tuple.
            output = [a + a.shape[0], a - a.shape[0]]
            # list += list
            output += [a - a.shape[0], a + a.shape[0]]
            # list += tuple
            output += (a + a.shape[0], a - a.shape[0])
            return output

        # expect 6 add / subs for static

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=6, expected_ops_dynamic=6
        )

    def test_list_iadd_side_effect(self):
        # [FUNCTION DOC] Checks that in-place list extension before a graph break is visible to
        # the caller and matches eager.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Appends b through in-place add, graph breaks, returns a.
            a += [b]
            torch._dynamo.graph_break()
            return a

        a = [1, 2, 3]
        b = torch.ones(2, 2)

        opt_fn = torch.compile(fn, backend="eager")

        exp = fn(a, b)

        a = [1, 2, 3]
        b = torch.ones(2, 2)
        act = opt_fn(a, b)

        self.assertEqual(exp, act)

    def test_class_binop(self):
        # [FUNCTION DOC] Checks a user class with custom addition compiles in one frame with 1
        # op, while an unsupported subtraction raises Unsupported under fullgraph.
        class Foo:
            # [CLASS DOC] HELPER: Wraps a tensor and supports addition returning a new wrapper.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new wrapper with summed tensors.
                return Foo(self.x + other.x)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        x = torch.randn(2)
        a, b = Foo(x), Foo(x + 1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(a, b).x, 2 * x + 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns a minus b, unsupported for the class.
            return a - b

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertRaises(torch._dynamo.exc.Unsupported, opt_fn, a, b)

    def test_user_getattr1(self):
        # [FUNCTION DOC] Checks attribute fallback on a dict subclass: attribute reads map to key
        # lookups; one frame, 2 ops.
        class MyConfig(dict):
            # [CLASS DOC] HELPER: Dict subclass that resolves missing attributes as keys.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: Returns the dict entry for name.
                return self[name]

        def fn(cfg, x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y plus cfg.offset.
            return x + y + cfg.offset

        x = torch.randn(10)
        cfg = MyConfig(offset=5)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_user_getattr2(self):
        # [FUNCTION DOC] Checks that class attributes and instance attributes take precedence
        # over a catch-all attribute hook; value is 1 minus 2 plus 3; 3 ops.
        class MyConfig:
            # [CLASS DOC] HELPER: Config with a class attribute, instance attribute and fallback
            # always returning 3.
            defined_on_class = 1

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the instance attribute to 2.
                self.defined_on_object = 2

            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: Returns 3 for any missing attribute.
                return 3

        def fn(cfg, x):
            # [FUNCTION DOC] HELPER: Combines all three attribute reads with x.
            return x + cfg.defined_on_class - cfg.defined_on_object + cfg.not_defined

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x), x + 1 - 2 + 3))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_getset_descriptor(self):
        # [FUNCTION DOC] Checks calling the get method of a getset descriptor for tensor shape
        # under fullgraph, and that a second call does not recompile.
        def fn(g, x):
            # [FUNCTION DOC] HELPER: Computes sine only to keep the frame, returns the descriptor
            # get applied to x.
            # Just to make Dynamo not skip the frame
            torch.sin(x)
            return g.__get__(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        g = torch.Tensor.shape

        res = opt_fn(g, torch.ones(2, 2))
        exp_res = fn(g, torch.ones(2, 2))
        self.assertEqual(res, exp_res)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            res = opt_fn(g, torch.ones(2, 2))

    def test_set_descriptor(self):
        # [FUNCTION DOC] Checks a data descriptor with a custom set method: assignment inside
        # compiled code runs the set; fields match eager.
        class Field:
            # [CLASS DOC] HELPER: Descriptor whose set adds twice the value to the instance
            # dict entry.
            def __set__(self, obj, value):
                # [FUNCTION DOC] HELPER: Adds twice value to the field entry.
                obj.__dict__["field"] += value * 2

        class Foo:
            # [CLASS DOC] HELPER: Holds the descriptor, storing 0 initial field in the
            # instance dict.
            field = Field()

            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets dict field 0.
                self.__dict__["field"] = 0

        def fn(x, foo):
            # [FUNCTION DOC] HELPER: Assigns 10 to the field and returns x plus the field.
            foo.field = 10
            return x + foo.field

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        x = torch.zeros(2)
        foo1, foo2 = Foo(), Foo()

        ref = fn(x, foo1)
        res = opt_fn(x, foo2)
        self.assertEqual(ref, res)
        self.assertEqual(foo1.field, foo2.field)

    def test_member_descriptor_set_delete_slot(self):
        # [FUNCTION DOC] Checks explicit calls to set and delete on a slot member descriptor
        # trace under fullgraph and match eager.
        # member_descriptor (a __slots__ field) exposes __set__/__delete__ via
        # the shared tp_descr_set slot; explicit calls should trace.
        class Slotted:
            # [CLASS DOC] HELPER: Class with one slot named a.
            __slots__ = ("a",)

        def set_fn(x):
            # [FUNCTION DOC] HELPER: Sets the slot via the descriptor and returns x plus it.
            o = Slotted()
            type(o).a.__set__(o, 9)
            return x + o.a

        def del_fn(x):
            # [FUNCTION DOC] HELPER: Sets then deletes slot via descriptor; returns x plus 0 or
            # 100 depending on hasattr.
            o = Slotted()
            o.a = 4
            type(o).a.__delete__(o)
            return x + (0 if hasattr(o, "a") else 100)

        for fn in (set_fn, del_fn):
            opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
            self.assertEqual(opt_fn(torch.zeros(2)), fn(torch.zeros(2)))

    def test_tuplegetter_readonly_slot(self):
        # [FUNCTION DOC] Checks that setting a namedtuple field via its descriptor raises
        # AttributeError inside compiled code and results match eager.
        # namedtuple field accessors are read-only; __set__/__delete__ raise
        # AttributeError through the tp_descr_set slot.
        P = collections.namedtuple("P", ["x", "y"])

        def fn(t):
            # [FUNCTION DOC] HELPER: Tries to set a field via descriptor; returns t plus 1 or the
            # error message with t.
            p = P(1, 2)
            try:
                type(p).x.__set__(p, 5)
                return t + 1
            except AttributeError as e:
                return t, str(e)

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        self.assertEqual(opt_fn(torch.zeros(2)), fn(torch.zeros(2)))

    def test_dict_with_descriptor(self):
        # [FUNCTION DOC] Checks a descriptor-backed attribute combined with direct instance dict
        # mutation; value 42 plus 99 stored correctly.
        class MyDescriptor:
            # [CLASS DOC] HELPER: Descriptor storing values in the instance dict under an
            # underscore key.
            def __get__(self, obj, objtype=None):
                # [FUNCTION DOC] HELPER: Returns the descriptor itself on class access, else the
                # stored value defaulting to 0.
                if obj is None:
                    return self
                return obj.__dict__.get("_value", 0)

            def __set__(self, obj, value):
                # [FUNCTION DOC] HELPER: Stores value in the instance dict.
                obj.__dict__["_value"] = value

        class MyClass:
            # [CLASS DOC] HELPER: Class using the descriptor and setting prop to 42 on creation.
            prop = MyDescriptor()

            def __init__(self):
                # [FUNCTION DOC] HELPER: Assigns 42 through the descriptor.
                self.prop = 42

        def fn(obj):
            # [FUNCTION DOC] HELPER: Writes extra 99 into the dict and returns prop plus extra.
            obj.__dict__["extra"] = 99
            return obj.prop + obj.__dict__["extra"]

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = MyClass()
        res = fn(obj)

        obj2 = MyClass()
        out = opt_fn(obj2)
        self.assertEqual(res, out)
        self.assertEqual(obj2.__dict__["_value"], 42)
        self.assertEqual(obj2.__dict__["extra"], 99)

    def test_dict_with_slots(self):
        # [FUNCTION DOC] Checks reading a slot-only attribute works in compiled code.
        class SlottedClass:
            # [CLASS DOC] HELPER: Slots class with field x.
            __slots__ = ("x",)

            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns double x.
            # SlottedClass doesn't have __dict__, so this should fail or be handled
            return obj.x * 2

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = SlottedClass(5)
        res = fn(obj)
        out = opt_fn(obj)
        self.assertEqual(res, out)

    def test_dict_with_slots_and_dict(self):
        # [FUNCTION DOC] Checks a class with slots including the dict slot: instance dict
        # mutation inside compiled code persists.
        class SlottedWithDict:
            # [CLASS DOC] HELPER: Class with slots x and a dict.
            __slots__ = ("x", "__dict__")

            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

        def fn(obj):
            # [FUNCTION DOC] HELPER: Writes custom 100 in the dict and returns x plus it.
            obj.__dict__["custom"] = 100
            return obj.x + obj.__dict__["custom"]

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = SlottedWithDict(7)
        res = fn(obj)

        obj2 = SlottedWithDict(7)
        out = opt_fn(obj2)
        self.assertEqual(res, out)
        self.assertEqual(obj2.__dict__["custom"], 100)

    def test_dict_descriptor_interaction(self):
        # [FUNCTION DOC] Checks interaction of a data descriptor with direct dict writes; the
        # final read returns the directly written value.
        class DescriptorWithDict:
            # [CLASS DOC] HELPER: Descriptor storing a prefixed string under an internal key.
            def __get__(self, obj, objtype=None):
                # [FUNCTION DOC] HELPER: Returns self on class access, else the internal value
                # with default.
                if obj is None:
                    return self
                return obj.__dict__.get("_internal", "default")

            def __set__(self, obj, value):
                # [FUNCTION DOC] HELPER: Stores a desc_ prefixed value.
                obj.__dict__["_internal"] = f"desc_{value}"

        class MyClass:
            # [CLASS DOC] HELPER: Class using the descriptor.
            desc = DescriptorWithDict()

            def __init__(self):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        def fn(obj):
            # [FUNCTION DOC] HELPER: Sets desc via descriptor, overwrites the internal dict key
            # directly, returns desc.
            obj.desc = "hello"
            obj.__dict__["_internal"] = "direct"
            return obj.desc

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = MyClass()
        ref = fn(obj)

        obj2 = MyClass()
        out = opt_fn(obj2)
        self.assertEqual(ref, out)
        self.assertEqual(out, "direct")

    def test_mutable_mapping_dict_update(self):
        # [FUNCTION DOC] Checks that dict mutation on an ordered-dict subclass instance attribute
        # dict works under fullgraph.
        """Test that MutableMappingVariable handles __dict__ updates correctly."""
        from collections import OrderedDict

        class CustomMapping(OrderedDict):
            # [CLASS DOC] HELPER: Empty ordered dict subclass.
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            # [FUNCTION DOC] HELPER: Writes custom_attr 42 into the attribute dict and reads
            # it back.
            mapping.__dict__["custom_attr"] = 42
            return mapping.__dict__["custom_attr"]

        mapping = CustomMapping()
        out = fn(mapping)
        self.assertEqual(out, 42)
        self.assertIn("custom_attr", mapping.__dict__)
        self.assertEqual(mapping.__dict__["custom_attr"], 42)

    def test_mutable_mapping_dict_access_pattern(self):
        # [FUNCTION DOC] Checks lazy init pattern with presence check in attribute dict of an
        # ordered dict subclass.
        """Test accessing attributes through __dict__ on MutableMapping subclasses."""
        from collections import OrderedDict

        class TrackedDict(OrderedDict):
            # [CLASS DOC] HELPER: Empty ordered dict subclass.
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(d):
            # [FUNCTION DOC] HELPER: Creates a tracker list if absent, appends 1, returns
            # its length.
            # Pattern: check if attribute exists in __dict__
            if "tracker" not in d.__dict__:
                d.__dict__["tracker"] = []
            d.__dict__["tracker"].append(1)
            return len(d.__dict__["tracker"])

        d = TrackedDict()
        out = fn(d)
        self.assertEqual(out, 1)
        self.assertIn("tracker", d.__dict__)
        self.assertEqual(d.__dict__["tracker"], [1])

    def test_mutable_mapping_lazy_dict_initialization(self):
        # [FUNCTION DOC] Checks lazy cache creation in the attribute dict of a dict subclass;
        # nested write persists.
        """Test lazy initialization pattern with MutableMapping __dict__."""
        from collections import defaultdict

        class LazyMapping(dict):
            # [CLASS DOC] HELPER: Empty dict subclass.
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            # [FUNCTION DOC] HELPER: Creates cache dict if no attribute, stores a key and
            # returns it.
            # Lazy initialization in __dict__
            if not hasattr(mapping, "_cache"):
                mapping.__dict__["_cache"] = {}
            mapping.__dict__["_cache"]["key"] = "value"
            return mapping.__dict__["_cache"]["key"]

        m = LazyMapping()
        out = fn(m)
        self.assertEqual(out, "value")
        self.assertIn("_cache", m.__dict__)
        self.assertEqual(m.__dict__["_cache"]["key"], "value")

    def test_mutable_mapping_dict_with_property_setter(self):
        # [FUNCTION DOC] Checks property setter on an ordered dict subclass writing the attribute
        # dict; value 100 stored.
        """Test MutableMapping with property setters that access __dict__."""
        from collections import OrderedDict

        class PropertyMapping(OrderedDict):
            # [CLASS DOC] HELPER: Ordered dict subclass with a value property backed by the
            # attribute dict.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls base init and sets internal value 0.
                super().__init__()
                self.__dict__["_value"] = 0

            @property
            def value(self):
                # [FUNCTION DOC] HELPER: Returns internal value defaulting to 0.
                return self.__dict__.get("_value", 0)

            @value.setter
            def value(self, v):
                # [FUNCTION DOC] HELPER: Stores v internally.
                self.__dict__["_value"] = v

        @torch.compile(fullgraph=True, backend="eager")
        def fn(m):
            # [FUNCTION DOC] HELPER: Sets value via the property and reads it back.
            m.value = 100
            return m.value

        m = PropertyMapping()
        out = fn(m)
        self.assertEqual(out, 100)
        self.assertIn("_value", m.__dict__)
        self.assertEqual(m.__dict__["_value"], 100)

    def test_mutable_mapping_dict_multiple_accesses(self):
        # [FUNCTION DOC] Checks several chained reads and writes of the attribute dict; result 6,
        # entries 1, 2, 3.
        """Test multiple accesses and mutations to MutableMapping __dict__."""
        from collections import OrderedDict

        class MultiAccessMapping(OrderedDict):
            # [CLASS DOC] HELPER: Empty ordered dict subclass.
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            # [FUNCTION DOC] HELPER: Sets a, b from a, c from b and sums them.
            # Multiple accesses to __dict__
            mapping.__dict__["a"] = 1
            mapping.__dict__["b"] = mapping.__dict__["a"] + 1
            mapping.__dict__["c"] = mapping.__dict__["b"] + 1
            return mapping.__dict__["a"] + mapping.__dict__["b"] + mapping.__dict__["c"]

        m = MultiAccessMapping()
        out = fn(m)
        self.assertEqual(out, 6)  # 1 + 2 + 3
        self.assertIn("a", m.__dict__)
        self.assertIn("b", m.__dict__)
        self.assertIn("c", m.__dict__)
        self.assertEqual(m.__dict__["a"], 1)
        self.assertEqual(m.__dict__["b"], 2)
        self.assertEqual(m.__dict__["c"], 3)

    def test_get_attr_function(self):
        # [FUNCTION DOC] Checks calling the get method of the tensor shape descriptor passed in
        # as an argument.
        def fn(g, x):
            # [FUNCTION DOC] HELPER: Calls g with x.
            return g(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        g = torch.Tensor.shape.__get__

        res = opt_fn(g, torch.ones(2, 2))
        exp_res = fn(g, torch.ones(2, 2))
        self.assertEqual(res, exp_res)

    def test_user_getattribute(self):
        # [FUNCTION DOC] Checks an object overriding attribute lookup that serves entries from a
        # custom dict, and a method using both normal and custom attributes.
        class MyObject:
            # [CLASS DOC] HELPER: Object with a dict entry and a number; its attribute lookup
            # consults the dict first.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the custom dict and the number 42.
                self.custom_dict = {"a": torch.rand((2, 2))}
                self.my_number = 42

            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: Returns dict entry if name is present, else
                # default lookup.
                custom_dict = super().__getattribute__("custom_dict")
                if name in custom_dict:
                    return custom_dict[name]
                return super().__getattribute__(name)

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns my_number times x plus a times x.
                return self.my_number * x + self.a * x

        def fn(obj, x):
            # [FUNCTION DOC] HELPER: Calls obj.run.
            return obj.run(x)

        obj = MyObject()
        x = torch.rand((2, 2))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj, x), fn(obj, x)))

    def test_nn_module_getattr(self):
        # [FUNCTION DOC] Checks nn.Module with fallback attribute hook serving a list from a
        # custom dict.
        # KNOWN ISSUE: frame and op counts use assertTrue so the asserts never compare values.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a custom dict, an attribute tensor and a fallback
            # attribute hook.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the dict holding a list of three tensors and a
                # tensor attribute.
                super().__init__()
                self.custom_dict = {"queue": [torch.rand((2, 2)) for _ in range(3)]}
                self.other_attr = torch.rand((2, 2))

            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: Returns dict entry if present else defers to the
                # module hook.
                custom_dict = self.custom_dict
                if name in custom_dict:
                    return custom_dict[name]
                return super().__getattr__(name)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Matrix-multiplies x with other_attr and adds the last
                # queue tensor.
                return x @ self.other_attr + self.queue[-1]

        x = torch.rand((2, 2))
        mod = MyMod()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnts)
        self.assertTrue(same(opt_mod(x), mod(x)))
        self.assertTrue(cnts.frame_count, 1)
        self.assertTrue(cnts.op_count, 2)

    def test_nn_module_getattribute(self):
        # [FUNCTION DOC] Checks nn.Module overriding full attribute lookup to serve a special
        # tensor attribute.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with number 42 whose lookup returns a fixed tensor
            # for special_attr.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets my_number.
                super().__init__()
                self.my_number = 42

            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: Returns the fixed tensor for the special name, else
                # default lookup.
                if name == "special_attr":
                    return torch.tensor([[1, 2], [3, 4]])
                return super().__getattribute__(name)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes my_number times x plus special_attr times x.
                return self.my_number * x + self.special_attr * x

        def fn(mod, x):
            # [FUNCTION DOC] HELPER: Calls the module.
            return mod(x)

        mod = MyMod()
        x = torch.rand((2, 2))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(mod, x), fn(mod, x)))

    def test_nn_module_getattribute_simple_delegation(self):
        # [FUNCTION DOC] Checks nn.Module whose full lookup hook aliases one name to
        # another compiles.
        # Test that nn.Module with __getattribute__ that overrides a
        # single attribute name compiles without graph break.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with scale 3.0 and an alias name.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets scale.
                super().__init__()
                self.scale = 3.0

            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: Returns scale for the alias name, else default lookup.
                if name == "my_scale":
                    return super().__getattribute__("scale")
                return super().__getattribute__(name)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies x by the alias.
                return x * self.my_scale

        mod = MyMod()
        x = torch.randn(2, 4)
        expected = mod(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mod, backend=cnts)
        result = opt_fn(x)
        self.assertTrue(same(result, expected))

    def test_nn_module_getattribute_graph_break(self):
        # [FUNCTION DOC] Checks that an untraceable lookup hook (an eval call) falls back to
        # correct results instead of crashing.
        # __getattribute__ that Dynamo cannot trace produces correct results
        # via eager fallback instead of crashing.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer and a lookup hook returning an
            # eval result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: Returns eval result for my_attr, else default lookup.
                if name == "my_attr":
                    return eval("len([1])")
                return super().__getattribute__(name)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads my_attr and adds it to the linear output.
                a = self.my_attr
                return self.linear(x) + a

        mod = MyMod()
        x = torch.randn(2, 4)
        expected = mod(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mod, backend=cnts)
        result = opt_fn(x)
        self.assertTrue(same(result, expected))

    def test_constant_getattr(self):
        # [FUNCTION DOC] Checks that getattr with default on None returns the default constant.
        # https://github.com/pytorch/pytorch/issues/97480
        def fn():
            # [FUNCTION DOC] HELPER: Returns getattr on None with default 3.
            return getattr(None, "arg", 3)

        cnt = torch._dynamo.testing.CompileCounter()
        optimized_fn = torch.compile(fn, backend=cnt)
        res = optimized_fn()
        self.assertTrue(same(res, 3))

    def test_user_property(self):
        # [FUNCTION DOC] Checks that a property on a user config class is read in compiled code:
        # one frame, 2 ops, result is twice x plus 5.
        class MyConfig:
            # [CLASS DOC] HELPER: Config class with a property.
            @property
            def prop5(self):
                # [FUNCTION DOC] HELPER: Property returning 5.
                return 5

        def fn(cfg, x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y plus the property value.
            return x + y + cfg.prop5

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_data_access_in_inference_mode(self):
        # [FUNCTION DOC] Checks reading the data attribute of a tensor inside inference mode
        # under fullgraph returns an equal tensor.
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x.data.
            y = x.data
            return y

        with torch.inference_mode():
            x = torch.randn(3)
            y = f(x)
        self.assertEqual(y, x)

    def test_dataclass_fields(self):
        # [FUNCTION DOC] Checks dataclass utilities in compiled code (fields, defaults, getattr,
        # hasattr) for two instances with different populated fields; 2 ops then 1 op; adding an
        # extra attribute later fails the guard and gives -2.
        @dataclasses.dataclass
        class MyDataClass:
            # [CLASS DOC] HELPER: Dataclass with a tensor a and four optional tensor fields
            # defaulting to None.
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        def fn(obj):
            # [FUNCTION DOC] HELPER: Asserts the fields and defaults, returns -1 or -2 on missing
            # or extra attribute, else sums all non-None field tensors.
            class_fields = dataclasses.fields(obj)
            assert len(class_fields)  # noqa: S101
            assert all(field.default is None for field in class_fields[1:])  # noqa: S101
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none  # noqa: S101

            if not hasattr(obj, "a"):
                return -1
            if hasattr(obj, "z"):
                return -2

            total = getattr(obj, class_fields[0].name)
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            return total

        obj1 = MyDataClass(torch.randn(10), torch.randn(10), torch.randn(10))
        obj2 = MyDataClass(torch.randn(10), e=torch.randn(10))
        correct1 = fn(obj1)
        correct2 = fn(obj2)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj1), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj2), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        # guard failure
        obj2.z = True
        self.assertEqual(opt_fn(obj2), -2)

    def test_dataclass_local_hasattr(self):
        # [FUNCTION DOC] Checks hasattr on a locally defined dataclass instance created inside
        # compiled code: returns the object, one frame, 2 ops.
        cnt = CompileCounter()
        x = torch.randn(10)

        @dataclasses.dataclass
        class MyDataClass:
            # [CLASS DOC] HELPER: Dataclass with two tensor fields.
            a: torch.Tensor
            b: torch.Tensor

        @torch.compile(backend=cnt, fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Creates the dataclass from the enclosing tensor, checks
            # attributes, returns the object.
            obj = MyDataClass(x + 1, x - 1)
            if not hasattr(obj, "a"):
                return -1
            if hasattr(obj, "z"):
                return -2
            return obj

        result = fn()
        self.assertIsInstance(result, MyDataClass)
        self.assertEqual(result.a, x + 1)
        self.assertEqual(result.b, x - 1)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_catch_watchings1(self):
        # [FUNCTION DOC] Checks that a warnings catch context manager used in compiled code
        # compiles in one frame.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Enters the warnings catch context and returns the sine of x.
            with warnings.catch_warnings(record=True):
                return x.sin()

        x = torch.randn(8)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(cnt.frame_count, 1)

    def test_catch_watchings2(self):
        # [FUNCTION DOC] Checks that creating a warnings catch object in compiled code returns
        # distinct instances on each call; one frame.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sine of x and a new warnings catch object.
            return x.sin(), warnings.catch_warnings(record=True)

        x = torch.randn(8)
        _, a = fn(x)
        _, b = fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsInstance(a, warnings.catch_warnings)
        self.assertIsInstance(b, warnings.catch_warnings)
        self.assertIsNot(a, b)

    def test_tensor_build_list_unpack(self):
        # [FUNCTION DOC] Checks concatenating an unpacked tensor along the last dim; one frame,
        # 2 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Unpacks tensor x into a list and concatenates along the
            # last dim.
            # seen in fastNLP_Bert
            return torch.cat([*x], dim=-1)

        val = torch.randn([1, 1, 473, 768])
        correct = fn(val)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(val), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_numpy_int_constant(self):
        # [FUNCTION DOC] Checks modulo with a numpy int64 constant under dynamic shapes compiles
        # once with 2 ops.
        def fn(x, a, b):
            # [FUNCTION DOC] HELPER: Adds a modulo b to x.
            return x + (a % b)

        args = [torch.randn(10), 4096, np.int64(8)]
        correct = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, dynamic=True, fullgraph=True)
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_numpy_subdtype(self):
        # [FUNCTION DOC] Checks that numpy issubdtype on the type of an int is evaluated at trace
        # time; one frame.
        def fn(x, n):
            # [FUNCTION DOC] HELPER: Adds the issubdtype result to x.
            return np.issubdtype(type(n), np.integer) + x

        args = [torch.randn(10), 4096]
        correct = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn(*args), correct)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_take_along_axis(self):
        # [FUNCTION DOC] Checks numpy take_along_axis on sample inputs from the take-along-dim
        # generator; compares with eager and expects one new frame per sample.
        def fn(x, i, a):
            # [FUNCTION DOC] HELPER: Calls numpy take_along_axis.
            return np.take_along_axis(x, i, a)

        def sample_to_args(s):
            # [FUNCTION DOC] HELPER: Converts a sample into an argument tuple, turning tensors
            # into numpy arrays.
            # KNOWN ISSUE: reads args from the loop variable sample in the enclosing scope
            # instead of the parameter s; works only because they are the same object.
            args = (s.input, *sample.args)
            return tuple(a.numpy() if isinstance(a, torch.Tensor) else a for a in args)

        samples = list(
            sample_inputs_take_along_dim(
                None, "cpu", torch.float32, requires_grad=False
            )
        )
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        i = 1
        for sample in samples:
            args = sample_to_args(sample)
            if len(args) < 3:
                # if axis is None, second argument is treated as 1d array
                args = (args[0], np.ravel(args[1]), None)
            self.assertEqual(fn(*args), opt_fn(*args))
            self.assertEqual(cnts.frame_count, i)
            i += 1

    def test_numpy_torch_operators(self):
        # [FUNCTION DOC] Checks all graph-capturable operators from the builtin variable table
        # with each mix of numpy or tensor operands: results equal eager and one frame each;
        # skips eq, ne, getitem, and in-place matmul on numpy, and ops raising in eager.
        def fn(op, t1, t2):
            # [FUNCTION DOC] HELPER: Applies op to the two operands.
            return op(t1, t2)

        from torch._dynamo.variables.builtin import BuiltinVariable

        operators = BuiltinVariable._fx_graph_functions()

        for op, t1_np, t2_np in itertools.product(
            operators, (True, False), (True, False)
        ):
            if op in [operator.eq, operator.ne]:
                # returns equivalent of torch.eq/ne
                continue
            if op is operator.getitem:
                # skip
                # Did you know that tensor[ndarray_of_floats] works?
                continue
            if op is operator.imatmul and (t1_np or t2_np):
                # skip
                # in numpy, in place matmul does not work single
                # dimensional arrays
                continue
            t1 = torch.rand(5)
            if t1_np:
                t1 = t1.numpy()
            t2 = torch.rand(5)
            if t2_np:
                t2 = t2.numpy()
            try:
                # TODO try a bit harder
                result = op(t1, t2)
            except (RuntimeError, TypeError, IndexError):
                continue
            cnts = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnts)
            self.assertEqual(
                result,
                opt_fn(op, t1, t2),
                msg=lambda msg: f"{msg}\n{op=} {t1_np=} {t2_np=}",
            )
            self.assertEqual(
                cnts.frame_count, 1, msg=lambda msg: f"{msg}\n{op=} {t1_np=} {t2_np=}"
            )
            torch._dynamo.reset()

    def test_numpy_ndarray_graph_break(self):
        # [FUNCTION DOC] Checks that a graph break after converting a tensor to numpy and taking
        # real gives two frames over ten runs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Converts to numpy, takes real part, graph breaks, multiplies
            # by 2.0.
            a = x.numpy()
            b = a.real
            torch._dynamo.graph_break()
            c = np.multiply(b, 2.0)
            return c

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_ndarray_graph_break_with_multiple_outputs(self):
        # [FUNCTION DOC] Same with two arrays and a graph break; two frames.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Converts two tensors to numpy, graph breaks, adds 1 to each.
            a = x.numpy()
            b = y.numpy()
            torch._dynamo.graph_break()
            return np.add(a, 1), np.add(b, 1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn([1, 3])
            y = torch.randn([1, 3])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_force(self):
        # [FUNCTION DOC] Checks tensor-to-numpy with force off and on (the latter with a
        # gradient-requiring tensor) returns numpy arrays in one frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.numpy with force off.
            return x.numpy(force=False)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(3)
        res = opt_fn(x)
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.numpy with force on.
            return x.numpy(force=True)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(3, requires_grad=True)
        res = opt_fn(x)
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_recompilation_scalar(self):
        # [FUNCTION DOC] Checks numpy where with different scalar values under dynamic compile
        # gives a single frame.
        def fn(x, a):
            # [FUNCTION DOC] HELPER: Returns numpy where of x below 0.5 else.
            return np.where(x < 0.5, a, x)

        x = np.random.randn(8)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, dynamic=True)

        ref = fn(x, 3)
        res = opt_fn(x, 3)
        self.assertEqual(ref, res)

        ref = fn(x, 4)
        res = opt_fn(x, 4)
        self.assertEqual(ref, res)

        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_interacts_with_numpy_ndarray(self):
        # [FUNCTION DOC] Checks mixed tensor/numpy interaction around a graph break; two frames.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Converts to numpy, builds ones-like arrays, graph breaks,
            # adds them.
            a = x.numpy()
            b = y.numpy()
            c = np.ones_like(a)
            d = np.ones_like(b)
            torch._dynamo.graph_break()
            return np.add(a, c), np.add(b, d)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn([1, 3])
            y = torch.randn([1, 3])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_ndarray_works_with_builtin_function(self):
        # [FUNCTION DOC] Checks numpy sum and len on an ndarray under fullgraph; one frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sum divided by length.
            v = x.sum() / len(x)
            return v

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        for _ in range(10):
            x = np.random.randn(2, 3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_array_of_arrays(self):
        # [FUNCTION DOC] Checks building an array from two numpy scalars and from two arrays;
        # second call with new shapes recompiles to 2 frames.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns numpy array of x and y.
            return np.array([x, y])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        x, y = np.float64(1), np.float64(2)
        res = opt_fn(x, y)
        self.assertEqual(res, np.array([1, 2], dtype=float))
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

        x, y = np.arange(2), np.arange(2) + 2
        res = opt_fn(x, y)
        self.assertEqual(res, np.array([[0, 1], [2, 3]]))
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_readonly(self):
        # [FUNCTION DOC] Checks a read-only numpy input passes through compilation without
        # warnings and the output is writeable (differs from numpy).
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.
            return x

        x = np.broadcast_to(np.arange(3), (2, 3))
        self.assertFalse(x.flags.writeable)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", category=DeprecationWarning)  # from asyncio
            y = fn(x)
        self.assertTrue(y.flags.writeable)  # XXX: differs from numpy

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_numpy_tolist(self):
        # [FUNCTION DOC] With scalar capture on, checks tolist on an array returns a Python list
        # under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.tolist.
            return x.tolist()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        x = np.arange(5)
        r = opt_fn(x)

        self.assertEqual(r, [0, 1, 2, 3, 4])
        self.assertEqual(type(r), list)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_size_attr(self):
        # [FUNCTION DOC] Checks the size attribute of an ndarray inside compiled code;
        # result ndarray.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.size plus x.
            return x.size + x

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        x = np.arange(5)
        r = opt_fn(x)

        self.assertEqual(r, fn(x))
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_no_raise(self):
        # [FUNCTION DOC] Checks that numpy nan_to_num on complex arrays with print inside a loop
        # compiles without error; frame count 2 static or 1 dynamic.
        def _inf_nan_preprocess(t, t_np):
            # [FUNCTION DOC] HELPER: Applies numpy nan_to_num to the array and returns tensor
            # and array.
            t_np = np.nan_to_num(t_np)
            return t, t_np

        def fn():
            # [FUNCTION DOC] HELPER: Loops over shapes, creates complex tensor and array,
            # preprocesses, prints.
            # shape, dims format
            test_cases = (
                (3, 3),
                (4, 4),
                (5, 5),
            )

            for shape in test_cases:
                t = torch.randn(shape, dtype=torch.complex64)
                t_np = np.random.randn(*shape).astype(np.complex64)

                _, t_np = _inf_nan_preprocess(t, t_np)
                print(t, t_np)  # Just a side effect so that compilation kicks in

        cnt = CompileCounterWithBackend("eager")
        fn = torch.compile(fn, backend=cnt)
        fn()
        self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))

    def test_mandelbrot_numpy(self):
        # [FUNCTION DOC] Checks a numpy Mandelbrot computation compiles under fullgraph for many
        # iteration counts, one frame per count (specialization), and matches eager.
        def mandelbrot_numpy(max_iter):
            # [FUNCTION DOC] HELPER: Builds the complex grid, iterates max_iter times updating
            # coordinates and accumulating the inside mask; returns mask.
            # Define the boundaries of the complex plane
            xn = 450
            yn = 375
            xmin = -2.25
            xmax = 0.75
            ymin = -1.25
            ymax = 1.25

            # Create the grid of complex numbers
            x_values = np.linspace(xmin, xmax, xn, dtype=np.float64)
            y_values = np.linspace(ymin, ymax, yn, dtype=np.float64)
            rx, iy = np.meshgrid(x_values, y_values, indexing="xy")

            x = rx.copy()
            y = iy.copy()
            mask = np.zeros_like(x)
            for i in range(max_iter):
                x_prev = x
                y_prev = y
                x = x_prev**2 - y_prev**2 + rx
                y = 2 * x_prev * y_prev + iy
                inside = np.sqrt(x**2 + y**2) <= 2
                mask += inside
            return mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mandelbrot_numpy, backend=cnts, fullgraph=True)
        n_iter = torch._dynamo.config.recompile_limit - 2
        for i in range(n_iter):
            x = i + 3
            ref = mandelbrot_numpy(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
        # We need to specialise the number as it's in a forloop
        self.assertEqual(cnts.frame_count, n_iter)

    def test_numpy_as_global(self):
        # [FUNCTION DOC] Checks a module-global numpy array used twice in compiled code; result
        # is three times it. Deletes the global afterwards.
        global x
        x = np.arange(10)

        @torch.compile(fullgraph=True, backend="eager")
        def fn(y):
            # [FUNCTION DOC] HELPER: Returns y plus global x plus global x.
            return y + x + x

        r = fn(np.arange(10))
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r, x * 3)
        del x

    def test_numpy_gt(self):
        # [FUNCTION DOC] Checks numpy array greater-or-equal comparison with scalar returns
        # an ndarray.
        x = np.arange(10)

        @torch.compile(backend="eager")
        def fn(y):
            # [FUNCTION DOC] HELPER: Returns y compared greater-or-equal to 3.
            return y >= 3

        r = fn(x)
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r, x >= 3)

    def test_numpy_min(self):
        # [FUNCTION DOC] Checks the builtin min on numpy arrays returns ndarrays equal to
        # numpy minimum.
        x = np.arange(10)

        @torch.compile(backend="eager")
        def fn(y):
            # [FUNCTION DOC] HELPER: Returns min of y and 3, and min of y and y minus 1.
            return min(y, 3), min(y, y - 1)

        r1, r2 = fn(x)
        self.assertEqual(type(r1), np.ndarray)
        self.assertEqual(type(r2), np.ndarray)
        self.assertEqual(r1, np.minimum(x, 3))
        self.assertEqual(r2, np.minimum(x, x - 1))

    def test_graph_break_correctly_when_passing_numpy_ndarray_to_torch_function(self):
        # [FUNCTION DOC] Checks a case modeled on big-bird: building a tensor from numpy arrays,
        # with in-place unsqueeze; two frames across ten sizes.
        # from transformers/models/big_bird/modeling_big_bird.py
        def fn(x: int, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Builds a ones array of shape 2 by x, stacks, makes a long
            # tensor, unsqueezes in place and adds y.
            ndarray_list = [np.ones([2, x])]
            ndarray = np.stack(ndarray_list, axis=0)
            tensor = torch.tensor(ndarray, dtype=torch.long)
            tensor.unsqueeze_(0)
            return tensor + y

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for x in range(1, 10):
            y = torch.randn([1, 2, x])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        # It's all traced once with x = 1 and then x = ks0
        # For dynamic it's x=ks0
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(str(cnts.frame_count), """2""")
        else:
            self.assertExpectedInline(str(cnts.frame_count), """2""")

    @skipIfWindows(
        msg="AssertionError: Object comparison failed: dtype('int64') != <class 'int'>"
    )
    def test_numpy_with_builtin_type(self):
        # [FUNCTION DOC] Checks chains of astype to builtin types on numpy arrays end with int
        # dtype in one frame. Skipped on Windows.
        x = np.random.rand(5)

        def fn(x):
            # [FUNCTION DOC] HELPER: Casts to bool, float, int and adds 8.
            return (x * 5).astype(bool).astype(float).astype(int) + 8

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn(x)
        self.assertEqual(r.dtype, int)
        self.assertEqual(cnts.frame_count, 1)

    def test_with_builtin_type(self):
        # [FUNCTION DOC] Checks chained to calls with python builtin types on a tensor end with
        # int64 in one frame.
        x = torch.randn(5)

        def fn(x):
            # [FUNCTION DOC] HELPER: Casts to bool, float, int and adds 8.
            return (x * 5).to(bool).to(float).to(int) + 8

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn(x)
        self.assertEqual(r.dtype, torch.int64)
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_unique_consecutive(self):
        # [FUNCTION DOC] With dynamic output shape ops enabled, checks unique_consecutive under
        # fullgraph matches eager.
        x = torch.tensor([1, 1, 2, 2, 1, 3])

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns unique consecutive of x.
            return torch.unique_consecutive(x)

        expected = fn(x)
        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        result = opt_fn(x)
        self.assertEqual(result, expected)

    def test_numpy_unique_f16(self):
        # [FUNCTION DOC] Checks numpy unique on a float16 array constructed inside the function
        # keeps its dtype; one frame.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a float16 array and returns its unique values.
            x = np.asarray([1, 1, 2, 2, 3], dtype=np.float16)
            return np.unique(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn()
        self.assertEqual(r.dtype, np.float16)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_fallback_on_eager(self):
        # [FUNCTION DOC] Checks numpy arrays of strings fall back to eager (zero frames) and
        # random choice of strings is not compiled.
        def fn():
            # [FUNCTION DOC] HELPER: Returns numpy array of two strings.
            return np.asarray(["L", "U"])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn()
        self.assertEqual(cnts.frame_count, 0)  # graph break
        self.assertEqual(r, np.asarray(["L", "U"]))

        # repeat with a different function
        def fn2():
            # [FUNCTION DOC] HELPER: Returns a random choice from two strings.
            return np.random.choice(["L", "U"])

        cnts2 = torch._dynamo.testing.CompileCounter()
        opt_fn2 = torch.compile(fn2, backend=cnts2)

        r2 = fn2()
        self.assertEqual(cnts.frame_count, 0)
        if r2 not in ("L", "U"):
            raise AssertionError(f"Expected r2 to be 'L' or 'U', got {r2}")

    def test_trace_ndarray_frame(self):
        # [FUNCTION DOC] Checks a function squaring a numpy array with a print gives two frames
        # and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares, prints, doubles.
            x = x**2
            print("graph break.")
            return 2 * x

        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter)

        x = np.arange(8)
        self.assertEqual(fn(x), compiled_fn(x))
        self.assertEqual(counter.frame_count, 2)

    @skipIfWindows(
        msg="AssertionError: The values for attribute 'dtype' do not match: torch.int32 != torch.int64."
    )
    def test_trace_ndarray_frame_2(self):
        # [FUNCTION DOC] Checks a function with no tensor inputs, taking an int and creating an
        # arange with print, compiles in one frame. Skipped on Windows.
        # no tensors/ndarray as inputs in the frame
        def fn(x):
            # [FUNCTION DOC] HELPER: Prints and returns twice numpy arange of x.
            print("graph break.")
            return 2 * np.arange(x)

        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter)

        x = 8
        self.assertEqual(fn(x), compiled_fn(x))
        self.assertEqual(counter.frame_count, 1)

    def test_numpy_non_torch_dtype(self):
        # [FUNCTION DOC] Checks numpy arrays and dtypes without torch equivalents (uint16) do not
        # become tensors: isinstance returns False and zero frames.
        # test that we gracefully graph break on dtypes
        # that do not have pytorch equivalents.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns whether x is a tensor.
            return isinstance(x, torch.Tensor)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        # torch does not have the `uint16` dtype
        for x in [np.array([42], dtype=np.uint16), np.uint16(42), np.dtype("uint16")]:
            r = opt_fn(x)

            self.assertEqual(r, False)
            self.assertEqual(cnts.frame_count, 0)  # graph break

    def test_numpy_iter(self):
        # [FUNCTION DOC] Checks that iterating a numpy array yields ndarrays, not tensors;
        # one frame.
        # test that iteration over an ndarray produces ndarrays not bare tensors
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a list of rows.
            return [bm for bm in x]

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        proba_map = np.arange(3)[:, None]
        res = opt_fn(proba_map)

        self.assertEqual([type(r) for r in res], [np.ndarray, np.ndarray, np.ndarray])
        self.assertEqual(res, [np.array([0]), np.array([1]), np.array([2])])
        self.assertEqual(cnts.frame_count, 1)

    # cache size limit needs to be larger than the `dtypes` list size
    @torch._dynamo.config.patch(recompile_limit=12)
    def test_dtypes_no_graphbreaks(self):
        # [FUNCTION DOC] With a raised recompile limit, checks many numpy dtype spellings used
        # for arange compile in one frame each without graph breaks.
        dtypes = [
            # floats
            float,
            np.float64,
            "float64",
            np.float32,
            "float32",
            # np.dtype('float64')   # XXX: this is not supported, yet
            # integers
            int,
            "int",
            np.intp,
            np.int32,
            np.uint8,
            # np.dtype('int')       # XXX: as above
        ]

        def fn(dt):
            # [FUNCTION DOC] HELPER: Returns numpy arange of 5 with dtype dt.
            return np.arange(5, dtype=dt)

        for dtyp in dtypes:
            cnts = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnts)

            val = fn(dtyp)
            opt_val = opt_fn(dtyp)

            self.assertEqual(cnts.frame_count, 1)  # no graph break

    # setting the config value makes the PRNG identical to numpy's
    # NB this may involve a graph break
    @torch._dynamo.config.patch(use_numpy_random_stream=True)
    def test_numpy_random_config_to_numpy(self):
        # [FUNCTION DOC] With numpy random stream config on, checks numpy uniform inside compiled
        # code returns the right shape.
        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Returns thirteen uniform random numbers.
            return np.random.uniform(size=13)

        self.assertEqual(fn().shape, (13,))

    def test_inplace_view_on_graph_input(self):
        # [FUNCTION DOC] Checks that in-place view methods on graph inputs (resize, transpose,
        # squeeze, and others) are compiled in one frame with one op and give the same shape
        # as eager.
        # graph break when calling methods with inplace_view tag on graph input
        func_args_map = {
            lambda x: x.resize_(6).mul_(2): torch.ones(4),
            lambda x: x.t_().mul_(2): torch.rand(2, 3),
            lambda x: x.transpose_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.squeeze_().mul_(2): torch.rand(1, 2, 3),
            lambda x: x.unsqueeze_(0).mul_(2): torch.rand(2, 3),
            lambda x: x.resize_as_(torch.rand(200, 300)): torch.rand(2, 3),
            lambda x: x.swapaxes_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.swapdims_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.as_strided_((3, 2), (2, 1)).mul_(2): torch.zeros(2, 3),
            lambda x: x.detach_().mul_(2): torch.zeros(2, 3),
        }
        for func, args in func_args_map.items():
            args_clone = args.clone()
            cnts = torch._dynamo.testing.CompileCounter()
            opt_f = torch.compile(func, backend=cnts)
            self.assertTrue(same(func(args).shape, opt_f(args_clone).shape))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 1)  # mul_

    def test_out_variants_with_resizing_on_graph_inputs(self):
        # [FUNCTION DOC] Checks out-variant that resizes an input out tensor compiles in one
        # frame and matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Computes cosh into y and adds 1.
            return torch.cosh(x, out=y) + 1

        x = torch.rand(2, 3)
        y = torch.rand(4)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(fn(x, y), opt_fn(x.clone(), y.clone())))
        self.assertEqual(cnts.frame_count, 1)

    def test_out_variants_with_resizing_on_graph_inputs_with_dynamic(self):
        # [FUNCTION DOC] Checks outer product with an out tensor passed through a dictionary for
        # several shapes under fullgraph.
        # https://github.com/pytorch/pytorch/issues/120482
        class CustomModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling outer with dictionary keyword arguments.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls base init.
                super().__init__()

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns outer product from the dict.
                return torch.outer(**inputs)

        compile_fn = torch.compile(CustomModel(), backend="eager", fullgraph=True)

        shapes = [(2, 1), (6, 1), (4, 1)]
        for shape in shapes:
            vec1, vec2 = shape
            input_tensor1 = torch.randn(vec1)
            input_tensor2 = torch.randn(vec2)
            out_tensor = torch.empty(shape)
            args = {"input": input_tensor1, "vec2": input_tensor2, "out": out_tensor}
            res = compile_fn(args)
            opt_res = res.clone()  # cuz this is out and we mutate it
            res = CustomModel()(args)
            self.assertEqual(res, opt_res)

    def test_out_variants_with_resizing_on_graph_inputs_with_dynamic1(self):
        # [FUNCTION DOC] Checks matrix-vector product, plain and with an out argument, through a
        # generic caller.
        mv_op = torch.mv

        def mv_out_op(a, b, c):
            # [FUNCTION DOC] HELPER: Computes mv into a and returns a.
            torch.mv(b, c, out=a)
            return a

        def fn(op, *args):
            # [FUNCTION DOC] HELPER: Calls op with args.
            return op(*args)

        opt_fn = torch.compile(fn, backend="eager")

        ref = fn(mv_op, torch.ones(3, 3), torch.ones(3))
        res = opt_fn(mv_op, torch.ones(3, 3), torch.ones(3))
        self.assertEqual(ref, res)

        ref = fn(mv_out_op, torch.empty(0), torch.ones(3, 3), torch.ones(3))
        res = opt_fn(mv_out_op, torch.empty(0), torch.ones(3, 3), torch.ones(3))
        self.assertEqual(ref, res)

    def test_mutable_mapping_multiple_inheritance(self):
        # [FUNCTION DOC] Checks a class inheriting from both MutableMapping and nn.Module can be
        # created from a dict inside compiled code under fullgraph.
        class MyWeirdDict(collections.abc.MutableMapping, torch.nn.Module):
            # [CLASS DOC] HELPER: Mapping-and-module class storing items in a dict, with all
            # mapping methods defined and a hash based on identity.
            def __init__(self, **kwargs):
                # [FUNCTION DOC] HELPER: Calls base init and stores keyword items.
                super().__init__()
                self._items = kwargs

            def keys(self):
                # [FUNCTION DOC] HELPER: Returns keys of items.
                return self._items.keys()

            def __getitem__(self, item):
                # [FUNCTION DOC] HELPER: Looks up item.
                return self._items[item]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Stores value.
                self._items[key] = value

            def __delitem__(self, item):
                # [FUNCTION DOC] HELPER: Deletes item.
                del self._items[item]

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns number of items.
                return len(self._items)

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Yields over items keys.
                yield from self._items

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash by identity.
                return hash(id(self))

            def items(self):
                # [FUNCTION DOC] HELPER: Yields key-value pairs.
                for k, v in self._items.items():
                    yield (k, v)

        @torch.compile(fullgraph=True, backend="eager")
        def to_weird_dict(td):
            # [FUNCTION DOC] HELPER: Builds the weird mapping from keyword expansion of td.
            return MyWeirdDict(**td)

        d = MyWeirdDict(a=1, b=2, c=3)
        res = to_weird_dict(d)
        self.assertEqual(tuple(d.items()), tuple(res.items()))

    def test_dunder_new_function_inlining(self):
        # [FUNCTION DOC] Checks a class whose new method returns an instance of a different
        # module class is inlined under fullgraph (issue 107460).
        # https://github.com/pytorch/pytorch/issues/107460

        counters.clear()

        class ModelA(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning tanh of x plus 1.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls base init.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns tanh of x plus 1.
                return torch.tanh(x + 1)

        class ModelB(torch.nn.Module):
            # [CLASS DOC] HELPER: Module class whose new returns a ModelA instance.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Returns a ModelA instance.
                return ModelA()

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer using ModelB instance.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the linear output and the other module output.
                other = ModelB()
                return self.layer(x) + other(x)

        x = torch.rand(2, 2)
        m = Model()

        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        ref = m(x)
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_dunder_new_function_inlining1(self):
        # [FUNCTION DOC] Checks a class defining new and init calling the base new is inlined;
        # result x times 5.
        class Mock:
            # [CLASS DOC] HELPER: Class with c equal to 5.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Returns new instance via base.
                return super().__new__(cls)

            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets c to 5.
                self.c = 5

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns x times c.
                return x * self.c

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates mock and runs.
            mock = Mock()
            return mock.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)

        self.assertEqual(fn(x), opt_fn(x))

    def test_dunder_new_function_inlining2(self):
        # [FUNCTION DOC] Checks inheritance chain with new methods calling base new and init
        # chains is inlined under fullgraph.
        class Vehicle:
            # [CLASS DOC] HELPER: Base class storing make, model, year.
            def __new__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Creates through base new.
                return super(Vehicle, cls).__new__(cls)

            def __init__(self, make, model, year):
                # [FUNCTION DOC] HELPER: Stores the fields.
                self.make = make
                self.model = model
                self.year = year

        class Car(Vehicle):
            # [CLASS DOC] HELPER: Subclass adding door count.
            def __new__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Creates through base new.
                return super(Car, cls).__new__(cls)

            def __init__(self, make, model, year, num_doors):
                # [FUNCTION DOC] HELPER: Calls base init and stores doors.
                super(Car, self).__init__(make, model, year)
                self.num_doors = num_doors

        class ElectricCar(Car):
            # [CLASS DOC] HELPER: Subclass adding battery.
            def __new__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Creates through base new.
                return super(ElectricCar, cls).__new__(cls)

            def __init__(self, make, model, year, num_doors, battery_capacity):
                # [FUNCTION DOC] HELPER: Calls base init and stores battery.
                super(ElectricCar, self).__init__(make, model, year, num_doors)
                self.battery_capacity = battery_capacity

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns sine of x.
                return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates an ElectricCar and runs.
            ev = ElectricCar("Tesla", "Model S", 2022, 4, "100 kWh")
            return ev.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)

        self.assertEqual(fn(x), opt_fn(x))

    def test_dunder_new_function_inlining3(self):
        # [FUNCTION DOC] Checks new returning an instance of a different class: init of that
        # other class is not called, so attribute is the value set in new (7).
        class Foo:
            # [CLASS DOC] HELPER: Class whose new sets a to 3 and init sets a to 5.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Creates instance, sets a to 3.
                instance = object.__new__(cls)
                instance.a = 3
                return instance

            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets a to 5.
                self.a = 5

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns sine of x times a.
                return torch.sin(x) * self.a

        class Bar:
            # [CLASS DOC] HELPER: Class whose new returns a Foo with a set to 7; its own init
            # would not run.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Creates a Foo instance with a equal 7.
                instance = object.__new__(Foo)  # not returning a new instance of Bar
                instance.a = 7
                return instance

            def __init__(self):
                # [FUNCTION DOC] HELPER: Would set a to 11; not called.
                self.a = 11  # not called in Bar()

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns sine of x times a.
                return torch.sin(x) * self.a

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a Bar and runs.
            bar = Bar()
            return bar.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_dunder_new_function_inlining4(self):
        # [FUNCTION DOC] Checks new taking extra args and calling object new is inlined.
        class Mock(object):
            # [CLASS DOC] HELPER: Class with new using object new and init setting a to 5.
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Creates via object new.
                return object.__new__(cls)

            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets a to 5.
                self.a = 5

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns sine of x times a.
                return torch.sin(x) * self.a

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates mock and runs.
            mock = Mock()
            return mock.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_user_defined_object_class_interaction(self):
        # [FUNCTION DOC] Checks class variable holding an object and a class method reading it
        # from compiled code.
        class Foo:
            # [CLASS DOC] HELPER: Class with x equal to 5.
            x = 5

        class Mock:
            # [CLASS DOC] HELPER: Class holding a Foo as class variable.
            # This is a class variable
            class_variable = Foo()

            @classmethod
            def get_class_variable(cls):
                # [FUNCTION DOC] HELPER: Class method returning the x of the class variable.
                # Accessing the class variable using the cls parameter
                return cls.class_variable.x

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns class variable value times x.
                return self.get_class_variable() * x

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates mock and runs.
            mock = Mock()
            return mock.run(x)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_multiple_inheritance(self):
        # [FUNCTION DOC] Checks a diamond-like multiple inheritance where init and new call super
        # chains; Base1 init requires Base2 to have set base2 first (raising ValueError on wrong
        # MRO tracing).
        class Base1:
            # [CLASS DOC] HELPER: First base; init checks base2 exists then sets base1 to 3.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Creates through super new.
                return super().__new__(cls)

            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls super init, raises ValueError if base2 missing,
                # sets base1.
                super().__init__()
                if not hasattr(self, "base2"):
                    raise ValueError("Wrong MRO tracing")
                self.base1 = 3

        class Base2:
            # [CLASS DOC] HELPER: Second base; init sets base2 to 5.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Creates through super new.
                return super().__new__(cls)

            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls super init and sets base2.
                super().__init__()
                self.base2 = 5

        class Derived(Base1, Base2):
            # [CLASS DOC] HELPER: Derived from both; derived equals 7.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Creates through super new.
                return super().__new__(cls)

            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls super init and sets derived.
                super().__init__()
                self.derived = 7

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns product of all three numbers and x.
                return self.base1 * self.base2 * self.derived * x

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates Derived and runs.
            o = Derived()
            return o.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_class_duner_mro(self):
        # [FUNCTION DOC] Checks membership test on the MRO of the class of a module instance
        # under fullgraph.
        class ModuleA(torch.nn.Module):
            # [CLASS DOC] HELPER: Empty module subclass.
            pass

        class ModuleB(ModuleA):
            # [CLASS DOC] HELPER: Subclass of ModuleA.
            pass

        def fn(x, mod):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if ModuleA is in the MRO of mod type, else
            # x minus 1.
            if ModuleA in type(mod).__mro__:
                return x + 1
            else:
                return x - 1

        x = torch.rand(2, 3)
        mod = ModuleB()
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        ref = fn(x, mod)
        res = opt_fn(x, mod)
        self.assertTrue(same(ref, res))

    def test_class_duner_flags(self):
        # [FUNCTION DOC] Checks the mapping type flag on a class and on a type of an instance,
        # for a module dict and mutable mapping subclass.
        class ModuleA(torch.nn.ModuleDict, collections.abc.MutableMapping):
            # [CLASS DOC] HELPER: Module dict subclass registered as a mutable mapping with
            # identity hash.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns identity based hash.
                return id(self)

        def fn(x, mod_class):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if the class has the mapping flag else x
            # minus 1.
            if mod_class.__flags__ & TPFLAGS_MAPPING:
                return x + 1
            else:
                return x - 1

        x = torch.rand(2, 3)
        mod_class = ModuleA
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        ref = fn(x, mod_class)
        res = opt_fn(x, mod_class)
        self.assertTrue(same(ref, res))

        def fn(x, mod):
            # [FUNCTION DOC] HELPER: Same flag test via the type of an instance.
            if type(mod).__flags__ & TPFLAGS_MAPPING:
                return x + 1
            else:
                return x - 1

        x = torch.rand(2, 3)
        mod = ModuleA()
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        ref = fn(x, mod)
        res = opt_fn(x, mod)
        self.assertTrue(same(ref, res))

    def test_nested_wraps(self):
        # [FUNCTION DOC] Checks functools wraps applied to nested functions (one and two levels
        # deep) are traced under fullgraph, giving sum and product results.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Defines an add function and a wrapped caller, returns the
            # wrapped call result.
            def add(x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

            @functools.wraps(add)
            def wrapped_call(x, y):
                # [FUNCTION DOC] HELPER: Wrapper that copies add metadata and calls add.
                return add(x, y)

            return wrapped_call(x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        self.assertEqual(o, x + y)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Defines a nested caller that returns the doubly nested
            # wrapped multiplication result.
            def nested_call(x, y):
                # [FUNCTION DOC] HELPER: Defines mul and a wrapped caller, returns the wrapped
                # call result.
                def mul(x, y):
                    # [FUNCTION DOC] HELPER: Returns x times y.
                    return x * y

                @functools.wraps(mul)
                def double_nested_call(x, y):
                    # [FUNCTION DOC] HELPER: Wrapper copying mul metadata and calling mul.
                    return mul(x, y)

                return double_nested_call(x, y)

            return nested_call(x, y)

        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        self.assertEqual(o, x * y)

    def test_module_deepcopy(self):
        # [FUNCTION DOC] Checks a function that deep-copies an nn.Sequential and applies it, for
        # two different modules, compiles once with 4 ops.
        m1 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        m2 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

        def fn(m, x):
            # [FUNCTION DOC] HELPER: Deep-copies m and applies it to x.
            m_copy = copy.deepcopy(m)
            return m_copy(x)

        v = torch.randn(10)
        correct1 = fn(m1, v)
        correct2 = fn(m2, v)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            self.assertTrue(same(opt_fn(m1, v), correct1))
        for _ in range(10):
            self.assertTrue(same(opt_fn(m2, v), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_deepcopy_dict(self):
        # [FUNCTION DOC] Checks deepcopy of a global dict then mutating the copy under fullgraph.
        MY_DICT = {"a": 1, "b": 2.0, "c": None}

        def fn(x):
            # [FUNCTION DOC] HELPER: Deep-copies the dict, sets b to 3.0, adds it to x.
            d = copy.deepcopy(MY_DICT)
            d["b"] = 3.0
            return x + d["b"]

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_nested_dict(self):
        # [FUNCTION DOC] Checks deepcopy of a nested dict containing a list.
        NESTED = {"a": {"b": 1.0}, "c": [2.0, 3.0]}

        def fn(x):
            # [FUNCTION DOC] HELPER: Deep-copies the nested dict and adds two nested values to x.
            d = copy.deepcopy(NESTED)
            return x + d["a"]["b"] + d["c"][0]

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_list(self):
        # [FUNCTION DOC] Checks deepcopy of a list then mutation of the copy.
        MY_LIST = [1.0, 2.0, 3.0]

        def fn(x):
            # [FUNCTION DOC] HELPER: Deep-copies the list, overwrites the first element, adds it
            # to x.
            lst = copy.deepcopy(MY_LIST)
            lst[0] = 5.0
            return x + lst[0]

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_user_defined_object(self):
        # [FUNCTION DOC] Checks deepcopy of a plain user config object attribute inside a module
        # forward under fullgraph.
        class MyConfig:
            # [CLASS DOC] HELPER: Config storing a hidden size.
            def __init__(self, hidden_size=64):
                # [FUNCTION DOC] HELPER: Stores hidden size.
                self.hidden_size = hidden_size

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a config and linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the config and linear layer.
                super().__init__()
                self.config = MyConfig()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Deep-copies config and scales the linear output by
                # hidden size.
                cfg = copy.deepcopy(self.config)
                return self.linear(x) * cfg.hidden_size

        m = MyModule()
        x = torch.randn(2, 64)
        result = torch.compile(m, fullgraph=True, backend="eager")(x)
        self.assertEqual(m(x), result)

    def test_deepcopy_user_defined_object_with_containers(self):
        # [FUNCTION DOC] Checks deepcopy of a user object with list, dict and tuple attributes:
        # mutations on the copy do not affect the original.
        class Config:
            # [CLASS DOC] HELPER: Config with list, dict and tuple attributes.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets sizes, mapping and flags.
                self.sizes = [1, 2, 3]
                self.mapping = {"a": 10, "b": 20}
                self.flags = (True, False)

        def fn(x, cfg):
            # [FUNCTION DOC] HELPER: Deep-copies cfg, mutates the copy, adds values to x.
            c = copy.deepcopy(cfg)
            c.sizes[0] = 99
            c.mapping["a"] = 77
            return x + c.sizes[0] + c.mapping["a"]

        cfg = Config()
        x = torch.randn(4)
        correct = fn(x, cfg)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x, cfg)
        self.assertEqual(result, correct)
        # Verify deepcopy didn't mutate original
        self.assertEqual(cfg.sizes[0], 1)
        self.assertEqual(cfg.mapping["a"], 10)

    def test_copy_namedtuple_and_user_object(self):
        # [FUNCTION DOC] Checks shallow and deep copy of a namedtuple and a plain object: types
        # and fields preserved, shallow copy shares the inner list, deep copy clones it.
        # namedtuple is a tuple subclass with __slots__ = () and no __dict__; it
        # must reduce via __getnewargs__ rather than obj.__dict__. Also exercise
        # a plain user object (has __dict__) to guard the common path.
        from collections import namedtuple

        NT = namedtuple("NT", "x y z")

        class Plain:
            # [CLASS DOC] HELPER: Plain object with an int and a list.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets a and b.
                self.a = 1
                self.b = [10, 20]

        def fn(o):
            # [FUNCTION DOC] HELPER: Returns shallow and deep copies of o.
            return copy.copy(o), copy.deepcopy(o)

        cfn = torch.compile(fn, fullgraph=True, backend="eager")

        nt = NT(10, 20, 30)
        c, d = cfn(nt)
        self.assertEqual(c, nt)
        self.assertEqual(d, nt)
        self.assertIs(type(c), NT)
        self.assertIs(type(d), NT)
        self.assertEqual(c._fields, nt._fields)

        p = Plain()
        c, d = cfn(p)
        self.assertEqual(c.a, 1)
        self.assertEqual(c.b, [10, 20])
        self.assertIs(c.b, p.b)  # shallow copy shares the list
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, [10, 20])
        self.assertIsNot(d.b, p.b)  # deep copy clones the list
        self.assertIs(type(d), Plain)

    def test_deepcopy_set(self):
        # [FUNCTION DOC] Checks deepcopy of a global set.
        MY_SET = {1, 2, 3}

        def fn(x):
            # [FUNCTION DOC] HELPER: Deep-copies the set and adds its length to x.
            s = copy.deepcopy(MY_SET)
            return x + len(s)

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_frozenset(self):
        # [FUNCTION DOC] Checks deepcopy of a global frozenset.
        MY_FROZENSET = frozenset([1, 2, 3])

        def fn(x):
            # [FUNCTION DOC] HELPER: Deep-copies the frozenset and adds its length to x.
            s = copy.deepcopy(MY_FROZENSET)
            return x + len(s)

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_user_defined_object_with_method(self):
        # [FUNCTION DOC] Checks deepcopy of a user object followed by a method call on the copy
        # inside a module forward.
        class MyConfig:
            # [CLASS DOC] HELPER: Config with size getter.
            def __init__(self, hidden_size=64):
                # [FUNCTION DOC] HELPER: Stores hidden size.
                self.hidden_size = hidden_size

            def get_size(self):
                # [FUNCTION DOC] HELPER: Returns hidden size.
                return self.hidden_size

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with config and linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds config and linear layer.
                super().__init__()
                self.config = MyConfig()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Deep-copies config and scales linear output by its size.
                cfg = copy.deepcopy(self.config)
                return self.linear(x) * cfg.get_size()

        m = MyModule()
        x = torch.randn(2, 64)
        correct = m(x)
        result = torch.compile(m, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_nested_user_defined_object(self):
        # [FUNCTION DOC] Checks deepcopy of nested user objects: mutating the copy inner leaves
        # original unchanged.
        class Inner:
            # [CLASS DOC] HELPER: Holder for a scale.
            def __init__(self, scale):
                # [FUNCTION DOC] HELPER: Stores scale.
                self.scale = scale

        class Outer:
            # [CLASS DOC] HELPER: Holder for an inner object and a bias.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates an Inner with scale 2.0 and bias 1.0.
                self.inner = Inner(2.0)
                self.bias = 1.0

        def fn(x, cfg):
            # [FUNCTION DOC] HELPER: Deep-copies cfg, sets inner scale to 3.0, returns x times
            # scale plus bias.
            c = copy.deepcopy(cfg)
            c.inner.scale = 3.0
            return x * c.inner.scale + c.bias

        cfg = Outer()
        x = torch.randn(4)
        correct = fn(x, cfg)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x, cfg)
        self.assertEqual(result, correct)
        # Verify deepcopy didn't mutate original
        self.assertEqual(cfg.inner.scale, 2.0)

    def test_deepcopy_with_getattribute_override(self):
        # [FUNCTION DOC] Regression test: deepcopy of a class that overrides attribute lookup
        # with an alias table must not graph break on the reduction method.
        # Regression test: classes that override __getattribute__ (like
        # HuggingFace PretrainedConfig) caused a graph break on
        # __reduce_ex__ because SuperVariable.call_method for
        # object.__getattribute__ bypassed the polyfill detection in
        # resolve_type_attr.
        class Config:
            # [CLASS DOC] HELPER: Config class with an empty alias map and custom
            # attribute lookup.
            attribute_map = {}

            def __init__(self, hidden_size=768, num_layers=6):
                # [FUNCTION DOC] HELPER: Stores hidden size and layer count.
                self.hidden_size = hidden_size
                self.num_layers = num_layers

            def __getattribute__(self, key):
                # [FUNCTION DOC] HELPER: Resolves key via the alias map if present, then
                # default lookup.
                if key != "attribute_map" and key in super().__getattribute__(
                    "attribute_map"
                ):
                    key = super().__getattribute__("attribute_map")[key]
                return super().__getattribute__(key)

        def fn(x, config):
            # [FUNCTION DOC] HELPER: Deep-copies config and returns x times hidden size
            # plus layers.
            c = copy.deepcopy(config)
            return x * c.hidden_size + c.num_layers

        x = torch.randn(3)
        config = Config()
        correct = fn(x, config)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x, config)
        self.assertEqual(result, correct)

    def test_global_state_guard_serialization(self):
        # [FUNCTION DOC] Checks that the global state guard round-trips through JSON: changing
        # any int or bool field makes the check fail, the unchanged state passes, and autocast
        # dtype changes make it fail.
        # KNOWN ISSUE: the autocast branch builds a variant but hits continue so it never adds
        # a sample.
        GlobalStateGuard = torch._C._dynamo.guards.GlobalStateGuard
        guards = GlobalStateGuard()
        serialized_guards = guards.__getstate__()
        json_guards = json.loads(serialized_guards)

        samples = []
        # Test on non autocast state and autocast cache states.
        self.assertIn("autocast_state", json_guards)
        for key, value in json_guards.items():
            if type(value) is int:
                variant = value + 1
            elif type(value) is bool:
                variant = not value
            elif isinstance(value, dict) and key == "autocast_state":
                variant = value.copy()
                variant["cached_enabled"] = not variant["cached_enabled"]
                continue
            else:
                self.fail(f"Unknown global state type {key}: {value}")
            new_dict = json_guards.copy()
            new_dict[key] = variant
            samples.append(new_dict)

        for sample in samples:
            guards.__setstate__(json.dumps(sample))
            self.assertFalse(guards.check())

        guards.__setstate__(json.dumps(json_guards))
        self.assertTrue(guards.check())

        # Test on autocast states.
        def _test_autocast(dtype):
            # [FUNCTION DOC] HELPER: Within a CPU autocast of the given dtype, rebuilds the guard
            # state, increments every enabled dtype entry in the JSON, loads it and asserts the
            # check fails.
            with torch.autocast("cpu", dtype):
                guards = GlobalStateGuard()
                serialized_guards = guards.__getstate__()
                json_guards = json.loads(serialized_guards)

                for i, enabled in enumerate(json_guards["autocast_state"]["enabled"]):
                    if enabled:
                        self.assertEqual(
                            type(json_guards["autocast_state"]["dtype"][i]), int
                        )
                        json_guards["autocast_state"]["dtype"][i] += 1
                        guards.__setstate__(json.dumps(json_guards))
                        self.assertFalse(guards.check())

        _test_autocast(torch.float16)
        _test_autocast(torch.float32)
        _test_autocast(torch.float64)
        _test_autocast(torch.bfloat16)

    def test_type_copy(self):
        # [FUNCTION DOC] Checks building the same sequence type as the input (list or tuple) with
        # computed elements; two frames, 6 ops, correct types.
        def fn(seq):
            # [FUNCTION DOC] HELPER: Unpacks a two-element sequence and returns a new sequence of
            # the same type with three tensors.
            a, b = seq
            return type(seq)([a + 1, b + 2, a + b])

        args1 = [torch.randn(10), torch.randn(10)]
        args2 = (torch.randn(10), torch.randn(10))
        correct1 = fn(args1)
        correct2 = fn(args2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(args1), correct1))
        self.assertTrue(same(opt_fn(args2), correct2))
        self.assertIsInstance(opt_fn(args1), list)
        self.assertIsInstance(opt_fn(args2), tuple)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_setattr_mutation1(self):
        # [FUNCTION DOC] Checks repeated attribute assignment on a user object input; same object
        # returned, values match eager, one frame, 12 ops.
        class MyObj:  # noqa: B903
            # [CLASS DOC] HELPER: Holder for two tensors.
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: Stores a and b.
                self.a = a
                self.b = b

        def fn(obj):
            # [FUNCTION DOC] HELPER: Reassigns c, b, a, c, b, a using tensor arithmetic and
            # returns the object.
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            obj.c = obj.a * obj.b + 4
            obj.b = obj.a * obj.c + 5
            obj.a = obj.b * obj.c + 6
            return obj

        x1 = torch.randn(10)
        x2 = torch.randn(10)
        obj1 = MyObj(x1, x2)
        obj2 = MyObj(x1, x2)
        fn(obj2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIs(opt_fn(obj1), obj1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 12)

    def test_setattr_mutation2(self):
        # [FUNCTION DOC] Checks a user object created in the function and mutated; one frame,
        # 9 ops.
        class MyObj:
            # [CLASS DOC] HELPER: Stores x plus 1 and x plus 2.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores a and b.
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Divides x by 3, builds the object, mutates attributes,
            # returns it.
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        obj1 = opt_fn(x1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_setattr_mutation3(self):
        # [FUNCTION DOC] Same with returning attribute tuple instead of the object; 9 ops.
        # TODO(jansel): dead code eliminate the object creation
        class MyObj:
            # [CLASS DOC] HELPER: Object storing two derived tensors.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Calls base init and stores a and b.
                super().__init__()
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the object, mutates fields and returns a, b, c.
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj.a, obj.b, obj.c

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        obj1 = opt_fn(x1)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_nesteduserfunction_setattr(self):
        # [FUNCTION DOC] Checks setting an attribute on a nested function created inside the
        # compiled region and returning it.
        x = 0

        def update(y):
            # [FUNCTION DOC] HELPER: Returns a wrapper closure.
            def wrapper():
                # [FUNCTION DOC] HELPER: Inner closure body that updates an enclosing variable
                # (never called).
                x += y

            return wrapper

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Creates a wrapper via update, sets its wrapped attribute,
            # returns sine and the wrapper.
            w = update(123)
            w.__wrapped__ = x
            return t.sin(), w

        t = torch.randn(2)
        y, w = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(w.__wrapped__, x)

    def test_object_setattr(self):
        # [FUNCTION DOC] Checks direct object-level attribute assignment on dataclasses: normal,
        # frozen without assignment (0 frames), frozen with assignment, and frozen with post-
        # init assignment.
        @dataclasses.dataclass
        class A:
            # [CLASS DOC] HELPER: Dataclass holding a tensor.
            x: torch.Tensor

        def fn1(x) -> None:
            # [FUNCTION DOC] HELPER: Creates the dataclass and sets the field to x plus 2 via
            # object-level assignment.
            a = A(x)
            object.__setattr__(a, "x", x + 2)
            return a

        x1 = torch.randn(10)
        obj11 = fn1(x1.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch.compile(fn1, backend=cnts, fullgraph=True)
        obj12 = opt_fn1(x1.clone())
        self.assertTrue(same(obj11.x, x1 + 2))
        self.assertTrue(same(obj12.x, x1 + 2))
        self.assertTrue(same(obj11.x, obj12.x))
        self.assertEqual(cnts.frame_count, 1)

        @dataclasses.dataclass(frozen=True)
        class B:
            # [CLASS DOC] HELPER: Frozen dataclass holding a tensor.
            x: torch.Tensor

        def fn2(x) -> None:
            # [FUNCTION DOC] HELPER: Creates and returns the frozen dataclass.
            b = B(x)
            return b

        x2 = torch.randn(10)
        obj21 = fn2(x2.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        obj22 = opt_fn2(x2.clone())
        self.assertTrue(same(obj21.x, x2))
        self.assertTrue(same(obj22.x, x2))
        self.assertTrue(same(obj21.x, obj22.x))
        self.assertEqual(cnts.frame_count, 0)

        @dataclasses.dataclass(frozen=True)
        class C:
            # [CLASS DOC] HELPER: Frozen dataclass holding a tensor.
            x: torch.Tensor

        def fn3(x) -> None:
            # [FUNCTION DOC] HELPER: Creates the frozen dataclass and overwrites the field via
            # object-level assignment.
            c = C(x)
            object.__setattr__(c, "x", x + 2)
            return c

        x3 = torch.randn(10)
        obj31 = fn3(x3.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn3 = torch.compile(fn3, backend=cnts, fullgraph=True)
        obj32 = opt_fn3(x3.clone())
        self.assertTrue(same(obj31.x, x3 + 2))
        self.assertTrue(same(obj32.x, x3 + 2))
        self.assertTrue(same(obj31.x, obj32.x))
        self.assertEqual(cnts.frame_count, 1)

        @dataclasses.dataclass(frozen=True)
        class D:
            # [CLASS DOC] HELPER: Frozen dataclass whose post-init stores x plus 2 as y.
            x: torch.Tensor

            def __post_init__(self):
                # [FUNCTION DOC] HELPER: Stores y as x plus 2 through object-level assignment.
                object.__setattr__(self, "y", self.x + 2)

        def fn4(x) -> None:
            # [FUNCTION DOC] HELPER: Creates and returns the frozen dataclass.
            d = D(x)
            return d

        x4 = torch.randn(10)
        obj41 = fn4(x4.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn4 = torch.compile(fn4, backend=cnts, fullgraph=True)
        obj42 = opt_fn4(x4.clone())
        self.assertTrue(same(obj41.x, x4))
        self.assertTrue(same(obj42.x, x4))
        self.assertTrue(same(obj41.x, obj42.x))
        self.assertTrue(same(obj41.y, x4 + 2))
        self.assertTrue(same(obj42.y, x4 + 2))
        self.assertTrue(same(obj41.y, obj42.y))
        self.assertEqual(cnts.frame_count, 1)

    def test_thread_local_setattr(self):
        # [FUNCTION DOC] Checks assigning an attribute on a thread-local object inside compiled
        # code is visible afterwards.
        from threading import local

        loc = local()

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, l):
            # [FUNCTION DOC] HELPER: Stores x on l and returns x plus 1.
            l.x = x
            return x + 1

        x = torch.ones(2, 2)
        fn(x, loc)

        self.assertTrue(loc.x is x)

    def test_setattr_wrong_args_raises(self):
        # [FUNCTION DOC] Checks that a setattr call with a wrong argument count raises TypeError
        # during tracing.
        # setattr() with wrong arg count should raise TypeError during tracing,
        # not silently graph-break (a graph break would just defer the same
        # failure to eager execution).
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls setattr with only two args.
            setattr(x, "foo")
            return x

        with self.assertRaises(TypeError):
            fn(torch.randn(4))

    def test_user_defined_class_name(self):
        # [FUNCTION DOC] Standard test helper for reading the class name of a locally defined
        # class instance.
        class MyClassFoo:
            # [CLASS DOC] HELPER: Empty class.
            pass

        def fn1(a, b, c):
            # [FUNCTION DOC] HELPER: Returns a minus b divided by c if the class name matches.
            tmp = MyClassFoo()
            if tmp.__class__.__name__ == "MyClassFoo":
                return a - b / c

        torch._dynamo.testing.standard_test(self, fn=fn1, nargs=3)

    def test_class_reassignment_graph_break(self):
        # [FUNCTION DOC] Checks reassigning the class attribute of an object inside compiled code
        # matches eager.
        class BaseClass:
            # [CLASS DOC] HELPER: Base with x.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

        class DerivedClass(BaseClass):
            # [CLASS DOC] HELPER: Subclass setting y.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Calls base init and sets y.
                super().__init__(x)
                self.y = x * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a base object, reassigns its class, returns x plus 1
            # and isinstance and hasattr flags.
            obj = BaseClass(5)
            obj.__class__ = DerivedClass
            is_derived = isinstance(obj, DerivedClass)
            has_y = hasattr(obj, "y")
            return x + 1, is_derived, has_y

        x = torch.ones(1)
        eager_result = fn(x)
        compiled_result = torch.compile(fn, backend="eager")(x)
        self.assertEqual(eager_result, compiled_result)

    def test_user_defined_class_python_type(self):
        # [FUNCTION DOC] Checks isinstance with a plain class and a class with a metaclass as the
        # instance argument.
        class MyClass1:
            # [CLASS DOC] HELPER: Empty class.
            pass

        class ExampleMeta(type):
            # [CLASS DOC] HELPER: Metaclass.
            pass

        class MyClass2(metaclass=ExampleMeta):
            # [CLASS DOC] HELPER: Class using the metaclass.
            pass

        def fn(x, c):
            # [FUNCTION DOC] HELPER: Returns x plus 1, 2 or 3 depending on isinstance results.
            if isinstance(c, MyClass1):
                return x + 1
            elif isinstance(c, MyClass2):
                return x + 2
            else:
                return x + 3

        x = torch.rand(3)
        opt_fn = torch.compile(fn, backend="eager")
        for c in [MyClass1, MyClass2]:
            ref = fn(x, c)
            res = opt_fn(x, c)
            self.assertTrue(same(ref, res))

    def test_super_calling_with_metaclass(self):
        # [FUNCTION DOC] Checks super calls in class methods of a class with a metaclass with a
        # graph break inside.
        class ExampleMeta(type):
            # [CLASS DOC] HELPER: Empty metaclass.
            pass

        class MyClass1(metaclass=ExampleMeta):
            # [CLASS DOC] HELPER: Class with a coefficient and a class method add.
            coeff = 4  # Force the constant guard to test source in guards

            @classmethod
            def add(cls, x):
                # [FUNCTION DOC] HELPER: Returns x plus 1.
                return x + 1

        class MyClass2(MyClass1):
            # [CLASS DOC] HELPER: Subclass overriding add.
            @classmethod
            def add(cls, x):
                # [FUNCTION DOC] HELPER: Graph breaks, returns x plus super add plus super coeff.
                torch._dynamo.graph_break()
                return x + super().add(x) + super().coeff

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns x plus the add method of obj applied to x.
            return x + obj.add(x)

        x = torch.rand(3)
        obj = MyClass2()
        opt_fn = torch.compile(fn, backend="eager")
        ref = fn(x, obj)
        res = opt_fn(x, obj)
        self.assertTrue(same(ref, res))

    def test_usr_cls_staticmethod(self):
        # [FUNCTION DOC] Standard test helper for a static method on a user class.
        class Foo:
            # [CLASS DOC] HELPER: Class with static method.
            @staticmethod
            def bar(a, b):
                # [FUNCTION DOC] HELPER: Returns a plus b.
                return a + b

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls static method and subtracts 1.
            return Foo.bar(a, b) - 1

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_usr_cls_classmethod(self):
        # [FUNCTION DOC] Standard test helper for a class method on a user class.
        class Foo:
            # [CLASS DOC] HELPER: Class with class method.
            @classmethod
            def bar(cls, a, b):
                # [FUNCTION DOC] HELPER: Returns a plus b.
                return a + b

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls class method and subtracts 1.
            return Foo.bar(a, b) - 1

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_dunder_methods(self):
        # [FUNCTION DOC] Standard test helper for user-defined arithmetic operators on wrapper
        # objects; 4 ops.
        class Foo:
            # [CLASS DOC] HELPER: Wrapper with four arithmetic operators returning wrappers.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Calls base init and stores val.
                super().__init__()
                self.val = val

            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Wrapper of summed values.
                return Foo(self.val + other.val)

            def __mul__(self, other):
                # [FUNCTION DOC] HELPER: Wrapper of product.
                return Foo(self.val * other.val)

            def __truediv__(self, other):
                # [FUNCTION DOC] HELPER: Wrapper of quotient.
                return Foo(self.val / other.val)

            def __sub__(self, other):
                # [FUNCTION DOC] HELPER: Wrapper of difference.
                return Foo(self.val - other.val)

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Combines wrappers in an expression and returns the wrapper.
            return Foo(a) + Foo(b) * Foo(c) / Foo(a) - Foo(b)

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=3, expected_ops=4)

    def test_function_annotation(self):
        # [FUNCTION DOC] Checks a nested function with a typing List annotation; two frames,
        # 2 ops.
        class Variable:
            # [CLASS DOC] HELPER: Empty class used in annotation.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Divides x by 3 and returns an inner closure.
            x = x / 3.0

            def inner(y: typing.List[Variable]):
                # [FUNCTION DOC] HELPER: Returns x plus 1.
                return x + 1

            return inner

        x1 = torch.randn(10)
        obj2 = fn(x1)([])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        opt_fn_inner = torch._dynamo.optimize_assert(cnts)(opt_fn(x1))
        obj1 = opt_fn_inner([])
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_function_generic_alias_annotation(self):
        # [FUNCTION DOC] Same but with a builtin generic list annotation; two frames, 2 ops.
        class Variable:
            # [CLASS DOC] HELPER: Empty class used in annotation.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Divides x by 3 and returns an inner closure.
            x = x / 3.0

            def inner(y: list[Variable]):
                # [FUNCTION DOC] HELPER: Returns x plus 1.
                return x + 1

            return inner

        x1 = torch.randn(10)
        obj2 = fn(x1)([])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        opt_fn_inner = torch._dynamo.optimize_assert(cnts)(opt_fn(x1))
        obj1 = opt_fn_inner([])
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_nested_closure(self):
        # [FUNCTION DOC] Checks four-level nested closures with defaults and varargs; two frames,
        # 9 ops.
        # KNOWN ISSUE: shape check uses assertTrue so the expected shape is only a message.
        v0 = torch.randn(10)

        def fn1():
            # [FUNCTION DOC] HELPER: Creates v1 and calls the next level with args.
            v1 = torch.randn(10)

            def fn2(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Asserts one positional and one keyword arg, creates v2,
                # returns fn3.
                assert len(args) == 1  # noqa: S101
                assert len(kwargs) == 1  # noqa: S101
                v2 = torch.randn(10) + args[0] + kwargs["b"]

                def fn3(v3=torch.randn(10)):
                    # [FUNCTION DOC] HELPER: Takes default v3 and returns fn4.
                    def fn4():
                        # [FUNCTION DOC] HELPER: Sums captured v0, v1, v2, v3 and 1.
                        return v0 + v1 + v2 + v3 + 1

                    return fn4

                return fn3

            return fn2(1, b=2)()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        tmp1 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        tmp2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        self.assertTrue(tmp1().shape, (10,))
        self.assertTrue(same(tmp1(), tmp1()))
        self.assertFalse(same(tmp1(), tmp2()))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    def test_nested_closure_mutation(self):
        # [FUNCTION DOC] Checks mutation of nonlocal variables through nested closures matches
        # eager across repeated calls; two frames, 11 ops.
        def fn1():
            # [FUNCTION DOC] HELPER: Creates v1, defines fn2, calls its result twice, returns it.
            v1 = torch.randn(10)

            def fn2():
                # [FUNCTION DOC] HELPER: Creates v2 and returns fn3.
                v2 = torch.randn(10)

                def fn3():
                    # [FUNCTION DOC] HELPER: Increments v1 and v2 in place and returns their sum.
                    nonlocal v1, v2
                    v1 += 1
                    v2 += 2
                    return v1 + v2

                return fn3

            rv = fn2()
            rv()
            rv()
            return rv

        torch.manual_seed(9000)
        counter1 = fn1()
        result1 = [counter1(), counter1(), counter1()]

        torch.manual_seed(9000)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        counter2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        result2 = [counter2(), counter2(), counter2()]
        result1.append(counter1())
        result2.append(counter2())

        self.assertTrue(same(result1, result2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 11)

    def test_write_to_closures_in_inlining(self):
        # [FUNCTION DOC] Checks closure writes during inlining: counter results with and without
        # compile are equal; compiled path yields one frame and 3 ops.
        out = []
        for use_dynamo in [False, True]:

            def make_counter():
                # [FUNCTION DOC] HELPER: Creates tensor x and returns a counter closure.
                x = torch.randn(10)

                def counter():
                    # [FUNCTION DOC] HELPER: Increments x by 1 and returns it.
                    nonlocal x
                    x = x + 1
                    return x

                return counter

            torch.manual_seed(0)
            counter = make_counter()
            if not use_dynamo:
                out.append(counter() + counter())
            else:
                cnts = torch._dynamo.testing.CompileCounter()

                @torch.compile(backend=cnts, fullgraph=True)
                def fn(counter):
                    # [FUNCTION DOC] HELPER: Calls counter twice and sums.
                    return counter() + counter()

                out.append(fn(counter))
                self.assertEqual(cnts.frame_count, 1)
                self.assertEqual(cnts.op_count, 3)
                self.assertFalse(same(counter() + counter(), out[-1]))

        self.assertTrue(same(out[0], out[1]))

    # When we unspecialize float, we wobble this test by changing
    # the op count since previously we would just specialize and constant
    # fold floats into the graph, whereas when we unspecialize we will have
    # ops for item, add, and all other tensorified operations. Since this
    # test really isn't testing that, we purposely specialize floats here.
    @torch._dynamo.config.patch(specialize_float=True)
    def test_closure_out_of_scope_cell(self):
        # [FUNCTION DOC] With float specialization, checks closure over out-of-scope cells (a
        # float and a tensor) through indirect functions; one frame, 1 op.
        cell1 = torch.rand(1).item()
        cell2 = torch.rand(3, 3)

        def indirect():
            # [FUNCTION DOC] HELPER: Calls direct.
            return direct()

        def direct():
            # [FUNCTION DOC] HELPER: Defines and calls inner.
            def inner():
                # [FUNCTION DOC] HELPER: Returns cell values each plus a constant.
                return cell1 + 1, cell2 + 3

            return inner()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(indirect, backend=cnts)
        result1, result2 = opt_fn()
        self.assertAlmostEqual(cell1 + 1, result1)
        self.assertTrue(torch.allclose(cell2 + 3, result2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    # When we unspecialize float, we wobble this test by changing
    # the op count since previously we would just specialize and constant
    # fold floats into the graph, whereas when we unspecialize we will have
    # ops for item, add, and all other tensorified operations. Since this
    # test really isn't testing that, we purposely specialize floats here.
    @torch._dynamo.config.patch(specialize_float=True)
    def test_closure_out_of_scope_cell_with_mutation(self):
        # [FUNCTION DOC] With float specialization, checks mutation of out-of-scope cells over
        # three runs under fullgraph; updated by 1 and 10 each time; 3 ops per run.
        cell1 = torch.rand(1).item()
        orig1 = cell1
        cell2 = torch.rand(3, 3)
        orig2 = cell2.clone()

        def indirect():
            # [FUNCTION DOC] HELPER: Calls direct.
            return direct()

        def direct():
            # [FUNCTION DOC] HELPER: Defines and calls inner.
            def inner():
                # [FUNCTION DOC] HELPER: Updates both cells and returns them with a
                # combined tensor.
                nonlocal cell1, cell2
                x = cell2 + 1
                cell1 += 1
                cell2 += 10
                x = x + cell2
                return cell1, cell2, x

            return inner()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(indirect, backend=cnts, fullgraph=True)
        for i in range(1, 4):
            result1, result2, _ = opt_fn()
            self.assertAlmostEqual(orig1 + 1 * i, result1)
            self.assertTrue(torch.allclose(orig2 + 10 * i, result2))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 3)
            cnts.clear()

    def test_closure_with_mutation_and_graph_break(self):
        # [FUNCTION DOC] Checks a closure variable assigned after a graph break (data-dependent
        # condition) still works; two frames.
        def fn():
            # [FUNCTION DOC] HELPER: Creates zeros, defines subfunc, evaluates a tensor
            # comparison (graph break), sets backup and calls subfunc.
            x = torch.zeros(1)

            def subfunc():
                # [FUNCTION DOC] HELPER: Writes backup into element 0 of x.
                x[0] = backup

            if x[0] >= -1e5:
                pass

            backup = 1
            subfunc()
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        expected = fn()
        actual = opt_fn()
        self.assertTrue(same(expected, actual))
        self.assertEqual(cnts.frame_count, 2)

    def test_closure_out_of_scope_cell_with_cond(self):
        # [FUNCTION DOC] Checks cond with branches reading closure variables in a nested module
        # structure, for both predicate values.
        # Test closure with out-of-scope cell variable, used in a cond
        # where the two branches read different closure variables
        from functorch.experimental.control_flow import cond

        def g(x):
            # [FUNCTION DOC] HELPER: Returns x.
            return x

        class ModuleCondDeep(torch.nn.Module):
            # [CLASS DOC] HELPER: Module delegating through two methods to a cond.
            def forward(self, pred, x):
                # [FUNCTION DOC] HELPER: Calls indirection helper.
                return self._indirection(pred, x)

            def _indirection(self, pred, x):
                # [FUNCTION DOC] HELPER: Calls indirection.
                return self.indirection(pred, x)

            def indirection(self, pred, x):
                # [FUNCTION DOC] HELPER: Defines branch functions and returns cond on pred
                # choosing shallow or deep.
                def true_fn(y):
                    # [FUNCTION DOC] HELPER: Returns y plus 2.
                    return y + 2

                def false_fn(y):
                    # [FUNCTION DOC] HELPER: Returns y minus 2.
                    return y - 2

                def shallow(x):
                    # [FUNCTION DOC] HELPER: Returns x times 2.
                    return x * 2

                def deep(x):
                    # [FUNCTION DOC] HELPER: Cond on first element sign choosing true or
                    # false function.
                    # y = g(x)
                    y = x
                    return cond(
                        x[0][0] > 0,
                        true_fn,
                        false_fn,
                        [y],
                    )

                return cond(pred, shallow, deep, [x])

        mod = ModuleCondDeep()
        opt_mod = torch.compile(mod, backend="eager")
        inp = torch.randn(3, 3)
        exp1 = mod(torch.tensor(False), inp)
        actual1 = opt_mod(torch.tensor(False), inp)
        exp2 = mod(torch.tensor(True), inp)
        actual2 = opt_mod(torch.tensor(True), inp)
        self.assertTrue(torch.allclose(exp1, actual1))
        self.assertTrue(torch.allclose(exp2, actual2))

    def test_closure_write_across_functions(self):
        # [FUNCTION DOC] Checks writes to enclosing variables from functions created and run
        # across frames update the original variables.
        z = 1
        k = 2

        def create_fn():
            # [FUNCTION DOC] HELPER: Returns a function.
            def fn(x):
                # [FUNCTION DOC] HELPER: Assigns z to k.
                nonlocal k, z
                k = z

            return fn

        def update_z_and_run_fn(fn, x):
            # [FUNCTION DOC] HELPER: Sets z to 3, runs fn, returns cosine of x.
            nonlocal z
            z = 3
            fn(x)
            return x.cos()

        @torch.compile(backend="eager")
        def foo(x):
            # [FUNCTION DOC] HELPER: Creates fn and delegates.
            fn = create_fn()
            return update_z_and_run_fn(fn, x)

        x = torch.randn(1)
        foo(x)
        self.assertEqual(3, z)
        self.assertEqual(3, k)

    def test_free_var_and_local_name_collision(self):
        # [FUNCTION DOC] Checks that a local variable and a free variable of a called function
        # with the same name do not collide.
        x = 10

        def make_func():
            # [FUNCTION DOC] HELPER: Returns a function.
            def func():
                # [FUNCTION DOC] HELPER: Returns outer x.
                return x

            return func

        @torch.compile(backend="eager")
        def root(t):
            # [FUNCTION DOC] HELPER: Sets local x to 0, calls the returned function, returns t
            # plus 1, x and the result.
            x = 0
            func = make_func()
            res = func()
            return t + 1, x, res

        res = root(torch.ones(1))
        self.assertTrue(torch.allclose(torch.ones(1) + 1, res[0]))
        self.assertEqual(0, res[1])
        self.assertEqual(10, res[2])

    def test_cell_captured_by_existing_func_but_not_root_frame(self):
        # [FUNCTION DOC] Checks a cell captured by an existing function but not by the compiled
        # root frame.
        x = torch.ones(1)

        def get_inner():
            # [FUNCTION DOC] HELPER: Defines inner, calls it, returns result and inner.
            def inner():
                # [FUNCTION DOC] HELPER: Returns x plus x.
                return x + x

            # Calling `inner` so Dynamo won't skip this frame.
            return inner(), inner

        @torch.compile(backend="eager")
        def root():
            # [FUNCTION DOC] HELPER: Calls get_inner.
            return get_inner()

        res, inner = root()
        self.assertTrue(torch.allclose(x + x, res))
        self.assertTrue(torch.allclose(inner(), res))

    def test_writes_to_cells_across_frames1(self):
        # [FUNCTION DOC] Regression test: write to a pre-existing cell by a called function must
        # be visible when read later in the same compiled region.
        # This regression test was added when Dynamo accidentally had both
        # unboxed and normal modeling for pre-existing cells, and failed to
        # account for buffered writes when we read from the unboxed value.
        x = 0

        def inc_x():
            # [FUNCTION DOC] HELPER: Increments the outer x.
            nonlocal x
            x += 1

        class MyObj:
            # [CLASS DOC] HELPER: Holder with method.
            def inc_x_then_return_x(self, fn):
                # [FUNCTION DOC] HELPER: Calls fn and returns outer x.
                fn()
                return x

        @torch.compile(backend="eager")
        def root(t):
            # [FUNCTION DOC] HELPER: Calls the method with inc_x and returns t plus 1 and
            # the result.
            obj = MyObj()
            res = obj.inc_x_then_return_x(inc_x)
            return t + 1, res

        res = root(torch.zeros(1))
        self.assertTrue(torch.allclose(res[0], torch.ones(1)))
        self.assertEqual(res[1], 1)
        self.assertEqual(x, 1)

    def test_writes_to_cells_across_frames2(self):
        # [FUNCTION DOC] Regression test: reuse of the same pre-existing cell variable across
        # calls so buffered writes are not lost.
        # This regression test was added when Dynamo didn't fully account for
        # already established `CellVariable` instance for pre-existing cell,
        # while encountering the same cell again (we should reuse the instance
        # rather than creating a new one). This caused buffered writes to escape
        # the newly created `CellVariable`.
        x = 0

        def inc_x_and_get_x(obj):
            # [FUNCTION DOC] HELPER: Increments x then returns obj.the getter.
            nonlocal x
            x += 1
            return obj.get_x()

        class MyObj:
            # [CLASS DOC] HELPER: Holder for method.
            def get_x(self):
                # [FUNCTION DOC] HELPER: Returns outer x.
                return x

        @torch.compile(backend="eager")
        def root(t):
            # [FUNCTION DOC] HELPER: Calls inc_x_and_get_x and returns t plus 1 and the result.
            obj = MyObj()
            res = inc_x_and_get_x(obj)
            return t + 1, res

        res = root(torch.zeros(1))
        self.assertTrue(torch.allclose(res[0], torch.ones(1)))
        self.assertEqual(res[1], 1)
        self.assertEqual(x, 1)

    def test_write_to_cells_with_name_shadowing(self):
        # [FUNCTION DOC] Checks cell writes with name shadowing: inner closure x is distinct from
        # outer x.
        x = 0
        y = x

        def make_x_get_set():
            # [FUNCTION DOC] HELPER: Creates inner x and returns getter and setter.
            # NOTE: this `x` is a different cell object than the outer `x`.
            x = y

            def set_x(v):
                # [FUNCTION DOC] HELPER: Assigns v to the inner x.
                nonlocal x
                x = v

            def get_x():
                # [FUNCTION DOC] HELPER: Returns inner x.
                return x

            return get_x, set_x

        get_x, set_x = make_x_get_set()

        @torch.compile(fullgraph=True, backend="eager")
        def fn(t):
            # [FUNCTION DOC] HELPER: Sets inner x to 42 and returns t plus outer x.
            set_x(42)  # This sets the `x` created within `make_x_get_set`
            res = t + x  # This uses the `x` outside `make_x_get_set`.
            return res

        result = fn(torch.ones(1))
        inner_x = get_x()
        self.assertTrue(torch.allclose(result, torch.ones(1)))
        self.assertEqual(inner_x, 42)

    def test_existing_func_that_creates_capturing_nested_func(self):
        # [FUNCTION DOC] Checks an existing function creating a nested function capturing the
        # outer variable, also read by root; later updates visible.
        x = 0  # Captured by both `make_get_x` and `root`

        def make_get_x():
            # [FUNCTION DOC] HELPER: Returns a getter.
            def get_x():
                # [FUNCTION DOC] HELPER: Returns outer x.
                return x

            return get_x

        @torch.compile(backend="eager", fullgraph=True)
        def root(t):
            # [FUNCTION DOC] HELPER: Builds getter, returns t plus x and getter.
            get_x = make_get_x()
            res = t + x
            return res, get_x

        res, get_x = root(torch.ones(1))
        self.assertTrue(torch.allclose(res, torch.ones(1)))
        self.assertEqual(0, get_x())
        x += 1
        self.assertEqual(1, get_x())

    def test_input_cell_mutation(self):
        # [FUNCTION DOC] Checks reassigned input captured by inner function under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Takes cosine then returns sine through inner.
            x = x.cos()

            def inner():
                # [FUNCTION DOC] HELPER: Returns sine of x.
                return x.sin()

            return inner()

        x = torch.ones(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_top_package_import(self):
        # [FUNCTION DOC] Checks import of a submodule inside compiled code and asserting on
        # proxy type.
        def fn(x):
            # [FUNCTION DOC] HELPER: Imports torch.fx, asserts x is not a proxy and returns sine.
            import torch.fx

            assert not isinstance(x, torch.fx.Proxy)  # noqa: S101
            return torch.sin(x)

        x = torch.randn(4, 5)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_typing_typevar(self):
        # [FUNCTION DOC] Checks typing type variable annotations on nested functions; one frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Defines sumt and foo with annotations and applies foo.
            def sumt(y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns sum of y.
                return torch.sum(y)

            def foo(c: typing.Callable[[T], T], y: T) -> T:
                # [FUNCTION DOC] HELPER: Calls c with y.
                return c(y)

            return foo(sumt, x)

        x = torch.randn(3)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)

    def test_typing_union_and_optional(self):
        # [FUNCTION DOC] Checks jit annotate with optional and union typed dict types inside
        # compiled code.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates two annotated empty dicts and returns them with x
            # plus 1.
            a = torch.jit.annotate(typing.Dict[str, typing.Optional[torch.Tensor]], {})
            b = torch.jit.annotate(
                typing.Dict[str, typing.Union[torch.Tensor, None]], {}
            )
            return a, b, x + 1

        x = torch.randn(3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_tying_union_new_syntax(self):
        # [FUNCTION DOC] Checks that functions annotated with pipe-style union types
        # (tensor or none, none or tensor, tensor or list) can be defined inside compiled
        # code under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Defines three inner functions with union annotations and
            # returns x plus 1.
            def inner1(y: torch.Tensor | None):
                # [FUNCTION DOC] HELPER: Returns its argument; annotated as tensor or none.
                return y

            def inner2(y: None | torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns its argument; annotated as none or tensor.
                return y

            def inner3(y: torch.Tensor | list[int]):
                # [FUNCTION DOC] HELPER: Returns its argument; annotated as tensor or list
                # of int.
                return y

            return x + 1

        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))

    @unittest.expectedFailure
    def test_typing_union_new_syntax_reconstruct(self):
        # [FUNCTION DOC] Expected failure: returning pipe-style union type objects from compiled
        # code (reconstruction of those types) is not supported.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 and three union type objects.
            return (
                x + 1,
                torch.Tensor | None,
                None | torch.Tensor,
                torch.Tensor | list[int],
            )

        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))

    def test_optimize_on_module(self):
        # [FUNCTION DOC] Checks compiling an nn.Module with fullgraph returns an object that
        # still forwards custom members to the original module.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ReLU and a no-op custom method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ReLU.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def custom_member(self):
                # [FUNCTION DOC] HELPER: Does nothing; used to check member forwarding.
                # Just for checking that Dynamo returned mod object can redirect
                # to this method
                pass

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies relu.
                return self.relu(x)

        cnts1 = torch._dynamo.testing.CompileCounter()
        mod = MockModule()
        optimized_mod = torch.compile(mod, backend=cnts1, fullgraph=True)

        a = torch.randn(10)
        ref = mod(a)
        res = optimized_mod(a)

        optimized_mod.custom_member()

        self.assertTrue(same(ref, res))

    def test_nested_optimize_decorator(self):
        # [FUNCTION DOC] Checks a function wrapped with the run decorator when called from a
        # compiled function that is called from another compiled function: inner compiles of fn2
        # are skipped (0 frames), outer compiles once with 4 ops.
        cnts2 = torch._dynamo.testing.CompileCounter()
        cnts3 = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.run()
        def fn1(x):
            # [FUNCTION DOC] HELPER: Returns sine of x times 10; wrapped with the run decorator.
            return torch.sin(x) * 10

        @torch.compile(backend=cnts2, fullgraph=True)
        def fn2(x):
            # [FUNCTION DOC] HELPER: Returns fn1(x) plus 1.
            return fn1(x) + 1

        @torch.compile(backend=cnts3, fullgraph=True)
        def fn3(x):
            # [FUNCTION DOC] HELPER: Returns relu of fn2(x).
            return torch.relu(fn2(x))

        fn3(torch.randn(4, 5))
        self.assertEqual(cnts2.frame_count, 0)
        self.assertEqual(cnts3.frame_count, 1)
        self.assertEqual(cnts3.op_count, 4)

    def test_nested_optimize_run(self):
        # [FUNCTION DOC] Checks the run wrapper on an already compiled function does not trigger
        # new compilations: frames 1, then 2 for a new shape, then still 2.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns relu of cos plus sin of x.
            return torch.relu(torch.cos(x) + torch.sin(x))

        fn(torch.randn(4))
        self.assertEqual(cnts.frame_count, 1)

        fn(torch.randn(4, 4))
        self.assertEqual(cnts.frame_count, 2)

        # Test that run works on a decorated fn
        fn = torch._dynamo.run(fn)
        fn(torch.randn(4, 4, 4))
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_optimize(self):
        # [FUNCTION DOC] Checks nested compile wrapping: the inner (first) compile is ignored
        # when called via the outer compile, and the run wrapper reuses the compiled code, in
        # both orders.
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns relu of cos plus sin of x.
            return torch.relu(torch.cos(x) + torch.sin(x))

        fn1 = torch.compile(fn, backend=cnts1, fullgraph=True)
        fn2 = torch.compile(fn1, backend=cnts2, fullgraph=True)

        # The first optimize in the nesting should be ignored
        fn2(torch.randn(4))
        self.assertEqual(cnts2.frame_count, 1)
        self.assertEqual(cnts1.frame_count, 0)

        # Since the fn code object is already compiled, calling fn1 should
        # directly call the compiled_fn callable.
        torch._dynamo.run()(fn1)(torch.randn(4))
        self.assertEqual(cnts1.frame_count, 0)

        # Test same behavior by reversing the calls
        torch._dynamo.reset()
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()
        fn1 = torch.compile(fn, backend=cnts1, fullgraph=True)
        fn2 = torch.compile(fn1, backend=cnts2, fullgraph=True)
        fn1(torch.randn(4))
        self.assertEqual(cnts1.frame_count, 1)
        torch._dynamo.run()(fn2)(torch.randn(4))
        self.assertEqual(cnts2.frame_count, 0)

    def test_torch_size(self):
        # [FUNCTION DOC] Checks use of torch.Size construction and star-unpacking into view
        # under fullgraph.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a Size of 10 by 10, views x with it, returns a tuple.
            output_size = torch.Size([10, 10])
            x = x.view(*output_size)
            return (x,)

        x = torch.randn(100, requires_grad=True)
        x_clone = x.clone()
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x_clone)

        self.assertTrue(same(ref, res))

    def test_torch_size_numel(self):
        # [FUNCTION DOC] Checks numel of a constructed Size under fullgraph.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn():
            # [FUNCTION DOC] HELPER: Returns numel of Size 10 by 8.
            return torch.Size([10, 8]).numel()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        num = torch.Size([10, 8]).numel()
        self.assertEqual(opt_fn(), num)

    def test_torch_size_numel_dynamic(self):
        # [FUNCTION DOC] Checks numel of the size of a tensor under fullgraph.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns numel of x.size.
            return x.size().numel()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.rand(10, 1, 8, 1)
        expect = fn(x)
        self.assertEqual(opt_fn(x), expect)

    def test_shape_type(self):
        # [FUNCTION DOC] Checks that the type of a shape equals the Size class at trace time.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds the boolean result of the type comparison to x.
            return x + (type(x.shape) == torch.Size)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.zeros(())
        self.assertEqual(opt_fn(x), fn(x))

    def test_size_dim(self):
        # [FUNCTION DOC] Checks size with dim keyword (positive and negative) returns 9.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, dim):
            # [FUNCTION DOC] HELPER: Returns x.size for given dim.
            return x.size(dim=dim)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.empty([4, 9, 8])
        self.assertEqual(opt_fn(x, 1), 9)
        self.assertEqual(opt_fn(x, -2), 9)

    def test_torch_size_tensor_index_scalar_constant(self):
        # [FUNCTION DOC] Checks indexing a shape by a scalar tensor under fullgraph and reshape
        # using it.
        def fn(x):
            # [FUNCTION DOC] HELPER: Indexes the shape with a tensor of 1 and reshapes x to
            # that width.
            idx = torch.tensor(1)
            dim_size = x.shape[idx]
            return x.reshape(-1, dim_size)

        x = torch.randn(2, 3, 4)
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)

    def test_torch_size_tensor_index_non_scalar(self):
        # [FUNCTION DOC] Checks indexing a shape with a non-scalar tensor matches eager
        # (TypeError caught, then sum).
        def fn(x):
            # [FUNCTION DOC] HELPER: Tries to index shape with a two-element tensor; on TypeError
            # returns the sum.
            idx = torch.tensor([1, 1])
            try:
                dim_size = x.shape[idx]
                return x * dim_size
            except TypeError:
                return x.sum()

        x = torch.randn(2, 3, 4)
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))

    def test_stride_dim(self):
        # [FUNCTION DOC] Checks stride with dim keyword: 72 for dim 0, 8 for dim -2.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, dim):
            # [FUNCTION DOC] HELPER: Returns x.stride for dim.
            return x.stride(dim=dim)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.empty([4, 9, 8])
        self.assertEqual(opt_fn(x, 0), 72)
        self.assertEqual(opt_fn(x, -2), 8)

    def test_torch_seed(self):
        # [FUNCTION DOC] Checks torch seed call is compiled (1 op, 1 frame) and manual_seed
        # causes a single graph break.
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes seed modulo max int, sets manual seed, returns x in
            # a tuple.
            attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(attention_seed)
            return (x,)

        x = torch.randn(10, requires_grad=True)
        ref = fn(x)

        # Python code is needed here, since torch.manual_seed graph-breaks.
        # Refs: https://github.com/pytorch/pytorch/issues/107187
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        # Only the torch.seed call is turned into an FX graph.
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        # Graph breaks at manual_seed.
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_manual_seed(self):
        # [FUNCTION DOC] Checks manual_seed on a generator argument graph-breaks once, with 1 op
        # and 1 frame.
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x, gen):
            # [FUNCTION DOC] HELPER: Sets generator seed to 3 and returns x plus 1.
            gen.manual_seed(3)
            return x + 1

        x = torch.randn(10)
        ref = fn(x, torch.Generator())

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x, torch.Generator())

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_initial_seed(self):
        # [FUNCTION DOC] Checks calling initial_seed on the default generator causes one graph
        # break; 1 op and 1 frame.
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 and the default generator initial seed.
            return x + 1, torch.default_generator.initial_seed()

        x = torch.randn(10)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_get_state_fullgraph(self):
        # [FUNCTION DOC] Checks get_state on the default generator raises Unsupported
        # under fullgraph.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the default generator state.
            return torch.default_generator.get_state()

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)()

    def test_is_tensor_like(self):
        # [FUNCTION DOC] Checks the is_tensor_like override helper under fullgraph for tensor and
        # int inputs.
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x):
            # [FUNCTION DOC] HELPER: Returns double x in a tuple if x is tensor-like, else ones
            # plus x in a tuple.
            if torch.overrides.is_tensor_like(x):
                return (x * 2,)
            return (torch.ones(10) + x,)

        x = torch.randn(10)
        ref0 = f(x)
        ref1 = f(4)
        opt_f = torch.compile(f, backend=cnts, fullgraph=True)
        res0 = opt_f(x)
        res1 = opt_f(4)
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_is_tensor_like2(self):
        # [FUNCTION DOC] Checks is_tensor_like on a custom class with a torch function hook where
        # max is intercepted to return 123.
        class MyTensor:
            # [CLASS DOC] HELPER: Class implementing the torch function protocol; max returns
            # tensor 123.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Torch function handler: returns tensor 123 for max,
                # otherwise calls the function.
                if kwargs is None:
                    kwargs = {}

                if func is torch.max:
                    return torch.tensor(123)
                return func(*args, **kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns max of x if tensor-like, otherwise zeros.
            if torch.overrides.is_tensor_like(x):
                return torch.max(x)
            else:
                return torch.zeros(1)

        x = MyTensor()
        ref0 = fn(x)
        ref1 = fn(4)
        opt_fn = torch.compile(fn, backend="eager")
        res0 = opt_fn(x)
        res1 = opt_fn(4)
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_tensor_data(self):
        # [FUNCTION DOC] Checks indexing a tensor by the data attribute of an int tensor
        # under fullgraph.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x indexed by y.data.
            return x[y.data]

        x = torch.rand(8)
        y = torch.ones(8).to(torch.int)
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_tensor_layout(self):
        # [FUNCTION DOC] Checks creating zeros with dtype, layout and device taken from another
        # tensor under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns zeros with shape and attributes of x.
            return torch.zeros(
                [x.size()[0], x.size()[1]],
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
            )

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_version_ci(self):
        # [FUNCTION DOC] Sanity check that the torch module has the subclasses attribute,
        # verifying the CI torch version.
        # temporary test to check that the ci torch version is set correctly
        self.assertTrue(hasattr(torch, "_subclasses"))

    def test_slice_input(self):
        # [FUNCTION DOC] Checks slice objects and int indices passed as inputs to a list indexing
        # function under fullgraph.
        cnts = torch._dynamo.testing.CompileCounter()

        def getitem(a, idx):
            # [FUNCTION DOC] HELPER: Indexes list by slice (appending 100) or int; returns tuple
            # with zeros.
            if isinstance(idx, slice):
                return (
                    torch.zeros(1),
                    a[idx]
                    + [
                        100,
                    ],
                )
            else:
                return (torch.zeros(1), a[idx])

        layers = list(range(10))
        ref0 = getitem(layers, slice(0, 2, 1))
        ref1 = getitem(layers, 2)
        ref2 = getitem(layers, slice(3, 8, 2))
        opt_getitem = torch.compile(getitem, backend=cnts, fullgraph=True)
        res0 = opt_getitem(layers, slice(0, 2, 1))
        res1 = opt_getitem(layers, 2)
        res2 = opt_getitem(layers, slice(3, 8, 2))

        self.assertTrue(ref0 == res0)
        self.assertTrue(ref1 == res1)
        self.assertTrue(ref2 == res2)

    def test_grad(self):
        # [FUNCTION DOC] Checks backward and gradient read inside compiled code matches eager.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Multiplies, runs backward, returns sigmoid of a.grad plus b.
            out = a * b
            out.sum().backward()
            real_out = torch.sigmoid(a.grad + b)
            return real_out

        inps = [torch.randn(4, requires_grad=True) for _ in range(2)]
        for inp in inps:
            inp.grad = None
        ref = fn(*inps)

        for inp in inps:
            inp.grad = None
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(*inps)

        self.assertTrue(same(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_source_non_input_grad_access(self):
        # [FUNCTION DOC] With module guarding on, checks reading grad of model parameters that
        # have a source but are not graph inputs; one frame, 3 ops.
        # This test creates a model, and accesses the grads
        # from its parameter. This means that within dynamo,
        # the tensor we are reading the grad from HAS a source,
        # but is not known to graphargs.
        cnts = torch._dynamo.testing.CompileCounter()

        class TrivialModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 2-to-1 linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear layer.
                super(TrivialModel, self).__init__()
                self.linear = torch.nn.Linear(2, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear.
                return self.linear(x)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Loops through global model parameters, collects ones with
            # grad sizes, returns list and last grad plus 1.
            outs = []
            for param in model.parameters():
                outs.append(torch.ones(param.grad.size()))
            return outs, param.grad + 1

        model = TrivialModel()
        # Eager
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()
        ref = fn(a, b)

        # Compiled
        model = TrivialModel()
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(a, b)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    @torch._dynamo.config.patch(graph_break_on_nn_param_ctor=False)
    def test_param_grad_in_forward(self):
        # [FUNCTION DOC] With parameter constructor graph break off, checks a parameter created
        # inside code can have its grad checked and set; one frame.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a parameter, sets grad to zeros if none, returns
            # flag and grad plus x.
            w = torch.nn.Parameter(torch.ones(4, 4))
            saw_none = w.grad is None
            if saw_none:
                w.grad = torch.zeros_like(w)
            return saw_none, w.grad + x

        x = torch.randn(4, 4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)

        self.assertTrue(res[0])
        self.assertEqual(ref[1], res[1])
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(
        graph_break_on_nn_param_ctor=False, trace_autograd_ops=True
    )
    def test_param_autograd_grad_in_forward(self):
        # [FUNCTION DOC] With parameter constructor graph break off and autograd tracing on,
        # checks autograd grad on a parameter created in code returns a grad and leaves
        # w.grad empty.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates parameter, computes grad of a sum, returns flags
            # and grad.
            w = torch.nn.Parameter(torch.ones(4, 4))
            y = (x @ w).sum()
            (grad,) = torch.autograd.grad(y, w)
            return grad is not None, w.grad is None, grad

        x = torch.randn(4, 4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(res[0])
        self.assertTrue(res[1])
        self.assertEqual(ref[2], res[2])

    def test_intermediary_tensor_grad_access(self):
        # [FUNCTION DOC] Checks accessing grad of an intermediate tensor which has no grad; one
        # frame, 2 ops.
        # This test creates a model, and accesses the grads
        # from its parameters and an entirely intermediary tensor.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Creates a tensor, adds, and returns a list holding its grad.
            intermediary = torch.ones(2, 2)
            c = a + intermediary
            outs = []
            outs.append(intermediary.grad)
            return outs

        # Eager
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        ref = fn(a, b)

        # Compiled
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(a, b)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_clone_sparse_input(self):
        # [FUNCTION DOC] Checks cloning input helper on sparse tensors of coo, csr, csc, bsr, bsc
        # layouts from the generator.
        for layout in [
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]:
            for sparse_input in self.generate_simple_inputs(
                layout,
                device="cpu",
                dtype=torch.float64,
                index_dtype=torch.int64,
            ):
                # Invoke the dynamo clone input method directly.
                sparse_copy = torch._dynamo.utils.clone_input(sparse_input)
                # Make sure sparse clone is successful.
                self.assertEqual(sparse_input, sparse_copy)

    @unittest.skipIf(not TEST_CUDA, "pinned CPU memory requires CUDA")
    @unittest.skipIf(
        PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property"
    )
    def test_clone_input_preserves_pinned_cpu_tensor_metadata(self):
        # [FUNCTION DOC] Requires CUDA. Checks clone_input preserves pinned memory, strides,
        # requires_grad, grad (also pinned) and the dynamic indices attribute (copied, not
        # shared); also pinned expanded tensor.
        base = torch.randn(4, 6, pin_memory=True)
        x = base[:, ::2].requires_grad_()
        x.grad = torch.ones_like(x).pin_memory()
        x._dynamo_dynamic_indices = {1}

        cloned = torch._dynamo.utils.clone_input(x)

        self.assertTrue(cloned.is_pinned())
        self.assertEqual(cloned.stride(), x.stride())
        self.assertEqual(cloned, x)
        self.assertNotEqual(cloned.data_ptr(), x.data_ptr())
        self.assertTrue(cloned.requires_grad)
        self.assertIsNotNone(cloned.grad)
        self.assertTrue(cloned.grad.is_pinned())
        self.assertEqual(cloned.grad, x.grad)
        self.assertEqual(cloned._dynamo_dynamic_indices, x._dynamo_dynamic_indices)
        self.assertIsNot(cloned._dynamo_dynamic_indices, x._dynamo_dynamic_indices)

        expanded = torch.arange(3, dtype=torch.float32, pin_memory=True).expand(2, 3)
        expanded_cloned = torch._dynamo.utils.clone_input(expanded)
        self.assertTrue(expanded_cloned.is_pinned())
        self.assertEqual(expanded_cloned, expanded)

    def test_clone_input_does_not_check_pinned_memory_for_wrapper_subclass(self):
        # [FUNCTION DOC] Checks clone_input does not query pinned status for wrapper subclasses
        # (would raise in dispatch handler).
        class WrapperSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass holding an inner tensor, with dispatch
            # handling that unwraps and rewraps.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates a wrapper subclass tensor matching the metadata
                # of elem.
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    requires_grad=elem.requires_grad,
                    device=elem.device,
                )

            def __init__(self, elem):
                # [FUNCTION DOC] HELPER: Stores elem.
                self.elem = elem

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Returns inner tensor names (elem) and no metadata.
                return ["elem"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Rebuilds wrapper from inner tensors with outer size and
                # stride; raises if metadata is not None.
                if metadata is not None:
                    raise AssertionError("Expected metadata to be None")
                return WrapperSubclass(
                    inner_tensors["elem"].as_strided(outer_size, outer_stride)
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Raises if is_pinned is dispatched; otherwise unwraps
                # wrapper args, calls func and wraps tensor outputs.
                if func is torch.ops.aten.is_pinned.default:
                    raise AssertionError(
                        "clone_input should not check pinning on wrapper subclasses"
                    )
                if kwargs is None:
                    kwargs = {}
                args = python_pytree.tree_map_only(
                    WrapperSubclass, lambda x: x.elem, args
                )
                kwargs = python_pytree.tree_map_only(
                    WrapperSubclass, lambda x: x.elem, kwargs
                )
                out = func(*args, **kwargs)
                return python_pytree.tree_map_only(torch.Tensor, WrapperSubclass, out)

        x = WrapperSubclass(torch.randn(4, 6))

        cloned = torch._dynamo.utils.clone_input(x)

        self.assertIsInstance(cloned, WrapperSubclass)
        self.assertEqual(cloned.elem, x.elem)
        self.assertNotEqual(cloned.elem.data_ptr(), x.elem.data_ptr())

    def test_tensor_is_contiguous(self):
        # [FUNCTION DOC] Checks is_contiguous with memory format on a conv output for contiguous
        # and channels-last formats matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convolves random input with a weight converted to memory
            # format x and returns output contiguity.
            input = torch.randn((1, 16, 1, 1))
            weight = torch.randn((8, 16, 3, 3))
            weight = weight.to(memory_format=x)
            output = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
            return output.is_contiguous(memory_format=x)

        opt_fn = torch.compile(fn, backend="eager")
        for x in [torch.contiguous_format, torch.channels_last]:
            self.assertEqual(fn(x), opt_fn(x))

    def test_python_slice(self):
        # [FUNCTION DOC] Checks slicing a list and a shape then enumerating with start 1; sums
        # are 8 and 9.
        def f1(input):
            # [FUNCTION DOC] HELPER: Sums elements of the list after the second one.
            y = 0
            for i, x in enumerate(input[2:], 1):
                y = y + x
            return y

        def f2(input):
            # [FUNCTION DOC] HELPER: Sums shape dims after the second one.
            y = 0
            for i, x in enumerate(input.shape[2:], 1):
                y = y + x
            return y

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f1 = torch.compile(f1, backend=cnts)
        opt_f2 = torch.compile(f2, backend=cnts)
        res1 = opt_f1([1, 2, 3, 5])
        res2 = opt_f2(torch.rand([2, 3, 4, 5]))

        self.assertEqual(res1, 8)
        self.assertEqual(res2, 9)

    def test_const_dict_variable_python_type(self):
        # [FUNCTION DOC] Checks dict and ordered dict constants used in compiled code
        # under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds constants from a dict and an ordered dict to x.
            d = {"a": 1, "b": 2}
            od = collections.OrderedDict([("x", 3), ("y", 4)])
            return x + d["a"] + od["x"]

        eager_result = fn(torch.ones(3))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.ones(3)
        )
        self.assertEqual(eager_result, compiled_result)

    def test_builtin_subclasses_as_method_on_class_type(self):
        # [FUNCTION DOC] Checks listing subclasses of a class from compiled code equals eager
        # (two subclasses).
        # KNOWN ISSUE: Foo init stores to a misspelled attribute name.
        class Foo:
            # [CLASS DOC] HELPER: Base class with a name and getter.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Stores name under a misspelled attribute (ame_).
                self.ame_ = name

            def get_name(self):
                # [FUNCTION DOC] HELPER: Returns Foo prefix plus name.
                return "Foo " + self.name_

        class Bar(Foo):
            # [CLASS DOC] HELPER: Subclass of Foo.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Stores name_.
                self.name_ = name

            def get_name(self):
                # [FUNCTION DOC] HELPER: Returns Bar prefix plus name.
                return "Bar " + self.name_

        class Baz(Foo):
            # [CLASS DOC] HELPER: Another subclass of Foo.
            def __init__(self, name):  # noqa: B903
                # [FUNCTION DOC] HELPER: Stores name_.
                self.name_ = name

            def get_name(self):
                # [FUNCTION DOC] HELPER: Returns Baz prefix plus name.
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()

        counter = CompileCounter()

        @torch._dynamo.optimize_assert(counter)
        def fn():
            # [FUNCTION DOC] HELPER: Returns subclasses of Foo.
            return Foo.__subclasses__()

        subs_of_foo_optim = fn()

        self.assertEqual(len(subs_of_foo_reg), 2)
        self.assertEqual(subs_of_foo_reg, subs_of_foo_optim)

    def test_builtin_subclasses_as_method_on_var(self):
        # [FUNCTION DOC] Checks subclasses of a subclass obtained from a compiled call
        # result list.
        class Foo:
            # [CLASS DOC] HELPER: Base class with name and getter.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Stores name_.
                self.name_ = name

            def get_name(self):
                # [FUNCTION DOC] HELPER: Returns Foo prefix plus name.
                return "Foo " + self.name_

        class Bar(Foo):
            # [CLASS DOC] HELPER: Subclass of Foo.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Stores name_.
                self.name_ = name

            def get_name(self):
                # [FUNCTION DOC] HELPER: Returns Bar prefix plus name.
                return "Bar " + self.name_

        class Baz(Bar):
            # [CLASS DOC] HELPER: Subclass of Bar.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Stores name_.
                self.name_ = name

            def get_name(self):
                # [FUNCTION DOC] HELPER: Returns Baz prefix plus name.
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()
        sub_of_foo_subclass_var_reg = subs_of_foo_reg[0].__subclasses__()

        sub_of_foo_subclass_var_optim = []
        counter = CompileCounter()

        @torch._dynamo.optimize_assert(counter)
        def fn():
            # [FUNCTION DOC] HELPER: Returns subclasses of Foo.
            return Foo.__subclasses__()

        @torch._dynamo.optimize_assert(counter)
        def fn_single(subs_of_foo_optim):
            # [FUNCTION DOC] HELPER: Returns subclasses of the first element.
            return subs_of_foo_optim[0].__subclasses__()

        subs_of_foo_optim = fn()
        sub_of_foo_subclass_var_optim = fn_single(subs_of_foo_optim)

        self.assertEqual(len(sub_of_foo_subclass_var_optim), 1)
        self.assertEqual(sub_of_foo_subclass_var_optim, sub_of_foo_subclass_var_reg)

    def test_builtin_str_on_user_defined_function(self):
        # [FUNCTION DOC] Checks that converting a user function to string within compiled code
        # works under fullgraph.
        def another_fn():
            # [FUNCTION DOC] HELPER: Empty function.
            pass

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether its name appears in its string form.
            return "another_fn" in str(another_fn)

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        self.assertTrue(opt_fn())

    def test_repeat_interleave_graphbreaks(self):
        # [FUNCTION DOC] Checks repeat_interleave: with an int repeat it compiles in one frame;
        # with a tensor repeat it graph breaks to two frames.
        def fn_no_breaks(x):
            # [FUNCTION DOC] HELPER: In-place add, repeat_interleave with int, in-place add.
            # no breaks on self_int
            x += 1
            x = torch.repeat_interleave(x, 2, 3)
            x += 1
            return x

        def fn_has_breaks(x):
            # [FUNCTION DOC] HELPER: In-place add, repeat_interleave with tensor, in-place add.
            # breaks on self_Tensor
            x += 1
            x = torch.repeat_interleave(x, torch.tensor(2), 3)
            x += 1
            return x

        x = torch.randn([4, 16, 1, 64])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn_no_breaks, backend=cnts)
        opt_fn(x)
        self.assertEqual(cnts.frame_count, 1)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn_has_breaks, backend=cnts)
        opt_fn(x)
        self.assertEqual(cnts.frame_count, 2)

    def test_inline_func_jump_on_tensor_condition(self):
        # [FUNCTION DOC] Checks inlining a function that branches on a tensor condition (graph
        # break); gives 3 and 1.
        def f1(input):
            # [FUNCTION DOC] HELPER: Returns input plus 1 if it equals 0 else plus 2.
            if input == 0:
                return input + 1
            else:
                return input + 2

        def f2(input):
            # [FUNCTION DOC] HELPER: Calls f1.
            return f1(input)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch.compile(f2, backend=cnts)
        res1 = opt_f2(torch.tensor([1.0]))
        res2 = opt_f2(torch.tensor([0.0]))

        self.assertEqual(res1, 3)
        self.assertEqual(res2, 1)

    def test_set_discard(self):
        # [FUNCTION DOC] Checks set discard of present and absent elements under fullgraph; 1 op.
        def fn(y):
            # [FUNCTION DOC] HELPER: Creates a set, discards two items, adds its length to y.
            x = set(["bar"])
            x.discard("bar")
            x.discard("foo")
            return y + len(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.randn(3)
        self.assertEqual(opt_fn(x), x)
        self.assertEqual(cnts.op_count, 1)

    def test_set_update(self):
        # [FUNCTION DOC] Checks set update with a mapped iterable mutates the input set and keeps
        # the list.
        @torch.compile(backend="eager", fullgraph=True)
        def run(x, int_set, int_list):
            # [FUNCTION DOC] HELPER: Updates int_set with ints from int_list, returns x plus 1.
            int_set.update(map(int, int_list))
            return x + 1

        int_set = set()
        int_list = [1, 2, 1]
        res = run(torch.ones(1), int_set, int_list)
        self.assertTrue(same(res, torch.ones(1) + 1))
        self.assertEqual(int_set, set([1, 2]))
        self.assertEqual(int_list, [1, 2, 1])

    def test_frozenset_torch_func_contains(self):
        # [FUNCTION DOC] Checks membership of torch functions in a frozenset constant; op counts
        # 2 and 1.
        funcs = frozenset([torch.add])

        def fn(x, func):
            # [FUNCTION DOC] HELPER: Adds 1 if func is in the set, then multiplies by 1.
            if func in funcs:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        x = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, torch.add)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, torch.mul)
        self.assertEqual(cnts.op_count, 1)

    def test_inline_list_mutation(self):
        # [FUNCTION DOC] Checks list append inside an inlined function is visible to the caller.
        def f1(x):
            # [FUNCTION DOC] HELPER: Appends ones to x and returns it.
            x.append(torch.ones(8))
            return x

        def f2():
            # [FUNCTION DOC] HELPER: Creates a list, calls f1, returns the list.
            x = [torch.ones(6)]
            f1(x)
            return x

        res1 = f2()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch.compile(f2, backend=cnts)
        res2 = opt_f2()
        self.assertTrue(same(res1, res2))

    def test_inline_dict_mutation(self):
        # [FUNCTION DOC] Checks dict item set and pop inside an inlined function.
        def f1(d):
            # [FUNCTION DOC] HELPER: Sets c to a plus popped b, returns d.
            d["c"] = d["a"] + d.pop("b")
            return d

        def f2():
            # [FUNCTION DOC] HELPER: Creates dict, calls f1, returns d.
            d = {"a": torch.ones(5), "b": torch.ones(5)}
            f1(d)
            return d

        res1 = f2()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch.compile(f2, backend=cnts)
        res2 = opt_f2()
        self.assertTrue(same(res1, res2))

    def test_inline_local_dict_clear(self):
        # [FUNCTION DOC] Checks clearing an input dict inside compiled code empties the
        # caller's dict.
        def f(d):
            # [FUNCTION DOC] HELPER: Clears d and returns it.
            d.clear()
            return d

        inp = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}
        out = torch.compile(f, backend="eager", fullgraph=True)(inp)
        self.assertEqual(len(out), 0)
        self.assertEqual(len(inp), 0)

    def test_inline_module_attr_dict_clear(self):
        # [FUNCTION DOC] Checks clearing a dict attribute of a module in forward under fullgraph.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a dict attribute of two tensors.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the dict attribute.
                super().__init__()
                self.a = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}

            def forward(self):
                # [FUNCTION DOC] HELPER: Clears dict a and returns it.
                self.a.clear()
                return self.a

        m = MyMod()
        out = torch.compile(m, backend="eager", fullgraph=True)()
        self.assertEqual(len(out), 0)
        self.assertEqual(len(m.a), 0)

    def test_inline_user_defined_dict_attr_clear(self):
        # [FUNCTION DOC] Checks clearing a dict attribute of a plain object after reading
        # its length.
        class MyMod:
            # [CLASS DOC] HELPER: Holder with dict attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates dict a.
                self.a = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}

        def f(obj, inp):
            # [FUNCTION DOC] HELPER: Adds dict length to inp, clears dict, returns dict and sum.
            ret = len(obj.a) + inp
            obj.a.clear()
            return obj.a, ret

        m = MyMod()
        before_len = len(m.a)
        t_inp = torch.ones(1)
        d, ret = torch.compile(f, backend="eager", fullgraph=True)(m, t_inp)
        self.assertEqual(len(m.a), 0)
        self.assertEqual(len(d), 0)
        self.assertEqual(ret, t_inp + before_len)

    def test_recursive_inline_list_mutation(self):
        # [FUNCTION DOC] Checks list append through three nested inlined function calls.
        def f1(x, y):
            # [FUNCTION DOC] HELPER: Appends one element to each list and returns both.
            x.append(torch.tensor([1.1]))
            y.append(torch.tensor([1.2]))
            return x, y

        def f2(x, y):
            # [FUNCTION DOC] HELPER: Appends to both lists, then calls f1.
            x.append(torch.tensor([2.1]))
            y.append(torch.tensor([2.2]))
            f1(x, y)
            return x, y

        def f3(x):
            # [FUNCTION DOC] HELPER: Appends to x, creates y, calls f2, returns both.
            x.append(torch.tensor([3.1]))
            y = [torch.tensor([3.2])]
            f2(x, y)
            return x, y

        def f4():
            # [FUNCTION DOC] HELPER: Creates x list and calls f3.
            x = [torch.tensor([4.1])]
            return f3(x)

        res1 = f4()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f4 = torch.compile(f4, backend=cnts)
        res2 = opt_f4()
        self.assertTrue(same(res1, res2))

    def test_sample_input(self):
        # [FUNCTION DOC] Checks a SampleInput object attribute with isinstance branch
        # matches eager.
        from torch.testing._internal.common_methods_invocations import SampleInput

        def fn(sample):
            # [FUNCTION DOC] HELPER: Returns double sample.input if it is a tensor, else zeros.
            if isinstance(sample.input, torch.Tensor):
                return sample.input * 2
            return torch.zeros(())

        sample = SampleInput(torch.ones(2))
        ref = fn(sample)

        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(sample)

        self.assertTrue(same(ref, res))

    @skipIfWindows(
        msg="TODO(xuhancn): confirm, AssertionError: tensor([0.0290, 0.4019, 0.2598, 0.3666]) is not None"
    )
    def test_release_input_memory(self):
        # [FUNCTION DOC] Checks that compiled call does not keep input tensor alive: after
        # deleting the input the weak reference is dead. Skipped on Windows.
        x = torch.rand([4])
        x_ref = weakref.ref(x)

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns x plus x.
            return x + x

        out = foo(x)
        self.assertTrue(same(out, x + x))
        del x
        self.assertIs(x_ref(), None)

    @skipIfWindows(
        msg="TODO: (xuhancn) conform, AssertionError: Linear(in_features=10, out_features=10, bias=True) is not None"
    )
    def test_release_module_memory(self):
        # [FUNCTION DOC] Checks that a module passed through compiled code is not retained by the
        # backend path; weak references to the module and its weight die. Skipped on Windows.
        mod = torch.nn.Linear(10, 10)
        x = torch.rand([10, 10])
        mod_weight_ref = weakref.ref(mod.weight)
        mod_ref = weakref.ref(mod)

        # Modules that are passed into torch._dynamo optimized functions
        # will normally be held onto through the generated GraphModule,
        # which contains the modules. remove the reference in this backend
        # and test that no additional references are being held.
        class NoLeakBackend:
            # [CLASS DOC] HELPER: Backend that clears the module reference on the graph module
            # and returns a dummy function.
            def __call__(self, gm: torch.fx.GraphModule, example_inputs):
                # [FUNCTION DOC] HELPER: Sets the mod attribute of gm to None and returns
                # a function.
                gm.mod = None

                def foo(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: Returns a tuple with 1.
                    return (1,)

                return foo

        no_leak_backend = NoLeakBackend()

        @torch.compile(backend=no_leak_backend)
        def foo(mod, x):
            # [FUNCTION DOC] HELPER: Applies the module to x.
            return mod(x)

        foo(mod, x)
        del mod
        del x
        self.assertIsNone(mod_ref(), None)
        self.assertIsNone(mod_weight_ref(), None)

    @skipIfWindows(msg="TODO: (xuhancn) conform, AssertionError: False is not true")
    def test_release_scope_memory(self):
        # [FUNCTION DOC] Checks tensor weak reference dies after deleting the variable even after
        # passing it to a compiled function. Skipped on Windows.
        def inner(y):
            # [FUNCTION DOC] HELPER: Evaluates y as an expression statement and returns nothing.
            y

        inner = torch.compile(inner, backend="eager")

        p_ref = None

        x = torch.randn((10, 10))
        inner(x)

        p_ref = weakref.ref(x)
        self.assertTrue(p_ref() is not None)
        del x
        self.assertTrue(p_ref() is None)

    def test_update_locals_and_stack_uses_shared_cache(self):
        # [FUNCTION DOC] Checks permutation building from list operations and a generator extend
        # under compile matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a permutation list from ranges and extends it with
            # missing dims.
            perm = [0, 3, 5]
            perm = list(range(min(perm))) + perm
            perm.extend(i for i in range(x.dim()) if i not in perm)
            return perm

        x = torch.rand([2, 2, 2, 2, 2, 2])
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_side_effects_codegen_update_mutated(self):
        # [FUNCTION DOC] Checks updating mutated lists and dicts with graph breaks in between
        # generates correct code.
        # codegen to update mutated variables with side effect
        # should after stack value's codegen
        def f1(x):
            # [FUNCTION DOC] HELPER: Appends to list, graph breaks on item calls, pops, returns.
            alist = [x]
            alist.append(x + 1)
            alist[0].sum().item()  # graph break
            res = alist.pop()
            res.sum().item()  # graph break
            return res

        def f2(a, b):
            # [FUNCTION DOC] HELPER: Builds dict, pops b, graph breaks, adds, sets c, returns d.
            d = {"a": a + 1, "b": b + 2}
            x = d.pop("b")
            x.sum().item()  # graph break
            y = d["a"] + x
            y.sum().item()  # graph break
            d["c"] = y
            return d

        x = torch.rand([2, 3])
        a = torch.rand([5, 6])
        b = torch.rand([5, 6])
        res11 = f1(x)
        res21 = f2(a, b)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f1 = torch.compile(f1, backend=cnts)
        opt_f2 = torch.compile(f2, backend=cnts)
        res12 = opt_f1(x)
        res22 = opt_f2(a, b)
        self.assertTrue(same(res11, res12))
        self.assertTrue(same(res21, res22))

    def test_replay_side_effects_config(self):
        # [FUNCTION DOC] Checks the replay side effects config: enabled by default the list
        # mutation replays; with replay disabled and warn policy it does not; with error policy a
        # RuntimeError names the source.
        # Test that replay_side_effects config controls mutation replay
        def fn(x, lst):
            # [FUNCTION DOC] HELPER: Appends x plus 1 to lst and returns double x.
            lst.append(x + 1)
            return x * 2

        x = torch.tensor([5.0])

        # Test with replay enabled (default)
        lst_with_replay = []
        opt_fn_with_replay = torch.compile(fn, backend="eager")
        result1 = opt_fn_with_replay(x, lst_with_replay)
        self.assertEqual(len(lst_with_replay), 1)  # Mutation should be replayed
        self.assertTrue(same(result1, x * 2))

        torch._dynamo.reset()

        # Test with replay disabled
        lst_without_replay = []
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="warn"
        ):
            opt_fn_without_replay = torch.compile(fn, backend="eager")
            result2 = opt_fn_without_replay(x, lst_without_replay)
            self.assertEqual(
                len(lst_without_replay), 0
            )  # Mutation should NOT be replayed
            self.assertTrue(same(result2, x * 2))

        torch._dynamo.reset()
        lst_without_replay = []
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="error"
        ):
            opt_fn_without_replay = torch.compile(fn, backend="eager")
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: [\"L['lst']\"]"
                ),
            ):
                _ = opt_fn_without_replay(x, lst_without_replay)

    def test_replay_side_effects_model_attr(self):
        # [FUNCTION DOC] Checks side effects on module attributes with replay disabled: error
        # policy raises RuntimeError about self; silent policy leaves attributes unchanged.
        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule with const 4 and cos forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets const 4.
                super().__init__()
                self.const = 4

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns cos of x.
                return x.cos()

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that mutates const and tensor in forward and uses
            # the submodule.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets const, tensor and submodule.
                super().__init__()
                self.const = 4
                self.tensor = None
                self.bar = Bar()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sets const to 5 and tensor to sin of x, combines cos,
                # submodule sum and tensor.
                self.const = 5
                self.tensor = x.sin()
                res = self.bar(x)
                return x.cos() + res.sum() + self.tensor

        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="error"
        ):
            foo = Foo()
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: [\"L['self']\"]"
                ),
            ):
                torch.compile(foo, fullgraph=True, backend="eager")(torch.randn(4, 4))

        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="silent"
        ):
            foo_v2_compile = Foo()
            foo_v2_eager = Foo()
            inp = torch.randn(4, 4)
            res = torch.compile(foo_v2_compile, fullgraph=True, backend="eager")(
                torch.randn(4, 4)
            )
            self.assertEqual(foo_v2_compile.tensor, None)
            self.assertEqual(foo_v2_compile.const, 4)
            self.assertEqual(foo_v2_compile.bar.const, 4)
            same(res, foo_v2_eager(inp))

    def test_replay_side_effects_input_mut(self):
        # [FUNCTION DOC] Checks that with side-effect replay disabled and the error policy, a
        # module that mutates its input in place still compiles under fullgraph, since the
        # mutation is captured in the graph.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with const and tensor attributes.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets const to 4 and tensor to None.
                super().__init__()
                self.const = 4
                self.tensor = None

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 5 to x in place and returns cosine of x.
                x.add_(5)
                return x.cos()

        # This is ok because we actually capture the graph which
        # has mutation. In export, we never retrace the actual
        # gm so we won't see any mutation applied to inputs
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="error"
        ):
            foo = Foo()
            torch.compile(foo, fullgraph=True, backend="eager")(torch.randn(4, 4))

    def test_list_append_return_none(self):
        # [FUNCTION DOC] Checks that a list append result (None) is returned and the list
        # updated, matching eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Appends x plus 1 to a list and returns the list with the
            # append result.
            alist = []
            blist = alist.append(x + 1)
            return alist, blist

        x = torch.tensor([2.3])
        res = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertEqual(res, res2)

    def test_function_return_none_creates_constant_variable(self):
        # [FUNCTION DOC] Regression test: a function returning nothing that is registered as a
        # reorderable logging function must produce a constant None value and not crash the
        # stack. Test passes if no crash.
        """
        Test that functions returning None properly return ConstantVariable.create(None)
        instead of raw None, which would violate the stack's type contract.

        Regression test for: Avoid using Optional[VariableTracker]
        """

        def gn(x):
            # [FUNCTION DOC] HELPER: Returns nothing.
            return

        torch._dynamo.config.reorderable_logging_functions.add(gn)

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1, branches on whether gn returns None, and adds 2
            # or 4.
            x = x + 1
            if gn(x) is None:
                return x + 2
            return x + 4

        # If this doesn't crash, the test passes
        fn(torch.ones(3))

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_tensor_ctor_list_of_tensor(self):
        # [FUNCTION DOC] With scalar capture on, checks building a tensor from a list holding a
        # 0-d tensor with dtype int64; one frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a long tensor from a one-element list holding x.
            return torch.tensor([x], dtype=torch.int64)

        x = torch.tensor(20)
        res = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertEqual(res, res2)
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_types(self):
        # [FUNCTION DOC] Checks creating empty tensors of each dtype and asserting isinstance of
        # the legacy tensor types, for ten dtype and type pairs.
        def fn(dtype, tensor_type):
            # [FUNCTION DOC] HELPER: Creates an empty tensor of dtype and asserts it is an
            # instance of tensor_type.
            x = torch.empty(4, dtype=dtype)
            assert isinstance(x, tensor_type)  # noqa: S101

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.float32, torch.FloatTensor)
        opt_fn(torch.float64, torch.DoubleTensor)
        opt_fn(torch.float16, torch.HalfTensor)
        opt_fn(torch.bfloat16, torch.BFloat16Tensor)
        opt_fn(torch.uint8, torch.ByteTensor)
        opt_fn(torch.int8, torch.CharTensor)
        opt_fn(torch.int64, torch.LongTensor)
        opt_fn(torch.int, torch.IntTensor)
        opt_fn(torch.int16, torch.ShortTensor)
        opt_fn(torch.bool, torch.BoolTensor)

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_or_union_type_opaque_class(self):
        # [FUNCTION DOC] Requires distributed. Checks the or operator on two placement classes
        # inside compiled code does not graph break (one frame).
        # Test that or_ on opaque class types (e.g. Shard | _StridedShard)
        # doesn't cause a graph break.
        from torch.distributed.tensor import Shard
        from torch.distributed.tensor.placement_types import _StridedShard

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Evaluates the union of the two placement classes, returns x
            # plus 1.
            _ = Shard | _StridedShard
            return x + 1

        x = torch.randn(4)
        result = fn(x)
        self.assertEqual(result, x + 1)
        self.assertEqual(cnt.frame_count, 1)

    def test_nan(self):
        # [FUNCTION DOC] Checks a float nan argument does not cause a recompile on the second
        # call; one frame.
        def f(x, n):
            # [FUNCTION DOC] HELPER: Returns x times 2 plus n.
            return x * 2 + n

        x = torch.randn(4)
        n = float("nan")

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnts)
        opt_f(x, n)
        opt_f(x, n)
        self.assertEqual(cnts.frame_count, 1)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item(self):
        # [FUNCTION DOC] With scalar capture, checks item of integer-cast max returns 11
        # under fullgraph.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning item of integer cast of max of x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes max, casts to int, returns item.
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        y = torch.compile(model, backend="eager", fullgraph=True)(x)

        self.assertEqual(y, 11)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item_changes(self):
        # [FUNCTION DOC] With scalar capture, checks that values from item are recomputed for
        # different inputs of the same shape (11 then 61).
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning item of integer cast of max of x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes max, casts to int, returns item.
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        y = opt_model(x)
        z = opt_model(torch.tensor([[y - 5, y + 10, y + 50]]))

        self.assertEqual(y, 11)
        self.assertEqual(z, 61)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item_changes_new_shape(self):
        # [FUNCTION DOC] With scalar capture, checks item for a changed input shape (11 then 61).
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning item of integer cast of max of x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes max, casts to int, returns item.
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        y = opt_model(x)
        z = opt_model(torch.tensor([[y - 5, y + 50], [y + 5, y - 50]]))

        self.assertEqual(y, 11)
        self.assertEqual(z, 61)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_foreach_tensor_scalar(self):
        # [FUNCTION DOC] With scalar capture, checks several foreach ops using a 0-d tensor
        # scalar (addcmul, addcdiv, add, mul, pow) trace under fullgraph and match eager.
        # Test that foreach ops with tensor scalar values can be traced fullgraph
        # when capture_scalar_outputs is enabled. The scalar's .item() creates an
        # unbacked symbol that should be properly ignored since it only affects
        # tensor values, not shapes.
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        val = torch.tensor(0.5, dtype=torch.float64)

        # Test _foreach_addcmul with value arg
        def fn_addcmul(a, b, c, val):
            # [FUNCTION DOC] HELPER: Foreach addcmul with value derived from val.
            return torch._foreach_addcmul([a], [b], [c], value=1 - val)

        expected = fn_addcmul(a, b, c, val)
        actual = torch.compile(fn_addcmul, backend="eager", fullgraph=True)(
            a, b, c, val
        )
        self.assertEqual(expected, actual)

        # Test _foreach_addcdiv with value arg
        def fn_addcdiv(a, b, c, val):
            # [FUNCTION DOC] HELPER: Foreach addcdiv with value derived from val.
            return torch._foreach_addcdiv([a], [b], [c], value=1 - val)

        expected = fn_addcdiv(a, b, c, val)
        actual = torch.compile(fn_addcdiv, backend="eager", fullgraph=True)(
            a, b, c, val
        )
        self.assertEqual(expected, actual)

        # Test _foreach_add with scalar arg
        def fn_add(a, val):
            # [FUNCTION DOC] HELPER: Foreach add with scalar derived from val.
            return torch._foreach_add([a], 1 - val)

        expected = fn_add(a, val)
        actual = torch.compile(fn_add, backend="eager", fullgraph=True)(a, val)
        self.assertEqual(expected, actual)

        # Test _foreach_mul with scalar arg
        def fn_mul(a, val):
            # [FUNCTION DOC] HELPER: Foreach mul with scalar derived from val.
            return torch._foreach_mul([a], 1 - val)

        expected = fn_mul(a, val)
        actual = torch.compile(fn_mul, backend="eager", fullgraph=True)(a, val)
        self.assertEqual(expected, actual)

        # Test _foreach_pow with scalar exponent
        def fn_pow(a, val):
            # [FUNCTION DOC] HELPER: Foreach pow with exponent from val item.
            return torch._foreach_pow([a.abs()], 1 - val.item())

        expected = fn_pow(a, val)
        actual = torch.compile(fn_pow, backend="eager", fullgraph=True)(a, val)
        self.assertEqual(expected, actual)

    # Test that ops with Scalar arguments (alpha, beta, value) work with
    # 0-d tensor inputs under fullgraph=True. The C++ arg parser calls
    # .item() on 0-d tensors to convert them to Scalars, creating unbacked
    # symbols that should be ignored since they only affect tensor values,
    # not shapes.
    @parametrize(
        "op",
        [
            "add_method",
            "add_positional",
            "add_func",
            "addcmul",
            "addcmul_positional",
            "addcdiv",
            "addcdiv_positional",
            "baddbmm",
        ],
    )
    def test_scalar_arg_0d_tensor(self, op):
        # [FUNCTION DOC] Parametrized over eight op names. Calls add, addcmul, addcdiv or baddbmm
        # variants with 0-d tensor alpha, beta or value arguments under fullgraph and compares
        # with eager.
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        alpha = torch.tensor(2.0)
        beta = torch.tensor(0.5)
        batch1 = torch.randn(2, 4, 3)
        batch2 = torch.randn(2, 3, 4)
        m_batch = torch.randn(2, 4, 4)

        cases = {
            "add_method": lambda: a.add(b, alpha=alpha),
            "add_positional": lambda: a.add(alpha, b),
            "add_func": lambda: torch.add(a, b, alpha=alpha),
            "addcmul": lambda: a.addcmul(b, c, value=beta),
            "addcmul_positional": lambda: a.addcmul(beta, b, c),
            "addcdiv": lambda: a.addcdiv(b, c.abs() + 1, value=beta),
            "addcdiv_positional": lambda: a.addcdiv(beta, b, c.abs() + 1),
            "baddbmm": lambda: m_batch.baddbmm(batch1, batch2, alpha=alpha, beta=beta),
        }

        fn = cases[op]
        expected = fn()
        actual = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(expected, actual)

    @unittest.skip("https://github.com/pytorch/pytorch/issues/99726")
    def test_cross_entropy_loss_fancy_ctor1(self):
        # [FUNCTION DOC] Skipped test (issue 99726): cross entropy loss module constructed with
        # weight, reduce off and label smoothing compiled vs eager.
        rand_5 = torch.randn(5)
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        opt_loss = torch.compile(loss, backend="eager", fullgraph=True)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_fancy_ctor2(self):
        # [FUNCTION DOC] Checks cross entropy loss module with reduce off and label smoothing
        # compiled under fullgraph matches eager.
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=0.5)
        opt_loss = torch.compile(loss, backend="eager", fullgraph=True)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=0.5)
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_simple_ctor(self):
        # [FUNCTION DOC] Checks default cross entropy loss module compiled under fullgraph
        # matches eager.
        output = None
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss()
        opt_loss = torch.compile(loss, backend="eager", fullgraph=True)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss()
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_prob_target_dynamic(self):
        # [FUNCTION DOC] Checks functional cross entropy with probability targets under
        # dynamic shapes over reductions, label smoothing values and optional weights,
        # comparing with eager.
        shapes = [(3, 5), (4, 7)]
        compiled = torch.compile(F.cross_entropy, dynamic=True, backend="eager")
        for N, C in shapes:
            x = torch.randn(N, C)
            target = torch.rand(N, C).softmax(dim=1)
            weight = torch.rand(C)
            for reduction in ["none", "mean", "sum"]:
                for label_smoothing in [0.0, 0.5]:
                    for w in [None, weight]:
                        expected = F.cross_entropy(
                            x,
                            target,
                            weight=w,
                            reduction=reduction,
                            label_smoothing=label_smoothing,
                        )
                        actual = compiled(
                            x,
                            target,
                            weight=w,
                            reduction=reduction,
                            label_smoothing=label_smoothing,
                        )
                        self.assertEqual(expected, actual)

    def test_repr(self):
        # [FUNCTION DOC] Checks calling repr on a user object with custom repr inside compiled
        # code under fullgraph.
        class Config:
            # [CLASS DOC] HELPER: Class with repr returning the Config repr text.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a fixed string.
                return "Config()"

        def forward(x, config):
            # [FUNCTION DOC] HELPER: Scales x by the length of the repr text.
            return x * len(repr(config))

        config = Config()
        x = torch.randn(2, 2)

        compiled = torch.compile(forward, fullgraph=True, backend="eager")
        compiled(x, config)

    def test_nn_functional_reduction(self):
        # [FUNCTION DOC] Checks use of the reduction enum helper with a string, with mean branch,
        # under fullgraph.
        def fn(loss, reduction):
            # [FUNCTION DOC] HELPER: Gets the reduction enum and returns loss, its mean or
            # its sum.
            reduction_enum = F._Reduction.get_enum(reduction)
            if reduction_enum == 0:
                return loss
            elif reduction_enum == 1:
                return loss.mean()
            elif reduction_enum == 2:
                return loss.sum()

        x = torch.rand([3, 5])
        y = "mean"
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertTrue(torch.allclose(ref, res))

    def test_large_reduction_list(self):
        # [FUNCTION DOC] Checks the sum of a large (200000 element) tensor equals the python sum
        # of its list conversion.
        dtype = torch.float32
        device = "cpu"

        def check_sum_all(tensor: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Converts the tensor to a list and asserts tensor sum equals
            # python sum.
            pylist = tensor.reshape(-1).tolist()
            self.assertTrue(same(tensor.sum(), torch.tensor(sum(pylist))))

        check_sum_all(torch.randn(200000, dtype=dtype, device=device))

    def test_raise_on_backend_error(self):
        # [FUNCTION DOC] Checks an error raised by a backend compiler surfaces
        # as BackendCompilerFailed.
        def my_compiler(gm, _):
            # [FUNCTION DOC] HELPER: Backend that always raises RuntimeError.
            raise RuntimeError("duck!")

        @torch.compile(backend=my_compiler)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns a plus b divided by a minus b.
            return a + b / (a - b)

        self.assertRaises(
            torch._dynamo.exc.BackendCompilerFailed,
            lambda: fn(torch.randn(10), torch.randn(10)),
        )

    def test_named_parameters(self):
        # [FUNCTION DOC] Checks named_parameters (with and without a prefix) called within
        # compiled code returns the same names and values as eager.
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class MyModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule with embedding, parameter and dropout.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates embedding, positional parameter and dropout.
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with embedding, parameter, dropout and a submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates its layers and the submodule.
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)
                self.submod2 = MyModel2()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        # Regular
        params = []
        mod = MyModel()
        actual_params = list(mod.named_parameters())

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Returns the list of named parameters.
            return list(mod.named_parameters())

        params = fn()

        self.assertEqual(len(actual_params), len(params))
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            self.assertTrue(torch.allclose(v_a, v))

        # Prefix
        params = []
        mod = MyModel()
        actual_params = list(mod.named_parameters(prefix="foo"))

        @torch.compile(backend="eager", fullgraph=True)
        def fn1():
            # [FUNCTION DOC] HELPER: Returns the list of named parameters with prefix.
            return list(mod.named_parameters(prefix="foo"))

        params = fn1()

        self.assertEqual(len(actual_params), len(params))
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            self.assertTrue(torch.allclose(v_a, v))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_module_complex_iter(self):
        # [FUNCTION DOC] With module guarding on, checks iterating named_modules and
        # named_parameters within a method of a compiled module yields the same names list as
        # eager, with and without a prefix.
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class FakeGPT(torch.nn.Module):
            # [CLASS DOC] HELPER: Small GPT-like module collecting parameter names in a list.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates embeddings, parameter, dropout, layer norm, head
                # and name list. Note it never creates blocks.
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)
                self.ln_f = torch.nn.LayerNorm(n_embd)
                self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)

                self.block_size = block_size
                self.names = []

            def forward(self, idx, targets=None):
                # [FUNCTION DOC] HELPER: GPT-style forward: embeds, applies blocks, norm, head
                # and optional loss.
                # KNOWN ISSUE: blocks is never defined, so this forward would fail if called; the
                # test only calls foo.
                b, t = idx.size()
                assert t <= self.block_size, (  # noqa: S101
                    "Cannot forward, model block size is exhausted."
                )

                # forward the GPT model
                token_embeddings = self.tok_emb(
                    idx
                )  # each index maps to a (learnable) vector
                position_embeddings = self.pos_emb[
                    :, :t, :
                ]  # each position maps to a (learnable) vector
                x = self.drop(token_embeddings + position_embeddings)
                x = self.blocks(x)
                x = self.ln_f(x)
                logits = self.head(x)

                # if we are given some desired targets also calculate the loss
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )

                return logits, loss

            def foo(self, memo=None, prefix="", remove_duplicate=False):
                # [FUNCTION DOC] HELPER: Appends dotted module and parameter names to the names
                # list for every module and parameter.
                for mn, m in self.named_modules(
                    memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
                ):
                    for pn, p in self.named_parameters():
                        fpn = f"{mn}.{pn}" if mn else pn
                        self.names.append(fpn)

        # Test plain recurse
        model_a = FakeGPT()
        model_a.foo()
        a_names = model_a.names

        model_b = FakeGPT()
        opt_model_b = torch.compile(model_b, backend="eager", fullgraph=True)
        opt_model_b.foo()

        self.assertEqual(a_names, model_b.names)

        # Test with prefix
        model_a = FakeGPT()
        model_a.foo(prefix="abc")
        a_names = model_a.names

        model_b = FakeGPT()
        opt_model_b = torch.compile(model_b, backend="eager", fullgraph=True)
        opt_model_b.foo(prefix="abc")

        self.assertEqual(a_names, model_b.names)

    def test_numpy_variable_isinstance(self):
        # [FUNCTION DOC] Checks isinstance on a numpy array input for both an array and a tensor.
        def fn(x, m):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if m is an ndarray, else x minus 1.
            if isinstance(m, np.ndarray):
                return x + 1
            else:
                return x - 1

        x = torch.tensor([2.3])
        m = np.array([1, 2, 3])
        ref = fn(x, m)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, m)
        self.assertEqual(ref, res)

        # Test now the other path
        ref = fn(x, x)
        res = opt_fn(x, x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_tensor_dot_grad_no_graph_break(self):
        # [FUNCTION DOC] With autograd tracing on, checks backward with gradient and zero_ on
        # grad compiles in one frame; grad of b is zero.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes a polynomial, runs backward, zeroes b grad and
            # returns both grads.
            y = 3 * a**3 - b**2
            y.backward(gradient=torch.tensor([1.0, 1.0]))
            b.grad.zero_()
            return a.grad, b.grad

        a = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor([6.0, 4.0], requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        _, b_grad = opt_fn(a, b)
        self.assertTrue(same(b_grad, torch.tensor([0.0, 0.0])))
        self.assertEqual(cnts.frame_count, 1)

    def test_torch_nn_parameter_isinstance(self):
        # [FUNCTION DOC] Checks isinstance of a parameter created inside code against tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a parameter and returns x plus 1 or x minus 1 based
            # on isinstance.
            a = torch.nn.Parameter(torch.rand(2, 3))
            if isinstance(a, torch.Tensor):
                return x + 1
            else:
                return x - 1

        x = torch.tensor([2.5])
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def _optimize_then_check_exp(
        self, foo, args, cnt, exp_out, exp_frame_count, exp_n_cached_backend
    ):
        # [FUNCTION DOC] HELPER: Compiles foo with the counting backend, calls it with args,
        # and asserts the output equals the expected output and the frame count equals the
        # expected one.
        # KNOWN ISSUE: the last parameter (expected cached backends) is accepted but never used.
        opt_out = torch.compile(foo, backend=cnt)(*args)
        self.assertEqual(exp_out, opt_out)
        self.assertEqual(cnt.frame_count, exp_frame_count)

    def test_backend_match_guard(self):
        # [FUNCTION DOC] Checks recompilation on backend change: for each of two backends,
        # repeated calls do not recompile; function without break gives 1 frame, with a break
        # 2 frames.
        x = torch.randn([3, 4])

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns sin plus cos of x.
            return x.sin() + x.cos()

        def foo_graph_break(x):
            # [FUNCTION DOC] HELPER: Sin, graph break, cos, sum.
            a = x.sin()
            torch._dynamo.graph_break()
            b = x.cos()
            return a + b

        eager_record_backend = torch._dynamo.testing.EagerAndRecordGraphs()
        backends = [eager_record_backend, "eager"]

        # We intentionally don't reset dynamo for each backend so that we can test
        # 1. dynamo doesn't recompile when backend stays the same, i.e. frame_count doesn't increase
        # 2. dynamo recompiles when backend changes, i.e. frame_count is non-zero for next backend
        def test_recompile(foo, *, exp_frame_count):
            # [FUNCTION DOC] HELPER: Runs the function eagerly, then for each backend creates a
            # counter and checks three calls.
            eager_result = foo(x)
            for i, backend in enumerate(backends):
                cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
                # Run opt_f multiple times to make sure dynamo doesn't recompile.
                # Specifically, frame_count doesn't increase
                # the number of cached backends is i + 2 because we have the optimizing backend + None
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )

        test_recompile(foo, exp_frame_count=1)
        torch._dynamo.reset()
        test_recompile(foo_graph_break, exp_frame_count=2)

    def test_multithread_compile_dynamic(self):
        # [FUNCTION DOC] Runs compiled functions from two threads with different sizes, checking
        # no exceptions (via a thread exception hook), for static and for dynamic-config-patched
        # compile paths.
        def f(x):
            # [FUNCTION DOC] HELPER: Asserts the first dimension is static at compile time and
            # returns x squared.
            comptime.assert_static(x.shape[0])
            return x * x

        def _do_test(func):
            # [FUNCTION DOC] HELPER: Runs func in two threads with different size offsets,
            # records any exception with the exception hook, restores the hook, asserts success.
            success = True

            def run(offset):
                # [FUNCTION DOC] HELPER: Calls func twenty times on random tensors of growing
                # size and prints results.
                for i in range(20):
                    print(func(torch.randn(i * 2 + offset)))

            t1 = threading.Thread(target=run, args=[0])
            t2 = threading.Thread(target=run, args=[1])

            def exc_hook(x):
                # [FUNCTION DOC] HELPER: Marks failure in the enclosing flag.
                nonlocal success
                success = False

            try:
                threading.excepthook = exc_hook
                t1.start()
                t2.start()

                t1.join()
                t2.join()
            finally:
                threading.excepthook = threading.__excepthook__
            self.assertTrue(success)

        _do_test(torch.compile(f, backend="eager", dynamic=False))
        torch._dynamo.reset()

        f_opt = torch.compile(f, backend="eager")

        def g(x):
            # [FUNCTION DOC] HELPER: Runs the optimized function with automatic dynamic off and
            # static default.
            with torch._dynamo.config.patch(
                automatic_dynamic_shapes=False, assume_static_by_default=True
            ):
                f_opt(x)

        _do_test(g)

    def test_backend_match_guard_multi_threads(self):
        # [FUNCTION DOC] Checks compile with different backends in separate threads each caches
        # one backend and has one frame.
        x = torch.randn([3, 4])

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns sin plus cos of x.
            return x.sin() + x.cos()

        def compile_then_check_exp(foo, args, cnt, eager_result, exp_frame_count):
            # [FUNCTION DOC] HELPER: Compiles and calls foo three times, asserts outputs and
            # frame count, records thread success.
            for i in range(3):
                opt_out = torch.compile(foo, backend=cnt)(*args)
                self.assertEqual(opt_out, eager_result)
            self.assertEqual(cnt.frame_count, exp_frame_count)
            thread_success[threading.current_thread()] = True

        eager_record_backend = torch._dynamo.testing.EagerAndRecordGraphs()
        backends = [eager_record_backend, "eager"]

        # Test dynamo recompiles but only caches a single backend for each thread
        eager_result = foo(x)
        # cnt and None
        exp_frame_count = 1
        threads = []
        thread_success = {}
        for i, backend in enumerate(backends):
            cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
            thread = threading.Thread(
                target=compile_then_check_exp,
                args=(
                    foo,
                    (x,),
                    cnt,
                    eager_result,
                    exp_frame_count,
                ),
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        self.assertEqual(len(thread_success), len(threads))

    def test_dynamo_min_operator_with_shape(self):
        # [FUNCTION DOC] Checks min of a shape element and an int under fullgraph.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x, a):
            # [FUNCTION DOC] HELPER: Returns min of first dim and a.
            return min(x.shape[0], a)

        result = f(torch.ones(6), 3)
        self.assertEqual(result, 3)

    def test_onnx_shape_as_tensor(self):
        # [FUNCTION DOC] Checks shape-as-tensor and its onnx variant under fullgraph on different
        # input ranks.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns 1 plus the first element of the shape tensor.
            return 1 + torch._shape_as_tensor(x)[0]

        gm, _ = torch._dynamo.export(f)(torch.ones(6))

        input_one_dim = torch.ones(6)
        input_two_dims = torch.ones(7, 4)
        self.assertEqual(f(input_one_dim), 7)
        self.assertEqual(f(input_two_dims), 8)
        self.assertEqual(f(input_two_dims), 8)

        @torch.compile(backend="eager", fullgraph=True)
        def f_onnx(x):
            # [FUNCTION DOC] HELPER: Returns 1 plus the first element of the onnx shape tensor.
            return 1 + torch.onnx.operators.shape_as_tensor(x)[0]

        self.assertEqual(f_onnx(input_one_dim), 7)
        self.assertEqual(f_onnx(input_two_dims), 8)
        self.assertEqual(f_onnx(input_two_dims), 8)

    def test_cond(self):
        # [FUNCTION DOC] Checks cond with sine and cosine branches for both predicate values.
        from functorch.experimental.control_flow import cond

        def true_fn(x):
            # [FUNCTION DOC] HELPER: Returns sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: Returns cosine of x.
            return x.cos()

        def f(pred, x):
            # [FUNCTION DOC] HELPER: Applies cond.
            return cond(pred, true_fn, false_fn, [x])

        opt_fn = torch.compile(f, backend="eager")
        a = opt_fn(torch.tensor(False), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.cos(torch.tensor([0.25, 0.25])), a))
        b = opt_fn(torch.tensor(True), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), b))

    def test_cond_with_quantization(self):
        # [FUNCTION DOC] Checks cond whose branches use a quantization-aware-trained module and
        # the plain module.
        from functorch.experimental.control_flow import cond

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer and its quantization
            # prepared version.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds linear and quantized models.
                super().__init__()
                example_inputs = (torch.randn(5, 5),)
                self.model = torch.nn.Linear(5, 5)
                self.quantized_model = prepare_qat_fx(
                    self.model, qconfig_dict, example_inputs=example_inputs
                )

            def forward(self, pred, x):
                # [FUNCTION DOC] HELPER: Applies cond with branches using the quantized or
                # plain model.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: Returns sine of x plus quantized model output.
                    return x.sin() + self.quantized_model(x)

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: Returns cosine of x plus plain model output.
                    return x.cos() + self.model(x)

                return cond(pred, true_fn, false_fn, [x])

        module = MyModule()
        opt_m = torch.compile(module, backend="eager", fullgraph=True)
        x = torch.rand((5, 5))
        pred = torch.tensor(True)
        self.assertTrue(same(module(pred, x), opt_m(pred, x)))
        pred = torch.tensor(False)
        self.assertTrue(same(module(pred, x), opt_m(pred, x)))

    def test_map_with_quantization(self):
        # [FUNCTION DOC] Checks map over rows with a body using a quantization-aware module.
        from functorch.experimental.control_flow import map

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with linear and quantized versions.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds linear and quantized models.
                super().__init__()
                example_inputs = (torch.randn(5, 5),)
                self.model = torch.nn.Linear(5, 5)
                self.quantized_model = prepare_qat_fx(
                    self.model, qconfig_dict, example_inputs=example_inputs
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Maps body over x.
                def body(x):
                    # [FUNCTION DOC] HELPER: Returns sine of x plus quantized model output.
                    return x.sin() + self.quantized_model(x)

                return map(body, x)

        module = MyModule()
        opt_m = torch.compile(module, backend="eager", fullgraph=True)
        x = torch.rand((5, 5))
        self.assertTrue(same(module(x), opt_m(x)))

    def test_cond_side_effects(self):
        # [FUNCTION DOC] Checks a nonlocal variable set before cond is read by the branches;
        # result 1.25.
        from functorch.experimental.control_flow import cond

        c = 0

        def true_fn(x):
            # [FUNCTION DOC] HELPER: Returns x minus c.
            return x - c

        def false_fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus c.
            return x + c

        def f(pred, x):
            # [FUNCTION DOC] HELPER: Sets c to 1 then applies cond.
            nonlocal c
            c = 1
            return cond(pred, true_fn, false_fn, [x])

        opt_fn = torch.compile(f, backend="eager")
        c = 0
        a = opt_fn(torch.tensor(False), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.tensor([1.25, 1.25]), a))

    def test_map_side_effects(self):
        # [FUNCTION DOC] Checks a side effect on a module attribute inside a map body raises
        # UncapturedHigherOrderOpError mentioning map_impl.
        from functorch.experimental.control_flow import map

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a tensor attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets w to tensor 1.
                super().__init__()
                self.w = torch.tensor(1)

            def forward(self, xs):
                # [FUNCTION DOC] HELPER: Maps body over xs.
                def body(x):
                    # [FUNCTION DOC] HELPER: Increments w in place and returns x.
                    self.w += 1
                    return x

                return map(body, xs)

        mod = Module()

        error_message = r"Higher Order Operator: torch\.ops\.higher_order\.map_impl"

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError, error_message
        ):
            opt_fn = torch.compile(mod, backend="eager", fullgraph=True)
            opt_fn(torch.randn(3, 2))

    def test_cond_nested(self):
        # [FUNCTION DOC] Checks nested cond over all predicate combinations with expected values;
        # asserts frame count.
        # KNOWN ISSUE: the final check uses assertTrue so it never compares counts.
        from functorch.experimental.control_flow import cond

        def true_fn_nested(x):
            # [FUNCTION DOC] HELPER: Returns x times 10.
            return x * 10

        def false_fn_nested(x):
            # [FUNCTION DOC] HELPER: Returns negative x.
            return x * -1

        def true_fn(pred2, x):
            # [FUNCTION DOC] HELPER: Returns sine of x.
            return x.sin()

        def false_fn(pred2, x):
            # [FUNCTION DOC] HELPER: Adds x and a nested cond.
            return x + cond(pred2, true_fn_nested, false_fn_nested, [x])

        def f(pred, pred2, x):
            # [FUNCTION DOC] HELPER: Outer cond.
            return cond(pred, true_fn, false_fn, [pred2, x])

        cc = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=cc)
        true_true_sin = opt_fn(
            torch.tensor(True), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_true_sin))

        true_false_sin = opt_fn(
            torch.tensor(True), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_false_sin))

        false_true_sum_mult = opt_fn(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([2.75, 2.75]), false_true_sum_mult)
        )  # * 10 then add x

        false_false_sum_neg = opt_fn(
            torch.tensor(False), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([0.0, 0.0]), false_false_sum_neg)
        )  # * -1 then add x
        self.assertTrue(cc.frame_count, 2)

    def test_cond_export(self):
        # [FUNCTION DOC] Checks nested cond is exported to a graph and gives correct values on
        # all four predicate combinations.
        from functorch.experimental.control_flow import cond

        def true_fn_nested(x):
            # [FUNCTION DOC] HELPER: Returns x times 10.
            return x * 10

        def false_fn_nested(x):
            # [FUNCTION DOC] HELPER: Returns negative x.
            return x * -1

        def true_fn(pred2, x):
            # [FUNCTION DOC] HELPER: Returns sine of x.
            return x.sin()

        def false_fn(pred2, x):
            # [FUNCTION DOC] HELPER: Adds x and a nested cond.
            return x + cond(pred2, true_fn_nested, false_fn_nested, [x])

        def f(pred, pred2, x):
            # [FUNCTION DOC] HELPER: Outer cond.
            return cond(pred, true_fn, false_fn, [pred2, x])

        graph, guard = torch._dynamo.export(f)(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        true_true_sin = graph(
            torch.tensor(True), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_true_sin))

        true_false_sin = graph(
            torch.tensor(True), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_false_sin))

        false_true_sum_mult = graph(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([2.75, 2.75]), false_true_sum_mult)
        )  # * 10 then add x

        false_false_sum_neg = graph(
            torch.tensor(False), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([0.0, 0.0]), false_false_sum_neg)
        )  # * -1 then add x

    def test_cond_export_single_arg(self):
        # [FUNCTION DOC] Checks exporting a cond and running the exported graph with different
        # shaped inputs.
        from functorch.experimental.control_flow import cond

        def true_fn(x):
            # [FUNCTION DOC] HELPER: Returns clone of x.
            return x.clone()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: Returns sine of x.
            return x.sin()

        def f(pred, x):
            # [FUNCTION DOC] HELPER: Applies cond.
            return cond(pred, true_fn, false_fn, [x])

        graph, guard = torch._dynamo.export(f)(
            torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        true_mirror = graph(torch.tensor(True), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.tensor([0.25, 0.25]), true_mirror))
        true_mirror_2 = graph(torch.tensor(True), torch.tensor([0.33, 0.33, 0.33]))
        self.assertTrue(same(torch.tensor([0.33, 0.33, 0.33]), true_mirror_2))

        false_sin = graph(torch.tensor(False), torch.tensor([0.5, 0.5]))
        self.assertTrue(same(torch.sin(torch.tensor([0.5, 0.5])), false_sin))

    def test_duplicate_graph_break_log(self):
        # [FUNCTION DOC] Checks the graph break log is emitted once per break with default
        # verbosity, and multiple times when verbose is on; resets logging at the end.
        torch._logging.set_logs(graph_breaks=True)

        @torch.compile(backend="eager")
        def f1(a, b):
            # [FUNCTION DOC] HELPER: Compiled function calling f2 inside try/finally.
            try:
                f2(a, b)
            finally:
                pass

        def f2(a, b):
            # [FUNCTION DOC] HELPER: Adds, prints (graph break), returns a sum.
            c = a + b
            print("break")
            return a + b + c

        @torch.compile(backend="eager")
        def g1(a, b):
            # [FUNCTION DOC] HELPER: Compiled function calling g2 inside try/finally.
            try:
                g2(a, b)
            finally:
                pass

        def g2(a, b):
            # [FUNCTION DOC] HELPER: Adds, prints (graph break), returns a sum.
            c = a + b
            print("break")
            return a + b + c

        def count_graph_break_msgs(msgs):
            # [FUNCTION DOC] HELPER: Counts messages containing the user-code graph break text.
            return sum("Graph break in user code" in msg for msg in msgs)

        with (
            self.assertLogs(logger="torch._dynamo", level=logging.DEBUG) as log,
            torch._dynamo.config.patch(verbose=True),
        ):
            f1(torch.randn(10), torch.randn(10))
            self.assertGreater(count_graph_break_msgs(log.output), 1)

        with (
            self.assertLogs(logger="torch._dynamo", level=logging.DEBUG) as log,
            torch._dynamo.config.patch(verbose=False),
        ):
            g1(torch.randn(10), torch.randn(10))
            self.assertEqual(count_graph_break_msgs(log.output), 1)

        # reset logging state
        torch._logging.set_logs()

    def test_inplace_param_update(self):
        # [FUNCTION DOC] Checks toggling grad mode around an in-place parameter update compiles
        # in one frame with 3 ops.
        def fn(param, y):
            # [FUNCTION DOC] HELPER: Toggles grad mode several times, adds y to param in place,
            # restores the previous grad mode.
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(False)
                torch.set_grad_enabled(True)
                torch.set_grad_enabled(False)
                param.add_(y)
            finally:
                torch.set_grad_enabled(prev_grad)

        y = torch.randn(4)
        x = torch.nn.Parameter(torch.randn(4))
        fn(x, y)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, y)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_generate_tensor_from_list_of_numpy_primitive_type(self):
        # [FUNCTION DOC] Checks creating a long tensor from a list of numpy int64 scalars
        # under fullgraph.
        # Test sth like torch.LongTensor(list(np.int64, np.int64, ...))
        def fn():
            # [FUNCTION DOC] HELPER: Builds numpy array, picks three elements, makes a
            # long tensor.
            x = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
            y = [x[0], x[2], x[4]]
            return torch.LongTensor(y)

        ref = fn()
        res = torch.compile(fullgraph=True, backend="eager")(fn)()
        self.assertEqual(ref, res)

    def test_object_classmethod(self):
        # [FUNCTION DOC] Checks calling a class method on an instance under fullgraph.
        class C:
            # [CLASS DOC] HELPER: Class with class method doubling x.
            @classmethod
            def fn(cls, x):
                # [FUNCTION DOC] HELPER: Returns x plus x.
                return x + x

        @torch.compile(backend="eager", fullgraph=True)
        def f():
            # [FUNCTION DOC] HELPER: Calls the class method on a ones tensor.
            return C().fn(torch.ones(2, 3))

        self.assertTrue(torch.allclose(f(), torch.tensor([2.0])))

    def test_object_staticmethod(self):
        # [FUNCTION DOC] Checks calling a static method on an instance under fullgraph.
        class C:
            # [CLASS DOC] HELPER: Class with static method doubling x.
            @staticmethod
            def fn(x):
                # [FUNCTION DOC] HELPER: Returns x plus x.
                return x + x

        @torch.compile(backend="eager", fullgraph=True)
        def f():
            # [FUNCTION DOC] HELPER: Calls the static method on a ones tensor.
            return C().fn(torch.ones(2, 3))

        self.assertTrue(torch.allclose(f(), torch.tensor([2.0])))

    def test_user_function_variable_supports_type_abcmeta_argument(self):
        # [FUNCTION DOC] Checks a default argument holding a tuple of ABC-metaclass classes works
        # as a function argument.
        class Foo(metaclass=abc.ABCMeta):
            # [CLASS DOC] HELPER: Abstract base using ABCMeta with an abstract class method.
            @abc.abstractclassmethod
            def read(self):  # noqa: B027
                # [FUNCTION DOC] HELPER: Abstract read method; does nothing.
                pass

        class Bar(Foo):
            # [CLASS DOC] HELPER: Concrete subclass.
            def read(self):
                # [FUNCTION DOC] HELPER: Returns the greeting string.
                return "Hello World!"

        class Baz:
            # [CLASS DOC] HELPER: Plain class.
            pass

        def gn(x, tys=(Bar, Baz)):
            # [FUNCTION DOC] HELPER: Returns x minus 1 if Bar is in tys else x plus 1.
            if Bar in tys:
                return x - 1
            else:
                return x + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls gn.
            return gn(x)

        x = torch.randn(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_user_function_variable_supports_function_argument(self):
        # [FUNCTION DOC] Checks default arguments that are user functions, torch functions and
        # builtin operators.
        # Test user defined function default arguments can be:
        # 1, user defined functions (e.g, add1)
        # 2, torch functions (e.g, torch.sin)
        # 3, python builtin functions (e.g, operator.neg)
        def add1(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        def gn(x, f1=add1, f2=torch.sin, f3=operator.neg):
            # [FUNCTION DOC] HELPER: Composes the three function arguments.
            return f3(f2(f1(x)))

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls gn.
            return gn(x)

        x = torch.randn(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_typing_variable_isinstance(self):
        # [FUNCTION DOC] Checks isinstance against a typing mapping class for a dict.
        def fn(x, m):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if m is a mapping, else x minus 1.
            if isinstance(m, typing.Mapping):
                return x + 1
            else:
                return x - 1

        x = torch.randn(2, 3)
        m = {"x": torch.randn(3)}
        ref = fn(x, m)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, m)
        self.assertTrue(torch.allclose(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_repro_graph_breaks_in__get_item_by_idx(self):
        # [FUNCTION DOC] With module guarding on, checks export of a module that indexes into a
        # sequential works.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a sequential of two linear layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the sequential.
                super().__init__()
                self.mod = torch.nn.Sequential(
                    torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first layer.
                return self.mod[0](x)

        m = Mod()
        graph, _ = torch._dynamo.export(m)(torch.randn(3, 3))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_nn_sequential_invocation(self):
        # [FUNCTION DOC] With module guarding on and a frozen RNG, checks export of a module
        # slicing a sequential (all but last) matches eager.
        with freeze_rng_state():

            class TestModel(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a sequential of four linear layers.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the four layers.
                    super().__init__()
                    self.linears = torch.nn.Sequential(
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Slices off the last layer and applies the rest.
                    all_but_last = self.linears[:-1]
                    return all_but_last(x)

            m = TestModel()
            x = torch.rand((2, 2))
            real = m(x)
            graph, _ = torch._dynamo.export(m)(x)
            dynamo_result = graph(x)
            self.assertTrue(same(real, dynamo_result))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_nn_sequential_invocation_reposition_indices(self):
        # [FUNCTION DOC] Same as above with a middle slice of layers 1 to 3.
        with freeze_rng_state():

            class TestModel(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a sequential of four linear layers.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the four layers.
                    super().__init__()
                    self.linears = torch.nn.Sequential(
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies layers 1 and 2.
                    all_but_last = self.linears[1:3]
                    return all_but_last(x)

            m = TestModel()
            x = torch.rand((2, 2))
            real = m(x)
            graph, _ = torch._dynamo.export(m)(x)
            dynamo_result = graph(x)
            self.assertTrue(same(real, dynamo_result))

    def test_error_on_nested_fx_trace(self):
        # [FUNCTION DOC] Checks FX symbolic tracing of a compiled function raises RuntimeError
        # saying FX use was detected.
        input = torch.rand(2, 3)

        def f(x):
            # [FUNCTION DOC] HELPER: Evaluates x plus x and returns nothing.
            x + x

        real = f(input)

        optimized = torch.compile(f, backend="eager")
        self.assertTrue(same(optimized(input), real))

        with self.assertRaisesRegex(RuntimeError, "Detected that you are using FX"):
            gm = torch.fx.symbolic_trace(optimized)

    @patch.object(torch._dynamo.config, "error_on_nested_fx_trace", False)
    def test_no_error_on_nested_fx_trace(self):
        # [FUNCTION DOC] With nested fx trace error config disabled, checks symbolic tracing of a
        # compiled function works and matches.
        input = torch.rand(2, 3)

        def f(x):
            # [FUNCTION DOC] HELPER: Evaluates x plus x and returns nothing.
            x + x

        real = f(input)

        optimized = torch.compile(f, backend="eager")
        self.assertTrue(same(optimized(input), real))

        # should not error
        gm = torch.fx.symbolic_trace(optimized)
        self.assertTrue(same(gm(input), real))

    def test_not_dynamic_scope(self):
        # [FUNCTION DOC] Checks lexical scoping: a closure returned from an inner function sees
        # its own x value (2), not the caller's.
        def f(y):
            # [FUNCTION DOC] HELPER: Defines g and returns y plus the result of calling the
            # closure from g.
            x = 1

            def g():
                # [FUNCTION DOC] HELPER: Returns a lambda capturing local 2.
                x = 2
                return lambda: x

            return y + g()()

        input = torch.zeros(1)
        real = f(input)
        optimized = torch.compile(f, backend="eager")
        opt = optimized(input)
        self.assertTrue(same(opt, real))

    def test_inference_mode(self):
        # [FUNCTION DOC] Checks a function decorated with inference mode compiled gives the same
        # result and leaves inputs equal.
        @torch.inference_mode()
        def func(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus 1.0 plus y.
            return x.add(1.0) + y

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=True)
        ref = func(x, y)
        opt_func = torch.compile(func, backend="eager")

        x1 = torch.ones(4, requires_grad=True)
        res = opt_func(x1, y)
        self.assertTrue(same(ref, res))
        self.assertTrue(same(x, x1))

    def test_inference_mode_param(self):
        # [FUNCTION DOC] Checks creating a parameter from a tensor in inference mode under
        # fullgraph matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a no-grad parameter of x and returns the product.
            p = torch.nn.Parameter(x, requires_grad=False)
            return x * p

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        with torch.inference_mode():
            x = torch.rand(4)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_if_cond_nn_mod1(self):
        # [FUNCTION DOC] Checks a module with optional submodule used as a truthiness condition,
        # with and without relu.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with optional relu.
            def __init__(self, output_relu=True):
                # [FUNCTION DOC] HELPER: Creates relu or none.
                super().__init__()
                self.relu = torch.nn.ReLU() if output_relu else None

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies sine, then relu if present.
                x = torch.sin(x)
                if self.relu:
                    x = self.relu(x)
                return x

        model = MockModule()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)

        x = torch.rand(4)
        ref = model(x)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

        model = MockModule(output_relu=False)
        opt_model = torch.compile(model, backend="eager", fullgraph=True)

        x = torch.rand(4)
        ref = model(x)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

    def test_if_cond_nn_mod2(self):
        # [FUNCTION DOC] Checks truthiness of an empty Sequential used as a branch condition
        # matches eager.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an empty sequential.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the empty sequential.
                super().__init__()
                self.layer = torch.nn.Sequential()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus 1 if the sequential is truthy (non-
                # empty), else x minus 1.
                if self.layer:
                    return x + 1
                else:
                    return x - 1

        model = MockModule()
        x = torch.rand(4)
        ref = model(x)
        opt_model = torch.compile(backend="eager")(model)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

    def test_if_cond_nn_mod3(self):
        # [FUNCTION DOC] Checks truthiness of an empty ModuleList created in compiled code.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if the new empty ModuleList is truthy, else
            # x minus 1.
            if torch.nn.ModuleList():
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_if_cond_user_defined_object(self):
        # [FUNCTION DOC] Checks truthiness of user objects: no bool method, bool method returning
        # a bool, and a bool attribute set on the instance; expects four frames over the sequence
        # of objects.
        # obj.__bool__ is not existed
        class A:  # noqa: B903
            # [CLASS DOC] HELPER: Plain object without a truth method.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

        # obj.__bool__ is function and returns bool type
        class B:
            # [CLASS DOC] HELPER: Object whose truth method returns whether x is positive.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

            def __bool__(self):
                # [FUNCTION DOC] HELPER: Returns x greater than zero.
                return self.x > 0

        # obj.__bool__ is non-function
        class C:
            # [CLASS DOC] HELPER: Object with a non-callable instance attribute named like the
            # truth method.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x and sets the attribute to False.
                self.x = x
                self.__bool__ = False

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if obj is falsy, else x minus 1.
            if not obj:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        obj1 = A(0.5)
        obj2 = B(0.5)
        obj3 = B(-0.5)
        obj4 = C(0.5)
        for obj in [obj1, obj2, obj3, obj4, obj3, obj2]:
            ref = fn(x, obj)
            res = opt_fn(x, obj)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 4)

    def test_if_cond_user_defined_object2(self):
        # [FUNCTION DOC] Checks that a truth method returning a non-bool raises TypeError with
        # the CPython-version-specific message.
        # obj.__bool__ is function and returns non-bool type
        class MyObj:
            # [CLASS DOC] HELPER: Object whose truth method sets x to 1.2 and returns it.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

            def __bool__(self):
                # [FUNCTION DOC] HELPER: Sets x to 1.2 and returns it (invalid return type).
                self.x = 1.2
                return self.x

        def fn(a, obj):
            # [FUNCTION DOC] HELPER: Branches on truthiness of obj and combines with obj.x.
            if not obj:
                return a + obj.x
            else:
                return a - obj.x

        x = torch.rand(4)
        obj = MyObj(0.5)
        opt_fn = torch.compile(fn, backend="eager")
        try:
            opt_fn(x, obj)
            self.assertFalse(True)
        except TypeError as e:
            if sys.version_info >= (3, 15):
                msg = "__bool__() must return a bool, not float"
            else:
                msg = "__bool__ should return bool, returned float"
            self.assertIn(msg, str(e))

    def test_unpack_tensor_shape_mismatch(self):
        # [FUNCTION DOC] Checks error cases when unpacking tensors: scalar tensor raises
        # AssertionError, a size-1 vector unpacked into two raises ValueError, and unpacking one
        # value from a size-1 vector works.
        @torch.compile(backend="eager")
        def f1(x):
            # [FUNCTION DOC] HELPER: Unpacks x into two values and returns the sine of the sum.
            a, b = x
            return torch.sin(a + b)

        x = torch.tensor(2.0)
        with self.assertRaisesRegex(AssertionError, "Can't unpack scalar tensors"):
            f1(x)

        x = torch.tensor([2.0])
        with self.assertRaisesRegex(
            ValueError, r"not enough values to unpack \(expected 2, got 1\)"
        ):
            f1(x)

        @torch.compile(backend="eager")
        def f2(x):
            # [FUNCTION DOC] HELPER: Unpacks x into one value and returns sine of it plus 1.
            (a,) = x
            return torch.sin(a + 1)

        x = torch.tensor(2.0)
        with self.assertRaisesRegex(AssertionError, "Can't unpack scalar tensors"):
            f2(x)

        x = torch.tensor([2.0])
        self.assertTrue(same(f2(x), torch.sin(x[0] + 1)))

    def test_if_cond_user_defined_object3(self):
        # [FUNCTION DOC] Checks truthiness through the length method when no truth method exists,
        # and that the truth method takes precedence.
        # obj.__bool__ is not existed, but obj.__len__ exists
        class A:  # noqa: B903
            # [CLASS DOC] HELPER: Object with only a length method.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns length of x.
                return len(self.x)

        # obj.__bool__ takes precedence over obj.__len__
        class B:
            # [CLASS DOC] HELPER: Object whose truth method returns False and which also has a
            # length method.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

            def __bool__(self):
                # [FUNCTION DOC] HELPER: Returns False.
                return False

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns length of x.
                return len(self.x)

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if obj is falsy, else x minus 1.
            if not obj:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        obj1 = A([1, 2, 3])
        obj2 = A([])
        obj3 = B([1, 2, 3])
        obj4 = B([])
        for obj in [obj1, obj2, obj3, obj4]:
            ref = fn(x, obj)
            res = opt_fn(x, obj)
            self.assertTrue(same(ref, res))

    def test_class_has_instancecheck_method(self):
        # [FUNCTION DOC] Checks isinstance with a metaclass defining an always-true
        # instance check.
        class A:
            # [CLASS DOC] HELPER: Plain class.
            pass

        class ExampleMeta(type):
            # [CLASS DOC] HELPER: Metaclass.
            def __instancecheck__(cls, instance):
                # [FUNCTION DOC] HELPER: Always reports True.
                return True

        class B(metaclass=ExampleMeta):
            # [CLASS DOC] HELPER: Class with the metaclass.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if isinstance of B else x minus 1.
            if isinstance(obj, B):
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        obj = A()
        ref = fn(x, obj)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, obj)
        self.assertTrue(same(ref, res))

    def test_tensor_isinstance_custom_instancecheck_graph_break(self):
        # [FUNCTION DOC] Checks isinstance with metaclasses having custom instance or subclass
        # checks: compiled call works non-fullgraph and with fullgraph raises Unsupported
        # mentioning custom type check; also for typing unions and pipe unions (version-
        # dependent) and tensor subclasses with custom instance check.
        shape_context = threading.local()

        class TensorWithShapeBaseMeta(type):
            # [CLASS DOC] HELPER: Metaclass whose instance check compares a tensor shape to a
            # thread-local or class expected shape.
            def __instancecheck__(cls, instance):
                # [FUNCTION DOC] HELPER: Returns whether the instance is a tensor with the
                # expected shape.
                try:
                    expected_shape = shape_context.value
                except AttributeError:
                    expected_shape = cls.expected_shape
                return (
                    isinstance(instance, torch.Tensor)
                    and tuple(instance.shape) == expected_shape
                )

        class TensorWithShapeMeta(TensorWithShapeBaseMeta):
            # [CLASS DOC] HELPER: Empty subclass of the base metaclass.
            pass

        class TensorWithShape(metaclass=TensorWithShapeMeta):
            # [CLASS DOC] HELPER: Class using the metaclass with expected shape (2, 3).
            expected_shape = (2, 3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if isinstance of the class else x minus 1.
            if isinstance(x, TensorWithShape):
                return x + 1
            return x - 1

        cnt = CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(2, 3)
        for expected_shape in [(2, 3), (2, 4)]:
            shape_context.value = expected_shape
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))

        torch._dynamo.reset()
        with self.assertRaisesRegex(Unsupported, "custom type check"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

        def fn_union(x):
            # [FUNCTION DOC] HELPER: Same check using a typing union with int.
            if isinstance(x, typing.Union[TensorWithShape, int]):
                return x + 1
            return x - 1

        torch._dynamo.reset()
        opt_fn_union = torch.compile(fn_union, backend="eager")
        for expected_shape in [(2, 3), (2, 4)]:
            shape_context.value = expected_shape
            self.assertTrue(same(fn_union(x), opt_fn_union(x)))

        torch._dynamo.reset()
        if sys.version_info >= (3, 12):
            with self.assertRaisesRegex(Unsupported, "custom type check"):
                torch.compile(fn_union, backend="eager", fullgraph=True)(x)
        else:
            self.assertTrue(
                same(
                    fn_union(x),
                    torch.compile(fn_union, backend="eager", fullgraph=True)(x),
                )
            )

        def fn_union_type(x):
            # [FUNCTION DOC] HELPER: Same check using a pipe union with int.
            if isinstance(x, TensorWithShape | int):
                return x + 1
            return x - 1

        torch._dynamo.reset()
        with self.assertRaisesRegex(Unsupported, "custom type check"):
            torch.compile(fn_union_type, backend="eager", fullgraph=True)(x)

        class TensorSubclassCheckMeta(type):
            # [CLASS DOC] HELPER: Metaclass with a custom subclass check driven by a thread-
            # local flag.
            def __subclasscheck__(cls, subclass):
                # [FUNCTION DOC] HELPER: Returns the thread-local allow flag, default False.
                return getattr(shape_context, "allow_subclass", False)

        class TensorSubclassCheck(metaclass=TensorSubclassCheckMeta):
            # [CLASS DOC] HELPER: Class using the metaclass.
            pass

        def fn_subclasscheck(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if isinstance of the class else x minus 1.
            if isinstance(x, TensorSubclassCheck):
                return x + 1
            return x - 1

        shape_context.allow_subclass = True
        self.assertTrue(issubclass(torch.Tensor, TensorSubclassCheck))
        self.assertFalse(isinstance(x, TensorSubclassCheck))
        torch._dynamo.reset()
        self.assertTrue(
            same(
                fn_subclasscheck(x),
                torch.compile(fn_subclasscheck, backend="eager")(x),
            )
        )

        torch._dynamo.reset()
        with self.assertRaisesRegex(Unsupported, "custom type check"):
            torch.compile(fn_subclasscheck, backend="eager", fullgraph=True)(x)

        def fn_union_subclasscheck(x):
            # [FUNCTION DOC] HELPER: Same check using a typing union with int.
            if isinstance(x, typing.Union[TensorSubclassCheck, int]):
                return x + 1
            return x - 1

        torch._dynamo.reset()
        with self.assertRaisesRegex(Unsupported, "custom type check"):
            torch.compile(fn_union_subclasscheck, backend="eager", fullgraph=True)(x)

        class TensorSubclassMeta(type(torch.Tensor)):
            # [CLASS DOC] HELPER: Tensor metaclass with instance check driven by a thread-
            # local flag.
            def __instancecheck__(cls, instance):
                # [FUNCTION DOC] HELPER: Returns the thread-local flag, default False.
                return getattr(shape_context, "is_tensor_subclass", False)

        class TensorSubclassWithInstanceCheck(
            torch.Tensor, metaclass=TensorSubclassMeta
        ):
            # [CLASS DOC] HELPER: Tensor subclass using the custom metaclass.
            pass

        def fn_tensor_subclass(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if isinstance of the subclass else x
            # minus 1.
            if isinstance(x, TensorSubclassWithInstanceCheck):
                return x + 1
            return x - 1

        torch._dynamo.reset()
        with self.assertRaisesRegex(Unsupported, "custom type check"):
            torch.compile(fn_tensor_subclass, backend="eager", fullgraph=True)(x)

    def test_tensor_isinstance_supported_tensor_types_no_graph_break(self):
        # [FUNCTION DOC] Checks isinstance with a float tensor type, parameter subclass, buffer
        # subclass, sequence ABC and ABC base compiles under fullgraph and matches eager.
        class ParameterSubclass(torch.nn.Parameter):
            # [CLASS DOC] HELPER: Empty subclass of Parameter.
            pass

        class BufferSubclass(torch.nn.Buffer):
            # [CLASS DOC] HELPER: Empty subclass of Buffer.
            pass

        def fn(x, p, b):
            # [FUNCTION DOC] HELPER: Returns a tuple of seven isinstance results.
            return (
                isinstance(x, torch.FloatTensor),
                isinstance(p, torch.nn.Parameter),
                isinstance(p, ParameterSubclass),
                isinstance(b, torch.nn.Buffer),
                isinstance(b, BufferSubclass),
                isinstance(x, collections.abc.Sequence),
                isinstance(x, abc.ABC),
            )

        x = torch.randn(2, 3)
        p = ParameterSubclass(torch.randn(2, 3))
        b = BufferSubclass(torch.randn(2, 3))
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, p, b), fn(x, p, b))

    def test_custom_instancecheck_does_not_cause_extra_init(self):
        # [FUNCTION DOC] Checks that when new returns an object that is not an instance of the
        # class (as the type), init is not called even if the metaclass instance check says yes.
        # When __new__ returns an object whose type is not a subclass of cls,
        # CPython's type.__call__ skips __init__. The polyfill
        # instantiate_user_defined_class_object must match this behavior even
        # when the metaclass defines a custom __instancecheck__ that would
        # return True for isinstance().
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass with instance check requiring Base instance with
            # matching tag.
            def __instancecheck__(cls, instance):
                # [FUNCTION DOC] HELPER: Returns whether instance is a Base with tag equal to the
                # class tag.
                return isinstance(instance, Base) and instance.tag == cls._tag

        class Base:
            # [CLASS DOC] HELPER: Base with a tag.
            def __init__(self, tag="default"):
                # [FUNCTION DOC] HELPER: Stores tag.
                self.tag = tag

        class Child(Base, metaclass=Meta):
            # [CLASS DOC] HELPER: Class with metaclass whose new returns a Base object
            # tagged child.
            _tag = "child"

            def __new__(cls):
                # [FUNCTION DOC] HELPER: Returns a Base with tag child.
                # Returns a Base (not a Child), like ByteStorage.__new__
                return Base(tag="child")

        def fn():
            # [FUNCTION DOC] HELPER: Creates Child and returns its tag.
            obj = Child()
            return obj.tag

        ref = fn()
        self.assertEqual(ref, "child")

        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn()
        self.assertEqual(res, "child")

    def test_custom_instancecheck_init_not_called(self):
        # [FUNCTION DOC] Checks that init is not called when new returns an unrelated object,
        # even if the metaclass instance check is always true.
        class AlwaysTrueMeta(type):
            # [CLASS DOC] HELPER: Metaclass with always-true instance check.
            def __instancecheck__(cls, instance):
                # [FUNCTION DOC] HELPER: Returns True.
                return True

        class Child(metaclass=AlwaysTrueMeta):
            # [CLASS DOC] HELPER: Class whose new returns a plain object; its init
            # raises AssertionError.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Returns a plain object.
                return object()

            def __init__(self):
                # [FUNCTION DOC] HELPER: Raises AssertionError (must not be called).
                raise AssertionError("should NOT be called")

        def fn():
            # [FUNCTION DOC] HELPER: Creates Child.
            return Child()

        ref = fn()
        self.assertIsInstance(ref, object)

        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn()
        self.assertIsInstance(res, object)

    def test_variable_tracker_recursively_contains(self):
        # [FUNCTION DOC] Checks mutation of a nested list built by repetition (aliased inner
        # lists) updates tracked containment correctly under fullgraph.
        # VariableTracker.recursively_contains should be updated correctly when mutation happens
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds aliased lists, fills the first with chained
            # additions, returns the last element.
            data = [[None] * 3] * 3
            for i in range(3):
                if i == 0:
                    data[0][i] = x
                else:
                    data[0][i] = data[0][i - 1] + 1
            return data[0][-1]

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_disable_flag(self):
        # [FUNCTION DOC] Checks the disable environment variable.
        # KNOWN ISSUE: the compile call has no function argument, so nothing is compiled or
        # called; the frame count check passes trivially.
        cnt = torch._dynamo.testing.CompileCounter()

        with patch.dict(os.environ, {"TORCH_COMPILE_DISABLE": "1"}):

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Defined but never called.
                x = x + 1
                y = y + 1

            opt_fn = torch.compile(backend=cnt)

        self.assertEqual(cnt.frame_count, 0)

    def test_is_compiling(self):
        # [FUNCTION DOC] Checks four is-compiling helper functions return ones when compiled and
        # zeros in eager.
        def f1():
            # [FUNCTION DOC] HELPER: Ones if the dynamo is_compiling is true else zeros.
            if torch._dynamo.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        def f2():
            # [FUNCTION DOC] HELPER: Ones if the utils is_compiling is true else zeros.
            if torch._utils.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        def f3():
            # [FUNCTION DOC] HELPER: Ones if the compiler is_compiling is true else zeros.
            if torch.compiler.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        def f4():
            # [FUNCTION DOC] HELPER: Ones if is_dynamo_compiling is true else zeros.
            if torch.compiler.is_dynamo_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        for f in [f1, f2, f3, f4]:
            opt_f = torch.compile(f, backend="eager")

            self.assertEqual(f(), torch.zeros(2, 2))
            self.assertEqual(opt_f(), torch.ones(2, 2))

    def test_is_exporting_not_true_during_compile(self):
        # [FUNCTION DOC] Checks the exporting flag is false during normal compile.
        def f():
            # [FUNCTION DOC] HELPER: Ones if exporting else zeros.
            if torch.compiler.is_exporting():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        opt_f = torch.compile(f, backend="eager")

        self.assertEqual(f(), torch.zeros(2, 2))
        self.assertEqual(opt_f(), torch.zeros(2, 2))

    def test_torch_generator_set_state(self):
        # [FUNCTION DOC] Checks default generator state save and restore across a graph break
        # reproduces the same random numbers.
        def fn():
            # [FUNCTION DOC] HELPER: Saves state, draws x, graph breaks, restores state, draws y.
            default_state = torch.default_generator.get_state()
            x = torch.rand([2, 3])
            if default_state.dtype != "float32":
                x = x * 2
            torch._dynamo.graph_break()
            torch.default_generator.set_state(default_state)
            y = torch.rand([2, 3])
            return x, y

        opt_fn = torch.compile(fn, backend="eager")
        x, y = opt_fn()
        self.assertEqual(x, y * 2)

    def test_torch_distributions_lazy_property(self):
        # [FUNCTION DOC] Checks a distributions entropy computation with lazy property compiled
        # matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Entropy of a categorical distribution.
            return torch.distributions.Categorical(probs=x).entropy()

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.rand([4, 4])
        self.assertEqual(opt_fn(x), fn(x))

    def test_torch_distributions_gamma_dynamic(self):
        # [FUNCTION DOC] Checks sampling a gamma distribution with a dynamic sample count under
        # fullgraph matches eager for the same seed.
        def fn(n: int):
            # [FUNCTION DOC] HELPER: Samples n values from a gamma distribution.
            distribution = torch.distributions.Gamma(
                concentration=torch.tensor(2.0),
                rate=torch.tensor(1.0),
            )
            return distribution.sample((n,))

        opt_fn = torch.compile(fn, backend="eager", dynamic=True, fullgraph=True)

        torch.manual_seed(42)
        expected = fn(5)

        torch.manual_seed(42)
        result = opt_fn(5)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_python_type_name_mirrors_cpython_tp_name(self):
        # [FUNCTION DOC] Checks python_type_name on constants returns the CPython type names
        # (int, str, NoneType, ellipsis).
        """Test python_type_name() mirrors CPython's tp_name slot."""
        from torch._dynamo.variables.constant import ConstantVariable

        # python_type_name() should return CPython's tp_name string
        test_cases = [
            (42, "int"),
            ("hello", "str"),
            (None, "NoneType"),
            (..., "ellipsis"),
        ]

        for value, expected_tp_name in test_cases:
            vt = ConstantVariable(value)
            self.assertEqual(vt.python_type_name(), expected_tp_name)

    def test_guard_failure_fn(self):
        # [FUNCTION DOC] Checks the guard failure callback is called after a recompile caused by
        # changed tensor shape and int value (depending on config).
        def fn(x, y, k):
            # [FUNCTION DOC] HELPER: Returns x plus 1 times y plus 1 times k.
            x = x + 1
            y = y + 1
            return x * y * k

        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])

        guard_failure = None

        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Records the failure.
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        x2 = torch.tensor([0.5, 0.5, 1.0])
        y2 = torch.tensor([0.5, 0.5, 0.5])

        opt_fn(x, y, 3)
        opt_fn(x2, y2, 5)

        if (
            not torch._dynamo.config.specialize_int
            and not torch._dynamo.config.assume_static_by_default
        ):
            # we didn't actually test guard_failure_fn here but whatever,
            # nice to see no guard failure on the test
            self.assertTrue(guard_failure is None)
        else:
            self.assertTrue(guard_failure is not None)

    def test_guard_failure_fn_shape_control(self):
        # [FUNCTION DOC] Checks the guard failure message for a shape-controlled branch reports
        # the size mismatch at index 0 (expected 2, actual 5).
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Branches on shape sizes and returns a product, sum or -1.
            if x.shape[0] < 4:
                if y.shape[0] < 3:
                    return x * y
                else:
                    return x + y
            else:
                return -1

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])

        guard_failure = None

        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Records the failure.
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        x2 = torch.randn([5, 5])
        y2 = torch.randn([5, 5])

        opt_fn(x, y)
        opt_fn(x2, y2)

        self.assertTrue(guard_failure is not None)
        first_guard_failure = guard_failure[0].partition("\n")[0]
        self.assertIn(
            """tensor 'x' size mismatch at index 0. expected 2, actual 5""",
            first_guard_failure,
        )

    def test_guard_failure_fn2(self):
        # [FUNCTION DOC] Checks the guard failure message for changed size when shapes are static
        # by default.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus 1 times y plus 1.
            x = x + 1
            y = y + 1
            return x * y

        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])

        guard_failure = None

        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Records the failure.
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        x2 = torch.tensor([0.5, 0.5, 1.0])
        y2 = torch.tensor([0.5, 0.5, 0.5])

        opt_fn(x, y)
        opt_fn(x2, y2)

        if torch._dynamo.config.assume_static_by_default:
            self.assertIn(
                """tensor 'x' size mismatch at index 0. expected 2, actual 3""",
                guard_failure[0],
            )
        else:
            self.assertTrue(guard_failure is None)

    def test_guard_failure_fn_tensor_iter(self):
        # [FUNCTION DOC] Checks the size mismatch message when iterating a tensor with different
        # first dim.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1 in place to each row and returns the last.
            for y in x:
                y.add_(1.0)
            return y

        guard_failure = None

        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Records the failure.
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        args1 = torch.randn(10, 10)
        out = fn(args1)
        opt_out = opt_fn(args1)
        self.assertTrue(same(out, opt_out))

        args2 = torch.randn(9, 10)
        out = fn(args2)
        opt_out = opt_fn(args2)
        self.assertTrue(same(out, opt_out))

        # guard is expected for both static and dynamic shapes
        self.assertTrue(guard_failure is not None)
        self.assertIn(
            """size mismatch at index 0. expected 10, actual 9""",
            guard_failure[0],
        )

    def test_no_guard_for_unused_sym_node_fstring(self):
        # [FUNCTION DOC] Checks that an unused formatted string of a symbolic size creates no
        # guard: changing the size produces no guard failure.
        def fn(x):
            # [FUNCTION DOC] HELPER: Formats the first dim (unused) and returns sine.
            f"{x.shape[0]}"
            return x.sin()

        guard_failure = None

        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Records the failure.
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", guard_fail_fn=guard_failures, dynamic=True
        )(fn)
        args1 = torch.randn(10, 11)
        out = fn(args1)
        opt_out = opt_fn(args1)
        self.assertEqual(out, opt_out)

        # We change x.shape[0] to test whether it's guarded
        args2 = torch.randn(9, 11)
        out = fn(args2)
        opt_out = opt_fn(args2)
        self.assertEqual(out, opt_out)
        self.assertEqual(guard_failure, None)

    def test_no_guard_for_unused_input(self):
        # [FUNCTION DOC] Checks inputs unused by the function create no guards: different shapes
        # and a non-tensor reuse the single compiled frame.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, dynamic=False)
        opt_fn(torch.randn(3), torch.randn(3))
        opt_fn(torch.randn(3), torch.randn(5))
        opt_fn(torch.randn(3), 42)
        self.assertEqual(cnt.frame_count, 1)

    def test_no_guard_for_unused_scalar_input(self):
        # [FUNCTION DOC] Same with an unused scalar input (int, int, string); one frame.
        def fn(x, n):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, dynamic=False)
        opt_fn(torch.randn(3), 5)
        opt_fn(torch.randn(3), 99)
        opt_fn(torch.randn(3), "hello")
        self.assertEqual(cnt.frame_count, 1)

    def test_no_guard_for_unused_input_when_locals_called(self):
        # [FUNCTION DOC] Checks that when the locals mapping is requested, otherwise unused
        # inputs get guards again, so a changed y shape recompiles (two frames).
        # locals() is handled at trace time by _call_frame_locals_snapshot, which
        # adds pruned locals back from tx.f_locals and installs guards on them.
        # So y still causes a recompile when its shape changes.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns locals x plus 1.
            return locals()["x"] + 1

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, dynamic=False)
        opt_fn(torch.randn(3), torch.randn(3))
        opt_fn(torch.randn(3), torch.randn(5))
        self.assertEqual(cnt.frame_count, 2)

    def test_locals_dict_access_of_unused_param_works(self):
        # [FUNCTION DOC] Checks reading an otherwise-unused parameter via locals works.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns locals y plus x.
            return locals()["y"] + x

        opt_fn = torch.compile(fn, backend="eager")
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_locals_alias_access_of_unused_param_works(self):
        # [FUNCTION DOC] Checks reading an unused parameter through an alias of locals works.
        # locals() called via an alias (locals2 = locals in global scope) must
        # still expose pruned locals; _call_frame_locals_snapshot adds them back.
        locals2 = locals

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns locals2 y plus x.
            return locals2()["y"] + x

        opt_fn = torch.compile(fn, backend="eager")
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_no_guard_for_unused_input_varargs(self):
        # [FUNCTION DOC] Checks varargs and kwargs inputs unused by the function are accepted.
        def fn(x, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(3)
        result = opt_fn(x, torch.randn(2), extra=torch.randn(1))
        self.assertEqual(result, fn(x))

    def test_no_guard_for_unused_input_still_recompiles_on_used(self):
        # [FUNCTION DOC] Checks unused y does not recompile but changed x does (frames 1,
        # then 2).
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, dynamic=False)
        opt_fn(torch.randn(3), torch.randn(3))
        opt_fn(torch.randn(3), torch.randn(5))
        self.assertEqual(cnt.frame_count, 1)
        opt_fn(torch.randn(4), torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)

    def test_no_guard_for_unused_input_with_graph_break(self):
        # [FUNCTION DOC] Checks unused input with a graph break does not add recompiles when
        # y changes.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds 1, graph breaks, adds 2.
            z = x + 1
            torch._dynamo.graph_break()
            return z + 2

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, dynamic=False)
        opt_fn(torch.randn(3), torch.randn(3))
        after_first = cnt.frame_count
        opt_fn(torch.randn(3), torch.randn(5))
        self.assertEqual(cnt.frame_count, after_first)

    def test_no_guard_for_input_unused_before_graph_break(self):
        # [FUNCTION DOC] Checks the base frame does not guard on an input used only after a graph
        # break; only the resume function recompiles on y shape changes.
        # y is not read before the break, so the first subgraph must not
        # guard on it. Only the resume function (which reads y) should guard.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds 1, graph breaks, adds sum of y.
            z = x + 1
            torch._dynamo.graph_break()
            return z + y.sum()

        recompiled_names = []

        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Records the original function name of the failing code.
            recompiled_names.append(failure.orig_code.co_name)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(
            cnt, nopython=False, guard_fail_fn=guard_failures
        )(fn)

        x = torch.randn(3)
        y1, y2, y3 = torch.randn(3), torch.randn(5), torch.randn(9)
        self.assertEqual(opt_fn(x, y1), fn(x, y1))

        self.assertEqual(opt_fn(x, y2), fn(x, y2))
        self.assertEqual(opt_fn(x, y3), fn(x, y3))
        # Resume function recompiled (y's shape changed), base frame did not.
        self.assertTrue(recompiled_names)
        self.assertNotIn(fn.__code__.co_name, recompiled_names)

    def test_guard_sym_node_fstring_when_used(self):
        # [FUNCTION DOC] Checks a formatted string of a symbolic size stored in a variable
        # creates a size guard that fails when size changes.
        def fn(x):
            # [FUNCTION DOC] HELPER: Formats first dim into a string and returns sine and
            # the string.
            # assign fstring to a variable causes the fstring to be used,
            # which realizes the variable tracker.
            f_str = f"{x.shape[0]}"
            return x.sin(), f_str

        guard_failure = None

        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Records the failure.
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", guard_fail_fn=guard_failures, dynamic=True
        )(fn)
        args1 = torch.randn(10, 11)
        out = fn(args1)
        opt_out = opt_fn(args1)
        self.assertEqual(out, opt_out)

        # We change x.shape[0] to test whether it's guarded
        args2 = torch.randn(9, 11)
        out = fn(args2)
        opt_out = opt_fn(args2)
        self.assertEqual(out, opt_out)
        self.assertTrue(guard_failure is not None)
        self.assertIn("""tensor 'x' size mismatch at index 0""", guard_failure[0])

    def test_fstring_dynamic_symint_across_graph_break(self):
        # [FUNCTION DOC] Checks formatted strings with dynamic symbolic sizes across a graph
        # break give right messages for three buffer sizes.
        class Arch(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer and message string.
            def __init__(self, n):
                # [FUNCTION DOC] HELPER: Registers buffer w of size n and empty message.
                super().__init__()
                self.register_buffer("w", torch.randn(n))
                self.msg = ""

            def forward(self, idx):
                # [FUNCTION DOC] HELPER: Clamps out-of-range indices and records a message with
                # max index and bad values; returns selected entries.
                max_idx = self.w.size(0) - 1
                idx = idx.long()
                mask = (idx < 0) | (idx > max_idx)
                if mask.any():
                    self.msg = f"max_idx={max_idx}, bad={idx[mask].tolist()}"
                    idx = torch.clamp(idx, 0, max_idx)
                return torch.index_select(self.w, 0, idx)

        @torch.compile(backend="eager")
        def run(arch, idx):
            # [FUNCTION DOC] HELPER: Calls arch.
            return arch(idx)

        idx = torch.tensor([0, -1, 2])
        # Distinct buffer sizes across the shared compiled forward force
        # automatic-dynamic, turning w.size(0) into a SymNodeVariable.
        for n in (8, 6, 7):
            arch = Arch(n)
            run(arch, idx)
            self.assertEqual(arch.msg, f"max_idx={n - 1}, bad=[-1]")

    def test_restore_graphstate(self):
        # [FUNCTION DOC] Checks guards created in a rolled-back branch of a nested function do
        # not show up in exported guards.
        # This function does some guard accumulation,
        # and then rolls back due to control flow.
        # The idea is that if one were printing guards as they appear,
        # they would see this insert a guard that does not show up in the final set of
        # guards as we rolled back from it.
        def nested_fn(s):
            # [FUNCTION DOC] HELPER: Squares s if x first element below 10.
            if x[0] < 10:
                return s * s
            return s

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds, calls nested_fn, adds 10, multiplies.
            x = x + 1
            y = nested_fn(y)
            y = y + 10
            return x * y

        all_guards = []

        def guard_export_print(guards):
            # [FUNCTION DOC] HELPER: Collects exported guards into the enclosing list.
            nonlocal all_guards
            all_guards.extend(guards)

        opt_fn = torch._dynamo.optimize("eager", guard_export_fn=guard_export_print)(fn)

        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])
        opt_fn(x, y)

        for guard in all_guards:
            # This guard was created
            self.assertTrue(guard.name != "nested_fn.__closure__[0].cell_contents")

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "Test requires CUDA or XPU.")
    def test_symint_as_device_kwarg_non_strict_export(self):
        # [FUNCTION DOC] Requires CUDA or XPU. Checks non-strict export with a symbolic size used
        # as a device index raises UserError about violated constraints.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating ones on a device given by the input size
            # minus two.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns ones on device size minus 2.
                # -2 to make device id 0 for easier testing on CI
                return torch.ones(10, device=x.size(0) - 2)

        x = torch.randn(2)
        m = Mod()
        d1 = torch.export.Dim("d1", max=2048)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError, r"Constraints violated \(d1\)"
        ):
            ep = torch.export.export(
                m, (x,), dynamic_shapes={"x": {0: d1}}, strict=False
            )

    def test_call_parent_non_class_methods_from_child(self):
        # [FUNCTION DOC] Checks calls to parent class methods from a child (super forms, explicit
        # class forms, class attributes); compiles in one frame, recompiles after a parent class
        # attribute change (two frames).
        class A:
            # [CLASS DOC] HELPER: Base class with attribute a and methods add and mul.
            a = 4

            def add(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus 10.
                return x + 10

            def mul(self, x):
                # [FUNCTION DOC] HELPER: Returns x times 0.1.
                return x * 0.1

        class B(A):
            # [CLASS DOC] HELPER: Subclass with coeff.
            coeff = 4

            def add(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus 20.
                return x + 20

            @classmethod
            def cube(cls, x):
                # [FUNCTION DOC] HELPER: Class method returning coeff times x cubed.
                return cls.coeff * x * x * x

            def mul(self, x):
                # [FUNCTION DOC] HELPER: Returns parent mul result times x times 0.2.
                return super().mul(x) * x * 0.2

        class C(B):
            # [CLASS DOC] HELPER: Grandchild class.
            def add(self, x):
                # [FUNCTION DOC] HELPER: Combines cube, parent add, mul of B, add of A and
                # attribute a times x.
                b = super().cube(x)
                c = A.add(self, x)
                d = B.mul(self, x)
                e = super(B, self).add(x)
                f = super().a * x
                return b + c + d + e + f

        x = torch.rand(4)
        fn = C().add
        ref = fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnt.frame_count, 1)

        # Check recompilation
        A.a = 5
        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        # Ensure that super guard checks are working as expected
        res = opt_fn(x)
        self.assertEqual(cnt.frame_count, 2)

    def test_builder_for_class_with_metaclass(self):
        # [FUNCTION DOC] Checks isinstance with a class having a metaclass for an instance.
        class ExampleMeta(type):
            # [CLASS DOC] HELPER: Metaclass.
            pass

        class MyClass(metaclass=ExampleMeta):
            # [CLASS DOC] HELPER: Class using the metaclass.
            pass

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if y is an instance else x minus 1.
            if isinstance(y, MyClass):
                return x + 1
            else:
                return x - 1

        x = torch.rand([4, 4])
        y = MyClass()
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_tuple_from_tuple_iter(self):
        # [FUNCTION DOC] Checks building tuples from lists and from an iterator in compiled code,
        # then forcing recompile with a different count.
        def inner_fn(*args):
            # [FUNCTION DOC] HELPER: Adds all args in place to a ones tensor.
            acc = torch.ones(10, 10)
            for arg in args:
                acc.add_(arg)

            return acc

        @torch.compile(backend="eager")
        def fn(inputs, params):
            # [FUNCTION DOC] HELPER: Concatenates two tuples and calls inner_fn with them.
            y = tuple(inputs) + tuple(params)
            return inner_fn(*y)

        inputs = [torch.randn(10, 10) for _ in range(3)]

        fn(inputs, iter(tuple(inputs)))

        def fn(params):
            # [FUNCTION DOC] HELPER: Converts iterator to tuple and calls inner_fn.
            y = tuple(params)
            return inner_fn(*y)

        opt_fn = torch.compile(fn, backend="eager")
        inputs = [torch.randn(10, 10) for _ in range(3)]
        self.assertTrue(same(fn(iter(tuple(inputs))), opt_fn(iter(tuple(inputs)))))

        # Force recompilation
        inputs = [torch.randn(10, 10) for _ in range(4)]
        self.assertTrue(same(fn(iter(tuple(inputs))), opt_fn(iter(tuple(inputs)))))

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_argwhere_with_dynamic_shapes(self):
        # [FUNCTION DOC] With dynamic output shape ops on, checks argwhere on a meshgrid equality
        # for index mapping under fullgraph matches eager.
        def fn(
            tensor: torch.Tensor,
            mapping: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Builds a meshgrid, finds equal positions, and writes mapped
            # indices into zeros.
            xx, yy = torch.meshgrid(mapping, tensor, indexing="ij")
            indices = torch.argwhere(xx == yy)

            mapped_values = torch.zeros_like(tensor)
            mapped_values[indices[:, 1]] = indices[:, 0]

            return mapped_values

        tensor = torch.tensor([1, 2, 3, 5, 6, 7])
        mapping = torch.tensor([0, 3, 4, 5, 7])
        opt = torch.compile(fn, fullgraph=True, backend="eager")
        self.assertEqual(fn(tensor, mapping), opt(tensor, mapping))

    def test_torch_package_working_with_trace(self):
        # [FUNCTION DOC] Checks a module saved through torch package and reloaded can be compiled
        # and run. Writes a file in the temp dir.
        # from torch._dynamo.test_case import run_tests

        inputs = [torch.randn([2, 2]), torch.randn([2, 2])]

        optimized_model = torch.compile(
            MyPickledModule(torch.randn([2, 2])), backend="eager"
        )
        from torch import package

        tmp_root = tempfile.gettempdir()
        path = os.path.join(tmp_root, "MyPickledModule.pt")
        package_name = "MyPickledModule"
        resource_name = "MyPickledModule.pkl"

        model = MyPickledModule(torch.randn([2, 2]))

        with package.PackageExporter(path) as exp:
            exp.extern("**")
            exp.save_pickle(package_name, resource_name, model)

        imp = package.PackageImporter(path)
        loaded_model = imp.load_pickle(package_name, resource_name)

        optimized_loaded_model = torch.compile(loaded_model, backend="eager")(*inputs)

    def test_precompile_entry_hit(self):
        # [FUNCTION DOC] Checks a precompile entry injected for a code object with an empty guard
        # manager is used instead of the original; after reset the original function runs.
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        def injected(x):
            # [FUNCTION DOC] HELPER: Returns x plus 42.
            return x + 42

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(),
            injected.__code__,
        )
        self.assertEqual(compiled_fn(*args), injected(*args))
        _reset_precompile_entries(fn.__code__)

        self.assertEqual(compiled_fn(*args), fn(*args))

    def test_precompile_entry_miss(self):
        # [FUNCTION DOC] Checks an injected precompile entry whose guard fails (not an int) is
        # skipped; the original function runs.
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        guard_manager = torch._dynamo.guards.RootGuardManager()
        guard_manager.add_lambda_guard(lambda L: isinstance(L["x"], int), [], None)

        def injected(x):
            # [FUNCTION DOC] HELPER: Returns x plus 42.
            return x + 42

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager),
            injected.__code__,
        )
        self.assertEqual(compiled_fn(*args), fn(*args))

    def test_precompile_entries(self):
        # [FUNCTION DOC] Checks multiple precompile entries guarded on bool, int, tensor and str
        # inputs; each input type dispatches to its own injected function; after reset the
        # original runs.
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        guard_manager_bool = torch._dynamo.guards.RootGuardManager()
        guard_manager_bool.add_lambda_guard(
            lambda L: isinstance(L["x"], bool), [], None
        )

        def injected_bool(x: bool):
            # [FUNCTION DOC] HELPER: Returns x plus 102.
            return x + 102

        guard_manager_int = torch._dynamo.guards.RootGuardManager()
        guard_manager_int.add_lambda_guard(lambda L: isinstance(L["x"], int), [], None)

        def injected_int(x: int):
            # [FUNCTION DOC] HELPER: Returns x plus 42.
            return x + 42

        guard_manager_tensor = torch._dynamo.guards.RootGuardManager()
        guard_manager_tensor.add_lambda_guard(
            lambda L: isinstance(L["x"], torch.Tensor), [], None
        )

        def injected_tensor(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Returns x plus 100.
            return x + 100

        guard_manager_str = torch._dynamo.guards.RootGuardManager()
        guard_manager_str.add_lambda_guard(lambda L: isinstance(L["x"], str), [], None)

        def injected_str(x: str):
            # [FUNCTION DOC] HELPER: Returns x plus the text 1.
            return x + "1"

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_bool),
            injected_bool.__code__,
        )

        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_int),
            injected_int.__code__,
        )

        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_tensor),
            injected_tensor.__code__,
        )

        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_str),
            injected_str.__code__,
        )

        self.assertEqual(compiled_fn(*args), injected_tensor(*args))
        self.assertEqual(compiled_fn(True), injected_bool(True))
        self.assertEqual(compiled_fn(10), injected_int(10))
        self.assertEqual(compiled_fn("10"), injected_str("10"))
        _reset_precompile_entries(fn.__code__)

        self.assertEqual(compiled_fn(*args), fn(*args))

    def test_precompile_fail_on_recompile(self):
        # [FUNCTION DOC] Checks a precompiled entry is used under the fail on recompile stance
        # when fn and injected share the same closure.
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        @torch.compiler.disable
        def graph(x, s0):
            # [FUNCTION DOC] HELPER: Disabled helper returning x plus s0.
            return x + s0

        def fn(x):
            # [FUNCTION DOC] HELPER: Declares the closure variable nonlocal and returns x
            # minus 1.
            nonlocal graph  # Forcing fn and injected to have the same closure.
            return x - 1

        def injected(x):
            # [FUNCTION DOC] HELPER: Gets the first dim size and calls graph.
            s0 = call_size(x, 0)
            return graph(x, s0)

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(),
            injected.__code__,
        )
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(*args), injected(*args))

    def test_fail_on_recompile_error_message(self):
        # [FUNCTION DOC] Checks the fail on recompile stance error message lists failed
        # precompiled guards. Resets entries afterwards.
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        guard_manager_bool = torch._dynamo.guards.RootGuardManager()
        guard_manager_bool.add_lambda_guard(
            lambda L: isinstance(L["x"], bool), ["isinstance(L['x'], bool)"], None
        )

        def injected_bool(x: bool):
            # [FUNCTION DOC] HELPER: Returns x plus 102.
            return x + 102

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_bool),
            injected_bool.__code__,
        )

        try:
            with torch.compiler.set_stance("fail_on_recompile"):
                with self.assertRaisesRegex(
                    RuntimeError, "Failed on the following precompiled guards:"
                ):
                    compiled_fn(*args)
        finally:
            _reset_precompile_entries(fn.__code__)

    def test_shape_and_tuple_equality(self):
        # [FUNCTION DOC] Checks comparing a tensor size with a tuple under fullgraph.
        def fn(x, y, t):
            # [FUNCTION DOC] HELPER: Multiplies, and returns cosine if size equals t else sine.
            z = x * y
            if x.size() == t:
                return z.cos()
            return z.sin()

        torch.compile(fn, backend="eager", fullgraph=True)(
            torch.randn([4, 4]), torch.randn([4, 4]), (4, 4)
        )

    def test_int_list(self):
        # [FUNCTION DOC] Checks int list elements specialization: frames 2 (auto dynamic) or 5
        # (static) or 1 (dynamic default), matching expected counts per config.
        # if assume_static_by_default == True: spec int list
        # otherwise: unspec int list
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns sine of x plus second element modulo 2.
            return torch.sin(x + y[1] % 2)

        x = torch.randn(6)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        for i in range(10, 25, 3):
            y = [i, i + 1, i + 2]
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, res))
        if torch._dynamo.config.assume_static_by_default:
            if torch._dynamo.config.automatic_dynamic_shapes:
                self.assertExpectedInline(cnt.frame_count, """2""")
            else:
                self.assertExpectedInline(cnt.frame_count, """5""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_patched_builtin_functions(self):
        # [FUNCTION DOC] Checks that compiled code respects a monkey-patched builtin isinstance
        # (patched to return False for one class), then recompiles after the patch is removed.
        # Restores the builtin in a finally block.
        import builtins

        # Cache the original builtin function ids
        torch._dynamo.trace_rules._builtin_function_ids()

        class MyClass:
            # [CLASS DOC] HELPER: Empty class.
            pass

        builtin_isinstance = builtins.isinstance

        def patched_isinstance(obj, classinfo) -> bool:
            # [FUNCTION DOC] HELPER: Patched isinstance returning False for MyClass instances,
            # else the real result.
            if builtin_isinstance(obj, MyClass):
                return False
            else:
                return builtin_isinstance(obj, classinfo)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if y is MyClass instance else x minus 1.
            if isinstance(y, MyClass):
                return x + 1
            else:
                return x - 1

        x = torch.ones(2, 3)
        y = MyClass()

        try:
            ref = fn(x, y)
            # Monkey patch builtin function
            builtins.isinstance = patched_isinstance
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, x + 1))
            self.assertTrue(same(res, x - 1))
        finally:
            builtins.isinstance = builtin_isinstance

        # check recompilation because builtins is now unpatched
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(res, x + 1))

    # specifically test for tensor.attribute -> torch.something()
    def test_real_imag_tensor_attribute(self):
        # [FUNCTION DOC] Checks the real and imaginary attribute of a complex tensor are traced
        # into the graph and match eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Multiplies the sum of real part and y with the
            # imaginary part.
            a = x.real
            b = x.imag
            return torch.mul(torch.add(a, y), b)

        x_real = torch.rand((4, 4))
        x_imag = torch.rand((4, 4))
        x = torch.complex(x_real, x_imag)
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_cast(self):
        # [FUNCTION DOC] Checks typing cast on a tensor result under fullgraph.
        from typing import cast

        def fn(x):
            # [FUNCTION DOC] HELPER: Casts the sum of x and 1.0 to Tensor.
            return cast(torch.Tensor, torch.add(x, 1.0))

        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        ref = fn(torch.ones(2, 2))
        res = opt_fn(torch.ones(2, 2))

        self.assertTrue(same(ref, res))

    def test_cast_with_different_module_types(self):
        # [FUNCTION DOC] Checks typing cast in a mixin used by two module classes compiles per
        # module type (two frames) with correct results.
        # typing.cast works correctly when used in a mixin pattern with
        # different module types, producing correct results without
        # graph breaks.
        from typing import cast

        class Mixin:
            # [CLASS DOC] HELPER: Mixin with a method that casts self to Module.
            def get_self_as_module(self):
                # [FUNCTION DOC] HELPER: Returns self cast to nn.Module.
                return cast(torch.nn.Module, self)

        class ModuleA(Mixin, torch.nn.Module):
            # [CLASS DOC] HELPER: Module with the mixin; forward calls the cast method and
            # adds 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the cast helper and returns x plus 1.
                self.get_self_as_module()
                return x + 1

        class ModuleB(Mixin, torch.nn.Module):
            # [CLASS DOC] HELPER: Module with the mixin; forward adds 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the cast helper and returns x plus 2.
                self.get_self_as_module()
                return x + 2

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(mod, x):
            # [FUNCTION DOC] HELPER: Calls the cast helper on mod and returns x plus 1.
            mod.get_self_as_module()
            return x + 1

        x = torch.randn(4)
        ref_a = fn.__wrapped__(ModuleA(), x)
        ref_b = fn.__wrapped__(ModuleB(), x)
        res_a = fn(ModuleA(), x)
        res_b = fn(ModuleB(), x)

        self.assertEqual(ref_a, res_a)
        self.assertEqual(ref_b, res_b)
        self.assertEqual(cnt.frame_count, 2)

    def test_cast_fullgraph_with_non_tensor(self):
        # [FUNCTION DOC] Checks typing cast with an int type on a shape element under fullgraph.
        # Verify typing.cast works with non-tensor values under fullgraph=True
        from typing import cast

        def fn(x):
            # [FUNCTION DOC] HELPER: Casts the first dim size to int and adds it to x.
            val = cast(int, x.shape[0])
            return x + val

        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        ref = fn(torch.ones(3, 4))
        res = opt_fn(torch.ones(3, 4))

        self.assertTrue(same(ref, res))

    def test_cast_no_recompile_after_graph_break(self):
        # [FUNCTION DOC] Checks that typing cast after a graph break does not cause extra
        # recompiles across module types: frame counts depend on the nested graph break config
        # (2) or are 1 with exactly 5 total frames.
        # In FSDP, cast(nn.Module, self) can be called after a
        # graph break. Without the polyfill + skip_code fix, PEP 523 compiles
        # typing.cast as a standalone frame with TYPE_MATCH guards on val,
        # causing recompilation when different module types pass through.
        # https://github.com/pytorch/pytorch/blob/0feb90404fbeb9b1594ae194f8fd47bbe7f5f245/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L376
        from typing import cast

        from torch._dynamo.utils import counters

        counters.clear()

        class Base(torch.nn.Module):
            # [CLASS DOC] HELPER: Module base with get_state.
            def get_state(self):
                # [FUNCTION DOC] HELPER: Graph breaks and then returns self cast to Module.
                torch._dynamo.decorators.graph_break()
                return cast(torch.nn.Module, self)

        class ModuleA(Base):
            # [CLASS DOC] HELPER: Empty subclass.
            pass

        class ModuleB(Base):
            # [CLASS DOC] HELPER: Empty subclass.
            pass

        cnt = torch._dynamo.testing.CompileCounter()
        a, b = ModuleA(), ModuleB()

        @torch.compile(backend=cnt)
        def fn(mod, x):
            # [FUNCTION DOC] HELPER: Calls get_state and returns x plus 1.
            mod.get_state()
            return x + 1

        x = torch.randn(4)
        fn(a, x)
        fn(b, x)
        if torch._dynamo.config.nested_graph_breaks:
            # NGB inlines get_state() and handles the graph break inline,
            # producing 2 subgraphs (before/after break) per call.
            self.assertEqual(cnt.frame_count, 2)
        else:
            self.assertEqual(cnt.frame_count, 1)
            # 5 frames: fn (x2), get_state before graph_break (x2),
            # get_state resume after graph_break (x1, no recompile).
            # Without skip_code, typing.cast would add 2 more frames (7 total).
            self.assertEqual(counters["frames"]["total"], 5)

    def test_T_tensor_attribute(self):
        # [FUNCTION DOC] Checks transpose attribute on a tensor in compiled code matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the transpose of x to y.
            a = x.T
            return torch.add(a, y)

        x = torch.rand((4, 4))
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_recursive_tensor_attribute(self):
        # [FUNCTION DOC] Checks chained attributes (real then transpose, imag) on a complex
        # tensor match eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Multiplies the sum of transposed real part and y with
            # imag part.
            a = x.real.T
            b = x.imag
            return torch.mul(torch.add(a, y), b)

        x_real = torch.rand((4, 4))
        x_imag = torch.rand((4, 4))
        x = torch.complex(x_real, x_imag)
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_assigning_function_to_object_attribute(self):
        # [FUNCTION DOC] Checks a function stored on an instance attribute is called without
        # receiving self.
        # user-defined functions which are object's attributes are not converted to bound methods
        def my_add(*args):
            # [FUNCTION DOC] HELPER: Unpacks two args and adds them.
            a, b = args
            return a + b

        class MyClass:
            # [CLASS DOC] HELPER: Holder storing a function attribute.
            def __init__(self, func):
                # [FUNCTION DOC] HELPER: Stores func as add.
                self.add = func

        obj = MyClass(my_add)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls obj.add with x and 2.
            return obj.add(x, 2)

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_assigning_function_to_class_attribute(self):
        # [FUNCTION DOC] Checks a function stored as a class attribute is bound to the instance
        # when called through it.
        # user-defined functions which are class's attributes are converted to bound methods
        def my_add(*args):
            # [FUNCTION DOC] HELPER: Unpacks object, a and b and returns x field plus a plus b.
            obj, a, b = args
            return obj.x + a + b

        class MyClass:
            # [CLASS DOC] HELPER: Class with a function class attribute.
            add = my_add

            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

        obj = MyClass(0.5)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls obj.add with x and 2.
            return obj.add(x, 2)

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_tagging_tensors_simple(self):
        # [FUNCTION DOC] Checks export keeps custom tag attributes of tensor inputs in
        # placeholder metadata (tags a and b).
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns product and both inputs.
            return x * y, x, y

        a = torch.randn([3, 3])
        a.tag = "a"
        b = torch.randn([3, 3])
        b.tag = "b"

        exported = torch._dynamo.export(foo)(a, b)
        out_graph = exported[0]

        nodes = list(out_graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        all_tags = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])

        self.assertEqual(all_tags, ["a", "b"])

    def test_tagging_tensors_mix_used_unused_structure(self):
        # [FUNCTION DOC] Checks export tags only for used inputs in a nested structure: only the
        # two state tensors' tags are in placeholders.
        def pre_attention_state_ops(input, mems, state):
            # [FUNCTION DOC] HELPER: Builds nested lists of tensors from the state entries.
            lc_key = state[0]
            lc_val = state[1]
            bar = []
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar.append(bar2)

            return bar

        mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
        state = [
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
        ]
        i = torch.tensor(
            [
                [0.0313, -0.1487, -0.3846, -0.5321],
                [-1.7073, 1.3331, -0.0890, -1.4935],
                [-0.8314, -0.1862, -0.5935, 1.5232],
            ]
        )

        mems.tag = "MEMS"
        i.tag = "FOO"
        state[0].tag = "STATE_0"
        state[1].tag = "HMMM"

        exported = torch._dynamo.export(pre_attention_state_ops)(i, mems, state)
        out_graph = exported[0]

        nodes = list(out_graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        all_tags = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])

        self.assertEqual(all_tags, ["STATE_0", "HMMM"])

    def test_get_custom_tensor_attribute(self):
        # [FUNCTION DOC] Checks reading a custom float attribute from a tensor in compiled code.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns custom attribute times x.
            return x.custom_attr * x

        x = torch.rand((2, 2))
        x.custom_attr = 3.14
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_set_custom_tensor_attribute(self):
        # [FUNCTION DOC] Checks setting then reading a custom attribute on a tensor in
        # compiled code.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sets custom attribute then returns it times x.
            x.custom_attr = 3.14
            return x.custom_attr * x

        x = torch.rand((2, 2))
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_unhandled_exception_in_dynamo(self):
        # [FUNCTION DOC] Checks that an exception raised in compiled code propagates with the
        # message visible in the formatted traceback.
        # traceback.format_exc() approximates an unhandled exception
        def f(a):
            # [FUNCTION DOC] HELPER: Adds 1 in place then raises RuntimeError.
            a += 1
            raise RuntimeError("smoge")
            return a

        opt_fn = torch.compile(f, backend="eager")
        try:
            opt_fn(torch.ones(2))
        except RuntimeError as e:
            self.assertIn("smoge", traceback.format_exc())

    def test_unhandled_exception_in_dynamo2(self):
        # [FUNCTION DOC] Checks error handling of nested exceptions in compiled code
        # (guards against a segfault): dense and sparse matmul errors are caught and a new
        # error is raised.
        # segfaults in python 3.11 if shadow frame is freed improperly
        from torch.testing import make_tensor

        def fn():
            # [FUNCTION DOC] HELPER: Defines test1 and calls it dense then sparse in nested try
            # blocks, raising RuntimeError on double failure.
            # test that the errors are the same for dense and sparse versions
            def test1(*, is_sparse):
                # [FUNCTION DOC] HELPER: Runs addmm with a dense or sparse csr argument on a
                # random matrix.
                # shapes must be compatible for matrix multiplication
                a = make_tensor((2, 3), dtype=torch.float32, device="cpu")
                if is_sparse:
                    a_sparse = a.to_sparse_csr()
                    return torch.addmm(a, a_sparse, a)
                else:
                    return torch.addmm(a, a, a)

            try:
                test1(is_sparse=False)
            except RuntimeError as msg:
                try:
                    test1(is_sparse=True)
                except RuntimeError as msg2:
                    raise RuntimeError("smoge")

        opt_fn = torch.compile(fn, backend="eager")
        try:
            opt_fn()
        except RuntimeError:
            self.assertIn("smoge", traceback.format_exc())

    def test_variable_access_in_exception(self):
        # [FUNCTION DOC] Checks reading and updating a variable in an except block
        # under fullgraph.
        def fn():
            # [FUNCTION DOC] HELPER: Raises and catches an error, increments x in the handler.
            x = torch.ones(1)
            try:
                raise RuntimeError("bad")
            except RuntimeError:
                x += 1
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), torch.tensor([2.0]))

    def test_nested_sequential_with(self):
        # [FUNCTION DOC] Checks sequential nested grad mode context managers produce the
        # right result.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1 in two sibling grad-mode contexts nested in an
            # outer one.
            with torch.set_grad_enabled(True):
                with torch.set_grad_enabled(False):
                    x = x + 1
                with torch.set_grad_enabled(True):
                    x = x + 1
                return x

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))

    def test_nested_sequential_try(self):
        # [FUNCTION DOC] Checks sequential and nested try blocks with bare excepts compile and
        # give 3.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1 inside nested try blocks twice.
            try:
                try:
                    x = x + 1
                except:
                    pass
                try:
                    try:
                        x = x + 1
                    except:
                        pass
                except:
                    pass
            except:
                pass
            return x

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))

    def test_sparse_output_inductor_should_break(self) -> None:
        # [FUNCTION DOC] Checks sparse conversion in a compiled function matches eager for
        # aot_eager and runs under inductor (issue 164823).
        # See https://github.com/pytorch/pytorch/issues/164823
        # We want consistent semantics here
        def forward(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Converts x to sparse and doubles.
            x_sparse = x.to_sparse()
            return x_sparse * 2

        test_tensor = torch.randn(10, 10)
        pt = forward(test_tensor)
        aot_eager = torch.compile(forward, backend="aot_eager")(test_tensor)
        self.assertEqual(pt, aot_eager)
        inductor = torch.compile(forward, backend="inductor")(test_tensor)

    def test_nested_sequential_try_with(self):
        # [FUNCTION DOC] Checks try blocks mixed with grad-mode contexts.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1 inside try blocks, one under a nested no-
            # grad context.
            with torch.set_grad_enabled(True):
                try:
                    x = x + 1
                except:
                    pass
                try:
                    with torch.set_grad_enabled(False):
                        x = x + 1
                except:
                    pass
            return x

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))

    def test_nested_sequential_try_with_graph_break(self):
        # [FUNCTION DOC] Checks graph breaks within nested try and grad-mode contexts: one frame
        # when taken path has no extra break, three frames when n is nonzero.
        def fn(x, n):
            # [FUNCTION DOC] HELPER: Adds 1 across nested no-grad contexts with breaks, one
            # dependent on n.
            with torch.set_grad_enabled(True):
                with torch.set_grad_enabled(False):
                    x = x + 1
                    torch._dynamo.graph_break()
                try:
                    with torch.set_grad_enabled(False):
                        x = x + 1
                        if n == 0:
                            torch._dynamo.graph_break()
                except:
                    pass
                with torch.set_grad_enabled(False):
                    x = x + 1
                    torch._dynamo.graph_break()
                x = x + 1
            return x

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        self.assertEqual(opt_fn(torch.ones(1), 0), torch.tensor([5.0]))
        self.assertEqual(counter.frame_count, 1)

        torch._dynamo.reset()
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        self.assertEqual(opt_fn(torch.ones(1), 1), torch.tensor([5.0]))
        self.assertEqual(counter.frame_count, 3)

    def test_ordered_dict_alias_reconstruct(self):
        # [FUNCTION DOC] Checks that an ordered dict created from a dict, mutated, and kept
        # across a graph break keeps its type; prints disassembly.
        od = collections.OrderedDict

        def fn():
            # [FUNCTION DOC] HELPER: Builds dict, converts to ordered dict, adds an item, graph
            # breaks, returns sum if type holds.
            d1 = dict()  # noqa: C408
            d1["a"] = 1
            d2 = od(d1)
            d2["b"] = 2
            torch._dynamo.graph_break()
            if isinstance(d2, od):
                return d2["a"] + d2["b"]
            else:
                return 0

        dis.dis(fn)
        self.assertEqual(torch.compile(fn, backend="eager")(), 3)

    # NOTE this test can be removed once multiline errors are in Python.
    # See https://github.com/python/cpython/issues/106922
    # Covered by test_logging.py:test_trace_call* tests in 3.13+
    @skipIfNotPy311
    @unittest.skipIf(sys.version_info >= (3, 13), "feature landed in 3.13")
    def test_get_instruction_source_311(self):
        # [FUNCTION DOC] Python 3.11 and 3.12 only. Checks the instruction source helper recovers
        # exact multi-line source snippets with underline markers for binary ops, subscripts,
        # calls and unicode text at chosen instruction offsets.
        def f():
            # [FUNCTION DOC] HELPER: Function never called; used only for its bytecode,
            # containing many formatted multi-line expressions.
            # flake8: noqa
            # fmt: off
            # test binary ops
            a = ( b   )   +   c
            a = (a + b) // (c - d)
            a = b    \
         +\
               c  # test
            a = (
                (b  # test +
                    )  \
                # +
            << (

                c  # test
                \
            )  # test
            )

            # test slice
            a = bbb   [  ccc    ]
            b = bbbbb \
                [  ccc # test

                 + ddd  \

                ] # test
            a = bbb[ccc][ddd][eee]

            # test nested and multiline function calls
            a = g(g(g(b)))
            a = g(h(
                g(b),
                c
            ))

            # test chained function calls
            a = (g(x).y)(
                z
            )(1)(2)

            # test unicode (match traceback behavior)
            a = ("🔥🔥🔥" +
                + "🔥🔥") + b

        from torch._dynamo.utils import get_instruction_source_311

        if sys.version_info >= (3, 12):
            # Offsets changed in 3.12, e.g. due to removal of PRECALL inst
            offsets = (3, 11, 15, 19, 23, 29, 35, 44, 53, 65)
        else:
            offsets = (3, 11, 15, 19, 23, 29, 35, 46, 58, 74)
        insts = list(dis.get_instructions(f))
        # uncomment to determine offsets
        # print(*enumerate(insts), sep="\n")
        all_sources = "\n".join(
            get_instruction_source_311(f.__code__, insts[offset]) for offset in offsets
        )
        self.assertExpectedInline(
            all_sources,
            """\
            a = ( b   )   +   c
                ~~~~~~~~~~^~~~~

            a = (a + b) // (c - d)
                ~~~~~~~~^^~~~~~~~~

            a = b    \\
                ~~~~~~
         +\\
         ^~
               c  # test
               ~

                (b  # test +
                ~~~~~~~~~~~~
                    )  \\
                    ~~~~
                # +
                ~~~
            << (
            ^^~~


                c  # test
                ~~~~~~~~~
                \\
                ~
            )  # test
            ~

            a = bbb   [  ccc    ]
                ~~~~~~^^^^^^^^^^^

            b = bbbbb \\
                ~~~~~~~
                [  ccc # test
                ^^^^^^^^^^^^^


                 + ddd  \\
                 ^^^^^^^^


                ] # test
                ^

            a = bbb[ccc][ddd][eee]
                ~~~~~~~~^^^^^

            a = g(g(g(b)))
                  ~^^^^^^

            a = g(h(
                  ~^
                g(b),
                ^^^^^
                c
                ^
            ))
            ^

            a = (g(x).y)(
                ~~~~~~~~~
                z
                ~
            )(1)(2)
            ~^^^
""",
        )
        # test unicode (since assertExpectedInline doesn't support unicode)
        op_offset = 74 if sys.version_info >= (3, 12) else 84
        self.assertEqual(
            get_instruction_source_311(f.__code__, insts[op_offset]),
            """\
            a = ("🔥🔥🔥" +
                ~~~~~~~~
                + "🔥🔥") + b
                ~~~~~~~~^~~
""",
        )

    def test_float_speculation_log_divergence(self):
        # [FUNCTION DOC] Checks recompile with different float scale factors on two interpolate
        # calls gives the same result as eager.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Interpolates x and y with scale z and multiplies.
            a = F.interpolate(x, scale_factor=z, mode="bilinear", align_corners=False)
            b = F.interpolate(y, scale_factor=z, mode="bilinear", align_corners=False)
            return a * b

        cnt = CompileCounterWithBackend("eager")
        fn_opt = torch.compile(fn, backend=cnt)
        y = torch.randn(3, 3, 3, 4)

        self.assertEqual(fn(y, y, 1.0), fn_opt(y, y, 1.0))
        self.assertEqual(fn(y, y, 2.0), fn_opt(y, y, 2.0))

    def test_raise_guard_full_constraint(self):
        # [FUNCTION DOC] Checks that a dimension marked dynamic but specialized by a branch on
        # equality raises ConstraintViolationError.
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            # [FUNCTION DOC] HELPER: Returns sine if first dim is 3, else cosine.
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(my_dyn_fn, backend="eager")(y)

    def test_raise_guard_indirect_full_constraint(self):
        # [FUNCTION DOC] Checks indirect constraint violation (comparisons forcing a specific
        # value) raises ConstraintViolationError.
        y = torch.randn([3, 3, 3])

        def dyn_fn(x):
            # [FUNCTION DOC] HELPER: Branches on first dim greater, less, else equal to 3.
            if x.shape[0] > 3:
                return x.cos()
            if x.shape[0] < 3:
                return x * 2
            return x.sin()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(dyn_fn, backend="eager")(y)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_empty_tensor(self):
        # [FUNCTION DOC] With scalar capture, checks empty tensor of an unbacked computed size
        # under fullgraph (sizes 1 and 0).
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Reads a scalar and returns an empty tensor of size (n - 1)
            # // 2.
            n = x.item()
            return torch.empty((n - 1) // 2)

        self.assertEqual(fn(torch.tensor([4])).size(0), 1)
        self.assertEqual(fn(torch.tensor([1])).size(0), 0)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_and_terms(self):
        # [FUNCTION DOC] With scalar capture, checks symbolic and of three conditions in a check
        # allows branch simplification, and a violating input raises RuntimeError.
        from torch.fx.experimental.symbolic_shapes import sym_and

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def fn(xs):
            # [FUNCTION DOC] HELPER: Unpacks list into two unbacked values, checks combined
            # bounds, counts true branch conditions and returns sum.
            u0, u1 = xs.tolist()
            torch._check(sym_and(u0 >= 3, u0 <= 10, u1 >= 2))

            # test individual checks
            n = 0
            if u0 >= 3:
                n += 1
            if u0 <= 11:
                n += 1
            if u1 >= 1:
                n += 1
            return u0 + u1 + n

        fn(torch.tensor([5, 6]))
        fn(torch.tensor([8, 7]))
        with self.assertRaises(RuntimeError):
            fn(torch.tensor([9, 0]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_min_simplifies_with_checked_upper_bound(self):
        # [FUNCTION DOC] With scalar capture, checks symbolic min simplifies given a checked
        # upper bound.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Reads a pair, checks x0 is at most 5 and compares symbolic
            # min with x0.
            x0, _ = x.tolist()
            torch._check(x0 <= 5)
            if torch.sym_min(x0, 5) == x0:
                return torch.tensor(True)
            return torch.tensor(False)

        self.assertEqual(fn(torch.tensor([3, 5])), torch.tensor(True))

    def test_unbacked_2d_expand(self):
        # [FUNCTION DOC] Checks expand with unbacked 2D shape inputs followed by new and changed
        # shapes does not error.
        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def func(a, b):
            # [FUNCTION DOC] HELPER: Expands a to the shape of b and returns a times 10.
            a.expand(b.shape)
            return a * 10

        a = torch.rand(1, 1)
        b = torch.rand(1, 1)

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(a, 1)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        torch._dynamo.decorators.mark_unbacked(b, 1)
        func(a, b)
        func(torch.rand(4, 5), torch.rand(4, 5))
        # This does not raise an error right now because of a recompilation.
        # https://github.com/pytorch/pytorch/issues/163785
        # with self.assertRaises(AssertionError):
        #     func(torch.rand(1, 1), torch.rand(2, 1))
        func(torch.rand(1, 1), torch.rand(2, 1))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_constrain_range_on_replaced_unbacked_symbol(self):
        # [FUNCTION DOC] With scalar capture, checks a runtime assertion on an unbacked
        # value replaced by symbolic sizes: valid input passes, an inconsistent input
        # raises RuntimeError.
        # Tests the following case:
        # Deferred runtime asserts adds sym_constrain_range(u0).
        # However, u0 is replaced with s0 + s1.
        # So, now we have sym_constrain_range(s0 + s1).
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Reads z item, checks size bounds and equality of u0 with sum
            # of sizes.
            z += 7  # to avoid creating unspecified symbol instead of unbacked symbol
            u0 = z.item()
            s0 = x.size(0)
            s1 = y.size(0)
            torch._check(s0 < 100)
            torch._check(s1 < 100)
            torch._check(u0 == s0 + s1)
            return x, y, z

        inputs = (
            x := torch.randn(16, 10),
            y := torch.randn(16, 10),
            torch.tensor(32 - 7),
        )
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        opt = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        opt(*inputs)
        with self.assertRaises(RuntimeError):
            inputs = (
                x := torch.randn(16, 10),
                y := torch.randn(16, 10),
                torch.tensor(32),
            )
            opt(*inputs)

    def test_sym_max_creates_graph_node(self):
        # [FUNCTION DOC] Checks symbolic max of two dims creates a graph node, once, and the
        # result is right.
        # Test that sym_max creates a graph node and works correctly
        from torch._dynamo.testing import EagerAndRecordGraphs

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sum plus symbolic max of dims.
            max_dim = torch.sym_max(x.size(0), x.size(1))
            return x.sum() + max_dim

        backend = EagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)

        x = torch.randn(4, 8)
        result = compiled_fn(x)
        expected = x.sum() + 8
        self.assertEqual(result, expected)

        # Verify sym_max appears in graph
        self.assertGreater(len(backend.graphs), 0)
        graph = backend.graphs[0]
        sym_max_nodes = [n for n in graph.graph.nodes if n.target is torch.sym_max]
        self.assertEqual(len(sym_max_nodes), 1, "sym_max should be in the graph")

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_symint_copy_into_unbacked_slice(self):
        # [FUNCTION DOC] With scalar capture and static default, checks copy into a slice with
        # unbacked padding does not crash (data-dependent guard).
        @torch.compile(backend="eager")
        def fn(a, x):
            # [FUNCTION DOC] HELPER: Pads a with an unbacked number of zeros, concatenates,
            # writes a random block into a slice and returns it.
            u0 = torch.tensor(x[0].to(torch.int64).item()).item()
            B, H, T, D = a.shape
            a_padding = torch.zeros((B, H, u0, D), dtype=torch.float64)
            b = torch.cat([a, a_padding], dim=2)
            c = torch.randn(B, H, 152, D)
            b[:, :, :152, :] = c
            return b

        x = torch.tensor([0])
        torch._dynamo.decorators.mark_unbacked(x, 0)
        a = torch.zeros((1, 16, 152, 96))

        # Previously would crash with guard on data dependent error
        fn(a, x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symint_fold_nontrivial_product_modulo(self):
        # [FUNCTION DOC] With scalar capture, checks a modulo of a product of the same unbacked
        # expression folds to true.
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Unpacks two scalars and returns true tensor if the product
            # modulo condition holds.
            u0, u1 = x.tolist()
            # The condition should fold to true.
            if ((u0 + 10) * (u0 + 10)) % (u0 + 10) == 0:
                return torch.tensor(True)
            return torch.tensor(False)

        res = f(torch.tensor([20, 21]))
        self.assertEqual(torch.tensor(True), res)

    # Translation validation changes the exception type, don't run with it
    @torch.fx.experimental._config.patch(translation_validation=False)
    def test_mark_dynamic_with_ranges(self):
        # [FUNCTION DOC] With translation validation off, checks mark_dynamic with min and max
        # and a conflicting branch raises ConstraintViolationError.
        y = torch.randn([8, 3, 3])

        def my_dyn_fn(x):
            # [FUNCTION DOC] HELPER: Returns sine if first dim is 3 else cosine.
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        torch._dynamo.mark_dynamic(y, 0, min=2, max=5)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(my_dyn_fn, backend="eager")(y)

    def test_mark_static(self):
        # [FUNCTION DOC] Checks mark_static causes two frames for sizes 3 and 4.
        counter = CompileCounter()

        def my_dyn_fn(x):
            # [FUNCTION DOC] HELPER: Returns cosine of x.
            return x.cos()

        y = torch.randn([3])
        torch._dynamo.mark_static(y, 0)
        torch.compile(my_dyn_fn, backend=counter)(y)

        z = torch.randn([4])
        torch.compile(my_dyn_fn, backend=counter)(z)

        self.assertEqual(counter.frame_count, 2)

    def test_no_raise_guard_partial_constraint(self):
        # [FUNCTION DOC] Checks a partial constraint (greater-than branch) does not raise when
        # marked dynamic.
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            # [FUNCTION DOC] HELPER: Returns sine if first dim above 3 else cosine.
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        torch.compile(my_dyn_fn, backend="eager")(y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        torch.compile(my_dyn_fn, backend="eager")(y)

    def test_no_raise_guard_partial_constraint_across_break(self):
        # [FUNCTION DOC] Checks partial constraint across a graph break does not raise.
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            # [FUNCTION DOC] HELPER: Multiplies, graph breaks, branches on first dim above 2.
            z = x * y

            torch._dynamo.graph_break()
            if z.shape[0] > 2:
                return z.cos()

            return x.cos()

        torch.compile(my_dyn_fn, backend="eager")(y, y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        torch.compile(my_dyn_fn, backend="eager")(y, y)

    def test_tolist(self):
        # [FUNCTION DOC] Checks tolist on an int tensor compiles with graph break; two frames.
        # This should compile with no faluire.
        cnt = CompileCounterWithBackend("eager")

        @torch.compile(fullgraph=False, backend=cnt)
        def func(a):
            # [FUNCTION DOC] HELPER: Scales a, unpacks five items, returns product.
            a = a * 100
            u0, u1, u2, u3, u4 = a.tolist()
            return a * u0 * u1

        func(torch.tensor([1, 2, 3, 4, 5]))
        self.assertEqual(cnt.frame_count, 2)

    # Sadly, this does not throw - we do not prop correctly across the graph break
    @unittest.expectedFailure
    def test_raise_guard_partial_constraint_across_break(self):
        # [FUNCTION DOC] Expected failure: a constraint violation across a graph break is
        # not detected.
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            # [FUNCTION DOC] HELPER: Multiplies, graph breaks, branches on first dim equal 3.
            z = x * y

            torch._dynamo.graph_break()
            if z.shape[0] == 3:
                return z.cos()

            return x.cos()

        torch.compile(my_dyn_fn, backend="eager")(y, y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            Exception,
        ):
            torch.compile(my_dyn_fn, backend="eager")(y, y)

    def test_raise_guard_partial_constraint_no_graph_break(self):
        # [FUNCTION DOC] Checks equality constraint without graph break
        # raises ConstraintViolationError.
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            # [FUNCTION DOC] HELPER: Multiplies, branches on first dim equal 3.
            z = x * y

            if z.shape[0] == 3:
                return z.cos()

            return x.cos()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(my_dyn_fn, backend="eager")(y, y)

    @torch._dynamo.config.patch(force_parameter_static_shapes=True)
    @torch._dynamo.config.patch(force_nn_module_property_static_shapes=True)
    @torch.compiler.config.patch(
        dynamic_sources="L['x'],L['y'],L['self']._modules['y'].x,L['self']._modules['y']._modules['c']._parameters['weight'],L['self']._modules['y']._modules['c']._parameters['bias']"
    )
    def test_dynamic_sources_force_parameter_static_shapes_and_property_static_shapes_override(
        self,
    ):
        # [FUNCTION DOC] With forced static parameter and property shapes but dynamic sources
        # listing input and nested module members, checks three models with different input sizes
        # compile once.
        builder._DYNAMIC_SOURCES = None

        counter = CompileCounter()

        class Y(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer and an int attribute.
            def __init__(self, n_input, n_output):
                # [FUNCTION DOC] HELPER: Creates linear layer and stores n_input.
                super().__init__()
                self.c = torch.nn.Linear(n_input, n_output)
                self.x = n_input

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear and scales by the int.
                return self.c(x) * self.x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding Y and sizes.
            def __init__(self, n_input, n_output):
                # [FUNCTION DOC] HELPER: Stores sizes, calls base init, builds Y.
                self.n_input = n_input
                self.n_output = n_output
                super().__init__()
                self.y = Y(n_input, n_output)

            @torch.compile(backend=counter)
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Compiled forward applying Y and multiplying by y.
                return self.y(x) * y

        model = M(3210, 30)
        model(torch.randn(1, 3210), 2)
        model = M(3211, 30)
        model(torch.randn(1, 3211), 3)
        model = M(3212, 30)
        model(torch.randn(1, 3212), 4)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['self']._modules['inner'].x")
    def test_dynamic_sources_precedence_over_int_specialization(self):
        # [FUNCTION DOC] Checks dynamic sources take precedence over int specialization for an
        # attribute on a built-in module; two models compile once.
        builder._DYNAMIC_SOURCES = None

        counter = CompileCounter()

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer carrying an extra attribute x.
            def __init__(self, x) -> None:
                # [FUNCTION DOC] HELPER: Creates inner linear and stores x on it.
                super().__init__()
                self.inner = torch.nn.Linear(10, 10)
                # attach attribute to builtin nn module.
                self.inner.x = x

            @torch.compile(fullgraph=True, backend=counter)
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Compiled forward multiplying a by the inner attribute.
                return a * self.inner.x

        m1 = Model(50)
        m2 = Model(60)
        with fresh_cache():
            m1(torch.rand(1, 2, 3))
            m2(torch.rand(1, 2, 3))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_int(self):
        # [FUNCTION DOC] Checks int argument listed as dynamic source gives one frame across
        # 3 values.
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies random tensor by x.
            return torch.randn(5) * x

        fn(1)
        fn(2)
        fn(3)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_tensor(self):
        # [FUNCTION DOC] Checks tensor argument listed as dynamic source gives one frame across
        # 3 sizes.
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        fn(torch.randn(2))
        fn(torch.randn(3))
        fn(torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_values="1111")
    def test_dynamic_values_int(self):
        # [FUNCTION DOC] Checks dynamic values config with a sentinel int marks int dynamic, so
        # one frame.
        builder._DYNAMIC_VALUES = None
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies random tensor by x.
            return torch.randn(5) * x

        # Warm up with the sentinel value 1111 → the int gets marked dynamic.
        fn(1111)
        fn(2)
        fn(3)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_values="1111")
    def test_dynamic_values_tensor(self):
        # [FUNCTION DOC] Checks dynamic values config marks a dim equal to the sentinel dynamic,
        # so one frame.
        builder._DYNAMIC_VALUES = None
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        # Warm up with a tensor whose dim equals 1111 → that dim becomes dynamic.
        fn(torch.randn(1111))
        fn(torch.randn(3))
        fn(torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(assume_static_by_default=True)
    @torch.compiler.config.patch(dynamic_values="1111")
    def test_dynamic_values_no_match_specializes(self):
        # [FUNCTION DOC] With static default, checks ints that do not match the dynamic value
        # specialize (two frames).
        builder._DYNAMIC_VALUES = None
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies random tensor by x.
            return torch.randn(5) * x

        fn(2)
        fn(3)
        fn(4)

        self.assertEqual(counter.frame_count, 2)

    @torch.compiler.config.patch(dynamic_values="1111")
    def test_dynamic_values_graph_break(self):
        # [FUNCTION DOC] Checks dynamic values re-mark dynamic after graph break; two frames with
        # no recompile.
        """Motivating use case: graph breaks drop dynamic-ness in eager code,
        but the sentinel value lets the second compiled region re-mark dynamic."""
        builder._DYNAMIC_VALUES = None
        counter = CompileCounter()

        def foo(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x, graph breaks, calls foo.
            x = x * x
            torch._dynamo.graph_break()
            return foo(x)

        fn(torch.randn(1111))
        fn(torch.randn(3))
        fn(torch.randn(4))

        # 2 frames (one per side of the graph break), no recompiles on either.
        self.assertEqual(counter.frame_count, 2)

    @torch.compiler.config.patch(dynamic_values="1111, 2222")
    def test_dynamic_values_multiple(self):
        # [FUNCTION DOC] Checks multiple sentinel values in dynamic values; one frame.
        builder._DYNAMIC_VALUES = None
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        # Both sentinel values trigger dynamic marking.
        fn(torch.randn(2222))
        fn(torch.randn(7))
        fn(torch.randn(9))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(unbacked_sources="L['x']")
    def test_unbacked_sources_tensor(self):
        # [FUNCTION DOC] Checks unbacked sources applies to tensor dims including zero and one;
        # one frame.
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        fn(torch.randn(0))
        fn(torch.randn(1))
        fn(torch.randn(2))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(unbacked_sources="L['x']")
    def test_unbacked_sources_scalar(self):
        # [FUNCTION DOC] Checks unbacked sources applies to an int input; one frame.
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies ones by x.
            return torch.ones(2) * x

        fn(0)
        fn(1)
        fn(2)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_graph_break(self):
        # [FUNCTION DOC] Checks dynamic sources with graph break; two frames, no recompile.
        counter = CompileCounter()

        def foo(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x, graph breaks, calls foo.
            x = x * x
            torch._dynamo.graph_break()
            return foo(x)

        fn(torch.randn(2))
        fn(torch.randn(3))
        fn(torch.randn(4))

        # 2 since graph break produces 2 graphs. NB: there are no recompiles
        self.assertEqual(counter.frame_count, 2)

    @torch.compiler.config.patch(dynamic_sources="L['x'], L['y']")
    def test_dynamic_sources_dynamic_override(self):
        # [FUNCTION DOC] Checks dynamic sources override dynamic mode off for int and tensor;
        # one frame.
        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Multiplies x and y.
            return x * y

        fn(2, torch.randn(2))
        fn(3, torch.randn(3))
        fn(4, torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L\\['x.*'\\], L\\['y.*'\\]")
    def test_dynamic_sources_dynamic_override_regex(self):
        # [FUNCTION DOC] Checks dynamic sources given as regex patterns; one frame.
        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x1, y1):
            # [FUNCTION DOC] HELPER: Multiplies x1 and y1.
            return x1 * y1

        fn(2, torch.randn(2))
        fn(3, torch.randn(3))
        fn(4, torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']:0")
    def test_dynamic_sources_per_dim_single(self):
        # [FUNCTION DOC] Checks per-dimension dynamic source (dim 0 only): varying dim 0 does not
        # recompile, varying dim 1 does (two frames).
        builder._DYNAMIC_SOURCES = None

        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        # Vary dim 0 only -> should not recompile (dim 0 is in the dynamic-sources list)
        fn(torch.randn(2, 4))
        fn(torch.randn(3, 4))
        fn(torch.randn(4, 4))

        self.assertEqual(counter.frame_count, 1)

        # Now vary dim 1 -> dim 1 is NOT in the dynamic-sources list, so we should recompile
        fn(torch.randn(4, 5))
        self.assertEqual(counter.frame_count, 2)

    @torch.compiler.config.patch(dynamic_sources="L['x']:0, L['x']:2")
    def test_dynamic_sources_per_dim_multiple(self):
        # [FUNCTION DOC] Checks per-dimension dynamic sources for dims 0 and 2; changing dim
        # 1 recompiles.
        builder._DYNAMIC_SOURCES = None

        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        # Varying dims 0 and 2 should not recompile
        fn(torch.randn(2, 4, 6))
        fn(torch.randn(3, 4, 7))
        fn(torch.randn(4, 4, 8))

        self.assertEqual(counter.frame_count, 1)

        # Varying dim 1 should recompile (not in the dynamic-sources list)
        fn(torch.randn(4, 5, 8))
        self.assertEqual(counter.frame_count, 2)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_no_dim_suffix_marks_all_dims(self):
        # [FUNCTION DOC] Checks a dynamic source without dimension suffix marks all dims dynamic;
        # one frame.
        # Backward-compat: an entry without ":N" still marks ALL dims dynamic.
        builder._DYNAMIC_SOURCES = None

        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        fn(torch.randn(2, 4))
        fn(torch.randn(3, 5))
        fn(torch.randn(4, 6))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(unbacked_sources="L['x']:0")
    def test_unbacked_sources_per_dim(self):
        # [FUNCTION DOC] Checks per-dimension unbacked sources; dim 0 variation does not
        # recompile, dim 1 does.
        builder._UNBACKED_SOURCES = None

        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x * x

        # Varying dim 0 should not recompile
        fn(torch.randn(2, 4))
        fn(torch.randn(3, 4))
        fn(torch.randn(4, 4))

        self.assertEqual(counter.frame_count, 1)

        # Varying dim 1 should recompile
        fn(torch.randn(4, 5))
        self.assertEqual(counter.frame_count, 2)

    def test_cannot_trace_mark_dynamic(self):
        # [FUNCTION DOC] Checks mark_dynamic cannot be traced inside compiled code
        # (AssertionError about forbidden callable).
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            # [FUNCTION DOC] HELPER: Calls mark_dynamic on x and squares.
            torch._dynamo.mark_dynamic(x, 0)
            return x * x

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            torch.compile(my_dyn_fn, backend="eager")(y)

    def test_cannot_trace_mark_dynamic_safe_unreached(self):
        # [FUNCTION DOC] Checks a mark_dynamic call in an unreached branch is not traced and
        # compiles fine.
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            # [FUNCTION DOC] HELPER: Returns x if first dim is 3, else prints the mark_dynamic
            # result and returns the square.
            if x.shape[0] == 3:
                return x
            print("Running", torch._dynamo.mark_dynamic(x, 0))
            return x * x

        torch.compile(my_dyn_fn, backend="eager")(y)

    def test_anomaly_aot_autograd(self):
        # [FUNCTION DOC] Checks anomaly detection information is attached: an exception in
        # backward of an allowed-in-graph function raises BackendCompilerFailed and a warning
        # mentions the forward call.
        def fail():
            # [FUNCTION DOC] HELPER: Raises AssertionError.
            raise AssertionError("fail")

        @allow_in_graph
        def h(a):
            # [FUNCTION DOC] HELPER: Allowed-in-graph function that sums a and registers a
            # backward hook which calls the failing function.
            r = a.sum()
            # Trigger an exception in backwards
            r.register_hook(lambda x: fail())
            return r

        @torch.compile(backend="aot_eager")
        def f(a):
            # [FUNCTION DOC] HELPER: Compiled wrapper calling h.
            return h(a)

        with (
            warnings.catch_warnings(record=True) as w,
            self.assertRaises(torch._dynamo.exc.BackendCompilerFailed),
        ):
            f(torch.randn(2, 2, requires_grad=True))

        # Suppress unrelated pkg_resources warnings
        self.assertIn("forward call that caused the error", str(w[-1].message))

    def test_py_guards_mark_dynamic(self):
        # [FUNCTION DOC] Checks marking different dims dynamic causes recompiles only for non-
        # subset dynamic sets: marking dim 0 gives one frame; unmarked reuses; marking dim 1
        # recompiles; after reset order tests subset logic (frames 1, 2, 3).
        def my_dyn_fn(a):
            # [FUNCTION DOC] HELPER: Returns cosine if first dim above 2 else sine.
            if a.shape[0] > 2:
                return a.cos()
            return a.sin()

        counter = CompileCounter()

        # Run with dynamic
        x0 = torch.randn([3, 3, 3])
        torch._dynamo.mark_dynamic(x0, 0)
        torch.compile(my_dyn_fn, backend=counter)(x0)
        self.assertEqual(counter.frame_count, 1)

        # Run without dynamic, no recompile
        x = torch.randn([3, 3, 3])
        torch.compile(my_dyn_fn, backend=counter)(x)
        self.assertEqual(counter.frame_count, 1)

        # Mark a new dim, 1, as dynamic
        x1 = torch.randn([3, 3, 3])
        torch._dynamo.mark_dynamic(x1, 1)
        torch.compile(my_dyn_fn, backend=counter)(x1)
        # Recompile triggered because we marked a new dym as dynamic
        self.assertEqual(counter.frame_count, 2)

        # Reset
        torch._dynamo.reset()
        # Reset counter
        counter = CompileCounter()

        # Run with dynamic 1
        torch.compile(my_dyn_fn, backend=counter)(x1)
        self.assertEqual(counter.frame_count, 1)

        # Run with dynamic 0, not subset
        torch.compile(my_dyn_fn, backend=counter)(x0)
        self.assertEqual(counter.frame_count, 2)

        # Run with dynamic 0, 1, 2, not subset
        x012 = torch.randn([3, 3, 3])
        torch._dynamo.mark_dynamic(x012, 0)
        torch._dynamo.mark_dynamic(x012, 1)
        torch._dynamo.mark_dynamic(x012, 2)
        torch.compile(my_dyn_fn, backend=counter)(x012)
        self.assertEqual(counter.frame_count, 3)

    @recover_orig_fp32_precision
    def test_recompile_on_global_state_change(self):
        # [FUNCTION DOC] Checks recompilation when global state (grad mode, deterministic
        # algorithms, tf32 flag) changes: each distinct state compiles once and is recorded by
        # the backend; state restored afterwards.
        last_state = []
        cnt = 0

        def my_compiler(gm, _):
            # [FUNCTION DOC] HELPER: Backend that counts compiles, records the global state at
            # compile time, and returns a wrapper saving it.
            nonlocal cnt
            cnt += 1
            state = read_state()

            def inner(*args):
                # [FUNCTION DOC] HELPER: Saves the state into last_state and calls the graph.
                last_state[:] = state
                return gm(*args)

            return inner

        def read_state():
            # [FUNCTION DOC] HELPER: Returns grad mode, deterministic flag, and tf32 flag as
            # a list.
            return [
                torch.is_grad_enabled(),
                torch.are_deterministic_algorithms_enabled(),
                torch._C._get_cublas_allow_tf32(),
            ]

        def write_state(state):
            # [FUNCTION DOC] HELPER: Sets the three global flags from a list.
            torch.set_grad_enabled(state[0])
            torch.use_deterministic_algorithms(state[1])
            torch._C._set_cublas_allow_tf32(state[2])

        @torch.compile(backend=my_compiler)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        initial_state = read_state()
        y = torch.randn(10)
        try:
            for round in range(3):
                for i in range(len(initial_state)):
                    new_state = [False] * len(initial_state)
                    new_state[i] = True
                    write_state(new_state)
                    if read_state() != new_state:
                        raise AssertionError(f"Expected read_state() == {new_state}")
                    last_state.clear()
                    fn(y)
                    if last_state != new_state:
                        raise AssertionError(f"Expected last_state == {new_state}")
                    if round == 0:
                        if cnt != i + 1:
                            raise AssertionError(f"Expected cnt == {i + 1}, got {cnt}")
                    else:
                        if cnt != len(initial_state):
                            raise AssertionError(
                                f"Expected cnt == {len(initial_state)}, got {cnt}"
                            )
        finally:
            write_state(initial_state)

    def test_grad_state_mutated(self):
        # [FUNCTION DOC] Checks that disabling grad mode inside a compiled function is
        # applied on exit and the state during the call is the previous one; only one frame
        # across two calls.
        prior = torch.is_grad_enabled()
        value = None
        cnt = CompileCounter()

        @torch._dynamo.allow_in_graph
        def check_state():
            # [FUNCTION DOC] HELPER: Allowed-in-graph function recording the current grad mode.
            nonlocal value
            value = torch.is_grad_enabled()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Checks state, disables grad, returns x plus 1.
            check_state()
            torch.set_grad_enabled(False)
            return x + 1

        try:
            torch.set_grad_enabled(True)
            fn(torch.randn(10))
            if value is not True:
                raise AssertionError(f"Expected value is True, got {value}")
            if torch.is_grad_enabled() is not False:
                raise AssertionError("Expected grad disabled after fn()")

            value = None
            torch.set_grad_enabled(True)
            fn(torch.randn(10))
            if value is not True:
                raise AssertionError(f"Expected value is True, got {value}")
            if torch.is_grad_enabled() is not False:
                raise AssertionError("Expected grad disabled after fn()")

            if cnt.frame_count != 1:
                raise AssertionError(f"Expected frame_count 1, got {cnt.frame_count}")
        finally:
            torch.set_grad_enabled(prior)

    def test_deterministic_algorithms_mutated(self):
        # [FUNCTION DOC] Checks deterministic algorithm mode (and warn-only) set before the call
        # is visible inside and changes made inside are applied after; one frame across two runs.
        prior = torch.are_deterministic_algorithms_enabled()
        prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        value = None
        warn_only = None
        cnt = CompileCounter()

        @torch._dynamo.allow_in_graph
        def check_state():
            # [FUNCTION DOC] HELPER: Allowed-in-graph function recording determinism flags.
            nonlocal value
            nonlocal warn_only
            value = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Records state, asserts deterministic is on, turns it off,
            # returns x plus 1.
            check_state()
            assert torch.are_deterministic_algorithms_enabled() is True  # noqa: S101
            torch.use_deterministic_algorithms(False, warn_only=False)
            return x + 1

        def run_fn():
            # [FUNCTION DOC] HELPER: Enables deterministic warn-only mode, runs fn and checks
            # recorded and final state.
            torch.use_deterministic_algorithms(True, warn_only=True)
            fn(torch.randn(10))
            if value is not True:
                raise AssertionError(f"Expected value is True, got {value}")
            if warn_only is not True:
                raise AssertionError(f"Expected warn_only is True, got {warn_only}")
            if torch.are_deterministic_algorithms_enabled() is not False:
                raise AssertionError(
                    "Expected deterministic algorithms disabled after fn()"
                )
            if torch.is_deterministic_algorithms_warn_only_enabled() is not False:
                raise AssertionError("Expected warn_only disabled after fn()")

        try:
            run_fn()
            value, warn_only = None, None
            run_fn()
            if cnt.frame_count != 1:
                raise AssertionError(f"Expected frame_count 1, got {cnt.frame_count}")
        finally:
            torch.use_deterministic_algorithms(prior, warn_only=prior_warn_only)

    def test_deterministic_algorithms_warn_only_mutated(self):
        # [FUNCTION DOC] Checks warn-only mode set inside a compiled function around scatter_add
        # is applied; result and final flags checked; one frame.
        prior = torch.are_deterministic_algorithms_enabled()
        prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        value = None
        warn_only = None
        cnt = CompileCounter()

        @torch._dynamo.allow_in_graph
        def check_state():
            # [FUNCTION DOC] HELPER: Allowed-in-graph function recording determinism flags.
            nonlocal value
            nonlocal warn_only
            value = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Enables deterministic warn-only mode, records, runs
            # scatter_add, then sets non-deterministic warn-only, returns result.
            torch.use_deterministic_algorithms(True, warn_only=True)
            check_state()
            res = x.scatter_add(
                0,
                torch.tensor([0, 1, 0]),
                torch.tensor([1.0, 2.0, 3.0]),
            )
            torch.use_deterministic_algorithms(False, warn_only=True)
            return res

        def run_fn():
            # [FUNCTION DOC] HELPER: Resets mode, runs fn, checks result and recorded and
            # final flags.
            torch.use_deterministic_algorithms(False, warn_only=False)
            result = fn(torch.zeros(2))
            self.assertEqual(result, torch.tensor([4.0, 2.0]))
            if value is not True:
                raise AssertionError(f"Expected value is True, got {value}")
            if warn_only is not True:
                raise AssertionError(f"Expected warn_only is True, got {warn_only}")
            if torch.are_deterministic_algorithms_enabled() is not False:
                raise AssertionError(
                    "Expected deterministic algorithms disabled after fn()"
                )
            if torch.is_deterministic_algorithms_warn_only_enabled() is not True:
                raise AssertionError("Expected warn_only enabled after fn()")

        try:
            run_fn()
            value, warn_only = None, None
            run_fn()
            if cnt.frame_count != 1:
                raise AssertionError(f"Expected frame_count 1, got {cnt.frame_count}")
        finally:
            torch.use_deterministic_algorithms(prior, warn_only=prior_warn_only)

    def test_deterministic_algorithms_warn_only_compile_time_value(self):
        # [FUNCTION DOC] Checks that the warn-only value passed as an argument is baked as a
        # compile-time constant in the graph call node, giving two graphs with True then False.
        prior = torch.are_deterministic_algorithms_enabled()
        prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x, warn_only):
            # [FUNCTION DOC] HELPER: Enables deterministic algorithms with the warn_only argument
            # and returns x plus 1.
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            return x + 1

        try:
            for expected_warn_only in (True, False):
                torch.use_deterministic_algorithms(False, warn_only=False)
                result = fn(torch.ones(2), expected_warn_only)
                self.assertEqual(result, torch.full((2,), 2.0))
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertEqual(
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    expected_warn_only,
                )

            self.assertEqual(len(backend.graphs), 2)
            warn_only_values = []
            for gm in backend.graphs:
                set_deterministic_nodes = [
                    node
                    for node in gm.graph.nodes
                    if node.op == "call_function"
                    and node.target is torch._C._set_deterministic_algorithms
                ]
                self.assertEqual(len(set_deterministic_nodes), 1)
                self.assertEqual(set_deterministic_nodes[0].args, (True,))
                warn_only_values.append(set_deterministic_nodes[0].kwargs["warn_only"])
            self.assertEqual(warn_only_values, [True, False])
        finally:
            torch.use_deterministic_algorithms(prior, warn_only=prior_warn_only)

    def test_torch_compile_ctx_on_forward_and_training_step(self):
        # [FUNCTION DOC] Checks the dynamo context wrapper can be applied to the forward and
        # training_step methods of a module and invoked.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a no-op forward and a training step calling itself.
            def forward(self): ...
                # [FUNCTION DOC] HELPER: Empty forward; written on one line.

            def training_step(self):
                # [FUNCTION DOC] HELPER: Calls the module itself.
                self()

        model = MyModel()
        compiled_model = torch.compile(model, backend="eager")

        model.forward = compiled_model.dynamo_ctx(model.forward)
        model.training_step = compiled_model.dynamo_ctx(model.training_step)

        model.training_step()

    def test_torch_guards_stack_frame_register_inlining(self):
        # [FUNCTION DOC] Checks the tracing context records one frame summary (the outer
        # function line) for an inlined call chain of depth one, via a patched current-frame
        # context manager.
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        def uwu_inline_me(x, y, z):
            # [FUNCTION DOC] HELPER: Concatenates and adds to produce two results.
            r = torch.cat((x, x)) + y
            r2 = torch.cat((y, y)) + z
            return r, r2

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Calls the inlined helper and squares both results.
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            # [FUNCTION DOC] HELPER: Context manager recording non-None frame summaries into the
            # enclosing list and yielding.
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(fn, backend="eager")(x, y, z)

        self.assertEqual(len(seen_frames), 1)
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[0].line, "r, r2 = uwu_inline_me(x, y, z)")

    def test_fullgraph_capture(self):
        # [FUNCTION DOC] Checks the full-graph capture API: guards built for a shape-dependent
        # function accept compatible inputs (and for static, reject others), the original
        # input passes, and the generated bytecode run with the backend graph module gives
        # eager results.
        from torch._dynamo.convert_frame import fullgraph_capture
        from torch._dynamo.utils import dynamo_timed, get_metrics_context

        def foo(x):
            # [FUNCTION DOC] HELPER: Adds or subtracts the first dim based on whether the second
            # dim is at least 3.
            if x.shape[1] >= 3:
                return x + x.shape[0]
            else:
                return x - x.shape[0]

        x = torch.randn(4, 3)
        with (
            get_metrics_context(),
            dynamo_timed(""),
        ):
            capture_output = fullgraph_capture(foo, (x,))
            graph_capture_output = capture_output.graph_capture_output
            fn = graph_capture_output.build_guards(foo.__code__)

            for guard in graph_capture_output.output_graph.guards:
                if guard.source == torch._guards.GuardSource.SHAPE_ENV:
                    dynamic = guard.code_list is not None
                    if dynamic:
                        self.assertEqual(
                            guard.code_list,
                            [
                                "L['x'].stride()[0] == L['x'].size()[1]",
                                "2 <= L['x'].size()[0]",
                                "3 <= L['x'].size()[1]",
                            ],
                        )
                        self.assertTrue(
                            fn.guard_manager.check({"x": torch.randn(3, 3)})
                        )
                        self.assertTrue(
                            fn.guard_manager.check({"x": torch.randn(4, 4)})
                        )
                    else:
                        self.assertFalse(
                            fn.guard_manager.check({"x": torch.randn(3, 3)})
                        )
                        self.assertFalse(
                            fn.guard_manager.check({"x": torch.randn(4, 4)})
                        )
                    self.assertFalse(fn.guard_manager.check({"x": torch.randn(4, 2)}))
                    self.assertFalse(fn.guard_manager.check({"x": torch.randn(1, 3)}))
                    break

            backend_input = capture_output.backend_input
            self.assertTrue(fn.guard_manager.check({"x": x}))
        import_sources = {
            alias: importlib.import_module(module_name)
            for alias, module_name in graph_capture_output.import_sources.items()
        }
        self.assertEqual(
            foo(x),
            types.FunctionType(
                graph_capture_output.bytecode,
                {
                    **import_sources,
                    backend_input.backend_id: backend_input.graph_module,
                },
            )(x),
        )

    def test_torch_guards_stack_frame_register_inlining_deep(self):
        # [FUNCTION DOC] Same as the previous frame-summary test but with a deeper chain (three
        # summaries: outer, middle, and the line of the second deep call).
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        def uwu_inline_me_deep(x, y):
            # [FUNCTION DOC] HELPER: Concatenates x with itself and adds y.
            return torch.cat((x, x)) + y

        def uwu_inline_me(x, y, z):
            # [FUNCTION DOC] HELPER: Calls the deep helper twice.
            r = uwu_inline_me_deep(x, y)
            r2 = uwu_inline_me_deep(y, z)
            return r, r2

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Calls the middle helper and squares both results.
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            # [FUNCTION DOC] HELPER: Context manager recording non-None frame summaries
            # and yielding.
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(fn, backend="eager")(x, y, z)

        self.assertEqual(len(seen_frames), 3)
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[1].name, "uwu_inline_me")
        self.assertEqual(seen_frames[2].line, "r2 = uwu_inline_me_deep(y, z)")

    def test_recompile_on_disable_1(self):
        # [FUNCTION DOC] Regression test for issue 157399: calling a function that defines
        # and calls a disabled inner function five times with error on recompile on must not
        # raise RecompileError.
        # fix https://github.com/pytorch/pytorch/issues/157399
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Defines a disabled inner adding 10 and returns its result
            # plus 1.
            @torch._dynamo.disable
            def inner(x):
                # [FUNCTION DOC] HELPER: Returns x plus 10.
                return x + 10

            return inner(x) + 1

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            try:
                for i in range(5):
                    fn(torch.rand(2, 3))
            except torch._dynamo.exc.RecompileError as e:
                self.fail("RecompileError raised unexpectedly: " + str(e))

    def test_recompile_on_disable_2(self):
        # [FUNCTION DOC] Defines functions for a recompile-on-disable scenario but never
        # calls them.
        # KNOWN ISSUE: the test body does not invoke the outer function or compile anything, so
        # it asserts nothing.
        def outer(x, cond):
            # [FUNCTION DOC] HELPER: Chooses one of two disabled functions by cond, graph breaks,
            # then calls it.
            @torch._dynamo.disable()
            def fn0(y):
                # [FUNCTION DOC] HELPER: Disabled function returning y plus 1.
                return y + 1

            @torch._dynamo.disable()
            def fn1(y):
                # [FUNCTION DOC] HELPER: Disabled function returning y plus 2.
                return y + 2

            if cond:
                f = fn0
            else:
                f = fn1

            torch._dynamo.graph_break()
            # there will be a resume function here
            return f(x)

    def test_error_on_recompile(self):
        # [FUNCTION DOC] Checks that with error on recompile enabled, calling a function with a
        # different argument type (tuple instead of tensor) raises RecompileError.
        @torch.compile(backend="eager")
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            with self.assertRaises(torch._dynamo.exc.RecompileError):
                fn(torch.rand(2, 3), torch.rand(2, 3))
                fn(torch.rand(2, 3), (1, 2, 3))

    def test_guards_strip_function_call(self):
        # [FUNCTION DOC] Checks the helper that strips function calls and extracts the base
        # object name from guard name strings for several patterns.
        from torch._dynamo.guards import strip_function_call

        test_case = [
            ("___odict_getitem(a, 1)", "a"),
            ("a.layers[slice(2)][0]._xyz", "a"),
            ("getattr(a.layers[slice(2)][0]._abc, '0')", "a"),
            ("getattr(getattr(a.x[3], '0'), '3')", "a"),
            ("a.layers[slice(None, -1, None)][0]._xyz", "a"),
            ("a.layers[func('offset', -1, None)][0]._xyz", "a"),
        ]
        # strip_function_call should extract the object from the string.
        for name, expect_obj in test_case:
            self.assertEqual(strip_function_call(name), expect_obj)

    def test_int_neg(self):
        # [FUNCTION DOC] Standard test helper for negating shape-derived ints.
        def int_neg(a, b):
            # [FUNCTION DOC] HELPER: Multiplies negated sizes by the tensors.
            x = a.shape[0]
            y = b.shape[0]
            return -x * -y * a * b

        torch._dynamo.testing.standard_test(self, int_neg, 2)

    def test_hash_getitem_slice(self):
        # [FUNCTION DOC] Checks source objects for slices hash and compare equal for equal slices
        # and differ for different step.
        s = GetItemSource(LocalSource("foo"), slice(None, -1, None))
        s2 = GetItemSource(LocalSource("foo"), slice(None, -1, None))
        s3 = GetItemSource(LocalSource("foo"), slice(None, -1, 2))
        some_set = set()

        self.assertTrue(s not in some_set)
        self.assertTrue(s2 not in some_set)
        self.assertTrue(s3 not in some_set)

        some_set.add(s)

        self.assertTrue(s in some_set)
        # s and s2 should hash the  same
        self.assertTrue(s2 in some_set)
        # s3 should be different
        self.assertTrue(s3 not in some_set)

        self.assertTrue(s == s2)
        self.assertTrue(s != s3)

    def test_inline_dict_function(self):
        # [FUNCTION DOC] Checks a dict lookup helper keyed by a type is inlined and gives a
        # float32 ones tensor.
        def _result_type_dict(dtype):
            # [FUNCTION DOC] HELPER: Looks up the dtype for a python type in a dict.
            return {bool: torch.float32}[dtype]

        @torch.compile(backend="eager")
        def f():
            # [FUNCTION DOC] HELPER: Creates ones with the looked-up dtype.
            return torch.ones(3, dtype=_result_type_dict(bool))

        self.assertEqual(f(), torch.ones(3, dtype=torch.float32))

    def test_inline_dict_function_passed_as_arg(self):
        # [FUNCTION DOC] Checks dict passed as argument and indexed by a type key chooses cosine
        # or sine.
        @torch.compile(backend="eager")
        def fn(d, x, y):
            # [FUNCTION DOC] HELPER: Branches on whether da one-element list holding x is float32
            # and returns cosine or sine of y.
            if d[x] is torch.float32:
                return y.cos()
            else:
                return y.sin()

        dd = {bool: torch.float32, int: torch.int64}
        self.assertEqual(fn(dd, bool, torch.ones(4)), torch.ones(4).cos())
        self.assertEqual(fn(dd, int, torch.ones(4)), torch.ones(4).sin())

    def test_add_sizes(self):
        # [FUNCTION DOC] Checks adding two shapes yields a Size in compiled code.
        def func(x):
            # [FUNCTION DOC] HELPER: Returns the size added to itself.
            y = x.size()
            return y + y

        eager_out = func(torch.ones(10, 10, 3))
        compile_out = torch.compile(func, backend="eager")(torch.ones(10, 10, 3))
        self.assertTrue(isinstance(compile_out, torch.Size))
        self.assertEqual(eager_out, compile_out)

    def test_nested_function_resuming_with_correct_globals(self):
        # [FUNCTION DOC] Regression test (issue 99665): a nested function from another module
        # wrapping a local function resumes with correct globals.
        # https://github.com/pytorch/pytorch/issues/99665
        try:
            from .utils import outer_func
        except ImportError:
            from utils import outer_func

        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the imported wrapper on gn.
            return outer_func(gn)(x, y)

        x = torch.rand([3])
        y = torch.rand([3])
        opt_fn = torch.compile(backend="eager")(fn)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_recursion_depth_guards(self):
        # [FUNCTION DOC] Checks compiling a function called with 2000 positional args under
        # dynamic shapes does not crash on recursion depth guards.
        @torch.compile(dynamic=True, backend="eager")
        def foo(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Returns 0 if sum of args is zero else 1.
            if sum(args) == 0:
                return 0
            return 1

        args = list(range(2000))
        foo(*args)
        # Previously would have crashed

    @dataclasses.dataclass
    class CSETestCase:
        # [CLASS DOC] HELPER: Dataclass test-case record with an expression string, a preface
        # list and an optional expected string; used by the CSE tests.
        expr: str
        preface: typing.List[str] = dataclasses.field(default_factory=list)
        expected: typing.Optional[str] = None

    def test_guards_cse_pass_single(self):
        # [FUNCTION DOC] Checks the guard expression common-subexpression-elimination pass on
        # single expressions: repeated sub-expressions beyond the threshold are hoisted into
        # numbered temporaries and replaced; others are left alone.
        from torch._dynamo.guards import PyExprCSEPass

        testcase = self.CSETestCase
        testcases = [
            # Nothing gets CSE-d, since the only repeated sub-expression is 'x'.
            # i.e. not a node type we are interested on.
            testcase(expr="x[0].a"),
            testcase(expr="x[1].a"),
            testcase(expr="x[2].a"),
            # 'a.b.c' gets CSE-d, since it's a sub-expression used more than 'PyExprCSEPass.USE_THRESHOLD'.
            testcase(
                expr="a.b.c[0].d.e",
                preface=["_var0 = a.b", "_var1 = _var0.c"],
                expected="_var1[0].d.e",
            ),
            testcase(expr="a.b.c[1].d.e", expected="_var1[1].d.e"),
            testcase(expr="a.b.c[2].d.e", expected="_var1[2].d.e"),
            # 'm.n[0]' gets CSE-d, since it is a sub-expression used more than 'PyExprCSEPass.USE_THRESHOLD'.
            testcase(
                expr="f(m.n[0], '0').x.y.z",
                preface=["_var2 = m.n", "_var3 = _var2[0]"],
                expected="f(_var3, '0').x.y.z",
            ),
            testcase(expr="f(m.n[0], '1').x.y.z", expected="f(_var3, '1').x.y.z"),
            testcase(expr="f(m.n[0], '2').x.y.z", expected="f(_var3, '2').x.y.z"),
            # The whole expression gets CSE-d, as well as all of its sub-expressions.
            testcase(
                expr="self.g(a, b).k",
                preface=["_var4 = self.g", "_var5 = _var4(a, b)", "_var6 = _var5.k"],
                expected="_var6",
            ),
            testcase(expr="self.g(a, b).k", expected="_var6"),
            testcase(expr="self.g(a, b).k", expected="_var6"),
        ]
        csepass = PyExprCSEPass()
        csepass.count([t.expr for t in testcases])

        for t in testcases:
            preface, expr = csepass.replace(t.expr)
            self.assertEqual(preface, t.preface)
            expected = t.expected if t.expected is not None else t.expr
            self.assertEqual(expr, expected)

    def test_guards_cse_pass_multiple(self):
        # [FUNCTION DOC] Checks the CSE pass on compound expressions combining
        # several sub-expressions.
        from torch._dynamo.guards import PyExprCSEPass

        testcase = self.CSETestCase
        testcases = [
            testcase(
                expr="x[0].a < x[1].a * (3 - x[2].a)",
                expected="x[0].a < x[1].a * (3 - x[2].a)",
            ),
            testcase(
                expr="a.b.c[0].d.e + a.b.c[1].d.e * a.b.c[2].d.e > 0",
                preface=["_var0 = a.b", "_var1 = _var0.c"],
                expected="_var1[0].d.e + _var1[1].d.e * _var1[2].d.e > 0",
            ),
            testcase(
                expr="f(m.n[0], '0').x.y.z * f(m.n[0], '1').x.y.z * f(m.n[0], '2').x.y.z < 512",
                preface=["_var2 = m.n", "_var3 = _var2[0]"],
                expected="f(_var3, '0').x.y.z * f(_var3, '1').x.y.z * f(_var3, '2').x.y.z < 512",
            ),
            testcase(
                expr="self.g(a, b).k + (1 - self.g(a, b).k) <= m[0].a + self.g(a, b).k",
                preface=["_var4 = self.g", "_var5 = _var4(a, b)", "_var6 = _var5.k"],
                expected="_var6 + (1 - _var6) <= m[0].a + _var6",
            ),
        ]

        csepass = PyExprCSEPass()
        csepass.count([t.expr for t in testcases])

        for t in testcases:
            preface, expr = csepass.replace(t.expr)
            self.assertEqual(preface, t.preface)
            expected = t.expected
            expected = expected if expected is not None else t.expr
            self.assertEqual(expr, expected)

    def test_guard_function_builder_with_cse(self):
        # [FUNCTION DOC] Checks building a guard function with CSE produces the exact expected
        # source text with temporaries defined before the check that uses them.
        from torch._dynamo.guards import build_guard_function

        exprs = [
            "x[0].a < x[1].a * (3 - x[2].a)",
            "a.b.c[0].d.e + a.b.c[1].d.e * a.b.c[2].d.e > 0",
            "f(m.n[0], '0').x.y.z * f(m.n[0], '1').x.y.z * f(m.n[0], '2').x.y.z < 512",
            "self.g(a, b).k + (1 - self.g(a, b).k) <= m[0].a + self.g(a, b).k",
        ]

        _, pycode = build_guard_function(exprs, "")
        expected = """\
def ___make_guard_fn():
    def guard(L):
        if not (x[0].a < x[1].a * (3 - x[2].a)):
            return False
        _var0 = a.b
        _var1 = _var0.c
        if not (_var1[0].d.e + _var1[1].d.e * _var1[2].d.e > 0):
            return False
        _var2 = m.n
        _var3 = _var2[0]
        if not (f(_var3, '0').x.y.z * f(_var3, '1').x.y.z * f(_var3, '2').x.y.z < 512):
            return False
        _var4 = self.g
        _var5 = _var4(a, b)
        _var6 = _var5.k
        if not (_var6 + (1 - _var6) <= m[0].a + _var6):
            return False
        return True
    return guard
"""

        self.assertEqual(expected, pycode)

    def test_dynamo_compiling_fake_tensor_to_vararg_int(self):
        # [FUNCTION DOC] Checks a module reshaping with a numpy int shape (wrapped as a fake
        # tensor) compiles and matches eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reshaping x with a numpy integer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base init.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reshapes x to numpy int 16.
                # use numpy int so it's wrapped as fake tensor in dynamo
                shape = np.int_(16)
                # test shape as fake tensor, which param type is
                # Sequence[Union[_int, SymInt]]
                return x.reshape(shape)

        x = torch.rand([4, 4])
        model = MyModule()
        orig_out = model(x)
        opt_model = torch.compile(MyModule(), backend="eager")
        opt_out = opt_model(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_compile_with_userland_fake_tensor_mode(self):
        # [FUNCTION DOC] Checks compile inside a user fake tensor mode works: forward and
        # backward on a linear module with aot_eager.
        # Test that torch.compile works when called inside a user's FakeTensorMode.
        # The user's fake tensors should be "refakified" to Dynamo's fake mode.
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            model = torch.nn.Linear(4, 4)
            inp = torch.rand(4, 4)
            loss = torch.compile(model, backend="aot_eager")(inp).sum()
            loss.backward()

    def test_scalar_tensor_is_equivalent_to_symint_argument(self):
        # [FUNCTION DOC] Checks a scalar tensor parameter used as top-k size argument compiles
        # and matches eager.
        class GumbelTopKSampler(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with temperature and k stored as frozen parameters.
            def __init__(self, T, k):
                # [FUNCTION DOC] HELPER: Stores T and k as non-grad parameters.
                super().__init__()
                self.T = torch.nn.Parameter(
                    torch.tensor(T, dtype=torch.float32), requires_grad=False
                )
                self.k = torch.nn.Parameter(
                    torch.tensor(k, dtype=torch.int32), requires_grad=False
                )

            def sample_discrete(self, logits):
                # [FUNCTION DOC] HELPER: Takes top-k threshold along last dim and returns float
                # mask of comparison.
                threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
                samples = torch.ge(logits.squeeze(1), threshold).float()
                return samples

            def forward(self, logits):
                # [FUNCTION DOC] HELPER: Returns sampled mask.
                dsamples = self.sample_discrete(logits)
                return dsamples

        x = torch.rand([4, 4, 4, 4])
        m = GumbelTopKSampler(T=4, k=4)
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_symint_list_argument(self):
        # [FUNCTION DOC] Checks scalar jitter values used as roll shift lists compile and
        # match eager.
        class Jitter(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a jitter value.
            def __init__(self, jitter_val):
                # [FUNCTION DOC] HELPER: Stores jitter.
                super().__init__()
                self.jitter_val = jitter_val

            def roll_tensor(self, input):
                # [FUNCTION DOC] HELPER: Rolls the input along dims 2 and 3 by jitter minus 1 and
                # plus 1.
                h_shift = self.jitter_val - 1
                w_shift = self.jitter_val + 1
                return torch.roll(
                    torch.roll(input, shifts=h_shift, dims=2), shifts=w_shift, dims=3
                )

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns rolled input.
                return self.roll_tensor(input)

        x = torch.rand([4, 4, 4, 4])
        m = Jitter(jitter_val=4)
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_int_list_argument(self):
        # [FUNCTION DOC] Checks a tensor used as permutation list unpacked into permute args
        # matches eager.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module permuting dims with an unpacked tensor.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns input permuted by 0, 2, 1.
                permute = torch.tensor([0, 2, 1])
                x = input.permute(*permute)
                return x

        x = torch.randn(2, 3, 4)
        m = MyModel()
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_torch_variable_hasattr(self):
        # [FUNCTION DOC] Checks hasattr on the torch.nn namespace under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x squared if torch.nn has Module else x plus 1.
            if hasattr(torch.nn, "Module"):
                return x * x
            return x + 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = torch.rand([4, 4])
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

    def test_list_hasattr1(self):
        # [FUNCTION DOC] Checks hasattr on a list input for a missing attribute.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns first element plus 1 if list has foo, else minus 1.
            if hasattr(x, "foo"):
                return x[0] + 1
            return x[0] - 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = [torch.randn(3)]
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

    def test_list_hasattr2(self):
        # [FUNCTION DOC] Checks hasattr for the length method on a list created inside.
        def fn():
            # [FUNCTION DOC] HELPER: Returns first element plus or minus 1 based on hasattr.
            x = [torch.zeros(3)]
            if hasattr(x, "__len__"):
                return x[0] + 1
            return x[0] - 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertTrue(same(fn_out, compiled_out))

    def test_tuple_hasattr(self):
        # [FUNCTION DOC] Checks hasattr on a tuple input for a missing attribute.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns first plus 1 if tuple has foo, else second minus 1.
            if hasattr(x, "foo"):
                return x[0] + 1
            return x[1] - 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = (torch.randn(3), torch.randn(3))
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

    def test_fn_hasattr__name__1(self):
        # [FUNCTION DOC] Checks a lambda has a name attribute inside compiled code.
        def fn():
            # [FUNCTION DOC] HELPER: Returns whether a lambda has a name.
            foo = lambda x: x + 1
            return hasattr(foo, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    def test_fn_hasattr__name__2(self):
        # [FUNCTION DOC] Checks a user function has a name attribute.
        def bar(x):
            # [FUNCTION DOC] HELPER: Returns sine of x.
            return torch.sin(x)

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether bar has a name.
            return hasattr(bar, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    def test_fn_hasattr__name__3(self):
        # [FUNCTION DOC] Checks a partial object has no name attribute.
        def bar(x, y):
            # [FUNCTION DOC] HELPER: Returns sum of sine and cosine.
            return torch.sin(x) + torch.cos(y)

        baz = functools.partial(bar, y=4)

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the partial has a name.
            return hasattr(baz, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertFalse(fn_out)

    def test_constant_hasattr_returns_bool(self):
        # [FUNCTION DOC] Checks hasattr on string, int and float constants gives true, false
        # pairs as a tuple.
        """Test that hasattr on constant values properly returns boolean ConstantVariable."""

        # Test various constant types
        def fn():
            # [FUNCTION DOC] HELPER: Computes six hasattr results on constants and returns them.
            # String constant
            s = "hello"
            result1 = hasattr(s, "upper")  # True
            result2 = hasattr(s, "nonexistent")  # False

            # Integer constant
            i = 42
            result3 = hasattr(i, "bit_length")  # True
            result4 = hasattr(i, "fake_method")  # False

            # Float constant
            f = 3.14
            result5 = hasattr(f, "is_integer")  # True
            result6 = hasattr(f, "missing_attr")  # False

            # Use all results to ensure they're compiled
            return (result1, result2, result3, result4, result5, result6)

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertEqual(fn_out, (True, False, True, False, True, False))

    def test_class_hasattr_sourceless_descriptor(self):
        # [FUNCTION DOC] Checks that hasattr on an owner class within a descriptor getter does
        # not graph break, for a class with and without a flag attribute.
        """Test that hasattr on sourceless UserDefinedClassVariable does not graph break."""

        class FlagDescriptor:
            # [CLASS DOC] HELPER: Descriptor returning 1 if the owner class has flag, else 0.
            def __get__(self, instance, owner):
                # [FUNCTION DOC] HELPER: Returns 1 or 0 depending on owner flag.
                if hasattr(owner, "flag"):
                    return 1
                return 0

        class WithFlag:
            # [CLASS DOC] HELPER: Class with flag and descriptor attribute.
            flag = True
            prop = FlagDescriptor()

        class WithoutFlag:
            # [CLASS DOC] HELPER: Class with descriptor but no flag.
            prop = FlagDescriptor()

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Adds the descriptor value to x.
            return x + obj.prop

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        x = torch.randn(3)
        self.assertEqual(fn(x, WithFlag()), compiled_fn(x, WithFlag()))
        self.assertEqual(fn(x, WithoutFlag()), compiled_fn(x, WithoutFlag()))

    def test_torch_objects_as_keys(self):
        # [FUNCTION DOC] Checks a dict keyed by a torch dtype is read in compiled code.
        remap = {torch.float16: torch.float32}

        def fn():
            # [FUNCTION DOC] HELPER: Returns random tensor with dtype looked up from dict.
            return torch.randn(3, dtype=remap[torch.float16])

        opt = torch.compile(fn, backend="eager")
        opt()

    def test_dynamic_one_hot(self):
        # [FUNCTION DOC] Checks one_hot has data-dependent output shape causing a graph break: 2
        # frames and 2 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1, one_hot, adds 1.
            x = x + 1
            # graph break from data-dependent output shape
            x = torch.nn.functional.one_hot(x)
            x = x + 1
            return x

        inp = torch.arange(20) % 4
        counter = CompileCounter()
        real_out = fn(inp)
        comp_out = torch.compile(fn, backend=counter)(inp)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 2)

    def test_jacfwd_one_hot_dynamic_compile(self):
        # [FUNCTION DOC] Checks jacobian forward-mode of a function using one_hot compiles
        # dynamically and matches eager.
        import torch.nn.functional as F

        MAX, BATCH = 3, 37

        def func(x, idxs):
            # [FUNCTION DOC] HELPER: Squares x and multiplies by one-hot of idxs.
            return x.square() * F.one_hot(idxs, MAX)

        def jacfunc(x, idxs):
            # [FUNCTION DOC] HELPER: Computes forward jacobian of func wrt x.
            return torch.func.jacfwd(func, argnums=(0,))(x, idxs)

        idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
        x = torch.rand((BATCH, MAX), dtype=torch.float64)
        eager = jacfunc(x, idxs)

        compiled = torch.compile(jacfunc, backend="eager", dynamic=True)
        out_comp = compiled(x, idxs)
        self.assertEqual(eager[0], out_comp[0])

    def test_tracing_nested_py_tree_mixed_all(self):
        # [FUNCTION DOC] Checks tree flatten and unflatten on a nested mix of dict, tuple and
        # list under fullgraph: one frame, 18 ops.
        def fn(xs):
            # [FUNCTION DOC] HELPER: Flattens inputs, clones leaves, unflattens.
            flat_xs, spec = python_pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            return python_pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}
        xsl = {
            "a": xs,
            "b": xsa,
            "c": xsb,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 18)

    def test_any_all_symnode(self):
        # [FUNCTION DOC] Checks any and all on symbolic booleans under dynamic shapes: one frame
        # for sizes 16 and 18 and another for size 5.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Decision chain using any and all of empty lists and symbolic
            # booleans and returns x shifted by a constant.
            t = x.size(0) >= 10
            f = x.size(0) >= 100
            if any([]) or any([f]) or any([f, f]):
                return x - 1
            if all([f]) or all([t, f]) or all([f, t]) or all([f, f]):
                return x - 2
            if not (all([]) and all([t]) and all([t, t])):
                return x - 3
            if not (any([t]) and any([t, f]) and any([f, t])):
                return x - 4
            return x + 1

        y1 = torch.randn(16)
        y2 = torch.randn(18)
        self.assertEqual(fn(y1), y1 + 1)
        self.assertEqual(fn(y2), y2 + 1)
        self.assertEqual(cnt.frame_count, 1)
        y3 = torch.randn(5)
        self.assertEqual(fn(y3), y3 - 3)
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_symint_split(self):
        # [FUNCTION DOC] With scalar and dynamic output capture, checks splitting by sizes from
        # tolist compiles.
        @torch.compile(backend="eager")
        def f(lengths, values):
            # [FUNCTION DOC] HELPER: Converts lengths to a list and splits values.
            sizes = lengths.tolist()
            return torch.split(values, sizes)

        f(torch.tensor([2, 3, 4]), torch.randn(9))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_out_variant_custom_op(self):
        # [FUNCTION DOC] With scalar and dynamic output capture, checks custom ops with an out
        # list or out tensor argument can be called under fullgraph (split with sizes copy,
        # chunk cat).
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
            )

            @torch.library.impl(lib, "split_with_sizes_copy", "Meta")
            @torch.library.impl(lib, "split_with_sizes_copy", "CPU")
            def split_with_sizes_copy(
                all_gather_output: torch.Tensor,
                all_gather_input_split_sizes: typing.List[int],
                dim: int,
                out: typing.List[torch.Tensor],
            ) -> None:
                # [FUNCTION DOC] HELPER: Implementation for meta and CPU calling
                # torch.split_with_sizes_copy writing into out.
                torch.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
                )

            @torch.compile(backend="eager", fullgraph=True)
            def f1(all_gather_output, all_gather_input_split_sizes, dim, out):
                # [FUNCTION DOC] HELPER: Calls the custom split op.
                return torch.ops.mylib.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim, out=out
                )

            all_gather_output = torch.randn(2, 272)
            all_gather_input_split_sizes = [128, 8, 128, 8]
            dim = 1
            out = [
                torch.empty(2, 128),
                torch.empty(2, 8),
                torch.empty(2, 128),
                torch.empty(2, 8),
            ]
            f1(all_gather_output, all_gather_input_split_sizes, dim, out)

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
            )

            @torch.library.impl(lib, "chunk_cat", "Meta")
            @torch.library.impl(lib, "chunk_cat", "CPU")
            def chunk_cat(
                tensors: typing.List[torch.Tensor],
                dim: int,
                num_chunks: int,
                out: torch.Tensor,
            ) -> None:
                # [FUNCTION DOC] HELPER: Implementation for meta and CPU calling torch._chunk_cat
                # with out.
                torch._chunk_cat(tensors, dim, num_chunks, out=out)

            @torch.compile(backend="eager", fullgraph=True)
            def f2(tensors, dim, num_chunks, out):
                # [FUNCTION DOC] HELPER: Calls the custom chunk cat op.
                return torch.ops.mylib.chunk_cat(tensors, dim, num_chunks, out=out)

            x = torch.zeros(100, dtype=torch.int64)
            tensors = [
                torch.randn(16, 16),
                torch.randn(16),
                torch.randn(16, 16),
                torch.randn(16),
            ]
            dim = 0
            num_chunks = 2
            out = torch.empty(2, 272)
            f2(tensors, dim, num_chunks, out)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_runtime_assert_replacement(self):
        # [FUNCTION DOC] With scalar capture, checks runtime assertion on item equals 3: passes
        # for 3, raises RuntimeError for 4.
        @torch.compile(backend="eager")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Reads item, checks it equals 3, returns x plus it.
            z = y.item()
            torch._check(z == 3)
            return x + z

        fn(torch.randn(4), torch.tensor([3]))
        self.assertRaises(RuntimeError, lambda: fn(torch.randn(4), torch.tensor([4])))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked(self):
        # [FUNCTION DOC] With scalar capture, checks concatenation with a ones tensor of unbacked
        # size raises RuntimeError for sizes 0 and 1.
        @torch.compile(backend="eager")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Reads item and concatenates x with ones of that size.
            z = y.item()
            return torch.cat([x, torch.ones(z)])

        self.assertRaises(
            RuntimeError, lambda: fn(torch.randn(2, 3), torch.tensor([0]))
        )
        self.assertRaises(
            RuntimeError, lambda: fn(torch.randn(2, 3), torch.tensor([1]))
        )

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_aot_autograd_propagate_unbacked_symints_shape(self):
        # [FUNCTION DOC] With scalar and dynamic output capture, checks aot_eager handles
        # nonzero's unbacked shape.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Returns nonzero of x.
            return torch.nonzero(x)

        f(torch.tensor([1, 0, 3, 2, 0]))

    def test_simple_set_usage(self):
        # [FUNCTION DOC] Checks building a set from two tensors and popping twice, under
        # fullgraph; one frame.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Creates a set, pops two items and multiplies them.
            setty = {x, y}
            return setty.pop() * setty.pop()

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter, fullgraph=True)
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        foo(x, y)
        self.assertEqual(counter.frame_count, 1)

    def test_add_to_set(self):
        # [FUNCTION DOC] Checks adding tensors to a set (duplicates collapse) and using its
        # length; one frame and result matches eager.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Adds three list elements and y to a set and scales y by
            # its length.
            setty = set()
            setty.add(x[0])
            setty.add(x[1])
            setty.add(x[2])
            setty.add(y)
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)
        eager_result = foo([x, x, x, x, y], y)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter, fullgraph=True)
        result = foo([x, x, x, x, y], y)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(result, eager_result)

    def test_remove_set(self):
        # [FUNCTION DOC] Checks removing an element from a set.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates set, removes 4, scales x by its length.
            set_a = set((4, 5))
            set_a.remove(4)
            return x * len(set_a)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_iter_set(self):
        # [FUNCTION DOC] Checks iterating a list and adding to a set.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Adds each element to a set and scales y by its length.
            setty = set()
            for t in x:
                setty.add(t)
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)
        eager_result = foo([x, x, x, x, y], y)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter, fullgraph=True)
        result = foo([x, x, x, x, y], y)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(result, eager_result)

    def test_reconstruct_set_across_graph_break(self):
        # [FUNCTION DOC] Checks a set is reconstructed across a graph break caused by print.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Builds set from list, prints, returns y scaled by
            # set length.
            setty = set()
            for t in x:
                setty.add(t)
            print("Break!")
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter)
        result = foo([x, x, x, x, y], y)

    def test_set_aliasing_recompiles(self):
        # [FUNCTION DOC] Checks recompilation when the aliasing pattern between a global tensor
        # and arguments changes (frames counted at each step, including after reset).
        g1 = torch.randn(10)
        g2 = torch.randn(10)
        g3 = torch.randn(10)
        g4 = torch.randn(10)

        def foo(a, b, c):
            # [FUNCTION DOC] HELPER: Builds a set of a global and three arguments and adds its
            # length to a.
            myset = {g1, a, b, c}
            return a + len(myset)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter)
        # first call with no aliasing
        foo(g2, g3, g4)
        self.assertEqual(counter.frame_count, 1)

        # no aliasing again
        foo(g3, g2, g4)
        # assert no recompile
        self.assertEqual(counter.frame_count, 1)

        # aliasing changes, we should recompile
        foo(g2, g2, g2)
        self.assertEqual(counter.frame_count, 2)

        # same aliasing, different tensor
        foo(g3, g3, g3)
        self.assertEqual(counter.frame_count, 2)

        # aliasing between global and arg, should recompile again
        foo(g1, g1, g1)
        self.assertEqual(counter.frame_count, 3)

        # Reset
        torch._dynamo.reset()

        # aliasing between global and arg, first call
        foo(g1, g1, g1)
        self.assertEqual(counter.frame_count, 4)

        # same aliasing, different tensor, all local, recompile
        foo(g3, g3, g3)
        self.assertEqual(counter.frame_count, 5)

        # aliasing same tensor, we shouldn't recompile
        foo(g2, g2, g2)
        self.assertEqual(counter.frame_count, 5)

        # No aliasing
        foo(g2, g3, g4)
        self.assertEqual(counter.frame_count, 6)

        # No aliasing again
        foo(g3, g2, g4)
        # assert no recompile
        self.assertEqual(counter.frame_count, 6)

    def test_str_format_return1(self):
        # [FUNCTION DOC] Checks a formatted string with shape slice and batch size built in
        # compiled code and returned along with the tensor result.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            # [FUNCTION DOC] HELPER: Computes sine, builds a message string from shape info,
            # returns the sum and message.
            x = torch.sin(img)
            y = f"shape {img.shape[-2:]} batch size {img.shape[0]}"
            return img + x, y

        img1 = torch.randn(1, 1, 8, 8)
        res, msg = fn(img1)
        self.assertEqual(msg, "shape torch.Size([8, 8]) batch size 1")
        self.assertEqual(res, img1 + torch.sin(img1))

    def test_str___iter__(self):
        # [FUNCTION DOC] Checks calling the iterator method on a string constant and taking next.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if the first character is a.
            s = "a"
            if next(s.__iter__()) == "a":
                return x + 1
            else:
                return x

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_str_format_return2(self):
        # [FUNCTION DOC] Checks the format method with positional and keyword fields (with float
        # format spec) of shape values.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            # [FUNCTION DOC] HELPER: Computes sine, builds a formatted message, returns the sum
            # and message.
            x = torch.sin(img)
            y = "shape {} batch size {y:.2f}".format(img.shape[-2:], y=img.shape[0])
            return img + x, y

        img1 = torch.randn(1, 1, 8, 8)
        res, msg = fn(img1)
        self.assertEqual(msg, "shape torch.Size([8, 8]) batch size 1.00")
        self.assertEqual(res, img1 + torch.sin(img1))

    def test_sourceless_namedtuple(self):
        # [FUNCTION DOC] Checks constructing a wrapper tensor subclass inside compiled code
        # carrying a namedtuple built in the function (without a source).
        from collections import namedtuple

        CustomDtype = namedtuple("CustomDtype", ["dtype", "higher_dtype"])

        class CustomTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass storing inner data and a custom dtype
            # namedtuple, flattening via metadata, passing dispatch through.
            _data: torch.Tensor
            custom_dtype: CustomDtype
            __torch_function__ = torch._C._disabled_torch_function_impl
            __slots__ = [
                "_data",
                "custom_dtype",
            ]

            def __new__(
                cls,
                data: torch.Tensor,
                custom_dtype: CustomDtype,
            ):
                # [FUNCTION DOC] HELPER: Creates a wrapper subclass with the custom dtype and
                # stores the inner data and dtype.
                self = torch.Tensor._make_wrapper_subclass(
                    cls,
                    data.size(),
                    strides=data.stride(),
                    storage_offset=data.storage_offset(),
                    dtype=custom_dtype.dtype,
                    layout=data.layout,
                    requires_grad=data.requires_grad,
                    device=data.device,
                )
                self._data = data
                self.custom_dtype = custom_dtype
                return self

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Returns inner tensor names and metadata with the
                # custom dtype.
                meta = {
                    "custom_dtype": self.custom_dtype,
                }
                return ["_data"], meta

            @staticmethod
            def __tensor_unflatten__(
                inner_tensors: dict, metadata, outer_size, outer_stride
            ):
                # [FUNCTION DOC] HELPER: Rebuilds the subclass from inner tensor and metadata.
                return CustomTensor(
                    inner_tensors["_data"],
                    metadata["custom_dtype"],
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs={}):
                # [FUNCTION DOC] HELPER: Calls the function directly.
                # KNOWN ISSUE: uses a mutable default dict for kwargs; harmless because it is
                # never mutated.
                return func(*args, **kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the subclass from x with a namedtuple dtype and
            # returns it with the dtype.
            y = CustomTensor(x, CustomDtype(torch.float32, torch.bfloat16))
            return y, y.custom_dtype

        fn(torch.ones(2, 2, device="cpu"))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_validate_outputs_unbacked(self):
        # [FUNCTION DOC] With scalar capture, checks a custom autograd function with unbacked
        # split sizes in backward compiles with aot_eager.
        class SillyCat(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function concatenating two tensors; backward splits
            # the gradient by sizes saved as a tensor.
            @staticmethod
            def forward(ctx, x0, x1, i):
                # [FUNCTION DOC] HELPER: Saves i and returns concatenation.
                ctx.save_for_backward(i)
                return torch.cat([x0, x1])

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Loads saved sizes, splits the gradient and returns
                # pieces plus None.
                (i,) = ctx.saved_tensors
                i0, i1 = i.tolist()
                g_x0, g_x1 = grad_out.split([i0, i1])
                return g_x0, g_x1, None

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, i):
            # [FUNCTION DOC] HELPER: Splits x by unbacked sizes and applies the function.
            i0, i1 = i.tolist()
            x0, x1 = x.split([i0, i1])
            return SillyCat.apply(x0, x1, i)

        f(torch.randn(9, requires_grad=True), torch.tensor([3, 6]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_validate_outputs_unbacked_by_custom_op(self):
        # [FUNCTION DOC] With scalar capture, checks a custom op with fake implementation used on
        # unbacked split pieces compiles with aot_eager.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo_validate_outputs_unbacked",
                "(Tensor a, Tensor b) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo_validate_outputs_unbacked", "cpu", lib=lib)
            @torch.library.register_fake(
                "mylib::foo_validate_outputs_unbacked", lib=lib
            )
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: Concatenates two tensors; used as both CPU and
                # fake implementation.
                return torch.cat([x, y])

            @torch.compile(backend="aot_eager", fullgraph=True)
            def f(x, i):
                # [FUNCTION DOC] HELPER: Splits x by unbacked sizes and calls the custom op.
                i0, i1 = i.tolist()
                x0, x1 = x.split([i0, i1])
                return torch.ops.mylib.foo_validate_outputs_unbacked(x0, x1)

            f(torch.randn(9, requires_grad=True), torch.tensor([3, 6]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_dim_order(self):
        # [FUNCTION DOC] With scalar capture, checks dim_order: permuted contiguous inputs
        # recover the original strides including unbacked dims, custom int strides give expected
        # order, and unbacked strides without ordering raise when the ambiguity check is on.
        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Permutes x and returns it with its dim order.
            x = x.permute(3, 0, 2, 1)
            return x, x.dim_order()

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def g(x):
            # [FUNCTION DOC] HELPER: Returns the dim order of x.
            return x.dim_order()

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def h0(xs, ambiguity_check=False):
            # [FUNCTION DOC] HELPER: Builds a strided tensor with unbacked strides and partial
            # ordering checks, returns its dim order (optionally checking ambiguity).
            u0, u1, u2 = xs.tolist()
            torch._check(u2 >= u0)
            torch._check(u1 >= u0)
            # stride ordering still isn't unique here, should raise
            y = torch.empty_strided([4, 4, 4], [u0, u1, u2])
            return y.dim_order(ambiguity_check=ambiguity_check)

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def h1(xs, ambiguity_check=False):
            # [FUNCTION DOC] HELPER: Builds a strided tensor with all equal unbacked strides and
            # returns its dim order.
            u0, u1, u2 = xs.tolist()
            y = torch.empty_strided([4, 4, 4], [u0, u0, u0])  # no ordering
            return y.dim_order(ambiguity_check=ambiguity_check)

        # check that for functions permuting contiguous input, the original stride is recovered with dim_order.
        def test(x):
            # [FUNCTION DOC] HELPER: Compares the input strides with the output strides reordered
            # by the returned dim order.
            stride_inp = tuple(x.stride())
            f_out, f_order = f(x)
            self.assertEqual(stride_inp, tuple(f_out.stride(i) for i in f_order))

        # shape: [4, u0, 5, u1]
        x0 = torch.randn(4, 1, 5, 2)
        torch._dynamo.decorators.mark_unbacked(x0, 1)
        torch._dynamo.decorators.mark_unbacked(x0, 3)
        test(x0)

        # shape: [u0, u1, u2, u3]
        x1 = torch.randn(4, 1, 5, 2)
        for i in range(x1.ndim):
            torch._dynamo.decorators.mark_unbacked(x1, i)
        test(x1)

        # custom strides (all integers)
        x2 = torch.randn(10000)
        x2 = x2.as_strided([4, 4, 4, 4], [1, 2, 4, 8])
        if g(x2) != (3, 2, 1, 0):
            raise AssertionError(f"Expected g(x2) == (3, 2, 1, 0), got {g(x2)}")

        # custom unbacked strides with no ordering: ambiguity check should raise
        xs = torch.tensor([2, 3, 4])
        h0(xs)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            r"The tensor does not have unique dim order.",
        ):
            h0(xs, ambiguity_check=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            r"The tensor does not have unique dim order.",
        ):
            h1(xs, ambiguity_check=True)

    def test_str_format_assert1(self):
        # [FUNCTION DOC] Checks torch assert with an f-string message built from a shape.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            # [FUNCTION DOC] HELPER: Computes sine, asserts the slice length is 2, returns
            # the sum.
            x = torch.sin(img)
            val = x.shape[-2:]
            torch._assert(len(val) == 2, f"shape {img.shape}")
            return img + x

        img1 = torch.randn(1, 1, 8, 8)
        res = fn(img1)
        self.assertEqual(res, img1 + torch.sin(img1))

    def test_str_format_assert2(self):
        # [FUNCTION DOC] Checks torch assert with a shape-based condition: compiled once for
        # valid shape; a different shape triggers recompile and AssertionError.
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(img):
            # [FUNCTION DOC] HELPER: Computes sine, asserts shape values with a message, returns
            # the sum.
            x = torch.sin(img)
            torch._assert(
                img.shape[-2] == 8 and img.shape[-1] == 16, f"shape {img.shape}"
            )
            return img + x

        img1 = torch.randn(1, 3, 8, 16)
        res = fn(img1)
        self.assertEqual(res, img1 + torch.sin(img1))
        self.assertEqual(cnt.frame_count, 1)

        # trigger a recompile and graph break
        img2 = torch.randn(1, 3, 8, 15)
        self.assertRaises(AssertionError, lambda: fn(img2))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_scalar(self):
        # [FUNCTION DOC] With scalar capture, checks tolist on a 1-element tensor and a loop
        # multiplying items, one frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies each item of the list by 4.
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([3])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_1d(self):
        # [FUNCTION DOC] Same for a 1-d tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies each item of the list by 4.
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([2, 1])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_kd(self):
        # [FUNCTION DOC] Same for a 3-d tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies each element of the outer list by 4.
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([[[2, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [2, 1]]])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @patch.object(torch._dynamo.config, "specialize_int", True)
    def test_tolist_0d(self):
        # [FUNCTION DOC] With scalar capture and int specialization, checks tolist of a
        # 0-d tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the scalar list value times 4 into a list.
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list

        x = torch.tensor(42)
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
    def test_tolist_kd_dynamic(self):
        # [FUNCTION DOC] With dynamic default and no automatic dynamic, checks tolist of a 2-d
        # tensor: value change reuses frame, size change recompiles (frames 1, 1, 2).
        def fn(x):
            # [FUNCTION DOC] HELPER: Collects scaled list and returns it with x times 10.
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list, x * 10

        x = torch.randint(3, 5, [5, 5])
        eager = fn(x)
        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=False)
        compiled = compiled_fn(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

        # Value change, no recompiles
        x = torch.randint(7, 9, [5, 5])
        compiled_fn(x)
        self.assertEqual(counter.frame_count, 1)

        # Size change, forced recompiles
        x = torch.randint(3, 5, [3, 3])
        compiled_fn(x)
        self.assertEqual(counter.frame_count, 2)

    def test_tolist_float(self):
        # [FUNCTION DOC] Checks tolist on a float tensor without fullgraph: nothing compiled
        # (zero frames).
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies each item by 4.
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor(
            [[[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]], [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]]
        )
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)(x)
        self.assertEqual(eager, compiled)
        # Nothing to compile here
        self.assertEqual(counter.frame_count, 0)

    def test_inline_closure_not_loaded_by_parent(self):
        # [FUNCTION DOC] Checks nested closures resolved from an enclosing function chain compile
        # in one frame.
        def outer(a):
            # [FUNCTION DOC] HELPER: Adds 1.
            return a + 1

        def indirect(x):
            # [FUNCTION DOC] HELPER: Calls direct.
            return direct(x)

        def direct(x):
            # [FUNCTION DOC] HELPER: Defines nested helpers and calls them.
            def deep2(c):
                # [FUNCTION DOC] HELPER: Calls outer.
                return outer(c)

            def deep(c):
                # [FUNCTION DOC] HELPER: Calls deep2.
                return deep2(c)

            return deep(x)

        x = torch.randn(3)
        eager = indirect(x)
        counter = CompileCounter()
        compiled = torch.compile(indirect, backend=counter)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_inline_closure_returned_by_another_function_and_captures(self):
        # [FUNCTION DOC] Checks inlining a function returned from another function that captured
        # a tensor from the test scope; result is 3.
        x = torch.ones(1)

        def fn():
            # [FUNCTION DOC] HELPER: Returns an inner closure.
            def inner():
                # [FUNCTION DOC] HELPER: Returns captured x plus 2.
                return x + 2

            return inner

        @torch.compile(backend="eager")
        def start():
            # [FUNCTION DOC] HELPER: Gets the closure, calls it and returns the result.
            # Obtain the `inner` function, which holds reference to `x`.
            inner = fn()

            # When we call `inner`, we end up looking up `x` from our inlining
            # tracer, Dynamo must make sure it still has some modeling of `x` at
            # that point.
            res = inner()
            return res

        res = start()
        self.assertEqual(torch.ones(1) * 3, res)

    def test_deque_input(self):
        # [FUNCTION DOC] Checks a deque input popped twice under fullgraph: input mutated like
        # eager, one frame.
        a = torch.randn([2, 3])
        b = torch.randn([2, 3])
        d1 = collections.deque(["foo", a, b])
        d2 = d1.copy()

        def fn(q):
            # [FUNCTION DOC] HELPER: Pops two entries and multiplies them.
            a = q.pop()
            b = q.pop()
            return a * b

        eager = fn(d1)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(d2)
        self.assertEqual(d1, d2)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_deque_append_left(self):
        # [FUNCTION DOC] Checks deque appendleft and popleft; equal state to eager, one frame,
        # tensor result.
        d1 = collections.deque(["foo", 10, 10])
        d2 = d1.copy()

        def fn(q, a, b):
            # [FUNCTION DOC] HELPER: Appends two values on the left, pops two from the
            # left, multiplies.
            q.appendleft(a)
            q.appendleft(b)
            return q.popleft() * q.popleft()

        a = torch.randn([3, 3])
        b = torch.randn([3, 3])
        eager = fn(d1, a, b)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(d2, a, b)
        self.assertEqual(d1, d2)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)
        self.assertTrue(isinstance(compiled, torch.Tensor))

    def test_yield_from(self):
        # [FUNCTION DOC] Checks yield from with a generator inside a function; one frame.
        def yield_from_fn(t_list, k):
            # [FUNCTION DOC] HELPER: Defines a generator, returns scaled items.
            def yield_from_gen(l):
                # [FUNCTION DOC] HELPER: Generator yielding scaled elements via yield from.
                l2 = [t * k for t in l]
                yield from l2

            return [t * k for t in yield_from_gen(t_list)]

        t_list = [torch.randn([2, 3]) for _ in range(3)]
        eager = yield_from_fn(t_list, 2)
        counter = CompileCounter()
        compiled = torch.compile(yield_from_fn, backend=counter)(t_list, 2)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_yield_from_in_a_loop(self):
        # [FUNCTION DOC] Checks yield from inside a loop of generators.
        def gen2():
            # [FUNCTION DOC] HELPER: Generator yielding 1.
            yield 1

        def gen1():
            # [FUNCTION DOC] HELPER: Generator that yields from gen2 five times.
            for value in range(5):
                yield from gen2()

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums yielded values and adds to x.
            c = 0
            for i in gen1():
                c = c + i
            return x + c

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.zeros(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_yield_gen_and_from(self):
        # [FUNCTION DOC] Checks a generator and a yield-from generator used inside a function;
        # one frame.
        def populate_and_multiply_sequence(n, multiplier):
            # [FUNCTION DOC] HELPER: Builds a list from a generator multiplied by a tensor and
            # returns items via another generator.
            # Inline generator
            def tensor_generator():
                # [FUNCTION DOC] HELPER: Generator yielding index tensors.
                for i in range(n):
                    yield torch.tensor([i])

            # Use 'yield from' to iterate over tensors and multiply
            t_list = [tensor * multiplier for tensor in tensor_generator()]

            def yield_from_gen():
                # [FUNCTION DOC] HELPER: Generator yielding from the list.
                yield from t_list

            return [t for t in yield_from_gen()]

        multiplier = torch.tensor([10])
        eager = populate_and_multiply_sequence(5, multiplier)
        counter = CompileCounter()
        compiled = torch.compile(populate_and_multiply_sequence, backend=counter)(
            5, multiplier
        )
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_yield_from_user_stop_iteration(self):
        # [FUNCTION DOC] Checks yield from over a user iterator raising StopIteration with a
        # value; no frames compiled (fallback).
        class MyIter:
            # [CLASS DOC] HELPER: Iterator over a sequence raising StopIteration with the index
            # at the end.
            def __init__(self, seq):
                # [FUNCTION DOC] HELPER: Stores sequence and index 0.
                self.seq = seq
                self.index = 0

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self

            def __next__(self):
                # [FUNCTION DOC] HELPER: Advances index and returns the element or
                # raises StopIteration.
                self.index += 1
                if self.index <= len(self.seq):
                    return self.seq[self.index - 1]
                raise StopIteration(self.index)

        def yield_from_iter_fn(seq):
            # [FUNCTION DOC] HELPER: Collects all values yielded from a generator over
            # the iterator.
            def gen(seq):
                # [FUNCTION DOC] HELPER: Generator yielding from the user iterator.
                yield from MyIter(seq)

            return [i for i in gen(seq)]

        seq = [torch.randn([2, 3]) for _ in range(3)]
        eager = yield_from_iter_fn(seq)
        counter = CompileCounter()
        compiled = torch.compile(yield_from_iter_fn, backend=counter)(seq)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 0)

    # just to be sure in case anyone tries to run this in older versions of Python
    def test_pep0479_convert_stopiteration(self):
        # [FUNCTION DOC] Checks StopIteration raised inside a generator is turned into
        # RuntimeError (PEP 479) in compiled code; result 100.
        # https://peps.python.org/pep-0479/
        def generator_with_stop_iteration():
            # [FUNCTION DOC] HELPER: Generator that yields 1 then raises StopIteration.
            yield 1
            # Explicitly raising StopIteration inside the generator
            raise StopIteration("StopIteration raised within generator")
            yield 2  # This should never be reached

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Consumes the generator twice and returns 100 on RuntimeError
            # or 200 on StopIteration.
            try:
                # Try to consume the generator
                gen = generator_with_stop_iteration()
                next(gen)
                next(gen)
            except RuntimeError as e:
                # Check that StopIteration was converted to RuntimeError
                # See STOPITERATION_ERROR opcode in symbolic_convert.py
                return 100
            except StopIteration:
                return 200

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, 100)

    def test_yield_send_to_subgenerator_graph_break(self):
        # [FUNCTION DOC] Checks send to a subgenerator via yield from in a generator function:
        # result matches eager and no frames are compiled.
        def subgenerator(tensor):
            # [FUNCTION DOC] HELPER: Generator receiving a multiplier then yielding the product.
            multiplier = yield
            yield tensor * multiplier

        def main_generator(t_list):
            # [FUNCTION DOC] HELPER: For each tensor, starts a subgenerator and yields from the
            # sent result.
            for tensor in t_list:
                subgen = subgenerator(tensor)
                next(subgen)
                yield from subgen.send(torch.tensor([10]))

        t_list = [torch.tensor([i]) for i in range(5)]
        eager = list(main_generator(t_list))

        counter = CompileCounter()
        compiled_fn = torch.compile(main_generator, backend=counter)
        compiled = list(compiled_fn(t_list))

        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 0)

    def test_derpy_nn_module_usage(self):
        # [FUNCTION DOC] Checks module methods replaced by functions that read other modules and
        # parameters; 6 ops under fullgraph.
        def ff1(x):
            # [FUNCTION DOC] HELPER: Forward for mod1: sigmoid of submodule output plus param1.
            self = mod1
            return torch.sigmoid(self.mod2(x) + self.param1)

        def ff2(x):
            # [FUNCTION DOC] HELPER: Forward for mod2: cosine of sine times param2 plus 10.
            self = mod2
            return torch.cos(torch.sin(x) * self.param2 + 10)

        mod1 = torch.nn.Module()
        mod2 = torch.nn.Module()
        mod1.register_module("mod2", mod2)
        mod1.register_parameter("param1", torch.nn.Parameter(torch.randn(10)))
        mod1.forward = ff1
        mod2.register_parameter("param2", torch.nn.Parameter(torch.randn(10)))
        mod2.forward = ff2
        mod1.eval()

        x = torch.randn(10)
        expected = mod1(x)
        counter = CompileCounter()
        actual = torch.compile(mod1, backend=counter, fullgraph=True)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(counter.op_count, 6)

    def test_default_args_device_dtype(self):
        # [FUNCTION DOC] Checks a class with torch dtype and device default argument values
        # creating a tensor under fullgraph.
        class Foo:
            # [CLASS DOC] HELPER: Holder creating a tensor from default dtype and device.
            def __init__(
                self,
                dtype: torch.dtype = torch.float16,
                device: torch.device = torch.device("cpu"),
            ) -> None:
                # [FUNCTION DOC] HELPER: Stores a scalar tensor made with dtype and device.
                self.value = torch.tensor(10, dtype=dtype, device=device)

        def fn():
            # [FUNCTION DOC] HELPER: Returns Foo value plus 1.
            return Foo().value + 1

        opt_func = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn()
        res = opt_func()
        self.assertEqual(ref, res)

    def test_torch_dtype_python_type(self):
        # [FUNCTION DOC] Checks dtype attribute and isinstance check at trace time; a constant
        # variable reports the dtype type.
        def fn(target):
            # [FUNCTION DOC] HELPER: Creates three zero tensors with the dtype and sums them.
            target_dtype = target.dtype
            a = torch.zeros(2, 3, dtype=target_dtype)
            # Constant assert at trace time
            assert isinstance(target_dtype, torch.dtype)  # noqa: S101
            b = torch.zeros(2, 3, dtype=target_dtype)
            c = torch.zeros(2, 3, dtype=target_dtype)
            return a + b + c

        from torch._dynamo.variables import ConstantVariable

        dtype = torch.float16
        expected_variable = ConstantVariable(dtype)
        self.assertEqual(expected_variable.python_type(), type(dtype))

        opt_func = torch.compile(fn, backend="eager", fullgraph=True)
        a = torch.tensor([2, 3], dtype=dtype)
        res = opt_func(a)
        self.assertIsInstance(res, torch.Tensor)

    def test_iterator_limit(self):
        # [FUNCTION DOC] Checks that an infinite generator consumed by list raises Unsupported
        # mentioning infinite generator.
        def fn(x):
            # [FUNCTION DOC] HELPER: Defines infinite generator and returns list of it.
            def gen():
                # [FUNCTION DOC] HELPER: Generator yielding x forever.
                while True:
                    yield x

            return list(gen())

        x = torch.randn([0, 1, 2, 3, 4, 5])
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "infinite generator"
        ):
            compiled_fn(x)

    def test_itertools_islice(self):
        # [FUNCTION DOC] Checks islice with start, stop and step under fullgraph with no
        # graph break.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns islice of x.
            return itertools.islice(x, 2, 5, 2)

        x = torch.randn([0, 1, 2, 3, 4, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_islice_default_step(self):
        # [FUNCTION DOC] Checks islice with start and stop.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns islice of x.
            return itertools.islice(x, 2, 5)

        x = torch.randn([0, 1, 2, 3, 4, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_islice_default_end(self):
        # [FUNCTION DOC] Checks islice with only stop.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns islice of x.
            return itertools.islice(x, 2)

        x = torch.randn([0, 1, 2, 3, 4, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_repeat(self):
        # [FUNCTION DOC] Checks bounded repeat iteration accumulating a constant.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 100 for each of five repeats.
            r = itertools.repeat(100.0, 5)
            for i in r:
                x += i
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_repeat(self):
        # [FUNCTION DOC] Checks infinite repeat with break after eleven iterations.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds repeated 100 until counter exceeds 10.
            r = itertools.repeat(100.0)
            idx = 0
            for i in r:
                x += i
                idx += 1
                if idx > 10:
                    break
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_repeat_mutation(self):
        # [FUNCTION DOC] Checks infinite repeat of a tensor object mutated in loop.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Repeats x, adds it to itself and mutates the loop variable,
            # with break.
            r = itertools.repeat(x)
            idx = 0
            for i in r:
                x += i
                i += 1
                idx += 1
                if idx > 10:
                    break
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_repeat_length_hint(self):
        # [FUNCTION DOC] Checks length hint of repeat objects (bounded, zero, negative, infinite
        # with default).
        def fn():
            # [FUNCTION DOC] HELPER: Returns four length hints.
            return (
                operator.length_hint(itertools.repeat(None, 50)),
                operator.length_hint(itertools.repeat(None, 0)),
                operator.length_hint(itertools.repeat(None, -3)),
                operator.length_hint(itertools.repeat(None), 12),
            )

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(), compiled_fn())
        self.assertEqual(compiled_fn(), (50, 0, 0, 12))

    def test_itertools_repeat_repr(self):
        # [FUNCTION DOC] Checks repr of bounded and unbounded repeat.
        def fn():
            # [FUNCTION DOC] HELPER: Returns reprs of two repeat objects.
            r = itertools.repeat("a", -1)
            bounded = repr(r)
            unbounded = repr(itertools.repeat(1 + 0j))
            return bounded, unbounded

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(), compiled_fn())
        self.assertEqual(compiled_fn(), ("repeat('a', 0)", "repeat((1+0j))"))

    def test_itertools_repeat_bounded_exhausts(self):
        # [FUNCTION DOC] Checks bounded repeats exhaust properly including negative count.
        def fn():
            # [FUNCTION DOC] HELPER: Returns concatenated lists of repeats.
            return list(itertools.repeat("a", 3)) + list(itertools.repeat("b", -2))

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(), compiled_fn())
        self.assertEqual(compiled_fn(), ["a", "a", "a"])

    def test_itertools_repeat_partial_iteration(self):
        # [FUNCTION DOC] Checks repeat length hint and repr after partial iteration.
        def fn():
            # [FUNCTION DOC] HELPER: Advances once and returns hint and repr.
            r = itertools.repeat("a", 3)
            next(r)
            return operator.length_hint(r), repr(r)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(), compiled_fn())
        self.assertEqual(compiled_fn(), (2, "repeat('a', 2)"))

    def test_itertools_infinite_count(self):
        # [FUNCTION DOC] Checks infinite count with no args, one arg and two args.
        for args in ([], [10], [5, -1]):
            counters.clear()

            def fn(x):
                # [FUNCTION DOC] HELPER: Adds counts into x with a break after eleven iterations.
                r = itertools.count(*args)
                idx = 0
                for i in r:
                    x += i
                    idx += 1
                    if idx > 10:
                        break
                return x

            x = torch.randn([2, 5])
            eager = fn(x)

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_count_from_uncompiled_region(self):
        # [FUNCTION DOC] Checks a count object created outside and advanced by compiled code is
        # advanced in the same way.
        counters.clear()
        counter = itertools.count()

        def fn(x):
            # [FUNCTION DOC] HELPER: Scales x by next count plus 1.
            return x * (next(counter) + 1)

        x = torch.randn([2, 5])
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)

        self.assertEqual(compiled_fn(x), x)
        self.assertEqual(compiled_fn(x), x * 2)
        self.assertEqual(next(counter), 2)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_count_already_advanced(self):
        # [FUNCTION DOC] Checks count advanced before compile continues from the
        # correct position.
        counters.clear()
        counter = itertools.count()
        # Advance the counter before entering the compiled region
        next(counter)  # 0
        next(counter)  # 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Scales x by next count plus 1.
            return x * (next(counter) + 1)

        x = torch.randn([2, 5])
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)

        self.assertEqual(compiled_fn(x), x * 3)  # next(counter) = 2
        self.assertEqual(compiled_fn(x), x * 4)  # next(counter) = 3
        self.assertEqual(next(counter), 4)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_count_kwargs(self):
        # [FUNCTION DOC] Checks count with keyword start and step arguments.
        counters.clear()

        def fn():
            # [FUNCTION DOC] HELPER: Returns zipped lists with three count variants.
            return (
                list(zip("abc", itertools.count(start=2, step=3))),
                list(zip("abc", itertools.count(step=-1))),
                list(zip("abc", itertools.count(2, step=3))),
            )

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled_fn(), fn())
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_count_repr(self):
        # [FUNCTION DOC] Checks repr of count objects with int and float step.
        counters.clear()

        def fn():
            # [FUNCTION DOC] HELPER: Builds counts, advances them and returns six reprs.
            c = itertools.count(3)
            r0 = repr(c)
            next(c)
            r1 = repr(c)
            d = itertools.count(3, 5)
            r2 = repr(d)
            next(d)
            r3 = repr(d)
            # step==1 (int) is suppressed; a float step (even 1.0) is shown.
            r4 = repr(itertools.count(10.5, 1))
            r5 = repr(itertools.count(10, 1.0))
            return r0, r1, r2, r3, r4, r5

        expected = (
            "count(3)",
            "count(4)",
            "count(3, 5)",
            "count(8, 5)",
            "count(10.5)",
            "count(10, 1.0)",
        )
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled_fn(), fn())
        self.assertEqual(compiled_fn(), expected)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_cycle(self):
        # [FUNCTION DOC] Checks cycle over empty, list, repeat and count iterators with a break.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: For each iterator, cycles and adds up to eleven values.
            for iterator in (
                iter([]),
                iter([10, 11.0]),
                itertools.repeat(-1, 3),
                itertools.count(10),
            ):
                r = itertools.cycle(iterator)
                idx = 0
                x += 1
                for i in r:
                    x += i
                    idx += 1
                    if idx > 10:
                        break
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_symint_default_sum(self):
        # [FUNCTION DOC] Checks accumulate over symbolic sizes (issue 110287).
        # https://github.com/pytorch/pytorch/issues/110287
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies x in place by each accumulated size.
            r = itertools.accumulate([x.size(0), x.size(1)])
            for i in r:
                x *= i
            return x

        x = torch.randn(2, 3)
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_tensors_default_sum(self):
        # [FUNCTION DOC] Checks accumulate over tensors with default sum.
        counters.clear()

        def fn(a, b, c, d, x):
            # [FUNCTION DOC] HELPER: Multiplies list entries by x and returns accumulate.
            l = [a, b, c, d, x]
            for i, t in enumerate(l):
                l[i] = t * x
            return itertools.accumulate(l)

        t_list = [torch.tensor([i + 1]) for i in range(4)]
        x = torch.tensor([[1, 2], [3, 4]])
        eager = fn(*t_list, x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(*t_list, x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_tensors_builtins(self):
        # [FUNCTION DOC] Checks accumulate with builtin operators mul, sub and pow.
        for builtin_op in [operator.mul, operator.sub, operator.pow]:
            counters.clear()

            def fn(a, b, c, d, x):
                # [FUNCTION DOC] HELPER: Multiplies list entries by x and returns accumulate with
                # the operator.
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, builtin_op)

            t_list = [torch.tensor([i + 1]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])
            eager = fn(*t_list, x)

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(*t_list, x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_tensors_kwargs(self):
        # [FUNCTION DOC] Checks accumulate with func and initial keyword combinations.
        from torch._dynamo.utils import counters

        for kwargs in [
            {"func": operator.mul},
            {"initial": 100},
            {"func": operator.sub, "initial": -1},
        ]:
            counters.clear()

            def fn(a, b, c, d, x):
                # [FUNCTION DOC] HELPER: Multiplies list entries by x and returns accumulate with
                # keyword args.
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, **kwargs)

            t_list = [torch.tensor([i + 1]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(*t_list, x)
            eager = fn(*t_list, x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_packaging_version_parse(self):
        # [FUNCTION DOC] Checks comparing parsed package versions at trace time.
        from packaging import version

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Returns ones if torch version parse is at least 2.0.0.
            x = torch.zeros(1)
            if version.parse(torch.__version__) >= version.parse("2.0.0"):
                return x + 1
            return x

        self.assertEqual(fn().item(), 1)

    def test_iter_version(self):
        # [FUNCTION DOC] Checks iterating over the version string.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sums digit characters of the version and adds to x
            # then sine.
            s = 0
            for i in torch.__version__:
                try:
                    s += int(i)
                except ValueError:
                    pass
            return (x + s).sin()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(fn(x), opt_fn(x))

    def test_itertools_accumulate_tensors_user_defined(self):
        # [FUNCTION DOC] Checks accumulate with three user-defined combining functions (constant,
        # random flag based, and stateful).
        def udo_fn_0(a, b):
            # [FUNCTION DOC] HELPER: Returns -1.
            return -1

        rando = random.randint(0, 1)

        def udo_fn_1(a, b):
            # [FUNCTION DOC] HELPER: Combines a and b scaled by a random flag.
            return a * rando + b * rando

        seen = []

        def udo_fn_2(a, b):
            # [FUNCTION DOC] HELPER: Records both values in a list and returns a times
            # list length.
            seen.append(a)
            seen.append(b)
            return a * len(seen)

        for udo_fn in [udo_fn_0, udo_fn_1, udo_fn_2]:
            counters.clear()
            torch._dynamo.reset()

            def fn(a, b, c, d, x):
                # [FUNCTION DOC] HELPER: Multiplies list entries by x and returns accumulate with
                # the user function.
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, udo_fn)

            t_list = [torch.tensor([i]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])
            eager = fn(*t_list, x)

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(*t_list, x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_pure_python_accumulate(self):
        # [FUNCTION DOC] Checks a pure python accumulate generator compiles in one frame.
        def accumulate(iterable, func=lambda x, y: x + y):
            # [FUNCTION DOC] HELPER: Generator yielding running accumulations with a default
            # sum function.
            it = iter(iterable)
            try:
                # Initialize the accumulator with the first value from the iterable
                accumulator = next(it)
            except StopIteration:
                # If the iterable is empty, return an empty generator
                return
            yield accumulator

            for element in it:
                accumulator = func(accumulator, element)
                yield accumulator

        def fn(it):
            # [FUNCTION DOC] HELPER: Returns the accumulate generator.
            return accumulate(it)

        t_list = [torch.tensor([i]) for i in range(4)]
        eager = fn(t_list)

        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter)
        compiled = compiled_fn(t_list)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(counter.frame_count, 1)

    def test_itertools_groupby_pure_python_default_identify_func(self):
        # [FUNCTION DOC] Checks groupby with identity key.
        counters.clear()

        def fn(l):
            # [FUNCTION DOC] HELPER: Returns groups as key and list pairs.
            return [(k, list(g)) for k, g in itertools.groupby(l)]

        l = [1, 2, 2, 3, 4, 4, 4, 1, 2]
        eager = fn(l)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(l)

        self.assertEqual(eager, compiled)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_groupby_pure_python_key_func(self):
        # [FUNCTION DOC] Checks groupby with a key function.
        counters.clear()

        def fn(l):
            # [FUNCTION DOC] HELPER: Returns groups with negation key.
            return [(k, list(g)) for k, g in itertools.groupby(l, key=operator.neg)]

        l = [1, 2, -2, 3, 4, 4, -4, 0, -2]
        eager = fn(l)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(l)

        self.assertEqual(eager, compiled)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_tee(self):
        # [FUNCTION DOC] Checks tee splits an iterator into two lists.
        counters.clear()

        def fn(l):
            # [FUNCTION DOC] HELPER: Returns two lists from tee.
            a, b = itertools.tee(l)
            return list(a), list(b)

        l = [1, 2, 2, 3, 4, 4, 4, 1, 2]
        eager = fn(l)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(l)

        self.assertEqual(eager, compiled)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_list_iterator_contains(self):
        # [FUNCTION DOC] Checks membership test on a list iterator after advancing once.
        def fn(x):
            # [FUNCTION DOC] HELPER: Advances iterator, tests membership, returns x plus 2 or 1.
            it = iter(["my_weight", "not_my_weight"])
            next(it)
            if "my_weight" in it:
                return x + 2
            return x + 1

        x = torch.zeros(3)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)

        self.assertEqual(fn(x), compiled_fn(x))

    def test_storage_return(self):
        # [FUNCTION DOC] Checks resizing storage inside compiled code and returning the
        # storage object.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes sine, resizes storage to zero, takes cosine and
            # returns result with storage.
            y = torch.sin(x + 1)
            storage = x.untyped_storage()
            storage.resize_(0)
            y = torch.cos(y)
            return y, storage

        x = torch.randn(10)
        expected = torch.cos(torch.sin(x + 1))
        y, s = fn(x)
        self.assertEqual(y, expected)
        self.assertEqual(x.untyped_storage().size(), 0)
        self.assertIs(s, x.untyped_storage())

    def test_flat_name_to_original_fqn(self):
        # [FUNCTION DOC] Checks export records metadata mapping flattened names to original
        # dotted parameter and buffer names.
        class FooBarModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter named 0, a buffer and a parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers parameter 0, buffer and parameter.
                super().__init__()
                self.register_parameter("0", torch.nn.Parameter(torch.randn(3, 4)))
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Combines input, buffer, parameter named 0 and divides
                # by param.
                return ((x + self.test_buf) * getattr(self, "0")) / self.test_param

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the submodule, a parameter and a buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates submodule, parameter and buffer.
                super().__init__()
                self.foo_bar = FooBarModule()
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Combines submodule output with param and buffer.
                return (self.foo_bar(x) + self.test_param) * self.test_buf

        gm, _ = torch._dynamo.export(TestModule(), torch.randn(3, 4))
        self.assertIn("dynamo_flat_name_to_original_fqn", gm.meta)
        expected_fqn = {
            "L__self___test_param": "test_param",
            "L__self___test_buf": "test_buf",
            "L__self___foo_bar_0": "foo_bar.0",
            "L__self___foo_bar_test_param": "foo_bar.test_param",
            "L__self___foo_bar_test_buf": "foo_bar.test_buf",
        }
        self.assertEqual(expected_fqn, gm.meta["dynamo_flat_name_to_original_fqn"])

    def test_proxy_frozen_dataclass(self):
        # [FUNCTION DOC] Checks a frozen dataclass created in compiled code and passed to an
        # allowed-in-graph function.
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass of two tensors.
            x: torch.Tensor
            y: torch.Tensor

        @allow_in_graph
        def inner_fn(dc):
            # [FUNCTION DOC] HELPER: Allowed-in-graph function summing fields.
            return dc.x + dc.y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds dataclass and passes it to inner_fn.
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_reconstruct_frozen_dataclass(self):
        # [FUNCTION DOC] Checks a frozen dataclass is reconstructed across a graph break.
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass of two tensors.
            x: torch.Tensor
            y: torch.Tensor

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds dataclass, graph breaks, sums fields.
            dc = TestDataClass(x, y)
            torch._dynamo.graph_break()
            return dc.x + dc.y

        fn_opt = torch.compile(backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

    def test_nested_dataclass_reconstruct(self):
        # [FUNCTION DOC] Checks a frozen dataclass with a nested frozen dataclass default is
        # reconstructed on output.
        @dataclasses.dataclass(frozen=True)
        class NestedDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass with int field default 2.
            x: int = 2

        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass with tensor and nested default.
            y: torch.Tensor
            ndc: NestedDataClass = NestedDataClass()

        def fn(y):
            # [FUNCTION DOC] HELPER: Builds dataclass and returns sum with the object.
            dc = TestDataClass(y)
            z = dc.y + dc.ndc.x
            return z, dc

        fn_opt = torch.compile(backend="eager")(fn)
        inps = (torch.ones(2, 2),)
        actual = fn_opt(*inps)
        expected = fn(*inps)

    def test_frozen_dataclass_default_value(self):
        # [FUNCTION DOC] Checks frozen dataclass default values passed to an allowed-in-
        # graph function.
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass with two tensors and two int defaults.
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(default=5)
            a: int = 6

        @allow_in_graph
        def inner_fn(dc):
            # [FUNCTION DOC] HELPER: Allowed-in-graph function summing all four fields.
            return dc.x + dc.y + dc.z + dc.a

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds dataclass and calls inner_fn.
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_default_factory(self):
        # [FUNCTION DOC] Checks frozen dataclass default factory fields (list and lambda) passed
        # to an allowed-in-graph function.
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass with two tensors and two fields using
            # default factories.
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(default_factory=list)
            a: int = dataclasses.field(default_factory=lambda: [5])

        @allow_in_graph
        def inner_fn(dc):
            # [FUNCTION DOC] HELPER: Allowed-in-graph function returning x plus y plus first
            # element of a.
            return dc.x + dc.y + dc.a[0]

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds the dataclass and calls inner_fn.
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_kw_only(self):
        # [FUNCTION DOC] Checks frozen dataclass with keyword-only fields passed to an allowed-
        # in-graph function.
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass with two tensors and two keyword-only
            # int fields.
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(kw_only=True)
            a: int = dataclasses.field(kw_only=True)

        @allow_in_graph
        def inner_fn(dc):
            # [FUNCTION DOC] HELPER: Allowed-in-graph function summing four fields.
            return dc.x + dc.y + dc.a + dc.z

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds dataclass with keyword args and calls inner_fn.
            dc = TestDataClass(x, y, z=5, a=2)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_attr_access(self):
        # [FUNCTION DOC] Checks attribute access on a frozen dataclass inside a regular helper
        # under fullgraph.
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass of two tensors and two ints.
            x: torch.Tensor
            y: torch.Tensor
            z: int
            a: int

        def inner_fn(dc):
            # [FUNCTION DOC] HELPER: Sums the four fields.
            return dc.x + dc.y + dc.a + dc.z

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds dataclass with keyword args and calls inner_fn.
            dc = TestDataClass(x, y, z=5, a=2)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_hashable(self):
        # [FUNCTION DOC] Checks a frozen dataclass (auto hashable) used as a dict key inside
        # compiled code.
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass of two floats and two ints.
            x: float
            y: float
            z: int
            a: int

        def inner_fn(dc, x, y):
            # [FUNCTION DOC] HELPER: Stores 2 under the dataclass key and sums fields, lookup and
            # x and y.
            d = {}
            d[dc] = 2
            return dc.x + dc.y + d[dc] + x + y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds the dataclass and calls inner_fn.
            dc = TestDataClass(x=3.2, y=2.5, z=5, a=2)
            return inner_fn(dc, x, y)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertEqual(actual, expected)

    def test_nested_frozen_dataclass_hashable(self):
        # [FUNCTION DOC] Checks a nested frozen dataclass used as a dict key.
        @dataclasses.dataclass(frozen=True)
        class TestDataClassInner:
            # [CLASS DOC] HELPER: Frozen dataclass of two floats.
            x: float
            y: float

        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: Frozen dataclass with a nested dataclass and two ints.
            b: TestDataClassInner
            z: int
            a: int

        def inner_fn(dc, x, y):
            # [FUNCTION DOC] HELPER: Stores 2 under the dataclass key and sums nested fields,
            # lookup and x and y.
            d = {}
            d[dc] = 2
            return dc.b.x + dc.b.y + d[dc] + x + y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds the nested dataclasses and calls inner_fn.
            dc = TestDataClass(b=TestDataClassInner(2.4, 4.4), z=5, a=2)
            return inner_fn(dc, x, y)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertEqual(actual, expected)

    def test_frozen_dataclass_in_compile(self):
        # [FUNCTION DOC] Checks pytree key objects (frozen dataclasses) can be built and
        # formatted inside compiled code.
        from torch.utils._pytree import MappingKey, SequenceKey

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds key path, formats message, returns double x
            # and message.
            path = (MappingKey("a"), SequenceKey(0))
            msg = f"path={path}"
            return x * 2, msg

        x = torch.randn(4, 4)
        eager_result = fn(x)
        compiled_result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1], compiled_result[1])

    def test_frozen_dataclass_treespec_method_and_fields(self):
        # [FUNCTION DOC] Checks tree spec method and fields (number of leaves, nodes, is_leaf)
        # are available in compiled code.
        from torch.utils._pytree import tree_flatten

        def fn(x):
            # [FUNCTION DOC] HELPER: Flattens a nested dict, returns sum of leaves and
            # spec statistics.
            d = {"a": x, "b": [x * 2, x * 3]}
            flat, spec = tree_flatten(d)
            is_leaf = spec.is_leaf()
            return sum(flat), spec.num_leaves, spec.num_nodes, is_leaf

        x = torch.randn(4)
        eager_result = fn(x)
        compiled_result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        for i in range(4):
            self.assertEqual(eager_result[i], compiled_result[i])

    def test_frozen_dataclass_with_len_constructed_in_compile(self):
        # [FUNCTION DOC] Checks constructing a frozen dataclass defining a length method inside
        # compile does not crash on truthiness checks of an uninitialized object.
        # Tests that constructing a frozen dataclass with __len__ inside
        # torch.compile doesn't crash. Exercises the truthiness check in
        # symbolic_convert.py call_function where `inner_fn and ...` would
        # trigger __len__ on an uninitialized backing object.
        @dataclasses.dataclass(frozen=True)
        class Layout:
            # [CLASS DOC] HELPER: Frozen dataclass layout with shape and stride tuples, length,
            # size, callable index mapping and cosize.
            shape: tuple
            stride: tuple

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns number of shape dims if shape is a tuple,
                # else 1.
                if isinstance(self.shape, tuple):
                    return len(self.shape)
                return 1

            def size(self):
                # [FUNCTION DOC] HELPER: Returns product of shape entries.
                result = 1
                for s in self.shape:
                    result *= s
                return result

            def __call__(self, idx):
                # [FUNCTION DOC] HELPER: Maps an index through the strides by repeated modulo
                # and division.
                # Map index through stride
                result = 0
                for sh, st in zip(self.shape, self.stride):
                    result += (idx % sh) * st
                    idx //= sh
                return result

            def cosize(self):
                # [FUNCTION DOC] HELPER: Returns mapped last index plus 1.
                return self(self.size() - 1) + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the layout and adds its cosize to x.
            layout = Layout((2, 3), (3, 1))
            return x + layout.cosize()

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2),)
        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertEqual(actual, expected)

    def test_frozen_dataclass_callable_with_hash_constructed_in_compile(self):
        # [FUNCTION DOC] Checks constructing a callable frozen dataclass with auto hash inside
        # compile does not crash.
        # Tests that constructing a frozen dataclass with __hash__ (auto from
        # frozen=True) inside torch.compile doesn't crash when the object is
        # callable. Exercises is_supported_random() which does
        # `self.value in set(...)` triggering __hash__ on an uninitialized
        # backing object.
        @dataclasses.dataclass(frozen=True)
        class Config:
            # [CLASS DOC] HELPER: Frozen dataclass with shape and stride tuples; callable mapping
            # an index.
            shape: tuple
            stride: tuple

            def __call__(self, idx):
                # [FUNCTION DOC] HELPER: Maps an index through the strides.
                result = 0
                for sh, st in zip(self.shape, self.stride):
                    result += (idx % sh) * st
                    idx //= sh
                return result

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds config and adds the config called with 2 to x.
            cfg = Config((2, 3), (3, 1))
            return x + cfg(2)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2),)
        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertEqual(actual, expected)

    def test_shape_env_no_recording(self):
        # [FUNCTION DOC] Checks a ShapeEnv created without event recording keeps no events after
        # creating symbolic sizes and evaluating a guard, and (if translation validation is on)
        # stores no recording metadata on graph nodes.
        main = ShapeEnv(should_record_events=False)

        # The main ShapeEnv should have no event recorded.
        self.assertEqual(len(main.events), 0)

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] == 3 (call evaluate_expr)
        #   - +1 guard entry
        #   - +1 replacement entry
        size = r[0]
        bool(size[0] == 3)

        # The main ShapeEnv should remain with no event recorded.
        self.assertEqual(len(main.events), 0)

        if torch.fx.experimental.validator.translation_validation_enabled():
            from torch.fx.experimental.symbolic_shapes import (
                CURRENT_NODE_KEY,
                SHAPEENV_EVENT_KEY,
            )

            # Check that we don't store any recording metadata on nodes
            # from the symbolic shape FX graph.
            for n in main.graph.nodes:
                self.assertFalse(SHAPEENV_EVENT_KEY in n.meta)
                self.assertFalse(CURRENT_NODE_KEY in n.meta)

    def _replay_and_check(self, shape_env: ShapeEnv):
        # [FUNCTION DOC] HELPER: If the environment records events, replays them into a new
        # environment and checks equality.
        if shape_env.should_record_events:
            replayed = replay_shape_env_events(shape_env.events)
            shape_env.check_equal(replayed)

    def test_shape_env_equal_empty(self):
        # [FUNCTION DOC] Checks two fresh ShapeEnv objects compare equal and replay works.
        main, other = ShapeEnv(), ShapeEnv()
        main.check_equal(other)
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_constructor(self):
        # [FUNCTION DOC] Runs only with translation validation. Checks a differing constructor
        # setting is reported by the equality check with the expected message.
        main, other = ShapeEnv(allow_scalar_outputs=False), ShapeEnv()
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> settings: values don't match.
  >  Left: ShapeEnvSettings(allow_scalar_outputs=False, allow_dynamic_output_shape_ops=True, assume_static_by_default=False, specialize_zero_one=True, duck_shape=True, prefer_deferred_runtime_asserts_over_guards=False, trace_asserts=False)
  > Right: ShapeEnvSettings(allow_scalar_outputs=True, allow_dynamic_output_shape_ops=True, assume_static_by_default=False, specialize_zero_one=True, duck_shape=True, prefer_deferred_runtime_asserts_over_guards=False, trace_asserts=False)
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_create_symbolic_sizes_strides_storage_offset(self):
        # [FUNCTION DOC] Runs only with translation validation. Checks the difference report
        # after creating symbolic sizes on one ShapeEnv only.
        main, other = ShapeEnv(), ShapeEnv()
        main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> backed_var_to_val: values don't match.
  >  Left: {s44: 2, s93: 3}
  > Right: {}
==> name_to_node: values don't match.
  >  Left: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {}
==> source_to_symbol: values don't match.
  >  Left: {x.size()[0]: x.size()[0], x.size()[1]: x.size()[1], x.storage_offset(): x.storage_offset(), x.stride()[0]: x.stride()[0], x.stride()[1]: x.stride()[1]}
  > Right: {}
==> source_to_var: values don't match.
  >  Left: {x.size()[0]: s93, x.size()[1]: s44}
  > Right: {}
==> unique_ids: values don't match.
  >  Left: {44, 93}
  > Right: {}
==> val_to_var: values don't match.
  >  Left: {2: s44, 3: s93}
  > Right: {}
==> var_to_range: values don't match.
  >  Left: {s44: VR[2, int_oo], s93: VR[2, int_oo]}
  > Right: {}
==> var_to_sources: values don't match.
  >  Left: {s44: [TensorPropertySource(base=ConstantSource(source_name='x'), prop=<TensorProperty.SIZE: 0>, idx=1)], s93: [TensorPropertySource(base=ConstantSource(source_name='x'), prop=<TensorProperty.SIZE: 0>, idx=0)]}
  > Right: {}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_unbacked(self):
        # [FUNCTION DOC] Runs only with translation validation. Checks the difference report
        # after creating unbacked symbols on one ShapeEnv.
        main, other = ShapeEnv(), ShapeEnv()
        main.create_unbacked_symint()
        main.create_unbacked_symfloat()
        main.create_unbacked_symbool()
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> name_to_node: values don't match.
  >  Left: {u0, u1, zuf0}
  > Right: {}
==> unbacked_symfloat_counter: values don't match.
  >  Left: 1
  > Right: 0
==> unbacked_symint_counter: values don't match.
  >  Left: 2
  > Right: 0
==> var_to_range: values don't match.
  >  Left: {u0: VR[-int_oo, int_oo], u1: VR[0, 1], zuf0: VR[-oo, oo]}
  > Right: {}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_evaluate_expr_divisible(self):
        # [FUNCTION DOC] Runs only with translation validation. Checks the difference report
        # after a divisibility guard on one ShapeEnv.
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] % 3 == 0 (only in the main ShapeEnv)
        #   - +1 guard entry
        #   - +1 divisible entry
        size = r[0]
        bool(size[0] % 3 == 0)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {(Mod(s93, 3)) < 0: False, (Mod(s93, 3)) <= 0: True, 0 < (Mod(s93, 3)): False, 0 <= (Mod(s93, 3)): True, Eq(0, Mod(s93, 3)): True, Eq(Mod(s93, 3), 0): True, Ne(0, Mod(s93, 3)): False, Ne(Mod(s93, 3), 0): False}
  > Right: {}
==> divisible: values don't match.
  >  Left: {Mod(s93, 3)}
  > Right: {}
==> guards: values don't match.
  >  Left: [Eq(Mod(s93, 3), 0)]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, eq, mod, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_evaluate_expr_replacement(self):
        # [FUNCTION DOC] Runs only with translation validation. Checks the difference report
        # after an equality guard creating a replacement.
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] == 3 (only in the main ShapeEnv)
        #   - +1 guard entry
        #   - +1 replacement entry
        size = r[0]
        bool(size[0] == 3)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {False: False, True: True}
  > Right: {}
==> guards: values don't match.
  >  Left: [Eq(s93, 3)]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, eq, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
==> replacements: values don't match.
  >  Left: {s93: 3}
  > Right: {}
==> var_to_range: values don't match.
  >  Left: {s44: VR[2, int_oo], s93: VR[3, 3]}
  > Right: {s44: VR[2, int_oo], s93: VR[2, int_oo]}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_evaluate_expr_refinement(self):
        # [FUNCTION DOC] Runs only with translation validation. Checks the difference report
        # after an inequality guard refining a range.
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] >= 3 (only in the main ShapeEnv)
        #   - +1 guard entry
        #   - +1 var_to_guard entry
        #   - Change: var_to_range
        size = r[0]
        bool(size[0] >= 3)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {3 <= s93: True, s93 < 3: False}
  > Right: {}
==> guards: values don't match.
  >  Left: [s93 >= 3]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, ge, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
==> var_to_range: values don't match.
  >  Left: {s44: VR[2, int_oo], s93: VR[3, int_oo]}
  > Right: {s44: VR[2, int_oo], s93: VR[2, int_oo]}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_runtime_assert(self):
        # [FUNCTION DOC] Runs only with translation validation. Checks the difference report
        # after a deferred runtime assertion on one ShapeEnv.
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_unbacked_symint on both of them.
        r = main.create_unbacked_symint()
        other.create_unbacked_symint()

        # Create a runtime assert: r % 3 == 0 (only in the main ShapeEnv)
        #   - +1 deferred_runtime_asserts entry
        #   - Change: num_deferred_runtime_asserts
        expect_true(r % 3 == 0)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {(PythonMod(u0, 3)) < 0: False, (PythonMod(u0, 3)) <= 0: True, 0 < (PythonMod(u0, 3)): False, 0 <= (PythonMod(u0, 3)): True, Eq(0, PythonMod(u0, 3)): True, Eq(PythonMod(u0, 3), 0): True, Ne(0, PythonMod(u0, 3)): False, Ne(PythonMod(u0, 3), 0): False}
  > Right: {}
==> deferred_runtime_asserts: values don't match.
  >  Left: {u0: [Eq(PythonMod(u0, 3), 0)]}
  > Right: {}
==> name_to_node: values don't match.
  >  Left: {_assert, eq, mod, u0}
  > Right: {u0}
==> num_deferred_runtime_asserts: values don't match.
  >  Left: 1
  > Right: 0
""",
        )
        self._replay_and_check(main)

    def test_shape_env_recorded_function_fallback(self):
        # [FUNCTION DOC] Checks recorded ShapeEnv functions fall back to normal behavior without
        # an environment: constrain range and unify succeed on ints, and range for size on an int
        # raises AssertionError.
        # Make sure the record/replay mechanism for ShapeEnv will fallback
        # if no ShapeEnv instance is found.
        constrain_range(5, min=2, max=10)
        constrain_unify(5, 5)

        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: _constrain_range_for_size(5, min=2, max=10),
            """can only constrain range for SymInt""",
        )

    def test_default_dtype_change(self):
        # [FUNCTION DOC] Checks type promotion result follows the default dtype both for float
        # and double defaults, inside compiled code.
        @torch.compile(backend="eager")
        def foo():
            # [FUNCTION DOC] HELPER: Calls inner on a scalar tensor and a python float.
            def inner(a, b, res_dtype):
                # [FUNCTION DOC] HELPER: Prints arguments and asserts result type equals the
                # expected dtype.
                print(a, b, res_dtype)
                self.assertEqual(torch.result_type(a, b), res_dtype)

            inner(torch.tensor(1, device="cpu"), 1.0, torch.get_default_dtype())

        with set_default_dtype(torch.float):
            foo()
        with set_default_dtype(torch.double):
            foo()

    def test_numpy_ufunc_out(self):
        # [FUNCTION DOC] Checks numpy sine with an out array returns the same out object inside
        # compiled code.
        @torch.compile(backend="eager")
        def foo():
            # [FUNCTION DOC] HELPER: Makes input and out arrays, calls sin with out,
            # asserts identity.
            x = np.arange(5)
            out = np.empty((x.shape[0], x.shape[0]))
            res_out = np.sin(x, out=out)
            assert res_out is out  # noqa: S101

        foo()

    # Unfortunately, we don't currently preserve the ids of
    # res_out and out correctly across the graph break
    @unittest.expectedFailure
    def test_numpy_ufunc_out_graph_break(self):
        # [FUNCTION DOC] Expected failure: array identity of the out argument is not preserved
        # across a graph break.
        @torch.compile(backend="eager")
        def foo():
            # [FUNCTION DOC] HELPER: Same as above with a graph break before the
            # identity assertion.
            x = np.arange(5)
            out = np.empty((x.shape[0], x.shape[0]))
            res_out = np.sin(x, out=out)
            torch._dynamo.graph_break()
            assert res_out is out  # noqa: S101

        foo()

    @wrapDeterministicFlagAPITest
    def test_backward_deterministic_mode_mismatch_warning(self):
        # [FUNCTION DOC] Checks the warning error when a compiled backward runs with
        # deterministic mode enabled but the forward had it disabled, for all four combinations.
        @torch.compile(backend="aot_eager")
        def func(a, b):
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        for forward_deterministic, backward_deterministic in itertools.product(
            [True, False], [True, False]
        ):
            torch.use_deterministic_algorithms(forward_deterministic)
            a = torch.randn(10, requires_grad=True)
            res = func(a, 1)
            grad = torch.ones_like(res)
            torch.use_deterministic_algorithms(backward_deterministic)

            if not forward_deterministic and backward_deterministic:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^This compiled backward function is being run with torch\.use_deterministic_algorithms",
                ):
                    res.backward(grad)

            else:
                res.backward(grad)

    @skipIfWindows(
        msg="AssertionError: False is not true : Encountered an unexpected fallback to 'aten pow' in dynamo compiled code"
    )
    @unittest.skipIf(
        torch._inductor.config.cpu_backend != "cpp",
        "Skip for non cpp backend CPU as comments contain 'aten.pow' ",
    )
    def test_torch_dynamo_codegen_pow(self):
        # [FUNCTION DOC] Checks the generated code for numpy power does not fall back to aten
        # pow. Skipped on Windows or non-cpp CPU backends.
        def pow(x):
            # [FUNCTION DOC] HELPER: Squares x.
            return x**2

        x = np.arange(8)
        pow_opt = torch.compile(pow, backend="eager")

        actual, source_code = run_and_get_code(pow_opt, x)
        expect = pow(x)

        self.assertEqual(expect, actual)

        self.assertTrue(
            all("aten.pow" not in code for code in source_code),
            msg="Encountered an unexpected fallback to 'aten pow' in dynamo compiled code",
        )

    def test_graph_break_compilation_metrics(self):
        # [FUNCTION DOC] Checks compilation metrics for graph breaks: first two compiles record
        # a restart reason with user-inserted graph break and timing ordering; last compile has
        # no restarts.
        def fn(x):
            # [FUNCTION DOC] HELPER: Cos, break, sin, break, cos.
            x.cos()
            torch._dynamo.graph_break()
            x.sin()
            torch._dynamo.graph_break()
            return x.cos()

        torch._dynamo.utils.clear_compilation_metrics()
        x = torch.rand((4, 4))
        f = torch.compile(fn, backend="eager")
        f(x)
        metrics = torch._dynamo.utils.get_compilation_metrics()
        # Should only be one restart per event
        (restart_reason,) = metrics[0].restart_reasons
        self.assertTrue(
            "User-inserted graph break" in restart_reason,
            "Should have logged graph break reason",
        )
        self.assertTrue(
            metrics[0].dynamo_time_before_restart_s
            <= metrics[0].entire_frame_compile_time_s
        )

        (restart_reason,) = metrics[1].restart_reasons
        self.assertTrue(
            "User-inserted graph break" in restart_reason,
            "Should have logged graph break reason",
        )
        self.assertTrue(
            metrics[1].dynamo_time_before_restart_s
            <= metrics[1].entire_frame_compile_time_s
        )

        # No restarts
        self.assertTrue(
            len(metrics[2].restart_reasons) == 0, "Last compile has no graph break"
        )
        self.assertTrue(metrics[2].dynamo_time_before_restart_s == 0)

    def test_graph_break_compilation_metrics_on_failure(self):
        # [FUNCTION DOC] Checks metrics are recorded with a fail reason when a backend raises and
        # errors are suppressed.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sine of x.
            return x.sin()

        def broken_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend raising RuntimeError.
            raise RuntimeError("broken backend")

        x = torch.rand((4, 4))
        f = torch.compile(fn, backend=broken_backend)
        with unittest.mock.patch("torch._dynamo.config.suppress_errors", True):
            torch._dynamo.utils.clear_compilation_metrics()
            f(x)
            metrics = torch._dynamo.utils.get_compilation_metrics()
            for metric in metrics:
                self.assertTrue(metric.dynamo_time_before_restart_s > 0)
                self.assertTrue(
                    "RuntimeError: broken backend" in metric.fail_reason,
                    "Should have logged fail reason",
                )

    def test_compilation_metrics_size_limit(self):
        # [FUNCTION DOC] Checks the compilation metrics list respects a configurable limit of 3
        # and the limit is restored afterwards.
        def fn1(x):
            # [FUNCTION DOC] HELPER: Returns relu of x.
            return x.relu()

        def fn2(x):
            # [FUNCTION DOC] HELPER: Returns cosine of x.
            return x.cos()

        def fn3(x):
            # [FUNCTION DOC] HELPER: Returns sine of x.
            return x.sin()

        def fn4(x):
            # [FUNCTION DOC] HELPER: Returns exp of x.
            return x.exp()

        import contextlib

        @contextlib.contextmanager
        def metrics_limit_ctx():
            # [FUNCTION DOC] HELPER: Context manager setting the metrics limit to 3 and restoring
            # the default in a finally block.
            try:
                torch._dynamo.utils.set_compilation_metrics_limit(3)
                yield
            finally:
                torch._dynamo.utils.set_compilation_metrics_limit(
                    torch._dynamo.utils.DEFAULT_COMPILATION_METRICS_LIMIT
                )

        x = torch.rand((4, 4))
        torch._dynamo.reset()
        torch.compile(fn1, backend="eager")(x)
        torch.compile(fn2, backend="eager")(x)
        torch.compile(fn3, backend="eager")(x)
        torch.compile(fn4, backend="eager")(x)

        with metrics_limit_ctx():
            torch._dynamo.utils.clear_compilation_metrics()
            torch._dynamo.reset()
            self.assertEqual(0, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn1, backend="eager")(x)
            self.assertEqual(1, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn2, backend="eager")(x)
            self.assertEqual(2, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn3, backend="eager")(x)
            self.assertEqual(3, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn4, backend="eager")(x)
            self.assertEqual(3, len(torch._dynamo.utils.get_compilation_metrics()))

    @skipIfWindows(
        msg="TypeError: sequence item 0: expected str instance, NoneType found"
    )
    def test_funcname_cache(self):
        # [FUNCTION DOC] Skipped on Windows. Writes a source file with nested classes and
        # functions to a temp file and checks the qualified function name looked up for
        # each line.
        src = """\
import torch
if True:
    test = 3

class AAA:
    class DUMMY:
        class DUMMY2:
            pass

    def dummy(self):
        def dummy2():
            pass
    class BBB:
        @staticmethod
        def CCC():
            class DDD:
                if True:
                    @staticmethod
                    def EEE():
                        x = [torch.ones(3, 3) for _ in range(5)]
                        return x
            return DDD
def fn():
    return 3
"""
        with WritableTempFile(mode="w") as f:
            f.write(src)
            f.flush()
            from torch._dynamo.funcname_cache import get_funcname

            names = [get_funcname(f.name, i + 1) for i in range(src.count("\n") + 1)]

        self.assertExpectedInline(
            "\n".join(names),
            """\




AAA
AAA.DUMMY
AAA.DUMMY.DUMMY2
AAA.DUMMY.DUMMY2
AAA.DUMMY.DUMMY2
AAA.dummy
AAA.dummy.dummy2
AAA.dummy.dummy2
AAA.BBB
AAA.BBB
AAA.BBB.CCC
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC
fn
fn
""",
        )

    def test_return_dict_with_graph_break_and_update(self):
        # [FUNCTION DOC] Checks dict unpacking of a dict created by a function containing a
        # graph break.
        def create():
            # [FUNCTION DOC] HELPER: Graph breaks and returns a dict with a tensor.
            torch._dynamo.graph_break()
            return {0: torch.tensor(3)}

        def fn():
            # [FUNCTION DOC] HELPER: Returns a dict unpacked from create.
            return {**create()}

        opt_fn = torch.compile(backend="eager")(fn)
        result = opt_fn()
        self.assertIn(0, result)
        self.assertTrue(same(result[0], torch.tensor(3)))

    def test_dynamo_reset_clears_cache(self):
        # [FUNCTION DOC] Checks dynamo reset clears the bytecode cache entries and the fake
        # tensor cache.
        """Test that dynamo bytecode and fake tensor caches are freed
        when dynamo reset is called
        """
        from torch._subclasses.fake_tensor import FakeTensorMode

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sine of x.
            return torch.sin(x)

        opt_fn = torch.compile(backend="eager")(fn)
        opt_fn(torch.randn(3, 3))

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 1)

        # Under symbolic shapes the dispatch cache lives on the ShapeEnv rather
        # than the process-global FakeTensorMode.cache, so only check the latter
        # is populated when shapes are static.
        if torch._dynamo.config.assume_static_by_default:
            self.assertGreater(len(FakeTensorMode.cache), 0)

        torch._dynamo.reset()
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c2), 0)
        self.assertEqual(len(FakeTensorMode.cache), 0)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_check_simplification(self):
        # [FUNCTION DOC] With scalar capture, checks a check on a floor-division expression
        # simplifies a later equal comparison.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Reads two scalars, checks the expression nonzero, then
            # branches on its being zero.
            u0, u1 = x.tolist()
            torch._check((2 * u0) // (u0 + u1) != 0)
            if (2 * u0) // (u0 + u1) == 0:
                return torch.tensor(True)
            else:
                return torch.tensor(False)

        fn(torch.tensor([3, 3]))

    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_mark_unbacked_strict(self):
        # [FUNCTION DOC] With static default, checks strict unbacked marking on dims of an input
        # raises RuntimeError about specialization.
        @torch.compile(backend="eager")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x times y.
            return torch.mul(x, y)

        x = torch.ones(5, 5)
        torch._dynamo.decorators.mark_unbacked(x, 0, strict=True)
        torch._dynamo.decorators.mark_unbacked(x, 1, strict=True)
        y = torch.randn(5, 5)

        with self.assertRaisesRegex(RuntimeError, "specialized"):
            fn(x, y)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_infer_unbacked_size_gt_zero(self):
        # [FUNCTION DOC] With scalar capture, checks creating zeros with an unbacked size and
        # comparing the size below zero compiles.
        # This code, in fact, does NOT work in eager
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates zeros of item size, asserts false if its size is
            # negative, returns it.
            y = torch.zeros(x.item())
            if y.size(0) < 0:
                assert False  # noqa: B011, S101
            return y

        self.assertEqual(fn(torch.tensor([0])), torch.zeros(0))

    @torch.fx.experimental._config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_strict_mode(self):
        # [FUNCTION DOC] With data-dependent graph break disabled, checks branching on an
        # unbacked dim raises RuntimeError about guarding on a data-dependent expression.
        @torch.compile(backend="eager")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns random tensor if first dim is 5 else product.
            if x.shape[0] == 5:
                return torch.randn(5)
            return torch.mul(x, y)

        x = torch.ones(5, 5)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        y = torch.randn(5, 5)
        with self.assertRaisesRegex(
            RuntimeError, "Could not guard on data-dependent expression"
        ):
            fn(x, y)

    def test_guard_size_oblivious_backed(self):
        # [FUNCTION DOC] Checks guard_size_oblivious on a backed size works for sizes 0 and 2.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns random tensors of size 1 or 2 based on a size-
            # oblivious comparison.
            y = x.size(0)
            # This doesn't actually do anything
            if guard_size_oblivious(y == 0):
                return torch.randn(1)
            else:
                return torch.randn(2)

        # Should not fail in either case
        self.assertEqual(f(torch.randn(0)).shape, (1,))
        self.assertEqual(f(torch.randn(2)).shape, (2,))

    def _test_compile_model_free(self, model_inp_ctr, weakref_watch):
        # [FUNCTION DOC] HELPER: Takes a constructor returning model and inputs and a function
        # selecting the object to watch; compiles the model, drops it, collects garbage and
        # asserts the finalizer ran.
        """
        Args:
        model_inp_ctr
            - constructor that returns a new model and inputs to that model
        weakref_watch
            - function that returns a layer of the model for weakref to
              finalize on, so we can check that the layer is freed after
              the model goes out of scope
        """
        cleared = False

        def finalize():
            # [FUNCTION DOC] HELPER: Marks cleared as True.
            nonlocal cleared
            cleared = True

        def run():
            # [FUNCTION DOC] HELPER: Builds model and inputs, registers a finalizer on the
            # watched layer, compiles and runs the model.
            mod, inp = model_inp_ctr()
            weakref.finalize(weakref_watch(mod), finalize)
            torch.compile(mod, backend="eager")(inp)

        run()
        gc.collect()
        self.assertTrue(cleared)

    def test_custom_module_free(self):
        # [FUNCTION DOC] Checks a custom module is freed after compile.
        """Test that a model is freed when it goes out of scope"""

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a linear layer.
                super(Mod, self).__init__()
                self.fc = torch.nn.Linear(100, 100)

            def forward(self, out):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.fc(out)

        self._test_compile_model_free(
            lambda: (Mod(), torch.randn(100, 100)),
            lambda mod: mod.fc,
        )

    def test_sequential_module_free(self):
        # [FUNCTION DOC] Checks a sequential module is freed (watching first layer).
        self._test_compile_model_free(
            lambda: (
                torch.nn.Sequential(
                    torch.nn.Linear(100, 100),
                    torch.nn.ReLU(),
                ),
                torch.randn(100, 100),
            ),
            lambda mod: mod[0],
        )

    def test_linear_module_free(self):
        # [FUNCTION DOC] Checks a linear module is freed.
        self._test_compile_model_free(
            lambda: (torch.nn.Linear(100, 100), torch.randn(100, 100)),
            lambda mod: mod,
        )

    def test_outside_linear_module_free(self):
        # [FUNCTION DOC] Checks a linear layer referenced by a module is freed after compile
        # without the shared helper.
        # Compared to test_linear_module_free, the linear
        # layer is not the code object that is directly compiled.

        # This test does not use _test_compile_model_free because of difficulty
        # in handling variable fc.

        cleared = False

        def finalize():
            # [FUNCTION DOC] HELPER: Marks cleared as True.
            nonlocal cleared
            cleared = True

        def run():
            # [FUNCTION DOC] HELPER: Creates a linear layer and module referencing it, registers
            # finalizer on layer, compiles and runs.
            fc = torch.nn.Linear(100, 100)

            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module holding a reference to the outer linear layer.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Stores fc.
                    super().__init__()
                    self.fc_ref = fc

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the stored layer.
                    return self.fc_ref(x)

            mod = Mod()
            inp = torch.randn(100, 100)
            weakref.finalize(fc, finalize)
            torch.compile(mod, backend="eager")(inp)

        run()
        # del fc  # This should delete all the references
        gc.collect()
        self.assertTrue(cleared)

    def test_parameter_free(self):
        # [FUNCTION DOC] Checks a parameter held by a module and kept alive by inputs is freed
        # after compile.
        def model_inp_ctr():
            # [FUNCTION DOC] HELPER: Builds a parameter, a module using it and inputs including
            # the parameter.
            param = torch.nn.Parameter(torch.randn(100, 100))

            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module referencing the parameter.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Stores param.
                    super().__init__()
                    self.param = param

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Multiplies the parameter by the first input.
                    return self.param * x[0]

            # return param to keep it alive in _test_compile_model_free
            return Mod(), (torch.randn(100, 100), param)

        self._test_compile_model_free(model_inp_ctr, lambda mod: mod.param)

    def test_conditional_list_comp_in_context(self):
        # [FUNCTION DOC] Checks a conditional list comprehension within try compiles.
        def fn(inp):
            # [FUNCTION DOC] HELPER: Returns sine for non-None entries, catching exceptions.
            try:
                return [torch.sin(x) for x in inp if x is not None]
            except Exception:
                pass

        inp = [torch.randn(3, 3) for _ in range(3)] + [None]
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(inp)

    def test_312_binary_slice_with_graph_break1(self):
        # [FUNCTION DOC] Checks sequence slicing on a sequential with a graph break and items on
        # the stack (3.12 binary slice).
        l1 = torch.nn.Linear(5, 5)
        l2 = torch.nn.Linear(5, 5)

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds sequential of two layers and applies from index 1.
            # causes a graph break with items in the stack
            n = torch.nn.Sequential(l1, l2)
            out = n[1:](x)
            return out

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(5, 5))

    def test_312_binary_slice_with_graph_break2(self):
        # [FUNCTION DOC] Checks graph breaks in slice store and slice load instructions of a
        # user object.
        class Foo:
            # [CLASS DOC] HELPER: Object with set and get item methods; get item graph breaks.
            def __setitem__(self, key, val):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Graph breaks and returns 1.
                torch._dynamo.graph_break()
                return 1

        foo = Foo()

        def fn(x):
            # [FUNCTION DOC] HELPER: Assigns a slice, adds foo slice, and branches with
            # identical arms.
            # graph break in a STORE_SLICE instruction
            foo[:] = x
            # graph break in BINARY_SLICE with has_backedge check
            x = x + foo[:]
            if x is None:
                x = x + 1
            else:
                x = x + 1
            return x

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(5, 5))

    def test_super_after_graph_break(self):
        # [FUNCTION DOC] Checks calling the parent initializer after a graph break in a subclass
        # of sequential.
        class Foo(torch.nn.Sequential):
            # [CLASS DOC] HELPER: Sequential subclass whose init graph breaks before
            # calling super.
            def __init__(self, layers):
                # [FUNCTION DOC] HELPER: Graph breaks then calls base init with layers.
                torch._dynamo.graph_break()
                super().__init__(*layers)

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds three layers, constructs Foo and applies it.
            layers = [torch.nn.Linear(3, 3) for _ in range(3)]
            mod = Foo(layers)
            return mod(x)

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(3, 3))

    def test_load_fast_and_clear_graph_break(self):
        # [FUNCTION DOC] Checks comprehension variable clearing instruction across a graph break
        # (could crash in 3.12+).
        # Can result in a segfault in 3.12+ if LOAD_FAST_AND_CLEAR
        # is not handled properly in a graph break
        def fn():
            # [FUNCTION DOC] HELPER: Concatenates comprehension results, graph breaks, repeats.
            out = torch.cat([torch.randn(r, 5) for r in range(3)])
            torch._dynamo.graph_break()
            out = torch.cat([torch.randn(r, 5) for r in range(3)])
            return out

        self.assertEqual(torch.compile(fn, backend="eager")().shape, (3, 5))

    def test_raises_importerror1(self):
        # [FUNCTION DOC] Checks import of a nonexistent module inside try/except ImportError in
        # compiled code.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Tries the failing import, falls through to sine.
            try:
                import some_module_that_surely_does_not_exist

                return
            except ImportError:
                pass
            return x.sin()

        x = torch.randn(8)
        self.assertEqual(fn(x), x.sin())

    def test_raises_importerror2(self):
        # [FUNCTION DOC] Checks an unhandled import error propagates from compiled code.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Imports a nonexistent module.
            import some_module_that_surely_does_not_exist

            return x + 1

        x = torch.randn(8)
        with self.assertRaises(ImportError):
            fn(x)

    def test_dynamo_cache_move_to_front(self):
        # [FUNCTION DOC] Checks cache entry order: a repeated call to the second guarded entry
        # moves it to the front.
        def fn(x, const):
            # [FUNCTION DOC] HELPER: Returns x plus const.
            return x + const

        # dynamic=False forces Dynamo to recompile
        opt_fn = torch.compile(fn, backend="eager", dynamic=False)

        inp = torch.randn(3, 3)

        # NOTE: assumes that each cache entry is guarded
        # on unique Mod instance
        opt_fn(inp, 1)
        opt_fn(inp, 2)
        opt_fn(inp, 3)

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 3)

        # move cache entry to front
        opt_fn(inp, 2)
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertIs(c1[1], c2[0])

    def test_inspect_signature_bind(self):
        # [FUNCTION DOC] Checks signature binding with defaults and arguments inside compiled
        # code matches eager.
        import inspect

        def inner(a, b, *ar, c=10, d=11, **kw):
            # [FUNCTION DOC] HELPER: Function with positional, vararg, keyword-default and
            # kwargs parameters.
            pass

        def fn(x, apply_defaults):
            # [FUNCTION DOC] HELPER: Binds arguments, modifies one, optionally applies defaults
            # and returns signature, bound object, parts and x plus 1.
            sig = inspect.signature(inner)
            bound = sig.bind(1, 2, 3, d=12, e=15)
            bound.arguments["d"] = 13
            if apply_defaults:
                bound.apply_defaults()
            return (
                sig,
                bound.signature,
                bound,
                bound.arguments,
                bound.args,
                bound.kwargs,
                x + 1,
            )

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        for apply_defaults in (True, False):
            _, _, bound0, arguments0, args0, kwargs0, _ = fn(
                torch.ones(3, 3), apply_defaults
            )
            _, _, bound1, arguments1, args1, kwargs1, _ = opt_fn(
                torch.ones(3, 3), apply_defaults
            )

            self.assertEqual(bound0, bound1)
            self.assertEqual(arguments0, arguments1)
            self.assertEqual(args0, args1)
            self.assertEqual(kwargs0, kwargs1)
            self.assertTrue(args1)
            self.assertTrue(kwargs1)

    def test_inspect_signature_bind_non_user_function(self):
        # [FUNCTION DOC] Checks signature binding of a class and of a skipped function with
        # keyword defaults.
        import inspect

        class Foo:
            # [CLASS DOC] HELPER: Class with a flexible init signature.
            def __init__(self, a, b, *ar, c=10, d=11, **kw):
                # [FUNCTION DOC] HELPER: Accepts a, b, varargs, keyword defaults and kwargs;
                # does nothing.
                pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Binds arguments to the class signature and returns bound and
            # x plus 1.
            sig = inspect.signature(Foo)
            bound = sig.bind(1, 2, 3, d=12, e=15)
            return bound, x + 1

        opt_fn = torch.compile(fn, backend="eager")
        bound0, _ = fn(torch.ones(3, 3))
        bound1, _ = opt_fn(torch.ones(3, 3))

        self.assertEqual(bound0, bound1)

        import traceback

        # choose a function that is skipped but has defaults
        self.assertTrue(hasattr(traceback.print_exc, "__kwdefaults__"))
        self.assertIs(
            torch._dynamo.trace_rules.lookup(traceback.print_exc),
            torch._dynamo.variables.UserFunctionVariable,
        )

        def gn(x):
            # [FUNCTION DOC] HELPER: Binds an empty argument set to print_exc signature.
            sig = inspect.signature(traceback.print_exc)
            bound = sig.bind()
            return bound, x + 1

        opt_gn = torch.compile(gn, backend="eager", fullgraph=True)
        bound0, _ = gn(torch.ones(3, 3))
        bound1, _ = opt_gn(torch.ones(3, 3))

        self.assertEqual(bound0, bound1)

    def test_sourceless_mapping_proxy(self):
        # [FUNCTION DOC] Checks a mapping proxy created while tracing via the type dict
        # descriptor can be measured.
        # Test that Dynamo can handle a sourceless MappingProxyType.
        # This occurs when type.__dict__['__dict__'].__get__ is called
        # and returns a mappingproxy that was created during tracing.
        _get_dunder_dict = type.__dict__["__dict__"].__get__

        class MyClass:
            # [CLASS DOC] HELPER: Class with two attributes.
            a = 1
            b = 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds length of the class dict mapping to x.
            d = _get_dunder_dict(MyClass)
            return x + len(d)

        t = torch.randn(3)
        ref = fn(t)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(t)
        self.assertEqual(ref, res)

    def test_sourceless_inspect_parameter(self):
        # [FUNCTION DOC] Checks reading an inspect Parameter object from the class dict mapping
        # inside compiled code.
        import inspect

        class MyClass:
            # [CLASS DOC] HELPER: Class holding a Parameter with default 42.
            param = inspect.Parameter(
                "x", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=42
            )

        _get_dunder_dict = type.__dict__["__dict__"].__get__

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads the parameter from the class dict and adds its default
            # to x.
            d = _get_dunder_dict(MyClass)
            p = d["param"]
            return x + p.default

        t = torch.randn(3)
        ref = fn(t)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(t)
        self.assertEqual(ref, res)

    def test_inspect_signature_parameters(self):
        # [FUNCTION DOC] Checks parameters mapping of an inspect signature of a
        # function argument.
        import inspect

        def fn(x, gn):
            # [FUNCTION DOC] HELPER: Branches on whether parameter a has no default.
            d = inspect.signature(gn).parameters
            if d["a"].default is inspect.Parameter.empty:
                return torch.sin(x + 1)
            else:
                return torch.cos(x + 1)

        def gn(a: torch.Tensor, b: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Typed function used as the signature source.
            return a + b

        x = torch.randn(2, 3)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        self.assertEqual(fn(x, gn), opt_fn(x, gn))

    def test_inspect_signature_caching(self):
        # [FUNCTION DOC] Checks signature results are cached per function within a compile:
        # patches output graph cleanup to capture the cache size; result 50 and 2 cache entries.
        """Test that inspect.signature results are cached for repeated calls."""
        import inspect
        from unittest.mock import patch

        def target_func(a, b, c=3):
            # [FUNCTION DOC] HELPER: Returns a plus b plus c.
            return a + b + c

        def other_func(x, y):
            # [FUNCTION DOC] HELPER: Returns x times y.
            return x * y

        def fn():
            # [FUNCTION DOC] HELPER: Loops ten times fetching signatures of both functions and
            # returns the total parameter count.
            results = []
            for _ in range(10):
                sig1 = inspect.signature(target_func)
                sig2 = inspect.signature(other_func)
                results.append(len(sig1.parameters))
                results.append(len(sig2.parameters))
            return sum(results)

        from torch._dynamo.output_graph import OutputGraph

        original_cleanup = OutputGraph.cleanup
        unique_calls = 0

        def capturing_cleanup(self):
            # [FUNCTION DOC] HELPER: Replacement cleanup recording the signature cache size and
            # calling the original.
            nonlocal unique_calls
            unique_calls = len(self.signature_cache)
            return original_cleanup(self)

        with patch.object(OutputGraph, "cleanup", capturing_cleanup):
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled_fn()

        # 10 iterations * (3 params from target_func + 2 params from other_func) = 50
        self.assertEqual(result, 50)
        # signature_cache should have exactly 2 entries (one per unique function)
        self.assertEqual(unique_calls, 2)

    def test_inspect_signature_caching_methods(self):
        # [FUNCTION DOC] Checks signature caching works for bound methods: result 40 and two
        # cache entries (bound methods drop self).
        """Test that inspect.signature caching works for methods."""
        import inspect
        from unittest.mock import patch

        class MyClass:
            # [CLASS DOC] HELPER: Class with two methods.
            def method_a(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns x plus y plus z.
                return x + y + z

            def method_b(self, a):
                # [FUNCTION DOC] HELPER: Returns double a.
                return a * 2

        obj = MyClass()

        def fn():
            # [FUNCTION DOC] HELPER: Loops ten times fetching signatures of two bound methods and
            # returns the total parameter count.
            results = []
            for _ in range(10):
                sig1 = inspect.signature(obj.method_a)
                sig2 = inspect.signature(obj.method_b)
                # Note: bound methods don't include 'self' in signature
                results.append(len(sig1.parameters))
                results.append(len(sig2.parameters))
            return sum(results)

        from torch._dynamo.output_graph import OutputGraph

        original_cleanup = OutputGraph.cleanup
        unique_calls = 0

        def capturing_cleanup(self):
            # [FUNCTION DOC] HELPER: Replacement cleanup recording the signature cache size and
            # calling the original.
            nonlocal unique_calls
            unique_calls = len(self.signature_cache)
            return original_cleanup(self)

        with patch.object(OutputGraph, "cleanup", capturing_cleanup):
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled_fn()

        # 10 iterations * (3 params from method_a + 1 param from method_b) = 40
        self.assertEqual(result, 40)
        # signature_cache should have exactly 2 entries (one per unique method)
        self.assertEqual(unique_calls, 2)

    def test_inspect_variable_redirect(self):
        # [FUNCTION DOC] Checks the inspect helper variable redirects property reads (parameters,
        # kind, name) on signature objects; patches the redirect method to record names.
        """Test that InspectVariable is used and redirects property accesses."""
        import inspect
        from unittest.mock import patch

        from torch._dynamo.variables.user_defined import InspectVariable

        redirected_attrs = []
        original_redirect = InspectVariable._redirect

        def tracking_redirect(self, tx, name):
            # [FUNCTION DOC] HELPER: Wrapper around the redirect method that records the
            # attribute name if it is a redirected property of the value type.
            redirects = self._PROPERTY_REDIRECTS.get(type(self.value), {})
            if name in redirects:
                redirected_attrs.append(name)
            return original_redirect(self, tx, name)

        def fn(x, gn):
            # [FUNCTION DOC] HELPER: Reads a signature, its parameters, a named parameter and
            # returns x plus kind plus name length.
            sig = inspect.signature(gn)
            params = sig.parameters
            param = params["a"]
            return x + param.kind + len(param.name)

        def gn(a: torch.Tensor, b: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Typed function used as the signature source.
            return a + b

        x = torch.randn(2, 3)
        with patch.object(InspectVariable, "_redirect", tracking_redirect):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            result = opt_fn(x, gn)

        self.assertEqual(result, fn(x, gn))
        self.assertIn("parameters", redirected_attrs)
        self.assertIn("kind", redirected_attrs)
        self.assertIn("name", redirected_attrs)

    def test_grad_none(self):
        # [FUNCTION DOC] Checks assigning and mutating grad of an input tensor which starts as
        # None matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sets x.grad to absolute value of y, adds y in place to it,
            # returns absolute y.
            x.grad = torch.abs(y)
            x.grad.add_(y)
            return torch.abs(y)

        y = torch.arange(4).reshape(2, 2).to(torch.float)
        x = torch.randn(2, 2)
        x.grad = None

        z = fn(x, y)
        ref_y = torch.clone(z).detach()
        ref_x_grad = torch.clone(x.grad).detach()

        y = torch.arange(4).reshape(2, 2).to(torch.float)
        x = torch.randn(2, 2)
        x.grad = None

        opt_fn = torch.compile(fn, backend="eager")
        z = opt_fn(x, y)
        self.assertEqual(z, ref_y)
        self.assertEqual(x.grad, ref_x_grad)

    def test_grad_non_none(self):
        # [FUNCTION DOC] Checks in-place update of an existing grad is part of the graph: the
        # graph output has exactly one input node and results match eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds y in place to x.grad and returns absolute y.
            x.grad.add_(y)
            return torch.abs(y)

        y = torch.ones(2, 2)
        x = torch.randn(2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        z = fn(x, y)
        ref_y = torch.clone(z).detach()
        ref_x_grad = torch.clone(x.grad).detach()

        y = torch.ones(2, 2)
        x = torch.randn(2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")
        opt_fn = torch.compile(fn, backend=cnt)
        z = opt_fn(x, y)

        # Ensure that the generated graph returns only one output. We want the
        # add_ on the grad to be part of the graph itself, so that inductor can
        # theoretically move the add_ and resulting copy_ nodes at the right
        # place to free memory.
        self.assertEqual(len(list(cnt.graphs[0].graph.nodes)[-1].all_input_nodes), 1)
        self.assertEqual(z, ref_y)
        self.assertEqual(x.grad, ref_x_grad)

    def test_new_with_int_list(self):
        # [FUNCTION DOC] Checks new with an unpacked size list matches eager in dtype, shape,
        # strides and offset.
        # Make sure torch.Tensor.new(int argument list) behaves the same on dynamo.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a new tensor from the size values and adds 5.
            return x.new(*x.size()) + 5

        optfn = torch.compile(backend="eager")(fn)

        x = torch.arange(10).view(2, 5)

        expected = fn(x)
        actual = optfn(x)

        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.stride(), actual.stride())
        self.assertEqual(expected.storage_offset(), actual.storage_offset())

    def test_dynamic_shapes_as_strided(self):
        # [FUNCTION DOC] Checks as_strided with a zero dim under dynamic shapes matches eager.
        def fn(t, new_size, new_stride):
            # [FUNCTION DOC] HELPER: Strided view flattened and summed, multiplied by t.
            tmp = t.as_strided(new_size, new_stride)
            tmp = tmp.view(-1)
            return t * tmp.sum()

        optfn = torch.compile(backend="eager", dynamic=True)(fn)

        x = torch.randn(3)
        new_size = [0, 3]
        new_stride = [3, 1]

        expected = fn(x, new_size, new_stride)
        actual = optfn(x, new_size, new_stride)

        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.stride(), actual.stride())
        self.assertEqual(expected.storage_offset(), actual.storage_offset())

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_hasattr_nn_module_guard(self):
        # [FUNCTION DOC] With module guarding on, checks hasattr on a module attribute
        # inside forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear layer a.
                super().__init__()
                self.a = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear if attribute a exists, else returns x.
                if hasattr(self, "a"):
                    return self.a(x)
                else:
                    return x

        m = M()
        x = torch.randn(3, 3)
        ref = m(x)

        opt_m = torch.compile(backend="eager")(m)
        res = opt_m(x)
        self.assertEqual(ref, res)

    def test_ordered_dict_move_to_end(self):
        # [FUNCTION DOC] Checks an ordered dict after move_to_end can be read without recompiling
        # on second call.
        d = {
            "foo": 1,
            "bar": 2,
        }

        d = collections.OrderedDict(d)
        d.move_to_end("foo")

        @torch.compile(backend="eager")
        def fn(x, d):
            # [FUNCTION DOC] HELPER: Multiplies x by foo and bar entries.
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_defaultdict(self):
        # [FUNCTION DOC] Checks defaultdict input without recompile on second call.
        d = collections.defaultdict()
        d["foo"] = 1
        d["bar"] = 2

        @torch.compile(backend="eager")
        def fn(x, d):
            # [FUNCTION DOC] HELPER: Multiplies x by foo and bar entries.
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_custom_dict(self):
        # [FUNCTION DOC] Checks dict subclass input without recompile on second call.
        class MyDict(dict):
            # [CLASS DOC] HELPER: Empty dict subclass.
            pass

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = MyDict(d)

        @torch.compile(backend="eager")
        def fn(x, d):
            # [FUNCTION DOC] HELPER: Multiplies x by foo and bar entries.
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_hash_hop(self):
        # [FUNCTION DOC] Checks a higher-order-op object can be used as a dict key
        # under fullgraph.
        associative_scan = importlib.import_module(
            "torch._higher_order_ops.associative_scan"
        )

        @torch.compile(fullgraph=True, backend="eager")
        def fn(y, s):
            # [FUNCTION DOC] HELPER: Stores y under key s and returns it plus 1.0.
            d = dict()
            d[s] = y
            return d[s] + 1.0

        fn(torch.ones(2, 2, device="cpu"), associative_scan.AssociativeScanOp())

    def test_iter_type(self):
        # [FUNCTION DOC] Checks an iterator object is not a list instance under fullgraph.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(y):
            # [FUNCTION DOC] HELPER: Returns y plus 1 for list instance else y plus 2.
            x = iter([])
            if isinstance(x, list):
                return y + 1
            else:
                return y + 2

        res = fn(torch.ones(2))
        self.assertEqual(torch.ones(2) + 2, res)

    def test_descriptor(self):
        # [FUNCTION DOC] Checks a non-data descriptor which sets an attribute on first access;
        # compiled first so the descriptor is hit, then both paths match.
        class lazy_property:
            # [CLASS DOC] HELPER: Lazy property descriptor computing once and caching it on the
            # instance by name.
            def __init__(self, wrapped):
                # [FUNCTION DOC] HELPER: Stores wrapped function.
                self.wrapped = wrapped

            def __get__(self, instance, obj_type=None):
                # [FUNCTION DOC] HELPER: Computes value, stores it as an instance attribute and
                # returns it.
                value = self.wrapped(instance)
                setattr(instance, self.wrapped.__name__, value)
                return value

        class UserDefined:
            # [CLASS DOC] HELPER: Class with a lazy length property and a run method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets a to 3.
                self.a = 3

            @lazy_property
            def length(self):
                # [FUNCTION DOC] HELPER: Returns 3 (wrapped by lazy property).
                return 3

            def run(self, x):
                # [FUNCTION DOC] HELPER: Multiplies x by length.
                return x * self.length

        obj = UserDefined()

        def fn(x):
            # [FUNCTION DOC] HELPER: Runs the global object.
            return obj.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        # Opt_fn is deliberately called first to trigger the __get__ function.
        # Otherwise, the setattr removes the lazy property.
        ref = opt_fn(x)
        res = fn(x)
        self.assertEqual(ref, res)
        ref = opt_fn(x)
        res = fn(x)
        self.assertEqual(ref, res)

    def test_descriptor_side_effect(self):
        # [FUNCTION DOC] Checks a read-only property with a writable dict value (scipy bunch
        # pattern) and an object returned from compiled code.
        # This pattern (readonly descriptor but writable value in `__dict__`) is
        # from scipy `_make_tuple_bunch`:
        # https://github.com/scipy/scipy/blob/maintenance/1.9.x/scipy/_lib/_bunch.py#L32-L226
        def fget(obj):
            # [FUNCTION DOC] HELPER: Reads the field from the instance dict.
            return obj.__dict__["field"]

        class MyClass:
            # [CLASS DOC] HELPER: Class writing field directly in the instance dict and exposing
            # a read-only property.
            def __init__(self, n):
                # [FUNCTION DOC] HELPER: Stores n in the dict.
                self.__dict__["field"] = n

            field = property(fget)

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates object and returns x plus field with the object.
            obj = MyClass(42)
            return x + obj.field, obj

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref_t, ref_obj = fn(x)
        res_t, res_obj = opt_fn(x)
        self.assertEqual(ref_t, res_t)
        self.assertEqual(ref_obj.field, res_obj.field)

    def test_data_descriptor_priority_over_instance_dict(self):
        # [FUNCTION DOC] Checks a property (data descriptor) wins over an instance dict entry of
        # the same name.
        # CPython: data descriptors on the type take priority over instance
        # dict values. Verify Dynamo follows this ordering.
        class Foo:
            # [CLASS DOC] HELPER: Class with property x returning 10.
            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Returns 10.
                return 10

        foo = Foo()
        # Manually put a different value in the instance dict.
        # The property (data descriptor) should still win.
        foo.__dict__["x"] = 999

        def fn(t):
            # [FUNCTION DOC] HELPER: Adds foo.x to t.
            return t + foo.x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        ref = fn(t)
        res = opt_fn(t)
        self.assertEqual(ref, res)
        self.assertEqual(ref, t + 10)

    def test_instance_dict_priority_over_non_data_descriptor(self):
        # [FUNCTION DOC] Checks an instance dict entry wins over a non-data descriptor (value 10,
        # not 700).
        # CPython: instance dict values take priority over non-data
        # descriptors (those with only __get__, no __set__/__delete__).
        class Desc:
            # [CLASS DOC] HELPER: Non-data descriptor returning stored value times 100.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores val.
                self.val = val

            def __get__(self, obj, cls):
                # [FUNCTION DOC] HELPER: Returns val times 100.
                return self.val * 100

        class Foo:
            # [CLASS DOC] HELPER: Class with descriptor attribute x.
            x = Desc(7)

        foo = Foo()
        # Instance dict value should shadow the non-data descriptor.
        foo.__dict__["x"] = 10

        def fn(t):
            # [FUNCTION DOC] HELPER: Adds foo.x to t.
            return t + foo.x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        ref = fn(t)
        res = opt_fn(t)
        self.assertEqual(ref, res)
        # CPython: instance dict wins → foo.x is 10, not 700
        self.assertEqual(ref, t + 10)

    def test_user_defined_data_descriptor(self):
        # [FUNCTION DOC] Checks a user data descriptor storing values in underscore-prefixed
        # attributes is used even when the instance sets the name.
        # A user-defined data descriptor (has __get__ + __set__) on the type
        # should be invoked even when the same name exists in the instance dict.
        class ValidatedAttr:
            # [CLASS DOC] HELPER: Data descriptor deriving storage name from its attribute name.
            def __set_name__(self, owner, name):
                # [FUNCTION DOC] HELPER: Records the storage name.
                self.storage_name = "_" + name

            def __get__(self, obj, cls):
                # [FUNCTION DOC] HELPER: Returns self on class access else the stored value.
                if obj is None:
                    return self
                return getattr(obj, self.storage_name)

            def __set__(self, obj, value):
                # [FUNCTION DOC] HELPER: Stores the value under storage name.
                setattr(obj, self.storage_name, value)

        class Foo:
            # [CLASS DOC] HELPER: Class using the descriptor.
            x = ValidatedAttr()

            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Assigns x through the descriptor.
                self.x = x

        foo = Foo(10)

        def fn(t):
            # [FUNCTION DOC] HELPER: Adds foo.x to t.
            return t + foo.x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        ref = fn(t)
        res = opt_fn(t)
        self.assertEqual(ref, res)
        self.assertEqual(ref, t + 10)

    def test_property_setter(self):
        # [FUNCTION DOC] Checks a property with a setter used from compiled code.
        class Box:
            # [CLASS DOC] HELPER: Class holding a tensor with a property returning value plus 1
            # and a setter that doubles.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores a float tensor.
                self._value = torch.tensor([value], dtype=torch.float32)

            @property
            def value(self):
                # [FUNCTION DOC] HELPER: Returns stored tensor plus 1.
                return self._value + 1

            @value.setter
            def value(self, new_value):
                # [FUNCTION DOC] HELPER: Stores new value times 2.
                self._value = new_value * 2

        def fn(b):
            # [FUNCTION DOC] HELPER: Sets value to 5 and returns it.
            b.value = 5
            return b.value

        b = Box(0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(Box(0))
        res = opt_fn(b)
        self.assertEqual(ref, res)

    def test_property_setter_in_init(self):
        # [FUNCTION DOC] Checks a property setter invoked from init clamps values.
        class Clamped:
            # [CLASS DOC] HELPER: Class with a clamping property.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Assigns val through the property.
                self.val = val

            @property
            def val(self):
                # [FUNCTION DOC] HELPER: Returns stored value.
                return self._val

            @val.setter
            def val(self, v):
                # [FUNCTION DOC] HELPER: Stores clamp of v in 0..100.
                self._val = torch.clamp(v, 0, 100)

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates object and returns val.
            obj = Clamped(x)
            return obj.val

        x = torch.tensor([200.0])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_assert_size_stride(self):
        # [FUNCTION DOC] Checks the size-and-stride assertion helper raises AssertionError
        # listing each mismatching dim with expected and actual values.
        x = torch.randn(2, 3, 4)
        with self.assertRaisesRegex(
            AssertionError,
            "expected size 2==5, stride 12==9 at dim=0; expected size 3==6, stride 4==9 at dim=1; expected size 4==7, stride 1==10 at dim=2",
        ):
            torch._C._dynamo.guards.assert_size_stride(x, (5, 6, 7), (9, 9, 10))

    def test_frozen_dict(self):
        # [FUNCTION DOC] Checks a frozen ordered dict subclass pattern from StableDiffusion
        # converted to dict inside compiled code.
        # KNOWN ISSUE: the freeze flag name-mangles so the freeze check in setattr and setitem
        # never triggers.
        # A pattern from StableDiffusion
        class FrozenDict(collections.OrderedDict):
            # [CLASS DOC] HELPER: Ordered dict subclass that copies keys to attributes, intended
            # to forbid mutation after init.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls base init, sets attributes for each item, sets
                # frozen flag.
                super().__init__(*args, **kwargs)

                for key, value in self.items():
                    setattr(self, key, value)

                self.__frozen = True

            def __delitem__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises.
                raise Exception(
                    f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
                )

            def setdefault(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises.
                raise Exception(
                    f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
                )

            def pop(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises.
                raise Exception(
                    f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
                )

            def update(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises.
                raise Exception(
                    f"You cannot use ``update`` on a {self.__class__.__name__} instance."
                )

            def __setattr__(self, name, value):
                # [FUNCTION DOC] HELPER: Raises if frozen, else sets attribute.
                # KNOWN ISSUE: the check looks up a literal frozen attribute name which does not
                # match the mangled attribute set in init, so it never raises.
                if hasattr(self, "__frozen") and self.__frozen:
                    raise Exception(
                        f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
                    )
                super().__setattr__(name, value)

            def __setitem__(self, name, value):
                # [FUNCTION DOC] HELPER: Raises if frozen, else sets the item.
                # KNOWN ISSUE: the same mangled-name check; also the message mentions setattr.
                if hasattr(self, "__frozen") and self.__frozen:
                    raise Exception(
                        f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
                    )
                super().__setitem__(name, value)

        d = {"a": 1}
        frozen_d = FrozenDict(d)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict from the frozen dict and returns sine of x.
            dict(frozen_d).items()
            return torch.sin(x)

        fn(torch.randn(4))

    def test_tuple_class(self):
        # [FUNCTION DOC] Checks the class of a tuple input reused to build a new tuple of
        # incremented tensors; one frame.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Increments each element and builds the same sequence class.
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)

        r = opt_fn((d1, d2))
        self.assertEqual(r.__class__, tuple)
        r1, r2 = r
        self.assertEqual(r1, torch.ones(2, 2))
        self.assertEqual(r2, torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_list_class(self):
        # [FUNCTION DOC] Same for a list input.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Increments each element and builds the same sequence class.
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)

        r = opt_fn([d1, d2])
        self.assertEqual(r.__class__, list)
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], torch.ones(2, 2))
        self.assertEqual(r[1], torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_namedtuple_class(self):
        # [FUNCTION DOC] Same for a namedtuple input built with unpacked values.
        import collections

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Increments each element and builds the same class from
            # unpacked values.
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(*updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)
        point = collections.namedtuple("Point", ["x", "y"])
        p = point(d1, d2)

        r = opt_fn(p)
        self.assertEqual(r.__class__, point)
        self.assertEqual(r.x, torch.ones(2, 2))
        self.assertEqual(r.y, torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_getattrvariable_as_python_constant(self):
        # [FUNCTION DOC] Checks that a patched random object attribute method is tracked as
        # python constant by the call-method variable (the patched method is called at
        # least once).
        from torch._dynamo.variables.misc import CallMethodVariable as CMV

        @torch.compile(backend="eager")
        def fn(x, rand1):
            # [FUNCTION DOC] HELPER: Copies state to a fresh random object and returns x plus a
            # random number.
            random.Random().setstate(rand1.getstate())
            return x + rand1.random()

        def get_rng():
            # [FUNCTION DOC] HELPER: Builds a seeded random generator with a wrapped
            # random method.
            rand1 = random.Random(1)
            orig_random = rand1.random
            rand1.random = lambda: orig_random()
            return rand1

        x = torch.randn(3, 3)
        expected = fn.__wrapped__(x, get_rng())

        with patch.object(CMV, "as_python_constant", autospec=True) as po:
            actual = fn(x, get_rng())

        self.assertEqual(expected, actual)
        self.assertGreater(po.call_count, 0)

    def test_data_ptr_graph_break_builtin(self):
        # [FUNCTION DOC] Checks adding two data pointers is unsupported under fullgraph, and
        # falls back with the right result otherwise.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns sum of data pointers of a and b.
            # builtin + not implemented for DataPtrVariable
            return a.data_ptr() + b.data_ptr()

        a = torch.randn(4)
        b = torch.randn(5)

        # make sure there is a graph break
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(f, backend="eager", fullgraph=True)(a, b)

        torch._dynamo.reset()

        expected = f(a, b)
        actual = torch.compile(f, backend="eager")(a, b)

        self.assertEqual(expected, actual)

    def test_data_ptr_detach_equality_fullgraph(self):
        # [FUNCTION DOC] Checks comparing data pointer, stride and shape of a tensor and its
        # detach under fullgraph.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns a long tensor of whether pointer, stride and
            # shape match.
            detached = x.detach()
            same_data = (
                x.data_ptr() == detached.data_ptr()
                and x.stride() == detached.stride()
                and x.shape == detached.shape
            )
            return torch.tensor([same_data], dtype=torch.long, device=x.device)

        x = torch.randn(1, 1, 1, 3)

        expected = f(x)
        actual = torch.compile(f, backend="eager", fullgraph=True)(x)

        self.assertEqual(expected, actual)

    def test_data_ptr_equality_after_mutation_graph_break(self):
        # [FUNCTION DOC] Checks comparing data pointers before and after an in-place mutation is
        # unsupported under fullgraph but works with graph breaks.
        def f(x):
            # [FUNCTION DOC] HELPER: Records pointer, mutates in place, compares with
            # detached pointer.
            ptr_before = x.data_ptr()
            x.add_(1)
            ptr_after = x.detach().data_ptr()
            return torch.tensor([ptr_before == ptr_after], dtype=torch.long)

        x = torch.randn(4)

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(f, backend="eager", fullgraph=True)(x.clone())

        expected = f(x.clone())
        actual = torch.compile(f, backend="eager")(x.clone())

        self.assertEqual(expected, actual)

    def test_data_ptr_constant_comparison_graph_break(self):
        # [FUNCTION DOC] Checks comparison of data pointers with constants (all four forms) is
        # unsupported under fullgraph and falls back with correct results.
        def ptr_eq_zero(x):
            # [FUNCTION DOC] HELPER: Returns whether pointer equals zero.
            return x.data_ptr() == 0

        def zero_eq_ptr(x):
            # [FUNCTION DOC] HELPER: Returns whether zero equals pointer.
            return 0 == x.data_ptr()

        def ptr_ne_zero(x):
            # [FUNCTION DOC] HELPER: Returns whether pointer differs from zero.
            return x.data_ptr() != 0

        def zero_ne_ptr(x):
            # [FUNCTION DOC] HELPER: Returns whether zero differs from pointer.
            return 0 != x.data_ptr()

        x = torch.randn(4)

        for fn in (ptr_eq_zero, zero_eq_ptr, ptr_ne_zero, zero_ne_ptr):
            with self.assertRaises(torch._dynamo.exc.Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(x)

            torch._dynamo.reset()

            expected = fn(x)
            actual = torch.compile(fn, backend="eager")(x)

            self.assertEqual(expected, actual)
            torch._dynamo.reset()

    def test_data_ptr_graph_break_aten(self):
        # [FUNCTION DOC] Checks adding a data pointer to a tensor graph breaks and gives
        # eager results.
        def f(a):
            # [FUNCTION DOC] HELPER: Adds the data pointer to a.
            # torch.add not implemented for DataPtrVariable
            return torch.add(a, a.data_ptr())

        a = torch.randn(4)

        counters.clear()

        expected = f(a)
        actual = torch.compile(f, backend="eager")(a)

        self.assertEqual(expected, actual)
        self.assertTrue(len(counters["graph_break"]) > 0)
        counters.clear()

    class AssertNumOutputBackend:
        # [CLASS DOC] HELPER: Backend which runs the graph and asserts the number of outputs
        # equals the expected count, then returns the graph.
        """
        A backend that checks the number of output for compiled graph, and
        return the graph as is.
        """

        def __init__(self, test_case, expected_num_output: int):
            # [FUNCTION DOC] HELPER: Stores the test case and expected output count.
            self.test_case = test_case
            self.expected_num_output = expected_num_output

        def __call__(self, gm: torch.fx.GraphModule, example_inputs):
            # [FUNCTION DOC] HELPER: Runs the graph module on the inputs, asserts output count
            # and returns the module.
            outputs = gm(*example_inputs)
            self.test_case.assertEqual(self.expected_num_output, len(outputs))
            return gm

    def test_returning_nested_func_with_captured_tensor(self):
        # [FUNCTION DOC] Checks a returned nested function capturing a tensor makes the compiled
        # graph output two values (the tensor and result).
        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def test():
            # [FUNCTION DOC] HELPER: Creates tensor and nested function, returns the function and
            # its call result.
            x = torch.rand(1)

            def func():
                # [FUNCTION DOC] HELPER: Returns captured tensor added to itself.
                return x + x

            # Returning `func` forces dynamo to output `x` in the compiled
            # graph, so that we can store it as `func`'s closure. The output of
            # compiled graph would be `(x, x + x)`.
            return func, func()

        test()

    def test_running_nested_func_with_captured_tensor(self):
        # [FUNCTION DOC] Checks a nested function that is only called leaves one graph output.
        @torch.compile(backend=self.AssertNumOutputBackend(self, 1))
        def test():
            # [FUNCTION DOC] HELPER: Creates tensor and nested function, returns the call result.
            x = torch.rand(1)

            def func():
                # [FUNCTION DOC] HELPER: Returns captured tensor added to itself.
                return x + x

            # `x` is no longer needed after running the compiled graph, so we
            # shouldn't return it. The output of compiled graph would be `(x +
            # x,)`.
            return func()

        test()

    def test_returning_func_with_captured_func_and_tensor(self):
        # [FUNCTION DOC] Checks a returned function capturing another function and tensor gives
        # two graph outputs.
        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def test():
            # [FUNCTION DOC] HELPER: Creates tensor and functions, returns the outer function and
            # its call result.
            x = torch.rand(1)

            def nested():
                # [FUNCTION DOC] HELPER: Returns captured tensor added to itself.
                return x + x

            def func():
                # [FUNCTION DOC] HELPER: Calls nested.
                return nested()

            # Returning `func` forces dynamo to output `x` in the compiled
            # graph, so that we can store it as `func`'s closure. The output of
            # compiled graph would be `(x, x + x)`.
            return func, func()

        test()

    def test_running_func_with_captured_func_and_tensor(self):
        # [FUNCTION DOC] Checks a called-only nested function chain gives one graph output.
        @torch.compile(backend=self.AssertNumOutputBackend(self, 1))
        def test():
            # [FUNCTION DOC] HELPER: Creates tensor and functions, returns the call result.
            x = torch.rand(1)

            def nested():
                # [FUNCTION DOC] HELPER: Returns captured tensor added to itself.
                return x + x

            def func():
                # [FUNCTION DOC] HELPER: Calls nested.
                return nested()

            # `x` is no longer needed after running the compiled graph, so we
            # shouldn't return it. The output of compiled graph would be `(x)`.
            return func()

        test()

    def test_escaping_closure_var_with_backward_hook(self):
        # [FUNCTION DOC] Checks a lambda registered as a backward hook captures a local that
        # escapes; two graph outputs; backward runs.
        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x, registers a hook returning a captured value,
            # returns the square.
            temp = x * x
            captured_var = temp + 1

            # This is where the lambda escapes the lifetime of `fn`, so
            # dynamo must generate proper bytecode to update `captured_var`.
            x.register_hook(lambda _: captured_var)

            # The output of compiled graph would be `(x * x, x * x + 1)`.
            return temp

        ones = torch.ones(4, requires_grad=True)
        fn(ones).sum().backward()

    def test_escaping_closure_var_with_nonlocal_var(self):
        # [FUNCTION DOC] Checks a closure escaping through a nonlocal variable captures a local;
        # two graph outputs; the escaped function is called afterwards.
        nonlocal_fn = None

        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x, creates a closure returning a captured value and
            # stores it in the enclosing variable.
            temp = x * x
            captured_var = x + 1

            def inner():
                # [FUNCTION DOC] HELPER: Returns the captured variable.
                return captured_var

            # This is where `inner` escapes the lifetime of `fn`, so dynamo must
            # generate proper bytecode to update `captured_var`.
            nonlocal nonlocal_fn
            nonlocal_fn = inner

            # The output of compiled graph would be `(x * x, x * x + 1)`.
            return temp

        ones = torch.ones(4, requires_grad=True)
        fn(ones)
        nonlocal_fn()

    def test_compare_tensor_with_none(self):
        # [FUNCTION DOC] Checks comparing a tensor to None produces false.
        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Returns a tensor holding the result of comparing x to None.
            return torch.tensor(x == None)

        res = f(torch.tensor(1))
        self.assertEqual(torch.tensor(False), res)

    def test_dataclass(self):
        # [FUNCTION DOC] Checks dataclass introspection and replace on a frozen dataclass in
        # compiled code.
        # KNOWN ISSUE: the tensor check uses assertTrue with a second argument, which is only
        # a message.
        @dataclasses.dataclass(frozen=True)
        class Foo:
            # [CLASS DOC] HELPER: Frozen dataclass with an int.
            x: int

        @torch.compile(backend="eager", fullgraph=True)
        def run(x, foo0):
            # [FUNCTION DOC] HELPER: Replaces the field with 1 if the object is a dataclass and
            # returns x plus 1 with the new object.
            if dataclasses.is_dataclass(foo0):
                foo1 = dataclasses.replace(foo0, **{"x": 1})
                return x + 1, foo1
            return x + 2, foo0

        res, foo = run(torch.zeros(1), Foo(0))
        self.assertTrue(res, torch.ones(1))
        self.assertEqual(foo.x, 1)

    def test_frozenset_of_non_literals(self):
        # [FUNCTION DOC] Checks a frozenset holding a non-literal object shares representation
        # with the same object so a mutation is visible via both.
        class Foo:
            # [CLASS DOC] HELPER: Empty class.
            pass

        foo = Foo()
        foo.x = 0
        s = frozenset([foo])

        @torch.compile(backend="eager")
        def run(x, s, foo0):
            # [FUNCTION DOC] HELPER: Reads first element of the set, increments foo0 attribute,
            # returns x plus 1 and the set element attribute.
            # Dynamo must have the same representation for `foo0` and `foo1`,
            # otherwise the update to `foo0.x` won't be reflected in the read of
            # `foo1.x`.
            foo1 = list(s)[0]
            foo0.x += 1
            return x + 1, foo1.x

        res = run(torch.ones(1), s, foo)
        self.assertTrue(same(res[0], torch.ones(1) + 1))
        self.assertEqual(res[1], 1)

    def test_ne_operator_with_custom_eq(self):
        # [FUNCTION DOC] Checks not-equal falls back on a custom equality method.
        class Foo:
            # [CLASS DOC] HELPER: Class with custom equality comparing x.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compares x fields.
                return self.x == other.x

        @torch.compile(fullgraph=True, backend="eager")
        def run(x):
            # [FUNCTION DOC] HELPER: Creates two objects and returns x plus 1 with
            # their inequality.
            f1 = Foo(0)
            f2 = Foo(0)
            # `x + 1` prevents Dynamo from skipping this frame.
            return x + 1, f1 != f2

        _, ne = run(torch.ones(1))
        self.assertFalse(ne)

    def test_ne_operator_with_custom_ne(self):
        # [FUNCTION DOC] Checks a custom not-equal method is traced (its flag is set).
        class Foo:
            # [CLASS DOC] HELPER: Class with custom not-equal setting a flag and
            # negating equality.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x and clears flag.
                self.x = x
                self.ne_called = False

            def __ne__(self, other):
                # [FUNCTION DOC] HELPER: Sets the flag and returns the negation of equality.
                # ne_called attr is later checked to ensure that overridden
                # `__ne__` is traced
                self.ne_called = True
                return not self.__eq__(other)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compares x fields.
                return self.x == other.x

        f1 = Foo(0)
        f2 = Foo(0)

        @torch.compile(fullgraph=True, backend="eager")
        def run(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 with the inequality of two outer objects.
            # `x + 1` prevents Dynamo from skipping this frame.
            return x + 1, f1 != f2

        _, ne = run(torch.ones(1))
        self.assertFalse(ne)
        self.assertTrue(f1.ne_called)

    def test_ne_operator_with_custom_graphbreak_eq(self):
        # [FUNCTION DOC] Checks a not-equal on a class whose equality method graph breaks traces
        # into it (one graph break).
        counters.clear()

        class Foo:
            # [CLASS DOC] HELPER: Class with equality that graph breaks.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Graph breaks and compares x fields.
                # This allows us to check that Dynamo actually traced into the
                # custom eq method.
                torch._dynamo.graph_break()
                return self.x == other.x

        @torch.compile(backend="eager")
        def run(x):
            # [FUNCTION DOC] HELPER: Creates two objects and returns x plus 1 with
            # their inequality.
            f1 = Foo(0)
            f2 = Foo(0)
            # `x + 1` prevents Dynamo from skipping this frame.
            return x + 1, f1 != f2

        _, ne = run(torch.ones(1))
        self.assertFalse(ne)
        self.assertEqual(len(counters["graph_break"]), 1)

    @unittest.skipIf(sys.version_info < (3, 12), "Python 3.12+")
    def test_CALL_INTRINSIC(self):
        # [FUNCTION DOC] Python 3.12+ only. Runs a shared helper test for intrinsic
        # call instructions.
        from torch.testing._internal.py312_intrinsics import Foo

        Foo.test_default_update(self)

    @unittest.skipIf(sys.version_info < (3, 11), "Python 3.11+")
    def test_RAISE_VARARGS_0(self):
        # [FUNCTION DOC] Python 3.11+ only. Checks a bare re-raise inside an except block is
        # caught by the matching handler in compiled code.
        def foo():
            # [FUNCTION DOC] HELPER: Raises ValueError and re-raises it in a bare handler.
            try:
                raise ValueError
            except:
                raise

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Returns sine on ValueError, cosine on other exceptions.
            try:
                foo()
            except ValueError:
                return t.sin()
            except Exception:
                return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())

    def test_overridden_getattribute(self):
        # [FUNCTION DOC] Checks classes overriding attribute lookup, assignment and fallback
        # with an alias map, a sentinel handled by the fallback and a nested object, compiled
        # under fullgraph.
        class Bar:
            # [CLASS DOC] HELPER: Holder with a value.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores v.
                self.v = v

        class Foo:
            # [CLASS DOC] HELPER: Class with alias map and custom attribute hooks.
            attribute_map = {}

            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets alias map and a nested Bar.
                self.attribute_map = {
                    "a_premap": "a",
                }
                # `bar` attribute requires propagating sources correctly through
                # object.__getattribute__
                self.bar = Bar(5)

            def __setattr__(self, key, value):
                # [FUNCTION DOC] HELPER: Maps key through the alias map and sets the attribute.
                if key in super().__getattribute__("attribute_map"):
                    key = super().__getattribute__("attribute_map")[key]
                super().__setattr__(key, value)

            def __getattribute__(self, key):
                # [FUNCTION DOC] HELPER: Raises AttributeError for sentinel, else resolves
                # through the alias map.
                if key == "sentinel":
                    raise AttributeError()
                if key != "attribute_map" and key in super().__getattribute__(
                    "attribute_map"
                ):
                    key = super().__getattribute__("attribute_map")[key]
                return super().__getattribute__(key)

            def __getattr__(self, key):
                # [FUNCTION DOC] HELPER: Returns 5 for sentinel, otherwise raises AttributeError.
                if key == "sentinel":
                    return 5
                raise AttributeError()

        def get_foo():
            # [FUNCTION DOC] HELPER: Creates Foo and sets aliased and normal attributes.
            f = Foo()
            f.a_premap = 2
            f.b = 3
            return f

        def fn(x, f):
            # [FUNCTION DOC] HELPER: Multiplies x by aliased, normal, sentinel and
            # nested attributes.
            return x * f.a_premap * f.a * f.b * f.sentinel * f.bar.v

        x = torch.randn(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, get_foo()), opt_fn(x, get_foo()))

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test___build_class__(self):
        # [FUNCTION DOC] With class-build tracing enabled, checks a class defined inside compiled
        # code is built and used under fullgraph.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(t):
            # [FUNCTION DOC] HELPER: Defines a class, creates an instance and calls its method.
            class NonTensor:
                # [CLASS DOC] HELPER: Class storing a tensor.
                def __init__(self, t):
                    # [FUNCTION DOC] HELPER: Stores t.
                    self.t = t

                def compute(self):
                    # [FUNCTION DOC] HELPER: Returns sine of stored tensor.
                    return self.t.sin()

            obj = NonTensor(t)
            return obj.compute()

        t = torch.randn(2)
        res = fn(t)
        self.assertEqual(res, t.sin())

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_return___build_class__(self):
        # [FUNCTION DOC] With class-build tracing enabled, checks a class defined in compiled
        # code can be returned with its name preserved.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(t):
            # [FUNCTION DOC] HELPER: Defines a class and returns it with sine of t.
            class NonTensor:
                # [CLASS DOC] HELPER: Class storing a tensor.
                def __init__(self, t):
                    # [FUNCTION DOC] HELPER: Stores t.
                    self.t = t

                def compute(self):
                    # [FUNCTION DOC] HELPER: Returns sine of stored tensor.
                    return self.t.sin()

            return NonTensor, t.sin()

        t = torch.randn(2)
        cls, res = fn(t)
        self.assertEqual(res, t.sin())
        self.assertEqual(cls.__name__, "NonTensor")

    @unittest.expectedFailure
    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_return_obj___build_class__(self):
        # [FUNCTION DOC] Expected failure: returning an instance of a class created in compiled
        # code cannot be reconstructed (ephemeral source).
        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn(t):
            # [FUNCTION DOC] HELPER: Defines a class and returns an instance of it.
            # class is created with an EphemeralSource and
            # UserDefinedClassVariable has no `reconstruct` method
            class NonTensor:
                # [CLASS DOC] HELPER: Class storing a tensor.
                def __init__(self, t):
                    # [FUNCTION DOC] HELPER: Stores t.
                    self.t = t

                def compute(self):
                    # [FUNCTION DOC] HELPER: Returns sine of stored tensor.
                    return self.t.sin()

            return NonTensor(t)

        t = torch.randn(2)
        obj = fn(t)
        self.assertEqual(obj.compute(), t.sin())

    @torch._dynamo.config.patch(enable_trace_load_build_class=False)
    def test___build_class___disabled(self):
        # [FUNCTION DOC] With class-build tracing disabled, checks defining a class in compiled
        # code raises Unsupported.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Defines a class, creates an instance and calls its method.
            class NonTensor:
                # [CLASS DOC] HELPER: Class storing a tensor.
                def __init__(self, t):
                    # [FUNCTION DOC] HELPER: Stores t.
                    self.t = t

                def compute(self):
                    # [FUNCTION DOC] HELPER: Returns sine of stored tensor.
                    return self.t.sin()

            obj = NonTensor(t)
            return obj.compute()

        t = torch.randn(2)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            fn(t)

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_dynamically_created_class_object_escape_unimplemented(self):
        # [FUNCTION DOC] Checks returning an instance of a dynamically created class raises
        # Unsupported about reconstructing without a source.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Defines a class and returns an instance.
            class DynamicClass:
                # [CLASS DOC] HELPER: Class storing a tensor.
                def __init__(self, tensor):
                    # [FUNCTION DOC] HELPER: Stores tensor.
                    self.tensor = tensor

                def compute(self):
                    # [FUNCTION DOC] HELPER: Returns sine of the stored tensor.
                    return self.tensor.sin()

            obj = DynamicClass(t)
            # The object escapes the compiled region (implicitly by being returned)
            return obj

        t = torch.randn(2)
        # Should raise Unsupported with the unimplemented error about
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Reconstruct user defined class without a source",
        ):
            fn(t)

    @expectedFailureDynamic
    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_build_class_closure_over_nonconstant(self):
        # [FUNCTION DOC] Expected failure under dynamic shapes. Checks a class defined in
        # compiled code closing over a non-constant object value is guarded: unchanged reuses the
        # frame, mutation recompiles (2 frames).
        class Outer:
            # [CLASS DOC] HELPER: Holder for scale 3.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets scale.
                self.scale = 3

        outer = Outer()

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Defines a wrapper class using outer scale and calls compute.
            class Wrapper:
                # [CLASS DOC] HELPER: Class storing a tensor.
                def __init__(self, t):
                    # [FUNCTION DOC] HELPER: Stores t.
                    self.t = t

                def compute(self):
                    # [FUNCTION DOC] HELPER: Returns sine of the tensor times outer scale.
                    return self.t.sin() * outer.scale

            obj = Wrapper(t)
            return obj.compute()

        t = torch.randn(2)
        self.assertEqual(fn(t), t.sin() * outer.scale)
        # Same value: the EQUALS_MATCH guard on outer.scale holds, no recompile.
        self.assertEqual(fn(t), t.sin() * outer.scale)
        self.assertEqual(cnt.frame_count, 1)
        # Mutating the closed-over non-constant trips the guard -> recompile with
        # the new value (proves scale is guarded, not baked in as a constant).
        outer.scale = 5
        self.assertEqual(fn(t), t.sin() * outer.scale)
        self.assertEqual(cnt.frame_count, 2)

    @expectedFailureDynamic
    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_build_class_closure_over_nonconstant_method(self):
        # [FUNCTION DOC] Expected failure under dynamic shapes. Same idea with a class method
        # calling an outer object method; mutation recompiles.
        class Outer:
            # [CLASS DOC] HELPER: Holder for scale 2 with a getter.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets scale.
                self.scale = 2

            def get_scale(self):
                # [FUNCTION DOC] HELPER: Returns scale.
                return self.scale

        outer = Outer()

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Defines wrapper and adds its keys result to the sum of t.
            class Wrapper:
                # [CLASS DOC] HELPER: Class with a keys method calling the outer getter.
                def keys(self):
                    # [FUNCTION DOC] HELPER: Returns the outer scale.
                    return outer.get_scale()

            return Wrapper().keys() + t.sum()

        t = torch.randn(2)
        self.assertEqual(fn(t), outer.get_scale() + t.sum())
        self.assertEqual(fn(t), outer.get_scale() + t.sum())
        self.assertEqual(cnt.frame_count, 1)
        # Mutating the closed-over value re-guards and recompiles.
        outer.scale = 7
        self.assertEqual(fn(t), outer.get_scale() + t.sum())
        self.assertEqual(cnt.frame_count, 2)

    def test_dunder_weakref(self):
        # [FUNCTION DOC] Checks the weak reference slot descriptor of a plain object is readable
        # in compiled code.
        class Foo:
            # [CLASS DOC] HELPER: Empty class.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates an object and returns cosine or sine based on its
            # weakref attribute.
            foo = Foo()
            # tests isgetsetdescriptor
            if foo.__weakref__:
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_guard_filter_fn_by_id(self):
        # [FUNCTION DOC] Checks a guard filter dropping identity-match guards allows a tensor
        # with a new identity without recompile and returns the first identity.
        def guard_filter_fn(entries):
            # [FUNCTION DOC] HELPER: Returns a list of booleans keeping all guard entries except
            # identity match.
            return [entry.guard_type != "ID_MATCH" for entry in entries]

        @torch.compile(
            fullgraph=True,
            options={"guard_filter_fn": guard_filter_fn},
            backend="eager",
        )
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns id of x.
            return id(x)

        inputs = (torch.randn(3, 2),)
        fn(*inputs)

        inputs_1 = (torch.randn(3, 2),)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs_1), id(inputs[0]))

    def test_guard_filter_fn_by_is_global(self):
        # [FUNCTION DOC] Checks a guard filter dropping global guards: changing the global does
        # not recompile.
        def guard_filter_fn(entries):
            # [FUNCTION DOC] HELPER: Returns keep flags excluding global entries.
            return [not entry.is_global for entry in entries]

        global GLOBAL_INT

        @torch.compile(
            fullgraph=True,
            options={"guard_filter_fn": guard_filter_fn},
            backend="eager",
        )
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds the global integer to x.
            return x + GLOBAL_INT

        GLOBAL_INT = 1
        fn(torch.randn(3, 2))

        GLOBAL_INT = 2
        inputs = (torch.randn(3, 2),)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs), inputs[0] + 1)

    def test_guard_filter_fn_by_name_and_value(self):
        # [FUNCTION DOC] Checks a guard filter dropping guards named y whose value is None lets a
        # tensor y reuse the compiled code.
        def guard_filter_fn(entries):
            # [FUNCTION DOC] HELPER: Returns keep flags dropping guards on y being None.
            return [
                not (entry.name == "y" and entry.value is None) for entry in entries
            ]

        @torch.compile(
            fullgraph=True,
            options={"guard_filter_fn": guard_filter_fn},
            backend="eager",
        )
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds y to x in place if y is not None.
            if y is not None:
                x += y
            return x

        fn(torch.randn(3, 2), None)

        inputs = (torch.randn(3, 2), torch.tensor(1))
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs), inputs[0])

    def test_guard_filter_inbuilt_nn_modules(self):
        # [FUNCTION DOC] Checks the helper that skips guards on built-in modules: changing a
        # layer norm eps does not recompile.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a layer norm.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates layer norm.
                super().__init__()
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies layer norm.
                return self.norm(x)

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            backend="eager",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.norm.eps = 1e-02
        # Since the guards are skipped on inbuilt nn modules, we should not recompile
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_filter_nn_modules(self):
        # [FUNCTION DOC] Checks skipping guards on all modules: changing a plain attribute of a
        # module does not recompile.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with int c and a layer norm.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets c and layer norm.
                super().__init__()
                self.c = 2
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies layer norm and adds c.
                return self.norm(x) + self.c

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            backend="eager",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.c = 3
        # Since the guards are skipped on all nn modules, we should not recompile
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_filter_tensors(self):
        # [FUNCTION DOC] Checks the keep-tensor-guards filter drops non-tensor guards, so
        # changing a float module attribute does not recompile.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with float c and a layer norm.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets c and layer norm.
                super().__init__()
                self.c = 2.0
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies layer norm and adds c.
                return self.norm(x) + self.c

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            backend="eager",
            options={
                "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.c = 3.0
        # Since the guards are skipped on all tensors
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_filter_globals(self):
        # [FUNCTION DOC] Checks the skip-global-guards filter: changing a global integer read in
        # forward does not recompile.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reading a global integer in forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets c and layer norm.
                super().__init__()
                self.c = 2
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies layer norm and adds c and the global integer.
                return self.norm(x) + self.c + GLOBAL_INT

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            backend="eager",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
            },
        )

        global GLOBAL_INT
        GLOBAL_INT = 1
        x = torch.rand(4, 8)
        opt_mod(x)

        GLOBAL_INT = 2
        # Since the guards are skipped on globals, we should not recompile
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symint(self):
        # [FUNCTION DOC] With scalar capture, checks bool of an item from an int tensor
        # under fullgraph.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns bool of the item.
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randint(10, (1,))

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_new_tensor_break(self):
        # [FUNCTION DOC] With scalar and dynamic output capture, checks new_tensor from lists
        # containing data-dependent counts (scalar, multi, mixed with shape, nested, with zero)
        # under fullgraph, run as subtests.
        a = torch.tensor([1, 0, 0, 5])

        cases = {
            "scalar": lambda a: a.new_tensor([a.nonzero().squeeze(-1).numel()]),
            "multi": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([n, n + 1, n * 2]),
            )[-1],
            "mixed_shape": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([n * a.shape[0], n + a.shape[0], a.shape[0] - n]),
            )[-1],
            "nested": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([[n, n + 1], [n * 2, n - 1]]),
            )[-1],
            "with_zero": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([0, n, n * n]),
            )[-1],
        }

        for name, fn in cases.items():
            with self.subTest(case=name):
                self.assertEqual(
                    torch.compile(fn, fullgraph=True, backend="eager")(a),
                    fn(a),
                )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_tensor(self):
        # [FUNCTION DOC] With scalar capture, checks bool on all-reduced tensors, scalar tensors
        # and error for multi-element tensors, compared with eager.
        def f_all(mask):
            # [FUNCTION DOC] HELPER: Returns whether all elements equal zero as bool.
            return bool((mask == 0).all())

        opt_f_all = torch.compile(f_all, backend="eager", fullgraph=True)

        mask_zeros = torch.zeros(2, 3)
        mask_nonzero = torch.tensor([[0, 1], [0, 0]])

        self.assertEqual(f_all(mask_zeros), opt_f_all(mask_zeros))
        self.assertEqual(f_all(mask_nonzero), opt_f_all(mask_nonzero))

        def f(x):
            # [FUNCTION DOC] HELPER: Returns bool of x.
            return bool(x)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)

        for val in [42, 0, 3.14, 0.0]:
            x = torch.tensor(val)
            self.assertEqual(f(x), opt_f(x))
        non_scalar = torch.tensor([1, 2, 3])
        with self.assertRaises(RuntimeError):
            f(non_scalar)
        with self.assertRaises(RuntimeError):
            opt_f(non_scalar)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symfloat(self):
        # [FUNCTION DOC] With scalar capture, checks bool of an item from a float tensor.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns bool of the item.
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(1)

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symbool(self):
        # [FUNCTION DOC] With scalar capture, checks bool of an item from a boolean tensor.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns bool of the item.
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(1) == 1

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)

    def test_builtin_complex(self):
        # [FUNCTION DOC] Checks the complex builtin with many argument forms and conjugate
        # under fullgraph.
        def f(x):
            # [FUNCTION DOC] HELPER: Builds eight complex numbers and returns x plus each.
            c = (
                complex(),
                complex(1),
                complex(2, 3),
                complex(imag=2),
                complex(real=1),
                complex(imag=1, real=2),
                complex("1+2j"),
                complex(1, 2).conjugate(),
            )
            return [x + z for z in c]

        x = torch.randn(1)
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)

    def test_builtin_complex_args(self):
        # [FUNCTION DOC] Checks the complex builtin with unsupported argument combinations
        # raises Unsupported.
        @torch.compile(backend="eager", fullgraph=True)
        def f(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Returns a tensor of the complex value built from arguments.
            return torch.tensor(complex(*args, **kwargs))

        self.assertRaises(Unsupported, f, 1, 1, 1)
        self.assertRaises(Unsupported, f, 1, 1, fake_arg=1)
        self.assertRaises(Unsupported, f, fake_arg=1)
        self.assertRaises(Unsupported, f, [])
        self.assertRaises(Unsupported, f, "1 + j")

    def test_builtin_class_method_constant_fold(self):
        # [FUNCTION DOC] Checks constant folding of class methods of bool, int and float (new,
        # from bytes, from hex, hex) under fullgraph.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Returns a tuple of eleven folded constants.
            return (
                bool.__new__(bool),
                bool.__new__(bool, 1),
                bool.__new__(bool, 0),
                bool.from_bytes(b"\x00" * 8, "big"),
                bool.from_bytes(b"abcd", "little"),
                int.__new__(int),
                int.__new__(int, 42),
                int.from_bytes(b"\x00\x03", "big"),
                int.from_bytes(b"\xff", byteorder="big", signed=True),
                float.fromhex("0x1.ffffp10"),
                float.hex(1.5),
            )

        res = fn()
        self.assertIs(res[0], False)
        self.assertIs(res[1], True)
        self.assertIs(res[2], False)
        self.assertIs(res[3], False)
        self.assertIs(res[4], True)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], 42)
        self.assertEqual(res[7], 3)
        self.assertEqual(res[8], -1)
        self.assertEqual(res[9], float.fromhex("0x1.ffffp10"))
        self.assertEqual(res[10], "0x1.8000000000000p+0")

    def test_builtin_constant_fold_str_conversions(self):
        # [FUNCTION DOC] Checks constant folding of hex, oct, bin, ascii and format results.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Concatenates the string conversions and adds their length
            # to x.
            s = hex(255) + oct(8) + bin(3) + ascii("hello") + format(42, "x")
            return x + len(s)

        x = torch.randn(4)
        res = fn(x)
        expected = hex(255) + oct(8) + bin(3) + ascii("hello") + format(42, "x")
        self.assertEqual(res, x + len(expected))

    def test_guard_string_escaped(self):
        # [FUNCTION DOC] Checks guards with frozenset keys in nested dicts are escaped properly.
        d = {frozenset({0}): {frozenset({0}): 1}}

        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Looks up nested frozenset keys and adds to x.
            return x + d[frozenset({0})][frozenset({0})]

        x = torch.ones(3)
        self.assertEqual(x + 1, f(x))

    def test_compiled_class_graph_break(self):
        # [FUNCTION DOC] Checks a class definition inside compiled function forces a graph break:
        # two frames.
        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=False)
        def f(x):
            # [FUNCTION DOC] HELPER: Adds in place, defines an empty class, returns sine.
            x += 1

            class C:
                # [CLASS DOC] HELPER: Empty class.
                pass

            return x.sin()

        x = torch.randn(3)
        f(x)
        self.assertEqual(counter.frame_count, 2)

    def test_debugmode(self):
        # [FUNCTION DOC] Checks custom op aliasing detection: inductor and aot_eager raise
        # RuntimeError when an op output aliases an input; repeated call on inductor does not
        # raise again.
        # Test that DebugMode works
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("alias_op(Tensor x) -> (Tensor, Tensor)")
            lib.impl(
                "alias_op",
                lambda x: (x.view_as(x), x.view_as(x)),
                "CompositeExplicitAutograd",
            )
            lib.impl("alias_op", lambda x: (x.view_as(x), x.view_as(x)), "Meta")

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the aliasing op and adds 1 to the first output.
                aliased, _ = torch.ops.mylib.alias_op(x)
                return aliased + 1

            x = torch.randn(10, 10)
            compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
            with torch._functorch.config.patch(
                check_custom_op_aliasing=True,
                error_on_custom_op_aliasing=True,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"The output of this custom operator \(1\) must not also be an input",
                ):
                    _ = compiled_fn(x)
                # Shouldn't error here because we already invoked once
                _ = compiled_fn(x)

                compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"The output of this custom operator \(1\) must not also be an input",
                ):
                    _ = compiled_fn(x)

    def test_debugmode_warns_outside_ci(self):
        # [FUNCTION DOC] Checks aliasing detection emits a warning with exact text, once, when
        # error mode is off.
        # Test that DebugMode emits warnings (not errors) when error_on_custom_op_aliasing=False
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("alias_op2(Tensor x) -> (Tensor, Tensor)")
            lib.impl(
                "alias_op2",
                lambda x: (x.view_as(x), x.view_as(x)),
                "CompositeExplicitAutograd",
            )
            lib.impl("alias_op2", lambda x: (x.view_as(x), x.view_as(x)), "Meta")

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the aliasing op and adds 1 to the first output.
                aliased, _ = torch.ops.mylib.alias_op2(x)
                return aliased + 1

            x = torch.randn(10, 10)
            compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
            # Use error_on_custom_op_aliasing=False to emit warnings instead of errors
            with (
                torch._functorch.config.patch(
                    check_custom_op_aliasing=True, error_on_custom_op_aliasing=False
                ),
                warnings.catch_warnings(record=True) as w,
            ):
                warnings.simplefilter("always")
                _ = compiled_fn(x)
                aliasing_warnings = [
                    x for x in w if "may not alias any inputs" in str(x.message)
                ]
                self.assertEqual(len(aliasing_warnings), 1)
                msg = str(aliasing_warnings[0].message)
                self.assertEqual(
                    msg,
                    "mylib::alias_op2 (with implementation in ???): "
                    "The output of this custom operator (1) must not also be an input "
                    "to this custom operator and (2) may not alias any inputs to this "
                    "custom operator or other returns. The most common way to trigger "
                    "this error is if we have y = custom_op(x) and y and x are the same "
                    "Tensor. Please instead return a clone of the offending output "
                    "tensor(s) (e.g. return x.clone()) or refactor the custom operator "
                    "to not return y. This is deprecated and will become an error in a future version of PyTorch.",
                )

    def test_debugmode_respects_custom_op_aliasing_env_override(self):
        # [FUNCTION DOC] Runs subprocesses to check the aliasing error flag defaults to true
        # under CI and can be overridden via environment variable; the second script compiles an
        # aliasing op and expects exactly one warning.
        default_script = """
from torch._functorch import config as functorch_config
assert functorch_config.error_on_custom_op_aliasing is True
"""
        default_env = os.environ.copy()
        default_env["CI"] = "1"
        default_env.pop("TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING", None)
        default_result = subprocess.run(
            [sys.executable, "-c", default_script],
            env=default_env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            default_result.returncode,
            0,
            msg=lambda msg: f"{msg}\nstdout:\n{default_result.stdout}\nstderr:\n{default_result.stderr}",
        )

        script = """
import warnings
import torch
from torch._functorch import config as functorch_config

assert functorch_config.error_on_custom_op_aliasing is False
with torch.library._scoped_library("mylib_ci", "FRAGMENT") as lib:
    lib.define("alias_op(Tensor x) -> Tensor")
    lib.impl("alias_op", lambda x: x.view_as(x), "CompositeExplicitAutograd")
    lib.impl("alias_op", lambda x: x.view_as(x), "Meta")

    def fn(x):
        return torch.ops.mylib_ci.alias_op(x) + 1

    compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled_fn(torch.randn(4))
    aliasing_warnings = [
        warning
        for warning in caught
        if "may not alias any inputs" in str(warning.message)
    ]
    assert len(aliasing_warnings) == 1, [
        str(warning.message) for warning in caught
    ]
"""
        env = os.environ.copy()
        env["CI"] = "1"
        env["TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING"] = "0"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=lambda msg: f"{msg}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_make_contiguous_strides_for_under_compile(self):
        # [FUNCTION DOC] Checks make_contiguous_strides_for with dynamic shapes is traceable.
        # is_nested_int and sym_max must be traceable under Dynamo.
        from torch._prims_common import make_contiguous_strides_for

        def fn(x):
            # [FUNCTION DOC] HELPER: Computes contiguous strides for the shape and returns
            # as_strided of x.
            strides = make_contiguous_strides_for(x.shape)
            return x.as_strided(x.shape, strides)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
        x = torch.randn(4, 8)
        result = compiled_fn(x)
        self.assertEqual(result, x)

        x2 = torch.randn(7, 8)
        result2 = compiled_fn(x2)
        self.assertEqual(result2, x2)

    def test_requires_grad_changes_dynamo_graph(self):
        # [FUNCTION DOC] Checks requires_grad_ on an input graph-breaks and gives the
        # branch result.
        # requires_grad_() on a graph input graph-breaks, so no fullgraph
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls requires_grad_ and returns double or plus 1.
            x.requires_grad_()
            if x.requires_grad:
                return x * 2
            return x + 1

        x = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager")
        result = opt_fn(x)
        self.assertEqual(result, x * 2)

    def test_requires_grad_backward_outside_compile(self):
        # [FUNCTION DOC] Checks requires_grad_ on an input then backward outside compile gives
        # the same gradient as eager.
        # requires_grad_() on a graph input graph-breaks, but eager fallback
        # produces correct results.
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls requires_grad_ and returns the sum of double x.
            x.requires_grad_()
            return (x * 2).sum()

        x_ref = torch.randn(3, 3)
        x_test = x_ref.clone()

        fn(x_ref).backward()
        torch.compile(fn, backend="eager")(x_test).backward()

        self.assertEqual(x_ref.grad, x_test.grad)

    def test_detach_inplace_on_intermediate_updates_metadata(self):
        # [FUNCTION DOC] Checks in-place detach on an intermediate updates requires_grad and
        # grad_fn metadata under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, detaches in place and returns y plus 1 with
            # metadata flags.
            y = x * 2
            y.detach_()
            return y + 1, y.requires_grad, y.grad_fn is None

        x = torch.randn(3, 3, requires_grad=True)
        ref = fn(x.clone())
        result = torch.compile(fn, backend="eager", fullgraph=True)(x.clone())

        self.assertEqual(ref, result)
        self.assertFalse(result[1])
        self.assertTrue(result[2])

    def test_requires_grad_on_intermediate(self):
        # [FUNCTION DOC] Checks requires_grad_ on an intermediate under fullgraph raises
        # Unsupported mentioning detach; without fullgraph, falls back to correct eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, calls requires_grad_ and returns it.
            y = x * 2
            y.requires_grad_()
            return y

        x = torch.randn(3, 3)

        # fullgraph=True should error with actionable message
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"requires_grad_\(\)(.|\n)*\.detach\(\)",
        ):
            torch.compile(fn, fullgraph=True)(x)  # noqa: UNSPECIFIED_BACKEND

        # Without fullgraph, falls back to eager and is correct
        result = torch.compile(fn, backend="eager")(x)
        self.assertTrue(result.requires_grad)
        self.assertEqual(fn(x), result)

    def test_requires_grad_on_intermediate_derived_returned(self):
        # [FUNCTION DOC] Same as above when a derived tensor is returned.
        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, calls requires_grad_ and returns thrice it.
            y = x * 2
            y.requires_grad_()
            return y * 3

        x = torch.randn(3, 3)

        # Derived tensor also loses requires_grad — should error with message
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"requires_grad_\(\)(.|\n)*\.detach\(\)",
        ):
            torch.compile(fn, fullgraph=True)(x)  # noqa: UNSPECIFIED_BACKEND

        # Without fullgraph, falls back to eager and is correct
        result = torch.compile(fn, backend="eager")(x)
        ref = fn(x)
        self.assertTrue(result.requires_grad)
        self.assertEqual(ref, result)

    def test_requires_grad_on_intermediate_partial_graph(self):
        # [FUNCTION DOC] Checks requires_grad_ on an intermediate leaking as output restarts with
        # a graph break at the call; ops before are captured in one graph with expected code
        # under static shapes.
        # When requires_grad_() on a source-less intermediate leaks as output,
        # Dynamo should restart and graph break at requires_grad_(), capturing
        # ops before it in a compiled graph (partial acceleration).
        def fn(x):
            # [FUNCTION DOC] HELPER: Sine, cosine, requires_grad_ and returns the result.
            a = x.sin()
            b = a.cos()
            b.requires_grad_()
            return b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        x = torch.randn(3, 3)
        result = torch.compile(fn, backend=backend)(x)
        self.assertEqual(result, fn(x))
        self.assertTrue(result.requires_grad)
        # The graph should capture the ops before requires_grad_()
        self.assertEqual(len(backend.graphs), 1)
        # Dynamic shapes adds shape guards to the graph, skip the exact check
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    sin = l_x_.sin();  l_x_ = None
    cos = sin.cos();  sin = None
    return (cos,)""",
            )

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_on_intermediate_not_returned(self):
        # [FUNCTION DOC] With autograd tracing, checks requires_grad_ on an intermediate then
        # backward and reading the grad under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, makes it require grad, runs backward through a
            # sum and returns y.grad.
            y = x * 2
            y.requires_grad_()
            loss = (y * 3).sum()
            loss.backward()
            return y.grad

        x = torch.randn(3, 3)

        ref = fn(x.clone())
        result = torch.compile(fn, fullgraph=True, backend="eager")(x.clone())
        self.assertEqual(ref, result)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_backward_grad_used_in_compute(self):
        # [FUNCTION DOC] With autograd tracing, checks the grad is used in further computation.
        # Use the grad result in further computation within compile
        def fn(x):
            # [FUNCTION DOC] HELPER: Backward through the sum of squares and returns grad times 2
            # plus 1.
            y = x * 2
            y.requires_grad_()
            loss = (y**2).sum()
            loss.backward()
            return y.grad * 2 + 1

        x = torch.randn(3, 3)

        ref = fn(x.clone())
        result = torch.compile(fn, fullgraph=True, backend="eager")(x.clone())
        self.assertEqual(ref, result)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_chunked_loss_backward(self):
        # [FUNCTION DOC] With autograd tracing, checks chunked loss pattern: detach into new
        # leaf, per-chunk cross entropy backward, then propagate back to the input.
        # Mirrors the TxtUnembedding pattern: forward compute, detach, make
        # new leaf, chunked loss with per-chunk backward, then propagate
        # accumulated grad back to the original input via h.backward().
        def fn(x, targets):
            # [FUNCTION DOC] HELPER: Forward compute, detach as leaf, loops chunks computing loss
            # and backward, accumulates loss, propagates grad and returns x.grad and total loss.
            # Forward computation before detach (e.g. transformer layers)
            h = x * 2 + 1
            x_detached = h.detach().requires_grad_()
            chunksz = x_detached.shape[0] // 2
            total_loss = torch.tensor(0.0)
            for start in range(0, x_detached.shape[0], chunksz):
                chunk = x_detached[start : start + chunksz]
                chunk_targets = targets[start : start + chunksz]
                logits = chunk @ torch.eye(chunk.shape[-1])
                loss = torch.nn.functional.cross_entropy(logits, chunk_targets)
                loss.backward()
                total_loss = total_loss + loss.detach()
            # Propagate chunked grad back through the forward computation
            h.backward(x_detached.grad)
            return x.grad, total_loss

        x_ref = torch.randn(4, 8, requires_grad=True)
        targets = torch.randint(0, 8, (4,))

        x_test = x_ref.clone().detach().requires_grad_(True)
        ref_grad, ref_loss = fn(x_ref, targets)
        compiled_grad, compiled_loss = torch.compile(
            fn, fullgraph=True, backend="eager"
        )(x_test, targets)
        self.assertEqual(ref_grad, compiled_grad)
        self.assertEqual(ref_loss, compiled_loss)
        # Verify grad propagated to the input
        self.assertEqual(x_ref.grad, x_test.grad)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_backward_and_return_detached(self):
        # [FUNCTION DOC] With autograd tracing, checks returning a detached output together with
        # the grad.
        # Returning a detached version of the tainted tensor is safe — detach()
        # strips requires_grad so AOTAutograd functionalization can't lose anything.
        def fn(x):
            # [FUNCTION DOC] HELPER: Backward through sum of thrice y, returns y.grad and
            # detached output.
            y = x * 2
            y.requires_grad_()
            out = y * 3
            loss = out.sum()
            loss.backward()
            return y.grad, out.detach()

        x = torch.randn(3, 3)

        ref_grad, ref_out = fn(x.clone())
        compiled_grad, compiled_out = torch.compile(
            fn, fullgraph=True, backend="eager"
        )(x.clone())
        self.assertEqual(ref_grad, compiled_grad)
        self.assertEqual(ref_out, compiled_out)
        self.assertFalse(compiled_out.requires_grad)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_metadata_checks(self):
        # [FUNCTION DOC] With autograd tracing, checks requires_grad and is_leaf metadata after
        # requires_grad_ drives control flow.
        # After requires_grad_() on an intermediate, requires_grad and is_leaf
        # should report correctly and be usable in control flow.
        def fn(x):
            # [FUNCTION DOC] HELPER: Branches on metadata and returns grad or y.
            y = x * 2
            y.requires_grad_()
            if y.requires_grad and y.is_leaf:
                loss = (y * 3).sum()
                loss.backward()
                return y.grad
            return y

        x = torch.randn(3, 3)
        ref = fn(x.clone())
        result = torch.compile(fn, fullgraph=True, backend="eager")(x.clone())
        self.assertEqual(ref, result)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_side_effect_global(self):
        # [FUNCTION DOC] With autograd tracing, checks storing the grad into a dict as
        # side effect.
        # requires_grad_() on intermediate, then store grad in a global
        saved = {}

        def fn(x):
            # [FUNCTION DOC] HELPER: Backward through squares, stores grad in the enclosing dict
            # and returns a clone.
            y = x * 2
            y.requires_grad_()
            loss = (y**2).sum()
            loss.backward()
            saved["grad"] = y.grad
            return y.grad.clone()

        x = torch.randn(3, 3)
        ref = fn(x.clone())
        saved_ref = saved["grad"].clone()
        saved.clear()

        result = torch.compile(fn, fullgraph=True, backend="eager")(x.clone())
        self.assertEqual(ref, result)
        self.assertEqual(saved_ref, saved["grad"])

    def test_import_user_defined_module(self):
        # [FUNCTION DOC] Regression test (issue 177682): importing a module registered in the
        # module table that is an instance of a module type subclass compiles. Registers a name
        # in the global module table.
        # testcase for https://github.com/pytorch/pytorch/issues/177682
        # Bad import result for types.ModuleType subclass in sys.modules
        class _ConfigModule(types.ModuleType):
            # [CLASS DOC] HELPER: Module subclass with class attribute x.
            x = 1

        _ConfigModule.__module__ = __name__
        sys.modules["my_config"] = _ConfigModule("my_config")

        def fn():
            # [FUNCTION DOC] HELPER: Imports the registered module and returns a tensor.
            import my_config  # noqa: F401

            return torch.tensor(1)

        compilefn = torch.compile(fn, fullgraph=True, backend="eager")

        ret1 = fn()
        ret2 = compilefn()
        self.assertEqual(ret1, ret2)

    def test_constant_subclass_guard_recompiles(self):
        # [FUNCTION DOC] Checks guards on subclasses of int, float and str compare by value
        # without calling their custom equality (which would raise): changing values recompiles,
        # repeating a value reuses the frame.
        class MyInt(int):
            # [CLASS DOC] HELPER: Int subclass whose equality raises RuntimeError.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always raises.
                raise RuntimeError("should not be called during guard check")

        class MyFloat(float):
            # [CLASS DOC] HELPER: Float subclass whose equality raises RuntimeError.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always raises.
                raise RuntimeError("should not be called during guard check")

        class MyStr(str):
            # [CLASS DOC] HELPER: Str subclass whose equality raises RuntimeError.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always raises.
                raise RuntimeError("should not be called during guard check")

        cnt = torch._dynamo.testing.CompileCounter()

        # int subclass
        @torch.compile(backend=cnt)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        r1 = f(torch.tensor(1), MyInt(5))
        self.assertEqual(r1.item(), 6)
        self.assertEqual(cnt.frame_count, 1)

        r2 = f(torch.tensor(1), MyInt(10))
        self.assertEqual(r2.item(), 11)
        self.assertEqual(cnt.frame_count, 2)

        r3 = f(torch.tensor(1), MyInt(5))
        self.assertEqual(r3.item(), 6)
        self.assertEqual(cnt.frame_count, 2)

        # float subclass
        cnt.clear()

        @torch.compile(backend=cnt)
        def g(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        r4 = g(torch.tensor(1.0), MyFloat(2.5))
        self.assertEqual(r4.item(), 3.5)
        self.assertEqual(cnt.frame_count, 1)

        r5 = g(torch.tensor(1.0), MyFloat(3.5))
        self.assertEqual(r5.item(), 4.5)
        self.assertEqual(cnt.frame_count, 2)

        r6 = g(torch.tensor(1.0), MyFloat(2.5))
        self.assertEqual(r6.item(), 3.5)
        self.assertEqual(cnt.frame_count, 2)

        # str subclass
        cnt.clear()

        @torch.compile(backend=cnt, fullgraph=True)
        def h(x, s):
            # [FUNCTION DOC] HELPER: Returns x plus length of s.
            return x + len(s)

        r7 = h(torch.tensor(1), MyStr("abc"))
        self.assertEqual(r7.item(), 4)
        self.assertEqual(cnt.frame_count, 1)

        r8 = h(torch.tensor(1), MyStr("abcde"))
        self.assertEqual(r8.item(), 6)
        self.assertEqual(cnt.frame_count, 2)

        r9 = h(torch.tensor(1), MyStr("abc"))
        self.assertEqual(r9.item(), 4)
        self.assertEqual(cnt.frame_count, 2)

    def test_sequence_iter_setstate_negative_clamps(self):
        # [FUNCTION DOC] Checks a sequence iterator state set to a negative number clamps to 0.
        # CPython iter_setstate clamps a negative index to 0.
        class Seq:
            # [CLASS DOC] HELPER: Sequence returning the index as the item.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Returns the index.
                return i

        def fn():
            # [FUNCTION DOC] HELPER: Sets iterator state to -42 and returns the next two items.
            it = iter(Seq())
            it.__setstate__(-42)
            return next(it), next(it)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt(), (0, 1))

    def test_sequence_iter_setstate_positive(self):
        # [FUNCTION DOC] Checks setting a sequence iterator state to 5 yields 5 and 6.
        class Seq:
            # [CLASS DOC] HELPER: Sequence returning the index as the item.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Returns the index.
                return i

        def fn():
            # [FUNCTION DOC] HELPER: Sets iterator state to 5 and returns the next two items.
            it = iter(Seq())
            it.__setstate__(5)
            return next(it), next(it)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt(), (5, 6))

    def test_sequence_iter_overflow(self):
        # [FUNCTION DOC] Checks a sequence iterator at the maximum index raises OverflowError on
        # the third next.
        # CPython iter_iternext raises OverflowError once the index reaches
        # sys.maxsize (PY_SSIZE_T_MAX) and cannot advance further.
        class Seq:
            # [CLASS DOC] HELPER: Sequence returning the index as the item.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Returns the index.
                return i

        def fn():
            # [FUNCTION DOC] HELPER: Sets state near the maximum, reads two items, catches
            # overflow and appends a marker.
            it = iter(Seq())
            it.__setstate__(sys.maxsize - 2)
            vals = [next(it), next(it)]
            try:
                next(it)
            except OverflowError:
                vals.append("overflow")
            return vals

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt(), [sys.maxsize - 2, sys.maxsize - 1, "overflow"])

    def test_callable_iter_reentrant_exhaustion(self):
        # [FUNCTION DOC] Regression for gh-101892: a callable iterator whose callable reentrantly
        # exhausts itself raises StopIteration on the outer next.
        # gh-101892: a two-argument iter() whose callable reentrantly exhausts
        # the iterator must raise StopIteration even when the outer call then
        # returns a non-sentinel value.
        HAS_MORE = 1
        NO_MORE = 2

        def fn():
            # [FUNCTION DOC] HELPER: Creates a state dict, a reentrant callable and iterator, and
            # returns stopped or no stop.
            state = {"recursive": False, "it": None}

            def spam():
                # [FUNCTION DOC] HELPER: First call exhausts the iterator via list and returns a
                # non-sentinel; reentrant call returns the sentinel.
                if state["recursive"]:
                    return NO_MORE
                state["recursive"] = True
                list(state["it"])
                return HAS_MORE

            state["it"] = iter(spam, NO_MORE)
            try:
                next(state["it"])
            except StopIteration:
                return "stopped"
            return "no stop"

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt(), "stopped")

    def test_re_module_constant_fold(self):
        # [FUNCTION DOC] Checks regex functions (search, match, no match, sub, findall, split,
        # compile and search, escape) fold to constants under fullgraph, with expected sums.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn_search(x):
            # [FUNCTION DOC] HELPER: Searches digits in a constant and adds the integer.
            m = re.search(r"(\d+)", "abc123def")
            if m:
                return x + int(m.group(1))
            return x

        @torch.compile(backend=cnt, fullgraph=True)
        def fn_match(x):
            # [FUNCTION DOC] HELPER: Matches a prefix and adds 1.
            m = re.match(r"hello", "hello world")
            if m:
                return x + 1
            return x

        @torch.compile(backend=cnt, fullgraph=True)
        def fn_no_match(x):
            # [FUNCTION DOC] HELPER: Searches a missing pattern and returns x unchanged.
            m = re.search(r"xyz", "hello world")
            if m:
                return x + 1
            return x

        @torch.compile(backend=cnt, fullgraph=True)
        def fn_sub(x):
            # [FUNCTION DOC] HELPER: Substitutes digits and adds 1 if text matches.
            s = re.sub(r"\d+", "0", "abc123def456")
            if s == "abc0def0":
                return x + 1
            return x

        @torch.compile(backend=cnt, fullgraph=True)
        def fn_findall(x):
            # [FUNCTION DOC] HELPER: Finds all digit runs and adds their count.
            matches = re.findall(r"\d+", "abc123def456")
            return x + len(matches)

        @torch.compile(backend=cnt, fullgraph=True)
        def fn_split(x):
            # [FUNCTION DOC] HELPER: Splits on whitespace and adds the part count.
            parts = re.split(r"\s+", "hello world foo")
            return x + len(parts)

        @torch.compile(backend=cnt, fullgraph=True)
        def fn_compile_and_search(x):
            # [FUNCTION DOC] HELPER: Compiles a pattern, searches, and adds the matched length.
            pattern = re.compile(r"(\w+)")
            m = pattern.search("hello world")
            if m:
                return x + len(m.group(0))
            return x

        @torch.compile(backend=cnt, fullgraph=True)
        def fn_escape(x):
            # [FUNCTION DOC] HELPER: Escapes a string and adds 1 if equal to the expected text.
            s = re.escape("hello.world")
            if s == r"hello\.world":
                return x + 1
            return x

        t = torch.tensor(10)

        self.assertEqual(fn_search(t), torch.tensor(133))
        self.assertEqual(fn_match(t), torch.tensor(11))
        self.assertEqual(fn_no_match(t), torch.tensor(10))
        self.assertEqual(fn_sub(t), torch.tensor(11))
        self.assertEqual(fn_findall(t), torch.tensor(12))
        self.assertEqual(fn_split(t), torch.tensor(13))
        self.assertEqual(fn_compile_and_search(t), torch.tensor(15))
        self.assertEqual(fn_escape(t), torch.tensor(11))

    @torch._dynamo.config.patch(recompile_limit=2)
    def test_compile_exec_function_no_cache_thrash(self):
        # [FUNCTION DOC] With a recompile limit of 2, checks compiling five functions made by
        # exec do not share a code object or thrash the cache (issue 124269).
        # Issue #124269: exec'd functions must not go through wrap_inline,
        # else they share one `inner` code object and hit recompile_limit.
        code = "def op(x):\n    return torch.mean(x)"
        x = torch.zeros(3)
        for _ in range(5):  # > recompile_limit
            scope: dict = {"torch": torch}
            exec(code, scope)
            compiled = torch.compile(scope["op"], fullgraph=True, backend="eager")
            self.assertEqual(compiled(x), torch.tensor(0.0))

    def test_str_join_with_list(self):
        # [FUNCTION DOC] Checks joining a list of strings under fullgraph.
        def fn():
            # [FUNCTION DOC] HELPER: Joins three strings with a comma.
            return ", ".join(["a", "b", "c"])

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt(), "a, b, c")

    def test_str_join_with_tuple(self):
        # [FUNCTION DOC] Checks joining a tuple of strings.
        def fn():
            # [FUNCTION DOC] HELPER: Joins three strings with a dash.
            return "-".join(("x", "y", "z"))

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt(), "x-y-z")

    def test_str_join_with_generator(self):
        # [FUNCTION DOC] Checks joining a generator of uppercase strings.
        def fn():
            # [FUNCTION DOC] HELPER: Joins uppercased letters with a comma.
            return ", ".join(s.upper() for s in ["a", "b", "c"])

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt(), "A, B, C")

    def test_str_join_with_map(self):
        # [FUNCTION DOC] Checks joining a map of string conversions.
        def fn():
            # [FUNCTION DOC] HELPER: Joins mapped integers with a colon.
            return ":".join(map(str, [1, 2, 3]))

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt(), "1:2:3")

    @unittest.skipIf(sys.version_info >= (3, 12), "comprehensions inlined in 3.12+")
    def test_listcomp_implicit_iterator_survives_graph_break(self):
        # [FUNCTION DOC] Skipped on Python 3.12+. Regression test: the implicit comprehension
        # iterator variable survives a graph break when resuming inside a comprehension.
        """Regression test: the implicit .0 iterator in a list comprehension
        must survive graph breaks under NGB.

        CPython names the comprehension's iterator ".0" in co_varnames.
        inspect.signature() renames it to "implicit0". This mismatch
        caused prune_dead_locals to drop the iterator and create_resume
        to omit it from its arguments. NGB is required to trigger this
        because without it, Dynamo un-inlines on graph break instead of
        building a resume function for the comprehension code object.
        """
        import random

        global _listcomp_helper

        def _listcomp_helper(x):
            # [FUNCTION DOC] HELPER: Module-level global function that graph breaks, adds, and
            # builds a list comprehension with random ints, returning x plus the sum.
            torch._dynamo.graph_break()
            x = x + 1
            x = x + 2
            result = [random.randint(1, 5) + s for s in [10, 20, 30]]
            return x + sum(result)

        opt = torch.compile(_listcomp_helper, backend="eager")
        x = torch.ones(3)
        result = opt(x)
        self.assertTrue(torch.all(result > x))

    def test_module_hooks_dict_reconstructed_as_ordered_dict(self):
        # [FUNCTION DOC] Checks that empty hook dicts on a module are rebuilt as ordered dicts so
        # that weak references to them work (spectral norm in compiled code).
        """Empty hooks dicts on nn.Module must be reconstructed as OrderedDict,
        not plain dict, so that weakref.ref() works in RemovableHandle.
        """

        def fn():
            # [FUNCTION DOC] HELPER: Creates a linear layer, applies spectral norm, returns
            # output for random input.
            m = torch.nn.Linear(5, 7)
            torch.nn.utils.spectral_norm(m)
            return m(torch.randn(3, 5))

        opt = torch.compile(fn, backend="eager")
        result = opt()
        self.assertEqual(result.shape, (3, 7))

    def test_module_hook_handle_references_real_dict(self):
        # [FUNCTION DOC] Checks a hook handle created inside compiled code references the
        # module's real hook dict: registered, referenced dict identity, and removal works.
        """A RemovableHandle created inside a compiled function must reference
        the real module's (empty) hooks dict. This requires reconstructing the
        empty hooks dict from its source; a sourceless fresh {} would make the
        handle weakref point at the wrong object so remove() silently no-ops.
        """

        def hook(mod, inp, out):
            # [FUNCTION DOC] HELPER: Hook returning the output unchanged.
            return out

        def fn(m, x):
            # [FUNCTION DOC] HELPER: Calls the module, registers the forward hook and returns
            # output and handle.
            y = m(x)
            h = m.register_forward_hook(hook)
            return y, h

        m = torch.nn.Linear(3, 3)
        opt = torch.compile(fn, backend="eager")
        x = torch.randn(2, 3)
        _, h = opt(m, x)
        self.assertEqual(len(m._forward_hooks), 1)
        self.assertIs(h.hooks_dict_ref(), m._forward_hooks)
        h.remove()
        self.assertEqual(len(m._forward_hooks), 0)

    def test_custom_op_register_fake_inside_traced_function(self):
        # [FUNCTION DOC] Checks a custom op defined with a registered fake inside a
        # traced function gives correct results (graph break to eager for the op,
        # surrounding ops compile).
        """Custom op defined with register_fake inside a traced function must
        produce correct results. register_fake mutates _abstract_fn on the real
        CustomOpDef as a side effect, so when the op is called before that
        side effect is applied, get_fake_value graph-breaks ("Custom op missing
        fake impl during tracing") and the op call falls back to eager. The
        surrounding tensor ops still compile (exercising the resume path).
        """
        from typing import Tuple

        def get_my_op():
            # [FUNCTION DOC] HELPER: Defines a custom op with fake registration and returns it.
            @torch.library.custom_op("test::ngb_id", mutates_args=[])
            def ngb_id(x: torch.Tensor) -> Tuple[torch.Tensor]:
                # [FUNCTION DOC] HELPER: Custom op returning a tuple holding a clone.
                return (x.clone(),)

            @ngb_id.register_fake
            def _(x: torch.Tensor) -> Tuple[torch.Tensor]:
                # [FUNCTION DOC] HELPER: Fake implementation returning a tuple holding a clone.
                return (x.clone(),)

            return ngb_id

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1, gets the op and calls it.
            x = x + 1  # non-empty prefix so a partial graph compiles
            my_op = get_my_op()
            return my_op(x)

        cnt = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt)
        x = torch.randn(3)
        result = opt(x)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], x + 1)
        # The op call graph-breaks to eager; the surrounding ops still compile.
        self.assertGreaterEqual(cnt.frame_count, 1)

    def test_custom_op_missing_fake_impl_hard_errors(self):
        # [FUNCTION DOC] Checks a custom op without any fake implementation raises RuntimeError
        # under compile.
        """A custom op genuinely missing its fake impl (register_fake never
        called) must still hard-error under compile. The missing-fake-impl
        graph break is scoped to the register_fake-side-effect-pending case, so
        it must NOT silently swallow this into an eager fallback.
        """

        @torch.library.custom_op("test::ngb_missing_fake", mutates_args=[])
        def ngb_missing_fake(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op returning a clone.
            return x.clone()

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the op and adds 1.
            return ngb_missing_fake(x) + 1

        with self.assertRaises(RuntimeError):
            fn(torch.randn(3))

    def test_custom_op_defined_inside_without_fake_hard_errors(self):
        # [FUNCTION DOC] Checks a custom op defined inside the traced function without fake
        # still raises RuntimeError even with an unrelated pending attribute mutation on the
        # op definition.
        """A custom op DEFINED INSIDE the traced function but never
        register_fake'd must still hard-error, and the gate must key off a
        pending `_abstract_fn` mutation SPECIFICALLY, not any pending mutation.

        Unlike test_custom_op_missing_fake_impl_hard_errors (op at module
        scope, whose opdef is untracked so the gate short-circuits at
        `vt is None`), the opdef here is tracked in side_effects. We store an
        unrelated attribute on it so it has a pending NON-`_abstract_fn`
        mutation: a gate that checked `has_pending_mutation(vt)` instead of
        `has_pending_mutation_of_attr(vt, "_abstract_fn")` would then wrongly
        graph-break and silently fall the genuinely-broken op back to eager.
        The specific check hard-errors as it must.
        """

        def get_op():
            # [FUNCTION DOC] HELPER: Defines and returns a custom op.
            @torch.library.custom_op("test::ngb_inside_nofake", mutates_args=[])
            def op(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op returning a clone.
                return x.clone()

            return op

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1, defines op, sets an unrelated attribute on it and
            # calls it.
            x = x + 1
            my_op = get_op()
            # pending non-_abstract_fn mutation on the tracked opdef
            my_op._unrelated_marker = 1
            return my_op(x)

        with self.assertRaises(RuntimeError):
            fn(torch.randn(3))


instantiate_parametrized_tests(MiscTests)


class MiscTestsPyTree(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Test case holding pytree tracing tests, each parametrized over the python, cxx
    # and optree pytree modules when available. No state beyond the parametrization.
    @parametrize_pytree_module
    def test_tracing_pytree(self, pytree):
        # [FUNCTION DOC] Checks flatten, clone of leaves and unflatten of a list under fullgraph;
        # one frame, 3 ops. The optree module takes the spec first when unflattening.
        def fn(xs):
            # [FUNCTION DOC] HELPER: Flattens the input, clones each leaf and unflattens
            # (argument order depends on the module).
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]

        counter = CompileCounter()
        torch.compile(fn, backend=counter, fullgraph=True)(xs)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

    @parametrize_pytree_module
    def test_tracing_nested_pytree(self, pytree):
        # [FUNCTION DOC] Same with a list of four lists; one frame, 12 ops.
        def fn(xs):
            # [FUNCTION DOC] HELPER: Flattens, clones leaves, unflattens.
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = [xs, xs, xs, xs]

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 12)

    @parametrize_pytree_module
    def test_tracing_nested_tuples(self, pytree):
        # [FUNCTION DOC] Same with a tuple of four lists; one frame, 12 ops.
        def fn(xs):
            # [FUNCTION DOC] HELPER: Flattens, clones leaves, unflattens.
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = (xs, xs, xs, xs)

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 12)

    @parametrize_pytree_module
    def test_tracing_nested_dicts(self, pytree):
        # [FUNCTION DOC] Same with a dict of three lists; one frame, 9 ops.
        def fn(xs):
            # [FUNCTION DOC] HELPER: Flattens, clones leaves, unflattens.
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = {
            "a": xs,
            "b": xs,
            "c": xs,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)

    @parametrize_pytree_module
    def test_tracing_nested_mixed_all(self, pytree):
        # [FUNCTION DOC] Same with a mixed nest of dicts, tuples and lists; one frame, 18 ops.
        def fn(xs):
            # [FUNCTION DOC] HELPER: Flattens, clones leaves, unflattens.
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}
        xsl = {
            "a": xs,
            "b": xsa,
            "c": xsb,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 18)

    @parametrize_pytree_module
    def test_tracing_nested_tensor_subclass(self, pytree):
        # [FUNCTION DOC] Checks flattening a tensor subclass (two-tensor) reaching the function
        # via a checkpoint (sourceless subclass); one frame, 2 ops.
        from torch.testing._internal.two_tensor import TwoTensor
        from torch.utils.checkpoint import checkpoint

        def fn(xs):
            # [FUNCTION DOC] HELPER: Flattens x and returns a clone of the first leaf.
            nested_xs = [[xs]]
            flat_xs, spec = pytree.tree_flatten(xs)
            return flat_xs[0].clone()

        # use checkpoint to trigger a "sourceless" tensor subclass
        def checkpoint_fn(xs):
            # [FUNCTION DOC] HELPER: Runs fn under reentrant checkpoint.
            return checkpoint(fn, xs, use_reentrant=True)

        xs = TwoTensor(torch.ones(2, 2), torch.ones(2, 2))

        counter = CompileCounter()
        torch.compile(checkpoint_fn, backend=counter, fullgraph=True)(xs)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)

    @parametrize_pytree_module
    def test_pytree_tree_leaves(self, pytree):
        # [FUNCTION DOC] Checks tree leaves of a nest including dict, list, tuple, deque with
        # maxlen, ordered dict, structseq and namedtuple matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the nested tree and returns its leaves.
            tree = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.linalg_qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            leaves = pytree.tree_leaves(tree)
            return leaves

        x = torch.randn(3, 2)
        expected = fn(x)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_flatten_unflatten(self, pytree):
        # [FUNCTION DOC] Checks flatten and unflatten of a complex nest with replaced leaves; for
        # optree one leaf is dropped since None is an internal node.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Flattens the tree, builds a new leaf list, unflattens,
            # returns leaves and the new tree.
            tree = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.linalg_qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            leaves, treespec = pytree.tree_flatten(tree)
            new_leaves = [
                x - 1,
                y,
                x * y,
                3.0,
                y - 2,
                1,
                torch.zeros(2, 2),
                2 * y,
                -y,
                x + y,
                x - y,
                torch.ones(3, 2),
                1,
            ]
            if pytree.__name__ == "optree":
                # `None` is a internal node rather than leaf in default OpTree / JAX PyTree
                new_leaves.pop()
                # The treespec argument comes first in OpTree / JAX PyTree
                new_tree = pytree.tree_unflatten(treespec, new_leaves)
            else:
                new_tree = pytree.tree_unflatten(new_leaves, treespec)
            return leaves, new_tree

        x = torch.randn(3, 2)
        y = torch.randn(3, 2)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_map(self, pytree):
        # [FUNCTION DOC] Checks tree_map over two trees with different dict orders matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds two trees and maps a pairing function over them.
            tree1 = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.linalg_qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            tree2 = collections.OrderedDict(
                [
                    ("c", (y, 3.0, collections.deque([1, -y, 10.0]))),
                    ("a", [y, y + 1]),
                    ("b", y + 2),
                    (
                        "d",
                        {
                            "f": MyTuple(torch.ones(4, 3), -y, y + 1),
                            "e": torch.return_types.linalg_qr((2 * y, None)),
                        },
                    ),
                ],
            )
            return pytree.tree_map(lambda u, v: (u, v), tree1, tree2)

        x = torch.randn(3, 2)
        y = torch.randn(3, 2)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_map_dict_order(self, pytree):
        # [FUNCTION DOC] Checks tree_map preserves key order for dict, ordered dict
        # and defaultdict.
        def fn(tree):
            # [FUNCTION DOC] HELPER: Maps identity over the tree and returns key and value lists.
            new_tree = pytree.tree_map(lambda x: x, tree)
            return list(new_tree.keys()), list(new_tree.values())

        x = torch.randn(3, 2)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)

        tree1 = {"b": x + 2, "a": x, "c": x - 1}
        expected1 = fn(tree1)
        actual1 = fn_opt(tree1)
        self.assertEqual(actual1, expected1)

        tree2 = collections.OrderedDict([("b", x + 2), ("a", x), ("c", x - 1)])
        expected2 = fn(tree2)
        actual2 = fn_opt(tree2)
        self.assertEqual(actual2, expected2)

        tree3 = collections.defaultdict(int, {"b": x + 2, "a": x, "c": x - 1})
        expected3 = fn(tree3)
        actual3 = fn_opt(tree3)
        self.assertEqual(actual3, expected3)

    @parametrize_pytree_module
    def test_pytree_tree_map_only(self, pytree):
        # [FUNCTION DOC] Checks tree_map_only on tensors in a nest containing a string; skipped
        # for modules without that function; one frame, 9 ops.
        if not callable(getattr(pytree, "tree_map_only", None)):
            # OpTree and JAX PyTree do not have `tree_map_only`
            return

        def fn(xs):
            # [FUNCTION DOC] HELPER: Maps a clone over tensor leaves only.
            def mapper(x):
                # [FUNCTION DOC] HELPER: Returns a clone of x.
                return x.clone()

            y = pytree.tree_map_only(torch.Tensor, mapper, xs)
            return y

        xs = [torch.tensor(i) for i in range(3)] + ["hi"]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsb)
        real_out = fn(xsb)

        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)

    def test_pytree_register_constant_with_side_effect(self):
        # [FUNCTION DOC] Checks registering a class as a pytree constant (hashable) and then
        # assigning an instance into a dict attribute on another object inside compiled code.
        class Foo:
            # [CLASS DOC] HELPER: Empty class.
            pass

        class Bar:
            # [CLASS DOC] HELPER: Class with custom equality via base and constant hash.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Defers to the base equality.
                return super().__eq__(other)

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns 0.
                return 0

        python_pytree.register_constant(Bar)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Sets a dict holding a Bar instance on obj and returns x
            # plus 1.
            obj.attr = {3: Bar()}
            return x + 1

        inp = torch.ones(3)
        self.assertEqual(fn(inp, Foo()), inp + 1)


class TestTracer(JitTestCase):
    # [CLASS DOC] Test case for a tracing interaction with compile.
    def test_jit_save(self):
        # [FUNCTION DOC] Checks tracing a scripted module with exported state methods inside a
        # function works eagerly and compiled.
        def fn():
            # [FUNCTION DOC] HELPER: Defines a module with exported state methods and traces it.
            class Foo(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with int attribute a and exported state hooks.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Sets a to 3.
                    super().__init__()
                    self.a = 3

                @torch.jit.export
                def __getstate__(self):
                    # [FUNCTION DOC] HELPER: Returns a tuple of 3 and the training flag.
                    return (3, self.training)

                @torch.jit.export
                def __setstate__(self, state):
                    # [FUNCTION DOC] HELPER: Restores a and training flag from the state tuple.
                    self.a = state[0]
                    self.training = state[1]

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns x plus a.
                    return x + self.a

            f = Foo()

            return torch.jit.trace(f, (torch.rand(3, 4),))

        fn()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()


class TestCustomFunction(torch.testing._internal.common_utils.TestCase):
    # [CLASS DOC] Test case for custom autograd function behaviors with compile.
    def test_autograd_function_with_matmul_folding_at_output(self):
        # [FUNCTION DOC] Checks a custom function returning a matmul that would fold into a view
        # can be modified in place and backpropagated; compares gradients with the plain version.
        # KNOWN ISSUE: the final gradient comparison is not asserted.
        """
        When tensor folding occurs during matmul operation returned tensor is a view.
        This can cause issues when matmul is used inside a custom function
        and such view is then returned as output. Then it cannot be modified inplace
        and causes errors.
        It can be especially problematic when after such function inplace allreduce
        is performed. This test recreates this behaviour.
        Issue is resolved when unsafe_view is returned from matmul instead.
        """

        class CustomFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function with matmul forward and backward saving the
            # second input and output shape.
            @staticmethod
            def forward(ctx, inp1, inp2):
                # [FUNCTION DOC] HELPER: Saves inp2 and the shape, returns matmul.
                ctx.save_for_backward(inp2)
                ctx.output_shape = inp1.size()
                return torch.matmul(inp1, inp2)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Computes the first input gradient by matmul with
                # transposed saved input, viewed to the saved shape, plus None.
                output_shape = ctx.output_shape
                (inp2,) = ctx.saved_tensors
                return (
                    torch.mm(grad_output.squeeze(), inp2.t()).view(output_shape),
                    None,
                )

        def outer_function(inp1, inp2):
            # [FUNCTION DOC] HELPER: Applies the custom function, adds 1 in place and sums.
            res = CustomFunction.apply(inp1, inp2)
            res.add_(1.0)
            return res.sum()

        def usual_function(inp1, inp2) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Plain matmul version of the same: add 1 in place and sum.
            res = torch.matmul(inp1, inp2)
            res.add_(1.0)
            return res.sum()

        inp1_custom = torch.randn(4, 1, 2, requires_grad=True)
        inp1_usual = inp1_custom.detach().clone().requires_grad_(True)

        inp2 = torch.randn(2, 4)
        c_custom_func = torch.compile(outer_function, backend="eager")
        c_usual_func = torch.compile(usual_function, backend="eager")

        result_custom = c_custom_func(inp1_custom, inp2)
        result_custom.backward()
        result_usual = c_usual_func(inp1_usual, inp2)
        result_usual.backward()

        torch.allclose(inp1_custom.grad, inp1_usual.grad)

    def test_retain_grad(self):
        # [FUNCTION DOC] Checks retain_grad inside a compiled function with aot_eager keeps a
        # gradient on the intermediate.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Retains grad on y and returns sine of y plus x.
            y.retain_grad()
            return torch.sin(y) + x

        opt_fn = torch.compile(fn, backend="aot_eager")
        x = torch.randn(4, requires_grad=True)
        y = torch.cos(x)
        opt_fn(x, y).sum().backward()
        self.assertTrue(y.grad is not None)


class MiscTestsDevice(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Device-generic test case instantiated per device (CPU excluded) for dynamo
    # tests that need a device argument.
    def test_rand(self, device):
        # [FUNCTION DOC] Checks random generation under a fixed seed matches between eager
        # and compiled.
        cnts = torch._dynamo.testing.CompileCounter()
        device = device

        def fn():
            # [FUNCTION DOC] HELPER: Returns a random normal tensor on the device.
            return torch.randn(10, device=device)

        torch.manual_seed(10)
        ref_run1 = fn()

        torch.manual_seed(10)
        ref_run2 = fn()
        self.assertTrue(same(ref_run1, ref_run2))

        torch.manual_seed(10)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn()

        self.assertTrue(same(res, ref_run1))

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_parsing_sdpa(self, device):
        # [FUNCTION DOC] Skipped without flash attention support. Checks multiple call forms
        # of scaled dot product attention (positional, keyword, mixed, with scale) compile
        # with inductor.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling attention in six argument styles.
            def forward(self, query, key, value):
                # [FUNCTION DOC] HELPER: Calls scaled dot product attention six ways and returns
                # the last.
                out = F.scaled_dot_product_attention(query, key, value, None, 0, True)
                out = F.scaled_dot_product_attention(
                    query, key, value, None, 0, True, scale=8
                )
                out = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                out = F.scaled_dot_product_attention(
                    query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                out = F.scaled_dot_product_attention(
                    query, key, value, None, dropout_p=0, is_causal=True
                )
                out = F.scaled_dot_product_attention(query, key, value, None, scale=8)
                return out

        device = device
        dtype = torch.float16
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        query = torch.ones(
            1, 8, seq_len_q, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        key = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        value = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        module = MyModule()
        opt_mod = torch.compile(module, backend="inductor")
        opt_mod(query, key, value)

    def test_torch_device_is_available(self, device):
        # [FUNCTION DOC] Checks accelerator availability check is evaluated at trace time.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if an accelerator is available, else x
            # minus 1.
            if torch.accelerator.is_available():
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @onlyCUDA
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable(self, device):
        # [FUNCTION DOC] CUDA with cudnn only. Checks the cudnn acceptability query on a tensor
        # under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if cudnn accepts the tensor, else x.
            if torch.backends.cudnn.is_acceptable(tensor=x):
                return x + 1
            return x

        x = torch.rand(4).to(device)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @onlyCUDA
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable_bad_inputs(self, device):
        # [FUNCTION DOC] CUDA with cudnn only. Checks invalid arguments to the cudnn
        # acceptability query raise AssertionError with the expected messages (non-tensor input,
        # extra argument).
        def fn1(x):
            # [FUNCTION DOC] HELPER: Calls the query with a string.
            if torch.backends.cudnn.is_acceptable("invalid"):
                return x + 1
            return x

        def fn2(x):
            # [FUNCTION DOC] HELPER: Calls the query with a tensor and an extra float.
            if torch.backends.cudnn.is_acceptable(x, 3.14):
                return x + 1
            return x

        with self.assertRaisesRegex(
            AssertionError, "Expect input to cudnn.is_acceptable to be a tensor"
        ):
            x1 = torch.rand(4).to(device)
            opt_fn1 = torch.compile(fn1, backend="eager", fullgraph=True)
            res1 = opt_fn1(x1)

        with self.assertRaisesRegex(
            AssertionError, "Expect 1 input to cudnn.is_acceptable"
        ):
            x2 = torch.rand(4).to(device)
            opt_fn2 = torch.compile(fn2, backend="eager", fullgraph=True)
            res = opt_fn2(x2)

    @onlyCUDA
    @torch._dynamo.config.patch(recompile_limit=999)
    def test_legacy_cuda_tensor(self, device):
        # [FUNCTION DOC] CUDA only with large recompile limit. Checks legacy CUDA tensor type
        # constructors with a list under fullgraph for nine types.
        typs = [
            torch.cuda.FloatTensor,
            torch.cuda.DoubleTensor,
            torch.cuda.HalfTensor,
            torch.cuda.BFloat16Tensor,
            torch.cuda.ByteTensor,
            torch.cuda.CharTensor,
            torch.cuda.IntTensor,
            torch.cuda.ShortTensor,
            torch.cuda.LongTensor,
        ]

        def f2(typ):
            # [FUNCTION DOC] HELPER: Builds a tensor of the given type from a list.
            return typ([1, 2, 3])

        compiled_f2 = torch.compile(f2, backend="eager", fullgraph=True)
        for typ in typs:
            output = compiled_f2(typ)
            expected = f2(typ)
            self.assertEqual(output, expected)

    def test_get_device(self, device):
        # [FUNCTION DOC] Checks get_device on tensors on the test device and on CPU.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds 1 to both and returns their device ids.
            x = x + 1
            y = y + 1
            return x.get_device(), y.get_device()

        x = torch.rand(4, device=device)
        y = torch.rand(4, device="cpu")
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_symint_as_device_kwarg(self, device):
        # [FUNCTION DOC] Checks using a symbolic size as a device index with dynamic shapes.
        def f(rank):
            # [FUNCTION DOC] HELPER: Creates ones on a device equal to the input size minus 2.
            # -2 to make device id 0 for easier testing on CI
            return torch.ones(10, device=rank.size(0) - 2)

        x = torch.randn(2)
        out = f(torch.randn(2))
        opt_out = torch.compile(backend="eager", dynamic=True, fullgraph=True)(f)(x)
        self.assertEqual(out, opt_out)

    def test_torch_device_python_type(self, device):
        # [FUNCTION DOC] Checks the device attribute python type and its type and index, for cpu
        # and the test device, with a trace-time assertion.
        device_type = torch.device(device).type
        for device, device_type, index in [
            ("cpu", "cpu", None),
            (device, device_type, 0),
        ]:

            def fn(target):
                # [FUNCTION DOC] HELPER: Creates three zero tensors on the target device and sums
                # them, asserting device type and index at trace time.
                target_device = target.device
                a = torch.zeros(2, 3, device=target_device)
                # Constant assert at trace time
                assert isinstance(target_device, torch.device)  # noqa: S101
                assert target_device.type == device_type  # noqa: S101
                assert target_device.index == index  # noqa: S101
                b = torch.zeros(2, 3, device=target_device)
                c = torch.zeros(2, 3, device=target_device)
                return a + b + c

            from torch._dynamo.variables import ConstantVariable

            device = torch.device(device)
            expected_variable = ConstantVariable(device)
            self.assertEqual(expected_variable.python_type(), type(device))

            opt_func = torch.compile(fn, backend="eager", fullgraph=True)
            a = torch.tensor([2, 3], device=device)
            res = opt_func(a)
            self.assertIsInstance(res, torch.Tensor)

    # Known CUDA memory leak: under propagate_real_tensors, a data-dependent
    # .tolist() retains the real input tensor (via FakeTensor.real_tensor held by
    # a TrackedFake) past torch._dynamo.reset(). See #190093.
    @skipCUDAMemoryLeakCheckIf(True)
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True)
    def test_interpolate_propagate_real_tensors(self, device):
        # [FUNCTION DOC] Skips the leak check on CUDA; with scalar capture and real tensor
        # propagation, checks interpolate to a data-dependent size compiles.
        @torch.compile(backend="eager", fullgraph=True)
        def f(mask, box):
            # [FUNCTION DOC] HELPER: Creates a random mask, reads the box dims and interpolates
            # to that size.
            # u0, u1 = mask.tolist()
            mask = torch.randn(1, 1, 30, 30, device=device)
            h, w = box.tolist()
            return torch.nn.functional.interpolate(
                mask, (h, w), mode="bilinear", align_corners=False
            )

        f(torch.tensor([30, 30], device=device), torch.tensor([68, 32], device=device))

    def test_scalar_isin_decomposition(self):
        # [FUNCTION DOC] Checks isin on scalar tensors compiles with inductor and matches eager.
        def f():
            # [FUNCTION DOC] HELPER: Returns isin of a scalar tensor with itself.
            x = torch.tensor(0)
            return torch.isin(x, x)

        opt_f = torch.compile(f, backend="inductor", fullgraph=True)
        ref = f()
        res = opt_f()
        self.assertEqual(ref, res)

    def test_randint_no_graphbreak(self):
        # [FUNCTION DOC] Checks randint_like with a tensor upper bound compiles without a
        # graph break.
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(actions, n_act, epsilon=0.1):
            # [FUNCTION DOC] HELPER: Returns random ints like actions bounded by n_act.
            actions_random = torch.randint_like(actions, n_act)

            return actions_random

        x = torch.ones([1], dtype=torch.int64)
        y = torch.tensor(5)
        f(x, y)

    def test_full_graph_capture_scalar_outputs(self):
        # [FUNCTION DOC] Checks an item call under fullgraph no longer raises.
        @torch.compile(fullgraph=True, backend="eager")
        def foo(a):
            # [FUNCTION DOC] HELPER: Multiplies random tensor by the item of a.
            return torch.randn(5) * a.item()

        # We expect to no longer raise here
        foo(torch.tensor(2.0))

    def test_full_graph_capture_dynamic_output_shape_ops(self):
        # [FUNCTION DOC] Checks a function with nonzero and slicing compiles under fullgraph
        # without raising.
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares nonzero indices, slices the last two columns, and
            # unsqueezes and squeezes.
            nz = torch.nonzero(x)
            squared = nz * nz
            sliced = torch.ops.aten.slice.Tensor(squared, dim=1, start=-2, end=None)
            view = sliced.unsqueeze(dim=0)
            return view.squeeze(dim=0)

        example_inputs = (torch.randn(1, 1, 1, 1),)
        # we expect to no longer raise here
        torch.compile(fn, fullgraph=True, backend="eager")(*example_inputs)

    def test_dynamic_fill_diagonal_(self):
        # [FUNCTION DOC] Checks fill_diagonal_ under dynamic shapes compiles.
        @torch.compile(dynamic=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Fills the diagonal with True.
            x.fill_diagonal_(True)

        x = torch.zeros(4, 4)
        f(x)

    def test_dynamic_float_scalar_tensor_coersion(self):
        # [FUNCTION DOC] Checks a large float config value coerced to a tensor dtype does not
        # overflow when the value changes (formerly crashed).
        # Minified version of https://github.com/pytorch/pytorch/issues/158376#issuecomment-3079591367
        class Foo:
            # [CLASS DOC] HELPER: Holder with a config class having a pad value and tolerance.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the config type.
                self.config = type(
                    "Config", (), {"pad_val": 1123581321.0, "tolerance": 1e-6}
                )

            @torch.compile(fullgraph=True, backend="eager")
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Compiled method using where on closeness to the pad
                # value with tensors from config values.
                outputs = torch.where(
                    torch.abs(input - self.config.pad_val) < self.config.tolerance,
                    torch.tensor(
                        self.config.pad_val, dtype=input.dtype, device=input.device
                    ),
                    torch.tensor(
                        self.config.pad_val + 1, dtype=input.dtype, device=input.device
                    ),
                )
                return outputs

        foo = Foo()
        inputs = torch.randn(3, 4)
        result = foo.forward(inputs)

        original_pad_val = foo.config.pad_val
        foo.config.pad_val += 1.0
        result2 = foo.forward(inputs)

        # Previously would crash with:
        #   RuntimeError: value cannot be converted to type at::Half without overflow

    def test_symbool_tensor_mul(self, device):
        # [FUNCTION DOC] Checks multiplying a boolean tensor by a scalar tensor with dynamic
        # shapes gives 2.0 for true and 0.0 for false.
        def symbool_mul_fn(x_bool, sentinel):
            # [FUNCTION DOC] HELPER: Returns x_bool times sentinel.
            result = x_bool * sentinel
            return result

        x_true = torch.tensor([True], device=device)
        x_false = torch.tensor([False], device=device)
        sentinel = torch.tensor(2.0, requires_grad=True, device=device)
        eager_result_true = symbool_mul_fn(x_true, sentinel)
        eager_result_false = symbool_mul_fn(x_false, sentinel)
        compiled_fn = torch.compile(
            symbool_mul_fn, fullgraph=True, dynamic=True, backend="eager"
        )
        compiled_result_true = compiled_fn(x_true, sentinel)
        compiled_result_false = compiled_fn(x_false, sentinel)
        self.assertEqual(eager_result_true, compiled_result_true)
        self.assertEqual(eager_result_false, compiled_result_false)
        self.assertEqual(compiled_result_true.item(), 2.0)
        self.assertEqual(compiled_result_false.item(), 0.0)

    def test_symbool_guard_or_false(self, device):
        # [FUNCTION DOC] Checks guard_or_false on a symbolic boolean from item with dynamic
        # shapes returns 10 times b or 100 times b.
        def symbool_guard_fn(a_bool_tensor, b):
            # [FUNCTION DOC] HELPER: Reads the item, branches on guard_or_false and scales b.
            u0 = a_bool_tensor.item()
            # Make sure guard_or_false still handles SymBool produced by .item()
            if guard_or_false(u0):
                return b * 10
            else:
                return b * 100

        compiled_guard_fn = torch.compile(
            symbool_guard_fn, backend="eager", dynamic=True
        )
        a_true = torch.tensor(True, device=device)
        a_false = torch.tensor(False, device=device)
        b = torch.randn(6, device=device)
        eager_res_true = symbool_guard_fn(a_true, b)
        compiled_res_true = compiled_guard_fn(a_true, b)
        self.assertEqual(eager_res_true, compiled_res_true)
        eager_res_false = symbool_guard_fn(a_false, b)
        compiled_res_false = compiled_guard_fn(a_false, b)
        self.assertEqual(eager_res_false, compiled_res_false)
        self.assertEqual(compiled_res_true, b * 10)
        self.assertEqual(compiled_res_false, b * 100)

    def test_symbool_tensor_mul_does_not_fail(self, device):
        # [FUNCTION DOC] Checks a fuzzer-found program (item of squeezed bool times sentinel)
        # compiles without error.
        def fuzzed_program(arg_0, sentinel):
            # [FUNCTION DOC] HELPER: Squeezes, reads the item, multiplies by sentinel, takes real
            # part if complex.
            var_node_2 = arg_0
            var_node_1 = torch.squeeze(var_node_2)
            var_node_0 = var_node_1.item()
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        sentinel = torch.tensor(1.0, requires_grad=True, device=device)
        arg_0 = torch.tensor([True], dtype=torch.bool, device=device)
        args = (arg_0,) + (sentinel,)
        try:
            compiled_program = torch.compile(
                fuzzed_program, fullgraph=True, dynamic=True, backend="eager"
            )
            compiled_program(*args)
        except Exception as e:
            self.fail(f"torch.compile failed with error: {e}")

    def test_module_to_with_shared_weights_compile(self, device):
        # [FUNCTION DOC] Checks moving a module with shared embedding weights to the input device
        # inside compiled code is correct for repeated calls.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an embedding that draws random ids and sums.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the embedding.
                super().__init__()
                self.embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Draws token ids on the input device and returns the sum
                # of x plus the embedding sum.
                token_ids = torch.randint(0, 10, (4,), device=x.device)
                embedded = self.embedding(token_ids).sum()
                return x.sum() + embedded.sum()

        class Container(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper module that moves its inner module to the input device
            # when not on CPU.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the inner model.
                super().__init__()
                self.mod = Model()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Moves the model to the device and calls it, else returns
                # the sum.
                if x.device.type != "cpu":
                    mod = self.mod.to(x.device)
                    return mod(x)
                else:
                    return x.sum()

        container = Container()
        container_eager = copy.deepcopy(container)
        with torch._dynamo.config.patch(graph_break_on_nn_param_ctor=False):
            compiled = torch.compile(container, backend="eager", fullgraph=True)

            inp1 = torch.randn(4, 4, 4, device=device)

            compiled_result1 = compiled(inp1)
            eager_result1 = container_eager(inp1)
            same(compiled_result1, eager_result1)

            compiled_result2 = compiled(inp1)
            eager_result2 = container_eager(inp1)
            same(compiled_result2, eager_result2)

    def test_module_to_move_compile(self, device):
        # [FUNCTION DOC] Checks moving a module to CPU inside compiled forward takes effect on
        # every call.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer that moves itself to CPU in forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.fc = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear, moves module to CPU and returns
                # the output.
                x = self.fc(x)
                self.to("cpu")
                return x

        mod = Model().to(device)
        with torch._dynamo.config.patch(graph_break_on_nn_param_ctor=False):
            fn = torch.compile(mod, backend="aot_eager", fullgraph=True)
            x = torch.randn(10, 10, device=device)
            ref = fn(x)
            self.assertEqual(str(mod.fc.weight.device), "cpu")
            mod.to(device)
            ref = fn(
                x
            )  # second time compile runs, we should also move the module to cpu device
            self.assertEqual(str(mod.fc.weight.device), "cpu")


instantiate_parametrized_tests(MiscTestsPyTree)

instantiate_device_type_tests(
    MiscTestsDevice, globals(), except_for=("cpu",), allow_xpu=True
)


class DynamoOpPromotionTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for tensorify behaviors on scalar values and symbolic ints
    # under compile.
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorify_track_item_symint(self):
        # [FUNCTION DOC] With scalar capture, checks a random resize to a data-dependent size
        # via item and interpolate compiles under fullgraph; output shape and divisibility by
        # 14 verified.
        def _random_resize(image: torch.Tensor):
            # [FUNCTION DOC] HELPER: Draws a random patch count with checks, interpolates the
            # image to that scaled size and returns the new size and image.
            image_metanet = image
            default_patch_size = 14
            rand_cnn_resolution = (224, 256)
            min_nump = rand_cnn_resolution[0] // default_patch_size
            max_nump = rand_cnn_resolution[1] // default_patch_size
            new_nump = torch.randint(min_nump, max_nump + 1, (1,)).item()
            torch._check(new_nump > 0)
            torch._check(new_nump * default_patch_size > 1)

            image_metanet = F.interpolate(
                image_metanet,
                size=(new_nump * default_patch_size, new_nump * default_patch_size),
                mode="bilinear",
                align_corners=True,
            )
            img_h_new, img_w_new = image_metanet.shape[2:]

            return (img_h_new, img_w_new), image_metanet

        _random_resize_compiled = torch.compile(fullgraph=True, backend="eager")(
            _random_resize
        )

        # Test the function
        input_tensor = torch.rand(1, 3, 224, 224)
        (h, w), output = _random_resize_compiled(input_tensor)

        # Verify output properties
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 3)
        self.assertEqual(output.shape[2], h)
        self.assertEqual(output.shape[3], w)
        self.assertTrue(h % 14 == 0)
        self.assertTrue(w % 14 == 0)
        self.assertTrue(224 <= h <= 256)
        self.assertTrue(224 <= w <= 256)

    def test_tensorify_scalar_only_bin_ops(self):
        # [FUNCTION DOC] Checks add, sub, mul and div on python floats under dynamic shapes
        # compile without graph break and match expected values.
        """When tensorify is enabled, torch bin_ops on scalar-only args should
        not graph-break. The scalars are lifted as 0-dim tensor placeholders by
        Dynamo; the tensorify pass converts them back to tensor ops."""
        for op, expected in [
            (torch.add, lambda a, b: a + b),
            (torch.sub, lambda a, b: a - b),
            (torch.mul, lambda a, b: a * b),
            (torch.div, lambda a, b: a / b),
        ]:
            with self.subTest(op=op.__name__):

                def fn(a, b):
                    # [FUNCTION DOC] HELPER: Applies the operator under test to the arguments.
                    return op(a, b)

                compiled = torch.compile(
                    fn, backend="eager", dynamic=True, fullgraph=True
                )
                result = compiled(3.5, 2.0)
                self.assertEqual(result, expected(3.5, 2.0))

    def test_tensorify_scalar_only_bin_ops_int(self):
        # [FUNCTION DOC] Checks add on python ints under dynamic shapes (result 8).
        """When tensorify is enabled, torch bin_ops on integer scalar-only
        args should not graph-break either. The ints are lifted as SymInts
        and symbolic arithmetic traces correctly."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns torch.add of a and b.
            return torch.add(a, b)

        compiled = torch.compile(fn, backend="eager", dynamic=True, fullgraph=True)
        result = compiled(5, 3)
        self.assertEqual(result, 8)

    def test_tensorify_symint_div_from_size(self):
        # [FUNCTION DOC] Checks torch div with truncation on a symbolic size and an int compiles;
        # result 4.
        """torch.div on SymInt args from tensor.size() should compile
        without graph break when tensorify is enabled."""

        def fn(x, divisor):
            # [FUNCTION DOC] HELPER: Divides the first dim size by divisor with trunc rounding.
            seq_len = x.size(0)
            return torch.div(seq_len, divisor, rounding_mode="trunc")

        compiled = torch.compile(fn, backend="eager", dynamic=True, fullgraph=True)
        result = compiled(torch.randn(1024), 256)
        self.assertEqual(result, 4)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorify_add_sub_alpha_from_item(self):
        # [FUNCTION DOC] With scalar capture, checks add and sub with non-unit alpha and an item
        # operand preserve alpha on aot_eager and inductor.
        """add/sub with a non-unit `alpha` and a `.item()` scalar operand must
        preserve `alpha`. The tensorify pass rebuilds the op from node.args
        only; `alpha` is keyword-only (node.kwargs) and was previously dropped,
        turning add(a, b, alpha=k) into a + b. Must run on aot_eager/inductor
        (the pass does not run under backend="eager")."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Reads item of b and returns add with alpha and sub with
            # alpha applied to sigmoid of a.
            s = b.item()
            return (
                torch.add(torch.sigmoid(a), s, alpha=-0.9248),
                torch.sub(torch.sigmoid(a), s, alpha=3.5),
            )

        a = torch.tensor(0.5)
        b = torch.tensor(2.0)
        expected = fn(a, b)
        for backend in ("aot_eager", "inductor"):
            with self.subTest(backend=backend):
                torch._dynamo.reset()
                got = torch.compile(fn, backend=backend, fullgraph=True)(a, b)
                self.assertEqual(got, expected)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
