# Owner(s): ["module: custom-operators"]
# ruff: noqa: F841
# [MODULE DOC] Tests for the custom operator APIs: the legacy torch._custom_op impl,
# torch.library.custom_op, torch.library.opcheck, Library registration, schema inference, fake
# profiles and related custom_op_db OpInfos. Defines a base TestCase that cleans up registered test
# ops, several TestCase groups, helper ops and subclasses, and device-generic instantiations at the
# bottom. Import time: defines MyList/MyTensor aliases used by string-type tests and computes
# device_type from the current accelerator, falling back to cpu.

import collections
import io
import itertools
import os
import re
import subprocess
import sys
import tempfile
import typing
import unittest
from functools import partial
from pathlib import Path
from typing import *  # noqa: F403
from unittest.mock import patch

import numpy as np
import yaml

import torch._custom_ops as custom_ops
import torch.distributed
import torch.testing._internal.optests as optests
import torch.utils._pytree as pytree
import torch.utils.cpp_extension
from torch import Tensor
from torch._custom_op.impl import CustomOp, infer_schema
from torch._library.fake_profile import (
    generate_yaml_from_profiles,
    load_op_profiles,
    MissingOpProfile,
    OpProfile,
    read_profiles_from_yaml,
    save_op_profiles,
    TensorMetadata,
)
from torch._library.infer_schema import tuple_to_list
from torch._utils_internal import get_file_path_2  # @manual
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal import custom_op_db
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
    PYTORCH_CUDA_MEMCHECK,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    scoped_load_inline,
    skipIfCrossRef,
    skipIfTorchDynamo,
    skipIfXpu,
    subtest,
    TemporaryFileName,
    TEST_ACCELERATOR,
    TEST_WITH_ASAN,
    TEST_WITH_SLOW,
    TEST_WITH_TORCHDYNAMO,
    TEST_XPU,
    TestCase,
)
from torch.testing._internal.custom_op_db import numpy_nonzero
from torch.testing._internal.optests.aot_autograd import _clone_input_for_aot_autograd
from torch.testing._internal.two_tensor import TwoTensor


# Shadowed by `torch.testing._internal.common_utils.custom_op`
from torch._custom_op.impl import custom_op  # usort: skip

# Needed by TestTypeConversion.test_string_type:
MyList = list
MyTensor = torch.Tensor

device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)


def requires_compile(fun):
    # [FUNCTION DOC] HELPER: Decorator that skips the wrapped test on Windows with a reason saying
    # torch.compile does not work there; returns the wrapped callable.
    fun = unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")(fun)
    return fun


class AOTAutogradCopyNoIsPinnedWrapperSubclass(torch.Tensor):
    # [CLASS DOC] HELPER: Tensor wrapper subclass holding an inner tensor in the attribute elem.
    # Its dispatch handler raises an AssertionError if is_pinned is ever called on it, so tests can
    # verify AOT input copying never checks pinning.
    @staticmethod
    def __new__(cls, elem):
        # [FUNCTION DOC] HELPER: Builds a wrapper subclass instance via _make_wrapper_subclass
        # copying size, stride, offset, dtype, layout, requires_grad and device from elem.
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
        # [FUNCTION DOC] HELPER: Stores the wrapped tensor in the attribute elem.
        self.elem = elem

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Flatten protocol: returns the inner attribute names (just elem)
        # and no metadata.
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Unflatten protocol: rebuilds the subclass from the inner
        # tensor elem restrided to the outer size and stride. Raises AssertionError if metadata
        # is not None.
        if metadata is not None:
            raise AssertionError("Expected metadata to be None")
        return AOTAutogradCopyNoIsPinnedWrapperSubclass(
            inner_tensors["elem"].as_strided(outer_size, outer_stride)
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Dispatch handler. Raises AssertionError if the op is
        # aten.is_pinned. Otherwise unwraps all subclass instances in args and kwargs, runs the op
        # and re-wraps tensor outputs in the subclass.
        if func is torch.ops.aten.is_pinned.default:
            raise AssertionError(
                "AOT input copying should not check pinning on wrapper subclasses"
            )
        if kwargs is None:
            kwargs = {}
        args = pytree.tree_map_only(
            AOTAutogradCopyNoIsPinnedWrapperSubclass, lambda x: x.elem, args
        )
        kwargs = pytree.tree_map_only(
            AOTAutogradCopyNoIsPinnedWrapperSubclass, lambda x: x.elem, kwargs
        )
        out = func(*args, **kwargs)
        return pytree.tree_map_only(
            torch.Tensor, AOTAutogradCopyNoIsPinnedWrapperSubclass, out
        )


class CustomOpTestCaseBase(TestCase):
    # [CLASS DOC] HELPER: Base TestCase for custom op tests. Uses the namespace in test_ns for all
    # ops, tracks Library objects created through lib in the libraries list, and tears down every
    # registered op and namespace afterwards.
    test_ns = "_test_custom_op"

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs base setUp and initializes an empty list of Library objects
        # to destroy during tearDown.
        super().setUp()
        self.libraries = []

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base tearDown, then destroys every CustomOp in the legacy
        # global registry whose key starts with the test namespace, deletes the namespace from
        # torch.ops if present, and destroys and drops tracked libraries.
        super().tearDown()
        import torch._custom_op

        keys = list(torch._custom_op.impl.global_registry.keys())
        for key in keys:
            if not key.startswith(f"{self.test_ns}::"):
                continue
            torch._custom_op.impl.global_registry[key]._destroy()
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        for lib in self.libraries:
            lib._destroy()
        del self.libraries

    def ns(self):
        # [FUNCTION DOC] HELPER: Returns the torch.ops namespace object for test_ns.
        return getattr(torch.ops, self.test_ns)

    def lib(self):
        # [FUNCTION DOC] HELPER: Creates a FRAGMENT Library in the test namespace, tracks it for
        # cleanup in tearDown and returns it.
        result = torch.library.Library(self.test_ns, "FRAGMENT")  # noqa: SCOPED_LIBRARY
        self.libraries.append(result)
        return result

    def get_op(self, qualname):
        # [FUNCTION DOC] HELPER: Looks up a legacy custom op by its qualified name using the
        # torch._custom_op.impl get_op function and returns it.
        return torch._custom_op.impl.get_op(qualname)


@requires_compile
class TestCustomOpTesting(CustomOpTestCaseBase):
    # [CLASS DOC] Device-generic tests of opcheck and the optests helpers (aot_autograd_check,
    # autograd_registration_check). Each test builds an op that is deliberately wrong and checks
    # the expected OpCheckError or AssertionError. Skipped on Windows via requires_compile.
    @parametrize("check_gradients", (False, "auto"))
    @parametrize("dynamic", (True, False))
    def test_aot_autograd_check_degenerate_cases(
        self, device, dynamic, check_gradients
    ):
        # [FUNCTION DOC] Checks aot_autograd_check passes without raising for degenerate functions
        # (clone, detached output, no differentiable output), for each combination of dynamic flag
        # and check_gradients setting (parametrized).
        def simple(x):
            # [FUNCTION DOC] HELPER: Returns a clone of x.
            return x.clone()

        # Should not raise
        x = torch.randn(3, device=device)
        optests.aot_autograd_check(
            simple, (x,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

        def outputs_dont_require_grad(x):
            # [FUNCTION DOC] HELPER: Returns x detached from the graph (defined but never passed to
            # the checker in this test).
            return x.detach()

        # Should not raise
        y = torch.randn(3, device=device, requires_grad=True)
        optests.aot_autograd_check(
            simple, (y,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

        def no_outputs(x):
            # [FUNCTION DOC] HELPER: Returns x detached, so the output never requires grad.
            return x.detach()

        # Should not raise
        x = torch.randn(3, device=device, requires_grad=True)
        y = torch.randn(3, device=device, requires_grad=False)
        optests.aot_autograd_check(
            no_outputs, (x,), {}, dynamic=dynamic, check_gradients=check_gradients
        )
        optests.aot_autograd_check(
            no_outputs, (y,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

    def test_incorrect_schema_mutation(self, device):
        # [FUNCTION DOC] Defines an op whose CPU/CUDA/XPU kernel mutates its input in place though
        # the schema says immutable; opcheck must raise OpCheckError saying the argument is not
        # defined as mutable but was mutated.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function that calls the op below autograd with dispatch
            # guard; backward passes the gradient through.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Calls the op under an autograd-below guard and removes the
                # guard afterwards in a finally block.
                guard = torch._C._AutoDispatchBelowAutograd()
                try:
                    return op(x)
                finally:
                    del guard

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged.
                return gx

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Backend kernel that mutates x with an in-place sin and returns
            # a clone, violating the declared schema.
            x.sin_()
            return x.clone()

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")

        x = torch.tensor(3.14159 / 3, requires_grad=True, device=device)
        with self.assertRaisesRegex(
            optests.OpCheckError, "Argument x is not defined as mutable but was mutated"
        ):
            torch.library.opcheck(op, (x,), {})

    def test_incorrect_schema_view(self, device):
        # [FUNCTION DOC] Defines an op whose kernel returns a view of the input though the schema
        # declares no alias; opcheck must raise OpCheckError that the argument is not defined to
        # alias the output but was aliasing it.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function calling the op with autograd and the
            # ADInplaceOrView key excluded; backward passes the gradient through.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Runs the op with autograd-below and ADInplaceOrView
                # excluded, emulating a guard not exposed to Python.
                # Emulate AutoDispatchBelowADInplaceOrView, which is not bound into python
                with torch._C._AutoDispatchBelowAutograd():
                    with torch._C._ExcludeDispatchKeyGuard(
                        torch._C.DispatchKeySet(torch._C.DispatchKey.ADInplaceOrView)
                    ):
                        return op(x)

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged.
                return gx

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: CPU kernel returning a view of x.
            return x.view_as(x)

        def foo_meta(x):
            # [FUNCTION DOC] HELPER: Meta kernel returning a view of x.
            return x.view_as(x)

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_meta, "Meta")

        x = torch.tensor(3.14159 / 3, requires_grad=True)
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "Argument x is not defined to alias output but was aliasing",
        ):
            torch.library.opcheck(op, (x,), {})

    # https://github.com/pytorch/pytorch/issues/149468
    def test_opcheck_cpu_stride_mismatch(self, device):
        # [FUNCTION DOC] Regression test for a stride mismatch: the fake implementation returns a
        # differently strided result than the real kernel, so opcheck raises OpCheckError
        # mentioning Stride mismatch.
        @torch.library.custom_op("test::stride_mismatch", mutates_args=())
        def stride_mismatch(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real kernel returning a clone permuted to dimension order 2,
            # 0, 1.
            return x.clone().permute(2, 0, 1)

        @stride_mismatch.register_fake
        def _(x):
            # [FUNCTION DOC] HELPER: Fake kernel returning an empty tensor with sizes taken in a
            # different order than the real result, giving other strides.
            c, h, w = x.shape[2], x.shape[0], x.shape[1]
            return x.new_empty(c, h, w)

        x = torch.randn(4, 4, 3, device=device)
        with self.assertRaisesRegex(optests.OpCheckError, "Stride mismatch"):
            torch.library.opcheck(stride_mismatch, (x,))

    # https://github.com/pytorch/pytorch/issues/142410
    def test_opcheck_unbacked_stride(self, device):
        # [FUNCTION DOC] Regression test that opcheck succeeds for an op whose fake implementation
        # returns a tensor with a dynamic (unbacked) second dimension.
        @torch.library.custom_op("test::f", mutates_args=[])
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real kernel returning zeros of shape batch by 18.
            return x.new_zeros((x.size(0), 18))

        @f.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake kernel allocating a new dynamic size from the library
            # context and returning an empty tensor with that second dimension.
            ctx = torch.library.get_ctx()
            s = ctx.new_dynamic_size()
            return torch.empty(x.shape[0], s, device=x.device, dtype=x.dtype)

        example = torch.zeros([10, 20], device=device)
        torch.library.opcheck(f, args=[example])

    # https://github.com/pytorch/pytorch/issues/150472
    def test_single_element_tuple_output(self, device):
        # [FUNCTION DOC] Regression test that a custom op returning a one-element tuple can be
        # registered with a fake kernel and called, returning a tuple of length one holding a value
        # equal to the input.
        # Helper function to register id_tuple custom and the fake tensor implementation
        # so that Dynamo has the fake tensor implementation
        def get_id_tuple():
            # [FUNCTION DOC] HELPER: Registers a custom op id_tuple with a fake kernel and returns
            # the op object.
            @torch.library.custom_op("test::id_tuple", mutates_args=[])
            def id_tuple(x: torch.Tensor) -> Tuple[torch.Tensor]:
                # [FUNCTION DOC] HELPER: Real kernel returning a one-tuple with a clone of x.
                return (x.clone(),)

            @id_tuple.register_fake
            def _(
                x: torch.Tensor,
            ) -> Tuple[torch.Tensor]:
                # [FUNCTION DOC] HELPER: Fake kernel returning a one-tuple with a clone of x.
                return (x.clone(),)

            return id_tuple

        id_tuple = get_id_tuple()
        x = torch.randn(3, device=device)
        ret = id_tuple(x)
        # Check if ret is a tuple and has exactly one and the same element
        self.assertIsInstance(ret, tuple)
        self.assertEqual(len(ret), 1)
        self.assertEqual(x, ret[0])

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/179898"
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TORCHDYNAMO or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/118253",
    )
    def test_missing_abstract_impl(self, device):
        # [FUNCTION DOC] Defines an op with a CPU kernel but no abstract (Meta) kernel; opcheck
        # must raise OpCheckError mentioning the op name. Skipped under several platform and
        # sanitizer conditions linked to GitHub issues.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function calling the op below autograd; backward doubles
            # the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Calls the op under an autograd-below guard.
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns twice the incoming gradient.
                return 2 * gx

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Kernel that goes through numpy and squares the values.
            return torch.tensor(x.cpu().numpy() ** 2, device=x.device)

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")

        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "_test_custom_op.foo.default",
        ):
            torch.library.opcheck(op, (x,), {})

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_incorrect_abstract_impl(self, device):
        # [FUNCTION DOC] Defines an op whose Meta kernel yields a different shape than the real
        # kernel; opcheck must raise OpCheckError with a shapes-not-equal message. Skipped
        # under Dynamo.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function calling the op with autograd and
            # ADInplaceOrView excluded through a combined with statement.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Runs the op with both dispatch guards active.
                # Emulate AutoDispatchBelowADInplaceOrView, which is not bound into python
                with (
                    torch._C._AutoDispatchBelowAutograd(),
                    torch._C._ExcludeDispatchKeyGuard(
                        torch._C.DispatchKeySet(torch._C.DispatchKey.ADInplaceOrView)
                    ),
                ):
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged.
                return gx

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Real kernel squaring x.
            return x**2

        def foo_meta(x):
            # [FUNCTION DOC] HELPER: Meta kernel returning the squared x with an extra unit
            # dimension, deliberately inconsistent with the real kernel.
            return x.unsqueeze(1) ** 2

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")
        lib.impl("foo", foo_meta, "Meta")

        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(optests.OpCheckError, "Shapes .* are not equal"):
            torch.library.opcheck(op, (x,), {})

    def test_missing_functionalization(self, device):
        # [FUNCTION DOC] Defines an op whose schema marks the argument as mutated and returned with
        # an alias annotation; opcheck must raise OpCheckError saying functionalization is only
        # supported for outputs without alias annotations.
        lib = self.lib()
        lib.define("foo(Tensor(a!) x) -> Tensor(a!)")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function that marks the input dirty and calls the op
            # below autograd.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Marks x dirty on ctx and runs the op below autograd.
                ctx.mark_dirty(x)
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged.
                return gx

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Kernel applying an in-place sin to x and returning it.
            return x.sin_()

        def foo_meta(x):
            # [FUNCTION DOC] HELPER: Meta kernel returning x unchanged.
            return x

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")
        lib.impl("foo", foo_meta, "Meta")

        x = torch.tensor([0, 1.0])
        y = x.clone()
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "We only support functionalizing operators whose outputs do not have alias annotations",
        ):
            torch.library.opcheck(op, (y,), {})

    def test_autograd_registered_at_backend(self, device):
        # [FUNCTION DOC] Registers the autograd function directly on backend keys rather than the
        # Autograd key; opcheck must raise OpCheckError saying the op has no autograd kernel.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function with a clone forward and a backward scaling the
            # gradient by one half.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Returns a clone of x.
                return x.clone()

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns half of the incoming gradient.
                return gx * 0.5

        lib.impl("foo", Foo.apply, "CPU")
        lib.impl("foo", Foo.apply, "CUDA")
        lib.impl("foo", Foo.apply, "XPU")
        lib.impl("foo", lambda x: x.clone(), "Meta")

        x = torch.randn([], requires_grad=True)

        with self.assertRaisesRegex(
            torch.testing._internal.optests.OpCheckError,
            "does not have an autograd kernel",
        ):
            torch.library.opcheck(op, (x,), {})

        # I'm not sure why this is necessary
        del lib

    def test_global_state_mutation(self, device):
        # [FUNCTION DOC] Registers a CompositeImplicitAutograd function whose forward depends on a
        # class counter incremented per call; opcheck must raise OpCheckError comparing eager and
        # AOTDispatcher, report absolute differences in the chained exception, and succeed when
        # atol and rtol overrides are large enough.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function with a class level counter named invoked that
            # increments on each forward call and scales the clone by it.
            invoked = 0

            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Increments the invoked counter and returns a clone of x
                # multiplied by the new counter value.
                Foo.invoked += 1
                return x.clone() * Foo.invoked

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged.
                return gx

        lib.impl("foo", Foo.apply, "CompositeImplicitAutograd")

        x = torch.tensor(3.14159 / 3, requires_grad=True)
        with self.assertRaisesRegex(
            optests.OpCheckError, "eager-mode PyTorch vs AOTDispatcher"
        ):
            torch.library.opcheck(op, (x,), {})

        # Test that we can actually see the absolute difference numbers
        try:
            torch.library.opcheck(op, (x,), {})
        except optests.OpCheckError as err:
            orig = err.__context__.__context__
            self.assertIn("Absolute difference:", str(orig))

        # Test atol/rtol overrides
        torch.library.opcheck(op, (x,), {}, atol=3, rtol=0.01)

    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/180338",
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/180323",
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/180301",
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/180262",
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/180214",
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/180196",
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/179992",
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/179909",
    )
    @ops(custom_op_db.custom_op_db, dtypes=OpDTypes.any_one)
    def test_opcheck_opinfo(self, device, dtype, op):
        # [FUNCTION DOC] Runs opcheck on every sample input of each OpInfo in custom_op_db (one
        # dtype per op), passing the input plus args and kwargs. Many skipIf decorators disable it
        # on Linux, macOS or ASAN, each citing a GitHub issue.
        for sample_input in op.sample_inputs(
            device, dtype, requires_grad=op.supports_autograd
        ):
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            torch.library.opcheck(op.op, args, kwargs)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/179991"
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TORCHDYNAMO or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/119159",
    )
    def test_opcheck_fails_basic(self, device):
        # [FUNCTION DOC] Defines a legacy custom op with a CPU/CUDA impl but no autograd formula;
        # opcheck on an input requiring grad must raise OpCheckError saying autograd has not been
        # implemented for the operator. Skipped on several platforms.
        @custom_op(f"{self.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor: ...
            # [FUNCTION DOC] HELPER: Legacy custom op declaration with only a schema stub.

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the sum of x.
            return x.sum()

        x = torch.randn(3, device=device, requires_grad=True)
        # Triggers the CustomOp autograd NYI error
        with self.assertRaisesRegex(
            optests.OpCheckError, "Autograd has not been implemented for operator"
        ):
            torch.library.opcheck(self.get_op(f"{self.test_ns}::foo"), (x,), {})

    def test_autograd_registration_check_autograd_kernel(self, device):
        # [FUNCTION DOC] Checks autograd_registration_check passes silently when the autograd
        # function is registered on the Autograd key.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function calling the op below autograd; backward passes
            # the gradient through.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Runs the op under an autograd-below guard.
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged.
                return gx

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Kernel returning the sine of x.
            return x.sin()

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")

        x = torch.randn(3, requires_grad=True, device=device)
        # Should not raise
        optests.autograd_registration_check(op, (x,), {})

    def test_autograd_registration_check_compositeimplicitautograd(self, device):
        # [FUNCTION DOC] Checks autograd_registration_check passes silently when the op is
        # registered as CompositeImplicitAutograd.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Composite kernel returning cos of sin of x.
            return x.sin().cos()

        lib.impl("foo", foo_impl, "CompositeImplicitAutograd")

        x = torch.randn(3, requires_grad=True, device=device)
        # Should not raise
        optests.autograd_registration_check(op, (x,), {})

    def test_autograd_registration_check_incorrect_composite(self, device):
        # [FUNCTION DOC] Registers the kernel as CompositeExplicitAutograd, which loses autograd;
        # autograd_registration_check must raise AssertionError mentioning incorrect registration.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Kernel returning cos of sin of x.
            return x.sin().cos()

        lib.impl("foo", foo_impl, "CompositeExplicitAutograd")

        x = torch.randn(3, requires_grad=True, device=device)
        with self.assertRaisesRegex(AssertionError, "incorrectly registered"):
            optests.autograd_registration_check(op, (x,), {})

    def test_autograd_registration_check_incorrect(self, device):
        # [FUNCTION DOC] Registers an autograd function only on backend keys; the
        # autograd_registration_check must raise AssertionError mentioning incorrect registration.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            # [FUNCTION DOC] HELPER: Autograd function whose forward is a plain sine; backward
            # passes the gradient through.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Returns the sine of x.
                return torch.sin(x)

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged.
                return gx

        lib.impl("foo", Foo.apply, "CPU")
        lib.impl("foo", Foo.apply, "CUDA")
        lib.impl("foo", Foo.apply, "XPU")

        x = torch.randn(3, requires_grad=True, device=device)
        with self.assertRaisesRegex(AssertionError, "incorrectly registered"):
            optests.autograd_registration_check(op, (x,), {})

    def test_assert_raises_regex(self, device):
        # [FUNCTION DOC] Tests the assert_raises_regex context manager from optests: it accepts a
        # matching exception and regex, and raises AssertionError when another exception type is
        # raised, when nothing is raised, or when the regex does not match.
        from torch.testing._internal.optests.aot_autograd import assert_raises_regex

        with assert_raises_regex(RuntimeError, "c"):
            raise RuntimeError("abcd")
        with assert_raises_regex(RuntimeError, "c.*"):
            raise RuntimeError("abcd")
        with self.assertRaisesRegex(AssertionError, "instead got"):
            with assert_raises_regex(RuntimeError, "c.*"):
                raise ValueError("abcd")
        with self.assertRaisesRegex(AssertionError, "Expected exception"):
            with assert_raises_regex(RuntimeError, "c.*"):
                pass
        with self.assertRaisesRegex(AssertionError, "to match regex"):
            with assert_raises_regex(RuntimeError, "f"):
                raise RuntimeError("abcd")


class TestCustomOp(CustomOpTestCaseBase):
    # [CLASS DOC] Main tests of the legacy and library custom op APIs in the test_ns namespace;
    # each test registers ops and relies on the base class tearDown for cleanup.
    test_ns = "_test_custom_op"

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    def test_library_impl_does_not_enable_pyobject_dispatch_by_default(self):
        # [FUNCTION DOC] Verifies that a Library impl registered for CompositeImplicitAutograd does
        # not enable the PyObject dispatcher by default: the op reports the dispatcher disabled,
        # results are correct and the composite function ran. Skipped under Dynamo.
        lib = self.lib()
        lib.define("pyobject_dispatch_composite(Tensor x) -> Tensor")

        calls = []

        def composite_impl(x):
            # [FUNCTION DOC] HELPER: Records a call in the shared list and returns x plus one.
            calls.append("composite")
            return x + 1

        lib.impl(
            "pyobject_dispatch_composite",
            composite_impl,
            "CompositeImplicitAutograd",
        )

        op = self.ns().pyobject_dispatch_composite.default
        x = torch.ones(2)
        self.assertFalse(op._is_pyobj_dispatcher_enabled())
        self.assertEqual(op(x), x + 1)
        self.assertTrue(calls)
        self.assertTrue(all(call == "composite" for call in calls))

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    def test_pyobject_dispatch_custom_op_enabled_by_default(self):
        # [FUNCTION DOC] Verifies a torch.library.custom_op enables the PyObject dispatcher by
        # default (the op reports it enabled), and that calling it runs the Python function exactly
        # once. Skipped under Dynamo.
        calls = []

        @torch.library.custom_op(
            f"{self.test_ns}::pyobject_dispatch_custom_op", mutates_args=()
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Custom op body that records a call marker and returns x
            # plus one.
            calls.append("custom")
            return x + 1

        op = self.ns().pyobject_dispatch_custom_op.default
        self.assertTrue(op._is_pyobj_dispatcher_enabled())

        x = torch.ones(2)
        self.assertEqual(f(x), x + 1)
        self.assertEqual(calls, ["custom"])

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    def test_pyobject_dispatch_custom_op_autograd(self):
        # [FUNCTION DOC] Verifies a custom op with a registered autograd formula runs backward
        # through the PyObject dispatcher: the backward function is invoked and the gradient equals
        # the cosine of the input. Skipped under Dynamo.
        @torch.library.custom_op(
            f"{self.test_ns}::pyobject_dispatch_custom_op_autograd",
            mutates_args=(),
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Custom op body returning the sine of x.
            return x.sin()

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup hook that unpacks the single input and saves it
            # for backward.
            (x,) = inputs
            ctx.save_for_backward(x)

        backward_called = False

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward formula; sets the enclosing flag to show it ran and
            # returns grad times the cosine of the saved input.
            nonlocal backward_called
            backward_called = True
            (x,) = ctx.saved_tensors
            return grad * x.cos()

        f.register_autograd(backward, setup_context=setup_context)

        op = self.ns().pyobject_dispatch_custom_op_autograd.default
        self.assertTrue(op._is_pyobj_dispatcher_enabled())

        x = torch.randn(3, requires_grad=True)
        f(x).sum().backward()
        self.assertTrue(backward_called)
        self.assertEqual(x.grad, x.detach().cos())

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    def test_pyobject_dispatch_normalizes_tensor_list_output(self):
        # [FUNCTION DOC] Verifies that a custom op returning a list of tensors (split results)
        # keeps the PyObject dispatcher disabled, hands the autograd backward function a list of
        # gradients, returns a list from the call, and produces the expected gradients (all ones).
        # Skipped under Dynamo.
        @torch.library.custom_op(
            f"{self.test_ns}::pyobject_dispatch_tensor_list_output",
            mutates_args=(),
        )
        def f(x: Tensor, sizes: Tensor) -> List[Tensor]:
            # [FUNCTION DOC] HELPER: Custom op body splitting x along dim 0 by the given sizes and
            # returning clones as a tuple.
            return tuple(t.clone() for t in torch.split(x, sizes.tolist(), dim=0))  # type: ignore[return-value]

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: No-op autograd setup hook.
            pass

        backward_called = False

        def backward(ctx, grad_outputs):
            # [FUNCTION DOC] HELPER: Backward formula; flags that it ran, asserts the incoming
            # gradients are a list, and returns their concatenation plus no gradient for sizes.
            nonlocal backward_called
            backward_called = True
            self.assertIsInstance(grad_outputs, list)
            return torch.cat(grad_outputs, dim=0), None

        f.register_autograd(backward, setup_context=setup_context)

        op = self.ns().pyobject_dispatch_tensor_list_output.default
        self.assertFalse(op._is_pyobj_dispatcher_enabled())

        x = torch.randn(6, 2, requires_grad=True)
        sizes = torch.tensor([1, 2, 3])
        y = f(x, sizes)
        self.assertIsInstance(y, list)
        self.assertEqual([part.shape[0] for part in y], sizes.tolist())

        sum(part.sum() for part in y).backward()
        self.assertTrue(backward_called)
        self.assertEqual(x.grad, torch.ones_like(x))

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    def test_pyobject_dispatch_normalizes_tensor_list_output_for_module(self):
        # [FUNCTION DOC] Verifies tensor-list outputs are normalized to a real list and that a
        # module with a forward pre-hook calling a submodule on the first element works on them,
        # while the extra tensor output is passed through unchanged. The PyObject dispatcher stays
        # disabled for this op. Skipped under Dynamo.
        @torch.library.custom_op(
            f"{self.test_ns}::pyobject_dispatch_module_tensor_list_output",
            mutates_args=(),
        )
        def f(x: Tensor, sizes: Tensor) -> Tuple[List[Tensor], Tensor]:
            # [FUNCTION DOC] HELPER: Custom op body returning the list of split clones plus a clone
            # of x.
            splits = tuple(t.clone() for t in torch.split(x, sizes.tolist(), dim=0))
            return typing.cast(List[Tensor], splits), x.clone()

        class ModuleExpectingTensor(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward requires a tensor input and returns an empty
            # tensor sized by the input's first dimension.
            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns an empty tensor with size equal to input's
                # first dimension.
                return torch.empty(input.size(0))

        class ListModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a forward pre-hook that calls a child
            # ModuleExpectingTensor on the first element of the first positional argument; forward
            # returns its list input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child module and registers the pre-hook.
                super().__init__()
                self.module_expecting_tensor = ModuleExpectingTensor()
                self.register_forward_pre_hook(self._forward_pre_hook)

            def _forward_pre_hook(self, module, inputs):
                # [FUNCTION DOC] HELPER: Pre-hook that feeds the first list element to the child
                # module via keyword argument and returns inputs unchanged.
                self.module_expecting_tensor(input=inputs[0][0])
                return inputs

            def forward(self, input_embs: List[Tensor]) -> List[Tensor]:
                # [FUNCTION DOC] HELPER: Returns the input list unchanged.
                return input_embs

        class Consumer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning a ListModule; wraps a non-list input in a list
            # before use.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child ListModule.
                super().__init__()
                self.list_module = ListModule()

            def forward(self, input_embs: List[Tensor]) -> Tensor:
                # [FUNCTION DOC] HELPER: Wraps non-list input in a list, passes the first element
                # through the list module in a one-element list and returns its first output.
                if not isinstance(input_embs, list):
                    input_embs = [input_embs]
                return self.list_module([input_embs[0]])[0]

        op = self.ns().pyobject_dispatch_module_tensor_list_output.default
        self.assertFalse(op._is_pyobj_dispatcher_enabled())

        x = torch.randn(6, 2)
        sizes = torch.tensor([1, 2, 3])
        y, passthrough = f(x, sizes)
        self.assertIsInstance(y, list)
        self.assertEqual(passthrough, x)
        self.assertEqual(Consumer()(y), y[0])

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    def test_pyobject_dispatch_normalizes_tensor_list_input(self):
        # [FUNCTION DOC] Verifies a custom op receiving a list of tensors gets a copy: when the op
        # appends to its argument, the caller's list keeps length one, both with the dispatcher at
        # default and after disabling PyObject dispatch. Skipped under Dynamo.
        @torch.library.custom_op(
            f"{self.test_ns}::pyobject_dispatch_tensor_list", mutates_args=()
        )
        def f(xs: list[Tensor]) -> Tensor:
            # [FUNCTION DOC] HELPER: Custom op body that appends a tensor to its list argument and
            # returns a clone of the first element.
            xs.append(torch.ones(()))
            return xs[0].clone()

        xs = [torch.zeros(())]
        f(xs)
        self.assertEqual(len(xs), 1)

        op = self.ns().pyobject_dispatch_tensor_list.default
        op._enable_pyobj_dispatch(False)
        xs = [torch.zeros(())]
        f(xs)
        self.assertEqual(len(xs), 1)

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    def test_pyobject_dispatch_passes_keyset_to_python_kernel(self):
        # [FUNCTION DOC] Registers a CPU kernel with the with_keyset option and a fallthrough
        # Autograd kernel; verifies the Python kernel receives a dispatch key set containing CPU
        # and the result is the input plus three. Skipped under Dynamo.
        lib = self.lib()
        lib.define("pyobject_dispatch_with_keyset(Tensor x) -> Tensor")

        seen_keysets = []

        def cpu_impl(keyset, x):
            # [FUNCTION DOC] HELPER: CPU kernel that records the key set and returns x plus three.
            seen_keysets.append(keyset)
            return x + 3

        lib.impl("pyobject_dispatch_with_keyset", cpu_impl, "CPU", with_keyset=True)
        lib.impl(
            "pyobject_dispatch_with_keyset",
            torch.library.fallthrough_kernel,
            "Autograd",
        )

        op = self.ns().pyobject_dispatch_with_keyset.default
        op._enable_pyobj_dispatch(True)

        x = torch.ones(2)
        self.assertEqual(op(x), torch.full((2,), 4.0))
        self.assertEqual(len(seen_keysets), 1)
        self.assertTrue(seen_keysets[0].has(torch._C.DispatchKey.CPU))

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    @skipIfCrossRef
    def test_pyobject_redispatch_requires_keyset(self):
        # [FUNCTION DOC] Verifies calling redispatch on the op's PyObject dispatcher without a key
        # set raises TypeError describing the expected arguments. Skipped under Dynamo and
        # cross-ref mode.
        @torch.library.custom_op(
            f"{self.test_ns}::pyobject_dispatch_redispatch_requires_keyset",
            mutates_args=(),
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Custom op body returning x.
            return x

        op = self.ns().pyobject_dispatch_redispatch_requires_keyset.default
        with self.assertRaisesRegex(
            TypeError,
            "redispatch\\(\\) expected redispatch\\(keyset, \\*args, \\*\\*kwargs\\)",
        ):
            op._pyobj_dispatcher.redispatch()

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    def test_pyobject_dispatch_respects_torch_function_mode(self):
        # [FUNCTION DOC] Verifies the prims abs op has the PyObject dispatcher enabled and that a
        # torch function mode (TorchRefsMode) is respected, giving the same result as the regular
        # abs. Skipped under Dynamo.
        import torch._refs
        from torch._prims.context import TorchRefsMode

        op = torch.ops.prims.abs.default
        self.assertTrue(op._is_pyobj_dispatcher_enabled())

        x = torch.randn(2)
        expected = x.abs()
        with TorchRefsMode():
            self.assertEqual(torch._refs.abs(x), expected)

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    def test_pyobject_dispatch_respects_torch_function_redispatch(self):
        # [FUNCTION DOC] Verifies a custom op called on a RedispatchTensor subclass goes through
        # torch function redispatch and returns a RedispatchTensor with correct values. Skipped
        # under Dynamo.
        from torch.testing._internal.common_subclass import RedispatchTensor

        @torch.library.custom_op(
            f"{self.test_ns}::pyobject_dispatch_torch_function_redispatch",
            mutates_args=(),
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Custom op body returning x plus one.
            return x + 1

        x = RedispatchTensor(torch.ones(2))
        y = f(x)
        self.assertIsInstance(y, RedispatchTensor)
        self.assertEqual(y, torch.full((2,), 2.0))

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    @scoped_load_inline
    def test_pyobject_dispatch_falls_back_to_cpp_dispatcher(self, load_inline):
        # [FUNCTION DOC] Compiles inline C++ defining a library op with CPU and fallthrough
        # Autograd kernels (CPU kernel adds two), enables PyObject dispatch for it and checks the
        # call falls back to the C++ dispatcher: ones become threes. Skipped under Dynamo.
        load_inline(
            name="test_pyobject_dispatch_cpp_fallback",
            cpp_sources="""
#include <torch/extension.h>

torch::Tensor pyobject_dispatch_cpp_fallback(const torch::Tensor& x) {
  return x + 2;
}

TORCH_LIBRARY(_test_pyobject_dispatch_cpp_fallback, m) {
  m.def("foo(Tensor x) -> Tensor");
  m.impl("foo", c10::DispatchKey::CPU, TORCH_FN(pyobject_dispatch_cpp_fallback));
  m.impl("foo", c10::DispatchKey::Autograd, torch::CppFunction::makeFallthrough());
}
""",
            is_python_module=False,
            verbose=True,
        )

        op = torch.ops._test_pyobject_dispatch_cpp_fallback.foo.default
        op._enable_pyobj_dispatch(True)

        x = torch.ones(2)
        self.assertEqual(op(x), torch.full((2,), 3.0))

    @skipIfTorchDynamo("PyObject dispatch test is eager-only")
    @scoped_load_inline
    def test_pyobject_dispatch_cpp_fallback_consumes_torch_function_skip(
        self, load_inline
    ):
        # [FUNCTION DOC] Compiles inline C++ defining a library op, enables PyObject dispatch and
        # calls it under a torch function mode that redispatches; verifies the result is still
        # computed by the C++ kernel (ones become threes). Skipped under Dynamo.
        load_inline(
            name="test_pyobject_dispatch_cpp_fallback_torch_function",
            cpp_sources="""
#include <torch/extension.h>

torch::Tensor pyobject_dispatch_cpp_fallback_torch_function(const torch::Tensor& x) {
  return x + 2;
}

TORCH_LIBRARY(_test_pyobject_dispatch_cpp_fallback_torch_function, m) {
  m.def("foo(Tensor x) -> Tensor");
  m.impl("foo", c10::DispatchKey::CPU, TORCH_FN(pyobject_dispatch_cpp_fallback_torch_function));
  m.impl("foo", c10::DispatchKey::Autograd, torch::CppFunction::makeFallthrough());
}
""",
            is_python_module=False,
            verbose=True,
        )

        class RedispatchMode(torch.overrides.TorchFunctionMode):
            # [CLASS DOC] HELPER: Torch function mode that forwards every call
            # to torch.overrides.redispatch_function.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Redispatches the function with the given types, args
                # and kwargs.
                return torch.overrides.redispatch_function(func, types, args, kwargs)

        op = torch.ops._test_pyobject_dispatch_cpp_fallback_torch_function.foo.default
        op._enable_pyobj_dispatch(True)

        x = torch.ones(2)
        with RedispatchMode():
            self.assertEqual(op(x), torch.full((2,), 3.0))

    @requires_compile
    def test_functionalize_error(self):
        # [FUNCTION DOC] Within a scoped library, defines an in-place op and a view op with alias
        # annotations, compiles functions using them with the aot_eager backend in full graph mode,
        # and verifies each raises RuntimeError saying functionalization only supports outputs
        # without alias annotations. Requires compile support.
        with torch.library._scoped_library(self.test_ns, "FRAGMENT") as lib:
            lib.define("foo(Tensor(a!) x) -> Tensor(a!)")

            def foo(x):
                # [FUNCTION DOC] HELPER: Kernel applying an in-place sin and returning x.
                return x.sin_()

            lib.impl("foo", foo, "CompositeExplicitAutograd")
            foo_op = self.get_op(f"{self.test_ns}::foo")

            lib.define("bar(Tensor(a) x) -> Tensor(a)")

            def bar(x):
                # [FUNCTION DOC] HELPER: Kernel returning a flattened view of x.
                return x.view(-1)

            lib.impl("bar", bar, "CompositeExplicitAutograd")
            bar_op = self.get_op(f"{self.test_ns}::bar")

            msg = r".*We only support functionalizing operators whose outputs do not have alias annotations"

            x = torch.randn(3)

            @torch.compile(backend="aot_eager", fullgraph=True)
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled wrapper that calls the in-place op.
                return foo_op(x)

            @torch.compile(backend="aot_eager", fullgraph=True)
            def g(x):
                # [FUNCTION DOC] HELPER: Compiled wrapper that calls the view op.
                return bar_op(x)

            with self.assertRaisesRegex(RuntimeError, msg):
                f(x)
            with self.assertRaisesRegex(RuntimeError, msg):
                g(x)

    def test_invalid_schemas(self):
        # [FUNCTION DOC] Verifies the legacy custom_ops.custom_op rejects a malformed schema string
        # by raising AssertionError naming an invalid function schema.
        # function schema validation goes through torchgen, so this is just a
        # basic test.
        with self.assertRaisesRegex(AssertionError, "Invalid function schema: foo"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(")

    def test_invalid_qualname(self):
        # [FUNCTION DOC] Verifies the legacy custom_op rejects a qualified name with an overload
        # suffix by raising ValueError mentioning overload.
        with self.assertRaisesRegex(ValueError, "overload"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo.Tensor", "() -> ()")

    def test_name_must_match(self):
        # [FUNCTION DOC] Verifies that decorating a function whose name differs from the operator
        # name raises ValueError saying the name is expected to match.
        with self.assertRaisesRegex(ValueError, "to have name"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def baz(x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Dummy function with a mismatched name; never runs.
                raise NotImplementedError

    def test_unsupported_schemas(self):
        # [FUNCTION DOC] Verifies the legacy custom_op raises ValueError for schemas it cannot
        # support: mutation, aliasing, a schema with no return value, and an argument named self.
        # Uses a module-level function foo.
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a!) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor x) -> ()")(
                foo
            )
        with self.assertRaisesRegex(ValueError, "self"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor self) -> ()")(
                foo
            )

    # Tests for the older custom_op API
    def test_schema_matches_signature(self):
        # [FUNCTION DOC] Verifies the older custom_op decorator raises ValueError when the schema
        # and Python signature disagree (names, order, varargs, default arguments) and accepts a
        # matching keyword-only signature.
        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(f"{TestCustomOp.test_ns}::blah", "(Tensor y) -> Tensor")
            def blah(x):
                # [FUNCTION DOC] HELPER: Decorated function whose argument name differs from
                # the schema.
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah2", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah2(x, y):
                # [FUNCTION DOC] HELPER: Decorated function whose keyword-only argument
                # is positional.
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah3",
                "(Tensor x, *, Tensor w, Tensor z) -> Tensor",
            )
            def blah3(x, *, y, z):
                # [FUNCTION DOC] HELPER: Decorated function with keyword-only names differing
                # from schema.
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah4",
                "(Tensor x, *, Tensor z, Tensor y) -> Tensor",
            )
            def blah4(x, *, y, z):
                # [FUNCTION DOC] HELPER: Decorated function with keyword-only names in a
                # different order.
                pass

        with self.assertRaisesRegex(ValueError, "not supported"):

            @custom_op(f"{TestCustomOp.test_ns}::blah5", "(Tensor x) -> Tensor")
            def blah5(*args):
                # [FUNCTION DOC] HELPER: Decorated function using variadic positional arguments.
                pass

        with self.assertRaisesRegex(ValueError, "not supported"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah6", "(*, Tensor z, Tensor y) -> Tensor"
            )
            def blah6(**kwargs):
                # [FUNCTION DOC] HELPER: Decorated function using variadic keyword arguments.
                pass

        with self.assertRaisesRegex(ValueError, "default arguments"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah7", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah7(x=1, *, y):
                # [FUNCTION DOC] HELPER: Decorated function with a default value on a
                # positional argument.
                pass

        with self.assertRaisesRegex(ValueError, "default arguments"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah8", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah8(x, *, y=1):
                # [FUNCTION DOC] HELPER: Decorated function with a default value on a
                # keyword-only argument.
                pass

        # kwonly-arg works
        @custom_op(
            f"{TestCustomOp.test_ns}::blah9", "(Tensor x, *, Tensor y) -> Tensor"
        )
        def blah9(x, *, y):
            # [FUNCTION DOC] HELPER: Valid decorated function with a keyword-only argument matching
            # the schema.
            pass

    def test_infer_schema_no_return(self):
        # [FUNCTION DOC] Verifies torch.library.custom_op raises ValueError saying no return type
        # annotation was provided when the function lacks one.
        with self.assertRaisesRegex(
            ValueError, "No return type annotation was provided. Please add one."
        ):

            @torch.library.custom_op("mylib::foo", mutates_args={})
            def foo(x: torch.Tensor, y: int):
                # [FUNCTION DOC] HELPER: Function missing a return annotation; never runs.
                return x * y

    def test_infer_schema_supported(self):
        # [FUNCTION DOC] Table-driven check of infer_schema output via assertExpectedInline for
        # many annotated functions: tensors, keyword-only args, scalars (SymInt, float, bool,
        # ScalarType, Device, Scalar), sequences, optionals, tuples, no return, mutation sets
        # (named, or all with the string unknown), default values, op_name prefix, inplace and out
        # tags, and string annotations resolved from this module.
        def a(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Single tensor in and out.
            return torch.empty([])

        self.assertExpectedInline(
            infer_schema(a, mutates_args=()), """(Tensor x) -> Tensor"""
        )

        def kwonly1(x: Tensor, *, y: int, z: float) -> Tensor:
            # [FUNCTION DOC] HELPER: Keyword-only int and float arguments.
            return torch.empty([])

        self.assertExpectedInline(
            infer_schema(kwonly1, mutates_args=()),
            """(Tensor x, *, SymInt y, float z) -> Tensor""",
        )

        def kwonly2(*, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Only a keyword-only tensor.
            return torch.empty([])

        self.assertExpectedInline(
            infer_schema(kwonly2, mutates_args=()), """(*, Tensor y) -> Tensor"""
        )

        def b(
            x: Tensor,
            y: int,
            z: bool,
            a: float,
            b: torch.dtype,
            c: torch.device,
            d: torch.types.Number,
        ) -> Tuple[Tensor, int, float, bool]:
            # [FUNCTION DOC] HELPER: Many scalar argument types and a tuple return.
            return torch.empty([]), 1, 0.1, True

        self.assertExpectedInline(
            infer_schema(b, mutates_args=()),
            """(Tensor x, SymInt y, bool z, float a, ScalarType b, Device c, Scalar d) -> (Tensor, SymInt, float, bool)""",
        )

        def c(
            x: Tensor,
            y: Sequence[Tensor],
            z: Optional[Tensor],
            w: Sequence[Optional[Tensor]],
        ) -> List[Tensor]:
            # [FUNCTION DOC] HELPER: Sequence, optional and list-of-optional tensor arguments.
            return [torch.empty([])]

        self.assertExpectedInline(
            infer_schema(c, mutates_args=()),
            """(Tensor x, Tensor[] y, Tensor? z, Tensor?[] w) -> Tensor[]""",
        )

        def d(x: Tensor) -> Tuple[List[Tensor], Tensor]:
            # [FUNCTION DOC] HELPER: Returns a tuple of a tensor list and a tensor.
            return [torch.empty([])], torch.empty([])

        self.assertExpectedInline(
            infer_schema(d, mutates_args=()), """(Tensor x) -> (Tensor[], Tensor)"""
        )

        def e() -> Tensor:
            # [FUNCTION DOC] HELPER: No arguments, returns a tensor.
            return torch.empty([])

        self.assertExpectedInline(infer_schema(e, mutates_args=()), """() -> Tensor""")

        def f(x: Tensor) -> None:
            # [FUNCTION DOC] HELPER: Tensor in, nothing returned.
            pass

        self.assertExpectedInline(
            infer_schema(f, mutates_args=()), """(Tensor x) -> ()"""
        )

        def g(
            x: Tensor, y: List[Tensor], z: List[Tensor], w: List[Optional[Tensor]]
        ) -> None:
            # [FUNCTION DOC] HELPER: Tensor and list arguments, no return; used with several
            # mutation sets.
            pass

        self.assertExpectedInline(
            infer_schema(g, mutates_args=()),
            """(Tensor x, Tensor[] y, Tensor[] z, Tensor?[] w) -> ()""",
        )

        self.assertExpectedInline(
            infer_schema(g, mutates_args={"x", "w", "z"}),
            """(Tensor(a0!) x, Tensor[] y, Tensor(a2!)[] z, Tensor(a3!)?[] w) -> ()""",
        )

        self.assertExpectedInline(
            infer_schema(g, mutates_args="unknown"),
            """(Tensor(a0!) x, Tensor(a1!)[] y, Tensor(a2!)[] z, Tensor(a3!)?[] w) -> ()""",
        )

        def h(
            x: Tensor,
            a: Optional[int] = None,
            b: float = 3.14,
            c: bool = True,
            d: int = 3,
            e: str = "foo",
            f: torch.dtype = torch.float,
            g: torch.dtype = torch.float32,
            h: torch.dtype = torch.int,
            i: torch.device = torch.device("cpu:0"),
            j: torch.device = "cpu",
        ) -> None:
            # [FUNCTION DOC] HELPER: Arguments with default values of assorted types.
            pass

        self.assertExpectedInline(
            infer_schema(h, mutates_args=()),
            (
                """(Tensor x, SymInt? a=None, float b=3.14, bool c=True, SymInt d=3, str e="foo", """
                """ScalarType f=float32, ScalarType g=float32, ScalarType h=int32, Device i="cpu:0", Device j="cpu") -> ()"""
            ),
        )

        def foo_impl(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return x.sin()

        schema = torch.library.infer_schema(foo_impl, op_name="myop", mutates_args={})
        self.assertExpectedInline(schema, "myop(Tensor x) -> Tensor")

        def inplace_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns x unchanged; schema inferred with the inplace tag.
            return x

        schema = torch.library.infer_schema(
            inplace_fn,
            mutates_args={"x"},
            tags=torch.Tag.inplace,
        )
        self.assertExpectedInline(schema, "(Tensor(a0!) x) -> Tensor(a0!)")

        def out_fn(x: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the keyword-only out tensor; schema inferred with the
            # out tag.
            return out

        schema = torch.library.infer_schema(
            out_fn,
            mutates_args={"out"},
            tags=torch.Tag.out,
        )
        self.assertExpectedInline(
            schema, "(Tensor x, *, Tensor(a1!) out) -> Tensor(a1!)"
        )

        # Ensure that a global in this file is properly found & evaluated.
        def stringy_fn(x: torch.Tensor) -> "MyList[torch.Tensor]":
            # [FUNCTION DOC] HELPER: Function whose return annotation is a string naming a
            # module-level alias.
            return [torch.randn_like(x)]

        schema = infer_schema(stringy_fn, mutates_args={})
        self.assertExpectedInline(schema, "(Tensor x) -> Tensor[]")

        # Make sure that substrings are evaluated properly.
        def substringy_fn(
            x: torch.Tensor,
        ) -> list["MyTensor"]:
            # [FUNCTION DOC] HELPER: Function whose return annotation holds a string name inside a
            # list type.
            return [torch.randn_like(x)]

        schema = infer_schema(substringy_fn, mutates_args={})
        self.assertExpectedInline(schema, "(Tensor x) -> Tensor[]")

    def test_infer_schema_unsupported(self):
        # [FUNCTION DOC] Verifies infer_schema raises ValueError for unsupported signatures:
        # varargs, varkwargs, missing annotation, variable-length tuple return, mutation of a
        # non-tensor argument, and a string annotation naming a symbol only defined in the
        # infer_schema module.
        with self.assertRaisesRegex(ValueError, "varargs"):

            def foo(*args):
                # [FUNCTION DOC] HELPER: Function with variadic positional arguments; never runs.
                raise NotImplementedError

            infer_schema(foo, mutates_args=())

        with self.assertRaisesRegex(ValueError, "varkwargs"):

            def foo(**kwargs):
                # [FUNCTION DOC] HELPER: Function with variadic keyword arguments; never runs.
                raise NotImplementedError

            infer_schema(foo, mutates_args=())

        with self.assertRaisesRegex(ValueError, "must have a type annotation"):

            def foo(x):
                # [FUNCTION DOC] HELPER: Function with an unannotated argument; never runs.
                raise NotImplementedError

            infer_schema(foo, mutates_args=())

        with self.assertRaisesRegex(ValueError, "unsupported"):

            def foo(x: Tensor) -> Tuple[Tensor, ...]:
                # [FUNCTION DOC] HELPER: Function returning a variable-length tuple; never runs.
                raise NotImplementedError

            infer_schema(foo, mutates_args=())

        with self.assertRaisesRegex(ValueError, "can be mutated"):

            def foo(x: Tensor, y: int) -> Tensor:
                # [FUNCTION DOC] HELPER: Function with a tensor and an int argument; never runs.
                raise NotImplementedError

            infer_schema(foo, mutates_args={"y"})

        # Ensure that a global defined in infer_schema's file ISN'T found.
        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation list\[_TestTensor\]\. It is not a type\.",
        ):

            def stringy_bad_type(
                x: torch.Tensor,
            ) -> "list[_TestTensor]":
                # [FUNCTION DOC] HELPER: Function whose return annotation names an undefined type;
                # never runs.
                return [torch.randn_like(x)]

            self.assertTrue(hasattr(torch._library.infer_schema, "_TestTensor"))
            schema = infer_schema(stringy_bad_type, mutates_args={})

    def _generate_examples(self, typ):
        # [FUNCTION DOC] HELPER: Returns a list of sample values for a type annotation: fixed
        # values for int, float, bool, str, GroupName (when distributed is available), dtype,
        # device, Number and Tensor; for Optional types the element examples plus None; for a list
        # type three copies of the element examples; for a Sequence type all ordered pairs of the
        # element examples. Raises AssertionError for malformed Optional/list/Sequence arguments
        # and NotImplementedError for any other type.
        if typ is int:
            return [17]
        if typ is float:
            return [3.14]
        if typ is bool:
            return [True]
        if typ is str:
            return ["foo"]
        if torch.distributed.is_available():
            from torch.distributed.distributed_c10d import GroupName

            if typ is GroupName:
                return ["group"]
        if typ is torch.dtype:
            return [torch.float32]
        if typ is torch.device:
            return [torch.device("cpu")]
        if typ == torch.types.Number:
            return [2.718]
        if typ is torch.Tensor:
            return [torch.tensor(3)]
        if typ == Optional[torch.types.Number]:
            return [None, 2.718]
        origin = typing.get_origin(typ)
        if origin is Union:
            args = typing.get_args(typ)
            if not (
                len(args) == 2 and (args[0] is type(None) or args[1] is type(None))
            ):
                raise AssertionError(f"expected Optional type, got {args}")
            elt = args[0] if args[1] is type(None) else args[1]
            return self._generate_examples(elt) + [None]
        if origin is list:
            args = typing.get_args(typ)
            if len(args) != 1:
                raise AssertionError(f"expected list with 1 arg, got {len(args)}")
            elt = args[0]
            return [
                self._generate_examples(elt),
                self._generate_examples(elt),
                self._generate_examples(elt),
            ]
        if origin is collections.abc.Sequence:
            args = typing.get_args(typ)
            if len(args) != 1:
                raise AssertionError(f"expected Sequence with 1 arg, got {len(args)}")
            examples = self._generate_examples(args[0])
            return list(itertools.product(examples, examples)) + []
        raise NotImplementedError(
            f"testrunner cannot generate instanstance of type {typ}"
        )

    def test_supported_return_types_single_return(self):
        # [FUNCTION DOC] For each type in SUPPORTED_RETURN_TYPES and each example value, defines a
        # legacy op returning that type, implements it to return the example, calls it and compares
        # the result; the op is destroyed in a finally block each iteration.
        for typ in torch._library.infer_schema.SUPPORTED_RETURN_TYPES:
            for example in self._generate_examples(typ):
                try:

                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> typ:
                        # [FUNCTION DOC] HELPER: Schema declaration whose return annotation is the
                        # type under test.
                        raise NotImplementedError

                    @custom_ops.impl(f"{self.test_ns}::foo")
                    def foo_impl(x: Tensor) -> typ:
                        # [FUNCTION DOC] HELPER: Impl returning the captured example value.
                        return example

                    op = self.get_op(f"{self.test_ns}::foo")
                    result = op(torch.randn([]))
                    self.assertEqual(
                        result, example, msg=lambda msg: f"{msg}\n{typ} {example}"
                    )
                finally:
                    custom_ops._destroy(f"{self.test_ns}::foo")

    def test_supported_return_types_multi_return(self):
        # [FUNCTION DOC] For each supported return type and example, defines a legacy op returning
        # a pair of that type, implements it to return the example twice and checks the call result
        # equals the pair; the op is destroyed after every iteration.
        for typ in torch._library.infer_schema.SUPPORTED_RETURN_TYPES:
            for example in self._generate_examples(typ):
                try:

                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> Tuple[typ, typ]:
                        # [FUNCTION DOC] HELPER: Schema declaration returning a pair of the type
                        # under test.
                        raise NotImplementedError

                    @custom_ops.impl(f"{self.test_ns}::foo")
                    def foo_impl(x: Tensor) -> Tuple[typ, typ]:
                        # [FUNCTION DOC] HELPER: Impl returning a pair of the example value.
                        return (example, example)

                    op = self.get_op(f"{self.test_ns}::foo")
                    result = op(torch.randn([]))
                    expected = (example, example)
                    self.assertEqual(
                        result, expected, msg=lambda msg: f"{msg}\n{typ} {example}"
                    )
                finally:
                    custom_ops._destroy(f"{self.test_ns}::foo")

    def test_supported_param_types(self):
        # [FUNCTION DOC] For each type in SUPPORTED_PARAM_TYPES, defines a legacy op taking a
        # tensor and a parameter of that type, implements it on CPU recording the parameter, calls
        # it with every generated example and checks the recorded value; the op is destroyed in a
        # finally block.
        for typ in torch._library.infer_schema.SUPPORTED_PARAM_TYPES:

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: typ) -> Tensor:
                # [FUNCTION DOC] HELPER: Schema declaration with a parameter of the type
                # under test.
                raise NotImplementedError

            yeet = None

            @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types=["cpu"])
            def foo_cpu(x, y):
                # [FUNCTION DOC] HELPER: CPU impl storing the received parameter in the enclosing
                # variable and returning a clone of x.
                nonlocal yeet
                yeet = y
                return x.clone()

            try:
                for example in self._generate_examples(typ):
                    op = self.get_op(f"{self.test_ns}::foo")
                    op(torch.randn([]), example)
                    self.assertEqual(
                        yeet, example, msg=lambda msg: f"{msg}\n{typ} {example}"
                    )
                    yeet = None
            finally:
                custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_sequences(self):
        # [FUNCTION DOC] Verifies an op declared with a Sequence of ints parameter accepts an
        # arbitrary user-defined Sequence subclass; the CPU impl is called once and sees the
        # dispatcher-normalized list of sizes.
        # Sequence[int] gets automagically turned into int[] in the schema.
        # This test checks that we actually do support arbitrary sequence types.
        class MySequence(collections.abc.Sequence):
            # [CLASS DOC] HELPER: Minimal Sequence implementation wrapping a list of 1, 2, 3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the backing list.
                self._container = [1, 2, 3]

            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: Returns the element or slice at the given index.
                return self._container[idx]

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns the length of the backing list.
                return len(self._container)

        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Schema declaration with a Sequence of ints argument.
            raise NotImplementedError

        called = 0

        @custom_ops.impl(f"{self.test_ns}::foo", device_types="cpu")
        def foo_cpu(x, sizes):
            # [FUNCTION DOC] HELPER: CPU impl that counts calls, asserts the sizes argument equals
            # a plain list of 1, 2, 3 and returns a clone of x.
            nonlocal called
            called += 1
            # Dispatcher will normalize the sequence type into a List
            self.assertEqual(sizes, [1, 2, 3])
            return x.clone()

        x = torch.randn([])
        seq = MySequence()
        op = self.get_op(f"{self.test_ns}::foo")
        op(x, seq)
        self.assertEqual(called, 1)

    def test_unsupported_param_types(self):
        # [FUNCTION DOC] Verifies that legacy and library custom op declarations raise ValueError
        # (unsupported type) for list of optional ints, tuples, Callable and a namedtuple
        # annotation; the tuple error under torch.library.custom_op suggests a list type, and the
        # error for a mixed tuple does not include an example.
        # Not comprehensive (it doesn't need to be), just a check that our mechanism works
        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: List[Optional[int]]) -> Tensor:
                # [FUNCTION DOC] HELPER: Declaration with a list of optional ints parameter;
                # deleted afterwards.
                raise NotImplementedError

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # int[N] in Dispatcher is a bit wild, so we don't try to support it.
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                # [FUNCTION DOC] HELPER: Declaration with a tuple parameter; deleted afterwards.
                raise NotImplementedError

            del foo

        with self.assertRaisesRegex(ValueError, r"For example, list\[int\]"):
            # test that we propose a correct and supported type.
            @torch.library.custom_op(f"{TestCustomOp.test_ns}::foo", mutates_args={})
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                # [FUNCTION DOC] HELPER: Library custom op with a tuple parameter; should fail with
                # a suggestion.
                raise NotImplementedError

            del foo

        with self.assertRaises(ValueError) as cm:

            @torch.library.custom_op(f"{TestCustomOp.test_ns}::foo", mutates_args={})
            def foo(x: Tensor, y: Tuple[int, float]) -> Tensor:
                # [FUNCTION DOC] HELPER: Library custom op with a mixed tuple parameter;
                # deleted afterwards.
                raise NotImplementedError

            del foo

            self.assertNotIn("example", str(cm.exception), "")

        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Callable) -> Tensor:
                # [FUNCTION DOC] HELPER: Declaration with a Callable parameter; deleted afterwards.
                raise NotImplementedError

            del foo

        # Define a named tuple for a Point with x and y coordinates
        Point = collections.namedtuple("Point", ["x", "y"])
        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Point) -> Tensor:
                # [FUNCTION DOC] HELPER: Declaration with a namedtuple parameter;
                # deleted afterwards.
                raise NotImplementedError

            del foo

    def test_supported_schemas(self):
        # [FUNCTION DOC] Sanity check that the legacy custom_op accepts a list of valid schema
        # strings (single or multiple outputs, optional, list and scalar types) and destroys each
        # op right after creation.
        # All of these should already be tested by PyTorch codegen
        # (we share the same mechanism), but here's a sanity check.
        schemas = [
            "(Tensor x) -> Tensor",
            "(Tensor x) -> Tensor y",
            "(Tensor[] x) -> Tensor y",
            "(Tensor x) -> (Tensor, Tensor)",
            "(Tensor x) -> (Tensor y, Tensor z)",
            "(Tensor x) -> (Tensor y, Tensor z)",
        ]
        other_schemas = [
            "(Tensor x, Tensor w) -> (Tensor y, Tensor z)",
            "(Tensor x, Tensor w) -> (Tensor, Tensor)",
            "(Tensor x, Tensor w) -> Tensor",
            "(Tensor? x, Tensor w) -> Tensor",
            "(Tensor? x, Tensor[] w) -> Tensor",
            "(Tensor x, int[] w) -> Tensor",
            "(Tensor x, SymInt[] w) -> Tensor",
            "(Tensor x, Scalar w) -> Tensor",
            "(Tensor x, float w) -> Tensor",
            "(Tensor x, float? w) -> Tensor",
            "(Tensor x, bool[] w) -> Tensor",
        ]

        for schema in schemas:
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", schema)
            custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        for schema in other_schemas:
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::bar", schema)
            custom_ops._destroy(f"{TestCustomOp.test_ns}::bar")

    def test_reserved_ns(self):
        # [FUNCTION DOC] Verifies each name in RESERVED_NS is rejected as an op namespace with
        # ValueError, both for the schema form and the decorator form.
        from torch._custom_op.impl import RESERVED_NS

        for ns in RESERVED_NS:
            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):
                custom_ops.custom_op(f"{ns}::foo", "(Tensor x) -> Tensor")

            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):

                @custom_ops.custom_op(f"{ns}::foo2")
                def foo2(x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Decorated stub in a reserved namespace; never runs.
                    raise NotImplementedError

    def test_private_ctor(self):
        # [FUNCTION DOC] Verifies constructing CustomOp directly raises RuntimeError saying the
        # constructor is private.
        with self.assertRaisesRegex(RuntimeError, "CustomOp constructor is private"):
            CustomOp(None, None, None, None, None)

    def test_lifetime(self):
        # [FUNCTION DOC] Verifies an op cannot be defined twice (RuntimeError mentioning multiple
        # times), but can be defined again after being destroyed.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub for the first definition.
            raise NotImplementedError

        custom_op = torch._custom_op.impl.get_op(f"{TestCustomOp.test_ns}::foo")

        # We can't define an op multiple times,
        with self.assertRaisesRegex(RuntimeError, "multiple times"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Duplicate declaration expected to fail.
                raise NotImplementedError

        # Unless we delete the original op.
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

        # Smoke test
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Redeclaration after the original was destroyed.
            raise NotImplementedError

        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_autograd_notimplemented(self):
        # [FUNCTION DOC] Verifies calling a legacy op with an input requiring grad (a single
        # tensor, a tensor sequence, or a mix) raises RuntimeError that autograd is not
        # implemented; the op is destroyed between cases.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a single tensor argument.
            raise NotImplementedError

        x = torch.randn(3, requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(x)
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        del foo

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: Sequence[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a tensor sequence argument.
            raise NotImplementedError

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op([y, x])
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        del foo

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with two tensor arguments.
            raise NotImplementedError

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(y, x)
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_autograd_notimplemented_gradmode(self):
        # [FUNCTION DOC] Verifies the autograd-not-implemented error is not raised under no_grad
        # mode even when an input requires grad.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with two tensor arguments.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            # [FUNCTION DOC] HELPER: Impl returning the elementwise product.
            return x * y

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with torch.no_grad():
            # Shouldn't raise, because we are in no_grad
            op(y, x)

    def test_impl_cpu(self):
        # [FUNCTION DOC] Verifies a CPU impl registered via device_types works and the op call
        # equals running the impl directly.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types="cpu")
        def foo_cpu(x):
            # [FUNCTION DOC] HELPER: CPU impl returning the sine of x.
            return x.sin()

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        result = op(x)
        self.assertEqual(result, foo_cpu(x))

    def test_impl_invalid_devices(self):
        # [FUNCTION DOC] Verifies every device type in SUPPORTED_DEVICE_TYPE_TO_KEY can be used
        # with impl, and that unsupported types (hip, xla, mkldnn, or a list containing hip) raise
        # ValueError saying only certain device types are supported.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x.
            return x.sin()

        from torch._custom_op.impl import SUPPORTED_DEVICE_TYPE_TO_KEY

        for device_type in SUPPORTED_DEVICE_TYPE_TO_KEY:
            # Smoke test: should not raise error
            custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types=device_type)(
                foo_impl
            )

        # Not supported by this API: we can either support them in the future
        # or provide some other CustomOp.def_* function. This depends on how
        # common the use cases are.
        for invalid_type in ["hip", "xla", "mkldnn", ["cpu", "hip"]]:
            with self.assertRaisesRegex(ValueError, "we only support device_type"):
                custom_ops.impl(
                    f"{TestCustomOp.test_ns}::foo", device_types=invalid_type
                )(foo_impl)

    def test_backward_partially_registered(self):
        # [FUNCTION DOC] Registers an impl and a backward but no save_for_backward; calling
        # backward raises RuntimeError saying no save_for_backward function was found.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x.
            return x.sin()

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning grad times cosine of the saved value.
            return grad * saved.cos()

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(
            RuntimeError, "unable to find a 'save_for_backward'"
        ):
            y = op(x)
            y.backward()

    def test_save_for_backward_inputs_are_namedtuple(self):
        # [FUNCTION DOC] Verifies the inputs passed to save_for_backward are a namedtuple whose
        # only key is x, that it runs once at forward time and not again at backward.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x.
            return x.sin()

        hit = 0

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Counts calls, checks inputs is a namedtuple with key x and
            # returns x.
            nonlocal hit
            hit += 1
            self.assertTrue(isinstance(inputs, tuple))
            self.assertEqual(list(inputs._asdict().keys()), ["x"])
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning a dict mapping x to grad times cosine
            # of saved.
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        self.assertEqual(hit, 1)
        y.backward()
        self.assertEqual(hit, 1)

    def test_backward_returns_dict(self):
        # [FUNCTION DOC] Verifies backward raises RuntimeError saying a dict is expected when the
        # backward function returns a plain tensor.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x.
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the x input.
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward wrongly returning a tensor instead of a dict.
            return grad * saved.cos()

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        with self.assertRaisesRegex(RuntimeError, "to be a dict"):
            y.backward()

    def test_backward_dict_invalid_keys(self):
        # [FUNCTION DOC] Verifies backward raises RuntimeError when the returned dict has keys
        # other than the op's differentiable input names.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x.
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the x input.
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning a dict with an extra key y.
            return {"x": grad * saved.cos(), "y": None}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        with self.assertRaisesRegex(RuntimeError, "to have keys {'x'}"):
            y.backward()

    def test_backward_dict_grad_for_nontensor(self):
        # [FUNCTION DOC] Verifies backward raises RuntimeError when the gradient dict contains a
        # key for a non-tensor argument (dim), even with value None.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a tensor and an int argument.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, dim):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x, ignoring dim.
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the x input.
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning a dict that includes the int argument dim.
            return {"x": grad * saved.cos(), "dim": None}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x, 32)
        with self.assertRaisesRegex(RuntimeError, "non-Tensor-like types"):
            y.backward()

    def test_backward_dict_requires_keys_for_input_tensors(self):
        # [FUNCTION DOC] Verifies backward raises RuntimeError when the gradient dict omits a key
        # for a tensor input (y).
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with two tensor arguments.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x.
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the x input.
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning a dict missing the y key.
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x, x)
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            y.backward()

    def test_backward_dict_requires_keys_for_input_optional_tensors(self):
        # [FUNCTION DOC] Verifies backward raises RuntimeError when the gradient dict omits a key
        # for an optional tensor input (y) even though None was passed.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a tensor and an optional tensor argument.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x.
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the x input.
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning a dict missing the y key.
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x, None)
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            y.backward()

    def test_backward_grads_are_tensor_or_none(self):
        # [FUNCTION DOC] Verifies backward raises RuntimeError saying gradients must be None or a
        # Tensor when the dict value for a tensor input is a tuple.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x.
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the x input.
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning a tuple holding the gradient as the
            # x entry.
            return {"x": (grad * saved.cos(),)}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        with self.assertRaisesRegex(RuntimeError, "either None or a Tensor"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads_with_same_numel(self):
        # [FUNCTION DOC] Verifies backward raises RuntimeError when the gradient list for a tensor
        # sequence input has two entries instead of three.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a tensor sequence argument.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(xs):
            # [FUNCTION DOC] HELPER: Impl returning the sine of the first tensor.
            return xs[0].sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the first tensor of the sequence.
            return inputs.xs[0]

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning a gradient list that is one entry
            # too short.
            return {"xs": [grad * saved.cos(), None]}

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(xs)
        with self.assertRaisesRegex(RuntimeError, "3 gradients but got 2"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads_none_or_Tensor(self):
        # [FUNCTION DOC] Verifies backward raises RuntimeError saying None or Tensor is required
        # when an entry of the gradient list is a tuple.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a tensor sequence argument.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(xs):
            # [FUNCTION DOC] HELPER: Impl returning the sine of the first tensor.
            return xs[0].sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the first tensor of the sequence.
            return inputs.xs[0]

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning a list whose last entry is a tuple.
            return {"xs": [grad * saved.cos(), None, (None,)]}

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(xs)
        with self.assertRaisesRegex(RuntimeError, "None or Tensor"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads(self):
        # [FUNCTION DOC] Verifies backward raises RuntimeError mentioning a list of gradients when
        # the gradient for a tensor sequence input is None rather than a list.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a tensor sequence argument.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(xs):
            # [FUNCTION DOC] HELPER: Impl returning the sine of the first tensor.
            return xs[0].sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the first tensor of the sequence.
            return inputs.xs[0]

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # [FUNCTION DOC] HELPER: Backward returning None for the sequence gradient.
            return {"xs": None}

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(xs)
        with self.assertRaisesRegex(RuntimeError, "list of gradients"):
            y.backward()

    def test_backward_output_differentiability_type(self):
        # [FUNCTION DOC] Verifies impl_backward raises RuntimeError when output_differentiability
        # is a bool instead of a list.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a tensor sequence argument.
            raise NotImplementedError

        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):

            @custom_ops.impl_backward(
                f"{TestCustomOp.test_ns}::foo", output_differentiability=True
            )
            def foo_backward(ctx, saved, grad):
                # [FUNCTION DOC] HELPER: Backward stub; registration fails before it is used.
                return {"xs": None}

    def test_backward_output_differentiability_numel(self):
        # [FUNCTION DOC] Verifies impl_backward raises RuntimeError when output_differentiability
        # has one entry for an op with two outputs.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Declaration returning two tensors.
            raise NotImplementedError

        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):

            @custom_ops.impl_backward(
                f"{TestCustomOp.test_ns}::foo", output_differentiability=[True]
            )
            def foo_backward(ctx, saved, grad):
                # [FUNCTION DOC] HELPER: Backward stub; registration fails before it is used.
                return {"xs": None}

    def test_backward_output_differentiability_tensorlist(self):
        # [FUNCTION DOC] Verifies output_differentiability for an op returning a tensor list plus a
        # tensor: the list elements do not require grad while the lone tensor does.
        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: Tensor) -> Tuple[List[Tensor], Tensor]:
            # [FUNCTION DOC] HELPER: Declaration returning a tensor list and a tensor.
            raise NotImplementedError

        @custom_ops.impl(f"{self.test_ns}::foo")
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning a list of two clones and another clone.
            return [x.clone(), x.clone()], x.clone()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves nothing (empty list).
            return []

        @custom_ops.impl_backward(
            f"{TestCustomOp.test_ns}::foo", output_differentiability=[False, True]
        )
        def foo_backward(ctx, saved, grad_lst, grad):
            # [FUNCTION DOC] HELPER: Backward receiving the list gradient and the tensor gradient,
            # returning the tensor gradient for x.
            return {"x": grad}

        op = self.get_op(f"{self.test_ns}::foo")
        x = torch.randn(3, requires_grad=True)
        [a, b], c = op(x)
        self.assertFalse(a.requires_grad)
        self.assertFalse(b.requires_grad)
        self.assertTrue(c.requires_grad)

    def test_backward_output_differentiability_non_tensor(self):
        # [FUNCTION DOC] Verifies that marking the int output as differentiable in
        # output_differentiability makes calling the op raise RuntimeError saying the output is not
        # a Tensor.
        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: Tensor) -> Tuple[Tensor, int]:
            # [FUNCTION DOC] HELPER: Declaration returning a tensor and an int.
            raise NotImplementedError

        @custom_ops.impl(f"{self.test_ns}::foo")
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning a clone and the int 3.
            return x.clone(), 3

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves nothing (empty list).
            return []

        @custom_ops.impl_backward(
            f"{TestCustomOp.test_ns}::foo", output_differentiability=[True, True]
        )
        def foo_backward(ctx, saved, grad0, grad1):
            # [FUNCTION DOC] HELPER: Backward taking two output gradients and returning the first
            # for x.
            return {"x": grad0}

        op = self.get_op(f"{self.test_ns}::foo")
        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "is not a Tensor"):
            op(x)

    @skipIfXpu(msg="Deprecated torch.custom_ops API")
    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires CUDA or XPU")
    def test_impl_separate(self):
        # [FUNCTION DOC] Defines a legacy op with separate CPU and accelerator impls (sine and
        # cosine) and checks a CPU input dispatches to the first and an accelerator input to the
        # second. Requires CUDA or XPU; skipped on XPU for the deprecated API.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types="cpu")
        def foo_cpu(x):
            # [FUNCTION DOC] HELPER: CPU impl returning the sine of x.
            return x.sin()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types=device_type)
        def foo_cuda(x):
            # [FUNCTION DOC] HELPER: Accelerator impl returning the cosine of x.
            return x.cos()

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        result = op(x)
        self.assertEqual(result, foo_cpu(x))

        x_cuda = x.to(device_type)
        op = self.get_op(f"{self.test_ns}::foo")
        result = op(x_cuda)
        self.assertEqual(result, foo_cuda(x_cuda))

    @skipIfXpu(msg="Deprecated torch.custom_ops API")
    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires CUDA or XPU")
    def test_impl_multiple(self):
        # [FUNCTION DOC] Registers a single impl without device restriction and checks it is used
        # for both CPU and accelerator inputs. Requires CUDA or XPU; skipped on XPU.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the cosine of x.
            return x.cos()

        op = self.get_op(f"{self.test_ns}::foo")
        x = torch.randn(3)
        result = op(x)
        self.assertEqual(result, foo_impl(x))

        x_cuda = x.to(device_type)
        result = op(x_cuda)
        self.assertEqual(result, foo_impl(x_cuda))

    def test_impl_abstract_overload(self):
        # [FUNCTION DOC] Registers a fake kernel (torch.empty_like) on an overloaded op name via
        # torch.library.register_fake and verifies calling the overload on a meta tensor succeeds.
        lib = self.lib()
        lib.define("sin.blah(Tensor x) -> Tensor")

        torch.library.register_fake(
            f"{self.test_ns}::sin.blah", torch.empty_like, lib=lib
        )

        op = self.ns().sin.blah
        x = torch.randn(3, device="meta")
        op(x)

    def test_impl_meta(self):
        # [FUNCTION DOC] Registers a fake kernel for a legacy op that drops one dimension and
        # checks the result shape on a meta input matches calling the kernel directly.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a tensor and an int dim.
            raise NotImplementedError

        @torch.library.register_fake(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
        def foo_meta(x, dim):
            # [FUNCTION DOC] HELPER: Fake kernel returning an empty tensor shaped like x without
            # dimension dim.
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        x = torch.randn(2, 3, device="meta")
        op = self.get_op(f"{self.test_ns}::foo")
        result = op(x, 1)
        self.assertEqual(result.shape, foo_meta(x, 1).shape)

    def test_duplicate_impl(self):
        # [FUNCTION DOC] Verifies registering a second fake kernel with allow_override disabled
        # raises RuntimeError whose message cites a location in this test file.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration with a tensor and an int dim.
            raise NotImplementedError

        @torch.library.register_fake(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
        def foo_meta(x, dim):
            # [FUNCTION DOC] HELPER: First fake kernel dropping dimension dim.
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        with self.assertRaisesRegex(RuntimeError, r"test_custom_ops.py:\d+"):

            @torch.library.register_fake(
                f"{TestCustomOp.test_ns}::foo", lib=self.lib(), allow_override=False
            )
            def foo_meta2(x, dim):
                # [FUNCTION DOC] HELPER: Second fake kernel, same behavior; registration expected
                # to fail.
                output_shape = list(x.shape)
                del output_shape[dim]
                return x.new_empty(output_shape)

    def test_new_data_dependent_symint(self):
        # [FUNCTION DOC] Exercises the fake kernel context's new_dynamic_size: min of 1 works, a
        # negative min and a max given as a plain int raise ValueError; the kernel returns a tensor
        # with the dynamic size, and make_fx with symbolic tracing runs it.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @torch.library.register_fake(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
        def foo_meta(x):
            # [FUNCTION DOC] HELPER: Fake kernel that creates a dynamic size, checks the two error
            # cases and returns an empty tensor of that size.
            ctx = torch.library.get_ctx()
            r = ctx.new_dynamic_size(min=1)
            with self.assertRaisesRegex(ValueError, "greater than or equal to 0"):
                ctx.new_dynamic_size(min=-1)
            with self.assertRaisesRegex(ValueError, "SymInt"):
                ctx.new_dynamic_size(max=x.numel())
            # NB: You must return dynamic sizes!
            return x.new_empty(r)

        x = torch.randn(2, 3, device="cpu")
        op = self.get_op(f"{self.test_ns}::foo")
        make_fx(op, tracing_mode="symbolic")(x)

    def test_meta_for_data_dependent_shape_operation(self):
        # [FUNCTION DOC] Verifies calling numpy_nonzero on a meta tensor raises RuntimeError
        # mentioning data-dependent shape.
        x = torch.randn(10, device="meta")
        with self.assertRaisesRegex(RuntimeError, "data-dependent shape"):
            numpy_nonzero(x)

    def test_basic_make_fx(self):
        # [FUNCTION DOC] Sanity check that make_fx with symbolic tracing of a legacy op with a fake
        # kernel produces a graph whose code mentions the op.
        # More serious tests are in our CustomOp opinfo db,
        # this one is just a sanity check.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        @torch.library.register_fake(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
        def foo_meta(x):
            # [FUNCTION DOC] HELPER: Fake kernel returning the sum of x.
            return x.sum()

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        gm = make_fx(op, tracing_mode="symbolic")(x)
        self.assertTrue(f"{TestCustomOp.test_ns}.foo" in gm.code)

    def test_not_implemented_error(self):
        # [FUNCTION DOC] Verifies error messages for an op with no impls: calling on CPU raises
        # NotImplementedError about a missing cpu impl; on meta, about no fake impl or Meta kernel;
        # an op without tensor inputs raises about no Tensor inputs.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration stub.
            raise NotImplementedError

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(NotImplementedError, "cpu impl registered"):
            op(x)

        x = torch.randn(3, device="meta")
        with self.assertRaisesRegex(NotImplementedError, "no fake impl or Meta kernel"):
            op(x)

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::bar")
        def bar(sizes: Sequence[int]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Declaration taking only a sequence of ints.
            raise NotImplementedError

        op = self.get_op(f"{self.test_ns}::bar")
        with self.assertRaisesRegex(NotImplementedError, "no Tensor inputs"):
            op((1, 2, 3))

    def test_data_dependent_basic(self):
        # [FUNCTION DOC] Traces numpy_nonzero with make_fx in symbolic mode and checks the
        # generated code mentions nonzero.
        x = torch.randn(5, 5)
        gm = make_fx(numpy_nonzero, tracing_mode="symbolic")(x)
        self.assertTrue("nonzero" in gm.code)

    def test_data_dependent_fake_tracing(self):
        # [FUNCTION DOC] Traces numpy_nonzero with make_fx in fake mode; passes if no error
        # is raised.
        x = torch.randn(5, 5)
        # We've updated to attempt to use unbacked symints even for fake
        # tracing
        make_fx(numpy_nonzero, tracing_mode="fake")(x)

    def test_symints(self):
        # [FUNCTION DOC] Traces numpy_view_copy with symbolic make_fx, checks the graph result
        # equals the eager call and compares the generated code against an expected inline string
        # that uses symbolic sizes of each dimension.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls the numpy_view_copy test op with x and its shape.
            return torch.ops._torch_testing.numpy_view_copy(x, x.shape)

        x = torch.randn(2, 3, 4)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        result = gm(x)
        self.assertEqual(result, f(x))
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(x_1, 2)
    numpy_view_copy = torch.ops._torch_testing.numpy_view_copy.default(x_1, [sym_size_int, sym_size_int_1, sym_size_int_2]);  x_1 = sym_size_int = sym_size_int_1 = sym_size_int_2 = None
    return numpy_view_copy""",
        )

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work on windows")
    def test_data_dependent_compile(self):
        # [FUNCTION DOC] Compiles a function calling numpy_nonzero with a counting backend and
        # checks exactly one graph break is recorded, once, with an expected message about a
        # dynamic shape operator. Not run on Windows.
        import torch._dynamo.testing
        from torch._dynamo.utils import counters

        counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function cloning the input, calling numpy_nonzero
            # and cloning.
            return numpy_nonzero(x.clone()).clone()

        f(torch.randn(10))

        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(next(iter(counters["graph_break"].values())), 1)
        self.assertExpectedInline(
            next(iter(counters["graph_break"].keys())).replace(";", "\n"),
            """\
Dynamic shape operator
  Explanation: Operator `_torch_testing.numpy_nonzero.default`'s output shape depends on input Tensor data.
  Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`

  Developer debug context: _torch_testing.numpy_nonzero.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0036.html""",
        )

    # pre-existing problem: torch.compile(dynamic=True) will, by default,
    # graph break on data-dependent operations. Eventually we'll make it so
    # that it never graph breaks on data-dependent operations.
    @unittest.expectedFailure
    def test_data_dependent_nms_dynamic_compile(self):
        # [FUNCTION DOC] Expected failure: compiling the numpy_nms op with dynamic shapes is meant
        # to produce zero graph breaks but currently breaks.
        import torch._dynamo.testing
        from torch._dynamo.utils import counters

        counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, dynamic=True)
        def f(x, s, i):
            # [FUNCTION DOC] HELPER: Compiled function calling the numpy_nms test op on a clone
            # of x.
            return torch.ops._torch_testing.numpy_nms(x.clone(), s, i).clone()

        f(torch.randn(20, 4), torch.randn(20), 0.1)

        self.assertEqual(len(counters["graph_break"]), 0)

    def test_impl_on_existing_op(self):
        # [FUNCTION DOC] Verifies torch._custom_ops.impl can attach a CPU impl to an op defined
        # through a Library; the result equals the sine of the input.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        @torch._custom_ops.impl(qualname)
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x.
            return x.sin()

        op = self.get_op(qualname)
        x = torch.randn(3)
        result = op(x)
        self.assertEqual(result, x.sin())

    @parametrize(
        "key", ["CPU", "CUDA", "CompositeImplicitAutograd", "CompositeExplicitAutograd"]
    )
    def test_impl_on_existing_op_with_cpu_registration(self, key):
        # [FUNCTION DOC] Parametrized over dispatch keys: with a kernel already registered on the
        # key, registering another impl through custom_ops.impl raises RuntimeError that an
        # implementation already exists.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Kernel returning the sine of x.
            return x.sin()

        lib.impl("foo", foo_impl, key)
        op = self.get_op(qualname)

        with self.assertRaisesRegex(RuntimeError, "already has an implementation"):
            custom_ops.impl(qualname, func=foo_impl)

    def test_abstract_impl_on_existing_op(self):
        # [FUNCTION DOC] Registers a fake kernel on a Library-defined op and checks that under
        # FakeTensorMode the result has the same shape and strides as the input.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        @torch.library.register_fake(qualname, lib=self.lib())
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Fake kernel returning the sine of x.
            return x.sin()

        op = self.get_op(qualname)
        with torch._subclasses.FakeTensorMode():
            x = torch.randn(3)
            result = op(x)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.stride(), x.stride())

    def test_abstract_impl_on_existing_op_with_meta(self):
        # [FUNCTION DOC] Verifies registering a fake kernel with allow_override disabled on an op
        # that already has a Meta kernel raises RuntimeError.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Kernel returning the sine of x.
            return x.sin()

        lib.impl("foo", foo_impl, "Meta")
        op = self.get_op(qualname)

        with self.assertRaisesRegex(RuntimeError, r"already has .*Meta implementation"):
            torch.library.register_fake(
                qualname, foo_impl, lib=self.lib(), allow_override=False
            )

    def test_abstract_impl_on_existing_op_with_CompositeImplicitAutograd(self):
        # [FUNCTION DOC] Verifies registering a fake kernel on an op with a
        # CompositeImplicitAutograd kernel raises RuntimeError mentioning that key.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Kernel returning the sine of x.
            return x.sin()

        lib.impl("foo", foo_impl, "CompositeImplicitAutograd")
        op = self.get_op(qualname)

        with self.assertRaisesRegex(RuntimeError, "CompositeImplicitAutograd"):
            torch.library.register_fake(qualname, foo_impl, lib=self.lib())

    def test_register_fake_does_not_leak_on_failure(self):
        # [FUNCTION DOC] Regression test: a failed register_fake (a composite kernel exists and
        # override is allowed) must raise RuntimeError and leave the process-wide fake impl holder
        # with no stored kernels, both before and after the failure.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Kernel returning the sine of x.
            return x.sin()

        # CompositeImplicitAutograd kernel makes lib.impl(..., "Meta", ...)
        # inside FakeImplHolder.register raise *after* the kernel has been
        # appended to the holder's singleton kernel list.
        lib.impl("foo", foo_impl, "CompositeImplicitAutograd")

        entry = torch._library.simple_registry.singleton.find(qualname)
        self.assertEqual(len(entry.fake_impl.kernels), 0)

        # allow_override=True bypasses the early guard and exercises the
        # exception-safety path in FakeImplHolder.register.
        with self.assertRaisesRegex(RuntimeError, "CompositeImplicitAutograd"):
            torch.library.register_fake(
                qualname, foo_impl, lib=self.lib(), allow_override=True
            )

        # The failed registration must not leave a dangling kernel in the
        # process-wide FakeImplHolder; otherwise teardown leaks it and
        # corrupts subsequent register_fake() calls for the same qualname.
        self.assertEqual(len(entry.fake_impl.kernels), 0)

    def test_abstract_impl_on_existing_op_with_CompositeExplicitAutograd(self):
        # [FUNCTION DOC] Verifies a fake kernel can be registered on an op with a
        # CompositeExplicitAutograd kernel; under FakeTensorMode the result has a scalar shape.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Kernel returning the sine of x.
            return x.sin()

        lib.impl("foo", foo_impl, "CompositeExplicitAutograd")
        op = self.get_op(qualname)

        torch.library.register_fake(qualname, lambda x: x.sum(), lib=self.lib())
        with torch._subclasses.FakeTensorMode():
            x = torch.randn(10)
            result = op(x)
            self.assertEqual(result.shape, ())

    def _test_backward_impl_raises(self, qualname, err_regex):
        # [FUNCTION DOC] HELPER: Asserts both impl_save_for_backward and impl_backward raise
        # RuntimeError matching err_regex when applied to the named existing op.
        with self.assertRaisesRegex(RuntimeError, err_regex):

            @custom_ops.impl_save_for_backward(qualname)
            def foo2(x):
                # [FUNCTION DOC] HELPER: Dummy save_for_backward function; registration is expected
                # to fail.
                return

        with self.assertRaisesRegex(RuntimeError, err_regex):

            @custom_ops.impl_backward(qualname)
            def foo3(x):
                # [FUNCTION DOC] HELPER: Dummy backward function; registration is expected to fail.
                return

    def test_backward_impl_on_existing_op_incorrect_schema_views(self):
        # [FUNCTION DOC] Backward registration on an op returning a view is rejected (error
        # mentions operators that return views).
        lib = self.lib()
        lib.define("foo(Tensor(a) x) -> Tensor(a)")
        qualname = f"{self.test_ns}::foo"
        self._test_backward_impl_raises(qualname, "operator that returns views")

    def test_backward_impl_on_existing_op_incorrect_schema_mutable(self):
        # [FUNCTION DOC] Backward registration on an op with a mutable argument is rejected (error
        # says non-functional).
        lib = self.lib()
        lib.define("foo(Tensor(a!) x) -> Tensor")
        qualname = f"{self.test_ns}::foo"
        self._test_backward_impl_raises(qualname, "non-functional")

    def test_backward_impl_on_existing_op_incorrect_schema_no_output(self):
        # [FUNCTION DOC] Backward registration on an op with no returns is rejected (error says
        # no returns).
        lib = self.lib()
        lib.define("foo(Tensor x) -> ()")
        qualname = f"{self.test_ns}::foo"
        self._test_backward_impl_raises(qualname, "no returns")

    def test_backward_impl_on_existing_op_CompositeImplicitAutograd(self):
        # [FUNCTION DOC] Backward registration on an op with a CompositeImplicitAutograd kernel is
        # rejected, with the key named in the error.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"
        lib.impl("foo", lambda x: x.sin().cos(), "CompositeImplicitAutograd")
        self._test_backward_impl_raises(qualname, "CompositeImplicitAutograd")

    @parametrize("key", ["Autograd", "AutogradCPU", "AutogradCUDA"])
    def test_backward_impl_on_existing_op_with_key(self, key):
        # [FUNCTION DOC] Parametrized over autograd keys: backward registration on an op that
        # already has a kernel on the key is rejected, with the key named in the error.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"
        lib.impl("foo", lambda x: x.sin().cos(), key)
        self._test_backward_impl_raises(qualname, key)

    def test_is_functional_schema(self):
        # [FUNCTION DOC] Checks torch._library.utils.is_functional_schema for several schema
        # strings, passing each as a string, a torchgen FunctionSchema and a parsed JIT schema; all
        # three must give the expected boolean.
        tests = {
            "foo(Tensor x) -> Tensor": True,
            "foo(Tensor(a) x) -> Tensor": True,
            "foo(Tensor(a!) x) -> Tensor": False,
            "foo(Tensor(a) x) -> Tensor(a)": False,
            "foo(Tensor x) -> ()": False,
        }
        for schema_str, expected in tests.items():
            res = torch._library.utils.is_functional_schema(schema_str)
            self.assertEqual(res, expected)

            from torchgen.model import FunctionSchema

            schema = FunctionSchema.parse(schema_str)
            res = torch._library.utils.is_functional_schema(schema)
            self.assertEqual(res, expected)

            schema = torch._C.parse_schema(schema_str)
            res = torch._library.utils.is_functional_schema(schema)
            self.assertEqual(res, expected)

    def test_incorrect_schema_types(self):
        # [FUNCTION DOC] Checks Library.define raises RuntimeError for unknown type specifiers and
        # for int64_t and double (with hints to use SymInt/int and float), and that a registered
        # opaque type name is accepted. Registers and unregisters an opaque type.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            with self.assertRaisesRegex(RuntimeError, "unknown type specifier"):
                lib.define("foo12(Tensor a) -> asdfasdf")
            with self.assertRaisesRegex(RuntimeError, "unknown type specifier"):
                lib.define("foo12(asdf a) -> Tensor")
            with self.assertRaisesRegex(RuntimeError, "Use `SymInt` or `int`"):
                lib.define("foo12(int64_t a) -> Tensor")
            with self.assertRaisesRegex(RuntimeError, "Use `float`"):
                lib.define("foo12(double a) -> Tensor")
            with self.assertRaisesRegex(RuntimeError, "unknown type specifier"):
                lib.define("foo12(PyObject a) -> Tensor")
            type_name = "torch.testing.OpaqueObjectForCustomOpsTest"
            if torch._C._is_opaque_type_registered(type_name):
                torch._C._unregister_opaque_type(type_name)
            torch._C._register_opaque_type(type_name)
            try:
                lib.define(f"foo12({type_name} a) -> Tensor")
            finally:
                torch._C._unregister_opaque_type(type_name)

    def test_is_tensorlist_like_type(self):
        # [FUNCTION DOC] Checks torch._library.utils.is_tensorlist_like_type is true for several
        # tensor-list types (also optional forms) and false for a plain Tensor and an int list.
        tensorlists = [
            # Tensor[]
            torch.ops.aten.where.default._schema.returns[0].type,
            # Tensor?[]
            torch.ops.aten.index.Tensor._schema.arguments[1].type,
            # Tensor[]?
            torch._C.parse_schema("foo(Tensor[]? x) -> ()").arguments[0].type,
            # Tensor?[]?
            torch._C.parse_schema("foo(Tensor?[]? x) -> ()").arguments[0].type,
        ]
        non_tensorlists = [
            # Tensor
            torch.ops.aten.sin.default._schema.arguments[0].type,
            # IntList
            torch.ops.aten.sum.dim_IntList._schema.arguments[1].type,
        ]
        for a in tensorlists:
            self.assertTrue(torch._library.utils.is_tensorlist_like_type(a))
        for a in non_tensorlists:
            self.assertFalse(torch._library.utils.is_tensorlist_like_type(a))

    def test_backward_impl_on_existing_op(self):
        # [FUNCTION DOC] Registers impl, save_for_backward and backward on an op defined in a
        # Library and verifies the gradient computed by autograd equals the cosine of the input.
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        @custom_ops.impl(qualname)
        def foo_impl(x):
            # [FUNCTION DOC] HELPER: Impl returning the sine of x under no_grad.
            with torch.no_grad():
                return x.sin()

        @custom_ops.impl_save_for_backward(qualname)
        def foo_save_for_backward(inputs, output):
            # [FUNCTION DOC] HELPER: Saves the x input.
            return inputs.x

        @custom_ops.impl_backward(qualname)
        def foo_backward(ctx, saved, grad_out):
            # [FUNCTION DOC] HELPER: Backward returning a dict mapping x to the incoming grad
            # times cosine.
            return {"x": grad_out * saved.cos()}

        op = self.get_op(qualname)
        x = torch.randn([], requires_grad=True)
        y = op(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, x.cos())

    @parametrize(
        "tags",
        [
            subtest(torch.Tag.pointwise, "single"),
            subtest((torch.Tag.pointwise,), "tuple"),
            subtest([torch.Tag.pointwise], "list"),
        ],
    )
    def test_define_with_tags(self, tags):
        # [FUNCTION DOC] Defines an op with torch.library.define passing tags and checks the op's
        # tags attribute is a list equal to the tags.
        # KNOWN ISSUE: the parametrized tags argument is overwritten at the start with a tuple
        # holding the pointwise tag, so the three parameter forms are not distinguished.
        lib = self.lib()
        tags = (torch.Tag.pointwise,)
        torch.library.define(
            f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib, tags=tags
        )
        actual = self.ns().foo.default.tags
        self.assertTrue(isinstance(actual, list))
        self.assertEqual(actual, list(tags))

    def test_builtin_aten_ops_are_pt2_compliant(self):
        # [FUNCTION DOC] Checks the sin and sum builtin ops carry the pt2_compliant tag.
        for op in [torch.ops.aten.sin.default, torch.ops.aten.sum.dim_IntList]:
            self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    def test_builtin_torchscript_ops(self):
        # [FUNCTION DOC] Checks the builtin complex sub and mul overloads carry the
        # pt2_compliant tag.
        for op in [torch.ops.aten.sub.complex, torch.ops.aten.mul.complex]:
            self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    def test_autogen_aten_ops_are_pt2_compliant(self):
        # [FUNCTION DOC] Checks the autogenerated fill out overload carries both the generated and
        # pt2_compliant tags.
        for op in [torch.ops.aten.fill.Tensor_out]:
            self.assertIn(torch.Tag.generated, op.tags)
            self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    def test_resolve_packet(self):
        # [FUNCTION DOC] Checks _jit_resolve_packet picks the default overload for sum, the
        # dim_IntList overload when dim is passed, and raises RuntimeError when no schema matches.
        x = torch.randn(3)
        result = torch._C._jit_resolve_packet("aten::sum", x)
        self.assertEqual(result, "default")

        result = torch._C._jit_resolve_packet("aten::sum", x, dim=1)
        self.assertEqual(result, "dim_IntList")

        with self.assertRaisesRegex(RuntimeError, "failed to match any schema"):
            result = torch._C._jit_resolve_packet("aten::sum", x, x, x)

    def test_define_bad_schema(self):
        # [FUNCTION DOC] Verifies torch.library.define with a name-containing schema raises
        # ValueError saying the schema should look like an argument list.
        lib = self.lib()
        with self.assertRaisesRegex(ValueError, "expected schema to look like"):
            torch.library.define(f"{self.test_ns}::foo", "foo(Tensor x) -> Tensor")

    def test_define_and_impl(self):
        # [FUNCTION DOC] Defines an op and registers a CPU impl through the torch.library.impl
        # decorator using numpy, then verifies the output equals the sine of the input.
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        @torch.library.impl(f"{self.test_ns}::foo", "CPU", lib=lib)
        def f(x):
            # [FUNCTION DOC] HELPER: CPU impl computing the sine via numpy.
            return torch.from_numpy(np.sin(x.numpy()))

        x = torch.randn(3)
        y = self.ns().foo(x)
        if not torch.allclose(y, x.sin()):
            raise AssertionError("expected y to equal x.sin()")

    def test_define_validation(self):
        # [FUNCTION DOC] Verifies torch.library.define with a name lacking a namespace
        # raises ValueError.
        with self.assertRaisesRegex(ValueError, "namespace"):
            torch.library.define("foo", "(Tensor x) -> Tensor")

    def test_legacy_define(self):
        # [FUNCTION DOC] Uses the legacy form of torch.library.define as a decorator taking a
        # Library and full schema string, then checks the op output equals the sine of the input.
        lib = self.lib()

        @torch.library.define(lib, "foo(Tensor x) -> Tensor")
        def f(x):
            # [FUNCTION DOC] HELPER: Impl computing the sine via numpy.
            return torch.from_numpy(np.sin(x.numpy()))

        x = torch.randn(3)
        y = self.ns().foo(x)
        if not torch.allclose(y, x.sin()):
            raise AssertionError("expected y to equal x.sin()")

    def test_impl_function(self):
        # [FUNCTION DOC] Defines an op and registers a CPU impl by calling torch.library.impl as a
        # function (not decorator), then checks the output equals the sine of the input.
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        def f(x):
            # [FUNCTION DOC] HELPER: Impl computing the sine via numpy.
            return torch.from_numpy(np.sin(x.numpy()))

        torch.library.impl(f"{self.test_ns}::foo", "CPU", f, lib=lib)
        x = torch.randn(3)
        y = self.ns().foo(x)
        if not torch.allclose(y, x.sin()):
            raise AssertionError("expected y to equal x.sin()")

    def test_legacy_impl(self):
        # [FUNCTION DOC] Uses the legacy impl decorator form taking a Library, op name and key,
        # then checks the output equals the sine of the input.
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        @torch.library.impl(lib, "foo", "CPU")
        def f(x):
            # [FUNCTION DOC] HELPER: Impl computing the sine via numpy.
            return torch.from_numpy(np.sin(x.numpy()))

        x = torch.randn(3)
        y = self.ns().foo(x)
        if not torch.allclose(y, x.sin()):
            raise AssertionError("expected y to equal x.sin()")

    def test_defined_in_python(self):
        # [FUNCTION DOC] Checks the _defined_in_python flag is false for builtin ops and true for
        # ops defined through torch.library.define, including one with an overload name.
        # KNOWN ISSUE: the qualified names passed to define are plain strings without an f prefix,
        # so the braces are literal text and the name does not use the test namespace.
        self.assertFalse(torch.ops.aten.sin.default._defined_in_python)
        self.assertFalse(torch.ops.aten.sum.dim_IntList._defined_in_python)

        lib = self.lib()
        torch.library.define("{self._test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)
        ns = self.ns()
        self.assertTrue(ns.foo.default._defined_in_python)

        torch.library.define(
            "{self._test_ns}::bar.overload", "(Tensor x) -> Tensor", lib=lib
        )
        self.assertTrue(ns.bar.overload._defined_in_python)

    def _test_impl_device(self, name, types, device):
        # [FUNCTION DOC] HELPER: Defines an op with the given name, registers an impl for the given
        # device types that computes the sine through numpy and moves the result to the input
        # device, then checks the output on the given device equals the sine of the input.
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::{name}", "(Tensor x) -> Tensor", lib=lib)

        @torch.library.impl(f"{self.test_ns}::{name}", types)
        def f(x):
            # [FUNCTION DOC] HELPER: Impl computing the sine via numpy on a CPU copy and
            # moving back.
            x_np = x.cpu().numpy()
            y = torch.from_numpy(np.sin(x_np))
            return y.to(device=x.device)

        x = torch.randn(3, device=device)
        y = getattr(self.ns(), name)(x)
        if not torch.allclose(y, x.sin()):
            raise AssertionError("expected y to equal x.sin()")

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_impl_device_cpu(self):
        # [FUNCTION DOC] Runs the device impl helper with CPU on a default, a list and a mixed CPU
        # and CUDA device type spec. Skipped under Dynamo.
        self._test_impl_device("foo1", "default", "cpu")
        self._test_impl_device("foo2", ["cpu"], "cpu")
        self._test_impl_device("foo3", ["cpu", "cuda"], "cpu")

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires cuda or xpu")
    def test_impl_device_cuda(self):
        # [FUNCTION DOC] Runs the device impl helper on the accelerator for default, list and mixed
        # specs. Skipped under Dynamo and without CUDA or XPU.
        self._test_impl_device("foo4", "default", device_type)
        self._test_impl_device("foo5", [device_type], device_type)
        self._test_impl_device("foo6", ["cpu", device_type], device_type)

    def test_impl_device_function(self):
        # [FUNCTION DOC] Registers an impl for the default device type by calling
        # torch.library.impl as a function and checks the output equals the sine of the input.
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        def f(x):
            # [FUNCTION DOC] HELPER: Impl computing the sine via numpy on a CPU copy and
            # moving back.
            x_np = x.cpu().numpy()
            y = torch.from_numpy(np.sin(x_np))
            return y.to(device=x.device)

        torch.library.impl(f"{self.test_ns}::foo", "default", f, lib=lib)
        x = torch.randn(3)
        y = self.ns().foo(x)
        if not torch.allclose(y, x.sin()):
            raise AssertionError("expected y to equal x.sin()")

    def test_impl_device_invalid(self):
        # [FUNCTION DOC] Verifies an invalid device type string given to torch.library.impl raises
        # RuntimeError listing the expected device names.
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu, cuda"):
            torch.library.impl("blah::blah", "somethingsomething")

    def test_override_impl(self):
        # [FUNCTION DOC] Verifies a CPU kernel can be overridden: registering a second one with
        # allow_override disabled raises RuntimeError, and the default overrides it (output changes
        # from three to six elements).
        lib = self.lib()
        op_name = f"{self.test_ns}::foo"
        torch.library.define(op_name, "(Tensor x) -> Tensor", lib=lib)
        op = self.ns().foo.default

        def foo_impl1(x):
            # [FUNCTION DOC] HELPER: Returns x times one.
            return x * 1

        # Register cpu impl to foo_impl1
        lib.impl("foo", foo_impl1, "CPU")
        self.assertEqual(op(torch.ones(3)), torch.ones(3))

        def foo_impl2(x):
            # [FUNCTION DOC] HELPER: Returns x concatenated with itself.
            return torch.cat([x, x])

        with self.assertRaisesRegex(RuntimeError, "already a kernel registered"):
            lib.impl("foo", foo_impl2, "CPU", allow_override=False)

        # Override cpu impl to foo_impl2
        lib.impl(op_name, foo_impl2, "CPU")
        self.assertEqual(op(torch.ones(3)), torch.ones(6))

    def test_override_fake(self):
        # [FUNCTION DOC] Verifies fake kernel override behavior: a second register_fake with
        # allow_override disabled raises RuntimeError; the default replaces it; a scoped library
        # temporarily registers another fake that is dropped on exit, restoring the previous one.
        # Checks shapes under FakeTensorMode and on meta tensors.
        lib = self.lib()
        op_name = f"{self.test_ns}::foo"
        torch.library.define(op_name, "(Tensor x) -> Tensor", lib=lib)
        op = self.ns().foo.default

        def foo_impl1(x):
            # [FUNCTION DOC] HELPER: Returns x times one.
            return x * 1

        # Register fake kernel to foo_impl1
        torch.library.register_fake(op_name, foo_impl1, lib=lib)

        with torch._subclasses.FakeTensorMode():
            self.assertEqual(op(torch.ones(3)).shape, [3])
        self.assertEqual(op(torch.ones(3, device="meta")).shape, [3])

        def foo_impl2(x):
            # [FUNCTION DOC] HELPER: Returns x concatenated with itself.
            return torch.cat([x, x])

        with self.assertRaisesRegex(RuntimeError, "already has a fake impl"):
            torch.library.register_fake(
                op_name, foo_impl2, lib=lib, allow_override=False
            )

        # Override fake kernel to foo_impl2
        torch.library.register_fake(op_name, foo_impl2, lib=lib)
        with torch._subclasses.FakeTensorMode():
            self.assertEqual(op(torch.ones(3)).shape, [6])
        self.assertEqual(op(torch.ones(3, device="meta")).shape, [6])

        # Use scoped_library to temporarily register Fake kernel to foo_impl1
        with torch.library._scoped_library(self.test_ns, "FRAGMENT") as lib2:
            torch.library.register_fake(op_name, foo_impl1, lib=lib2)
            with torch._subclasses.FakeTensorMode():
                self.assertEqual(op(torch.ones(3)).shape, [3])
            self.assertEqual(op(torch.ones(3, device="meta")).shape, [3])

        # Fake kernel should go back to foo_impl2
        with torch._subclasses.FakeTensorMode():
            self.assertEqual(op(torch.ones(3)).shape, [6])
        self.assertEqual(op(torch.ones(3, device="meta")).shape, [6])

    def test_override_meta(self):
        # [FUNCTION DOC] Verifies Meta kernel override behavior: allow_override disabled raises
        # RuntimeError, default replaces the kernel, and a scoped library registration is
        # temporary, after which the previous kernel returns.
        lib = self.lib()
        op_name = f"{self.test_ns}::foo"
        torch.library.define(op_name, "(Tensor x) -> Tensor", lib=lib)
        op = self.ns().foo.default

        def foo_impl1(x):
            # [FUNCTION DOC] HELPER: Returns x times one.
            return x * 1

        # Register Meta kernel to foo_impl1
        lib.impl("foo", foo_impl1, "Meta")
        self.assertEqual(op(torch.ones(3, device="meta")).shape, [3])

        def foo_impl2(x):
            # [FUNCTION DOC] HELPER: Returns x concatenated with itself.
            return torch.cat([x, x])

        with self.assertRaisesRegex(RuntimeError, "already a kernel registered"):
            lib.impl("foo", foo_impl2, "Meta", allow_override=False)

        # Override Meta kernel to foo_impl2
        lib.impl("foo", foo_impl2, "Meta")
        self.assertEqual(op(torch.ones(3, device="meta")).shape, [6])

        # Use scoped_library to temporarily register Meta kernel to foo_impl1
        with torch.library._scoped_library(self.test_ns, "FRAGMENT") as lib2:
            lib2.impl("foo", foo_impl1, "Meta")
            self.assertEqual(op(torch.ones(3, device="meta")).shape, [3])

        # Meta kernel should go back to foo_impl2
        self.assertEqual(op(torch.ones(3, device="meta")).shape, [6])

        # Use register_fake to override Meta kernel to foo_impl1
        torch.library.register_fake(op_name, foo_impl1, lib=lib)
        self.assertEqual(op(torch.ones(3, device="meta")).shape, [3])

    def test_override_cea(self):
        # [FUNCTION DOC] Registers a CompositeExplicitAutograd kernel (identity-like), checks
        # output shapes under fake and meta, then registers a fake kernel that doubles the size and
        # checks both fake and meta paths now return the larger shape.
        lib = self.lib()
        op_name = f"{self.test_ns}::foo"
        torch.library.define(op_name, "(Tensor x) -> Tensor", lib=lib)
        op = self.ns().foo.default

        def foo_impl1(x):
            # [FUNCTION DOC] HELPER: Returns x times one.
            return x * 1

        # Register CEA impl to foo_impl1
        lib.impl("foo", foo_impl1, "CompositeExplicitAutograd")
        with torch._subclasses.FakeTensorMode():
            self.assertEqual(op(torch.ones(3)).shape, [3])
        self.assertEqual(op(torch.ones(3, device="meta")).shape, [3])

        def foo_impl2(x):
            # [FUNCTION DOC] HELPER: Returns x concatenated with itself.
            return torch.cat([x, x])

        # Override Meta/fake kernel with foo_impl2
        torch.library.register_fake(op_name, foo_impl2, lib=lib)
        with torch._subclasses.FakeTensorMode():
            self.assertEqual(op(torch.ones(3)).shape, [6])
        self.assertEqual(op(torch.ones(3, device="meta")).shape, [6])

    @scoped_load_inline
    def test_autograd_function_backed_op(self, load_inline):
        # [FUNCTION DOC] Compiles inline C++ that defines an op backed by a traceable autograd
        # Function (identity forward, pass-through backward), calls it on a tensor requiring grad
        # and verifies backward yields a gradient of ones.
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

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

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_autograd_function_backed_op, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = load_inline(
            name="test_autograd_function_backed_op",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        x = torch.ones(2, 2, requires_grad=True)
        temp = x.detach().clone()
        out = (
            torch.ops.test_autograd_function_backed_op.custom_op_backed_by_autograd_fn(
                x
            )
        )
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad, temp)

    # Using a non-existent DSO is a quick way to trigger an OSError,
    # which can be used to not break BC.
    def test_load_library(self):
        # [FUNCTION DOC] Verifies torch.ops.load_library on a nonexistent shared library raises
        # OSError with a message about being unable to load it.
        with self.assertRaisesRegex(
            OSError, "Could not load this library: .*libnoexist.so"
        ):
            torch.ops.load_library("libnoexist.so")

    def test_list_scalar_type(self):
        # [FUNCTION DOC] Defines an op with a list-of-ScalarType argument and a CPU impl that
        # stores what it receives, then checks the dtypes arrive intact.
        lib = self.lib()
        lib.define("scalar_list(Tensor x, ScalarType[] dts) -> Tensor")

        received = None

        @torch.library.impl(lib, "scalar_list", "CPU")
        def _(x, dts):
            # [FUNCTION DOC] HELPER: CPU impl recording dts in the enclosing variable and returning
            # a clone.
            nonlocal received
            received = dts
            return x.clone()

        x = torch.randn(3)
        torch.ops._test_custom_op.scalar_list(x, [torch.float32, torch.bfloat16])
        self.assertEqual(received, [torch.float32, torch.bfloat16])

    def test_list_layout(self):
        # [FUNCTION DOC] Defines an op with a list-of-Layout argument and checks a CPU impl
        # receives the layouts (strided and sparse COO) unchanged.
        lib = self.lib()
        lib.define("layout_list(Tensor x, Layout[] layouts) -> Tensor")

        received = None

        @torch.library.impl(lib, "layout_list", "CPU")
        def _(x, layouts):
            # [FUNCTION DOC] HELPER: CPU impl recording layouts in the enclosing variable and
            # returning a clone.
            nonlocal received
            received = layouts
            return x.clone()

        x = torch.randn(3)
        torch.ops._test_custom_op.layout_list(x, [torch.strided, torch.sparse_coo])
        self.assertEqual(received, [torch.strided, torch.sparse_coo])

    def test_list_memory_format(self):
        # [FUNCTION DOC] Defines an op with a list-of-MemoryFormat argument and checks a CPU impl
        # receives the formats (contiguous and channels last) unchanged.
        lib = self.lib()
        lib.define("memfmt_list(Tensor x, MemoryFormat[] fmts) -> Tensor")

        received = None

        @torch.library.impl(lib, "memfmt_list", "CPU")
        def _(x, fmts):
            # [FUNCTION DOC] HELPER: CPU impl recording fmts in the enclosing variable and
            # returning a clone.
            nonlocal received
            received = fmts
            return x.clone()

        x = torch.randn(3)
        torch.ops._test_custom_op.memfmt_list(
            x, [torch.contiguous_format, torch.channels_last]
        )
        self.assertEqual(received, [torch.contiguous_format, torch.channels_last])


def op_with_incorrect_schema(testcase, name):
    # [FUNCTION DOC] HELPER: Defines in the test case's library an op with the given name whose
    # CompositeExplicitAutograd kernel returns a slice of x (a view), despite a non-aliasing
    # schema, and returns the op looked up through the test case.
    lib = testcase.lib()
    lib.define(f"{name}(Tensor x) -> Tensor")
    qualname = f"{testcase.test_ns}::{name}"
    lib.impl(name, lambda x: x[:], "CompositeExplicitAutograd")
    return testcase.get_op(qualname)


class MiniOpTest(CustomOpTestCaseBase):
    # [CLASS DOC] Tests running ops (builtin and deliberately flawed Library ops) in the namespace
    # mini_op_test; setUp builds two ops, one lacking a fake impl and one whose autograd backward
    # always raises NotImplementedError. Likely used by the opcheck test generator for the
    # dontGenerate marker.
    test_ns = "mini_op_test"

    def _init_op_delayed_backward_error(self):
        # [FUNCTION DOC] HELPER: Defines op delayed_error: a CompositeExplicitAutograd clone kernel
        # plus an Autograd kernel wrapping a function whose backward raises NotImplementedError.
        # Returns the op.
        name = "delayed_error"
        qualname = f"{self.test_ns}::{name}"
        lib = self.lib()
        lib.define(f"{name}(Tensor x) -> Tensor")
        lib.impl(name, lambda x: x.clone(), "CompositeExplicitAutograd")
        op = self.get_op(qualname)

        class Op(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function calling the op below autograd; backward
            # raises NotImplementedError.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Runs the op under an autograd-below guard.
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Always raises NotImplementedError.
                raise NotImplementedError

        def autograd_impl(x):
            # [FUNCTION DOC] HELPER: Autograd key kernel applying the autograd function.
            return Op.apply(x)

        lib.impl(name, autograd_impl, "Autograd")
        return op

    def _init_op_with_no_abstract_impl(self):
        # [FUNCTION DOC] HELPER: Defines op no_abstract with the pt2_compliant tag and only a CPU
        # clone kernel (no fake or Meta), and returns the op found by lookup_op.
        name = "no_abstract"
        qualname = f"{self.test_ns}::{name}"
        lib = self.lib()
        lib.define(f"{name}(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,))
        lib.impl(name, lambda x: x.clone(), "CPU")
        return torch._library.utils.lookup_op(qualname)

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs base setUp and stores the two ops created by the
        # init helpers.
        super().setUp()
        self._op_with_no_abstract_impl = self._init_op_with_no_abstract_impl()
        self._op_delayed_backward_error = self._init_op_delayed_backward_error()

    @optests.dontGenerateOpCheckTests("Testing this API")
    def test_dont_generate(self):
        # [FUNCTION DOC] Calls an op with an incorrect schema; the dontGenerateOpCheckTests marker
        # excludes it from generated opcheck tests. Passes if the call does not raise.
        op = op_with_incorrect_schema(self, "incorrect_schema")
        x = torch.randn(3)
        op(x)

    def test_mm(self):
        # [FUNCTION DOC] Checks aten mm gives the same result as the matmul operator.
        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(3, 5)
        result = torch.ops.aten.mm.default(x, y)
        self.assertEqual(result, x @ y)

    def test_mm_meta(self):
        # [FUNCTION DOC] Checks aten mm on meta tensors yields the expected output shape.
        x = torch.randn(2, 3, requires_grad=True, device="meta")
        y = torch.randn(3, 5, device="meta")
        result = torch.ops.aten.mm.default(x, y)
        self.assertEqual(result.shape, (x @ y).shape)

    def test_mm_fake(self):
        # [FUNCTION DOC] Checks aten mm under FakeTensorMode yields the expected output shape.
        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.randn(2, 3, requires_grad=True, device="cpu")
            y = torch.randn(3, 5, device="cpu")
            result = torch.ops.aten.mm.default(x, y)
            self.assertEqual(result.shape, (x @ y).shape)

    def test_mm_errors(self):
        # [FUNCTION DOC] Verifies aten mm with incompatible shapes raises RuntimeError that the
        # matrices cannot be multiplied.
        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(4, 5)
        with self.assertRaisesRegex(RuntimeError, "cannot be multiplied"):
            result = torch.ops.aten.mm.default(x, y)

    def test_nonzero(self):
        # [FUNCTION DOC] Checks aten nonzero returns the indices of nonzero elements as a column.
        x = torch.tensor([0, 1, 2, 0, 0])
        y = torch.ops.aten.nonzero.default(x)
        self.assertEqual(y, torch.tensor([[1], [2]]))

    def test_inplace(self):
        # [FUNCTION DOC] Checks the in-place aten sin_ modifies x to equal the sine of its
        # old value.
        x = torch.randn(3)
        x_clone = x.clone()
        y = torch.ops.aten.sin_(x)
        self.assertEqual(x, x_clone.sin())

    def test_incorrect_schema(self):
        # [FUNCTION DOC] Calls the op with an incorrect schema; passes if no error is raised.
        op = op_with_incorrect_schema(self, "incorrect_schema")
        x = torch.randn(3)
        op(x)

    def test_no_abstract(self):
        # [FUNCTION DOC] Calls the op that has no fake impl on a CPU tensor; passes if no error
        # is raised.
        op = self._op_with_no_abstract_impl
        x = torch.randn(3)
        op(x)

    def test_delayed_error(self):
        # [FUNCTION DOC] Verifies backward through the op whose autograd function raises
        # NotImplementedError propagates that exception.
        op = self._op_delayed_backward_error
        x = torch.randn([], requires_grad=True)
        y = op(x)
        with self.assertRaises(NotImplementedError):
            y.sum().backward()

    def test_delayed_error_no_requires_grad(self):
        # [FUNCTION DOC] Calls the delayed-error op on a tensor without requires_grad; passes since
        # the backward error is never triggered.
        op = self._op_delayed_backward_error
        x = torch.randn([])
        y = op(x)


class TestCustomOpAPI(TestCase):
    # [CLASS DOC] Tests for torch.library.custom_op: tags handling (including the inplace and out
    # tags and their validation), aliasing schema generation, fake mode behavior and compile
    # integration. Plain TestCase; ops are registered in the _torch_testing namespace.
    @parametrize(
        "tags, opname",
        [
            subtest((torch.Tag.pointwise, "single"), name="single"),
            subtest(([torch.Tag.pointwise], "list"), name="list"),
            subtest(((torch.Tag.pointwise,), "tuple"), name="tuple"),
        ],
    )
    def test_custom_op_tags(self, tags, opname):
        # [FUNCTION DOC] Parametrized over a single tag, a list and a tuple: creates a custom op
        # with the tags and checks the op's tags include both pt2_compliant and the given tag.
        @torch.library.custom_op(
            f"_torch_testing::tags_{opname}",
            mutates_args=(),
            tags=tags,
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning a clone of x.
            return x.clone()

        self.assertIn(torch.Tag.pt2_compliant_tag, f._opoverload.tags)
        self.assertIn(torch.Tag.pointwise, f._opoverload.tags)

    def test_custom_op_tags_dedup_pt2_compliant(self):
        # [FUNCTION DOC] Checks that passing the pt2_compliant tag explicitly does not duplicate it
        # in the op's tag list, and the other tag is kept.
        @torch.library.custom_op(
            "_torch_testing::tags_dedup_pt2",
            mutates_args=(),
            tags=[torch.Tag.pt2_compliant_tag, torch.Tag.pointwise],
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning a clone of x.
            return x.clone()

        tags = f._opoverload.tags
        self.assertEqual(tags.count(torch.Tag.pt2_compliant_tag), 1)
        self.assertIn(torch.Tag.pointwise, tags)

    def test_custom_op_inplace_out_tags_mutually_exclusive(self):
        # [FUNCTION DOC] Verifies that giving both the inplace and out tags raises AssertionError
        # saying they are mutually exclusive.
        def f(x: Tensor, *, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Function with a keyword-only out tensor, returning it.
            return out

        with self.assertRaisesRegex(AssertionError, "mutually exclusive"):
            torch.library.custom_op(
                "_torch_testing::inplace_out_tags_mutually_exclusive",
                f,
                mutates_args={"x", "out"},
                tags=[torch.Tag.inplace, torch.Tag.out],
            )

    def test_custom_op_inplace_tag(self):
        # [FUNCTION DOC] Tests the inplace tag: the schema marks the first argument and the return
        # as written with the same alias set, the op adds y into x and returns x itself (also under
        # FakeTensorMode, returning the fake input).
        @torch.library.custom_op(
            "_torch_testing::inplace_tag",
            mutates_args={"x"},
            tags=torch.Tag.inplace,
        )
        def f(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Adds y into x in place and returns x.
            x.add_(y)
            return x

        self.assertIn(torch.Tag.inplace, f._opoverload.tags)
        schema = f._opoverload._schema
        self.assertTrue(schema.arguments[0].alias_info.is_write)
        self.assertTrue(schema.returns[0].alias_info.is_write)
        self.assertEqual(
            schema.arguments[0].alias_info.before_set,
            schema.returns[0].alias_info.before_set,
        )

        x = torch.randn(3)
        y = torch.randn(3)
        expected = x + y
        result = f(x, y)
        self.assertIs(result, x)
        self.assertEqual(x, expected)

        with torch._subclasses.fake_tensor.FakeTensorMode():
            fake_x = torch.randn(3)
            fake_y = torch.randn(3)
            fake_result = f(fake_x, fake_y)
            self.assertIs(fake_result, fake_x)

    def test_custom_op_inplace_tag_returns_wrong_tensor(self):
        # [FUNCTION DOC] Verifies an inplace-tagged op whose body returns a different tensor raises
        # RuntimeError saying it must return its first argument.
        @torch.library.custom_op(
            "_torch_testing::inplace_tag_returns_wrong_tensor",
            mutates_args={"x"},
            tags=torch.Tag.inplace,
        )
        def f(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Adds y into x but returns y.
            x.add_(y)
            return y

        with self.assertRaisesRegex(RuntimeError, "must return its first argument"):
            f(torch.randn(3), torch.randn(3))

    def test_custom_op_inplace_tag_mutates_args_iterable(self):
        # [FUNCTION DOC] Verifies mutates_args can be a generator for an inplace-tagged op: the
        # schema marks argument and return as written, and calling returns the input itself.
        def mutates_args():
            # [FUNCTION DOC] HELPER: Generator yielding the single argument name x.
            yield "x"

        @torch.library.custom_op(
            "_torch_testing::inplace_tag_mutates_args_iterable",
            mutates_args=mutates_args(),
            tags=torch.Tag.inplace,
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns x.
            return x

        schema = f._opoverload._schema
        self.assertTrue(schema.arguments[0].alias_info.is_write)
        self.assertTrue(schema.returns[0].alias_info.is_write)
        self.assertEqual(
            schema.arguments[0].alias_info.before_set,
            schema.returns[0].alias_info.before_set,
        )

        x = torch.randn(3)
        self.assertIs(f(x), x)

    def test_custom_op_inplace_tag_autograd(self):
        # [FUNCTION DOC] Verifies an inplace-tagged op applied to a tensor requiring grad performs
        # the mutation, returns a result requiring grad, and backward raises RuntimeError that
        # there is no autograd formula.
        @torch.library.custom_op(
            "_torch_testing::inplace_tag_autograd",
            mutates_args={"x"},
            tags=torch.Tag.inplace,
        )
        def f(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Adds y into x in place and returns x.
            x.add_(y)
            return x

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        expected = x.detach() + y
        result = f(x, y)
        self.assertEqual(x, expected)
        self.assertTrue(result.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "no autograd formula"):
            result.sum().backward()

    def test_custom_op_inplace_tag_register_fake_warns(self):
        # [FUNCTION DOC] Verifies registering a fake kernel for an inplace-tagged op emits a
        # UserWarning mentioning the tag, and the fake kernel's output under fake mode is not the
        # input object.
        @torch.library.custom_op(
            "_torch_testing::inplace_tag_register_fake_warns",
            mutates_args={"x"},
            tags=torch.Tag.inplace,
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns x.
            return x

        with self.assertWarnsRegex(UserWarning, "torch.Tag.inplace"):

            @f.register_fake
            def _(x):
                # [FUNCTION DOC] HELPER: Fake kernel returning a clone of x.
                return x.clone()

        with torch._subclasses.fake_tensor.FakeTensorMode():
            fake_x = torch.randn(3)
            self.assertIsNot(f(fake_x), fake_x)

    @skipIfTorchDynamo("recursive dynamo")
    @requires_compile
    @parametrize(
        "backend, fullgraph",
        [
            subtest(("eager", False), name="eager"),
            subtest(("eager", True), name="eager_fullgraph"),
            subtest(("inductor", True), name="inductor_fullgraph"),
        ],
    )
    def test_custom_op_inplace_tag_compile(self, backend, fullgraph):
        # [FUNCTION DOC] Parametrized over eager, eager fullgraph and inductor fullgraph backends:
        # compiles a function calling an inplace-tagged op and checks it returns the same tensor
        # object it mutated, with the expected sum. Skipped under Dynamo.
        name = f"{backend}_{'fullgraph' if fullgraph else 'default'}"

        @torch.library.custom_op(
            f"_torch_testing::inplace_tag_compile_{name}",
            mutates_args={"x"},
            tags=torch.Tag.inplace,
        )
        def f(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Adds y into x in place and returns x.
            x.add_(y)
            return x

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled wrapper calling the op.
            return f(x, y)

        x = torch.randn(3)
        y = torch.randn(3)
        expected = x + y
        result = torch.compile(fn, backend=backend, fullgraph=fullgraph)(x, y)
        self.assertIs(result, x)
        self.assertEqual(x, expected)

    def test_custom_op_inplace_tag_no_positional_arg(self):
        # [FUNCTION DOC] Verifies the inplace tag with a function that has only a keyword-only
        # argument raises ValueError mentioning the tag.
        def f(*, x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Function with only a keyword-only tensor argument.
            return x

        with self.assertRaisesRegex(ValueError, "torch.Tag.inplace"):
            torch.library.custom_op(
                "_torch_testing::inplace_tag_no_positional_arg",
                f,
                mutates_args={"x"},
                tags=torch.Tag.inplace,
            )

    def test_custom_op_inplace_tag_first_arg_not_tensor(self):
        # [FUNCTION DOC] Verifies the inplace tag with a non-tensor first argument raises
        # ValueError about the first positional argument.
        def f(x: int, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Function whose first argument is an int.
            return y

        with self.assertRaisesRegex(ValueError, "first positional argument"):
            torch.library.custom_op(
                "_torch_testing::inplace_tag_first_arg_not_tensor",
                f,
                mutates_args={"y"},
                tags=torch.Tag.inplace,
            )

    def test_custom_op_inplace_tag_mutates_non_first_arg(self):
        # [FUNCTION DOC] Verifies the inplace tag with mutates_args not covering the first argument
        # raises ValueError mentioning mutates_args.
        def f(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Function returning its second tensor.
            return y

        with self.assertRaisesRegex(ValueError, "mutates_args"):
            torch.library.custom_op(
                "_torch_testing::inplace_tag_mutates_non_first_arg",
                f,
                mutates_args={"y"},
                tags=torch.Tag.inplace,
            )

    def test_custom_op_inplace_tag_return_none(self):
        # [FUNCTION DOC] Verifies the inplace tag with a function annotated to return nothing
        # raises ValueError mentioning the return annotation.
        def f(x: Tensor) -> None:
            # [FUNCTION DOC] HELPER: Function with no return value.
            pass

        with self.assertRaisesRegex(ValueError, "return annotation"):
            torch.library.custom_op(
                "_torch_testing::inplace_tag_return_none",
                f,
                mutates_args={"x"},
                tags=torch.Tag.inplace,
            )

    def test_custom_op_inplace_tag_multiple_returns(self):
        # [FUNCTION DOC] Verifies the inplace tag with a function returning two tensors raises
        # ValueError mentioning the return annotation.
        def f(x: Tensor) -> tuple[Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Function returning a pair of tensors.
            return x, x

        with self.assertRaisesRegex(ValueError, "return annotation"):
            torch.library.custom_op(
                "_torch_testing::inplace_tag_multiple_returns",
                f,
                mutates_args={"x"},
                tags=torch.Tag.inplace,
            )

    def test_custom_op_inplace_tag_manual_schema(self):
        # [FUNCTION DOC] Verifies the inplace tag works with an explicit schema string using
        # alias annotations: schema aliasing is as expected, calls return the mutated input, also
        # under FakeTensorMode.
        @torch.library.custom_op(
            "_torch_testing::inplace_tag_manual_schema",
            mutates_args={"x"},
            schema="(Tensor(a!) x, Tensor y) -> Tensor(a!)",
            tags=torch.Tag.inplace,
        )
        def f(x, y):
            # [FUNCTION DOC] HELPER: Adds y into x in place and returns x.
            x.add_(y)
            return x

        self.assertIn(torch.Tag.inplace, f._opoverload.tags)
        schema = f._opoverload._schema
        self.assertTrue(schema.arguments[0].alias_info.is_write)
        self.assertTrue(schema.returns[0].alias_info.is_write)
        self.assertEqual(
            schema.arguments[0].alias_info.before_set,
            schema.returns[0].alias_info.before_set,
        )

        x = torch.randn(3)
        y = torch.randn(3)
        expected = x + y
        result = f(x, y)
        self.assertIs(result, x)
        self.assertEqual(x, expected)

        with torch._subclasses.fake_tensor.FakeTensorMode():
            fake_x = torch.randn(3)
            fake_y = torch.randn(3)
            self.assertIs(f(fake_x, fake_y), fake_x)

    def test_custom_op_inplace_tag_bad_manual_schema(self):
        # [FUNCTION DOC] Verifies an explicit schema whose return does not alias the mutated first
        # argument raises ValueError saying the return must alias.
        with self.assertRaisesRegex(ValueError, "return must alias"):

            @torch.library.custom_op(
                "_torch_testing::inplace_tag_bad_manual_schema",
                mutates_args={"x"},
                schema="(Tensor(a!) x) -> Tensor",
                tags=torch.Tag.inplace,
            )
            def f(x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

    def test_custom_op_out_tag(self):
        # [FUNCTION DOC] Tests the out tag: the out argument is keyword-only and written with the
        # same alias set as the return, the call returns the out tensor holding the sum, also
        # under FakeTensorMode.
        @torch.library.custom_op(
            "_torch_testing::out_tag",
            mutates_args={"out"},
            tags=torch.Tag.out,
        )
        def f(x: Tensor, y: Tensor, *, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Copies the sum of x and y into out and returns out.
            out.copy_(x + y)
            return out

        self.assertIn(torch.Tag.out, f._opoverload.tags)
        schema = f._opoverload._schema
        self.assertTrue(schema.arguments[2].kwarg_only)
        self.assertTrue(schema.arguments[2].alias_info.is_write)
        self.assertTrue(schema.returns[0].alias_info.is_write)
        self.assertEqual(
            schema.arguments[2].alias_info.before_set,
            schema.returns[0].alias_info.before_set,
        )

        x = torch.randn(3)
        y = torch.randn(3)
        out = torch.empty_like(x)
        result = f(x, y, out=out)
        self.assertIs(result, out)
        self.assertEqual(out, x + y)

        with torch._subclasses.fake_tensor.FakeTensorMode():
            fake_x = torch.randn(3)
            fake_y = torch.randn(3)
            fake_out = torch.empty_like(fake_x)
            fake_result = f(fake_x, fake_y, out=fake_out)
            self.assertIs(fake_result, fake_out)

    @skipIfTorchDynamo("recursive dynamo")
    @requires_compile
    @parametrize(
        "backend, fullgraph",
        [
            subtest(("eager", False), name="eager"),
            subtest(("eager", True), name="eager_fullgraph"),
            subtest(("inductor", True), name="inductor_fullgraph"),
        ],
    )
    def test_custom_op_out_tag_compile(self, backend, fullgraph):
        # [FUNCTION DOC] Parametrized over eager, eager fullgraph and inductor fullgraph: compiles
        # a function calling an out-tagged op, checks out holds the sum, and for the eager backend
        # that the returned object is out. Skipped under Dynamo.
        name = f"{backend}_{'fullgraph' if fullgraph else 'default'}"

        @torch.library.custom_op(
            f"_torch_testing::out_tag_compile_{name}",
            mutates_args={"out"},
            tags=torch.Tag.out,
        )
        def f(x: Tensor, y: Tensor, *, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Copies the sum of x and y into out and returns out.
            out.copy_(x + y)
            return out

        def fn(x, y, out):
            # [FUNCTION DOC] HELPER: Compiled wrapper calling the op with out passed by keyword.
            return f(x, y, out=out)

        x = torch.randn(3)
        y = torch.randn(3)
        out = torch.empty_like(x)
        expected = x + y
        result = torch.compile(fn, backend=backend, fullgraph=fullgraph)(x, y, out)
        if backend == "eager":
            self.assertIs(result, out)
        self.assertEqual(out, expected)
        self.assertEqual(result, expected)

    def test_custom_op_out_tag_multiple_returns(self):
        # [FUNCTION DOC] Tests an out-tagged op with two out arguments and a pair return: both are
        # keyword-only and written, aliasing the matching returns; the call returns the two out
        # tensors (also under FakeTensorMode) holding the sum and product.
        @torch.library.custom_op(
            "_torch_testing::out_tag_multiple_returns",
            mutates_args={"add_out", "mul_out"},
            tags=torch.Tag.out,
        )
        def f(
            x: Tensor, y: Tensor, *, add_out: Tensor, mul_out: Tensor
        ) -> tuple[Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Writes the sum and product into the two out tensors and
            # returns them.
            add_out.copy_(x + y)
            mul_out.copy_(x * y)
            return add_out, mul_out

        schema = f._opoverload._schema
        out_args = schema.arguments[2:]
        self.assertTrue(all(arg.kwarg_only for arg in out_args))
        self.assertEqual(len(schema.returns), 2)
        for arg, ret in zip(out_args, schema.returns):
            self.assertTrue(arg.alias_info.is_write)
            self.assertTrue(ret.alias_info.is_write)
            self.assertEqual(
                arg.alias_info.before_set,
                ret.alias_info.before_set,
            )

        x = torch.randn(3)
        y = torch.randn(3)
        add_out = torch.empty_like(x)
        mul_out = torch.empty_like(x)
        result = f(x, y, add_out=add_out, mul_out=mul_out)
        self.assertIs(result[0], add_out)
        self.assertIs(result[1], mul_out)
        self.assertEqual(add_out, x + y)
        self.assertEqual(mul_out, x * y)

        with torch._subclasses.fake_tensor.FakeTensorMode():
            fake_x = torch.randn(3)
            fake_y = torch.randn(3)
            fake_add_out = torch.empty_like(fake_x)
            fake_mul_out = torch.empty_like(fake_x)
            fake_result = f(
                fake_x,
                fake_y,
                add_out=fake_add_out,
                mul_out=fake_mul_out,
            )
            self.assertIs(fake_result[0], fake_add_out)
            self.assertIs(fake_result[1], fake_mul_out)

    def test_custom_op_out_tag_returns_wrong_tensor(self):
        # [FUNCTION DOC] Verifies an out-tagged op that returns an input instead of its out
        # argument raises RuntimeError about mutable keyword-only arguments.
        @torch.library.custom_op(
            "_torch_testing::out_tag_returns_wrong_tensor",
            mutates_args={"out"},
            tags=torch.Tag.out,
        )
        def f(x: Tensor, *, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns x instead of out.
            return x

        with self.assertRaisesRegex(RuntimeError, "mutable keyword-only arguments"):
            f(torch.randn(3), out=torch.empty(3))

    def test_custom_op_out_tag_autograd(self):
        # [FUNCTION DOC] Verifies an out-tagged op rejects inputs or out requiring grad with
        # RuntimeError (no automatic differentiation with out arguments), but works under no_grad
        # leaving out without a grad function.
        @torch.library.custom_op(
            "_torch_testing::out_tag_autograd",
            mutates_args={"out"},
            tags=torch.Tag.out,
        )
        def f(x: Tensor, *, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Copies the sine of x into out and returns out.
            out.copy_(x.sin())
            return out

        error_msg = (
            r"functions with out=\.\.\. arguments don't support automatic "
            "differentiation"
        )
        with self.assertRaisesRegex(RuntimeError, error_msg):
            f(torch.randn(3, requires_grad=True), out=torch.empty(3))

        with self.assertRaisesRegex(RuntimeError, error_msg):
            f(torch.randn(3), out=torch.empty(3, requires_grad=True))

        x = torch.randn(3, requires_grad=True)
        out = torch.empty(3, requires_grad=True)
        with torch.no_grad():
            result = f(x, out=out)
        self.assertIs(result, out)
        self.assertTrue(out.requires_grad)
        self.assertIsNone(out.grad_fn)
        self.assertEqual(out, x.sin())

    def test_custom_op_out_tag_register_fake_warns(self):
        # [FUNCTION DOC] Verifies registering a fake kernel for an out-tagged op emits a
        # UserWarning mentioning the out tag, and in fake mode the fake kernel result is not the
        # passed out tensor.
        @torch.library.custom_op(
            "_torch_testing::out_tag_register_fake_warns",
            mutates_args={"out"},
            tags=torch.Tag.out,
        )
        def f(x: Tensor, *, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Copies x into out and returns out.
            out.copy_(x)
            return out

        with self.assertWarnsRegex(UserWarning, "torch.Tag.out"):

            @f.register_fake
            def _(x, *, out):
                # [FUNCTION DOC] HELPER: Fake kernel returning a clone of out.
                return out.clone()

        with torch._subclasses.fake_tensor.FakeTensorMode():
            fake_x = torch.randn(3)
            fake_out = torch.empty_like(fake_x)
            self.assertIsNot(f(fake_x, out=fake_out), fake_out)

    def test_custom_op_register_fake_ordinary_op(self):
        # [FUNCTION DOC] Verifies a fake kernel registered for an ordinary (untagged) op is called
        # under FakeTensorMode and yields a new tensor, not the input.
        @torch.library.custom_op(
            "_torch_testing::register_fake_ordinary_op",
            mutates_args=(),
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns a clone of x.
            return x.clone()

        called = False

        @f.register_fake
        def _(x):
            # [FUNCTION DOC] HELPER: Fake kernel; sets the enclosing called flag and returns an
            # empty tensor shaped like x.
            nonlocal called
            called = True
            return x.new_empty(x.shape)

        with torch._subclasses.fake_tensor.FakeTensorMode():
            fake_x = torch.randn(3)
            fake_result = f(fake_x)
            self.assertIsNot(fake_result, fake_x)
        self.assertTrue(called)

    def test_custom_op_out_tag_register_autograd(self):
        # [FUNCTION DOC] Verifies register_autograd on an out-tagged op, through both the op
        # method and the torch.library function, raises RuntimeError saying out variants do not
        # support autograd.
        @torch.library.custom_op(
            "_torch_testing::out_tag_register_autograd",
            mutates_args={"out"},
            tags=torch.Tag.out,
        )
        def f(x: Tensor, *, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Copies the sine of x into out and returns out.
            out.copy_(x.sin())
            return out

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward returning the gradient unchanged.
            return grad

        with self.assertRaisesRegex(
            RuntimeError, "Out variants do not support autograd"
        ):
            f.register_autograd(backward)

        with self.assertRaisesRegex(
            RuntimeError, "Out variants do not support autograd"
        ):
            torch.library.register_autograd(f, backward)

    def test_library_register_autograd_out_tag_low_level(self):
        # [FUNCTION DOC] Verifies torch.library.register_autograd on a low-level Library op with
        # the out tag raises RuntimeError saying out variants do not support autograd.
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            lib.define(
                "out_tag_register_autograd_low_level(Tensor x, *, Tensor(a!) out) -> Tensor(a!)",
                tags=torch.Tag.out,
            )

            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward returning the gradient unchanged.
                return grad

            with self.assertRaisesRegex(
                RuntimeError, "Out variants do not support autograd"
            ):
                torch.library.register_autograd(
                    "_torch_testing::out_tag_register_autograd_low_level",
                    backward,
                    lib=lib,
                )

    def test_custom_op_out_tag_positional_mutable_arg(self):
        # [FUNCTION DOC] Verifies the out tag with a positional mutable out argument raises
        # ValueError requiring keyword-only.
        def f(x: Tensor, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Function with positional out, returning it.
            return out

        with self.assertRaisesRegex(ValueError, "keyword-only"):
            torch.library.custom_op(
                "_torch_testing::out_tag_positional_mutable_arg",
                f,
                mutates_args={"out"},
                tags=torch.Tag.out,
            )

    def test_custom_op_out_tag_no_mutable_out(self):
        # [FUNCTION DOC] Verifies the out tag with no mutated arguments raises ValueError requiring
        # at least one.
        def f(x: Tensor, *, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Function with keyword-only out, returning it.
            return out

        with self.assertRaisesRegex(ValueError, "at least one"):
            torch.library.custom_op(
                "_torch_testing::out_tag_no_mutable_out",
                f,
                mutates_args=(),
                tags=torch.Tag.out,
            )

    def test_custom_op_out_tag_optional_mutable_out(self):
        # [FUNCTION DOC] Verifies the out tag with an optional tensor out argument raises
        # ValueError saying only some types are supported.
        def f(x: Tensor, *, out: Optional[Tensor]) -> Tensor:
            # [FUNCTION DOC] HELPER: Function with optional out, returning x if absent else out.
            return x if out is None else out

        with self.assertRaisesRegex(ValueError, "only supports"):
            torch.library.custom_op(
                "_torch_testing::out_tag_optional_mutable_out",
                f,
                mutates_args={"out"},
                tags=torch.Tag.out,
            )

    def test_custom_op_out_tag_list_mutable_out(self):
        # [FUNCTION DOC] Verifies the out tag with a list-of-tensors out argument raises ValueError
        # saying only some types are supported.
        def f(x: Tensor, *, out: List[Tensor]) -> Tensor:
            # [FUNCTION DOC] HELPER: Function with out list, returning its first element.
            return out[0]

        with self.assertRaisesRegex(ValueError, "only supports"):
            torch.library.custom_op(
                "_torch_testing::out_tag_list_mutable_out",
                f,
                mutates_args={"out"},
                tags=torch.Tag.out,
            )

    def test_custom_op_out_tag_manual_schema_requires_tensor_out(self):
        # [FUNCTION DOC] Verifies an out-tagged op with a manual schema whose out is an int raises
        # ValueError saying only Tensor out arguments are supported.
        with self.assertRaisesRegex(ValueError, "only supports Tensor"):

            @torch.library.custom_op(
                "_torch_testing::out_tag_manual_schema_requires_tensor_out",
                mutates_args={"out"},
                schema="(Tensor x, *, int(a!) out) -> int(a!)",
                tags=torch.Tag.out,
            )
            def f(x, *, out):
                # [FUNCTION DOC] HELPER: Returns out.
                return out

    def test_custom_op_out_tag_return_mismatch(self):
        # [FUNCTION DOC] Verifies the out tag with two out arguments but a single tensor return
        # annotation raises ValueError mentioning the return annotation.
        def f(x: Tensor, *, out1: Tensor, out2: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Function with two outs returning the first.
            return out1

        with self.assertRaisesRegex(ValueError, "return annotation"):
            torch.library.custom_op(
                "_torch_testing::out_tag_return_mismatch",
                f,
                mutates_args={"out1", "out2"},
                tags=torch.Tag.out,
            )

    def test_custom_op_out_tag_single_return_tuple_mismatch(self):
        # [FUNCTION DOC] Verifies the out tag with one out argument but a one-tuple return
        # annotation raises ValueError mentioning the return annotation.
        def f(x: Tensor, *, out: Tensor) -> tuple[Tensor]:
            # [FUNCTION DOC] HELPER: Function returning a one-tuple holding out.
            return (out,)

        with self.assertRaisesRegex(ValueError, "return annotation"):
            torch.library.custom_op(
                "_torch_testing::out_tag_single_return_tuple_mismatch",
                f,
                mutates_args={"out"},
                tags=torch.Tag.out,
            )

    def test_custom_op_out_tag_manual_schema(self):
        # [FUNCTION DOC] Verifies the out tag with an explicit schema: keyword-only aliasing out
        # and return, call returns the out tensor holding the sum, also under FakeTensorMode.
        @torch.library.custom_op(
            "_torch_testing::out_tag_manual_schema",
            mutates_args={"out"},
            schema="(Tensor x, Tensor y, *, Tensor(a!) out) -> Tensor(a!)",
            tags=torch.Tag.out,
        )
        def f(x, y, *, out):
            # [FUNCTION DOC] HELPER: Copies the sum of x and y into out and returns out.
            out.copy_(x + y)
            return out

        self.assertIn(torch.Tag.out, f._opoverload.tags)
        schema = f._opoverload._schema
        self.assertTrue(schema.arguments[2].kwarg_only)
        self.assertTrue(schema.arguments[2].alias_info.is_write)
        self.assertTrue(schema.returns[0].alias_info.is_write)
        self.assertEqual(
            schema.arguments[2].alias_info.before_set,
            schema.returns[0].alias_info.before_set,
        )

        x = torch.randn(3)
        y = torch.randn(3)
        out = torch.empty_like(x)
        result = f(x, y, out=out)
        self.assertIs(result, out)
        self.assertEqual(out, x + y)

        with torch._subclasses.fake_tensor.FakeTensorMode():
            fake_x = torch.randn(3)
            fake_y = torch.randn(3)
            fake_out = torch.empty_like(fake_x)
            self.assertIs(f(fake_x, fake_y, out=fake_out), fake_out)

    def test_custom_op_out_tag_bad_manual_schema(self):
        # [FUNCTION DOC] Verifies an explicit schema with a positional out argument raises
        # ValueError requiring keyword-only.
        with self.assertRaisesRegex(ValueError, "keyword-only"):

            @torch.library.custom_op(
                "_torch_testing::out_tag_bad_manual_schema",
                mutates_args={"out"},
                schema="(Tensor x, Tensor(a!) out) -> Tensor(a!)",
                tags=torch.Tag.out,
            )
            def f(x, out):
                # [FUNCTION DOC] HELPER: Returns out.
                return out

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_basic(self):
        # [FUNCTION DOC] Creates a custom op adding a float via numpy, verifies the default
        # result, then registers a CPU kernel with register_kernel and verifies it is used.
        # Skipped under Dynamo.
        @torch.library.custom_op("_torch_testing::add", mutates_args=())
        def add(x: Tensor, y: float) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body adding y to x through numpy and moving to x's device.
            x_np = x.numpy(force=True)
            out_np = x_np + y
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = 3.14
        z = add(x, y)
        self.assertEqual(z, x + y)

        cpu_called = False

        @add.register_kernel("cpu")
        def _(x, y):
            # [FUNCTION DOC] HELPER: CPU kernel; sets the enclosing flag and returns x plus y as
            # a tensor.
            nonlocal cpu_called
            cpu_called = True
            x_np = x.numpy()
            out_np = x_np + y
            return torch.from_numpy(out_np)

        z = add(x, y)
        self.assertEqual(z, x + y)
        self.assertTrue(cpu_called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_no_grad_skips_autograd(self):
        # [FUNCTION DOC] Verifies the autograd setup_context is not called under no_grad or
        # when no input requires grad, but is called once when an input requires grad. Skipped
        # under Dynamo.
        @torch.library.custom_op("_torch_testing::add", mutates_args=())
        def add(x: Tensor, y: float) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body adding y to x through numpy.
            x_np = x.numpy(force=True)
            out_np = x_np + y
            return torch.from_numpy(out_np).to(x.device)

        called = 0

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup hook counting its invocations.
            nonlocal called
            called += 1

        def backward(ctx, grad):
            # [FUNCTION DOC] HELPER: Backward that raises AssertionError because it must not run.
            raise AssertionError("should not be reached")

        add.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        with torch.no_grad():
            y = add(x, 2.0)
        self.assertEqual(called, 0)
        self.assertEqual(y, x + 2.0)

        x.requires_grad_(False)
        y = add(x, 2.0)
        self.assertEqual(called, 0)
        self.assertEqual(y, x + 2.0)

        x = torch.randn(3, requires_grad=True)
        y = add(x, 2.0)
        self.assertEqual(called, 1)
        self.assertEqual(y, x + 2.0)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @parametrize(
        "op_factory, opname",
        [
            subtest((torch.library.custom_op, "custom"), name="custom_op"),
            subtest((torch.library.triton_op, "triton"), name="triton_op"),
        ],
    )
    def test_register_autograd_incorrect_num_gradients(self, op_factory, opname):
        # [FUNCTION DOC] Parametrized over custom_op and triton_op factories: a backward returning
        # an empty tuple for a one-input op raises RuntimeError reporting expected one and got zero
        # gradients. Skipped under Dynamo.
        @op_factory(
            f"_torch_testing::incorrect_num_gradients_{opname}",
            mutates_args=(),
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning the sine of x.
            return x.sin()

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: No-op setup hook.
            pass

        def backward(ctx, grad_output):
            # [FUNCTION DOC] HELPER: Backward returning an empty tuple.
            return ()

        f.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"incorrect number of gradients \(expected 1, got 0\)",
        ):
            f(x).sum().backward()

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @parametrize(
        "op_factory, opname",
        [
            subtest((torch.library.custom_op, "custom"), name="custom_op"),
            subtest((torch.library.triton_op, "triton"), name="triton_op"),
        ],
    )
    def test_register_autograd_defaulted_inputs_full_gradients(
        self, op_factory, opname
    ):
        # [FUNCTION DOC] Parametrized over custom_op and triton_op: for an op with defaulted
        # inputs, the setup hook sees all four inputs including defaults, backward returning one
        # gradient per input (None for others) works, needs_input_grad covers only the passed
        # input, and the gradient is the cosine of x. Skipped under Dynamo.
        @op_factory(
            f"_torch_testing::defaulted_input_gradients_{opname}",
            mutates_args=(),
        )
        def f(
            x: Tensor,
            y: Optional[Tensor] = None,
            scale: float = 2.0,
            enabled: bool = False,
        ) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body with optional and defaulted arguments returning the
            # sine of x.
            return x.sin()

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup hook asserting four inputs with the default values and
            # saving x.
            self.assertEqual(len(inputs), 4)
            self.assertIs(inputs[1], None)
            self.assertEqual(inputs[2], 2.0)
            self.assertFalse(inputs[3])
            (x, _, _, _) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad_output):
            # [FUNCTION DOC] HELPER: Checks needs_input_grad has only the first entry and returns
            # the x gradient followed by three Nones.
            self.assertEqual(ctx.needs_input_grad, (True,))
            (x,) = ctx.saved_tensors
            return grad_output * x.cos(), None, None, None

        f.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        f(x).sum().backward()
        self.assertEqual(x.grad, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @parametrize(
        "op_factory, opname",
        [
            subtest((torch.library.custom_op, "custom"), name="custom_op"),
            subtest((torch.library.triton_op, "triton"), name="triton_op"),
        ],
    )
    def test_register_autograd_defaulted_inputs_actual_gradients(
        self, op_factory, opname
    ):
        # [FUNCTION DOC] Parametrized over custom_op and triton_op: with defaulted inputs, backward
        # returning only the gradient for x works and equals the cosine. Skipped under Dynamo.
        @op_factory(
            f"_torch_testing::defaulted_input_actual_gradients_{opname}",
            mutates_args=(),
        )
        def f(
            x: Tensor,
            y: Optional[Tensor] = None,
            scale: float = 2.0,
            enabled: bool = False,
        ) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body with optional and defaulted arguments returning the
            # sine of x.
            return x.sin()

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup hook asserting four inputs and saving x.
            self.assertEqual(len(inputs), 4)
            (x, _, _, _) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad_output):
            # [FUNCTION DOC] HELPER: Checks needs_input_grad is the one-entry form and returns the
            # x gradient.
            self.assertEqual(ctx.needs_input_grad, (True,))
            (x,) = ctx.saved_tensors
            return grad_output * x.cos()

        f.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        f(x).sum().backward()
        self.assertEqual(x.grad, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @parametrize(
        "op_factory, opname",
        [
            subtest((torch.library.custom_op, "custom"), name="custom_op"),
            subtest((torch.library.triton_op, "triton"), name="triton_op"),
        ],
    )
    def test_register_autograd_defaulted_inputs_too_few_gradients(
        self, op_factory, opname
    ):
        # [FUNCTION DOC] Parametrized over custom_op and triton_op: returning three gradients for
        # an op with four inputs (one passed) raises RuntimeError reporting expected one or four,
        # got three. Skipped under Dynamo.
        @op_factory(
            f"_torch_testing::defaulted_input_too_few_gradients_{opname}",
            mutates_args=(),
        )
        def f(
            x: Tensor,
            y: Optional[Tensor] = None,
            scale: float = 2.0,
            enabled: bool = False,
        ) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body with optional and defaulted arguments returning the
            # sine of x.
            return x.sin()

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup hook saving x.
            (x, _, _, _) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad_output):
            # [FUNCTION DOC] HELPER: Backward returning three gradient entries.
            (x,) = ctx.saved_tensors
            return grad_output * x.cos(), None, None

        f.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"incorrect number of gradients \(expected 1 or 4, got 3\)",
        ):
            f(x).sum().backward()

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @parametrize(
        "op_factory, opname",
        [
            subtest((torch.library.custom_op, "custom"), name="custom_op"),
            subtest((torch.library.triton_op, "triton"), name="triton_op"),
        ],
    )
    def test_register_autograd_defaulted_inputs_non_none_gradient(
        self, op_factory, opname
    ):
        # [FUNCTION DOC] Parametrized over custom_op and triton_op: backward returning a
        # non-None gradient for a defaulted input that was not passed raises RuntimeError.
        # Skipped under Dynamo.
        @op_factory(
            f"_torch_testing::defaulted_input_non_none_gradient_{opname}",
            mutates_args=(),
        )
        def f(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body with an optional argument returning the sine of x.
            return x.sin()

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup hook saving x.
            (x, _) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad_output):
            # [FUNCTION DOC] HELPER: Backward returning a gradient for x and a gradient for the
            # unused input.
            (x,) = ctx.saved_tensors
            return grad_output * x.cos(), grad_output

        f.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "returned a non-None gradient for an input that was not passed",
        ):
            f(x).sum().backward()

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_pt_metadata_ctx_attr_is_not_tensorlist(self):
        # [FUNCTION DOC] Verifies a user attribute named _pt_metadata set on ctx in setup_context
        # (a string, not a tensor list) survives to backward and gradients are correct. Skipped
        # under Dynamo.
        @torch.library.custom_op(
            "_torch_testing::pt_metadata_ctx_attr_is_not_tensorlist",
            mutates_args=(),
        )
        def f(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning the sine of x.
            return x.sin()

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup hook storing the string attribute and saving x.
            ctx._pt_metadata = "user attr"
            (x, _) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad_output):
            # [FUNCTION DOC] HELPER: Backward checking the string attribute and returning the x
            # gradient and None.
            self.assertEqual(ctx._pt_metadata, "user attr")
            (x,) = ctx.saved_tensors
            return grad_output * x.cos(), None

        f.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        f(x).sum().backward()
        self.assertEqual(x.grad, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_tensorlist_defaulted_input(self):
        # [FUNCTION DOC] Op taking a tensor list and a defaulted scale: checks gradients for
        # both tensors and needs_input_grad shape, with the scale defaulted and passed. Skipped
        # under Dynamo.
        @torch.library.custom_op(
            "_torch_testing::tensorlist_defaulted_input_gradients",
            mutates_args=(),
        )
        def f(xs: list[Tensor], scale: float = 2.0) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning the scaled sum of sines.
            return sum(x.sin() for x in xs) * scale

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup hook asserting scale is 2.0 or 3.0, saving the tensors
            # and the scale.
            xs, scale = inputs
            self.assertIn(scale, (2.0, 3.0))
            ctx.save_for_backward(*xs)
            ctx.scale = scale

        def backward(ctx, grad_output):
            # [FUNCTION DOC] HELPER: Backward checking needs_input_grad and returning per-tensor
            # gradients and None for the scale.
            if ctx.scale == 2.0:
                self.assertEqual(ctx.needs_input_grad, ([True, True],))
            else:
                self.assertEqual(ctx.needs_input_grad, ([True, True], False))
            grads = [grad_output * x.cos() * ctx.scale for x in ctx.saved_tensors]
            return grads, None

        f.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        f([x, y]).sum().backward()
        self.assertEqual(x.grad, x.cos() * 2.0)
        self.assertEqual(y.grad, y.cos() * 2.0)

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        f([x, y], 3.0).sum().backward()
        self.assertEqual(x.grad, x.cos() * 3.0)
        self.assertEqual(y.grad, y.cos() * 3.0)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_tensorlist_return_defaulted_input(self):
        # [FUNCTION DOC] Op returning a tensor list with a defaulted scale: checks the combined
        # gradient of both outputs for the default and an explicit scale. Skipped under Dynamo.
        @torch.library.custom_op(
            "_torch_testing::tensorlist_return_defaulted_input_gradients",
            mutates_args=(),
        )
        def f(x: Tensor, scale: float = 2.0) -> list[Tensor]:
            # [FUNCTION DOC] HELPER: Op body returning the scaled sine and cosine as a list.
            return [x.sin() * scale, x.cos() * scale]

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup hook saving x and the scale.
            x, scale = inputs
            ctx.save_for_backward(x)
            ctx.scale = scale

        def backward(ctx, grad_outputs):
            # [FUNCTION DOC] HELPER: Backward combining the two output gradients into the x
            # gradient, None for scale.
            if ctx.scale == 2.0:
                self.assertEqual(ctx.needs_input_grad, (True,))
            else:
                self.assertEqual(ctx.needs_input_grad, (True, False))
            (x,) = ctx.saved_tensors
            grad_x = (
                grad_outputs[0] * x.cos() * ctx.scale
                - grad_outputs[1] * x.sin() * ctx.scale
            )
            return grad_x, None

        f.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        y0, y1 = f(x, 3.0)
        (y0.sum() + y1.sum()).backward()
        self.assertEqual(x.grad, (x.cos() - x.sin()) * 3.0)

        x = torch.randn(3, requires_grad=True)
        y0, y1 = f(x)
        (y0.sum() + y1.sum()).backward()
        self.assertEqual(x.grad, (x.cos() - x.sin()) * 2.0)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_manual_schema(self):
        # [FUNCTION DOC] Checks custom_op with an explicit schema string, both for a
        # functional op and for an in-place op marked as mutating x with a no-return schema.
        # Skipped under Dynamo.
        @torch.library.custom_op(
            "_torch_testing::add",
            mutates_args=(),
            schema="(Tensor x, float y) -> Tensor",
        )
        def add(x, y):
            # [FUNCTION DOC] HELPER: Adds y to x through numpy and returns the tensor.
            x_np = x.numpy(force=True)
            out_np = x_np + y
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = 3.14
        z = add(x, y)
        self.assertEqual(z, x + y)

        @torch.library.custom_op(
            "_torch_testing::sin_",
            mutates_args=["x"],
            schema="(Tensor(a!) x) -> ()",
        )
        def sin_(x):
            # [FUNCTION DOC] HELPER: Applies sine to x in place through numpy; returns nothing.
            x_np = x.numpy()
            np.sin(x_np, out=x_np)

        x = torch.randn(3)
        expected = x.sin()
        sin_(x)
        self.assertEqual(x, expected)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_subclass_accessor_view_error(self):
        # [FUNCTION DOC] Verifies that custom ops returning views with SymInt or SymInt-list
        # arguments raise RuntimeError saying they do not support SymInt when called. Skipped
        # under Dynamo.
        @torch.library.custom_op(
            "_torch_testing::_failing_two_tensor_accessor",
            mutates_args=(),
            schema="(Tensor(a) tx, SymInt idx) -> Tensor(a)",
        )
        def _failing_two_tensor_accessor(tx, idx):
            # [FUNCTION DOC] HELPER: Op body returning a view of tx.
            return tx.view_as(tx)

        def noop(*args):
            # [FUNCTION DOC] HELPER: Accepts any arguments and does nothing.
            pass

        _failing_two_tensor_accessor.register_autograd(noop, setup_context=noop)

        t = torch.rand(2)
        with self.assertRaisesRegex(
            RuntimeError, "Custom ops that are views do not support SymInt."
        ):
            torch.ops._torch_testing._failing_two_tensor_accessor(t, 2)

        @torch.library.custom_op(
            "_torch_testing::_failing_two_tensor_accessor_list",
            mutates_args=(),
            schema="(Tensor(a) tx, SymInt[] idx) -> Tensor(a)",
        )
        def _failing_two_tensor_accessor_list(tx, idx):
            # [FUNCTION DOC] HELPER: Op body returning a view of tx.
            return tx.view_as(tx)

        def noop(*args):
            # [FUNCTION DOC] HELPER: Accepts any arguments and does nothing.
            pass

        _failing_two_tensor_accessor_list.register_autograd(noop, setup_context=noop)

        t = torch.rand(2)
        with self.assertRaisesRegex(
            RuntimeError, "Custom ops that are views do not support SymInt."
        ):
            torch.ops._torch_testing._failing_two_tensor_accessor_list(t, (2,))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_subclass_accessor_view(self):
        # [FUNCTION DOC] Defines a view-returning accessor op for a TwoTensor subclass with
        # autograd, checks results are fresh views of the inner tensors with the base set to the
        # subclass, gradients flow into the right component, and in-place edits on a view affect
        # the non-leaf but not the leaf. Skipped under Dynamo.
        class MyTwoTensor(TwoTensor):
            # [CLASS DOC] HELPER: TwoTensor subclass whose dispatch handler serves the accessor op
            # by returning a fresh view of the first or second inner tensor, and defers all other
            # ops to the parent class.
            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: For the accessor op asserts args and returns a view of
                # inner a or b selected by the index; otherwise calls the parent handler.
                if func is torch.ops._torch_testing._two_tensor_accessor.default:
                    self.assertIsInstance(args[0], MyTwoTensor)
                    self.assertIn(args[1], (0, 1))
                    if args[1] == 0:
                        res = args[0].a
                    else:
                        res = args[0].b
                    # Always return a fresh Tensor!
                    return res.view_as(res)
                return super().__torch_dispatch__(func, types, args, kwargs)

        @torch.library.custom_op(
            "_torch_testing::_two_tensor_accessor",
            mutates_args=(),
            schema="(Tensor(a) tx, int idx) -> Tensor(a)",
        )
        def _two_tensor_accessor(tx, idx):
            # [FUNCTION DOC] HELPER: Op body that raises RuntimeError because the dispatch handler
            # serves it.
            raise RuntimeError("Should never be called")

        def backward(ctx, gO):
            # [FUNCTION DOC] HELPER: Backward building a MyTwoTensor gradient with the incoming
            # gradient in the selected slot and zeros in the other; None for the index.
            gI = gO.clone()
            if ctx.idx == 0:
                return MyTwoTensor(gI, torch.zeros_like(gO)), None
            else:
                return MyTwoTensor(torch.zeros_like(gO), gI), None

        def setup_ctx(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup hook marking the op as a pure view and saving the index
            # on ctx.
            ctx._is_pure_view = True
            ctx.idx = inputs[1]

        _two_tensor_accessor.register_autograd(backward, setup_context=setup_ctx)

        x = torch.rand(3)
        y = torch.rand(3)
        z = MyTwoTensor(x, y, requires_grad=True)
        res = torch.ops._torch_testing._two_tensor_accessor(z, 0)
        res.sum().backward()
        self.assertEqual(res, x)
        self.assertTrue(res._is_view())
        self.assertTrue(res._base is z)
        self.assertEqual(z.grad, torch.ones_like(z.grad))

        res = torch.ops._torch_testing._two_tensor_accessor(z, 1)
        res.sum().backward()
        self.assertEqual(res, y)
        self.assertTrue(res._is_view())
        self.assertTrue(res._base is z)
        self.assertEqual(z.grad, TwoTensor(torch.ones(3), torch.ones(3)))

        leaf = MyTwoTensor(torch.rand(3), torch.rand(3), requires_grad=True)
        non_leaf = leaf.clone()
        view_a = torch.ops._torch_testing._two_tensor_accessor(non_leaf, 0)
        self.assertTrue(view_a._is_view())
        self.assertTrue(view_a._base is non_leaf)
        view_a *= 2
        self.assertEqual(non_leaf.a, view_a)
        self.assertNotEqual(leaf.a, view_a)
        non_leaf.sum().backward()
        self.assertEqual(leaf.grad, MyTwoTensor(2 * torch.ones(3), torch.ones(3)))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_kwarg_only_tensors(self):
        # [FUNCTION DOC] Verifies keyword-only tensor arguments (plain, optional or list) are not
        # supported by custom_op, register_autograd or register_vmap: each raises
        # NotImplementedError mentioning kwarg-only Tensor args. Skipped under Dynamo.
        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):

            @torch.library.custom_op("_torch_testing::foo", mutates_args=())
            def foo(x: Tensor, *, y: int, z: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Stub with a keyword-only tensor argument.
                pass

        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):

            @torch.library.custom_op("_torch_testing::foo", mutates_args=())
            def foo2(x: Tensor, *, y: int, z: Optional[Tensor]) -> Tensor:
                # [FUNCTION DOC] HELPER: Stub with a keyword-only optional tensor argument.
                pass

        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):

            @torch.library.custom_op("_torch_testing::foo", mutates_args=())
            def foo3(x: Tensor, *, y: int, z: List[Tensor]) -> Tensor:
                # [FUNCTION DOC] HELPER: Stub with a keyword-only tensor list argument.
                pass

        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            lib.define("foo(Tensor x, *, Tensor y) -> Tensor")
            with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):
                torch.library.register_autograd(
                    "_torch_testing::foo",
                    lambda grad: grad,
                    setup_context=lambda ctx, inputs, keyword_only_inputs, output: None,
                )

            with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):
                torch.library.register_vmap(
                    "_torch_testing::foo",
                    lambda info, in_dims, x, *, y: (x, 0),
                )

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_kwargonly_low_level(self):
        # [FUNCTION DOC] Defines a low-level Library op with a keyword-only float, registers a CPU
        # kernel and autograd through torch.library.register_autograd; verifies setup_context
        # receives the keyword-only inputs as a mapping, backward runs, and the gradient equals the
        # keyword value. Skipped under Dynamo.
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            lib.define("foo(Tensor x, *, float y) -> Tensor")
            called = False

            def foo_impl(x, *, y):
                # [FUNCTION DOC] HELPER: CPU kernel returning x multiplied by the keyword-only y.
                return x * y

            lib.impl("foo", foo_impl, "CPU")

            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward flagging that it ran and returning grad
                # times ctx.y.
                nonlocal called
                called = True
                return grad * ctx.y

            def setup_context(ctx, inputs, keyword_only_inputs, output):
                # [FUNCTION DOC] HELPER: Setup hook; raises AssertionError unless the keyword-only
                # inputs hold exactly the key y, then stores its value on ctx.
                if tuple(keyword_only_inputs.keys()) != ("y",):
                    raise AssertionError(
                        f"expected keyword_only_inputs.keys() == ('y',), got {tuple(keyword_only_inputs.keys())}"
                    )
                ctx.y = keyword_only_inputs["y"]

            torch.library.register_autograd(
                "_torch_testing::foo", backward, setup_context=setup_context, lib=lib
            )

            x = torch.randn(3, requires_grad=True)
            torch.ops._torch_testing.foo(x, y=3.14).sum().backward()
            self.assertTrue(called)
            self.assertEqual(x.grad, torch.tensor([3.14, 3.14, 3.14]))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_defaults(self):
        # [FUNCTION DOC] Defines an op with a defaulted positional int and defaulted keyword-only
        # ints; verifies setup_context sees the defaults filled in (two positional inputs and the
        # keyword mapping with the passed value) and the gradient is the product of the factors.
        # Skipped under Dynamo.
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            lib.define("foo(Tensor w, int x = 2, *, int y = 3, int z) -> Tensor")

            def foo_impl(w, x=2, *, y=3, z):
                # [FUNCTION DOC] HELPER: CPU kernel returning the product of w, x, y and z.
                return w * x * y * z

            lib.impl("foo", foo_impl, "CPU")

            called = False

            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward flagging that it ran and returning grad
                # times ctx.c.
                nonlocal called
                called = True
                return grad * ctx.c

            def setup_context(ctx, inputs, keyword_only_inputs, output):
                # [FUNCTION DOC] HELPER: Setup hook; raises AssertionError unless the positional
                # inputs and the keyword-only mapping have the expected defaulted contents, then
                # stores their product.
                if len(inputs) != 2:
                    raise AssertionError(
                        f"expected len(inputs) == 2, got {len(inputs)}"
                    )
                if inputs[1] != 2:
                    raise AssertionError(f"expected inputs[1] == 2, got {inputs[1]}")
                if keyword_only_inputs != {"y": 3, "z": 42}:
                    raise AssertionError(
                        f"expected keyword_only_inputs == {{'y': 3, 'z': 42}}, got {keyword_only_inputs}"
                    )
                ctx.c = keyword_only_inputs["y"] * keyword_only_inputs["z"] * inputs[1]

            torch.library.register_autograd(
                "_torch_testing::foo", backward, setup_context=setup_context, lib=lib
            )

            w = torch.randn(3, requires_grad=True)
            torch.ops._torch_testing.foo(w, z=42).sum().backward()
            self.assertTrue(called)
            self.assertEqual(w.grad, torch.full_like(w, 2 * 3 * 42))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_manual_schema_error(self):
        # [FUNCTION DOC] Verifies an explicit schema that mutates x with mutates_args left empty
        # raises ValueError saying the op mutates x. Skipped under Dynamo.
        with self.assertRaisesRegex(ValueError, "the op mutates {'x'}"):

            @torch.library.custom_op(
                "_torch_testing::sin_",
                mutates_args=(),
                schema="(Tensor(a!) x) -> ()",
            )
            def sin_(x):
                # [FUNCTION DOC] HELPER: Applies sine to x in place through numpy.
                x_np = x.numpy()
                np.sin(x_np, out=x_np)

    def test_supports_tensorlist(self):
        # [FUNCTION DOC] Tests the supports_tensorlist wrapper for autograd Functions taking a list
        # of tensors: multiple independent applies, multiple backwards with retained graph,
        # NotImplementedError for direct forward/backward calls, and recursive use of the wrapped
        # function in forward and in backward (checking the resulting gradients).
        @torch._library.autograd.supports_tensorlist
        class Stack(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function stacking a list of tensors; backward checks
            # needs_input_grad and returns the unbound gradients as a list.
            @staticmethod
            def forward(ctx, xs):
                # [FUNCTION DOC] HELPER: Saves the number of inputs and returns their stack.
                ctx.num_xs = len(xs)
                return torch.stack(xs)

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Asserts needs_input_grad is a one-entry tuple of an
                # all-True list and returns the gradient split along dim 0.
                expected = ([True] * ctx.num_xs,)
                self.assertEqual(ctx.needs_input_grad, expected)
                return list(grad.unbind(0))

        # call two applies, do a backward on the first
        def t():
            # [FUNCTION DOC] HELPER: Returns a scalar random tensor requiring grad.
            return torch.randn([], requires_grad=True)

        xs0 = [t(), t(), t()]
        xs1 = [t(), t(), t(), t()]
        y0 = Stack.apply(xs0)
        y1 = Stack.apply(xs1)
        grads = torch.autograd.grad(y0.sum(), xs0)
        self.assertEqual(grads, [torch.tensor(1.0) for _ in range(3)])

        # call one apply, do multiple backwards
        xs = [t(), t(), t()]
        y = Stack.apply(xs)
        _ = torch.autograd.grad(y.sum(), xs, retain_graph=True)
        _ = torch.autograd.grad(y.sum(), xs, retain_graph=True)
        grads = torch.autograd.grad(y.sum(), xs, retain_graph=True)
        self.assertEqual(grads, [torch.tensor(1.0) for _ in range(3)])

        # error: on access forward, backward directly
        with self.assertRaisesRegex(NotImplementedError, "Function.forward directly"):
            Stack.forward(None, xs)
        with self.assertRaisesRegex(NotImplementedError, "Function.backward directly"):
            Stack.backward(None, xs)

        # the recursive case
        @torch._library.autograd.supports_tensorlist
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose forward recursively applies itself to all but the
            # first list element; only the last element's sine is the result.
            @staticmethod
            def forward(ctx, xs):
                # [FUNCTION DOC] HELPER: Recurses on the tail when longer than one, else saves the
                # length and returns the sine of the single element.
                if len(xs) > 1:
                    return Foo.apply(xs[1:])
                ctx.len_xs = len(xs)
                return xs[0].sin()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Returns a list of None with the last entry set to the
                # cosine of grad.
                result = [None] * ctx.len_xs
                result[-1] = grad.cos()
                return result

        # should work
        result = Foo.apply(xs)
        expected = xs[-1].sin()
        self.assertEqual(result, expected)

        # recursive on backward
        @torch._library.autograd.supports_tensorlist
        class Bar(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function that adds each element's index to it and, in backward,
            # applies itself recursively to two slices of the gradients.
            @staticmethod
            def forward(ctx, xs):
                # [FUNCTION DOC] HELPER: Returns a list of each element plus its index.
                return [xs[i] + i for i in range(len(xs))]

            @staticmethod
            def backward(ctx, grads):
                # [FUNCTION DOC] HELPER: Applies the function to the first two and the remaining
                # gradients and returns the concatenated lists.
                f1 = Bar.apply(grads[:2])
                f2 = Bar.apply(grads[2:])
                return f1 + f2

        xs = [torch.tensor(0.0, requires_grad=True) for _ in range(5)]
        ys = Bar.apply(xs)
        sum(ys).backward()
        result = [xi.grad for xi in xs]
        self.assertEqual(result, torch.tensor([1.0, 2, 1, 2, 3]).unbind(0))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_default_values(self):
        # [FUNCTION DOC] Defines a custom op with many defaulted arguments (none, float, bool, int,
        # str, dtypes, devices), checks the Python function receives those defaults and that the
        # schema records default values (dtypes as enum integers). Skipped under Dynamo.
        defaults = []

        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(
            x: Tensor,
            a: Optional[int] = None,
            b: float = 3.14,
            c: bool = True,
            d: int = 3,
            e: str = "foo",
            f: torch.dtype = torch.float,
            g: torch.dtype = torch.float32,
            h: torch.dtype = torch.int,
            i: torch.device = torch.device("cpu:0"),
            j: torch.device = "cpu",
        ) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body appending all defaulted arguments to the enclosing
            # list and returning a clone of x.
            defaults.extend([a, b, c, d, e, f, g, h, i, j])
            return x.clone()

        x = torch.randn(3)
        f(x)
        self.assertEqual(
            defaults,
            [
                None,
                3.14,
                True,
                3,
                "foo",
                torch.float,
                torch.float32,
                torch.int,
                torch.device("cpu:0"),
                "cpu",
            ],
        )
        default_values = [
            arg.default_value
            for arg in torch.ops._torch_testing.f.default._schema.arguments
        ]
        # enum values taken from c10/core/ScalarType.h
        type_enum = {
            "float": 6,
            "int": 3,
        }
        self.assertEqual(
            default_values,
            [
                None,
                None,
                3.14,
                True,
                3,
                "foo",
                type_enum["float"],
                type_enum["float"],
                type_enum["int"],
                torch.device("cpu:0"),
                torch.device("cpu"),
            ],
        )

    def test_mutated_error(self):
        # [FUNCTION DOC] Verifies mutates_args naming a nonexistent parameter raises ValueError.
        with self.assertRaisesRegex(
            ValueError, r".*{'y'} in mutates_args were not found"
        ):

            @torch.library.custom_op(
                "_torch_testing::numpy_sin_inplace",
                mutates_args={"y"},
                device_types="cpu",
            )
            def numpy_sin_inplace(x: Tensor) -> None:
                # [FUNCTION DOC] HELPER: Applies sine to x in place through numpy.
                x_np = x.numpy()
                np.sin(x_np, out=x_np)

    def test_mutated(self):
        # [FUNCTION DOC] Verifies mutated arguments get their version counters bumped: a simple op
        # mutating x, and an op declaring tensor, optional, list and list-of-optional mutated
        # arguments, where the unmutated first argument keeps its version.
        @torch.library.custom_op(
            "_torch_testing::numpy_sin_inplace", mutates_args={"x"}, device_types="cpu"
        )
        def numpy_sin_inplace(x: Tensor) -> None:
            # [FUNCTION DOC] HELPER: Applies sine to x in place through numpy.
            x_np = x.numpy()
            np.sin(x_np, out=x_np)

        x = torch.randn(3)
        version = x._version
        expected = x.sin()
        numpy_sin_inplace(x)
        self.assertEqual(x, expected)
        self.assertGreater(x._version, version)

        @torch.library.custom_op("_torch_testing::f", mutates_args={"y", "z", "w"})
        def f(
            x: Tensor, y: Optional[Tensor], z: List[Tensor], w: List[Optional[Tensor]]
        ) -> None:
            # [FUNCTION DOC] HELPER: Op body doing nothing; its mutated-argument declaration drives
            # the check.
            return

        x = torch.randn(3)
        y = torch.randn(3)
        z = [torch.randn(3), torch.randn(3)]
        w = [torch.randn(3), None, torch.randn(3)]
        initial_versions = pytree.tree_map_only(
            torch.Tensor, lambda x: x._version, (x, y, z, w)
        )
        f(x, y, z, w)
        new_versions = pytree.tree_map_only(
            torch.Tensor, lambda x: x._version, (x, y, z, w)
        )

        self.assertEqual(initial_versions[0], new_versions[0])
        initial_versions, _ = pytree.tree_flatten(initial_versions[1:])
        new_versions, _ = pytree.tree_flatten(new_versions[1:])
        for prev, after in zip(initial_versions, new_versions):
            if prev is None and after is None:
                continue
            self.assertGreater(after, prev)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_mutated_optional_arg_default_none(self):
        # [FUNCTION DOC] Verifies an op mutating an optional out argument defaulting to None
        # works with and without it, and bumps the version of out when it is passed. Skipped
        # under Dynamo.
        @torch.library.custom_op(
            "_torch_testing::copy_optional_out", mutates_args={"out"}
        )
        def copy_optional_out(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
            # [FUNCTION DOC] HELPER: Copies x into out and returns an empty tensor if out is given,
            # else returns a clone of x.
            if out is not None:
                out.copy_(x)
                return x.new_empty(0)
            return x.clone()

        x = torch.randn(3)
        self.assertEqual(copy_optional_out(x), x)

        out = torch.empty_like(x)
        version = out._version
        result = copy_optional_out(x, out=out)
        self.assertEqual(result.numel(), 0)
        self.assertEqual(out, x)
        self.assertGreater(out._version, version)

    def test_mutated_version_bump_does_not_fill_all_defaults(self):
        # [FUNCTION DOC] Verifies the version bump of mutated arguments does not call fill_defaults
        # (the function is patched to raise AssertionError) for a keyword-call op, an out-tagged op
        # and an op with a defaulted optional out; versions of mutated tensors increase.
        @torch.library.custom_op(
            "_torch_testing::mutated_fastpath", mutates_args={"x", "ys"}
        )
        def mutated_fastpath(
            x: Tensor,
            a: Tensor,
            b: Tensor,
            c: Tensor,
            d: Tensor,
            ys: List[Tensor],
        ) -> None:
            # [FUNCTION DOC] HELPER: Op body doing nothing (mutates x and ys by declaration).
            pass

        @torch.library.custom_op(
            "_torch_testing::mutated_fastpath_out",
            mutates_args={"out"},
            tags=torch.Tag.out,
        )
        def mutated_fastpath_out(x: Tensor, *, out: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Copies x into out and returns out.
            out.copy_(x)
            return out

        @torch.library.custom_op(
            "_torch_testing::mutated_fastpath_default", mutates_args={"out"}
        )
        def mutated_fastpath_default(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns a clone of x, or copies x into out and returns out
            # when given.
            if out is None:
                return x.clone()
            out.copy_(x)
            return out

        x = torch.randn(3)
        args = [torch.randn(3) for _ in range(4)]
        ys = [torch.randn(3), torch.randn(3)]
        out = torch.empty_like(x)

        initial_versions = pytree.tree_map_only(
            torch.Tensor, lambda t: t._version, (x, ys, out)
        )
        with patch(
            "torch._library.utils.fill_defaults",
            side_effect=AssertionError("unexpected fill_defaults"),
        ):
            mutated_fastpath(
                x=x,
                a=args[0],
                b=args[1],
                c=args[2],
                d=args[3],
                ys=ys,
            )
            mutated_fastpath_out(x, out=out)
            self.assertEqual(mutated_fastpath_default(x), x)
        new_versions = pytree.tree_map_only(
            torch.Tensor, lambda t: t._version, (x, ys, out)
        )

        self.assertGreater(new_versions[0], initial_versions[0])
        for new_version, initial_version in zip(new_versions[1], initial_versions[1]):
            self.assertGreater(new_version, initial_version)
        self.assertGreater(new_versions[2], initial_versions[2])

    def test_mutated_no_warning(self):
        # [FUNCTION DOC] Runs a script in a subprocess (the warning is emitted only once per
        # process) that defines a mutating custom op with warnings always recorded; the test fails
        # with the subprocess output if any warning was emitted.
        # Run in subprocess since the warning is emitted only once
        script = """\
import warnings
import torch
from torch import Tensor

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    torch.set_warn_always(True)

    @torch.library.custom_op("mylib::func", mutates_args=("x",))
    def func(x: Tensor) -> None:
        x.add_(1)

    if len(w) > 0:
        raise AssertionError(f"Unexpected warning: {w[0].message}")
"""
        try:
            subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            self.fail(e.output.decode("utf-8"))

    def test_mutated_unknown(self):
        # [FUNCTION DOC] Verifies mutates_args set to the string unknown bumps versions of all
        # tensor arguments (single and nested lists with optional entries) and that any other
        # string raises ValueError.
        @torch.library.custom_op(
            "_torch_testing::f", mutates_args="unknown", device_types="cpu"
        )
        def f(x: Tensor) -> None:
            # [FUNCTION DOC] HELPER: Applies sine to x in place through numpy.
            x_np = x.numpy()
            np.sin(x_np, out=x_np)

        x = torch.randn(3)
        version = x._version
        expected = x.sin()
        f(x)
        self.assertEqual(x, expected)
        self.assertGreater(x._version, version)

        @torch.library.custom_op("_torch_testing::f2", mutates_args="unknown")
        def f2(
            x: Tensor, y: Optional[Tensor], z: List[Tensor], w: List[Optional[Tensor]]
        ) -> None:
            # [FUNCTION DOC] HELPER: Op body doing nothing.
            return

        x = torch.randn(3)
        y = torch.randn(3)
        z = [torch.randn(3), torch.randn(3)]
        w = [torch.randn(3), None, torch.randn(3)]
        initial_versions = pytree.tree_map_only(
            torch.Tensor, lambda x: x._version, (x, y, z, w)
        )
        f2(x, y, z, w)
        new_versions = pytree.tree_map_only(
            torch.Tensor, lambda x: x._version, (x, y, z, w)
        )

        initial_versions, _ = pytree.tree_flatten(initial_versions)
        new_versions, _ = pytree.tree_flatten(new_versions)
        for prev, after in zip(initial_versions, new_versions):
            if prev is None and after is None:
                continue
            self.assertGreater(after, prev)

        with self.assertRaisesRegex(ValueError, "string"):

            @torch.library.custom_op("_torch_testing::f3", mutates_args="x")
            def f3(x: Tensor) -> None:
                # [FUNCTION DOC] HELPER: Op body doing nothing; creation expected to fail.
                return

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_torch_dispatch_rule_subclass(self):
        # [FUNCTION DOC] Registers a torch dispatch rule for a subclass (TwoTensor) on a custom op
        # via the Library and verifies it is called once for the op, not for unrelated ops. Skipped
        # under Dynamo.
        from torch.testing._internal.two_tensor import TwoTensor

        @torch.library.custom_op("mylib::foo", mutates_args={})
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Op body returning the sine of x.
            return x.sin()

        x = torch.randn(3)
        y = torch.randn(3)
        z = TwoTensor(x, y)

        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            called = 0

            def TwoTensor_foo(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch rule asserting the class is TwoTensor, counting
                # the call and returning the sine of the captured x.
                nonlocal called
                if cls is not TwoTensor:
                    raise AssertionError(f"expected cls is TwoTensor, got {cls}")
                called += 1
                return x.sin()

            m._register_torch_dispatch_rule("foo", TwoTensor, TwoTensor_foo)

            out = f(z)
            out2 = z.cos()

        self.assertEqual(called, 1)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_torch_dispatch_rule_mode(self):
        # [FUNCTION DOC] Registers a torch dispatch rule for a dispatch mode (TwoTensorMode)
        # on a custom op and verifies it runs once inside the mode for the op only. Skipped
        # under Dynamo.
        from torch.testing._internal.two_tensor import TwoTensorMode

        @torch.library.custom_op("mylib::foo", mutates_args={})
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Op body returning the sine of x.
            return x.sin()

        x = torch.randn(3)

        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            called = 0

            def TwoTensor_foo(mode, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch rule counting the call and returning the sine of
                # the captured x.
                nonlocal called
                called += 1
                return x.sin()

            m._register_torch_dispatch_rule("foo", TwoTensorMode, TwoTensor_foo)

            with TwoTensorMode():
                out = f(x)
                out2 = x.cos()

        self.assertEqual(called, 1)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @parametrize("idx", [0, 1, 2, 3, 4, 5])
    def test_library_register_fake_source(self, idx):
        # [FUNCTION DOC] Parametrized over six test ops: verifies the recorded fake kernel source
        # location of the op in the test namespace is known and mentions custom_op_db.py. Skipped
        # under Dynamo.
        opname = f"source{idx}"
        op = getattr(torch.ops._torch_testing, opname).default
        entry = torch._library.simple_registry.singleton.find(op._name)
        source = entry.fake_impl.kernel.source
        if source is None:
            raise AssertionError("expected source to be not None")
        self.assertTrue("custom_op_db.py" in source)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_fake(self):
        # [FUNCTION DOC] Loops over registering a fake kernel via the op function, qualified name
        # and overload object; each time verifies the decorator is returned, the fake kernel is
        # used under FakeTensorMode and the output shape matches. Skipped under Dynamo.
        for mode in ["function", "qualname", "opoverload"]:

            @torch.library.custom_op("_torch_testing::add", mutates_args=())
            def add(x: Tensor, y: float) -> Tensor:
                # [FUNCTION DOC] HELPER: Op body adding y to x through numpy.
                x_np = x.cpu().numpy()
                out_np = x_np + y
                return torch.from_numpy(out_np).to(x.device)

            called = False

            if mode == "function":
                dec = torch.library.register_fake(add)
                self.assertIsNotNone(dec)
            elif mode == "qualname":
                dec = torch.library.register_fake("_torch_testing::add")
                self.assertIsNotNone(dec)
            elif mode == "opoverload":
                dec = torch.library.register_fake(torch.ops._torch_testing.add.default)
                self.assertIsNotNone(dec)
            else:
                raise AssertionError("should not get here")

            @dec
            def _(x, y):
                # [FUNCTION DOC] HELPER: Fake kernel setting the enclosing flag and returning an
                # empty tensor.
                nonlocal called
                called = True
                return torch.empty_like(x)

            with torch._subclasses.fake_tensor.FakeTensorMode():
                x = torch.randn(3)
                y = 3.14
                z = add(x, y)
                self.assertEqual(z.shape, x.shape)
                self.assertTrue(called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_torch_dispatch(self):
        # [FUNCTION DOC] Loops over registering a torch dispatch rule for a mode class via op
        # function, qualified name and overload; checks the rule runs when the op is called inside
        # the mode. Skipped under Dynamo.
        for mode in ["function", "qualname", "opoverload"]:

            class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
                # [CLASS DOC] HELPER: Dispatch mode that just forwards every op unchanged.
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    # [FUNCTION DOC] HELPER: Calls the function with its args and kwargs.
                    return func(*args, **kwargs)

            @torch.library.custom_op("_torch_testing::add", mutates_args=())
            def add(x: Tensor, y: float) -> Tensor:
                # [FUNCTION DOC] HELPER: Op body adding y to x through numpy.
                x_np = x.cpu().numpy()
                out_np = x_np + y
                return torch.from_numpy(out_np).to(x.device)

            called = False

            if mode == "function":
                dec = torch.library.register_torch_dispatch(add, MyMode)
                self.assertIsNotNone(dec)
            elif mode == "qualname":
                dec = torch.library.register_torch_dispatch(
                    "_torch_testing::add", MyMode
                )
                self.assertIsNotNone(dec)
            elif mode == "opoverload":
                dec = torch.library.register_torch_dispatch(
                    torch.ops._torch_testing.add.default, MyMode
                )
                self.assertIsNotNone(dec)
            else:
                raise AssertionError("should not get here")

            @dec
            def _(mode, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch rule setting the enclosing flag and forwarding
                # the call.
                nonlocal called
                called = True
                return func(*args, **kwargs)

            with MyMode():
                x = torch.randn(3)
                y = 3.14
                z = add(x, y)
                self.assertEqual(z.shape, x.shape)
                self.assertTrue(called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_torch_dispatch_low_level(self):
        # [FUNCTION DOC] Combinatorial test of register_torch_dispatch on a Library-defined op
        # across op spec forms (qualified name, overload), decorator or function style, and device
        # type options; verifies the rule adds the inputs when called inside the mode. Skipped
        # under Dynamo.
        modes = ["qualname", "opoverload"]
        calls = ["decorator", "function"]
        device_types_options = [("cpu", "cuda"), "cpu", None]

        for mode, call, device_types in itertools.product(
            modes, calls, device_types_options
        ):
            with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
                lib.define("add10(Tensor x, float y) -> Tensor")

                if mode == "qualname":
                    op = "_torch_testing::add10"
                else:
                    if mode != "opoverload":
                        raise AssertionError(
                            f"expected mode == 'opoverload', got {mode!r}"
                        )
                    op = torch.ops._torch_testing.add10.default

                called = False

                class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
                    # [CLASS DOC] HELPER: Dispatch mode that just forwards every op unchanged.
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        # [FUNCTION DOC] HELPER: Calls the function with its args and kwargs.
                        return func(*args, **kwargs)

                if call == "decorator":

                    @torch.library.register_torch_dispatch(op, MyMode, lib=lib)
                    def _(mode, func, types, args, kwargs):
                        # [FUNCTION DOC] HELPER: Dispatch rule setting the enclosing flag and
                        # returning x plus y.
                        x, y = args
                        nonlocal called
                        called = True
                        return x + y

                else:
                    if call != "function":
                        raise AssertionError(
                            f"expected call == 'function', got {call!r}"
                        )

                    def add_stuff(mode, func, types, args, kwargs):
                        # [FUNCTION DOC] HELPER: Dispatch rule setting the enclosing flag and
                        # returning x plus y.
                        x, y = args
                        nonlocal called
                        called = True
                        return x + y

                    torch.library.register_torch_dispatch(
                        op, MyMode, add_stuff, lib=lib
                    )

                x = torch.randn(3)
                y = 3.14
                with MyMode():
                    z = torch.ops._torch_testing.add10.default(x, y)
                self.assertEqual(z, x + y)
                self.assertTrue(called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_kernel(self):
        # [FUNCTION DOC] Combinatorial test of register_kernel on a custom op defined for CUDA
        # across op spec forms (function, name, overload), decorator or function style, and CPU or
        # default device types; verifies the registered kernel is called and result equals the sum.
        # Skipped under Dynamo.
        modes = ["function", "qualname", "opoverload"]
        calls = ["decorator", "function"]
        device_types_options = ["cpu", None]

        for mode, call, device_types in itertools.product(
            modes, calls, device_types_options
        ):

            @torch.library.custom_op(
                "_torch_testing::add", mutates_args=(), device_types="cuda"
            )
            def add(x: Tensor, y: float) -> Tensor:
                # [FUNCTION DOC] HELPER: Op body adding y to x through numpy on CPU.
                x_np = x.cpu().numpy()
                out_np = x_np + y
                return torch.from_numpy(out_np).to(x.device)

            if mode == "function":
                op = add
            elif mode == "qualname":
                op = "_torch_testing::add"
            else:
                if mode != "opoverload":
                    raise AssertionError(f"expected mode == 'opoverload', got {mode!r}")
                op = torch.ops._torch_testing.add.default

            called = False

            if call == "decorator":

                @torch.library.register_kernel(op, device_types)
                def _(x, y):
                    # [FUNCTION DOC] HELPER: CPU kernel setting the enclosing flag and adding y
                    # via numpy.
                    nonlocal called
                    called = True
                    x_np = x.numpy()
                    out_np = x_np + y
                    return torch.from_numpy(out_np)

            else:
                if call != "function":
                    raise AssertionError(f"expected call == 'function', got {call!r}")

                def add_cpu(x, y):
                    # [FUNCTION DOC] HELPER: CPU kernel setting the enclosing flag and adding y
                    # via numpy.
                    nonlocal called
                    called = True
                    x_np = x.numpy()
                    out_np = x_np + y
                    return torch.from_numpy(out_np)

                torch.library.register_kernel(op, device_types, add_cpu)

            x = torch.randn(3)
            y = 3.14
            z = add(x, y)
            self.assertEqual(z, x + y)
            self.assertTrue(called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_kernel_low_level(self):
        # [FUNCTION DOC] Combinatorial test of register_kernel on a Library-defined op across op
        # spec forms, decorator or function style and device type options (including a tuple);
        # verifies the kernel is called. Skipped under Dynamo.
        modes = ["qualname", "opoverload"]
        calls = ["decorator", "function"]
        device_types_options = [("cpu", "cuda"), "cpu", None]

        for mode, call, device_types in itertools.product(
            modes, calls, device_types_options
        ):
            with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
                lib.define("add9(Tensor x, float y) -> Tensor")

                if mode == "qualname":
                    op = "_torch_testing::add9"
                else:
                    if mode != "opoverload":
                        raise AssertionError(
                            f"expected mode == 'opoverload', got {mode!r}"
                        )
                    op = torch.ops._torch_testing.add9.default

                called = False

                if call == "decorator":

                    @torch.library.register_kernel(op, device_types, lib=lib)
                    def _(x, y):
                        # [FUNCTION DOC] HELPER: CPU kernel setting the enclosing flag and adding y
                        # via numpy.
                        nonlocal called
                        called = True
                        x_np = x.numpy()
                        out_np = x_np + y
                        return torch.from_numpy(out_np)

                else:
                    if call != "function":
                        raise AssertionError(
                            f"expected call == 'function', got {call!r}"
                        )

                    def add_cpu(x, y):
                        # [FUNCTION DOC] HELPER: CPU kernel setting the enclosing flag and adding y
                        # via numpy.
                        nonlocal called
                        called = True
                        x_np = x.numpy()
                        out_np = x_np + y
                        return torch.from_numpy(out_np)

                    torch.library.register_kernel(op, device_types, add_cpu, lib=lib)

                x = torch.randn(3)
                y = 3.14
                z = torch.ops._torch_testing.add9.default(x, y)
                self.assertEqual(z, x + y)
                self.assertTrue(called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_ACCELERATOR, "requires accelerator.")
    def test_library_register_autocast(self):
        # [FUNCTION DOC] For the current accelerator and CPU, and for registering via function,
        # name and overload, registers autocast to float16 for a custom op and checks that the op
        # output under autocast is float16 for a float32 input. Needs an accelerator; skipped
        # under Dynamo.
        for device in [torch.accelerator.current_accelerator().type, "cpu"]:
            for mode in ["function", "qualname", "opoverload"]:

                @torch.library.custom_op("mylib::my_sin", mutates_args=())
                def my_sin(x: Tensor) -> Tensor:
                    # [FUNCTION DOC] HELPER: Op body returning the sine of x.
                    return torch.sin(x)

                if mode == "function":
                    torch.library.register_autocast(my_sin, device, torch.float16)
                elif mode == "qualname":
                    torch.library.register_autocast(
                        "mylib::my_sin", device, torch.float16
                    )
                elif mode == "opoverload":
                    torch.library.register_autocast(
                        torch.ops.mylib.my_sin.default, device, torch.float16
                    )

                x = torch.randn(3, dtype=torch.float32, device=device)
                with torch.autocast(device, dtype=torch.float16):
                    y = torch.ops.mylib.my_sin(x)
                self.assertEqual(y.dtype, torch.float16)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_ACCELERATOR, "requires accelerator.")
    def test_library_register_autocast_low_level(self):
        # [FUNCTION DOC] Same autocast check for a Library-defined op with an impl on the device
        # key, registering via name and overload with the lib argument; output must be float16.
        # Needs an accelerator; skipped under Dynamo.
        for device in [torch.accelerator.current_accelerator().type, "cpu"]:
            for mode in ["qualname", "opoverload"]:
                with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
                    lib.define("my_sin(Tensor x) -> Tensor")

                    def my_sin(x: Tensor) -> Tensor:
                        # [FUNCTION DOC] HELPER: Kernel returning the sine of x.
                        return torch.sin(x)

                    lib.impl("my_sin", my_sin, device.upper())

                    if mode == "qualname":
                        torch.library.register_autocast(
                            "_torch_testing::my_sin", device, torch.float16, lib=lib
                        )
                    elif mode == "opoverload":
                        torch.library.register_autocast(
                            torch.ops._torch_testing.my_sin.default,
                            device,
                            torch.float16,
                            lib=lib,
                        )

                    x = torch.randn(3, dtype=torch.float32, device=device)
                    with torch.autocast(device, dtype=torch.float16):
                        y = torch.ops._torch_testing.my_sin(x)
                    self.assertEqual(y.dtype, torch.float16)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_ACCELERATOR, "requires accelerator.")
    def test_library_register_autocast_list_input(self):
        # [FUNCTION DOC] Autocast registration for an op taking a list of tensors: with each
        # registration form and device, the output under autocast is float16. Needs an accelerator;
        # skipped under Dynamo.
        for device in [torch.accelerator.current_accelerator().type, "cpu"]:
            for mode in ["function", "qualname", "opoverload"]:

                @torch.library.custom_op("mylib::my_add_sin", mutates_args=())
                def my_add_sin(x: List[Tensor]) -> Tensor:
                    # [FUNCTION DOC] HELPER: Op body returning the sine of the sum of the first two
                    # list elements.
                    return torch.sin(x[0] + x[1])

                if mode == "function":
                    torch.library.register_autocast(my_add_sin, device, torch.float16)
                elif mode == "qualname":
                    torch.library.register_autocast(
                        "mylib::my_add_sin", device, torch.float16
                    )
                elif mode == "opoverload":
                    torch.library.register_autocast(
                        torch.ops.mylib.my_add_sin.default, device, torch.float16
                    )

                lst = [
                    torch.randn(3, dtype=torch.float32, device=device) for _ in range(2)
                ]
                with torch.autocast(device, dtype=torch.float16):
                    y = torch.ops.mylib.my_add_sin(lst)
                self.assertEqual(y.dtype, torch.float16)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_ACCELERATOR, "requires accelerator.")
    def test_library_register_autocast_multiple_times(self):
        # [FUNCTION DOC] Registers autocast twice for the same op and device and checks float16
        # output after each registration, verifying repeated registration does not error. Needs an
        # accelerator; skipped under Dynamo.
        for device in [torch.accelerator.current_accelerator().type, "cpu"]:

            @torch.library.custom_op("mylib::my_sin", mutates_args=())
            def my_sin(x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Op body returning the sine of x.
                return torch.sin(x)

            torch.library.register_autocast(my_sin, device, torch.float16)

            x = torch.randn(3, dtype=torch.float32, device=device)
            with torch.autocast(device, dtype=torch.float16):
                y1 = my_sin(x)
            self.assertEqual(y1.dtype, torch.float16)

            # Ensure calling register_autocast multiple times does not error out.
            torch.library.register_autocast(my_sin, device, torch.float16)

            with torch.autocast(device, dtype=torch.float16):
                y2 = my_sin(x)
            self.assertEqual(y2.dtype, torch.float16)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_ACCELERATOR, "requires accelerator.")
    def test_library_register_autocast_multiple_times_different_devices(self):
        # [FUNCTION DOC] Registers autocast for the accelerator and CPU, each twice, checking
        # float16 output each time, then nests both autocast contexts to verify no infinite
        # redispatch. Needs an accelerator; skipped under Dynamo.
        @torch.library.custom_op("mylib::my_sin", mutates_args=())
        def my_sin(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning the sine of x.
            return torch.sin(x)

        # Register autocast for the current accelerator
        device_type = torch.accelerator.current_accelerator().type
        torch.library.register_autocast(my_sin, device_type, torch.float16)

        x1 = torch.randn(3, dtype=torch.float32, device=device_type)
        with torch.autocast(device_type, dtype=torch.float16):
            y1 = my_sin(x1)
        self.assertEqual(y1.dtype, torch.float16)

        # Register autocast for CPU
        torch.library.register_autocast(my_sin, "cpu", torch.float16)

        x2 = torch.randn(3, dtype=torch.float32, device="cpu")
        with torch.autocast("cpu", dtype=torch.float16):
            y2 = my_sin(x2)
        self.assertEqual(y2.dtype, torch.float16)

        # Register accelerator autocast for the second time
        torch.library.register_autocast(my_sin, device_type, torch.float16)

        with torch.autocast(device_type, dtype=torch.float16):
            y3 = my_sin(x1)
        self.assertEqual(y3.dtype, torch.float16)

        # Register CPU autocast for the second time
        torch.library.register_autocast(my_sin, "cpu", torch.float16)

        with torch.autocast("cpu", dtype=torch.float16):
            y4 = my_sin(x2)
        self.assertEqual(y4.dtype, torch.float16)

        # Nested autocast: verifies no infinite redispatch when both keys are active.
        with (
            torch.autocast("cpu", dtype=torch.float16),
            torch.autocast(device_type, dtype=torch.float16),
        ):
            y5 = my_sin(x1)
            y6 = my_sin(x2)
        self.assertEqual(y5.dtype, torch.float16)
        self.assertEqual(y6.dtype, torch.float16)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_autograd(self):
        # [FUNCTION DOC] Registers autograd for a custom op via function, name and overload;
        # verifies the backward function runs and the gradient equals the cosine of the input.
        # Skipped under Dynamo.
        for mode in ["function", "qualname", "opoverload"]:

            @torch.library.custom_op("mylib::numpy_sin", mutates_args=())
            def numpy_sin(x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Op body computing the sine through numpy and returning it
                # on the device of x.
                x_np = x.cpu().numpy()
                y_np = np.sin(x_np)
                return torch.from_numpy(y_np).to(device=x.device)

            def setup_context(ctx, inputs, output) -> Tensor:
                # [FUNCTION DOC] HELPER: Setup hook saving the single input for backward.
                (x,) = inputs
                ctx.save_for_backward(x)

            called = False

            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward flagging that it ran and returning grad times
                # cosine of the saved input.
                nonlocal called
                called = True
                (x,) = ctx.saved_tensors
                return grad * x.cos()

            if mode == "function":
                torch.library.register_autograd(
                    numpy_sin, backward, setup_context=setup_context
                )
            elif mode == "qualname":
                torch.library.register_autograd(
                    "mylib::numpy_sin", backward, setup_context=setup_context
                )
            elif mode == "opoverload":
                torch.library.register_autograd(
                    torch.ops.mylib.numpy_sin.default,
                    backward,
                    setup_context=setup_context,
                )

            x = torch.randn(3, requires_grad=True)
            y = numpy_sin(x)
            (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
            self.assertTrue(called)
            self.assertEqual(grad_x, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_autograd_low_level(self):
        # [FUNCTION DOC] Registers autograd for a Library-defined op via name and overload with
        # the lib argument; verifies backward runs and the gradient equals the cosine. Skipped
        # under Dynamo.
        for mode in ["qualname", "opoverload"]:
            with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
                lib.define("sin5(Tensor x) -> Tensor")

                def numpy_sin(x: Tensor) -> Tensor:
                    # [FUNCTION DOC] HELPER: CPU kernel computing the sine through numpy on a
                    # detached CPU copy.
                    x_np = x.cpu().detach().numpy()
                    y_np = np.sin(x_np)
                    return torch.from_numpy(y_np).to(device=x.device)

                def setup_context(ctx, inputs, output) -> Tensor:
                    # [FUNCTION DOC] HELPER: Setup hook saving the single input for backward.
                    (x,) = inputs
                    ctx.save_for_backward(x)

                called = False

                def backward(ctx, grad):
                    # [FUNCTION DOC] HELPER: Backward flagging that it ran and returning grad times
                    # cosine of the saved input.
                    nonlocal called
                    called = True
                    (x,) = ctx.saved_tensors
                    return grad * x.cos()

                lib.impl("sin5", numpy_sin, "CPU")

                called = False

                if mode == "qualname":
                    torch.library.register_autograd(
                        "_torch_testing::sin5",
                        backward,
                        setup_context=setup_context,
                        lib=lib,
                    )
                elif mode == "opoverload":
                    torch.library.register_autograd(
                        torch.ops._torch_testing.sin5.default,
                        backward,
                        setup_context=setup_context,
                        lib=lib,
                    )
                x = torch.randn(3, requires_grad=True)
                y = torch.ops._torch_testing.sin5(x)
                (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
                self.assertTrue(called)
                self.assertEqual(grad_x, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_fake(self):
        # [FUNCTION DOC] Verifies the missing-fake-implementation behavior: under FakeTensorMode
        # the op raises RuntimeError with an expected message (address normalized), compiled use
        # raises RuntimeError (not on Windows); after register_fake the fake kernel is used with
        # the right shape. Skipped under Dynamo.
        @torch.library.custom_op("_torch_testing::add", mutates_args=())
        def add(x: Tensor, y: float) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body adding y to x through numpy.
            x_np = x.cpu().numpy()
            out_np = x_np + y
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = 3.14
        z = add(x, y)
        self.assertEqual(z, x + y)

        try:
            with torch._subclasses.fake_tensor.FakeTensorMode():
                x = torch.randn(3)
                add(x, y)
            raise AssertionError("should not be hit")
        except RuntimeError as e:
            abstract_impl_error_msg = str(e)
        abstract_impl_error_msg = re.sub(
            r"0x.*>\)>", "0xDEADBEEF>)>", abstract_impl_error_msg
        ).replace(". ", ".\n")
        self.assertExpectedInline(
            abstract_impl_error_msg,
            """\
There was no fake impl registered for <CustomOpDef(_torch_testing::add)>.
This is necessary for torch.compile/export/fx tracing to work.
Please use `add.register_fake` to add an fake impl.""",
        )

        if not IS_WINDOWS:

            @torch.compile(backend="eager")
            def f(x, y):
                # [FUNCTION DOC] HELPER: Compiled wrapper calling the op.
                return add(x, y)

            x = torch.randn(3)
            with self.assertRaisesRegex(RuntimeError, "no fake impl"):
                f(x, y)

        abstract_called = False

        @add.register_fake
        def _(x, y):
            # [FUNCTION DOC] HELPER: Fake kernel setting the enclosing flag and returning an empty
            # tensor like x.
            nonlocal abstract_called
            abstract_called = True
            return torch.empty_like(x)

        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.randn(3)
            z = add(x, y)
            self.assertEqual(z.shape, x.shape)
            self.assertTrue(abstract_called)

    @skipIfTorchDynamo("recursive dynamo")
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work on windows")
    def test_compile(self):
        # [FUNCTION DOC] Compiles a custom linear op with the eager backend in full graph mode;
        # checks the real and fake kernels are both called and the output matches the functional
        # linear. Skipped under Dynamo and on Windows.
        called_impl = False
        called_abstract = False

        @torch.library.custom_op("_torch_testing::linear", mutates_args=())
        def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body running linear through numpy and flagging that it ran.
            nonlocal called_impl
            called_impl = True
            x_np = x.numpy()
            w_np = weight.numpy()
            b_np = bias.numpy()
            out_np = np.add(x_np @ w_np.T, bias)
            return out_np

        @custom_linear.register_fake
        def _(x, weight, bias):
            # [FUNCTION DOC] HELPER: Fake kernel validating ranks, shapes and devices of the
            # arguments (raising AssertionError on mismatch) and returning an empty tensor of
            # output shape.
            nonlocal called_abstract
            called_abstract = True
            if x.dim() != 2:
                raise AssertionError(f"expected x.dim() == 2, got {x.dim()}")
            if weight.dim() != 2:
                raise AssertionError(f"expected weight.dim() == 2, got {weight.dim()}")
            if bias.dim() != 1:
                raise AssertionError(f"expected bias.dim() == 1, got {bias.dim()}")
            if x.shape[1] != weight.shape[1]:
                raise AssertionError(
                    f"expected x.shape[1] == weight.shape[1], got {x.shape[1]} vs {weight.shape[1]}"
                )
            if weight.shape[0] != bias.shape[0]:
                raise AssertionError(
                    f"expected weight.shape[0] == bias.shape[0], got {weight.shape[0]} vs {bias.shape[0]}"
                )
            if x.device != weight.device:
                raise AssertionError(
                    f"expected x.device == weight.device, got {x.device} vs {weight.device}"
                )
            return x.new_empty(x.size(0), weight.size(0))

        x = torch.randn(2, 2)
        weight = torch.randn(2, 2)
        bias = torch.randn(2)
        out = torch.compile(custom_linear, backend="eager", fullgraph=True)(
            x, weight, bias
        )
        self.assertEqual(out, torch.nn.functional.linear(x, weight, bias))
        self.assertTrue(called_impl)
        self.assertTrue(called_abstract)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_error_cases(self):
        # [FUNCTION DOC] Verifies backward through a custom op with no registered autograd formula
        # raises RuntimeError saying there is no autograd formula. Skipped under Dynamo.
        @torch.library.custom_op("_torch_testing::g", mutates_args=())
        def g(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning the sine of x.
            return x.sin()

        x = torch.randn(3, requires_grad=True)
        y = g(x)
        with self.assertRaisesRegex(RuntimeError, "no autograd formula"):
            y.sum().backward()

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_replacement(self):
        # [FUNCTION DOC] Verifies redefining a custom op with the same name replaces the earlier
        # one (output changes from sine to cosine). Skipped under Dynamo.
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning the sine of x.
            return x.sin()

        x = torch.randn(3)
        y = f(x)
        self.assertEqual(y, x.sin())

        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Replacement op body returning the cosine of x.
            return x.cos()

        y = f(x)
        self.assertEqual(y, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_split_device(self):
        # [FUNCTION DOC] Defines an op for CPU, registers a CUDA kernel with register_kernel,
        # and checks each device calls only its own kernel (call counters). Needs CUDA; skipped
        # under Dynamo.
        cpu_call_count = 0
        cuda_call_count = 0

        @torch.library.custom_op(
            "_torch_testing::f", mutates_args=(), device_types="cpu"
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: CPU op body counting calls and computing the sine
            # through numpy.
            nonlocal cpu_call_count
            cpu_call_count += 1
            x_np = x.numpy()
            out_np = np.sin(x_np)
            return torch.from_numpy(out_np)

        @f.register_kernel("cuda")
        def _(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: CUDA kernel counting calls and computing the sine
            # through numpy.
            nonlocal cuda_call_count
            cuda_call_count += 1
            x_np = x.cpu().numpy()
            out_np = np.sin(x_np)
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = f(x)
        self.assertEqual(y, x.sin())
        self.assertEqual(cpu_call_count, 1)
        self.assertEqual(cuda_call_count, 0)

        x = x.cuda()
        y = f(x)
        self.assertEqual(y, x.sin())
        self.assertEqual(cpu_call_count, 1)
        self.assertEqual(cuda_call_count, 1)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_multi_types(self):
        # [FUNCTION DOC] Defines an op for both CPU and CUDA device types with one body and checks
        # both produce the sine. Needs CUDA; skipped under Dynamo.
        @torch.library.custom_op(
            "_torch_testing::f", mutates_args=(), device_types=("cpu", "cuda")
        )
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body computing the sine through numpy on a CPU copy.
            x_np = x.cpu().numpy()
            out_np = np.sin(x_np)
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = f(x)
        self.assertEqual(y, x.sin())
        x = x.cuda()
        y = f(x)
        self.assertEqual(y, x.sin())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_overloading(self):
        # [FUNCTION DOC] Defines an op and then an overload with a different signature in the same
        # namespace and verifies each call form reaches its own function (call counters). Skipped
        # under Dynamo.
        called_f = 0
        called_f1 = 0

        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body counting calls and returning a clone of x.
            nonlocal called_f
            called_f += 1
            return x.clone()

        x = torch.randn(2, 3)
        torch.ops._torch_testing.f(x)
        self.assertEqual(called_f, 1)

        @torch.library.custom_op("_torch_testing::f.overload", mutates_args=())
        def f1(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Overload body counting calls and returning a clone of x.
            nonlocal called_f1
            called_f1 += 1
            return x.clone()

        torch.ops._torch_testing.f(x, x)
        self.assertEqual(called_f1, 1)

    def test_disallows_output_aliasing(self):
        # [FUNCTION DOC] Verifies a custom op whose body returns a view of its input, the input
        # itself, or a mutated input raises RuntimeError saying outputs may not alias.
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns a flattened view of x.
            return x.view(-1)

        x = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            f(x)

        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns x itself.
            return x

        x = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            f(x)

        @torch.library.custom_op(
            "_torch_testing::f", mutates_args={"x"}, device_types="cpu"
        )
        def numpy_sin_inplace(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Applies sine in place through numpy and returns x.
            x_np = x.numpy()
            np.sin(x_np, out=x_np)
            return x

        x = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            numpy_sin_inplace(x)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_factory_function(self):
        # [FUNCTION DOC] Tests ops without tensor inputs: they need a device argument, a call for
        # an unregistered device raises RuntimeError, definition without a device argument raises
        # ValueError, and registering a kernel without one raises ValueError. Skipped under Dynamo.
        # KNOWN ISSUE: the final calls use the first op f rather than the later op f4, so f4 and
        # its kernel are never called.
        @torch.library.custom_op(
            "_torch_testing::f", mutates_args={}, device_types="cpu"
        )
        def f(device: torch.device) -> Tensor:
            # [FUNCTION DOC] HELPER: CPU op body taking a device and returning ones of size three.
            return torch.ones(3)

        result = f(device="cpu")
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(result, torch.ones(3))

        with self.assertRaisesRegex(
            RuntimeError, "f does not have a kernel registered for cuda"
        ):
            f("cuda")

        with self.assertRaisesRegex(
            ValueError,
            "Functions without tensor inputs are required to have a `device: torch.device` argument",
        ):

            @torch.library.custom_op(
                "_torch_testing::f2", mutates_args={}, device_types="cpu"
            )
            def f2() -> Tensor:
                # [FUNCTION DOC] HELPER: Op body without arguments; definition is expected to fail.
                return torch.ones(3)

        @torch.library.custom_op("_torch_testing::f3", mutates_args={})
        def f3() -> Tensor:
            # [FUNCTION DOC] HELPER: Op body without arguments that raises NotImplementedError.
            raise NotImplementedError("NYI")

        with self.assertRaisesRegex(
            ValueError,
            "Functions without tensor inputs are required to have a `device: torch.device` argument",
        ):

            @f3.register_kernel("cpu")
            def _():
                # [FUNCTION DOC] HELPER: Kernel without a device argument returning zeros;
                # registration expected to fail.
                return torch.zeros(3)

            result = f(x)

        @torch.library.custom_op("_torch_testing::f4", mutates_args={})
        def f4(device: torch.device) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body with a device argument that
            # raises NotImplementedError.
            raise NotImplementedError("NYI")

        @f4.register_kernel("cpu")
        def _(device: torch.device):
            # [FUNCTION DOC] HELPER: CPU kernel with a device argument returning zeros of
            # size three.
            return torch.zeros(3)

        result = f(device="cpu")
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(result, torch.ones(3))

    def test_library_schema_infer(self):
        # [FUNCTION DOC] Checks torch.library.infer_schema output with and without an op name
        # against expected inline strings.
        def foo_impl(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return x.sin()

        schema = torch.library.infer_schema(foo_impl, op_name="myop", mutates_args={})
        self.assertExpectedInline(schema, "myop(Tensor x) -> Tensor")

        schema = torch.library.infer_schema(foo_impl, mutates_args={})
        self.assertExpectedInline(schema, "(Tensor x) -> Tensor")

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_set_kernel_enabled(self):
        # [FUNCTION DOC] Tests set_kernel_enabled: disabling an unregistered device logs that no
        # kernel was registered, enabling an already enabled kernel logs it, disabling twice logs
        # it, and the CPU kernel is used only while enabled (otherwise the default body). Skipped
        # under Dynamo.
        x = torch.ones(1)

        @torch.library.custom_op("mylib::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Default op body returning x plus one.
            return x + 1

        self.assertEqual(f(x), x + 1)
        with self.assertLogs("torch._library.custom_ops") as captured:
            with f.set_kernel_enabled("gpu", enabled=False):
                self.assertEqual(f(x), x + 1)
            self.assertIn(
                "no kernel was registered for this device type", captured.output[0]
            )

        @f.register_kernel("cpu")
        def _(x):
            # [FUNCTION DOC] HELPER: CPU kernel returning x plus two.
            return x + 2

        self.assertEqual(f(x), x + 2)

        with self.assertLogs("torch._library.custom_ops") as captured:
            with f.set_kernel_enabled("cpu", enabled=True):
                self.assertEqual(f(x), x + 2)
            self.assertIn("already enabled", captured.output[0])

        with f.set_kernel_enabled("cpu", enabled=False):
            self.assertEqual(f(x), x + 1)

            with self.assertLogs("torch._library.custom_ops") as captured:
                with f.set_kernel_enabled("cpu", enabled=False):
                    self.assertEqual(f(x), x + 1)
                self.assertIn("already disabled", captured.output[0])

            self.assertEqual(f(x), x + 1)

        with f.set_kernel_enabled("cpu", enabled=True):
            self.assertEqual(f(x), x + 2)

        with f.set_kernel_enabled("cpu", enabled=False):
            self.assertEqual(f(x), x + 1)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_vmap_kwargonly_low_level(self):
        # [FUNCTION DOC] Registers a vmap rule for a low-level op with a keyword-only float and
        # verifies vmap over the op calls the rule and yields the expected values. Skipped
        # under Dynamo.
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            lib.define("foo(Tensor x, *, float y) -> Tensor")
            called = False

            def foo_impl(x, *, y):
                # [FUNCTION DOC] HELPER: CPU kernel returning x times y.
                return x * y

            lib.impl("foo", foo_impl, "CPU")

            def vmap(info, in_dims, x, *, y):
                # [FUNCTION DOC] HELPER: Vmap rule flagging that it ran and returning x times y
                # with batch dim 0.
                nonlocal called
                called = True
                return x * y, 0

            torch.library.register_vmap("_torch_testing::foo", vmap, lib=lib)

            x = torch.ones(3)
            result = torch.vmap(torch.ops._torch_testing.foo)(x, y=3.14)
            self.assertTrue(called)
            self.assertEqual(result, torch.tensor([3.14, 3.14, 3.14]))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_vmap_defaults(self):
        # [FUNCTION DOC] Registers a vmap rule for an op with defaulted positional and keyword-only
        # ints and verifies vmap passes the defaults through and the result is the product. Skipped
        # under Dynamo.
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            lib.define("foo(Tensor w, int x = 2, *, int y = 3, int z) -> Tensor")

            def foo_impl(w, x=2, *, y=3, z):
                # [FUNCTION DOC] HELPER: CPU kernel returning the product of w, x, y and z.
                return w * x * y * z

            lib.impl("foo", foo_impl, "CPU")

            called = False

            def vmap(info, in_dims, w, x=2, *, y=3, z):
                # [FUNCTION DOC] HELPER: Vmap rule flagging that it ran and returning the product
                # with batch dim 0.
                nonlocal called
                called = True
                return w * x * y * z, 0

            torch.library.register_vmap("_torch_testing::foo", vmap, lib=lib)

            w = torch.ones(3)
            result = torch.vmap(torch.ops._torch_testing.foo)(w, z=42)
            self.assertTrue(called)
            self.assertEqual(result, w * 2 * 3 * 42)

    def test_layout_constraint_tags(self):
        # [FUNCTION DOC] Checks get_layout_constraint_tag returns the most constrained of the op's
        # layout tags, and needs_exact_strides by default when none are given.
        needs_exact_strides = torch._C.Tag.needs_exact_strides
        needs_fixed_stride_order = torch._C.Tag.needs_fixed_stride_order
        flexible_layout = torch._C.Tag.flexible_layout
        # (tags, the result of the tag inference)
        tests = [
            ({needs_fixed_stride_order}, needs_fixed_stride_order),
            ({flexible_layout}, flexible_layout),
            # If no tags are provided, then the following is the default
            (set(), needs_exact_strides),
            # If multiple tags are provided, then we use the most constrained tag.
            ({flexible_layout, needs_fixed_stride_order}, needs_fixed_stride_order),
        ]
        from torch._library.utils import get_layout_constraint_tag

        for tags, expected in tests:
            with torch.library._scoped_library("mylib", "FRAGMENT") as m:
                m.define("foobar(Tensor x) -> Tensor", tags=tags)
                result = get_layout_constraint_tag(torch.ops.mylib.foobar.default)
                self.assertEqual(result, expected)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_vmap(self):
        # [FUNCTION DOC] Registers a vmap rule via function, name, overload and the op method;
        # verifies vmap with default, output dim and input dim options calls the rule and gives the
        # elementwise product (transposed as needed). Skipped under Dynamo.
        for mode in ["function", "qualname", "opoverload", "c_opdef"]:

            @torch.library.custom_op("mylib::f", mutates_args=())
            def f(x: Tensor, y: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Op body returning x times y.
                return x * y

            called = False

            def fvmap(info, in_dims, x, y):
                # [FUNCTION DOC] HELPER: Vmap rule moving batch dims to the end (or adding one),
                # multiplying, moving the batch dim to front and returning it with batch dim 0;
                # flags it ran.
                nonlocal called
                called = True
                x_bdim, y_bdim = in_dims
                x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
                y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
                result = x * y
                result = result.movedim(-1, 0)
                return result, 0

            if mode == "function":
                torch.library.register_vmap(f, fvmap)
            elif mode == "qualname":
                torch.library.register_vmap("mylib::f", fvmap)
            elif mode == "opoverload":
                torch.library.register_vmap(torch.ops.mylib.f.default, fvmap)
            elif mode == "c_opdef":
                f.register_vmap(fvmap)

            x = torch.randn(2, 2)
            y = torch.randn(2, 2)

            result = torch.vmap(f)(x, y)
            self.assertTrue(called)
            self.assertEqual(result, x * y)

            called = False
            result = torch.vmap(f, out_dims=1)(x, y)
            self.assertEqual(result, (x * y).T)
            self.assertTrue(called)

            called = False
            result = torch.vmap(f, in_dims=1)(x, y)
            self.assertEqual(result, (x * y).T)
            self.assertTrue(called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_vmap_library_decorator(self):
        # [FUNCTION DOC] Registers a vmap rule using the library decorator with the op name,
        # verifies single vmap and a nested vmap. Skipped under Dynamo.
        @torch.library.custom_op("mylib::f", mutates_args=())
        def f(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning x times y.
            return x * y

        called = False

        @torch.library.register_vmap("mylib::f")
        def fvmap(info, in_dims, x, y):
            # [FUNCTION DOC] HELPER: Vmap rule multiplying with batch dims handled, flagging that
            # it ran.
            nonlocal called
            called = True
            x_bdim, y_bdim = in_dims
            x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
            y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
            result = x * y
            result = result.movedim(-1, 0)
            return result, 0

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        result = torch.vmap(f)(x, y)
        self.assertTrue(called)
        self.assertEqual(result, x * y)

        x = torch.randn(3)
        y = torch.randn(3)
        result = torch.vmap(torch.vmap(f, in_dims=(0, None)), in_dims=(None, 0))(x, y)
        self.assertEqual(result, y.unsqueeze(-1) * x)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_vmap_op_decorator(self):
        # [FUNCTION DOC] Registers a vmap rule using the op's register_vmap decorator, verifies
        # single vmap and a nested vmap over inputs of different sizes. Skipped under Dynamo.
        @torch.library.custom_op("mylib::f", mutates_args=())
        def f(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning x times y.
            return x * y

        called = False

        @f.register_vmap
        def fvmap(info, in_dims, x, y):
            # [FUNCTION DOC] HELPER: Vmap rule multiplying with batch dims handled, flagging that
            # it ran.
            nonlocal called
            called = True
            x_bdim, y_bdim = in_dims
            x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
            y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
            result = x * y
            result = result.movedim(-1, 0)
            return result, 0

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        result = torch.vmap(f)(x, y)
        self.assertTrue(called)
        self.assertEqual(result, x * y)

        x = torch.randn(3)
        y = torch.randn(2)
        result = torch.vmap(torch.vmap(f, in_dims=(0, None)), in_dims=(None, 0))(x, y)
        self.assertEqual(result, y.unsqueeze(-1) * x)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_vmap_register_multiple_times(self):
        # [FUNCTION DOC] Registers a vmap rule with the op's register_vmap decorator, checks vmap
        # uses it, then registers a second rule on the same op (addition instead of product) and
        # verifies the new rule replaces the first. Skipped under Dynamo.
        @torch.library.custom_op("mylib::f", mutates_args=())
        def f(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning x times y.
            return x * y

        called = False

        @f.register_vmap
        def fvmap(info, in_dims, x, y):
            # [FUNCTION DOC] HELPER: Vmap rule multiplying with batch dims handled, flagging that
            # it ran.
            nonlocal called
            called = True
            x_bdim, y_bdim = in_dims
            x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
            y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
            result = x * y
            result = result.movedim(-1, 0)
            return result, 0

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        result = torch.vmap(f)(x, y)
        self.assertTrue(called)
        self.assertEqual(result, x * y)
        called = False

        @f.register_vmap
        def fvmap2(info, in_dims, x, y):
            # [FUNCTION DOC] HELPER: Replacement vmap rule that adds instead of multiplies,
            # flagging that it ran.
            nonlocal called
            called = True
            x_bdim, y_bdim = in_dims
            x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
            y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
            result = x + y
            result = result.movedim(-1, 0)
            return result, 0

        result = torch.vmap(f)(x, y)
        self.assertTrue(called)
        self.assertEqual(result, x + y)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_vmap_register_multiple_times_2(self):
        # [FUNCTION DOC] Same as the previous test but registers both vmap rules with the
        # torch.library register_vmap decorator by op name; the second registration replaces the
        # first. Skipped under Dynamo.
        @torch.library.custom_op("mylib::f", mutates_args=())
        def f(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Op body returning x times y.
            return x * y

        called = False

        @torch.library.register_vmap("mylib::f")
        def fvmap(info, in_dims, x, y):
            # [FUNCTION DOC] HELPER: Vmap rule multiplying with batch dims handled, flagging that
            # it ran.
            nonlocal called
            called = True
            x_bdim, y_bdim = in_dims
            x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
            y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
            result = x * y
            result = result.movedim(-1, 0)
            return result, 0

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        result = torch.vmap(f)(x, y)
        self.assertTrue(called)
        self.assertEqual(result, x * y)
        called = False

        @torch.library.register_vmap("mylib::f")
        def fvmap2(info, in_dims, x, y):
            # [FUNCTION DOC] HELPER: Replacement vmap rule that adds instead of multiplies,
            # flagging that it ran.
            nonlocal called
            called = True
            x_bdim, y_bdim = in_dims
            x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
            y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
            result = x + y
            result = result.movedim(-1, 0)
            return result, 0

        result = torch.vmap(f)(x, y)
        self.assertTrue(called)
        self.assertEqual(result, x + y)

    @skipIfTorchDynamo("Skip due to sys.refcount")
    def test_any_requires_grad(self):
        # [FUNCTION DOC] Tests the internal _any_requires_grad helper: no reference leak on keyword
        # arguments (reference count of the tensor is unchanged) and it returns true only if a
        # tensor in the args, a top-level tensor list, or a keyword argument requires grad (nested
        # lists are not searched). Skipped under Dynamo.
        test_fn = torch._C._any_requires_grad
        # Regression test on not leaking kwargs
        t = torch.randn(2, 2)
        t_refcount = sys.getrefcount(t)
        test_fn(t, a=t)
        self.assertEqual(sys.getrefcount(t), t_refcount)

        self.assertTrue(
            test_fn(
                torch.zeros(1, requires_grad=True), torch.ones(1, requires_grad=True)
            )
        )
        self.assertFalse(test_fn(torch.ones(1), torch.zeros(1)))
        self.assertTrue(
            test_fn(
                [torch.zeros(1, requires_grad=True), torch.ones(1, requires_grad=True)]
            )
        )
        # _C_any_requires_grad supports only List[Tensor] in args, not List[List[Tensor]]
        self.assertFalse(test_fn([[torch.zeros(1, requires_grad=True)]], torch.ones(1)))
        self.assertFalse(test_fn([torch.zeros(1), torch.ones(1)]))
        self.assertTrue(test_fn(torch.zeros(1), a=torch.ones(1, requires_grad=True)))
        self.assertFalse(test_fn(torch.zeros(1), a=torch.ones(1)))
        self.assertTrue(
            test_fn([torch.zeros(1, requires_grad=True), torch.ones(1)], torch.zeros(1))
        )
        self.assertFalse(test_fn([torch.zeros(1), torch.ones(1)], torch.zeros(1)))

    @skipIfTorchDynamo("Skip due to sys.refcount")
    def test_any_output_is_alias_to_input_or_output(self):
        # [FUNCTION DOC] Tests the internal _any_output_is_alias_to_input_or_output helper with
        # args, kwargs and outputs containing views, transposes, slices, empty tensors and nested
        # lists; also checks no reference leak. Skipped under Dynamo.
        test_fn = torch._C._any_output_is_alias_to_input_or_output
        # Regression test on not leaking kwargs
        t = torch.randn(2, 2)
        t_refcount = sys.getrefcount(t)
        test_fn((t,), {"a": t}, ())
        if sys.getrefcount(t) != t_refcount:
            raise AssertionError(f"refcount leak: {sys.getrefcount(t)} != {t_refcount}")

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        self.assertTrue(
            test_fn(
                (x,),
                {},
                (x.t(),),
            )
        )
        self.assertFalse(test_fn((x,), None, (2 * x,)))
        self.assertTrue(
            test_fn(
                (),
                {"a": x.view(-1)},
                (x,),
            )
        )
        self.assertTrue(
            test_fn(
                (),
                {"a": x.view(-1)},
                (x.t(),),
            )
        )
        self.assertTrue(test_fn((y,), {}, (y[1:],)))
        self.assertFalse(
            test_fn(
                (x,),
                {"a": x},
                (),
            )
        )
        self.assertFalse(
            test_fn(
                (torch.tensor([]),),
                {},
                (torch.tensor([]),),
            )
        )
        self.assertTrue(
            test_fn(
                ([x], x + 1),
                {},
                (x.t(),),
            )
        )
        self.assertTrue(
            test_fn(
                ([x], x + 1),
                {},
                ([x.t()], x + 1),
            )
        )
        self.assertTrue(
            test_fn(
                ([x], x),
                {},
                ([x.t()], x + 1),
            )
        )
        self.assertTrue(
            test_fn(
                ([x, 1], x),
                {},
                ([x.t()], x + 1),
            )
        )
        self.assertTrue(
            test_fn(
                ([[x]], x),
                {},
                ([x.t()], x + 1),
            )
        )
        self.assertTrue(
            test_fn(
                ([[1, x], 2], 3),
                {},
                ([x.t()], x + 1),
            )
        )

    def test_library_get_kernel(self):
        # [FUNCTION DOC] Registers a CPU kernel (with key set) for aten arange.start in a scoped
        # library, fetches it with torch.library.get_kernel and calls it boxed, expecting a
        # tensor of ones; after the scope ends the kernel handle raises RuntimeError saying it
        # was invalidated.
        """Test registering a custom kernel, using it, then deregistering and verifying error."""

        # Register a dummy kernel for arange to the CPU key that returns a tensor of ones
        def dummy_arange_cpu(
            dispatch_keys,
            start,
            end,
            dtype=None,
            layout=torch.strided,
            device=None,
            pin_memory=False,
        ):
            # [FUNCTION DOC] HELPER: Dummy arange kernel taking a key set and ignoring the range
            # values, returning ones of size end minus start (at least zero).
            size = max(0, int(end - start))
            return torch.ones(size, dtype=dtype, device=device)

        with torch.library._scoped_library("aten", "IMPL") as lib:
            lib.impl("arange.start", dummy_arange_cpu, "CPU", with_keyset=True)

            kernel = torch.library.get_kernel("aten::arange.start", "CPU")
            dispatch_keys = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
            result = kernel.call_boxed(dispatch_keys, 0, 5)

            self.assertEqual(result, torch.ones(5))

        # The kernel should now be invalidated after exiting the scoped_library context
        with self.assertRaisesRegex(RuntimeError, "has been invalidated"):
            kernel.call_boxed(dispatch_keys, 0, 5)

    def test_library_get_kernel_with_conditional_dispatch(self):
        # [FUNCTION DOC] Stacks two scoped-library CPU overrides of arange.start that each call the
        # previously fetched kernel for even bounds and otherwise return a fixed fill; checks the
        # combined results, then that destroying the inner library invalidates the first kernel
        # chain (RuntimeError) while the outer still works.
        """Test registering a custom kernel with conditional dispatch logic."""

        def conditional_arange_cpu1(
            original_kernel,
            dispatch_keys,
            start,
            end,
            dtype=None,
            layout=torch.strided,
            device=None,
            pin_memory=False,
        ):
            # [FUNCTION DOC] HELPER: Calls the wrapped original kernel boxed when end is even, else
            # returns ones of the requested size.
            # If end is even, use the original kernel, otherwise return ones tensor
            if end % 2 == 0:
                op_handle = torch.ops.aten.arange.start._handle
                return original_kernel.call_boxed(
                    dispatch_keys,
                    start,
                    end,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            else:
                size = max(0, int(end - start))
                return torch.ones(size, dtype=dtype, device=device)

        def conditional_arange_cpu2(
            original_kernel,
            dispatch_keys,
            start,
            end,
            dtype=None,
            layout=torch.strided,
            device=None,
            pin_memory=False,
        ):
            # [FUNCTION DOC] HELPER: Calls the wrapped original kernel boxed when start is even,
            # else returns a tensor of twos of the requested size.
            # If start is even, use the original kernel, otherwise return twos tensor
            if start % 2 == 0:
                op_handle = torch.ops.aten.arange.start._handle
                return original_kernel.call_boxed(
                    dispatch_keys,
                    start,
                    end,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            else:
                size = max(0, int(end - start))
                return torch.empty(size, dtype=dtype, device=device).fill_(2)

        original_kernel = torch.library.get_kernel("aten::arange.start", "CPU")
        expected_result1, expected_result2 = torch.ones(5), torch.arange(0, 6)
        expected_result3, expected_result4, expected_result5 = (
            torch.ones(5),
            torch.arange(0, 6),
            torch.ones(5).fill_(2),
        )

        with torch.library._scoped_library("aten", "IMPL") as lib2:
            with torch.library._scoped_library("aten", "IMPL") as lib1:
                lib1.impl(
                    "arange.start",
                    partial(conditional_arange_cpu1, original_kernel),
                    "CPU",
                    with_keyset=True,
                )

                self.assertEqual(torch.arange(0, 5), expected_result1)
                self.assertEqual(torch.arange(0, 6), expected_result2)
                new_original_kernel = torch.library.get_kernel(
                    "aten::arange.start", "CPU"
                )
                lib2.impl(
                    "arange.start",
                    partial(conditional_arange_cpu2, new_original_kernel),
                    "CPU",
                    with_keyset=True,
                )

                self.assertEqual(torch.arange(0, 5), expected_result3)
                self.assertEqual(torch.arange(0, 6), expected_result4)
                self.assertEqual(torch.arange(1, 6), expected_result5)

            # The kernel should now be invalidated after destroying lib1
            with self.assertRaisesRegex(RuntimeError, "has been invalidated"):
                torch.arange(0, 5)

            # Should still work after destroying lib1
            self.assertEqual(torch.arange(1, 6), expected_result5)

    def test_library_get_kernel_invalid(self):
        # [FUNCTION DOC] Verifies get_kernel works for an existing CPU kernel and raises
        # RuntimeError saying no kernel for CUDA exists for an op with only a CPU kernel.
        """Test that get_kernel raises an error when no kernel is available."""
        with torch.library._scoped_library("test_invalid_kernel", "DEF") as lib:
            lib.define("cpu_only_op(Tensor x) -> Tensor")
            lib.impl("cpu_only_op", lambda x: x * 2, "CPU")

            cpu_kernel = torch.library.get_kernel(
                "test_invalid_kernel::cpu_only_op", "CPU"
            )
            self.assertIsNotNone(cpu_kernel)

            # CUDA should fail at the isValid() check since no CUDA kernel exists
            with self.assertRaisesRegex(
                RuntimeError, "no kernel for CUDA for test_invalid_kernel::cpu_only_op"
            ):
                torch.library.get_kernel("test_invalid_kernel::cpu_only_op", "CUDA")


class TestLibrarySourceLocation(TestCase):
    # [CLASS DOC] Test for the source location recorded by torch.library.Library, run in
    # a subprocess.
    def test_library_source_location(self):
        # [FUNCTION DOC] Runs a subprocess script that creates a DEF library twice for the same
        # namespace and prints the RuntimeError; asserts exit code zero and that the message
        # mentions the script's own location on line two, which shows the caller frame is used.
        # Library.__init__ uses sys._getframe(1) to capture the caller's
        # filename and line number. Verify this works correctly by creating
        # a Library and checking the source location in the error message
        # that appears when a duplicate DEF library is created.
        script = """\
import torch
lib1 = torch.library.Library("_test_loc", "DEF")
lib1.define("foo(Tensor x) -> Tensor")
try:
    lib2 = torch.library.Library("_test_loc", "DEF")
except RuntimeError as e:
    print(str(e))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # The error message should reference <string>:2, since
        # lib1 = torch.library.Library(...) is on line 2 of the script.
        self.assertIn("<string>:2", result.stdout)


class MiniOpTestOther(CustomOpTestCaseBase):
    # [CLASS DOC] Second TestCase in namespace mini_op_test whose tests get opcheck variants
    # generated by generate_opcheck_tests below.
    test_ns = "mini_op_test"

    def test_nonzero_again(self):
        # [FUNCTION DOC] Checks aten nonzero returns the indices of nonzero elements as a column.
        x = torch.tensor([0, 1, 2, 0, 0])
        y = torch.ops.aten.nonzero.default(x)
        self.assertEqual(y, torch.tensor([[1], [2]]))


optests.generate_opcheck_tests(
    MiniOpTest,
    ["aten", "mini_op_test"],
    get_file_path_2(os.path.dirname(__file__), "minioptest_failures_dict.json"),
    additional_decorators={
        "test_pt2_compliant_tag_mini_op_test_no_abstract": [unittest.expectedFailure]
    },
    test_utils=optests.generate_tests.DEPRECATED_DEFAULT_TEST_UTILS,
)

optests.generate_opcheck_tests(
    MiniOpTestOther,
    ["aten", "mini_op_test"],
    get_file_path_2(os.path.dirname(__file__), "minioptest_failures_dict.json"),
    test_utils=optests.generate_tests.DEPRECATED_DEFAULT_TEST_UTILS,
)


class TestGenerateOpcheckTests(CustomOpTestCaseBase):
    # [CLASS DOC] Tests of the opcheck test generator and the torch.library.opcheck API.
    def test_MiniOpTest(self):
        # [FUNCTION DOC] Verifies generated test methods exist on MiniOpTest for each default test
        # utility and for the tests test_mm and test_nonzero.
        for orig_test in ["test_mm", "test_nonzero"]:
            for (
                test
            ) in torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS:
                expected_test = f"{test}__{orig_test}"
                self.assertTrue(hasattr(MiniOpTest, expected_test), msg=expected_test)

    def test_generate_repro_save_data(self):
        # [FUNCTION DOC] Checks generate_repro output in dry-run mode with data saving: the script
        # loads args and kwargs from a saved file (path normalized) and calls opcheck; compared
        # with an expected inline string.
        from torch.testing._internal.optests.generate_tests import generate_repro

        args = (torch.ones(2, 2),)
        kwargs = {"mat2": torch.zeros(2, 2)}
        actual = generate_repro(
            "test_schema",
            torch.ops.aten.sin.default,
            args,
            kwargs,
            save_data=True,
            dry_run=True,
        )
        actual = re.sub(r"torch.load\(\".*\.pt\"\)", 'torch.load("repro.pt")', actual)
        self.assertExpectedInline(
            actual,
            """\
# =========================================================
# BEGIN REPRO SCRIPT
# =========================================================
import torch
from torch.testing._internal.optests import opcheck

# Make sure you have loaded the library that contains the op
# via an import or torch.ops.load_library(...)
op = torch.ops.aten.sin.default

args, kwargs = torch.load("repro.pt")
opcheck(op, args, kwargs, test_utils="test_schema")
# =========================================================
# END REPRO SCRIPT
# =========================================================
""",
        )

    def test_generate_repro_no_save_data(self):
        # [FUNCTION DOC] Checks generate_repro output in dry-run mode without data saving: the
        # script has placeholder empty args and kwargs; compared with an expected inline string.
        from torch.testing._internal.optests.generate_tests import generate_repro

        args = (torch.ones(2, 2),)
        kwargs = {"mat2": torch.zeros(2, 2)}
        actual = generate_repro(
            "test_schema",
            torch.ops.aten.sin.default,
            args,
            kwargs,
            save_data=False,
            dry_run=True,
        )
        self.assertExpectedInline(
            actual,
            """\
# =========================================================
# BEGIN REPRO SCRIPT
# =========================================================
import torch
from torch.testing._internal.optests import opcheck

# Make sure you have loaded the library that contains the op
# via an import or torch.ops.load_library(...)
op = torch.ops.aten.sin.default

# If you rerun your test with PYTORCH_OPCHECK_PRINT_BETTER_REPRO=1
# we will fill them in same (args, kwargs) as in your test
args = ()  # args to the operator
kwargs = {}  # kwargs to the operator
opcheck(op, args, kwargs, test_utils="test_schema")
# =========================================================
# END REPRO SCRIPT
# =========================================================
""",
        )

    def test_failures_dict_validation(self):
        # [FUNCTION DOC] Checks validate_failures_dict_structure raises RuntimeError for a failure
        # entry with status success, a name not beginning with a known test utility, and a test
        # name that does not exist on the TestCase.
        from torch.testing._internal.optests.generate_tests import (
            FailuresDict,
            validate_failures_dict_structure,
        )

        failures = {
            "mini_op_test::incorrect_schema": {
                "MiniOpTest.test_aot_dispatch_dynamic__test_delayed_error": {
                    "comment": "",
                    "status": "success",
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "got status=success"):
            validate_failures_dict_structure(
                FailuresDict("", failures),
                torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS,
                MiniOpTest,
            )

        failures = {
            "mini_op_test::incorrect_schema": {
                "MiniOpTest.test_aot_dispatch__test_delayed_error": {
                    "comment": "",
                    "status": "xfail",
                },
            }
        }
        with self.assertRaisesRegex(RuntimeError, "should begin with one of"):
            validate_failures_dict_structure(
                FailuresDict("", failures),
                torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS,
                MiniOpTest,
            )

        failures = {
            "mini_op_test::incorrect_schema": {
                "MiniOpTest.test_aot_dispatch_dynamic__test_delayed_error_nopenopenope": {
                    "comment": "",
                    "status": "xfail",
                },
            }
        }
        with self.assertRaisesRegex(RuntimeError, "does not exist on the TestCase"):
            validate_failures_dict_structure(
                FailuresDict("", failures),
                torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS,
                MiniOpTest,
            )

    def test_dont_generate_decorator(self):
        # [FUNCTION DOC] Verifies the dontGenerateOpCheckTests marker leaves the original test but
        # generates no schema variant for it on MiniOpTest.
        self.assertTrue(hasattr(MiniOpTest, "test_dont_generate"))
        self.assertFalse(hasattr(MiniOpTest, "test_schema__test_dont_generate"))

    def test_opcheck(self):
        # [FUNCTION DOC] Tests torch.library.opcheck on aten sin: rejects a non-OpOverload
        # (ValueError), an invalid test_utils name (ValueError), returns SUCCESS for all four
        # checks by default, and honors a single or list of test_utils.
        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "OpOverload"):
            torch.library.opcheck(torch.sin, (x,))
        with self.assertRaisesRegex(ValueError, "test_utils to be subset of"):
            torch.library.opcheck(torch.ops.aten.sin.default, (x,), test_utils="blah")
        result = torch.library.opcheck(torch.ops.aten.sin.default, (x,))

        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
                "test_autograd_registration": "SUCCESS",
                "test_faketensor": "SUCCESS",
                "test_aot_dispatch_dynamic": "SUCCESS",
            },
        )

        result = torch.library.opcheck(
            torch.ops.aten.sin.default, (x,), test_utils="test_schema"
        )
        self.assertEqual(result, {"test_schema": "SUCCESS"})

        result = torch.library.opcheck(
            torch.ops.aten.sin.default,
            (x,),
            test_utils=["test_schema", "test_faketensor"],
        )
        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
                "test_faketensor": "SUCCESS",
            },
        )

    @unittest.skipIf(not TEST_CUDA, "pinned CPU memory requires CUDA")
    @unittest.skipIf(
        PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property"
    )
    def test_opcheck_preserves_pinned_memory_for_schema_check(self):
        # [FUNCTION DOC] Verifies opcheck preserves pinned memory for the schema check: the
        # CPU kernel raises RuntimeError unless its input is pinned. Needs CUDA; skipped
        # under CUDA memcheck.
        lib = self.lib()
        lib.define("requires_pinned(Tensor x) -> Tensor")
        op = self.ns().requires_pinned.default

        def requires_pinned_impl(x):
            # [FUNCTION DOC] HELPER: CPU kernel raising RuntimeError if x is not pinned, else
            # returning a clone.
            if not x.is_pinned():
                raise RuntimeError("expected pinned input")
            return x.clone()

        lib.impl("requires_pinned", requires_pinned_impl, "CPU")

        x = torch.arange(12, dtype=torch.float32, pin_memory=True).view(3, 4)
        torch.library.opcheck(op, (x,), test_utils="test_schema")

    @unittest.skipIf(not TEST_CUDA, "pinned CPU memory requires CUDA")
    @unittest.skipIf(
        PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property"
    )
    def test_opcheck_preserves_pinned_memory_by_default(self):
        # [FUNCTION DOC] Verifies opcheck by default preserves pinned memory for a custom op with a
        # fake kernel, and all four checks succeed. Needs CUDA; skipped under CUDA memcheck.
        @torch.library.custom_op(
            f"{self.test_ns}::requires_pinned_default", mutates_args=()
        )
        def requires_pinned_default(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Op body raising RuntimeError if x is not pinned, else
            # returning x plus one.
            if not x.is_pinned():
                raise RuntimeError("expected pinned input")
            return x + 1

        @requires_pinned_default.register_fake
        def _(x):
            # [FUNCTION DOC] HELPER: Fake kernel returning an empty tensor like x.
            return torch.empty_like(x)

        x = torch.arange(12, dtype=torch.float32, pin_memory=True).view(3, 4)
        result = torch.library.opcheck(requires_pinned_default, (x,))

        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
                "test_autograd_registration": "SUCCESS",
                "test_faketensor": "SUCCESS",
                "test_aot_dispatch_dynamic": "SUCCESS",
            },
        )

    @skipIfTorchDynamo("recursive dynamo")
    def test_safe_aot_autograd_check_checks_gradients_for_non_leaf_inputs(self):
        # [FUNCTION DOC] Patches torch.testing.assert_close to record tuple comparisons and checks
        # that safe_aot_autograd_check on a non-leaf input compares gradients exactly once. Skipped
        # under Dynamo.
        original_assert_close = torch.testing.assert_close
        gradient_asserts = []

        def assert_close(actual, expected, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Records tuple-versus-tuple comparisons and delegates to the
            # original assert_close.
            if isinstance(actual, tuple) and isinstance(expected, tuple):
                gradient_asserts.append((actual, expected))
            return original_assert_close(actual, expected, *args, **kwargs)

        leaf = torch.randn(3, requires_grad=True)
        non_leaf = leaf * 2
        with patch("torch.testing.assert_close", assert_close):
            optests.generate_tests.safe_aot_autograd_check(
                torch.ops.aten.sin.default,
                (non_leaf,),
                {},
                dynamic=True,
            )

        self.assertEqual(len(gradient_asserts), 1)

    def test_aot_autograd_copy_does_not_check_pinned_memory_for_wrapper_subclass(
        self,
    ):
        # [FUNCTION DOC] Verifies cloning an input for AOT autograd preserves the wrapper subclass,
        # copies its inner tensor value and gives fresh storage, without calling the pin check.
        x = AOTAutogradCopyNoIsPinnedWrapperSubclass(torch.randn(4, 6))

        cloned = _clone_input_for_aot_autograd(x)

        self.assertIsInstance(cloned, AOTAutogradCopyNoIsPinnedWrapperSubclass)
        self.assertEqual(cloned.elem, x.elem)
        self.assertNotEqual(cloned.elem.data_ptr(), x.elem.data_ptr())

    @unittest.skipIf(not TEST_CUDA, "pinned CPU memory requires CUDA")
    @unittest.skipIf(
        PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property"
    )
    def test_safe_schema_check_copy_inputs_preserves_pinned_memory_and_copies(self):
        # [FUNCTION DOC] Verifies safe_schema_check keeps pinned memory and strides for copied
        # inputs and leaves the original unmodified, whereas without copying it mutates the
        # original tensor. Needs CUDA; skipped under CUDA memcheck.
        lib = self.lib()
        lib.define("check_and_mutate(Tensor(a!) x) -> ()")
        op = self.ns().check_and_mutate.default
        seen_inputs = []

        def check_and_mutate_impl(x):
            # [FUNCTION DOC] HELPER: CPU kernel recording whether x is pinned, its data pointer and
            # strides, then adding one in place.
            seen_inputs.append((x.is_pinned(), x.data_ptr(), x.stride()))
            x.add_(1)

        lib.impl("check_and_mutate", check_and_mutate_impl, "CPU")

        x = torch.zeros(4, 6, pin_memory=True)[:, ::2]
        optests.generate_tests.safe_schema_check(op, (x,), {}, copy_inputs=True)

        self.assertEqual(x, torch.zeros_like(x))
        self.assertEqual(len(seen_inputs), 1)
        self.assertTrue(seen_inputs[-1][0])
        self.assertNotEqual(seen_inputs[-1][1], x.data_ptr())
        self.assertEqual(seen_inputs[-1][2], x.stride())

        optests.generate_tests.safe_schema_check(op, (x,), {}, copy_inputs=False)

        self.assertEqual(x, torch.ones_like(x))
        self.assertEqual(len(seen_inputs), 2)
        self.assertEqual(seen_inputs[-1][1], x.data_ptr())

    def test_opcheck_customopdef(self):
        # [FUNCTION DOC] Runs opcheck on the numpy_cube CustomOpDef with CPU (and CUDA if
        # available) inputs, with and without requires_grad.
        sample_inputs = [
            (torch.randn(3),),
            (torch.randn(3, requires_grad=True),),
        ]
        if torch.cuda.is_available():
            sample_inputs.extend(
                [
                    (torch.randn(3, device="cuda"),),
                    (torch.randn(3, device="cuda", requires_grad=True),),
                ]
            )
        for args in sample_inputs:
            torch.library.opcheck(custom_op_db.numpy_cube, args)

    def test_is_inside_opcheck_mode(self):
        # [FUNCTION DOC] Verifies is_inside_opcheck_mode is false normally and true
        # inside OpCheckMode.
        self.assertFalse(optests.is_inside_opcheck_mode())
        with optests.generate_tests.OpCheckMode(
            ["foo"], "bar", lambda x: x, None, "baz", "brr"
        ):
            self.assertTrue(optests.is_inside_opcheck_mode())

    @patch("torch._functorch.config.check_custom_op_aliasing", False)
    def test_opcheck_bad_op(self):
        # [FUNCTION DOC] With the custom op aliasing config check disabled, verifies opcheck on an
        # op with an incorrect schema raises an exception about not aliasing the output, and with
        # raise_exception off returns a mapping where the schema check result is a RuntimeError and
        # the other three checks report SUCCESS.
        op = op_with_incorrect_schema(self, "foo")
        x = torch.randn(3)
        with self.assertRaisesRegex(Exception, "is not defined to alias output"):
            torch.library.opcheck(op, (x,))

        result = torch.library.opcheck(op, (x,), raise_exception=False)
        self.assertTrue(isinstance(result["test_schema"], RuntimeError))
        del result["test_schema"]
        self.assertEqual(
            result,
            {
                "test_autograd_registration": "SUCCESS",
                "test_faketensor": "SUCCESS",
                "test_aot_dispatch_dynamic": "SUCCESS",
            },
        )

    def test_opcheck_does_not_require_extra_deps(self):
        # [FUNCTION DOC] Runs opcheck on aten sin in a fresh subprocess and asserts the expecttest
        # module and the internal common_utils module were not imported, so the public API needs
        # only install-time dependencies.
        # torch.testing._internal.common_utils comes with a lot of additional
        # test-time dependencies. Since opcheck is public API, it should be
        # usable only with pytorch install-time dependencies.
        cmd = [
            sys.executable,
            "-c",
            "import torch; import sys; \
               x = torch.randn(3, requires_grad=True); \
               torch.library.opcheck(torch.ops.aten.sin.default, (x,)); \
               assert 'expecttest' not in sys.modules; \
               assert 'torch.testing._internal.common_utils' not in sys.modules",
        ]
        subprocess.check_output(cmd, shell=False)


class TestTypeConversion(TestCase):
    # [CLASS DOC] Tests for tuple_to_list, the helper that suggests a list annotation for tuple
    # annotations when inferring schemas; supported base types are set up in setUp.
    """In infer_schema(), we try to suggest a correct type when the type annotation is wrong."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs base setUp and stores the list of supported base types (int,
        # float, bool, str, device, Tensor, dtype, Number).
        super().setUp()
        self.supported_base_types = [
            int,
            float,
            bool,
            str,
            torch.device,
            torch.Tensor,
            torch.dtype,
            torch.types.Number,
        ]

    def test_simple_tuple(self):
        # [FUNCTION DOC] Checks a bare Tuple annotation converts to the plain list type.
        self.assertEqual(list, tuple_to_list(Tuple))

    def test_supported_types(self):
        # [FUNCTION DOC] For each supported base type, checks homogeneous tuples of one or three
        # elements convert to a list of that type.
        for t in self.supported_base_types:
            result_type = tuple_to_list(Tuple[t, t, t])
            self.assertEqual(result_type, list[t])

            result_type = tuple_to_list(Tuple[t])
            self.assertEqual(result_type, list[t])

    def test_optional(self):
        # [FUNCTION DOC] For each supported base type, checks tuples with a trailing optional
        # element convert to a list of optional, and variable-length tuples to a list of the type.
        for t in self.supported_base_types:
            result_type = tuple_to_list(Tuple[t, Optional[t]])
            self.assertEqual(result_type, list[Optional[t]])

            result_type = tuple_to_list(Tuple[t, t, Optional[t]])
            self.assertEqual(result_type, list[Optional[t]])

            result_type = tuple_to_list(Tuple[t, ...])
            self.assertEqual(result_type, list[t])

    def test_mixed_types(self):
        # [FUNCTION DOC] Checks tuples of mixed types convert to a list of the union of the
        # element types.
        result_type = tuple_to_list(Tuple[int, float])
        self.assertEqual(result_type, list[int | float])

        result_type = tuple_to_list(Tuple[int, float, str])
        self.assertEqual(result_type, list[int | float | str])


class TestOpProfiles(TestCase):
    # [CLASS DOC] Tests for fake op profiles (generated fake kernels from recorded tensor metadata)
    # and their YAML and file serialization.
    def get_sample_op_profile(self, opname) -> dict[str, set[OpProfile]]:
        # [FUNCTION DOC] HELPER: Returns a mapping from the given op name to a set holding
        # one profile for two rank-2 float32 CPU strided inputs and a rank-2 float32 CPU
        # strided output.
        return {
            opname: {
                OpProfile(
                    args_profile=(
                        TensorMetadata(
                            rank=2,
                            dtype=torch.float32,
                            device=torch.device("cpu"),
                            layout=torch.strided,
                        ),
                        TensorMetadata(
                            rank=2,
                            dtype=torch.float32,
                            device=torch.device("cpu"),
                            layout=torch.strided,
                        ),
                    ),
                    out_profile=TensorMetadata(
                        rank=2,
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                        layout=torch.strided,
                    ),
                )
            }
        }

    def test_fake_registration(self):
        # [FUNCTION DOC] Verifies that an op without a fake kernel raises
        # UnsupportedOperatorException under fake mode, works inside unsafe_generate_fake_kernels
        # with a matching profile, raises MissingOpProfile for inputs not in the profile, and is
        # unsupported again after leaving the context.
        fm = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(allow_dynamic_output_shape_ops=True)
        )
        t1 = fm.from_tensor(torch.ones(3, 3))
        t2 = fm.from_tensor(torch.ones(3, 3))

        op_profiles = self.get_sample_op_profile("mylib.foo2.default")

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo2",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo2", "cpu", lib=lib)
            def foo_impl(a, b):
                # [FUNCTION DOC] HELPER: CPU kernel returning a plus b.
                return a + b

            with (
                self.assertRaisesRegex(
                    torch._subclasses.fake_tensor.UnsupportedOperatorException,
                    "mylib.foo2.default",
                ),
                fm,
            ):
                torch.ops.mylib.foo2(t1, t2)

            with (
                torch._library.fake_profile.unsafe_generate_fake_kernels(op_profiles),
                fm,
            ):
                torch.ops.mylib.foo2(t1, t2)

                with self.assertRaisesRegex(MissingOpProfile, "mylib::foo2"):
                    torch.ops.mylib.foo2(torch.ones(3, 3, 3), torch.ones(3, 3, 3))

            with (
                self.assertRaisesRegex(
                    torch._subclasses.fake_tensor.UnsupportedOperatorException,
                    "mylib.foo2.default",
                ),
                fm,
            ):
                torch.ops.mylib.foo2(t1, t2)

    def test_duplicate_registration_impl(self):
        # [FUNCTION DOC] Verifies profile-generated fake kernels temporarily override an
        # existing fake kernel: output dtype is bfloat16 normally, float32 inside the context,
        # bfloat16 after.
        fm = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(allow_dynamic_output_shape_ops=True)
        )
        t1 = fm.from_tensor(torch.ones(3, 3))
        t2 = fm.from_tensor(torch.ones(3, 3))

        op_profiles = self.get_sample_op_profile("mylib.foo3.default")

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo3",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo3", "cpu", lib=lib)
            def foo3_impl(a, b):
                # [FUNCTION DOC] HELPER: CPU kernel returning a plus b.
                return a + b

            @torch.library.register_fake("mylib::foo3", lib=lib)
            def foo3_impl_fake(a, b):
                # [FUNCTION DOC] HELPER: Fake kernel returning a plus b converted to bfloat16.
                return (a + b).to(dtype=torch.bfloat16)

            with fm:
                self.assertEqual(torch.ops.mylib.foo3(t1, t2).dtype, torch.bfloat16)

            with torch._library.fake_profile.unsafe_generate_fake_kernels(op_profiles):
                with fm:
                    self.assertEqual(torch.ops.mylib.foo3(t1, t2).dtype, torch.float32)

            with fm:
                self.assertEqual(torch.ops.mylib.foo3(t1, t2).dtype, torch.bfloat16)

    def test_duplicate_registration_custom_op(self):
        # [FUNCTION DOC] Same temporary override check as above but for a torch.library.custom_op
        # with a registered fake kernel returning bfloat16.
        fm = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(allow_dynamic_output_shape_ops=True)
        )
        t1 = fm.from_tensor(torch.ones(3, 3))
        t2 = fm.from_tensor(torch.ones(3, 3))

        op_profiles = self.get_sample_op_profile("mylib.foo1.default")

        @torch.library.custom_op("mylib::foo1", mutates_args=())
        def foo_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Op body returning a plus b.
            return a + b

        @torch.library.register_fake("mylib::foo1")
        def foo_impl_fake(a, b):
            # [FUNCTION DOC] HELPER: Fake kernel returning an empty bfloat16 tensor like a.
            return torch.empty_like(a, dtype=torch.bfloat16)

        with fm:
            self.assertEqual(torch.ops.mylib.foo1(t1, t2).dtype, torch.bfloat16)

        with torch._library.fake_profile.unsafe_generate_fake_kernels(op_profiles):
            with fm:
                self.assertEqual(torch.ops.mylib.foo1(t1, t2).dtype, torch.float32)

        with fm:
            self.assertEqual(torch.ops.mylib.foo1(t1, t2).dtype, torch.bfloat16)

    def test_yaml(self):
        # [FUNCTION DOC] Verifies profiles survive a round trip through generate_yaml_from_profiles
        # and read_profiles_from_yaml.
        op_profiles = self.get_sample_op_profile("mylib.foo.default")
        yaml_str = generate_yaml_from_profiles(op_profiles)
        loaded = read_profiles_from_yaml(yaml_str)
        self.assertEqual(op_profiles, loaded)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_save_to_file(self):
        # [FUNCTION DOC] Verifies save_op_profiles and load_op_profiles round-trip a profile
        # through an in-memory buffer, a named temporary file path and a pathlib Path. Not run
        # on Windows.
        op_profile = self.get_sample_op_profile("mylib.foo.default")

        # Saving with buffer
        buffer = io.BytesIO()
        save_op_profiles(op_profile, buffer)
        buffer.seek(0)
        loaded = load_op_profiles(buffer)
        self.assertEqual(op_profile, loaded)

        # Saving with file
        with tempfile.NamedTemporaryFile() as f:
            save_op_profiles(op_profile, f.name)
            f.seek(0)
            loaded = load_op_profiles(f.name)
            self.assertEqual(op_profile, loaded)

        # Saving with Path
        with TemporaryFileName() as fname:
            path = Path(fname)
            save_op_profiles(op_profile, path)
            loaded = load_op_profiles(path)
            self.assertEqual(op_profile, loaded)

    def test_version(self):
        # [FUNCTION DOC] Verifies loading a YAML profile whose torch version is changed to an old
        # value raises RuntimeError saying the profile is outdated.
        op_profiles = self.get_sample_op_profile("mylib.foo.default")
        yaml_str = generate_yaml_from_profiles(op_profiles)
        loaded = yaml.safe_load(yaml_str)
        loaded["torch_version"] = "2.7"
        yaml_str = yaml.dump(loaded, sort_keys=False)
        with self.assertRaisesRegex(RuntimeError, "Unable to load outdated profile"):
            loaded = read_profiles_from_yaml(yaml_str)


only_for = ("cpu", "cuda", "xpu")
instantiate_device_type_tests(
    TestCustomOpTesting, globals(), only_for=only_for, allow_xpu=True
)
instantiate_parametrized_tests(TestCustomOp)
instantiate_parametrized_tests(TestCustomOpAPI)

if __name__ == "__main__":
    run_tests()
