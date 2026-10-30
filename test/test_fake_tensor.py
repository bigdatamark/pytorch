# Owner(s): ["module: meta tensors"]
# ruff: noqa: F841
# [MODULE DOC] Tests for FakeTensor and FakeTensorMode (torch._subclasses.fake_tensor):
# fake tensor creation from real tensors, device handling, caching of op results, dynamic
# and unbacked shapes via ShapeEnv, meta-kernel and fake-impl behavior, sparse and subclass
# inputs, and op-db driven consistency checks. Importing sets two dynamo config flags (fake
# tensor cache and cache crosscheck enabled) globally, and defines the module-level
# device_type used by device-generic tests. Run directly it executes the unittest runner.


import contextlib
import copy
import dataclasses
import gc
import inspect
import io
import itertools
import pickle
import subprocess
import sys
import unittest
import weakref
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo
import torch._functorch.config
import torch._prims as prims
import torch.testing._internal.optests as optests
import torch.utils._pytree as pytree
from torch import distributed as dist
from torch._C._functorch import _add_batch_dim, get_unwrapped, is_batchedtensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.source import LocalSource
from torch._dynamo.testing import make_test_cls_with_patches, rand_strided
from torch._guards import tracing, TracingContext
from torch._higher_order_ops.scan import scan
from torch._subclasses.fake_tensor import (
    _CacheKeyState,
    _check_for_subclass_arg,
    DynamicOutputShapeException,
    extract_tensor_metadata,
    FakeTensor,
    FakeTensorConverter,
    FakeTensorDeviceMismatchError,
    FakeTensorMode,
    is_fake_tensor,
    MetadataMismatchError,
    unset_fake_temporarily,
    UnsupportedOperatorException,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    free_symbols,
    free_unbacked_symbols,
    ShapeEnv,
    ShapeEnvSettings,
    StatefulSymbolicContext,
    StatelessSymbolicContext,
    statically_known_true,
)
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_dtype import all_types_complex_float8_and
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    skipIfWindows,
    skipIfXpu,
    TemporaryFileName,
    TEST_ACCELERATOR,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.jit_utils import RUN_CUDA
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

torch._dynamo.config.fake_tensor_cache_enabled = True
torch._dynamo.config.fake_tensor_cache_crosscheck_enabled = True

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def expectedFailurePropagateRealTensors(fn):
    # [FUNCTION DOC] HELPER: Decorator that marks a test function with an attribute
    # flagging it as an expected failure when fake_tensor_propagate_real_tensors is
    # on. Mutates and returns the same function; does not wrap it.
    fn._expected_failure_propagate_real_tensors = True
    return fn


class FakeTensorTest(TestCase):
    # [CLASS DOC] Main test case for FakeTensor/FakeTensorMode behavior on CPU and (where noted)
    # CUDA: construction, device semantics, metadata, caching, dispatch and error cases. Holds
    # no state of its own; helper methods build fake inputs and check fake tensor type/shape.
    def fake_with_unbacked_batch(self, *tensors):
        # [FUNCTION DOC] HELPER: Builds a FakeTensorMode with a fresh ShapeEnv and converts each
        # given real tensor into a fake tensor whose first dimension is unbacked and remaining
        # dimensions are static. Conversion happens with fresh unbacked symbols ignored so inputs
        # are not treated as op outputs. Returns the pair of the mode and the list of fake tensors.
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        fakes = []
        # These are fake inputs created before tracing, not fresh op outputs for make_fx.
        with fake_mode, shape_env.ignore_fresh_unbacked_symbols():
            for i, tensor in enumerate(tensors):
                fakes.append(
                    fake_mode.from_tensor(
                        tensor,
                        source=LocalSource(f"arg{i}", is_input=True),
                        symbolic_context=StatelessSymbolicContext(
                            dynamic_sizes=[
                                DimDynamic.UNBACKED,
                                *[DimDynamic.STATIC] * (tensor.dim() - 1),
                            ],
                        ),
                    )
                )
        return fake_mode, fakes

    def checkType(self, t, device_str, size):
        # [FUNCTION DOC] HELPER: Asserts the given tensor is a fake tensor, that its device
        # type equals the given device string, and that its size equals the given list of ints.
        self.assertTrue(is_fake_tensor(t))
        self.assertEqual(t.device.type, device_str)
        self.assertEqual(list(t.size()), size)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cuda_initialized(self):
        # [FUNCTION DOC] Smoke test (CUDA only): runs a matmul, square, sum and backward on real
        # CUDA tensors inside a FakeTensorMode and checks nothing raises. No explicit assertions.
        # doesn't error
        with FakeTensorMode():
            p = torch.randn(4, 2, requires_grad=True, device="cuda")
            x = torch.randn(8, 4, device="cuda")
            y = torch.mm(x, p).square().sum()
            y.backward()

    def test_basic(self):
        # [FUNCTION DOC] Basic check: converts two CPU tensors to fake tensors, adds them under the
        # mode, and verifies the result is a fake tensor with the broadcast shape and CPU device.
        x = torch.empty(2, 2, device="cpu")
        y = torch.empty(4, 2, 2, device="cpu")
        with FakeTensorMode() as mode:
            x = mode.from_tensor(x)
            y = mode.from_tensor(y)
            z = x + y
            self.assertEqual(z.shape, (4, 2, 2))
            self.assertEqual(z.device, torch.device("cpu"))
            self.assertTrue(is_fake_tensor(z))

    def test_nansum_nanmean_empty_dim(self):
        # [FUNCTION DOC] Regression test for nansum and nanmean with an empty dim given as
        # tuple or list: for each keepdim value, the fake output shape must equal the eager
        # output shape (full reduction), since the meta kernel once preserved the input shape.
        # nansum/nanmean reduce over all dimensions when dim=() or dim=[] is
        # passed, matching eager. The meta kernel used to preserve the input
        # shape instead. See https://github.com/pytorch/pytorch/issues/191188
        x = torch.randn(2, 3)
        for op in (torch.nansum, torch.nanmean):
            for dim in ((), []):
                for keepdim in (False, True):
                    eager = op(x, dim=dim, keepdim=keepdim)
                    with FakeTensorMode() as mode:
                        fake = op(mode.from_tensor(x), dim=dim, keepdim=keepdim)
                    self.assertEqual(fake.shape, eager.shape)

    def test_inplace_non_broadcastable_raises(self):
        # [FUNCTION DOC] Regression test: in-place ops (pow_, atan2_, eq_, fmod_,
        # remainder_, add_) whose other operand has a non-broadcastable-to-self
        # shape must raise RuntimeError mentioning the broadcast shape under
        # FakeTensorMode, and must leave self's shape unchanged.
        # Ops decomposed via _make_inplace used to silently resize the fake
        # self tensor to the broadcast shape instead of raising like eager.
        # See https://github.com/pytorch/pytorch/issues/191283
        with FakeTensorMode():
            for name in ["pow_", "atan2_", "eq_", "fmod_", "remainder_", "add_"]:
                a = torch.empty(1)
                b = torch.empty(2, 1, 2, 1)
                msg = "doesn't match the broadcast shape"
                with self.assertRaisesRegex(RuntimeError, msg):
                    getattr(a, name)(b)
                self.assertEqual(a.shape, (1,))

    def test_mm_out_dtype(self):
        # [FUNCTION DOC] Tests mm and bmm with an out_dtype of float32 on float8 inputs. On CPU
        # fake tensors the result has the requested dtype and expected shape; on CUDA device
        # fake tensors the meta kernel must raise for float8 with mismatched out_dtype, while
        # bfloat16 to float32 is allowed. Skipped when propagate_real_tensors is enabled.
        # The out_dtype dtype restriction in mm/bmm/baddbmm is a property of the
        # in-tree CUDA/XPU backends. Out-of-tree backends may support arbitrary
        # out_dtype combinations

        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")

        with FakeTensorMode():
            a = torch.empty((8, 2048), dtype=torch.float8_e4m3fn, device="cpu")
            b = torch.empty((2048, 768), dtype=torch.float8_e4m3fn, device="cpu")
            out = torch.mm(a, b, out_dtype=torch.float32)
            self.assertEqual(out.dtype, torch.float32)
            self.assertEqual(out.shape, (8, 768))

            a3 = torch.empty((2, 8, 2048), dtype=torch.float8_e4m3fn, device="cpu")
            b3 = torch.empty((2, 2048, 768), dtype=torch.float8_e4m3fn, device="cpu")
            out3 = torch.bmm(a3, b3, out_dtype=torch.float32)
            self.assertEqual(out3.dtype, torch.float32)
            self.assertEqual(out3.shape, (2, 8, 768))

        # On CUDA the restriction must still be enforced by the meta kernel.
        with FakeTensorMode(allow_non_fake_inputs=True):
            a = torch.empty((8, 2048), dtype=torch.float8_e4m3fn, device="cuda")
            b = torch.empty((2048, 768), dtype=torch.float8_e4m3fn, device="cuda")
            with self.assertRaisesRegex(RuntimeError, "out_dtype must be the same"):
                torch.mm(a, b, out_dtype=torch.float32)

            # fp16/bf16 -> fp32 is still allowed on CUDA.
            c = torch.empty((8, 2048), dtype=torch.bfloat16, device="cuda")
            d = torch.empty((2048, 768), dtype=torch.bfloat16, device="cuda")
            out = torch.mm(c, d, out_dtype=torch.float32)
            self.assertEqual(out.dtype, torch.float32)

    def test_sparse_spdiags(self):
        # [FUNCTION DOC] Checks that sparse spdiags under a fake mode
        # returns a FakeTensor with COO layout and the requested shape.
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()):
            out = torch.sparse.spdiags(torch.randn(2, 3), torch.tensor([0, -1]), (2, 3))

        self.assertIsInstance(out, FakeTensor)
        self.assertEqual(out.layout, torch.sparse_coo)
        self.assertEqual(out.shape, (2, 3))

    def test_sparse_spdiags_compressed_layouts(self):
        # [FUNCTION DOC] Checks spdiags with CSR and CSC layouts under fake mode,
        # with both constant (static) offsets and dynamically shaped inputs: result
        # layout is as requested, and the compressed index tensors have the expected
        # sizes and matching values sizes. Skipped when propagate_real_tensors is on.
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("propagate_real_tensors does not support sparse strides")

        real_offsets = torch.tensor([0, -1])
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, real_offsets, make_constant=True
            )
            csr_out = torch.sparse.spdiags(
                torch.randn(2, 3),
                offsets,
                (2, 3),
                layout=torch.sparse_csr,
            )
            csc_out = torch.sparse.spdiags(
                torch.randn(2, 3),
                offsets,
                (2, 3),
                layout=torch.sparse_csc,
            )
            dynamic_offsets = torch.tensor([-1, -2])
            dynamic_csr_out = torch.sparse.spdiags(
                torch.randn(2, 100),
                dynamic_offsets,
                (100, 2),
                layout=torch.sparse_csr,
            )
            dynamic_csc_out = torch.sparse.spdiags(
                torch.randn(2, 100),
                dynamic_offsets,
                (100, 2),
                layout=torch.sparse_csc,
            )

        self.assertEqual(csr_out.layout, torch.sparse_csr)
        self.assertEqual(csc_out.layout, torch.sparse_csc)
        self.assertEqual(dynamic_csr_out.layout, torch.sparse_csr)
        self.assertEqual(dynamic_csc_out.layout, torch.sparse_csc)
        self.assertEqual(dynamic_csr_out.crow_indices().shape, (101,))
        self.assertEqual(
            dynamic_csr_out.col_indices().shape, dynamic_csr_out.values().shape
        )
        self.assertEqual(dynamic_csc_out.ccol_indices().shape, (3,))
        self.assertEqual(
            dynamic_csc_out.row_indices().shape, dynamic_csc_out.values().shape
        )

    def test_sparse_spdiags_dynamic_nnz_bound(self):
        # [FUNCTION DOC] Checks that spdiags with a dynamically sized input produces
        # an nnz symbol whose recorded value range in the shape env is lower bound 0
        # and upper bound 200. Skipped when propagate_real_tensors is on.
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("propagate_real_tensors resolves the real nnz")

        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            out = torch.sparse.spdiags(
                torch.randn(2, 100), torch.tensor([-1, -2]), (100, 2)
            )

        nnz = out._values().shape[0]
        nnz_range = mode.shape_env.var_to_range[nnz.node.expr]
        self.assertEqual(nnz_range.lower, 0)
        self.assertEqual(nnz_range.upper, 200)

    def test_sparse_spdiags_coalesced_metadata(self):
        # [FUNCTION DOC] Checks the coalesced flag of spdiags outputs with constant offsets:
        # the zero-nnz and one-nnz results are coalesced, while the two-nnz one is not.
        zero_nnz_offsets = torch.tensor([3])
        one_nnz_offsets = torch.tensor([0])
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            zero_nnz_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, zero_nnz_offsets, make_constant=True
            )
            one_nnz_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, one_nnz_offsets, make_constant=True
            )
            zero_nnz = torch.sparse.spdiags(torch.randn(1, 3), zero_nnz_offsets, (3, 3))
            one_nnz = torch.sparse.spdiags(torch.randn(1, 1), one_nnz_offsets, (1, 1))
            two_nnz = torch.sparse.spdiags(torch.randn(1, 2), one_nnz_offsets, (2, 2))

        self.assertTrue(zero_nnz.is_coalesced())
        self.assertTrue(one_nnz.is_coalesced())
        self.assertFalse(two_nnz.is_coalesced())

    def test_sparse_spdiags_zero_dim_static_offsets(self):
        # [FUNCTION DOC] Checks spdiags with a zero-sized output dimension and constant
        # offsets: duplicate offsets raise RuntimeError, while an out-of-range offset
        # yields an empty COO coalesced result of shape zero by three.
        duplicate_offsets = torch.tensor([0, 0])
        out_of_range_offsets = torch.tensor([4])
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            duplicate_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, duplicate_offsets, make_constant=True
            )
            out_of_range_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, out_of_range_offsets, make_constant=True
            )

            with self.assertRaisesRegex(
                RuntimeError, "Offset tensor contains duplicate values"
            ):
                torch.sparse.spdiags(torch.randn(2, 3), duplicate_offsets, (0, 3))

            out = torch.sparse.spdiags(torch.randn(1, 3), out_of_range_offsets, (0, 3))

        self.assertEqual(out.layout, torch.sparse_coo)
        self.assertEqual(out.shape, (0, 3))
        self.assertTrue(out.is_coalesced())

    def test_sparse_spdiags_static_offset_errors(self):
        # [FUNCTION DOC] Checks error reporting for spdiags with constant
        # offsets: duplicate offsets raise a RuntimeError about duplicate values,
        # and an out-of-range offset raises one about a negative dimension.
        duplicate_offsets = torch.tensor([0, 0])
        out_of_range_offsets = torch.tensor([4])
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            duplicate_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, duplicate_offsets, make_constant=True
            )
            out_of_range_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, out_of_range_offsets, make_constant=True
            )
            with self.assertRaisesRegex(
                RuntimeError, "Offset tensor contains duplicate values"
            ):
                torch.sparse.spdiags(torch.randn(2, 3), duplicate_offsets, (2, 3))

            with self.assertRaisesRegex(RuntimeError, "negative dimension"):
                torch.sparse.spdiags(torch.randn(1, 3), out_of_range_offsets, (3, 3))

    def test_sparse_spdiags_cuda_not_supported(self):
        # [FUNCTION DOC] Checks that spdiags on fake tensors claiming a CUDA device
        # raises RuntimeError stating it is only implemented for CPU tensors.
        with FakeTensorMode() as fake_mode:
            diagonals = FakeTensor(
                fake_mode, torch.empty(2, 3, device="meta"), torch.device("cuda")
            )
            offsets = FakeTensor(
                fake_mode,
                torch.empty(2, dtype=torch.long, device="meta"),
                torch.device("cuda"),
            )
            with self.assertRaisesRegex(
                RuntimeError, "_spdiags is only implemented for CPU tensors"
            ):
                torch.sparse.spdiags(diagonals, offsets, (2, 3))

    def test_sparse_spdiags_compile_backend_eager(self):
        # [FUNCTION DOC] Checks that spdiags compiled with the eager backend
        # returns the same result as running the function uncompiled.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a 2x3 float tensor and returns the
            # sparse tensor made by spdiags with offsets zero and minus one.
            diags = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            return torch.sparse.spdiags(
                diags,
                offsets=torch.tensor([0, -1]),
                shape=(diags.size(0), diags.size(1)),
            )

        self.assertEqual(torch.compile(fn, backend="eager")(), fn())

    def test_custom_op_fallback(self):
        # [FUNCTION DOC] Checks that a custom op with only a CPU kernel (no fake impl), when
        # fallback kernels are allowed, still raises UnsupportedOperatorException naming the
        # op under FakeTensorMode. Uses a temporary scoped library that is removed afterwards.
        from torch.library import _scoped_library, impl

        with _scoped_library("my_test_op", "DEF") as test_lib:
            test_lib.define("foo(Tensor self) -> Tensor")

            @impl(test_lib, "foo", "CPU")
            def foo_impl(self):
                # [FUNCTION DOC] HELPER: CPU kernel for the
                # test op; returns the cosine of its input.
                return self.cos()

            x = torch.empty(2, 2, device="cpu")
            with self.assertRaisesRegex(
                UnsupportedOperatorException, "my_test_op.foo.default"
            ):
                with FakeTensorMode(allow_fallback_kernels=True) as mode:
                    x = mode.from_tensor(x)
                    torch.ops.my_test_op.foo(x)

    def test_custom_op_fake_impl_with_parameter_subclass(self):
        # [FUNCTION DOC] Checks a custom op with a registered fake impl called
        # with a Parameter subclass argument: raises AssertionError asking to
        # convert tensors to fake when non-fake inputs are disallowed, and
        # returns a FakeTensor with matching shape and CPU device when allowed.
        class ParameterSubclass(torch.nn.Parameter):
            # [CLASS DOC] HELPER: Plain Parameter subclass without any custom dispatch behavior.
            pass

        @torch.library.custom_op(
            "_torch_testing::fake_tensor_parameter_subclass", mutates_args=()
        )
        def custom_op(weight: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op (no mutation) whose eager
            # implementation returns a clone of its weight argument.
            return weight.clone()

        @custom_op.register_fake
        def custom_op_fake(weight: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of the
            # custom op; returns an empty tensor like the weight.
            return torch.empty_like(weight)

        weight = ParameterSubclass(torch.randn(10, 10))
        with self.assertRaisesRegex(
            AssertionError,
            "Please convert all Tensors to FakeTensors first",
        ):
            with FakeTensorMode():
                torch.ops._torch_testing.fake_tensor_parameter_subclass(weight)

        with FakeTensorMode(allow_non_fake_inputs=True):
            result = torch.ops._torch_testing.fake_tensor_parameter_subclass(weight)

        self.assertIsInstance(result, FakeTensor)
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.device, torch.device("cpu"))

    def test_parameter_subclass_with_torch_dispatch_gets_subclass_dispatch(self):
        # [FUNCTION DOC] Checks that a Parameter subclass defining a torch dispatch
        # method receives the subclass dispatch (recorded op equals the custom op
        # overload) instead of the registered fake impl: the fake impl call list
        # stays empty, the result is a FakeTensor with matching shape and CPU device.
        dispatch_calls = []
        fake_impl_calls = []

        @torch.library.custom_op(
            "_torch_testing::fake_tensor_dispatch_parameter_subclass",
            mutates_args=(),
        )
        def custom_op(weight: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op whose eager
            # implementation returns a clone of the weight.
            return weight.clone()

        @custom_op.register_fake
        def custom_op_fake(weight: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake impl that records that it ran in
            # a shared list and returns an empty tensor like the weight.
            fake_impl_calls.append(True)
            return torch.empty_like(weight)

        class DispatchParameter(torch.nn.Parameter):
            # [CLASS DOC] HELPER: Parameter subclass that records every
            # dispatched function in a shared list and, for the test custom op,
            # returns an empty tensor of the weight's shape, dtype and device.
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Torch dispatch hook; appends func to
                # the shared list, returns a freshly allocated empty tensor for
                # the custom op, and NotImplemented for any other function.
                dispatch_calls.append(func)
                if (
                    func
                    is torch.ops._torch_testing.fake_tensor_dispatch_parameter_subclass.default
                ):
                    weight = args[0]
                    return torch.empty(
                        weight.shape, dtype=weight.dtype, device=weight.device
                    )
                return NotImplemented

        weight = DispatchParameter(torch.randn(10, 10))
        with FakeTensorMode():
            result = torch.ops._torch_testing.fake_tensor_dispatch_parameter_subclass(
                weight
            )

        self.assertIsInstance(result, FakeTensor)
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(
            dispatch_calls,
            [torch.ops._torch_testing.fake_tensor_dispatch_parameter_subclass.default],
        )
        self.assertEqual(fake_impl_calls, [])

    def test_check_for_subclass_arg_parameter_subclass(self):
        # [FUNCTION DOC] Checks the subclass-argument detector: it is false for plain tensors,
        # plain Parameters, Parameter subclasses without a dispatch hook and fake tensors, and true
        # for Parameters with a dispatch hook or wrapping a tensor subclass such as TwoTensor.
        class ParameterSubclass(torch.nn.Parameter):
            # [CLASS DOC] HELPER: Parameter subclass with no dispatch hook.
            pass

        class DispatchParameter(torch.nn.Parameter):
            # [CLASS DOC] HELPER: Parameter subclass that defines a torch dispatch classmethod.
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook that always returns NotImplemented.
                return NotImplemented

        self.assertFalse(_check_for_subclass_arg(torch.randn(10)))
        self.assertFalse(_check_for_subclass_arg(torch.nn.Parameter(torch.randn(10))))
        self.assertFalse(_check_for_subclass_arg(ParameterSubclass(torch.randn(10))))
        self.assertTrue(_check_for_subclass_arg(DispatchParameter(torch.randn(10))))

        with FakeTensorMode() as mode:
            fake = mode.from_tensor(torch.randn(10))
        self.assertFalse(_check_for_subclass_arg(fake))

        tt_param = torch.nn.Parameter(TwoTensor(torch.randn(4, 4), torch.randn(4, 4)))
        self.assertTrue(_check_for_subclass_arg(tt_param))

    def test_parameter_instantiation(self):
        # [FUNCTION DOC] Checks that creating a Parameter from a
        # tensor under FakeTensorMode yields an instance of Parameter.
        with FakeTensorMode():
            x = torch.rand([4])
            y = torch.nn.parameter.Parameter(x)
            self.assertTrue(isinstance(y, torch.nn.Parameter))

    def test_parameter_state_reconstruction_from_dict(self):
        # [FUNCTION DOC] Checks rebuilding a fake Parameter from its instance dict plus
        # requires_grad and a meta-converted tensor: result is a FakeTensor and Parameter with
        # same requires_grad, fake device and mode, the dict holds the private device attribute
        # and not the public name, and converting to meta changes its fake device to meta.
        with FakeTensorMode():
            param = torch.nn.Conv2d(3, 4, 1).weight

        fake_device = param.fake_device
        param_cls = type(param)
        kwargs = dict(param.__dict__)
        kwargs["requires_grad"] = param.requires_grad

        reconstructed = param_cls(param.to("meta"), **kwargs)

        self.assertIsInstance(reconstructed, FakeTensor)
        self.assertIsInstance(reconstructed, torch.nn.Parameter)
        self.assertEqual(reconstructed.requires_grad, param.requires_grad)
        self.assertEqual(reconstructed.fake_device, fake_device)
        self.assertEqual(reconstructed.device, fake_device)
        self.assertIs(reconstructed.fake_mode, param.fake_mode)
        self.assertIn("_fake_device", reconstructed.__dict__)
        self.assertNotIn("fake_device", reconstructed.__dict__)
        self.assertEqual(reconstructed.to("meta").fake_device, torch.device("meta"))

    def test_parameter_state_reconstruction_rejects_device_conflict(self):
        # [FUNCTION DOC] Checks that constructing a FakeTensor with both a device and a
        # fake_device that disagree raises ValueError about conflicting device values.
        with FakeTensorMode() as mode:
            elem = torch.empty(2, 2, device="meta")
            with self.assertRaisesRegex(ValueError, "conflicting device values"):
                FakeTensor(
                    mode,
                    elem,
                    device=torch.device("cpu"),
                    fake_device=torch.device("cuda:0"),
                )

    def test_parameter_state_reconstruction_accepts_normalized_device_aliases(self):
        # [FUNCTION DOC] Checks that a FakeTensor built with device cuda and fake_device
        # cuda:0 is accepted (aliases normalize to the same device) and reports cuda:0.
        with FakeTensorMode() as mode:
            elem = torch.empty(2, 2, device="meta")
            fake = FakeTensor(
                mode,
                elem,
                device=torch.device("cuda"),
                fake_device=torch.device("cuda:0"),
            )

        self.assertEqual(fake.fake_device, torch.device("cuda:0"))

    @parametrize("device_kwarg_name", ["fake_device", "device"])
    def test_parameter_state_reconstruction_device_names(self, device_kwarg_name):
        # [FUNCTION DOC] Parametrized over the keyword name (fake_device or device):
        # rebuilds a fake Parameter from its instance dict with the device passed under
        # that keyword and checks the same properties as the dict-based reconstruction
        # test (type, requires_grad, devices, mode, dict keys, meta conversion).
        with FakeTensorMode():
            param = torch.nn.Conv2d(3, 4, 1).weight

        kwargs = dict(param.__dict__)
        fake_device = kwargs.pop("_fake_device")
        kwargs[device_kwarg_name] = fake_device
        kwargs["requires_grad"] = param.requires_grad

        reconstructed = type(param)(param.to("meta"), **kwargs)

        self.assertIsInstance(reconstructed, FakeTensor)
        self.assertIsInstance(reconstructed, torch.nn.Parameter)
        self.assertEqual(reconstructed.requires_grad, param.requires_grad)
        self.assertEqual(reconstructed.fake_device, fake_device)
        self.assertEqual(reconstructed.device, fake_device)
        self.assertIs(reconstructed.fake_mode, param.fake_mode)
        self.assertIn("_fake_device", reconstructed.__dict__)
        self.assertNotIn("fake_device", reconstructed.__dict__)
        self.assertEqual(reconstructed.to("meta").fake_device, torch.device("meta"))

    @unittest.skipIf(not dist.is_available(), "requires distributed")
    def test_fsdp_flat_param(self):
        # [FUNCTION DOC] Checks (when distributed is available) that creating
        # FSDP FlatParameter data under a fake mode gives an object that is
        # simultaneously a FlatParameter, a Parameter and a FakeTensor.
        from torch.distributed.fsdp._flat_param import FlatParameter

        with FakeTensorMode() as m:
            data = torch.randn(2, 2)
            param = FlatParameter(data, requires_grad=True)
        self.assertIsInstance(param, FlatParameter)
        self.assertIsInstance(param, torch.nn.Parameter)
        self.assertIsInstance(param, FakeTensor)

    def test_non_parameter_grad(self):
        # [FUNCTION DOC] Checks that converting a real tensor
        # requiring grad to fake preserves its requires_grad flag.
        mode = FakeTensorMode()
        t = torch.rand([4], requires_grad=True)
        fake_t = mode.from_tensor(t)
        self.assertEqual(fake_t.requires_grad, t.requires_grad)

    def test_non_parameter_grad_tensor_subclass_stateful_context(self):
        # [FUNCTION DOC] Checks converting a tensor requiring grad whose grad is
        # a TwoTensor, using a stateful symbolic context with static sizes: the
        # fake grad stays a TwoTensor whose inner components are FakeTensors.
        mode = FakeTensorMode(shape_env=ShapeEnv())
        t = torch.ones(2, requires_grad=True)
        t.grad = TwoTensor(torch.ones(2), torch.ones(2))
        source = LocalSource("t", is_input=True)
        symbolic_context = StatefulSymbolicContext(
            dynamic_sizes=[DimDynamic.STATIC] * t.dim(),
            constraint_sizes=[None] * t.dim(),
            tensor_source=source,
        )

        fake_t = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)

        self.assertIsInstance(fake_t.grad, TwoTensor)
        self.assertIsInstance(fake_t.grad.a, FakeTensor)
        self.assertIsInstance(fake_t.grad.b, FakeTensor)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    @parametrize(
        "dtype",
        all_types_complex_float8_and(),
    )
    def test_index_cuda_with_cpu(self, dtype):
        # [FUNCTION DOC] Parametrized over many dtypes (CUDA only, skipped under
        # dynamo): indexes a fake CUDA tensor with a CPU int64 index tensor and checks
        # the result is fake on cuda with the indexed size and the original dtype.
        with FakeTensorMode():
            x = torch.ones([2048], device="cuda", dtype=dtype)
            out = x[torch.zeros([36], dtype=torch.int64)]
            self.checkType(out, "cuda", [36])
            self.assertEqual(out.dtype, dtype)

    @skipIfTorchDynamo("test directly checks FakeTensorMode outputs")
    def test_scalar_tensor_index(self):
        # [FUNCTION DOC] For several integer dtypes, indexes a fake
        # tensor by a zero-dim index tensor of that dtype and checks
        # the result is a fake CPU tensor with the remaining dimension.
        for dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                with FakeTensorMode() as mode:
                    x, idx = map(
                        mode.from_tensor,
                        (torch.empty(3, 4), torch.tensor(0, dtype=dtype)),
                    )
                    out = x[idx]
                self.checkType(out, "cpu", [4])

    def test_static_scalar_tensor_with_shape_env_has_no_concrete_item_memo(self):
        # [FUNCTION DOC] Checks that a statically shaped scalar tensor converted
        # with a shape env present has no item memo (None), for int64 and float64.
        shape_env = ShapeEnv()
        for dtype in (torch.int64, torch.float64):
            with self.subTest(dtype=dtype):
                scalar = torch.tensor(0, dtype=dtype)
                with FakeTensorMode(shape_env=shape_env) as mode:
                    scalar = mode.from_tensor(
                        scalar,
                        static_shapes=True,
                    )
                    self.assertIsNone(scalar.item_memo)

    def test_scalar_tensor_item_memo_invalidates_on_epoch(self):
        # [FUNCTION DOC] Checks that the item memo of a converted scalar is a SymInt equal
        # to the value, and becomes None after the mode's epoch counter is incremented.
        shape_env = ShapeEnv()
        scalar = torch.tensor(2, dtype=torch.int64)
        with FakeTensorMode(shape_env=shape_env) as mode:
            scalar = mode.from_tensor(scalar, source=LocalSource("scalar"))
            item_memo = scalar.item_memo
            self.assertIsInstance(item_memo, torch.SymInt)
            self.assertEqual(int(item_memo), 2)

            mode.epoch += 1

            self.assertIsNone(scalar.item_memo)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_shape_take_not_device(self):
        # [FUNCTION DOC] CUDA only: resize_as_ of a CPU fake tensor to a CUDA
        # fake tensor takes the shape but keeps the CPU device and stays fake.
        with FakeTensorMode():
            x = torch.empty(1, device="cpu")
            y = torch.empty(8, 8, device="cuda")
            out = x.resize_as_(y)
            self.assertEqual(out.shape, (8, 8))
            self.assertEqual(out.device.type, "cpu")
            self.assertTrue(is_fake_tensor(out))

    def test_repr(self):
        # [FUNCTION DOC] Checks the repr string of fake tensors:
        # it omits the device for CPU and includes it for meta.
        with FakeTensorMode():
            x = torch.empty(2, 2, device="cpu")
            self.assertEqual(repr(x), "FakeTensor(..., size=(2, 2))")
            x = torch.empty(2, 2, device="meta")
            self.assertEqual(repr(x), "FakeTensor(..., device='meta', size=(2, 2))")

    def test_fake_device_property_normalization(self):
        # [FUNCTION DOC] Checks that the fake_device property normalizes devices:
        # CPU and indexed CUDA stay unchanged, device-without-index for mps
        # becomes index zero; assigning via the property setter normalizes mps and
        # cuda to index zero. Also checks the fake_device constructor keyword.
        """Test that fake_device property normalizes device on assignment."""
        with FakeTensorMode() as mode:
            # Test CPU device - should remain unchanged
            cpu_tensor = torch.empty(2, 2, device="meta")
            fake_cpu = FakeTensor(mode, cpu_tensor, torch.device("cpu"))
            self.assertEqual(fake_cpu.fake_device, torch.device("cpu"))
            fake_cpu_from_alias = FakeTensor(
                mode, cpu_tensor, fake_device=torch.device("cpu")
            )
            self.assertEqual(fake_cpu_from_alias.fake_device, torch.device("cpu"))

            # Test device with explicit index - should remain unchanged
            cuda_device_with_index = torch.device("cuda:0")
            fake_cuda = FakeTensor(mode, cpu_tensor, cuda_device_with_index)
            self.assertEqual(fake_cuda.fake_device, torch.device("cuda:0"))

            # Test MPS device without index - should normalize to mps:0
            mps_device = torch.device("mps")
            fake_mps = FakeTensor(mode, cpu_tensor, mps_device)
            self.assertEqual(fake_mps.fake_device, torch.device("mps:0"))

            # Test property setter normalization with MPS
            fake_tensor = FakeTensor(mode, cpu_tensor, torch.device("cpu"))
            fake_tensor.fake_device = torch.device("mps")
            self.assertEqual(fake_tensor.fake_device, torch.device("mps:0"))

            # Test property setter normalization with CUDA
            fake_tensor.fake_device = torch.device("cuda")
            self.assertEqual(fake_tensor.fake_device, torch.device("cuda:0"))

    def test_convert_fake_to_real(self):
        # [FUNCTION DOC] Checks that try_convert_fake_to_real turns a slice of a real tensor back
        # into an equal real tensor after the tensor was used under a non-fake-input-allowing mode.
        x = torch.ones([20])
        with FakeTensorMode(allow_non_fake_inputs=True) as m:
            _ = x + 1

        out = torch._subclasses.fake_utils.try_convert_fake_to_real([x[0:10]])

        self.assertEqual(torch.ones([10]), out[0])

    def test_conv_nhwc(self):
        # [FUNCTION DOC] Checks that a convolution op on channels-last inputs
        # executed in a fake mode gives the same output strides as eager execution.
        x = torch.randn([1, 1024, 16, 16]).to(memory_format=torch.channels_last)
        w = torch.randn([256, 1024, 4, 4]).to(memory_format=torch.channels_last)
        b = torch.randn([256])

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that wraps the aten convolution
            # op with fixed stride, padding, dilation and groups.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor only.
                super().__init__()

            def forward(self, x, w, b):
                # [FUNCTION DOC] HELPER: Applies the aten convolution to
                # input, weight and bias with stride one, no padding, dilation
                # one, not transposed, no output padding and one group.
                return torch.ops.aten.convolution(
                    x, w, b, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
                )

        model = Model()
        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            fake_out = model.forward(x, w, b)
        eager_out = model.forward(x, w, b)
        self.assertEqual(fake_out.stride(), eager_out.stride())

    def test_private_convolution_symint(self):
        # [FUNCTION DOC] Checks the private convolution op (default and deprecated overloads)
        # on a fake input with dynamic batch-excluded dims and a static fake weight, stride
        # two: the output keeps batch one and channel 32 and its spatial sizes are statically
        # known to equal the symbolic convolution formula for kernel three, stride two.
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        x = fake_mode.from_tensor(
            torch.randn(1, 3, 16, 16),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[
                    DimDynamic.DYNAMIC,
                    DimDynamic.STATIC,
                    DimDynamic.DYNAMIC,
                    DimDynamic.DYNAMIC,
                ],
                constraint_sizes=[None, None, None, None],
            ),
        )
        w = fake_mode.from_tensor(torch.randn(32, 3, 3, 3), static_shapes=True)

        with fake_mode:
            out_default = aten._convolution.default(
                x,
                w,
                None,
                [2, 2],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                False,
                False,
                True,
                True,
            )
            out_deprecated = aten._convolution.deprecated(
                x,
                w,
                None,
                [2, 2],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                False,
                False,
                True,
            )

        for out in (out_default, out_deprecated):
            self.assertEqual(out.shape[:2], (1, 32))
            self.assertTrue(
                statically_known_true(out.shape[2] == (x.shape[2] - 3) // 2 + 1)
            )
            self.assertTrue(
                statically_known_true(out.shape[3] == (x.shape[3] - 3) // 2 + 1)
            )

    def test_conv_rejects_mismatched_fake_devices(self):
        # [FUNCTION DOC] Checks that a convolution mixing a CPU input with CUDA weight and bias
        # under fake mode raises RuntimeError saying all tensors must be on the same device and
        # naming the weight on cuda:0. Skipped when propagate_real_tensors is on without CUDA.
        if torch._functorch.config.fake_tensor_propagate_real_tensors and not RUN_CUDA:
            self.skipTest("propagate_real_tensors requires real CUDA tensors")
        with FakeTensorMode():
            x = torch.empty(1, 3, 8, 8)
            w = torch.empty(3, 3, 3, 3, device="cuda")
            b = torch.empty(3, device="cuda")

            with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device.*weight is on cuda:0",
            ):
                torch.ops.aten.convolution.default(
                    x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
                )

    def test_conv_allows_unspecified_fake_device_index(self):
        # [FUNCTION DOC] Checks that a convolution succeeds when fake tensors name
        # the same CPU device with and without an explicit index (cpu versus cpu:0).
        with FakeTensorMode() as mode:
            x = FakeTensor(
                mode,
                torch.empty(1, 3, 8, 8, device="meta"),
                torch.device("cpu"),
            )
            w = FakeTensor(
                mode,
                torch.empty(3, 3, 3, 3, device="meta"),
                torch.device("cpu:0"),
            )
            b = FakeTensor(
                mode,
                torch.empty(3, device="meta"),
                torch.device("cpu"),
            )

            torch.ops.aten.convolution.default(
                x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
            )

    @unittest.skipUnless(RUN_CUDA, "requires cuda")
    def test_conv_mismatched_device_error_matches_eager(self):
        # [FUNCTION DOC] CUDA only: runs the same CPU/CUDA mixed-device convolution
        # both eagerly and under FakeTensorMode, using one subtest per mode, and
        # checks both raise the same device mismatch RuntimeError message.
        error = "Expected all tensors to be on the same device.*weight is on cuda:0"

        for name, ctx in (
            ("eager", contextlib.nullcontext()),
            ("fake", FakeTensorMode()),
        ):
            with self.subTest(name), ctx:
                x = torch.empty(1, 3, 8, 8)
                w = torch.empty(3, 3, 3, 3, device="cuda")
                b = torch.empty(3, device="cuda")

                with self.assertRaisesRegex(RuntimeError, error):
                    torch.ops.aten.convolution.default(
                        x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
                    )

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_zero_dim(self):
        # [FUNCTION DOC] CUDA only: adding a CPU zero-dim tensor to a CUDA tensor in fake
        # mode succeeds and yields a fake tensor with the larger shape on the CUDA device.
        with FakeTensorMode() as mode:
            x = torch.tensor(0.0)
            y = torch.rand([4, 4], device="cuda")
            out = x + y
            self.assertEqual(out.shape, (4, 4))
            self.assertEqual(out.device, y.device)
            self.assertTrue(is_fake_tensor(out))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_op_with_zero_dim_bypassed(self):
        # [FUNCTION DOC] CUDA only: nextafter of a CUDA zero-dim fake tensor with a CPU
        # zero-dim fake tensor must still raise FakeTensorDeviceMismatchError (zero-dim
        # bypass does not apply to this op). Skipped under propagate_real_tensors.
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")
        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        x = torch.tensor(1.0, device="cuda")
        y = torch.tensor(2.0)
        fake_x = mode.from_tensor(x)
        fake_y = mode.from_tensor(y)

        with self.assertRaisesRegex(
            FakeTensorDeviceMismatchError,
            "Expected all tensors to be on the same device",
        ) as exc:
            torch.nextafter(fake_x, fake_y)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_diagonal_scatter_one_dim_single_elem_cpu_with_cuda_tensor(self):
        # [FUNCTION DOC] CUDA only: diagonal_scatter of a one-element CPU source into
        # a CUDA base is allowed; the result is fake with the base's shape and device.
        with FakeTensorMode():
            base = torch.zeros((1, 2), device="cuda")
            src = torch.tensor([1.0])
            out = torch.diagonal_scatter(base, src, dim1=0, dim2=1)
            self.assertEqual(out.shape, (1, 2))
            self.assertEqual(out.device, base.device)
            self.assertTrue(is_fake_tensor(out))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_diagonal_scatter_two_dim_cpu_with_cuda_tensor(self):
        # [FUNCTION DOC] CUDA only: diagonal_scatter of a CUDA source into a CPU base is
        # allowed under fake mode; the result is fake with the base's shape and device.
        with FakeTensorMode():
            base = torch.zeros((3, 3, 3))
            src = torch.ones((3, 3), device="cuda")
            out = torch.diagonal_scatter(base, src)
            self.assertEqual(out.shape, (3, 3, 3))
            self.assertEqual(out.device, base.device)
            self.assertTrue(is_fake_tensor(out))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_add_one_dim_single_elem_cpu_with_cuda_tensor(self):
        # [FUNCTION DOC] CUDA only: adding a one-element CPU tensor to a CUDA tensor raises
        # FakeTensorDeviceMismatchError for this case. Skipped under propagate_real_tensors.
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")
        with FakeTensorMode():
            x = torch.randn([1])
            y = torch.randn(10, device="cuda")

            with self.assertRaisesRegex(
                FakeTensorDeviceMismatchError,
                "Expected all tensors to be on the same device",
            ) as exc:
                x + y

    def test_nan_to_num(self):
        # [FUNCTION DOC] Checks nan_to_num on float16 and float32 fake tensors, with an
        # explicit None nan argument and a positional replacement, preserves the dtype.
        with FakeTensorMode():
            for dtype in [torch.float16, torch.float32]:
                x = torch.rand([4], dtype=dtype)
                y = torch.nan_to_num(x, nan=None)
                z = torch.nan_to_num(x, 0.0)
                self.assertEqual(dtype, y.dtype)
                self.assertEqual(dtype, z.dtype)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_throw(self):
        # [FUNCTION DOC] CUDA only: lerp of a CPU zero-dim fake tensor
        # with a CUDA tensor and a CPU tensor must raise some exception.
        x = torch.tensor(0.0)  # TODO: tensor() errors
        with FakeTensorMode() as mode:
            x_conv = mode.from_tensor(x)
            y = torch.rand([4, 4], device="cuda")
            z = torch.rand([4, 4], device="cpu")
            self.assertRaises(Exception, lambda: torch.lerp(x_conv, y, z))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_type_as(self):
        # [FUNCTION DOC] CUDA only: type_as from a CPU fake
        # tensor to a CUDA one gives a fake result on CUDA.
        with FakeTensorMode():
            x = torch.rand([16, 1], device="cpu")
            y = torch.rand([4, 4], device="cuda")
            out = x.type_as(y)
            self.assertEqual(out.device.type, "cuda")
            self.assertTrue(is_fake_tensor(out))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_setitem(self):
        # [FUNCTION DOC] CUDA only: in-place item assignment on a
        # fake tensor works on both CPU and CUDA without error.
        for device in ["cpu", "cuda"]:
            with FakeTensorMode():
                x = torch.rand([16, 1], device=device)
                x[..., 0] = 0

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_device_inplace_copy(self):
        # [FUNCTION DOC] CUDA only: copy_ between CPU and CUDA fake tensors keeps
        # the destination's device; raises AssertionError if the device changes.
        with FakeTensorMode():
            x = torch.rand([8, 8], device="cpu")
            y = torch.rand([8, 8], device="cuda")
            if x.copy_(y).device.type != "cpu":
                raise AssertionError("expected cpu device")
            if y.copy_(x).device.type != "cuda":
                raise AssertionError("expected cuda device")

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_fake_device(self):
        # [FUNCTION DOC] CUDA only: sets the fake_device of a fake tensor
        # to CUDA, converts that fake tensor again in a second mode, and
        # checks the new tensor's device equals the original's.
        t = torch.ones(3)
        t = t.view(1, 3)

        fake_mode1 = FakeTensorMode(allow_non_fake_inputs=True)
        fake_t = fake_mode1.from_tensor(t)
        fake_t.fake_device = torch.device("cuda")

        fake_mode2 = FakeTensorMode(allow_non_fake_inputs=True)
        new_fake_t = fake_mode2.from_tensor(fake_t)

        self.assertEqual(new_fake_t.device, fake_t.device)

    def test_fake_dispatch_keys(self):
        # [FUNCTION DOC] Checks the dispatch key set of a fake CPU tensor includes CPU,
        # in-place/view, autograd and autocast keys, and that under inference mode the result of
        # an add has CPU and autocast keys but neither the in-place/view key nor any autograd key.
        with FakeTensorMode():
            x = torch.rand([4])
            f = (
                FileCheck()
                .check("CPU")
                .check("ADInplaceOrView")
                .check("AutogradCPU")
                .check("AutocastCPU")
            )
            f.run(torch._C._dispatch_key_set(x))

            with torch.inference_mode():
                x = torch.rand([4])
                y = x + x
                FileCheck().check("CPU").check("AutocastCPU").run(
                    torch._C._dispatch_key_set(y)
                )
                FileCheck().check_not("ADInplaceOrView").check_not("Autograd").run(
                    torch._C._dispatch_key_set(y)
                )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN not available")
    def test_mkldnn_to_dense(self):
        # [FUNCTION DOC] Checks MKLDNN fake tensors (skipped without MKLDNN or under
        # propagate_real tensors): to_dense yields strided non-mkldnn output with strides three
        # and one via method, op overload, detach, alias, a float8 target dtype, an op-produced
        # mkldnn tensor, and through functional tensor mode with unwrapped results.
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")
        real = torch.randn(2, 3).to_mkldnn()
        with FakeTensorMode() as fake_mode:
            x = fake_mode.from_tensor(real)
            self.assertTrue(x.is_mkldnn)
            self.assertEqual(x.layout, torch._mkldnn)  # type: ignore[attr-defined]

            y = x.to_dense()
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            y = torch.ops.aten.to_dense.default(x)
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            y = x.to_dense(torch.float8_e4m3fn)
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.dtype, torch.float8_e4m3fn)
            self.assertEqual(y.stride(), (3, 1))

            detached = x.detach()
            self.assertTrue(detached.is_mkldnn)
            y = detached.to_dense()
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            aliased = torch.ops.aten.alias.default(x)
            self.assertTrue(aliased.is_mkldnn)
            y = aliased.to_dense()
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            dense = torch.randn(2, 3)
            mkldnn = torch.ops.aten.to_mkldnn.default(dense)
            self.assertTrue(mkldnn.is_mkldnn)
            y = torch.ops.aten.to_dense.default(mkldnn)
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            with FunctionalTensorMode():
                functional = FunctionalTensor.to_functional(x)
                y = functional.detach().to_dense()
                y_unwrapped = torch._from_functional_tensor(y.elem)
                self.assertFalse(y_unwrapped.is_mkldnn)
                self.assertEqual(y_unwrapped.layout, torch.strided)
                self.assertEqual(y_unwrapped.stride(), (3, 1))

                functional = FunctionalTensor.to_functional(x)
                y = torch.ops.aten.alias.default(functional).to_dense()
                y_unwrapped = torch._from_functional_tensor(y.elem)
                self.assertFalse(y_unwrapped.is_mkldnn)
                self.assertEqual(y_unwrapped.layout, torch.strided)
                self.assertEqual(y_unwrapped.stride(), (3, 1))

    def test_compare_tensor_meta_unbacked_numel(self):
        # [FUNCTION DOC] Checks that two fake tensors sharing an unbacked, size-constrained
        # leading dimension pass the tensor metadata comparison including strides.
        from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            u0 = shape_env.create_unbacked_symint()
            _constrain_range_for_size(u0, min=0)
            a = torch.empty(u0, 16, device="cpu")
            b = torch.empty(u0, 16, device="cpu")
        prims.utils.compare_tensor_meta(a, b, check_strides=True)

    def test_stack_symbolic_shapes(self):
        # [FUNCTION DOC] Regression test: stacking tensors with two unbacked
        # dimensions along a middle dim must not require concrete sizes in the
        # inner view; result has three dims and size two in the stacked dim.
        # torch.stack, for a non-trailing dim, is computed as
        # cat(...).view(result_sizes). The inner view must not require concrete
        # sizes when the stacked tensors have symbolic shapes, otherwise it
        # raises "SymIntArrayRef expected to contain only concrete integers".
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            u0 = shape_env.create_unbacked_symint()
            u1 = shape_env.create_unbacked_symint()
            t = torch.empty(u0, u1, device="cpu")
            out = torch.stack([t, t], dim=1)
            self.assertEqual(out.dim(), 3)
            self.assertEqual(out.size(1), 2)

    def test_batch_tensor(self):
        # [FUNCTION DOC] Checks conversion of batched tensors (created via
        # functorch batch-dim helpers) to fake: metadata matches for singly and
        # doubly batched tensors, the fake result is still a batched tensor at
        # each level, and the innermost unwrapped value is a FakeTensor.
        x = torch.rand((3, 4, 5))
        b = _add_batch_dim(x, 0, 0)
        mode = FakeTensorMode()
        fake_b = mode.from_tensor(b)
        prims.utils.compare_tensor_meta(b, fake_b, check_strides=True)

        b1 = _add_batch_dim(x, 1, 1)
        b2 = _add_batch_dim(b1, 0, 2)
        fake_b2 = mode.from_tensor(b2)
        prims.utils.compare_tensor_meta(b2, fake_b2, check_strides=True)
        self.assertTrue(is_batchedtensor(fake_b2))
        fake_b1 = get_unwrapped(fake_b2)
        self.assertTrue(is_batchedtensor(fake_b1))
        fake_tensor = get_unwrapped(fake_b1)
        self.assertIsInstance(fake_tensor, FakeTensor)

    def test_constructor(self):
        # [FUNCTION DOC] Checks that a random tensor created in fake mode is fake and on CPU.
        with FakeTensorMode():
            x = torch.rand([4, 4], device="cpu")

        self.assertTrue(is_fake_tensor(x))
        self.assertTrue(x.device.type == "cpu")

    def test_constructor_like_custom_op_without_device_arg(self):
        # [FUNCTION DOC] Regression test for issue 163196: a custom op without
        # a device argument that builds a tensor is traced symbolically and
        # its recorded value is a fake CPU tensor of shape three.
        with torch.library._scoped_library(
            "fake_tensor_issue_163196", "FRAGMENT"
        ) as lib:
            lib.define("moo(int x) -> Tensor")

            @torch.library.impl(
                "fake_tensor_issue_163196::moo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def moo(x):
                # [FUNCTION DOC] HELPER: Kernel for the test op; returns
                # an empty tensor of length three filled with x.
                return torch.empty(3).fill_(x)

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the test op with the item of its tensor argument.
                return torch.ops.fake_tensor_issue_163196.moo(x.item())

            gm = make_fx(f, tracing_mode="symbolic")(torch.tensor(4))
            target = torch.ops.fake_tensor_issue_163196.moo.default
            (node,) = [n for n in gm.graph.nodes if n.target is target]
            self.assertIsInstance(node.meta["val"], FakeTensor)
            self.assertEqual(node.meta["val"].shape, (3,))
            self.assertEqual(node.meta["val"].device.type, "cpu")

    def test_constructor_like_custom_op_with_device_arg(self):
        # [FUNCTION DOC] Checks a custom constructor-like op with an optional
        # device keyword under fake mode: output is a fake CPU tensor and the
        # kernel was invoked with the meta device at least once.
        with torch.library._scoped_library(
            "fake_tensor_issue_163196_device_arg", "FRAGMENT"
        ) as lib:
            lib.define("moo(int x, *, Device? device=None) -> Tensor")
            seen_devices = []

            @torch.library.impl(
                "fake_tensor_issue_163196_device_arg::moo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def moo(x, device=None):
                # [FUNCTION DOC] HELPER: Kernel that records the device argument in a shared
                # list and returns an empty tensor of length three on that device filled with x.
                seen_devices.append(device)
                return torch.empty(3, device=device).fill_(x)

            with FakeTensorMode():
                out = torch.ops.fake_tensor_issue_163196_device_arg.moo(4, device="cpu")

            self.assertIsInstance(out, FakeTensor)
            self.assertEqual(out.device.type, "cpu")
            self.assertTrue(torch.device("meta") in seen_devices)

    def test_constructor_like_custom_op_with_non_device_arg_named_device(self):
        # [FUNCTION DOC] Checks a custom op whose string argument is merely
        # named device (not a Device type) is not treated as constructor-like:
        # symbolic tracing yields a fake CPU tensor of shape three.
        with torch.library._scoped_library(
            "fake_tensor_issue_163196_non_device_arg", "FRAGMENT"
        ) as lib:
            lib.define("moo(str device) -> Tensor")

            @torch.library.impl(
                "fake_tensor_issue_163196_non_device_arg::moo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def moo(device):
                # [FUNCTION DOC] HELPER: Kernel returning an empty
                # tensor whose length is that of the string.
                return torch.empty(len(device))

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the test op with the string cpu, ignoring its input.
                return torch.ops.fake_tensor_issue_163196_non_device_arg.moo("cpu")

            gm = make_fx(f, tracing_mode="symbolic")(torch.tensor(4))
            target = torch.ops.fake_tensor_issue_163196_non_device_arg.moo.default
            (node,) = [n for n in gm.graph.nodes if n.target is target]
            self.assertIsInstance(node.meta["val"], FakeTensor)
            self.assertEqual(node.meta["val"].shape, (3,))
            self.assertEqual(node.meta["val"].device.type, "cpu")

    def test_mode(self):
        # [FUNCTION DOC] Checks that an add under fake mode produces a fake tensor.
        with FakeTensorMode():
            y = torch.rand([4], device="cpu")
            out = y + y

        self.assertTrue(is_fake_tensor(out))

    def test_full(self):
        # [FUNCTION DOC] Smoke test: torch.full under the cross-ref fake mode runs without error.
        # Test torch.full returns tensor with correct dtype
        with torch._subclasses.CrossRefFakeMode():
            y = torch.full((4, 4), 1)

    def check_function_with_fake(self, fn):
        # [FUNCTION DOC] HELPER: Runs the given zero-arg function once normally
        # and once under fake mode, and compares metadata (including strides) of
        # corresponding output leaves; non-tensor outputs must stay non-tensors.
        out = fn()
        with torch._subclasses.FakeTensorMode():
            out_fake = fn()

        for a, b in zip(pytree.tree_leaves(out), pytree.tree_leaves(out_fake)):
            if not isinstance(a, torch.Tensor):
                self.assertTrue(not isinstance(b, torch.Tensor))
                continue

            prims.utils.compare_tensor_meta(a, b, check_strides=True)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_non_kwarg_device(self):
        # [FUNCTION DOC] CUDA only: moving a CPU fake tensor to CPU by positional device
        # returns the same object, and to a CUDA device gives a CUDA fake tensor.
        with FakeTensorMode():
            x = torch.rand([16, 1], device="cpu")
            y = x.to(torch.device("cpu"))
            self.assertIs(x, y)
            z = x.to(torch.device("cuda"))
            self.assertEqual(z.device.type, "cuda")

    def test_non_overlapping_stride_zero(self):
        # [FUNCTION DOC] Checks that casting an empty-strided tensor with
        # overlapping zero stride to half gives metadata matching real execution.
        def foo():
            # [FUNCTION DOC] HELPER: Builds an empty strided tensor
            # with a zero stride and returns it converted to half.
            x = torch.empty_strided([1, 3, 427, 640], (0, 1, 1920, 3))
            return x.half()

        self.check_function_with_fake(foo)

    def test_fake_mode_error(self):
        # [FUNCTION DOC] Checks that indexing a real (non-fake) tensor under
        # fake mode raises an exception asking to convert all tensors.
        x = torch.rand([4, 4])

        with self.assertRaisesRegex(Exception, "Please convert all Tensors"):
            with FakeTensorMode():
                y = x[0]

    def test_no_tag_func(self):
        # [FUNCTION DOC] Checks that flex attention with an identity score modifier and no block
        # mask runs under a fake mode that allows non-fake inputs, with bfloat16 CPU inputs.
        import functools

        from torch.nn.attention.flex_attention import _identity, flex_attention

        def create_attention(score_mod, block_mask, enable_gqa=False):
            # [FUNCTION DOC] HELPER: Returns a partial of flex attention with
            # the given score modifier, block mask and grouped-query flag bound.
            return functools.partial(
                flex_attention,
                score_mod=score_mod,
                block_mask=block_mask,
                enable_gqa=enable_gqa,
            )

        input_shape = (4, 16, 128, 64)
        q = torch.randn(
            input_shape,
            dtype=torch.bfloat16,
            device="cpu",
            requires_grad=False,
        )
        k = torch.randn(
            input_shape,
            dtype=torch.bfloat16,
            device="cpu",
            requires_grad=False,
        )
        v = torch.randn(
            input_shape,
            dtype=torch.bfloat16,
            device="cpu",
            requires_grad=False,
        )
        sdpa_partial = create_attention(_identity, None)
        with FakeTensorMode(allow_non_fake_inputs=True):
            sdpa_partial(q, k, v, return_lse=False)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_fake_grad_copy(self):
        # [FUNCTION DOC] Checks that converting a tensor with a grad to fake preserves metadata
        # of both the tensor and its grad, and that the fake grad is fake. Skipped under dynamo.
        x = torch.rand([4, 4], requires_grad=True)
        x.grad = torch.rand([4, 4])
        mode = FakeTensorMode()
        fake_x = mode.from_tensor(x)
        prims.utils.compare_tensor_meta(fake_x, x)
        prims.utils.compare_tensor_meta(fake_x.grad, x.grad)

        self.assertTrue(is_fake_tensor(fake_x.grad))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_index_put_error(self):
        # [FUNCTION DOC] CUDA only: index assignment of a CPU value into a CUDA tensor raises
        # RuntimeError both in real and fake mode, for both operator and aten index_put,
        # while a zero-dim CPU value index_put (functional and in-place) does not raise.
        mode = FakeTensorMode()
        for context in [contextlib.nullcontext, lambda: mode]:
            with context():
                y = torch.randn(2, 2, 3)
                x = torch.randn(2, 2, 3).to("cuda")
                with self.assertRaises(RuntimeError):
                    x[[1, 1]] = y

                with self.assertRaises(RuntimeError):
                    torch.ops.aten.index_put(x, torch.tensor([1, 1], device="cuda"), y)

                # no error
                torch.ops.aten.index_put(
                    x, torch.tensor([1, 1], device="cuda"), torch.tensor(5.0)
                )
                torch.ops.aten.index_put_(
                    x, torch.tensor([1, 1], device="cuda"), torch.tensor(5.0)
                )

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_like_constructor(self):
        # [FUNCTION DOC] CUDA only: ones_like of a CPU fake tensor is
        # fake on CPU, and with a CUDA device keyword is fake on CUDA.
        with FakeTensorMode():
            x = torch.rand([4, 4])
            y = torch.ones_like(x)
            self.assertTrue(is_fake_tensor(y))
            self.assertEqual(y.device.type, "cpu")
            z = torch.ones_like(x, device="cuda")
            self.assertTrue(is_fake_tensor(z))
            self.assertEqual(z.device.type, "cuda")

    def test_binary_op_type_promotion(self):
        # [FUNCTION DOC] Checks that dividing a float fake tensor
        # by an int64 one promotes to float and stays on CPU.
        with FakeTensorMode():
            x = torch.empty([2, 2], dtype=torch.float)
            y = torch.empty([2, 2], dtype=torch.int64)
            out = x / y
            self.assertEqual(out.dtype, torch.float)
            self.assertEqual(out.device.type, "cpu")

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_from_numpy(self):
        # [FUNCTION DOC] Checks that building a tensor from a numpy array under fake
        # mode gives a fake CPU tensor of the same shape. Skipped under dynamo.
        with FakeTensorMode():
            x = torch.tensor(np.zeros([4, 4]))
            self.checkType(x, "cpu", [4, 4])

    def test_randperm(self):
        # [FUNCTION DOC] Checks randperm under fake mode produces tensors whose
        # metadata matches real randperm output, with and without an explicit device.
        x = torch.randperm(10)
        y = torch.randperm(5, device="cpu")
        with FakeTensorMode():
            x1 = torch.randperm(10)
            prims.utils.compare_tensor_meta(x, x1)
            y1 = torch.randperm(5, device="cpu")
            prims.utils.compare_tensor_meta(y, y1)

    def test_print_in_fake_mode(self):
        # [FUNCTION DOC] Checks that converting a real tensor to a string within fake mode does
        # not fail and its text does not mention FakeTensor; raises AssertionError otherwise.
        x = torch.zeros(2)
        # does not fail
        with FakeTensorMode():
            out = str(x)
        if "FakeTensor" in out:
            raise AssertionError("FakeTensor should not be in output")

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_upsample_bilinear_small_channels(self):
        # [FUNCTION DOC] CUDA only: bilinear upsample of a small-channel
        # strided CUDA tensor is run eagerly and in fake mode; the fake
        # output is contiguous and matches the real output metadata.
        out = []
        mode = FakeTensorMode()
        for context in [contextlib.nullcontext, lambda: mode]:
            with context():
                arg0_1 = torch.empty_strided(
                    (3, 427, 640), (1, 1920, 3), dtype=torch.float32, device="cuda"
                )
                unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 0)
                out.append(
                    torch.ops.aten.upsample_bilinear2d.default(
                        unsqueeze, [800, 1199], False
                    )
                )

        self.assertTrue(out[1].is_contiguous())
        self.checkMetaProps(out[0], out[1])

    def test_split_return_self(self):
        # [FUNCTION DOC] Checks that splitting an empty tensor with split size zero under fake
        # mode and the python dispatcher returns a new tensor (not self), matching eager metadata.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the first piece
            # of splitting the input with split size zero.
            return torch.functional.split(x, 0)[0]

        # meta should not return self
        with FakeTensorMode(), enable_python_dispatcher():
            out_fake = fn(torch.empty((0,)))

        out_eager = fn(torch.empty((0,)))
        self.checkMetaProps(out_fake, out_eager)

    def test_split_empty_dim(self):
        # [FUNCTION DOC] Checks that splitting an empty tensor with a positive split
        # size under fake mode and the python dispatcher matches eager output metadata.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the first piece
            # of splitting the input with split size two.
            return torch.split(x, 2)[0]

        with FakeTensorMode(), enable_python_dispatcher():
            out_fake = fn(torch.empty((0,)))

        out_eager = fn(torch.empty((0,)))
        self.checkMetaProps(out_fake, out_eager)

    def test_as_strided_negative_stride_error(self):
        # [FUNCTION DOC] Checks that as_strided with a negative stride
        # on a fake tensor raises RuntimeError with the message that
        # negative strides are unsupported and lists the strides.
        error = (
            r"as_strided: Negative strides are not supported at the "
            r"moment, got strides: \[-?[0-9]+(, -?[0-9]+)*\]"
        )
        with FakeTensorMode():
            x = torch.empty(0)
            with self.assertRaisesRegex(RuntimeError, error):
                torch.as_strided(x, (17, 18), (-80, 1), 1)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cpu_fallback(self):
        # [FUNCTION DOC] CUDA only: with fallback kernels disabled conv2d on fake CUDA
        # tensors yields the right shape; with fallback enabled, invalid shapes raise
        # RuntimeError and valid shapes still return the right fake shape and device.
        with FakeTensorMode(allow_fallback_kernels=False):
            filters = torch.randn(8, 4, 3, 3).cuda()
            inputs = torch.randn(1, 4, 5, 5).cuda()
            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            self.assertEqual(out.device.type, "cuda")
            self.assertEqual(list(out.size()), [1, 8, 5, 5])

        with FakeTensorMode(allow_fallback_kernels=True):
            # intentionally bad inputs
            filters = torch.randn(8, 20, 3, 3).cuda()
            inputs = torch.randn(1, 7, 10, 5).cuda()
            with self.assertRaises(RuntimeError):
                torch.nn.functional.conv2d(inputs, filters, padding=1)

        with FakeTensorMode(allow_fallback_kernels=True):
            filters = torch.randn(8, 4, 3, 3).cuda()
            inputs = torch.randn(1, 4, 5, 5).cuda()

            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            self.assertEqual(out.device.type, "cuda")
            self.assertEqual(list(out.size()), [1, 8, 5, 5])

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_out_multi_device(self):
        # [FUNCTION DOC] CUDA only: ops combining CPU and CUDA fake tensors via an out
        # argument or an in-place add raise an exception stating that two devices were found.
        with FakeTensorMode():
            x = torch.rand([4])
            y = torch.rand([4], device="cuda")

            with self.assertRaisesRegex(Exception, "found.+two.+devices"):
                torch.sin(x, out=y)

            with self.assertRaisesRegex(Exception, "found.+two.+devices"):
                x.add_(y)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_normalize_device(self):
        # [FUNCTION DOC] CUDA only: adding two fake CUDA tensors, one
        # given with a bare device name and one with the explicit current
        # index, succeeds and gives a fake CUDA result of size one.
        with FakeTensorMode():
            x = torch.empty(1, device="cuda")
            y = torch.empty(1, device=f"cuda:{torch.cuda.current_device()}")
            out = x + y
        self.checkType(out, "cuda", [1])

    def test_recursive_invocation(self):
        # [FUNCTION DOC] Checks that after setting the mode's in-kernel
        # invocation flag manually, an add inside the mode leaves the flag set.
        mode = FakeTensorMode()
        with mode:
            x = torch.tensor(2)
            mode.in_kernel_invocation = True
            y = x + x
            self.assertTrue(mode.in_kernel_invocation)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @parametrize(
        "allow_fallback_kernels",
        [False, True],
        lambda a: "with_fallback" if a else "without_fallback",
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cudnn_rnn(self, allow_fallback_kernels):
        # [FUNCTION DOC] CUDA only, parametrized over whether fallback kernels are allowed:
        # runs the cudnn (or miopen on ROCm) rnn op with large CUDA weights both for real and
        # under fake mode, once with all inputs and once with the final cell-state input set to
        # None. Checks that the weight buffer output is the input object (cudnn) or non-None
        # (hip), that outputs live on cuda, and that fake-mode outputs are fake tensors.
        def fn(
            a0,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
            a3,
            a4,
            a5,
        ):
            # [FUNCTION DOC] HELPER: Packs sixteen weight tensors into a list and
            # calls the cudnn rnn op (or the miopen rnn op on ROCm builds) with fixed
            # hidden size, layer count and flags, returning its tuple of outputs.
            a1 = [
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                b10,
                b11,
                b12,
                b13,
                b14,
                b15,
            ]

            if torch.version.cuda:
                return torch.ops.aten._cudnn_rnn(
                    a0,
                    a1,
                    4,
                    a3,
                    a4,
                    a5,
                    2,
                    2048,
                    0,
                    2,
                    False,
                    0.0,
                    False,
                    True,
                    [],
                    None,
                )
            else:
                return torch.ops.aten.miopen_rnn(
                    a0,
                    a1,
                    4,
                    # weight_buf is cudnn only
                    a4,
                    a5,
                    2,
                    2048,
                    # proj_size is cudnn only
                    2,
                    False,
                    0.0,
                    False,
                    True,
                    [],
                    None,
                )

        mode = FakeTensorMode(allow_fallback_kernels=allow_fallback_kernels)
        for i, context in enumerate([contextlib.nullcontext, lambda: mode]):
            with context():
                inps1 = [
                    torch.randn([92, 8, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 4096]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 4096]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([167837696]).cuda(),
                    torch.randn([4, 8, 2048]).cuda(),
                    torch.randn([4, 8, 2048]).cuda(),
                ]
                inps2 = inps1
                inps2[len(inps2) - 1] = None  # argument `cx` can be None

                for inps in [inps1, inps2]:
                    out = fn(*inps)
                    # weight_buf is cudnn only <- this is a3 and inps[-3]
                    # in cudnn_rnn, it passthroughs the inps[-3] to out[4],
                    #   so the test expects self.assertIs(out[4], inps[-3])
                    # in miopen_rnn, it does not need the input argument weight_buf,
                    #   the miopen_rnn generate a new weight_buf instead,
                    #   and it returns via out[4]
                    if torch.version.hip:
                        self.assertIsNotNone(out[4])
                    else:
                        self.assertIs(out[4], inps[-3])
                    for ten in out:
                        if i == 1:
                            self.assertTrue(is_fake_tensor(ten))
                        self.assertEqual(ten.device.type, "cuda")

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cuda_lstm(self):
        # [FUNCTION DOC] CUDA only: runs a two-layer LSTM with projection (cudnn
        # disabled) on CUDA tensors in fake mode with fallbacks off, backpropagates,
        # and checks the output and final hidden and cell state shapes.
        # Ensure CUDA (non-cuDNN) impl succeeds with fake tensors.
        with torch.backends.cudnn.flags(enabled=False):
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False)
            with fake_tensor_mode:
                N = 5
                L = 4
                H_in = 2
                hidden_size = 3
                proj_size = 2
                num_layers = 2
                bidir = False
                D = 2 if bidir else 1
                H_out = proj_size if proj_size > 0 else hidden_size

                lstm = torch.nn.LSTM(
                    input_size=H_in,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    proj_size=proj_size,
                    batch_first=False,
                    bias=True,
                    bidirectional=bidir,
                    device="cuda",
                )

                h_0 = torch.randn((num_layers * D, N, H_out), device="cuda")
                c_0 = torch.randn((num_layers * D, N, hidden_size), device="cuda")
                inp = torch.randn((L, N, H_in), device="cuda")
                (output, (h_n, c_n)) = lstm(inp, (h_0, c_0))
                output.sum().backward()

                self.assertEqual(output.shape, (L, N, D * H_out))
                self.assertEqual(h_n.shape, (D * num_layers, N, H_out))
                self.assertEqual(c_n.shape, (D * num_layers, N, hidden_size))

    def test_data_dependent_operator(self):
        # [FUNCTION DOC] Checks that nonzero on a fake tensor with
        # fallbacks disabled raises DynamicOutputShapeException.
        with FakeTensorMode(allow_fallback_kernels=False):
            x = torch.rand([10, 10])

            self.assertRaises(DynamicOutputShapeException, lambda: torch.nonzero(x))

    def test_parameter_view(self):
        # [FUNCTION DOC] Checks that a view of a Parameter converts to a
        # non-Parameter fake tensor while the Parameter itself stays a Parameter.
        x = torch.nn.Parameter(torch.randn(4))
        x_view = x.view(4)
        mode = FakeTensorMode()
        fake_x_view = mode.from_tensor(x_view)
        fake_x = mode.from_tensor(x)
        self.assertFalse(isinstance(fake_x_view, torch.nn.Parameter))
        self.assertTrue(isinstance(fake_x, torch.nn.Parameter))

    def test_tolist(self):
        # [FUNCTION DOC] Smoke test: tolist on a fake tensor
        # with a shape env and no fallbacks does not raise.
        shape_env = ShapeEnv()
        with FakeTensorMode(allow_fallback_kernels=False, shape_env=shape_env):
            x = torch.rand([10])
            x.tolist()

    # Propagate real tensors doesn't work with fake-on-fake
    @expectedFailurePropagateRealTensors
    def test_same_shape_env_preserved(self):
        # [FUNCTION DOC] Checks that re-converting a fake tensor in a second
        # mode sharing the same shape env yields a distinct fake tensor owned by
        # the second mode whose first dimension is still the same dynamic symbol
        # in the same shape env. Expected to fail with propagate_real_tensors.
        shape_env = ShapeEnv()
        mode1 = FakeTensorMode(shape_env=shape_env)
        t1 = mode1.from_tensor(
            torch.randn(10),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC], constraint_sizes=[None]
            ),
        )
        mode2 = FakeTensorMode(shape_env=shape_env)
        t2 = mode2.from_tensor(t1)
        # t2.size(0) is still dynamic, even though we didn't pass DYNAMIC here
        self.assertIsNot(t2, t1)
        self.assertIs(t1.fake_mode, mode1)
        self.assertIs(t2.fake_mode, mode2)
        self.assertIs(t2.size(0).node.shape_env, t1.size(0).node.shape_env)
        self.assertEqual(str(t2.size(0)), str(t1.size(0)))

    def test_linear_rank3_unbacked_batch_dim(self):
        # [FUNCTION DOC] Checks a rank-3 linear op with unbacked leading dim: under fake mode the
        # output keeps unbacked first dim and static trailing dims; a symbolic make_fx trace of
        # linear with a dimension marked unbacked can be replayed with a different batch size.
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        fake_x = fake_mode.from_tensor(
            torch.randn(4, 16, 256),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[
                    DimDynamic.UNBACKED,
                    DimDynamic.STATIC,
                    DimDynamic.STATIC,
                ],
                constraint_sizes=[None, None, None],
            ),
        )
        fake_weight = fake_mode.from_tensor(torch.randn(3072, 256), static_shapes=True)

        def f(x):
            # [FUNCTION DOC] HELPER: Applies linear with the fixed fake weight to the input.
            return torch.nn.functional.linear(x, fake_weight)

        self.assertTrue(free_unbacked_symbols(fake_x.shape[0]))
        with fake_mode:
            out = f(fake_x)

        self.assertEqual(out.shape[1:], (16, 3072))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

        x = torch.randn(4, 16, 256)
        weight = torch.randn(3072, 256)
        torch._dynamo.decorators.mark_unbacked(
            x, 0, min=2, max=8, hint_override=x.shape[0]
        )

        def trace_f(x, weight):
            # [FUNCTION DOC] HELPER: Applies linear with the given weight to the input.
            return torch.nn.functional.linear(x, weight)

        gm = make_fx(trace_f, tracing_mode="symbolic")(x, weight)
        replay_out = gm(torch.randn(5, 16, 256), weight)
        self.assertEqual(replay_out.shape, (5, 16, 3072))

    def test_view_unbacked_contiguous_relation_from_hints(self):
        # [FUNCTION DOC] Checks that a view of an as_strided tensor over an unbacked
        # sequence dimension (with an optimization hint override) succeeds under
        # fake mode and its leading output dimension retains unbacked symbols.
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        with fake_mode:
            seq = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(seq, 16)
            base = torch.empty(4, seq, 256)
            chunked = torch.as_strided(
                base,
                (4, 2 * (seq // 2), 256),
                (256 * seq, 256, 1),
            )
            out = chunked.view(4 * 2 * (seq // 2), 256)

        self.assertEqual(out.shape, (4 * 2 * (seq // 2), 256))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

    def test_meta_storage_trace_uses_hint_for_symbolic_size(self):
        # [FUNCTION DOC] Checks that the JSON metadata for a meta storage descriptor
        # with an unbacked symbolic size records the size as an expression string
        # and the size hint equal to three times the overridden hint (48).
        from torch._subclasses.meta_utils import MetaStorageDesc, MetaStorageId

        shape_env = ShapeEnv()
        size = shape_env.create_unbacked_symint()
        torch._dynamo.override_optimization_hint(size, 16)

        metadata = MetaStorageDesc(MetaStorageId(0), 3 * size, None).as_json(1)
        self.assertEqual(metadata["size"], "3*u0")
        self.assertEqual(metadata["size_hint"], 48)

    def test_matmul_rank4_unbacked_batch_dim(self):
        # [FUNCTION DOC] Checks a traced matmul of rank-4 tensors with an unbacked
        # batch dim keeps the unbacked batch symbol and the static remaining dims.
        fake_mode, (q, k) = self.fake_with_unbacked_batch(
            torch.randn(4, 2, 8, 16),
            torch.randn(4, 2, 8, 16),
        )

        def f(q, k):
            # [FUNCTION DOC] HELPER: Matmul of q with k transposed over its last two dims.
            return torch.matmul(q, k.transpose(-2, -1))

        with fake_mode:
            gm = make_fx(f)(q, k)
            out = gm(q, k)

        self.assertEqual(out.shape[1:], (2, 8, 8))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

    def test_sdpa_math_unbacked_batch_dim(self):
        # [FUNCTION DOC] Checks that traced scaled dot product attention with the
        # math backend and causal mask preserves the unbacked batch dimension.
        fake_mode, (q, k, v) = self.fake_with_unbacked_batch(
            torch.randn(4, 2, 8, 16),
            torch.randn(4, 2, 8, 16),
            torch.randn(4, 2, 8, 16),
        )

        def f(q, k, v):
            # [FUNCTION DOC] HELPER: Causal scaled dot product
            # attention using only the math backend with set priority.
            with sdpa_kernel([SDPBackend.MATH], set_priority=True):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )

        with fake_mode:
            gm = make_fx(f)(q, k, v)
            out = gm(q, k, v)

        self.assertEqual(out.shape[1:], (2, 8, 16))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

    def test_size_default_unbacked_batch_dim(self):
        # [FUNCTION DOC] Checks that reshaping to the size given by the aten
        # size op on an unbacked-batch input traces to the exact expected
        # graph text (sym_size then view) and keeps the unbacked first dim.
        fake_mode, (x,) = self.fake_with_unbacked_batch(torch.randn(4, 2))

        def f(x):
            # [FUNCTION DOC] HELPER: Reshapes the input to
            # the shape returned by the aten size overload.
            return x.reshape(torch.ops.aten.size.default(x))

        with fake_mode:
            gm = make_fx(f)(x)
            out = gm(x)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    view = torch.ops.aten.view.default(x_1, [sym_size_int, 2]);  x_1 = sym_size_int = None
    return view""",
        )
        self.assertEqual(out.shape[1:], (2,))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

    def test_cudnn_sdpa_unbacked_batch_dim(self):
        # [FUNCTION DOC] Checks the cudnn scaled dot product attention op on fake
        # CUDA tensors with an unbacked batch dim: output and symbolic size keep
        # the unbacked first dim and trailing shape two by eight by sixteen.
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        batch = shape_env.create_unbacked_symint()

        def fake_cuda_attn_tensor(*shape):
            # [FUNCTION DOC] HELPER: Makes a float16 fake tensor of the
            # given shape claiming a CUDA device from an empty meta tensor.
            meta = torch.empty(shape, device="meta", dtype=torch.float16)
            return FakeTensor(fake_mode, meta, torch.device("cuda"))

        q = fake_cuda_attn_tensor(batch, 2, 8, 24)
        k = fake_cuda_attn_tensor(batch, 2, 8, 24)
        v = fake_cuda_attn_tensor(batch, 2, 8, 16)
        self.assertTrue(free_unbacked_symbols(q.shape[0]))

        with fake_mode:
            out = torch.ops.aten._scaled_dot_product_cudnn_attention.default(
                q, k, v, None, False, 0.0, True
            )[0]
            out_size = torch.ops.aten.sym_size.default(out)

        self.assertEqual(out.shape[1:], (2, 8, 16))
        self.assertEqual(out_size[1:], (2, 8, 16))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))
        self.assertTrue(free_unbacked_symbols(out_size[0]))

    # TODO: Support NJT.  There's also some funny business with dynamic shapes
    # which would need to be dealt with as well
    @expectedFailurePropagateRealTensors
    def test_jagged_fake_to_fake_preserved(self):
        # [FUNCTION DOC] Checks converting a jagged nested tensor to fake in two modes
        # sharing a shape env: the second result is a distinct object whose offsets belong to
        # the second mode, symbols are free, and the shape env of a size entry matches the
        # first. Expected to fail with propagate_real_tensors (nested tensors unsupported).
        from torch.nested._internal.nested_tensor import jagged_from_list

        S0, S1, S2 = 3, 4, 5
        D = 4
        a = torch.randn(S0, D, requires_grad=True, dtype=torch.float64)
        b = torch.randn(S1, D, requires_grad=True, dtype=torch.float64)
        c = torch.randn(S2, D, requires_grad=True, dtype=torch.float64)
        offsets = None
        jt, _ = jagged_from_list([a, b, c], offsets)
        shape_env = ShapeEnv()
        mode1 = FakeTensorMode(shape_env=shape_env)
        t1 = mode1.from_tensor(jt)
        mode2 = FakeTensorMode(shape_env=shape_env)
        t2 = mode2.from_tensor(t1)
        # It's not obvious that the invocation above makes it dynamic but it
        # does!
        self.assertTrue(free_symbols(t1.size()))
        self.assertIsNot(t2, t1)
        self.assertIs(t1.offsets().fake_mode, mode1)
        self.assertIs(t2.offsets().fake_mode, mode2)
        self.assertIs(t2.size(1).node.shape_env, t1.size(1).node.shape_env)
        self.assertEqual(str(t2.size(1)), str(t1.size(1)))

    def checkMetaProps(self, t1, t2):
        # [FUNCTION DOC] HELPER: Compares metadata of two tensors
        # including strides via the prims utility; raises on mismatch.
        prims.utils.compare_tensor_meta(t1, t2, check_strides=True)

    @skipIfCrossRef
    def test_deepcopy(self):
        # [FUNCTION DOC] Checks deep-copying a BatchNorm module under the fake copy mode
        # yields fake parameters and buffers with equal metadata, parameter type and
        # requires_grad; also checks that deep-copying a module with aliased tensors
        # preserves aliasing (same object, same storage handle). Skipped under cross-ref.
        with FakeTensorMode() as mode:
            pass
        mod = torch.nn.BatchNorm2d(10)
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            mod_copied = copy.deepcopy(mod)

        def check_copy(mod, mod_copied):
            # [FUNCTION DOC] HELPER: For each parameter and buffer of the original
            # module, checks the same-named entry of the copy has matching metadata,
            # is fake, and agrees on being a Parameter and on requires_grad.
            for name, param in itertools.chain(
                mod.named_parameters(), mod.named_buffers()
            ):
                param_copied = getattr(mod_copied, name)
                self.checkMetaProps(param, param_copied)
                self.assertTrue(is_fake_tensor(param_copied))
                self.assertEqual(
                    isinstance(param, torch.nn.Parameter),
                    isinstance(param_copied, torch.nn.Parameter),
                )
                self.assertEqual(param.requires_grad, param_copied.requires_grad)

        check_copy(mod, mod_copied)

        class ModuleNew(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a random tensor plus two aliases
            # to it (the same tensor and a slice of it) as plain attributes.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the random tensor and
                # attributes a, b (the same object) and c (its first row).
                super().__init__()
                self.a = torch.rand([10, 2])
                self.b = self.a
                self.c = self.a[0]

        mod = ModuleNew()
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            mod_copied = copy.deepcopy(mod)

        self.assertIs(mod_copied.a, mod_copied.b)
        self.assertEqual(mod_copied.b.storage()._cdata, mod_copied.a.storage()._cdata)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_new(self):
        # [FUNCTION DOC] CUDA only: checks the legacy new constructor on fake tensors with sizes, a
        # list, a device keyword and a tensor, giving fake tensors with expected device and size.
        with FakeTensorMode():
            a = torch.rand([16, 1])
            self.checkType(a.new(10, 10), "cpu", [10, 10])
            self.checkType(a.new([1, 2, 3, 4]), "cpu", [4])
            b = torch.rand([4, 4], device="cuda")
            self.checkType(b.new(device="cuda"), "cuda", [0])
            self.checkType(a.new(torch.rand([1])), "cpu", [1])

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_scalar_inputs(self):
        # [FUNCTION DOC] Checks div of Python scalars and multiplication of an int tensor by a
        # float under fake mode: scalar result has empty size, tensor result has float dtype.
        with FakeTensorMode():
            self.checkType(torch.div(3, 2), "cpu", [])
            ten = torch.zeros(2, dtype=torch.int32) * 2.0
            self.assertEqual(ten.dtype, torch.float)
            self.checkType(ten, "cpu", [2])

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_allow_meta(self):
        # [FUNCTION DOC] Checks that operating on meta-device tensors under fake mode works
        # while the config allowing meta is on and raises an exception when it is patched off.
        def run_meta():
            # [FUNCTION DOC] HELPER: Under fake mode creates a
            # meta random tensor and returns its sum with itself.
            with FakeTensorMode():
                x = torch.rand([4], device="meta")
                return x + x

        self.checkType(run_meta(), "meta", [4])

        with patch.object(torch._functorch.config, "fake_tensor_allow_meta", False):
            self.assertRaises(Exception, run_meta)

    def test_embedding_bag_meta(self):
        # [FUNCTION DOC] Checks that an EmbeddingBag module created on the meta device gives
        # output tensors with the same sizes and devices when run for real and under fake mode.
        def f():
            # [FUNCTION DOC] HELPER: Builds a small meta EmbeddingBag
            # and runs it on fixed index and offset tensors.
            # This behavior was originally unintentional but we see people
            # relying on it
            embedding = torch.nn.EmbeddingBag(10, 3, mode="sum", device="meta")
            input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
            offsets = torch.tensor([0, 4], dtype=torch.long)
            return embedding(input, offsets)

        real_out = f()
        with FakeTensorMode():
            fake_out = f()

        for r, f in zip(real_out, fake_out):
            self.assertEqual(r.size(), f.size())
            self.assertEqual(r.device, f.device)

    def test_embedding_meta_indices(self):
        # [FUNCTION DOC] Checks embedding with fake weights and meta or CPU or
        # CUDA index tensors gives fake results of shape indices by 8 that
        # follow the weight's dtype and device; CUDA cases run only if allowed;
        # mixing CPU weight with CUDA indices raises a device mismatch error.
        with FakeTensorMode():
            weight = torch.empty(20, 8)
            # Indices must be in range: under propagate_real_tensors the real
            # embedding gather runs, and uninitialized (torch.empty) indices read
            # out of bounds -- a hard GPU fault on some backends (e.g. ROCm).
            indices = torch.zeros(2, 3, dtype=torch.long, device="meta")
            out = torch.nn.functional.embedding(indices, weight)

            meta_weight = torch.empty(20, 8, dtype=torch.float64, device="meta")
            cpu_indices = torch.zeros(4, 5, dtype=torch.long)
            meta_weight_out = torch.nn.functional.embedding(cpu_indices, meta_weight)

            run_cuda_cases = (
                not torch._functorch.config.fake_tensor_propagate_real_tensors
                or RUN_CUDA
            )
            if run_cuda_cases:
                cuda_weight = torch.empty(20, 8, device="cuda")
                cuda_indices = torch.zeros(2, 3, dtype=torch.long, device="cuda")
                cuda_out = torch.nn.functional.embedding(cuda_indices, cuda_weight)

        self.assertIsInstance(out, FakeTensor)
        self.assertEqual(out.shape, (2, 3, 8))
        self.assertEqual(out.dtype, weight.dtype)
        self.assertEqual(out.device, weight.device)
        self.assertEqual(out.fake_device, weight.fake_device)

        self.assertIsInstance(meta_weight_out, FakeTensor)
        self.assertEqual(meta_weight_out.shape, (4, 5, 8))
        self.assertEqual(meta_weight_out.dtype, meta_weight.dtype)
        self.assertEqual(meta_weight_out.device, meta_weight.device)
        self.assertEqual(meta_weight_out.fake_device, meta_weight.fake_device)

        if run_cuda_cases:
            self.assertIsInstance(cuda_out, FakeTensor)
            self.assertEqual(cuda_out.shape, (2, 3, 8))
            self.assertEqual(cuda_out.device, cuda_weight.device)
            self.assertEqual(cuda_out.fake_device, cuda_weight.fake_device)

        if run_cuda_cases:
            with FakeTensorMode():
                cpu_weight = torch.empty(20, 8)
                cuda_indices = torch.empty(2, 3, dtype=torch.long, device="cuda")
                with self.assertRaisesRegex(
                    (FakeTensorDeviceMismatchError, RuntimeError),
                    "same device|different from other tensors",
                ):
                    torch.nn.functional.embedding(cuda_indices, cpu_weight)

    @skipIfTorchDynamo("uses torch.compile")
    def test_embedding_meta_indices_torch_compile(self):
        # [FUNCTION DOC] Checks that a module doing embedding lookup of positions moved to meta
        # gives the same output shape and device when compiled with the eager backend as in
        # eager; resets dynamo afterwards. Skipped with propagate_real_tensors or under dynamo.
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("uses default FakeTensorMode propagation")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an embedding layer that looks
            # up positions generated by arange and moved to the meta device.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent
                # constructor and creates the embedding layer.
                super().__init__()
                self.position_embedding = torch.nn.Embedding(20, 8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a range of positions matching the
                # input's shape, moves it to meta, and returns the embedding of it.
                positions = (
                    torch.arange(0, x.shape[1])
                    .expand(x.shape[0], x.shape[1])
                    .to("meta")
                )
                return self.position_embedding(positions)

        m = M().eval()
        x = torch.randint(0, 20, (2, 3))
        eager = m(x)
        try:
            compiled = torch.compile(m, backend="eager")(x)
        finally:
            torch._dynamo.reset()

        self.assertEqual(compiled.shape, eager.shape)
        self.assertEqual(compiled.device, eager.device)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_mixed_real_and_fake_inputs(self):
        # [FUNCTION DOC] Checks that a module mixing real parameters
        # with a real input runs under a fake mode that allows non-fake
        # inputs and produces a fake CPU result of the expected shape.
        class _TestPattern(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution and a batch norm that applies
            # the batch norm folding computation pattern: scales conv weights, runs conv
            # with zero bias, undoes the scale, adds the bias, then applies batch norm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the one-channel
                # convolution and batch norm submodules.
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = torch.nn.BatchNorm2d(1)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Runs the folded conv plus batch
                # norm computation on the input and returns the result.
                running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
                scale_factor = self.bn.weight / running_std
                weight_shape = [1] * len(self.conv.weight.shape)
                weight_shape[0] = -1
                bias_shape = [1] * len(self.conv.weight.shape)
                bias_shape[1] = -1
                scaled_weight = self.conv.weight * scale_factor.reshape(weight_shape)
                zero_bias = torch.zeros_like(self.conv.bias, dtype=input.dtype)
                conv = self.conv._conv_forward(input, scaled_weight, zero_bias)
                conv_orig = conv / scale_factor.reshape(bias_shape)
                conv_orig = conv_orig + self.conv.bias.reshape(bias_shape)
                conv = self.bn(conv_orig)
                return conv

        example_inputs = (torch.randn(1, 1, 3, 3),)
        mod = _TestPattern()
        with FakeTensorMode(allow_non_fake_inputs=True):
            out = mod(torch.randn(1, 1, 3, 3))
        self.checkType(out, "cpu", (1, 1, 3, 3))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_aten_copy_multi_device(self):
        # [FUNCTION DOC] CUDA only: the aten copy op (functional and out) across CPU and CUDA
        # fake tensors returns results on the self tensor's device; checks types and sizes.
        with FakeTensorMode():
            x1 = torch.rand(4, device="cpu")
            x2 = torch.rand(4, device="cuda")
            copy1 = torch.ops.aten.copy.default(x1, x2)
            copy2 = torch.ops.aten.copy.default(x2, x1)
            out = torch.empty(4, device="cpu")
            torch.ops.aten.copy.out(x1, x2, out=out)
        self.checkType(copy1, "cpu", (4,))
        self.checkType(copy2, "cuda", (4,))
        self.checkType(out, "cpu", (4,))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_aten_index_multi_device(self):
        # [FUNCTION DOC] CUDA only: the aten index op with CUDA self and CPU indices, and index_put
        # with mixed-device values, returns fake tensors on the self device with expected shapes.
        with FakeTensorMode():
            x1 = torch.rand(4, 4, device="cpu")
            x2 = torch.rand(4, 4, device="cuda")
            i1 = torch.tensor([0, 1], device="cuda")
            i2 = torch.tensor([0, 1], device="cpu")
            # NB: This one does not work: cuda indices not allowed on cpu
            # tensor
            # r1 = torch.ops.aten.index(x1, i1)
            r2 = torch.ops.aten.index(x2, i2)

            y1 = torch.rand(4, device="cpu")
            y2 = torch.rand(4, device="cuda")
            j1 = torch.tensor([2], device="cuda")
            j2 = torch.tensor([2], device="cpu")
            r3 = torch.ops.aten.index_put.default(x1, j1, y1)
            r4 = torch.ops.aten.index_put.default(x2, j2, y2)
        # self.checkType(r1, "cpu", ())
        self.checkType(r2, "cuda", ())
        self.checkType(r3, "cpu", (4, 4))
        self.checkType(r4, "cuda", (4, 4))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_aten_slice_scatter_multi_device(self):
        # [FUNCTION DOC] CUDA only: slice_scatter across CPU and CUDA fake tensors (default and
        # out overloads) returns results on the base tensor's device with shape four by four.
        with FakeTensorMode():
            x1 = torch.rand(4, 4, device="cpu")
            y1 = torch.rand(2, 4, device="cuda")
            x2 = torch.rand(4, 4, device="cuda")
            y2 = torch.rand(2, 4, device="cpu")
            out = torch.empty(4, 4, device="cpu")
            r1 = torch.ops.aten.slice_scatter.default(x1, y1, start=2)
            r2 = torch.ops.aten.slice_scatter.default(x2, y2, start=2)
            r3 = torch.ops.aten.slice_scatter.out(x1, y1, out=out, start=2)
        self.checkType(r1, "cpu", (4, 4))
        self.checkType(r2, "cuda", (4, 4))
        self.checkType(r3, "cpu", (4, 4))
        self.checkType(out, "cpu", (4, 4))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_aten_set_data_multi_device(self):
        # [FUNCTION DOC] CUDA only: the aten shallow-copy-data op moves a
        # fake tensor to the device of the source tensor in both directions.
        with FakeTensorMode():
            x1 = torch.rand(4, device="cpu")
            x2 = torch.rand(4, device="cuda")
            # cpu -> cuda
            torch.ops.aten.shallow_copy_data_(x1, x2)
            self.checkType(x1, "cuda", (4,))
            # cuda -> cpu
            x3 = torch.rand(4, device="cuda")
            x4 = torch.rand(4, device="cpu")
            torch.ops.aten.shallow_copy_data_(x3, x4)
            self.checkType(x3, "cpu", (4,))

    def test__adaptive_avg_pool2d_backward(self):
        # [FUNCTION DOC] Checks that the adaptive average pool 2d backward op on a
        # channels-last input gives a gradient whose suggested memory format is channels-last.
        with FakeTensorMode():
            grad_out = torch.rand(2, 3, 4, 4)
            inp = torch.rand(2, 3, 4, 4).to(memory_format=torch.channels_last)
            grad_in = torch.ops.aten._adaptive_avg_pool2d_backward(grad_out, inp)
            self.assertTrue(
                torch._prims_common.suggest_memory_format(grad_in)
                == torch.channels_last
            )

    def test_suggest_memory_format_with_degenerate_dimensions(self):
        # [FUNCTION DOC] Regression test: a tensor with degenerate unit spatial dims and ambiguous
        # strides must have suggested memory format contiguous (not channels-last), in eager and
        # fake mode, and bilinear upsampling of it gives a contiguous output in both modes.
        """
        Test that suggest_memory_format correctly returns contiguous_format for
        tensors with degenerate dimensions (H=1, W=1) that have ambiguous strides.
        This is a regression test for a bug where the Python implementation checked
        strides[d] > 1 instead of shape[d] > 1, causing incorrect channels_last
        classification. See c10/core/MemoryFormat.h is_channels_last_strides_2d_s4.
        """
        # Create a tensor that mimics the problematic case:
        # shape (1, 512, 1, 1) with strides (512, 1, 512, 1)
        # This comes from permute(0, 2, 1).contiguous().unsqueeze(-1)
        x = torch.randn(1, 1, 512)
        x = x.permute(0, 2, 1).contiguous()  # shape (1, 512, 1), strides (512, 1, 512)
        x = x.unsqueeze(-1)  # shape (1, 512, 1, 1), strides (512, 1, 512, 1)

        # In eager mode, this should be contiguous_format
        self.assertEqual(
            torch._prims_common.suggest_memory_format(x),
            torch.contiguous_format,
            "Eager tensor with degenerate dims should be contiguous_format",
        )

        # In FakeTensor mode, it should also be contiguous_format (not channels_last)
        with FakeTensorMode() as mode:
            fake_x = mode.from_tensor(torch.randn(1, 1, 512))
            fake_x = fake_x.permute(0, 2, 1).contiguous()
            fake_x = fake_x.unsqueeze(-1)

            self.assertEqual(
                torch._prims_common.suggest_memory_format(fake_x),
                torch.contiguous_format,
                "FakeTensor with degenerate dims should be contiguous_format",
            )

        # Test that upsample output has matching strides in eager vs FakeTensor
        upsample = torch.nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        eager_out = upsample(x)
        self.assertTrue(
            eager_out.is_contiguous(), "Eager upsample output should be contiguous"
        )

        with FakeTensorMode() as mode:
            fake_x = mode.from_tensor(torch.randn(1, 1, 512))
            fake_x = fake_x.permute(0, 2, 1).contiguous().unsqueeze(-1)
            fake_out = upsample(fake_x)

            self.assertTrue(
                fake_out.is_contiguous(),
                lambda msg: f"{msg}\nFakeTensor upsample output should be contiguous, got strides {fake_out.stride()}",
            )

    def test_export_numpy(self):
        # [FUNCTION DOC] Checks that exporting a model that converts to numpy and adds
        # random noise, under fake mode in strict mode, returns an exported program.
        class MyNumpyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward converts the input
            # to numpy and adds numpy random noise of the same shape.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the numpy form of the input plus random noise.
                input = input.numpy()
                return input + np.random.randn(*input.shape)

        with FakeTensorMode():
            ep = torch.export.export(
                MyNumpyModel(), args=(torch.randn(1000),), strict=True
            )
            self.assertTrue(isinstance(ep, torch.export.ExportedProgram))

    def test_unsqueeze_copy(self):
        # [FUNCTION DOC] Checks that unsqueeze_copy keeps
        # a dynamic leading dim equal to the input's.
        shape_env = ShapeEnv()
        t1 = torch.ones(2, 2, 768)
        with FakeTensorMode(shape_env=shape_env) as fake_mode:
            t = fake_mode.from_tensor(
                t1,
                symbolic_context=StatelessSymbolicContext(
                    dynamic_sizes=[
                        DimDynamic.DYNAMIC,
                        DimDynamic.STATIC,
                        DimDynamic.STATIC,
                    ],
                ),
            )

        self.assertEqual(t.shape[0], torch.ops.aten.unsqueeze_copy(t, 1).shape[0])

    def test_alias_call(self):
        # [FUNCTION DOC] Checks that multiplying a forward-mode autograd
        # dual tensor under fake mode gives a fake tensor of size three.
        fwAD = torch.autograd.forward_ad

        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by a fixed large integer constant.
            return 4312491 * x

        with torch._subclasses.fake_tensor.FakeTensorMode():
            with fwAD.dual_level():
                x = torch.randn(3, device="cpu")
                y = torch.ones_like(x)
                dual = fwAD.make_dual(x, y)
                r = f(dual)

        self.assertIsInstance(r, FakeTensor)
        self.assertEqual(r.size(), [3])

    @parametrize("reverse", [False, True])
    def test_scan(self, reverse):
        # [FUNCTION DOC] Parametrized over reverse: runs the scan higher-order op under fake mode
        # with an add combine function along dim one and checks both outputs are fake tensors.
        def add(x, y):
            # [FUNCTION DOC] HELPER: Combine function returning the sum twice (carry and output).
            return x + y, x + y

        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.randn((3, 5, 7), device="cpu")
            init = torch.randn((3, 7), device="cpu")
            r = scan(add, init, x, dim=1, reverse=reverse)

        self.assertIsInstance(r[0], FakeTensor)
        self.assertIsInstance(r[1], FakeTensor)

    def test_fast_div_int_to_float(self):
        # [FUNCTION DOC] Checks the fast-path implementation of
        # tensor division on two int32 fake tensors returns float32.
        mode = FakeTensorMode()
        with mode:
            x = torch.empty(2, 2, device="cpu", dtype=torch.int32)
            y = torch.empty(2, 2, device="cpu", dtype=torch.int32)
        from torch._subclasses.fake_impls import get_fast_op_impls

        fast_div = get_fast_op_impls()[torch.ops.aten.div.Tensor]
        z = fast_div(mode, x, y)
        self.assertEqual(z.dtype, torch.float32)

    def test_fast_div(self):
        # [FUNCTION DOC] Checks the fast-path division of an
        # int32 fake tensor by a Python scalar returns float32.
        mode = FakeTensorMode()
        with mode:
            x = torch.empty(2, 2, device="cpu", dtype=torch.int32)
        from torch._subclasses.fake_impls import get_fast_op_impls

        fast_div = get_fast_op_impls()[torch.ops.aten.div.Tensor]
        y = fast_div(mode, x, 2)
        self.assertEqual(y.dtype, torch.float32)

    def test_nanmean_out(self):
        # [FUNCTION DOC] Regression test: nanmean with an out argument
        # under fake mode does not raise and out keeps the input dtype.
        # Regression test to ensure we don't error out.
        with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
            x = torch.randn(10)
            out = torch.empty(())
            torch.nanmean(x, out=out)

        self.assertEqual(out.dtype, x.dtype)

    def test_unbind_copy_out(self):
        # [FUNCTION DOC] Regression test: unbind_copy with a tuple of out tensors
        # under fake mode does not raise and each out keeps the input dtype.
        # Regression test to ensure we don't error out.
        with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
            eye = torch.eye(3)
            out = (torch.zeros(3), torch.zeros(3), torch.zeros(3))
            torch.unbind_copy(eye, out=out)

        self.assertEqual(out[0].dtype, eye.dtype)
        self.assertEqual(out[1].dtype, eye.dtype)
        self.assertEqual(out[2].dtype, eye.dtype)

    @unittest.skipIf(not torch.cuda._is_compiled(), "requires CUDA-compiled PyTorch")
    def test_fake_device_guard_no_use_after_free(self):
        # [FUNCTION DOC] Regression test (needs CUDA-compiled build): runs a
        # script in a subprocess with CUDA devices hidden that compiles a function
        # from several threads, resets dynamo and repeats, and asserts exit code
        # zero (guards against a dangling fake device guard implementation).
        # Regression test: when CUDA is compiled but no devices are visible,
        # FakeTensorMode installs a FakeGuardImpl into the global
        # device_guard_impl_registry.  The impl must outlive all threads that
        # use it; previously it was stored as a thread-local unique_ptr, so
        # the pointer became dangling after the thread exited, causing a
        # segfault when a later thread called deviceCount() on it.
        #
        # CUDA_VISIBLE_DEVICES must be set before torch is imported, so we
        # run the repro in a subprocess.

        script = """\
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import threading
import torch

def f(x):
    return x * x

def run_compile():
    g = torch.compile(f, backend="eager")
    for i in range(10):
        g(torch.randn(i))

threads = [threading.Thread(target=run_compile) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()

torch._dynamo.reset()

threads = [threading.Thread(target=run_compile) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=lambda msg: f"{msg}\nsubprocess failed:\n{result.stderr.decode()}",
        )

    @unittest.skipIf(not torch.cuda._is_compiled(), "requires CUDA-compiled PyTorch")
    def test_cuda_fake_tensor_new_methods_no_device_init(self):
        # [FUNCTION DOC] Needs CUDA-compiled build: runs a subprocess with CUDA hidden that creates
        # a fake CUDA tensor and its new_empty-style derived tensors, asserting they are fake on
        # cuda:0 with float32 dtype and CUDA never gets initialized; asserts exit code zero.
        # CUDA_VISIBLE_DEVICES must be set before torch is imported, so run
        # the repro in a subprocess.

        script = """\
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import numpy before torch to avoid an MKL threading-layer conflict in this
# subprocess environment.
import numpy
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake_tensor

fake_mode = FakeTensorMode()
with fake_mode:
    x = torch.randn(3, device="cuda")

assert not torch.cuda.is_initialized()

new_tensors = [
    x.new_empty([2, 1], device="cuda:0", dtype=torch.float32),
    x.new_empty_strided([2, 1], [1, 2], device="cuda:0", dtype=torch.float32),
    x.new_full([2, 1], 1.0, device="cuda:0", dtype=torch.float32),
    x.new_ones([2, 1], device="cuda:0", dtype=torch.float32),
    x.new_zeros([2, 1], device="cuda:0", dtype=torch.float32),
]

for y in new_tensors:
    assert is_fake_tensor(y)
    assert y.device == torch.device("cuda:0")
    assert y.dtype is torch.float32

assert not torch.cuda.is_initialized()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=lambda msg: f"{msg}\nsubprocess failed:\n{result.stderr.decode()}",
        )

    @unittest.skipIf(
        TEST_ACCELERATOR, "Only execute when an accelerator is not present"
    )
    def test_avoid_device_init_without_backends(self):
        # [FUNCTION DOC] Checks that when no accelerator is present, a new
        # FakeTensorMode reports that it avoids device initialization.
        fake_mode = FakeTensorMode()

        self.assertTrue(
            fake_mode.avoid_device_init,
            "Expected avoid_device_init to return True when no backends are registered",
        )

    @parametrize("is_available", [False, True])
    @skipIfTorchDynamo(
        "TorchDynamo exposes https://github.com/pytorch/pytorch/issues/166696"
    )
    @unittest.skipIf(
        TEST_ACCELERATOR, "Only execute when an accelerator is not present"
    )
    def test_avoid_device_init_with_privateuse1_backend(self, is_available):
        # [FUNCTION DOC] Parametrized over availability: registers a dummy
        # privateuseone device module whose availability query returns the given
        # flag, and checks the mode's avoid-device-init property equals the
        # negation of that flag. Always unregisters the module and removes it from
        # the loaded modules. Skipped when an accelerator exists or under dynamo.
        class _DummyPrivateUse1Module:
            # [CLASS DOC] HELPER: Dummy backend module exposing only a static availability
            # query that returns the enclosing test's availability parameter.
            @staticmethod
            def is_available() -> bool:
                # [FUNCTION DOC] HELPER: Returns the enclosing test's availability flag.
                return is_available

        backend_name = "privateuseone"

        try:
            torch._register_device_module(backend_name, _DummyPrivateUse1Module)

            fake_mode = FakeTensorMode()

            self.assertEqual(fake_mode.avoid_device_init, not is_available)
        finally:
            delattr(torch, backend_name)
            del sys.modules[f"torch.{backend_name}"]

    def test_unique_output_dtype(self):
        # [FUNCTION DOC] Checks that torch.unique under fake mode (with inverse
        # and counts, and with a dim argument) returns outputs whose dtypes equal
        # those of the real call, for float32, float16 and bfloat16 inputs.
        shape_env = ShapeEnv()
        for input_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x_real = torch.randn(10, dtype=input_dtype)
            real_unique, real_inverse, real_counts = torch.unique(
                x_real, return_inverse=True, return_counts=True
            )
            with FakeTensorMode(shape_env=shape_env):
                x = torch.randn(10, dtype=input_dtype)
                fake_unique, fake_inverse, fake_counts = torch.unique(
                    x, return_inverse=True, return_counts=True
                )
                self.assertEqual(fake_unique.dtype, real_unique.dtype)
                self.assertEqual(fake_inverse.dtype, real_inverse.dtype)
                self.assertEqual(fake_counts.dtype, real_counts.dtype)

        # Also test with dim argument
        x_real = torch.randn(3, 4, dtype=torch.float32)
        real_unique, real_inverse, real_counts = torch.unique(
            x_real, dim=0, return_inverse=True, return_counts=True
        )
        with FakeTensorMode(shape_env=shape_env):
            x = torch.randn(3, 4, dtype=torch.float32)
            fake_unique, fake_inverse, fake_counts = torch.unique(
                x, dim=0, return_inverse=True, return_counts=True
            )
            self.assertEqual(fake_unique.dtype, real_unique.dtype)
            self.assertEqual(fake_inverse.dtype, real_inverse.dtype)
            self.assertEqual(fake_counts.dtype, real_counts.dtype)

    def test_assert_tensor_metadata_with_list(self):
        # [FUNCTION DOC] Checks the aten tensor-metadata assertion op
        # accepts list-valued size and stride arguments under fake mode.
        with FakeTensorMode():
            x = torch.empty(10)
            torch.ops.aten._assert_tensor_metadata.default(x, size=[10], stride=[1])

    def test_unique_default_dynamic_output_shape(self):
        # [FUNCTION DOC] Checks the aten unique op on a fake tensor: raises
        # DynamicOutputShapeException by default; with dynamic-output-shape ops
        # allowed in the shape env it returns outputs with the input dtype, int64
        # inverse of the input shape, and an unbacked first output dimension.
        x = torch.tensor([2, 1, 4, 2, 2])
        with FakeTensorMode() as mode:
            fake_x = mode.from_tensor(x)
            with self.assertRaises(DynamicOutputShapeException):
                torch.ops.aten._unique.default(fake_x, False, False)

        shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
        with FakeTensorMode(shape_env=shape_env) as mode:
            fake_x = mode.from_tensor(x)
            fake_unique, fake_inverse = torch.ops.aten._unique.default(
                fake_x, False, True
            )

        self.assertEqual(fake_unique.dtype, x.dtype)
        self.assertEqual(fake_inverse.dtype, torch.int64)
        self.assertEqual(fake_inverse.shape, x.shape)
        self.assertTrue(free_unbacked_symbols(fake_unique.shape[0]))

    def test_select_out_of_bounds(self):
        # [FUNCTION DOC] Checks that select with an out-of-range
        # positive or negative index under fake mode raises IndexError.
        with FakeTensorMode():
            x = torch.randn(3, 4)
            with self.assertRaisesRegex(IndexError, "index .* out of range"):
                torch.select(x, dim=1, index=10)
            with self.assertRaisesRegex(IndexError, "index .* out of range"):
                torch.select(x, dim=1, index=-10)


instantiate_parametrized_tests(FakeTensorTest)


def make_propagate_real_tensors_cls(cls):
    # [FUNCTION DOC] HELPER: Builds a variant of the given test class named with the
    # PropagateRealTensors prefix in which the real-tensor propagation config is on (via the dynamo
    # test-class patching helper, skipped under dynamo, honoring the expected-failure marker),
    # assigns it this module's file and name, and registers it in this module's global namespace.
    cls = make_test_cls_with_patches(
        cls,
        "PropagateRealTensors",
        "_propagate_real_tensors",
        (torch._functorch.config, "fake_tensor_propagate_real_tensors", True),
        xfail_prop="_expected_failure_propagate_real_tensors",
        decorator=skipIfTorchDynamo("propagate_real_tensors affects Dynamo"),
    )
    cls.__file__ = __file__
    cls.__module__ = __name__
    globals()[cls.__name__] = cls


make_propagate_real_tensors_cls(FakeTensorTest)


class FakeTensorConstHandling(TestCase):
    # [CLASS DOC] Tests that fake tensors created from Python scalars/constructors track a constant
    # real value (the constant attribute) and that in-place or aliasing writes invalidate it.
    def assertConst(self, *args):
        # [FUNCTION DOC] HELPER: Asserts every argument has a non-None constant attribute.
        for arg in args:
            self.assertTrue(arg.constant is not None)

    def assertNotConst(self, *args):
        # [FUNCTION DOC] HELPER: Asserts every argument has a None constant attribute.
        for arg in args:
            self.assertTrue(arg.constant is None)

    def test_simple(self):
        # [FUNCTION DOC] Checks the item of a tensor
        # created under fake mode returns the real value.
        with FakeTensorMode():
            x = torch.tensor(4.0)
            self.assertEqual(x.item(), 4.0)

    def test_inplace_add(self):
        # [FUNCTION DOC] Checks in-place add on a constant fake tensor updates
        # its constant value for both the tensor and the returned alias.
        with FakeTensorMode():
            x = torch.tensor(4.0)
            y = x.add_(1)
            self.assertEqual(x.item(), 5.0)
            self.assertEqual(y.item(), 5.0)
            self.assertConst(x, y)

    def test_shared_storages(self):
        # [FUNCTION DOC] Checks that a slice of a constant fake
        # tensor shares storage with it, and so do their constants.
        with FakeTensorMode():
            x = torch.tensor([4.0])
            y = x[:]

            self.assertEqual(x.storage()._cdata, y.storage()._cdata)
            self.assertEqual(x.constant.storage()._cdata, y.constant.storage()._cdata)

    def test_constant_invalidation(self):
        # [FUNCTION DOC] Checks that an in-place add with
        # a non-constant tensor clears the constant.
        with FakeTensorMode():
            x = torch.tensor([1.0])
            self.assertConst(x)
            y = torch.rand([1])
            x.add_(y)
            self.assertNotConst(x)

    def test_inplace_view_invalidation(self):
        # [FUNCTION DOC] Checks that resizing a constant fake
        # tensor in place changes its size and clears the constant.
        with FakeTensorMode():
            x = torch.tensor([1])
            self.assertConst(x)
            x.resize_([2])
            self.assertEqual(x.size(0), 2)
            self.assertNotConst(x)

    def test_fake_tensor_in_intlist_repro(self):
        # [FUNCTION DOC] Checks that a function building a batch shape from a
        # tensor's values raises DataDependentOutputException under fake mode.
        def fn(tensors):
            # [FUNCTION DOC] HELPER: Builds a batch shape list from the count of
            # tensors, the leading dims of the first tensor, and a real tensor's
            # elements, and returns a new tensor of zeros of that shape.
            max_size = torch.tensor([800, 1216], dtype=torch.int64)
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            return tensors[0].new_full(batch_shape, 0.0)

        with self.assertRaises(
            torch._subclasses.fake_tensor.DataDependentOutputException
        ):
            with torch._subclasses.fake_tensor.FakeTensorMode():
                a = torch.randn(3, 800, 1199)
                b = torch.randn(3, 800, 800)
                inputs = [a, b]
                ref = fn(inputs)

    def test_fake_tensor_batch_norm_cpu(self):
        # [FUNCTION DOC] Smoke test: a sequential batch norm and relu in eval mode
        # under the cross-ref fake mode runs on a random input without error.
        with torch._subclasses.CrossRefFakeMode():
            m = torch.nn.Sequential(
                torch.nn.BatchNorm2d(10),
                torch.nn.ReLU(),
            )
            m.eval()
            out = m(torch.randn([2, 10, 8, 8]))

    def test_shared_storage_invalidation(self):
        # [FUNCTION DOC] Checks that writing in place to a slice of a
        # constant fake tensor clears the constant of both tensors.
        with FakeTensorMode():
            x = torch.tensor([1.0])
            y = x[:]
            self.assertConst(x, y)
            y.add_(torch.rand([1]))
            self.assertNotConst(x, y)

    def test_aliased_const_write(self):
        # [FUNCTION DOC] Checks that a broadcast-expanded tensor of a constant is non-constant,
        # and that writing through the expanded tensor leaves the original non-constant too.
        with FakeTensorMode():
            x = torch.tensor([1])
            y = x.expand([4])
            self.assertNotConst(y)
            y[0] = 1
            self.assertNotConst(x)

    def test_constant_propagate_through_functions(self):
        # [FUNCTION DOC] Checks that integer division with truncating rounding
        # of Python scalars under fake mode gives a constant fake tensor.
        with FakeTensorMode():
            y = torch.div(4, 4, rounding_mode="trunc")
            self.assertConst(y)


make_propagate_real_tensors_cls(FakeTensorConstHandling)


def contains_type(type: torch.Type, maybe_contained_type: torch.Type):
    # [FUNCTION DOC] HELPER: Returns whether the second type is a subtype of the first or
    # of any type recursively contained in it (checked through the contained-types list).
    return maybe_contained_type.isSubtypeOf(type) or any(
        contains_type(e, maybe_contained_type) for e in type.containedTypes()
    )


class FakeTensorOpInfoTest(TestCase):
    # [CLASS DOC] Device-generic tests over the custom op database; instantiated per
    # device (CPU and CUDA) and also in a real-tensor-propagating variant (CPU).
    @ops(custom_op_db, dtypes=OpDTypes.any_one)
    def test_fake(self, device, dtype, op):
        # [FUNCTION DOC] For each sample input of the custom op under test, runs the
        # optests fake check on the op with its arguments and keyword arguments.
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            args = (sample_input.input,) + sample_input.args
            kwargs = sample_input.kwargs
            optests.fake_check(op, args, kwargs)


make_propagate_real_tensors_cls(FakeTensorOpInfoTest)
instantiate_device_type_tests(FakeTensorOpInfoTest, globals(), only_for=("cpu", "cuda"))
instantiate_device_type_tests(
    PropagateRealTensorsFakeTensorOpInfoTest,  # noqa: F821
    globals(),
    only_for=("cpu",),
)


class FakeTensorConverterTest(TestCase):
    # [CLASS DOC] Tests for FakeTensorConverter: memoization, storage sharing between views, weak
    # reference cleanup, parameter storage handling, grad dtype preservation and mode ownership.
    def test_memoized_conversion_to_meta(self):
        # [FUNCTION DOC] Checks converting the same real tensor
        # twice with one mode returns the same fake object.
        x = torch.rand(2, 2, 2)
        mode = FakeTensorMode()
        self.assertTrue(mode.from_tensor(x) is mode.from_tensor(x))

    def test_memoized_conversion_from_meta(self):
        # [FUNCTION DOC] Checks that converting a meta tensor to a CPU fake
        # tensor twice with the same converter returns the same object.
        x = torch.rand(2, 2).to(device="meta")
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        self.assertTrue(
            converter.from_meta_and_device(mode, x, "cpu")
            is converter.from_meta_and_device(mode, x, "cpu")
        )

    def test_separate_tensor_storages_view(self):
        # [FUNCTION DOC] Checks that converting a tensor and a
        # view of it gives fake tensors with the same storage id.
        x = torch.rand(2, 2, 2)
        y = x[0]
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        x_conv = converter.from_real_tensor(mode, x)
        y_conv = converter.from_real_tensor(mode, y)
        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))

    @xfailIfTorchDynamo
    def test_separate_tensor_storages_non_view(self):
        # [FUNCTION DOC] Checks that two tensors sharing storage but not related as views
        # convert to the same fake storage; after deleting each pair, the converter's tensor
        # and storage memos shrink from one entry to zero. Expected failure under dynamo.
        x = torch.rand(2, 2, 2)
        y = torch.rand(4, 2)
        y.set_(x.storage())
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        x_conv = converter.from_real_tensor(mode, x)
        y_conv = converter.from_real_tensor(mode, y)
        stor_id = torch._C._storage_id(x_conv)
        self.assertEqual(stor_id, torch._C._storage_id(y_conv))
        del x
        del x_conv
        self.assertEqual(len(converter.tensor_memo), 1)
        self.assertEqual(len(converter.meta_converter.storage_memo), 1)
        del y
        del y_conv
        self.assertEqual(len(converter.tensor_memo), 0)
        self.assertEqual(len(converter.meta_converter.storage_memo), 0)

    def test_parameter_views_keep_full_storage(self):
        # [FUNCTION DOC] Checks that Parameters made from non-overlapping slices of a base
        # tensor convert (static shapes) to fake tensors sharing a storage id with the full
        # base storage size and the original storage offset; under real-tensor propagation
        # also checks the real tensors keep the full storage and equal the originals.
        base = torch.arange(12, dtype=torch.float16)
        x = torch.nn.Parameter(base[:6].view(2, 3))
        y = torch.nn.Parameter(base[6:].view(2, 3))
        self.assertFalse(x._is_view())
        self.assertFalse(y._is_view())

        mode = FakeTensorMode()
        x_conv = mode.from_tensor(x, static_shapes=True)
        y_conv = mode.from_tensor(y, static_shapes=True)

        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))
        self.assertEqual(
            x_conv.untyped_storage().nbytes(), base.untyped_storage().nbytes()
        )
        self.assertEqual(y_conv.storage_offset(), y.storage_offset())
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.assertEqual(
                x_conv.real_tensor.untyped_storage().nbytes(),
                base.untyped_storage().nbytes(),
            )
            self.assertEqual(y_conv.real_tensor.storage_offset(), y.storage_offset())
            self.assertEqual(x_conv.real_tensor, x)
            self.assertEqual(y_conv.real_tensor, y)

    def test_parameter_views_keep_full_storage_symbolic_propagate_real(self):
        # [FUNCTION DOC] Same storage and offset checks as the previous
        # parameter views test but with a symbolic shape env and real-tensor
        # propagation switched on via a config patch while converting.
        base = torch.arange(12, dtype=torch.float16)
        x = torch.nn.Parameter(base[:6].view(2, 3))
        y = torch.nn.Parameter(base[6:].view(2, 3))

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            mode = FakeTensorMode(shape_env=ShapeEnv())
            x_conv = mode.from_tensor(x)
            y_conv = mode.from_tensor(y)

        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))
        self.assertEqual(
            x_conv.real_tensor.untyped_storage().nbytes(),
            base.untyped_storage().nbytes(),
        )
        self.assertEqual(
            y_conv.real_tensor.untyped_storage().nbytes(),
            base.untyped_storage().nbytes(),
        )
        self.assertEqual(y_conv.real_tensor.storage_offset(), y.storage_offset())
        self.assertEqual(x_conv.real_tensor, x)
        self.assertEqual(y_conv.real_tensor, y)

    def test_strided_parameter_views_keep_real_storage_data(self):
        # [FUNCTION DOC] Checks that strided Parameters made from even and odd
        # elements of a base tensor convert with real-tensor propagation on to fake
        # tensors sharing a storage id whose real tensors equal the originals.
        base = torch.arange(11, dtype=torch.float32)
        x = torch.nn.Parameter(base[::2])
        y = torch.nn.Parameter(base[1::2])
        self.assertFalse(x._is_view())
        self.assertFalse(y._is_view())

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            mode = FakeTensorMode()
            x_conv = mode.from_tensor(x, static_shapes=True)
            y_conv = mode.from_tensor(y, static_shapes=True)

        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))
        self.assertEqual(x_conv.real_tensor, x)
        self.assertEqual(y_conv.real_tensor, y)

    def test_functional_tensor_offset_storage_copy(self):
        # [FUNCTION DOC] Checks that a Parameter with a storage offset, faked, made
        # functional, then faked again in another mode (with real-tensor propagation
        # on), keeps its storage offset and a real tensor equal to the original.
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        base = torch.arange(12, dtype=torch.float32)
        y = torch.nn.Parameter(base[6:].view(2, 3))

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            mode = FakeTensorMode()
            y_fake = mode.from_tensor(y, static_shapes=True)
            with FunctionalTensorMode():
                y_functional = FunctionalTensor.to_functional(y_fake)
                refake_mode = FakeTensorMode()
                y_refake = refake_mode.from_tensor(y_functional, static_shapes=True)

        self.assertEqual(y_refake.storage_offset(), y.storage_offset())
        self.assertEqual(y_refake.real_tensor, y)

    def test_refake_unbacked_storage_static_shapes(self):
        # [FUNCTION DOC] Checks that re-faking (static shapes, new mode) a
        # fake tensor whose first dim is unbacked keeps an unbacked first dim
        # and a static second dim of three, with real-tensor propagation off.
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=False):
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)
            with shape_env.ignore_fresh_unbacked_symbols():
                fake = fake_mode.from_tensor(
                    torch.randn(2, 3),
                    source=LocalSource("x", is_input=True),
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[DimDynamic.UNBACKED, DimDynamic.STATIC],
                        constraint_sizes=[None, None],
                    ),
                )

            refake_mode = FakeTensorMode()
            refake = refake_mode.from_tensor(fake, static_shapes=True)

        self.assertTrue(free_unbacked_symbols(refake.shape[0]))
        self.assertEqual(refake.shape[1], 3)

    def test_dead_weak_ref(self):
        # [FUNCTION DOC] Checks that after deleting a fake tensor, its
        # source tensor is no longer in the converter memo, and converting
        # a view afterwards reuses the stored fake storage object.
        x = torch.rand(2, 2, 2)
        y = x[0]
        mode = FakeTensorMode()
        converter = FakeTensorConverter()
        x_conv = converter.from_real_tensor(mode, x)
        x_conv_storage = x_conv.untyped_storage()
        del x_conv
        self.assertFalse(x in converter.tensor_memo)
        y_conv = converter.from_real_tensor(mode, y)
        self.assertIs(x_conv_storage, y_conv.untyped_storage())

    @xfailIfTorchDynamo
    def test_dead_key(self):
        # [FUNCTION DOC] Checks the converter memo holds one entry after converting,
        # returns the same object on repeated conversion, and is empty once the real
        # tensor and fake tensors are deleted; raises AssertionError if the second
        # conversion is a different object. Expected failure under dynamo.
        x = torch.rand(2, 2, 2)
        mode = FakeTensorMode()
        converter = FakeTensorConverter()
        x_conv = converter.from_real_tensor(mode, x)
        self.assertEqual(len(converter.tensor_memo), 1)
        x_conv2 = converter.from_real_tensor(mode, x)
        if x_conv2 is not x_conv:
            raise AssertionError("expected x_conv2 is x_conv")
        del x
        del x_conv
        del x_conv2
        self.assertEqual(len(converter.tensor_memo), 0)

    def test_no_active_mode(self):
        # [FUNCTION DOC] Checks a fake tensor created inside a mode but used outside it: adding
        # two such tensors outside the mode still yields a fake tensor owned by that mode, on CPU.
        with FakeTensorMode() as mode:
            x = torch.empty(2, 2, device="cpu")
            y = torch.empty(2, 2, device="cpu")

        out = x + y
        self.assertEqual(mode, out.fake_mode)
        self.assertTrue(is_fake_tensor(out))
        self.assertEqual(out.device.type, "cpu")

    def test_multiple_modes(self):
        # [FUNCTION DOC] Checks that adding fake tensors owned by two different
        # nested modes raises an exception mentioning mixing fake modes.
        t = torch.rand([4])
        t2 = torch.rand([4])
        with FakeTensorMode() as m:
            with FakeTensorMode() as m2:
                t_fake = m.from_tensor(t)
                t2_fake = m2.from_tensor(t2)

                with self.assertRaisesRegex(Exception, "Mixing fake modes"):
                    t_fake + t2_fake

    def test_separate_mode_error(self):
        # [FUNCTION DOC] Creates one tensor in each of two separate fake modes, then asserts
        # that an exception is raised by the assertion helper given a no-argument function
        # returning the first tensor and the second tensor as an extra call argument.
        # KNOWN ISSUE: vacuous test: the lambda takes no arguments but is called
        # with the second tensor as an extra argument, so a TypeError is raised
        # regardless of fake modes, and the two tensors are never combined.
        with FakeTensorMode():
            x = torch.empty(2, 2, device="cpu")
        with FakeTensorMode():
            y = torch.empty(2, 2, device="cpu")
        self.assertRaises(Exception, lambda: x, y)

    @xfailIfTorchDynamo
    def test_no_ref_cycle(self):
        # [FUNCTION DOC] Checks the mode and its fake tensor can be garbage collected (memo
        # has one entry, then both weak references are dead after deleting the strong
        # ones); raises AssertionError if either survives. Expected failure under dynamo.
        # KNOWN ISSUE: the second weak reference is created from the mode, not
        # the tensor, so the fake tensor liveness is never actually tested.
        x = torch.rand([4])
        mode = FakeTensorMode()
        y = mode.from_tensor(x)
        self.assertEqual(len(mode.fake_tensor_converter.tensor_memo), 1)
        mode_weak = weakref.ref(mode)
        y_weak = weakref.ref(mode)
        del mode
        del y
        if mode_weak() is not None:
            raise AssertionError("expected mode_weak() is None")
        if y_weak() is not None:
            raise AssertionError("expected y_weak() is None")

    def test_grad_dtype_preserved(self):
        # [FUNCTION DOC] Checks that a custom grad dtype on a
        # requires-grad tensor is preserved in the fake tensor.
        t = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
        t.grad_dtype = torch.float32

        mode = FakeTensorMode()
        fake_t = mode.from_tensor(t)
        self.assertEqual(fake_t.grad_dtype, torch.float32)

    def test_grad_dtype_none_preserved(self):
        # [FUNCTION DOC] Checks that a grad dtype of None is preserved in the fake tensor.
        t = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
        t.grad_dtype = None

        mode = FakeTensorMode()
        fake_t = mode.from_tensor(t)
        self.assertIsNone(fake_t.grad_dtype)

    def test_grad_dtype_functional_tensor_no_crash(self):
        # [FUNCTION DOC] Checks that re-faking a functional tensor wrapping a fake tensor with
        # custom grad dtype under functional mode does not crash and still requires grad.
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        t = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
        t.grad_dtype = torch.float32

        mode = FakeTensorMode()
        fake_t = mode.from_tensor(t)
        self.assertEqual(fake_t.grad_dtype, torch.float32)

        with FunctionalTensorMode():
            func_t = FunctionalTensor.to_functional(fake_t)
            # Re-fakifying a FunctionalTensor should not crash even though
            # the inner tensor has a custom grad_dtype.
            re_faked = mode.from_tensor(func_t)
        self.assertTrue(re_faked.requires_grad)

    @skipIfTorchDynamo("make_fx tracing is incompatible with dynamo")
    def test_grad_dtype_make_fx(self):
        # [FUNCTION DOC] Checks that tracing a backward computation with make_fx
        # from a fake input that had a custom grad dtype gives a graph whose
        # gradient dtype equals the eager gradient dtype. Skipped under dynamo.
        def train_step(w):
            # [FUNCTION DOC] HELPER: Sums a float32 version of the doubled
            # input and returns its gradient with respect to the input.
            y = (w.float() * 2).sum()
            (g,) = torch.autograd.grad(y, w)
            return g

        w = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
        w.grad_dtype = torch.float32

        g_eager = train_step(w)

        fake_mode = FakeTensorMode()
        w_fake = fake_mode.from_tensor(w)
        with fake_mode:
            gm = make_fx(train_step)(w_fake)
        g_traced = gm(w)

        self.assertEqual(g_eager.dtype, g_traced.dtype)


make_propagate_real_tensors_cls(FakeTensorConverterTest)


class FakeTensorOperatorInvariants(TestCase):
    # [CLASS DOC] Invariant checks over all registered aten operator schemas versus the fake tensor
    # module's operator tables (device-not-kwarg ops, constructors, like ops), plus assorted fake
    # mode regression tests (storage repr, embedding bag, loss, attention, module moves).
    def get_aten_op(self, schema):
        # [FUNCTION DOC] HELPER: Maps an operator schema in the aten namespace
        # to the corresponding callable op overload object (default overload
        # when none named); raises AssertionError if the namespace is not aten.
        namespace, name = schema.name.split("::")
        overload = schema.overload_name if schema.overload_name else "default"
        if namespace != "aten":
            raise AssertionError(f"expected namespace 'aten', got {namespace!r}")
        return getattr(getattr(torch.ops.aten, name), overload)

    def get_all_aten_schemas(self):
        # [FUNCTION DOC] HELPER: Generator over every registered schema whose namespace is aten.
        for schema in torch._C._jit_get_all_schemas():
            namespace = schema.name.split("::")[0]
            if namespace != "aten":
                continue
            yield schema

    def test_non_kwarg_only_device(self):
        # [FUNCTION DOC] For every aten schema involving tensors, checks that any
        # schema with a non-keyword-only optional device argument has its op listed
        # in the fake tensor module's set of ops with positional device arguments.
        for schema in self.get_all_aten_schemas():
            ten_type = torch._C.TensorType.get()
            if not any(
                contains_type(arg.type, ten_type)
                for arg in itertools.chain(schema.arguments, schema.returns)
            ):
                continue

            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            has_non_kwarg_device = any(
                not arg.kwarg_only and arg.type.isSubtypeOf(opt_device)
                for arg in schema.arguments
            )
            if has_non_kwarg_device:
                self.assertTrue(
                    self.get_aten_op(schema)
                    in torch._subclasses.fake_tensor._device_not_kwarg_ops
                )

    def test_tensor_constructors_all_have_kwarg_device(self):
        # [FUNCTION DOC] For each aten op that the fake tensor module classifies
        # as a tensor constructor, checks its schema has a keyword-only optional
        # device argument (the list-to-tensor op excepted).
        for schema in self.get_all_aten_schemas():
            op = self.get_aten_op(schema)
            if not torch._subclasses.fake_tensor._is_tensor_constructor(op):
                continue

            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            has_kwarg_device = any(
                arg.kwarg_only and arg.type.isSubtypeOf(opt_device)
                for arg in schema.arguments
            )

            self.assertTrue(
                has_kwarg_device or op == torch.ops.aten._list_to_tensor.default
            )

    @unittest.expectedFailure
    def test_sparse_new(self):
        # [FUNCTION DOC] Expected-failure test: creating a sparse tensor via
        # the legacy new constructor with indices, values and extra size
        # under fake mode, which still raises an error (it used to crash).
        with FakeTensorMode():
            indices = torch.randn(1, 1, dtype=torch.int64)
            values = torch.randn(1)
            extra = (2,)
            sparse = torch.randn(1).to_sparse()
            # This used to segfault, now it does not, but it still raises an
            # error
            sparse2 = sparse.new(indices, values, extra)

    def test_tensor_new(self):
        # [FUNCTION DOC] Checks that the legacy Tensor
        # constructor under fake mode returns a FakeTensor.
        with FakeTensorMode():
            x = torch.Tensor([1, 2, 3])
        self.assertIsInstance(x, FakeTensor)

    def test_like_ops(self):
        # [FUNCTION DOC] Checks that every aten schema whose name ends in the like suffix
        # has its op present in the fake tensor module's like-constructor collection.
        for schema in self.get_all_aten_schemas():
            if "_like" == schema.name[-5:]:
                op = self.get_aten_op(schema)
                self.assertIn(
                    op, torch._subclasses.fake_tensor._like_tensor_constructors
                )

    def test_str_storage(self):
        # [FUNCTION DOC] Checks the string forms of the storage of a real tensor
        # (shows values) and of its fake counterpart (shows ellipsis with meta
        # device), both inside and after the mode, using inline expected strings.
        x = torch.zeros(3)
        with FakeTensorMode() as m:
            y = m.from_tensor(x)
            self.assertExpectedInline(
                str(x.storage()),
                """\
 0.0
 0.0
 0.0
[torch.storage.TypedStorage(dtype=torch.float32, device=cpu) of size 3]""",
            )
            self.assertExpectedInline(
                str(y.storage()),
                """\
...
[torch.storage.TypedStorage(dtype=torch.float32, device=meta) of size 3]""",
            )

        self.assertExpectedInline(
            str(y.storage()),
            """\
...
[torch.storage.TypedStorage(dtype=torch.float32, device=meta) of size 3]""",
        )

    # at::_embedding_bag has no op info,
    # and returns extra tensors that at::embedding bag throws away
    def test_embedding_bag_private(self):
        # [FUNCTION DOC] Checks that the private aten embedding bag op under fake mode
        # returns the same number of outputs with the same sizes as the real call.
        args = [
            torch.ones(6, 1),
            torch.ones(6, dtype=torch.int64),
            torch.arange(2, dtype=torch.int64),
            False,
            2,  # mode = max
        ]

        ref_out = torch.ops.aten._embedding_bag(*args)
        with FakeTensorMode() as m:
            meta_args = [
                m.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
            ]
            meta_out = torch.ops.aten._embedding_bag(*meta_args)

        self.assertEqual(len(ref_out), len(meta_out))
        for ref_o, meta_o in zip(ref_out, meta_out):
            self.assertEqual(ref_o.size(), meta_o.size())

    def test_cross_entropy_loss(self):
        # [FUNCTION DOC] Checks that cross entropy with label smoothing
        # under fake mode, with and without class weights, gives an
        # output of the same size as the real call without smoothing.
        inp = torch.randn(3, 5)
        target = torch.randint(5, (3,), dtype=torch.long)
        weight = torch.rand(5)
        fn = torch.nn.functional.cross_entropy
        for w in (weight, None):
            args = (inp, target, w)
            ref = fn(*args)
            with FakeTensorMode() as m:
                meta_args = [
                    m.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
                ]
                meta_out = torch.nn.functional.cross_entropy(
                    *meta_args, label_smoothing=0.5
                )

            self.assertEqual(ref.size(), meta_out.size())

    @skipIfXpu(msg="MetadataMismatchError, torch-xpu-ops: 2802")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support SDPA or pre-SM80 hardware",
    )
    def test_flash_attention(self):
        # [FUNCTION DOC] Runs a module calling the flash attention op on strided
        # half-precision inputs under the cross-ref fake mode (skipped without flash
        # support or on XPU). A metadata mismatch is tolerated only if it is not about
        # the first output and its message matches the expected form for the
        # real-tensor-propagation class or the plain class; other exceptions propagate.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls the flash
            # attention op with a fixed scale and returns nothing.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor only.
                super().__init__()

            def forward(self, arg1, arg2, arg3):
                # [FUNCTION DOC] HELPER: Calls the flash attention op on the
                # three inputs with fixed scale and discards the result.
                torch.ops.aten._scaled_dot_product_flash_attention(
                    arg1, arg2, arg3, scale=0.17677669529663687
                )

        args_new = [
            [
                ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, device_type),
                ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, device_type),
                ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, device_type),
            ],
            [
                ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, device_type),
                ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, device_type),
                ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, device_type),
            ],
        ]
        for args_list in args_new:
            args = [
                rand_strided(bsz, num_heads, seq_len, head_dim)
                for (bsz, num_heads, seq_len, head_dim) in args_list
            ]
            try:
                with torch._subclasses.CrossRefFakeMode():
                    Repro()(*args)
            except MetadataMismatchError as e:
                # We expect the cross ref to succeed for the first output to fail
                # for the rng state, see Note [Seed and Offset]
                self.assertTrue("output[0]" not in str(e))
                if self.__class__.__name__.startswith("PropagateRealTensors"):
                    self.assertTrue(
                        "Real tensor propagation found a metadata mismatch" in str(e)
                    )
                else:
                    self.assertTrue(
                        "found mismatched tensor metadata for output" in str(e)
                    )

    # IMPORTANT!!! Always run even if CUDA is not available
    def test_fake_gpu_no_init(self):
        # [FUNCTION DOC] Always runs: under a fake mode allowing non-fake inputs, empty, ones,
        # zeros, rand and tensor constructors with the GPU device type give tensors with that
        # device type without initializing the device. Skipped under real-tensor propagation.
        # Skip this test, we will try to run CUDA operations to real prop so
        # it clearly will not work on CPU runner
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")

        with FakeTensorMode(allow_non_fake_inputs=True):
            self.assertEqual(torch.empty(10, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(torch.ones(10, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(torch.zeros(10, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(torch.rand(10, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(torch.tensor(3.14, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(
                torch.tensor([[3.14, 2], [1, 2]], device=GPU_TYPE).device.type, GPU_TYPE
            )

    @unittest.skipIf(not torch.backends.cuda.is_built(), "requires CUDA build")
    def test_move_module_under_fake(self):
        # [FUNCTION DOC] Checks that moving a module and input to a GPU device
        # inside fake mode makes all parameters, buffers and the output fake tensors
        # on that device. Needs a CUDA build. Skipped under real-tensor propagation.
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer, a buffer and a
            # parameter whose forward sums the linear output, buffer and parameter.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer,
                # a two by two random buffer and parameter.
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.buffer = torch.nn.Buffer(torch.rand(2, 2))
                self.param = torch.nn.Parameter(torch.rand(2, 2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear of the
                # input plus the buffer plus the parameter.
                return self.linear(x) + self.buffer + self.param

        m = Module()
        input = torch.rand(2, 2)
        gpu_device = torch.device(GPU_TYPE, 0)

        with FakeTensorMode(allow_non_fake_inputs=True):
            m.to(device=gpu_device)
            arg = input.to(device=gpu_device)
            out = m(arg)

        for p in m.parameters():
            self.assertTrue(is_fake_tensor(p))
            self.assertEqual(p.device, gpu_device)
        for b in m.buffers():
            self.assertTrue(is_fake_tensor(b))
            self.assertEqual(b.device, gpu_device)

        self.assertTrue(is_fake_tensor(out))
        self.assertEqual(out.device, gpu_device)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_move_meta_tensor(self):
        # [FUNCTION DOC] CUDA only: moving a real meta tensor to CPU and to the GPU device inside
        # fake mode yields tensors of those device types. Skipped under real-tensor propagation.
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")

        meta_tensor = torch.ones(2, device="meta")
        with FakeTensorMode(allow_non_fake_inputs=True):
            self.assertEqual(meta_tensor.to(device="cpu").device.type, "cpu")
            self.assertEqual(meta_tensor.to(device=GPU_TYPE).device.type, GPU_TYPE)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_conv_c1_backward(self):
        # [FUNCTION DOC] CUDA only: runs a module calling convolution_backward
        # on strided half-precision CUDA inputs under the cross-ref fake mode;
        # no explicit assertion, passes if no mismatch error is raised.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls the convolution backward
            # op on its three inputs with fixed arguments and discards the result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor only.
                super().__init__()

            def forward(self, arg1, arg2, arg3):
                # [FUNCTION DOC] HELPER: Invokes the convolution backward op with fixed
                # stride, padding, dilation, and output mask, discarding the result.
                torch.ops.aten.convolution_backward.default(
                    arg1,
                    arg2,
                    arg3,
                    [1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                    [True, True, False],
                )

        args_new = [
            ((16, 1, 128, 128), (16384, 16384, 128, 1), torch.float16, "cuda"),
            ((16, 64, 128, 128), (1048576, 1, 8192, 64), torch.float16, "cuda"),
            ((1, 64, 3, 3), (576, 9, 3, 1), torch.float16, "cuda"),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args_new]

        with torch._subclasses.CrossRefFakeMode():
            Repro()(*args)

    def test_convolution_backward_channels_last_memory_format(self):
        # [FUNCTION DOC] Regression test (issue 171622): the meta convolution
        # backward must give channels-last gradients for channels-last inputs
        # and contiguous gradients for contiguous inputs.
        """Regression test: meta convolution_backward must predict channels_last
        output strides when inputs are channels_last, matching CUDA/MPS backends.
        See https://github.com/pytorch/pytorch/issues/171622
        """
        with FakeTensorMode():
            # channels_last inputs: outputs should be channels_last
            grad_out = torch.randn(2, 3, 4, 4).to(memory_format=torch.channels_last)
            inp = torch.randn(2, 3, 4, 4).to(memory_format=torch.channels_last)
            weight = torch.randn(3, 3, 3, 3).to(memory_format=torch.channels_last)
            grad_input, grad_weight, _ = torch.ops.aten.convolution_backward(
                grad_out,
                inp,
                weight,
                [3],
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            self.assertTrue(grad_input.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(
                grad_weight.is_contiguous(memory_format=torch.channels_last)
            )

            # contiguous inputs: outputs should be contiguous
            grad_out_c = torch.randn(2, 3, 4, 4)
            inp_c = torch.randn(2, 3, 4, 4)
            weight_c = torch.randn(3, 3, 3, 3)
            grad_input_c, grad_weight_c, _ = torch.ops.aten.convolution_backward(
                grad_out_c,
                inp_c,
                weight_c,
                [3],
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            self.assertTrue(grad_input_c.is_contiguous())
            self.assertTrue(grad_weight_c.is_contiguous())

    def test_no_dispatch_with_like_function(self):
        # [FUNCTION DOC] Checks that a like-constructor call inside the no-dispatch context
        # under fake mode is not seen by an enclosing counting dispatch mode (count stays zero).
        class CountingMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that counts how many ops were
            # dispatched through it and forwards each to the real function.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the dispatch count to zero.
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Increments the count and calls the
                # intercepted function with the given arguments, returning its result.
                self.count += 1
                return func(*args, **kwargs)

        with FakeTensorMode():
            x = torch.randn(2)
            with CountingMode() as mode:
                with no_dispatch():
                    torch.zeros_like(x)

        self.assertEqual(mode.count, 0)

    # PropagateRealTensors installs weakrefs
    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/165387",
    )
    @expectedFailurePropagateRealTensors
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_module_to(self):
        # [FUNCTION DOC] CUDA only (skipped on Linux, ROCm and slow runs; expected failure
        # under real-tensor propagation): checks that moving a Linear module to CUDA
        # inside fake mode changes every state dict entry's device type from cpu to cuda.
        def _check_device(sd, device_type):
            # [FUNCTION DOC] HELPER: Asserts every value of
            # the given mapping has the given device type.
            for v in sd.values():
                self.assertEqual(v.device.type, device_type)

        with FakeTensorMode():
            m = torch.nn.Linear(2, 2)
            _check_device(m.state_dict(), "cpu")
            m.to("cuda")
            _check_device(m.state_dict(), "cuda")


make_propagate_real_tensors_cls(FakeTensorOperatorInvariants)


class FakeTensorPropTest(TestCase):
    # [CLASS DOC] Tests running FakeTensorProp over graph modules.
    def test_fake_tensor_prop_on_nn_module(self):
        # [FUNCTION DOC] Checks FakeTensorProp on a symbolically traced two-layer module
        # whose parameters are faked with the same mode: propagation with that mode gives a
        # fake result of shape five by two; propagation with a fresh, different mode must
        # raise AssertionError (device mismatch), which the test catches and verifies.
        class ToyNnModuleWithParameters(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module with two linear
            # layers (4 to 3 to 2) and a relu in between.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two linear layers.
                super().__init__()
                self.layer1 = torch.nn.Linear(4, 3)
                self.layer2 = torch.nn.Linear(3, 2)

            def forward(self, value):
                # [FUNCTION DOC] HELPER: Applies first linear
                # layer, relu, then second linear layer.
                value = self.layer1(value)
                value = torch.relu(value)
                value = self.layer2(value)
                return value

        model = ToyNnModuleWithParameters()
        value = torch.randn(5, 4)
        # Convert nn.Module to GraphModule so that FakeTensorProp runs.
        graph_model = torch.fx.symbolic_trace(model, (value,))
        # The following block runs FakeTensorProp on graph_module w/to the same FakeTensorMode
        #
        # TODO(wschin): there should be an API to run FakeTensorProp for GraphModule
        # with parameters and buffers.
        with FakeTensorMode() as fake_tensor_mode:

            def to_fake_tensor(x):
                # [FUNCTION DOC] HELPER: Converts a real tensor that is not yet fake into
                # a fake tensor using the enclosing mode; returns anything else unchanged.
                if isinstance(x, torch.Tensor) and not is_fake_tensor(x):
                    return fake_tensor_mode.from_tensor(x)
                return x

            fake_parameters_and_buffers = {
                k: to_fake_tensor(v)
                for k, v in itertools.chain(
                    graph_model.named_parameters(), graph_model.named_buffers()
                )
            }
            with torch.nn.utils.stateless._reparametrize_module(
                graph_model, fake_parameters_and_buffers
            ):
                # This case uses the **same** fake tensor mode to
                #  1. create fake parameters and fake buffers, and
                #  2. run FakeTensorProp
                # The result should be correct.
                result = FakeTensorProp(graph_model, fake_tensor_mode).propagate(value)
                self.assertTrue(is_fake_tensor(result))
                self.assertEqual(result.shape, (5, 2))
                # This case uses the **different** fake tensor modes to
                #  1. create fake parameters and fake buffers, and
                #  2. run FakeTensorProp
                # The following code should fail.
                failed = False
                try:
                    FakeTensorProp(graph_model).propagate(value)
                except AssertionError:
                    # AssertionError: tensor's device must be `meta`, got cpu instead
                    failed = True
                self.assertTrue(failed)

    def test_fake_tensor_prop_on_nn_module_with_optional_args(self):
        # [FUNCTION DOC] Checks FakeTensorProp on a symbolically traced module whose forward
        # has several optional arguments passed as None in the middle, inside a fake mode that
        # allows non-fake inputs and disables fallback kernels; propagation must not raise.
        class OptionalArgumentInBetween(torch.nn.Module):
            # [CLASS DOC] HELPER: Module mimicking models with optional
            # forward arguments: fills missing optional values with random
            # tensors, sums them with the input and squares the result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two unused linear layers.
                super().__init__()
                self.layer1 = torch.nn.Linear(4, 3)
                self.layer2 = torch.nn.Linear(3, 2)

            def forward(self, value, another_value=None, another_optional_value=None):
                # [FUNCTION DOC] HELPER: Replaces missing optional arguments
                # with random tensors like the input, adds them to the
                # input, and returns the product of the sum with itself.
                # Mimic huggingface's `forward` methods which have several optional arguments.
                # For example, GPT accepts forward(self, input_ids, None, attention_mask, ...).
                # To apply FakeTensorProp, its from_real_tensor(...) needs to accept None.
                if another_value is None:
                    another_value = torch.rand_like(value)
                if another_optional_value is None:
                    another_optional_value = torch.rand_like(value)
                value = value + another_value + another_optional_value
                return value * value

        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True, allow_fallback_kernels=False
        )
        with fake_mode:
            model = OptionalArgumentInBetween()
            value = torch.randn(5, 4)
            another_optional_value = torch.randn(5, 4)
            graph_model = torch.fx.symbolic_trace(
                model, (value, None, another_optional_value)
            )
            FakeTensorProp(graph_model, fake_mode).propagate(
                value, None, another_optional_value
            )

    def test_unbacked_shape_realloc(self):
        # [FUNCTION DOC] Checks that FakeTensorProp re-run over a traced graph with a
        # data-dependent nonzero node produces a new unbacked size symbol object
        # (reallocation) that is nevertheless statically known equal to the original.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the nonzero indices of the input.
            return x.nonzero()

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            value = torch.randn(5)
            gm = make_fx(f)(value)
        nonzero_nodes = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.nonzero.default
        ]
        self.assertEqual(len(nonzero_nodes), 1)
        self.assertIsInstance(nonzero_nodes[0].meta["val"].shape[0], torch.SymInt)
        u0 = nonzero_nodes[0].meta["val"].shape[0]
        FakeTensorProp(gm, fake_mode).propagate(value)
        u1 = nonzero_nodes[0].meta["val"].shape[0]
        # Test that this test is actually doing something in that the
        # FakeTensorProp actually triggered a reallocation.  If this assert is
        # failing, it could be because we started memoizing the nnz count for
        # nonzero, which is nice in some sense (no reallocation) but not
        # helpful for this test, which is checking what we do when we have
        # to reallocate.  If so, you need to make this example more
        # complicated (e.g., maybe have a nontrivial computation on the input
        # before feeding it into nonzero, or have some sort of randomness)
        self.assertIsNot(u0, u1)
        self.assertTrue(statically_known_true(u0 == u1))

    def test_nonzero_stride(self):
        # [FUNCTION DOC] Checks the transposed nonzero result of
        # a fake tensor has the same contiguity as the real one.
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            value = torch.ones(5)
            fake_r = value.nonzero()

        r = torch.ones(5).nonzero()

        self.assertEqual(fake_r.T.is_contiguous(), r.T.is_contiguous())

    def test_nan_to_num(self):
        # [FUNCTION DOC] Checks that nan_to_num on a transposed
        # fake tensor keeps the input's size and strides.
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            x = torch.randn(5, 10).t()
            y = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        self.assertEqual(x.size(), y.size())
        self.assertEqual(x.stride(), y.stride())

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_torch_load_with_fake_mode(self):
        # [FUNCTION DOC] CUDA only: saves a state dict (including a TwoTensor entry) to
        # a temp file, loads it under fake mode with default and CUDA map locations,
        # and verifies for every entry the device type and the recorded storage size
        # and checkpoint offset by reading the raw file bytes; then repeats with the
        # saved tensors on CUDA and loading to the default and CPU locations.
        model = torch.nn.Linear(5, 10)
        sd = model.state_dict()
        sd["tt"] = TwoTensor(torch.randn(2), torch.randn(2))

        def _read_tensor_and_check(key, sd_loaded, sd_orig, all_bytes, device):
            # [FUNCTION DOC] HELPER: For one key, checks the loaded tensor's device type matches,
            # and that its fake storage byte sizes equal the originals (for both components of a
            # TwoTensor); then reads the float values from the file bytes at the recorded
            # checkpoint offset, reshapes them, and checks they equal the original value.
            dtype = torch.float32
            t = sd_loaded[key]
            self.assertEqual(t.device.type, device)
            if isinstance(t, TwoTensor):
                untyped_storage_a, untyped_storage_b = (
                    t.a.untyped_storage(),
                    t.b.untyped_storage(),
                )
                self.assertEqual(
                    untyped_storage_a.nbytes(),
                    sd_orig[key].a.untyped_storage().nbytes(),
                )
                self.assertEqual(
                    untyped_storage_b.nbytes(),
                    sd_orig[key].b.untyped_storage().nbytes(),
                )
                offset_a, offset_b = (
                    untyped_storage_a._checkpoint_offset,
                    untyped_storage_b._checkpoint_offset,
                )
                nbytes_a, nbytes_b = (
                    untyped_storage_a.nbytes() // 4,
                    untyped_storage_b.nbytes() // 4,
                )
                result_a = torch.frombuffer(
                    all_bytes, dtype=dtype, count=nbytes_a, offset=offset_a
                ).resize_(t.a.size())
                result_b = torch.frombuffer(
                    all_bytes, dtype=dtype, count=nbytes_b, offset=offset_b
                ).resize_(t.b.size())
                self.assertEqual(TwoTensor(result_a, result_b), sd_orig[key])
            else:
                untyped_storage = t.untyped_storage()
                self.assertEqual(
                    untyped_storage.nbytes(), sd_orig[key].untyped_storage().nbytes()
                )
                offset = untyped_storage._checkpoint_offset
                nbytes = untyped_storage.nbytes() // 4
                result = torch.frombuffer(
                    all_bytes, dtype=dtype, count=nbytes, offset=offset
                ).resize_(t.size())
                self.assertEqual(result, sd_orig[key])

        with TemporaryFileName() as f, torch.serialization.safe_globals([TwoTensor]):
            # Create state_dict to be loaded later
            torch.save(sd, f)
            with open(f, "rb") as g:
                all_bytes = g.read()

            fake_mode = FakeTensorMode()
            with fake_mode:
                sd_loaded = torch.load(f)
            for k in sd:
                _read_tensor_and_check(k, sd_loaded, sd, all_bytes, "cpu")
            with fake_mode:
                sd_loaded = torch.load(f, map_location="cuda")
            for k in sd:
                _read_tensor_and_check(k, sd_loaded, sd, all_bytes, "cuda")

        for k in sd:
            sd[k] = sd[k].to("cuda")

        with TemporaryFileName() as f, torch.serialization.safe_globals([TwoTensor]):
            torch.save(sd, f)
            with open(f, "rb") as g:
                all_bytes = g.read()

            fake_mode = FakeTensorMode()
            with fake_mode:
                sd_loaded = torch.load(f)
            for k in sd:
                _read_tensor_and_check(k, sd_loaded, sd, all_bytes, "cuda")
            with fake_mode:
                sd_loaded = torch.load(f, map_location="cpu")
            for k in sd:
                _read_tensor_and_check(k, sd_loaded, sd, all_bytes, "cpu")

    def test_inference_mode_nonzero_memo(self):
        # [FUNCTION DOC] Regression test: under inference mode in fake mode, boolean
        # mask indexing of two tensors with the same mask yields leading dims statically
        # known equal, i.e. the nonzero memo is stored for inference fake tensors.
        """D106102842: nonzero() memo must be stored for inference FakeTensors.

        Without the fix, SymNumberMemoDescriptor.__set__ skipped memo storage
        for inference tensors, so each boolean indexing (x[mask]) allocated a
        fresh unbacked symint. Two tensors filtered by the same mask would get
        independent symbols, causing GuardOnDataDependentSymNode failures.
        """
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            with torch.inference_mode():
                x = torch.randn(10, 3)
                y = torch.randn(10, 5)
                mask = torch.randn(10) > 0

                self.assertIsInstance(x, FakeTensor)
                self.assertTrue(x.is_inference())

                filtered_x = x[mask]
                filtered_y = y[mask]

                sx = filtered_x.shape[0]
                sy = filtered_y.shape[0]

                # Both shapes should be the same unified symbol because
                # mask.nonzero() is memoized.
                self.assertTrue(statically_known_true(sx == sy))


make_propagate_real_tensors_cls(FakeTensorPropTest)


class FakeTensorSerialization(TestCase):
    # [CLASS DOC] Tests pickling of real tensors while a fake mode is active.
    def test_serialization(self):
        # [FUNCTION DOC] Checks that unpickling a pickled real CPU tensor inside
        # fake mode produces a FakeTensor on meta, while inside the temporary
        # unset-fake context it round-trips to a real tensor on the original device.
        x = torch.tensor([0], device="cpu")
        with FakeTensorMode():
            y = pickle.loads(pickle.dumps(x))
            self.assertEqual(type(y), FakeTensor)
            self.assertEqual(y.device.type, "meta")

            with unset_fake_temporarily():
                y = pickle.loads(pickle.dumps(x))
                self.assertEqual(x.device, y.device)

    def test_serialization_with_tracing(self):
        # [FUNCTION DOC] Checks that unpickling a tensor inside a tracing
        # context holding a fake mode yields a tensor on the original device.
        x = torch.tensor([0], device="cpu")
        with tracing(TracingContext(FakeTensorMode())):
            y = pickle.loads(pickle.dumps(x))
            self.assertEqual(x.device, y.device)


class FakeTensorDispatchCache(TestCase):
    # [CLASS DOC] Tests of the FakeTensorMode dispatch cache: cache key computation, hit, miss
    # and bypass counters, and correctness of cached results across modes of the environment.
    def test_shape_env_settings(self):
        # [FUNCTION DOC] Checks that every boolean default parameter
        # of the shape env's internal initializer has a matching field
        # in the ShapeEnvSettings dataclass, so cache keys include it.
        """
        Validation that any boolean settings in ShapeEnv are present in the
        ShapeEnvSettings. We hope to ensure that any new settings that might
        affect FakeTensor dispatch are included in the cache key calculation.
        If this test fails, consider updating ShapeEnvSettings or change this
        test to omit checking for the new field.
        """
        init_sig = inspect.signature(ShapeEnv._init)
        args = [
            name
            for name, param in init_sig.parameters.items()
            if type(param.default) is bool
        ]

        settings = [f.name for f in dataclasses.fields(ShapeEnvSettings)]
        for arg in args:
            self.assertTrue(arg in settings)

    def _test_cache_key(self, fm, x, y, z):
        # [FUNCTION DOC] HELPER: Computes the cache keys of the add
        # operator for the three inputs with a fresh key state, and asserts
        # the keys of the first two are equal and differ from the third.
        """
        Helper for all test_cache_key_* tests below. Assert that the
        cache keys for inputs x and y are the same, but z is different.
        """
        func = aten.add.Tensor
        state = _CacheKeyState()
        key_x = fm._cache_key(state, func, [x], {})
        key_y = fm._cache_key(state, func, [y], {})
        key_z = fm._cache_key(state, func, [z], {})

        self.assertEqual(key_x, key_y)
        self.assertNotEqual(key_x, key_z)

    def test_cache_key_dtype(self):
        # [FUNCTION DOC] Cache key test: same dtype gives
        # the same key, a different dtype a different one.
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3, dtype=torch.float16)
            y = torch.randn(4, 3, dtype=torch.float16)
            z = x.to(dtype=torch.float32)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_shape(self):
        # [FUNCTION DOC] Cache key test: same shape gives
        # the same key, a different shape a different one.
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = torch.randn(4, 2)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_stride(self):
        # [FUNCTION DOC] Cache key test: same strides gives
        # the same key, different strides a different one.
        with FakeTensorMode() as fm:
            x = torch.randn(4, 2)
            y = torch.randn(4, 2)
            z = x.as_strided((4, 2), (1, 2))
            self._test_cache_key(fm, x, y, z)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cache_key_device(self):
        # [FUNCTION DOC] CUDA only cache key test: CPU
        # tensors share a key and a CUDA tensor differs.
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = x.to(device="cuda")
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_memory_format(self):
        # [FUNCTION DOC] Cache key test: contiguous inputs
        # share a key, a channels-last input differs.
        with FakeTensorMode() as fm:
            x = torch.randn(1, 2, 3, 4)
            y = torch.randn(1, 2, 3, 4)
            z = x.to(memory_format=torch.channels_last)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_storage_offset(self):
        # [FUNCTION DOC] Cache key test: equal storage offsets
        # share a key, a tensor without the offset differs.
        with FakeTensorMode() as fm:
            x = torch.randn(3)[1:]
            y = torch.randn(3)[1:]
            z = torch.randn(2)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_requires_grad(self):
        # [FUNCTION DOC] Cache key test: a tensor requiring grad gets a different key.
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = torch.randn(4, 3, requires_grad=True)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_is_conj(self):
        # [FUNCTION DOC] Cache key test: a complex tensor with
        # the conjugate bit flipped gets a different key.
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3, dtype=torch.complex64)
            y = torch.randn(4, 3, dtype=torch.complex64)
            z = torch.randn(4, 3, dtype=torch.complex64)
            torch._C._set_conj(z, not z.is_conj())
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_is_neg(self):
        # [FUNCTION DOC] Cache key test: a complex tensor
        # with the negative bit flipped gets a different key.
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3, dtype=torch.complex64)
            y = torch.randn(4, 3, dtype=torch.complex64)
            z = torch.randn(4, 3, dtype=torch.complex64)
            torch._C._set_neg(z, not z.is_neg())
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_is_inference(self):
        # [FUNCTION DOC] Cache key test: a fake tensor made from an
        # inference-mode tensor gets a different key from normal tensors.
        with torch.inference_mode(True):
            t = torch.randn(4, 3)
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = fm.from_tensor(t)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_constants(self):
        # [FUNCTION DOC] Cache key test: scalar constants of float and int type that hash
        # equal (one point zero versus one, zero point zero versus zero) get different keys.
        with FakeTensorMode() as fm:
            # Python hashes 1.0 to the same value as 1. Make sure the
            # cache key calculation differentiates them.
            self._test_cache_key(fm, 1.0, 1.0, 1)
            self._test_cache_key(fm, 0.0, 0.0, 0)

    def test_empty_list(self):
        # [FUNCTION DOC] Checks the cache key for the any op with an empty
        # list argument differs from the key with that argument omitted.
        with FakeTensorMode() as fm:
            func = aten.any.dims
            state = _CacheKeyState()
            x = torch.ones((2, 3))
            key_x = fm._cache_key(state, func, [x, []], {})
            key_y = fm._cache_key(state, func, [x], {})

        self.assertNotEqual(key_x, key_y)

    def assertHitsMisses(self, hits, misses):
        # [FUNCTION DOC] HELPER: Reads the mode's global cache statistics
        # and asserts the hit and miss counts equal the given numbers.
        """
        Helper to assert on the number of recorded hits and misses.
        """
        info = FakeTensorMode.cache_info()
        self.assertEqual(info.hits, hits)
        self.assertEqual(info.misses, misses)

    def assertBypasses(self, reason, count):
        # [FUNCTION DOC] HELPER: Reads the cache statistics and asserts
        # that the bypass counter for the given reason equals the count;
        # when the count is zero it asserts the reason is absent.
        """
        Helper to assert on the number of recorded bypasses.
        """
        info = FakeTensorMode.cache_info()
        if count > 0:
            self.assertIn(reason, info.bypasses)
            self.assertEqual(info.bypasses[reason], count)
        else:
            self.assertNotIn(reason, info.bypasses)

    def test_cache_hit(self):
        # [FUNCTION DOC] Checks that the first add after clearing the cache
        # is a miss and the second a hit, with equal result metadata.
        """
        Test that cache hit/miss counters are updated correctly.
        """
        with FakeTensorMode():
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)
            res1 = x + y
            self.assertHitsMisses(0, 1)
            res2 = x + y
            self.assertHitsMisses(1, 1)

            self.assertEqual(
                extract_tensor_metadata(res1),
                extract_tensor_metadata(res2),
            )

    def test_cache_bypass_prims_as_strided(self):
        # [FUNCTION DOC] Checks that the as_strided primitive on fake tensors bypasses the cache
        # each time, counted under the primitive's reason string (two bypasses for two calls).
        x = torch.empty(0, 8)
        y = torch.empty(0, 8)
        mode = FakeTensorMode(shape_env=ShapeEnv())
        x = mode.from_tensor(x, static_shapes=True)
        y = mode.from_tensor(y, static_shapes=True)

        with mode:
            FakeTensorMode.cache_clear()
            prims.as_strided(x, (0, 8), (8, 1), 0)
            prims.as_strided(y, (0, 8), (8, 1), 0)

            self.assertBypasses("prims.as_strided", 2)

    def test_cache_bypass(self):
        # [FUNCTION DOC] Checks the bypass counter for in-place view ops
        # starts at zero and becomes one after an in-place unsqueeze.
        """
        Test that cache bypass counters are updated correctly.
        """
        with FakeTensorMode():
            x = torch.randn(1, 2)

            FakeTensorMode.cache_clear()
            self.assertBypasses("inplace view", 0)

            x.unsqueeze_(0)
            self.assertBypasses("inplace view", 1)

    def test_cache_default_dtype(self):
        # [FUNCTION DOC] Checks the cache respects the global default dtype:
        # adding a float to an int tensor misses under each new default (float32,
        # float16) with the matching result dtype and hits when switching back to
        # float32. Changes the global default dtype as a side effect.
        """
        Test that the default dtype is respected when serving cached results.
        """
        with FakeTensorMode():
            x = torch.tensor([1, 2], dtype=torch.int32)
            torch.set_default_dtype(torch.float32)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            y = x + 1.0
            self.assertEqual(y.dtype, torch.float32)
            self.assertHitsMisses(0, 1)

            torch.set_default_dtype(torch.float16)
            y = x + 1.0
            self.assertEqual(y.dtype, torch.float16)
            self.assertHitsMisses(0, 2)

            torch.set_default_dtype(torch.float32)
            y = x + 1.0
            self.assertEqual(y.dtype, torch.float32)
            self.assertHitsMisses(1, 2)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cache_default_device(self):
        # [FUNCTION DOC] CUDA only: checks the cache respects the global default device:
        # results follow the default device (cpu, cuda, cpu) with misses for the first two
        # and a hit when returning to cpu. Resets the default device to none afterwards.
        """
        Test that the default device is respected when serving cached results.
        """
        with FakeTensorMode():
            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            try:
                torch.set_default_device("cpu")
                x = torch.tensor([1, 2])
                y = x + 1.0
                self.assertEqual(y.device.type, "cpu")
                self.assertHitsMisses(0, 1)

                torch.set_default_device("cuda")
                x = torch.tensor([1, 2])
                y = x + 1.0
                self.assertEqual(y.device.type, "cuda")
                self.assertHitsMisses(0, 2)

                torch.set_default_device("cpu")
                x = torch.tensor([1, 2])
                y = x + 1.0
                self.assertEqual(y.device.type, "cpu")
                self.assertHitsMisses(1, 2)
            finally:
                torch.set_default_device(None)

    def test_cache_inplace_op(self):
        # [FUNCTION DOC] Checks that an in-place add served from the cache still returns
        # the same input object (identity preserved) on both the miss and the hit.
        """
        Test that inplace ops served from the cache correctly reference the
        input parameter.
        """
        with FakeTensorMode():
            x = torch.randn(1, 2)
            y = torch.randn(1, 2)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            z = x.add_(y)
            self.assertHitsMisses(0, 1)
            self.assertEqual(id(x), id(z))

            w = x.add_(y)
            self.assertHitsMisses(1, 1)
            self.assertEqual(id(x), id(w))

    def test_cache_view_op(self):
        # [FUNCTION DOC] Checks that in-place multiplication on a non-view and on a view
        # tensor report the right view status regardless of which was cached first.
        """
        Test that view ops are handled correctly when served from the cache.
        """
        with FakeTensorMode():
            x1 = torch.ones(2, requires_grad=True).clone()
            x2 = torch.ones(2, requires_grad=True).clone()
            y2 = x2.view(-1)

            # Test operating on a non-view tensor, then the same operation
            # on a view tensor. Assert that the view property is set correctly.
            z1 = x1.mul_(2)
            self.assertFalse(z1._is_view())

            z2 = y2.mul_(2)
            self.assertTrue(z2._is_view())

            # Now the other way around: first operate on a view tensor, then
            # the same operation on a non-view tensor.
            z2 = y2.mul_(2)
            self.assertTrue(z2._is_view())

            z1 = x1.mul_(2)
            self.assertFalse(z1._is_view())

    def test_cache_dispatch_key_set(self):
        # [FUNCTION DOC] Checks that creating efficient zero tensors, which
        # change the dispatch key set, bypasses the cache each time under
        # the dispatch-key-set mismatch reason (counts one then two).
        """
        Test that operations that change the dispatch key set bypass caching.
        """
        with FakeTensorMode():
            FakeTensorMode.cache_clear()
            self.assertBypasses("dispatch_key_set mismatch", 0)

            x = torch._efficientzerotensor(3)
            self.assertTrue(x._is_zerotensor())
            self.assertBypasses("dispatch_key_set mismatch", 1)

            y = torch._efficientzerotensor(3)
            self.assertTrue(y._is_zerotensor())
            self.assertBypasses("dispatch_key_set mismatch", 2)

    def test_fft_hfft2_issue145522(self):
        # [FUNCTION DOC] Regression test for issue 145522: the hermitian
        # fft 2d op with out argument, size, dim and orthonormal norm
        # under fake mode returns a result with the out tensor's shape.
        with FakeTensorMode():
            s0 = 5
            s1 = 6
            s2 = 7
            s3 = 3
            s4 = 10
            s5 = 2
            x = torch.randn(s0, s1, s2)
            out = torch.randn(s0, s3, s4)
            kwargs = {
                "s": (s3, s4),
                "dim": (1, s5),
                "norm": "ortho",
            }
            r = torch._C._fft.fft_hfft2(x, **kwargs, out=out)
            self.assertEqual(r.shape, out.shape)

    def test_inference_mode(self):
        # [FUNCTION DOC] Checks cache behavior with inference mode: the first add inside
        # and outside inference mode both miss with matching inference flags, repeated
        # calls hit, and cached results have the same metadata as the originals.
        """
        Test that caching handles inference mode correctly.
        """
        with FakeTensorMode():
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            # Expect a miss when the inference mode is different
            res1 = x + y
            with torch.inference_mode():
                res2 = x + y

            self.assertHitsMisses(0, 2)
            self.assertFalse(res1.is_inference())
            self.assertTrue(res2.is_inference())

            # Second tries should see hits
            res3 = x + y

            self.assertHitsMisses(1, 2)
            self.assertFalse(res3.is_inference())
            self.assertEqual(
                extract_tensor_metadata(res1),
                extract_tensor_metadata(res3),
            )

            with torch.inference_mode():
                res4 = x + y

            self.assertHitsMisses(2, 2)
            self.assertTrue(res4.is_inference())
            self.assertEqual(
                extract_tensor_metadata(res2),
                extract_tensor_metadata(res4),
            )

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_wrapper_tensor_subclass_different_device(self):
        # [FUNCTION DOC] CUDA only: converts a wrapper tensor subclass that reports a
        # CPU outer device with an inner CUDA tensor, and checks the fake copy remains
        # of the same subclass, with CPU outer device and non-CPU inner tensor.
        class DifferentDeviceTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass whose outer device is always CPU
            # while it holds an inner tensor (possibly on CUDA); dispatch unwraps wrapper
            # arguments and runs the op on the inner tensors, returning plain tensors.
            @staticmethod
            def __new__(cls, a):
                # [FUNCTION DOC] HELPER: Creates the wrapper with the inner tensor's size, strides,
                # storage offset, layout, requires_grad and dtype but forcing the CPU device.
                kwargs = {}
                kwargs["strides"] = a.stride()
                kwargs["storage_offset"] = a.storage_offset()
                kwargs["device"] = torch.device("cpu")
                kwargs["layout"] = a.layout
                kwargs["requires_grad"] = a.requires_grad
                kwargs["dtype"] = a.dtype
                out = torch.Tensor._make_wrapper_subclass(cls, a.size(), **kwargs)
                return out

            def __init__(self, a):
                # [FUNCTION DOC] HELPER: Stores the wrapped tensor as the inner tensor.
                self.inner_tensor = a

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a string showing
                # the class name around the inner tensor's repr.
                return f"DifferentDeviceTensor({repr(self.inner_tensor)})"

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Reports the inner tensor attribute name
                # list and no metadata, for the tensor subclass flattening protocol.
                return ["inner_tensor"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Rebuilds the wrapper from the inner tensor
                # mapping; raises AssertionError if the metadata is not None.
                if meta is not None:
                    raise AssertionError(f"expected meta is None, got {meta}")
                return DifferentDeviceTensor(inner_tensors["inner_tensor"])

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch hook: replaces any wrapper
                # arguments (positional and keyword) with their inner tensors
                # and runs the function, returning its unwrapped result.
                if kwargs is None:
                    kwargs = {}
                args = pytree.tree_map_only(
                    DifferentDeviceTensor, lambda x: x.inner_tensor, args
                )
                kwargs = pytree.tree_map_only(
                    DifferentDeviceTensor, lambda x: x.inner_tensor, kwargs
                )
                # Returns unwrapped tensor
                return func(*args, **kwargs)

        a = torch.ones(2, 2, 768, device="cuda")
        wrapped_a = DifferentDeviceTensor(a)

        # Outer Tensor is on cpu, inner is on cuda
        self.assertTrue(wrapped_a.is_cpu)
        self.assertFalse(wrapped_a.inner_tensor.is_cpu)

        with FakeTensorMode() as fake_mode:
            fake_wrapped_a = fake_mode.from_tensor(wrapped_a)

        self.assertTrue(fake_wrapped_a.is_cpu)
        if not isinstance(fake_wrapped_a, DifferentDeviceTensor):
            raise AssertionError(
                f"expected DifferentDeviceTensor, got {type(fake_wrapped_a)}"
            )
        self.assertFalse(fake_wrapped_a.inner_tensor.is_cpu)

    def test__upsample_bilinear2d_aa_backward_dynamic_shapes(self):
        # [FUNCTION DOC] Checks that antialiased bilinear interpolation
        # backward on a fake input with dynamic spatial dims and the python
        # dispatcher works and leaves the gradient with the input's shape.
        def f(x):
            # [FUNCTION DOC] HELPER: Bilinear antialiased interpolation
            # of the input to 256 by 256 without corner alignment.
            return torch.nn.functional.interpolate(
                x,
                size=[256, 256],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        shape_env = ShapeEnv()
        fake_m = FakeTensorMode(shape_env=shape_env)
        x = fake_m.from_tensor(
            torch.randn(1, 3, 2005, 1920, requires_grad=True),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[
                    DimDynamic.STATIC,
                    DimDynamic.STATIC,
                    DimDynamic.DYNAMIC,
                    DimDynamic.DYNAMIC,
                ],
                constraint_sizes=[None, None, None, None],
            ),
        )
        with fake_m, enable_python_dispatcher():
            out = f(x)
            out.sum().backward()
            self.assertEqual(x.shape, x.grad.shape)

    def test_from_buffer(self):
        # [FUNCTION DOC] Checks that building an untyped storage from a pickled byte
        # buffer, then a byte tensor from it, under fake mode gives a fake CPU tensor.
        with FakeTensorMode():
            obj = [1, 2]
            f = io.BytesIO()
            pickle.Pickler(f).dump(obj)
            storage = torch.UntypedStorage.from_buffer(f.getvalue(), dtype=torch.uint8)

            t = torch.ByteTensor(storage)
            self.assertTrue(is_fake_tensor(t))
            self.assertEqual(t.device, torch.device("cpu"))

    def test_meta_tensor_to_fake_cpu(self):
        # [FUNCTION DOC] Checks that moving a real meta tensor to CPU in
        # a mode allowing non-fake inputs produces a fake CPU tensor.
        x = torch.randn(4, 4, device="meta")
        with FakeTensorMode(allow_non_fake_inputs=True):
            x_cpu = x.to(device="cpu")
        self.assertTrue(is_fake_tensor(x_cpu))
        self.assertEqual(x_cpu.device, torch.device("cpu"))

    def test_cache_tuple_outputs(self):
        # [FUNCTION DOC] Checks that split (an op with tuple output) is cached: the first call
        # misses, the second with a different input hits, and the pieces have equal metadata.
        """
        Test to check that ops with tuple outputs work.
        """
        with FakeTensorMode():
            x = torch.randn(6, 4)
            y = torch.randn(6, 4)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            ref = torch.split(x, 2)
            self.assertHitsMisses(0, 1)

            res = torch.split(y, 2)
            self.assertHitsMisses(1, 1)
            self.assertEqual(len(ref), len(res))
            for a, b in zip(ref, res):
                self.assertEqual(
                    extract_tensor_metadata(a),
                    extract_tensor_metadata(b),
                )

    def test_cache_aten_index(self):
        # [FUNCTION DOC] Checks the aten index op with a None entry and two
        # index tensors records three misses then a hit with equal metadata;
        # boolean or int8 index tensors raise DynamicOutputShapeException.
        with FakeTensorMode():
            x = torch.randn(4, 4, 4)
            idx_tensor1 = torch.tensor([0, 2, 3])
            idx_tensor2 = torch.tensor([0, 1, 2])

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            ref = torch.ops.aten.index(x, [None, idx_tensor1, idx_tensor2])
            self.assertHitsMisses(0, 3)

            res = torch.ops.aten.index(x, [None, idx_tensor1, idx_tensor2])
            self.assertHitsMisses(1, 3)
            self.assertEqual(extract_tensor_metadata(ref), extract_tensor_metadata(res))

        with FakeTensorMode():
            x = torch.randn(4, 4, 4)
            idx_tensor1 = torch.tensor([True, True, False, True])
            self.assertRaises(
                DynamicOutputShapeException,
                lambda: torch.ops.aten.index(x, [None, idx_tensor1]),
            )

            idx_tensor1 = torch.tensor([1, -2, 3, -4], dtype=torch.int8)
            self.assertRaises(
                DynamicOutputShapeException,
                lambda: torch.ops.aten.index(x, [None, idx_tensor1]),
            )

    @skipIfWindows(
        msg="weird bug - cache may not be cleared after https://github.com/pytorch/pytorch/pull/154283"
    )
    @skipIfTorchDynamo("cache hit/miss changes with invoke_subgraph caching")
    def test_invoke_subgraph(self):
        # [FUNCTION DOC] Checks caching of the invoke-subgraph higher-order op: a plain
        # function argument bypasses the cache each call (counted) while inner ops hit; a
        # recorded graph module caches with hits on rerun; after dynamo reset and garbage
        # collection no invoke-subgraph cache keys remain. Skipped on Windows and under dynamo.
        """
        Tests invoke subgraph
        """
        invoke_subgraph = torch._higher_order_ops.invoke_subgraph

        def run():
            # [FUNCTION DOC] HELPER: Runs the invoke-subgraph checks described in
            # the test: plain function bypass counts, graph module hit counts, equal
            # results, and that at least one subgraph cache key exists afterwards.
            def fn(x, y):
                # [FUNCTION DOC] HELPER: Returns a one-element tuple with x plus twice y.
                return (x + y * 2,)

            # Ensure there is no caching for non-Fx graph module inputs
            with FakeTensorMode():
                x = torch.randn(6, 4)
                y = torch.randn(6, 4)

                FakeTensorMode.cache_clear()
                self.assertHitsMisses(0, 0)

                ref = invoke_subgraph(fn, "subgraph", x, y)
                self.assertHitsMisses(0, 2)
                self.assertBypasses("function argument", 1)

                res = invoke_subgraph(fn, "subgraph", x, y)
                # The hits are from the ops inside fn
                self.assertHitsMisses(2, 2)
                self.assertBypasses("function argument", 2)

                res = invoke_subgraph(fn, "subgraph", x, y)
                # The hits are from the ops inside fn
                self.assertHitsMisses(4, 2)
                self.assertBypasses("function argument", 3)

            # Get the mod as if its going through torch.compile
            backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
            x = torch.randn(6, 4)
            y = torch.randn(6, 4)
            torch.compile(fn, backend=backend, fullgraph=True)(x, y)
            self.assertEqual(len(backend.fw_graphs), 1)
            mod = backend.fw_graphs[0]

            # Ensure that we see hits every time
            with FakeTensorMode():
                x = torch.randn(6, 4)
                y = torch.randn(6, 4)

                FakeTensorMode.cache_clear()
                self.assertHitsMisses(0, 0)

                ref = invoke_subgraph(mod, "subgraph", x, y)
                self.assertHitsMisses(0, 3)

                res = invoke_subgraph(mod, "subgraph", x, y)
                # The hits are from re-running the subgraph
                self.assertHitsMisses(1, 3)

                res = invoke_subgraph(mod, "subgraph", x, y)
                # The hits are from re-running the subgraph
                self.assertHitsMisses(2, 3)

                self.assertEqual(len(ref), len(res))
                for a, b in zip(ref, res):
                    self.assertEqual(
                        extract_tensor_metadata(a),
                        extract_tensor_metadata(b),
                    )
            self.assertTrue(count_invoke_subgraph_keys() > 0)

        def count_invoke_subgraph_keys():
            # [FUNCTION DOC] HELPER: Counts the cache keys in the mode's
            # global cache whose first key element is a higher-order operator.
            invoke_subgraph_keys = 0
            for cache_key in FakeTensorMode.cache:
                if isinstance(cache_key.key[0], torch._ops.HigherOrderOperator):
                    invoke_subgraph_keys += 1
            return invoke_subgraph_keys

        # Check that the graph gc clears the cache
        run()
        torch.compiler.reset()
        gc.collect()
        gc.collect()  # second collection needed for 3.14t
        self.assertTrue(count_invoke_subgraph_keys() == 0)

    @skipIfTorchDynamo("cache hit/miss changes with invoke_subgraph caching")
    def test_invoke_subgraph_cacheable_inplace(self):
        # [FUNCTION DOC] Checks that a recorded graph containing an in-place view
        # op is still cached when invoked through invoke-subgraph: one run misses
        # three times, later runs hit once more each time. Skipped under dynamo.
        invoke_subgraph = torch._higher_order_ops.invoke_subgraph

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies aten cos, an in-place transpose
            # and a multiply with y, returning a one-element tuple.
            # aten ops are used so that eager backend graph is suitable for fake
            # tensor testing
            cos = torch.ops.aten.cos.default(x)
            # inplace-view - this should cause the whole invoke_subgraph to not
            # being able to cache
            t = torch.ops.aten.t_.default(cos)
            mul = torch.ops.aten.mul.Tensor(t, y)
            return (mul,)

        # Get the mod as if its going through torch.compile
        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        torch.compile(fn, backend=backend, fullgraph=True)(x, y)
        self.assertEqual(len(backend.graphs), 1)
        mod = backend.graphs[0]

        # Ensure that invoke_subgraph result is still cached
        with FakeTensorMode():
            x = torch.randn(4, 4)
            y = torch.randn(4, 4)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            ref = invoke_subgraph(mod, "subgraph", x, y)
            self.assertHitsMisses(0, 3)

            res = invoke_subgraph(mod, "subgraph", x, y)
            # The hits are from the ops inside fn and not the subgraph
            self.assertHitsMisses(1, 3)

            res = invoke_subgraph(mod, "subgraph", x, y)
            # The hits are from the ops inside fn and not the subgraph
            self.assertHitsMisses(2, 3)

            self.assertEqual(len(ref), len(res))
            for a, b in zip(ref, res):
                self.assertEqual(
                    extract_tensor_metadata(a),
                    extract_tensor_metadata(b),
                )

    @skipIfTorchDynamo("cache hit/miss changes with invoke_subgraph caching")
    def test_unbacked_output(self):
        # [FUNCTION DOC] Checks that exporting and decomposing a module whose
        # op has an output symbol but no input symbols bypasses the cache
        # twice under the unrepresented-symbol reason. Skipped under dynamo.
        # The point of this test is to have an op which has no symbols as input
        # but a symbol as an output and make sure that we skip caching it.
        class LengthsGather(torch.nn.Module):
            # [CLASS DOC] HELPER: Module gathering offsets by indices and repeating
            # them by gathered lengths, producing a data-dependent output size.
            def forward(
                self,
                input: torch.Tensor,
                lengths: torch.Tensor,
                indices: torch.Tensor,
                offsets: torch.Tensor,
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Gathers bias and lengths by indices and
                # returns the repeat-interleave of the bias by the selected lengths.
                bias = torch.gather(offsets, 0, indices)
                lengths_selected = torch.gather(lengths, 0, indices)
                index = torch.repeat_interleave(bias, lengths_selected, dim=0)
                return index

        input = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        lengths = torch.tensor([0, 2, 3, 1, 4])
        indices = torch.tensor([2, 3, 4, 6, 7, 8, 9])
        offsets = torch.cumsum(lengths, 0)
        ep = torch.export.export(
            LengthsGather(), (input, lengths, indices, offsets), strict=False
        )

        FakeTensorMode.cache_clear()
        ep.run_decompositions({})
        self.assertBypasses("unrepresented symbol in output", 2)


class FakeTensorPreferDeviceType(TestCase):
    # [CLASS DOC] Tests for the fake_tensor_prefer_device_type functorch config.
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_fake_tensor_prefer_device_type(self):
        # [FUNCTION DOC] CUDA only: checks that by default a function mixing a CUDA tensor and a
        # CPU-created tensor raises a same-device RuntimeError under fake mode; with the config set
        # to cuda it succeeds and returns a fake CUDA tensor of the expected shape; same-device
        # CUDA and CPU ops are unaffected; and after leaving the config context the error returns.
        """
        Test that fake_tensor_prefer_device_type configuration works correctly
        for device mismatch scenarios.
        """

        # Create a custom operation that would normally cause device mismatch
        def mixed_device_op(a, b):
            # [FUNCTION DOC] HELPER: Adds the first argument to an unsqueezed CPU range with
            # length equal to the first argument's leading size; the second argument is unused.
            # This simulates an operation where 'a' is on MTIA/CUDA but 'b' is created on CPU
            cpu_tensor = torch.arange(a.shape[0], device="cpu")
            return a + cpu_tensor.unsqueeze(-1)

        with FakeTensorMode():
            # Test default behavior (should raise error on device mismatch)
            cuda_tensor = torch.randn(3, 4, device="cuda")

            # Without the config, this should raise a device mismatch error
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device",
            ):
                mixed_device_op(cuda_tensor, None)

        # Test with prefer_device_type set to "cuda"
        with torch._functorch.config.patch(fake_tensor_prefer_device_type="cuda"):
            with FakeTensorMode():
                cuda_tensor = torch.randn(3, 4, device="cuda")

                # This should now work and prefer the CUDA device
                result = mixed_device_op(cuda_tensor, None)

                # The result should be on CUDA device (preferred device type)
                self.assertEqual(result.device.type, "cuda")
                self.assertEqual(result.shape, (3, 4))
                self.assertTrue(is_fake_tensor(result))

        # Test that the configuration doesn't affect normal operations
        with torch._functorch.config.patch(fake_tensor_prefer_device_type="cuda"):
            with FakeTensorMode():
                # Normal same-device operations should work as before
                x = torch.randn(2, 3, device="cuda")
                y = torch.randn(2, 3, device="cuda")
                result = x + y
                self.assertEqual(result.device.type, "cuda")

                # CPU operations should still work
                x_cpu = torch.randn(2, 3, device="cpu")
                y_cpu = torch.randn(2, 3, device="cpu")
                result_cpu = x_cpu + y_cpu
                self.assertEqual(result_cpu.device.type, "cpu")

        # Test that the configuration is properly scoped
        with FakeTensorMode():
            cuda_tensor = torch.randn(3, 4, device="cuda")

            # After exiting the config context, should raise error again
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device",
            ):
                mixed_device_op(cuda_tensor, None)

    def test_fake_tensor_prefer_device_type_cpu_only(self):
        # [FUNCTION DOC] Checks that with the prefer-device-type config set
        # to cuda, adding two CPU fake tensors still gives a CPU fake tensor.
        """
        Test that fake_tensor_prefer_device_type works correctly when only CPU tensors are involved.
        """
        with torch._functorch.config.patch(fake_tensor_prefer_device_type="cuda"):
            with FakeTensorMode():
                # When all tensors are CPU, the result should still be CPU
                x = torch.randn(2, 3, device="cpu")
                y = torch.randn(2, 3, device="cpu")
                result = x + y
                self.assertEqual(result.device.type, "cpu")
                self.assertTrue(is_fake_tensor(result))


class FakeTensorMetaDevicePropagation(TestCase):
    # [CLASS DOC] Tests device propagation for ops with a meta-device operand.
    @parametrize("device", ["cpu", "cuda"])
    def test_inplace_add_with_meta_rhs_keeps_destination_device(self, device):
        # [FUNCTION DOC] Parametrized over cpu and cuda (skips cuda if unavailable): an in-place
        # add of a meta tensor into a fake tensor keeps the destination's device and stays fake.
        if device == "cuda" and not RUN_CUDA:
            self.skipTest("requires cuda")

        with FakeTensorMode():
            log_det = torch.zeros(2, device=device)
            log_det += torch.zeros(2, device="meta")

            self.assertEqual(log_det.device.type, device)
            self.assertTrue(is_fake_tensor(log_det))


instantiate_parametrized_tests(FakeTensorMetaDevicePropagation)


class FakeTensorViewCopy(TestCase):
    # [CLASS DOC] Tests that view_copy on expanded tensors behaves like eager mode.
    def test_expand_then_view_copy_matches_eager_mode(self):
        # [FUNCTION DOC] Checks view_copy of an expanded tensor to a flat shape
        # gives the same shape as eager and no base (non-aliasing) in both modes.
        x = torch.arange(7)
        y = x.expand(12, 7)

        # Eager baseline
        eager = torch.ops.aten.view_copy.default(y, [84])
        self.assertEqual(eager.shape, (84,))
        self.assertIsNone(eager._base)  # non-aliasing

        # FakeTensor behavior should match eager
        with FakeTensorMode():
            xf = torch.arange(7)
            yf = xf.expand(12, 7)
            fake = torch.ops.aten.view_copy.default(yf, [84])
            self.assertEqual(fake.shape, (84,))
            self.assertIsNone(fake._base)  # non-aliasing

    def test_expand_then_view_still_not_allowed(self):
        # [FUNCTION DOC] Checks that an ordinary view of an expanded
        # tensor to a flat shape under fake mode still raises ValueError.
        with FakeTensorMode():
            xf = torch.arange(7)
            yf = xf.expand(12, 7)
            with self.assertRaisesRegex(
                ValueError, "Cannot view a tensor with shape *"
            ):
                _ = yf.view(-1)

    def test_expand_then_view_copy_unbacked_matches_eager_mode(self):
        # [FUNCTION DOC] Same as the expand then view_copy test but
        # with backed-size-oblivious mode enabled via a config patch.
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            with FakeTensorMode():
                xf = torch.arange(7)
                yf = xf.expand(12, 7)
                fake = torch.ops.aten.view_copy.default(yf, [84])
                self.assertEqual(fake.shape, (84,))
                self.assertIsNone(fake._base)  # non-aliasing

    def test_expand_then_view_unbacked_still_not_allowed(self):
        # [FUNCTION DOC] Same as the expand then view test but with backed-size-oblivious
        # mode enabled via a config patch: the plain view must still raise ValueError.
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            with FakeTensorMode():
                xf = torch.arange(7)
                yf = xf.expand(12, 7)
                with self.assertRaisesRegex(
                    ValueError, "Cannot view a tensor with shape *"
                ):
                    _ = yf.view(-1)


if __name__ == "__main__":
    run_tests()
