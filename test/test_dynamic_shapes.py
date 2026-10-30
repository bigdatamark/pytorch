# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] Tests for symbolic-shape support: SymInt, SymFloat, SymBool and ShapeEnv. Defines a
# fake wrapper-subclass tensor with registered meta functions, helper constructors for symbolic
# ints/floats/bools backed by a ShapeEnv, and test classes covering arithmetic and magic methods on
# symbolic numbers, floor division, DimConstraints solving, guard expressions, unbacked-symbol
# handling under torch.compile, fast comparison evaluation, and transferring symbols between
# ShapeEnvs. Import-time side effects: creates the module-level aten alias and the meta_funcs
# registry, registers meta functions, and applies parametrization to selected test classes. The
# file runs its tests through run_tests when executed as a script.
import contextlib
import copy
import itertools
import math
import operator
import unittest

import numpy as np
import sympy

import torch
import torch.fx
import torch.nn.functional as F
from torch import sym_int, SymBool, SymFloat, SymInt
from torch._C import _disabled_torch_function_impl
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounter,
    CompileCounterWithBackend,
)
from torch._inductor.utils import fresh_cache
from torch.fx.experimental import sym_node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.sym_node import method_to_operator, SymNode, to_node
from torch.fx.experimental.symbolic_shapes import (
    _constrain_range_for_size,
    _iterate_exprs,
    DimConstraints,
    DimDynamic,
    expect_true,
    free_symbols,
    guard_bool,
    guard_float,
    guard_int,
    guarding_hint_or_throw,
    GuardOnDataDependentSymNode,
    has_free_symbols,
    is_symbolic,
    rebind_unbacked,
    ShapeEnv,
    StatelessSymbolicContext,
    statically_known_false,
    statically_known_true,
    SYMPY_INTERP,
)
from torch.testing._internal.common_dtype import all_types_and
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
    TEST_XPU,
    TestCase,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._sympy.functions import (
    CleanDiv,
    FloorDiv,
    IsNonOverlappingAndDenseIndicator,
    Max,
    Min,
    Mod,
)
from torch.utils._sympy.value_ranges import ValueRangeError


aten = torch.ops.aten

meta_funcs = {}


def register_meta(op):
    # [FUNCTION DOC] HELPER: Decorator factory that registers a meta implementation for one
    # operator or a pytree of operators. Returns a decorator which stores the decorated
    # function in the module-level meta_funcs mapping under each given operator and returns the
    # function unchanged.
    def decorator(f):
        # [FUNCTION DOC] HELPER: Decorator produced by the factory. Walks the operator argument
        # with pytree and records the function for every leaf operator, then returns the function
        # itself so the module-level name still refers to it.
        def add_func(op):
            # [FUNCTION DOC] HELPER: Pytree leaf callback; stores the enclosed function in
            # meta_funcs keyed by the given operator overload. Mutates global state and
            # returns nothing.
            meta_funcs[op] = f

        pytree.tree_map_(add_func, op)
        return f

    return decorator


@register_meta([aten.add.Tensor, aten.sub.Tensor])
def binary_meta(a, b):
    # [FUNCTION DOC] HELPER: Meta function for elementwise add and sub overloads. Ignores any
    # broadcasting and returns an empty fake tensor with the shape of the first operand.
    return a.new_empty(a.shape)


@register_meta(aten.cat.default)
def cat_meta(tensors, dim=0):
    # [FUNCTION DOC] HELPER: Meta function for concatenation. Sums the sizes along the given
    # dimension across all input tensors, checks every other dimension matches the first tensor,
    # and returns an empty fake tensor of the combined shape. Raises AssertionError on a size
    # mismatch in a non-concatenated dimension.
    concat_length = 0
    shape = tensors[0].shape
    for tensor in tensors:
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                if length != common_length:
                    raise AssertionError(
                        f"length mismatch: {length} != {common_length}"
                    )
    new_shape = list(shape)
    new_shape[dim] = concat_length
    return tensors[0].new_empty(new_shape)


@register_meta([aten.narrow_copy.default])
def narrow_copy_symint_meta(a, dim, start, length, **kwargs):
    # [FUNCTION DOC] HELPER: Meta function for narrow_copy with symbolic sizes. Builds the output
    # shape equal to the input shape except that the narrowed dimension gets the given length;
    # start and extra keyword arguments are ignored. Returns an empty fake tensor of that shape.
    shape = []
    for i, x in enumerate(a.shape):
        if i == dim:
            shape.append(length)
        else:
            shape.append(x)
    return a.new_empty(tuple(shape))


@register_meta([aten.expand.default])
def expand_symint_meta(a, size, implicit=False):
    # [FUNCTION DOC] HELPER: Meta function for expand. Returns an empty fake tensor with the
    # requested size, ignoring the implicit flag and the input's own shape.
    return a.new_empty(size)


def create_contiguous(shape):
    # [FUNCTION DOC] HELPER: Computes contiguous strides for a shape (sequence of ints or symbolic
    # ints). Accumulates running products from the last dimension backward and returns them as a
    # list ordered to match the dimensions.
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))


class FakeSymbolicTensor(torch.Tensor):
    # [CLASS DOC] Minimal tensor wrapper subclass used to give tests tensors whose sizes are
    # symbolic. It holds no storage (created via a wrapper-subclass constructor), disables torch
    # function interception, and routes all operators through __torch_dispatch__, which only
    # supports ops with registered meta functions plus new_empty.
    @staticmethod
    def __new__(
        cls,
        sym_shape,
        sym_strides,
        dtype,
        layout,
        requires_grad,
        device,
        storage_offset=0,
    ):
        # [FUNCTION DOC] Constructs the wrapper tensor from a (possibly symbolic) shape. The
        # sym_strides argument is ignored: strides are always recomputed as contiguous strides from
        # the shape (marked as a TODO in the code). Passes dtype, layout, requires_grad, device and
        # storage offset to the wrapper constructor and returns the new tensor.
        # TODO: this is wrong in general
        sym_stride = create_contiguous(sym_shape)
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            sym_shape,
            sym_stride,
            storage_offset,
            dtype=dtype,
            layout=layout,
            requires_grad=requires_grad,
            device=device,
        )
        return r

    __torch_function__ = _disabled_torch_function_impl

    def new_empty(self, shape):
        # [FUNCTION DOC] HELPER: Creates a new FakeSymbolicTensor of the given shape that copies
        # this tensor's dtype, layout, requires_grad and device. Strides are not passed.
        return FakeSymbolicTensor(
            shape, None, self.dtype, self.layout, self.requires_grad, self.device
        )

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        # [FUNCTION DOC] Dispatch hook. If the overload has a registered meta function, calls it
        # with the given args and kwargs (kwargs must be a mapping in that case). For the new_empty
        # overload, returns a FakeSymbolicTensor with the requested shape and the source tensor's
        # dtype, layout, requires_grad and device. Any other overload raises RuntimeError saying
        # the operator is not supported.
        if func_overload in meta_funcs:
            return meta_funcs[func_overload](*args, **kwargs)

        if func_overload == torch.ops.aten.new_empty.default:
            self = args[0]
            shape = args[1]
            return FakeSymbolicTensor(
                shape,
                self.stride(),
                self.dtype,
                self.layout,
                self.requires_grad,
                self.device,
            )

        raise RuntimeError(f"operator {func_overload} not supported")


def create_symbolic_tensor(name, arg, shape_env, source=None, dynamic_dims=None):
    # [FUNCTION DOC] HELPER: Builds a FakeSymbolicTensor whose sizes, strides and storage offset
    # are symbolic, created by the ShapeEnv from a real tensor. The source defaults to a
    # ConstantSource with the given name; dynamic_dims defaults to duck-sized for every dimension;
    # no per-dimension constraints are applied. Returns the fake tensor with the original's dtype,
    # layout, requires_grad and device.
    from torch._dynamo.source import ConstantSource

    if source is None:
        source = ConstantSource(name)
    constraint_dims = [None] * arg.dim()
    if dynamic_dims is None:
        dynamic_dims = [DimDynamic.DUCK] * arg.dim()
    (
        sym_shapes,
        sym_strides,
        sym_storage_offset,
    ) = shape_env.create_symbolic_sizes_strides_storage_offset(
        arg,
        source=source,
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=dynamic_dims, constraint_sizes=constraint_dims
        ),
    )
    return FakeSymbolicTensor(
        sym_shapes,
        sym_strides,
        arg.dtype,
        arg.layout,
        arg.requires_grad,
        arg.device,
        sym_storage_offset,
    )


def create_fake_tensor_with_dynamic_size(x, shape_env, dynamic_sizes, dynamic_strides):
    # [FUNCTION DOC] HELPER: Converts a real tensor into a fake tensor inside a FakeTensorMode that
    # shares the given ShapeEnv, using a stateless symbolic context with the supplied per-dimension
    # dynamic sizes and strides. Returns the resulting fake tensor.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(shape_env=shape_env) as fake_mode:
        return fake_mode.from_tensor(
            x,
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=dynamic_sizes,
                dynamic_strides=dynamic_strides,
            ),
        )


def create_symtype(cls, pytype, shape_env, val, duck=True, **kwargs):
    # [FUNCTION DOC] HELPER: Shared factory for symbolic scalars. Creates a ShapeEnv symbol for the
    # value (with a testing-only constant source whose name counts existing backed symbols,
    # duck-sized when duck is true, otherwise dynamic) and wraps it in a SymNode of the given
    # python type with the value as hint. Returns an instance of the given symbolic class. Extra
    # keyword arguments go to symbol creation. Side effect: adds a symbol to the shape env.
    from torch._dynamo.source import ConstantSource

    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(f"__testing_only{len(shape_env.backed_var_to_val)}"),
        dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
        constraint_dim=None,
        **kwargs,
    )
    return cls(SymNode(symbol, shape_env, pytype, hint=val))


# TODO: default duck to False
def create_symint(shape_env, i: int, duck=True, **kwargs) -> SymInt:
    # [FUNCTION DOC] HELPER: Creates a symbolic int with the given hint in the ShapeEnv; duck
    # sizing is on by default so equal values share one symbol.
    return create_symtype(SymInt, int, shape_env, i, duck=duck, **kwargs)


def create_symbool(shape_env, b: bool) -> SymBool:
    # [FUNCTION DOC] HELPER: Creates a symbolic bool with the given hint in the ShapeEnv, using
    # duck sizing.
    return create_symtype(SymBool, bool, shape_env, b)


def create_symfloat(shape_env, f: float) -> SymFloat:
    # [FUNCTION DOC] HELPER: Creates a symbolic float with the given hint in the ShapeEnv, using
    # duck sizing.
    return create_symtype(SymFloat, float, shape_env, f)


@skipIfTorchDynamo(
    "Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)"
)
class TestPySymInt(TestCase):
    # [CLASS DOC] Tests for symbolic integers, floats and bools created from a ShapeEnv:
    # arithmetic, bitwise ops, tracing, guards, expect_true, unbacked substitution and renaming,
    # simplification and meta-kernel behavior with unbacked sizes. Skipped under TorchDynamo
    # because ShapeEnv creation is not expected inside traced code.
    def test_arith_ops(self):
        # [FUNCTION DOC] Checks that add, sub, floordiv, mul and mod of symbolic ints from
        # different values match the same operations on plain ints for all ordered pairs.
        shape_env = ShapeEnv()
        symints = []
        for i in range(2, 5):
            symints.append((i, create_symint(shape_env, i)))

        ops = [
            operator.add,
            operator.sub,
            operator.floordiv,
            operator.mul,
            operator.mod,
        ]

        for op in ops:
            for args in itertools.permutations(symints, 2):
                if not isinstance(args[0][1], int) and (
                    (op != operator.mod or op != operator.floordiv) and args[1][0] != 0
                ):
                    self.assertTrue(
                        op(args[0][1], args[1][1]) == op(args[0][0], args[1][0])
                    )

    def test_reverse_arith_ops(self):
        # [FUNCTION DOC] Checks that python-int floordiv and multiplication with a symbolic int on
        # the right produce results equal to the plain int computation.
        shape_env = ShapeEnv()

        a = create_symint(shape_env, 2)
        self.assertTrue(5 // a == 5 // 2)

        a = create_symint(shape_env, 2)
        self.assertTrue(5 * a == 5 * 2)

    def test_sympify_symint(self):
        # [FUNCTION DOC] Verifies sympify of a SymInt, SymFloat and SymBool returns the identical
        # underlying sympy expression object of its node.
        shape_env = ShapeEnv()
        a = create_symint(shape_env, 2)
        self.assertIs(sympy.sympify(a), a.node.expr)
        b = create_symfloat(shape_env, 3.0)
        self.assertIs(sympy.sympify(b), b.node.expr)
        c = create_symbool(shape_env, True)
        self.assertIs(sympy.sympify(c), c.node.expr)

    def test_roundtrip(self):
        # [FUNCTION DOC] Checks that sizes, strides and storage offset of a symbolic tensor are
        # SymInt objects (not SymNode), compare equal to the original values, and can be simplified
        # to python ints after a guard, using both shape and size accessors.
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)

        self.assertTrue(not isinstance(x.shape[0], SymNode))
        self.assertTrue(isinstance(x.shape[0], SymInt))

        self.assertTrue(x.shape[0] == 5)
        self.assertTrue(x.shape[1] == 4)
        self.assertTrue(x.shape[2], 3)

        self.assertTrue(x.size()[0], 5)
        self.assertTrue(x.size()[1], 4)
        # Should be simplifiable to an integer.
        # Ref: https://github.com/pytorch/pytorch/pull/107492
        self.assertTrue(isinstance(x.size()[1], SymInt))
        self.assertTrue(
            isinstance(x.size()[1].node.maybe_as_int(), int)
        )  # due to guard above
        self.assertTrue(x.size()[2] == 3)

        self.assertTrue(x.size(0) == 5)
        self.assertTrue(x.size(1) == 4)
        self.assertTrue(x.size(2) == 3)
        self.assertTrue(isinstance(x.size(2), SymInt))
        self.assertTrue(isinstance(x.size(2).node.maybe_as_int(), int))

        y = create_symbolic_tensor("y", torch.randn(5, 4, 3)[1:], shape_env)
        self.assertTrue(isinstance(y.storage_offset(), SymInt))
        self.assertTrue(y.storage_offset() == 12)

    def test_binary(self):
        # [FUNCTION DOC] Checks that elementwise add of symbolic tensors, including a broadcast
        # operand, yields the expected output sizes.
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        y = create_symbolic_tensor("y", torch.randn(5, 4, 3), shape_env)

        z = x + y
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # broadcasting
        y = create_symbolic_tensor("y2", torch.randn(1, 4, 1), shape_env)
        z = x + y
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

    def test_symint_args(self):
        # [FUNCTION DOC] Checks narrow_copy accepts a SymInt, a difference of two SymInts, and a
        # SymInt minus a python int as the length argument, giving the expected resulting size.
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        y = create_symbolic_tensor("y", torch.randn(5, 4, 1), shape_env)
        LAST_DIM = 2
        z = x.narrow_copy(LAST_DIM, 0, y.shape[LAST_DIM])
        self.assertTrue(z.shape[2] == y.shape[2])

        # arithmetic expr with two symints
        z = x.narrow_copy(LAST_DIM, 0, x.shape[LAST_DIM] - y.shape[LAST_DIM])
        self.assertTrue(z.shape[2] == 2)

        # arithmetic expr with a symint and python int
        z = x.narrow_copy(LAST_DIM, 0, x.shape[LAST_DIM] - 1)
        self.assertTrue(z.shape[2] == 2)

    def test_symint_vargs(self):
        # [FUNCTION DOC] Checks expand with varargs and with shape sequences, with all-symbolic,
        # mixed symbolic/int and all-int arguments, and with a single dimension, all produce the
        # expected sizes.
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        y = create_symbolic_tensor("y", torch.randn(1, 4, 1), shape_env)

        # varargs
        z = y.expand(x.shape[0], y.shape[1], x.shape[2])
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # shape list
        z = y.expand((x.shape[0], y.shape[1], x.shape[2]))
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python symints and ints
        z = y.expand(x.shape[0], y.shape[1], 3)
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python symints and ints in a list
        z = y.expand((x.shape[0], y.shape[1], 3))
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python symints and ints
        z = y.expand(5, y.shape[1], x.shape[2])
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python ints and symints in a list
        z = y.expand((5, y.shape[1], x.shape[2]))
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        z = y.expand((y.shape[1],))
        z = y.expand(y.shape[1])

    def test_symint_bitwise_and(self):
        # [FUNCTION DOC] Checks bitwise and on SymInt pairs, SymInt with SymBool, SymInt with int
        # and SymBool pairs: results compare to the expected values, the main result stays a
        # SymInt, and the recorded guard prints as the expected equality on the bitwise function.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 0b1100)
        b0 = create_symint(shape_env, 0b1010)
        res_and = a0 & b0
        self.assertEqual(res_and, 0b1000)
        self.assertIsInstance(res_and, torch.SymInt, msg=type(res_and))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(BitwiseFn_bitwise_and(s97, s26), 8)"""
        )

        a1 = create_symint(shape_env, 3)
        b1 = create_symbool(shape_env, True)
        self.assertEqual(a1 & b1, 1)

        a2 = create_symint(shape_env, 0b1100)
        self.assertEqual(a2 & 0b1010, 0b1000)

        a3 = create_symbool(shape_env, True)
        b3 = create_symbool(shape_env, True)
        self.assertEqual(a3 & b3, True)

    def test_symint_bitwise_or(self):
        # [FUNCTION DOC] Checks bitwise or on two SymInts gives the right value, stays a SymInt,
        # and records the expected guard expression string.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 0b1100)
        b0 = create_symint(shape_env, 0b1010)
        res_or = a0 | b0
        self.assertEqual(res_or, 0b1110)
        self.assertIsInstance(res_or, torch.SymInt, msg=type(res_or))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(BitwiseFn_bitwise_or(s97, s26), 14)"""
        )

    def test_symint_bitwise_xor(self):
        # [FUNCTION DOC] Checks bitwise xor on two SymInts gives the right value, stays a SymInt,
        # and records the expected guard expression string.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 0b1100)
        b0 = create_symint(shape_env, 0b1010)
        res_xor = a0 ^ b0
        self.assertEqual(res_xor, 0b0110)
        self.assertIsInstance(res_xor, torch.SymInt, msg=type(res_xor))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(BitwiseFn_bitwise_xor(s97, s26), 6)"""
        )

    def test_stride(self):
        # [FUNCTION DOC] Checks that the stride of a symbolic tensor is a SymInt.
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 5), shape_env)
        self.assertIsInstance(x.stride()[0], SymInt)

    def test_size_expressions(self):
        # [FUNCTION DOC] Checks that branching on a comparison of an expanded symbolic size records
        # a strict-greater-than guard whose recorded size-oblivious flag is false.
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        expand_x = x.expand(x.shape[0], x.shape[0])
        if expand_x.shape[0] > 3:
            result = expand_x + expand_x
        else:
            result = expand_x + expand_x

        gt_op, _bt, is_size_obv = shape_env.guards[-1]
        self.assertTrue(isinstance(gt_op, sympy.core.relational.StrictGreaterThan))
        self.assertTrue(str(x.shape[0]), str(gt_op.args[0]))
        self.assertTrue(str(expand_x.shape[1]), str(x.shape[0]))
        self.assertTrue(str(expand_x.shape[1]), str(result.shape[0]))
        self.assertFalse(is_size_obv)

    def test_floordiv_static(self):
        # [FUNCTION DOC] Triggers several divisibility checks on a symbolic int and asserts that
        # static-known-true evaluation can prove floor division of the value by half of itself
        # equals two. The pattern was extracted from an inductor cpp wrapper test.
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 8)
        # This was extracted from
        # python test/inductor/test_cuda_cpp_wrapper.py -k
        # DynamicShapesCudaWrapperCudaTests.test_insignificant_strides_cuda_dynamic_shapes_cuda_wrapper
        bool(s0 % 2 == 0)
        bool(s0 % (s0 // 2) == 0)
        bool(2 * (s0 // 2) == s0)
        self.assertTrue(statically_known_true(s0 // (s0 // 2) == 2))

    def test_numel(self):
        # [FUNCTION DOC] Checks numel is a SymInt for symbolic tensors (method and function form)
        # and a plain int for ordinary tensors.
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        self.assertIsInstance(x.numel(), torch.SymInt)
        self.assertIsInstance(torch.numel(x), torch.SymInt)

        x = torch.rand(3, 3)
        self.assertIsInstance(x.numel(), int)
        self.assertIsInstance(torch.numel(x), int)

    def test_int_to_float(self):
        # [FUNCTION DOC] Checks that converting a symbolic size to float yields a SymFloat.
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        r = torch.sym_float(x.shape[0])
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))

    def test_aten_ops(self):
        # [FUNCTION DOC] Smoke test that narrow_copy and expand overloads run on symbolic tensors
        # with symbolic size arguments without raising.
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        torch.ops.aten.narrow_copy.default(x, 0, 0, x.shape[0])

        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x2", torch.randn(5, 4, 3), shape_env)
        torch.ops.aten.expand.default(x, [x.shape[0], x.shape[1], x.shape[2]])

    def test_fx_trace_intlist(self):
        # [FUNCTION DOC] Checks that fx symbolic tracing of a module that unpacks sizes and calls
        # pad with computed padding does not raise a type error about tuples.
        class CustomModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward unpacks four input sizes and pads the last
            # two dims by the parity of width and height.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Unpacks batch, channel, height and width from the input
                # and returns the input padded by width mod 2 on the right and height mod 2 on
                # the bottom.
                bs, c, h, w = x.shape
                return F.pad(x, (0, w % 2, 0, h % 2, 0, 0))

        m = CustomModule()
        x = torch.rand(1, 3, 4, 4)
        # should not TypeError: pad(): argument 'pad' (position 2) must be
        # tuple of ints, not tuple
        torch.fx.symbolic_trace(m)

    def test_meta_symint(self):
        # [FUNCTION DOC] Checks that creating an empty meta tensor from a SymInt gives a size that
        # is a SymInt.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        r = torch.empty(a0, device="meta")
        self.assertIsInstance(r.shape[0], SymInt)

    def test_hash_size(self):
        # [FUNCTION DOC] Regression test for issue 168254: hashing a tensor size containing a
        # symbolic value must raise TypeError.
        # See issue #168254
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        r = torch.empty(a0, device="meta")
        self.assertRaises(TypeError, lambda: hash(r.shape))

    def test_guard_int(self):
        # [FUNCTION DOC] Checks guard_int returns the hint value and records an equality guard on
        # the symbol.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        self.assertEqual(guard_int(a0), 2)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 2)""")

    def test_sym_sum(self):
        # [FUNCTION DOC] Checks that torch sym_sum over symbolic ints gives the same expression as
        # chained addition.
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 2)
        s1 = create_symint(shape_env, 3)
        s2 = create_symint(shape_env, 4)
        self.assertEqual(
            (s0 + s1 + s2).node.expr, torch.sym_sum([s0, s1, s2]).node.expr
        )

    def test_prefer_deferred_runtime_assertions_over_guards(self):
        # [FUNCTION DOC] With deferred runtime asserts preferred over guards, checks guard_int
        # still adds a guard, while expect_true adds no guard and records one deferred runtime
        # assert with the expected expression.
        shape_env = ShapeEnv(prefer_deferred_runtime_asserts_over_guards=True)
        s0 = create_symint(shape_env, 2)
        self.assertEqual(guard_int(s0), 2)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 2)""")

        shape_env = ShapeEnv(prefer_deferred_runtime_asserts_over_guards=True)
        s0 = create_symint(shape_env, 2)
        self.assertTrue(expect_true(s0 == 2))
        self.assertEqual(len(shape_env.guards), 0)
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[None]]),
            """[Eq(s97, 2)]""",
        )

    def test_sym_int(self):
        # [FUNCTION DOC] Checks sym_int on a SymInt, on a true division result and on a scaled
        # float conversion: results are SymInts with the expected values and the recorded guards
        # are truncation expressions.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = sym_int(a0)
        self.assertEqual(r, 5)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 5)""")

        a1 = create_symint(shape_env, 7)
        r = sym_int(a1 / 2)
        self.assertEqual(guard_int(r), 3)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]), """Eq(TruncToInt(IntTrueDiv(s26, 2)), 3)"""
        )

        a3 = create_symint(shape_env, 3)
        r = sym_int(2.0 * torch.sym_float(a3))
        self.assertEqual(guard_int(r), 6)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[2][0]), """Eq(TruncToInt(2.0*ToFloat(s57)), 6)"""
        )

    def test_sym_log2(self):
        # [FUNCTION DOC] Checks the symbolic log2 produces a SymFloat with the right value and
        # records an opaque unary function guard.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 4)
        r = torch._sym_log2(a0)
        self.assertEqual(r, 2.0)
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(OpaqueUnaryFn_log2(ToFloat(s97)), 2.0)"""
        )

    def test_sym_sqrt(self):
        # [FUNCTION DOC] Checks the symbolic sqrt produces a SymFloat with the right value and
        # records an opaque unary function guard.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 4)
        r = torch._sym_sqrt(a0)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(OpaqueUnaryFn_sqrt(ToFloat(s97)), 2.0)"""
        )

    def test_sym_floor(self):
        # [FUNCTION DOC] Checks math floor of symbolic true division and of float-scaled symbolic
        # ints returns SymInts with right values and floor-to-int guards.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.floor(a0 / 2)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(FloorToInt(IntTrueDiv(s97, 2)), 2)""",
        )
        r = math.floor(3.0 * a0)
        self.assertEqual(r, 15)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(FloorToInt(3.0*ToFloat(s97)), 15)""",
        )

    def test_sym_trunc(self):
        # [FUNCTION DOC] Checks math trunc of a symbolic true division and sym_int of a symbolic
        # sqrt return SymInts with the right values and truncation guards.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.trunc(a0 / 2)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(TruncToInt(IntTrueDiv(s97, 2)), 2)"""
        )
        r = torch.sym_int(torch.sym_sqrt(a0))
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(TruncToInt(OpaqueUnaryFn_sqrt(ToFloat(s97))), 2)""",
        )

    def test_sym_ceil(self):
        # [FUNCTION DOC] Checks math ceil of a symbolic true division gives a SymInt with a
        # ceil-to-int guard, then floors a float-scaled value and checks its guard.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.ceil(a0 / 2)
        self.assertEqual(r, 3)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(CeilToInt(IntTrueDiv(s97, 2)), 3)""",
        )
        r1 = 3.0 * a0
        r = math.floor(r1)
        self.assertEqual(r, 15)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(FloorToInt(3.0*ToFloat(s97)), 15)""",
        )

    def test_sym_ite(self):
        # [FUNCTION DOC] Checks sym_ite: python bool conditions return the chosen operand object
        # itself; a symbolic condition yields a result of the same type with the right value and no
        # extra guard from the condition, and the guard list holds a piecewise expression for it.
        shape_env = ShapeEnv()
        t = create_symint(shape_env, 5)
        f = create_symint(shape_env, 4)
        b1 = True
        r1 = torch.sym_ite(b1, t, f)
        self.assertTrue(r1 is t)
        b2 = False
        r2 = torch.sym_ite(b2, t, f)
        self.assertTrue(r2 is f)
        b3 = t == 5
        r3 = torch.sym_ite(b3, t, f)
        self.assertEqual(len(shape_env.guards), 0)
        self.assertEqual(r3, 5)
        self.assertEqual(type(t), type(r3))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(Piecewise((s97, Eq(s97, 5)), (s26, True)), 5)""",
        )
        b4 = f == 5
        r4 = torch.sym_ite(b4, t, f)
        self.assertEqual(len(shape_env.guards), 1)
        self.assertEqual(r4, 4)
        self.assertEqual(type(f), type(r4))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(Piecewise((s97, Eq(s26, 5)), (s26, True)), 4)""",
        )

    def test_tracing_sym_ite(self):
        # [FUNCTION DOC] Traces a function returning sym_ite of a size equality using symbolic
        # make_fx, asserts no guards were recorded, compares the generated graph code with an
        # expected string, then runs the graph on two inputs and checks both return int 5.
        def f(x):
            # [FUNCTION DOC] HELPER: Compares the first size to 5 and returns sym_ite choosing the
            # first size when equal, otherwise the second size.
            b = x.shape[0] == 5
            ret = torch.sym_ite(b, x.shape[0], x.shape[1])
            return ret

        gm = make_fx(f, tracing_mode="symbolic")(torch.ones(4, 5))
        self.assertEqual(len(gm.shape_env.guards), 0)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 5
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1);  x_1 = None
    sym_ite = torch.sym_ite(eq, sym_size_int, sym_size_int_1);  eq = sym_size_int = sym_size_int_1 = None
    return sym_ite""",
        )
        r1 = gm(torch.ones(4, 5))
        self.assertIsInstance(r1, int)
        self.assertEqual(r1, 5)
        r2 = gm(torch.ones(5, 4))
        self.assertIsInstance(r2, int)
        self.assertEqual(r2, 5)

    def test_int_conversion(self):
        # [FUNCTION DOC] Checks that converting a duck-sized symbolic int to a python int adds an
        # equality guard on its symbol.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        int(a0)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 2)""")

    def test_data_dependent_guard(self):
        # [FUNCTION DOC] Checks that branching on equality of an unbacked symbolic int
        # raises GuardOnDataDependentSymNode.
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        self.assertRaises(GuardOnDataDependentSymNode, lambda: bool(s0 == 0))

    def test_data_dependent_guard_propagate_real_tensors(self):
        # [FUNCTION DOC] Checks that with a real value (zero) registered for an unbacked symbol,
        # evaluating an equality to zero as a bool succeeds and returns True rather than raising.
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        shape_env.set_real_tensor_prop_unbacked_vals(s0.node.expr, 0)
        self.assertEqual(bool(s0 == 0), True)

    def test_expect_true_basic(self):
        # [FUNCTION DOC] Checks expect_true on an unbacked equality returns true without error,
        # records a replacement of the symbol by zero in the ShapeEnv, and that the same equality
        # can then be evaluated as a bool.
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i0_sym = i0.node.expr
        # This doesn't error
        self.assertTrue(expect_true(i0 == 0))
        # This generates a deferred runtime assert via replacement
        self.assertEqual(shape_env.replacements[i0_sym], 0)
        # After expecting true, guards now resolve given the runtime assert
        bool(i0 == 0)

    def test_expect_true_with_s0(self):
        # [FUNCTION DOC] Checks expect_true of an unbacked symbol being less than a backed one
        # stores one deferred runtime assert under the unbacked symbol, after which that comparison
        # and its implied != hold while the opposite comparisons evaluate false.
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 5)
        i0 = shape_env.create_unbacked_symint()
        self.assertTrue(expect_true(i0 < s0))
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[i0.node.expr]]),
            """[u0 < s97]""",
        )
        self.assertTrue(i0 < s0)
        self.assertTrue(i0 != s0)
        self.assertFalse(i0 > s0)
        self.assertFalse(i0 >= s0)

    def test_expect_true_prefer_later(self):
        # [FUNCTION DOC] Checks expect_true of a sum of two unbacked symbols equal to a constant is
        # filed as a deferred runtime assert under the later symbol, and that the same equality
        # then evaluates true. The trailing note says the sum is not substituted by the constant.
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i1 = shape_env.create_unbacked_symint()
        i1_sym = i1.node.expr
        self.assertTrue(expect_true(i0 + i1 == 10))
        # Importantly, this is put in i1, not i0!
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[i1_sym]]),
            """[Eq(u0 + u1, 10)]""",
        )
        self.assertTrue(i0 + i1 == 10)
        # NB: We currently don't support deriving that we can substitute
        # i0 + i1 with 10; maybe we should, but this means our rewriting
        # system is no longer confluent (it's probably OK though, because
        # you're unlikely to get other equalities like this on the
        # unbacked SymInts.)

    def test_unbacked_substitution(self):
        # [FUNCTION DOC] Checks that expecting equality between a size-constrained unbacked int and
        # four times another (in both operand orders) leaves the larger multiple printing under its
        # own name (u0 and u3), i.e. it is not rewritten in terms of the other symbol.
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i1 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i0)
        _constrain_range_for_size(i1)
        self.assertTrue(expect_true(i0 == i1 * 4))
        self.assertExpectedInline(str(i0), """u0""")

        i2 = shape_env.create_unbacked_symint()
        i3 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i2)
        _constrain_range_for_size(i3)
        self.assertTrue(expect_true(i2 * 4 == i3))
        self.assertExpectedInline(str(i3), """u3""")

    def test_rename_unbacked_to_unifies_unbacked_dest(self):
        # [FUNCTION DOC] Checks the internal rename-unbacked-to routine when the original symbol
        # already maps to an unbacked destination: afterwards the original maps to the new symbol
        # and the new symbol maps to that destination.
        shape_env = ShapeEnv()
        orig = shape_env.create_unbacked_symint().node.expr
        new = shape_env.create_unbacked_symint().node.expr
        dest = shape_env.create_unbacked_symint().node.expr
        shape_env._set_replacement(orig, dest, "test-setup")
        shape_env._rename_unbacked_to(orig, new)
        self.assertEqual(shape_env.replacements[orig], new)
        self.assertEqual(shape_env.replacements[new], dest)

    def test_rename_unbacked_to_raises_on_disjoint_ranges(self):
        # [FUNCTION DOC] Checks the rename routine raises ValueRangeError when the original's
        # replacement destination has a range (0 to 5) disjoint from the new symbol's (10 to 20).
        shape_env = ShapeEnv()
        orig = shape_env.create_unbacked_symint()
        new = shape_env.create_unbacked_symint()
        dest = shape_env.create_unbacked_symint()
        _constrain_range_for_size(dest, min=0, max=5)
        _constrain_range_for_size(new, min=10, max=20)
        shape_env._set_replacement(orig.node.expr, dest.node.expr, "test-setup")
        with self.assertRaises(ValueRangeError):
            shape_env._rename_unbacked_to(orig.node.expr, new.node.expr)

    def test_rename_unbacked_to_preserves_existing_backed_replacement(self):
        # [FUNCTION DOC] Checks the rename routine keeps an existing backed replacement of the new
        # symbol: the original maps to the new symbol, and both the new symbol and the old
        # destination end up mapped to the backed symbol.
        shape_env = ShapeEnv()
        orig = shape_env.create_unbacked_symint().node.expr
        new = shape_env.create_unbacked_symint().node.expr
        dest = shape_env.create_unbacked_symint().node.expr
        backed = create_symint(shape_env, 4).node.expr
        shape_env._set_replacement(new, backed, "test-setup")
        shape_env._set_replacement(orig, dest, "test-setup")
        shape_env._rename_unbacked_to(orig, new)
        self.assertEqual(shape_env.replacements[orig], new)
        self.assertEqual(shape_env.replacements[new], backed)
        self.assertEqual(shape_env.replacements[dest], backed)

    def test_rename_unbacked_to_unifies_existing_unbacked(self):
        # [FUNCTION DOC] Checks the rename routine when the new symbol already aliases another
        # unbacked symbol: the original maps to the new one and the new one and the other alias are
        # all unified onto the original's destination.
        shape_env = ShapeEnv()
        orig = shape_env.create_unbacked_symint().node.expr
        new = shape_env.create_unbacked_symint().node.expr
        dest = shape_env.create_unbacked_symint().node.expr
        other = shape_env.create_unbacked_symint().node.expr
        shape_env._set_replacement(new, other, "test-setup")
        shape_env._set_replacement(orig, dest, "test-setup")
        shape_env._rename_unbacked_to(orig, new)
        self.assertEqual(shape_env.replacements[orig], new)
        self.assertEqual(shape_env.replacements[new], dest)
        self.assertEqual(shape_env.replacements[other], dest)

    def test_rename_unbacked_to_preserves_transitive_backed_replacement(self):
        # [FUNCTION DOC] Checks the rename routine with a chain new -> other -> backed: the find of
        # the new symbol is the backed symbol, and the other symbol and old destination also
        # resolve to the backed one.
        shape_env = ShapeEnv()
        orig = shape_env.create_unbacked_symint().node.expr
        new = shape_env.create_unbacked_symint().node.expr
        other = shape_env.create_unbacked_symint().node.expr
        dest = shape_env.create_unbacked_symint().node.expr
        backed = create_symint(shape_env, 4).node.expr
        shape_env._set_replacement(other, backed, "test-setup")
        shape_env._set_replacement(new, other, "test-setup")
        shape_env._set_replacement(orig, dest, "test-setup")
        shape_env._rename_unbacked_to(orig, new)
        self.assertEqual(shape_env._find(new), backed)
        self.assertEqual(shape_env.replacements[other], backed)
        self.assertEqual(shape_env.replacements[dest], backed)

    def test_rename_unbacked_to_unifies_unbacked_alias_onto_backed_dest(self):
        # [FUNCTION DOC] Checks the rename routine when the original maps to a backed destination
        # and the new symbol aliases another unbacked symbol: original, new and alias all resolve
        # to the backed destination.
        shape_env = ShapeEnv()
        orig = shape_env.create_unbacked_symint().node.expr
        new = shape_env.create_unbacked_symint().node.expr
        alias = shape_env.create_unbacked_symint().node.expr
        backed_dest = create_symint(shape_env, 4).node.expr
        shape_env._set_replacement(new, alias, "test-setup")
        shape_env._set_replacement(orig, backed_dest, "test-setup")
        shape_env._rename_unbacked_to(orig, new)
        self.assertEqual(shape_env._find(orig), backed_dest)
        self.assertEqual(shape_env._find(new), backed_dest)
        self.assertEqual(shape_env._find(alias), backed_dest)

    def test_rename_unbacked_to_skips_range_widening_alias(self):
        # [FUNCTION DOC] Checks that the rename routine does not alias onto a replacement that
        # would widen ranges: with a wide new symbol aliasing a narrow one, new and original
        # resolve to the destination while the narrow alias gets no replacement entry.
        shape_env = ShapeEnv()
        orig = shape_env.create_unbacked_symint()
        new = shape_env.create_unbacked_symint()
        alias = shape_env.create_unbacked_symint()
        dest = shape_env.create_unbacked_symint()
        _constrain_range_for_size(new, min=0, max=100)
        _constrain_range_for_size(alias, min=0, max=5)
        _constrain_range_for_size(dest, min=0, max=20)
        shape_env._set_replacement(new.node.expr, alias.node.expr, "test-setup")
        shape_env._set_replacement(orig.node.expr, dest.node.expr, "test-setup")
        shape_env._rename_unbacked_to(orig.node.expr, new.node.expr)
        self.assertEqual(shape_env._find(new.node.expr), dest.node.expr)
        self.assertEqual(shape_env._find(orig.node.expr), dest.node.expr)
        self.assertNotIn(alias.node.expr, shape_env.replacements)

    def test_rename_unbacked_to_unifies_two_backed_terminals(self):
        # [FUNCTION DOC] Checks the rename routine when both the original and new symbols point at
        # different backed (non-duck) symbols: the new symbol maps to the original's backed
        # destination, and both resolve to the same representative.
        shape_env = ShapeEnv()
        orig = shape_env.create_unbacked_symint().node.expr
        new = shape_env.create_unbacked_symint().node.expr
        backed_dest = create_symint(shape_env, 4, duck=False).node.expr
        backed_new = create_symint(shape_env, 4, duck=False).node.expr
        shape_env._set_replacement(new, backed_new, "test-setup")
        shape_env._set_replacement(orig, backed_dest, "test-setup")
        shape_env._rename_unbacked_to(orig, new)
        self.assertEqual(shape_env.replacements[new], backed_dest)
        self.assertEqual(shape_env._find(orig), shape_env._find(new))

    def test_rename_unbacked_to_unifies_on_double_propagation(self):
        # [FUNCTION DOC] Regression test: exports a module that calls unique and uses the unbacked
        # count, then runs the unbacked-symint propagation interpreter twice over the exported
        # graph under its fake mode, which previously asserted. Checks the exported module still
        # equals the eager module on the original input.
        from torch._guards import detect_fake_mode
        from torch.fx.experimental.symbolic_shapes import PropagateUnbackedSymInts

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the sum of unique values of the
            # input plus the sum of a zeros tensor sized by the number of unique values (an
            # unbacked size).
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes unique values of x, then returns their sum added
                # to the sum of a new zero tensor with one element per unique value.
                u = torch.unique(x)  # aten._unique2 -> unbacked count
                return u.sum() + x.new_zeros(u.shape[0]).sum()

        m = M()
        inp = (torch.tensor([1, 1, 2, 3, 3]),)
        ep = torch.export.export(m, inp)
        gm = ep.module()
        ph_vals = [
            n.meta["val"]
            for n in gm.graph.nodes
            if n.op == "placeholder" and "val" in n.meta
        ]
        fm = detect_fake_mode(ph_vals)
        self.assertIsNotNone(fm)
        with fm:
            PropagateUnbackedSymInts(gm).run(*ph_vals)
            PropagateUnbackedSymInts(gm).run(*ph_vals)
        # After the (previously-asserting) re-propagation, the exported program
        # still runs and matches eager.
        self.assertEqual(gm(*inp), m(*inp))

    def test_avoid_unbacked_substitution(self):
        # [FUNCTION DOC] Checks that expecting an unbacked symbol equal to a constant minus another
        # unbacked symbol does not substitute away the first symbol, whose printed form stays the
        # original name.
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i0)
        i1 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i1)
        self.assertTrue(expect_true(i0 == 10 - i1))
        self.assertExpectedInline(str(i0), """u0""")

    def test_expect_true_double_digits(self):
        # [FUNCTION DOC] Checks symbol naming with eleven unbacked ints (last is u10) and that
        # expect_true on their sum equal to twenty files exactly one deferred runtime assert under
        # the last symbol.
        shape_env = ShapeEnv()
        ia = [shape_env.create_unbacked_symint() for _ in range(11)]  # allocate 10
        self.assertEqual(str(ia[-1]), "u10")
        self.assertTrue(expect_true(sum(ia) == 20))
        self.assertEqual(len(shape_env.deferred_runtime_asserts[ia[-1].node.expr]), 1)

    def test_expect_true_refine_range(self):
        # [FUNCTION DOC] Checks that expect_true on lower bound relations (greater than 4 and
        # equivalent forms) and upper bound relations (less than 4 and equivalent forms) on fresh
        # unbacked symbols refines their value ranges, tested via statically-known-true on
        # inequalities, with a subtest for each form.
        shape_env = ShapeEnv()
        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = shape_env.create_unbacked_symint()
                self.assertTrue(expect_true(rel(i0)))
                self.assertTrue(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertFalse(statically_known_true(i0 != 5))
                self.assertFalse(statically_known_true(i0 != 6))
                self.assertTrue(statically_known_true(i0 > 4))
                self.assertTrue(statically_known_true(i0 >= 5))

        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = shape_env.create_unbacked_symint()
                self.assertTrue(expect_true(rel(i0)))
                self.assertFalse(statically_known_true(i0 != 2))
                self.assertFalse(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertTrue(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 < 4))
                self.assertTrue(statically_known_true(i0 <= 5))

    def test_guard_refine_range(self):
        # [FUNCTION DOC] Checks that evaluating bounds relations as booleans on non-duck backed
        # symints (hints 10 and 2) refines ranges so later inequality queries are statically known
        # true or false accordingly, for four relation forms in both outcomes, using subtests.
        shape_env = ShapeEnv()
        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 10, duck=False)
                self.assertTrue(bool(rel(i0)))
                self.assertTrue(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertFalse(statically_known_true(i0 != 5))
                self.assertFalse(statically_known_true(i0 != 6))
                self.assertTrue(statically_known_true(i0 > 4))
                self.assertTrue(statically_known_true(i0 >= 5))

        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 2, duck=False)
                self.assertFalse(bool(rel(i0)))
                self.assertFalse(statically_known_true(i0 != 3))
                self.assertFalse(statically_known_true(i0 != 4))
                self.assertTrue(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 != 6))
                self.assertTrue(statically_known_true(i0 <= 4))
                self.assertTrue(statically_known_true(i0 < 5))

        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 2, duck=False)
                self.assertTrue(bool(rel(i0)))
                self.assertFalse(statically_known_true(i0 != 2))
                self.assertFalse(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertTrue(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 < 4))
                self.assertTrue(statically_known_true(i0 <= 3))

        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 10, duck=False)
                self.assertFalse(bool(rel(i0)))
                self.assertTrue(statically_known_true(i0 != 2))
                self.assertTrue(statically_known_true(i0 != 3))
                self.assertFalse(statically_known_true(i0 != 4))
                self.assertFalse(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 >= 4))
                self.assertTrue(statically_known_true(i0 > 3))

    def test_mul_int_oo_nan(self):
        # [FUNCTION DOC] Regression smoke test: evaluating a product of a symbol and a floor
        # division by itself compared to another symbol must not raise (integer infinity times nan
        # case in range analysis).
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 5, duck=False)
        s1 = create_symint(shape_env, 6, duck=False)
        s2 = create_symint(shape_env, 5, duck=False)
        bool(s0 * (s1 // s0) == s2)

    def test_non_overlapping_and_dense_backed(self):
        # [FUNCTION DOC] Checks the non-overlapping-and-dense operator returns true for a meta
        # tensor with backed symbolic size and transposed strides.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = torch.empty_strided((a0, 7), (1, a0), device="meta")
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(r))

    def test_non_overlapping_and_dense_unbacked(self):
        # [FUNCTION DOC] Checks the non-overlapping-and-dense indicator function and operator on
        # shapes with an unbacked size: simple 2D, 1D and singleton cases give true, as does a 4D
        # case with strides built from the maximum of one and the unbacked size.
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        cf = torch.ops.aten.is_non_overlapping_and_dense.default

        self.assertEqual(IsNonOverlappingAndDenseIndicator(u0.node.expr, 2, 2, 1), 1)
        self.assertEqual(IsNonOverlappingAndDenseIndicator(2, u0.node.expr, 1, 2), 1)
        self.assertTrue(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")))
        self.assertTrue(cf(torch.empty_strided((2, u0), (1, 2), device="meta")))

        self.assertEqual(IsNonOverlappingAndDenseIndicator(u0.node.expr, 1), 1)
        self.assertEqual(IsNonOverlappingAndDenseIndicator(1, u0.node.expr), 1)
        self.assertTrue(cf(torch.empty_strided((u0,), (1,), device="meta")))
        self.assertTrue(cf(torch.empty_strided((1,), (u0,), device="meta")))

        Max = torch.sym_max
        # NB: This only works because we're able to determine this tensor is
        # contiguous. transpose(0, 1) makes it stop working
        self.assertTrue(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            )
        )

    def test_sympy_interp_is_non_overlapping_and_dense_flat_args(self):
        # [FUNCTION DOC] Checks the SYMPY_INTERP indicator function accepts flat arguments (sizes
        # followed by strides) as emitted in guard code: 1D, 2D and 4D dense layouts give 1 and
        # gapped layouts give 0.
        # SYMPY_INTERP is used as the eval() namespace for guard code strings.
        # Guard code prints IsNonOverlappingAndDenseIndicator(s0, s1, ..., st0, st1, ...)
        # with flat args, so the SYMPY_INTERP function must accept flat args.
        interp_fn = SYMPY_INTERP["IsNonOverlappingAndDenseIndicator"]

        # 1D contiguous: sizes=(5,), strides=(1,)
        self.assertEqual(interp_fn(5, 1), 1)
        # 1D non-contiguous: sizes=(5,), strides=(2,)
        self.assertEqual(interp_fn(5, 2), 0)
        # 1D single element: sizes=(1,), strides=(42,)
        self.assertEqual(interp_fn(1, 42), 1)

        # 2D contiguous: sizes=(3, 4), strides=(4, 1)
        self.assertEqual(interp_fn(3, 4, 4, 1), 1)
        # 2D non-contiguous: sizes=(3, 4), strides=(5, 1) -- gap in memory
        self.assertEqual(interp_fn(3, 4, 5, 1), 0)
        # 2D transposed but still dense: sizes=(4, 3), strides=(1, 4)
        self.assertEqual(interp_fn(4, 3, 1, 4), 1)

        # 4D contiguous (the exact scenario from the MAST job failure):
        # sizes=(2, 3, 4, 5), strides=(60, 20, 5, 1)
        self.assertEqual(interp_fn(2, 3, 4, 5, 60, 20, 5, 1), 1)
        # 4D non-contiguous:
        self.assertEqual(interp_fn(2, 3, 4, 5, 100, 20, 5, 1), 0)

    def test_sympy_interp_guard_eval_simulation(self):
        # [FUNCTION DOC] Simulates guard evaluation by evaluating guard code strings with the
        # SYMPY_INTERP namespace: a dense 4D layout compares true and a gapped one compares false.
        # Simulate the actual guard eval() path: guard code strings use
        # IsNonOverlappingAndDenseIndicator as a function name, and SYMPY_INTERP
        # provides the binding in the eval namespace.
        guard_code = "IsNonOverlappingAndDenseIndicator(2, 3, 4, 5, 60, 20, 5, 1) == 1"
        result = eval(guard_code, SYMPY_INTERP)  # noqa: P204
        self.assertTrue(result)

        guard_code_false = (
            "IsNonOverlappingAndDenseIndicator(2, 3, 4, 5, 100, 20, 5, 1) == 1"
        )
        result_false = eval(guard_code_false, SYMPY_INTERP)  # noqa: P204
        self.assertFalse(result_false)

    def test_prims_is_non_overlapping_and_dense_or_false(self):
        # [FUNCTION DOC] Checks the or-false variant of the dense check on meta tensors: true for
        # backed and unbacked dense layouts, false for a permuted unbacked layout and for layouts
        # with arbitrary unbacked strides, since it returns false when it cannot prove density.
        shape_env = ShapeEnv()
        cf = torch._prims_common.is_non_overlapping_and_dense_or_false

        # backed case
        a0 = create_symint(shape_env, 5)
        self.assertTrue(cf(torch.empty_strided((a0, 7), (1, a0), device="meta")))

        # unbacked
        u0 = shape_env.create_unbacked_symint()
        self.assertTrue(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")))
        self.assertTrue(cf(torch.empty_strided((2, u0), (1, 2), device="meta")))
        self.assertTrue(cf(torch.empty_strided((u0,), (1,), device="meta")))
        self.assertTrue(cf(torch.empty_strided((1,), (u0,), device="meta")))

        Max = torch.sym_max
        self.assertTrue(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            )
        )
        self.assertFalse(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (Max(1, u0), Max(1, u0), 1, 3 * Max(1, u0)),
                    device="meta",
                )
            )
        )

        # return False on arbitrary strides
        u1 = shape_env.create_unbacked_symint()
        self.assertFalse(
            cf(
                torch.empty_strided(
                    (2 * u0, u0, 1),
                    (u1, u0, u0 + u1),
                    device="meta",
                )
            )
        )
        self.assertFalse(
            cf(
                torch.empty_strided(
                    (2, 3, u0),
                    (u1, 3, 1),
                    device="meta",
                )
            )
        )

    def test_sympy_optimized_add_binary_search(self):
        # [FUNCTION DOC] Checks the sorted-insert helper for symbolic sum arguments: inserting into
        # an empty list yields a one-item list, inserting an element already present returns None,
        # and new elements are placed in sorted name order, including at the front.
        import sympy

        from torch.fx.experimental.sym_node import _binary_search_insert_arg

        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        c = sympy.Symbol("c")

        args = []
        args = _binary_search_insert_arg([], b)
        self.assertEqual(args, [b])

        self.assertEqual(_binary_search_insert_arg(args, b), None)

        args = _binary_search_insert_arg(args, a)
        self.assertEqual(args, [a, b])

        self.assertEqual(_binary_search_insert_arg(args, b), None)
        self.assertEqual(_binary_search_insert_arg(args, a), None)

        args = _binary_search_insert_arg(args, c)
        self.assertEqual(args, [a, b, c])

        self.assertEqual(_binary_search_insert_arg(args, a), None)
        self.assertEqual(_binary_search_insert_arg(args, b), None)
        self.assertEqual(_binary_search_insert_arg(args, c), None)

        a1 = sympy.Symbol("a1")
        a2 = sympy.Symbol("a2")

        args = _binary_search_insert_arg(args, a1)
        self.assertEqual(args, [a, a1, b, c])

        args = _binary_search_insert_arg(args, a2)
        self.assertEqual(args, [a, a1, a2, b, c])

        c1 = sympy.Symbol("c1")
        args = _binary_search_insert_arg(args, c1)
        self.assertEqual(args, [a, a1, a2, b, c, c1])

        # insert to front
        _a = sympy.Symbol("_a")
        args = _binary_search_insert_arg(args, _a)
        self.assertEqual(args, [_a, a, a1, a2, b, c, c1])

    def test_floor_clean_div_axioms(self):
        # [FUNCTION DOC] Checks that an axiom recorded for a floor division by three is still found
        # after the ShapeEnv learns divisibility: the floor form evaluates true, the clean division
        # form is not statically known at first, then after asserting the modulus is zero the floor
        # form simplifies to the clean-division form and both evaluate true.
        # Test that if we add an axiom that have FloorDiv, after which the
        # shapeEnv changed such that it can be simplified it to CleanDiv, then
        # We still correctly replace CleanDiv with the axiom value of FloorDiv.
        shape_env = ShapeEnv()
        a = shape_env.create_unbacked_symint()

        shape_env.guard_or_defer_runtime_assert((a // 3 == 1).node.expr, " test")

        from sympy import Eq

        test1 = Eq(FloorDiv(a.node.expr, 3), 1)
        test2 = Eq(CleanDiv(a.node.expr, 3), 1)

        self.assertTrue(shape_env.evaluate_expr(test1))
        self.assertEqual(shape_env._maybe_evaluate_static(test2), None)

        # After this FloorDiv(a, 3) is simplified to CleanDiv(a, 3)
        shape_env.guard_or_defer_runtime_assert(Eq(Mod(a, 3), 0), " test")
        self.assertEqual(test2, shape_env.simplify(test1))

        self.assertTrue(shape_env.evaluate_expr(test1))
        self.assertTrue(shape_env.evaluate_expr(test2))

    def test_sympy_optimized_add(self):
        # [FUNCTION DOC] Checks the optimized-summation flag on sums of symbolic ints: a sum of
        # distinct symbols is flagged and stays flagged when a new unique symbol is added; adding a
        # duplicate symbol, a constant, or combining two long optimized sums clears it.
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 2)
        s1 = create_symint(shape_env, 3)
        s2 = create_symint(shape_env, 4)
        sum = s0 + s1

        self.assertTrue(sum.node._optimized_summation)

        def assert_optimized(sym):
            # [FUNCTION DOC] HELPER: Asserts the symbolic value's node has the optimized-summation
            # flag set.
            self.assertTrue(sym.node._optimized_summation)

        def assert_not_optimized(sym):
            # [FUNCTION DOC] HELPER: Asserts the node's optimized-summation flag is absent
            # or false.
            self.assertFalse(getattr(sym.node, "_optimized_summation", False))

        assert_optimized(sum)

        # add duplicate symbol
        assert_not_optimized(sum + s0)

        # add constant.
        assert_not_optimized(sum + 1)

        # add new unique symbol, should maintain _optimized_summation property.
        assert_optimized(sum + s2)

        assert_optimized(s2 + sum)

        # add x + (a+b) with no  _optimized_summation on the rhs sum.
        a = create_symint(shape_env, 10)
        b = create_symint(shape_env, 11)
        two_sum = torch.sym_sum([a, b])
        assert_not_optimized(two_sum)
        assert_optimized(sum + two_sum)

        # adding two expressions of length >2 that are _optimized_summation.
        a = s0 + s1 + s2
        s3 = create_symint(shape_env, 10)
        s4 = create_symint(shape_env, 20)
        s5 = create_symint(shape_env, 30)
        b = s3 + s4 + s5
        assert_optimized(a)
        assert_optimized(b)
        assert_not_optimized(a + b)
        assert_not_optimized(b + a)
        assert_not_optimized(b + a + b)

    def test_max_of_unique_summation_opt(self):
        # [FUNCTION DOC] Checks symmetric max over sums of unbacked symbols: max of two sums of
        # distinct symbols and max with further disjoint sums carry unique-summation symbol info,
        # while other forms are computed (see known issue below).
        # KNOWN ISSUE: the not-optimized helper only reads the attribute and never asserts, so the
        # not-optimized calls in this test verify nothing.
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        s1 = shape_env.create_unbacked_symint()
        s2 = shape_env.create_unbacked_symint()
        s3 = shape_env.create_unbacked_symint()
        s4 = shape_env.create_unbacked_symint()
        s5 = shape_env.create_unbacked_symint()
        s7 = shape_env.create_unbacked_symint()

        def assert_optimized(sym):
            # [FUNCTION DOC] HELPER: Asserts the expression has unique-summation symbols recorded.
            self.assertTrue(sym.node.expr.unique_summations_symbols is not None)

        def assert_not_optimized(sym):
            # [FUNCTION DOC] HELPER: Intended to check absence of unique-summation symbols but only
            # reads the attribute and asserts nothing.
            # KNOWN ISSUE: contains no assertion, so it can never fail.
            getattr(sym.node.expr, "unique_summations_symbols", None)

        mx1 = torch.sym_max(s0, s1)
        assert_not_optimized(mx1)

        mx2 = torch.sym_max(s0 + s1, s2 + s3)
        assert_optimized(mx2)

        mx3 = torch.sym_max(mx2, s4 + s5)
        assert_optimized(mx3)
        assert_optimized(torch.sym_max(s4 + s5, mx2))

        assert_not_optimized(torch.sym_max(mx3, s7))
        assert_not_optimized(torch.sym_max(mx3, 10))
        assert_not_optimized(torch.sym_max(mx3, s3 + s7))
        assert_not_optimized(torch.sym_max(mx3, s7 * 2))

    def test_build_proxy_for_optimized_add(self):
        # [FUNCTION DOC] Checks that rebuilding a proxy for a flattened three-argument sum
        # expression works. With no output given and with an output given, the rebuilt graph must
        # sum all three inputs; an fx graph module built from placeholder proxies is executed on 2,
        # 3, 4 and must return 9.
        """
        Test that _build_proxy_for_sym_expr correctly handles flattened
        sympy.Add expressions with more than 2 arguments.

        When sympy.Add operations are optimized by _optimized_add
        (in torch/fx/experimental/sym_node.py), they can be flattened into a
        single Add node with 3 or more arguments (e.g., Add(s0, s1, s2)
        instead of Add(Add(s0, s1), s2)).

        _build_proxy_for_sym_expr must handle this by using torch.sym_sum
        (which natively supports n-ary addition) instead of operator.add
        (which is binary and would either raise TypeError or silently drop
        arguments via FX codegen's 2-placeholder format string).
        """
        import torch.fx as fx
        from torch.fx.experimental.proxy_tensor import (
            _build_proxy_for_sym_expr,
            _SympyExprTrackerValue,
            PythonKeyTracer,
            set_meta,
        )
        from torch.utils._thunk import Thunk

        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 2)
        s1 = create_symint(shape_env, 3)
        s2 = create_symint(shape_env, 4)

        # This triggers _optimized_add and produces a flattened sympy.Add
        # with 3 arguments: Add(s0_symbol, s1_symbol, s2_symbol)
        optimized_sum = s0 + s1 + s2
        self.assertTrue(optimized_sum.node._optimized_summation)
        # Confirm the underlying sympy expression has 3 args (flattened).
        self.assertEqual(len(optimized_sum.node.expr.args), 3)

        # ------------------------------------------------------------------
        # Case 1: out=None
        #
        # Previously this would call operator.add(a, b, c) and raise
        # TypeError. Now uses torch.sym_sum which handles n-ary addition.
        # ------------------------------------------------------------------
        tracer_none = PythonKeyTracer()
        for sym in [s0, s1, s2]:
            tracer_none.sympy_expr_tracker[sym.node.expr] = _SympyExprTrackerValue(
                proxy=sym, value=sym
            )

        result = _build_proxy_for_sym_expr(tracer_none, optimized_sum.node.expr)
        self.assertEqual(int(result), 9)

        # ------------------------------------------------------------------
        # Case 2: out=optimized_sum  (the typical path from get_proxy_slot)
        #
        # Previously _sym_register would record operator.add with 3 args,
        # and FX codegen's "{} + {}" format string would silently drop the
        # third argument. Now uses torch.sym_sum which correctly handles
        # all arguments.
        # ------------------------------------------------------------------
        tracer = PythonKeyTracer()
        tracer.root = torch.nn.Module()
        tracer.graph = fx.Graph(tracer_cls=PythonKeyTracer)

        for sym, name in [(s0, "s0"), (s1, "s1"), (s2, "s2")]:
            node = tracer.graph.placeholder(name)
            proxy = fx.Proxy(node, tracer)
            set_meta(proxy, sym)
            tracer.sympy_expr_tracker[sym.node.expr] = _SympyExprTrackerValue(
                proxy=proxy, value=sym
            )
            tracer.symnode_tracker[sym] = Thunk(lambda p=proxy: p)

        _build_proxy_for_sym_expr(tracer, optimized_sum.node.expr, out=optimized_sum)

        out_proxy = tracer.symnode_tracker[optimized_sum].force()
        tracer.graph.output(out_proxy.node)

        gm = fx.GraphModule(tracer.root, tracer.graph)

        result = gm(2, 3, 4)  # should be 2 + 3 + 4 = 9
        self.assertEqual(result, 9)

    def test_build_proxy_for_nary_min_max(self):
        # [FUNCTION DOC] Checks that rebuilding proxies for flattened three-argument max and min
        # expressions of unbacked symints works with no output and with an output given; the
        # resulting graph executed on 2, 3, 4 must return 4 for max and 2 for min. Uses subtests.
        """
        Test that _build_proxy_for_sym_expr correctly handles flattened
        sympy.Max/sympy.Min expressions with more than 2 arguments.

        sympy flattens Max(a, Max(b, c)) -> Max(a, b, c) (and likewise for
        Min). _build_proxy_for_sym_expr must rebuild these even though
        torch.sym_max/sym_min are binary; a binary handler would raise
        TypeError on a 3-arg Max/Min.
        """
        import torch.fx as fx
        from torch.fx.experimental.proxy_tensor import (
            _build_proxy_for_sym_expr,
            _SympyExprTrackerValue,
            PythonKeyTracer,
            set_meta,
        )
        from torch.utils._thunk import Thunk

        # min/max of (2, 3, 4) -> 2 for min, 4 for max
        for sym_op, expected in [(torch.sym_max, 4), (torch.sym_min, 2)]:
            with self.subTest(op=sym_op.__name__):
                shape_env = ShapeEnv()
                # Unbacked symints keep the expression symbolic (backed symints
                # would specialize away the Max/Min).
                u0 = shape_env.create_unbacked_symint()
                u1 = shape_env.create_unbacked_symint()
                u2 = shape_env.create_unbacked_symint()

                flattened = sym_op(sym_op(u0, u1), u2)
                self.assertEqual(len(flattened.node.expr.args), 3)

                # Case 1: out=None must not raise TypeError on the 3-arg expr.
                tracer_none = PythonKeyTracer()
                for sym in [u0, u1, u2]:
                    tracer_none.sympy_expr_tracker[sym.node.expr] = (
                        _SympyExprTrackerValue(proxy=sym, value=sym)
                    )
                result = _build_proxy_for_sym_expr(tracer_none, flattened.node.expr)
                self.assertEqual(result.node.expr, flattened.node.expr)

                # Case 2: out=flattened (the typical get_proxy_slot path). The
                # rebuilt node must execute correctly (a binary handler would
                # have dropped the third argument or crashed).
                tracer = PythonKeyTracer()
                tracer.root = torch.nn.Module()
                tracer.graph = fx.Graph(tracer_cls=PythonKeyTracer)
                for sym, name in [(u0, "u0"), (u1, "u1"), (u2, "u2")]:
                    node = tracer.graph.placeholder(name)
                    proxy = fx.Proxy(node, tracer)
                    set_meta(proxy, sym)
                    tracer.sympy_expr_tracker[sym.node.expr] = _SympyExprTrackerValue(
                        proxy=proxy, value=sym
                    )
                    tracer.symnode_tracker[sym] = Thunk(lambda p=proxy: p)

                _build_proxy_for_sym_expr(tracer, flattened.node.expr, out=flattened)
                out_proxy = tracer.symnode_tracker[flattened].force()
                tracer.graph.output(out_proxy.node)
                gm = fx.GraphModule(tracer.root, tracer.graph)
                self.assertEqual(gm(2, 3, 4), expected)

    def test_build_proxy_for_pow(self):
        # [FUNCTION DOC] Checks that rebuilding proxies for a product containing a power of an
        # unbacked symbol (exponents 2 and 3) works with no output and with an output given; the
        # resulting graph executed on 2 and 3 must equal the plain integer product.
        """
        Test that _build_proxy_for_sym_expr correctly handles sympy.Pow.

        sympy canonicalizes x * x into Pow(x, 2), so a reduction numel over a
        tensor whose two dims duck-share a symbol becomes a nonlinear product
        like s0 * s1**2. _build_proxy_for_sym_expr must rebuild it; without a
        Pow handler the Pow factor has no proxy and get_proxy_slot raises
        "is not tracked with proxy".
        """
        import sympy

        import torch.fx as fx
        from torch.fx.experimental.proxy_tensor import (
            _build_proxy_for_sym_expr,
            _SympyExprTrackerValue,
            PythonKeyTracer,
            set_meta,
        )
        from torch.utils._thunk import Thunk

        # Cover more than just the square: the exponent is passed straight to
        # the handler, so any natural power must rebuild (s1**2, s1**3, ...).
        for exp in (2, 3):
            with self.subTest(exp=exp):
                shape_env = ShapeEnv()
                # Unbacked symints keep the expression symbolic (backed symints
                # would specialize the power away).
                u0 = shape_env.create_unbacked_symint()
                u1 = shape_env.create_unbacked_symint()

                prod = u0 * u1**exp
                self.assertTrue(prod.node.expr.has(sympy.Pow))

                # Case 1: out=None must rebuild the Pow expr, not return None.
                tracer_none = PythonKeyTracer()
                for sym in [u0, u1]:
                    tracer_none.sympy_expr_tracker[sym.node.expr] = (
                        _SympyExprTrackerValue(proxy=sym, value=sym)
                    )
                result = _build_proxy_for_sym_expr(tracer_none, prod.node.expr)
                self.assertEqual(result.node.expr, prod.node.expr)

                # Case 2: out=prod (the typical get_proxy_slot path). The rebuilt
                # node must execute correctly: u0 * u1**exp with u0=2, u1=3.
                tracer = PythonKeyTracer()
                tracer.root = torch.nn.Module()
                tracer.graph = fx.Graph(tracer_cls=PythonKeyTracer)
                for sym, name in [(u0, "u0"), (u1, "u1")]:
                    node = tracer.graph.placeholder(name)
                    proxy = fx.Proxy(node, tracer)
                    set_meta(proxy, sym)
                    tracer.sympy_expr_tracker[sym.node.expr] = _SympyExprTrackerValue(
                        proxy=proxy, value=sym
                    )
                    tracer.symnode_tracker[sym] = Thunk(lambda p=proxy: p)

                _build_proxy_for_sym_expr(tracer, prod.node.expr, out=prod)
                out_proxy = tracer.symnode_tracker[prod].force()
                tracer.graph.output(out_proxy.node)
                gm = fx.GraphModule(tracer.root, tracer.graph)
                self.assertEqual(gm(2, 3), 2 * 3**exp)

    def test_sym_max_multi_max_simplify(self):
        # [FUNCTION DOC] Checks that max of one and max of 257 and an unbacked symbol is statically
        # known equal to max of 257 and that symbol.
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        self.assertTrue(
            statically_known_true(
                torch.sym_max(1, torch.sym_max(257, u0)) == torch.sym_max(257, u0)
            )
        )

    def test_min_max_simplify_with_value_ranges(self):
        # [FUNCTION DOC] Checks that simplify removes redundant min/max terms using value ranges
        # established by checks (upper bound 5 on one symbol, lower bound 10 on another), and
        # leaves an unconstrained symbol's min/max unchanged.
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        u1 = shape_env.create_unbacked_symint()
        torch._check(u0 <= 5)
        torch._check(u1 >= 10)

        u0_expr = u0.node.expr
        u1_expr = u1.node.expr
        self.assertEqual(shape_env.simplify(Min(u0_expr, 5)), u0_expr)
        self.assertEqual(shape_env.simplify(Max(u1_expr, 10)), u1_expr)
        self.assertEqual(shape_env.simplify(Min(u0_expr, u1_expr, 20)), u0_expr)
        self.assertEqual(shape_env.simplify(Max(u1_expr, u0_expr, -1)), u1_expr)

        u2 = shape_env.create_unbacked_symint()
        u2_expr = u2.node.expr
        self.assertEqual(shape_env.simplify(Min(u2_expr, 5)), Min(5, u2_expr))
        self.assertEqual(shape_env.simplify(Max(u2_expr, 5)), Max(5, u2_expr))

    def test_numpy_sym_max(self):
        # [FUNCTION DOC] Checks sym_max with numpy scalar and python int/float mixes returns the
        # larger value (floats where a float is involved).
        self.assertEqual(torch.sym_max(np.int64(10), 12), 12)
        self.assertEqual(torch.sym_max(np.int64(12), 10), 12)
        self.assertEqual(torch.sym_max(np.int64(10), 12.5), 12.5)
        self.assertEqual(torch.sym_max(np.int64(14), 12.5), 14.0)
        self.assertEqual(torch.sym_max(np.float64(14.0), 12), 14.0)
        self.assertEqual(torch.sym_max(np.float64(14.0), 16), 16.0)

    def test_numpy_sym_min(self):
        # [FUNCTION DOC] Checks sym_min with numpy scalar and python int/float mixes returns the
        # smaller value (floats where a float is involved).
        self.assertEqual(torch.sym_min(np.int64(10), 12), 10)
        self.assertEqual(torch.sym_min(np.int64(12), 10), 10)
        self.assertEqual(torch.sym_min(np.int64(10), 12.5), 10.0)
        self.assertEqual(torch.sym_min(np.int64(14), 12.5), 12.5)
        self.assertEqual(torch.sym_min(np.float64(14.0), 12), 12.0)
        self.assertEqual(torch.sym_min(np.float64(14.0), 16), 14.0)

    def test_debug_has_internal_overlap_unbacked(self):
        # [FUNCTION DOC] Checks the internal-overlap debug function on meta tensors with unbacked
        # sizes: values 0 for non-overlapping layouts and 2 (too hard to tell) for broadcast-like
        # strides or layouts depending on unbacked sizes.
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        cf = torch._debug_has_internal_overlap
        self.assertEqual(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")), 0)
        self.assertEqual(cf(torch.empty_strided((2, u0), (1, 2), device="meta")), 0)
        self.assertEqual(cf(torch.empty_strided((u0,), (1,), device="meta")), 0)
        self.assertEqual(cf(torch.empty_strided((1,), (u0,), device="meta")), 2)
        Max = torch.sym_max
        self.assertEqual(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            ),
            2,
        )

        # Wobbling these to zero is OK too
        self.assertEqual(cf(torch.empty_strided((u0, 2), (3, 1), device="meta")), 2)
        self.assertEqual(cf(torch.empty_strided((2, u0), (1, 3), device="meta")), 2)

    def test_specialize_zero_one(self):
        # [FUNCTION DOC] Checks the specialize-zero-one option: comparing a size 5 to one adds no
        # guard when enabled, and adds one guard when disabled.
        shape_env = ShapeEnv(specialize_zero_one=True)
        a0 = create_symint(shape_env, 5)
        if a0 == 1:
            raise AssertionError("expected a0 != 1")
        self.assertEqual(len(shape_env.guards), 0)

        shape_env = ShapeEnv(specialize_zero_one=False)
        a0 = create_symint(shape_env, 5)
        if a0 == 1:
            raise AssertionError("expected a0 != 1")
        self.assertEqual(len(shape_env.guards), 1)

    def test_duck_shape(self):
        # [FUNCTION DOC] Checks duck shaping: two sizes with equal value compare equal with no
        # guard when duck shaping is on, and need one guard when it is off.
        shape_env = ShapeEnv(duck_shape=True)
        a0 = create_symint(shape_env, 5)
        a1 = create_symint(shape_env, 5)
        if a0 != a1:
            raise AssertionError("expected a0 == a1")
        self.assertEqual(len(shape_env.guards), 0)

        shape_env = ShapeEnv(duck_shape=False)
        a0 = create_symint(shape_env, 5)
        a1 = create_symint(shape_env, 5)
        if a0 != a1:
            raise AssertionError("expected a0 == a1")
        self.assertEqual(len(shape_env.guards), 1)

    def test_int_bool(self):
        # [FUNCTION DOC] Regression test for issue 95981: truthiness of a duck-sized symbolic int
        # works and adds no guard.
        # See https://github.com/pytorch/pytorch/issues/95981
        shape_env = ShapeEnv(duck_shape=True)
        a0 = create_symint(shape_env, 5)
        if not a0:
            raise AssertionError("expected a0 to be truthy")
        self.assertEqual(len(shape_env.guards), 0)

    def test_symint_as_scalar(self):
        # [FUNCTION DOC] Checks that a SymInt passed as a scalar argument (alpha) of add reaches
        # the dispatch mode with the same node, by recording that fact in an outer flag.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)

        sym_int_encountered = False

        class TestSymInt(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that intercepts add, records whether the alpha
            # keyword holds the original symbolic node, then calls the op with alpha zeroed.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Raises AssertionError if the op is not the aten add Tensor
                # overload. Sets the enclosing flag from comparing the alpha node identity,
                # overwrites alpha with 0 in the kwargs mapping, and calls the op with positional
                # args only.
                if func != torch.ops.aten.add.Tensor:
                    raise AssertionError(
                        f"expected torch.ops.aten.add.Tensor, got {func}"
                    )

                nonlocal sym_int_encountered
                # WARNING: do not do identity tests on the outer
                # SymInt/SymFloat, they are NOT STABLE
                sym_int_encountered = kwargs["alpha"].node is a0.node
                kwargs["alpha"] = 0
                return func(*args)

        x = torch.rand([4, 4])
        with TestSymInt():
            y = torch.add(x, x, alpha=a0)

        self.assertTrue(sym_int_encountered)

    def test_deepcopy(self):
        # [FUNCTION DOC] Checks deep-copying a ShapeEnv after one guard was recorded preserves that
        # guard in the copy.
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        if not (a0 < 4):
            raise AssertionError(f"expected a0 < 4, got a0={a0}")
        new_shape_env = copy.deepcopy(shape_env)
        self.assertEqual(len(new_shape_env.guards), 1)

    def test_print_readable_with_symints(self):
        # [FUNCTION DOC] Traces a function with symbolic make_fx on two tensors, prints the graph
        # readable form without output, and compares it with an expected string containing symbolic
        # size annotations.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Builds a new empty tensor sized by the sums of both
            # inputs' sizes and applies native dropout (training mode) to it, returning the
            # dropout outputs.
            dim0 = a.shape[0] + b.shape[0]
            dim1 = a.shape[1] + b.shape[1]
            d = a.new_empty(dim0, dim1)
            d = torch.ops.aten.native_dropout(d, 0.5, train=True)
            return d

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(5, 3), torch.randn(4, 3))
        out = fx_g.print_readable(print_output=False)

        self.assertExpectedInline(
            out.strip(),
            """\
class f(torch.nn.Module):
    def forward(self, a_1: "f32[s75, s96]", b_1: "f32[s57, s96]"):
        # No stacktrace found for following nodes
        sym_size_int: "Sym(s75)" = torch.ops.aten.sym_size.int(a_1, 0)
        sym_size_int_1: "Sym(s57)" = torch.ops.aten.sym_size.int(b_1, 0)
        add: "Sym(s57 + s75)" = sym_size_int + sym_size_int_1;  sym_size_int = sym_size_int_1 = None
        sym_size_int_2: "Sym(s96)" = torch.ops.aten.sym_size.int(a_1, 1)
        sym_size_int_3: "Sym(s96)" = torch.ops.aten.sym_size.int(b_1, 1);  b_1 = None
        add_1: "Sym(2*s96)" = sym_size_int_2 + sym_size_int_3;  sym_size_int_2 = sym_size_int_3 = None
        new_empty: "f32[s57 + s75, 2*s96]" = torch.ops.aten.new_empty.default(a_1, [add, add_1], pin_memory = False);  a_1 = add = add_1 = None
        native_dropout = torch.ops.aten.native_dropout.default(new_empty, 0.5, True);  new_empty = None
        getitem: "f32[s57 + s75, 2*s96]" = native_dropout[0]
        getitem_1: "b8[s57 + s75, 2*s96]" = native_dropout[1];  native_dropout = None
        return (getitem, getitem_1)""",
        )

    def test_statically_known_true(self):
        # [FUNCTION DOC] Checks statically_known_true on symbolic expressions: true only for facts
        # provable without hints, false for facts that are false, or true/false only for the hint
        # values; also asserts no guards were recorded.
        shape_env = ShapeEnv()
        s2, s3, s4 = (create_symint(shape_env, i) for i in range(2, 5))

        # Statically known true
        self.assertTrue(statically_known_true(True))
        self.assertTrue(statically_known_true(s2 == s2))
        self.assertTrue(statically_known_true(s2 * s3 > s3))
        self.assertTrue(statically_known_true(s3 * s4 > s4))
        self.assertTrue(statically_known_true((s3 + s3) % 2 == 0))

        # Statically known false
        self.assertFalse(statically_known_true(False))
        self.assertFalse(statically_known_true(s3 * s4 <= s4))
        self.assertFalse(statically_known_true((s3 + s3) % 2 == 1))

        # True for hints, but not known statically
        self.assertFalse(statically_known_true(s2 + s2 == s4))
        self.assertFalse(statically_known_true(s4 % s2 == 0))
        self.assertFalse(statically_known_true(s2 != s3))
        self.assertFalse(statically_known_true(s3 * s4 > s2))

        # False for hints, but not known statically
        self.assertFalse(statically_known_true(s2 == s3))
        self.assertFalse(statically_known_true(s2 > s3))
        self.assertFalse(statically_known_true(s3 + s3 == s4))

        # No guards should be generated
        self.assertEqual(len(shape_env.guards), 0)

    def test_statically_known_false(self):
        # [FUNCTION DOC] Mirror of the statically-known-true test for statically_known_false: true
        # only for facts provably false without hints; hint-only truths or falsities are not
        # enough. Also asserts no guards were recorded.
        shape_env = ShapeEnv()
        s2, s3, s4 = (create_symint(shape_env, i) for i in range(2, 5))

        # Statically known true
        self.assertFalse(statically_known_false(True))
        self.assertFalse(statically_known_false(s2 == s2))
        self.assertFalse(statically_known_false(s2 * s3 > s3))
        self.assertFalse(statically_known_false(s3 * s4 > s4))
        self.assertFalse(statically_known_false((s3 + s3) % 2 == 0))

        # Statically known false
        self.assertTrue(statically_known_false(False))
        self.assertTrue(statically_known_false(s3 * s4 <= s4))
        self.assertTrue(statically_known_false((s3 + s3) % 2 == 1))

        # True for hints, but not known statically
        self.assertFalse(statically_known_false(s2 + s2 == s4))
        self.assertFalse(statically_known_false(s4 % s2 == 0))
        self.assertFalse(statically_known_false(s2 != s3))
        self.assertFalse(statically_known_false(s3 * s4 > s2))

        # False for hints, but not known statically
        self.assertFalse(statically_known_false(s2 == s3))
        self.assertFalse(statically_known_false(s2 > s3))
        self.assertFalse(statically_known_false(s3 + s3 == s4))

        # No guards should be generated
        self.assertEqual(len(shape_env.guards), 0)

    def test_ephemeral_source_simplification(self):
        # [FUNCTION DOC] For each combination of which tensor gets an ephemeral source and the
        # order of comparison, creates two dynamic symbolic tensors, checks the ephemeral tensor
        # has symbols with ephemeral sources, asserts equality of sizes, strides and storage offset
        # via torch check, and verifies no ephemeral-source symbols remain.
        from torch._dynamo.source import EphemeralSource

        # For full robustness, ensure the ephemeral source symbols are simplified out regardless
        # of construction order or check order.
        for construct_ephemeral_first, x_first_in_check in itertools.product(
            [False, True], [False, True]
        ):
            shape_env = ShapeEnv()
            shape = (5, 10)
            dynamic_dims = [DimDynamic.DYNAMIC for _ in shape]
            x = create_symbolic_tensor(
                "x",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if construct_ephemeral_first else None),
                dynamic_dims=dynamic_dims,
            )
            y = create_symbolic_tensor(
                "y",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if not construct_ephemeral_first else None),
                dynamic_dims=dynamic_dims,
            )
            t_with_ephemeral = x if construct_ephemeral_first else y

            def _get_ephemeral_source_symbols(t):
                # [FUNCTION DOC] HELPER: Returns the sympy expressions of all symbolic sizes,
                # strides and storage offset of a tensor whose symbols have at least one ephemeral
                # source in the shape env's source table.
                return [
                    s.node.expr
                    for s in itertools.chain(t.shape, t.stride(), (t.storage_offset(),))
                    if isinstance(s, torch.SymInt)
                    and s.node.expr in shape_env.var_to_sources
                    and any(
                        source.is_ephemeral()
                        for source in shape_env.var_to_sources[s.node.expr]
                    )
                ]

            # these checks should simplify out the ephemeral symbols, regardless of the
            # ordering x == y or y == x
            self.assertTrue(len(_get_ephemeral_source_symbols(t_with_ephemeral)) > 0)
            if x_first_in_check:
                torch._check(x.size() == y.size())
                torch._check(x.stride() == y.stride())
                torch._check(x.storage_offset() == y.storage_offset())
            else:
                torch._check(y.size() == x.size())
                torch._check(y.stride() == x.stride())
                torch._check(y.storage_offset() == x.storage_offset())
            self.assertEqual(len(_get_ephemeral_source_symbols(t_with_ephemeral)), 0)

    def test_ephemeral_source_unified_with_non_ephemeral_source(self):
        # [FUNCTION DOC] With duck sizing and equal shapes, creates two symbolic tensors, one with
        # an ephemeral source, in both construction orders. Checks the first source of every symbol
        # is non-ephemeral and that sizes, strides and storage offsets compare equal.
        from torch._dynamo.source import EphemeralSource

        for construct_ephemeral_first in (False, True):
            shape_env = ShapeEnv()
            shape = (5, 10)
            # use duck sizing here to ensure symbol reuse across x and y
            duck_dims = [DimDynamic.DUCK for _ in shape]
            x = create_symbolic_tensor(
                "x",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if construct_ephemeral_first else None),
                dynamic_dims=duck_dims,
            )
            y = create_symbolic_tensor(
                "y",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if not construct_ephemeral_first else None),
                dynamic_dims=duck_dims,
            )

            # regardless of construction order, non-ephemeral sources should be preferred
            # first in the var_to_sources list for potential guarding later on
            for source_list in shape_env.var_to_sources.values():
                self.assertFalse(source_list[0].is_ephemeral())

            self.assertEqual(x.size(), y.size())
            self.assertEqual(x.stride(), y.stride())
            self.assertEqual(x.storage_offset(), y.storage_offset())

    def test_tensor_factory_with_symint(self):
        # [FUNCTION DOC] Checks torch tensor and the legacy tensor constructors accept a list of
        # SymInts for every dtype tried and produce values equal to those from plain ints.
        args = list(range(3))
        expected = torch.tensor(args)

        shape_env = ShapeEnv()
        sym_args = [create_symint(shape_env, i) for i in args]

        # test tensor factories
        for dt in all_types_and(torch.half, torch.bfloat16):
            res = torch.tensor(sym_args, dtype=dt)
            self.assertEqual(res, expected, exact_dtype=False)

        # test legacy tensor factories
        legacy_ctors = [
            torch.Tensor,
            torch.LongTensor,
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.IntTensor,
            torch.ShortTensor,
            torch.HalfTensor,
            torch.ByteTensor,
        ]
        for Tensor in legacy_ctors:
            res = Tensor(sym_args)
            self.assertEqual(res, expected, exact_dtype=False)

    def test_backed_size_oblivious_01_spec(self):
        # [FUNCTION DOC] Under the backed-size-oblivious config, checks a compiled function that
        # branches on a size-oblivious equality of a size to 1 takes the other branch even when the
        # size is 1 (returns 20 rather than 10).
        from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

        @torch.compile(dynamic=True, fullgraph=True)
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns b times 10 if the size-oblivious check says the first
            # size of a equals one, else b times 20.
            if guard_size_oblivious(a.size(0) == 1):
                return b * 10
            else:
                return b * 20

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            # always go to the >= 2 branch.
            self.assertEqual(
                f(torch.tensor([1]), torch.tensor([1])), torch.tensor([20])
            )

    @fresh_cache()
    def test_slice_backed_size_oblivious(self):
        # [FUNCTION DOC] Compiles slicing the first five rows with dynamic shapes through inductor
        # under the backed-size-oblivious config and runs it on a ten by ten input, expecting no
        # error. Uses a fresh inductor cache.
        @torch.compile(backend="inductor", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first five rows of x.
            return x[:5]

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            f(torch.randn(10, 10))

    def test_baddbmm_symint(self):
        # [FUNCTION DOC] Smoke test: baddbmm on meta tensors whose batch and matrix dimensions are
        # unbacked symbols under a fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)

        B, M, K, N = [shape_env.create_unbacked_symint() for _ in range(4)]

        with fake_mode:
            A = torch.empty((B, M, K), device="meta")
            Bmat = torch.empty((B, K, N), device="meta")
            bias3 = torch.empty((B, M, N), device="meta")

            _ = torch.baddbmm(bias3, A, Bmat)

    def test_lu_unpack_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: lu_unpack on a meta tensor with unbacked sizes under a fake
        # mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        m, n = [shape_env.create_unbacked_symint() for _ in range(2)]

        with fake_mode:
            LU = torch.empty((m, n), device="meta")
            pivots = torch.empty((1,), device="meta", dtype=torch.int32)
            P, L, U = torch.lu_unpack(LU, pivots)

    def test_linalg_qr_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: QR decomposition on a meta tensor with unbacked sizes under a
        # fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        m, n = [shape_env.create_unbacked_symint() for _ in range(2)]

        with fake_mode:
            A = torch.empty((m, n), device="meta")
            Q, R = torch.linalg.qr(A)

    def _make_unbacked_size(self, shape_env):
        # [FUNCTION DOC] HELPER: Returns a fresh unbacked symbolic int from the given shape env.
        return shape_env.create_unbacked_symint()

    def test_avg_pool2d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: avg_pool2d with unbacked spatial sizes on a meta tensor under
        # a fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        u = self._make_unbacked_size
        with fake_mode:
            x = torch.empty((1, 3, u(shape_env), u(shape_env)), device="meta")
            torch.ops.aten.avg_pool2d(x, [2, 2], [2, 2], [0, 0])

    def test_max_pool2d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: max_pool2d with indices and unbacked spatial sizes under a
        # fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        u = self._make_unbacked_size
        with fake_mode:
            x = torch.empty((1, 3, u(shape_env), u(shape_env)), device="meta")
            torch.ops.aten.max_pool2d_with_indices(x, [2, 2], [2, 2])

    def test_avg_pool3d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: avg_pool3d with three unbacked spatial sizes under a fake mode
        # must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        u = self._make_unbacked_size
        with fake_mode:
            x = torch.empty(
                (1, 3, u(shape_env), u(shape_env), u(shape_env)), device="meta"
            )
            torch.ops.aten.avg_pool3d(x, [2, 2, 2], [2, 2, 2])

    def test_max_pool3d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: max_pool3d with indices and three unbacked spatial sizes under
        # a fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        u = self._make_unbacked_size
        with fake_mode:
            x = torch.empty(
                (1, 3, u(shape_env), u(shape_env), u(shape_env)), device="meta"
            )
            torch.ops.aten.max_pool3d_with_indices(x, [2, 2, 2], [2, 2, 2])

    def test_avg_pool1d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: avg_pool1d with an unbacked length under a fake mode must
        # not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)

        with fake_mode:
            x = torch.empty((1, 3, self._make_unbacked_size(shape_env)), device="meta")
            torch.ops.aten.avg_pool1d(x, [2], [2])

    def test_pixel_shuffle_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: pixel_shuffle with an unbacked channel count (upscale factor
        # 1) under a fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        u = self._make_unbacked_size
        with fake_mode:
            x = torch.empty((1, u(shape_env), 3, 3), device="meta")
            torch.ops.aten.pixel_shuffle(x, 1)

    def test_pdist_forward_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: pairwise distance forward on a matrix with an unbacked row
        # count under a fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            x = torch.empty((self._make_unbacked_size(shape_env), 5), device="meta")
            torch.ops.aten._pdist_forward(x, 2.0)

    def test_reflection_pad2d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: 2D reflection padding with unbacked spatial sizes under a fake
        # mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        u = self._make_unbacked_size
        with fake_mode:
            x = torch.empty((1, 3, u(shape_env), u(shape_env)), device="meta")
            torch.ops.aten.reflection_pad2d(x, [1, 1, 1, 1])

    def test_replication_pad2d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: 2D replication padding with unbacked spatial sizes under a
        # fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        u = self._make_unbacked_size
        with fake_mode:
            x = torch.empty((1, 3, u(shape_env), u(shape_env)), device="meta")
            torch.ops.aten.replication_pad2d(x, [1, 1, 1, 1])

    def test_reflection_pad1d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: 1D reflection padding with an unbacked length under a fake
        # mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            x = torch.empty((1, 3, self._make_unbacked_size(shape_env)), device="meta")
            torch.ops.aten.reflection_pad1d(x, [1, 1])

    def test_replication_pad1d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: 1D replication padding with an unbacked length under a fake
        # mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            x = torch.empty((1, 3, self._make_unbacked_size(shape_env)), device="meta")
            torch.ops.aten.replication_pad1d(x, [1, 1])

    def test_reflection_pad3d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: 3D reflection padding with three unbacked spatial sizes under
        # a fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        u = self._make_unbacked_size
        with fake_mode:
            x = torch.empty(
                (1, 3, u(shape_env), u(shape_env), u(shape_env)), device="meta"
            )
            torch.ops.aten.reflection_pad3d(x, [1, 1, 1, 1, 1, 1])

    def test_replication_pad3d_unbacked_symint(self):
        # [FUNCTION DOC] Smoke test: 3D replication padding with three unbacked spatial sizes under
        # a fake mode must not raise.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        u = self._make_unbacked_size
        with fake_mode:
            x = torch.empty(
                (1, 3, u(shape_env), u(shape_env), u(shape_env)), device="meta"
            )
            torch.ops.aten.replication_pad3d(x, [1, 1, 1, 1, 1, 1])


@skipIfTorchDynamo(
    "Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)"
)
class TestSymNumberMagicMethods(TestCase):
    # [CLASS DOC] Tests that magic methods (arithmetic, comparison, bitwise, math functions) on
    # symbolic ints, floats and bools agree with the plain python results, over a grid of small
    # values, with the symbolic value on the left, on the right or on both sides. Test methods are
    # expanded per magic method name by the parametrization call below the class. Also covers
    # hashing, deep copy and constant/nested-int symbolic nodes.
    def _do_test(self, fn, inp1, inp2, shape_env, is_unary_fn):
        # [FUNCTION DOC] HELPER: Runs the comparison for one function and input pair inside a
        # subtest labelled with the function and inputs; returns whatever the worker returns.
        with self.subTest(fn=fn, inp1=inp1, inp2=inp2, is_unary_fn=is_unary_fn):
            return self._do_test2(fn, inp1, inp2, shape_env, is_unary_fn)

    def _do_test2(self, fn, inp1, inp2, shape_env, is_unary_fn):
        # [FUNCTION DOC] HELPER: Core check for one magic method. Computes the reference result on
        # plain values, then repeats with the first input made symbolic, the second made symbolic,
        # and both symbolic (the latter two skipped for unary functions), asserting each result is
        # a symbolic number and that its guarded value equals the reference. Expected python errors
        # are tolerated through an exception-expecting context. Some function and input
        # combinations (negative base for float pow, negative exponent for natural pow) return
        # early without checking.
        # Helper function
        # NB: don't use one as that will get specialized
        # TODO: We don't have to circuitously create the float, can just
        # create a symfloat directly
        seed_node = (create_symint(shape_env, 2) / 2.0).node
        bool_seed_node = (create_symint(shape_env, 2) == 2).node

        def get_sym_inp(inp):
            # [FUNCTION DOC] HELPER: Wraps a python value into a symbolic node of the matching
            # kind (SymBool for bool, SymInt for int, otherwise SymFloat) reusing the seed node's
            # shape env.
            # NB: this must come before int
            if isinstance(inp, bool):
                return torch.SymBool(to_node(bool_seed_node, inp))
            elif isinstance(inp, int):
                return torch.SymInt(to_node(seed_node, inp))
            else:
                return torch.SymFloat(to_node(seed_node, inp))

        if fn == "float_pow":
            if inp1 < 0:
                return

        if fn == "pow_by_natural":
            if isinstance(inp1, float) or isinstance(inp2, float):
                return
            if inp2 < 0:
                return

        def maybe_xfail(inp1, inp2):
            # [FUNCTION DOC] HELPER: Returns the context manager for one call: an assertRaises
            # context for the error python is expected to raise (math domain error for negative
            # square root, zero division for division, modulo and zero base with negative exponent,
            # a runtime error for complex pow results, TypeError for shifts on non-ints, ValueError
            # for negative shift counts), otherwise a null context.
            if fn == "sym_sqrt" and inp1 < 0:
                # ValueError: math domain error
                return self.assertRaises((ValueError,))
            elif (
                fn in ("float_truediv", "int_truediv", "int_floordiv", "mod")
                and inp2 == 0
            ):
                # ZeroDivisionError: division by zero
                return self.assertRaises((ZeroDivisionError,))
            elif fn in ["float_pow", "pow_by_natural"] and inp1 == 0 and inp2 < 0:
                # ZeroDivisionError: 0.0 cannot be raised to a negative power
                return self.assertRaises((ZeroDivisionError,))
            elif (
                # TODO: dear catastrophe waitress,
                # this doesn't work
                fn in ["float_pow", "pow_by_natural"]
                and inp1 < 0
                and (
                    type(inp1) is (SymInt, SymFloat) or type(inp2) is (SymInt, SymFloat)
                )
                and (type(inp1) is (SymFloat, float) or type(inp2) is (SymFloat, float))
            ):
                # Complex result, which we do not support:
                # TypeError: Cannot convert complex to float
                return self.assertRaises((RuntimeError,))
            elif fn in ("lshift", "rshift") and not (
                isinstance(inp1, (SymInt, int)) and isinstance(inp2, (SymInt, int))
            ):
                # TypeError: unsupported operand type(s)
                return self.assertRaises((TypeError,))
            elif fn in ("lshift", "rshift") and inp2 < 0:
                # ValueError: math domain error
                return self.assertRaises((ValueError,))
            else:
                return contextlib.nullcontext()

        lambda_apply = method_to_operator(fn)

        def guard_fn(v):
            # [FUNCTION DOC] HELPER: Converts a symbolic or plain value to a plain python value by
            # calling the guard function matching its type (bool, float, or int).
            if type(v) in (SymBool, bool):
                return guard_bool(v)
            elif type(v) in (SymFloat, float):
                return guard_float(v)
            else:  # SymInt, int
                return guard_int(v)

        # Get reference result
        with maybe_xfail(inp1, inp2):
            if is_unary_fn:
                ref_out = lambda_apply(inp1)
            else:
                ref_out = lambda_apply(inp1, inp2)

        # Symified first arg
        sym_inp1 = get_sym_inp(inp1)
        with maybe_xfail(sym_inp1, inp2):
            if is_unary_fn:
                out = lambda_apply(sym_inp1)
            else:
                out = lambda_apply(sym_inp1, inp2)
            self.assertTrue(isinstance(out, (SymInt, SymFloat, SymBool)))
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

        if is_unary_fn:
            return

        # Symified second arg
        sym_inp2 = get_sym_inp(inp2)
        with maybe_xfail(inp1, sym_inp2):
            out = lambda_apply(inp1, sym_inp2)
            self.assertTrue(isinstance(out, (SymInt, SymFloat, SymBool)))
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

        # Symified both args
        with maybe_xfail(sym_inp1, sym_inp2):
            out = lambda_apply(sym_inp1, sym_inp2)
            self.assertTrue(isinstance(out, (SymInt, SymFloat, SymBool)))
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

    @parametrize("fn", list(sym_node.magic_methods.keys()))
    def test_bool_method(self, fn):
        # [FUNCTION DOC] Parametrized over every magic method name. Skips methods that are not
        # boolean magic methods and sym_ite; otherwise checks the method with the bool inputs true
        # and false on a fresh shape env.
        # sym_ite has its own tests
        if fn not in sym_node.bool_magic_methods or fn == "sym_ite":
            self.skipTest(f"{fn} is non-bool")

        is_unary_fn = fn in sym_node.unary_methods
        shape_env = ShapeEnv()
        self._do_test(fn, True, False, shape_env, is_unary_fn)

    @parametrize("fn", list(sym_node.magic_methods.keys()))
    @parametrize("first_type", ["int", "float"])
    @parametrize("second_type", ["int", "float"])
    def test_method(self, fn, first_type, second_type):
        # [FUNCTION DOC] Parametrized over magic method name and the input types. Skips every case
        # with a float first type, float-only methods with an int argument, mod with a float second
        # argument, bitwise ops with float inputs, repeated unary cases and bool methods. Otherwise
        # runs the check on all pairs of values from zero, one and a middle value (with negatives)
        # on a fresh shape env, converting to int when the type is int.
        if first_type == "float":
            # TODO: Hmm, this looks like we skip all floats
            self.skipTest(f"{fn} is not a float magic method")

        if (
            first_type == "int" or second_type == "int"
        ) and fn in sym_node.only_float_magic_methods:
            self.skipTest(f"{fn} is not an int method")

        if second_type == "float" and fn == "mod":
            self.skipTest(f"{fn} only handles int")

        if fn in sym_node.bitwise_ops and (first_type != "int" or second_type != "int"):
            self.skipTest(f"{fn} is a bitwise op, only handles int")

        is_unary_fn = fn in sym_node.unary_methods or fn == "round"
        # Second argument is ignored for unary function. So only run for one type
        if is_unary_fn and second_type == "float":
            self.skipTest(f"{fn} is unary and already tested")

        if fn in sym_node.bool_magic_methods:
            self.skipTest(f"{fn} is bool")

        # Only floats here since these will be converted to int if necessary.
        # We also ignore complex and bool.
        values = (
            0.0,
            1.0,
            0.5 if fn in ("sym_acos", "sym_asin") else 2.5,  # avoid math domain error
        )

        neg_values = tuple(-x for x in values)

        for inp1, inp2 in itertools.chain(
            itertools.product(values, values),
            itertools.product(values, neg_values),
            itertools.product(neg_values, values),
            itertools.product(neg_values, neg_values),
        ):
            if first_type == "int":
                inp1 = int(inp1)
            if second_type == "int":
                inp2 = int(inp2)

            shape_env = ShapeEnv()

            self._do_test(fn, inp1, inp2, shape_env, is_unary_fn)

    def get_constant_bool(self, val):
        # [FUNCTION DOC] HELPER: Returns a SymBool wrapping a constant boolean symbolic node.
        return SymBool(torch._C._get_constant_bool_symnode(val))

    @unittest.expectedFailure
    def test_symint_hashing(self):
        # [FUNCTION DOC] Expected-failure test: hashing a symbolic int backed by a ShapeEnv is
        # meant to fail, so the test passes only if hashing raises.
        shape_env = ShapeEnv()
        hash(create_symint(shape_env, 3))

    def test_symnode_hashing(self):
        # [FUNCTION DOC] Checks hashing behavior: symbolic bool and float hashing is allowed (and
        # specializes); nested ints and constant symbolic bools with equal value compare equal
        # and hash equal, different ones compare unequal and hash differently; symbolic nodes can
        # be hashed.
        shape_env = ShapeEnv()

        # These all trigger specialization when hashed
        hash(create_symbool(shape_env, True))
        # We should be passing in float here, but create_symbol currently
        # only supports int
        hash(create_symfloat(shape_env, 3.0))

        # NestedInt (SymInt), constant SymBool, SymNode are hashable
        j1 = torch._C._get_nested_int(1, 1)
        j1_copy = torch._C._get_nested_int(1, 1)
        j2 = torch._C._get_nested_int(2, 1)
        t = self.get_constant_bool(True)
        t_copy = self.get_constant_bool(True)
        f = self.get_constant_bool(False)
        n = create_symint(shape_env, 3).node
        m = self.get_constant_bool(True).node

        self.assertIs(j1 == j1_copy, True)
        self.assertEqual(hash(j1), hash(j1_copy))
        self.assertIs(j1 == j2, False)
        self.assertNotEqual(hash(j1), hash(j2))
        self.assertIs(t == t_copy, True)
        self.assertEqual(hash(t), hash(t_copy))
        self.assertIs(t == f, False)
        self.assertNotEqual(hash(t), hash(f))

        hash(n)
        hash(m)

    def test_symint_deepcopy(self):
        # [FUNCTION DOC] Checks that deep-copying a tuple holding a nested int gives an
        # equal tuple.
        shape_env = ShapeEnv()

        symnodes = (torch._C._get_nested_int(1, 1),)
        deepcopied_symnodes = copy.deepcopy(symnodes)
        self.assertEqual(symnodes, deepcopied_symnodes)

    def test_non_symbolic_symnode(self):
        # [FUNCTION DOC] Checks nested ints and constant symbolic bools: nested ints are SymInt but
        # not int, support equality but raise RuntimeError on addition or ordering; constant
        # SymBool supports not and logical and with bools and symbolic bools (results stay symbolic
        # and guard to True); torch.Size objects holding nested ints compare as expected.
        j1 = torch._C._get_nested_int(1, 1)
        j2 = torch._C._get_nested_int(1, 1)
        j3 = torch._C._get_nested_int(3, 1)

        self.assertIsInstance(j1, torch.SymInt)
        self.assertNotIsInstance(j1, int)

        with self.assertRaisesRegex(
            RuntimeError, "add not supported by NestedIntSymNode"
        ):
            j1 + 3

        self.assertFalse(j1 == 3)
        with self.assertRaisesRegex(RuntimeError, "indeterminate"):
            self.assertFalse(3 >= j2)

        self.assertIs(j1 == j1, True)
        self.assertIs(j1 == j2, True)
        self.assertIs(j1 == j3, False)
        self.assertIs(j1 != j3, True)
        self.assertIs(j1 != j2, False)

        x = self.get_constant_bool(True)
        #
        # Unary
        #
        # op(constant SymBool)
        self.assertIs(x.__sym_not__(), False)

        #
        # Binary
        #
        # op(constant SymBool, bool)
        # op(constant SymBool, constant SymBool)
        # op(bool, constant SymBool)
        self.assertIs(operator.and_(x, True), True)
        self.assertIs(operator.and_(x, x), True)
        self.assertIs(operator.and_(True, x), True)

        # op(symbolic SymBool, constant Symbool)
        # op(constant SymBool, symbolic Symbool)
        shape_env = ShapeEnv()
        a = create_symint(shape_env, 2)
        b = create_symint(shape_env, 2)
        c = a == b  # symbolic SymBool
        d = self.get_constant_bool(True)
        e = operator.and_(c, d)
        f = operator.and_(d, c)
        self.assertTrue(is_symbolic(e))
        self.assertTrue(is_symbolic(f))
        self.assertIs(e.node.guard_bool("", 0), True)
        self.assertIs(f.node.guard_bool("", 0), True)

        # Comparing sizes
        sz1 = torch.Size([j1, j1, j1])
        sz2 = torch.Size([j1, j1, j1])
        self.assertIs(sz1 == sz2, True)

        sz1 = torch.Size([3, j1, 4])
        sz2 = torch.Size([3, j2, 4])
        self.assertIs(sz1 == sz2, True)
        self.assertIs(sz1 != sz2, False)

    def test_stride_symnode(self):
        # [FUNCTION DOC] Checks combinations of static, dynamic and inferred sizes and strides on
        # fake tensors: all-static gives plain ints; dynamic sizes with inferred strides give
        # symbolic strides tied to the sizes; static sizes with a dynamic stride give one symbolic
        # stride; dynamic sizes and strides give a stride symbol different from the size symbol.
        shape_env = ShapeEnv()

        # check everything static
        t = create_fake_tensor_with_dynamic_size(
            torch.ones(3, 6),
            shape_env,
            dynamic_sizes=[
                DimDynamic.STATIC,
                DimDynamic.STATIC,
            ],
            dynamic_strides=[
                DimDynamic.INFER_STRIDE,
                DimDynamic.INFER_STRIDE,
            ],
        )
        self.assertTrue(all(isinstance(size, int) for size in t.size()))
        self.assertTrue(all(isinstance(stride, int) for stride in t.stride()))

        # check dynamic size but static dims
        t = create_fake_tensor_with_dynamic_size(
            torch.ones(3, 6),
            shape_env,
            dynamic_sizes=[
                DimDynamic.DYNAMIC,
                DimDynamic.DYNAMIC,
            ],
            dynamic_strides=[
                DimDynamic.INFER_STRIDE,
                DimDynamic.INFER_STRIDE,
            ],
        )
        # Expect stride to be inferred
        s0, s1 = t.size()
        s2, s3 = t.stride()
        self.assertTrue(isinstance(s0, torch.SymInt))
        self.assertTrue(isinstance(s1, torch.SymInt))
        self.assertTrue(isinstance(s2, torch.SymInt))
        self.assertTrue(s1 == s2)
        self.assertEqual(s3, 1)

        # Check dynamic stride but static dims
        t = create_fake_tensor_with_dynamic_size(
            torch.ones(3, 6),
            shape_env,
            dynamic_sizes=[
                DimDynamic.STATIC,
                DimDynamic.STATIC,
            ],
            dynamic_strides=[
                DimDynamic.DYNAMIC,
                DimDynamic.INFER_STRIDE,
            ],
        )
        s0, s1 = t.size()
        s2, s3 = t.stride()
        self.assertTrue(isinstance(s0, int))
        self.assertTrue(isinstance(s1, int))
        self.assertTrue(isinstance(s2, torch.SymInt))
        self.assertTrue(isinstance(s3, int))

        # Check dynamic sizes and dims, and ensure different symbol
        t = create_fake_tensor_with_dynamic_size(
            torch.ones(3, 6),
            shape_env,
            dynamic_sizes=[
                DimDynamic.DYNAMIC,
                DimDynamic.DYNAMIC,
            ],
            dynamic_strides=[
                DimDynamic.DYNAMIC,
                DimDynamic.INFER_STRIDE,
            ],
        )
        s0, s1 = t.size()
        s2, s3 = t.stride()
        self.assertTrue(isinstance(s0, torch.SymInt))
        self.assertTrue(isinstance(s1, torch.SymInt))
        self.assertTrue(isinstance(s2, torch.SymInt))
        self.assertTrue(isinstance(s3, int))
        self.assertTrue(str(s1.node.expr) != str(s2.node.expr))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_dynamic_int_basic_compile(self, backend):
        # [FUNCTION DOC] Parametrized over inductor and eager backends with scalar-output capture
        # enabled and a fresh cache. Compiles a function taking scalar ints, some wrapped as
        # dynamic ints, and checks results against eager across different values with a single
        # frame (no recompiles) while one case recompiles; then compiles a module holding a dynamic
        # int attribute, overrides it, and checks results and frame count.
        from torch.fx.experimental.sym_node import DynamicInt

        cnt = CompileCounterWithBackend(backend)

        # test scalar inputs to function
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Builds a tensor holding the sum of three scalars and adds the
            # sum of a zeros tensor sized by the absolute value of the first scalar plus two.
            out = torch.tensor([x + y + z])
            out = out + torch.zeros(abs(x) + 2).sum()  # test out tensor construction
            return out

        fn = torch.compile(f, fullgraph=True, backend=cnt)
        x = DynamicInt(1)
        z = DynamicInt(3)
        self.assertEqual(fn(x, x, z), f(1, 1, 3))  # guard: x == y
        self.assertEqual(fn(2, 2, 0), f(2, 2, 0))
        self.assertEqual(fn(-1, -1, 2), f(-1, -1, 2))
        self.assertEqual(cnt.frame_count, 1)  # no recompiles

        self.assertEqual(fn(3, 4, 5), f(3, 4, 5))  # now we recompile
        self.assertEqual(cnt.frame_count, 2)

        # test nn module property
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a dynamic int attribute; forward returns a tensor
            # holding the input plus that attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and sets the attribute i to a
                # dynamic int with value one.
                super().__init__()
                self.i = DynamicInt(1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a one-element tensor with x plus the attribute i.
                return torch.tensor([x + self.i])

        cnt.clear()
        m = Foo()
        mc = torch.compile(m, backend=cnt, fullgraph=True)

        self.assertEqual(mc(DynamicInt(0)), m(0))
        mc.i = -2  # override attribute
        self.assertEqual(mc(-1), m(-1))
        self.assertEqual(cnt.frame_count, 1)

    def test_dynamic_int_eager_usage(self):
        # [FUNCTION DOC] Checks DynamicInt used eagerly: arithmetic, shifts, bitwise or, min, pow
        # (including three-argument pow) keep the DynamicInt type and expected values; modulo by
        # zero raises ZeroDivisionError; math, numpy and torch functions (arange, add with alpha,
        # Linear sizes, sum) accept it as a plain number; int/float/bool conversions behave as for
        # ints, and true division returns a plain float.
        from torch.fx.experimental.sym_node import DynamicInt

        w = DynamicInt(-1)
        x = DynamicInt(0)
        y = DynamicInt(1)
        z = DynamicInt(2)

        def check(l, r):
            # [FUNCTION DOC] HELPER: Asserts the left value is a DynamicInt and equals the
            # right value.
            self.assertTrue(isinstance(l, DynamicInt))
            self.assertEqual(l, r)

        # test arithmetic
        check(2 * y + z, 4)
        check((10 - z) // 2, 4)
        check(1 // z, 0)
        check(-w + w**2, 2)
        check(x % z, 0)
        check(1 << z, 4)
        check(z | y, 3)
        check(min(y, z), 1)
        self.assertTrue(z > -2)
        with self.assertRaises(ZeroDivisionError):
            y % x

        # test pow operations preserve DynamicInt type
        check(w**2, 1)  # DynamicInt ** int
        check(2**z, 4)  # int ** DynamicInt
        check(y**x, 1)  # DynamicInt ** DynamicInt
        check(pow(z, 2), 4)  # pow(DynamicInt, int)
        self.assertTrue(isinstance(pow(y, 3, 5), DynamicInt))  # pow with modulo

        # math, numpy
        self.assertEqual(math.cos(x), y)
        self.assertEqual(math.prod([z, z], start=z), 8)
        self.assertEqual(np.arange(z)[y], 1)
        self.assertTrue(np.allclose(np.ones([y, z]).sum(axis=x), np.ones(z)))

        # test conversions
        self.assertTrue(isinstance(x + 2, int))
        self.assertTrue(isinstance(x + 2, DynamicInt))
        self.assertEqual(y / 2.0, 0.5)  # this could return DynamicFloat in future
        self.assertEqual(float(z), 2.0)
        self.assertFalse(bool(x))
        self.assertEqual(DynamicInt(x).real, x.real)

        # torch functions, scalar inputs
        self.assertEqual(torch.arange(z)[:w][x], 0)
        self.assertEqual(torch.add(torch.tensor(w), torch.tensor(w), alpha=z), -3)
        self.assertEqual(
            list(torch.nn.Linear(z, y)(torch.randn(z * 2, z)).shape), [4, 1]
        )
        self.assertEqual(z * torch.ones(z).sum(dim=x), 4)


instantiate_parametrized_tests(TestSymNumberMagicMethods)


class TestFloorDiv(TestCase):
    # [CLASS DOC] Tests that the symbolic FloorDiv function agrees with python floor division on
    # integer inputs of both signs, simplifies and evaluates constants, avoids non-integer
    # rationals, and has the expected integer/real assumptions.
    @staticmethod
    def python_floordiv(x, y):
        # [FUNCTION DOC] HELPER: Static method returning python floor division of x by y.
        return x // y

    @staticmethod
    def torch_floordiv(x, y):
        # [FUNCTION DOC] HELPER: Static method that creates a fresh ShapeEnv and fully evaluates
        # the FloorDiv expression for x and y, returning the result.
        # Note: we fully evaluate here since FloorDiv might not always do
        # that.
        shape_env = ShapeEnv()
        return shape_env.evaluate_expr(FloorDiv(x, y))

    @staticmethod
    def yield_test_cases(values, negate=True):
        # [FUNCTION DOC] HELPER: Generator over input pairs: yields each (x, y) pair, and when
        # negate is true also the three sign-flipped variants of it.
        for x, y in values:
            yield (x, y)
            if negate:
                yield (-x, y)
                yield (x, -y)
                yield (-x, -y)

    def test_floordiv_float_int(self):
        # [FUNCTION DOC] Checks the symbolic FloorDiv matches python floor division for 7 and 2 and
        # their sign-flipped variants.
        values = ((7, 2),)

        for x, y in TestFloorDiv.yield_test_cases(values):
            self.assertEqual(
                TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y)
            )

    def test_floordiv_div_by_one(self):
        # [FUNCTION DOC] Checks the symbolic FloorDiv matches python floor division when dividing 2
        # by 1 and for sign-flipped variants.
        values = ((2, 1),)

        for x, y in TestFloorDiv.yield_test_cases(values):
            self.assertEqual(
                TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y)
            )

    def test_floordiv_div_does_not_generate_non_int_rational(self):
        # [FUNCTION DOC] Builds a nested FloorDiv expression with positive integer symbols
        # and checks every rational number in the result has denominator one (no
        # non-integer rationals).
        s14 = sympy.Symbol("s14", integer=True, positive=True)
        s37 = sympy.Symbol("s37", integer=True, positive=True)

        inner_expr = FloorDiv(s14, 2016)
        middle_expr = (24 * s37 + 672) * inner_expr
        numerator = middle_expr + 21
        denominator = 22
        result = FloorDiv(numerator, denominator)
        rationals = result.atoms(sympy.Rational)
        all_rationals_ints = all(r.q == 1 for r in rationals)
        self.assertTrue(all_rationals_ints)

    def test_floordiv_simplify(self):
        # [FUNCTION DOC] Checks that an expression containing FloorDiv of constants equals 21 and
        # evaluates to 21 via equality, doit (shallow and deep), sympy simplify, ShapeEnv simplify
        # and ShapeEnv evaluate.
        # Tests how we simplify or evaluate FloorDiv without free variables
        shape_env = ShapeEnv()
        result = 21
        exprs = (7 * FloorDiv(6, 2),)

        for expr in exprs:
            self.assertEqual(expr, result)
            self.assertEqual(expr.doit(deep=False), result)
            self.assertEqual(expr.doit(deep=True), result)
            self.assertEqual(sympy.simplify(expr), result)
            self.assertEqual(shape_env.simplify(expr), result)
            self.assertEqual(shape_env.evaluate_expr(expr), result)

    def test_floordiv_assumptions(self):
        # [FUNCTION DOC] For every pair of integer symbols, checks FloorDiv construction and its
        # assumptions: complex-typed operands are expected to raise TypeError with a specific
        # message (no symbol in the list is complex, so that branch is not taken); otherwise the
        # result is integer and real when both are integers or the operands are the identical
        # symbol, else real with unknown integer-ness.
        cases = (
            sympy.Symbol("i1", integer=True),
            sympy.Symbol("i2", integer=True),
        )

        for base, divisor in itertools.product(cases, repeat=2):

            def op():
                # [FUNCTION DOC] HELPER: Builds the FloorDiv of the enclosing loop's base
                # and divisor.
                return FloorDiv(base, divisor)

            def is_complex(x):
                # [FUNCTION DOC] HELPER: True if the sympy value is known non-integer, non-real
                # and complex.
                return x.is_integer is False and x.is_real is False and x.is_complex

            if is_complex(base) or is_complex(divisor):
                self.assertRaisesRegex(
                    TypeError,
                    (
                        r"unsupported operand type\(s\) for //: 'Symbol' and 'Symbol',"
                        r" expected integer or real"
                    ),
                    op,
                )
                continue

            op = op()

            # In regular Python, x//x == 1.0 if x is a float, but FloorDiv
            # always returns an integer 1 when both args are the same object.
            # This even works for Symbols with no assumptions specified.
            if base is divisor:
                self.assertTrue(op.is_integer)
                self.assertTrue(op.is_real)
            elif base.is_integer and divisor.is_integer:
                self.assertTrue(op.is_integer)
                self.assertTrue(op.is_real)
            else:
                self.assertEqual(op.is_integer, None)
                self.assertTrue(op.is_real)


class TestDimConstraints(TestCase):
    # [CLASS DOC] Tests for DimConstraints, the dynamic-dimension constraint solver: simplifying
    # max expressions, congruence reduction, inequality reduction, printing errors and a large
    # end-to-end solve.
    @skipIfTorchDynamo("mark_dynamic not supported")
    def test_simplify_max_1_0(self):
        # [FUNCTION DOC] Marks a tensor dimension dynamic with bounds 5 to 20, compiles a function
        # that asserts statically_known_true on a max expression for argument values 0 and 1 and
        # branches on it; checks both calls give the multiplied-by-400 result without a constraint
        # violation error. Skipped under TorchDynamo.
        x = torch.rand(10)
        torch._dynamo.mark_dynamic(x, 0, max=20, min=5)

        @torch.compile(fullgraph=True)
        def func(x, v):
            # [FUNCTION DOC] HELPER: Raises AssertionError if v is 0 or 1 and the max expression is
            # not statically known equal to the half-size expression; then returns x times 400 if
            # the comparison holds, else x times 1000.
            # test that statically_known_true
            if (v == 0 or v == 1) and not statically_known_true(
                max(v, (-1 + x.size()[0] // 2)) == (-1 + x.size()[0] // 2)
            ):
                raise AssertionError("error")

            if max(v, (-1 + x.size()[0] // 2)) == (-1 + x.size()[0] // 2):
                return x * 400
            else:
                return (x * 10) * 100

        # testing that this does not throw constraint violation error.
        self.assertEqual(func(x, 1), x * 400)
        self.assertEqual(func(x, 0), x * 400)

    def test_dim_constraints_reduce_congruences_simple(self):
        # [FUNCTION DOC] Fills the congruence set of a symbol with several modular constraints and
        # checks the reduction merges them into the single congruence modulo 64 with offset 32.
        from sympy import Symbol

        s = Symbol("s", positive=True, integer=True)
        dim_constraints = DimConstraints({}, {}, set(), {})
        dim_constraints._congruences[s] = {
            (s / 2) % 2,
            (s / 2) % 8,
            (s / 2) % 4,
            s % 2,
            ((s / 16) + 2) % 4,
        }
        congruences = dim_constraints._reduce_congruences()
        self.assertEqual(congruences[s], {(s + 32) % 64})

    def test_dim_constraints_reduce_inequalities_simple(self):
        # [FUNCTION DOC] Checks that sympy inequality reduction of a set of bounds and not-equal
        # constraints gives the half-open interval from 4 to 16, and gives the set containing 8
        # once an equality is added.
        from sympy import Eq, Interval, Ne, Symbol
        from sympy.solvers.inequalities import reduce_inequalities

        s = Symbol("s", positive=True, integer=True)
        exprs = {
            s >= 2,
            Ne(8 * s, 16),
            Ne(s / 2, 1),
            Ne(16 * s, 32),
            s < 16,
            Ne(s, 2),
            s / 2 < 16,
            s / 2 > 1,
            s / 2 >= 2,
            Ne(3 * s / 2, 3),
        }
        solution = reduce_inequalities(exprs, s).as_set()
        self.assertEqual(solution, Interval.Ropen(4, 16))

        exprs.add(Eq(s / 2, 4))
        solution = reduce_inequalities(exprs, s).as_set()
        self.assertEqual(solution, {8})

    def test_dim_constraints_reduce_inequalities_error(self):
        # [FUNCTION DOC] Reduces cubic inequalities to a solution and prints it with the dynamic
        # dimension printer; expects AssertionError about an unknown symbol created by the
        # constraints solver.
        from collections import defaultdict

        from sympy import Symbol
        from sympy.solvers.inequalities import reduce_inequalities

        from torch._dynamo.source import (
            LocalSource,
            TensorProperty,
            TensorPropertySource,
        )
        from torch.fx.experimental.symbolic_shapes import DynamicDimConstraintPrinter

        s0 = Symbol("s0", positive=True, integer=True)
        exprs = {
            4 * s0**3 - 4 * s0**2 + s0 <= 2147483647,
            s0 >= 2,
            s0**3 <= 2147483647,
            s0 <= 2147483647,
        }
        answer = reduce_inequalities(exprs, s0)

        symbol_to_source = defaultdict(list)
        symbol_to_source[s0].append(
            TensorPropertySource(
                base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=0
            )
        )
        dcp = DynamicDimConstraintPrinter(symbol_to_source, {})
        with self.assertRaisesRegex(
            AssertionError,
            "Unknown symbol.*created by constraints solver",
        ):
            dcp.doprint(answer)

    def test_dim_constraints_solve_full(self):
        # [FUNCTION DOC] End-to-end solve: builds a source table for four symbols covering many
        # tensor size sources, registers equalities (including constants) and a long list of
        # inequality, not-equal and modular constraints involving nested floor divisions, runs the
        # solver, and checks the exact sets of static results (fixed sizes) and dynamic results
        # (lower bound and equalities for the remaining symbol).
        from sympy import Eq, Integer, Ne, Symbol

        from torch._dynamo.source import (
            LocalSource,
            TensorProperty,
            TensorPropertySource,
        )

        src0 = TensorPropertySource(
            base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=0
        )
        src2 = TensorPropertySource(
            base=LocalSource(local_name="b"), prop=TensorProperty.SIZE, idx=0
        )
        src3 = TensorPropertySource(
            base=LocalSource(local_name="c"), prop=TensorProperty.SIZE, idx=0
        )
        src4 = TensorPropertySource(
            base=LocalSource(local_name="d"), prop=TensorProperty.SIZE, idx=0
        )

        src1 = TensorPropertySource(
            base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=2
        )
        src7 = TensorPropertySource(
            base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=3
        )

        src5 = TensorPropertySource(
            base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=1
        )
        src8 = TensorPropertySource(
            base=LocalSource(local_name="b"), prop=TensorProperty.SIZE, idx=1
        )

        src6 = TensorPropertySource(
            base=LocalSource(local_name="c"), prop=TensorProperty.SIZE, idx=1
        )
        src9 = TensorPropertySource(
            base=LocalSource(local_name="d"), prop=TensorProperty.SIZE, idx=1
        )
        src10 = TensorPropertySource(
            base=LocalSource(local_name="e"), prop=TensorProperty.SIZE, idx=1
        )

        src11 = TensorPropertySource(
            base=LocalSource(local_name="f"), prop=TensorProperty.SIZE, idx=1
        )
        src12 = TensorPropertySource(
            base=LocalSource(local_name="b"), prop=TensorProperty.SIZE, idx=2
        )

        s0 = Symbol("s0", positive=True, integer=True)
        s1 = Symbol("s1", positive=True, integer=True)
        s5 = Symbol("s5", positive=True, integer=True)
        s6 = Symbol("s6", positive=True, integer=True)
        symbol_to_source = {
            s0: [src0, src2, src3, src4],
            s1: [src1, src7],
            s5: [src5, src8],
            s6: [src6, src9, src10],
        }
        var_to_val = {s0: 8, s1: 96, s5: 22, s6: 21}
        marked_dynamic = {s0, s1, s5, s6}
        dim_constraints = DimConstraints(
            symbol_to_source, var_to_val, marked_dynamic, {}
        )
        dim_constraints.add_equality(src2, s0)
        dim_constraints.add_equality(src3, s0)
        dim_constraints.add_equality(src4, s0)
        dim_constraints.add_equality(src7, s1)
        dim_constraints.add_equality(src8, s5)
        dim_constraints.add_equality(src9, s6)
        dim_constraints.add_equality(src10, s6)
        dim_constraints.add_equality(src11, Integer(1))
        dim_constraints.add_equality(src12, Integer(3))

        dim_constraints.add(s1**2 <= 2147483647)
        dim_constraints.add(32 * s1**2 <= 2147483647)
        dim_constraints.add(s0 < 16)
        dim_constraints.add(Eq(Mod(s1, 2), 0))
        dim_constraints.add(Ne(FloorDiv(s1, 2), 1))
        dim_constraints.add(Ne((FloorDiv(s1, 2)) ** 2, 1))
        dim_constraints.add(32 * (FloorDiv(s1, 2)) ** 2 <= 2147483647)
        dim_constraints.add((FloorDiv(s1, 2)) ** 2 > 1)
        dim_constraints.add(Ne(FloorDiv(s1, 2), 1))
        dim_constraints.add(
            64 * (FloorDiv((FloorDiv(s1, 2) - 1), 2)) ** 2
            + 128 * (FloorDiv((FloorDiv(s1, 2) - 1), 2))
            + 64
            <= 2147483647
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 2) + 1, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 2)) ** 2
                + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 2))
                + 1,
                1,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 2) + 1, 1))
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 2)) ** 2
            + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 2))
            + 1
            > 1
        )
        dim_constraints.add(
            128 * (FloorDiv((FloorDiv(s1, 2) - 1), 4)) ** 2
            + 256 * (FloorDiv((FloorDiv(s1, 2) - 1), 4))
            + 128
            <= 2147483647
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 4) + 1, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 4)) ** 2
                + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 4))
                + 1,
                1,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 4) + 1, 1))
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 4)) ** 2
            + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 4))
            + 1
            > 1
        )
        dim_constraints.add(
            256 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            + 512 * (FloorDiv((FloorDiv(s1, 2) - 1), 8))
            + 256
            <= 2147483647
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) + 1, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 8))
                + 1,
                1,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) + 1, 1))
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 8))
            + 1
            > 1
        )
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) + 1 >= 3)
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            <= 2147483647
        )
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1 >= 0)
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1 >= 1)
        dim_constraints.add(
            Ne(
                60 * s0 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * s0 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * s0,
                0,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 1,
                1,
            )
        )
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 1,
                0,
            )
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 1
            >= 0
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, 0))
        dim_constraints.add(
            1
            < 60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, -1))
        dim_constraints.add(
            Ne(
                60 * s0 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * s0 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * s0,
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
            )
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            > 0
        )
        dim_constraints.add(
            Eq(
                60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2 * (Mod(s0, 2))
                - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8) * Mod(s0, 2)
                + 60 * (Mod(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            Ne(
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
                0,
            )
        )
        dim_constraints.add(
            Ne(
                60
                * (FloorDiv(s0, 2))
                * (FloorDiv(s0, (FloorDiv(s0, 2))))
                * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120
                * FloorDiv(s0, 2)
                * FloorDiv(s0, (FloorDiv(s0, 2)))
                * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * (FloorDiv(s0, 2)) * (FloorDiv(s0, (FloorDiv(s0, 2)))),
                0,
            )
        )
        dim_constraints.add(Ne(FloorDiv(s0, 2), 1))
        dim_constraints.add(
            Ne(
                60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60,
                0,
            )
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            >= 0
        )
        dim_constraints.add(
            1
            < 60
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, (FloorDiv(s0, 2))) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, (FloorDiv(s0, 2))))
        )
        dim_constraints.add(Ne(16 * s0, 32))
        dim_constraints.add(Eq(16 * (Mod(s0, 2)), 0))
        dim_constraints.add(Ne(16 * s0, 32))
        dim_constraints.add(Eq(16 * (Mod(s0, 2)), 0))
        dim_constraints.add(FloorDiv(s0, 2) >= 2)
        dim_constraints.add(Ne(FloorDiv(s0, 2), 1))
        dim_constraints.add(1 < FloorDiv(s0, 2))
        dim_constraints.add(Ne(s0, 2))
        dim_constraints.add(
            60
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, (FloorDiv(s0, 2))) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, (FloorDiv(s0, 2))))
            >= 60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
        )
        dim_constraints.add(
            60
            * (FloorDiv(s0, 2))
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120
            * FloorDiv(s0, 2)
            * FloorDiv(s0, (FloorDiv(s0, 2)))
            * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, 2)) * (FloorDiv(s0, (FloorDiv(s0, 2))))
            > 0
        )
        dim_constraints.add(
            Ne(
                60
                * (FloorDiv(s0, 2))
                * (FloorDiv(s0, (FloorDiv(s0, 2))))
                * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120
                * FloorDiv(s0, 2)
                * FloorDiv(s0, (FloorDiv(s0, 2)))
                * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * (FloorDiv(s0, 2)) * (FloorDiv(s0, (FloorDiv(s0, 2)))),
                3 * (FloorDiv(s0, 2)) * (FloorDiv(s0, (FloorDiv(s0, 2)))),
            )
        )
        dim_constraints.add(
            Ne(
                20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 20,
                0,
            )
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            >= 0
        )
        dim_constraints.add(
            Ne(
                20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 20,
                20,
            )
        )
        dim_constraints.add(
            Ne(
                20
                * (
                    Mod(
                        1,
                        (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 1,
                    )
                ),
                0,
            )
        )
        dim_constraints.add(
            Ne(
                20
                * (FloorDiv((FloorDiv(s1, 2) - 1), 8))
                * (
                    Mod(
                        1,
                        (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                        - 2
                        * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                        + 1 / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1),
                    )
                )
                - 20
                * Mod(
                    1,
                    (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                    / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                    - 2
                    * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                    / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                    + 1 / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1),
                ),
                0,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, 1))
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 1
            >= 1
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            >= 0
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            >= 1
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            >= 2
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            > 1
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            < 60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
        )
        dim_constraints.add(
            Ne(
                60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60,
                60,
            )
        )
        dim_constraints.add(
            Ne(
                FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1,
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 1,
            )
        )
        dim_constraints.add(
            Eq(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8))
                * (
                    Mod(
                        (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                        - 2
                        * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                        + 1 / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1),
                        1,
                    )
                )
                - Mod(
                    (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                    / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                    - 2
                    * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                    / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                    + 1 / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1),
                    1,
                ),
                0,
            )
        )
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 1,
                FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1,
            )
        )
        dim_constraints.add(Ne(8 * s0, 16))
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            >= (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 1
        )
        dim_constraints.add(
            60
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, (FloorDiv(s0, 2))) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, (FloorDiv(s0, 2))))
            <= 2147483647
        )
        dim_constraints.add(
            90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 90
            <= 2147483647
        )
        dim_constraints.add(FloorDiv(s0, 2) < 16)
        dim_constraints.add(FloorDiv(s0, 2) > 1)
        dim_constraints.add(
            Ne(
                90 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 180 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 90 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < 90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 90
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 1
            > 1
        )
        dim_constraints.add(
            60
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, (FloorDiv(s0, 2))) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, (FloorDiv(s0, 2))))
            > 1
        )
        dim_constraints.add(
            Ne(
                60 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 90
            > 1
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            > 1
        )
        dim_constraints.add(
            Ne(
                60 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * (FloorDiv(s0, 2)),
                3 * (FloorDiv(s0, 2)),
            )
        )
        dim_constraints.add(
            60 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, 2))
            > 0
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            > 0
        )
        dim_constraints.add(
            Ne(
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
                0,
            )
        )
        dim_constraints.add(
            1
            < 120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
        )
        dim_constraints.add(
            Ne(
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
                6,
            )
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            > 0
        )
        dim_constraints.add(
            Ne(
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
                0,
            )
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            <= 2147483647
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            <= 20480
        )
        dim_constraints.add(
            Ne(
                90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 90,
                0,
            )
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            > 1
        )
        dim_constraints.add(
            90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 90
            <= 20480
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            <= 20480
        )
        dim_constraints.add(
            Ne(
                240 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 480 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 240,
                0,
            )
        )
        dim_constraints.add(Eq(6 * s5, 132))
        dim_constraints.add(Eq(4, FloorDiv(s0, 2)))
        dim_constraints.add(Eq(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, 4))
        dim_constraints.add(
            Ne(
                64 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 128 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 64 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < 64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 128 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 64
        )
        dim_constraints.add(
            64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 128 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 64
            <= 2147483647
        )
        dim_constraints.add(
            64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 128 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 64
            > 1
        )
        dim_constraints.add(
            62 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 124 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 62
            <= 2147483647
        )
        dim_constraints.add(
            Ne(
                62 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 124 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 62 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < 62 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 124 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 62
        )
        dim_constraints.add(Ne(3 * (FloorDiv(s0, 2)), 3))
        dim_constraints.add(Ne(3 * (FloorDiv(s0, 2)), 3))
        dim_constraints.add(Eq(FloorDiv(s0, 2), 4))
        dim_constraints.add(Eq(4, FloorDiv(s0, 2)))
        dim_constraints.add(Eq(FloorDiv(s0, 2), 4))
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1 >= 3)
        dim_constraints.add(
            64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 384 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 576
            <= 2147483647
        )
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3 >= 0)
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3 >= 1)
        dim_constraints.add(
            Ne(
                64 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 384 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 576 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 9,
                1,
            )
        )
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 9,
                0,
            )
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
            >= 0
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3, 0))
        dim_constraints.add(
            1
            < 64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 384 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 576
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3, 1))
        dim_constraints.add(
            Ne(
                64 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 384 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 576 * (FloorDiv(s0, 2)),
                256,
            )
        )
        dim_constraints.add(
            Eq(
                64
                * (
                    Mod(
                        (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 6 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 9 * (FloorDiv(s0, 2)),
                        4,
                    )
                ),
                0,
            )
        )
        dim_constraints.add(
            Eq(
                FloorDiv(s0, 2),
                FloorDiv(
                    (
                        (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 6 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 9 * (FloorDiv(s0, 2))
                    ),
                    4,
                ),
            )
        )
        dim_constraints.add(
            Eq(
                FloorDiv(
                    (
                        (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 6 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 9 * (FloorDiv(s0, 2))
                    ),
                    4,
                ),
                FloorDiv(s0, 2),
            )
        )
        dim_constraints.add(
            Ne(64 * (Mod(FloorDiv((FloorDiv(s1, 2) - 1), 8) + 1, 4)), 0)
        )
        dim_constraints.add(
            Eq(
                64
                * (
                    Mod(
                        (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 1,
                        4,
                    )
                ),
                0,
            )
        )
        dim_constraints.add(
            64 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 384 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 576 * (FloorDiv(s0, 2))
            > 0
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
            >= 1
        )
        dim_constraints.add(
            Eq(
                64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 384 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 576,
                256,
            )
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 360 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 540
            <= 2147483647
        )
        dim_constraints.add(
            Ne(
                60 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 360 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 540 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < 60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 360 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 540
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
            <= 2147483647
        )
        dim_constraints.add(
            Ne(
                (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 6 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 9 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
            > 1
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 360 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 540
            > 1
        )
        dim_constraints.add(s0 >= 2)
        dim_constraints.add(s1 >= 2)
        dim_constraints.add(s6 >= 2)
        dim_constraints.add(s5 >= 2)

        dim_constraints.solve()
        self.assertEqual(
            dim_constraints._static_results,
            {
                "L['c'].size()[0] == 8",
                "L['d'].size()[0] == 8",
                "L['a'].size()[2] == 96",
                "L['f'].size()[1] == 1",
                "L['a'].size()[3] == 96",
                "L['b'].size()[2] == 3",
                "L['b'].size()[1] == 22",
                "L['b'].size()[0] == 8",
                "L['a'].size()[1] == 22",
                "L['a'].size()[0] == 8",
            },
        )
        self.assertEqual(
            dim_constraints._dynamic_results,
            {
                "2 <= L['c'].size()[1]",
                "L['d'].size()[1] == L['c'].size()[1]",
                "L['e'].size()[1] == L['c'].size()[1]",
            },
        )


class TestGuardsExpressions(TestCase):
    # [CLASS DOC] Tests for producing and evaluating guard expression strings from a ShapeEnv (as
    # used by the inductor FX graph cache) and for guard_or_true / guard_or_false behavior and
    # recompilation under torch.compile. Has no state of its own beyond TestCase.
    """
    Tests the guards-related methods used by the inductor FX graph cache.
    """

    def test_guards_gt_lt(self):
        # [FUNCTION DOC] Records guards from comparisons on a symbolic int, produces the combined
        # guard expression, and checks it evaluates true for the original hint and false for hints
        # that violate either bound.
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 6)
        s1 = create_symint(shape_env, 7)
        s2 = create_symint(shape_env, 5)

        guard_int(sym_int(s0 > 5))
        guard_int(sym_int(s0 < 7))

        guards = shape_env.produce_guards_expression([s0])

        self.assertTrue(
            shape_env.evaluate_guards_expression(guards, [guarding_hint_or_throw(s0)])
        )
        self.assertFalse(
            shape_env.evaluate_guards_expression(guards, [guarding_hint_or_throw(s1)])
        )
        self.assertFalse(
            shape_env.evaluate_guards_expression(guards, [guarding_hint_or_throw(s2)])
        )

    def test_guards_float_print(self):
        # [FUNCTION DOC] Records a guard on a float division and checks the produced expression
        # text can be evaluated and holds for the original hint.
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 3)
        guard_bool(2 / s0 == 2 / 3)
        guards = shape_env.produce_guards_expression([s0])
        self.assertTrue(
            shape_env.evaluate_guards_expression(guards, [guarding_hint_or_throw(s0)])
        )

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_guard_or_true(self):
        # [FUNCTION DOC] Tests guard_or_true with a scalar item: eager gives each branch by value;
        # compiled with unbacked data the true branch is always taken; a statically true
        # self-inequality makes the false branch always taken; under backed-size-oblivious mode the
        # compiled function takes the true branch for a size of 9 where eager takes the other.
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        def func(a, b):
            # [FUNCTION DOC] HELPER: Reads a scalar item from a; returns b times 10 when the
            # guard_or_true of equality to one holds, else b times 20.
            x = a.item()
            if guard_or_true(x == 1):
                return b * 10
            else:
                return b * 20

        # eager.
        self.assertEqual(func(torch.tensor([1]), torch.tensor([1])), torch.tensor([10]))
        self.assertEqual(func(torch.tensor([2]), torch.tensor([1])), torch.tensor([20]))

        # compile with unbacked.
        unbacked_func = torch.compile(func, dynamic=True, fullgraph=True)
        a = torch.tensor([1])
        b = torch.tensor([1])
        unbacked_func(a, b)

        # always return b*10
        self.assertEqual(
            unbacked_func(torch.tensor([1]), torch.tensor([1])), torch.tensor([10])
        )
        self.assertEqual(
            unbacked_func(torch.tensor([2]), torch.tensor([1])), torch.tensor([10])
        )

        # Test that statically known true works.
        def func2(a, b):
            # [FUNCTION DOC] HELPER: Reads a scalar item from a; returns b times 10 when
            # guard_or_true of the item differing from itself holds, else b times 20.
            x = a.item()
            if guard_or_true(x != x):
                return b * 10
            else:
                return b * 20

        unbacked_func2 = torch.compile(func2, dynamic=True, fullgraph=True)
        a = torch.tensor([1])
        b = torch.tensor([1])
        unbacked_func2(a, b)
        # always return b*20
        self.assertEqual(
            unbacked_func2(torch.tensor([1]), torch.tensor([1])), torch.tensor([20])
        )
        self.assertEqual(
            unbacked_func2(torch.tensor([2]), torch.tensor([1])), torch.tensor([20])
        )

        # Test backed_size_oblivious
        with torch.fx.experimental._config.patch("backed_size_oblivious", True):

            def func3(a, b):
                # [FUNCTION DOC] HELPER: Returns b times 10 when guard_or_true says the first size
                # of a is not 9, else b times 20.
                if guard_or_true(a.size()[0] != 9):
                    return b * 10
                else:
                    return b * 20

            compiled = torch.compile(func3, dynamic=True, fullgraph=True)
            a = torch.rand(9, 2)
            b = torch.rand(3, 4)

            self.assertEqual(func3(a, b), b * 20)
            self.assertEqual(compiled(a, b), b * 10)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_guard_or_false(self):
        # [FUNCTION DOC] Tests guard_or_false with a scalar item: eager gives each branch by value;
        # compiled with unbacked data the false branch is always taken; a statically true
        # self-equality makes the true branch always taken; under backed-size-oblivious mode the
        # compiled function takes the false branch for a size of 9 where eager takes the true one.
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        def func(a, b):
            # [FUNCTION DOC] HELPER: Reads a scalar item from a; returns b times 10 when
            # guard_or_false of equality to one holds, else b times 20.
            x = a.item()
            if guard_or_false(x == 1):
                return b * 10
            else:
                return b * 20

        # eager.
        self.assertEqual(func(torch.tensor([1]), torch.tensor([1])), torch.tensor([10]))
        self.assertEqual(func(torch.tensor([2]), torch.tensor([1])), torch.tensor([20]))

        # compile with unbacked.
        unbacked_func = torch.compile(func, dynamic=True, fullgraph=True)
        a = torch.tensor([1])
        b = torch.tensor([1])
        unbacked_func(a, b)

        # always return b*20
        self.assertEqual(
            unbacked_func(torch.tensor([1]), torch.tensor([1])), torch.tensor([20])
        )
        self.assertEqual(
            unbacked_func(torch.tensor([2]), torch.tensor([1])), torch.tensor([20])
        )

        # Test that statically known true works.
        def func2(a, b):
            # [FUNCTION DOC] HELPER: Reads a scalar item from a; returns b times 10 when
            # guard_or_false of the item equal to itself holds, else b times 20.
            x = a.item()
            if guard_or_false(x == x):
                return b * 10
            else:
                return b * 20

        unbacked_func2 = torch.compile(func2, dynamic=True, fullgraph=True)
        a = torch.tensor([1])
        b = torch.tensor([1])
        unbacked_func2(a, b)
        # always return b*10
        self.assertEqual(
            unbacked_func2(torch.tensor([1]), torch.tensor([1])), torch.tensor([10])
        )
        self.assertEqual(
            unbacked_func2(torch.tensor([2]), torch.tensor([1])), torch.tensor([10])
        )

        # Test backed_size_oblivious
        with torch.fx.experimental._config.patch("backed_size_oblivious", True):

            def func3(a, b):
                # [FUNCTION DOC] HELPER: Returns b times 10 when guard_or_false says the first size
                # of a is 9, else b times 20.
                if guard_or_false(a.size()[0] == 9):
                    return b * 10
                else:
                    return b * 20

            compiled = torch.compile(func3, dynamic=True, fullgraph=True)
            a = torch.rand(9, 2)
            b = torch.rand(3, 4)

            self.assertEqual(func3(a, b), b * 10)
            self.assertEqual(compiled(a, b), b * 20)

    def test_guards_float_div(self):
        # [FUNCTION DOC] Records a guard on a truncated float division, and checks the guard string
        # contains truncation and float conversion, is true for the original hint and false for
        # another value.
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 8)
        s1 = create_symint(shape_env, 7)

        guard_int(sym_int(s0 / 2.0))
        guards = shape_env.produce_guards_expression([s0])

        self.assertIn("math.trunc(", guards)
        self.assertIn("float(", guards)
        self.assertTrue(
            shape_env.evaluate_guards_expression(guards, [guarding_hint_or_throw(s0)])
        )
        self.assertFalse(
            shape_env.evaluate_guards_expression(guards, [guarding_hint_or_throw(s1)])
        )

    @unittest.skipIf(
        TEST_XPU, "Skipped on XPU"
    )  # https://github.com/intel/torch-xpu-ops/issues/2169"
    @skipIfTorchDynamo("Attempt to trace generator")
    @torch.fx.experimental._config.patch("use_duck_shape", False)
    def test_size_comparison_no_recompile(self):
        # [FUNCTION DOC] Compiles a function that compares the whole sizes of two tensors with duck
        # shaping disabled, calls it twice with different shapes and checks only one compilation
        # happened. Skipped on XPU and under TorchDynamo.
        """
        Test that size comparisons don't cause recompilation.
        When comparing x.size() == b.size() with different sizes,
        the compiled function should only compile once.
        We should not guard in sizes of the inner elements.
        """
        cnt = CompileCounter()

        @torch.compile(fullgraph=True, dynamic=True, backend=cnt)
        def f(x, b):
            # [FUNCTION DOC] HELPER: Returns x if its size equals the size of b, else x doubled.
            if x.size() == b.size():
                return x
            return x * 2

        # First call: shapes differ (1, 2) vs (2, 4, 9), so if branch is False
        f(torch.rand(10, 2), torch.rand(20, 4, 9))

        # Second call: shapes differ again (1, 2) vs (1, 4, 9), so if branch is False
        f(torch.rand(10, 2), torch.rand(10, 4, 9))

        # Should only compile once despite different input shapes
        self.assertEqual(
            cnt.frame_count,
            1,
            lambda msg: f"{msg}\nExpected 1 compilation, got {cnt.frame_count}. "
            f"Size comparison should not cause recompilation.",
        )

    @skipIfTorchDynamo("Test uses torch.compile")
    def test_python_mod_padding_no_recompile(self):
        # [FUNCTION DOC] Compiles a function that pads a tensor to a multiple of four and slices
        # the padding off before a view; checks two different input sizes compile once, then after
        # a dynamo reset, with unbacked dimensions, checks results and a total of two frames.
        """
        Test that padding with PythonMod and slicing back to original size
        doesn't cause guards or recompilation.
        """
        cnt = CompileCounter()

        @torch.compile(backend=cnt, dynamic=True)
        def func(x):
            # [FUNCTION DOC] HELPER: Allocates an empty tensor padded along the first dimension up
            # to a multiple of four, multiplies it by 100, slices back to the input's first size,
            # and returns a view with the input's size.
            # Padding to align to multiple of 4
            padding = (-x.size()[0]) % 4
            aligned_size = x.size()[0] + padding
            tensor = torch.empty(aligned_size, x.size()[1])
            # Do some work on padded tensor
            tensor = tensor * 100
            # Remove padding with slicing
            remove_padding = tensor[0 : x.size()[0], ...]
            # View should work without guards since shapes match
            return remove_padding.view(x.size())

        # First call we pass something does not need padding
        result1 = func(torch.rand(4, 10))
        self.assertEqual(result1.shape, (4, 10))

        # Second call with different size needs padding
        result1 = func(torch.rand(10, 12))
        self.assertEqual(result1.shape, (10, 12))

        # Should only compile once despite different input shapes
        self.assertEqual(
            cnt.frame_count,
            1,
            lambda msg: f"{msg}\nExpected 1 compilation, got {cnt.frame_count}. "
            f"PythonMod padding should not cause recompilation.",
        )

        # test with unbacked.
        torch._dynamo.reset()

        a = torch.rand(10, 10)
        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(a, 1)
        result1 = func(a)
        self.assertEqual(result1.shape, (10, 10))

        result1 = func(torch.rand(4, 10))
        self.assertEqual(result1.shape, (4, 10))

        self.assertEqual(cnt.frame_count, 2)

    def test_remove_symbols_without_guarding(self):
        # [FUNCTION DOC] Checks the partitioner helper that replaces symbolic sizes and strides of
        # a fake tensor with their hint values: a fake tensor printing symbolic sizes and strides
        # prints concrete (5, 8) and strides (8, 1) after cleaning.
        from torch._functorch.partitioners import _remove_symbols_without_guarding

        shape_env = ShapeEnv()

        x = create_fake_tensor_with_dynamic_size(
            torch.randn(5, 8),
            shape_env,
            dynamic_sizes=[
                DimDynamic.DYNAMIC,
                DimDynamic.DYNAMIC,
            ],
            dynamic_strides=[
                DimDynamic.INFER_STRIDE,
                DimDynamic.INFER_STRIDE,
            ],
        )

        self.assertEqual(f"{x.stride()}", "(s49, 1)")
        self.assertEqual(f"{x.shape}", "torch.Size([s26, s49])")

        x_clean = _remove_symbols_without_guarding(x, 4096)

        self.assertEqual(f"{x_clean.stride()}", "(8, 1)")
        self.assertEqual(f"{x_clean.shape}", "torch.Size([5, 8])")


def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
    # [FUNCTION DOC] HELPER: Inductor pre-pass hook used by a test. Scans graph nodes for the one
    # named arg3_1 and raises AssertionError if its fake value's first size is not 2. Returns the
    # graph unchanged.
    for node in graph.nodes:
        if node.name == "arg3_1":
            if node.meta["val"].size()[0] != 2:
                raise AssertionError(
                    f"expected size()[0] == 2, got {node.meta['val'].size()[0]}"
                )
    return graph


class TestUnbacked(TestCase):
    # [CLASS DOC] Tests of unbacked symbol handling: rebinding, deferred runtime asserts under
    # torch.compile for several backends, unbacked input marking, division simplification with
    # equal sizes, ignorable fresh unbacked symbols and related regressions.
    def test_rebind_unbacked_to_symbolic_expression(self):
        # [FUNCTION DOC] Creates an unbacked binding annotation on a graph node and calls
        # rebind_unbacked with a symbolic expression, then checks the old unbacked symbol is
        # replaced by that expression in the ShapeEnv replacements.
        shape_env = ShapeEnv()
        fake_mode = torch._subclasses.FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=shape_env
        )
        graph = torch.fx.Graph()
        node = graph.call_function(operator.add, args=(1, 1))

        with fake_mode:
            old = shape_env.create_unbacked_symint()
            base = shape_env.create_unbacked_symint()
            result = (base + 1) // 2

        old_expr = old.node.expr
        result_expr = result.node.expr
        node.meta["unbacked_bindings"] = {old_expr: ()}

        rebind_unbacked(shape_env, node, result)

        self.assertEqual(shape_env.replacements[old_expr], result_expr)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/156135")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_deferred_neq_assert(self, backend):
        # [FUNCTION DOC] Parametrized over inductor and eager backends. A compiled function checks
        # an item is not 5; passing 100 works, passing 5 raises RuntimeError.
        @torch.compile(fullgraph=True, backend=backend)
        def func(a):
            # [FUNCTION DOC] HELPER: Asserts via torch check that the scalar item is not 5 and
            # returns the item times ten.
            torch._check(a.item() != 5)
            return a.item() * 10

        func(torch.tensor([100]))

        with self.assertRaises(RuntimeError):
            func(torch.tensor([5]))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_deferred_symfloat_assert(self, backend):
        # [FUNCTION DOC] Parametrized over inductor and eager. A compiled function checks a float
        # item times two equals 11; 5.5 passes and returns 55, while 1.8 raises RuntimeError whose
        # message matches the failed float equality expression.
        @torch.compile(fullgraph=True, backend=backend)
        def func(a, b):
            # [FUNCTION DOC] HELPER: Checks the float item of b doubled equals 11 and returns b
            # times 10.
            torch._check(b.item() * 2 == 11)
            return b * 10

        with fresh_cache():
            self.assertEqual(
                func(torch.tensor([100]), torch.tensor([5.5])), torch.tensor([55.0])
            )
            with self.assertRaisesRegex(RuntimeError, r"Eq\(2\.0\*zuf\d+, 11\.0\)"):
                func(torch.tensor([5]), torch.tensor([1.8]))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_deferred_symfloat_eq_assert(self, backend):
        # [FUNCTION DOC] Parametrized over inductor and eager. A compiled function checks a float
        # item equals 5.5; 5.5 passes and returns a times ten, while 1.5 raises RuntimeError whose
        # message mentions a lower-bound float expression.
        @torch.compile(fullgraph=True, backend=backend)
        def func(a, b):
            # [FUNCTION DOC] HELPER: Checks the float item of b equals 5.5 and returns a times 10.
            torch._check(b.item() == 5.5)
            return a * 10

        with fresh_cache():
            self.assertEqual(
                func(torch.tensor([5]), torch.tensor([5.5])), torch.tensor([50])
            )
            with self.assertRaisesRegex(RuntimeError, r"zuf\d+ >= 5\.5"):
                func(torch.tensor([100]), torch.tensor([1.5]))

    # Test a situation where we generate a runtime assert i.e: u1==s1, then we specialize s1
    # later on to a constant.
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_post_specialize_runtime_assert1(self, backend):
        # [FUNCTION DOC] Parametrized over backends. A dynamic-shape compiled function asserts an
        # item plus both sizes sums to 102 and expects the first size to specialize to 2; passing
        # 50 works, passing 51 raises RuntimeError.
        @torch.compile(dynamic=True, backend=backend)
        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads y's item, checks item plus both sizes of x equals 102,
            # raises AssertionError if the first size is not 2, returns x times 10.
            u0 = y.item()
            s0 = x.size()[0]
            s1 = x.size()[1]
            torch._check(u0 + s0 + s1 == 102)
            if s0 != 2:
                raise AssertionError(f"expected s0 == 2, got {s0}")
            return x * 10

        func(torch.rand(2, 50), torch.tensor([50]))
        with self.assertRaises(RuntimeError):
            func(torch.rand(2, 50), torch.tensor([51]))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch(post_grad_custom_pre_pass=custom_pass)
    @parametrize("backend", ["inductor", "eager"])
    def test_post_specialize_runtime_assert2(self, backend):
        # [FUNCTION DOC] Same scenario as the first post-specialization test, but with the custom
        # inductor pre-pass installed which verifies the first size of arg3_1 is 2. Passing 50
        # works, passing 51 raises RuntimeError.
        @torch.compile(dynamic=True, backend=backend)
        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads y's item, checks item plus both sizes of x equals 102,
            # returns x times 10.
            u0 = y.item()
            s0 = x.size()[0]
            s1 = x.size()[1]
            torch._check(u0 + s0 + s1 == 102)
            return x * 10

        func(torch.rand(2, 50), torch.tensor([50]))
        with self.assertRaises(RuntimeError):
            func(torch.rand(2, 50), torch.tensor([51]))

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/156135")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_deferred_sym_or_assert(self, backend):
        # [FUNCTION DOC] Parametrized over backends. A compiled function checks that either of two
        # items equals 5; both calls with one item equal to 5 must pass.
        @torch.compile(fullgraph=True, backend=backend)
        def func(a, b):
            # [FUNCTION DOC] HELPER: Checks the logical or of two item equalities to 5 and returns
            # a's item times ten.
            torch._check(operator.or_(a.item() == 5, b.item() == 5))
            return a.item() * 10

        func(torch.tensor([5]), torch.tensor([100]))
        func(torch.tensor([100]), torch.tensor([5]))

    def test_has_free_symbols(self):
        # [FUNCTION DOC] Checks has_free_symbols returns false for constants and true for
        # expressions containing named symbols.
        self.assertFalse(has_free_symbols(sympy.S.true))
        self.assertFalse(has_free_symbols(sympy.Max(1, 10, evaluate=False)))

        self.assertFalse(has_free_symbols(sympy.sympify("1")))
        self.assertFalse(has_free_symbols(sympy.sympify("1.1")))
        self.assertTrue(has_free_symbols(sympy.sympify("a")))
        self.assertTrue(has_free_symbols(sympy.sympify("a*2")))
        self.assertTrue(has_free_symbols(sympy.sympify("a+b")))

    def test_iterate_exprs_dict(self):
        # [FUNCTION DOC] Checks the expression iterator handles dictionaries: a dict with a
        # concrete tensor value yields nothing, a dict with sympy keys yields both keys, free
        # symbols are collected from keys, and an empty dict yields nothing.
        """Test that _iterate_exprs handles dict values (e.g. from triton_kernel_wrapper_functional)."""
        a, b = sympy.symbols("a b")

        # dict with tensor values and string keys — should not crash
        t = torch.randn(3, 4)
        result = list(_iterate_exprs({"Out": t}))
        # concrete tensor has no symbolic exprs
        self.assertEqual(len(result), 0)

        # dict with sympy keys — should iterate over both keys and values
        result = list(_iterate_exprs({a: 1, b: 2}))
        self.assertEqual(len(result), 2)
        self.assertIn(a, result)
        self.assertIn(b, result)

        # free_symbols works on dicts with sympy keys
        self.assertEqual(free_symbols({a + b: 1}), {a, b})

        # empty dict
        self.assertEqual(list(_iterate_exprs({})), [])

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/156135")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_deferred_sym_eq_assert(self, backend):
        # [FUNCTION DOC] Parametrized over backends. A compiled function checks an item equals 5;
        # passing 5 works, passing 1 raises RuntimeError.
        @torch.compile(fullgraph=True, backend=backend)
        def func(a, b):
            # [FUNCTION DOC] HELPER: Checks the item of b equals 5 and returns a times 10.
            torch._check(b.item() == 5)
            return a * 10

        func(torch.tensor([5]), torch.tensor([5]))
        with self.assertRaises(RuntimeError):
            func(torch.tensor([100]), torch.tensor([1]))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_deferred_with_unbacked_input(self, backend):
        # [FUNCTION DOC] Parametrized over backends. Two inputs are marked unbacked in dim zero and
        # a compiled function checks their first sizes equal; a mismatching second call must raise
        # AssertionError or RuntimeError (inductor may raise its own assertion).
        @torch.compile(fullgraph=True, dynamic=True, backend=backend)
        def func(a, b):
            # [FUNCTION DOC] HELPER: Checks the first sizes of a and b are equal and returns a
            # times 10.
            torch._check(a.size()[0] == b.size()[0])
            return a * 10

        a = torch.rand(1, 1)
        b = torch.rand(1, 1)
        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        func(a, b)

        # inductor adds the check sometimes itself so it will be reflected
        # as AssertionError.
        with self.assertRaises((AssertionError, RuntimeError)):
            func(a, torch.rand(2, 1))

    @unittest.skipIf(
        TEST_WITH_ASAN
        or IS_LINUX
        or IS_MACOS
        or TEST_WITH_ROCM
        or TEST_WITH_SLOW
        or IS_WINDOWS,
        "https://github.com/pytorch/pytorch/issues/163953",
    )
    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_do_not_guard_unbacked_inputs(self):
        # [FUNCTION DOC] Compiles through inductor with both inputs fully unbacked, runs it, and
        # checks no symbolic shape guard appears in the guard log. Effectively disabled: the
        # skip condition is true on Linux, macOS, Windows and several other configs, so it only
        # runs elsewhere.
        @torch.compile(fullgraph=True, dynamic=True, backend="inductor")
        def func(a, b):
            # [FUNCTION DOC] HELPER: Expands a to the shape of b, discarding the result, and
            # returns a times 10.
            a.expand(b.shape)
            return a * 10

        a = torch.rand(1, 1)
        b = torch.rand(1, 1)

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(a, 1)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        torch._dynamo.decorators.mark_unbacked(b, 1)

        log_stream, ctx = logs_to_string("torch._dynamo.guards", "guards")
        with ctx():
            func(a, b)
            func(torch.rand(4, 5), torch.rand(4, 5))

        guards = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertFalse("SYMBOLIC_SHAPE_GUARD" in guards)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_div_unbacked_eq_input_tensors(self):
        # [FUNCTION DOC] Two unbacked-marked inputs are checked to have equal first sizes inside a
        # compiled function that divides sizes; compilation must succeed.
        @torch.compile(fullgraph=True)
        def func(a, b):
            # [FUNCTION DOC] HELPER: Checks first sizes of a and b are equal, then multiplies a by
            # 10 if the size ratio is 1 and by 20 if twice the size over the other equals 2.
            x = a.size()[0]
            y = b.size()[0]
            torch._check(x == y)
            if x // y == 1:
                a = a * 10
            if 2 * x // y == 2:
                a = a * 20
            return a

        a = torch.randn(10, 10)
        b = torch.randn(10, 20)

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        func(a, b)

    @torch.compiler.config.patch(unbacked_sources="L['x'],L['y']")
    def test_div_unbacked_eq_input_ints(self):
        # [FUNCTION DOC] With integer arguments treated as unbacked via config, a compiled function
        # checks equality and divides; compilation and a call with 10 and 10 must succeed.
        @torch.compile(fullgraph=True)
        def func(x, y):
            # [FUNCTION DOC] HELPER: Checks x equals y, then scales a fresh random tensor by 10 if
            # the integer quotient is 1 and by 20 if twice x over y is 2.
            a = torch.rand(1)
            torch._check(x == y)
            if x // y == 1:
                a = a * 10
            if 2 * x // y == 2:
                a = a * 20
            return a

        func(10, 10)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    @torch.compiler.config.patch(unbacked_sources="L['y']")
    def test_div_unbacked_eq_globals(self):
        # [FUNCTION DOC] Marks a global tensor dimension unbacked and sets a global int as
        # unbacked via config; a compiled function checks they are equal and divides; it must
        # compile and run.
        tensor = torch.rand(10, 44)
        y = 10

        @torch.compile(fullgraph=True, dynamic=True)
        def func():
            # [FUNCTION DOC] HELPER: Checks the global tensor's first size equals global y, then
            # scales a fresh random tensor by 10 if the integer quotient is 1 and by 20 if twice x
            # over y is 2.
            a = torch.rand(1)
            x = tensor.size()[0]
            torch._check(x == y)
            if x // y == 1:
                a = a * 10
            if 2 * x // y == 2:
                a = a * 20
            return a

        torch._dynamo.decorators.mark_unbacked(tensor, 0)
        func()

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_div_unbacked_eq_item(self):
        # [FUNCTION DOC] With scalar capture enabled, a compiled function reads two items, checks
        # they are equal and that quotients equal expected values, and branches on them; must
        # compile and run with items 1 and 1.
        @torch.compile(fullgraph=True)
        def func(a, b):
            # [FUNCTION DOC] HELPER: Reads items x and y from a and b, checks relations between
            # them, then scales a by 10 and 20 depending on the integer quotients.
            x = a.item()
            y = b.item()
            torch._check(x == y)
            # TODO we should not need those torch checks.
            torch._check(x // y == 1)
            torch._check(2 * x // y == 2)
            if x // y == 1:
                a = a * 10
            if 2 * x // y == 2:
                a = a * 20
            return a

        a = torch.tensor([1])
        b = torch.tensor([1])
        func(a, b)

    def test_ignorable_fresh_unbacked_symbols(self):
        # [FUNCTION DOC] Defines a scoped custom op whose fake kernel creates a dynamic size and
        # returns a clone; with the flag true the fresh symbol creation is wrapped in the ignore
        # context, with false it is not. The compiled flag-true function runs and returns its input
        # while the flag-false one raises an error about pending unbacked symbols not in outputs.
        # Test that symbols created during fake tensor tracing but not returned
        # can be marked as ignorable to avoid unbound symbol errors

        # Positive case: marking symbol as ignorable should work
        with torch.library._scoped_library("test_ignorable", "FRAGMENT") as lib:
            torch.library.define(
                "test_ignorable::custom_op",
                "(Tensor x, bool y) -> Tensor",
                lib=lib,
            )

            @torch.library.register_fake("test_ignorable::custom_op", lib=lib)
            def custom_op_fake(x, y):
                # [FUNCTION DOC] HELPER: Fake kernel: creates a dynamic size, inside the ignore
                # context when y is true, and returns a clone of x without using the new size.
                ctx = torch._custom_op.impl.get_ctx()
                # Create a symint during fake tensor tracing
                if y:
                    with ctx._shape_env.ignore_fresh_unbacked_symbols():
                        sz = ctx.new_dynamic_size()
                else:
                    sz = ctx.new_dynamic_size()
                # Return something else (not the symint we created)
                return x.clone()

            @torch.library.impl(
                "test_ignorable::custom_op",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def custom_op_impl(x, y):
                # [FUNCTION DOC] HELPER: Real kernel of the custom op; returns a clone of x.
                return x.clone()

            @torch.compile(fullgraph=True)
            def func_positive(x):
                # [FUNCTION DOC] HELPER: Calls the custom op with the ignore flag true.
                return torch.ops.test_ignorable.custom_op(x, True)

            @torch.compile(fullgraph=True)
            def func_negative(x):
                # [FUNCTION DOC] HELPER: Calls the custom op with the ignore flag false.
                return torch.ops.test_ignorable.custom_op(x, False)

            x = torch.randn(3, 4)

            # This should not raise an error about unbound symbols
            result = func_positive(x)
            self.assertEqual(result, x)

            # This should raise an error about unbound symbols
            with self.assertRaisesRegex(
                Exception, "Pending unbacked symbols.*not in returned outputs"
            ):
                func_negative(x)

    def test_meta_copy(self):
        # [FUNCTION DOC] Checks the meta implementation of copy_ accepts self and source tensors
        # with different unbacked first sizes under a fake mode without raising.
        """
        Test that meta_copy_ does not raise when self and src have different
        unbacked symint sizes.
        """
        from torch._meta_registrations import meta_copy_
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            u0 = shape_env.create_unbacked_symint()
            u1 = shape_env.create_unbacked_symint()

            self_tensor = torch.empty((u0, 256))
            src_tensor = torch.empty((u1, 256))

            meta_copy_(self_tensor, src_tensor)

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_symbool_propagate_real_tensors(self):
        # [FUNCTION DOC] Regression test: with real-tensor propagation on, a compiled function
        # branches on a boolean item. Calling it with a false item gives the else-branch result;
        # calling it later with a true item must raise RuntimeError about a failed runtime
        # assertion on an inequality, instead of silently taking the wrong branch.
        """
        Test that propagate_real_tensors properly handles SymBool from boolean .item() calls.

        When tracing with propagate_real_tensors=True, if a boolean .item() returns False
        during tracing, a runtime assertion should be generated that throws when the
        boolean becomes True at runtime.

        This is a regression test for a bug where int(real_t) was used instead of real_t,
        causing boolean values to be incorrectly converted to integers and losing the
        proper runtime assertion.
        """

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x if any element equals 0.1, else x plus one.
            if x.eq(0.1).any().item():
                return x
            return x + 1

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            # First call with 0.2 - item() returns False, traces the x + 1 branch
            result1 = f(torch.ones(2) * 0.2)
            torch.testing.assert_close(result1, torch.ones(2) * 1.2)

            # Second call with 0.1 - item() returns True, should throw runtime assertion
            # because the traced graph assumed False
            with self.assertRaisesRegex(
                RuntimeError,
                r"Runtime assertion failed for expression Ne\(u0, 1\)",
            ):
                f(torch.ones(2) * 0.1)

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    @torch._dynamo.config.patch("suppress_errors", True)
    def test_sparse_csr_tensor_no_data_dependent_guard(self):
        # [FUNCTION DOC] Compiles (allowing graph breaks) a model constructing a sparse CSR tensor
        # with scalar capture, dynamic-shape-op capture and error suppression enabled, and checks
        # layout, shape and values match eager (no data-dependent-guard error from printing).
        """
        Test that sparse CSR tensor construction doesn't trigger
        GuardOnDataDependentSymNode errors due to str() being called
        on FakeTensors during graph break handling.
        """

        class MinimalSparseModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward builds a sparse CSR tensor from fixed row
            # pointers and column indices and the given values for a two by two size.
            def forward(self, values):
                # [FUNCTION DOC] HELPER: Builds fixed int64 row and column index tensors and
                # returns the CSR sparse tensor made from them and values with size two by two.
                crow_indices = torch.tensor([0, 2, 4], dtype=torch.int64)
                col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
                size = (2, 2)
                sparse = torch.ops.aten.sparse_csr_tensor(
                    crow_indices, col_indices, values, size
                )
                return sparse

        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        model = MinimalSparseModel()
        eager_out = model(values)

        compiled_model = torch.compile(model, fullgraph=False)
        compiled_out = compiled_model(values)

        self.assertEqual(eager_out.layout, compiled_out.layout)
        self.assertEqual(eager_out.shape, compiled_out.shape)
        torch.testing.assert_close(eager_out.values(), compiled_out.values())


class TestUbackedOps(TestCase):
    # [CLASS DOC] Tests of view, reshape, slice, select, narrow, split and related operators
    # compiled with unbacked sizes or indices (data-dependent values from scalar capture or
    # mark_unbacked), checking results against eager, recompilation counts and, in several tests,
    # the exact graph produced. Also covers mark_dynamic/unbacked/static recompilation behavior and
    # shape ids.
    @fresh_cache()
    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_reshape1(self):
        # [FUNCTION DOC] Compiles a function doing view, reshape and view_copy of a non-contiguous
        # unbacked 1D input to a square shape given by an item. Checks results match eager, the
        # generated AOT graph against an expected string for the strided case and again for a
        # contiguous input (which recompiles for a total of two frames), and that a later call with
        # another contiguous size does not recompile.
        cnt = CompileCounterWithBackend("inductor")

        # Reshape happens in place reshape (no-clone)
        # reshape u1 -> (u0*u0)
        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads the item f from y, then returns the input viewed to f by
            # f and multiplied by 10, the same via reshape multiplied by 10, and a view copy to
            # that shape.
            f = y.item()
            t1 = x.view((f, f))
            t2 = x.reshape((f, f))
            t3 = torch._ops.ops.aten.view_copy(x, (f, f))
            return t1 * 10, t2 * 10, t3

        compiled_func = torch.compile(
            fullgraph=True,
            backend=cnt,
            dynamic=True,
        )(func)

        # create a non-contiguous with data being even numbers in [0:cnt-1]
        # and reshape it into sqrt(cnt)*sqrt(cnt)
        def make_non_contiguous_tensor_and_test(cnt):
            # [FUNCTION DOC] HELPER: Builds a strided 1D tensor taking every second element of a
            # range of twice the count, marks its dimension unbacked, runs compiled and eager with
            # the square root of the count as the target size, and asserts equal results.
            # create a non-contiguous tensor x that is skipping odd indices.
            x = torch.arange(cnt * 2)
            x = x.as_strided((x.size()[0] // 2,), (2,))

            torch._dynamo.decorators.mark_unbacked(x, 0)
            sz = torch.tensor([int(math.sqrt(cnt))])
            compiled_result = compiled_func(x, sz)
            eager_result = func(x, sz)
            self.assertEqual(compiled_result, eager_result)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            make_non_contiguous_tensor_and_test(4)
        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[1][1]cpu", arg1_1: "Sym(u1)", arg2_1: "Sym(s7)", arg3_1: "i64[u1][s7]cpu"):
        ge: "Sym(u1 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u1 >= 0 on node 'ge'");  ge = _assert_scalar = None
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        ge_1: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
        pow_1: "Sym(u0**2)" = _local_scalar_dense ** 2
        eq: "Sym(Eq(u1, u0**2))" = arg1_1 == pow_1;  arg1_1 = pow_1 = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0**2) on node 'eq'");  eq = _assert_scalar_2 = None
        view: "i64[u0, u0][s7*u0, s7]cpu" = torch.ops.aten.view.default(arg3_1, [_local_scalar_dense, _local_scalar_dense])
        view_1: "i64[u0, u0][s7*u0, s7]cpu" = torch.ops.aten.view.default(arg3_1, [_local_scalar_dense, _local_scalar_dense])
        view_2: "i64[u0, u0][s7*u0, s7]cpu" = torch.ops.aten.view.default(arg3_1, [_local_scalar_dense, _local_scalar_dense]);  arg3_1 = _local_scalar_dense = None
        clone: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.clone.default(view_2);  view_2 = None
        mul_11: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.mul.Tensor(view, 10);  view = None
        mul_14: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.mul.Tensor(view_1, 10);  view_1 = None
        return (mul_11, mul_14, clone)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        make_non_contiguous_tensor_and_test(49)
        self.assertEqual(cnt.frame_count, 1)

        # Pass in a contiguous tensor, it will recompile due to stride being 1 (0/1 specialization).
        # marking strides unbacked would have avoided the recompilation here.
        x = torch.arange(100)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            compiled_result = compiled_func(x, torch.tensor([10]))
            eager_result = func(x, torch.tensor([10]))
            self.assertEqual(compiled_result, eager_result)
            self.assertEqual(cnt.frame_count, 2)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[1][1]cpu", arg1_1: "Sym(u1)", arg2_1: "i64[u1][1]cpu"):
        ge: "Sym(u1 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u1 >= 0 on node 'ge'");  ge = _assert_scalar = None
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        ge_1: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
        pow_1: "Sym(u0**2)" = _local_scalar_dense ** 2
        eq: "Sym(Eq(u1, u0**2))" = arg1_1 == pow_1;  arg1_1 = pow_1 = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0**2) on node 'eq'");  eq = _assert_scalar_2 = None
        view: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.view.default(arg2_1, [_local_scalar_dense, _local_scalar_dense])
        view_1: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.view.default(arg2_1, [_local_scalar_dense, _local_scalar_dense])
        view_2: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.view.default(arg2_1, [_local_scalar_dense, _local_scalar_dense]);  arg2_1 = _local_scalar_dense = None
        clone: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.clone.default(view_2);  view_2 = None
        mul_6: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.mul.Tensor(view, 10);  view = None
        mul_9: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.mul.Tensor(view_1, 10);  view_1 = None
        return (mul_6, mul_9, clone)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        x = torch.arange(25)
        compiled_result = compiled_func(x, torch.tensor([5]))
        eager_result = func(x, torch.tensor([5]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_reshape2(self):
        # [FUNCTION DOC] Compiles a reshape from a transposed (non-contiguous) tensor with both
        # dimensions unbacked to a shape given by two items. Checks the AOT graph (a clone then a
        # view with a runtime size-equality assertion), results versus eager over several inputs
        # with one frame, then that a contiguous input triggers a second compilation.
        cnt = CompileCounterWithBackend("inductor")

        # This reshape requires a clone when the input is not contiguous and we can't compute strides.
        # reshape (u2, u3) -> (u0, u1)
        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads two ints from y and returns the input reshaped to them
            # times 10.
            u0, u1 = y.tolist()

            result1 = torch.reshape(x, (u0, u1))
            return result1 * 10

        compiled_func = torch.compile(fullgraph=True, backend=cnt, dynamic=True)(func)

        x = torch.randn(10, 10)
        # make x not contiguous.
        x = x.t_()
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            result_eager = func(x, torch.tensor([5, 20]))
            result_compiled = compiled_func(x, torch.tensor([5, 20]))
            self.assertEqual(result_compiled, result_eager)
            self.assertEqual(cnt.frame_count, 1)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[2][1]cpu", arg1_1: "Sym(u2)", arg2_1: "Sym(u3)", arg3_1: "f32[u2, u3][1, u2]cpu"):
        ge: "Sym(u2 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u2 >= 0 on node 'ge'");  ge = _assert_scalar = None
        ge_1: "Sym(u3 >= 0)" = arg2_1 >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u3 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
        select: "i64[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 0)
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(select);  select = None
        ge_2: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u0 >= 0 on node 'ge_2'");  ge_2 = _assert_scalar_2 = None
        sym_sum: "Sym(u0 + 1)" = torch.sym_sum((1, _local_scalar_dense))
        gt: "Sym(u0 + 1 > 0)" = sym_sum > 0;  sym_sum = None
        _assert_scalar_3 = torch.ops.aten._assert_scalar.default(gt, "Runtime assertion failed for expression 0 < u0 + 1 on node 'gt'");  gt = _assert_scalar_3 = None
        select_1: "i64[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        _local_scalar_dense_1: "Sym(u1)" = torch.ops.aten._local_scalar_dense.default(select_1);  select_1 = None
        ge_3: "Sym(u1 >= 0)" = _local_scalar_dense_1 >= 0
        _assert_scalar_4 = torch.ops.aten._assert_scalar.default(ge_3, "Runtime assertion failed for expression u1 >= 0 on node 'ge_3'");  ge_3 = _assert_scalar_4 = None
        sym_sum_1: "Sym(u1 + 1)" = torch.sym_sum((1, _local_scalar_dense_1))
        gt_1: "Sym(u1 + 1 > 0)" = sym_sum_1 > 0;  sym_sum_1 = None
        _assert_scalar_5 = torch.ops.aten._assert_scalar.default(gt_1, "Runtime assertion failed for expression 0 < u1 + 1 on node 'gt_1'");  gt_1 = _assert_scalar_5 = None
        mul: "Sym(u2*u3)" = arg1_1 * arg2_1;  arg1_1 = arg2_1 = None
        mul_1: "Sym(u0*u1)" = _local_scalar_dense * _local_scalar_dense_1
        eq: "Sym(Eq(u2*u3, u0*u1))" = mul == mul_1;  mul = mul_1 = None
        _assert_scalar_6 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u2*u3, u0*u1) on node 'eq'");  eq = _assert_scalar_6 = None
        clone: "f32[u2, u3][Max(1, u3), 1]cpu" = torch.ops.aten.clone.default(arg3_1, memory_format = torch.contiguous_format);  arg3_1 = None
        view: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.view.default(clone, [_local_scalar_dense, _local_scalar_dense_1]);  clone = _local_scalar_dense = _local_scalar_dense_1 = None
        mul_21: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.mul.Tensor(view, 10);  view = None
        return (mul_21,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        result_eager = func(x, torch.tensor([2, 50]))
        result_compiled = compiled_func(x, torch.tensor([2, 50]))
        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(cnt.frame_count, 1)

        x = torch.randn(4, 4).t_()
        result_eager = func(x, torch.tensor([2, 8]))
        result_compiled = compiled_func(x, torch.tensor([2, 8]))
        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(cnt.frame_count, 1)

        # Pass a contiguous tensor. A recompilation will happen due to 0/1 specialization on stride.
        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            # This used to hit could guard on data-dependent expression Eq(10, u3) x.stride[0]==10. and x.size()=[u2, u3].
            # but not anymore since we use  contiguous_or_false .
            # We need a way to mark strides unbacked to avoid the recompilation here.
            x = torch.randn(10, 10)
            torch._dynamo.decorators.mark_unbacked(x, 0)
            torch._dynamo.decorators.mark_unbacked(x, 1)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        result_compiled = compiled_func(x, torch.tensor([2, 50]))
        result_eager = func(x, torch.tensor([2, 50]))

        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(cnt.frame_count, 2)

        x = torch.randn(4, 4)

        result_eager = func(x, torch.tensor([2, 8]))
        result_compiled = compiled_func(x, torch.tensor([2, 8]))
        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(cnt.frame_count, 2)

    @fresh_cache()
    @skipIfTorchDynamo("test inspects inner torch.compile/AOT backend graph")
    def test_user_check_input_shape_bound_preserved_in_aot_graph(self):
        # [FUNCTION DOC] Compiles a function with a user shape check carrying a custom message
        # using the AOT-eager recording backend and checks the forward graph keeps exactly one
        # assertion node with that message, runs correctly for size 4, and raises RuntimeError with
        # that message for size 6. Also checks a lazy message callable that raises is not called
        # when the check passes at trace time.
        def f(x):
            # [FUNCTION DOC] HELPER: Checks the first size is at most 5 with a custom message
            # callable and returns x plus one.
            torch._check(x.shape[0] <= 5, lambda: "custom shape check failed")
            return x + 1

        backend = AotEagerAndRecordGraphs()
        compiled_f = torch.compile(f, fullgraph=True, dynamic=True, backend=backend)
        x = torch.ones(4)
        self.assertEqual(compiled_f(x), f(x))

        self.assertEqual(len(backend.fw_graphs), 1)
        fw_graph = backend.fw_graphs[0]
        assert_nodes = [
            node
            for node in fw_graph.graph.nodes
            if node.target is torch.ops.aten._assert_scalar.default
        ]
        self.assertEqual(len(assert_nodes), 1)
        self.assertEqual(assert_nodes[0].args[1], "custom shape check failed")
        self.assertEqual(fw_graph(4, x)[0], f(x))

        with self.assertRaisesRegex(RuntimeError, "custom shape check failed"):
            fw_graph(6, torch.ones(6))

        def g(x):
            # [FUNCTION DOC] HELPER: Same check as the first function but with a message callable
            # that raises if invoked; returns x plus one.
            def lazy_message():
                # [FUNCTION DOC] HELPER: Message callable that always raises RuntimeError; used to
                # verify the message is evaluated lazily.
                raise RuntimeError("message should be lazy")

            torch._check(x.shape[0] <= 5, lazy_message)
            return x + 1

        backend = AotEagerAndRecordGraphs()
        compiled_g = torch.compile(g, fullgraph=True, dynamic=True, backend=backend)
        x = torch.ones(4)
        self.assertEqual(compiled_g(x), g(x))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_slice(self):
        # [FUNCTION DOC] Compiles three slicing cases with unbacked bounds: a range with checks
        # proving the size equals end minus start (statically known), a negative start with a
        # proven size, and an unchecked general slice whose post-grad graph is compared to an
        # expected string over negative, out-of-bounds and mixed inputs with one frame. Also checks
        # violated checks raise RuntimeError.
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        # standard slice
        def f1(x, xs):
            # [FUNCTION DOC] HELPER: Reads start and stop from a tensor, adds torch checks bounding
            # them, slices, raises AssertionError unless the output size is statically known to be
            # stop minus start, and returns the slice.
            u0, u1 = xs.tolist()
            # in this test we add the torch checks not to avoid DDE but to ensure
            # that we pick specific path during compilation.
            torch._check(u0 >= 0)
            torch._check(u0 <= x.size(0))
            torch._check(u1 >= 0)
            torch._check(u1 <= x.size(0))
            torch._check(u0 <= u1)
            out = x[u0:u1]
            if not statically_known_true(out.size(0) == (u1 - u0)):
                raise AssertionError(
                    "expected statically_known_true(out.size(0) == (u1 - u0))"
                )
            return out

        x, xs = torch.randn(10), torch.tensor([3, 6])
        fn1 = torch.compile(f1, fullgraph=True, backend="inductor")
        self.assertEqual(fn1(x, xs).size(0), 3)
        self.assertTrue(torch.allclose(fn1(x, xs), f1(x, xs)))
        with self.assertRaises(RuntimeError):
            fn1(x, torch.tensor([-1, 5]))

        # known negative slice
        def f2(x, n):
            # [FUNCTION DOC] HELPER: Reads n, checks it is greater than 1 and within the size,
            # slices the last n items, raises AssertionError unless the size is statically known to
            # be n, and returns the slice.
            u0 = n.item()
            torch._check(u0 > 1)
            torch._check(u0 <= x.size(0))
            out = x[-u0:]
            if not statically_known_true(out.size(0) == u0):
                raise AssertionError(
                    "expected statically_known_true(out.size(0) == u0)"
                )
            return out

        x, n = torch.randn(10), torch.tensor([5])
        fn2 = torch.compile(f2, fullgraph=True, backend="inductor")
        self.assertEqual(fn2(x, n).size(0), 5)
        self.assertTrue(torch.allclose(fn2(x, n), f2(x, n)))
        with self.assertRaises(RuntimeError):
            fn2(x, torch.tensor([-5]))

        # general case: no known info
        def f3(x, xs):
            # [FUNCTION DOC] HELPER: Reads start and stop from a tensor and returns the slice with
            # no other information.
            u0, u1 = xs.tolist()
            return x[u0:u1]

        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        cnts = CompileCounterWithBackend("inductor")
        x, xs = torch.randn(10), torch.tensor([3, 6])
        with ctx():
            fn3 = torch.compile(f3, fullgraph=True, backend=cnts)
            xs = torch.tensor([-9, -1])  # negative case
            self.assertTrue(torch.allclose(fn3(x, xs), f3(x, xs)))
            xs = torch.tensor([-1000, 1000])  # out of bounds
            self.assertTrue(torch.allclose(fn3(x, xs), f3(x, xs)))
            xs = torch.tensor([2, -2])  # mixed
            self.assertTrue(torch.allclose(fn3(x, xs), f3(x, xs)))
            self.assertEqual(cnts.frame_count, 1)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
        select: "i64[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 0)
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(select);  select = None
        select_1: "i64[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        _local_scalar_dense_1: "Sym(u1)" = torch.ops.aten._local_scalar_dense.default(select_1);  select_1 = None
        slice_1: "f32[u2][1]cpu" = torch.ops.aten.slice.Tensor(arg1_1, 0, _local_scalar_dense, _local_scalar_dense_1);  arg1_1 = _local_scalar_dense = _local_scalar_dense_1 = None
        sym_size_int: "Sym(u2)" = torch.ops.aten.sym_size.int(slice_1, 0)
        ge_1: "Sym(u2 >= 0)" = sym_size_int >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u2 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        le: "Sym(u2 <= 10)" = sym_size_int <= 10;  sym_size_int = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u2 <= 10 on node 'le'");  le = _assert_scalar_1 = None
        sym_storage_offset_default: "Sym(u3)" = torch.ops.aten.sym_storage_offset.default(slice_1)
        ge_2: "Sym(u3 >= 0)" = sym_storage_offset_default >= 0;  sym_storage_offset_default = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u3 >= 0 on node 'ge_1'");  ge_2 = _assert_scalar_2 = None
        return (slice_1,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_unbacked_slice_cpp_wrapper(self):
        # [FUNCTION DOC] Reruns the unbacked slice test with the C++ wrapper config enabled.
        self.test_unbacked_slice()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_slice_with_step(self):
        # [FUNCTION DOC] Compiles slicing with unbacked bounds and a step of 5 through inductor and
        # checks the result matches eager for a negative stop.
        def f1(x, xs):
            # [FUNCTION DOC] HELPER: Reads start and stop from a tensor and returns the
            # strided slice.
            u0, u1 = xs.tolist()
            out = x[u0:u1:5]
            return out

        x, xs = torch.randn(10), torch.tensor([2, -2])
        fn1 = torch.compile(f1, fullgraph=True, backend="inductor")
        self.assertTrue(torch.allclose(fn1(x, xs), f1(x, xs)))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_unbacked_slice_with_step_cpp_wrapper(self):
        # [FUNCTION DOC] Reruns the unbacked slice-with-step test with the C++ wrapper
        # config enabled.
        self.test_unbacked_slice_with_step()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_slice_noop_with_unbacked_bindings(self):
        # [FUNCTION DOC] Regression test: slicing from zero to an unbacked end followed by checks
        # that make the slice a no-op at lowering time must still compile and equal eager.
        # Regression test: the no-op early return in slice_ lowering
        # (start==0, size<=end, step==1) must register DynamicSliceSize
        # for unbacked bindings created at trace time.
        # torch._check after the slice adds facts to ShapeEnv that make
        # statically_known_leq(size, end) True at lowering time.
        def f(x, n):
            # [FUNCTION DOC] HELPER: Slices x up to the item n, then checks n is non-negative and
            # at least the first size, and returns the slice.
            u0 = n.item()
            result = x[:u0]
            torch._check(u0 >= 0)
            torch._check(u0 >= x.size(0))
            return result

        x = torch.randn(10, 16)
        n = torch.tensor([10])

        eager_result = f(x, n)

        compiled_fn = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result = compiled_fn(x, n)

        self.assertEqual(eager_result, compiled_result)

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_slice_with_tensor_indices(self):
        # [FUNCTION DOC] For both dynamic and static compilation, checks slicing with zero-dim
        # tensors as start, stop and step (combinations of them, negative start, and a 2D case)
        # compile through inductor and match eager results.
        for d in [True, False]:
            # Test slicing with tensor start/stop/step on RHS (reading)

            # Test 1: Basic slice with tensor start and stop
            def f1(x, start_t, stop_t):
                # [FUNCTION DOC] HELPER: Returns x sliced by tensor start and stop.
                return x[start_t:stop_t]

            x = torch.randn(20)
            start_t = torch.tensor(5)
            stop_t = torch.tensor(15)
            fn1 = torch.compile(f1, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(
                torch.allclose(fn1(x, start_t, stop_t), f1(x, start_t, stop_t))
            )

            # Test 2: Slice with tensor step
            def f2(x, start_t, stop_t, step_t):
                # [FUNCTION DOC] HELPER: Returns x sliced by tensor start, stop and step.
                return x[start_t:stop_t:step_t]

            step_t = torch.tensor(2)
            fn2 = torch.compile(f2, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(
                torch.allclose(
                    fn2(x, start_t, stop_t, step_t), f2(x, start_t, stop_t, step_t)
                )
            )

            # Test 3: Slice with only tensor start
            def f3(x, start_t):
                # [FUNCTION DOC] HELPER: Returns x sliced from tensor start to the end.
                return x[start_t:]

            fn3 = torch.compile(f3, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(torch.allclose(fn3(x, start_t), f3(x, start_t)))

            # Test 4: Slice with only tensor stop
            def f4(x, stop_t):
                # [FUNCTION DOC] HELPER: Returns x sliced from the beginning to tensor stop.
                return x[:stop_t]

            fn4 = torch.compile(f4, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(torch.allclose(fn4(x, stop_t), f4(x, stop_t)))

            # Test 5: Negative indices with tensors
            def f5(x, start_t):
                # [FUNCTION DOC] HELPER: Returns x sliced from tensor start up to the last
                # element exclusive.
                return x[start_t:-1]

            start_t_neg = torch.tensor(-10)
            fn5 = torch.compile(f5, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(torch.allclose(fn5(x, start_t_neg), f5(x, start_t_neg)))

            # Test 6: Multidimensional slice with tensor indices
            def f6(x, start_t, stop_t):
                # [FUNCTION DOC] HELPER: Returns x sliced on dimension one by tensor start
                # and stop.
                return x[:, start_t:stop_t]

            x_2d = torch.randn(10, 20)
            fn6 = torch.compile(f6, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(
                torch.allclose(fn6(x_2d, start_t, stop_t), f6(x_2d, start_t, stop_t))
            )

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_slice_with_tensor_indices_cpp_wrapper(self):
        # [FUNCTION DOC] Reruns the tensor-index slice test with the C++ wrapper config enabled.
        self.test_slice_with_tensor_indices()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_tensor_split(self):
        # [FUNCTION DOC] Compiles tensor_split with split points taken from a tensor and
        # checks each output piece matches eager for sorted, partly negative and all negative
        # index tensors.
        def f1(x, xs):
            # [FUNCTION DOC] HELPER: Converts the index tensor through a list back to a tensor and
            # returns tensor_split of x at those indices.
            xs = torch.tensor(xs.tolist())
            return torch.tensor_split(x, xs)

        x = torch.randn(20)
        xs = torch.tensor([5, 10, 15])
        fn = torch.compile(f1, fullgraph=True, backend="inductor")

        def compare(x, xs):
            # [FUNCTION DOC] HELPER: Compares each piece from eager and compiled tensor_split
            # for equality.
            for i, j in zip(f1(x, xs), fn(x, xs)):
                self.assertTrue(torch.allclose(i, j))

        compare(x, xs)
        xs = torch.tensor([-15, 9, 10, 11])
        compare(x, xs)
        xs = torch.tensor([-15, -10, -5, -2])
        compare(x, xs)

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_tensor_split_cpp_wrapper(self):
        # [FUNCTION DOC] Reruns the tensor_split test with the C++ wrapper config enabled.
        self.test_tensor_split()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_nonzero_slice(self):
        # [FUNCTION DOC] Compiles taking nonzero indices and slicing off the last row
        # (data-dependent size) and checks equality with eager for random and all-zero input.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the nonzero index matrix without its last row.
            nz = x.nonzero()
            return nz[:-1]

        x = torch.randn(3, 4)
        fn = torch.compile(f, fullgraph=True, backend="inductor")
        self.assertTrue(torch.allclose(f(x), fn(x)))
        y = torch.zeros(3, 4)
        self.assertTrue(torch.allclose(f(y), fn(y)))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_nonzero_slice_cpp_wrapper(self):
        # [FUNCTION DOC] Reruns the nonzero slice test with the C++ wrapper config enabled.
        self.test_nonzero_slice()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_nonzero_select(self):
        # [FUNCTION DOC] Compiles selecting the last and first rows of a nonzero index matrix and
        # checks the sum equals eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the last and first rows of the nonzero
            # index matrix.
            nz = x.nonzero()
            return nz[-1] + nz[0]

        x = torch.randn(3, 4)
        fn = torch.compile(f, fullgraph=True, backend="inductor")
        self.assertTrue(torch.allclose(f(x), fn(x)))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_nonzero_select_cpp_wrapper(self):
        # [FUNCTION DOC] Reruns the nonzero select test with the C++ wrapper config enabled.
        self.test_nonzero_select()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_padnd(self):
        # [FUNCTION DOC] Compiles padding a matrix by item amounts followed by a matrix multiply
        # with unbacked padding; only checks it compiles and runs.
        import torch.nn.functional as F

        def f(x, xs, y):
            # [FUNCTION DOC] HELPER: Reads two pad amounts from xs, checks both non-negative, pads
            # x on all four sides, and returns the padded matrix multiplied by y.
            u0, u1 = xs.tolist()
            for u in [u0, u1]:
                torch._check(u >= 0)
            z = F.pad(x, (u0, u1, u0, u1))
            return z @ y

        x = torch.randn(8, 8)
        xs = torch.tensor([2, 2])
        y = torch.randn(12, 4)
        fn = torch.compile(f, fullgraph=True, backend="inductor")
        fn(x, xs, y)

    @unittest.skip("this test fails due to inductor/autograd issue #153041")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_non_contigious_reshape_failing(self):
        # [FUNCTION DOC] Skipped test (inductor/autograd issue): view and reshape of a
        # non-contiguous unbacked tensor to a square, compared to eager.
        # reshape u1 -> (u0*u0)
        # this result in the tensor "i64[u0, u0][s7*u0, s7].
        # reshape happens in place reshape (no-clone)
        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads item f, returns the input viewed and reshaped to f by f.
            f = y.item()
            t1 = x.view((f, f))
            t2 = x.reshape((f, f))
            return t1, t2

        # create a non-contiguous with data being even numbers in [0:cnt-1]
        def make_non_contiguous_tensor(cnt):
            # [FUNCTION DOC] HELPER: Returns a tensor that takes every second element of a range of
            # twice the count.
            # create a non-contiguous tensor x that is skipping odd indices.
            x = torch.arange(cnt * 2)
            x = x.as_strided((x.size()[0] // 2,), (2,))
            return x

        x = make_non_contiguous_tensor(4)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        compiled_func = torch.compile(
            fullgraph=True,
            backend="inductor",
        )(func)

        compiled_result = compiled_func(x, torch.tensor([2]))
        eager_result = func(x, torch.tensor([2]))
        self.assertEqual(compiled_result, eager_result)

    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_invalid_view_unbacked_view(self):
        # [FUNCTION DOC] Checks that an invalid view of a transposed unbacked tensor to two
        # item sizes cannot be compiled: calling raises torch dynamo UserError for the
        # data-dependent error.
        cnt = CompileCounterWithBackend("inductor")

        # This view (u2, u3) -> (u0, u1) can't happen in general unless we know that input is contiguous or we have
        # hints to compute strides.
        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads two ints from y and returns the input viewed to them
            # times 10.
            u0, u1 = y.tolist()
            result2 = x.view(u0, u1) * 10
            return result2

        compiled_func = torch.compile(fullgraph=True, backend=cnt, dynamic=True)(func)

        x = torch.randn(10, 10)
        # make x not contiguous.
        x = x.t_()
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        with self.assertRaises(torch._dynamo.exc.UserError):
            # throws a data dependent error.
            compiled_func(x, torch.tensor([5, 20]))

    @skipIfTorchDynamo()
    def test_unbind_not_dynamic(self):
        # [FUNCTION DOC] Checks that unbind along dimension 2 under dynamic compilation keeps that
        # dimension static: changing other sizes does not recompile, but changing dim 2 recompiles
        # (frame counts 1, 1 and then 3).
        cnt = CompileCounter()

        @torch.compile(fullgraph=True, dynamic=True, backend=cnt)
        def func(y):
            # [FUNCTION DOC] HELPER: Returns the unbind of y along dimension two and y times 10.
            return y.unbind(dim=2), y * 10

        func(torch.ones(5, 6, 7, 8))
        self.assertEqual(cnt.frame_count, 1)
        # it can be dynamic in all dimensions except dim=2
        func(torch.ones(4, 9, 7, 10))
        self.assertEqual(cnt.frame_count, 1)

        func(torch.ones(5, 6, 8, 8))
        func(torch.ones(5, 6, 9, 8))
        self.assertEqual(cnt.frame_count, 3)

    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @fresh_cache()
    def test_unbacked_contiguous(self):
        # [FUNCTION DOC] Compiles contiguous plus arithmetic on a transposed unbacked input, checks
        # the post-grad graph with a clone, another call without recompiling, then a contiguous
        # unbacked input that recompiles (two frames) and produces a graph without the clone.
        cnt = CompileCounterWithBackend("inductor")

        def func(x):
            # [FUNCTION DOC] HELPER: Makes the input contiguous, adds one and multiplies by 100.
            contig = x.contiguous()
            return (contig + 1) * 100

        compiled_func = torch.compile(fullgraph=True, backend=cnt, dynamic=True)(func)

        x = torch.randn(10, 10)
        # make x not contiguous.
        x = x.t_()
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        with ctx():
            compiled_func(x)
            self.assertEqual(compiled_func(x), func(x))
            y = torch.rand(20, 20).t()
            self.assertEqual(compiled_func(y), func(y))
            self.assertEqual(cnt.frame_count, 1)
        output = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            output,
            """\
        ge: "Sym(u0 >= 0)" = arg0_1 >= 0;  arg0_1 = None
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        ge_1: "Sym(u1 >= 0)" = arg1_1 >= 0;  arg1_1 = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
        clone: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.clone.default(arg2_1, memory_format = torch.contiguous_format);  arg2_1 = None
        add_3: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.add.Tensor(clone, 1);  clone = None
        mul_6: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.mul.Tensor(add_3, 100);  add_3 = None
        return (mul_6,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        with ctx():
            # recompilation will happen due to stride specialization.
            y = torch.rand(20, 20)
            torch._dynamo.decorators.mark_unbacked(y, 0)
            torch._dynamo.decorators.mark_unbacked(y, 1)
            self.assertEqual(compiled_func(y), func(y))
            self.assertEqual(cnt.frame_count, 2)

        output = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()

        # No clone this time since input is contiguous.
        self.assertExpectedInline(
            output,
            """\
        ge: "Sym(u0 >= 0)" = arg0_1 >= 0;  arg0_1 = None
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        ge_1: "Sym(u1 >= 0)" = arg1_1 >= 0;  arg1_1 = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
        add: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.add.Tensor(arg2_1, 1);  arg2_1 = None
        mul_5: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.mul.Tensor(add, 100);  add = None
        return (mul_5,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_select_index(self):
        # [FUNCTION DOC] Compiles selects with an item index on three dimensions, checks the
        # post-grad graph, and that zero, positive and negative indices run in one frame; then
        # compiles a static version selecting with a sum of two items and checks results for all
        # signs with a second frame.
        cnt = CompileCounterWithBackend("inductor")

        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads item u0 and returns selects of x at that index on
            # dims 0,1,2.
            u0 = y.item()
            return (
                torch.select(x, 0, u0),
                torch.select(x, 1, u0),
                torch.select(x, 2, u0),
            )

        compiled_func = torch.compile(fullgraph=True, backend=cnt, dynamic=True)(func)
        x = torch.rand(3, 3, 3)
        zero = torch.tensor([0])
        pos = torch.tensor([1])
        # code can handle both negative and positive indices.
        neg = torch.tensor([-1])

        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        with ctx():
            self.assertEqual(compiled_func(x, zero), func(x, zero))
        output = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            output,
            """\
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        select: "f32[s77, s77][s77, 1]cpu" = torch.ops.aten.select.int(arg2_1, 0, _local_scalar_dense)
        select_1: "f32[s77, s77][s77**2, 1]cpu" = torch.ops.aten.select.int(arg2_1, 1, _local_scalar_dense)
        select_2: "f32[s77, s77][s77**2, s77]cpu" = torch.ops.aten.select.int(arg2_1, 2, _local_scalar_dense);  arg2_1 = _local_scalar_dense = None
        return (select, select_1, select_2)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        self.assertEqual(compiled_func(x, pos), func(x, pos))
        self.assertEqual(compiled_func(x, neg), func(x, neg))
        self.assertEqual(cnt.frame_count, 1)

        def func2(x, y):
            # [FUNCTION DOC] HELPER: Reads two items and returns the select of x on dim 0 at
            # their sum.
            u0, u1 = y.tolist()
            return torch.select(x, 0, u0 + u1)

        compiled_func2 = torch.compile(fullgraph=True, backend=cnt, dynamic=False)(
            func2
        )
        zero = torch.tensor([0, 0])
        pos = torch.tensor([1, 1])
        neg = torch.tensor([-1, -1])

        self.assertEqual(compiled_func2(x, pos), func2(x, pos))
        self.assertEqual(compiled_func2(x, neg), func2(x, neg))
        self.assertEqual(compiled_func2(x, zero), func2(x, zero))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_select_2(self):
        # [FUNCTION DOC] Compiles a module that selects the last row of a nonzero result and checks
        # equality with eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the last row of the nonzero
            # index matrix.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the last row of the nonzero index matrix of x.
                nz = x.nonzero()
                return nz[-1]

        mod = M()
        x = torch.randn(4)
        self.assertEqual(torch.compile(mod)(x), mod(x))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_select_index_with_check(self):
        # [FUNCTION DOC] Compiles select with an item index after a non-negativity check and
        # compares to eager for positive and zero index. Also prints one compiled result.
        def func3(x, y):
            # [FUNCTION DOC] HELPER: Reads item u0, checks it is non-negative, returns a tuple with
            # the select at dim 1 and that index.
            u0 = y.item()
            # Test that taking the non-unbacked path works fine also.
            torch._check(u0 >= 0)
            return (torch.select(x, 1, u0),)

        compiled_func3 = torch.compile(
            fullgraph=True, backend="inductor", dynamic=True
        )(func3)
        x = torch.rand(3, 3, 3)
        zero = torch.tensor([0])
        pos = torch.tensor([1])
        print(compiled_func3(x, pos))

        self.assertEqual(compiled_func3(x, pos), func3(x, pos))
        self.assertEqual(compiled_func3(x, zero), func3(x, zero))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_unbacked_select_index_cpp_wrapper(self):
        # [FUNCTION DOC] Reruns the unbacked select index test with the C++ wrapper config enabled.
        self.test_unbacked_select_index()

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_select2(self):
        # [FUNCTION DOC] Compiles selecting a matrix by an item index and squaring it by matrix
        # multiply, comparing to eager.
        def f(idx, x):
            # [FUNCTION DOC] HELPER: Selects index idx from dim 0 of x and returns the matrix
            # product of the result with itself.
            x = x.select(0, idx.item())
            return x @ x

        x = torch.randn(3, 3, 3)
        idx = torch.tensor(1, dtype=torch.int64)
        out = torch.compile(f)(idx, x)
        self.assertEqual(out, f(idx, x))

    def test_trunc_int_div_true(self):
        # [FUNCTION DOC] Compiles with dynamic shapes a function with checks on three int arguments
        # and a branch on an integer-truncated expression; only checks it compiles and runs.
        @torch.compile(backend="inductor", dynamic=True, fullgraph=True)
        def f(x, s13, s57, s77):
            # [FUNCTION DOC] HELPER: Checks three ints non-negative, then returns x times 2 if the
            # truncated value of a mixed division expression is at least 1, else x times 100.
            torch._check(s13 >= 0)
            torch._check(s57 >= 0)
            torch._check(s77 >= 0)
            if int(s13 * ((s57 // s13) + (s77 // s13)) / s13) >= 1:
                return x * 2
            else:
                return x * 100

        # ensure we compile this with no errors.
        x = torch.rand(10)
        f(x, 4, 4096, 3920)

    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_reshape3(self):
        # [FUNCTION DOC] Compiles viewing a strided input of unbacked first size with extra
        # dimension, comparing with eager.
        def func(x):
            # [FUNCTION DOC] HELPER: Restrides x to a fixed second size of 1536 with outer stride
            # 2048, views it with a trailing size of 128, and multiplies by 10.
            x = x.as_strided([x.size()[0], 1536], [2048, 1])
            result1 = x.view(x.size()[0], -1, 128)
            return result1 * 10

        compiled = torch.compile(fullgraph=True, backend="inductor")(func)
        x = torch.randn(10, 2048)

        torch._dynamo.decorators.mark_unbacked(x, 0)
        self.assertEqual(func(x), compiled(x))

    @fresh_cache()
    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_reshape_copy(self):
        # [FUNCTION DOC] Same scenario as the first reshape test but for reshape copy: checks the
        # AOT graph for a non-contiguous unbacked input, compile counts for a second strided and a
        # contiguous input, and the graph for the contiguous case.
        cnt = CompileCounterWithBackend("inductor")

        # Reshape happens in place reshape (no-clone)
        # reshape u1 -> (u0*u0)
        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads item f and returns the reshape-copy of x to f by f.
            f = y.item()
            t3 = torch._ops.ops.aten._reshape_copy(x, (f, f))
            return t3

        compiled_func = torch.compile(
            fullgraph=True,
            backend=cnt,
            dynamic=True,
        )(func)

        # create a non-contiguous with data being even numbers in [0:cnt-1]
        # and reshape it into sqrt(cnt)*sqrt(cnt)
        def make_non_contiguous_tensor_and_test(cnt):
            # [FUNCTION DOC] HELPER: Builds a strided unbacked 1D tensor from a range, runs
            # compiled and eager reshape with the square root of the count as size, and asserts
            # equal results.
            # create a non-contiguous tensor x that is skipping odd indices.
            x = torch.arange(cnt * 2)
            x = x.as_strided((x.size()[0] // 2,), (2,))

            torch._dynamo.decorators.mark_unbacked(x, 0)
            sz = torch.tensor([int(math.sqrt(cnt))])
            compiled_result = compiled_func(x, sz)
            eager_result = func(x, sz)
            self.assertEqual(compiled_result, eager_result)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            make_non_contiguous_tensor_and_test(4)
        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[1][1]cpu", arg1_1: "Sym(u1)", arg2_1: "Sym(s7)", arg3_1: "i64[u1][s7]cpu"):
        ge: "Sym(u1 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u1 >= 0 on node 'ge'");  ge = _assert_scalar = None
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        ge_1: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
        pow_1: "Sym(u0**2)" = _local_scalar_dense ** 2
        eq: "Sym(Eq(u1, u0**2))" = arg1_1 == pow_1;  arg1_1 = pow_1 = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0**2) on node 'eq'");  eq = _assert_scalar_2 = None
        _reshape_copy: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten._reshape_copy.default(arg3_1, [_local_scalar_dense, _local_scalar_dense]);  arg3_1 = _local_scalar_dense = None
        return (_reshape_copy,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        make_non_contiguous_tensor_and_test(49)
        self.assertEqual(cnt.frame_count, 1)

        # Pass in a contiguous tensor, it will recompile due to stride being 1 (0/1 specialization).
        # marking strides unbacked would have avoided the recompilation here.
        x = torch.arange(100)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            compiled_result = compiled_func(x, torch.tensor([10]))
            eager_result = func(x, torch.tensor([10]))
            self.assertEqual(compiled_result, eager_result)
            self.assertEqual(cnt.frame_count, 2)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[1][1]cpu", arg1_1: "Sym(u1)", arg2_1: "i64[u1][1]cpu"):
        ge: "Sym(u1 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u1 >= 0 on node 'ge'");  ge = _assert_scalar = None
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        ge_1: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
        pow_1: "Sym(u0**2)" = _local_scalar_dense ** 2
        eq: "Sym(Eq(u1, u0**2))" = arg1_1 == pow_1;  arg1_1 = pow_1 = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0**2) on node 'eq'");  eq = _assert_scalar_2 = None
        _reshape_copy: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten._reshape_copy.default(arg2_1, [_local_scalar_dense, _local_scalar_dense]);  arg2_1 = _local_scalar_dense = None
        return (_reshape_copy,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        x = torch.arange(25)
        compiled_result = compiled_func(x, torch.tensor([5]))
        eager_result = func(x, torch.tensor([5]))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_item(self):
        # [FUNCTION DOC] Compiles a function that selects a masked element from a small tensor and
        # returns its item, with dynamic shapes, and checks the compiled result equals eager.
        def func():
            # [FUNCTION DOC] HELPER: Builds a two-element int64 tensor and a boolean mask that
            # is true only for the first element, selects the masked element and returns its
            # python scalar.
            _x_ms = torch.tensor([True, False], dtype=torch.int64)
            _mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)
            _mask_ms[:1] = True
            var_node_2 = torch.masked_select(_x_ms, _mask_ms)
            var_node_0 = var_node_2.item()
            return var_node_0

        result_original = func()
        compiled_program = torch.compile(func, fullgraph=True, dynamic=True)
        result_compiled = compiled_program()
        self.assertEqual(result_original, result_compiled)

    def test_unbacked_item_set_item(self):
        # [FUNCTION DOC] Compiles a histogram-like loop that reads item indices from an int tensor
        # and increments positions of a zeros tensor; checks the compiled result equals eager.
        def my_arithmetic(a, b):
            # [FUNCTION DOC] HELPER: For each position of a, reads the index item from b at
            # that position and adds one to that position of a zeros work tensor; returns the
            # work tensor.
            wrk = torch.zeros(a.size(0))
            for i in range(a.size(0)):
                idx = b[i].item()
                wrk[idx] += 1

            return wrk

        compiled = torch.compile(my_arithmetic, fullgraph=True, disable=False)
        a = torch.randn([9])
        b = torch.ones(9, dtype=torch.int32)
        compiled(a, b)
        self.assertEqual(compiled(a, b), my_arithmetic(a, b))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_item_set_item2(self):
        # [FUNCTION DOC] Compiles an accumulation reading a start item and adding three following
        # rows to the row at start; checks equality with eager for start 1.
        def accumulate(X0, start):
            # [FUNCTION DOC] HELPER: Converts start to a python scalar, takes the row at start,
            # adds the next three rows to it in place, and returns it.
            start = start.item()
            N = 3
            result = X0[start]
            for i in range(N):
                result += X0[start + 1 + i]
            return result

        compiled = torch.compile(accumulate, fullgraph=True)
        X0 = torch.randn(10, 10)
        self.assertEqual(
            accumulate(X0, torch.tensor([1])), compiled(X0, torch.tensor([1]))
        )

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_item_set_item3(self):
        # [FUNCTION DOC] Compiles an in-place assignment of zero into a tensor at an item index and
        # checks the compiled result equals eager.
        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads item u0 from y, sets x at that index to zero in place,
            # and returns x.
            u0 = y.item()
            x[u0] = 0
            return x

        compiled = torch.compile(func, fullgraph=True, disable=False)
        b = torch.tensor([0])
        a = torch.ones(9, dtype=torch.int32)

        compiled(a, b)
        self.assertEqual(compiled(a, b), func(a, b))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_select_scatter_unbacked_index(self):
        # [FUNCTION DOC] Compiles select_scatter of a scalar value at an item index with dynamic
        # shapes through inductor and checks equality with eager.
        def func(x, y):
            # [FUNCTION DOC] HELPER: Reads item u0 from y and returns x with the scalar 42
            # scattered at that index of dimension zero.
            u0 = y.item()
            # Create a scalar tensor to scatter into the selected index
            scalar_src = torch.tensor(42, dtype=x.dtype)
            return x.select_scatter(scalar_src, 0, u0)

        compiled = torch.compile(func, fullgraph=True, dynamic=True, backend="inductor")
        b = torch.tensor([0])
        a = torch.ones(9, dtype=torch.int32)

        self.assertEqual(compiled(a, b), func(a, b))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_narrow_unbacked_start(self):
        # [FUNCTION DOC] Compiles narrow with an item as start and a python length through inductor
        # and compares against eager over twelve start and length cases covering negative starts,
        # positive starts and edge cases, using subtests.
        def func(x, start, length):
            # [FUNCTION DOC] HELPER: Reads the start as an item and returns the narrow of x on
            # dimension zero from that start for the given length.
            # unbacked start
            u0 = start.item()
            return torch.narrow(x, 0, u0, length)

        compiled_func = torch.compile(func, fullgraph=True, backend="inductor")

        x = torch.tensor([1, 2, 3, 4, 5, 6])

        # Test cases: (start, length)
        test_cases = [
            # Negative starts
            (-2, 2),  # Start from second-to-last element
            (-1, 1),  # Start from last element
            (-3, 3),  # Start from third-to-last element
            (-6, 2),  # Start from beginning (negative)
            (-4, 1),  # Start from fourth-to-last element
            # Positive starts
            (0, 2),  # Start from beginning
            (1, 3),  # Start from second element
            (2, 2),  # Start from third element
            (4, 2),  # Start near end
            # Edge cases
            (0, 6),  # Full tensor
            (0, 1),  # Single element from start
            (5, 1),  # Single element from end
        ]

        for start_val, length in test_cases:
            with self.subTest(start=start_val, length=length):
                start = torch.tensor([start_val])

                # Test with compiled function
                result_compiled = compiled_func(x, start, length)

                # Test with eager function (expected behavior)
                result_eager = func(x, start, length)

                # Compare results
                self.assertEqual(result_compiled, result_eager)

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_narrow_unbacked_start_cpp_wrapper(self):
        # [FUNCTION DOC] Reruns the narrow unbacked start test with the C++ wrapper config enabled.
        """Test narrow with unbacked start with cpp_wrapper"""
        self.test_narrow_unbacked_start()

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_narrow_with_tensor_start(self):
        # [FUNCTION DOC] Compiles narrow with a zero-dim tensor start and zero length
        # through inductor and checks the result has shape of a single zero. Uses a CUDA
        # tensor if available.
        @torch.compile(backend="inductor", fullgraph=True)
        def f(x, start, end):
            # [FUNCTION DOC] HELPER: Returns narrow of x on dimension zero with the given start
            # and length.
            return torch.narrow(x, 0, start, end)

        x = torch.tensor(
            [False], device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        start = torch.tensor(0)
        res = f(x, start, 0)
        self.assertEqual(res.shape, torch.Size([0]))

    @skipIfTorchDynamo("mark_unbacked not supported")
    def test_unbacked_norm_no_dde(self):
        # [FUNCTION DOC] With all dimensions of the inputs marked unbacked, compiles vector norm,
        # distance and Frobenius norm each with the eager backend after a dynamo reset; the test
        # passes if none raise a data-dependent error.
        def vector_norm(x):
            # [FUNCTION DOC] HELPER: Returns the linear algebra vector norm of x.
            return torch.linalg.vector_norm(x)

        def dist_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the distance between x and y.
            return torch.dist(x, y)

        def norm_fro(x):
            # [FUNCTION DOC] HELPER: Returns the Frobenius norm of x.
            return torch.norm(x, p="fro")

        x = torch.randn(4, 5)
        y = torch.randn(4, 5)
        for dim in range(x.ndim):
            torch._dynamo.decorators.mark_unbacked(x, dim)
            torch._dynamo.decorators.mark_unbacked(y, dim)

        for fn, args in [
            (vector_norm, (x,)),
            (dist_fn, (x, y)),
            (norm_fro, (x,)),
        ]:
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, backend="eager")
            compiled(*args)

    @skipIfTorchDynamo()
    @torch.fx.experimental._config.patch("backed_size_oblivious", True)
    def test_backed_size_oblivious_expand(self):
        # [FUNCTION DOC] With backed-size-oblivious on, compiles expand with a shape argument and
        # checks compile counts: matching dims compile once; an input dimension of 1 expanded to
        # larger sizes specializes and reuses the graph until a non-1 input forces a recompile;
        # multi-dimension broadcast is also reused.
        cnt = CompileCounterWithBackend("inductor")
        torch._dynamo.reset()

        def func(a, shape):
            # [FUNCTION DOC] HELPER: Returns a expanded to the given shape.
            return a.expand(*shape)

        compiled = torch.compile(func, fullgraph=True, backend=cnt, dynamic=True)

        def run(a, shape):
            # [FUNCTION DOC] HELPER: Asserts compiled and eager expand results are equal.
            self.assertEqual(compiled(a, shape), func(a, shape))

        # No specialization - matching dimensions
        run(torch.rand(2, 10), (2, 10))
        run(torch.rand(5, 5), (5, 5))
        self.assertEqual(cnt.frame_count, 1)  # Single compilation

        cnt.clear()
        torch._dynamo.reset()

        # Specialize input dim to 1 (broadcasting)
        # Input dim is 1, needs to expand to larger size
        run(torch.rand(1, 10), (9, 10))  # Compile with input[0] == 1
        run(torch.rand(1, 5), (7, 5))  # Reuse same compiled graph
        self.assertEqual(cnt.frame_count, 1)
        # Changing input from 1 triggers recompilation
        run(torch.rand(5, 10), (5, 10))
        self.assertEqual(cnt.frame_count, 2)

        cnt.clear()
        torch._dynamo.reset()

        # Multi-dimensional broadcasting
        # Multiple dimensions with different broadcast semantics
        run(torch.rand(1, 1, 10), (5, 7, 10))  # Broadcast first two dims
        run(torch.rand(1, 1, 5), (3, 4, 5))  # Reuse pattern
        self.assertEqual(cnt.frame_count, 1)

    @skipIfTorchDynamo()
    @torch.fx.experimental._config.patch("backed_size_oblivious", True)
    def test_backed_size_oblivious_broadcast(self):
        # [FUNCTION DOC] With backed-size-oblivious on, compiles an add after a broadcast-shapes
        # call and checks compile counts across inputs: equal shapes share one compile, a broadcast
        # later forces a second; specialization of first-input or second-input dimension to 1
        # recompiles when that dimension changes. The code notes duck sizing may be a problem in
        # the last scenario.
        cnt = CompileCounterWithBackend("inductor")
        torch._dynamo.reset()

        def func(a, b):
            # [FUNCTION DOC] HELPER: Calls broadcast shapes on the sizes of a and b (result
            # discarded) and returns their sum.
            torch.broadcast_shapes(a.size(), b.size())
            return a + b

        compiled = torch.compile(func, fullgraph=True, backend=cnt, dynamic=True)

        def run(a, b):
            # [FUNCTION DOC] HELPER: Asserts compiled and eager results are equal.
            self.assertEqual(compiled(a, b), func(a, b))

        # No 0/1 specializations, no broadcasts.
        # but a[0] == b[0] and a[1] == b[1] are asserted.
        run(torch.rand(1, 10), torch.rand(1, 10))
        run(torch.rand(1, 1), torch.rand(1, 1))
        run(torch.rand(10, 10), torch.rand(10, 10))

        self.assertEqual(cnt.frame_count, 1)
        run(torch.rand(10, 10), torch.rand(1, 10))
        self.assertEqual(cnt.frame_count, 2)

        cnt.clear()
        torch._dynamo.reset()

        # specialize a[0] == 1. b[0] not specialized.
        run(torch.rand(1, 10), torch.rand(9, 10))
        run(torch.rand(1, 10), torch.rand(1, 10))
        self.assertEqual(cnt.frame_count, 1)
        # if we change a[0] we get recompilation.
        run(torch.rand(10, 10), torch.rand(10, 10))
        self.assertEqual(cnt.frame_count, 2)

        cnt.clear()
        torch._dynamo.reset()

        # TODO duck sizing shall be disabled when backed_size_oblivious
        # is on probably.
        # specialize b[0] == 1. a[0] not specialized.
        run(torch.rand(10, 11), torch.rand(1, 11))
        run(torch.rand(1, 10), torch.rand(1, 10))
        self.assertEqual(cnt.frame_count, 1)
        run(torch.rand(2, 10), torch.rand(2, 10))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_unbacked_view_extra(self):
        # [FUNCTION DOC] With dynamic output shape capture, compiles a function building a zeros
        # tensor sized by the count of nonzero elements and viewing it with an inferred dimension;
        # checks equality with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Counts nonzeros in x, builds zeros of that many rows by 192,
            # and returns a view of shape 12, inferred, 192.
            i0 = x.nonzero().size(0)
            y = torch.zeros((i0, 192))
            return y.view([12, -1, 192])

        res1 = torch.compile(fn, fullgraph=True)(torch.ones((12,)))
        res2 = fn(torch.ones((12,)))
        self.assertEqual(res1, res2)

    @skipIfTorchDynamo("mark_dynamic not supported")
    def test_hint_override_consistent_stride1(self):
        # [FUNCTION DOC] Marks two dimensions dynamic with hint overrides (2 and 6) on a three
        # dimensional tensor and compiles a function that reads the guarding hints of sizes and
        # strides, asserting via checks they are consistent (6 and 120). Runs it once.
        @torch.compile(fullgraph=True, dynamic=True)
        def func(x):
            # [FUNCTION DOC] HELPER: Reads guarding hints of the third size and second stride,
            # checks both equal 6, then hints of the product of second and third sizes and first
            # stride and checks both equal 120; returns the two hints and x times 100.
            a = torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw(
                x.size()[2]
            )
            b = torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw(
                x.stride()[1]
            )
            torch._check(a == b)
            torch._check(a == 6)

            a = torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw(
                x.size()[1] * x.size()[2]
            )
            b = torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw(
                x.stride()[0]
            )
            torch._check(a == b)
            torch._check(a == 120)

            return a, b, x * 100

        x = torch.rand(10, 20, 30)

        torch._dynamo.mark_dynamic(x, 0, hint_override=2)
        torch._dynamo.mark_dynamic(x, 2, hint_override=6)

        func(x)

    @skipIfTorchDynamo("mark_dynamic not supported")
    def test_hint_override_consistent_stride2(self):
        # [FUNCTION DOC] Marks all three dimensions dynamic with hint overrides 2, 4 and 6 and
        # compiles a function that checks the product of the last two sizes equals the first stride
        # and equals 24; runs it once.
        @torch.compile(fullgraph=True, dynamic=True)
        def func(x):
            # [FUNCTION DOC] HELPER: Reads guarding hints of the product of the last two sizes and
            # of the first stride, checks both equal 24, returns the two hints and x times 100.
            # only one of the sizes has hint overridden.
            a = torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw(
                x.size()[1] * x.size()[2]
            )
            b = torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw(
                x.stride()[0]
            )
            torch._check(a == b)
            torch._check(a == 24)

            return a, b, x * 100

        x = torch.rand(10, 20, 30)

        torch._dynamo.mark_dynamic(x, 0, hint_override=2)
        torch._dynamo.mark_dynamic(x, 1, hint_override=4)
        torch._dynamo.mark_dynamic(x, 2, hint_override=6)
        func(x)

    def test_optimization_hint(self):
        # [FUNCTION DOC] Compiles a function that checks optimization hints: for a size from an
        # item the fallback 300 is returned, for a size twice another item the hint is 600. Runs
        # with 33.
        @torch.compile(fullgraph=True)
        def func(x):
            # [FUNCTION DOC] HELPER: Builds ones tensors sized by an item and by twice the item,
            # checks the optimization hint of the first size is the fallback 300 and of the second
            # is 600, returns the first times 10.
            u0 = x.item()
            a = torch.ones([u0])
            torch._check(
                torch.fx.experimental.symbolic_shapes.optimization_hint(
                    a.size()[0], fallback=300
                )
                == 300
            )
            b = torch.ones([x.item() * 2])
            torch._check(
                torch.fx.experimental.symbolic_shapes.optimization_hint(
                    b.size()[0], fallback=300
                )
                == 600
            )

            return a * 10

        func(torch.tensor([33]))

    def test_guarding_hint_or_throw(self):
        # [FUNCTION DOC] Checks that calling the guarding hint helper on an unbacked size raises
        # GuardOnDataDependentSymNode when running a compiled function.
        @torch.compile(fullgraph=True)
        def func(x):
            # [FUNCTION DOC] HELPER: Builds a ones tensor sized by an item and asserts its guarding
            # hint is 300 (which raises for unbacked sizes), returns the tensor times 10.
            u0 = x.item()
            a = torch.ones([u0])
            torch._check(
                torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw(
                    a.size()[0]
                )
                == 300
            )

            return a * 10

        self.assertRaises(GuardOnDataDependentSymNode, lambda: func(torch.tensor([33])))

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_shape_id_unifies_unbacked_symbols(self):
        # [FUNCTION DOC] Compares sizes of two tensors with the first dimension marked unbacked and
        # expects a UserError when compiling with full graph. The second part of the test compiles
        # on two plain tensors and checks the add result is chosen.
        # KNOWN ISSUE: the second section is commented as using shape_id to share one symbol, but
        # it never marks the tensors with shape_id (or as unbacked), so it only exercises ordinary
        # static compilation and does not test shape_id unification.
        """
        Test that shape_id parameter in mark_unbacked causes tensors to share
        the same unbacked symbol, allowing equality comparisons without DDE.
        """

        def func(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum if the first sizes are equal, else
            # the difference.
            if x.size()[0] == y.size()[0]:
                return x + y
            else:
                return x - y

        # Test 1: Without shape_id, comparing unbacked symbols raises DDE
        x1 = torch.rand(4, 3)
        y1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0)
        torch._dynamo.decorators.mark_unbacked(y1, 0)

        torch._dynamo.reset()
        with self.assertRaises(torch._dynamo.exc.UserError):
            compiled_func = torch.compile(func, fullgraph=True, backend="eager")
            compiled_func(x1, y1)

        # Test 2: With shape_id, same symbol is used - no DDE, fullgraph succeeds
        x2 = torch.rand(4, 3)
        y2 = torch.rand(4, 3)

        torch._dynamo.reset()
        compiled_func = torch.compile(func, fullgraph=True, backend="eager")
        result = compiled_func(x2, y2)
        # Since both use same symbol, x.size()[0] == y.size()[0] is always True
        self.assertTrue(torch.allclose(result, x2 + y2))

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_shape_id_runtime_assertion_on_mismatch(self):
        # [FUNCTION DOC] Compiles an add with the first dimension of both inputs marked unbacked
        # with one shape id; a call with equal sizes works and matches eager. A second call with
        # the same shape id but different batch sizes (4 and 5) must raise AssertionError.
        """
        Test that shape_id with different actual sizes at runtime
        raises an assertion error during tracing. This ensures that duck shaping
        violations are caught rather than silently producing incorrect results.
        """

        def func(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y.
            return x + y

        torch._dynamo.reset()
        compiled_func = torch.compile(func, fullgraph=True, backend="eager")

        # First, compile with valid inputs (same batch size)
        x1 = torch.rand(4, 3)
        y1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0, shape_id="batch")
        torch._dynamo.decorators.mark_unbacked(y1, 0, shape_id="batch")
        result = compiled_func(x1, y1)
        self.assertTrue(torch.allclose(result, x1 + y1))

        # Now pass tensors with different batch sizes but same shape_id
        # This triggers recompilation, and during tracing the torch._check
        # equality assertion will fail because the sizes don't match
        x2 = torch.rand(4, 3)
        y2 = torch.rand(5, 3)  # Different batch size!
        torch._dynamo.decorators.mark_unbacked(x2, 0, shape_id="batch")
        torch._dynamo.decorators.mark_unbacked(y2, 0, shape_id="batch")

        # Should raise an AssertionError during guard building because batch sizes don't match
        with self.assertRaises(AssertionError):
            compiled_func(x2, y2)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_shape_id_recompilation(self):
        # [FUNCTION DOC] Counts compilations while changing the shape id annotation: first call
        # with a shape id compiles once, same id reuses the frame, and a call without an id
        # recompiles (two frames).
        """
        Test that changing _dynamo_shape_ids triggers recompilation.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with shape_id
        x1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0, shape_id="batch")
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with same shape_id - no recompilation
        x2 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x2, 0, shape_id="batch")
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

        # Third call without shape_id - should recompile
        x3 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x3, 0)
        compiled_func(x3)
        self.assertEqual(counter.frame_count, 2)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_shape_id_no_recompile_without_unbacked_indices(self):
        # [FUNCTION DOC] Checks a plain tensor without unbacked annotations reuses the frame
        # compiled for a tensor annotated with a shape id (one frame).
        """
        Test that passing a tensor without _dynamo_unbacked_indices after
        compiling with shape_ids does NOT trigger recompilation.
        The guard on shape_ids only applies when the runtime tensor
        also has _dynamo_unbacked_indices.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with shape_id (has _dynamo_unbacked_indices)
        x1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0, shape_id="batch")
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with regular tensor (no _dynamo_unbacked_indices)
        # Should NOT recompile - guard passes when no _dynamo_unbacked_indices
        x2 = torch.rand(4, 3)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_shape_id_recompile_with_different_id(self):
        # [FUNCTION DOC] Checks that changing the shape id string between calls of an unbacked
        # tensor triggers a recompilation (two frames).
        """
        Test that passing a tensor with same _dynamo_dynamic_indices but
        different shape_id DOES trigger recompilation.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with shape_id="batch"
        x1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0, shape_id="batch")
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with different shape_id - should recompile
        x2 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x2, 0, shape_id="other")
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 2)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_unbacked_bounds_recompilation(self):
        # [FUNCTION DOC] Checks recompilation on unbacked bounds: same min and max reuses the
        # frame, no bounds recompiles, and different max recompiles again (three frames total).
        """
        Test that changing _dynamo_unbacked_bounds triggers recompilation.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with min/max bounds
        x1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0, min=1, max=100)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with same bounds - no recompilation
        x2 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x2, 0, min=1, max=100)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

        # Third call without bounds - should recompile
        x3 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x3, 0)
        compiled_func(x3)
        self.assertEqual(counter.frame_count, 2)

        # Fourth call with different bounds - should recompile
        x4 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x4, 0, min=1, max=200)
        compiled_func(x4)
        self.assertEqual(counter.frame_count, 3)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_unbacked_bounds_no_recompile_without_unbacked_indices(self):
        # [FUNCTION DOC] Checks a plain tensor reuses the frame compiled with an unbacked tensor
        # that had bounds (still one frame).
        """
        Test that passing a tensor without _dynamo_unbacked_indices after
        compiling with bounds does NOT trigger recompilation.
        The guard on bounds only applies when the runtime tensor
        also has _dynamo_unbacked_indices.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with bounds (has _dynamo_unbacked_indices)
        x1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0, min=1, max=100)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with regular tensor (no _dynamo_unbacked_indices)
        # Should NOT recompile - guard passes when no _dynamo_unbacked_indices
        x2 = torch.rand(4, 3)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_unbacked_no_bounds_then_bounds(self):
        # [FUNCTION DOC] Checks a function compiled for unbacked without bounds is reused for the
        # same marking but recompiled when bounds are then supplied (two frames).
        """
        Test that compiling without bounds then calling with bounds
        triggers recompilation.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with mark_unbacked but no bounds
        x1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with same (no bounds) - no recompilation
        x2 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x2, 0)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

        # Third call with min=1, max=100 - should recompile
        # (compiled without bounds, now calling with bounds)
        x3 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x3, 0, min=1, max=100)
        compiled_func(x3)
        self.assertEqual(counter.frame_count, 2)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_unbacked_indices_recompilation(self):
        # [FUNCTION DOC] Checks subset matching of unbacked indices: after compiling with indices 0
        # and 1, same, subset, plain tensor and empty list inputs reuse the frame; a three
        # dimension tensor with indices 0, 1, 2 recompiles; an empty list afterwards does not set
        # the attribute and reuses the new frame.
        """
        Test that changing _dynamo_unbacked_indices triggers recompilation.
        Uses subset match semantics: runtime indices must be a subset of compiled indices,
        unless the runtime tensor has no attribute (unspecified = don't care).
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with unbacked indices [0, 1]
        x1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, [0, 1])
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with same unbacked indices - no recompilation
        x2 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x2, [0, 1])
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

        # Third call with subset [0] - should NOT recompile (subset of {0, 1})
        x3 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x3, 0)
        compiled_func(x3)
        self.assertEqual(counter.frame_count, 1)

        # Fourth call with no unbacked indices (plain tensor) - should NOT recompile
        # (no attribute = unspecified = don't care, reuse existing frame)
        x4 = torch.rand(4, 3)
        compiled_func(x4)
        self.assertEqual(counter.frame_count, 1)

        # Fifth call with superset [0, 1, 2] - should recompile (not a subset of {0, 1})
        x5 = torch.rand(4, 3, 5)
        torch._dynamo.decorators.mark_unbacked(x5, [0, 1, 2])
        compiled_func(x5)
        self.assertEqual(counter.frame_count, 2)

        # Sixth call with empty list [] - should NOT recompile
        # (empty list is a no-op, no attribute set, same as plain tensor)
        x6 = torch.rand(4, 3, 5)
        torch._dynamo.decorators.mark_unbacked(x6, [])
        self.assertFalse(hasattr(x6, "_dynamo_unbacked_indices"))
        compiled_func(x6)
        self.assertEqual(counter.frame_count, 2)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_unbacked_indices_no_recompile_to_unbacked(self):
        # [FUNCTION DOC] Checks that a tensor with no unbacked annotation compiles once, and an
        # unbacked-annotated tensor afterwards recompiles (two frames).
        """
        Test that compiling without _dynamo_unbacked_indices and then passing
        a tensor with _dynamo_unbacked_indices DOES trigger recompilation.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call without unbacked indices
        x1 = torch.rand(4, 3)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with unbacked indices - should recompile
        x2 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x2, 0)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 2)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_unbacked_no_shape_id_then_shape_id(self):
        # [FUNCTION DOC] Checks that after compiling unbacked tensors with no shape id, repeating
        # reuses the frame but giving a shape id recompiles (two frames).
        """
        Test that compiling without shape_id then calling with shape_id
        triggers recompilation.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with mark_unbacked but no shape_id
        x1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with same (no shape_id) - no recompilation
        x2 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x2, 0)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

        # Third call with shape_id="batch" - should recompile
        # (compiled without shape_id, now calling with shape_id)
        x3 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x3, 0, shape_id="batch")
        compiled_func(x3)
        self.assertEqual(counter.frame_count, 2)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_mark_unbacked_empty_list_is_noop(self):
        # [FUNCTION DOC] Checks that marking unbacked with an empty list sets no attribute on a
        # fresh tensor and does not clear an already marked dimension.
        """
        Test that mark_unbacked(x, []) is a no-op:
        - On a fresh tensor, no attribute is set.
        - After mark_unbacked(x, 0), calling mark_unbacked(x, []) does NOT clear dim 0.
        """
        # Empty list on fresh tensor is a no-op
        x0 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x0, [])
        self.assertFalse(hasattr(x0, "_dynamo_unbacked_indices"))

        # Empty list after marking a dim does not clear it
        x = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        self.assertEqual(x._dynamo_unbacked_indices, {0})
        torch._dynamo.decorators.mark_unbacked(x, [])
        self.assertEqual(x._dynamo_unbacked_indices, {0})

    @skipIfTorchDynamo("mark_dynamic is not traceable")
    def test_mark_dynamic_empty_list_is_noop(self):
        # [FUNCTION DOC] Checks that marking dynamic with an empty list sets no attribute on a
        # fresh tensor and does not clear an already marked dimension.
        """
        Test that mark_dynamic(x, []) is a no-op:
        - On a fresh tensor, no attribute is set.
        - After mark_dynamic(x, 0), calling mark_dynamic(x, []) does NOT clear dim 0.
        """
        # Empty list on fresh tensor is a no-op
        x0 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_dynamic(x0, [])
        self.assertFalse(hasattr(x0, "_dynamo_dynamic_indices"))

        # Empty list after marking a dim does not clear it
        x = torch.rand(4, 3)
        torch._dynamo.decorators.mark_dynamic(x, 0)
        self.assertIn(0, x._dynamo_dynamic_indices)
        torch._dynamo.decorators.mark_dynamic(x, [])
        self.assertIn(0, x._dynamo_dynamic_indices)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_mark_unbacked_to_mark_static_recompilation(self):
        # [FUNCTION DOC] Checks that after compiling with an unbacked mark, a tensor marked static
        # triggers a recompilation (two frames).
        """
        Test that compiling with mark_unbacked and then calling with mark_static
        triggers recompilation.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with mark_unbacked
        x1 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x1, 0)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with mark_static - should recompile
        x2 = torch.rand(4, 3)
        torch._dynamo.mark_static(x2, 0)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 2)

    @skipIfTorchDynamo("mark_static is not traceable")
    def test_mark_static_to_mark_unbacked_recompilation(self):
        # [FUNCTION DOC] Checks that after compiling with a static mark, a tensor marked unbacked
        # triggers a recompilation (two frames).
        """
        Test that compiling with mark_static and then calling with mark_unbacked
        triggers recompilation.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with mark_static
        x1 = torch.rand(4, 3)
        torch._dynamo.mark_static(x1, 0)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with mark_unbacked - should recompile
        x2 = torch.rand(4, 3)
        torch._dynamo.decorators.mark_unbacked(x2, 0)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 2)

    @skipIfTorchDynamo("mark_dynamic is not traceable")
    def test_mark_dynamic_to_mark_static_recompilation(self):
        # [FUNCTION DOC] Checks that after compiling with a dynamic mark, a tensor marked static
        # triggers a recompilation (two frames).
        """
        Test that compiling with mark_dynamic and then calling with mark_static
        triggers recompilation.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with mark_dynamic
        x1 = torch.rand(4, 3)
        torch._dynamo.mark_dynamic(x1, 0)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with mark_static - should recompile
        x2 = torch.rand(4, 3)
        torch._dynamo.mark_static(x2, 0)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 2)

    @skipIfTorchDynamo("mark_dynamic is not traceable")
    def test_dynamic_indices_exact_match_recompilation(self):
        # [FUNCTION DOC] Checks subset matching for dynamic indices: after marking dims 0 and 1
        # dynamic, same marks, a subset, a plain tensor, and an empty-list marking all reuse the
        # single frame.
        """
        Test that dynamic indices use subset match semantics.
        - Compile with mark_dynamic(x, [0, 1]) then call with mark_dynamic(x, [0]) → no recompile (subset)
        - Compile with mark_dynamic(x, [0, 1]) then call with mark_dynamic(x, [2]) → recompile (not subset)
        - Plain tensor (no attribute) = unspecified = don't care → no recompile
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with mark_dynamic on dims 0 and 1
        x1 = torch.rand(4, 3)
        torch._dynamo.mark_dynamic(x1, 0)
        torch._dynamo.mark_dynamic(x1, 1)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with same dynamic indices - should NOT recompile (exact match)
        x2 = torch.rand(4, 3)
        torch._dynamo.mark_dynamic(x2, 0)
        torch._dynamo.mark_dynamic(x2, 1)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

        # Third call with only dim 0 dynamic - should NOT recompile (subset of {0, 1})
        x3 = torch.rand(4, 3)
        torch._dynamo.mark_dynamic(x3, 0)
        compiled_func(x3)
        self.assertEqual(counter.frame_count, 1)

        # Fourth call with plain tensor (no attribute) - should NOT recompile
        # (unspecified = don't care, reuse existing frame)
        x4 = torch.rand(4, 3)
        self.assertFalse(hasattr(x4, "_dynamo_dynamic_indices"))
        compiled_func(x4)
        self.assertEqual(counter.frame_count, 1)

        # Fifth call with empty list [] - should NOT recompile
        # (empty list is a no-op, no attribute set, same as plain tensor)
        x5 = torch.rand(4, 3)
        torch._dynamo.mark_dynamic(x5, [])
        self.assertFalse(hasattr(x5, "_dynamo_dynamic_indices"))
        compiled_func(x5)
        self.assertEqual(counter.frame_count, 1)

    @skipIfTorchDynamo("maybe_mark_dynamic is not traceable")
    def test_weak_dynamic_indices_exact_match_recompilation(self):
        # [FUNCTION DOC] Checks subset matching for weakly dynamic indices (maybe mark dynamic):
        # after marking dims 0 and 1, same marks, a subset, a plain tensor and an empty-list
        # marking all reuse the single frame.
        """
        Test that weak dynamic indices (from maybe_mark_dynamic) use subset match semantics.
        - Compile with maybe_mark_dynamic(x, [0, 1]) then call with maybe_mark_dynamic(x, [0]) → no recompile (subset)
        - Plain tensor (no attribute) = unspecified = don't care → no recompile
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with maybe_mark_dynamic on dims 0 and 1
        x1 = torch.rand(4, 3)
        torch._dynamo.maybe_mark_dynamic(x1, 0)
        torch._dynamo.maybe_mark_dynamic(x1, 1)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with same weak dynamic indices - should NOT recompile (exact match)
        x2 = torch.rand(4, 3)
        torch._dynamo.maybe_mark_dynamic(x2, 0)
        torch._dynamo.maybe_mark_dynamic(x2, 1)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

        # Third call with only dim 0 weak dynamic - should NOT recompile (subset of {0, 1})
        x3 = torch.rand(4, 3)
        torch._dynamo.maybe_mark_dynamic(x3, 0)
        compiled_func(x3)
        self.assertEqual(counter.frame_count, 1)

        # Fourth call with plain tensor (no attribute) - should NOT recompile
        # (unspecified = don't care, reuse existing frame)
        x4 = torch.rand(4, 3)
        self.assertFalse(hasattr(x4, "_dynamo_weak_dynamic_indices"))
        compiled_func(x4)
        self.assertEqual(counter.frame_count, 1)

        # Fifth call with empty list [] - should NOT recompile
        # (empty list is a no-op, no attribute set, same as plain tensor)
        x5 = torch.rand(4, 3)
        torch._dynamo.maybe_mark_dynamic(x5, [])
        self.assertFalse(hasattr(x5, "_dynamo_weak_dynamic_indices"))
        compiled_func(x5)
        self.assertEqual(counter.frame_count, 1)

    @skipIfTorchDynamo("mark_static is not traceable")
    def test_static_indices_exact_match_recompilation(self):
        # [FUNCTION DOC] Checks subset matching for static indices: after marking dims 0 and 1
        # static, same marks, a subset, a plain tensor and an empty-list marking all reuse the
        # single frame.
        """
        Test that static indices use subset match semantics.
        - Compile with mark_static(x, [0, 1]) then call with mark_static(x, [0]) → no recompile (subset)
        - If you want dim 1 NOT static, explicitly mark it dynamic.
        """
        counter = CompileCounter()

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled_func = torch.compile(func, backend=counter)

        # First call with mark_static on dims 0 and 1
        x1 = torch.rand(4, 3)
        torch._dynamo.mark_static(x1, 0)
        torch._dynamo.mark_static(x1, 1)
        compiled_func(x1)
        self.assertEqual(counter.frame_count, 1)

        # Second call with same static indices - should NOT recompile (exact match)
        x2 = torch.rand(4, 3)
        torch._dynamo.mark_static(x2, 0)
        torch._dynamo.mark_static(x2, 1)
        compiled_func(x2)
        self.assertEqual(counter.frame_count, 1)

        # Third call with only dim 0 static - should NOT recompile (subset of {0, 1})
        x3 = torch.rand(4, 3)
        torch._dynamo.mark_static(x3, 0)
        compiled_func(x3)
        self.assertEqual(counter.frame_count, 1)

        # Fourth call with plain tensor (no attribute) - should NOT recompile
        # (unspecified = don't care, reuse existing frame)
        x4 = torch.rand(4, 3)
        self.assertFalse(hasattr(x4, "_dynamo_static_indices"))
        compiled_func(x4)
        self.assertEqual(counter.frame_count, 1)

        # Fifth call with empty list [] - should NOT recompile
        # (empty list is a no-op, no attribute set, same as plain tensor)
        x5 = torch.rand(4, 3)
        torch._dynamo.mark_static(x5, [])
        self.assertFalse(hasattr(x5, "_dynamo_static_indices"))
        compiled_func(x5)
        self.assertEqual(counter.frame_count, 1)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_unbacked_exec_fft_reshape_no_dde(self):
        # [FUNCTION DOC] Compiles an FFT of a complex tensor whose batch dimension is unbacked with
        # the eager backend and full graph, and checks no data-dependent error is raised and the
        # result matches eager.
        """
        Test that view/reshape operations from with in meta python function
        on tensors with unbacked dimensions do not raise GuardOnDataDependentSymNode
        errors. This tests the unbacked handling in the view decomposition via
        _exec_fft.
        """

        def func(x):
            # [FUNCTION DOC] HELPER: Returns the FFT of x.
            # FFT triggers _exec_fft which calls reshape with unbacked batch dims
            return torch.fft.fft(x)

        x = torch.rand(4, 8, dtype=torch.complex64)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        torch._dynamo.reset()
        # This should not raise GuardOnDataDependentSymNode
        compiled_func = torch.compile(func, fullgraph=True, backend="eager")
        result = compiled_func(x)
        expected = torch.fft.fft(x)
        self.assertTrue(torch.allclose(result, expected))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_inductor_issue_164428(self):
        # [FUNCTION DOC] Regression test for issue 164428: runs a fuzzed program (division, matmul,
        # flatten, subtraction, contiguous view) with strided float64 inputs and a sentinel tensor
        # that requires grad; compiled with dynamic shapes must match the eager result.
        torch.manual_seed(6804)

        def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
            # [FUNCTION DOC] HELPER: Fuzzer-generated pipeline: divides a broadcast-strided input
            # by a constant tensor, flattens it, subtracts the difference of a matrix product and
            # arg_2, makes the result contiguous, views it as sixteen by fourteen, multiplies by
            # the sentinel, and returns the real part if the result is complex.
            var_node_4 = arg_0
            var_node_5 = torch.full((7, 1, 32), -1.195053522845565, dtype=torch.float64)
            var_node_3 = torch.div(var_node_4, var_node_5)
            var_node_2 = torch.flatten(var_node_3)
            var_node_8 = torch.full((2,), -0.8316502130341195, dtype=torch.float64)
            var_node_9 = arg_1
            var_node_7 = torch.matmul(
                var_node_8.to(torch.float64), var_node_9.to(torch.float64)
            )
            var_node_10 = arg_2
            var_node_6 = torch.sub(var_node_7, var_node_10)
            var_node_1 = torch.sub(var_node_2, var_node_6)
            var_node_0 = var_node_1.contiguous().view([16, 14])
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        sentinel = torch.tensor(1.0, requires_grad=True)

        arg_0 = torch.as_strided(
            torch.randn(7).to(torch.float64), (7, 1, 32), (1, 1, 0)
        )
        arg_1 = torch.as_strided(torch.randn(448).to(torch.float64), (2, 224), (224, 1))
        arg_2 = torch.as_strided(torch.randn(224).to(torch.float64), (224,), (1,))

        args = (arg_0, arg_1, arg_2) + (sentinel,)
        result_original = fuzzed_program(*args)
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)
        self.assertTrue(torch.allclose(result_compiled, result_original))

    def test_inductor_view_issue_172693(self):
        # [FUNCTION DOC] Regression test for issue 172693: a module reshaping a strided key with an
        # inferred dimension, then a graph break, then adding one is compiled with dynamic shapes
        # in inference mode; the compiled output must match eager.
        class MinimalBugModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a head count attribute whose forward reshapes the key
            # to batch, heads and an inferred last dim, forces a graph break, and adds one.
            def __init__(self, num_heads: int = 8):
                # [FUNCTION DOC] HELPER: Stores the number of heads (default 8) on the module.
                super().__init__()
                self.num_heads = num_heads

            def forward(self, key):
                # [FUNCTION DOC] HELPER: Reshapes key to its first size by num_heads by inferred,
                # calls a dynamo graph break, and returns the reshaped tensor plus one.
                reshaped_key = key.reshape(key.shape[0], self.num_heads, -1)
                torch._dynamo.graph_break()
                return reshaped_key + 1

        num_heads = 8

        bug_module = MinimalBugModule(num_heads=num_heads)

        key = torch.empty_strided((2, 16, 9), (288, 9, 1)).normal_()

        with torch.inference_mode():
            eager_result = bug_module(key)
            bug_module.compile(dynamic=True)
            compiled_result = bug_module(key)

        self.assertEqual(compiled_result, eager_result)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_view_as_complex_unbacked_no_dde(self):
        # [FUNCTION DOC] Compiles view-as-complex on a tensor whose batch dimension is unbacked
        # with the eager backend and full graph; no data-dependent error must be raised and the
        # real view of the result must match eager.
        """
        Test that view_as_complex does not raise GuardOnDataDependentSymNode
        when operating on tensors with unbacked dimensions.
        """

        def func(x):
            # [FUNCTION DOC] HELPER: Returns x viewed as complex.
            return torch.view_as_complex(x)

        # Create a tensor with shape [..., 2] where batch dim is unbacked
        x = torch.rand(4, 8, 2)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        torch._dynamo.reset()
        # This should not raise GuardOnDataDependentSymNode
        compiled_func = torch.compile(func, fullgraph=True, backend="eager")
        result = compiled_func(x)
        expected = torch.view_as_complex(x)
        self.assertTrue(
            torch.allclose(torch.view_as_real(result), torch.view_as_real(expected))
        )

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_view_as_real_backward_unbacked_no_dde(self):
        # [FUNCTION DOC] Compiles a function summing view-as-real of a complex tensor requiring
        # grad with an unbacked batch dimension, then runs backward; must not raise a
        # data-dependent error.
        """
        Test that view_as_real backward (which calls view_as_complex) does not
        raise GuardOnDataDependentSymNode when operating on tensors with
        unbacked dimensions.
        """

        def func(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the real view of x.
            y = torch.view_as_real(x)
            return y.sum()

        x = torch.rand(4, 8, dtype=torch.complex64, requires_grad=True)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        torch._dynamo.reset()
        # This should not raise GuardOnDataDependentSymNode during backward
        compiled_func = torch.compile(func, fullgraph=True, backend="eager")
        result = compiled_func(x)
        result.backward()

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_mark_unbacked_view_input(self):
        # [FUNCTION DOC] Marks a tensor's dims unbacked, passes a slice view of it to a compiled
        # function and checks the output shape is one by one.
        @torch.compile()
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x times 100 plus 1.
            return x * 100 + 1

        x = torch.rand(10, 10)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        y = x[0:1, 0:1]  # y is a view, y._base is x
        result = fn(y)
        self.assertEqual(result.shape, (1, 1))


instantiate_parametrized_tests(TestUnbacked)


class TestMaybeFastEvalComparison(TestCase):
    # [CLASS DOC] Tests for the ShapeEnv fast path for comparing sums of non-negative symbols to
    # zero, plus printing of min/max expressions for guards and an unbacked slice assignment
    # regression test.
    """Tests for _maybe_fast_eval_comparison fast path optimization."""

    def test_sum_of_nonneg_ge_zero(self):
        # [FUNCTION DOC] Checks the fast comparison returns the true expression for the sum of two
        # non-negative unbacked symbols being at least zero.
        """Test that sum of non-negative symbols >= 0 returns True."""
        shape_env = ShapeEnv()
        # Create unbacked symbols and constrain them to be positive
        u0 = shape_env.create_unbacked_symint()
        u1 = shape_env.create_unbacked_symint()
        torch._check(u0 >= 0)
        torch._check(u1 >= 0)

        # u0 + u1 >= 0 should be True (both have range [0, inf])
        expr = sympy.GreaterThan(u0.node.expr + u1.node.expr, 0)
        result = shape_env._maybe_fast_eval_comparison(expr)
        self.assertEqual(result, sympy.true)

    def test_single_nonneg_symbol_ge_zero(self):
        # [FUNCTION DOC] Checks the fast comparison returns true for a single non-negative unbacked
        # symbol being at least zero.
        """Test that a single non-negative symbol >= 0 returns True."""
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        torch._check(u0 >= 0)

        # u0 >= 0 should be True (u0 has range [0, inf])
        expr = sympy.GreaterThan(u0.node.expr, 0)
        result = shape_env._maybe_fast_eval_comparison(expr)
        self.assertEqual(result, sympy.true)

    def test_zero_le_sum_of_nonneg(self):
        # [FUNCTION DOC] Checks the fast comparison returns true for zero being at most the sum of
        # two non-negative symbols.
        """Test that 0 <= sum of non-negative symbols returns True."""
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        u1 = shape_env.create_unbacked_symint()
        torch._check(u0 >= 0)
        torch._check(u1 >= 0)

        # 0 <= u0 + u1 should be True
        expr = sympy.Le(0, u0.node.expr + u1.node.expr)
        result = shape_env._maybe_fast_eval_comparison(expr)
        self.assertEqual(result, sympy.true)

    def test_nonneg_constant_in_sum(self):
        # [FUNCTION DOC] Checks the fast comparison returns true when a non-negative symbol plus
        # the constant 5 is compared to zero.
        """Test that sum with non-negative constant >= 0 returns True."""
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        torch._check(u0 >= 0)

        # u0 + 5 >= 0 should be True
        expr = sympy.GreaterThan(u0.node.expr + 5, 0)
        result = shape_env._maybe_fast_eval_comparison(expr)
        self.assertEqual(result, sympy.true)

    def test_negative_constant_returns_none(self):
        # [FUNCTION DOC] Checks the fast comparison returns None when a negative constant is added
        # since the result is undecidable.
        """Test that sum with negative constant returns None (undecidable)."""
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        torch._check(u0 >= 0)

        # u0 - 5 >= 0 is undecidable (u0 could be 0, making it -5)
        expr = sympy.GreaterThan(u0.node.expr - 5, 0)
        result = shape_env._maybe_fast_eval_comparison(expr)
        self.assertIsNone(result)

    def test_sympy_interp_min_max_are_symbol_safe(self):
        # [FUNCTION DOC] Prints max and min expressions with the guard printer and checks the text
        # uses the symbol-safe two-argument max and min calls; evaluates them in the interpreter
        # namespace on an unbacked symbolic int and compares the sympy results; also checks a
        # three-argument max prints as nested binary calls and evaluates to the right expression.
        from torch._dynamo.source import LocalSource
        from torch.fx.experimental.symbolic_shapes import ShapeGuardPythonPrinter
        from torch.utils._sympy.functions import Max, Min

        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        sym = u0.node.expr
        source = LocalSource("u0")
        printer = ShapeGuardPythonPrinter(
            {sym: [source]}, lambda source: source.name, {sym: [source]}
        )

        # Printer emits torch.sym_max / torch.sym_min so the printed string is
        # symbol-safe to eval on unbacked SymInts.
        max_expr = printer.doprint(sympy.Max(1, sym))
        min_expr = printer.doprint(sympy.Min(1, sym))
        self.assertEqual(max_expr, "torch.sym_max(1, L['u0'])")
        self.assertEqual(min_expr, "torch.sym_min(1, L['u0'])")

        locals_ = {"L": {"u0": u0}}
        max_result = eval(max_expr, SYMPY_INTERP, locals_).node.expr  # noqa: P204
        min_result = eval(min_expr, SYMPY_INTERP, locals_).node.expr  # noqa: P204
        self.assertEqual(max_result, Max(1, sym))
        self.assertEqual(min_result, Min(1, sym))

        # Three-arg form right-folds into nested binary calls.
        u1 = shape_env.create_unbacked_symint()
        sym1 = u1.node.expr
        printer3 = ShapeGuardPythonPrinter(
            {sym: [source], sym1: [LocalSource("u1")]},
            lambda s: s.name,
            {sym: [source], sym1: [LocalSource("u1")]},
        )
        max3 = printer3.doprint(sympy.Max(1, sym, sym1))
        self.assertIn("torch.sym_max", max3)
        locals3 = {"L": {"u0": u0, "u1": u1}}
        max3_result = eval(max3, SYMPY_INTERP, locals3).node.expr  # noqa: P204
        self.assertEqual(max3_result, Max(1, sym, sym1))

    def test_non_zero_rhs_returns_none(self):
        # [FUNCTION DOC] Checks the fast comparison returns None when the right side is a
        # nonzero constant.
        """Test that comparison with non-zero RHS returns None."""
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        torch._check(u0 >= 0)

        # u0 >= 5 is not handled by this fast path
        expr = sympy.GreaterThan(u0.node.expr, 5)
        result = shape_env._maybe_fast_eval_comparison(expr)
        self.assertIsNone(result)

    def test_unhandled_expr_type_returns_none(self):
        # [FUNCTION DOC] Checks the fast comparison returns None for an equality expression, which
        # it does not handle.
        """Test that unhandled expression types return None."""
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        torch._check(u0 >= 0)

        # Eq is not handled by the current fast path
        expr = sympy.Eq(u0.node.expr, 0)
        result = shape_env._maybe_fast_eval_comparison(expr)
        self.assertIsNone(result)

    def test_unbacked_symbol_with_default_range_returns_none(self):
        # [FUNCTION DOC] Checks the fast comparison returns None for an unbacked symbol with the
        # default unbounded range.
        """Test that unbacked symbols with default range [-inf, inf] return None."""
        shape_env = ShapeEnv()
        # Default unbacked symint has range [-int_oo, int_oo]
        u0 = shape_env.create_unbacked_symint()

        # u0 >= 0 is undecidable since u0 could be negative
        expr = sympy.GreaterThan(u0.node.expr, 0)
        result = shape_env._maybe_fast_eval_comparison(expr)
        self.assertIsNone(result)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_slice_assignment_same_bounds(self):
        # [FUNCTION DOC] Regression test (MoE sequential expert matmul pattern): compiles with
        # scalar capture and the eager backend a loop that slices rows by data-dependent start and
        # end tensors, multiplies, and writes the result into the same slice of an output; must not
        # fail with a data-dependent guard and must match eager.
        # Repro from sequential_experts_gemm (MoE pattern in HuggingFace Aria).
        # output[start:end] = out should not fail with a data-dependent guard
        # because `out` was produced from token_states[start:end], so both
        # slices share the same (start, end) and thus the same size.
        def sequential_experts_gemm(token_states, expert_weights, tokens_per_expert):
            # [FUNCTION DOC] HELPER: Allocates a zeros output, computes cumulative token offsets
            # with a leading zero, and for each expert multiplies its slice of tokens by that
            # expert's weights and writes the result back into the same slice of the output;
            # returns the output.
            num_tokens = token_states.shape[0]
            out_features = expert_weights.shape[-1]
            output = torch.zeros(
                num_tokens,
                out_features,
                dtype=token_states.dtype,
                device=token_states.device,
            )

            cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
            zero_tensor = torch.zeros(
                1, dtype=torch.long, device=cumsum_num_tokens.device
            )
            cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

            for expert_num in range(expert_weights.shape[0]):
                start = cumsum_num_tokens[expert_num]
                end = cumsum_num_tokens[expert_num + 1]
                tokens = token_states[start:end]
                out = torch.matmul(tokens, expert_weights[expert_num])
                output[start:end] = out
            return output

        num_tokens = 10
        in_features = 16
        out_features = 32
        num_experts = 3

        token_states = torch.randn(num_tokens, in_features)
        expert_weights = torch.randn(num_experts, in_features, out_features)
        tokens_per_expert = torch.tensor([3, 2, 5])

        eager_result = sequential_experts_gemm(
            token_states, expert_weights, tokens_per_expert
        )

        compiled_fn = torch.compile(
            sequential_experts_gemm, fullgraph=True, backend="eager"
        )
        compiled_result = compiled_fn(token_states, expert_weights, tokens_per_expert)

        self.assertEqual(eager_result, compiled_result)


class TestTransferSymbolsFromForeignShapeEnv(TestCase):
    # [CLASS DOC] Tests for transferring symbols from a foreign ShapeEnv into another one, covering
    # backed, static and unbacked symbols, derived expressions, hints, and a flex attention
    # end-to-end case. Helper methods build sources, transfer values and create symbols.
    """Tests for ShapeEnv.transfer_symbols_from_foreign_shape_env."""

    def _make_source(self, name="t"):
        # [FUNCTION DOC] HELPER: Returns a constant source with the given name (default t).
        from torch._dynamo.source import ConstantSource

        return ConstantSource(name)

    @staticmethod
    def _transfer_symint(env, value, source):
        # [FUNCTION DOC] HELPER: Static method mimicking how a raw foreign unbacked SymInt input is
        # wrapped: transfers the expression as an unbacked symbol into the given env and returns a
        # SymInt node in that env, without a hint, built with the given source.
        """Test helper that mirrors what builder.py does when wrapping a raw
        foreign unbacked SymInt input: transfer the expression and wrap as a
        SymInt in the local env."""
        new_expr = env._transfer_foreign_expr_as_unbacked(value, source=source)
        return env.create_symintnode(new_expr, hint=None, source=source)

    def _create_backed_symbols(self, shape_env, tensor, source_name="foreign"):
        # [FUNCTION DOC] HELPER: Creates backed symbolic sizes, strides and storage offset for a
        # real tensor in the given shape env using duck sizes and inferred strides with a constant
        # source of the given name (default foreign). Returns the three results of the env call.
        """Create backed symbolic sizes/strides/offset using _create_symbolic_sizes_strides_storage_offset."""
        src = self._make_source(source_name)
        return shape_env._create_symbolic_sizes_strides_storage_offset(
            tensor.size(),
            tensor.stride(),
            tensor.storage_offset(),
            [True] * tensor.dim(),
            src,
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DUCK] * tensor.dim(),
                dynamic_strides=[DimDynamic.INFER_STRIDE] * tensor.dim(),
            ),
        )

    def test_backed_symbols_transferred_as_duck(self):
        # [FUNCTION DOC] Creates backed symbols in a foreign env for a 4 by 8 tensor, transfers
        # them into a new env, and checks the new sizes are symbolic nodes owned by the new env
        # with hints 4 and 8.
        """Backed (guarding-hinted) symbols should become DUCK dims in the new env."""
        foreign_env = ShapeEnv()
        x = torch.randn(4, 8)
        sizes, strides, offset = self._create_backed_symbols(foreign_env, x)
        # Sanity: foreign sizes are symbolic
        self.assertTrue(is_symbolic(sizes[0]))
        self.assertTrue(is_symbolic(sizes[1]))

        local_env = ShapeEnv()
        new_sizes, new_strides, new_offset = (
            local_env.transfer_symbols_from_foreign_shape_env(
                sizes,
                strides,
                offset,
                source=self._make_source("local"),
            )
        )
        # New symbols should exist in local_env, not foreign_env
        for s in new_sizes:
            self.assertTrue(is_symbolic(s))
            self.assertIs(s.node.shape_env, local_env)
        # Hints should match the original concrete values
        self.assertEqual(guarding_hint_or_throw(new_sizes[0].node), 4)
        self.assertEqual(guarding_hint_or_throw(new_sizes[1].node), 8)

    def test_static_dims_stay_static(self):
        # [FUNCTION DOC] Transfers plain int sizes and strides into a new env and checks sizes stay
        # plain ints equal to the inputs.
        """Non-symbolic (plain int) sizes should remain static ints."""
        local_env = ShapeEnv()

        sizes = (3, 5)
        strides = (5, 1)
        storage_offset = 0

        new_sizes, new_strides, new_offset = (
            local_env.transfer_symbols_from_foreign_shape_env(
                sizes,
                strides,
                storage_offset,
                source=self._make_source("local"),
            )
        )
        # Static dims should stay as plain ints
        for s in new_sizes:
            self.assertFalse(is_symbolic(s))
        self.assertEqual(new_sizes, (3, 5))

    def test_unbacked_symbols_transferred(self):
        # [FUNCTION DOC] Transfers two foreign unbacked symbols and checks the new sizes are
        # symbolic, owned by the new env, and have no guarding hint.
        """Unbacked symbols should become UNBACKED dims in the new env."""
        foreign_env = ShapeEnv()
        u0 = foreign_env.create_unbacked_symint()
        u1 = foreign_env.create_unbacked_symint()

        # Use plain int strides; unbacked symbols as strides would need
        # valid positive hints, and INFER_STRIDE handles stride creation.
        sizes = (u0, u1)
        strides = (1, 1)
        storage_offset = 0

        local_env = ShapeEnv()
        new_sizes, new_strides, new_offset = (
            local_env.transfer_symbols_from_foreign_shape_env(
                sizes,
                strides,
                storage_offset,
                source=self._make_source("local"),
            )
        )
        # New symbols should be unbacked in local_env
        for s in new_sizes:
            self.assertTrue(is_symbolic(s))
            self.assertIs(s.node.shape_env, local_env)
            self.assertFalse(local_env.has_guarding_hint(s.node.expr))

    def test_shared_unbacked_symbol_preserved(self):
        # [FUNCTION DOC] Transfers sizes that reuse one foreign unbacked symbol twice, with that
        # symbol also as a stride, and checks both new sizes share one symbol and the stride equals
        # the shared size symbol.
        """When two dims share the same unbacked symbol (e.g. size=[u0, u0]),
        the transfer preserves that sharing."""
        foreign_env = ShapeEnv()
        u0 = foreign_env.create_unbacked_symint()

        sizes = (u0, u0)
        strides = (u0, 1)
        storage_offset = 0

        local_env = ShapeEnv()
        new_sizes, new_strides, new_offset = (
            local_env.transfer_symbols_from_foreign_shape_env(
                sizes,
                strides,
                storage_offset,
                source=self._make_source("local"),
            )
        )
        # Both dims should map to the same new symbol since they were
        # the same unbacked symbol in the foreign env.
        self.assertTrue(is_symbolic(new_sizes[0]))
        self.assertTrue(is_symbolic(new_sizes[1]))
        self.assertEqual(
            new_sizes[0].node.expr,
            new_sizes[1].node.expr,
            "Shared unbacked symbol should be preserved across dims",
        )
        # Stride[0] should equal the shared size symbol (contiguous relationship)
        self.assertTrue(is_symbolic(new_strides[0]))
        self.assertEqual(
            new_strides[0].node.expr,
            new_sizes[0].node.expr,
            "Stride should preserve relationship with size via substitution",
        )

    def test_foreign_stride_symbol_not_in_sizes(self):
        # [FUNCTION DOC] Transfers a tensor whose stride is a foreign unbacked symbol absent from
        # the sizes and checks both size and stride become symbols owned by the new env, and they
        # are different symbols.
        """If a stride references a foreign unbacked symbol not in any size dim
        (e.g. from as_strided), a fresh unbacked symbol should be created."""
        foreign_env = ShapeEnv()
        u_size = foreign_env.create_unbacked_symint()
        u_stride = foreign_env.create_unbacked_symint()  # independent from sizes

        sizes = (u_size,)
        strides = (u_stride,)  # stride symbol not in sizes
        storage_offset = 0

        local_env = ShapeEnv()
        new_sizes, new_strides, new_offset = (
            local_env.transfer_symbols_from_foreign_shape_env(
                sizes,
                strides,
                storage_offset,
                source=self._make_source("local"),
            )
        )
        # Size should be unbacked in local_env
        self.assertTrue(is_symbolic(new_sizes[0]))
        self.assertIs(new_sizes[0].node.shape_env, local_env)
        # Stride should also be unbacked in local_env (fresh symbol, not foreign)
        self.assertTrue(is_symbolic(new_strides[0]))
        self.assertIs(new_strides[0].node.shape_env, local_env)
        # Stride should be a different symbol than size
        self.assertNotEqual(new_sizes[0].node.expr, new_strides[0].node.expr)

    def test_derived_unbacked_expression_preserved(self):
        # [FUNCTION DOC] Transfers a size and a floor-half derived size of the same foreign
        # symbol and checks the derived dimension is built from the new base symbol rather
        # than independent.
        """If a size is a derived expression (e.g., u0 // 2) from a base symbol
        that appears in another dim, it should be expressed in terms of the
        new symbol rather than creating an independent fresh symbol."""
        foreign_env = ShapeEnv()
        u0 = foreign_env.create_unbacked_symint()
        # Create a derived expression: u0 // 2
        u0_half = u0 // 2

        sizes = (u0, u0_half)
        strides = (1, 1)
        storage_offset = 0

        local_env = ShapeEnv()
        new_sizes, new_strides, new_offset = (
            local_env.transfer_symbols_from_foreign_shape_env(
                sizes,
                strides,
                storage_offset,
                source=self._make_source("local"),
            )
        )
        # dim 0 should be a fresh unbacked symbol
        self.assertTrue(is_symbolic(new_sizes[0]))
        new_u0 = new_sizes[0].node.expr
        # dim 1 should be derived from new_u0 (e.g., floor(new_u0/2))
        self.assertTrue(is_symbolic(new_sizes[1]))
        # The derived expression should reference the new base symbol
        self.assertIn(new_u0, new_sizes[1].node.expr.free_symbols)

    def test_unbacked_hint_overrides_transferred(self):
        # [FUNCTION DOC] Sets a hint override of 42 on a foreign unbacked symbol and checks the
        # transferred symbol in the new env has the same override.
        """Hint overrides on unbacked symbols in the foreign env should be
        transferred to the new env."""
        foreign_env = ShapeEnv()
        u0 = foreign_env.create_unbacked_symint()
        # Set a hint override on u0 in the foreign env
        foreign_env.var_to_hint_override[u0.node.expr] = 42

        sizes = (u0,)
        strides = (1,)
        storage_offset = 0

        local_env = ShapeEnv()
        new_sizes, new_strides, new_offset = (
            local_env.transfer_symbols_from_foreign_shape_env(
                sizes,
                strides,
                storage_offset,
                source=self._make_source("local"),
            )
        )
        # The hint override should be transferred to local_env
        new_sym = new_sizes[0]
        self.assertTrue(is_symbolic(new_sym))
        self.assertIn(new_sym.node.expr, local_env.var_to_hint_override)
        self.assertEqual(local_env.var_to_hint_override[new_sym.node.expr], 42)

    def test_mixed_static_backed_unbacked(self):
        # [FUNCTION DOC] Transfers a static size, a backed symbol and an unbacked symbol with a
        # hint override of 99 together and checks each is handled: static stays an int, the backed
        # one is symbolic with hint 4 in the new env, and the unbacked one has no guarding hint but
        # carries the override.
        """A mix of static, backed, and unbacked dims should all be handled correctly."""
        foreign_env = ShapeEnv()
        # Create a backed symbol
        x = torch.randn(4)
        backed_sizes, _, _ = self._create_backed_symbols(foreign_env, x)
        backed_sym = backed_sizes[0]  # backed, hint=4

        # Create an unbacked symbol
        unbacked_sym = foreign_env.create_unbacked_symint()
        foreign_env.var_to_hint_override[unbacked_sym.node.expr] = 99

        sizes = (7, backed_sym, unbacked_sym)  # static, backed, unbacked
        strides = (1, 1, 1)
        storage_offset = 0

        local_env = ShapeEnv()
        new_sizes, new_strides, new_offset = (
            local_env.transfer_symbols_from_foreign_shape_env(
                sizes,
                strides,
                storage_offset,
                source=self._make_source("local"),
            )
        )
        # dim 0: static
        self.assertFalse(is_symbolic(new_sizes[0]))
        self.assertEqual(new_sizes[0], 7)

        # dim 1: backed/duck
        self.assertTrue(is_symbolic(new_sizes[1]))
        self.assertIs(new_sizes[1].node.shape_env, local_env)
        self.assertEqual(guarding_hint_or_throw(new_sizes[1].node), 4)

        # dim 2: unbacked with hint override
        self.assertTrue(is_symbolic(new_sizes[2]))
        self.assertIs(new_sizes[2].node.shape_env, local_env)
        self.assertFalse(local_env.has_guarding_hint(new_sizes[2].node.expr))
        self.assertEqual(local_env.var_to_hint_override[new_sizes[2].node.expr], 99)

    def test_raw_unbacked_symint_transfer_shares_tensor_symbol(self):
        # [FUNCTION DOC] Transfers a tensor dimension and then raw symbolic ints (the same symbol
        # and that symbol plus one) from one foreign env and checks they share the same new symbol
        # (plus one for the derived one) and the hint override 32 is carried over.
        """Tensor dims and raw SymInts from one foreign ShapeEnv share transfer."""
        foreign_env = ShapeEnv()
        u0 = foreign_env.create_unbacked_symint()
        foreign_env.var_to_hint_override[u0.node.expr] = 32

        local_env = ShapeEnv()
        new_sizes, _, _ = local_env.transfer_symbols_from_foreign_shape_env(
            (u0,),
            (1,),
            0,
            source=self._make_source("tensor"),
        )
        raw_u0 = self._transfer_symint(local_env, u0, source=self._make_source("raw"))
        raw_u0_plus_1 = self._transfer_symint(
            local_env, u0 + 1, source=self._make_source("raw_plus_1")
        )

        tensor_symbol = new_sizes[0].node.expr
        self.assertEqual(raw_u0.node.expr, tensor_symbol)
        self.assertEqual(raw_u0_plus_1.node.expr, tensor_symbol + 1)
        self.assertEqual(local_env.var_to_hint_override[tensor_symbol], 32)

    def test_raw_unbacked_symint_transfer_preserves_multi_symbol_expr(self):
        # [FUNCTION DOC] Transfers two foreign unbacked symbols and their sum as tensor sizes, then
        # transfers the sum as a raw symbolic int, and checks the sum expression is rebuilt from
        # the new base symbols in both paths and hint overrides 32 and 16 are carried over.
        """Derived raw SymInts are rebuilt from transferred base symbols."""
        foreign_env = ShapeEnv()
        u0 = foreign_env.create_unbacked_symint()
        u1 = foreign_env.create_unbacked_symint()
        foreign_env.var_to_hint_override[u0.node.expr] = 32
        foreign_env.var_to_hint_override[u1.node.expr] = 16

        local_env = ShapeEnv()
        new_sizes, _, _ = local_env.transfer_symbols_from_foreign_shape_env(
            (u0, u1, u0 + u1),
            (1, 1, 1),
            0,
            source=self._make_source("tensor"),
        )
        raw_sum = self._transfer_symint(
            local_env, u0 + u1, source=self._make_source("raw_sum")
        )

        new_u0 = new_sizes[0].node.expr
        new_u1 = new_sizes[1].node.expr
        self.assertEqual(new_sizes[2].node.expr, new_u0 + new_u1)
        self.assertEqual(raw_sum.node.expr, new_u0 + new_u1)
        self.assertEqual(local_env.var_to_hint_override[new_u0], 32)
        self.assertEqual(local_env.var_to_hint_override[new_u1], 16)

    def test_raw_unbacked_symint_transfer_falls_back_for_backed_leftover(self):
        # [FUNCTION DOC] Transfers a raw symbolic int mixing a foreign unbacked and backed
        # symbol and checks the result is a fresh unbacked local symbol with only local unbacked
        # free symbols.
        """Mixed backed+unbacked foreign expr falls back to fresh local unbacked."""
        foreign_env = ShapeEnv()
        u0 = foreign_env.create_unbacked_symint()
        # Create a backed foreign symbol that this env cannot translate.
        s0 = foreign_env.create_symbol(
            8,
            self._make_source("s0"),
            DimDynamic.DYNAMIC,
        )
        s0_symint = foreign_env.create_symintnode(s0, hint=8)
        mixed = u0 + s0_symint

        local_env = ShapeEnv()
        raw = self._transfer_symint(
            local_env, mixed, source=self._make_source("raw_mixed")
        )

        # Result is a fresh local unbacked symbol; no foreign symbols leaked.
        result_expr = raw.node.expr
        self.assertTrue(local_env.is_unbacked_symint(result_expr))
        # All free symbols are owned by local_env.
        for sym in result_expr.free_symbols:
            self.assertTrue(local_env.is_unbacked_symint(sym))

    def test_transfer_is_self_short_circuit(self):
        # [FUNCTION DOC] Checks transferring an expression already owned by the same env returns it
        # unchanged and adds nothing to the foreign unbacked symbol cache.
        """When the SymInt already belongs to this ShapeEnv,
        _transfer_foreign_expr_as_unbacked is a no-op (returns the input expr
        unchanged, no new symbol minted)."""
        local_env = ShapeEnv()
        local_u0 = local_env.create_unbacked_symint()

        cache_size_before = len(local_env.foreign_unbacked_symbol_cache)
        result = local_env._transfer_foreign_expr_as_unbacked(
            local_u0, source=self._make_source("local")
        )
        self.assertEqual(result, local_u0.node.expr)
        # Nothing should have been added to the cache.
        self.assertEqual(
            len(local_env.foreign_unbacked_symbol_cache), cache_size_before
        )

    def test_transfer_constrains_size_on_cache_hit(self):
        # [FUNCTION DOC] Transfers a foreign symbol as a raw symbolic int (not size-like), then as
        # a tensor size dimension, and checks the cached local symbol is reused and now appears in
        # the env's size-like set.
        """When is_size=True and the expression resolves to a single Symbol
        from the cache (cached via a non-size path earlier),
        _constrain_range_for_size is still applied; the symbol ends up in
        size_like."""
        foreign_env = ShapeEnv()
        u0 = foreign_env.create_unbacked_symint()

        local_env = ShapeEnv()
        # First: transfer u0 via the raw SymInt path (is_size=False).
        raw_symint = self._transfer_symint(
            local_env, u0, source=self._make_source("raw")
        )
        raw = raw_symint.node.expr
        self.assertNotIn(raw, local_env.size_like)

        # Now: transfer the same u0 as a tensor size dim (is_size=True via
        # transfer_symbols_from_foreign_shape_env's UNBACKED path).
        new_sizes, _, _ = local_env.transfer_symbols_from_foreign_shape_env(
            (u0,), (1,), 0, source=self._make_source("tensor")
        )
        # Same cached local symbol reused.
        self.assertEqual(new_sizes[0].node.expr, raw)
        # And now it carries the size lower-bound constraint.
        self.assertIn(raw, local_env.size_like)

    def test_foreign_unbacked_transfer_preserves_derived_tensor_expr(self):
        # [FUNCTION DOC] Transfers derived tensor dims (a halved symbol) and strides from a foreign
        # env with hint overrides, then transfers the same derived expression as a raw int, and
        # checks it reuses the same local symbol and the stride equals the second size.
        """Derived tensor dims are minted as opaque symbols; raw SymInt with
        the same expression reuses that symbol via the (env, expr) cache."""
        foreign_env = ShapeEnv()
        global_tokens = foreign_env.create_unbacked_symint()
        hidden = foreign_env.create_unbacked_symint()
        foreign_env.var_to_hint_override[global_tokens.node.expr] = 64
        foreign_env.var_to_hint_override[hidden.node.expr] = 128

        derived_tokens = global_tokens // 2
        local_env = ShapeEnv()
        new_sizes, new_strides, _ = local_env.transfer_symbols_from_foreign_shape_env(
            (derived_tokens, hidden),
            (hidden, 1),
            0,
            source=self._make_source("derived_tensor"),
        )
        raw_derived_tokens = self._transfer_symint(
            local_env, derived_tokens, source=self._make_source("raw_derived_tokens")
        )

        self.assertEqual(raw_derived_tokens.node.expr, new_sizes[0].node.expr)
        self.assertEqual(new_strides[0].node.expr, new_sizes[1].node.expr)
        self.assertEqual(len(raw_derived_tokens.node.expr.free_symbols), 1)
        derived_token_sym = next(iter(raw_derived_tokens.node.expr.free_symbols))
        # `derived_tokens` (= global_tokens // 2) is minted as one fresh
        # opaque symbol; its hint is the foreign expression's hint (64 // 2).
        self.assertEqual(local_env.var_to_hint_override[derived_token_sym], 32)
        self.assertEqual(local_env.var_to_hint_override[new_sizes[1].node.expr], 128)

    def test_foreign_unbacked_transfer_preserves_shared_token_grid(self):
        # [FUNCTION DOC] Transfers three overlapping tensor shapes (token grid, labels, derived
        # activation using a halved sequence and hidden size) and two raw symbolic ints from one
        # foreign env into a new env, and checks that shared batch, sequence and hidden symbols are
        # the same across transfers, strides follow sizes, and the derived raw expressions reuse
        # the symbols created for the tensor dims.
        """Separate tensor transfers preserve shared token-grid symbols."""
        foreign_env = ShapeEnv()
        batch = foreign_env.create_unbacked_symint()
        seq = foreign_env.create_unbacked_symint()
        hidden = foreign_env.create_unbacked_symint()
        foreign_env.var_to_hint_override[batch.node.expr] = 4
        foreign_env.var_to_hint_override[seq.node.expr] = 128
        foreign_env.var_to_hint_override[hidden.node.expr] = 64

        derived_seq = seq // 2
        local_env = ShapeEnv()
        token_grid_sizes, token_grid_strides, _ = (
            local_env.transfer_symbols_from_foreign_shape_env(
                (batch, seq),
                (seq, 1),
                0,
                source=self._make_source("token_grid"),
            )
        )
        label_sizes, label_strides, _ = (
            local_env.transfer_symbols_from_foreign_shape_env(
                (batch, seq),
                (seq, 1),
                0,
                source=self._make_source("labels"),
            )
        )
        derived_sizes, derived_strides, _ = (
            local_env.transfer_symbols_from_foreign_shape_env(
                (batch, derived_seq, hidden),
                (derived_seq * hidden, hidden, 1),
                0,
                source=self._make_source("derived_activation"),
            )
        )
        raw_derived_seq = self._transfer_symint(
            local_env, derived_seq, source=self._make_source("raw_derived_seq")
        )
        raw_seq_plus_hidden = self._transfer_symint(
            local_env, seq + hidden, source=self._make_source("raw_seq_plus_hidden")
        )

        self.assertEqual(token_grid_sizes[0].node.expr, label_sizes[0].node.expr)
        self.assertEqual(token_grid_sizes[0].node.expr, derived_sizes[0].node.expr)
        self.assertEqual(token_grid_sizes[1].node.expr, label_sizes[1].node.expr)
        self.assertEqual(token_grid_strides[0].node.expr, token_grid_sizes[1].node.expr)
        self.assertEqual(label_strides[0].node.expr, label_sizes[1].node.expr)
        self.assertEqual(raw_derived_seq.node.expr, derived_sizes[1].node.expr)
        self.assertEqual(
            derived_strides[0].node.expr,
            derived_sizes[1].node.expr * derived_sizes[2].node.expr,
        )
        self.assertEqual(
            raw_seq_plus_hidden.node.expr,
            token_grid_sizes[1].node.expr + derived_sizes[2].node.expr,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_flex_attention_foreign_fake_e2e(self):
        # [FUNCTION DOC] End-to-end test, skipped without CUDA. Builds a flex attention block mask
        # with unbacked index dimensions and a CUDA mask function, makes fake tensors in an outer
        # fake mode, then traces a compiled flex attention call with make_fx inside that mode
        # (nested-compile error disabled temporarily). Checks tracing works and the captured graph
        # placeholders keep at least four unbacked, hint-free symbolic dimensions. Registers pytree
        # support for BlockMask and the mask wrapper if missing (global side effect).
        """E2E test: trace flex_attention with BlockMask containing unbacked dims
        through a fresh FakeTensorMode, exercising the foreign ShapeEnv transfer path."""
        from contextlib import contextmanager

        import torch.utils._pytree as pytree
        from torch._dynamo.decorators import mark_unbacked
        from torch._guards import tracing, TracingContext
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import (
            RelaxedUnspecConstraint,
            TrackedFake,
        )
        from torch.fx.traceback import preserve_node_meta
        from torch.nn.attention.flex_attention import (
            _MaskModWrapper,
            AuxRequest,
            BlockMask,
            flex_attention,
        )

        if BlockMask not in pytree.SUPPORTED_NODES:
            pytree.register_pytree_node(
                BlockMask,
                BlockMask._flatten,
                BlockMask._unflatten,
                flatten_with_keys_fn=BlockMask._flatten_with_keys,
                serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
            )
        if _MaskModWrapper not in pytree.SUPPORTED_NODES:
            pytree.register_constant(_MaskModWrapper)

        @contextmanager
        def skip_nested_compile():
            # [FUNCTION DOC] HELPER: Context manager that turns off the dynamo config error on
            # nested fx tracing for its body and restores the previous value afterward, even
            # on exceptions.
            prev = torch._dynamo.config.error_on_nested_fx_trace
            torch._dynamo.config.error_on_nested_fx_trace = False
            try:
                yield
            finally:
                torch._dynamo.config.error_on_nested_fx_trace = prev

        def copy_marks(src, dst):
            # [FUNCTION DOC] HELPER: Copies dynamo marking attributes (unbacked indices, strict
            # unbacked indices, bounds, shape ids, hint overrides, specialize-on) that exist on the
            # source tensor onto the destination tensor and returns the destination.
            for key in (
                "_dynamo_unbacked_indices",
                "_dynamo_strict_unbacked_indices",
                "_dynamo_unbacked_bounds",
                "_dynamo_shape_ids",
                "_dynamo_hint_overrides",
                "_specialize_on",
            ):
                if hasattr(src, key):
                    setattr(dst, key, getattr(src, key))
            return dst

        def _symbolic_context(t):
            # [FUNCTION DOC] HELPER: Builds a stateless symbolic context for a tensor from its
            # dynamo marks: returns None if no dimension is marked unbacked or strictly unbacked;
            # otherwise dimensions are static except marked ones, which are unbacked (strict ones
            # also get a relaxed unspecialized constraint), carrying shape ids and bounds.
            marked = getattr(t, "_dynamo_unbacked_indices", set())
            strict = getattr(t, "_dynamo_strict_unbacked_indices", set())
            if not marked and not strict:
                return None
            dynamic_sizes = [DimDynamic.STATIC] * t.dim()
            constraint_sizes = [None] * t.dim()
            for i in range(t.dim()):
                if i in marked:
                    dynamic_sizes[i] = DimDynamic.UNBACKED
                elif i in strict:
                    dynamic_sizes[i] = DimDynamic.UNBACKED
                    constraint_sizes[i] = RelaxedUnspecConstraint(warn_only=False)
            return StatelessSymbolicContext(
                dynamic_sizes=dynamic_sizes,
                constraint_sizes=constraint_sizes,
                specialize_on=[[] for _ in range(t.dim())],
                shape_ids=getattr(t, "_dynamo_shape_ids", None),
                unbacked_bounds=getattr(t, "_dynamo_unbacked_bounds", None),
            )

        def fakeify(fake_mode, x, name):
            # [FUNCTION DOC] HELPER: Converts a value to a fake tensor in the given fake mode.
            # Non-tensors are returned unchanged. A tensor with unbacked marks is faked with an
            # input local source and its symbolic context, and recorded in the shape env's tracked
            # fakes; any other tensor is faked with static shapes.
            if not isinstance(x, torch.Tensor):
                return x
            from torch._dynamo.source import LocalSource

            source = LocalSource(name, is_input=True)
            ctx = _symbolic_context(x)
            if ctx is not None:
                fake = fake_mode.from_tensor(x, source=source, symbolic_context=ctx)
                fake_mode.shape_env.tracked_fakes.append(TrackedFake(fake, source, ctx))
                return fake
            return fake_mode.from_tensor(x, static_shapes=True)

        ntoks, block_size = 8192, 128
        nblocks = (ntoks + block_size - 1) // block_size
        width = 2

        kv_indices = (
            torch.arange(width, dtype=torch.int32).expand(nblocks, width).clone()
        )
        full_kv_indices = kv_indices.clone()
        q_indices = kv_indices.clone()
        full_q_indices = kv_indices.clone()
        kv_num_blocks = torch.full((nblocks,), width, dtype=torch.int32)
        full_kv_num_blocks = kv_num_blocks.clone()
        q_num_blocks = kv_num_blocks.clone()
        full_q_num_blocks = kv_num_blocks.clone()

        mark_unbacked(kv_indices, 1)
        mark_unbacked(full_kv_indices, 1)
        mark_unbacked(q_indices, 1)
        mark_unbacked(full_q_indices, 1)

        attn_regions = torch.arange(ntoks, dtype=torch.int32, device="cuda")
        document_ids = torch.zeros(ntoks, dtype=torch.int32, device="cuda")

        def mask_mod(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: Block mask function combining causal order with equal
            # attention region and equal document id; returns the combined boolean mask.
            return (
                (q_idx >= kv_idx)
                & (attn_regions[q_idx] == attn_regions[kv_idx])
                & (document_ids[q_idx] == document_ids[kv_idx])
            )

        block_mask = BlockMask(
            kv_num_blocks=kv_num_blocks.to("cuda"),
            kv_indices=copy_marks(kv_indices, kv_indices.to("cuda")),
            full_kv_num_blocks=full_kv_num_blocks.to("cuda"),
            full_kv_indices=copy_marks(full_kv_indices, full_kv_indices.to("cuda")),
            q_num_blocks=q_num_blocks.to("cuda"),
            q_indices=copy_marks(q_indices, q_indices.to("cuda")),
            full_q_num_blocks=full_q_num_blocks.to("cuda"),
            full_q_indices=copy_marks(full_q_indices, full_q_indices.to("cuda")),
            BLOCK_SIZE=(block_size, block_size),
            mask_mod=mask_mod,
            seq_lengths=(ntoks, ntoks),
        )

        q = torch.randn(1, 4, ntoks, 128, device="cuda")
        k = torch.randn(1, 4, ntoks, 128, device="cuda")
        v = torch.randn(1, 4, ntoks, 128, device="cuda")
        cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)

        flat_args, spec = pytree.tree_flatten([q, k, v, block_mask])
        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(tracked_fakes=[]),
        )
        fake_args = tuple(
            fakeify(fake_mode, x, f"inp_{i}") for i, x in enumerate(flat_args)
        )
        torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings(
            fake_mode.shape_env,
            fake_args,
        )

        def wrapped(*flat):
            # [FUNCTION DOC] HELPER: Unflattens the arguments into query, key, value and block
            # mask, runs the compiled flex attention requesting max scores, and returns the
            # detached sum of the output and the detached maximum of the max scores.
            q1, k1, v1, mask1 = pytree.tree_unflatten(list(flat), spec)
            out, aux = cflex(
                q1,
                k1,
                v1,
                block_mask=mask1,
                return_aux=AuxRequest(max_scores=True),
            )
            return out.sum().detach(), aux.max_scores.max().detach()

        # This should not raise — the fix ensures unbacked symbols from
        # the outer FakeTensorMode are correctly transferred into the
        # inner ShapeEnv created by torch.compile.
        with (
            fake_mode,
            tracing(TracingContext(fake_mode)),
            preserve_node_meta(),
            skip_nested_compile(),
            torch.compiler._non_strict_tracing_context(),
        ):
            gm = make_fx(
                wrapped,
                record_stack_traces=True,
                record_module_stack=False,
            )(*fake_args)

        # Verify unbacked dims are preserved in the captured graph.
        # Check placeholder shapes for unbacked symbols (u0, u1, etc.)
        unbacked_count = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val", None)
                if val is not None and hasattr(val, "size"):
                    for s in val.size():
                        if is_symbolic(s) and not s.node.has_hint():
                            unbacked_count += 1
        self.assertGreaterEqual(
            unbacked_count,
            4,
            lambda msg: f"{msg}\nExpected at least 4 unbacked dims but found {unbacked_count}",
        )


if __name__ == "__main__":
    run_tests()
