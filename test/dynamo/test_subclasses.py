# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for tensor subclasses: torch function subclasses, wrapper
# subclasses implementing the traceable-subclass protocol (flatten/unflatten, dispatch),
# nested/jagged tensors, ScaledTensor and TwoTensor, guards on subclass metadata,
# aliasing/view/mutation handling under aot_autograd, and dynamic shapes. The top of the
# module defines helper subclasses, recording backends and compile-check helpers shared
# by the test classes below. Importing it only defines classes and builds the view test
# case table; the module is run through the standard PyTorch test runner.
import abc
import functools
import itertools
import unittest
from functools import partial

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._functorch._aot_autograd.utils import make_boxed_compiler
from torch._functorch.compilers import min_cut_rematerialization_partition
from torch._higher_order_ops.wrap import wrap
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.nested._internal.nested_tensor import (
    jagged_from_list,
    jagged_from_tensor_and_lengths,
    nested_view_from_values_offsets,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    NestedTensorTestCase,
    parametrize,
    subtest,
)
from torch.testing._internal.triton_utils import requires_gpu_and_triton
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import return_and_correct_aliasing


def nontraceable_subclass(c):
    # [FUNCTION DOC] HELPER: Returns a Dynamo config patch (usable as decorator or
    # context manager) that registers the given class c as the sole entry in the
    # nontraceable_tensor_subclasses setting, so Dynamo treats instances of it as opaque.
    return torch._dynamo.config.patch("nontraceable_tensor_subclasses", {c})


def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
    # [FUNCTION DOC] HELPER: Test assertion helper. Runs the module-level recompile
    # probe on the given function with two input sets, then asserts on the test
    # case that the observed outcome equals expected_recompiles (a boolean).
    actual_recompiles = _recompiles_for_inputs(fn, inputs1, inputs2)
    self.assertEqual(actual_recompiles, expected_recompiles)


def get_jagged_tensor(nested_size, offsets, requires_grad=True):
    # [FUNCTION DOC] HELPER: Builds a jagged nested tensor. nested_size is a pair whose first
    # item gives the length of each constituent and whose second is the shared feature dim D.
    # Each constituent is a random float64 tensor of shape length by D with the given
    # requires_grad; they are combined with jagged_from_list using the given offsets. Returns
    # whatever jagged_from_list returns (callers unpack it as a pair of tensor and offsets).
    # Makes a jagged tensor with N constituent tensors with size
    # as specified ((S0, S1, S2), D)
    D = nested_size[1]
    out = []
    for s in nested_size[0]:
        out.append(torch.randn(s, D, requires_grad=requires_grad, dtype=torch.float64))
    return jagged_from_list(out, offsets)


def get_view_test_cases():
    # [FUNCTION DOC] HELPER: Generator yielding pairs of a zero-argument factory and a
    # case name, covering view-of-jagged-tensor scenarios for both a nested-tensor base
    # and a dense base (basic non-leaf view, leaf views over all requires_grad
    # combinations, obscure intermediate-leaf view), plus subclass-to-dense and
    # dense/subclass chains. Factories are lazy so no tensors are created at collection
    # time; loop variables are bound with partial to avoid closure capture bugs.
    # Test all cases with both an NT base and a dense base
    # Subclass -> Subclass
    # Dense -> Subclass

    # NB: Don't close over loop variables, they will not get copied into the
    # closure
    #
    # NB: These return functions so we don't generate tensors during test
    # collection time

    def mk_basic(base_is_nt):
        # [FUNCTION DOC] HELPER: Factory for the basic view case: makes a requires_grad jagged
        # tensor (cloned when base_is_nt so it is a non-leaf), checks that it is not a leaf
        # (raises AssertionError otherwise) and returns an unsqueezed view of it on the last dim.
        # There are three cases to consider here based on the logic in
        # meta_utils.py
        #
        # (1) basic case:
        # view is not a leaf and has the same requires grad as its basic case
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)
        x = x.clone() if base_is_nt else x
        if x.is_leaf:
            raise AssertionError("Expected x to not be a leaf")
        return x.unsqueeze(-1)

    def mk_leaf(base_is_nt, requires_grad_1, requires_grad_2):
        # [FUNCTION DOC] HELPER: Factory for the leaf-view case: builds a jagged base with
        # requires_grad_1 (cloned if base_is_nt), then under no_grad takes an unsqueezed view
        # and sets its requires_grad flag to requires_grad_2, so the returned view is a leaf.
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=requires_grad_1)
        x = x.clone() if base_is_nt else x
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
            # The issue is this doesn't quite work
            x_view.requires_grad_(requires_grad_2)

        return x_view

    def mk_obscure(base_is_nt):
        # [FUNCTION DOC] HELPER: Factory for the obscure case: base without grad (cloned if
        # base_is_nt), a view taken under no_grad that is then made requires_grad, and a
        # further unsqueezed view of that view; returns this non-leaf view of a leaf view.
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=False)
        x = x.clone() if base_is_nt else x
        # intermediate leaf view
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
        x_view.requires_grad_(True)
        x_view_view = x_view.unsqueeze(-1)
        return x_view_view

    for base_is_nt in [False, True]:
        prefix = f"base_is_nt_{base_is_nt}"

        yield partial(mk_basic, base_is_nt), f"{prefix}_basic"

        # (2) leaf view case:
        # the view has to be a leaf (w/ requires_grad True or requires_grad False)
        # base w/ requires_grad True or requires_grad False
        for requires_grad_1, requires_grad_2 in itertools.product(
            [True, False], repeat=2
        ):
            yield (
                partial(mk_leaf, base_is_nt, requires_grad_1, requires_grad_2),
                f"{prefix}_leaf_{requires_grad_1}_{requires_grad_2}",
            )

        # (3) obscure case:
        # view is not a leaf (implies requires_grad True)
        # base w/ requires_grad False)
        yield partial(mk_obscure, base_is_nt), f"{prefix}_obscure"

    # Subclass -> Dense
    yield (
        lambda: get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[0].clone(),
        "subclass_dense",
    )

    # Dense -> Subclass -> Dense -> Subclass
    def mk_dense_subclass_dense_subclass():
        # [FUNCTION DOC] HELPER: Factory for a dense to subclass to dense to subclass chain:
        # nests a jagged tensor built from random dense values and fixed offsets inside
        # another one built from the values of the first, and returns the outer jagged tensor.
        values = torch.randn(10, 5)
        offsets = torch.tensor([0, 3, 6, 10])
        return nested_view_from_values_offsets(
            nested_view_from_values_offsets(values, offsets).values(), offsets
        )

    yield mk_dense_subclass_dense_subclass, "dense_subclass_dense_subclass"

    def mk_subclass_dense_subclass_dense():
        # [FUNCTION DOC] HELPER: Factory for a subclass to dense to subclass to
        # dense chain: clones a jagged tensor, builds a new jagged view from its
        # values and detached offsets and takes its values. Note it has no return
        # statement, so it returns None and the final dense tensor is discarded.
        x = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[0].clone()
        offsets2 = x.offsets().detach().clone()
        nested_view_from_values_offsets(x.values(), offsets2).values()

    yield mk_subclass_dense_subclass_dense, "subclass_dense_subclass_dense"


VIEW_TEST_CASES = {k: v for v, k in get_view_test_cases()}


compile_full_eager = torch.compile(backend="eager", fullgraph=True)


class BaseTorchFunction(torch.Tensor):
    # [CLASS DOC] HELPER: Plain tensor subclass that only overrides
    # __torch_function__ to delegate to the default Tensor implementation;
    # used to test tracing of a trivial torch-function subclass.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Classmethod that forwards the call to the
        # base class torch function handler unchanged and returns its result.
        return super().__torch_function__(func, types, args, kwargs)


class MockSubclass(torch.Tensor):
    # [CLASS DOC] HELPER: Tensor subclass whose __torch_function__ simply delegates
    # to the default implementation. Used as a minimal subclass in global test sets.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Delegates to the default Tensor
        # torch function handling and returns the result.
        return super().__torch_function__(func, types, args, kwargs)


class AttrSubclass(torch.Tensor):
    # [CLASS DOC] HELPER: Tensor subclass carrying two class-level int
    # attributes named x and size (both 10); size shadows the Tensor
    # size method. Its torch function just delegates to the default.
    x: int = 10
    size: int = 10

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Delegates to the default Tensor
        # torch function handling and returns the result.
        return super().__torch_function__(func, types, args, kwargs)


class DummyNDim(torch.Tensor):
    # [CLASS DOC] HELPER: Tensor subclass that makes the ndim property
    # report 10 for every instance by intercepting the property getter in
    # __torch_function__; all other functions use the default handling.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: If func is the getter for the Tensor ndim attribute, returns
        # the constant 10; otherwise delegates to the default torch function implementation.
        if func == torch.Tensor.ndim.__get__:
            return 10

        return super().__torch_function__(func, types, args, kwargs)


class WrapperSubclass:
    # [CLASS DOC] HELPER: Non-tensor class that holds a tensor and implements __torch_function__
    # by unwrapping every instance of itself in args and kwargs to the held tensor and calling
    # the function on the unwrapped values. The result is a plain tensor, not re-wrapped.
    def __init__(self, tensor):
        # [FUNCTION DOC] HELPER: Stores the wrapped tensor in the tensor attribute.
        self.tensor = tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Classmethod protocol hook. Defaults kwargs to an empty
        # mapping, replaces WrapperSubclass instances found anywhere in args and kwargs
        # with their tensors using pytree, then calls func on the result and returns it.
        if kwargs is None:
            kwargs = {}

        args = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, args)
        kwargs = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, kwargs)

        return func(*args, **kwargs)


class SigmoidToExpSubclass(torch.Tensor):
    # [CLASS DOC] HELPER: Tensor subclass that rewrites sigmoid into exp: its __torch_function__
    # redirects calls to Tensor.sigmoid to Tensor.exp and delegates everything else unchanged.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: If func is Tensor.sigmoid, runs the default handler with
        # Tensor.exp substituted; otherwise runs it with the original func. Returns the result.
        if func == torch.Tensor.sigmoid:
            return super().__torch_function__(torch.Tensor.exp, types, args, kwargs)

        return super().__torch_function__(func, types, args, kwargs)


# Wrapper subclass with two inner tensors: data and scale
# data has same shape as outer, and scale has single dim size
class ScaledTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Traceable wrapper subclass holding two inner tensors, _data (same
    # shape as the outer tensor) and _scale, plus an int _constant kept as metadata.
    # Implements __tensor_flatten__/__tensor_unflatten__ and a __torch_dispatch__ that
    # applies each op to _data only and rewraps the result with the same scale and constant.
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        *,
        constant: int = 0,
    ):
        # [FUNCTION DOC] HELPER: Creates the wrapper subclass object via _make_wrapper_subclass
        # with size, strides, storage offset, dtype, layout, requires_grad and device copied from
        # data. The scale and constant arguments are not used here (they are consumed by __init__).
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, constant: int = 0):
        # [FUNCTION DOC] HELPER: Stores data, scale and constant
        # in the attributes _data, _scale and _constant.
        self._data = data
        self._scale = scale
        self._constant = constant

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Flatten hook of the traceable subclass
        # protocol. Returns the list of inner tensor attribute names (_data
        # and _scale) and a context mapping holding the _constant value.
        ctx = {"_constant": self._constant}
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Static unflatten hook. Requires exactly two inner tensors
        # (raises AssertionError otherwise) and rebuilds a ScaledTensor from the _data and
        # _scale entries and the _constant from metadata. Outer size and stride are ignored.
        if len(inner_tensors) != 2:
            raise AssertionError(f"Expected 2 inner tensors, got {len(inner_tensors)}")
        return ScaledTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            constant=metadata["_constant"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # [FUNCTION DOC] HELPER: Dispatch hook. Takes the first argument as the
        # scaled tensor, calls func on its _data with the remaining args and kwargs,
        # and wraps the result in a new ScaledTensor reusing the original scale and
        # constant. Assumes kwargs is a mapping (not None) because it is unpacked.
        scaled_tensor = args[0]
        out = func(scaled_tensor._data, *args[1:], **kwargs)
        return ScaledTensor(out, scaled_tensor._scale, constant=scaled_tensor._constant)

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string with the repr of the
        # data tensor, a newline, then the repr of the scale tensor.
        return f"{self._data.__repr__()}\n{self._scale.__repr__()}"


class OptionalScaledTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Variant of ScaledTensor whose scale may be None. The flatten
    # hook lists _scale as an inner tensor only when present, the unflatten hook tolerates
    # its absence, and dispatch multiplies the op result by the scale when there is one.
    def __new__(
        cls,
        data,
        scale,
        *,
        constant: int = 0,
    ):
        # [FUNCTION DOC] HELPER: Creates the wrapper subclass object with
        # geometry (size, strides, offset, dtype, layout, requires_grad, device)
        # copied from data; scale and constant are used only by __init__.
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, scale, constant: int = 0):
        # [FUNCTION DOC] HELPER: Stores data, scale (possibly None) and constant on the instance.
        self._data = data
        self._scale = scale
        self._constant = constant

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Flatten hook: returns inner tensor names (_data
        # plus _scale when scale is not None) and a context mapping with _constant.
        ctx = {"_constant": self._constant}
        if self._scale is not None:
            return ["_data", "_scale"], ctx
        else:
            return ["_data"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Static unflatten hook: rebuilds the subclass from the _data
        # entry, the optional _scale entry (None when missing) and the _constant metadata.
        return OptionalScaledTensor(
            inner_tensors["_data"],
            inner_tensors.get("_scale", None),
            constant=metadata["_constant"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # [FUNCTION DOC] HELPER: Dispatch hook: applies func to the inner _data with the remaining
        # arguments, multiplies the result by the scale when a scale exists, and wraps it in a new
        # OptionalScaledTensor with the same scale and constant. kwargs must be a mapping.
        scaled_tensor = args[0]
        out = func(scaled_tensor._data, *args[1:], **kwargs)
        if scaled_tensor._scale is not None:
            out = out * scaled_tensor._scale
        return OptionalScaledTensor(
            out, scaled_tensor._scale, constant=scaled_tensor._constant
        )

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string showing the class name with the
        # data repr, a newline and the scale repr (which prints None when absent).
        return (
            f"OptionalScaledTensor({self._data.__repr__()}\n{self._scale.__repr__()})"
        )


class CtxSubclassTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Wrapper subclass wrapping one tensor a together with an int
    # constant stored as metadata, used to verify that Dynamo/AOT guard on subclass metadata.
    # Dispatch unwraps all subclass args, runs the op, rewraps outputs with the largest
    # constant among the inputs, and for the mul Tensor op adds that constant to the result.
    """
    Class used to verify guarding on the subclass metadata
    """

    @staticmethod
    def __new__(cls, a, constant):
        # [FUNCTION DOC] HELPER: Static constructor: builds a wrapper subclass
        # object with shape, strides, storage offset, device, layout,
        # requires_grad and dtype taken from a. The constant is ignored here.
        shape = a.shape
        kwargs = {}
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, a, constant):
        # [FUNCTION DOC] HELPER: Stores the wrapped tensor in a and the metadata value in constant.
        self.a = a
        self.constant = constant

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string of the form
        # class name wrapping the repr of the inner tensor.
        a_repr = repr(self.a)
        return f"CtxSubclassTensor({a_repr})"

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Flatten hook: inner tensor attribute list with
        # only a, and a one-element tuple holding the constant as metadata.
        return ["a"], (self.constant,)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, sizes, strides):
        # [FUNCTION DOC] HELPER: Static unflatten hook: reads the constant
        # from the first metadata item and the tensor from the inner tensors,
        # returning a new CtxSubclassTensor. Sizes and strides are unused.
        constant = meta[0]
        a = inner_tensors["a"]
        return CtxSubclassTensor(a, constant)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] HELPER: Dispatch hook. Defaults kwargs to an empty mapping, computes
        # the max constant over all CtxSubclassTensor leaves in args (raises ValueError if
        # there are none), unwraps args and kwargs, calls func, wraps every tensor output into
        # a CtxSubclassTensor with that max constant, adds the constant to the output for the
        # aten mul Tensor overload, and returns the result via return_and_correct_aliasing.
        if kwargs is None:
            kwargs = {}
        biggest_constant = max(
            [
                x.constant
                for x in pytree.tree_flatten(args)[0]
                if isinstance(x, CtxSubclassTensor)
            ]
        )
        args_a = pytree.tree_map(
            lambda x: x.a if isinstance(x, CtxSubclassTensor) else x, args
        )
        kwargs_a = pytree.tree_map(
            lambda x: x.a if isinstance(x, CtxSubclassTensor) else x, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        out = pytree.tree_map(
            lambda x: (
                CtxSubclassTensor(x, biggest_constant)
                if isinstance(x, torch.Tensor)
                else x
            ),
            out_a,
        )

        if func == torch.ops.aten.mul.Tensor:
            out = out + out.constant

        return return_and_correct_aliasing(func, args, kwargs, out)


class TensorMetadataSubclass(torch.Tensor):
    # [CLASS DOC] HELPER: Wrapper subclass with one inner tensor _data and an extra tensor of
    # weights stored as flatten metadata (not as inner tensor). Dispatch unwraps to _data,
    # and rewraps tensor outputs with the weights of the first subclass argument found.
    @staticmethod
    def __new__(cls, data, weights):
        # [FUNCTION DOC] HELPER: Static constructor: creates a wrapper subclass with
        # shape, dtype, device and requires_grad of data; weights are ignored here.
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
            requires_grad=data.requires_grad,
        )

    def __init__(self, data, weights):
        # [FUNCTION DOC] HELPER: Stores data in _data and weights in _weights.
        self._data = data
        self._weights = weights

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Flatten hook: returns the single inner attribute
        # name _data and a metadata mapping holding the weights tensor.
        return ["_data"], {"weights": self._weights}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Classmethod unflatten hook: builds an instance of cls from the
        # _data inner tensor and the weights entry of the metadata; sizes and strides are unused.
        return cls(inner_tensors["_data"], meta["weights"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Dispatch hook. Unwraps subclass instances in args and kwargs to
        # their _data, runs func, returns non-tensor results as is, and otherwise wraps the output
        # in cls using the weights of the first subclass argument found among the args leaves.
        if kwargs is None:
            kwargs = {}

        def unwrap(t):
            # [FUNCTION DOC] HELPER: Returns the _data tensor if t is
            # a TensorMetadataSubclass, otherwise returns t unchanged.
            return t._data if isinstance(t, TensorMetadataSubclass) else t

        unwrapped_args = pytree.tree_map(unwrap, args)
        unwrapped_kwargs = pytree.tree_map(unwrap, kwargs)
        out = func(*unwrapped_args, **unwrapped_kwargs)

        if not isinstance(out, torch.Tensor):
            return out

        first_subclass_arg = next(
            t for t in pytree.tree_leaves(args) if isinstance(t, TensorMetadataSubclass)
        )
        return cls(out, first_subclass_arg._weights)


class TensorMetadataSubclassWithCustomGuard(TensorMetadataSubclass):
    # [CLASS DOC] HELPER: TensorMetadataSubclass that supplies a custom
    # metadata guard, so guard checks on its flatten metadata compare the
    # weights tensors by value rather than by default metadata equality.
    @classmethod
    def __metadata_guard__(cls, original_metadata, current_metadata):
        # [FUNCTION DOC] HELPER: Classmethod guard hook. Returns whether the weights tensors in
        # the original and current metadata mappings are equal in value (torch.equal result).
        return torch.equal(original_metadata["weights"], current_metadata["weights"])


class DeferredInitSubclass(torch.Tensor):
    # [CLASS DOC] HELPER: Traceable wrapper subclass whose __init__ calls the
    # parent initializer before assigning attributes (as torchao style
    # subclasses do). Stores _data as an inner tensor and _scale as metadata.
    """
    A traceable wrapper subclass that calls super().__init__() BEFORE
    setting instance attributes (similar to torchao patterns).
    """

    @staticmethod
    def __new__(cls, data, scale):
        # [FUNCTION DOC] HELPER: Static constructor: creates a wrapper
        # subclass with shape, dtype and device from data; scale is ignored.
        return torch.Tensor._make_wrapper_subclass(
            cls, data.shape, dtype=data.dtype, device=data.device
        )

    def __init__(self, data, scale):
        # [FUNCTION DOC] HELPER: Calls the parent initializer
        # first, then stores data in _data and scale in _scale.
        super().__init__()
        self._data = data
        self._scale = scale

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Flatten hook: returns inner attribute
        # list with _data and a metadata mapping holding _scale.
        return ["_data"], {"_scale": self._scale}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, ctx, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Classmethod unflatten hook: rebuilds an instance from the _data
        # inner tensor and the _scale entry of the context; outer size and stride are unused.
        return cls(inner_tensors["_data"], ctx["_scale"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # [FUNCTION DOC] HELPER: Dispatch hook. Defaults kwargs to an empty
        # mapping, replaces DeferredInitSubclass args by their _data, runs
        # func, and wraps tensor results in DeferredInitSubclass using the
        # scale of the first argument; non-tensor results are returned as is.
        if kwargs is None:
            kwargs = {}
        args_inner = pytree.tree_map_only(DeferredInitSubclass, lambda x: x._data, args)
        out = func(*args_inner, **kwargs)
        if isinstance(out, torch.Tensor):
            return DeferredInitSubclass(out, args[0]._scale)
        return out


def func(a):
    # [FUNCTION DOC] HELPER: Simple test function returning the sine of its input tensor.
    return a.sin()


class EagerRecordGraphAndInputs:
    # [CLASS DOC] HELPER: Callable backend for torch.compile that
    # returns the graph module unchanged while recording each compiled
    # graph module in graphs and its example inputs in example_inputs.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes empty lists graphs and example_inputs.
        self.graphs = []
        self.example_inputs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        # [FUNCTION DOC] HELPER: Backend entry point. Appends gm and example_inputs
        # to the recorded lists and returns gm as the compiled callable.
        self.graphs.append(gm)
        self.example_inputs.append(example_inputs)
        return gm


GLOBAL_TEST_SUBCLASSES = {
    MockSubclass,
    DummyNDim,
    SigmoidToExpSubclass,
    BaseTorchFunction,
}


# Returns True if the function recompiles between inputs1 and inputs2 with the
# specified dynamic setting.
def _recompiles_for_inputs(fn, inputs1, inputs2, dynamic=True):
    # [FUNCTION DOC] HELPER: Compiles fn with fullgraph and a counting backend (with the given
    # dynamic setting), calls it on inputs1 then on inputs2 (each an argument sequence), and
    # returns True when the backend was invoked more than once, i.e. the second call recompiled.
    compile_count = [0]

    def counter(gm, example_inputs):
        # [FUNCTION DOC] HELPER: Counting backend: increments the first slot of
        # the enclosing compile_count list and returns the graph module unchanged.
        compile_count[0] += 1
        return gm

    compiled_f = torch.compile(fn, fullgraph=True, backend=counter, dynamic=dynamic)
    compiled_f(*inputs1)
    compiled_f(*inputs2)
    return compile_count[0] > 1


class _SubclassCompileCheckMixin:
    # [CLASS DOC] HELPER: Mixin for test cases providing _compile_check, which compares eager and
    # compiled results using an aot_autograd backend that records forward and backward graphs.
    """Shared compile-check helper for subclass tests."""

    def _compile_check(
        self,
        fn,
        inps,
        *,
        dynamic=True,
        fullgraph=True,
        call_backward=False,
        call_backward_fn=None,
        assert_equal_fn=None,
    ):
        # [FUNCTION DOC] HELPER: Compiles fn with an aot_autograd backend (min-cut partitioner,
        # input mutations kept, recording forward and backward graphs) under the given dynamic and
        # fullgraph settings. For each input tuple in inps it runs eager and compiled (reseeding
        # the RNG before the compiled run) and compares with assert_equal_fn (default assertEqual).
        # With call_backward it also applies call_backward_fn (default: sum then backward retaining
        # the graph) to every tensor requiring grad in both outputs and compares those results.
        # Returns the recorded forward graphs and the backward graphs (None unless call_backward).
        if call_backward_fn is None:

            def call_backward_fn(t):
                # [FUNCTION DOC] HELPER: Default backward driver: sums the tensor and
                # runs backward with retain_graph, returning that backward result.
                return t.sum().backward(retain_graph=True)

        if assert_equal_fn is None:
            assert_equal_fn = self.assertEqual

        torch.manual_seed(0)
        fw_compiler = EagerRecordGraphAndInputs()
        bw_compiler = EagerRecordGraphAndInputs()
        compiler_fn = aot_autograd(
            fw_compiler=make_boxed_compiler(fw_compiler),
            bw_compiler=make_boxed_compiler(bw_compiler),
            partition_fn=min_cut_rematerialization_partition,
            keep_inference_input_mutations=True,
        )

        c = torch.compile(backend=compiler_fn, dynamic=dynamic, fullgraph=fullgraph)(fn)
        for inp in inps:
            expected = fn(*inp)
            torch.manual_seed(0)
            got = c(*inp)
            assert_equal_fn(expected, got)

            if call_backward:
                re = pytree.tree_map_only(
                    lambda x: isinstance(x, torch.Tensor) and x.requires_grad,
                    call_backward_fn,
                    expected,
                )
                rg = pytree.tree_map_only(
                    lambda x: isinstance(x, torch.Tensor) and x.requires_grad,
                    call_backward_fn,
                    got,
                )
                assert_equal_fn(re, rg)

        if call_backward:
            return fw_compiler.graphs, bw_compiler.graphs
        return fw_compiler.graphs, None


class SubclassTests(_SubclassCompileCheckMixin, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Main test class for Dynamo support of tensor subclasses (torch function,
    # wrapper and nested subclasses). Inherits the compile-check mixin and the Dynamo TestCase;
    # tests define their own subclasses inline and compare eager against compiled behavior.
    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown override that only closes the class-
        # wide exit stack created by the Dynamo base setUpClass. Unlike the
        # base implementation it does not call the parent tearDownClass.
        cls._exit_stack.close()

    def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
        # [FUNCTION DOC] HELPER: Method wrapper delegating to the module-level
        # recompile assertion helper with this test case as first argument.
        _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles)

    def test_no_call_to_new(self):
        # [FUNCTION DOC] Verifies that tracing a torch function subclass never calls
        # its constructor: a subclass whose __new__ raises RuntimeError is created by
        # as_subclass, then passed through a fullgraph eager compiled add; the call
        # must succeed and the result must still be an instance of the subclass.
        class BadNewTorchFunction(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose __new__ always raises
            # RuntimeError; used to prove Dynamo does not construct instances. Its torch
            # function delegates to the default with kwargs defaulted to an empty mapping.
            def __new__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError with a fixed message.
                raise RuntimeError("Oops!")

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs to an empty mapping
                # and delegates to the default torch function handler.
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function adding one to its input via torch.add.
            return torch.add(x, 1)

        input = torch.ones(2, 2).as_subclass(BadNewTorchFunction)

        res = fn(input)
        self.assertIsInstance(res, BadNewTorchFunction)

    def test_no_torch_function_recompiles(self):
        # [FUNCTION DOC] Verifies that a non-tensor object class implementing __torch_function__
        # (a toy nested tensor) does not cause recompilation: a sin function compiled with the
        # eager backend is called on two instances with different values and the same offsets
        # while the error_on_recompile config is patched on, so any recompile would raise.
        class NJT:
            # [CLASS DOC] HELPER: Toy non-tensor nested-tensor class holding
            # _values and _offsets and handling only torch.sin in its
            # __torch_function__ (any other function raises AssertionError).
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a string using the (nonexistent
                # here) shape attribute; not exercised by the test.
                return f"NJT(shape={self.shape})"

            def __init__(self, values, offsets):
                # [FUNCTION DOC] HELPER: Stores values in _values and offsets in _offsets.
                self._values = values
                self._offsets = offsets

            def sin(self):
                # [FUNCTION DOC] HELPER: Method that calls torch.sin on
                # this object, dispatching through __torch_function__.
                return torch.sin(self)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Classmethod hook. Defaults kwargs; for
                # torch.sin returns a new NJT with sin applied to the values and
                # the same offsets; for any other func raises AssertionError.
                if kwargs is None:
                    kwargs = {}
                if func == torch.sin:
                    self = args[0]
                    return NJT(func(self._values), self._offsets)
                raise AssertionError("should not get here")

        values1 = torch.randn(10, 3, 4, requires_grad=True)
        values2 = torch.randn(10, 3, 4, requires_grad=True)
        offsets = torch.tensor([0, 3, 10])
        njt1 = NJT(values1, offsets)
        njt2 = NJT(values2, offsets)

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function
            # returning the sine of its input via torch.sin.
            return torch.sin(x)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            f(njt1)
            f(njt2)

    def test_base_torch_function_tracing(self):
        # [FUNCTION DOC] Checks tracing of a subclass with the default torch function:
        # eager and fullgraph-eager compiled add of one on a BaseTorchFunction input
        # must both return the subclass type and produce equal values.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test: adds one to its input.
            return torch.add(x, 1)

        input = torch.ones(2, 2).as_subclass(BaseTorchFunction)
        out = fn(input)
        out_opt = compile_full_eager(fn)(input)
        self.assertIsInstance(out, BaseTorchFunction)
        self.assertEqual(out, out_opt)

    def test_torch_function_subclass_with_mode(self):
        # [FUNCTION DOC] Regression test that, with a TorchFunctionMode active, a subclass
        # torch function must still be inlined by Dynamo so it does not fire twice. Computes
        # torch.add of two subclass tensors eagerly under a no-op mode, resets Dynamo,
        # compiles add with the eager backend under the same mode and asserts equal results.
        # Subclass __torch_function__ must still be inlined when a
        # TorchFunctionMode is active, otherwise the runtime wrapper
        # (DisableTorchFunctionSubclass) is not applied and the subclass
        # dispatch fires twice.
        class ScaledTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Local tensor subclass created by _make_subclass whose
            # __torch_function__ doubles the result of torch.add and leaves other ops unchanged.
            @staticmethod
            def __new__(cls, data):
                # [FUNCTION DOC] HELPER: Static constructor: makes
                # a subclass instance from data via _make_subclass.
                return torch.Tensor._make_subclass(cls, data)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the default handler (kwargs
                # defaulted to empty) and multiplies the result by 2.0 when it is
                # a tensor and func is torch.add; otherwise returns it unchanged.
                kwargs = kwargs or {}
                raw = super().__torch_function__(func, types, args, kwargs)
                if isinstance(raw, torch.Tensor) and func is torch.add:
                    return raw * 2.0
                return raw

        class NoopMode(torch.overrides.TorchFunctionMode):
            # [CLASS DOC] HELPER: Torch function mode that passes every
            # call through unchanged (calls func with the args and kwargs).
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Calls func with args and kwargs (empty
                # if None) and returns the result, performing no modification.
                return func(*args, **(kwargs or {}))

        a = ScaledTensor(torch.tensor([1.0, 2.0]))
        b = ScaledTensor(torch.tensor([3.0, 4.0]))

        with NoopMode():
            eager = torch.add(a, b)
            torch._dynamo.reset()
            compiled = torch.compile(torch.add, backend="eager")(a, b)
        self.assertEqual(eager, compiled)

    def test_torch_function_reentrant_dispatch(self):
        # [FUNCTION DOC] Checks that a subclass torch function which re-enters
        # dispatch behaves the same compiled as eager: computes torch.add on
        # two doubling-subclass tensors eagerly, resets Dynamo, compiles add
        # with the eager backend and asserts equal results (no mode is active).
        class ScaledTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Local tensor subclass whose __torch_function__
            # doubles the result of torch.add; other ops are unchanged.
            @staticmethod
            def __new__(cls, data):
                # [FUNCTION DOC] HELPER: Static constructor: makes
                # a subclass instance from data via _make_subclass.
                return torch.Tensor._make_subclass(cls, data)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the default handler (kwargs
                # defaulted to empty) and multiplies the result by 2.0 when it is
                # a tensor and func is torch.add; otherwise returns it unchanged.
                kwargs = kwargs or {}
                raw = super().__torch_function__(func, types, args, kwargs)
                if isinstance(raw, torch.Tensor) and func is torch.add:
                    return raw * 2.0
                return raw

        a = ScaledTensor(torch.tensor([1.0, 2.0]))
        b = ScaledTensor(torch.tensor([3.0, 4.0]))

        eager = torch.add(a, b)
        torch._dynamo.reset()
        compiled = torch.compile(torch.add, backend="eager")(a, b)
        self.assertEqual(eager, compiled)

    def test_tensorify_under_disabled_torch_function(self):
        # [FUNCTION DOC] Regression test (issue 180906): the tensorify scalars pass must work
        # while subclass torch function dispatch is disabled, since it relies on a proxy
        # object torch function to intercept calls. Under fake tensor mode builds an fx graph
        # with a float tensor placeholder, a scale placeholder, a scalar extraction node with
        # an unbacked symbolic float, and a multiply node; runs tensorify_python_scalars
        # inside the disable-subclass context; then asserts that at least one
        # convert_element_type node was inserted and that the first one has float64 dtype.
        # Fixes #180906
        # The checks tensorify_python_scalars works under dispatch
        # as it relies on MetaProxy's __torch_function__ to intercept calls
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.fx.passes._tensorify_python_scalars import tensorify_python_scalars

        # Build a minimal graph containing _local_scalar_dense (i.e. .item())
        # on a floating-point placeholder — just enough for tensorify to act.
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env) as fake_mode:
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            x.meta["val"] = torch.randn(4)

            scale_ph = graph.placeholder("scale")
            scale_ph.meta["val"] = torch.tensor(1.0)

            item_node = graph.call_function(
                torch.ops.aten._local_scalar_dense.default, (scale_ph,)
            )
            # tensorify needs a backed SymFloat with a sympy expression
            item_node.meta["val"] = shape_env.create_unbacked_symfloat()

            mul_node = graph.call_function(torch.ops.aten.mul.Tensor, (x, item_node))
            mul_node.meta["val"] = torch.randn(4)
            graph.output(mul_node)

            gm = torch.fx.GraphModule(torch.nn.Module(), graph)

            # Run tensorify_python_scalars with __torch_function__
            # disabled — without the _EnableTorchFunction fix, this raises:
            #   RuntimeError: prims::convert_element_type() Expected
            #   a value of type 'Tensor' ... found type 'MetaProxy'.
            with torch._C.DisableTorchFunctionSubclass():
                tensorify_python_scalars(gm, shape_env, fake_mode)

            # The pass should have inserted a convert_element_type node
            # that upcasts the scale placeholder to float64.
            convert_nodes = [
                n
                for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is torch.ops.prims.convert_element_type.default
            ]
            self.assertTrue(len(convert_nodes) > 0)
            # The first convert_element_type is the float64 upcast;
            # verify it carries correct metadata.
            self.assertEqual(convert_nodes[0].meta["val"].dtype, torch.float64)

    def test_torch_function_state_graph_break(self):
        # [FUNCTION DOC] Checks that the torch-function-enabled state set by the
        # disable-subclass context survives a graph break: the eager-backend compiled
        # function disables subclass torch function, forces a graph break and returns
        # the enabled flag plus an add. The test asserts the flag is False.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Inside the disable-subclass context, graph breaks, then
            # returns a pair of the torch-function-enabled flag and the sum of x and 1.0.
            with torch._C.DisableTorchFunctionSubclass():
                torch._dynamo.graph_break()
                return torch._C._is_torch_function_enabled(), torch.add(x, 1.0)

        input = torch.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_disable_all_torch_function(self):
        # [FUNCTION DOC] Checks that the full disable-torch-function context
        # is observed across a graph break: asserts the enabled flag is
        # False and the all-disabled flag is True inside the context.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Inside the full disable context, graph breaks, then
            # returns the enabled flag, the all-disabled flag and an add of x and 1.0.
            with torch._C.DisableTorchFunction():
                torch._dynamo.graph_break()
                return (
                    torch._C._is_torch_function_enabled(),
                    torch._C._is_torch_function_all_disabled(),
                    torch.add(x, 1.0),
                )

        input = torch.ones(2, 2)
        res1, res2, _ = fn(input)
        self.assertFalse(res1)
        self.assertTrue(res2)

    def test_disable_all_torch_function_restore_values(self):
        # [FUNCTION DOC] Checks that leaving the full disable context
        # restores the all-disabled flag: the value captured inside the
        # context is True while the value read after the context is False.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Reads the all-disabled flag inside the full disable
            # context into x, then returns x, the flag read after exiting, and an add using x.
            with torch._C.DisableTorchFunction():
                x = torch._C._is_torch_function_all_disabled()

            return (
                x,
                torch._C._is_torch_function_all_disabled(),
                torch.add(x, 1.0),
            )

        input = torch.ones(2, 2)
        res1, res2, _ = fn(input)
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_disable_all_torch_function_restore_values_graph_break(self):
        # [FUNCTION DOC] Same restore check as the previous test but with a graph break inside
        # the full disable context: flag captured inside is True, flag read after exit is False.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Inside the full disable context graph
            # breaks, reads the all-disabled flag into x, then after exit
            # returns x, the new flag value and an add using x.
            with torch._C.DisableTorchFunction():
                torch._dynamo.graph_break()
                x = torch._C._is_torch_function_all_disabled()

            return (
                x,
                torch._C._is_torch_function_all_disabled(),
                torch.add(x, 1.0),
            )

        input = torch.ones(2, 2)
        res1, res2, _ = fn(input)
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_torch_function_state_nested(self):
        # [FUNCTION DOC] Checks nesting of the disable-subclass context:
        # after the inner context exits, the state must return to the outer
        # state (still disabled), so the returned enabled flag must be False.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Nests two disable-subclass contexts around an
            # add, then within the outer one returns the enabled flag and another add.
            with torch._C.DisableTorchFunctionSubclass():
                with torch._C.DisableTorchFunctionSubclass():
                    x = x + 1
                # Should reset to the outer state (disabled) after exiting ctx manager
                return torch._C._is_torch_function_enabled(), torch.add(x, 1.0)

        input = torch.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_torch_function_state_tracing(self):
        # [FUNCTION DOC] Verifies that a function entering the disable-
        # subclass context and calling torch.add compiles under fullgraph
        # with the eager backend without graph breaks, and runs it once.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1.0 to x inside
            # the disable-subclass context; returns nothing.
            with torch._C.DisableTorchFunctionSubclass():
                torch.add(x, 1.0)

        input = torch.ones(2, 2)

        fn(input)

    def test_torch_function_state_guards(self):
        # [FUNCTION DOC] Verifies that the torch function enabled state is guarded: the
        # same compiled function is called once under the disable-subclass context and once
        # outside it, and the compile counter's frame_count must equal 2 (a recompile).
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Performs an add of 1.0 on x and returns nothing.
            torch.add(x, 1.0)

        input = torch.ones(2, 2)

        with torch._C.DisableTorchFunctionSubclass():
            fn(input)

        fn(input)

        self.assertEqual(cnt.frame_count, 2)

    def test_return_subclass(self):
        # [FUNCTION DOC] Verifies that a function building a MockSubclass from a tensor and
        # multiplying it compiles with fullgraph and returns an instance of that subclass.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the sum of x and 1.0 in MockSubclass and doubles it.
            return MockSubclass(torch.add(x, 1.0)) * 2

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, MockSubclass)

    def test_return_as_subclass(self):
        # [FUNCTION DOC] Verifies that casting a result with as_subclass to MockSubclass
        # inside a fullgraph compiled function yields an instance of that subclass.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 1.0 to x, converts it
            # with as_subclass to MockSubclass and doubles it.
            return torch.add(x, 1.0).as_subclass(MockSubclass) * 2

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, MockSubclass)

    def test_return_local_subclass(self):
        # [FUNCTION DOC] Same as the previous returned-subclass tests but
        # with a subclass class defined locally inside the test; asserts
        # the compiled result is an instance of the local subclass.
        class LocalSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Local tensor subclass delegating
            # __torch_function__ to the default implementation.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Delegates to the default torch function handler.
                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps x plus 1.0 in LocalSubclass and doubles it.
            return LocalSubclass(torch.add(x, 1.0)) * 2

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, LocalSubclass)

    def test_torch_function_list_args(self):
        # [FUNCTION DOC] Checks that a non-tensor class implementing
        # __torch_function__ can handle a function taking a list of arguments:
        # torch.stack of two such objects is routed to a registered handler that
        # sums their foo values, and the compiled result's foo must equal 2.
        HANDLED_FUNCTIONS = {}

        class MyClass:
            # [CLASS DOC] HELPER: Plain class with a foo attribute whose torch function
            # looks up the function in a handled-functions registry; returns
            # NotImplemented if the function is not registered or any involved type is
            # neither a tensor class nor this class, else calls the registered handler.
            def __init__(self, foo):
                # [FUNCTION DOC] HELPER: Stores foo.
                self.foo = foo

            @classmethod
            def __torch_function__(
                cls,
                func,
                types,
                args=(),
                kwargs=None,
            ):
                # [FUNCTION DOC] HELPER: Classmethod hook. Defaults kwargs to empty;
                # returns NotImplemented unless func has a registered handler and
                # every type in types is a subclass of tensor or MyClass; otherwise
                # returns the handler result for the args and kwargs.
                if kwargs is None:
                    kwargs = {}
                if func not in HANDLED_FUNCTIONS or not all(
                    [  # noqa: C419
                        issubclass(t, (torch.Tensor, MyClass)) for t in types
                    ]
                ):
                    return NotImplemented
                return HANDLED_FUNCTIONS[func](*args, **kwargs)

        def _stack(input, dim=0, *, out=None):
            # [FUNCTION DOC] HELPER: Registered handler for torch.stack:
            # returns a MyClass whose foo is the sum of foo over the items
            # of the input list. The dim and out arguments are ignored.
            return MyClass(sum([x.foo for x in input]))

        HANDLED_FUNCTIONS[torch.stack] = _stack

        @torch.compile(backend="eager", fullgraph=True)
        def fn(v0, v1):
            # [FUNCTION DOC] HELPER: Stacks the two inputs using torch.stack on a list.
            return torch.stack([v0, v1])

        ret = fn(MyClass(1), MyClass(1))
        self.assertEqual(ret.foo, 2)

    @parametrize(
        "comparison",
        [
            subtest(isinstance, "isinstance"),
            subtest(lambda instance, type_: type(instance) is type_, "equality"),
            subtest(lambda instance, type_: type(instance) is type_, "identity"),
        ],
    )
    @parametrize(
        "input_type",
        [
            subtest(torch.Tensor, "tensor"),
            subtest(DummyNDim, "subclass"),
        ],
    )
    def test_type_check(self, comparison, input_type):
        # [FUNCTION DOC] Parametrized test over three comparison styles (isinstance,
        # and two lambdas comparing exact type, both written identically) and two input
        # types (plain tensor, DummyNDim subclass). Checks that compiled fullgraph
        # eager gives the same result as eager for a type check against DummyNDim.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a 1x1 ones tensor when the
            # comparison says x is a DummyNDim, else a 2x2 zeros tensor.
            if comparison(x, DummyNDim):
                return torch.ones(1, 1)
            else:
                return torch.zeros(2, 2)

        input = torch.ones(2, 2).as_subclass(input_type)
        exp_res = fn(input)
        act_res = torch.compile(backend="eager", fullgraph=True)(fn)(input)
        self.assertEqual(exp_res, act_res)

    def test_torch_function_call_on_method(self):
        # [FUNCTION DOC] Checks that calling a method that is not overridden in the
        # subclass torch function (exp) on a SigmoidToExpSubclass tensor behaves
        # the same eager and compiled, and equals the result on a plain tensor.
        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        z = torch.ones(2, 2)
        wrapped = x.as_subclass(SigmoidToExpSubclass)
        wrapped2 = y.as_subclass(SigmoidToExpSubclass)

        def fn(w):
            # [FUNCTION DOC] HELPER: Returns the exp method applied to w.
            return w.exp()

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped2)
        res_exp2 = z.exp()

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_exp, res_exp2)

    def test_torch_function_call_on_method_arg(self):
        # [FUNCTION DOC] Checks that torch function rewriting applies when a subclass is passed
        # as the argument of a plain tensor method: the local subclass turns in-place add into
        # in-place sub, and the compiled and eager results on a plain base must match.
        class LocalSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Local tensor subclass whose torch function swaps the in-place
            # add function for in-place sub; also defines a sigmoid method returning None.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Replaces add_ with sub_ when func is
                # add_, defaults kwargs and delegates to the default handler.
                if func == torch._C.TensorBase.add_:
                    func = torch._C.TensorBase.sub_

                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            def sigmoid(self):
                # [FUNCTION DOC] HELPER: Overriding method that returns None.
                return None

        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        z = torch.ones(2, 2)
        wrapped = y.as_subclass(LocalSubclass)
        wrapped2 = z.as_subclass(LocalSubclass)

        def fn(a, w):
            # [FUNCTION DOC] HELPER: In-place adds w to a and returns a.
            a.add_(w)
            return a

        fn_opt = torch.compile(fn, backend="eager")

        res_exp = fn(x, wrapped)
        res_act = fn_opt(y, wrapped2)

        self.assertEqual(res_exp, res_act)

    def test_user_overridden_method_unsupported(self):
        # [FUNCTION DOC] Checks behavior when a subclass overrides a tensor
        # method (sigmoid returning None) and the compiled fullgraph function
        # calls it: the compiled and eager results (both None) must be equal.
        class LocalSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Local subclass delegating torch
            # function to default and overriding sigmoid to return None.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Delegates to the default torch function handler.
                return super().__torch_function__(func, types, args, kwargs)

            def sigmoid(self):
                # [FUNCTION DOC] HELPER: Overridden method returning None.
                return None

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls sigmoid on x and returns nothing.
            x.sigmoid()

        x = torch.ones(2, 2).as_subclass(LocalSubclass)
        fn_opt = compile_full_eager(fn)

        res_exp = fn(x)
        res_act = fn_opt(x)

        self.assertEqual(res_exp, res_act)

    def test_user_overridden_attr_unsupported(self):
        # [FUNCTION DOC] Checks that tracing an attribute overridden in a subclass body
        # (ndim defined as a plain class attribute) raises torch._dynamo.exc.Unsupported
        # with the message that only certain overridden attribute types are supported.
        class LocalSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Local subclass whose class body defines
            # a plain ndim attribute equal to 10, overriding the property.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs and delegates to the default handler.
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            ndim = 10

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the ndim attribute of x.
            return x.ndim

        msg = "`torch.compile` only support tracing certain types of overridden tensor subclass attributes"
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    def test_user_overridden_property_unsupported(self):
        # [FUNCTION DOC] Checks that a subclass overriding the ndim
        # property (with getter and setter backed by an instance attribute)
        # gives the same result eager and compiled when used in arithmetic.
        class LocalSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Local tensor subclass with an instance attribute _ndim set to
            # 10 in its initializer and an ndim property with getter and setter operating on it.
            def __init__(self, *args, **kwargs) -> None:
                # [FUNCTION DOC] HELPER: Sets _ndim to 10, ignoring all arguments.
                self._ndim = 10

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Delegates to the default torch function handler.
                return super().__torch_function__(func, types, args, kwargs)

            @property
            def ndim(self):
                # [FUNCTION DOC] HELPER: Property getter returning the stored _ndim value.
                return self._ndim

            @ndim.setter
            def ndim(self, value):
                # [FUNCTION DOC] HELPER: Property setter storing value into _ndim.
                self._ndim = value

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus its ndim property value.
            return x + x.ndim

        x = LocalSubclass(torch.ones(2, 2))
        fn_opt = compile_full_eager(fn)

        res_exp = fn(x)
        res_act = fn_opt(x)

        self.assertEqual(res_exp, res_act)

    def test_overridden_method_guarding(self):
        # [FUNCTION DOC] Checks guarding on overridden methods of a subclass: with
        # error_on_recompile enabled, calling the compiled sigmoid function repeatedly on instances
        # of the same subclass must not recompile; then replacing sigmoid on the class by False
        # must make the compiled call raise TypeError saying the bool object is not callable.
        class LocalSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Local subclass with default delegating torch function and no
            # other overrides; the test later assigns a non-callable to its sigmoid attribute.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs and delegates to the default handler.
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sigmoid method applied to x.
            return x.sigmoid()

        with torch._dynamo.config.patch(error_on_recompile=True):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)
            fn(x)
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

        with self.assertRaisesRegex(TypeError, "'bool' object is not callable"):
            LocalSubclass.sigmoid = False
            fn(x)

    def test_torch_function_call_on_attr(self):
        # [FUNCTION DOC] Checks attribute access intercepted by __torch_function__:
        # for a DummyNDim tensor ndim reports 10, so the function returning that plus
        # a ones tensor must be equal eager and compiled and equal to 10 plus ones.
        x = torch.ones(2, 2)
        wrapped = x.as_subclass(DummyNDim)

        def fn(w):
            # [FUNCTION DOC] HELPER: Returns the ndim
            # attribute of w added to a ones tensor of size 2.
            return w.ndim + torch.ones(2)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_exp, torch.ones(2) + 10)

    def test_torch_function_wrapper_class(self):
        # [FUNCTION DOC] Checks that a non-tensor wrapper class
        # (WrapperSubclass) passed to torch.add is unwrapped via its torch
        # function, with equal eager and fullgraph compiled results.
        x = torch.ones(2, 2)
        wrapped = WrapperSubclass(x)

        def fn(w):
            # [FUNCTION DOC] HELPER: Returns torch.add of w and 1.0.
            return torch.add(w, 1.0)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)
        self.assertEqual(res_exp, res_act)

    def test_no_torch_function_on_size_bytecode(self):
        # [FUNCTION DOC] Checks that a subclass whose torch function runs the op with
        # subclass dispatch disabled and doubles the output for clone works with a dynamic-
        # marked input under fullgraph compile; the output must equal a ones tensor times 2.
        class TestTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Local subclass whose torch function
            # runs the function with subclass torch function disabled,
            # and doubles the result when the function is torch.clone.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs, evaluates func
                # inside the disable-subclass context, returns twice the
                # result for torch.clone, else the plain result.
                if kwargs is None:
                    kwargs = {}
                with torch._C.DisableTorchFunctionSubclass():
                    out = func(*args, **kwargs)

                    if func == torch.clone:
                        return out * 2
                    else:
                        return out

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns torch.clone of x.
            return torch.clone(x)

        inp = torch.ones(4, 4)
        x = inp.as_subclass(TestTensor)
        torch._dynamo.mark_dynamic(x, 0)
        compiled_fn = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        out = compiled_fn(x)
        self.assertEqual(out, torch.ones(4, 4) * 2)

    def test_tensor_subclass_unpack(self):
        # [FUNCTION DOC] Checks that iterating over a tensor subclass (list
        # conversion) yields subclass instances when the class is registered in the
        # traceable subclasses config: the registration is added before and
        # discarded after, even on failure. Asserts three elements, all of type Foo.
        class Foo(torch.Tensor):
            # [CLASS DOC] HELPER: Empty tensor subclass used to test unpacking.
            pass

        torch._dynamo.config.traceable_tensor_subclasses.add(Foo)
        try:

            @torch.compile(backend="eager", fullgraph=True)
            def fn_list(x):
                # [FUNCTION DOC] HELPER: Returns the list produced by iterating x.
                return list(x)

            x = torch.ones(3).as_subclass(Foo)
            res_list = fn_list(x)

            for elem in res_list:
                self.assertIsInstance(elem, Foo)
            self.assertEqual(len(res_list), 3)
        finally:
            torch._dynamo.config.traceable_tensor_subclasses.discard(Foo)

    def test_torch_function_wrapper_class_with_kwargs(self):
        # [FUNCTION DOC] Same as the wrapper class test but passes an extra keyword
        # argument alpha to torch.add; compiled and eager results must match.
        x = torch.ones(2, 2)
        wrapped = WrapperSubclass(x)

        def fn(w):
            # [FUNCTION DOC] HELPER: Returns torch.add of w and 1.0 with alpha 2.0.
            return torch.add(w, 1.0, alpha=2.0)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)
        self.assertEqual(res_exp, res_act)

    def test_tensor_subclass_with_non_classmethod_torch_function(self):
        # [FUNCTION DOC] Checks that a tensor subclass defining __torch_function__
        # as a regular instance method rather than a classmethod still works:
        # compiled and eager results of adding one are compared.
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Local subclass whose torch function is a plain method
            # (not classmethod) that runs the function with subclass dispatch disabled.
            def __torch_function__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs and returns func
                # evaluated on args inside the disable-subclass context.
                if kwargs is None:
                    kwargs = {}
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        fn_opt = compile_full_eager(fn)

        x = torch.randn(2, 2).as_subclass(MySubclass)
        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_tensor_subclass_custom_attr(self):
        # [FUNCTION DOC] Checks that a custom class-level attribute on a tensor subclass
        # can be read inside a fullgraph compiled function, comparing against eager.
        class AttrSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Local subclass with a class
            # attribute x equal to 10 and a delegating torch function.
            x: int = 10

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs and delegates to the default handler.
                if kwargs is None:
                    kwargs = {}

                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the x attribute of the input plus a 2x2 ones tensor.
            return x.x + torch.ones(2, 2)

        input = torch.ones(2, 2).as_subclass(AttrSubclass)
        fn_opt = compile_full_eager(fn)

        res_exp = fn(input)
        res_act = fn_opt(input)
        self.assertEqual(res_exp, res_act)

    def test_make_subclass(self):
        # [FUNCTION DOC] Checks that the _make_subclass call is traceable and
        # that aliasing is modeled correctly: three views (original, subclass
        # view and plain view) share data, are mutated in place and combined;
        # eager and compiled results and the mutated base must match.
        # Make sure `torch.Tensor._make_subclass` is traceable, and Dynamo
        # models its aliasing relationships correctly.
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Empty tensor subclass used as the target of _make_subclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Makes two aliasing views of x via
            # _make_subclass (one subclass, one plain), mutates x, y and z
            # in place with different amounts, and returns x times y plus z.
            # Downcast then upcast
            y = torch.Tensor._make_subclass(MySubclass, x)
            z = torch.Tensor._make_subclass(torch.Tensor, x)
            # Now `x, y, z` should have the same underlying data.
            x += 1
            y += 2
            z += 3
            res = x * y + z
            return res

        x0 = torch.randn(2, 2)
        x1 = x0.clone()

        fn_opt = compile_full_eager(fn)

        res_exp = fn(x0)
        res_act = fn_opt(x1)
        self.assertEqual(res_exp, res_act)
        self.assertEqual(x0, x1)

    # ACT (AsyncCollectiveTensor) can be constructed directly, so the guard
    # relaxation for ACT inputs is exercised here on CPU without a process group.
    # The end-to-end path with a real collective + inductor is covered by
    # test/distributed/test_inductor_collectives.py.
    @unittest.skipIf(not torch.distributed.is_available(), "requires torch.distributed")
    def test_async_collective_tensor_input_polymorphism(self):
        # [FUNCTION DOC] Skipped without torch.distributed. Checks input
        # polymorphism for AsyncCollectiveTensor: a graph traced with a wrapper
        # input is reused for a plain tensor and vice versa. Calls the function
        # four times alternating wrapper and plain input, checking each result
        # equals the expected matmul sum, and asserts exactly one compiled frame.
        # A graph traced on an ACT input must be reused when the resolved plain
        # Tensor is passed at runtime (and vice versa), with no class recompile.
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, w):
            # [FUNCTION DOC] HELPER: Returns the sum of the matrix product of x and w.
            return (x @ w).sum()

        x = torch.randn(4, 4)
        base = torch.randn(4, 4)
        for i in range(4):
            w = AsyncCollectiveTensor(base) if i % 2 == 0 else base
            self.assertEqual(fn(x, w), (x @ base).sum())
        self.assertEqual(cnt.frame_count, 1)

    @unittest.skipIf(not torch.distributed.is_available(), "requires torch.distributed")
    def test_async_collective_tensor_input_polymorphism_aot_eager(self):
        # [FUNCTION DOC] Skipped without torch.distributed. Same alternating-input polymorphism
        # check as the previous test but through the aot_eager counting backend, so the runtime
        # wrapper unwrapping an AsyncCollectiveTensor is exercised; expects exactly one frame.
        # Same as above but through AOTAutograd (aot_eager), which exercises the
        # runtime wrapper that unwraps/waits an ACT input and passes a plain
        # Tensor through unchanged -- the mechanism a bare eager backend skips.
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, w):
            # [FUNCTION DOC] HELPER: Returns the sum of the matrix product of x and w.
            return (x @ w).sum()

        x = torch.randn(4, 4)
        base = torch.randn(4, 4)
        for i in range(4):
            w = AsyncCollectiveTensor(base) if i % 2 == 0 else base
            self.assertEqual(fn(x, w), (x @ base).sum())
        self.assertEqual(cnt.frame_count, 1)

    @unittest.skipIf(not torch.distributed.is_available(), "requires torch.distributed")
    def test_async_collective_tensor_inner_access(self):
        # [FUNCTION DOC] Skipped without torch.distributed. Checks that reading the inner
        # elem tensor of an AsyncCollectiveTensor under fullgraph traces and reconstructs
        # correctly. Alternates wrapper and plain inputs, checking each result equals the
        # sum of the base; because the function branches on isinstance, expects two frames.
        # Reading the ACT inner tensor (w.elem) under fullgraph must trace and
        # reconstruct it correctly -- for an ACT input the inner tensor is guarded
        # through UnwrapCollectiveTensorSource, so this exercises that source's
        # reconstruct() (whose absence previously hard-errored). The isinstance
        # check discriminates ACT from Tensor, so the two classes compile separate
        # graphs (no reuse here); the plain-Tensor reuse path is covered by
        # test_async_collective_tensor_input_polymorphism.
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(w):
            # [FUNCTION DOC] HELPER: Returns the sum of the elem inner tensor
            # when w is an AsyncCollectiveTensor, otherwise the sum of w itself.
            return (w.elem if isinstance(w, AsyncCollectiveTensor) else w).sum()

        base = torch.randn(4, 4)
        for i in range(4):
            w = AsyncCollectiveTensor(base) if i % 2 == 0 else base
            self.assertEqual(fn(w), base.sum())
        # isinstance(w, ACT) discriminates, so the two classes need separate graphs.
        self.assertEqual(cnt.frame_count, 2)

    @unittest.skipIf(not torch.distributed.is_available(), "requires torch.distributed")
    def test_async_collective_tensor_reverse_direction_still_guards(self):
        # [FUNCTION DOC] Skipped without torch.distributed. Reverse of the polymorphism test: the
        # first trace uses a plain tensor, then an AsyncCollectiveTensor is passed on odd steps.
        # Each result is checked against the expected matmul sum, and the compile counter must
        # report two frames since the relaxation only applies when the traced value is the wrapper.
        # Reverse direction: trace on a plain Tensor, then pass an ACT at runtime.
        # This must recompile -- the relaxation only applies when the traced value
        # is an ACT (a Tensor-traced graph may not await the collective).
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, w):
            # [FUNCTION DOC] HELPER: Returns the sum of the matrix product of x and w.
            return (x @ w).sum()

        x = torch.randn(4, 4)
        base = torch.randn(4, 4)
        for i in range(4):
            # Step 0 traces with the plain Tensor; odd steps pass the ACT.
            w = base if i % 2 == 0 else AsyncCollectiveTensor(base)
            self.assertEqual(fn(x, w), (x @ base).sum())
        self.assertEqual(cnt.frame_count, 2)

    @unittest.skipIf(not torch.distributed.is_available(), "requires torch.distributed")
    @parametrize("channel", ["isinstance", "class", "type", "match"])
    def test_async_collective_tensor_type_introspection_recompiles(self, channel):
        # [FUNCTION DOC] Skipped without torch.distributed; parametrized over four ways of
        # observing the class of the second input (isinstance, class attribute identity,
        # type identity, structural match). For each, a function that doubles the result
        # when the input is an AsyncCollectiveTensor must compile twice across alternating
        # wrapper and plain inputs, while a control function without class check compiles
        # once, which pins that the extra compile comes from the observation.
        # A region that observes the ACT class must recompile on the class change
        # and stay correct: the relaxed class guard is reinstalled on observation.
        # The non-observing control below (same matmul, no class check) reuses one
        # graph across the ACT/Tensor alternation, so the extra compile is caused
        # by the observation, not by the input class per se. This is what pins the
        # reinstall-on-observation behavior: without the relaxation the control
        # would recompile too (ACT vs Tensor guards differ regardless), so a plain
        # "observing recompiles" assertion would pass even if the relaxation
        # regressed.
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        def make_fn(channel):
            # [FUNCTION DOC] HELPER: Factory returning a function of x and w whose body
            # checks whether w is an AsyncCollectiveTensor using the technique named by
            # channel (one of isinstance, class, type, otherwise a match statement) and
            # returns twice the matmul sum if so, else the plain matmul sum.
            if channel == "isinstance":

                def fn(x, w):
                    # [FUNCTION DOC] HELPER: Doubles the matmul sum when w is an
                    # AsyncCollectiveTensor per an isinstance check, else returns the plain sum.
                    if isinstance(w, AsyncCollectiveTensor):
                        return (x @ w).sum() * 2
                    return (x @ w).sum()

            elif channel == "class":

                def fn(x, w):
                    # [FUNCTION DOC] HELPER: Same result logic using a comparison
                    # of the class attribute of w with AsyncCollectiveTensor.
                    if w.__class__ is AsyncCollectiveTensor:
                        return (x @ w).sum() * 2
                    return (x @ w).sum()

            elif channel == "type":

                def fn(x, w):
                    # [FUNCTION DOC] HELPER: Same result logic using an
                    # exact type comparison of w with AsyncCollectiveTensor.
                    if type(w) is AsyncCollectiveTensor:
                        return (x @ w).sum() * 2
                    return (x @ w).sum()

            else:  # match

                def fn(x, w):
                    # [FUNCTION DOC] HELPER: Same result logic using a structural match statement
                    # on w with a class pattern for AsyncCollectiveTensor and a wildcard fallback.
                    match w:
                        case AsyncCollectiveTensor():
                            return (x @ w).sum() * 2
                        case _:
                            return (x @ w).sum()

            return fn

        def control(x, w):
            # [FUNCTION DOC] HELPER: Control function with no class
            # observation: returns the sum of the matrix product of x and w.
            return (x @ w).sum()

        def run(f):
            # [FUNCTION DOC] HELPER: Resets Dynamo, compiles f under fullgraph
            # with a counting backend, runs it four times with random inputs
            # alternating a wrapper and a plain tensor, asserts each compiled
            # result equals eager f, and returns the compile counter frame count.
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()
            step = torch.compile(f, backend=cnt, fullgraph=True)
            x = torch.randn(4, 4)
            base = torch.randn(4, 4)
            for i in range(4):
                w = AsyncCollectiveTensor(base) if i % 2 == 0 else base
                self.assertEqual(step(x, w), f(x, w))
            return cnt.frame_count

        # Observing the class recompiles across the ACT<->Tensor change...
        self.assertEqual(run(make_fn(channel)), 2)
        # ...while the same computation without the class check reuses one graph.
        self.assertEqual(run(control), 1)

    @unittest.skipIf(not torch.distributed.is_available(), "requires torch.distributed")
    @parametrize("channel", ["isinstance", "isinstance_tuple", "match"])
    def test_async_collective_tensor_nondiscriminating_checks_reuse(self, channel):
        # [FUNCTION DOC] Skipped without torch.distributed; parametrized over isinstance,
        # isinstance with a tuple of classes, and match. A class check that is true for
        # both an AsyncCollectiveTensor and a plain tensor must not cause recompilation:
        # after four alternating calls with correct results the frame count must be 1.
        # A class check that is True for both an ACT and the resolved Tensor
        # (e.g. isinstance(w, torch.Tensor)) must NOT recompile.
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        def make_fn(channel):
            # [FUNCTION DOC] HELPER: Factory returning a function of x and w that
            # returns the matmul sum when w passes a non-discriminating check
            # against the tensor class (isinstance, isinstance with a tuple, or a
            # match class pattern chosen by channel) and otherwise returns w.
            if channel == "isinstance":

                def fn(x, w):
                    # [FUNCTION DOC] HELPER: Returns the matmul
                    # sum if w is a tensor per isinstance, else w.
                    return (x @ w).sum() if isinstance(w, torch.Tensor) else w

            elif channel == "isinstance_tuple":

                def fn(x, w):
                    # [FUNCTION DOC] HELPER: Returns the matmul sum
                    # if w is an instance of tensor or int, else w.
                    return (x @ w).sum() if isinstance(w, (torch.Tensor, int)) else w

            else:  # match

                def fn(x, w):
                    # [FUNCTION DOC] HELPER: Uses a match statement with a tensor
                    # class pattern to return the matmul sum, otherwise returns w.
                    match w:
                        case torch.Tensor():
                            return (x @ w).sum()
                        case _:
                            return w

            return fn

        cnt = torch._dynamo.testing.CompileCounter()
        fn = make_fn(channel)
        step = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.randn(4, 4)
        base = torch.randn(4, 4)
        for i in range(4):
            w = AsyncCollectiveTensor(base) if i % 2 == 0 else base
            self.assertEqual(step(x, w), (x @ base).sum())
        self.assertEqual(cnt.frame_count, 1)

    @unittest.skipIf(not torch.distributed.is_available(), "requires torch.distributed")
    def test_async_collective_tensor_dynamic_shapes(self):
        # [FUNCTION DOC] Skipped without torch.distributed. Checks dynamic shape behavior
        # with AsyncCollectiveTensor inputs: with static shapes alternating wrapper and
        # plain inputs share one compiled entry (one frame), while with dynamic shapes they
        # split once into two entries and more repetitions (4 or 8) add no more compiles.
        # Documents a known imperfection in the symbolic guard source for the inner tensor.
        # The unwrapped inner tensor is guarded through UnwrapCollectiveTensorSource;
        # dynamic shapes over that source must stay correct across ACT/Tensor.
        #
        # Reuse is imperfect under dynamic=True. The ShapeEnv derives the ACT
        # inner-dim symbol from a plain ".elem" AttrSource, not from
        # UnwrapCollectiveTensorSource, so the symbolic-shape guard evaluates
        # L['w'].elem.size() and fails with AttributeError on a resolved plain
        # Tensor. The ACT and the Tensor therefore land in two separate cache
        # entries. This split is one-time and stable (each class always hits its
        # own entry); it is NOT a per-alternation recompile, so 4 and 8 reps both
        # compile exactly twice. Static shapes reuse perfectly (one entry).
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        def run(reps, dynamic):
            # [FUNCTION DOC] HELPER: Resets Dynamo, compiles a matmul-sum
            # function with the given dynamic flag and a counting backend, calls
            # it reps times alternating wrapper and plain second input with a 3
            # by 4 first input, checks results, and returns the frame count.
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()

            @torch.compile(backend=cnt, fullgraph=True, dynamic=dynamic)
            def fn(x, w):
                # [FUNCTION DOC] HELPER: Returns the sum of the matrix product of x and w.
                return (x @ w).sum()

            base = torch.randn(4, 4)
            x = torch.randn(3, 4)
            for i in range(reps):
                w = AsyncCollectiveTensor(base) if i % 2 == 0 else base
                self.assertEqual(fn(x, w), (x @ base).sum())
            return cnt.frame_count

        # Static shapes: ACT and Tensor share one cache entry (perfect reuse).
        self.assertEqual(run(reps=4, dynamic=False), 1)
        # Dynamic shapes: a one-time, stable ACT/Tensor split into two entries --
        # more reps add no further compiles.
        self.assertEqual(run(reps=4, dynamic=True), 2)
        self.assertEqual(run(reps=8, dynamic=True), 2)

    def test_tensor_dunder_class_provenance(self):
        # [FUNCTION DOC] Checks that reading the class attribute of a plain tensor
        # and branching on it still traces under fullgraph: the function scales
        # the sum of x plus one by 1 for a plain tensor class. Two different
        # tensors must give correct results with a single compiled frame.
        # var_getattr("__class__") now attaches a source (TypeSource) for all
        # tensors. Reading a plain tensor's __class__ and branching on it must
        # still trace and produce the correct result.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Chooses a factor of 1 if the class attribute of x is
            # the tensor class, else 2, and returns the sum of x plus one times the factor.
            factor = 1 if x.__class__ is torch.Tensor else 2
            return (x + 1).sum() * factor

        x = torch.randn(4)
        y = torch.randn(4)
        self.assertEqual(fn(x), (x + 1).sum())
        self.assertEqual(fn(y), (y + 1).sum())
        self.assertEqual(cnt.frame_count, 1)

    def test_subclass_override_shape_and_to(self):
        # [FUNCTION DOC] Checks a subclass that overrides the shape property and the
        # to method (modeled on a quantizer utility): compiled and eager results
        # (including recorded shape attributes) and the parameters themselves must
        # be equal, as must the tensor_shape attributes set as side effects.
        # This is a slight variabtion of
        # https://github.com/huggingface/diffusers/blob/fbf6b856cc61fd22ad8635547bff4aafe05723f3/src/diffusers/quantizers/gguf/utils.py#L398-L435
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass overriding to so that the result
            # carries a tensor_shape attribute copied from the source (or the data
            # shape), and a shape property that lazily caches the size in tensor_shape.
            def to(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the base to, sets the tensor_shape
                # attribute of the result to this tensor's existing value
                # (falling back to the result data shape) and returns the result.
                new = super().to(*args, **kwargs)
                new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
                return new

            @property
            def shape(self):
                # [FUNCTION DOC] HELPER: Property getter: sets tensor_shape
                # from the size when missing, then returns it.
                if not hasattr(self, "tensor_shape"):
                    self.tensor_shape = self.size()
                return self.tensor_shape

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads the shape of x, converts x to cpu and returns
            # a tuple of x plus one, the converted tensor plus two, the saved shape, and
            # the tensor_shape attributes of both x and the converted tensor.
            x_shape = x.shape
            y = x.to("cpu")
            return x + 1, y + 2, x_shape, x.tensor_shape, y.tensor_shape

        x0 = torch.nn.Parameter(torch.randn(2, 2).as_subclass(MySubclass))
        x1 = torch.nn.Parameter(x0.clone().as_subclass(MySubclass))

        fn_opt = compile_full_eager(fn)

        res_exp = fn(x0)
        res_act = fn_opt(x1)
        self.assertEqual(res_exp, res_act)
        self.assertEqual(x0, x1)
        self.assertEqual(x0.tensor_shape, x1.tensor_shape)

    def test_subclass_dont_invoke_torch_function_on_overridden_method(self):
        # [FUNCTION DOC] Checks that torch function is not fired for a method the
        # subclass overrides itself: the overridden to method multiplies by the
        # length of the device string, and the torch function would graph break
        # if it ran under fullgraph. Eager and compiled results must match.
        # We shouldn't fire `__torch_function__` for overridden tensor methods.
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Subclass overriding to with a version
            # multiplying itself by the length of its device argument, and a torch
            # function that graph breaks when called for the Tensor to function.
            def to(self, device):
                # [FUNCTION DOC] HELPER: Overriding to method: returns
                # self times the length of the device argument.
                return self * len(device)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Triggers a Dynamo graph break when
                # func is Tensor.to, then delegates to the default handler.
                if func is torch.Tensor.to:
                    torch._dynamo.graph_break()
                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Converts x to the cpu device string.
            return x.to("cpu")

        x = torch.nn.Parameter(torch.randn(2, 2).as_subclass(MySubclass))

        fn_opt = compile_full_eager(fn)

        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_subclass_dont_invoke_torch_function_on_overridden_attr(self):
        # [FUNCTION DOC] Checks that torch function is not fired for an attribute the subclass
        # overrides itself (ndim defined as a method returning 42). The torch function graph
        # breaks if it sees a method-wrapper named ndim; eager and fullgraph results must match.
        from types import MethodWrapperType

        # We shouldn't fire `__torch_function__` for overridden tensor attrs.
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Subclass defining an ndim method returning 42 and a torch
            # function that graph breaks if the function is a method wrapper named ndim.
            def ndim(self):
                # [FUNCTION DOC] HELPER: Returns the constant 42.
                return 42

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Graph breaks for a method wrapper
                # function called ndim, then delegates to the default handler.
                if type(func) is MethodWrapperType and func.__name__ == "ndim":
                    torch._dynamo.graph_break()
                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus the result of calling its ndim method.
            return x + x.ndim()

        x = torch.nn.Parameter(torch.randn(2, 2).as_subclass(MySubclass))

        fn_opt = compile_full_eager(fn)

        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_parameter_subclass_with_old_torch_function(self):
        # [FUNCTION DOC] Checks a Parameter subclass with default torch function through transpose
        # and the T attribute under fullgraph; eager and compiled outputs must be equal.
        class MySubclass(torch.nn.Parameter):
            # [CLASS DOC] HELPER: Empty subclass of Parameter.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Transposes x with the method,
            # then the T property, and returns the result plus one.
            x = x.t()
            x = x.T
            return x + 1

        fn_opt = compile_full_eager(fn)

        x = torch.randn(2, 2).as_subclass(MySubclass)
        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_subclass_with_disabled_torch_function(self):
        # [FUNCTION DOC] Same transpose test for a tensor subclass whose torch function is
        # the disabled implementation; eager and fullgraph compiled outputs must match.
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose torch function is set to the
            # disabled torch function implementation, turning off subclass dispatch.
            __torch_function__ = torch._C._disabled_torch_function_impl

        def fn(x):
            # [FUNCTION DOC] HELPER: Transposes x with the method,
            # then the T property, and returns the result plus one.
            x = x.t()
            x = x.T
            return x + 1

        fn_opt = compile_full_eager(fn)

        x = torch.randn(2, 2).as_subclass(MySubclass)
        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_redispatch_function_with_dynamo(self):
        # [FUNCTION DOC] Checks redispatch_function integration with Dynamo:
        # asserts the skip-torch-function flag is False before, compiles a function
        # that dispatches through handle_torch_function on subclass inputs, and
        # checks the result is 3.0 and the skip flag is False again afterward.
        from torch.overrides import (
            handle_torch_function,
            has_torch_function,
            redispatch_function,
        )

        self.assertFalse(torch._C._peek_should_skip_torch_function())

        class SimpleTFTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Subclass whose torch function redispatches
            # the call past subclass handling using redispatch_function.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs and returns the redispatched call result.
                if kwargs is None:
                    kwargs = {}
                return redispatch_function(func, types, args, kwargs)

        def my_func(a, b):
            # [FUNCTION DOC] HELPER: If either argument has a torch function, routes through
            # handle_torch_function with itself as the public API; otherwise returns the sum.
            if has_torch_function((a, b)):
                return handle_torch_function(my_func, (a, b), a, b)
            return a + b

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls my_func on the two inputs.
            return my_func(x, y)

        x = torch.tensor([1.0]).as_subclass(SimpleTFTensor)
        y = torch.tensor([2.0]).as_subclass(SimpleTFTensor)
        result = fn(x, y)
        self.assertEqual(result, torch.tensor([3.0]))
        self.assertFalse(torch._C._peek_should_skip_torch_function())

    def test_redispatch_function_graph_break(self):
        # [FUNCTION DOC] Same as the redispatch test but my_func graph
        # breaks after the torch function check on the non-dispatch path;
        # checks the result is 3.0 and the skip flag is restored to False.
        from torch.overrides import (
            handle_torch_function,
            has_torch_function,
            redispatch_function,
        )

        self.assertFalse(torch._C._peek_should_skip_torch_function())

        class SimpleTFTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Subclass whose torch
            # function redispatches via redispatch_function.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs and returns the redispatched result.
                if kwargs is None:
                    kwargs = {}
                return redispatch_function(func, types, args, kwargs)

        def my_func(a, b):
            # [FUNCTION DOC] HELPER: Dispatches through handle_torch_function when an
            # input has a torch function; otherwise graph breaks and returns the sum.
            if has_torch_function((a, b)):
                return handle_torch_function(my_func, (a, b), a, b)
            torch._dynamo.graph_break()
            return a + b

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls my_func on the two inputs.
            return my_func(x, y)

        x = torch.tensor([1.0]).as_subclass(SimpleTFTensor)
        y = torch.tensor([2.0]).as_subclass(SimpleTFTensor)
        result = fn(x, y)
        self.assertEqual(result, torch.tensor([3.0]))
        self.assertFalse(torch._C._peek_should_skip_torch_function())

    def test_redispatch_function_graph_break_before_has_torch_function(self):
        # [FUNCTION DOC] Variant where the graph break occurs before the torch function
        # check, exercising the case where the skip-next flag was set but not yet consumed
        # when the break fires. Result must be 3.0 and the skip flag must be False afterward.
        from torch.overrides import (
            handle_torch_function,
            has_torch_function,
            redispatch_function,
        )

        self.assertFalse(torch._C._peek_should_skip_torch_function())

        class SimpleTFTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Subclass whose torch
            # function redispatches via redispatch_function.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs and returns the redispatched result.
                if kwargs is None:
                    kwargs = {}
                return redispatch_function(func, types, args, kwargs)

        def my_func(a, b):
            # [FUNCTION DOC] HELPER: Graph breaks first, then dispatches through
            # handle_torch_function if an input has a torch function, else returns the sum.
            # Graph break BEFORE has_torch_function — this tests the case
            # where skip_next is set by _skip_one_hop_torch_function but
            # not yet consumed when the graph break fires.
            torch._dynamo.graph_break()
            if has_torch_function((a, b)):
                return handle_torch_function(my_func, (a, b), a, b)
            return a + b

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls my_func on the two inputs.
            return my_func(x, y)

        x = torch.tensor([1.0]).as_subclass(SimpleTFTensor)
        y = torch.tensor([2.0]).as_subclass(SimpleTFTensor)
        result = fn(x, y)
        self.assertEqual(result, torch.tensor([3.0]))
        self.assertFalse(torch._C._peek_should_skip_torch_function())

    def test_parameter_subclass_custom_torch_func_and_dynamic_attr(self):
        # [FUNCTION DOC] Checks that a Parameter subclass carrying quantization
        # metadata preserved through a custom torch function, with a method
        # converting to a plain tensor, compiles under fullgraph with the eager
        # backend without graph breaks; compiled and eager results must match.
        # This is a slight variation of
        # https://github.com/huggingface/diffusers/blob/fbf6b856cc61fd22ad8635547bff4aafe05723f3/src/diffusers/quantizers/gguf/utils.py#L398-L435
        # which basically
        # 1. uses tensor subclass to attach quantization metadata onto tensors
        # 2. preserve them across torch ops
        # 3. use the metadata to dequantize the tensor
        # 4. convert it to a regular tensor.
        #
        # The test is meant to make sure Dynamo won't graph break over it.
        class GGUFParameter(torch.nn.Parameter):
            # [CLASS DOC] HELPER: Parameter subclass with a quant_type attribute that its
            # torch function re-attaches to tensor results (and to tensors in tuple or list
            # results) using the quant_type of the first GGUFParameter argument found.
            def __new__(cls, data, requires_grad=False, quant_type=None):
                # [FUNCTION DOC] HELPER: Constructs the subclass from data (an empty
                # tensor if None) with requires_grad via _make_subclass and returns it.
                data = data if data is not None else torch.empty(0)
                self = torch.Tensor._make_subclass(cls, data, requires_grad)
                return self

            def __init__(self, *args, quant_type=None, **kwargs):
                # [FUNCTION DOC] HELPER: Stores the keyword quant_type on the instance.
                self.quant_type = quant_type

            def as_tensor(self):
                # [FUNCTION DOC] HELPER: Returns a plain tensor sharing
                # data via _make_subclass, preserving requires_grad.
                return torch.Tensor._make_subclass(
                    torch.Tensor, self, self.requires_grad
                )

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the default handler; finds quant_type
                # from the first argument that is a list starting with a GGUFParameter
                # or is a GGUFParameter itself; wraps tensor results in this class
                # with that quant_type, wraps tensor members of tuple or list results
                # preserving the container type, and returns other results unchanged.
                if kwargs is None:
                    kwargs = {}

                result = super().__torch_function__(func, types, args, kwargs)

                quant_type = None
                for arg in args:
                    if isinstance(arg, list) and isinstance(arg[0], GGUFParameter):
                        quant_type = arg[0].quant_type
                        break
                    if isinstance(arg, GGUFParameter):
                        quant_type = arg.quant_type
                        break
                if isinstance(result, torch.Tensor):
                    return cls(result, quant_type=quant_type)
                # Handle tuples and lists
                elif isinstance(result, (tuple, list)):
                    # Preserve the original type (tuple or list)
                    wrapped = [
                        (
                            cls(x, quant_type=quant_type)
                            if isinstance(x, torch.Tensor)
                            else x
                        )
                        for x in result
                    ]
                    return type(result)(wrapped)
                else:
                    return result

        def f(x):
            # [FUNCTION DOC] HELPER: Doubles x, adds its quant_type
            # to it, converts to a plain tensor and triples it.
            tmp = x * 2
            tmp = tmp + tmp.quant_type
            tmp = tmp.as_tensor()
            return tmp * 3

        opt_f = torch.compile(f, backend="eager", fullgraph=True)

        x = GGUFParameter(torch.ones(2), quant_type=42)
        res = f(x)
        ref = opt_f(x)
        self.assertEqual(res, ref)

    def test_newly_constructed_tensor_subclass_attr_mutation(self):
        # [FUNCTION DOC] Checks that an attribute set on a newly constructed tensor subclass object
        # inside a compiled function is applied during tracing and visible outside; compares tuple
        # results and the bar attribute of the returned object between eager and compiled.
        # Make sure the attribute mutation for newly constructed tensor subclass
        # object (from constructor call) is handled both during Dynamo tracing
        # and codegen-ed to be visible outside `torch.compile`.
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Empty tensor subclass.
            pass

        def f():
            # [FUNCTION DOC] HELPER: Constructs the subclass from ones, sets attribute
            # bar to 42 and returns the object together with itself multiplied by bar.
            x = MySubclass(torch.ones(2))
            x.bar = 42
            return x, x * x.bar

        opt_f = compile_full_eager(f)

        res = f()
        ref = opt_f()

        self.assertEqual(res, ref)
        self.assertEqual(res[0].bar, ref[0].bar)

    def test_as_subclass_attr_mutation(self):
        # [FUNCTION DOC] Same as the previous test but the subclass object is
        # created with as_subclass; checks results and the bar attribute match.
        # Make sure the attribute mutation for newly constructed tensor subclass
        # object (from as_subclass call) is handled both during Dynamo tracing
        # and codegen-ed to be visible outside `torch.compile`.
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Empty tensor subclass.
            pass

        def f():
            # [FUNCTION DOC] HELPER: Creates a ones tensor converted with as_subclass,
            # sets attribute bar to 42 and returns it with itself multiplied by bar.
            x = torch.ones(2).as_subclass(MySubclass)
            x.bar = 42
            return x, x * x.bar

        opt_f = compile_full_eager(f)

        res = f()
        ref = opt_f()

        self.assertEqual(res, ref)
        self.assertEqual(res[0].bar, ref[0].bar)

    def test_subclass_method_override(self):
        # [FUNCTION DOC] Checks that a subclass overriding sum is respected when compiled with the
        # eager backend (graph breaks allowed): result equals eager and has the subclass type.
        class MyTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass overriding sum to call
            # the base sum and convert the result back to this subclass.
            def sum(self, dim=None, keepdim=False):
                # [FUNCTION DOC] HELPER: Calls the base sum with dim
                # and keepdim and returns the result as MyTensor.
                return super().sum(dim=dim, keepdim=keepdim).as_subclass(MyTensor)

        def fn(x):
            # [FUNCTION DOC] HELPER: Converts x to MyTensor and returns its sum over dimension 1.
            y = x.as_subclass(MyTensor)
            return y.sum(dim=1)

        x = torch.randn(4, 10)
        fn_opt = torch.compile(fn, backend="eager", fullgraph=False)

        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)
        self.assertIsInstance(res_act, MyTensor)

    def test_tensor_subclass_attr_codegen_tos(self):
        # [FUNCTION DOC] Regression test for the interaction between attribute mutation
        # codegen of torch function override variables and the code generator top-of-stack
        # tracking: a function that builds a new subclass from the elem attribute of its
        # input must give equal tensors, equal elem and equal type compared to eager.
        # This repros a very subtle interaction between
        # `TensorWithTFOverrideVariable` attribute mutation codegen and
        # `PyCodegen.top_of_stack`. It was uncovered from
        # `test_tensor_subclass_deepcopy`.
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose constructor makes an empty subclass
            # instance via _make_subclass and stores the given elem as an attribute.
            def __new__(cls, elem, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the instance from a ones
                # tensor of size 0, sets attribute elem and returns it.
                r = torch.Tensor._make_subclass(cls, torch.ones(0))
                r.elem = elem
                return r

        def f(t):
            # [FUNCTION DOC] HELPER: Builds a new subclass
            # instance from a clone of the elem attribute of t.
            return MySubclass(t.elem.clone())

        opt_f = compile_full_eager(f)

        t = MySubclass(torch.ones(2))
        res = f(t)
        ref = opt_f(t)

        self.assertEqual(res, ref)
        self.assertEqual(res.elem, ref.elem)
        self.assertEqual(type(res), type(ref))

    def test_nontraceable_tensor_subclass(self):
        # [FUNCTION DOC] Checks that the nontraceable subclass configuration
        # prevents Dynamo from wrapping the instance as a tensor variable (which
        # would call the torch function and raise): with the config patched, the
        # compiled function reading a custom attribute foo must equal eager.
        # This will error if Dynamo tries to wrap it as a tensor variable,
        # because that involves calling certain methods to inspect the tensor
        # property, which will blow up in the overridden `__torch_function__`.
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose torch function always raises RuntimeError.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError("one shall not pass")

        def f(t):
            # [FUNCTION DOC] HELPER: Returns the foo attribute of t plus a ones tensor of size 10.
            return t.foo + torch.ones(10)

        opt_f = torch.compile(f, backend="eager", fullgraph=False)

        t = MySubclass(torch.ones(2))
        t.foo = 42
        # Make sure the `nontraceable_tensor_subclasses` config prevents Dynamo
        # from wrapping `t`.
        with nontraceable_subclass(MySubclass):
            res = f(t)
            ref = opt_f(t)

        self.assertEqual(res, ref)

    def test_compile_with_fake_tensor_dynamic_dim(self):
        # [FUNCTION DOC] Checks compilation of fake tensors created with a given per-
        # dimension dynamism policy. An inner driver resets Dynamo, runs the function
        # eagerly on real inputs, then compiles it under fullgraph with a counting
        # backend and feeds it two fake tensors from a fake mode with a stateless
        # symbolic context, asserting frame count and op count. The test runs it for
        # the dynamic, duck and static policies, each expecting one frame and one op.
        x = torch.randn([3, 4])

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return torch.sin(x)

        def test_dynamic_dim(f, x, dim_dynamic, exp_frame_count, exp_op_count):
            # [FUNCTION DOC] HELPER: Driver. Resets Dynamo, makes a counting backend and compiled
            # version of f, calls f eagerly on x and a differently shaped tensor, then inside a
            # fake tensor mode with a fresh shape env converts x and a random tensor of the same
            # shape to fake tensors with every dimension set to dim_dynamic and calls the compiled
            # function on both. Asserts frame_count and op_count equal the expected values.
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()

            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            x1 = torch.rand_like(x)
            f(x)
            f(torch.randn([4, 3]))
            shape_env = ShapeEnv()
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                x_fake = fake_mode.from_tensor(
                    x,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[dim_dynamic for i in range(x.dim())]
                    ),
                )
                x1_fake = fake_mode.from_tensor(
                    x1,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[dim_dynamic for i in range(x.dim())]
                    ),
                )
                opt_f(x_fake)
                opt_f(x1_fake)

            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        test_dynamic_dim(f, x, DimDynamic.DYNAMIC, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.DUCK, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.STATIC, 1, 1)

    def test_compile_with_fake_tensor_automatic_dynamic(self):
        # [FUNCTION DOC] Tests automatic dynamic shape promotion with fake tensors from
        # several shape sets. With dynamic or duck policies the inputs already have
        # symbolic sizes so only one compile occurs; with the static policy dimensions
        # that change across inputs become dynamic via recompilation (2 frames/ops for the
        # first set, 3 for the other two sets). Uses an inner driver to run each case.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return torch.sin(x)

        def test_automatic_dynamic(f, inps, dim_dynamic, exp_frame_count, exp_op_count):
            # [FUNCTION DOC] HELPER: Driver. Resets Dynamo, compiles f with a counting
            # backend under fullgraph, then inside a fake tensor mode converts each
            # input in inps to a fake tensor with every dimension set to dim_dynamic
            # (the rank is taken from the enclosing variable x) and calls the compiled
            # function. Asserts frame_count and op_count equal the expected values.
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()
            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            shape_env = ShapeEnv()
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                for inp in inps:
                    fake_inp = fake_mode.from_tensor(
                        inp,
                        symbolic_context=StatelessSymbolicContext(
                            [dim_dynamic for i in range(x.dim())]
                        ),
                    )
                    opt_f(fake_inp)
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        x = torch.randn([3, 4])
        y = torch.randn([4, 5])
        z = torch.randn([5, 6])
        a = torch.randn([3, 5])
        b = torch.randn([4, 4])
        # When inputs' DimDynamic is DYNAMIC or DUCK, the inputs
        # to opt_f will be tensors with SymInt sizes. Dynamo will treat input
        # as dynamic automatically and will only compile once
        for dim_dynamic in [DimDynamic.DYNAMIC, DimDynamic.DUCK]:
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 1, 1)

        for dim_dynamic in [DimDynamic.STATIC]:
            # Recompile once, first with dim 0 and 1 become Dynamic
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 2, 2)
            # Recompile 2 times, first with dim 1 become Dynamic, second with dim 0 becomes Dynamic.
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 3, 3)
            # Recompile 2 times, first with dim 0 become Dynamic, second with dim 1 becomes Dynamic.
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 3, 3)

    def test_parametrized_subclass_param_nested_graph_break(self):
        # [FUNCTION DOC] Regression test: under nested graph breaks, a module with a
        # tensor-subclass parameter (a wrapper tensor with custom sizes from the shared
        # subclass database) and a parametrization must not crash Dynamo when the value
        # stack is realized (previously failed because the parameter variable had no
        # graph proxy). Asserts the nested graph breaks config is on and that the
        # result of optimize with the eager backend is a WrapperTensorWithCustomSizes.
        # Regression test: under nested graph breaks, realizing the value stack
        # for a partial subgraph routes a module with a tensor-subclass parameter
        # through wrap_module -> mark_static_input. The subclass param is tracked
        # as a UserDefinedObjectVariable (no graph proxy node), which previously
        # crashed with "'UserDefinedObjectVariable' object has no attribute 'proxy'".
        from torch.nn.utils.parametrize import (
            register_parametrization,
            remove_parametrizations,
        )
        from torch.testing._internal.common_subclass import (
            subclass_db,
            WrapperTensorWithCustomSizes,
        )

        create_fn = subclass_db[WrapperTensorWithCustomSizes].create_fn

        def body():
            # [FUNCTION DOC] HELPER: Defines a module with a subclass weight and a parametrization
            # negating it, registers the parametrization, runs the module on a subclass input,
            # removes the parametrization keeping the parametrized value, and returns the output.
            class MyModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Module holding a parameter built from a wrapper
                # subclass tensor of shape 3 by 3; forward adds the weight to the input.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Calls the parent initializer and
                    # creates the weight parameter from a 3 by 3 subclass tensor.
                    super().__init__()
                    self.weight = torch.nn.Parameter(create_fn((3, 3)))

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the weight plus x.
                    return self.weight + x

            class MyParametrization(torch.nn.Module):
                # [CLASS DOC] HELPER: Parametrization module that negates its input.
                def forward(self, X):
                    # [FUNCTION DOC] HELPER: Returns the negation of X.
                    return -X

            m = MyModule()
            register_parametrization(m, "weight", MyParametrization())
            out = m(create_fn((3, 3)))
            remove_parametrizations(m, "weight", leave_parametrized=True)
            return out

        self.assertTrue(torch._dynamo.config.nested_graph_breaks)
        out = torch._dynamo.optimize("eager")(body)()
        self.assertIsInstance(out, WrapperTensorWithCustomSizes)

    def test_compile_with_functionalization(self):
        # [FUNCTION DOC] Checks Dynamo compilation of in-place ops with functionalization
        # active. Compiles a function doing in-place add and in-place relu on its input with a
        # recording backend and verifies the captured graph text, frame and op counts. Then runs
        # it under functorch functionalize (graph recompiled, input a functional tensor), under
        # a manual enable-functionalization wrapper, checks results are equal, and finally that
        # passing a view of a functional tensor raises RuntimeError about being unable to fakify
        # a view. The expected graph strings are compared with assertExpectedInline.
        x = torch.randn([3, 4])
        x_clone = x.clone()
        x_clone2 = x.clone()
        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: In-place adds 1.0 to x
            # and adds the result to the in-place relu of x.
            return x.add_(1.0) + torch.nn.functional.relu_(x)

        f_out = f(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.example_inputs), 1)

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        return (add,)
""",
        )

        ff = torch.func.functionalize(f)
        ff_out = ff(x_clone)

        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 6)
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(len(backend.example_inputs), 2)
        actual = normalize_gm(backend.graphs[1].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        return (add,)
""",
        )
        self.assertTrue(torch._is_functional_tensor(backend.example_inputs[1][0]))

        # Cannot reuse the version from AOTAutograd, since that uses python functional tensors.
        def to_fun(x):
            # [FUNCTION DOC] HELPER: Converts x to a functional tensor and mirrors
            # its autograd metadata onto it; returns the functional tensor.
            x_functional = torch._to_functional_tensor(x)
            torch._mirror_autograd_meta_to(x, x_functional)
            return x_functional

        def aot_f_wrapper(func):
            # [FUNCTION DOC] HELPER: Decorator that wraps func so it runs with functionalization
            # enabled (views not reapplied): converts positional and keyword arguments to
            # functional tensors, calls func, and always disables functionalization afterwards.
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Wrapper preserving func metadata. Enables
                # functionalization, maps the arguments to functional tensors via pytree,
                # returns the call result, and disables functionalization in a finally clause.
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(to_fun, args)
                    func_kwargs = pytree.tree_map(to_fun, kwargs)
                    return func(*func_args, **func_kwargs)
                finally:
                    torch._disable_functionalization()

            return wrapper

        aot_ff = aot_f_wrapper(f)
        aot_ff_out = aot_ff(x_clone2)

        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 9)
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(len(backend.example_inputs), 3)
        actual = normalize_gm(backend.graphs[2].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        return (add,)
""",
        )
        self.assertTrue(torch._is_functional_tensor(backend.example_inputs[1][0]))

        self.assertEqual(f_out, ff_out)
        self.assertEqual(f_out, aot_ff_out)

        try:
            torch._enable_functionalization(reapply_views=False)
            xf = pytree.tree_map(to_fun, x)
            x_view = xf.t()
            with self.assertRaisesRegex(RuntimeError, "Cannot safely fakify a view"):
                f(x_view)
        finally:
            torch._disable_functionalization()

    def test_compile_higher_order_with_functionalization(self):
        # [FUNCTION DOC] Same kind of functionalization test for a higher-order op:
        # compiles a function calling the wrap higher-order op around an in-place
        # add, then recompiles under functorch functionalize and under manual
        # functionalization, checking frame counts (1, 2, 3), op counts (2, 4, 6),
        # number of recorded graphs and the graph text each time via a helper.
        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Applies the wrap higher-order op to a
            # lambda that in-place adds 1.0 to its input and returns the result.
            return wrap(lambda x: x.add_(1.0), x)

        def check_count_and_graph(
            exp_frame_count, exp_op_count, exp_n_graph, exp_graph
        ):
            # [FUNCTION DOC] HELPER: Asserts the counter frame and op counts and the number
            # of recorded graphs, then compares the normalized readable text of the last
            # recorded graph to the expected string, passing skip 1 to the expect helper.
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)
            self.assertEqual(len(backend.graphs), exp_n_graph)
            actual = normalize_gm(
                backend.graphs[exp_n_graph - 1].print_readable(print_output=False)
            )
            self.assertExpectedInline(actual, exp_graph, skip=1)

        t = torch.randn([3, 4])
        t_clone = t.clone()
        t_clone2 = t.clone()
        f(t)

        check_count_and_graph(
            1,
            2,
            1,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 4]" = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
        )

        ff = torch.func.functionalize(f)
        ff_out = ff(t_clone)  # noqa: F841
        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(
            2,
            4,
            2,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 4]" = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
        )

        try:
            x = torch._to_functional_tensor(t_clone2)
            torch._mirror_autograd_meta_to(t_clone2, x)
            torch._enable_functionalization(reapply_views=False)
            aot_f_out = f(x)  # noqa: F841
        finally:
            torch._disable_functionalization()

        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(
            3,
            6,
            3,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 4]" = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
        )

    def test_has_torch_function(self):
        # [FUNCTION DOC] Checks the has_torch_function unary and variadic predicates under
        # compile. For both a non-tensor class with a classmethod torch function and a tensor
        # subclass, compares eager and compiled results on an instance and on the int 4.
        class MyTensor:
            # [CLASS DOC] HELPER: Plain class with a torch function returning a 123
            # tensor for torch.max and otherwise calling the function directly.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs; returns tensor of
                # 123 for torch.max, else calls func on args and kwargs.
                if kwargs is None:
                    kwargs = {}

                if func is torch.max:
                    return torch.tensor(123)
                return func(*args, **kwargs)

        class LocalSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose torch function calls
            # func directly on args and kwargs (with kwargs defaulted to empty).
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs and calls func directly.
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a pair of the unary
            # and variadic has-torch-function results for x.
            return torch.overrides.has_torch_function_unary(
                x
            ), torch.overrides.has_torch_function_variadic(x)

        for test_class in [MyTensor, LocalSubclass]:
            x = test_class()
            ref0 = fn(x)
            ref1 = fn(4)
            opt_fn = torch.compile(fn, backend="eager")
            res0 = opt_fn(x)
            res1 = opt_fn(4)
            self.assertEqual(ref0, res0)
            self.assertEqual(ref1, res1)

    def test_wrapper_subclass_guards_on_inner_tensor(self):
        # [FUNCTION DOC] Checks symbolic shape guards for a wrapper subclass whose outer
        # size is twice the inner tensor and whose dispatch adds an extra op when the inner
        # first dim exceeds 3. Compiles a branching function with a capturing backend,
        # marks the outer dim dynamic, then asserts the symbol to value map, symbol to
        # source name map, and the guard list (compared as inline expected text).
        # Holds an inner tensor, that has a distinct shape from the outer wrapper tensor.
        # Also adds additional guards on the inner tensor's sizes.
        # When the first input to an op has x.shape[0] > 5, we insert an extra add node.
        class DoubleSizeMaybeAddGeThreeTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper subclass holding inner_elem whose outer first dimension
            # is twice the inner one; its dispatch unwraps, runs the op, adds 2 to the output
            # when the inner first dim exceeds 3 (adding a guard on the inner size) and rewraps.
            @staticmethod
            def __new__(cls, inner):
                # [FUNCTION DOC] HELPER: Static constructor: builds a wrapper subclass
                # with doubled first dimension, inner strides, dtype, layout, device and
                # requires_grad via the positional overload of _make_wrapper_subclass.
                # KNOWN ISSUE: the in-code TODO notes the positional overload always
                # specializes sizes, so dynamic shape interaction is limited.
                # Double the outer-most dimension
                outer_shape = (inner.shape[0] * 2,) + inner.shape[1:]
                return torch.Tensor._make_wrapper_subclass(
                    # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
                    # Calling the overload that has kwargs causes us to go down the first overload path,
                    # which will **always** specialize sizes.
                    # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
                    cls,
                    outer_shape,
                    inner.stride(),
                    None,
                    None,
                    inner.dtype,
                    inner.layout,
                    inner.device,
                    False,
                    inner.requires_grad,
                )

            def __init__(self, inner):
                # [FUNCTION DOC] HELPER: Stores the inner tensor as inner_elem.
                self.inner_elem = inner

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Flatten hook: inner
                # attribute names (only inner_elem) and no metadata.
                return ["inner_elem"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, _, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Static unflatten hook: rebuilds the subclass
                # from the inner_elem entry; metadata, outer size and stride are unused.
                return DoubleSizeMaybeAddGeThreeTensor(inner_tensors["inner_elem"])

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a string with the repr of the inner
                # tensor (the class name in the string has a typo with an extra letter).
                return f"DoubleSizeMayberAddGeThreeTensor({repr(self.inner_elem)})"

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook: defaults kwargs, unwraps subclass
                # args to inner tensors, runs func, adds 2 in place to the output when
                # the first arg's first dim exceeds 3, and rewraps the output.
                if kwargs is None:
                    kwargs = {}

                args_inner = torch.utils._pytree.tree_map_only(
                    DoubleSizeMaybeAddGeThreeTensor, lambda x: x.inner_elem, args
                )
                out_inner = func(*args_inner, **kwargs)

                # Add guards on the  inner tensor's sizes
                if args_inner[0].shape[0] > 3:
                    out_inner += 2

                return DoubleSizeMaybeAddGeThreeTensor(out_inner)

        curr_var_to_val = None
        curr_var_to_sources = None
        guards = None

        def backend(gm, args):
            # [FUNCTION DOC] HELPER: Capturing backend. Reads the current tracing
            # context and stores into enclosing variables the guard expressions as
            # strings, the backed variable to value map and the variable to first source
            # name map from its shape env, then returns the graph module unchanged.
            context = torch._guards.TracingContext.get()

            # Grab info on sources and guards from the shapeenv
            nonlocal curr_var_to_val
            nonlocal curr_var_to_sources
            nonlocal guards

            guards = [str(g.expr) for g in context.fake_mode.shape_env.guards]
            curr_var_to_val = {
                str(k): v
                for k, v in context.fake_mode.shape_env.backed_var_to_val.items()
            }
            curr_var_to_sources = {
                str(k): v[0].name
                for k, v in context.fake_mode.shape_env.var_to_sources.items()
            }
            return gm

        @torch.compile(backend=backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x times x when its
            # first dimension is less than 13, else x divided by x.
            if x.shape[0] < 13:
                return torch.mul(x, x)
            else:
                return torch.div(x, x)

        inp = torch.ones(4, 4)

        x = DoubleSizeMaybeAddGeThreeTensor(inp)
        torch._dynamo.mark_dynamic(x, 0)
        res = fn(x)  # noqa: F841
        # During fakeifying, we end up allocating a separate symint
        # for the outer and inner tensor (in this test, s0 is unused).
        expected_var_to_val = {
            "s50": 4,
            "s77": 8,
        }
        expected_var_to_sources = {
            "s50": "L['x'].inner_elem.size()[0]",
            "s77": "L['x'].size()[0]",
        }
        self.assertEqual(curr_var_to_val, expected_var_to_val)
        self.assertEqual(curr_var_to_sources, expected_var_to_sources)
        self.assertExpectedInline(
            "\n".join(guards),
            """\
Eq(2*s50, s77)
2*s50 < 13
s50 > 3""",
        )

    def test_wrapper_subclass_with_same_sized_inner_tensor(self):
        # [FUNCTION DOC] Checks recompilation of a wrapper subclass whose scale tensor has the same
        # size as the outer dim: no recompile for different sizes with dynamic enabled; recompile
        # with different data size and dynamic disabled; no recompile when dimensions of the outer
        # tensor are marked dynamic first, since that carries over to the same-sized inner tensor.
        # shouldn't recompile for different sizes when dynamic=True
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(6))
        sub2 = ScaledTensor(torch.randn(3, 5), torch.randn(7))
        self.assertFalse(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=True))

        # should recompile for different data size when dynamic=False
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(6))
        sub2 = ScaledTensor(torch.randn(3, 5), torch.randn(6))
        self.assertTrue(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

        # avoid recompile using manual mark_dynamic() for different data size
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(6))
        # NB: mark_dynamic() on outer tensor should translate to inner tensors of the same size
        torch._dynamo.mark_dynamic(sub1, 0)
        torch._dynamo.mark_dynamic(sub1, 1)
        sub2 = ScaledTensor(torch.randn(3, 5), torch.randn(6))
        self.assertFalse(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

    def test_wrapper_subclass_with_differently_sized_inner_tensor(self):
        # [FUNCTION DOC] Checks that when the scale size differs from the outer size, changing
        # the scale size recompiles with dynamic disabled, and marking outer dimensions dynamic
        # does not propagate to the differently sized inner tensor so it still recompiles.
        # should recompile for different scale size when dynamic=False
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(3))
        sub2 = ScaledTensor(torch.randn(2, 4), torch.randn(5))
        self.assertTrue(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

        # still recompiles using manual mark_dynamic() on outer for different scale size
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(3))
        # NB: mark_dynamic() on outer tensor doesn't translate to inner tensors of different size
        torch._dynamo.mark_dynamic(sub1, 0)
        torch._dynamo.mark_dynamic(sub1, 1)
        sub2 = ScaledTensor(torch.randn(2, 4), torch.randn(5))
        self.assertTrue(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

    def test_recompiles_with_optional_inner_tensor(self):
        # [FUNCTION DOC] Checks recompilation with a subclass having an optional inner
        # tensor: same-structure inputs do not recompile, switching between with and without
        # the optional scale does, and the aot_eager compiled results equal eager for both.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        # sub1 does not have the optional tensor specified while sub2 does
        sub1 = OptionalScaledTensor(torch.randn(2, 4), None)
        sub2 = OptionalScaledTensor(torch.randn(2, 4), torch.randn(2, 4))

        # sanity check; don't recompile for same input
        self.assertFalse(_recompiles_for_inputs(f, (sub1,), (sub1,), dynamic=True))
        self.assertFalse(_recompiles_for_inputs(f, (sub2,), (sub2,), dynamic=True))

        # these should recompile; optional tensor changes between specified and unspecified
        self.assertTrue(_recompiles_for_inputs(f, (sub1,), (sub2,), dynamic=True))
        self.assertTrue(_recompiles_for_inputs(f, (sub2,), (sub1,), dynamic=True))

        f_compiled = torch.compile(f, backend="aot_eager")
        self.assertEqual(f(sub1)._data, f_compiled(sub1)._data)
        self.assertEqual(f(sub2)._data, f_compiled(sub2)._data)

    def test_torch_dispatch_subclass_guard_recompile(self):
        # [FUNCTION DOC] Checks guards for a dispatch subclass: compiles an
        # add on a TwoTensor, results equal eager; a second TwoTensor with
        # different values must not recompile (error on recompile is enabled),
        # while a plain tensor input must recompile and still match eager.
        x = torch.ones(2, 2)
        x_two = TwoTensor(x.clone(), x.clone())

        def fn(w):
            # [FUNCTION DOC] HELPER: Returns torch.add of w and 1.0.
            return torch.add(w, 1.0)

        fn_opt = torch.compile(backend="eager")(fn)

        ref = fn(x_two)
        res = fn_opt(x_two)
        self.assertEqual(ref, res)

        # ensure no recompilation on same input type
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn_opt(TwoTensor(x + 1, x + 2))

        # recompile!
        ref = fn(x)
        res = fn_opt(x)
        self.assertEqual(ref, res)

    def test_tensor_subclass_ctx_guards(self):
        # [FUNCTION DOC] Checks guards on subclass metadata: two instances with
        # equal constant do not recompile, one with a different constant does.
        x = CtxSubclassTensor(torch.ones(2), 3)
        x2 = CtxSubclassTensor(torch.ones(2), 3)
        x3 = CtxSubclassTensor(torch.ones(2), 4)
        _check_recompiles(self, lambda x: x * x, (x,), (x2,), False)
        _check_recompiles(self, lambda x: x * x, (x,), (x3,), True)

    def test_tensor_subclass_ctx_recursive_guards(self):
        # [FUNCTION DOC] Checks guards on nested subclass metadata: two TwoTensors
        # whose inner subclass instances differ in constant must trigger a recompile.
        x0 = torch.ones(2, 2)
        x1 = CtxSubclassTensor(x0.clone(), 2)
        x2 = CtxSubclassTensor(x0.clone(), 3)
        tt0 = TwoTensor(x0.clone(), x1)
        tt1 = TwoTensor(x0.clone(), x2)

        _check_recompiles(self, lambda x: x * x, (tt0,), (tt1,), True)

    def test_tensor_subclass_ctx_custom_guards_override(self):
        # [FUNCTION DOC] Checks that a subclass can override the metadata guard
        # classmethod: a guard allowing the new constant to be at least the original
        # means a larger constant does not recompile while a smaller one does.
        class CtxSubclassTensorCustomGuardFn(CtxSubclassTensor):
            # [CLASS DOC] HELPER: CtxSubclassTensor variant with a custom metadata guard
            # that passes when the original constant is less than or equal to the other.
            @classmethod
            def __metadata_guard__(cls, orig_data, other):
                # [FUNCTION DOC] HELPER: Classmethod guard hook returning whether the first item
                # of the original metadata is at most the first item of the other metadata.
                return orig_data[0] <= other[0]

        x = CtxSubclassTensorCustomGuardFn(torch.ones(2), 2)
        x2 = CtxSubclassTensorCustomGuardFn(torch.ones(2), 3)
        x3 = CtxSubclassTensorCustomGuardFn(torch.ones(2), 1)
        _check_recompiles(self, lambda x: x * x, (x,), (x2,), False)
        _check_recompiles(self, lambda x: x * x, (x,), (x3,), True)

    def test_tensor_subclass_ctx_custom_guards_error_arg_num(self):
        # [FUNCTION DOC] Checks that a custom metadata guard taking the wrong
        # number of arguments (one instead of two) is rejected: compiling a
        # function on such an instance must raise InternalTorchDynamoError with
        # a message saying the method must take exactly two metadata arguments.
        import torch._dynamo.exc

        class CtxSubclassTensorCustomGuardFn(CtxSubclassTensor):
            # [CLASS DOC] HELPER: CtxSubclassTensor variant whose metadata
            # guard classmethod wrongly accepts a single argument.
            @classmethod
            def __metadata_guard__(cls, y):
                # [FUNCTION DOC] HELPER: Malformed guard hook with one argument;
                # it always returns False and is expected never to run.
                # Shouldn't reach here
                return False

        x = CtxSubclassTensorCustomGuardFn(torch.ones(2), 3)
        self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            "Tensor subclass method __metadata_guard__ must take exactly two subclass metadata arguments",
            lambda: torch.compile(lambda x: x * x, backend="eager")(x),
        )

    def test_tensor_subclass_ctx_custom_guards_error_not_classmethod(self):
        # [FUNCTION DOC] Checks that defining the metadata guard as an
        # instance method rather than a classmethod makes compilation
        # raise InternalTorchDynamoError saying it must be a classmethod.
        import torch._dynamo.exc

        class CtxSubclassTensorCustomGuardFn(CtxSubclassTensor):
            # [CLASS DOC] HELPER: CtxSubclassTensor variant whose
            # metadata guard is an ordinary instance method.
            def __metadata_guard__(self, x, y):
                # [FUNCTION DOC] HELPER: Instance-method guard
                # (invalid) taking two arguments and returning False.
                return False

        x = CtxSubclassTensorCustomGuardFn(torch.ones(2), 3)
        self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            "Tensor subclass method __metadata_guard__ must be a classmethod",
            lambda: torch.compile(lambda x: x * x, backend="eager")(x),
        )

    def test_tensor_subclass_tensor_metadata_requires_custom_guard(self):
        # [FUNCTION DOC] Checks that a subclass whose flatten metadata contains
        # a tensor but has no custom guard raises InternalTorchDynamoError
        # mentioning that returned tensor metadata does not define a metadata
        # guard, and that the error text does not include a guard failure note.
        import torch._dynamo.exc

        x = TensorMetadataSubclass(torch.randn(4), torch.randn(5))
        with self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            "returned Tensor metadata.*does not define __metadata_guard__",
        ) as cm:
            torch.compile(lambda x: x * 2, backend="eager")(x)

        self.assertNotIn("Guard failed on the same frame", str(cm.exception))

    def test_tensor_subclass_tensor_metadata_custom_guard(self):
        # [FUNCTION DOC] Checks the custom guard that compares weights by value: metadata
        # with equal weights values does not recompile, metadata with different values does.
        x = TensorMetadataSubclassWithCustomGuard(torch.randn(4), torch.randn(5))
        same_metadata = TensorMetadataSubclassWithCustomGuard(
            torch.randn(4), x._weights.clone()
        )
        different_metadata = TensorMetadataSubclassWithCustomGuard(
            torch.randn(4), torch.randn(5)
        )

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x times 2.
            return x * 2

        _check_recompiles(self, fn, (x,), (same_metadata,), False)
        _check_recompiles(self, fn, (x,), (different_metadata,), True)

    def test_tensor_subclass_registered_pytree_metadata_without_key_flatten(self):
        # [FUNCTION DOC] Checks recompile behavior for subclass metadata of a
        # custom class registered as a pytree node without key-path
        # flattening: equal metadata (by value comparison) must not recompile
        # and different must. The registration is removed in a finally clause.
        class MetadataWithoutKeyFlatten:
            # [CLASS DOC] HELPER: Simple value holder with equality comparing
            # the held value; registered as a pytree node by the test.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores value.
                self.value = value

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: true only if
                # other is the same class and values are equal.
                return (
                    isinstance(other, MetadataWithoutKeyFlatten)
                    and self.value == other.value
                )

        pytree.register_pytree_node(
            MetadataWithoutKeyFlatten,
            lambda x: ([x.value], None),
            lambda values, _: MetadataWithoutKeyFlatten(values[0]),
        )

        try:
            x = TensorMetadataSubclass(torch.randn(4), MetadataWithoutKeyFlatten(5))
            same_metadata = TensorMetadataSubclass(
                torch.randn(4), MetadataWithoutKeyFlatten(5)
            )
            different_metadata = TensorMetadataSubclass(
                torch.randn(4), MetadataWithoutKeyFlatten(6)
            )

            def fn(x):
                # [FUNCTION DOC] HELPER: Returns x times 2.
                return x * 2

            _check_recompiles(self, fn, (x,), (same_metadata,), False)
            _check_recompiles(self, fn, (x,), (different_metadata,), True)
        finally:
            pytree._deregister_pytree_node(MetadataWithoutKeyFlatten)

    def test_tensor_subclass_metadata_with_symint(self):
        # [FUNCTION DOC] Checks that metadata containing a symbolic integer does not
        # over-guard: a fake tensor yields a dynamic symbolic size which is used as
        # constant metadata of a subclass; after compiling with dynamic shapes and
        # fullgraph, a second instance with a plain int constant must not trigger a
        # recompile because symbolic positions are replaced by wildcard comparisons.
        # TENSOR_SUBCLASS_METADATA_MATCH replaces SymInts in metadata with
        # _AnyCompare sentinels so that (a) deepcopy doesn't pull in the
        # ShapeEnv, (b) dynamic dims aren't over-guarded, and (c) unbacked
        # SymInts don't cause errors.
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC],
        )
        fake_t = fake_mode.from_tensor(torch.randn(8), symbolic_context=sym_ctx)
        sym_int = fake_t.shape[0]  # SymInt with hint 8

        # Construct a real tensor whose metadata contains a SymInt.
        x1 = CtxSubclassTensor(torch.randn(8, 8), sym_int)

        @torch.compile(backend="eager", dynamic=True, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x times x.
            return x * x

        f(x1)

        # Without the fix, the guard stores the SymInt's hint value (8)
        # and recompiles when it sees a different constant. With the fix,
        # the SymInt position is replaced by _AnyCompare so the guard
        # passes for any constant value.
        x2 = CtxSubclassTensor(torch.randn(8, 8), 3)
        _check_recompiles(self, f, (x1,), (x2,), False)

    def test_subclass_constructor_proxying(self):
        # [FUNCTION DOC] Checks that constructors of subclasses taking frozen dataclass or
        # namedtuple metadata arguments are traceable: defines a metadata dataclass, a namedtuple,
        # and a wrapper subclass, then compiles two functions (with eager backend, fullgraph) that
        # build nested metadata of each kind, construct the subclass and multiply it by itself.
        import dataclasses
        from collections import namedtuple
        from typing import Any

        @dataclasses.dataclass(frozen=True)
        class SubclassTensorArgs:
            # [CLASS DOC] HELPER: Frozen dataclass metadata record
            # with original_shape, device and inner_meta fields.
            original_shape: torch.Size
            device: torch.device
            inner_meta: Any

        SubclassTensorArgs2 = namedtuple(
            "SubclassTensorArgs2",
            [
                "original_shape",
                "device",
                "inner_meta",
            ],
        )

        class SubclassTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper subclass holding an inner tensor a plus
            # arbitrary metadata, flattening with the metadata as context. Its
            # dispatch unwraps, runs the op and wraps outputs with namedtuple
            # metadata built from output shape and device, then fixes aliasing.
            @staticmethod
            def __new__(cls, a, meta):
                # [FUNCTION DOC] HELPER: Static constructor creating a
                # wrapper subclass with the size, strides, offset, device,
                # layout, requires_grad and dtype of a; meta ignored.
                shape = a.shape
                kwargs = {}
                kwargs["strides"] = a.stride()
                kwargs["storage_offset"] = a.storage_offset()
                kwargs["device"] = a.device
                kwargs["layout"] = a.layout
                kwargs["requires_grad"] = a.requires_grad
                kwargs["dtype"] = a.dtype
                out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
                return out

            def __init__(self, a, meta):
                # [FUNCTION DOC] HELPER: Stores a and meta.
                self.a = a
                self.meta = meta

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a string with
                # the class name wrapping the inner tensor repr.
                a_repr = repr(self.a)
                return f"SubclassTensor({a_repr})"

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Flatten hook: inner
                # attribute a and the meta object as context.
                return ["a"], self.meta

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, _, __):
                # [FUNCTION DOC] HELPER: Static unflatten hook: rebuilds the
                # subclass from inner a and the meta; other arguments unused.
                a = inner_tensors["a"]
                return SubclassTensor(a, meta)

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch hook: unwraps args and kwargs
                # to inner tensors, runs func, wraps tensor outputs with fresh
                # namedtuple metadata and returns through the aliasing helper.
                if kwargs is None:
                    kwargs = {}
                args_a = pytree.tree_map(
                    lambda x: x.a if isinstance(x, SubclassTensor) else x, args
                )
                kwargs_a = pytree.tree_map(
                    lambda x: x.a if isinstance(x, SubclassTensor) else x, kwargs
                )
                out_a = func(*args_a, **kwargs_a)
                out = pytree.tree_map(
                    lambda x: (
                        SubclassTensor(x, SubclassTensorArgs2(x.shape, x.device, None))
                        if isinstance(x, torch.Tensor)
                        else x
                    ),
                    out_a,
                )
                return return_and_correct_aliasing(func, args, kwargs, out)

        @torch.compile(fullgraph=True, backend="eager")
        def f1(x):
            # [FUNCTION DOC] HELPER: Compiled function (first definition)
            # building nested dataclass metadata from the input shape and
            # device, constructing a subclass with it and returning its square.
            meta = SubclassTensorArgs(
                x.shape, x.device, SubclassTensorArgs(x.shape, x.device, None)
            )
            out = SubclassTensor(x, meta)
            return out * out

        x = torch.randn(3, 3)
        f1(x)

        @torch.compile(fullgraph=True, backend="eager")
        def f1(x):
            # [FUNCTION DOC] HELPER: Compiled function (second definition,
            # shadowing the first) doing the same with the namedtuple metadata.
            meta = SubclassTensorArgs2(
                x.shape, x.device, SubclassTensorArgs2(x.shape, x.device, None)
            )
            out = SubclassTensor(x, meta)
            return out * out

        x = torch.randn(3, 3)
        f1(x)

    def test_torch_function_subclass_survives_into_aot_autograd(self):
        # [FUNCTION DOC] Checks that torch function behavior of a subclass survives
        # into AOTAutograd, such that logic requiring torch function inside it is not
        # unexpectedly disabled after redispatch. Compiles with dynamic shapes an
        # unflatten of the last dim on a wrapper subclass whose torch function disables
        # subclass dispatch and calls the function; only checks it runs without error.
        # If you have a tensor subclass that relies on dispatch into the same op
        # without unwrapping and calling torch._C.DisableTorchFunctionSubclass(),
        # the torch function-ness will survive into AOTAutograd. Today, NestedTensor
        # actually relies on this behavior! Because that torch function logic
        # runs during AOTAutograd, this test tests that there is no logic below
        # that relies torch function that gets unexpectedly disabled after we
        # redispatch from the subclass's torch function.
        class SubTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper subclass holding an inner tensor _t. Its
            # torch function runs the call with subclass torch function disabled;
            # its dispatch unwraps, runs the op and rewraps tensor outputs.
            @staticmethod
            def __new__(cls, t):
                # [FUNCTION DOC] HELPER: Static constructor using the long positional overload
                # of _make_wrapper_subclass with the geometry of t and a sizes dispatch policy.
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    t.shape,
                    t.stride(),
                    t.storage_offset(),
                    torch.contiguous_format,
                    t.dtype,
                    torch.strided,
                    t.device,
                    False,
                    t.requires_grad,
                    "sizes",
                    False,
                    False,
                    None,
                )

            def __init__(self, t):
                # [FUNCTION DOC] HELPER: Calls the parent initializer then stores t as _t.
                super().__init__()
                self._t = t

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Flatten hook: inner attribute _t and empty context.
                return ["_t"], {}

            @staticmethod
            def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Static unflatten hook:
                # rebuilds the subclass from the _t inner tensor.
                t = inner_tensors["_t"]
                return SubTensor(t)

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a string with
                # the class name wrapping the inner tensor.
                return f"SubTensor({self._t})"

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Defaults kwargs and calls
                # func with subclass torch function disabled.
                if kwargs is None:
                    kwargs = {}

                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook: defaults kwargs, unwraps SubTensor
                # args, runs func, and wraps every tensor output in a SubTensor.
                kwargs = {} if kwargs is None else kwargs
                new_args = pytree.tree_map_only(SubTensor, lambda s: s._t, args)
                output = func(*new_args, **kwargs)
                output = pytree.tree_map_only(
                    torch.Tensor, lambda t: SubTensor(t), output
                )
                return output

        @torch.compile(dynamic=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Unflattens the last dim of x into sizes 2 and 5.
            return x.unflatten(-1, [2, 5])

        s = SubTensor(torch.randn(3, 10))
        f(s)

    # Guard validation upsets the guard
    # https://github.com/pytorch/pytorch/issues/129936
    @unittest.expectedFailure
    def test_recompile_with_symbool_inputs(self):
        # [FUNCTION DOC] Expected-failure test (guard validation issue 129936) of
        # recompilation when the input is a symbolic boolean derived from comparing a
        # fake dynamic size with a number. For two orderings of sizes it checks recorded
        # graphs, frame counts and shape env guard lists after each call. KNOWN ISSUE:
        # decorated expectedFailure because guard validation upsets the guard.
        def f(pred: bool):
            # [FUNCTION DOC] HELPER: Returns a ones tensor
            # of shape 3 by 4 if pred is true else 4 by 3.
            if pred:
                return torch.ones([3, 4])
            else:
                return torch.ones([4, 3])

        def test_recompilation(
            f, x, sizes, exp_graphs, exp_frame_count, exp_shape_env_guards
        ):
            # [FUNCTION DOC] HELPER: Driver. Resets Dynamo, compiles f with a recording
            # counting backend under fullgraph, builds a fake tensor with all dims dynamic,
            # and for each size computes a symbolic boolean from the first dim equal to the
            # size and calls the compiled function. After each call it checks the expected
            # graph text, frame count and the shape env guard expression strings.
            torch._dynamo.reset()
            shape_env = ShapeEnv()
            backend = torch._dynamo.testing.EagerAndRecordGraphs()
            cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
            f_cond = torch.compile(f, backend=cnt, fullgraph=True)
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                fake_inp = fake_mode.from_tensor(
                    x,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[DimDynamic.DYNAMIC for i in range(x.dim())]
                    ),
                )
                for i, size in enumerate(sizes):
                    pred = fake_inp.size(0) == size
                    f_cond(pred)
                    actual = normalize_gm(
                        backend.graphs[exp_frame_count[i] - 1].print_readable(
                            print_output=False
                        )
                    )
                    actual_guard_str = [str(guard.expr) for guard in shape_env.guards]
                    self.assertExpectedInline(actual, exp_graphs[i])
                    self.assertEqual(cnt.frame_count, exp_frame_count[i])
                    self.assertEqual(actual_guard_str, exp_shape_env_guards[i])

        true_graph = """\
class GraphModule(torch.nn.Module):
    def forward(self):
        ones: "f32[3, 4]" = torch.ones([3, 4])
        return (ones,)
"""
        false_graph = """\
class GraphModule(torch.nn.Module):
    def forward(self):
        ones: "f32[4, 3]" = torch.ones([4, 3])
        return (ones,)
"""
        test_recompilation(
            f,
            torch.randn([3, 4]),
            [3, 3, 4, 5],
            exp_graphs=[true_graph, true_graph, false_graph, false_graph],
            exp_frame_count=[1, 1, 2, 2],
            exp_shape_env_guards=[
                [],
                # s0 is specialized and guarded in outer shape_env when dynamo checks the guards
                ["Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)"],
                [
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 4)), (0, True)), 1)",
                ],
                [
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 4)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                ],
            ],
        )

        test_recompilation(
            f,
            torch.randn([3, 4]),
            [4, 5, 3, 3],
            exp_graphs=[false_graph, false_graph, true_graph, true_graph],
            exp_frame_count=[1, 1, 2, 2],
            exp_shape_env_guards=[
                [],
                # s0 is specialized and guarded in outer shape_env when dynamo checks the guards
                ["Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)"],
                [
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                ],
                [
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                ],
            ],
        )

    def test_wrapper_subclass_dynamo_attribute_access_on_intermediate(self):
        # [FUNCTION DOC] Checks attribute access on an intermediate wrapper subclass with
        # aot_eager under fullgraph. KNOWN ISSUE: the inner function takes x_subclass but uses the
        # enclosing variable x, so it ignores its argument and relies on the outer ScaledTensor.
        def f(x_subclass):
            # [FUNCTION DOC] HELPER: Adds 1 to the enclosing ScaledTensor named x (not its
            # parameter) and returns the product of the result's scale and constant attributes.
            tmp_subclass = torch.add(x, 1)
            return torch.mul(tmp_subclass._scale, tmp_subclass._constant)

        x = ScaledTensor(torch.randn(2, 4), torch.randn(3), constant=2)
        out_ref = f(x)
        out_test = torch.compile(f, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(out_ref, out_test)

    def test_support_bases(self):
        # [FUNCTION DOC] Checks that reading base classes of a class created with a custom
        # metaclass is supported under fullgraph: the tuple of bases of the instance type
        # must equal the abstract base, and the single base attribute likewise.
        import torch.fx._symbolic_trace

        class Meta(abc.ABCMeta, torch.fx._symbolic_trace.ProxyableClassMeta):
            # [CLASS DOC] HELPER: Metaclass combining ABC and the proxyable class
            # metaclass whose class creation sets attribute attr to 100 on each new class.
            def __new__(cls, name, bases, dct):
                # [FUNCTION DOC] HELPER: Creates the class via the parent
                # and sets attribute attr to 100 before returning it.
                x = super().__new__(cls, name, bases, dct)
                x.attr = 100
                return x

        class Multistreamable(abc.ABC):  # noqa: B024
            # [CLASS DOC] HELPER: Empty abstract base class.
            pass

        class Foo(Multistreamable, metaclass=Meta):
            # [CLASS DOC] HELPER: Class using the custom
            # metaclass and deriving from the abstract base.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Accesses and returns the
            # tuple of bases of the type of a new Foo instance.
            typ = type(Foo())
            typ.__bases__
            return typ.__bases__

        self.assertEqual(f(torch.randn(1)), (Multistreamable,))

        @torch.compile(backend="eager", fullgraph=True)
        def g(x):
            # [FUNCTION DOC] HELPER: Accesses and returns the
            # single base class of the type of a new Foo instance.
            typ = type(Foo())
            typ.__base__
            return typ.__base__

        self.assertEqual(g(torch.randn(1)), Multistreamable)

    @parametrize("dynamic", [False, True])
    def test_subclass_views(self, dynamic):
        # [FUNCTION DOC] Parametrized by dynamic flag. Takes various views (narrow,
        # split, split with sizes, expand, select, slicing, view) of a TwoTensor,
        # passes each to a doubling function compiled with aot_eager and fullgraph
        # and compares to eager. For the slicing view with dynamic shapes an
        # AssertionError is expected (issue 128649). Dynamo is reset per view.
        def _get_views(t):  # returns (view: Tensor, expects_raises_false)
            # [FUNCTION DOC] HELPER: Generator over views of t yielding pairs of a
            # view tensor and a flag telling whether compiling it is expected to
            # raise AssertionError (true only for the slicing view when dynamic).
            # Note that any closed-over SymInts will be symbolicized during fake-ification.
            yield t.narrow(dim=-1, start=3, length=8), False
            yield t.split(5, -1)[2], False
            yield t.split_with_sizes([9, 6], -1)[1], False
            yield t.unsqueeze(-1).expand(4, 15, 10), False
            yield t.select(-1, 6), False
            # https://github.com/pytorch/pytorch/issues/128649
            yield t[2:3, 5:9], dynamic
            yield t.view(-1, 15), False

        def f(x):
            # [FUNCTION DOC] HELPER: Returns x times 2.
            return x * 2

        compiled_f = torch.compile(
            f, backend="aot_eager", fullgraph=True, dynamic=dynamic
        )

        # Take a view of a subclass to pass as input.
        t = TwoTensor(torch.randn(4, 15), torch.randn(4, 15))
        for view, expects_raises in _get_views(t):
            torch._dynamo.reset()
            out_ref = f(view)
            if expects_raises:
                with self.assertRaises(AssertionError):
                    out_test = compiled_f(view)
            else:
                out_test = compiled_f(view)
                self.assertEqual(out_ref, out_test)

    @parametrize("dynamic", [True, False])
    def test_mark_static_with_subclass_desugaring(self, dynamic):
        # [FUNCTION DOC] Parametrized by dynamic flag. Marks a TwoTensor as a
        # static address and compiles through compile_fx with a custom inner
        # compile that asserts which input indexes were treated as static (indexes
        # 2 and 4 when dynamic, 1 and 2 otherwise) and returns the graph unchanged.
        from collections.abc import Callable
        from typing import Any

        from torch._dynamo.decorators import mark_static_address
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.cudagraph_utils import BoxedDeviceIndex
        from torch._inductor.utils import BoxedBool

        x_inner = torch.ones(4)
        x = TwoTensor(x_inner, x_inner)
        mark_static_address(x, guard=False)

        def inner_compile(
            gm: torch.fx.GraphModule,
            example_inputs: list[torch.Tensor],
            cudagraphs: BoxedBool | None = None,
            static_input_idxs: list[int] | None = None,
            is_backward: bool = False,
            graph_id: int | None = None,
            cpp_wrapper: bool = False,
            aot_mode: bool = False,
            is_inference: bool = False,
            boxed_forward_device_index: BoxedDeviceIndex | None = None,
            layout_opt: bool | None = None,
            extern_node_serializer: Callable[[list[Any]], Any] | None = None,
            **kwargs: Any,
        ):
            # [FUNCTION DOC] HELPER: Stand-in inner compiler accepting the inductor
            # inner compile keyword arguments; asserts static_input_idxs equals the
            # expected list for the dynamic setting and returns gm unchanged.
            if dynamic:
                self.assertEqual(static_input_idxs, [2, 4])
            else:
                self.assertEqual(static_input_idxs, [1, 2])
            return gm

        compiler = functools.partial(compile_fx, inner_compile=inner_compile)

        @torch.compile(backend=compiler, dynamic=dynamic)
        def fn(t0, t1, t2):
            # [FUNCTION DOC] HELPER: Returns the sum of three inputs plus 2.
            return t0 + t1 + t2 + 2

        fn(torch.ones(4), x, torch.ones(4))

    def test_subclass_parameters_are_static_under_training(self):
        # [FUNCTION DOC] Checks that parameters that are subclass tensors are marked
        # static under training: compiles a linear module whose weight is a TwoTensor
        # parameter through compile_fx with a stand-in inner compiler asserting
        # exactly three static input indexes (two weight parts and the bias).
        from collections.abc import Callable
        from typing import Any

        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.cudagraph_utils import BoxedDeviceIndex
        from torch._inductor.utils import BoxedBool

        def inner_compile(
            gm: torch.fx.GraphModule,
            example_inputs: list[torch.Tensor],
            cudagraphs: BoxedBool | None = None,
            static_input_idxs: list[int] | None = None,
            is_backward: bool = False,
            graph_id: int | None = None,
            cpp_wrapper: bool = False,
            aot_mode: bool = False,
            is_inference: bool = False,
            boxed_forward_device_index: BoxedDeviceIndex | None = None,
            layout_opt: bool | None = None,
            extern_node_serializer: Callable[[list[Any]], Any] | None = None,
            **kwargs: Any,
        ):
            # [FUNCTION DOC] HELPER: Stand-in inner compiler accepting the inductor keyword
            # arguments; asserts the number of static input indexes is 3 and returns gm unchanged.
            # There are 3 params: linear.weight.a, linear.weight.b, linear.bias.
            # They should all be marked static. The specific indices depend
            # on graph input ordering (which canonicalization may change).
            self.assertEqual(len(static_input_idxs), 3)
            return gm

        compiler = functools.partial(compile_fx, inner_compile=inner_compile)

        mod = torch.nn.Linear(4, 4)
        w_a = torch.randn(4, 4)
        w_b = torch.randn(4, 4)
        w = torch.nn.Parameter(TwoTensor(w_a, w_b).requires_grad_())
        mod.weight = w

        mod = torch.compile(mod, backend=compiler)

        mod(torch.randn(4))

    def test_deferred_init_subclass_init_not_traced(self):
        # [FUNCTION DOC] Regression test: compiling the initializer of a traceable
        # wrapper subclass directly (partially initialised self) must not raise
        # from the flatten hook. Calls the compiled initializer on a shell object
        # from the constructor and checks _data and _scale are set afterward.
        """
        Tracing a function that constructs a DeferredInitSubclass must not crash.

        The bug: when Dynamo's frame hook intercepts __init__ as a root frame,
        self is partially initialised (attributes not yet set by __init__).
        Previously, wrap_tensor would call __tensor_flatten__ on this
        partially-initialised self and raise AttributeError.

        The fix skips tracing __init__ of traceable wrapper subclasses at the
        frame level (convert_frame.py), so __init__ runs eagerly like
        @torch._disable_dynamo would.
        """
        # Compile __init__ directly, simulating the root-frame interception
        # scenario that occurs in practice (e.g. Diffusers + TorchAO + Dynamo).
        compiled_init = torch.compile(
            DeferredInitSubclass.__init__, backend="eager", fullgraph=False
        )
        data = torch.randn(4, 4)
        shell = DeferredInitSubclass.__new__(DeferredInitSubclass, data, 2.0)

        # Should not raise AttributeError from __tensor_flatten__ on partial self
        compiled_init(shell, data, 2.0)

        self.assertEqual(shell._data, data)
        self.assertEqual(shell._scale, 2.0)

    def test_tensor_subclass_super_new(self):
        # [FUNCTION DOC] Checks that calling the parent new with a tensor inside a
        # tensor subclass constructor is traceable under fullgraph: the compiled
        # function returns an instance of the subclass equal to the input.
        # super().__new__(cls, tensor) should be traceable in Tensor subclasses
        class MyTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose
            # constructor calls the parent constructor with x.
            def __new__(cls, x):
                # [FUNCTION DOC] HELPER: Constructor delegating to the parent new with x.
                return super().__new__(cls, x)

        @torch.compile(backend="eager", fullgraph=True)
        def forward(x):
            # [FUNCTION DOC] HELPER: Constructs and returns MyTensor from x.
            return MyTensor(x)

        x = torch.randn(4, 10)
        result = forward(x)
        self.assertIsInstance(result, MyTensor)
        self.assertEqual(result, x)


instantiate_parametrized_tests(SubclassTests)


class TestTwoTensorSubclass(
    _SubclassCompileCheckMixin, torch._dynamo.test_case.TestCase
):
    # [CLASS DOC] Tests for TwoTensor wrapper subclass tracing under Dynamo,
    # comparing eager against AOT-compiled behavior and the generated forward and
    # backward graphs. Uses the compile-check mixin with the Dynamo TestCase.
    """Tests for TwoTensor wrapper subclass tracing under dynamo."""

    def test_tensor_subclass_TwoTensor_simple(self):
        # [FUNCTION DOC] Compiles a function multiplying a TwoTensor by its first
        # dimension size (dynamic shapes, backward called) using the mixin helper,
        # and compares the printed forward and backward graph modules to expected
        # inline text, covering the subclass inputs, symbolic sizes and strides.
        def f(tt):
            # [FUNCTION DOC] HELPER: Multiplies tt by the first dimension of its size.
            return tt * tt.size()[0]

        a = torch.ones(3, 4, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a')
        primals_2: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_3: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=1)
        primals_4: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_5: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b')
        primals_6: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_7: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=1)
        primals_8: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_9: "Sym(s47)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_10: "Sym(s16)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=1)
        primals_11: "Sym(s16)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
    ):
        mul: "f32[s47, s16]" = torch.ops.aten.mul.Tensor(primals_1, primals_12);  primals_1 = None
        mul_3: "f32[s47, s16]" = torch.ops.aten.mul.Tensor(primals_5, primals_12);  primals_5 = None
        return (
            mul,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=0)
            mul_3,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_12,  # SavedForBackwardsAOTOutput(idx=0)
            primals_13,  # SavedForBackwardsAOTOutput(idx=1)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
        tangents_1: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='a')
        tangents_2: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='b')
    ):
        mul_8: "f32[s47, s16]" = torch.ops.aten.mul.Tensor(tangents_1, primals_12);  tangents_1 = None
        mul_9: "f32[s47, s16]" = torch.ops.aten.mul.Tensor(tangents_2, primals_12);  tangents_2 = None
        return (
            mul_8,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            mul_9,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            None,  # None
            None,  # None
        )
""",
        )

    def test_tensor_subclass_TwoTensor_clone_view(self):
        # [FUNCTION DOC] Compiles a function that clones a TwoTensor and views it
        # with swapped dimensions (dynamic shapes, with backward) and compares
        # the forward and backward graph text to expected inline strings.
        def f(tt):
            # [FUNCTION DOC] HELPER: Clones tt and returns a view with the dimensions swapped.
            y = tt.clone()
            return y.view(y.shape[1], y.shape[0])

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a')
        primals_2: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_3: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=1)
        primals_4: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_5: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b')
        primals_6: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_7: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=1)
        primals_8: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_9: "Sym(s47)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_10: "Sym(s16)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=1)
        primals_11: "Sym(s16)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
    ):
        clone: "f32[s47, s16]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[s47, s16]" = torch.ops.aten.clone.default(primals_5);  primals_5 = None

        view: "f32[s16, s47]" = torch.ops.aten.view.default(clone, [primals_13, primals_12]);  clone = None
        view_1: "f32[s16, s47]" = torch.ops.aten.view.default(clone_1, [primals_13, primals_12]);  clone_1 = None
        return (
            view,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a')
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=1)
            primals_12,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=0)
            view_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b')
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=1)
            primals_12,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=1)
            primals_12,  # SubclassStrideAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_12,  # SavedForBackwardsAOTOutput(idx=0)
            primals_13,  # SavedForBackwardsAOTOutput(idx=1)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
        tangents_1: "f32[s16, s47]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='a')
        tangents_2: "f32[s16, s47]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='b')
    ):
        view_2: "f32[s47, s16]" = torch.ops.aten.view.default(tangents_1, [primals_12, primals_13]);  tangents_1 = None
        view_3: "f32[s47, s16]" = torch.ops.aten.view.default(tangents_2, [primals_12, primals_13]);  tangents_2 = None
        return (
            view_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            view_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            None,  # None
            None,  # None
        )
""",
        )

    def test_tensor_subclass_TwoTensor_mul(self):
        # [FUNCTION DOC] Compiles a function multiplying a TwoTensor by several
        # size values taken from two other tensors (dynamic shapes, with backward)
        # and compares the forward and backward graph text to expected strings.
        def f(tt, a, b):
            # [FUNCTION DOC] HELPER: Unpacks the sizes of a and b
            # into four ints and returns tt multiplied by all four.
            s0, s1 = a.size()
            s2, s3 = b.size()
            # return tt * a.size()[1]
            return tt * s0 * s1 * s2 * s3

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt, a, b)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "Sym(s97)",  # PlainAOTInput(idx=0)
        primals_2: "Sym(s98)",  # PlainAOTInput(idx=1)
        primals_3: "f32[s97, s98]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=2), attr='a')
        primals_4: "Sym(s97)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=2), attr='a'), idx=0)
        primals_5: "Sym(s98)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=2), attr='a'), idx=1)
        primals_6: "Sym(s98)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=2), attr='a'), idx=0)
        primals_7: "f32[s97, s98]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=2), attr='b')
        primals_8: "Sym(s97)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=2), attr='b'), idx=0)
        primals_9: "Sym(s98)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=2), attr='b'), idx=1)
        primals_10: "Sym(s98)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=2), attr='b'), idx=0)
        primals_11: "Sym(s97)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=2), idx=0)
        primals_12: "Sym(s98)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=2), idx=1)
        primals_13: "Sym(s98)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=2), idx=0)
    ):
        mul: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(primals_3, primals_1);  primals_3 = None
        mul_3: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(primals_7, primals_1);  primals_7 = None
        mul_8: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul, primals_2);  mul = None
        mul_11: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_3, primals_2);  mul_3 = None
        mul_16: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_8, primals_1);  mul_8 = None
        mul_19: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_11, primals_1);  mul_11 = None
        mul_24: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_16, primals_2);  mul_16 = None
        mul_27: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_19, primals_2);  mul_19 = None
        return (
            mul_24,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a')
            primals_11,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=0)
            mul_27,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b')
            primals_11,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=0)
            primals_11,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_1,  # SavedForBackwardsAOTOutput(idx=0)
            primals_2,  # SavedForBackwardsAOTOutput(idx=1)
            primals_11,  # SavedForBackwardsAOTOutput(idx=2)
            primals_13,  # SavedForBackwardsAOTOutput(idx=3)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "Sym(s97)",  # PlainAOTInput(idx=0)
        primals_2: "Sym(s98)",  # PlainAOTInput(idx=1)
        primals_11: "Sym(s97)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=2), idx=0)
        primals_13: "Sym(s98)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=2), idx=0)
        tangents_1: "f32[s97, s98]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='a')
        tangents_2: "f32[s97, s98]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='b')
    ):
        mul_32: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  tangents_1 = None
        mul_33: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(tangents_2, primals_2);  tangents_2 = None
        mul_34: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_32, primals_1);  mul_32 = None
        mul_35: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_33, primals_1);  mul_33 = None
        mul_36: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_34, primals_2);  mul_34 = None
        mul_37: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_35, primals_2);  mul_35 = primals_2 = None
        mul_38: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_36, primals_1);  mul_36 = None
        mul_39: "f32[s97, s98]" = torch.ops.aten.mul.Tensor(mul_37, primals_1);  mul_37 = primals_1 = None
        return (
            None,  # None
            None,  # None
            mul_38,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), attr='a')
            primals_11,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), attr='a'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), attr='a'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), attr='a'), idx=0)
            mul_39,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), attr='b')
            primals_11,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), attr='b'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), attr='b'), idx=0)
            primals_11,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=2)), idx=0)
        )
""",
        )

    def test_tensor_subclass_TwoTensor_view(self):
        # [FUNCTION DOC] Compiles a function that clones a TwoTensor and views
        # it with its own dimensions (dynamic shapes, with backward) and
        # compares the forward and backward graph text to expected strings.
        def f(tt):
            # [FUNCTION DOC] HELPER: Clones tt and returns a view using its own two dimensions.
            y = tt.clone()
            return y.view(y.shape[0], y.shape[1])

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a')
        primals_2: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_3: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=1)
        primals_4: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_5: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b')
        primals_6: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_7: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=1)
        primals_8: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_9: "Sym(s47)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_10: "Sym(s16)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=1)
        primals_11: "Sym(s16)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
    ):
        clone: "f32[s47, s16]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[s47, s16]" = torch.ops.aten.clone.default(primals_5);  primals_5 = None

        view: "f32[s47, s16]" = torch.ops.aten.view.default(clone, [primals_12, primals_13]);  clone = None
        view_1: "f32[s47, s16]" = torch.ops.aten.view.default(clone_1, [primals_12, primals_13]);  clone_1 = None
        return (
            view,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=0)
            view_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_12,  # SavedForBackwardsAOTOutput(idx=0)
            primals_13,  # SavedForBackwardsAOTOutput(idx=1)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
        tangents_1: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='a')
        tangents_2: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='b')
    ):
        view_2: "f32[s47, s16]" = torch.ops.aten.view.default(tangents_1, [primals_12, primals_13]);  tangents_1 = None
        view_3: "f32[s47, s16]" = torch.ops.aten.view.default(tangents_2, [primals_12, primals_13]);  tangents_2 = None
        return (
            view_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            view_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            None,  # None
            None,  # None
        )
""",
        )

    def test_tensor_subclass_TwoTensor_view_mul(self):
        # [FUNCTION DOC] Compiles a function that clones a TwoTensor and views it as one
        # dimension whose size is the product of the dimensions (dynamic shapes, with
        # backward) and compares the forward and backward graph text to expected strings.
        def f(tt):
            # [FUNCTION DOC] HELPER: Clones tt and returns a flattened
            # view sized as the product of its two dimensions.
            y = tt.clone()
            return y.view(y.shape[0] * y.shape[1])

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a')
        primals_2: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_3: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=1)
        primals_4: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_5: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b')
        primals_6: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_7: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=1)
        primals_8: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_9: "Sym(s47)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_10: "Sym(s16)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=1)
        primals_11: "Sym(s16)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
    ):
        clone: "f32[s47, s16]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[s47, s16]" = torch.ops.aten.clone.default(primals_5);  primals_5 = None

        mul_6: "Sym(s16*s47)" = primals_12 * primals_13
        view: "f32[s16*s47]" = torch.ops.aten.view.default(clone, [mul_6]);  clone = None
        view_1: "f32[s16*s47]" = torch.ops.aten.view.default(clone_1, [mul_6]);  clone_1 = None
        return (
            view,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a')
            mul_6,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a'), idx=0)
            view_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b')
            mul_6,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b'), idx=0)
            mul_6,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_12,  # SavedForBackwardsAOTOutput(idx=0)
            primals_13,  # SavedForBackwardsAOTOutput(idx=1)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
        tangents_1: "f32[s16*s47]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='a')
        tangents_2: "f32[s16*s47]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='b')
    ):
        view_2: "f32[s47, s16]" = torch.ops.aten.view.default(tangents_1, [primals_12, primals_13]);  tangents_1 = None
        view_3: "f32[s47, s16]" = torch.ops.aten.view.default(tangents_2, [primals_12, primals_13]);  tangents_2 = None
        return (
            view_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            view_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            None,  # None
            None,  # None
        )
""",
        )

    def test_tensor_subclass_TwoTensor_return_tensor_and_subclass(self):
        # [FUNCTION DOC] Compiles a function returning an inner tensor of a cloned
        # TwoTensor together with a flattened view of the TwoTensor (dynamic shapes, with
        # backward) and compares forward and backward graph text to expected strings.
        def f(tt):
            # [FUNCTION DOC] HELPER: Clones tt and returns its
            # a component and a flattened view of the clone.
            y = tt.clone()
            return y.a, y.view(y.shape[0] * y.shape[1])

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a')
        primals_2: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_3: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=1)
        primals_4: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_5: "f32[s47, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b')
        primals_6: "Sym(s47)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_7: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=1)
        primals_8: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_9: "Sym(s47)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_10: "Sym(s16)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=1)
        primals_11: "Sym(s16)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
    ):
        clone: "f32[s47, s16]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[s47, s16]" = torch.ops.aten.clone.default(primals_5);  primals_5 = None

        mul_6: "Sym(s16*s47)" = primals_12 * primals_13
        view: "f32[s16*s47]" = torch.ops.aten.view.default(clone, [mul_6])
        view_1: "f32[s16*s47]" = torch.ops.aten.view.default(clone_1, [mul_6]);  clone_1 = None
        return (
            clone,  # PlainAOTOutput(idx=0)
            view,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='a')
            mul_6,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='a'), idx=0)
            view_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='b')
            mul_6,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='b'), idx=0)
            mul_6,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=1), idx=0)
            primals_12,  # SavedForBackwardsAOTOutput(idx=0)
            primals_13,  # SavedForBackwardsAOTOutput(idx=1)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_12: "Sym(s47)",  # PlainAOTInput(idx=1)
        primals_13: "Sym(s16)",  # PlainAOTInput(idx=2)
        tangents_1: "f32[s16*s47]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=1)), attr='a')
        tangents_2: "f32[s16*s47]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=1)), attr='b')
    ):
        view_2: "f32[s47, s16]" = torch.ops.aten.view.default(tangents_1, [primals_12, primals_13]);  tangents_1 = None
        view_3: "f32[s47, s16]" = torch.ops.aten.view.default(tangents_2, [primals_12, primals_13]);  tangents_2 = None
        return (
            view_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            view_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b')
            primals_12,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_12,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            primals_13,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=1)
            primals_13,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            None,  # None
            None,  # None
        )
""",
        )

    @unittest.expectedFailure
    def test_tensor_subclass_TwoTensor_return_multiple(self):
        # [FUNCTION DOC] Expected failure test: a function returning several components
        # and views of cloned TwoTensors is compiled with dynamic shapes and backward
        # and its graphs compared to expected strings. KNOWN ISSUE: decorated
        # expectedFailure so the expected graphs presumably do not match current output.
        def f(tt):
            # [FUNCTION DOC] HELPER: Clones tt twice and returns the a part and a flattened view
            # of the first clone, its b part, and a view of the second clone with inferred size.
            y = tt.clone()
            z = tt.clone()
            return y.a, y.view(y.shape[0] * y.shape[1]), y.b, z.view(-1)

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 4]", primals_2: "f32[3, 4]", primals_3: "Sym(3)", primals_4: "Sym(4)", primals_5: "Sym(3)", primals_6: "Sym(4)"):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[3, 4]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        mul: "Sym(12)" = primals_5 * primals_6
        view: "f32[12]" = torch.ops.aten.view.default(clone, [mul])
        view_1: "f32[12]" = torch.ops.aten.view.default(clone_1, [mul]);  clone_1 = None
        return [clone, view, view_1, mul, primals_5, primals_6]
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_5: "Sym(3)", primals_6: "Sym(4)", tangents_1: "f32[12]", tangents_2: "f32[12]"):
        view_2: "f32[3, 4]" = torch.ops.aten.view.default(tangents_1, [primals_5, primals_6]);  tangents_1 = None
        view_3: "f32[3, 4]" = torch.ops.aten.view.default(tangents_2, [primals_5, primals_6]);  tangents_2 = primals_5 = primals_6 = None
        return [view_2, view_3, None, None]
""",
        )

    def test_tensor_subclass_TwoTensor_automatic_dynamic_shapes(self):
        # [FUNCTION DOC] Compiles a function over two TwoTensors of different
        # second dimension with automatic dynamic shapes (dynamic setting None),
        # so the second call recompiles with a dynamic dimension; compares the
        # two forward graphs and two backward graphs to expected strings.
        def f(tt):
            # [FUNCTION DOC] HELPER: Clones tt and returns
            # its a part, a flattened view and its b part.
            y = tt.clone()
            return y.a, y.view(-1), y.b

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt1 = TwoTensor(a, b)

        a = torch.ones(3, 5, requires_grad=True)
        b = a.clone()
        tt2 = TwoTensor(a, b)

        fw, bw = self._compile_check(
            f, [(tt1,), (tt2,)], dynamic=None, call_backward=True
        )

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f32[3, 4]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a')
        primals_2: "f32[3, 4]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b')
    ):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[3, 4]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        view: "f32[12]" = torch.ops.aten.view.default(clone, [-1])
        view_1: "f32[12]" = torch.ops.aten.view.default(clone_1, [-1])
        return (
            clone,  # PlainAOTOutput(idx=0)
            view,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='a')
            view_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='b')
            clone_1,  # PlainAOTOutput(idx=2)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(fw[1].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f32[3, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a')
        primals_2: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=1)
        primals_3: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_4: "f32[3, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b')
        primals_5: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=1)
        primals_6: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_7: "Sym(s16)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=1)
        primals_8: "Sym(s16)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_9: "Sym(s16)",  # PlainAOTInput(idx=1)
    ):
        clone: "f32[3, s16]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[3, s16]" = torch.ops.aten.clone.default(primals_4);  primals_4 = None

        view: "f32[3*s16]" = torch.ops.aten.view.default(clone, [-1])
        sym_size_int_2: "Sym(3*s16)" = torch.ops.aten.sym_size.int(view, 0)
        view_1: "f32[3*s16]" = torch.ops.aten.view.default(clone_1, [-1])
        sym_size_int_3: "Sym(3*s16)" = torch.ops.aten.sym_size.int(view_1, 0)
        return (
            clone,  # PlainAOTOutput(idx=0)
            view,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='a')
            sym_size_int_2,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='a'), idx=0)
            view_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='b')
            sym_size_int_3,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='b'), idx=0)
            sym_size_int_2,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=1), idx=0)
            clone_1,  # PlainAOTOutput(idx=2)
            primals_9,  # SavedForBackwardsAOTOutput(idx=0)
            primals_2,  # SavedForBackwardsAOTOutput(idx=1)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        tangents_1: "f32[12]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=1)), attr='a')
        tangents_2: "f32[12]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=1)), attr='b')
    ):
        view_2: "f32[3, 4]" = torch.ops.aten.view.default(tangents_1, [3, 4]);  tangents_1 = None
        view_3: "f32[3, 4]" = torch.ops.aten.view.default(tangents_2, [3, 4]);  tangents_2 = None
        return (
            view_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a')
            view_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b')
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[1].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_9: "Sym(s16)",  # PlainAOTInput(idx=1)
        primals_2: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=1)
        tangents_1: "f32[3*s16]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=1)), attr='a')
        tangents_2: "f32[3*s16]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=1)), attr='b')
    ):
        view_2: "f32[3, s16]" = torch.ops.aten.view.default(tangents_1, [3, primals_9]);  tangents_1 = None
        view_3: "f32[3, s16]" = torch.ops.aten.view.default(tangents_2, [3, primals_9]);  tangents_2 = None
        return (
            view_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a')
            primals_9,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=1)
            primals_9,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            view_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b')
            primals_9,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=1)
            primals_9,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_9,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=1)
            primals_9,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            None,  # None
        )
""",
        )

    def test_tensor_subclass_TwoTensor_mark_dynamic_shapes(self):
        # [FUNCTION DOC] Compiles a function over a TwoTensor whose second
        # dimension is marked dynamic (dynamic setting None, with backward)
        # and compares the forward and backward graphs to expected strings.
        def f(tt):
            # [FUNCTION DOC] HELPER: Clones tt and returns
            # its a part, a flattened view and its b part.
            y = tt.clone()
            return y.a, y.view(-1), y.b

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)
        torch._dynamo.mark_dynamic(tt, 1)

        fw, bw = self._compile_check(
            f,
            [
                (tt,),
            ],
            dynamic=None,
            call_backward=True,
        )

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f32[3, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a')
        primals_2: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=1)
        primals_3: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=0)
        primals_4: "f32[3, s16]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b')
        primals_5: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=1)
        primals_6: "Sym(s16)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b'), idx=0)
        primals_7: "Sym(s16)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=1)
        primals_8: "Sym(s16)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_9: "Sym(s16)",  # PlainAOTInput(idx=1)
    ):
        clone: "f32[3, s16]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[3, s16]" = torch.ops.aten.clone.default(primals_4);  primals_4 = None

        view: "f32[3*s16]" = torch.ops.aten.view.default(clone, [-1])
        sym_size_int_2: "Sym(3*s16)" = torch.ops.aten.sym_size.int(view, 0)
        view_1: "f32[3*s16]" = torch.ops.aten.view.default(clone_1, [-1])
        sym_size_int_3: "Sym(3*s16)" = torch.ops.aten.sym_size.int(view_1, 0)
        return (
            clone,  # PlainAOTOutput(idx=0)
            view,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='a')
            sym_size_int_2,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='a'), idx=0)
            view_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='b')
            sym_size_int_3,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=1), attr='b'), idx=0)
            sym_size_int_2,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=1), idx=0)
            clone_1,  # PlainAOTOutput(idx=2)
            primals_9,  # SavedForBackwardsAOTOutput(idx=0)
            primals_2,  # SavedForBackwardsAOTOutput(idx=1)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_9: "Sym(s16)",  # PlainAOTInput(idx=1)
        primals_2: "Sym(s16)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a'), idx=1)
        tangents_1: "f32[3*s16]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=1)), attr='a')
        tangents_2: "f32[3*s16]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=1)), attr='b')
    ):
        view_2: "f32[3, s16]" = torch.ops.aten.view.default(tangents_1, [3, primals_9]);  tangents_1 = None
        view_3: "f32[3, s16]" = torch.ops.aten.view.default(tangents_2, [3, primals_9]);  tangents_2 = None
        return (
            view_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a')
            primals_9,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=1)
            primals_9,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a'), idx=0)
            view_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b')
            primals_9,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=1)
            primals_9,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b'), idx=0)
            primals_9,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=1)
            primals_9,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            None,  # None
        )
""",
        )

    def test_tensor_subclass_TwoTensor_different_shape(self):
        # [FUNCTION DOC] Compiles a function that views a cloned TwoTensor of
        # one-dimensional 24 elements as three dimensional (dynamic shapes, with
        # backward) and compares forward and backward graphs to expected strings.
        def f(tt):
            # [FUNCTION DOC] HELPER: Clones tt and returns a view of shape 3 by 2 by 4.
            y = tt.clone()
            return y.view(3, 2, 4)

        a = torch.ones((2 * 4 * 3), requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f32[24]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='a')
        primals_2: "f32[24]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='b')
    ):
        clone: "f32[24]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[24]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        view: "f32[3, 2, 4]" = torch.ops.aten.view.default(clone, [3, 2, 4]);  clone = None
        view_1: "f32[3, 2, 4]" = torch.ops.aten.view.default(clone_1, [3, 2, 4]);  clone_1 = None
        return (
            view,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='a')
            view_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='b')
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        tangents_1: "f32[3, 2, 4]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='a')
        tangents_2: "f32[3, 2, 4]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='b')
    ):
        view_2: "f32[24]" = torch.ops.aten.view.default(tangents_1, [24]);  tangents_1 = None
        view_3: "f32[24]" = torch.ops.aten.view.default(tangents_2, [24]);  tangents_2 = None
        return (
            view_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='a')
            view_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='b')
        )
""",
        )

    def test_tensor_subclass_TwoTensor_return_shape(self):
        # [FUNCTION DOC] Checks that a function compiled with aot_eager and dynamic shapes
        # flattens a TwoTensor via a view sized from its dimensions: the result equals the
        # eager view of the original, and its shape is a one-element tuple of 6.
        @torch.compile(backend="aot_eager", dynamic=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a clone of x viewed as one
            # dimension of size equal to the product of its two dimensions.
            return x.clone().view(x.shape[0] * x.shape[1])

        a = torch.ones(2, 3)
        b = a.clone()
        tt = TwoTensor(a, b)
        out = fn(tt)
        self.assertEqual(tt.view(2 * 3), out)
        self.assertEqual(out.shape, (6,))

    def test_tensor_subclass_TwoTensor_nested(self):
        # [FUNCTION DOC] Checks compilation (aot_eager, dynamic) with nested TwoTensors:
        # a function takes a TwoTensor, a plain tensor and a TwoTensor of TwoTensors,
        # and results and gradients of compiled calls are computed. KNOWN ISSUE: the
        # closeness comparisons of outputs and gradients are not wrapped in assertions,
        # so their results are discarded and the test does not actually verify equality.
        @torch.compile(backend="aot_eager", dynamic=True)
        def f(x, i, y):
            # [FUNCTION DOC] HELPER: Sums the sines of the three inputs and scales the
            # result by the product of the first dims of x and y and the second dim of i.
            out1 = x.sin() + i.sin() + y.sin()
            val1 = x.shape[0] * i.shape[1] * y.shape[0]
            return out1 * val1

        i = torch.randn(2, 2, requires_grad=True)
        x = TwoTensor(i, i.clone())
        y = TwoTensor(x.clone(), x.clone())

        out = f(x, i, y)

        x_test = x.detach().clone().requires_grad_(True)
        i_test = i.detach().clone().requires_grad_(True)
        y_test = y.detach().clone().requires_grad_(True)

        out_test = f(x_test, i_test, y_test)
        torch.allclose(out, out_test)

        out.sum().backward()
        out_test.sum().backward()
        torch.allclose(x.grad, x_test.grad)
        torch.allclose(i.grad, i_test.grad)
        torch.allclose(y.grad, y_test.grad)

    def test_subclass_TwoTensor_TwoTensor_TwoTensor(self):
        # [FUNCTION DOC] Checks compilation (aot_eager, dynamic) of a sine
        # over a three-level nested TwoTensor whose components are
        # themselves TwoTensors, and that the result equals the eager sine.
        @torch.compile(backend="aot_eager", dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return x.sin()

        data = torch.randn(2, 3)
        s = TwoTensor(data, data.clone())
        y = TwoTensor(s, s.clone())
        z = TwoTensor(s, y)
        out = f(z)
        self.assertEqual(out, z.sin())

    def test_subclass_TwoTensor_nested_diff_sizes(self):
        # [FUNCTION DOC] Checks dynamic compilation of nested subclasses whose inner
        # tensors have different sizes than the outer one: builds nested instances
        # of a TwoTensor derived class where inner elements have unrelated shapes,
        # compiles a function applying sine and summing both, and compares to eager.
        class TT(TwoTensor):
            # [CLASS DOC] HELPER: TwoTensor subclass whose constructor accepts an explicit
            # outer size and stride (defaulting to those of a) so the outer shape can differ
            # from inner shapes. The unflatten hook passes outer size and stride through.
            @staticmethod
            def __new__(cls, a, b, outer_size=None, outer_stride=None):
                # [FUNCTION DOC] HELPER: Static constructor. Defaults outer size
                # and stride from a, asserts the two inner tensors agree in device,
                # layout, requires_grad and dtype, and creates a wrapper subclass
                # with the outer size, outer stride and the properties of a.
                if outer_size is None:
                    outer_size = a.size()
                if outer_stride is None:
                    outer_stride = a.stride()

                assert (  # noqa: S101
                    a.device == b.device
                    and a.layout == b.layout
                    and a.requires_grad == b.requires_grad
                    and a.dtype == b.dtype
                )
                shape = outer_size
                kwargs = {}
                kwargs["strides"] = outer_stride
                kwargs["storage_offset"] = a.storage_offset()
                kwargs["device"] = a.device
                kwargs["layout"] = a.layout
                kwargs["requires_grad"] = a.requires_grad
                kwargs["dtype"] = a.dtype
                out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
                return out

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Static unflatten hook: asserts there is no
                # metadata, gets inner a and b, asserts outer size and stride are
                # provided when a is a plain tensor, and builds a new instance with them.
                assert meta is None  # noqa: S101
                a, b = inner_tensors["a"], inner_tensors["b"]
                if type(a) is torch.Tensor:
                    assert outer_size is not None  # noqa: S101
                    assert outer_stride is not None  # noqa: S101
                return TT(a, b, outer_size, outer_stride)

        @torch.compile(dynamic=True, backend="eager")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the sums of the sine of x and the sine of y as a pair.
            tmp1 = x.sin()
            tmp2 = y.sin()
            return tmp1.sum(), tmp2.sum()

        x = TT(
            TT(
                torch.randn(3, 4),
                torch.randn(5, 6, 7),
            ),
            TT(
                torch.randn(4),
                torch.randn(2, 3),
            ),
        )

        y = TT(
            torch.randn(2, 3, 4, 5),
            TT(
                torch.randn(3, 4),
                torch.randn(5),
            ),
        )

        out = f(x, y)
        self.assertEqual(out, (x.sin().sum(), y.sin().sum()))


# Tests for the issubclass() builtin
# Mirrors CPython's object_issubclass algorithm in Objects/abstract.c.


class _IssubclassBase:
    # [CLASS DOC] HELPER: Empty root class used as a base in issubclass tests.
    pass


class _IssubclassSub(_IssubclassBase):
    # [CLASS DOC] HELPER: Empty subclass of the issubclass test base.
    pass


class _IssubclassSubSub(_IssubclassSub):
    # [CLASS DOC] HELPER: Empty subclass of the issubclass
    # test sub class (two levels below the base).
    pass


class _IssubclassOther:
    # [CLASS DOC] HELPER: Empty class unrelated to the issubclass test hierarchy.
    pass


# Records every __subclasscheck__ call made on _IssubclassMetaWithCheck so
# tests can verify the metaclass method was actually traced (not silently
# constant-folded by generic_issubclass).
_issubclass_metacheck_calls: list = []


class _IssubclassMetaWithCheck(type):
    # [CLASS DOC] HELPER: Metaclass whose subclass check appends each
    # checked class to the module-level call record and always answers True.
    def __subclasscheck__(cls, subclass):
        # [FUNCTION DOC] HELPER: Records subclass in the
        # module-level metaclass call list and returns True.
        _issubclass_metacheck_calls.append(subclass)
        return True


class _IssubclassClassWithMeta(metaclass=_IssubclassMetaWithCheck):
    # [CLASS DOC] HELPER: Empty class using the recording metaclass.
    pass


class _IssubclassMetaNonBool(type):
    # [CLASS DOC] HELPER: Metaclass whose subclass check
    # always returns the integer 42 instead of a bool.
    def __subclasscheck__(cls, subclass):
        # [FUNCTION DOC] HELPER: Returns 42 (non-bool) to test coercion to truth value.
        # Non-bool return; CPython coerces via PyObject_IsTrue.
        return 42


class _IssubclassClassWithNonBoolMeta(metaclass=_IssubclassMetaNonBool):
    # [CLASS DOC] HELPER: Empty class using the metaclass that returns a non-bool.
    pass


class _IssubclassMyABC(metaclass=abc.ABCMeta):  # noqa: B024
    # [CLASS DOC] HELPER: Empty abstract base class (ABCMeta) with a registered virtual subclass.
    pass


class _IssubclassRegisteredViaABC:
    # [CLASS DOC] HELPER: Empty class registered as a virtual subclass
    # of the abstract test class right after the definitions.
    pass


_IssubclassMyABC.register(_IssubclassRegisteredViaABC)


@torch._dynamo.config.patch(enable_trace_unittest=True)
class TestIssubclass(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for Dynamo's handling of the builtin issubclass, modeled on
    # the CPython algorithm: builtin types, user hierarchies, tuples, union types,
    # metaclass subclass checks, ABC registration, error cases and graph breaks.
    # The whole class is decorated to enable tracing of unittest assertions, and
    # most tests use the Dynamo test decorator so their bodies are traced.
    # --- Built-in types ---

    @make_dynamo_test
    def test_int_is_int(self):
        # [FUNCTION DOC] Asserts int is a subclass of int when traced.
        self.assertTrue(issubclass(int, int))

    @make_dynamo_test
    def test_bool_is_int(self):
        # [FUNCTION DOC] Asserts bool is a subclass of int.
        self.assertTrue(issubclass(bool, int))

    @make_dynamo_test
    def test_int_not_str(self):
        # [FUNCTION DOC] Asserts int is not a subclass of str.
        self.assertFalse(issubclass(int, str))

    @make_dynamo_test
    def test_object_supertype(self):
        # [FUNCTION DOC] Asserts int and str are both subclasses of object.
        self.assertTrue(issubclass(int, object))
        self.assertTrue(issubclass(str, object))

    # --- User-defined classes ---

    @make_dynamo_test
    def test_sub_of_base(self):
        # [FUNCTION DOC] Asserts a direct user subclass is a subclass of its base.
        self.assertTrue(issubclass(_IssubclassSub, _IssubclassBase))

    @make_dynamo_test
    def test_subsub_of_base(self):
        # [FUNCTION DOC] Asserts a two-level descendant is a subclass of the root base.
        self.assertTrue(issubclass(_IssubclassSubSub, _IssubclassBase))

    @make_dynamo_test
    def test_subsub_of_sub(self):
        # [FUNCTION DOC] Asserts the two-level descendant is a subclass of the middle class.
        self.assertTrue(issubclass(_IssubclassSubSub, _IssubclassSub))

    @make_dynamo_test
    def test_base_not_sub(self):
        # [FUNCTION DOC] Asserts a base class is not a subclass of its subclass.
        self.assertFalse(issubclass(_IssubclassBase, _IssubclassSub))

    @make_dynamo_test
    def test_unrelated_classes(self):
        # [FUNCTION DOC] Asserts unrelated classes are not
        # subclasses of each other in either direction.
        self.assertFalse(issubclass(_IssubclassSub, _IssubclassOther))
        self.assertFalse(issubclass(_IssubclassOther, _IssubclassSub))

    # --- Tuple of classes ---

    @make_dynamo_test
    def test_tuple_hit_first(self):
        # [FUNCTION DOC] Asserts a match on the first entry of a class tuple succeeds.
        self.assertTrue(issubclass(int, (int, str)))

    @make_dynamo_test
    def test_tuple_hit_second(self):
        # [FUNCTION DOC] Asserts a match on the second entry of a class tuple succeeds.
        self.assertTrue(issubclass(int, (str, int)))

    @make_dynamo_test
    def test_tuple_miss(self):
        # [FUNCTION DOC] Asserts a tuple with no matching class gives False.
        self.assertFalse(issubclass(int, (str, float)))

    @make_dynamo_test
    def test_tuple_empty(self):
        # [FUNCTION DOC] Asserts an empty tuple gives False.
        self.assertFalse(issubclass(int, ()))

    @make_dynamo_test
    def test_tuple_with_user_classes(self):
        # [FUNCTION DOC] Asserts tuples of user classes
        # give the right answer both for a hit and a miss.
        self.assertTrue(issubclass(_IssubclassSub, (_IssubclassOther, _IssubclassBase)))
        self.assertFalse(
            issubclass(_IssubclassOther, (_IssubclassBase, _IssubclassSub))
        )

    # --- PEP 604 Union (X | Y) ---

    @make_dynamo_test
    def test_union_hit(self):
        # [FUNCTION DOC] Asserts a union type containing the class matches.
        self.assertTrue(issubclass(int, int | str))

    @make_dynamo_test
    def test_union_miss(self):
        # [FUNCTION DOC] Asserts a union type without the class does not match.
        self.assertFalse(issubclass(int, str | bytes))

    @make_dynamo_test
    def test_union_with_user_classes(self):
        # [FUNCTION DOC] Asserts a union of user classes matches a subclass of one member.
        self.assertTrue(issubclass(_IssubclassSub, _IssubclassOther | _IssubclassBase))

    # --- Metaclass with __subclasscheck__ ---

    def test_custom_metaclass_subclasscheck(self):
        # [FUNCTION DOC] Checks that a metaclass subclass check is actually traced:
        # clears the call record, compiles a function calling issubclass for int
        # and str against a class with the recording metaclass under fullgraph,
        # expects results True twice and that the metaclass recorded int then str.
        _issubclass_metacheck_calls.clear()

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Returns a pair of issubclass results for
            # int and str against the class with the recording metaclass.
            return (
                issubclass(int, _IssubclassClassWithMeta),
                issubclass(str, _IssubclassClassWithMeta),
            )

        self.assertEqual(fn(), (True, True))
        self.assertEqual(_issubclass_metacheck_calls, [int, str])

    @make_dynamo_test
    def test_custom_metaclass_subclasscheck_non_bool(self):
        # [FUNCTION DOC] Asserts that a non-bool value returned by a metaclass
        # subclass check is coerced to a real True boolean (identity with True).
        # __subclasscheck__ returning a non-bool should be coerced via
        # PyObject_IsTrue (abstract.c L2812) — exercises the generic_is_true
        # branch at the bottom of generic_issubclass.
        result = issubclass(int, _IssubclassClassWithNonBoolMeta)
        self.assertIs(result, True)

    # --- abc.ABCMeta ---
    # ABCMeta.__subclasscheck__ lives in <frozen abc> which Dynamo skips; we
    # constant-fold via Python's issubclass when both args are concrete.

    @make_dynamo_test
    def test_abc_register(self):
        # [FUNCTION DOC] Asserts a class registered as virtual
        # subclass of an abstract base is reported as a subclass.
        self.assertTrue(issubclass(_IssubclassRegisteredViaABC, _IssubclassMyABC))

    @make_dynamo_test
    def test_abc_not_registered(self):
        # [FUNCTION DOC] Asserts an unregistered class is not a subclass of the abstract base.
        self.assertFalse(issubclass(_IssubclassOther, _IssubclassMyABC))

    # --- TypeError cases ---

    @make_dynamo_test
    def test_non_class_arg2_doesnt_raise(self):
        # [FUNCTION DOC] Asserts that a non-class entry after a matching class
        # in the tuple does not raise, because matching stops at the first hit.
        self.assertTrue(issubclass(int, (int, 1)))

    @make_dynamo_test
    def test_non_class_arg1_raises(self):
        # [FUNCTION DOC] Asserts a non-class first argument raises
        # TypeError with a message saying arg 1 must be a class.
        with self.assertRaisesRegex(TypeError, r"issubclass\(\) arg 1 must be a class"):
            issubclass(1, int)

    @make_dynamo_test
    def test_non_class_arg2_raises(self):
        # [FUNCTION DOC] Asserts a non-class second argument raises TypeError whose
        # message says arg 2 must be a class, a tuple of classes, or a union.
        with self.assertRaisesRegex(
            TypeError,
            r"issubclass\(\) arg 2 must be a class, a tuple of classes, or a union",
        ):
            issubclass(int, 1)

    @make_dynamo_test
    def test_non_class_in_tuple_raises(self):
        # [FUNCTION DOC] Asserts a tuple whose first entry is not a class raises TypeError.
        # CPython: TypeError when a tuple element isn't a class.
        with self.assertRaises(TypeError):
            issubclass(int, (1, str))

    # --- Identity ---

    @make_dynamo_test
    def test_class_is_subclass_of_itself(self):
        # [FUNCTION DOC] Asserts a class is a subclass of itself.
        self.assertTrue(issubclass(_IssubclassSub, _IssubclassSub))

    # --- Graph break path ---

    def test_invalid_args_graph_break(self):
        # [FUNCTION DOC] Checks that issubclass on a tensor (non-class) argument under fullgraph
        # raises the Dynamo Unsupported error with a message matching issubclass unsupported.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Returns issubclass of t against int.
            return issubclass(t, int)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "issubclass.*unsupported",
        ):
            fn(torch.randn(3))

    def test_wrapper_subclass_vmap_compile(self):
        # [FUNCTION DOC] Checks that vmap over a function adding a tensor and a
        # wrapper subclass (whose torch function is disabled) works compiled under
        # fullgraph with the eager backend and equals the eager vmap result.
        class TransparentTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Transparent wrapper subclass storing a tensor in
            # _data, with torch function disabled and a dispatch that unwraps arguments
            # (including lists and tuples), runs the op and rewraps tensor results.
            __torch_function__ = torch._C._disabled_torch_function_impl

            @staticmethod
            def __new__(cls, data):
                # [FUNCTION DOC] HELPER: Static constructor. Returns data as is if
                # already of this class; converts non-tensors via as_tensor; creates
                # a wrapper subclass with the geometry of data (strides and storage
                # offset only for strided layout) and stores the data in _data.
                if isinstance(data, cls):
                    return data
                if not isinstance(data, torch.Tensor):
                    data = torch.as_tensor(data)
                kwargs = {}
                if data.layout == torch.strided:
                    kwargs["strides"] = data.stride()
                    kwargs["storage_offset"] = data.storage_offset()
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    data.shape,
                    dtype=data.dtype,
                    layout=data.layout,
                    device=data.device,
                    requires_grad=data.requires_grad,
                    **kwargs,
                )
                r._data = data
                return r

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Flatten hook: inner attribute _data and empty metadata.
                return ["_data"], {}

            @classmethod
            def __tensor_unflatten__(
                cls, inner_tensors, metadata, outer_size, outer_stride
            ):
                # [FUNCTION DOC] HELPER: Classmethod unflatten hook:
                # builds an instance from the _data inner tensor.
                return cls(inner_tensors["_data"])

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook: defaults kwargs,
                # unwraps wrapped args and kwargs, calls func, and wraps tensor
                # results (or tensors in tuple or list results) in this class.
                if kwargs is None:
                    kwargs = {}

                def unwrap(x):
                    # [FUNCTION DOC] HELPER: Recursively replaces instances of this subclass by
                    # their _data, going into lists and tuples while keeping the container type.
                    if isinstance(x, TransparentTensor):
                        return x._data
                    if isinstance(x, (list, tuple)):
                        return type(x)(unwrap(a) for a in x)
                    return x

                result = func(*unwrap(args), **unwrap(kwargs))
                if isinstance(result, torch.Tensor):
                    return cls(result)
                if isinstance(result, (tuple, list)):
                    return type(result)(
                        cls(r) if isinstance(r, torch.Tensor) else r for r in result
                    )
                return result

        def fn(x, tag):
            # [FUNCTION DOC] HELPER: Returns x plus tag.
            return x + tag

        vmapped = torch.vmap(fn, in_dims=(0, None))

        x = torch.randn(8, 4)
        tag = TransparentTensor(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        result_eager = vmapped(x, tag)

        compiled = torch.compile(vmapped, backend="eager", fullgraph=True)
        result_compiled = compiled(x, tag)
        self.assertEqual(result_eager, result_compiled)


class TestNestedTensor(
    _SubclassCompileCheckMixin,
    torch._dynamo.test_case.TestCase,
    NestedTensorTestCase,
):
    # [CLASS DOC] Tests for Dynamo with nested (jagged) tensors, using the compile-check
    # mixin, the Dynamo TestCase and the nested tensor test case helpers. Covers recompilation
    # behavior, views, guards, dynamic shapes and graph structure for jagged tensors.
    def _get_jagged_tensor(self, nested_size, offsets, requires_grad=True):
        # [FUNCTION DOC] HELPER: Method wrapper that builds a jagged
        # tensor through the module-level helper with the same arguments.
        return get_jagged_tensor(nested_size, offsets, requires_grad)

    def _njt_compile_check(self, fn, inps, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the mixin compile check with a backward function
        # that converts nested outputs to their jagged values buffer before summing
        # and calling backward, and an equality function ignoring nested ints; extra
        # keyword arguments are forwarded. Returns forward and backward graph lists.
        def _njt_call_backward_fn(t):
            # [FUNCTION DOC] HELPER: Converts nested tensors to their values buffer, then
            # sums and runs backward retaining the graph; returns the backward result.
            if t.is_nested:
                from torch.nested._internal.nested_tensor import buffer_from_jagged

                t = buffer_from_jagged(t)
            return t.sum().backward(retain_graph=True)

        return self._compile_check(
            fn,
            inps,
            call_backward_fn=_njt_call_backward_fn,
            assert_equal_fn=self.assertEqualIgnoringNestedInts,
            **kwargs,
        )

    def _get_nc_jagged_tensor(self, inner_dim, starts, lengths, requires_grad=True):
        # [FUNCTION DOC] HELPER: Builds a non-contiguous jagged tensor from float64 random
        # values of shape (number of rows, max of starts plus lengths, inner_dim) and the given
        # starts and lengths via jagged_from_tensor_and_lengths; returns the jagged tensor.
        # Makes a jagged tensor with N constituent tensors with size
        # as specified ((S0, S1, S2), D)
        max_dim = (starts + lengths).max()
        values_tensor = torch.randn(
            starts.shape[0],
            max_dim.item(),
            inner_dim,
            requires_grad=requires_grad,
            dtype=torch.float64,
        )
        return jagged_from_tensor_and_lengths(values_tensor, starts, lengths)

    def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
        # [FUNCTION DOC] HELPER: Method wrapper delegating
        # to the module-level recompile assertion helper.
        _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles)

    def test_unary_does_not_recompile(self):
        # [FUNCTION DOC] Checks that a unary op on jagged tensors of
        # different batch sizes and feature dims does not recompile.
        nt1, _ = self._get_jagged_tensor(((2, 3, 4), 3), None)
        nt2, _ = self._get_jagged_tensor(((3, 4, 5, 6), 4), None)
        self._check_recompiles(lambda nt1: nt1.sin(), (nt1,), (nt2,), False)

    def test_binary_does_not_recompile(self):
        # [FUNCTION DOC] Checks that a binary op on jagged tensors does not recompile
        # when moving from one pair of tensors sharing offsets to a different pair
        # (different batch and feature sizes) that also shares its own offsets;
        # feature dim 5 is used to avoid duck sizing equating batch and last dims.
        def binary(nt1, nt2):
            # [FUNCTION DOC] HELPER: Returns the sum if the two jagged
            # tensors have equal shape, otherwise the sine of the first.
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # NB: If we have shape e.g. (3, j0, 3), duck sizing will give us (s0, s1, s0).
        # This causes a recompile later on when it realizes the batch and last dim
        # should not always be equal. To avoid that, we use (3, j0, 5) here.
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 5), offsets)
        nt3, offsets = self._get_jagged_tensor(((3, 4, 5), 4), None)
        nt4, _ = self._get_jagged_tensor(((3, 4, 5), 4), offsets)
        self._check_recompiles(binary, (nt1, nt2), (nt3, nt4), False)

    def test_binary_recompiles(self):
        # [FUNCTION DOC] Checks that a binary op on jagged tensors does
        # recompile when the second operand has different offsets (distinct
        # nested ints) than in the first call, where both shared offsets.
        def binary(nt1, nt2):
            # [FUNCTION DOC] HELPER: Returns the sum if the two jagged
            # tensors have equal shape, otherwise the sine of the first.
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # Binary recompiles because singleton ints no longer match
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 5), offsets)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)
        self._check_recompiles(binary, (nt1, nt2), (nt1, nt3), True)

    def _validate_compile(self, fn, arg_fn):
        # [FUNCTION DOC] HELPER: Validation harness. Within a branched nested-tensor global
        # state, compiles fn with nopython aot_eager recording exported guard code and
        # failed guards, checks that compilation does not change the eager nested-int
        # registry or the tensor id counter, runs the compiled function on arguments from
        # arg_fn and collects gradients with respect to inputs requiring grad. Then, in a
        # separate branched state, runs eager and collects reference outputs and gradients.
        # Compares outputs (elementwise for lists/tuples) and gradients ignoring nested
        # ints. Returns the list of exported guard strings and the list of failed guards.
        def _gen_grad_outputs(out_val):
            # [FUNCTION DOC] HELPER: Returns a tuple of ones-like tensors for each item of a
            # list or tuple output, otherwise a one-element tuple with a ones-like tensor.
            if isinstance(out_val, (list, tuple)):
                return tuple(torch.ones_like(c) for c in out_val)
            else:
                return (torch.ones_like(out_val),)

        with self.branch_nested_state():
            from torch.nested._internal.nested_tensor import _tensor_symint_registry

            # Validate that compilation does not modify eager state
            registry_before = list(_tensor_symint_registry.items())
            count_before = torch.nested._internal.nested_tensor._tensor_id_counter

            guards_exported = []
            guards_failed = []

            def append_guard_export(guards):
                # [FUNCTION DOC] HELPER: Guard export callback: appends the first
                # code line of each guard that has a code list to the exported list.
                for g in guards:
                    if g.code_list is not None:
                        guards_exported.append(g.code_list[0])

            def append_guard_fail(guards):
                # [FUNCTION DOC] HELPER: Guard failure callback:
                # extends the failed guard list with the given guards.
                guards_failed.extend(guards)

            compiled = torch._dynamo.optimize(
                nopython=True,
                backend="aot_eager",
                guard_export_fn=append_guard_export,
                guard_fail_fn=append_guard_fail,
            )(fn)
            registry_after = list(_tensor_symint_registry.items())
            count_after = torch.nested._internal.nested_tensor._tensor_id_counter
            self.assertEqual(registry_before, registry_after)
            self.assertEqual(count_before, count_after)

            args = arg_fn()
            compile_out = compiled(*args)
            compile_grads = []
            g_args = [arg for arg in args if arg.requires_grad]
            if len(g_args) > 0:
                compile_grad_outputs = _gen_grad_outputs(compile_out)
                compile_grads = torch.autograd.grad(
                    compile_out, inputs=g_args, grad_outputs=compile_grad_outputs
                )

        with self.branch_nested_state():
            args = arg_fn()
            ref_out = fn(*args)
            ref_grads = []
            g_args = [arg for arg in args if arg.requires_grad]
            if len(g_args) > 0:
                ref_grad_outputs = _gen_grad_outputs(ref_out)
                ref_grads = torch.autograd.grad(
                    ref_out, inputs=g_args, grad_outputs=ref_grad_outputs
                )

        # Validate correctness forward
        if isinstance(compile_out, (list, tuple)):
            # TODO: Fix assertEqual() to support NJTs so this isn't necessary
            self.assertEqual(len(compile_out), len(ref_out))
            for c, r in zip(compile_out, ref_out):
                self.assertEqualIgnoringNestedInts(c, r)
        else:
            self.assertEqualIgnoringNestedInts(compile_out, ref_out)

        # Validate correctness backward
        for compile_grad, ref_grad in zip(compile_grads, ref_grads):
            self.assertEqualIgnoringNestedInts(compile_grad, ref_grad)

        return guards_exported, guards_failed

    def test_in_graph_is_nested_call(self):
        # [FUNCTION DOC] Checks that a nested-ness query inside a compiled function is resolved
        # at trace time: compiles a function branching on is_nested of a jagged tensor with
        # aot_eager and fullgraph, runs forward and backward, asserts one frame and one graph,
        # and compares the printed graph text, which must contain no is_nested call.
        def f(nt):
            # [FUNCTION DOC] HELPER: Adds 2 to nt if it is nested, otherwise adds 1.
            if nt.is_nested:
                return nt + 2
            else:
                return nt + 1

        cnt = CompileCounterWithBackend("aot_eager")
        compiled_f = torch.compile(f, backend=cnt, fullgraph=True)
        nt, offsets = self._get_jagged_tensor(((2, 3, 4), 5), None)
        output = compiled_f(nt)
        output.backward(torch.ones_like(output))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(cnt.graphs), 1)
        graph = cnt.graphs[0]
        norm_graph = normalize_gm(graph.print_readable(print_output=False))

        # expect -no- is_nested calls within the graph
        self.assertExpectedInline(
            norm_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_nt_: "NestedTensor(f64[3, s71, 5])", s71: "Sym(s71)"):
        l_nt_ = L_nt_

        add: "NestedTensor(f64[3, s71, 5])" = l_nt_ + 2;  l_nt_ = None
        return (add,)
""",
        )

    # Note: [What kind of guards are involved in nested tensor compilation]
    #
    # Until we implement UnionFind, dynamic shapes guards are not involved.
    # we rely only on dynamo's tensor aliasing guards.
    #
    # This is possible because dynamo able to generate tensor aliasing guards
    # not only for the outer tensor, but also for the inner tensor.
    #
    # The case where dynamic shapes guards would eventually come into play is
    # when my inputs are (1) two non-aliased tensors, but (2) declared as
    # equal using a "trust me assert equal" API.

    # Note: [Compiling nested tensor global state]
    #
    # Today there are two pieces of global eager state that NJTs deals with:
    # - tensor_id_counter: a global counter that assigns unique ids to tensors
    # - tensor_symint_registry: maps tensor to nested int
    #   - this is used in eager only (we should get rid of this because it is
    #     not necessary to cache nested int in eager)
    #   - during tracing, we DO need to cache nested int, but we do so on
    #     the FakeTensor.
    #
    # Ideally we would like to satisfy the following:
    # - (1) The eager state is not mutated during tracing
    # - (2) Running the compiled function should mutate the eager state in the
    #       same way that running the eager function would
    #       (a) The global counter should be incremented
    #       (b) The registry is updated in the same way
    #
    # Today we can satisfy (1) and (2a) but cannot satisfy (2b)
    #
    # Today, (1) is satisfied because we maintain a separate counter during
    # tracing, and cache nested int on FakeTensor instead of relying on
    # tensor_symint_registry.
    #
    # (2) is cannot be completely satisfied because we trace away the
    # side-effectful operations (which we can fix this by wrapping the
    # side-effectful operations in a custom op, and threading through effect
    # tokens.) The current plan is to do that in the UnionFind impl.
    #
    # Interestingly, despite this, the state is mutated in a way that is somewhat
    # close to what we want, e.g. if I construct a nested tensor using an
    # offsets in the compiled region and return it, AOTAutograd runtime wrapper
    # must rewrap the inner->inner graph outputs back into subclass. This
    # triggers the eager logic to run, updating the counter and registry.
    #
    # Notably however, compile differs in two ways from eager:
    # (1) The order in which the offsets are assigned ids is different
    #     the registry would be set in the order the offsets are returned
    #     which is not necessarily the same order as they were constructed.
    # (2) If a NestedTensor is not returned, then the AOTAutograd wrapping
    #     logic will not be triggered.
    #
    # I claim that correctness is not affected by these differences today.
    # e.g. there is never the case where two distinct offsets silently share
    # the same id.
    #
    # (1) is clearly not a problem, and (2) should only be a problem if
    # the nested int is returned on its own, without the corresponding NJT
    # being returned. This is not a problem in the current implementation
    # because returning only a shape is not supported!

    # Note: [Creating symbolic nested int]
    #
    # We must create a symbolic nested int when we construct a nested tensor
    # from a tensor. There are two main cases:
    #
    # 1. The offsets has NOT been used to construct a NJT
    #    - Create a new plain nested int with current val of fake nt id counter
    #    - Increment the fake nt id counter
    #    - Create a new symint with plain nested int as hint
    # 2. The offsets HAS been used to construct a NJT
    #    - Create a new symint with plain nested int as hint
    #
    # More details on case 2:
    # - During fakification of the offsets, we check the eager registry, and
    #   if the tensor HAS been used to construct a NJT,
    #   we create a symint, with the existing nested int as hint, and cache
    #   it on to the FakeTensor.
    #
    # [ Always use ephemeral source ]
    #
    # We create the new symint ALWAYS with ephemeral source whether that is
    # in case (1) or (2) even though we could've had a proper source for case (2).
    # Using a proper source would enable a few more (edge) cases, but since
    # we plan to handle things more holistically in the future anyway, we don't
    # bother doing so today.
    #
    # Using an ephemeral source has some consequences. But we are happy if
    # - We do not silently miss recompiles, e.g. we guard when necessary.
    #   We know that this is true, because dynamo guards alone are already
    #   sufficient.
    # - We are not producing errors for the cases we care about
    #
    # The main case we care about is when we guard that two shapes are equal.
    # In this case, the replacements logic would simplify away the ephemeral
    # symbol, and there is no error produced.
    # The unsupported case is when we guard that two shapes are not equal, in
    # which, we will try and fail to generate a guard.

    #
    # Case 1: in-graph construction where the offsets are passed as inputs
    #
    def test_in_graph_construction_from_input(self):
        # [FUNCTION DOC] Case 1: in-graph construction of a jagged tensor from values and
        # offsets passed as inputs. Validates compile against eager, then with different
        # offsets values under error on recompile to check the offsets are not specialized.
        # The offsets is passed as an input
        def fn(values, offsets):
            # [FUNCTION DOC] HELPER: Builds a jagged tensor from
            # doubled values and the offsets, and doubles it.
            return torch.nested.nested_tensor_from_jagged(values * 2, offsets) * 2

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        self._validate_compile(fn, arg_fn=lambda: (values, offsets))

        # Do not specialize on the offsets
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            different_offsets = torch.tensor([0, 1, 5, 10], dtype=torch.int64)
            self._validate_compile(fn, arg_fn=lambda: (values, different_offsets))

    def test_in_graph_construction_from_input_2(self):
        # [FUNCTION DOC] Builds two jagged tensors from inputs. With distinct offsets,
        # no guard failures and no identity guard between the offsets; with the same
        # offsets object, a duplicate tensors guard failure is recorded and an
        # identity guard appears; a clone used for both with error on recompile must
        # not recompile; finally a binary op over two jagged tensors is validated.
        # Construct two NJTs, both are passed as inputs
        def fn(values, offsets1, offsets2):
            # [FUNCTION DOC] HELPER: Builds two jagged tensors from scaled
            # values and two offsets and returns them in swapped order.
            nt1 = torch.nested.nested_tensor_from_jagged(values * 2, offsets1)
            nt2 = torch.nested.nested_tensor_from_jagged(values * 3, offsets2)
            return nt2, nt1

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        offsets2 = torch.tensor([0, 1, 4, 10], dtype=torch.int64)
        # 1. Offsets are different
        guards_exported, guards_failed = self._validate_compile(
            fn, arg_fn=lambda: (values, offsets, offsets2)
        )
        self.assertEqual(len(guards_failed), 0)
        self.assertNotIn("L['offsets1'] is L['offsets2']", guards_exported)

        # TODO
        # 2. Offsets are the same
        new_guards_exported, _ = self._validate_compile(
            fn, arg_fn=lambda: (values, offsets, offsets)
        )
        self.assertTrue(any("Duplicate tensors found" in g for g in guards_failed))
        self.assertIn("L['offsets1'] is L['offsets2']", new_guards_exported)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            offsets3 = offsets.clone()
            self._validate_compile(fn, arg_fn=lambda: (values, offsets3, offsets3))

        # Do a binary op
        def fn(values, offsets, offsets2):
            # [FUNCTION DOC] HELPER: Redefinition used for a binary op
            # test: builds two jagged tensors from values scaled by 2
            # and 3 with the given offsets and returns their product.
            nt1 = torch.nested.nested_tensor_from_jagged(values * 2, offsets)
            nt2 = torch.nested.nested_tensor_from_jagged(values * 3, offsets2)
            return nt1 * nt2

        self._validate_compile(fn, arg_fn=lambda: (values, offsets, offsets))

    def test_in_graph_construction_from_input_4(self):
        # [FUNCTION DOC] Checks construction where offsets come from a
        # jagged tensor input: validates compile, then with a cloned offsets
        # tensor under error on recompile to ensure no specialization.
        # The offsets is taken from an NJT input
        def fn(nt, other_values):
            # [FUNCTION DOC] HELPER: Builds a second jagged tensor from
            # other_values and the offsets of nt, and returns the sum.
            nt2 = torch.nested.nested_tensor_from_jagged(other_values, nt.offsets())
            return nt + nt2

        values = torch.randn(9, 5, requires_grad=True)
        other_values = torch.randn(9, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 9], dtype=torch.int64)

        def arg_fn(values=values, other_values=other_values, offsets=offsets):
            # [FUNCTION DOC] HELPER: Argument factory with defaults bound at definition:
            # builds a jagged tensor from values and offsets and returns it with other_values.
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            return nt, other_values

        self._validate_compile(fn, arg_fn=arg_fn)

        # Do not specialize on the offsets
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            different_offsets = offsets.clone()

            def arg_fn(
                values=values, other_values=other_values, offsets=different_offsets
            ):
                # [FUNCTION DOC] HELPER: Argument factory building a jagged
                # tensor from the cloned offsets (the offsets default parameter
                # is unused in the body) and returning it with other_values.
                nt = torch.nested.nested_tensor_from_jagged(values, different_offsets)
                return nt, other_values

            self._validate_compile(fn, arg_fn=arg_fn)

    def test_in_graph_construction_from_input_5(self):
        # [FUNCTION DOC] Checks construction from lengths instead
        # of offsets and applying sine, validated against eager.
        # Construct from lengths instead of offsets
        def fn(values, lengths):
            # [FUNCTION DOC] HELPER: Builds a jagged tensor
            # from values with lengths and returns its sine.
            nt = torch.nested.nested_tensor_from_jagged(values, lengths=lengths)
            return nt.sin()

        values = torch.randn(9, 5, requires_grad=True)
        lengths = torch.tensor([2, 4, 3])
        self._validate_compile(fn, arg_fn=lambda: (values, lengths))

    def test_in_graph_construction_from_input_6(self):
        # [FUNCTION DOC] Checks construction with a symbolic max sequence length under dynamic
        # compile (fullgraph, eager backend): rebuilding from the offsets and the stored max
        # seqlen of the first tensor gives a result equal to eager ignoring nested ints.
        # Construct with symbolic int.
        def fn(values, offsets, max_seqlen):
            # [FUNCTION DOC] HELPER: Builds a jagged tensor with offsets and max_seqlen,
            # then builds another from its offsets and stored max seqlen, and returns it.
            t = torch.nested.nested_tensor_from_jagged(
                values, offsets, max_seqlen=max_seqlen
            )
            return torch.nested.nested_tensor_from_jagged(
                values, t.offsets(), max_seqlen=t._maybe_max_seqlen
            )

        opt_fn = torch.compile(fn, fullgraph=True, dynamic=True, backend="eager")
        values = torch.randn(10, 5)
        offsets = torch.tensor([0, 2, 4, 7, 10])
        max_seqlen = 5

        ref = fn(values, offsets, max_seqlen)
        res = opt_fn(values, offsets, max_seqlen)
        self.assertEqualIgnoringNestedInts(ref, res)

    #
    # Case 2: in-graph construction where offsets are graph intermediates
    #
    def test_in_graph_construction_from_intermediate(self):
        # [FUNCTION DOC] Case 2: offsets are a graph intermediate computed
        # from lengths. Validates compile against eager and checks that using
        # a cloned lengths tensor under error on recompile does not recompile.
        # offsets is an intermediate computed from lengths
        def fn(values, lengths):
            # [FUNCTION DOC] HELPER: Computes offsets as zero followed by the
            # cumulative sum of lengths, builds two jagged tensors from values
            # with these offsets and returns the sine of their product.
            offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            nt2 = torch.nested.nested_tensor_from_jagged(values, offsets)
            return (nt * nt2).sin()

        values = torch.randn(9, 5, requires_grad=True)
        lengths = torch.tensor([2, 4, 3])
        self._validate_compile(fn, arg_fn=lambda: (values, lengths))

        # Do not specialize on the lengths
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            different_lengths = lengths.clone()
            self._validate_compile(fn, arg_fn=lambda: (values, different_lengths))

    def test_in_graph_construction_from_intermediate_2(self):
        # [FUNCTION DOC] Case 2 variant: offsets is a cloned input used to
        # build a jagged tensor from doubled values; validated against eager.
        def fn(values, offsets):
            # [FUNCTION DOC] HELPER: Builds a jagged tensor
            # from doubled values and a clone of offsets.
            return torch.nested.nested_tensor_from_jagged(values * 2, offsets.clone())

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        self._validate_compile(fn, arg_fn=lambda: (values, offsets))

    def test_in_graph_construction_from_intermediate_3(self):
        # [FUNCTION DOC] Case 2 variant with two jagged tensors built
        # from separate clones of the same offsets (clones may be merged
        # by common subexpression elimination); validated against eager.
        # Note that due to CSE, clone is not necessarily called twice!
        def fn(values, offsets):
            # [FUNCTION DOC] HELPER: Builds two jagged tensors from scaled values
            # with separate offset clones and returns them in swapped order.
            nt1 = torch.nested.nested_tensor_from_jagged(values * 2, offsets.clone())
            nt2 = torch.nested.nested_tensor_from_jagged(values * 3, offsets.clone())
            return nt2, nt1

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        self._validate_compile(fn, arg_fn=lambda: (values, offsets))

    def test_in_graph_construction_from_intermediate_4(self):
        # [FUNCTION DOC] Case 2 variant where both jagged tensors share one
        # intermediate offsets tensor created in the function; validated against eager.
        # Shared intermediate (should be same as case #1)
        def fn(values):
            # [FUNCTION DOC] HELPER: Creates fixed offsets, builds two jagged tensors
            # (from values and from ones) with them and returns their product.
            offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            values2 = torch.ones_like(values)
            nt2 = torch.nested.nested_tensor_from_jagged(values2, offsets)
            return nt * nt2

        values = torch.randn(10, 5).requires_grad_(True)
        self._validate_compile(fn, arg_fn=lambda: (values,))

    # AssertionError: s2 (could be from ['<ephemeral: intermediate_offsets_or_lengths>',
    @unittest.expectedFailure
    def test_in_graph_construction_from_intermediate_5(self):
        # [FUNCTION DOC] Expected failure: two jagged tensors built from non-
        # shared intermediate offsets, one a clone, with a branch on their shape
        # comparison. KNOWN ISSUE: decorated expectedFailure because an ephemeral
        # symbol source assertion error occurs (see the comment above it).
        # non-shared intermediate
        def fn(values):
            # [FUNCTION DOC] HELPER: Builds a jagged tensor from values and
            # offsets, another from ones and a clone of the offsets, and returns
            # nt times 2 when their second dims differ, otherwise nt times 3.
            offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            values2 = torch.ones_like(values)
            nt2 = torch.nested.nested_tensor_from_jagged(values2, offsets.clone())
            if nt2.shape[1] != nt.shape[1]:
                return nt * 2
            else:
                return nt * 3

        values = torch.randn(10, 5).requires_grad_(True)
        self._validate_compile(fn, arg_fn=lambda: (values,))

    #
    # Case 3: in-graph construction where offsets are both direct graph inputs
    #         and passed in as part of an NJT's offsets.
    #
    def test_in_graph_construction_mixed(self):
        # [FUNCTION DOC] Case 3: offsets used both as a direct graph input
        # and inside a jagged tensor input; validated against eager.
        def fn(nt, values, offsets):
            # [FUNCTION DOC] HELPER: Builds a second jagged tensor from the
            # values and offsets inputs and returns the product with nt.
            nt2 = torch.nested.nested_tensor_from_jagged(values, offsets)
            return nt * nt2

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)

        def arg_fn(values=values, offsets=offsets):
            # [FUNCTION DOC] HELPER: Argument factory with defaults bound: builds a jagged
            # tensor from values and offsets and returns it together with values and offsets.
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            return nt, values, offsets

        self._validate_compile(fn, arg_fn)

    # See Note: [Creating symbolic nested int]
    # AssertionError: s2 (could be from ['<ephemeral: intermediate_offsets_or_lengths>',
    @unittest.expectedFailure
    def test_in_graph_construction_mixed_2(self):
        # [FUNCTION DOC] Expected failure test of a mixed case where an
        # intermediate offsets clone has an ephemeral source and a shape inequality
        # guard is created. KNOWN ISSUE: decorated expectedFailure because of an
        # assertion error about the symbol source (see the comment above it).
        def fn(nt, values, offsets, nt2):
            # [FUNCTION DOC] HELPER: Builds a jagged tensor from values and a clone of offsets,
            # and doubles nt if the second dim of nt2 differs from it, then returns nt.
            # Intermediate offsets has ephemeral source
            intermediate_nt = torch.nested.nested_tensor_from_jagged(
                values, offsets.clone()
            )
            # This creates a dynamic shapes neq guard
            if nt2.shape[1] != intermediate_nt.shape[1]:
                # We should always go here.
                nt = nt * 2
            return nt

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        offsets2 = torch.tensor([0, 1, 4, 10], dtype=torch.int64)

        def arg_fn(values=values, offsets=offsets, offsets2=offsets2):
            # [FUNCTION DOC] HELPER: Argument factory: builds two jagged tensors sharing values
            # with different offsets and returns the first, values, offsets and the second.
            # Values is shared, but it shouldn't matter
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            nt2 = torch.nested.nested_tensor_from_jagged(values, offsets2)
            return nt, values, offsets, nt2

        self._validate_compile(fn, arg_fn)

    def test_in_graph_construction_mixed_3(self):
        # [FUNCTION DOC] A more involved mixed case: two jagged tensors built in the function
        # from values and shared offsets plus a jagged tensor input; validated against eager.
        # More involved mixed case
        def fn(nt, values, offsets):
            # [FUNCTION DOC] HELPER: Builds two jagged tensors from scaled
            # values with the offsets and returns their sum with nt.
            nt1 = torch.nested.nested_tensor_from_jagged(values * 2, offsets)
            nt2 = torch.nested.nested_tensor_from_jagged(values * 3, offsets)
            return nt1 + nt2 + nt

        values = torch.randn(9, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 9], dtype=torch.int64)

        def arg_fn(values=values, offsets=offsets):
            # [FUNCTION DOC] HELPER: Argument factory: builds a jagged tensor
            # from values and offsets and returns it with values and offsets.
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            return nt, values, offsets

        self._validate_compile(fn, arg_fn)

    def test_return_shape(self):
        # [FUNCTION DOC] Checks that a compiled function returning only the shape of
        # a jagged tensor result works under fullgraph with aot_eager (only runs it).
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)

        def fn(nt):
            # [FUNCTION DOC] HELPER: Returns the shape of nt times 2.
            return (nt * 2).shape

        compiled = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled(nt)

    def test_inference_tensor(self):
        # [FUNCTION DOC] Checks that a jagged tensor created under inference mode
        # can be passed to a compiled function (eager backend); only runs it.
        with torch.inference_mode():
            nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)

        def fn(n):
            # [FUNCTION DOC] HELPER: Returns n times 2.
            return n * 2

        torch.compile(fn, backend="eager")(nt)

    # TODO: cannot parametrize this test class with device for some reason
    def _test_autograd(self, backend):
        # [FUNCTION DOC] HELPER: Shared body for autograd tests: builds three float64
        # leaf tensors, makes two jagged tensors from them (public API and internal
        # helper), compiles an add-sin-cos function with the given backend, fullgraph
        # and dynamic shapes, takes gradients of the summed values buffer with
        # respect to the leaves and checks they match eager gradients with allclose.
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        # TODO: Switch to public API when it exists
        nt2, _ = jagged_from_list([a, b, c], nt.offsets())

        def fn1(nt1, nt2):
            # [FUNCTION DOC] HELPER: Returns cos of sin of the sum of the two inputs.
            return (nt1 + nt2).sin().cos()

        compiled_f = torch.compile(fn1, fullgraph=True, backend=backend, dynamic=True)
        out = compiled_f(nt, nt2)
        out_buffer = out.values()
        ga, gb, gc = torch.autograd.grad(out_buffer.sum(), (a, b, c))

        out_ref = fn1(nt, nt2)
        out_buffer_ref = out_ref.values()
        ga_ref, gb_ref, gc_ref = torch.autograd.grad(out_buffer_ref.sum(), (a, b, c))

        self.assertTrue(torch.allclose(ga, ga_ref))
        self.assertTrue(torch.allclose(gb, gb_ref))
        self.assertTrue(torch.allclose(gc, gc_ref))

    def test_basic_autograd(self):
        # [FUNCTION DOC] Runs the shared autograd test with the aot_eager backend.
        self._test_autograd("aot_eager")

    @requires_gpu_and_triton
    def test_basic_autograd_inductor(self):
        # [FUNCTION DOC] Runs the shared autograd test with the
        # inductor backend; skipped without a GPU and Triton.
        self._test_autograd("inductor")

    def test_subclass_with_mutation_in_graph(self):
        # [FUNCTION DOC] Regression test for a subclass graph with an in-graph mutation
        # (in-place sine on a dense input) compiled with inductor: forward results,
        # values and gradients of the jagged output must equal eager, guarding against
        # a mismatch in the number of forward outputs for the joint graph.
        # In this graph, we have an in-graph mutation, i.e. a mutation that is allowed
        # to remain in the graph. Normally this is allowed, but it's not allowed if
        # the graph handles subclasses at all.
        # Whether the mutation is allowed or not allowed in the graph alters the number
        # of outputs from the forward graph. Previously, a bug in this handling meant
        # that sometimes the expected number and actual number of outputs from the
        # joint graph did not match, causing assertion failures.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the cos of sin of
            # x and the cos of y after an in-place sine on y.
            z = x.sin()
            y.sin_()
            return z.cos(), y.cos()

        fn_c = torch.compile(fn, backend="inductor")

        values = [torch.rand((i, 8), requires_grad=True) for i in range(1, 6)]
        values_copy = [x.detach().clone().requires_grad_(True) for x in values]

        nt, offsets = jagged_from_list(values, None)
        nt_copy, offsets = jagged_from_list(values_copy, offsets)
        y = torch.rand((4, 8))
        y_copy = y.clone()

        ret = fn_c(nt, y)[0]
        ref = fn(nt_copy, y_copy)[0]

        self.assertEqual(ret.values(), ref.values())

        ret.values().sum().backward()
        ref.values().sum().backward()
        for ref_v, res_v in zip(values_copy, values):
            self.assertEqual(ref_v.grad, res_v.grad)

    @torch._dynamo.config.patch({"capture_scalar_outputs": True})
    def test_unbind(self):
        # [FUNCTION DOC] Checks unbind on jagged tensors with scalar output capture
        # enabled: compiled output equals eager, and no recompile for another tensor
        # with the same number of constituents but recompile when the count differs.
        # NB: If we have shape e.g. (3, j0, 3), duck sizing will give us (s0, s1, s0).
        # This causes a recompile later on when it realizes the batch and last dim
        # should not always be equal. To avoid that, we use (3, j0, 5) here.
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 5), 2), None)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4, 5), 3), None)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the unbind of x.
            return x.unbind()

        compiled_f = torch.compile(fn, fullgraph=True, backend="eager", dynamic=True)
        out = compiled_f(nt)

        out_ref = fn(nt)

        # correctness
        self.assertEqual(len(out), len(out_ref))
        for x, x_ref in zip(out, out_ref):
            self.assertTrue(torch.allclose(x, x_ref))

        # We specialize on the length of offsets, e.g. (1) we recompile if the
        # length of the offsets is different. (2) we don't recompile if the
        # length of the offsets is the same, even if the size of the constituent
        # tensors are different.
        self._check_recompiles(fn, (nt,), (nt2,), False)
        self._check_recompiles(fn, (nt,), (nt3,), True)

    def test_inline_nested_tensor_from_jagged(self):
        # [FUNCTION DOC] Checks that building a jagged tensor from the values and offsets
        # of another inside a compiled function (aot_eager, fullgraph) works; only runs it.
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a jagged tensor
            # from doubled values and the offsets of x.
            return torch.nested.nested_tensor_from_jagged(x.values() * 2, x.offsets())

        torch.compile(fn, fullgraph=True, backend="aot_eager")(nt)

    # The test here: nn.Parameters that are secretly subclasses
    # have a metaclass that overrides __isinstance__,
    # that dynamo needs to respect when it inlines the if statement.
    def test_param_subclass_isinstance_input(self):
        # [FUNCTION DOC] Checks that an isinstance check of a parameter holding a subclass against
        # Parameter is respected by Dynamo (the metaclass overrides the instance check): a module
        # weight that is a TwoTensor parameter makes the compiled function equal to eager.
        x_inner = torch.randn(16, 16, requires_grad=True)
        x = torch.nn.Parameter(TwoTensor(x_inner, x_inner))
        m = torch.nn.Linear(16, 16)
        m.weight = x

        def fn():
            # [FUNCTION DOC] HELPER: Returns weight plus 1
            # if the weight is a Parameter, else plus 2.
            if isinstance(m.weight, torch.nn.Parameter):
                return m.weight + 1
            else:
                return m.weight + 2

        out_ref = fn()
        out_test = torch.compile(fn, backend="aot_eager")()
        self.assertEqual(out_ref, out_test)

    def test_buffer_subclass_isinstance_input(self):
        # [FUNCTION DOC] Checks isinstance against the Buffer class under
        # fullgraph aot_eager: a Buffer input takes the first branch and a plain
        # tensor (after resetting Dynamo) the second, each matching eager.
        from torch.nn.parameter import Buffer

        buf = Buffer(torch.ones(5))

        def fn(b):
            # [FUNCTION DOC] HELPER: Returns b plus 1 if b is a Buffer, else b plus 2.
            if isinstance(b, torch.nn.Buffer):
                return b + 1
            else:
                return b + 2

        out_ref = fn(buf)
        out_test = torch.compile(fn, backend="aot_eager", fullgraph=True)(buf)
        self.assertEqual(out_ref, out_test)

        torch._dynamo.reset()

        tensor = torch.ones(5)
        out_ref_tensor = fn(tensor)
        out_test_tensor = torch.compile(fn, backend="aot_eager", fullgraph=True)(tensor)
        self.assertEqual(out_ref_tensor, out_test_tensor)

    def test_buffer_subclass_check(self):
        # [FUNCTION DOC] Checks that an isinstance check against Buffer
        # compiled with inductor under fullgraph returns True for a
        # Buffer and False for a plain tensor (Dynamo reset in between).
        from torch.nn.parameter import Buffer

        def check_buffer(x):
            # [FUNCTION DOC] HELPER: Returns whether x is an instance of Buffer.
            return isinstance(x, torch.nn.Buffer)

        buf = Buffer(torch.ones(5))
        compiled_fn = torch.compile(check_buffer, fullgraph=True, backend="inductor")
        self.assertTrue(compiled_fn(buf))

        torch._dynamo.reset()

        tensor = torch.ones(5)
        compiled_fn = torch.compile(check_buffer, fullgraph=True, backend="inductor")
        self.assertFalse(compiled_fn(tensor))

    def _input_view_test(self, nt_view_name):
        # [FUNCTION DOC] HELPER: Shared body for view input tests. Builds the view tensor
        # for the named case from the module table of view cases, computes the eager
        # sine, compiles it with aot_eager (fullgraph, dynamic) and checks size, stride
        # and values (of the inner values buffer for nested outputs). Then recompiles
        # with a custom backend that checks the symbolic shape guards recorded against
        # expected inline text chosen by the case name, raising NotImplementedError for
        # unknown names, to verify no upper or lower bound guards appear.
        nt_view = VIEW_TEST_CASES[nt_view_name]()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return x.sin()

        out_ref = fn(nt_view)
        torch._dynamo.reset()
        compile_fn = torch.compile(
            fn, fullgraph=True, backend="aot_eager", dynamic=True
        )
        out = compile_fn(nt_view)

        # Check metadata and values are correct
        self.assertTrue(out.size() == out_ref.size())
        self.assertTrue(out.stride() == out_ref.stride())
        if out.is_nested:
            self.assertTrue(torch.allclose(out.values(), out_ref.values()))
        else:
            self.assertTrue(torch.allclose(out, out_ref))

        # Check that no upper/lower bound guards are incurred
        def backend(gm, args):
            # [FUNCTION DOC] HELPER: Guard checking backend: collects the shape env guard
            # expressions from the tracing context, joins them with newlines and asserts the
            # text matches the expectation for the current view case name (expected strings
            # differ by case; unknown name raises NotImplementedError). Returns gm unchanged.
            context = torch._guards.TracingContext.get()
            guards = [str(g.expr) for g in context.fake_mode.shape_env.guards]

            # varies based on the type of view
            guard_str = "\n".join(guards)

            if nt_view_name == "base_is_nt_False_basic":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s85 - 1, s64)
Eq(s20, s64)
Eq(s80 - 1, s77)
Eq(s72, s71)""",
                )
            elif nt_view_name == "base_is_nt_False_leaf_False_False":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s85 - 1, s64)
Eq(s80 - 1, s77)
Eq(s72, s71)""",
                )
            elif nt_view_name == "base_is_nt_False_leaf_False_True":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s85 - 1, s64)
Eq(s20, s64)
Eq(s80 - 1, s77)
Eq(s72, s71)""",
                )
            elif nt_view_name == "base_is_nt_False_leaf_True_False":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s85 - 1, s64)
Eq(s20, s64)
Eq(s80 - 1, s77)
Eq(s72, s71)""",
                )
            elif nt_view_name == "base_is_nt_False_leaf_True_True":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s85 - 1, s64)
Eq(s20, s64)
Eq(s80 - 1, s77)
Eq(s72, s71)""",
                )
            elif nt_view_name == "base_is_nt_False_obscure":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s85 - 1, s64)
Eq(s20, s64)
Eq(s80 - 1, s77)
Eq(s72, s71)""",
                )
            elif nt_view_name == "base_is_nt_True_basic":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s17 - 1, s83)
Eq(s20, s83)""",
                )
            elif nt_view_name == "base_is_nt_True_leaf_False_False":
                self.assertExpectedInline(
                    guard_str,
                    """Eq(s17 - 1, s83)""",
                )
            elif nt_view_name == "base_is_nt_True_leaf_False_True":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s17 - 1, s83)
Eq(s20, s83)""",
                )
            elif nt_view_name == "base_is_nt_True_leaf_True_False":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s17 - 1, s83)
Eq(s20, s83)""",
                )
            elif nt_view_name == "base_is_nt_True_leaf_True_True":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s17 - 1, s83)
Eq(s20, s83)""",
                )
            elif nt_view_name == "base_is_nt_True_obscure":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s17 - 1, s83)
Eq(s20, s83)""",
                )
            elif nt_view_name == "dense_subclass_dense_subclass":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s85 - 1, s77)
Eq(s80 - 1, s78)
Eq(s72, s71)""",
                )
            elif nt_view_name == "subclass_dense":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s85 - 1, s77)
Eq(s20, s77)""",
                )
            else:
                raise NotImplementedError
            return gm

        torch._dynamo.reset()
        compile_fn = torch.compile(fn, fullgraph=True, backend=backend, dynamic=True)
        out = compile_fn(nt_view)

    @parametrize(
        "nt_view_name",
        [k for k in VIEW_TEST_CASES if k != "subclass_dense_subclass_dense"],
    )
    def test_inputs_to_compiled_fn_are_views(self, nt_view_name):
        # [FUNCTION DOC] Parametrized over all view cases except the subclass to dense
        # to subclass to dense chain; runs the shared view input test for each name.
        self._input_view_test(nt_view_name)

    def test_subclass_gives_static_shapes_when_dynamic_false(self):
        # [FUNCTION DOC] Checks that compiling a subclass view with dynamic disabled gives static
        # sizes: a custom backend asserts every dimension of the first placeholder example value
        # is a plain int, and the test calls the compiled function on a view of a TwoTensor.
        def check_graph(gm, *args):
            # [FUNCTION DOC] HELPER: Backend that asserts all sizes of the first graph
            # node's example value are ints (static), then returns gm unchanged.
            first_node_example_val = next(iter(gm.graph.nodes)).meta["example_value"]
            # We compiled with dynamic=False, expect no SymInt sizes on our placeholders
            self.assertTrue(
                all(isinstance(x, int) for x in first_node_example_val.shape)
            )
            return gm

        @torch.compile(backend=check_graph, dynamic=False)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.
            return x + 1

        x_inner = torch.ones(4)
        x = TwoTensor(x_inner, x_inner)
        x_view = x.view(2, 2)
        out = f(x_view)  # noqa: F841

    # NJT1 -> Dense -> NJT2 -> Dense view
    # During view replay, the Dense -> NJT2 part will construct an intermediate,
    # symbolically-sized NJT that is immediately deconstructed to return the final dense
    # view. To construct this intermediate properly, we need the associated nested int
    # to be symbolic. This view is expected to fail compilation until symbolic nested ints
    # are cached onto fake offsets to solve this problem.
    @unittest.expectedFailure
    def test_subclass_dense_subclass_dense_view(self):
        # [FUNCTION DOC] Expected failure test running the shared view input test on
        # the subclass to dense to subclass to dense case. KNOWN ISSUE: decorated
        # expectedFailure until symbolic nested ints are cached onto fake offsets, as
        # view replay needs a symbolic nested int for the intermediate jagged tensor.
        self._input_view_test("subclass_dense_subclass_dense")

    def test_njt_subclass_simple(self):
        # [FUNCTION DOC] Compiles a function cloning a jagged tensor and multiplying
        # by its batch size (dynamic shapes, with backward) using the nested compile
        # check, and compares the forward and backward graph text to expected strings.
        def f(nt):
            # [FUNCTION DOC] HELPER: Clones nt and multiplies by the size of dimension 0.
            y = nt.clone()
            return y * y.size(0)

        nt, _ = get_jagged_tensor(((2, 3, 4), 5), None, True)

        fw, bw = self._njt_compile_check(f, [(nt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f64[s64, s55]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values')
        primals_2: "Sym(s64)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=0)
        primals_3: "Sym(s55)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=1)
        primals_4: "Sym(s55)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=0)
        primals_5: "i64[s51 + 1]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_offsets')
        primals_6: "Sym(s51 + 1)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_offsets'), idx=0)
        primals_7: "f32[s0, 0]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_min_seqlen_tensor')
        primals_8: "Sym(s0)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_min_seqlen_tensor'), idx=0)
        primals_9: "f32[s83, 0]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_max_seqlen_tensor')
        primals_10: "Sym(s83)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_max_seqlen_tensor'), idx=0)
        primals_11: "Sym(s51)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_12: "Sym(s55)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=2)
        primals_13: "Sym(s55)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=1)
        primals_14: "Sym(s51)",  # PlainAOTInput(idx=1)
        primals_15: "Sym(s71)",  # PlainAOTInput(idx=2)
        primals_16: "Sym(s55)",  # PlainAOTInput(idx=3)
    ):
        clone: "f64[s64, s55]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None

        mul: "f64[s64, s55]" = torch.ops.aten.mul.Tensor(clone, primals_14);  clone = None
        return (
            mul,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values')
            primals_2,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values'), idx=0)
            primals_16,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values'), idx=1)
            primals_16,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values'), idx=0)
            primals_5,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_offsets')
            primals_6,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_offsets'), idx=0)
            primals_7,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_min_seqlen_tensor')
            primals_8,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_min_seqlen_tensor'), idx=0)
            primals_9,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_max_seqlen_tensor')
            primals_10,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_max_seqlen_tensor'), idx=0)
            primals_14,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            primals_16,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=2)
            primals_16,  # SubclassStrideAOTOutput(base=PlainAOTOutput(idx=0), idx=1)
            primals_2,  # SavedForBackwardsAOTOutput(idx=0)
            primals_6,  # SavedForBackwardsAOTOutput(idx=1)
            primals_8,  # SavedForBackwardsAOTOutput(idx=2)
            primals_10,  # SavedForBackwardsAOTOutput(idx=3)
            primals_14,  # SavedForBackwardsAOTOutput(idx=4)
            primals_16,  # SavedForBackwardsAOTOutput(idx=5)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_2: "Sym(s64)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=0)
        primals_6: "Sym(s51 + 1)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_offsets'), idx=0)
        primals_8: "Sym(s0)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_min_seqlen_tensor'), idx=0)
        primals_10: "Sym(s83)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_max_seqlen_tensor'), idx=0)
        primals_14: "Sym(s51)",  # PlainAOTInput(idx=1)
        primals_16: "Sym(s55)",  # PlainAOTInput(idx=3)
        tangents_1: "f64[s64, s55]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='_values')
        tangents_2: "i64[s51 + 1]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='_offsets')
        tangents_3: "f32[s0, 0]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='_min_seqlen_tensor')
        tangents_4: "f32[s83, 0]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='_max_seqlen_tensor')
    ):
        mul_1: "f64[s64, s55]" = torch.ops.aten.mul.Tensor(tangents_1, primals_14);  tangents_1 = None
        return (
            mul_1,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_values')
            primals_2,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_values'), idx=0)
            primals_16,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_values'), idx=1)
            primals_16,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_values'), idx=0)
            tangents_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_offsets')
            primals_6,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_offsets'), idx=0)
            tangents_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_min_seqlen_tensor')
            primals_8,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_min_seqlen_tensor'), idx=0)
            tangents_4,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_max_seqlen_tensor')
            primals_10,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_max_seqlen_tensor'), idx=0)
            primals_14,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            primals_16,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=2)
            primals_16,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=1)
            None,  # None
            None,  # None
            None,  # None
        )
""",
        )

    def test_njt_subclass_from_cat(self):
        # [FUNCTION DOC] Compiles a function concatenating a cloned jagged
        # tensor with itself along the last dim (dynamic shapes, with backward)
        # and compares the forward and backward graph text to expected strings.
        def f(nt):
            # [FUNCTION DOC] HELPER: Clones nt and returns the
            # concatenation of the clone with itself along the last dim.
            y = nt.clone()
            z = torch.cat([y, y], dim=-1)
            return z

        nt, _ = get_jagged_tensor(((2, 3, 4), 5), None, True)

        fw, bw = self._njt_compile_check(f, [(nt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_1: "f64[s64, s55]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values')
        primals_2: "Sym(s64)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=0)
        primals_3: "Sym(s55)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=1)
        primals_4: "Sym(s55)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=0)
        primals_5: "i64[s51 + 1]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_offsets')
        primals_6: "Sym(s51 + 1)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_offsets'), idx=0)
        primals_7: "f32[s0, 0]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_min_seqlen_tensor')
        primals_8: "Sym(s0)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_min_seqlen_tensor'), idx=0)
        primals_9: "f32[s83, 0]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_max_seqlen_tensor')
        primals_10: "Sym(s83)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_max_seqlen_tensor'), idx=0)
        primals_11: "Sym(s51)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=0)
        primals_12: "Sym(s55)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=2)
        primals_13: "Sym(s55)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=1)
        primals_14: "Sym(s51)",  # PlainAOTInput(idx=1)
        primals_15: "Sym(s71)",  # PlainAOTInput(idx=2)
        primals_16: "Sym(s55)",  # PlainAOTInput(idx=3)
    ):
        clone: "f64[s64, s55]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None

        cat: "f64[s64, 2*s55]" = torch.ops.aten.cat.default([clone, clone], 1);  clone = None

        add_2: "Sym(2*s55)" = primals_16 + primals_16
        return (
            cat,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values')
            primals_2,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values'), idx=0)
            add_2,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values'), idx=1)
            add_2,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values'), idx=0)
            primals_5,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_offsets')
            primals_6,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_offsets'), idx=0)
            primals_7,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_min_seqlen_tensor')
            primals_8,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_min_seqlen_tensor'), idx=0)
            primals_9,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_max_seqlen_tensor')
            primals_10,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_max_seqlen_tensor'), idx=0)
            primals_14,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=0)
            add_2,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=2)
            add_2,  # SubclassStrideAOTOutput(base=PlainAOTOutput(idx=0), idx=1)
            primals_2,  # SavedForBackwardsAOTOutput(idx=0)
            primals_6,  # SavedForBackwardsAOTOutput(idx=1)
            primals_8,  # SavedForBackwardsAOTOutput(idx=2)
            primals_10,  # SavedForBackwardsAOTOutput(idx=3)
            primals_14,  # SavedForBackwardsAOTOutput(idx=4)
            primals_16,  # SavedForBackwardsAOTOutput(idx=5)
            add_2,  # SavedForBackwardsAOTOutput(idx=6)
        )
""",
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class GraphModule(torch.nn.Module):
    def forward(
        self,
        primals_2: "Sym(s64)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=0)
        primals_6: "Sym(s51 + 1)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_offsets'), idx=0)
        primals_8: "Sym(s0)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_min_seqlen_tensor'), idx=0)
        primals_10: "Sym(s83)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_max_seqlen_tensor'), idx=0)
        primals_14: "Sym(s51)",  # PlainAOTInput(idx=1)
        primals_16: "Sym(s55)",  # PlainAOTInput(idx=3)
        add_2: "Sym(2*s55)",
        tangents_1: "f64[s64, 2*s55]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='_values')
        tangents_2: "i64[s51 + 1]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='_offsets')
        tangents_3: "f32[s0, 0]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='_min_seqlen_tensor')
        tangents_4: "f32[s83, 0]",  # SubclassGetAttrAOTInput(base=TangentAOTInput(output=PlainAOTOutput(idx=0)), attr='_max_seqlen_tensor')
    ):
        slice_1: "f64[s64, s55]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 0, primals_16)
        slice_2: "f64[s64, s55]" = torch.ops.aten.slice.Tensor(tangents_1, 1, primals_16, add_2);  tangents_1 = add_2 = None
        add_4: "f64[s64, s55]" = torch.ops.aten.add.Tensor(slice_1, slice_2);  slice_1 = slice_2 = None
        return (
            add_4,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_values')
            primals_2,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_values'), idx=0)
            primals_16,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_values'), idx=1)
            primals_16,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_values'), idx=0)
            tangents_2,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_offsets')
            primals_6,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_offsets'), idx=0)
            tangents_3,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_min_seqlen_tensor')
            primals_8,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_min_seqlen_tensor'), idx=0)
            tangents_4,  # SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_max_seqlen_tensor')
            primals_10,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), attr='_max_seqlen_tensor'), idx=0)
            primals_14,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=0)
            primals_16,  # SubclassSizeAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=2)
            primals_16,  # SubclassStrideAOTOutput(base=GradAOTOutput(grad_of=PlainAOTInput(idx=0)), idx=1)
            None,  # None
            None,  # None
            None,  # None
        )
""",
        )

    def test_njt_subclass_from_buffer(self):
        # [FUNCTION DOC] Compiles a function that builds a new jagged tensor from a buffer inside
        # the function, concatenates it with the input and applies sine and size scaling (dynamic
        # shapes, no backward) and compares the forward graph text to the expected string.
        def f(nt):
            # [FUNCTION DOC] HELPER: Builds a second jagged tensor without
            # grad inside the function, concatenates it with nt along the
            # last dim, and returns its sine times its batch size.
            nested_size = ((2, 3, 4), 5)
            offsets = None
            nt2, _ = get_jagged_tensor(nested_size, offsets, requires_grad=False)
            nt3 = torch.cat([nt2, nt], dim=-1)
            return nt3.sin() * nt3.size(0)

        nested_size = ((2, 3, 4), 5)
        offsets = None
        nt, _ = get_jagged_tensor(nested_size, offsets, requires_grad=False)

        fw, _ = self._njt_compile_check(
            f,
            [(nt,)],
            dynamic=True,
            call_backward=False,
        )

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False, expanded_def=True)),
            """\
class <lambda>(torch.nn.Module):
    def forward(
        self,
        arg0_1: "f64[9, s55]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values')
        arg1_1: "Sym(s55)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=1)
        arg2_1: "Sym(s55)",  # SubclassStrideAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_values'), idx=0)
        arg3_1: "i64[s51 + 1]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_offsets')
        arg4_1: "Sym(s51 + 1)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_offsets'), idx=0)
        arg5_1: "f32[s0, 0]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_min_seqlen_tensor')
        arg6_1: "Sym(s0)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_min_seqlen_tensor'), idx=0)
        arg7_1: "f32[s83, 0]",  # SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_max_seqlen_tensor')
        arg8_1: "Sym(s83)",  # SubclassSizeAOTInput(base=SubclassGetAttrAOTInput(base=PlainAOTInput(idx=0), attr='_max_seqlen_tensor'), idx=0)
        arg9_1: "Sym(s51)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=0)
        arg10_1: "Sym(s55)",  # SubclassSizeAOTInput(base=PlainAOTInput(idx=0), idx=2)
        arg11_1: "Sym(s55)",  # SubclassStrideAOTInput(base=PlainAOTInput(idx=0), idx=1)
        arg12_1: "Sym(s51)",  # PlainAOTInput(idx=1)
        arg13_1: "Sym(s71)",  # PlainAOTInput(idx=2)
        arg14_1: "Sym(s55)",  # PlainAOTInput(idx=3)
    ):
        randn: "f64[2, 5]" = torch.ops.aten.randn.default([2, 5], dtype = torch.float64, device = device(type='cpu'), pin_memory = False)
        randn_1: "f64[3, 5]" = torch.ops.aten.randn.default([3, 5], dtype = torch.float64, device = device(type='cpu'), pin_memory = False)
        randn_2: "f64[4, 5]" = torch.ops.aten.randn.default([4, 5], dtype = torch.float64, device = device(type='cpu'), pin_memory = False)

        cat: "f64[9, 5]" = torch.ops.aten.cat.default([randn, randn_1, randn_2]);  randn = randn_1 = randn_2 = None
        zeros: "i64[1]" = torch.ops.aten.zeros.default([1], dtype = torch.int64, device = device(type='cpu'), pin_memory = False)
        _tensor_constant0: "i64[3]" = self._tensor_constant0
        lift_fresh_copy: "i64[3]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        cumsum: "i64[3]" = torch.ops.aten.cumsum.default(lift_fresh_copy, 0);  lift_fresh_copy = None
        cat_1: "i64[4]" = torch.ops.aten.cat.default([zeros, cumsum]);  zeros = cumsum = None
        zeros_1: "f32[2, 0]" = torch.ops.aten.zeros.default([2, 0], device = device(type='cpu'), pin_memory = False)
        zeros_2: "f32[4, 0]" = torch.ops.aten.zeros.default([4, 0], device = device(type='cpu'), pin_memory = False)

        cat_2: "f64[9, s55 + 5]" = torch.ops.aten.cat.default([cat, arg0_1], 1);  cat = arg0_1 = None

        sin: "f64[9, s55 + 5]" = torch.ops.aten.sin.default(cat_2)
        mul: "f64[9, s55 + 5]" = torch.ops.aten.mul.Tensor(sin, 3);  sin = None

        sym_size_int: "Sym(s55 + 5)" = torch.ops.aten.sym_size.int(cat_2, 1);  cat_2 = None
        sym_stride_int: "Sym(s55 + 5)" = torch.ops.aten.sym_stride.int(mul, 0)
        return (
            mul,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values')
            sym_size_int,  # SubclassSizeAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values'), idx=1)
            sym_stride_int,  # SubclassStrideAOTOutput(base=SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_values'), idx=0)
            cat_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_offsets')
            zeros_1,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_min_seqlen_tensor')
            zeros_2,  # SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), attr='_max_seqlen_tensor')
            sym_size_int,  # SubclassSizeAOTOutput(base=PlainAOTOutput(idx=0), idx=2)
            sym_stride_int,  # SubclassStrideAOTOutput(base=PlainAOTOutput(idx=0), idx=1)
        )
""",
        )


instantiate_parametrized_tests(TestNestedTensor)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
