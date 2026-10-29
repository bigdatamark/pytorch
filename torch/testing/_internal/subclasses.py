# mypy: ignore-errors
# [MODULE DOC] Test tensor subclass WrapperSubclass, a wrapper tensor subclass that holds one
# inner tensor and forwards every dispatched operation to it, supporting flatten/unflatten
# for compile tracing. No import-time side effects.
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import is_fake
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import return_and_correct_aliasing


class WrapperSubclass(torch.Tensor):
    # [CLASS DOC] Wrapper tensor subclass with one inner tensor attribute a. The outer
    # tensor mirrors the inner tensor's metadata (or explicit outer size and stride) but
    # owns no data; operations are dispatched by unwrapping to the inner tensor and
    # rewrapping tensor results. Supports the traceable subclass protocol through
    # __tensor_flatten__ and __tensor_unflatten__.
    @staticmethod
    def __new__(cls, a, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] Creates the wrapper tensor object. Size and stride default to those
        # of a; storage offset, device, layout, requires_grad and dtype are copied from a.
        # Returns the new wrapper (without setting the inner tensor; see __init__).
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, outer_size, **kwargs)

        return out

    def __init__(self, a, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] Stores the inner tensor in attribute a; the size and stride
        # arguments are already used by __new__ and ignored here.
        self.a = a

    def __repr__(self):
        # [FUNCTION DOC] Returns text of the form WrapperSubclass followed by the inner
        # tensor's repr in parentheses.
        return f"WrapperSubclass({repr(self.a)})"

    def __tensor_flatten__(self):
        # [FUNCTION DOC] Returns the names of inner tensor attributes (just a) and no extra
        # metadata (None).
        return ["a"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] Rebuilds a WrapperSubclass from a mapping holding the inner tensor a.
        # Raises AssertionError if meta is not None, or if the inner tensor is fake and the
        # outer size or stride is missing. Returns the new wrapper.
        if meta is not None:
            raise AssertionError("Expected meta to be None")
        a = inner_tensors["a"]
        if is_fake(a):
            if outer_size is None:
                raise AssertionError("Expected outer_size to not be None")
            if outer_stride is None:
                raise AssertionError("Expected outer_stride to not be None")
        return WrapperSubclass(a, outer_size, outer_stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] Dispatch handler: unwraps every WrapperSubclass in args and kwargs
        # to its inner tensor, calls func, wraps each tensor in the (pytree) result in a new
        # WrapperSubclass, and returns the result. For the cond higher-order op returns it
        # directly; otherwise fixes aliasing and mutation relationships between inputs and
        # outputs via return_and_correct_aliasing.
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(WrapperSubclass, lambda x: x.a, args)

        kwargs_a = pytree.tree_map_only(WrapperSubclass, lambda x: x.a, kwargs)

        out_a = func(*args_a, **kwargs_a)
        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_flat = [
            WrapperSubclass(o_a) if isinstance(o_a, torch.Tensor) else o_a
            for o_a in out_a_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        from torch._higher_order_ops.cond import cond_op

        if func is cond_op:
            return out
        else:
            return return_and_correct_aliasing(func, args, kwargs, out)

    def __coerce_same_metadata_as_tangent__(
        self, expected_metadata: Any, expected_type: type | None = None
    ):
        # [FUNCTION DOC] Coerces this tangent to the expected type: returns the inner tensor
        # if expected_type is the inner tensor's type, a TwoTensor of the inner tensor and
        # its clone if expected_type is TwoTensor, otherwise None. expected_metadata is
        # unused.
        if expected_type is type(self.a):
            return self.a
        elif expected_type is TwoTensor:
            return TwoTensor(self.a, self.a.clone())

        return None
