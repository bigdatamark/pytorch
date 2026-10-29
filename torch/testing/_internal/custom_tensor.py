# mypy: ignore-errors
# [MODULE DOC] Tensor subclasses for testing wrapper-subclass support in tracing and compilation:
# ConstantExtraMetadataTensor (wraps one tensor plus a constant attribute and rewraps dispatch
# results) and CustomTensorPlainOut (wraps two tensors, runs ops on both and returns plain
# tensors), together with the FancyNamedTuple type returned by a metadata accessor.


from collections import namedtuple

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


FancyNamedTuple = namedtuple("FancyNamedTuple", ["foo", "bar"])


# A simple tensor subclass that holds a tensor with custom metadata and custom method
class ConstantExtraMetadataTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Wrapper tensor subclass around one inner tensor (attribute elem) that
    # mirrors its shape, strides, offset, device, layout, dtype and requires_grad. Carries an
    # integer attribute constant_attribute (initially 4) that is flattened as subclass metadata
    # and survives unflattening. Dispatch unwraps inputs, runs the op, and rewraps tensor outputs.
    @staticmethod
    def __new__(cls, elem):
        # [FUNCTION DOC] HELPER: Creates the wrapper subclass instance with the same metadata
        # (size, strides, storage offset, device, layout, requires_grad, dtype) as elem, without
        # allocating new storage. The inner tensor is stored later by the initializer.
        shape = elem.shape
        kwargs = {}
        kwargs["strides"] = elem.stride()
        kwargs["storage_offset"] = elem.storage_offset()
        kwargs["device"] = elem.device
        kwargs["layout"] = elem.layout
        kwargs["requires_grad"] = elem.requires_grad
        kwargs["dtype"] = elem.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, elem):
        # [FUNCTION DOC] HELPER: Stores the inner tensor as elem and sets constant_attribute to 4.
        self.elem = elem
        self.constant_attribute = 4

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string showing the inner tensor repr inside the text
        # CustomTensor(...). (The text does not use the class's own name.)
        inner_repr = repr(self.elem)
        return f"CustomTensor({inner_repr})"

    def get_complicated_metadata(self):
        # [FUNCTION DOC] HELPER: Returns a FancyNamedTuple whose two fields both hold the current
        # constant_attribute value.
        return FancyNamedTuple(self.constant_attribute, self.constant_attribute)

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Subclass flatten protocol: returns the list of inner tensor
        # attribute names (just elem) and the constant_attribute value as metadata.
        return ["elem"], self.constant_attribute

    def add_constant(self, a):
        # [FUNCTION DOC] HELPER: Mutates the instance by increasing constant_attribute by a.
        # Returns None.
        self.constant_attribute += a

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Subclass unflatten protocol: rebuilds an instance from the
        # mapping of inner tensors (key elem) and restores constant_attribute from meta. Raises an
        # AssertionError if meta is None. The outer size and stride arguments are unused.
        if meta is None:
            raise AssertionError("Expected meta to not be None")
        elem = inner_tensors["elem"]
        out = ConstantExtraMetadataTensor(elem)
        out.constant_attribute = meta
        return out

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] HELPER: Dispatch hook. Replaces every wrapper in args and kwargs by its
        # inner tensor, runs the op, wraps each tensor in the (flattened) result in a new
        # wrapper, leaves non-tensor results as is, and fixes up aliasing for view and in-place
        # ops. Returns the rewrapped output structure. Note that a rewrapped result starts with a
        # fresh constant_attribute (4), not the input's value.
        if kwargs is None:
            kwargs = {}
        args_inner = pytree.tree_map_only(
            ConstantExtraMetadataTensor, lambda x: x.elem, args
        )

        kwargs_inner = pytree.tree_map_only(
            ConstantExtraMetadataTensor, lambda x: x.elem, kwargs
        )

        out_inner = func(*args_inner, **kwargs_inner)
        out_inner_flat, spec = pytree.tree_flatten(out_inner)
        # for aten ops that return non-tensors, just assume that
        # our cust inner tensors return the same value
        out_flat = [
            ConstantExtraMetadataTensor(o_inner)
            if isinstance(o_inner, torch.Tensor)
            else o_inner
            for o_inner in out_inner_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        return return_and_correct_aliasing(func, args, kwargs, out)


# A simple tensor subclass that always returns plain tensor during __torch_dispatch__
# It is similar to TwoTensor and is used to simulate torchao quantized tensors
class CustomTensorPlainOut(torch.Tensor):
    # [CLASS DOC] HELPER: Wrapper tensor subclass holding two inner tensors (elem1, elem2) with
    # metadata taken from elem1. Dispatch runs each op on both inner sets and returns, for
    # non-view ops, the elementwise sum of the two results as plain tensors (simulating quantized
    # tensors that dequantize on use). View ops return wrapped results. Not meant for view ops.
    @staticmethod
    def __new__(cls, elem1, elem2):
        # [FUNCTION DOC] HELPER: Creates the wrapper instance using the shape, strides, storage
        # offset, device, layout, requires_grad and dtype of elem1; elem2 is ignored here.
        shape = elem1.shape
        kwargs = {}
        kwargs["strides"] = elem1.stride()
        kwargs["storage_offset"] = elem1.storage_offset()
        kwargs["device"] = elem1.device
        kwargs["layout"] = elem1.layout
        kwargs["requires_grad"] = elem1.requires_grad
        kwargs["dtype"] = elem1.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, elem1, elem2):
        # [FUNCTION DOC] HELPER: Stores the two inner tensors as elem1 and elem2.
        self.elem1 = elem1
        self.elem2 = elem2

    def get_elem(self):
        # [FUNCTION DOC] HELPER: Returns the first inner tensor, elem1.
        return self.elem1

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string with the class name and the reprs of both inner
        # tensors.
        inner_repr_1 = repr(self.elem1)
        inner_repr_2 = repr(self.elem2)
        return f"CustomTensorPlainOut({inner_repr_1}, {inner_repr_2})"

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Subclass flatten protocol: returns the inner attribute names
        # (elem1, elem2) and no metadata.
        return ["elem1", "elem2"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Subclass unflatten protocol: rebuilds an instance from the
        # mapping of inner tensors (keys elem1, elem2). Metadata, size and stride are unused.
        elem1 = inner_tensors["elem1"]
        elem2 = inner_tensors["elem2"]
        out = CustomTensorPlainOut(elem1, elem2)
        return out

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] HELPER: Dispatch hook. Runs the op once on the elem1 parts and once on the
        # elem2 parts of the inputs. If the op is a view, pairs the two results into new wrappers
        # and fixes aliasing; otherwise returns the elementwise sum of the paired results as
        # plain tensors in the original output structure.
        # KNOWN ISSUE: both the view and non-view paths pass a generator (not a list) to the
        # unflatten routine, and non-tensor outputs are combined with plus, which only suits
        # numbers.
        # Don't use this tensor with view ops
        if kwargs is None:
            kwargs = {}
        args_inner_1 = pytree.tree_map_only(
            CustomTensorPlainOut, lambda x: x.elem1, args
        )

        kwargs_inner_1 = pytree.tree_map_only(
            CustomTensorPlainOut, lambda x: x.elem1, kwargs
        )

        args_inner_2 = pytree.tree_map_only(
            CustomTensorPlainOut, lambda x: x.elem2, args
        )

        kwargs_inner_2 = pytree.tree_map_only(
            CustomTensorPlainOut, lambda x: x.elem2, kwargs
        )

        out_inner_1 = func(*args_inner_1, **kwargs_inner_1)
        out_inner_2 = func(*args_inner_2, **kwargs_inner_2)

        out_inner_flat_1, spec = pytree.tree_flatten(out_inner_1)
        out_inner_flat_2, spec = pytree.tree_flatten(out_inner_2)

        if func.is_view:
            new_out = pytree.tree_unflatten(
                (
                    CustomTensorPlainOut(tensor1, tensor2)
                    for tensor1, tensor2 in zip(
                        out_inner_flat_1, out_inner_flat_2, strict=True
                    )
                ),
                spec,
            )
            return return_and_correct_aliasing(func, args, kwargs, new_out)

        out_new = (
            out_inner_flat_1[ix] + out_inner_flat_2[ix]
            for ix in range(len(out_inner_flat_1))
        )

        return pytree.tree_unflatten(out_new, spec)
