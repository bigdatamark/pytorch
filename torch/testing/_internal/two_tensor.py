# mypy: ignore-errors

# [MODULE DOC] Test-only tensor subclass TwoTensor, a wrapper holding two inner tensors of
# identical metadata and running every op on both, plus TwoTensorMode, a dispatch mode that
# turns tensor-constructor outputs into TwoTensors. No import-time side effects beyond imports.

import torch
import torch.utils._pytree as pytree
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.utils._python_dispatch import return_and_correct_aliasing


# A simple tensor subclass that holds two tensors internally, and runs every op on both tensors.
class TwoTensor(torch.Tensor):
    # [CLASS DOC] Wrapper tensor subclass with two inner tensors, attributes a and b, which
    # must agree in device, layout, requires_grad, dtype, shape, stride and storage offset.
    # Outer metadata mirrors a (or explicit outer size/stride). Every dispatched op is run
    # separately on a and b and the results are re-wrapped. Supports flatten/unflatten for
    # compile and export.
    @staticmethod
    def __new__(cls, a, b, outer_size=None, outer_stride=None, *, requires_grad=None):
        # [FUNCTION DOC] Creates the wrapper tensor. Takes inner tensors a and b, optional
        # outer size and stride (default to those of a) and optional requires_grad (or-ed
        # with a's). Returns the wrapper subclass instance with a's device/layout/dtype.
        # Raises AssertionError if a and b differ in metadata, shape, stride or storage offset.
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        if not (
            a.device == b.device
            and a.layout == b.layout
            and a.requires_grad == b.requires_grad
            and a.dtype == b.dtype
        ):
            raise AssertionError(
                "Expected a and b to have same device, layout, requires_grad, and dtype"
            )
        # I guess it would be more accurate to represent the shape as torch.cat(a, b).shape
        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = requires_grad or a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

        if a.shape != b.shape:
            raise AssertionError(
                f"Expected a.shape == b.shape, got {a.shape} != {b.shape}"
            )
        if a.stride() != b.stride():
            raise AssertionError(
                f"Expected a.stride() == b.stride(), got {a.stride()} != {b.stride()}"
            )
        if a.storage_offset() != b.storage_offset():
            raise AssertionError(
                f"Expected a.storage_offset() == b.storage_offset(), "
                f"got {a.storage_offset()} != {b.storage_offset()}"
            )
        return out

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, a, b, outer_size=None, outer_stride=None, *, requires_grad=None):
        # [FUNCTION DOC] Stores the inner tensors as attributes a and b; the other arguments
        # are only consumed by the creation method. Not traced by dynamo.
        self.a = a
        self.b = b

    def __repr__(self):
        # [FUNCTION DOC] Returns a string showing the reprs of both inner tensors.
        a_repr = repr(self.a)
        b_repr = repr(self.b)
        return f"TwoTensor({a_repr}, {b_repr})"

    def __tensor_flatten__(self):
        # [FUNCTION DOC] Returns the names of the inner tensor attributes (a, b) and no
        # extra metadata (None).
        return ["a", "b"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] Rebuilds a TwoTensor from a mapping holding inner tensors a and b.
        # Raises AssertionError if meta is not None, or if the inner tensors are plain
        # tensors and outer size or stride is missing.
        if meta is not None:
            raise AssertionError("Expected meta to be None")
        a, b = inner_tensors["a"], inner_tensors["b"]
        if type(a) is torch.Tensor:
            if outer_size is None:
                raise AssertionError("Expected outer_size to not be None")
            if outer_stride is None:
                raise AssertionError("Expected outer_stride to not be None")
        return TwoTensor(a, b, outer_size, outer_stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] Runs the op once with every TwoTensor argument replaced by its a
        # tensor and once with b, then pairs up the flattened outputs: tensor outputs are
        # wrapped into new TwoTensors, non-tensor outputs are taken from the a run. For the
        # conditional higher-order op the output is returned as is; otherwise aliasing is
        # corrected against the inputs.
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, args)
        args_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, args)

        kwargs_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, kwargs)
        kwargs_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, kwargs)

        out_a = func(*args_a, **kwargs_a)
        out_b = func(*args_b, **kwargs_b)
        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_b_flat = pytree.tree_leaves(out_b)
        # for aten ops that return non-tensors, just assume that
        # our two inner tensors return the same value
        out_flat = [
            cls(o_a, o_b) if isinstance(o_a, torch.Tensor) else o_a
            for o_a, o_b in zip(out_a_flat, out_b_flat, strict=True)
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        from torch._higher_order_ops.cond import cond_op

        if func is cond_op:
            return out
        else:
            return return_and_correct_aliasing(func, args, kwargs, out)

    def get_elem_a(self):
        # [FUNCTION DOC] Returns the first inner tensor.
        return self.a


class TwoTensorMode(torch.utils._python_dispatch.TorchDispatchMode):
    # [CLASS DOC] Dispatch mode that wraps the result of every tensor-constructor op into a
    # TwoTensor whose second half is a clone of the first. Stateless.
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Runs the op; if it is a tensor constructor, returns a TwoTensor of
        # the result and its clone, otherwise returns the result unchanged. Note kwargs is
        # unpacked directly, so a None kwargs would fail.
        out = func(*args, **kwargs)
        if torch._subclasses.fake_tensor._is_tensor_constructor(func):
            out = TwoTensor(out, out.clone())
        return out
