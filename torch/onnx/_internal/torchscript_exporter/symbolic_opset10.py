# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
# [MODULE DOC] ONNX symbolic functions for opset 10 of the TorchScript exporter. Registers
#   (via the opset-10 registration partial) converters from aten and quantized ops to ONNX
#   graph nodes: div/floor_divide, sort, topk, max/avg pooling, upsample/interpolate (Resize),
#   slice, flip, fmod, embedding_bag, fake quantization, isinf/isfinite, nan_to_num and the
#   quantized:: ops (implemented as dequantize, opset9 op, requantize). Many ops delegate to
#   opset9. Import time: defines the registration partial only; the registration decorators
#   add entries to the global symbolic registry.
from __future__ import annotations

import functools
import sys
import warnings
from typing import TYPE_CHECKING

import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


if TYPE_CHECKING:
    from collections.abc import Sequence


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# This file exports ONNX ops for opset 10
# Opset 10 is supported by ONNX release 1.5.0
# release on 04/24/19


__all__ = [
    "dequantize",
    "div",
    "embedding_bag",
    "fake_quantize_per_tensor_affine",
    "flip",
    "fmod",
    "isfinite",
    "isinf",
    "nan_to_num",
    "quantize_per_tensor",
    "quantized_add_relu",
    "quantized_add",
    "quantized_cat",
    "quantized_conv1d_relu",
    "quantized_conv2d_relu",
    "quantized_conv3d_relu",
    "quantized_conv1d",
    "quantized_conv2d",
    "quantized_conv3d",
    "quantized_conv_transpose1d",
    "quantized_conv_transpose2d",
    "quantized_conv_transpose3d",
    "quantized_group_norm",
    "quantized_hardswish",
    "quantized_instance_norm",
    "quantized_layer_norm",
    "quantized_leaky_relu",
    "quantized_linear",
    "quantized_linear_relu",
    "quantized_mul",
    "quantized_sigmoid",
    "slice",
    "sort",
    "topk",
]


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=10)


@_onnx_symbolic("aten::div")
def div(g: jit_utils.GraphContext, self, other, *args):
    # [FUNCTION DOC] Symbolic for aten::div. With no extra argument it does true division via
    #   opset9; with a rounding_mode argument it forwards all extra arguments to the rounding-
    #   mode helper. Returns the output graph value.
    if len(args) == 0:
        return opset9.true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


@symbolic_helper.parse_args("v", "v", "s")
def _div_rounding_mode(g: jit_utils.GraphContext, self, other, rounding_mode):
    # [FUNCTION DOC] HELPER: Handles div with a rounding_mode string (parsed from the graph).
    #   Mode floor goes to the opset-10 floor division; any other mode (none, trunc) is
    #   delegated to the opset9 helper. Returns the graph value.
    if rounding_mode == "floor":
        return _floor_divide(g, self, other)
    else:
        return opset9._div_rounding_mode(g, self, other, rounding_mode)


@_onnx_symbolic("aten::_floor_divide")
def _floor_divide(g: jit_utils.GraphContext, self, other):
    # [FUNCTION DOC] HELPER: symbolic for aten::_floor_divide. If either operand is
    #   floating point, emits true division followed by Floor. Otherwise emits integer Div
    #   (which truncates toward zero) and subtracts one where the signs differ and the
    #   remainder is nonzero, selected with Where. Returns the result value.
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        out = opset9.true_divide(g, self, other)
        return g.op("Floor", out)
    else:
        # Integer division does truncation rounding
        div = g.op("Div", self, other)
        # Division is negative if: self < 0 != other < 0
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        negative = g.op("Xor", g.op("Less", self, zero), g.op("Less", other, zero))

        # For negative numbers with self % other != 0, subtract 1 to round down instead of up
        mod = g.op("Mod", self, other, fmod_i=0)
        fixup_mask = g.op("And", negative, g.op("Not", g.op("Equal", mod, zero)))

        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        fixup = g.op("Sub", div, one)
        return g.op("Where", fixup_mask, fixup, div)


@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def sort(g: jit_utils.GraphContext, self, dim, descending, out=None):
    # [FUNCTION DOC] Symbolic for aten::sort. Args: self tensor, dim, descending flag and an
    #   optional out (ignored by the helper). Delegates to the shared sort helper (TopK based)
    #   and returns its (values, indices) result.
    return symbolic_helper._sort_helper(g, self, dim, descending=descending, out=out)


@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    # [FUNCTION DOC] Symbolic for aten::topk. Args: self, k (graph value), dim, largest,
    #   sorted, optional out. Delegates to the shared topk helper and returns its (values,
    #   indices) result.
    return symbolic_helper._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out
    )


def _aten_max_pool_onnx(
    g: jit_utils.GraphContext,
    self: _C.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
) -> _C.Value:
    # [FUNCTION DOC] HELPER: Emits a MaxPool node (without returning indices) on the input,
    #   using the given kernel, strides, pads, dilations and ceil_mode. If the input is
    #   unbatched it is unsqueezed with a leading batch axis before pooling and squeezed
    #   afterwards. Returns the pooled value.
    # KNOWN ISSUE: the rank check compares a graph value (a Size node output) with a Python
    #   int, so it is never true during tracing and the unbatched handling is effectively dead
    #   code.
    self_rank = g.op("Size", g.op("Shape", self))
    if self_rank == unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = g.op(
            "Unsqueeze",
            self,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    pool_result, _ = g.op(
        "MaxPool",
        self,
        outputs=2,
        ceil_mode_i=ceil_mode,
        dilations_i=dilations,
        kernel_shape_i=kernel_shape,
        pads_i=pads,
        strides_i=strides,
    )

    if self_rank == unbatched_rank:
        pool_result = g.op(
            "Squeeze",
            pool_result,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    return pool_result


# For MaxPool
def _adjust_attributes_of_max_pool(
    expand_size: int,
    kernel_size: Sequence[int] | int,
    stride: Sequence[int] | int,
    padding: Sequence[int] | int,
    dilation: Sequence[int] | int,
) -> tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    # [FUNCTION DOC] HELPER: Normalizes max pool attributes (kernel_size, stride, padding,
    #   dilation, each an int or sequence) to ONNX form for expand_size spatial dims. Returns
    #   the tuple (kernel_shape, strides, pads, dilation). Scalars are broadcast; padding of
    #   length 1, 2 or 3 is repeated to give begin and end pads; an empty stride defaults to
    #   the kernel shape.
    # KNOWN ISSUE: the docstring says it adjusts avg_pool attributes, but this function is for
    #   max pool.
    """Adjust attributes of avg_pool to match ONNX specification."""

    if isinstance(dilation, int):
        dilation = [dilation] * expand_size

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size  # type: ignore[assignment]

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 1:
        pads = padding * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 2:
        # 2D padding
        pads = padding * 2  # type: ignore[operator, assignment]
    elif len(padding) == 3:
        # 3D padding
        pads = padding * 2  # type: ignore[operator, assignment]
    else:
        # When padding is already done for all dimensions,
        # we don't need to double it
        # eg: (1, 1, 1, 1, 1, 1)
        pads = padding  # type: ignore[assignment]

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride  # type: ignore[assignment]

    # pyrefly: ignore [bad-return]
    return (kernel_shape, strides, pads, dilation)


def _aten_max_pool_with_indices_onnx(
    g: jit_utils.GraphContext,
    self: _C.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
    n_dims_one: Sequence[int],
    n_dims_zero: Sequence[int],
    n_dims_axes: Sequence[int],
) -> tuple[_C.Value, Sequence[int]]:
    # [FUNCTION DOC] HELPER: Emits a MaxPool returning both pooled output and flat indices. A
    #   second MaxPool with unit kernels is used to compute per-position offsets, which are
    #   sliced (axes n_dims_axes) and subtracted from the raw indices so indices are relative
    #   to each channel plane. Unbatched inputs are unsqueezed and squeezed back. Returns
    #   (pool_result, indices).
    # KNOWN ISSUE: the unbatched rank check compares a graph value with an int, so it is never
    #   true during tracing; also the return annotation says the second item is a sequence of
    #   ints but it is a graph value.
    self_rank = g.op("Size", g.op("Shape", self))
    if self_rank == unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = g.op(
            "Unsqueeze",
            self,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    pool_result, indices = g.op(
        "MaxPool",
        self,
        outputs=2,
        ceil_mode_i=ceil_mode,
        dilations_i=dilations,
        kernel_shape_i=kernel_shape,
        pads_i=pads,
        strides_i=strides,
    )
    _, flatten_indices = g.op(
        "MaxPool",
        self,
        outputs=2,
        dilations_i=dilations,
        kernel_shape_i=n_dims_one,
        strides_i=n_dims_one,
    )

    ends = g.op("Constant", value_t=torch.tensor(n_dims_one))
    starts = g.op("Constant", value_t=torch.tensor(n_dims_zero))
    axes = g.op("Constant", value_t=torch.tensor(n_dims_axes))

    delta = g.op("Slice", flatten_indices, starts, ends, axes)
    indices = g.op("Sub", indices, delta)

    if self_rank == unbatched_rank:
        pool_result = g.op(
            "Squeeze", pool_result, value_t=torch.tensor([0], dtype=torch.int64)
        )
        indices = g.op("Squeeze", indices, value_t=torch.tensor([0], dtype=torch.int64))

    return (pool_result, indices)


@_onnx_symbolic(
    "aten::max_pool1d",
    decorate=[symbolic_helper._apply_params("max_pool1d", 1, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool2d",
    decorate=[symbolic_helper._apply_params("max_pool2d", 2, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool3d",
    decorate=[symbolic_helper._apply_params("max_pool3d", 3, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool1d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool1d_with_indices",
            1,
            return_indices=True,
        )
    ],
)
@_onnx_symbolic(
    "aten::max_pool2d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool2d_with_indices",
            2,
            return_indices=True,
        )
    ],
)
@_onnx_symbolic(
    "aten::max_pool3d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool3d_with_indices",
            3,
            return_indices=True,
        )
    ],
)
def _max_pool(name: str, expand_size: int, return_indices: bool):
    # [FUNCTION DOC] HELPER: Factory registered for max_pool1d/2d/3d and their _with_indices
    #   variants. Args: name, expand_size (spatial dims), return_indices. Returns a symbolic
    #   function taking input, kernel_size, stride, padding, dilation and ceil_mode that
    #   normalizes the attributes and emits a MaxPool graph, with or without indices.
    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
    def symbolic_fn(
        g: jit_utils.GraphContext,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: int | Sequence[int],
        dilation: Sequence[int],
        ceil_mode: bool,
    ):
        # [FUNCTION DOC] HELPER: Generated symbolic: adjusts attributes, then builds the max
        #   pool subgraph (with indices when return_indices was set). Returns the pooled value
        #   or a (values, indices) pair.
        kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
            expand_size, kernel_size, stride, padding, dilation
        )

        if return_indices:
            return _aten_max_pool_with_indices_onnx(
                g,
                input,
                kernel_shape,
                strides,
                pads,
                dilations,
                ceil_mode,
                expand_size + 1,
                ([1] * expand_size),
                ([0] * expand_size),
                ([2 + i for i in range(expand_size)]),
            )
        else:
            return _aten_max_pool_onnx(
                g,
                input,
                kernel_shape,
                strides,
                pads,
                dilations,
                ceil_mode,
                expand_size + 1,
            )

    return symbolic_fn


# For AvgPool
def _adjust_attributes_of_avg_pool(
    expand_size: int,
    kernel_size: Sequence[int] | int,
    stride: Sequence[int] | int,
    padding: Sequence[int] | int,
) -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
    # [FUNCTION DOC] HELPER: Normalizes avg pool attributes (kernel_size, stride, padding; int
    #   or sequence) to ONNX form for expand_size spatial dims. Returns (kernel_shape,
    #   strides, pads). Scalars are broadcast, short padding is repeated, and an empty stride
    #   defaults to the kernel shape.
    """Adjust attributes of avg_pool to match ONNX specification."""

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size  # type: ignore[assignment]

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 2:
        pads = padding * expand_size  # type: ignore[operator, assignment]
    else:
        pads = padding * 2  # type: ignore[operator, assignment]

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride  # type: ignore[assignment]

    # pyrefly: ignore [bad-return]
    return (kernel_shape, strides, pads)


@_onnx_symbolic(
    "aten::avg_pool1d",
    decorate=[symbolic_helper._apply_params("avg_pool1d", 1)],
)
@_onnx_symbolic(
    "aten::avg_pool2d",
    decorate=[symbolic_helper._apply_params("avg_pool2d", 2)],
)
@_onnx_symbolic(
    "aten::avg_pool3d",
    decorate=[symbolic_helper._apply_params("avg_pool3d", 3)],
)
def _avg_pool(name, expand_size):
    # [FUNCTION DOC] HELPER: Factory registered for avg_pool1d/2d/3d. Args: name, expand_size.
    #   Returns a symbolic function that emits an AveragePool node.
    @symbolic_helper.quantized_args(True, False, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    def symbolic_fn(
        g,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: int | Sequence[int],
        ceil_mode: int,
        count_include_pad: int,
        divisor_override=None,
    ):
        # [FUNCTION DOC] HELPER: Generated symbolic: normalizes attributes and emits
        #   AveragePool with ceil_mode and count_include_pad. Returns the result value.
        # KNOWN ISSUE: divisor_override is accepted but silently ignored.
        kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
            expand_size, kernel_size, stride, padding
        )

        result = g.op(
            "AveragePool",
            input,
            ceil_mode_i=ceil_mode,
            count_include_pad_i=count_include_pad,
            kernel_shape_i=kernel_shape,
            pads_i=pads,
            strides_i=strides,
        )

        return result

    return symbolic_fn


@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[symbolic_helper._apply_params("upsample_linear1d", 3, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear")],
)
def _interpolate(name, dim, interpolate_mode):
    # [FUNCTION DOC] HELPER: Factory registered for the upsample_* ops. Args: name, dim (rank
    #   of input), interpolate_mode (nearest or linear). Returns a symbolic function that
    #   emits a Resize node.
    @symbolic_helper.quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        # [FUNCTION DOC] HELPER: Generated symbolic for upsample ops. Computes scales (from
        #   the arguments or from output_size), emits an interpolation warning, and returns a
        #   Resize node. Unsupported (reported through the unimplemented helper) when
        #   align_corners is true.
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        symbolic_helper._interpolate_warning(interpolate_mode)
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True", input)
        if scales is None:
            scales = symbolic_helper._interpolate_size_to_scales(
                g, input, output_size, dim
            )
        return g.op("Resize", input, scales, mode_s=interpolate_mode)

    return symbolic_fn


@_onnx_symbolic("aten::__interpolate")
def __interpolate(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    # [FUNCTION DOC] HELPER: Symbolic for aten::__interpolate. Derives scales and mode from
    #   size, scale_factor, mode and align_corners via the shared helper and emits a Resize
    #   node. recompute_scale_factor and antialias are ignored. Returns the result.
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    return g.op("Resize", input, scales, mode_s=mode)


def _slice(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    axes: list | torch.Tensor | torch._C.Value,
    starts: list | torch.Tensor | torch._C.Value,
    ends: list | torch.Tensor | torch._C.Value,
    steps: list | torch.Tensor | torch._C.Value | None = None,
):
    # [FUNCTION DOC] HELPER: Builds a Slice node for input over axes, starts, ends and
    #   optional steps; each may be a list, tensor or graph value. Returns the input unchanged
    #   when the slice is a no-op (start 0, end INT64_MAX, step 1). Missing starts/ends
    #   default to 0 and INT64_MAX. Raises SymbolicValueError if a graph-valued argument has
    #   rank other than 0 or 1.
    def is_none_value(value):
        # [FUNCTION DOC] HELPER: Returns true if the value is Python None or a graph constant
        #   of None type.
        if value is None:
            return True
        return (
            isinstance(value, torch._C.Value)
            and value.node().kind() == "prim::Constant"
            and isinstance(value.type(), _C.NoneType)
        )

    def to_slice_input(list_or_value, default_value=None):
        # [FUNCTION DOC] HELPER: Converts a list, tensor or graph value to a 1-D graph value
        #   (constant node or unsqueezed scalar), using default_value when the input is none.
        #   Raises SymbolicValueError if rank is not 0 or 1.
        # Convert input param into a 1D torch.Value.
        if is_none_value(list_or_value) and default_value is not None:
            list_or_value = [default_value]

        if isinstance(list_or_value, torch.Tensor):
            return g.op("Constant", value_t=list_or_value.clone().detach())
        elif isinstance(list_or_value, list):
            return g.op("Constant", value_t=torch.tensor(list_or_value))

        rank = symbolic_helper._get_tensor_rank(list_or_value)
        if rank == 0:
            return symbolic_helper._unsqueeze_helper(g, list_or_value, [0])
        if rank == 1:
            return list_or_value
        raise errors.SymbolicValueError(
            f"Rank must be 0 or 1, not {rank}", list_or_value
        )

    def get_const_value(list_or_value):
        # [FUNCTION DOC] HELPER: Returns the single constant integer of a one-element
        #   list/tensor or a constant graph value, else None.
        if isinstance(list_or_value, (list, torch.Tensor)):
            if len(list_or_value) == 1:
                return list_or_value[0]
            return None
        return symbolic_helper._maybe_get_const(list_or_value, "i")

    # Check if slice is a no-op
    if (
        get_const_value(starts) == 0
        and get_const_value(ends) == _constants.INT64_MAX
        and (steps is None or get_const_value(steps) == 1)
    ):
        return input

    axes = to_slice_input(axes)
    starts = to_slice_input(starts, default_value=0)
    ends = to_slice_input(ends, default_value=_constants.INT64_MAX)
    if steps is None:
        return g.op("Slice", input, starts, ends, axes)
    steps = to_slice_input(steps, default_value=1)
    return g.op("Slice", input, starts, ends, axes, steps)


@_onnx_symbolic("aten::slice")
def slice(g: jit_utils.GraphContext, self, *args):
    # [FUNCTION DOC] Symbolic for aten::slice. Accepts either 4 extra args (dim, start, end,
    #   step) for tensors or 3 (start, end, step) for lists, where the dim is then zero.
    #   Delegates to the shared slice helper. Raises SymbolicValueError on any other argument
    #   count.
    if len(args) == 4:
        # aten::slice(Tensor self, int dim, int? start=None, int? end=None, int step=1) -> Tensor
        dims, start, end, step = args
    elif len(args) == 3:
        # aten::slice(t[] l, int? start=None, int? end=None, int step=1) -> t[]
        start, end, step = args
        dims = [0]
    else:
        raise errors.SymbolicValueError("Unknown aten::slice signature", self)

    return symbolic_helper._slice_helper(
        g,
        self,
        axes=dims,
        starts=start,
        ends=end,
        steps=step,
    )


@_onnx_symbolic("aten::flip")
@symbolic_helper.parse_args("v", "is")
def flip(g: jit_utils.GraphContext, input, dims):
    # [FUNCTION DOC] Symbolic for aten::flip. Implemented as a Slice over the given dims with
    #   start -1, end minus INT64_MAX and step -1. Returns the flipped value.
    return symbolic_helper._slice_helper(
        g,
        input,
        axes=dims,
        starts=[-1] * len(dims),
        ends=[-_constants.INT64_MAX] * len(dims),
        steps=[-1] * len(dims),
    )


@_onnx_symbolic("aten::fmod")
def fmod(g: jit_utils.GraphContext, input, other):
    # [FUNCTION DOC] Symbolic for aten::fmod. Emits a Mod node with fmod semantics (sign
    #   follows the dividend). Returns the value.
    return g.op("Mod", input, other, fmod_i=1)


@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    # [FUNCTION DOC] Symbolic for aten::embedding_bag, supported only when the offsets shape
    #   is static. Unrolls over bags: slices indices per offset range, gathers embeddings,
    #   optionally multiplies by per_sample_weights, reduces by mode (0 sum, 1 mean, otherwise
    #   max) and concatenates. Returns (output, None, None, None). Warns that dynamic shapes
    #   need opset 11+. Raises RuntimeError if padding_idx is non-negative; reports
    #   unsupported for training scale_grad_by_freq or unknown offsets length.
    if scale_grad_by_freq and GLOBALS.export_training:
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with scale_grad_by_freq for training mode"
        )
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")

    warnings.warn(
        "Export of embedding_bag with dynamic input/offsets shape is not supported in opset 10. "
        "Please use opset 11 or higher to export model for dynamic input shape.'",
        stacklevel=2,
    )
    offsets_dim_0 = symbolic_helper._get_tensor_dim_size(offsets, 0)
    if offsets_dim_0 is not None:
        if include_last_offset:
            offset_len = offsets_dim_0 - 1
            offsets_extended = offsets
        else:
            offset_len = offsets_dim_0
            offsets_extended = [
                offsets,
                g.op("Constant", value_t=torch.tensor([sys.maxsize])),
            ]
            offsets_extended = g.op("Concat", *offsets_extended, axis_i=0)
        list_ = []
        for i in range(offset_len):
            start_ = symbolic_helper._unsqueeze_helper(
                g,
                opset9.select(g, offsets_extended, torch.tensor(0), torch.tensor(i)),
                [0],
            )
            end_ = symbolic_helper._unsqueeze_helper(
                g,
                opset9.select(
                    g, offsets_extended, torch.tensor(0), torch.tensor(i + 1)
                ),
                [0],
            )
            axes_ = g.op("Constant", value_t=torch.tensor([0]))
            indices_row = g.op("Slice", indices, start_, end_, axes_)

            embeddings = g.op("Gather", embedding_matrix, indices_row)
            if not symbolic_helper._is_none(per_sample_weights):
                per_sample_weights_row = g.op(
                    "Slice", per_sample_weights, start_, end_, axes_
                )
                per_sample_weights_row = symbolic_helper._unsqueeze_helper(
                    g, per_sample_weights_row, [1]
                )
                embeddings = g.op("Mul", embeddings, per_sample_weights_row)
            if mode == 0:
                embeddings = symbolic_helper._reducesum_helper(
                    g, embeddings, axes_i=[0], keepdims_i=0
                )
            elif mode == 1:
                embeddings = g.op("ReduceMean", embeddings, axes_i=[0], keepdims_i=0)
            else:
                embeddings = g.op("ReduceMax", embeddings, axes_i=[0], keepdims_i=0)

            embeddings = symbolic_helper._unsqueeze_helper(g, embeddings, [0])
            list_.append(embeddings)

        output = g.op("Concat", *list_, axis_i=0)
        # aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
        # But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
        return output, None, None, None
    else:
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with unknown shape of offsets for opset 10 is not supported. "
            "please use opset 11 or higher."
        )


@_onnx_symbolic("aten::fake_quantize_per_tensor_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i")
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
):
    # [FUNCTION DOC] Symbolic for aten::fake_quantize_per_tensor_affine. Emits QuantizeLinear
    #   followed by DequantizeLinear using a constant scale and a zero point cast to uint8
    #   (quant_min 0) or int8. Raises SymbolicValueError unless the range is (0, 255) or
    #   (-128, 127); the range (0, 127) and non-constant scale are reported as unsupported
    #   before opset 13. Returns the dequantized value.
    # NOTE: (0, 127) is a special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) == (0, 127):
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Quantize range (0, 127) not supported, requires opset 13 Clip",
            inputs,
        )
    if (quant_min, quant_max) not in [(0, 255), (-128, 127)]:
        raise errors.SymbolicValueError(
            f"For (quant_min, quant_max), ONNX allows only (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    scale = symbolic_helper._maybe_get_scalar(scale)
    if scale is None:
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Non-constant scale not supported",
            inputs,
        )

    scale = scale.float().data  # Avoid exporter generating double type
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    return g.op(
        "DequantizeLinear",
        g.op("QuantizeLinear", inputs, scale, zero_point),
        scale,
        zero_point,
    )


@_onnx_symbolic("aten::isinf")
def isinf(g: jit_utils.GraphContext, input):
    # [FUNCTION DOC] Symbolic for aten::isinf. Casts the input to double and emits IsInf.
    #   Returns a boolean value.
    return g.op("IsInf", g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.DOUBLE))


@_onnx_symbolic("aten::isfinite")
def isfinite(g: jit_utils.GraphContext, input):
    # [FUNCTION DOC] Symbolic for aten::isfinite. Computed as not (isinf or isnan) using
    #   opset-10 isinf and opset9 helpers. Returns a boolean value.
    inf_node = isinf(g, input)
    nan_node = opset9.isnan(g, input)
    return opset9.__not_(g, opset9.__or_(g, inf_node, nan_node))


@_onnx_symbolic("aten::quantize_per_tensor")
def quantize_per_tensor(g: jit_utils.GraphContext, input, scale, zero_point, dtype):
    # [FUNCTION DOC] Symbolic for aten::quantize_per_tensor. Casts zero_point to the ONNX type
    #   of the given dtype and scale to float, then quantizes through the shared helper.
    #   Returns the quantized value.
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # TODO(justinchuby): Extract all the cast ops into a helper function.
    zero_point = g.op(
        "Cast", zero_point, to_i=_type_utils.JitScalarType(dtype).onnx_type()
    )
    scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return symbolic_helper.quantize_helper(g, input, scale, zero_point)


@_onnx_symbolic("aten::dequantize")
def dequantize(g: jit_utils.GraphContext, input):
    # [FUNCTION DOC] Symbolic for aten::dequantize. Delegates to the shared dequantize helper
    #   and returns only the dequantized tensor.
    return symbolic_helper.dequantize_helper(g, input)[0]


@_onnx_symbolic("aten::nan_to_num")
@symbolic_helper.parse_args("v", "f", "f", "f")
def nan_to_num(g: jit_utils.GraphContext, input, nan, posinf, neginf):
    # [FUNCTION DOC] Symbolic for aten::nan_to_num. Non-floating inputs are returned
    #   unchanged. Otherwise replaces nan with the nan value (default 0), then positive
    #   infinity with posinf (default dtype max) and negative infinity with neginf (default
    #   dtype min), each via IsInf/Less/Greater tests and Where. Returns the result value.
    # Cannot create an int type tensor with inf/nan values, so we simply
    # return the original tensor
    if not symbolic_helper._is_fp(input):
        return input
    input_dtype = _type_utils.JitScalarType.from_value(input).dtype()
    if nan is None:
        nan = 0.0
    nan_cond = opset9.isnan(g, input)
    nan_result = g.op(
        "Where",
        nan_cond,
        g.op("Constant", value_t=torch.tensor([nan], dtype=input_dtype)),
        input,
    )

    # For None values of posinf, neginf we use the greatest/lowest finite
    # value representable by input's dtype.
    finfo = torch.finfo(input_dtype)
    if posinf is None:
        posinf = finfo.max
    posinf_cond = opset9.logical_and(
        g,
        isinf(g, nan_result),
        opset9.gt(g, nan_result, g.op("Constant", value_t=torch.LongTensor([0]))),
    )
    nan_posinf_result = g.op(
        "Where",
        posinf_cond,
        g.op("Constant", value_t=torch.tensor([posinf], dtype=input_dtype)),
        nan_result,
    )

    if neginf is None:
        neginf = finfo.min
    neginf_cond = opset9.logical_and(
        g,
        isinf(g, nan_posinf_result),
        opset9.lt(
            g, nan_posinf_result, g.op("Constant", value_t=torch.LongTensor([0]))
        ),
    )
    return g.op(
        "Where",
        neginf_cond,
        g.op("Constant", value_t=torch.tensor([neginf], dtype=input_dtype)),
        nan_posinf_result,
    )


# Quantized symbolics ---------------------------------------------------------
# https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter#quantized-model-export
# Support starts from opset 10 because `DequantizeLinear` and `QuantizeLinear` were
# introduced in opset version 10.
@_onnx_symbolic("quantized::linear")
def quantized_linear(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    # [FUNCTION DOC] Quantized symbolic for quantized::linear. Dequantizes input and weight,
    #   requantizes the bias to the product of their scales and dequantizes it, applies opset9
    #   linear, then requantizes with op_scale and op_zero_point. Returns the quantized
    #   output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.linear(g, input, weight, bias)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::linear_relu")
def quantized_linear_relu(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    # [FUNCTION DOC] Quantized symbolic for quantized::linear_relu. Same as quantized linear
    #   but applies relu before requantizing. Returns the quantized output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.linear(g, input, weight, bias)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::add")
def quantized_add(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    # [FUNCTION DOC] Quantized symbolic for quantized::add. Dequantizes both operands, adds
    #   them via opset9, and requantizes with op_scale and op_zero_point. Returns the
    #   quantized sum.
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    output = opset9.add(g, x, y)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::add_relu")
def quantized_add_relu(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    # [FUNCTION DOC] Quantized symbolic for quantized::add_relu. Dequantizes both operands,
    #   adds, applies relu, and requantizes. Returns the quantized result.
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    output = opset9.add(g, x, y)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::mul")
def quantized_mul(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    # [FUNCTION DOC] Quantized symbolic for quantized::mul. Dequantizes both operands,
    #   multiplies via opset9, and requantizes. Returns the quantized product.
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    output = opset9.mul(g, x, y)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::hardswish")
def quantized_hardswish(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    # [FUNCTION DOC] Quantized symbolic for quantized::hardswish. Dequantizes x, applies
    #   hardswish, requantizes with op_scale and op_zero_point.
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.hardswish(g, x)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::sigmoid")
def quantized_sigmoid(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    # [FUNCTION DOC] Quantized symbolic for quantized::sigmoid. Dequantizes x, applies
    #   sigmoid, requantizes with op_scale and op_zero_point.
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.sigmoid(g, x)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::leaky_relu")
def quantized_leaky_relu(
    g: jit_utils.GraphContext, x, negative_slope, inplace, op_scale, op_zero_point
):
    # [FUNCTION DOC] Quantized symbolic for quantized::leaky_relu. Dequantizes x, applies
    #   leaky_relu with negative_slope and inplace, requantizes.
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.leaky_relu(g, x, negative_slope, inplace)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::layer_norm")
def quantized_layer_norm(
    g: jit_utils.GraphContext,
    x,
    normalized_shape,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::layer_norm. Dequantizes x, applies
    #   opset9 layer_norm (cudnn flag off) with the given normalized_shape, weight, bias and
    #   eps, then requantizes.
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.layer_norm(g, x, normalized_shape, weight, bias, eps, False)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::group_norm")
def quantized_group_norm(
    g: jit_utils.GraphContext,
    x,
    num_groups,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::group_norm. Dequantizes x, applies
    #   opset9 group_norm over num_groups (cudnn flag off), then requantizes.
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.group_norm(g, x, num_groups, weight, bias, eps, False)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::instance_norm")
@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v")
def quantized_instance_norm(
    g: jit_utils.GraphContext,
    q_input,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::instance_norm. Dequantizes q_input,
    #   applies opset9 instance_norm without running stats (training off, momentum 0), then
    #   requantizes.
    input, _, _, _ = symbolic_helper.dequantize_helper(g, q_input)

    output = opset9.instance_norm(
        g, input, weight, bias, None, None, False, 0.0, eps, False
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv1d_relu")
def quantized_conv1d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::conv1d_relu. Dequantizes input and
    #   weight, requantizes and dequantizes the bias, applies the opset9 convolution with
    #   stride, padding, dilation and groups, then relu, and requantizes with op_scale and
    #   op_zero_point. Returns the quantized output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv1d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv2d_relu")
def quantized_conv2d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::conv2d_relu. Dequantizes input and
    #   weight, requantizes and dequantizes the bias, applies the opset9 convolution with
    #   stride, padding, dilation and groups, then relu, and requantizes with op_scale and
    #   op_zero_point. Returns the quantized output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv3d_relu")
def quantized_conv3d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::conv3d_relu. Dequantizes input and
    #   weight, requantizes and dequantizes the bias, applies the opset9 convolution with
    #   stride, padding, dilation and groups, then relu, and requantizes with op_scale and
    #   op_zero_point. Returns the quantized output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv3d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv1d")
def quantized_conv1d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::conv1d. Dequantizes input and weight,
    #   requantizes and dequantizes the bias, applies the opset9 convolution with stride,
    #   padding, dilation and groups, and requantizes with op_scale and op_zero_point. Returns
    #   the quantized output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv1d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv2d")
def quantized_conv2d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::conv2d. Dequantizes input and weight,
    #   requantizes and dequantizes the bias, applies the opset9 convolution with stride,
    #   padding, dilation and groups, and requantizes with op_scale and op_zero_point. Returns
    #   the quantized output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv3d")
def quantized_conv3d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::conv3d. Dequantizes input and weight,
    #   requantizes and dequantizes the bias, applies the opset9 convolution with stride,
    #   padding, dilation and groups, and requantizes with op_scale and op_zero_point. Returns
    #   the quantized output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv3d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv_transpose1d")
def quantized_conv_transpose1d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::conv_transpose1d. Dequantizes input,
    #   weight and bias like the other quantized convs, applies a transposed convolution and
    #   requantizes. Returns the quantized output.
    # KNOWN ISSUE: uses the 2d transposed convolution implementation for the 1d op.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose2d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv_transpose2d")
def quantized_conv_transpose2d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::conv_transpose2d. Dequantizes input,
    #   weight and bias, applies opset9 conv_transpose2d, and requantizes. Returns the
    #   quantized output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose2d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv_transpose3d")
def quantized_conv_transpose3d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC] Quantized symbolic for quantized::conv_transpose3d. Dequantizes input,
    #   weight and bias, applies opset9 conv_transpose3d, and requantizes. Returns the
    #   quantized output.
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose3d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::cat")
@symbolic_helper.parse_args("v", "i", "v", "v")
def quantized_cat(
    g: jit_utils.GraphContext,
    q_inputs: _C.Value,
    dim: int,
    op_scale: _C.Value,
    op_zero_point: _C.Value,
) -> _C.Value:
    # [FUNCTION DOC] Quantized symbolic for quantized::cat. Unpacks the list of quantized
    #   inputs, dequantizes each, concatenates along dim, and requantizes with op_scale and
    #   op_zero_point. Returns the quantized result.
    unpacked_inputs = symbolic_helper._unpack_list(q_inputs)
    dequantized = [
        symbolic_helper.dequantize_helper(g, input)[0] for input in unpacked_inputs
    ]
    concatenated = g.op("Concat", *dequantized, axis_i=dim)
    return symbolic_helper.quantize_helper(g, concatenated, op_scale, op_zero_point)
