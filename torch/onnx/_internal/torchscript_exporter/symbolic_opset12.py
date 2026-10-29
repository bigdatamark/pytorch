# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
# [MODULE DOC] ONNX symbolic functions for opset 12 of the TorchScript exporter. Registers
#   (via the opset-12 registration partial) converters for einsum/outer/tensordot, dropout and
#   native_dropout, nll_loss variants, cross_entropy_loss, binary_cross_entropy_with_logits,
#   celu, argmax/argmin, pow, ge/le and unfold, using ops new in opset 12 (Einsum,
#   NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss, Celu, GreaterOrEqual, LessOrEqual,
#   Dropout with training flag). Import time: defines the registration partial; decorators add
#   to the global symbolic registry.
from __future__ import annotations

import functools
import sys

import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
    utils,
)


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# This file exports ONNX ops for opset 12

__all__ = [
    "argmax",
    "argmin",
    "binary_cross_entropy_with_logits",
    "celu",
    "cross_entropy_loss",
    "dropout",
    "einsum",
    "ge",
    "le",
    "native_dropout",
    "nll_loss",
    "nll_loss2d",
    "nll_loss_nd",
    "outer",
    "pow",
    "tensordot",
    "unfold",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=12)


def _einsum_helper(g: jit_utils.GraphContext, equation, tensors):
    # [FUNCTION DOC] HELPER: Emits an Einsum node over the tensors with the equation. Boolean
    #   inputs are cast to int64 first (ONNX Einsum lacks bool) and the result cast back to
    #   bool. Raises RuntimeError if the tensor list is empty.
    if not tensors:
        raise RuntimeError("Einsum inputs are empty.")
    # ONNX does not support bool for Einsum inputs.
    if symbolic_helper._is_bool(tensors[0]):
        tensors = [
            g.op("Cast", tensor, to_i=_C_onnx.TensorProtoDataType.INT64)
            for tensor in tensors
        ]
        return g.op(
            "Cast",
            g.op("Einsum", *tensors, equation_s=equation),
            to_i=_C_onnx.TensorProtoDataType.BOOL,
        )
    else:
        return g.op("Einsum", *tensors, equation_s=equation)


@_onnx_symbolic("aten::einsum")
@symbolic_helper.parse_args("s", "v", "is")
def einsum(g: jit_utils.GraphContext, equation, tensor_list, path=None):
    # [FUNCTION DOC] Symbolic for aten::einsum. Unpacks the tensor list and calls the einsum
    #   helper with the equation; the optional path is ignored. Returns the result.
    tensors = symbolic_helper._unpack_list(tensor_list)
    return _einsum_helper(g, equation, tensors)


@_onnx_symbolic("aten::outer")
@symbolic_helper.parse_args("v", "v")
def outer(g: jit_utils.GraphContext, input, other):
    # [FUNCTION DOC] Symbolic for aten::outer. Casts other to the type of input if they
    #   differ, then computes the outer product via einsum with equation i,j->ij.
    # make sure to cast other to self's type
    if _type_utils.JitScalarType.from_value(
        other, _type_utils.JitScalarType.UNDEFINED
    ) != _type_utils.JitScalarType.from_value(input):
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_value(input).onnx_type(),
        )
    return _einsum_helper(g, "i,j->ij", [input, other])


def _dropout_returns_masked_input_and_mask(
    g: jit_utils.GraphContext, input: torch._C.Value, p: float, train: bool
) -> tuple[torch._C.Value, torch._C.Value | None]:
    # [FUNCTION DOC] HELPER: Shared dropout lowering. Checks the training mode against the
    #   export mode (may warn or raise via the check helper). In eval mode returns (input,
    #   None). In training mode emits Dropout with ratio p and a true training flag and
    #   returns (output, mask).
    symbolic_helper.check_training_mode(train, "dropout")
    # In eval mode, dropout is non-op. That is, if the node's
    # train param is set to False, dropout just returns its inputs.
    if not train:
        return input, None
    p = g.op("Constant", value_t=torch.tensor(p))
    t = g.op("Constant", value_t=torch.tensor(train, dtype=torch.bool))
    r, mask = g.op("Dropout", input, p, t, outputs=2)
    return r, mask


@_onnx_symbolic("aten::dropout")
@symbolic_helper.parse_args("v", "f", "b")
def dropout(g: jit_utils.GraphContext, input, p, train):
    # [FUNCTION DOC] Symbolic for aten::dropout. Returns only the dropout output (the input
    #   itself when not training).
    masked, _ = _dropout_returns_masked_input_and_mask(g, input, p, train)
    return masked


@_onnx_symbolic("aten::native_dropout")
@symbolic_helper.parse_args("v", "f", "b")
def native_dropout(g: jit_utils.GraphContext, input, p, train):
    # [FUNCTION DOC] Symbolic for aten::native_dropout. Returns the pair (output, mask); mask
    #   is None when not training.
    return _dropout_returns_masked_input_and_mask(g, input, p, train)


@_onnx_symbolic("aten::nll_loss")
def nll_loss(g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index):
    # [FUNCTION DOC] Symbolic for aten::nll_loss. Maps the integer reduction code (0 none, 1
    #   mean, 2 sum) to a string and emits NegativeLogLikelihoodLoss with the explicit
    #   ignore_index, passing weight only if it is not none. Returns the loss.
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]

    # in onnx NegativeLogLikelihoodLoss specification, ignore_index is optional without default value.
    # therefore we need to set ignore_index attribute even if it is not specified (e.g. ignore_index=-100).
    ignore_index = symbolic_helper._maybe_get_const(ignore_index, "i")
    if weight.node().mustBeNone():
        nllloss = g.op(
            "NegativeLogLikelihoodLoss",
            self,
            target,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )
    else:
        nllloss = g.op(
            "NegativeLogLikelihoodLoss",
            self,
            target,
            weight,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )

    return nllloss


@_onnx_symbolic("aten::nll_loss2d")
def nll_loss2d(
    g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index
):
    # [FUNCTION DOC] Symbolic for aten::nll_loss2d. Forwards to nll_loss.
    return nll_loss(g, self, target, weight, reduction, ignore_index)


@_onnx_symbolic("aten::nll_loss_nd")
def nll_loss_nd(
    g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index
):
    # [FUNCTION DOC] Symbolic for aten::nll_loss_nd. Forwards to nll_loss.
    return nll_loss(g, self, target, weight, reduction, ignore_index)


@_onnx_symbolic("aten::cross_entropy_loss")
def cross_entropy_loss(
    g: jit_utils.GraphContext,
    self,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
):
    # [FUNCTION DOC] Symbolic for aten::cross_entropy_loss. Maps the reduction code to a
    #   string and emits SoftmaxCrossEntropyLoss with ignore_index, passing weight only if not
    #   none. Raises SymbolicValueError if label_smoothing is positive. Returns the loss.
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]

    label_smoothing = symbolic_helper._maybe_get_const(label_smoothing, "f")
    if label_smoothing is not None and label_smoothing > 0.0:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX does not support label_smoothing", self
        )

    # in onnx SoftmaxCrossEntropyLoss specification, ignore_index is optional without default value.
    # therefore we need to set ignore_index attribute even if it is not specified (e.g. ignore_index=-100).
    ignore_index = symbolic_helper._maybe_get_const(ignore_index, "i")
    if weight.node().mustBeNone():
        celoss = g.op(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )
    else:
        celoss = g.op(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            weight,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )

    return celoss


@_onnx_symbolic("aten::binary_cross_entropy_with_logits")
@symbolic_helper.parse_args("v", "v", "v", "v", "i")
def binary_cross_entropy_with_logits(
    g: jit_utils.GraphContext, input, target, weight, pos_weight, reduction
):
    # [FUNCTION DOC] Symbolic for aten::binary_cross_entropy_with_logits. Builds the loss from
    #   sigmoid and log nodes: negative of target times log sigmoid (times pos_weight if
    #   given) plus one minus target times log of one minus sigmoid, multiplied by weight if
    #   given. Reduction none returns elementwise, mean uses ReduceMean, sum uses ReduceSum;
    #   other values are reported unsupported.
    # KNOWN ISSUE: the constant one is created as an int64 tensor and subtracted from float
    #   values, which may yield mismatched operand types in the exported graph.
    p = g.op("Constant", value_t=torch.tensor([1]))
    sig_x = opset9.sigmoid(g, input)
    log_sig_x = opset9.log(g, sig_x)
    sub_1_x = opset9.sub(g, p, sig_x)
    sub_1_y = opset9.sub(g, p, target)
    log_1_x = opset9.log(g, sub_1_x)
    if pos_weight is None or symbolic_helper._is_none(pos_weight):
        output = opset9.neg(
            g,
            opset9.add(
                g, opset9.mul(g, target, log_sig_x), opset9.mul(g, sub_1_y, log_1_x)
            ),
        )
    else:
        output = opset9.neg(
            g,
            opset9.add(
                g,
                opset9.mul(g, opset9.mul(g, target, log_sig_x), pos_weight),
                opset9.mul(g, sub_1_y, log_1_x),
            ),
        )

    if weight is not None and not symbolic_helper._is_none(weight):
        output = opset9.mul(g, weight, output)

    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    if reduction == 0:
        return output
    elif reduction == 1:
        return g.op("ReduceMean", output, keepdims_i=0)
    elif reduction == 2:
        return g.op("ReduceSum", output, keepdims_i=0)
    else:
        return symbolic_helper._onnx_unsupported(
            "binary_cross_entropy_with_logits with reduction other than none, mean, or sum",
            input,
        )


@_onnx_symbolic("aten::celu")
def celu(g: jit_utils.GraphContext, self, alpha):
    # [FUNCTION DOC] Symbolic for aten::celu. Emits Celu with the constant alpha; double
    #   inputs are cast to float and the result cast back to double.
    alpha = symbolic_helper._maybe_get_const(alpha, "f")
    # if the input is of type double cast it to float
    if (
        _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.DOUBLE
    ):
        self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        out = g.op("Celu", self, alpha_f=alpha)
        return g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.DOUBLE)

    return g.op("Celu", self, alpha_f=alpha)


@_onnx_symbolic("aten::argmax")
@symbolic_helper.parse_args("v", "v", "b")
def argmax(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    # [FUNCTION DOC] Symbolic for aten::argmax. Delegates to the shared argmin/argmax helper
    #   with the ArgMax operator.
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMax")


@_onnx_symbolic("aten::argmin")
@symbolic_helper.parse_args("v", "v", "b")
def argmin(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    # [FUNCTION DOC] Symbolic for aten::argmin. Delegates to the shared argmin/argmax helper
    #   with the ArgMin operator.
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMin")


@_onnx_symbolic("aten::pow")
def pow(g: jit_utils.GraphContext, self, exponent):
    # [FUNCTION DOC] Symbolic for aten::pow. Emits the Pow node (opset 12 supports mixed
    #   integer and float types).
    return g.op("Pow", self, exponent)


@_onnx_symbolic("aten::ge")
def ge(g: jit_utils.GraphContext, input, other):
    # [FUNCTION DOC] Symbolic for aten::ge. Emits GreaterOrEqual.
    return g.op("GreaterOrEqual", input, other)


@_onnx_symbolic("aten::le")
def le(g: jit_utils.GraphContext, input, other):
    # [FUNCTION DOC] Symbolic for aten::le. Emits LessOrEqual.
    return g.op("LessOrEqual", input, other)


@_onnx_symbolic("aten::unfold")
@symbolic_helper.parse_args("v", "i", "v", "v")
def unfold(g: jit_utils.GraphContext, input, dimension, size, step):
    # [FUNCTION DOC] Symbolic for aten::unfold. Constant size and step defer to opset9. With
    #   dynamic size or step but a known dimension size, builds a Loop over window start/end
    #   indices (from Range nodes), slices each window, transposes and unsqueezes it,
    #   concatenates the results and permutes/squeezes the loop output. Reports unimplemented
    #   if the dimension size is unknown. Raises AssertionError if the input rank is unknown.
    # KNOWN ISSUE: the loop body slices along a hardcoded axis 2 and the output permutation
    #   assumes exactly 5 dims, so it only works for 4-D inputs unfolded in a limited way; the
    #   loop-carried condition input is unused.
    const_size = symbolic_helper._maybe_get_const(size, "i")
    const_step = symbolic_helper._maybe_get_const(step, "i")
    if not symbolic_helper._is_value(const_size) and not symbolic_helper._is_value(
        const_step
    ):
        return opset9.unfold(g, input, dimension, const_size, const_step)

    sizedim = symbolic_helper._get_tensor_dim_size(input, dimension)
    if sizedim is not None:
        low_start = g.op("Constant", value_t=torch.tensor(0))
        low_end = g.op("Constant", value_t=torch.tensor(sizedim))
        hi_end = g.op("Constant", value_t=torch.tensor(sizedim + 1))
        low_indices = g.op("Range", low_start, low_end, step)
        hi_indices = g.op("Range", size, hi_end, step)

        low_size = symbolic_helper._size_helper(
            g, low_indices, g.op("Constant", value_t=torch.tensor(0))
        )
        hi_size = symbolic_helper._size_helper(
            g, hi_indices, g.op("Constant", value_t=torch.tensor(0))
        )

        ndim = symbolic_helper._get_tensor_rank(input)
        if ndim is None:
            raise AssertionError("ndim must be non-None")
        perm = list(range(ndim))
        perm.append(perm.pop(dimension))

        unsqueeze_list = []
        loop_condition = g.op("Constant", value_t=torch.tensor(1))
        loop_condition = g.op(
            "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
        )
        loop_len = g.op("Min", low_size, hi_size)

        loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
            g, "Loop", loop_len, loop_condition, n_blocks=1
        )

        loop_block = loop_context.block
        block_input_iter = utils._add_input_to_block(loop_block)
        cond = utils._add_input_to_block(loop_block)  # noqa: F841

        starts = loop_context.op("Gather", low_indices, block_input_iter)
        ends = loop_context.op("Gather", hi_indices, block_input_iter)
        axes = loop_context.op("Constant", value_t=torch.tensor([2]))
        starts = symbolic_helper._unsqueeze_helper(loop_context, starts, [0])
        ends = symbolic_helper._unsqueeze_helper(loop_context, ends, [0])
        stack = loop_context.op("Slice", input, starts, ends, axes)

        unsqueeze = symbolic_helper._unsqueeze_helper(
            loop_context, loop_context.op("Transpose", stack, perm_i=perm), [dimension]
        )
        unsqueeze_list.append(unsqueeze)
        concat = loop_context.op("Concat", *unsqueeze_list, axis_i=0)

        cond_out = loop_context.op(
            "Cast",
            loop_condition,
            # pyrefly: ignore [bad-argument-type]
            _C_onnx.TensorProtoDataType.BOOL,
        )
        utils._add_output_to_block(loop_block, cond_out)
        utils._add_output_to_block(loop_block, concat)

        loop_output = loop.node().output()
        perm = [0, 1, 2, 3, 4]
        perm[0], perm[dimension + 1] = perm[dimension + 1], perm[0]
        transpose = g.op("Transpose", loop_output, perm_i=perm)
        squeeze = symbolic_helper._squeeze_helper(g, transpose, [0])

        return squeeze

    return symbolic_helper._unimplemented("Unfold", "input size not accessible")


@_onnx_symbolic("aten::tensordot")
@symbolic_helper.parse_args("v", "v", "is", "is", "v")
def tensordot(g: jit_utils.GraphContext, input_a, input_b, dims_a, dims_b, out=None):
    # [FUNCTION DOC] Symbolic for aten::tensordot. Normalizes negative dims, permutes input_a
    #   to put free dims first and contracted dims last (input_b the reverse), reshapes both
    #   to 2-D matrices, multiplies them with einsum ij,jk->ik and reshapes to the
    #   concatenation of the free dim sizes. Raises SymbolicValueError for unknown input
    #   ranks; out is reported unsupported.
    # KNOWN ISSUE: the out check calls the unimplemented helper without returning; the first
    #   reshape of output_a is recomputed from new_input_a and its result discarded.
    if out is not None:
        symbolic_helper._unimplemented(
            "Tensordot", "Out parameter is not supported for tensordot."
        )

    dim_count_a = symbolic_helper._get_tensor_rank(input_a)
    if dim_count_a is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of tensordot for tensor(input_a) of unknown rank.",
            input_a,
        )

    dim_count_b = symbolic_helper._get_tensor_rank(input_b)
    if dim_count_b is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of tensordot for tensor(input_b) of unknown rank.",
            input_b,
        )

    dims_a = [
        (dims_a[i] + dim_count_a) if (dims_a[i] < 0) else dims_a[i]
        for i in range(len(dims_a))
    ]
    dims_b = [
        (dims_b[i] + dim_count_b) if (dims_b[i] < 0) else dims_b[i]
        for i in range(len(dims_b))
    ]

    left_dims_a = [i for i in range(dim_count_a) if (i not in dims_a)]
    left_dims_b = [i for i in range(dim_count_b) if (i not in dims_b)]

    new_input_a = opset9.permute(g, input_a, left_dims_a + dims_a)
    new_input_b = opset9.permute(g, input_b, dims_b + left_dims_b)

    input_shape = g.op("Shape", new_input_a)
    left_sizes_a = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[0], ends=[len(left_dims_a)]
    )
    shape_sizes = [
        left_sizes_a,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
    ]
    output_a = opset9._reshape_from_tensor(g, new_input_a, shape_sizes)

    input_shape = g.op("Shape", output_a)
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[-1], ends=[sys.maxsize]
    )
    shape_sizes = [
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
        slices,
    ]
    output_a = opset9._reshape_from_tensor(g, new_input_a, shape_sizes)

    input_shape = g.op("Shape", new_input_b)
    left_sizes_b = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[len(dims_b)], ends=[sys.maxsize]
    )
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[0], ends=[len(dims_b)]
    )
    shape_sizes = [
        slices,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
    ]
    output_b = opset9._reshape_from_tensor(g, new_input_b, shape_sizes)

    input_shape = g.op("Shape", output_b)
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[-1], ends=[sys.maxsize]
    )
    shape_sizes = [
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
        slices,
    ]
    output_b = opset9._reshape_from_tensor(g, new_input_b, shape_sizes)

    output = einsum(g, "ij,jk->ik", g.op("prim::ListConstruct", *[output_a, output_b]))

    shape_sizes = [left_sizes_a, left_sizes_b]
    return opset9._reshape_from_tensor(g, output, shape_sizes)
