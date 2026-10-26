# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# [MODULE DOC] Inductor FX passes for quantization fusion on CPU/oneDNN: registers
# pattern-matcher lowerings that turn matched quantized qconv/qlinear (unary and
# binary post-op) ops, quantized max_pool2d, cat and reshape patterns into Inductor
# IR, plus weight-only-quantization (WOQ) int8 mm and concat-linear patterns, an
# int4 concat-linear graph pass, and a quant lift-up pass. Also defines pattern
# builders and validity checks used by those registrations. Import-time side
# effects: creates aten/prims/quantized op aliases and module-level pattern objects;
# lowering registration happens only when the register functions are called.
import copy
import itertools
import math
import operator

import torch
from torch._dynamo.utils import counters
from torch.fx.node import map_arg

from .. import config
from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import (
    Arg,
    CallFunction,
    filter_nodes,
    KeywordArg,
    ListOf,
    Match,
    stable_topological_sort,
)
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern


aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
quantized = torch.ops.quantized

# Only for per tensor quant since permute may change the channel idx
_PER_TENSOR_QUANTIZE_OPS = [
    quantized_decomposed.quantize_per_tensor.default,
    quantized_decomposed.quantize_per_tensor.tensor,
]

_VIEW_OPS = [
    aten.transpose.int,
    aten.permute.default,
    aten.view.default,
    aten.reshape.default,
]

"""
The quantization.py file primarily incorporates passes related to quantization fusion
in inductor, includes:
1. Lowering of qconv/qlinear (unary and binary post-op variants) from the
   `onednn::qconv_pointwise` / `onednn::qlinear_pointwise` ops to Inductor IR.
2. Pointwise operators' quantization fusion like: qmaxpool2d, qcat, qreshape and more.
3. Weight-only quantization (WOQ) int8 mm / concat-linear patterns lowered to
   `aten._weight_int8pack_mm` and the int4 concat-linear pass for
   `aten._weight_int4pack_mm_for_cpu`.

Note: the dequant promotion, qconv/qlinear weight prepack and qconv/qlinear
fusion patterns that used to live here have been moved to TorchAO.
"""


def _get_pattern_output_dtype(match: Match):
    # [FUNCTION DOC] HELPER: Return the dtype of the single output node of a match,
    # read from the node's fake value. Raises AssertionError if the match does not
    # have exactly one output node, the node is not an FX node, or the dtype is not
    # one of int8, uint8, float32, bfloat16, float8_e4m3fn.
    """
    Get the pattern's output dtype from node's meta
    Assume only 1 output node in this matched pattern.
    """
    pattern_output_nodes = match.output_nodes()
    if len(pattern_output_nodes) != 1:
        raise AssertionError(
            f"expected exactly 1 output node, got {len(pattern_output_nodes)}"
        )
    output_node = pattern_output_nodes[0]
    if not isinstance(output_node, torch.fx.Node):
        raise AssertionError(f"expected torch.fx.Node, got {type(output_node)}")
    output_dtype = output_node.meta["val"].dtype
    if output_dtype not in [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ]:
        raise AssertionError(f"unexpected output dtype {output_dtype}")
    return output_dtype


def _may_generate_pattern_with_dtype_convert(
    pattern, dtype=Arg(), with_dtype_convert=True, users=1
):
    # [FUNCTION DOC] HELPER: Wrap a pattern in a convert_element_type call pattern
    # (with the given dtype pattern and user count) when with_dtype_convert is true;
    # otherwise return the pattern unchanged.
    if with_dtype_convert:
        return CallFunction(
            prims.convert_element_type.default,
            pattern,
            dtype,
            _users=users,
        )
    else:
        return pattern


def _may_generate_pattern_with_reshape(pattern, reshape_size=Arg(), with_reshape=True):
    # [FUNCTION DOC] HELPER: Wrap a pattern in a reshape call pattern with the given
    # size pattern when with_reshape is true; otherwise return it unchanged.
    if with_reshape:
        return CallFunction(
            torch.ops.aten.reshape.default,
            pattern,
            reshape_size,
        )
    else:
        return pattern


def _generate_linear_t_pattern(
    _dequant_per_channel_pattern,
    dtype,
):
    # [FUNCTION DOC] HELPER: Build the transposed-weight pattern: a permute of the
    # dequantized per-channel weight pattern, with an optional dtype convert
    # (keyword autocast_wgt_dtype) only when dtype is bfloat16. Raises
    # AssertionError if dtype is not float32 or bfloat16.
    if dtype not in [torch.float32, torch.bfloat16]:
        raise AssertionError(f"expected float32 or bfloat16, got {dtype}")
    t_pattern = CallFunction(
        aten.permute.default,
        _may_generate_pattern_with_dtype_convert(
            _dequant_per_channel_pattern,
            KeywordArg("autocast_wgt_dtype"),
            dtype == torch.bfloat16,
        ),
        KeywordArg("permute_axes"),
    )
    return t_pattern


def _unary_fusion_pattern(unary_fusion, call_fn, users, is_bf16):
    # [FUNCTION DOC] HELPER: Build a unary-fusion pattern: wrap call_fn with an
    # optional dtype convert (inserted only when is_bf16, keyword to_float, given
    # user count) and apply the unary_fusion pattern builder to it.
    # only insert to_dtype if is_bf16 is True
    computation_call = _may_generate_pattern_with_dtype_convert(
        call_fn, dtype=KeywordArg("to_float"), with_dtype_convert=is_bf16, users=users
    )
    return unary_fusion(computation_call)


def get_dequantize_per_tensor_activation_pattern(is_tensor_overload=False):
    # [FUNCTION DOC] Return a pattern matching dequantize_per_tensor on an
    # activation x with scale, zero point, quant min/max and dtype keywords; uses
    # the tensor-scale overload when is_tensor_overload is true.
    dequantize_per_tensor_activation_pattern = CallFunction(
        quantized_decomposed.dequantize_per_tensor.tensor
        if is_tensor_overload
        else quantized_decomposed.dequantize_per_tensor.default,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("x_quant_min"),
        KeywordArg("x_quant_max"),
        KeywordArg("x_dq_dtype"),
    )
    return dequantize_per_tensor_activation_pattern


dequantize_per_channel_weight_pattern = CallFunction(
    quantized_decomposed.dequantize_per_channel.default,
    KeywordArg("q_weight"),
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_axis"),
    KeywordArg("w_quant_min"),
    KeywordArg("w_quant_max"),
    KeywordArg("w_dtype"),
)

dequantize_per_channel_to_bf16_weight_pattern = (
    _may_generate_pattern_with_dtype_convert(
        dequantize_per_channel_weight_pattern,
        KeywordArg("autocast_wgt_dtype"),
    )
)

dequantize_per_channel_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)

dequantize_per_channel_to_bf16_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_to_bf16_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)


def get_qconv_pt2e_pattern(x_scale_zp_are_tensors=False, users=1):
    # [FUNCTION DOC] Return a pattern matching a oneDNN qconv_pointwise op (tensor
    # overload when x_scale_zp_are_tensors) with all keyword arguments for
    # activation and weight qparams, conv params, output qparams and post-op,
    # constrained to the given number of users.
    qconv_op = (
        torch.ops.onednn.qconv_pointwise.tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qconv_pointwise.default
    )
    return CallFunction(
        qconv_op,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("b"),
        KeywordArg("stride"),
        KeywordArg("padding"),
        KeywordArg("dilation"),
        KeywordArg("groups"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("postop_name"),
        KeywordArg("postop_args"),
        KeywordArg("postop_algorithm"),
        _users=users,
    )


def get_qconv2d_binary_pt2e_pattern(x_scale_zp_are_tensors=False, users=1):
    # [FUNCTION DOC] Return a pattern matching a oneDNN qconv2d_pointwise binary op
    # (tensor overload when x_scale_zp_are_tensors) with keywords for qparams, the
    # accumulation input, conv params, binary and unary post-op settings,
    # constrained to the given number of users.
    qconv_op = (
        torch.ops.onednn.qconv2d_pointwise.binary_tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qconv2d_pointwise.binary
    )
    return CallFunction(
        qconv_op,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("accum"),
        KeywordArg("b"),
        KeywordArg("stride"),
        KeywordArg("padding"),
        KeywordArg("dilation"),
        KeywordArg("groups"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("accum_scale"),
        KeywordArg("accum_zero_point"),
        KeywordArg("binary_op_name"),
        KeywordArg("alpha"),
        KeywordArg("unary_op_name"),
        KeywordArg("unary_op_args"),
        KeywordArg("unary_op_algorithm"),
        _users=users,
    )


def get_qlinear_pt2e_pattern(x_scale_zp_are_tensors, users=1):
    # [FUNCTION DOC] Return a pattern matching a oneDNN qlinear_pointwise op (tensor
    # overload when x_scale_zp_are_tensors) with keywords for activation, weight and
    # output qparams, bias and post-op settings, constrained to the given number of
    # users.
    qlinear_op = (
        torch.ops.onednn.qlinear_pointwise.tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qlinear_pointwise.default
    )
    return CallFunction(
        qlinear_op,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("b"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("postop_name"),
        KeywordArg("postop_args"),
        KeywordArg("postop_algorithm"),
        _users=users,
    )


def get_qlinear_binary_pt2e_pattern(x_scale_zp_are_tensors, users=1):
    # [FUNCTION DOC] Return a pattern matching a oneDNN qlinear_pointwise binary op
    # (tensor overload when x_scale_zp_are_tensors) with keywords for qparams, the
    # second input x_2 and its qparams, binary and unary post-op settings,
    # constrained to the given number of users.
    qlinear_op = (
        torch.ops.onednn.qlinear_pointwise.binary_tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qlinear_pointwise.binary
    )
    return CallFunction(
        qlinear_op,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("x_2"),
        KeywordArg("b"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("x2_scale"),
        KeywordArg("x2_zp"),
        KeywordArg("binary_op_name"),
        KeywordArg("alpha"),
        KeywordArg("unary_op_name"),
        KeywordArg("unary_op_args"),
        KeywordArg("unary_op_algorithm"),
        _users=users,
    )


dequantize_accum_pattern = CallFunction(
    quantized_decomposed.dequantize_per_tensor.default,
    KeywordArg("accum"),
    KeywordArg("accum_scale"),
    KeywordArg("accum_zp"),
    Arg(),
    Arg(),
    KeywordArg("accum_dq_dtype"),
)


def generate_pattern_with_binary(
    binary_post_op,
    computation_call,
    extra_input_pattern,
    dtype_convert=False,
    swap_inputs=False,
):
    # [FUNCTION DOC] Build a binary post-op pattern over the computation call and an
    # extra input pattern, with operand order swapped when swap_inputs is set,
    # optionally followed by a dtype convert (keyword
    # convert_dtype_after_inplace_add) when dtype_convert is true.
    binary_pattern = (
        CallFunction(
            binary_post_op,
            extra_input_pattern,
            computation_call,
        )
        if swap_inputs
        else CallFunction(
            binary_post_op,
            computation_call,
            extra_input_pattern,
        )
    )
    return _may_generate_pattern_with_dtype_convert(
        binary_pattern,
        KeywordArg("convert_dtype_after_inplace_add"),
        dtype_convert,
    )


def generate_pattern_with_unary(computation_call, unary_post_op):
    # [FUNCTION DOC] Apply a unary post-op to the computation call pattern, or
    # return the computation call pattern unchanged when unary_post_op is None.
    if unary_post_op is not None:
        return CallFunction(
            unary_post_op,
            computation_call,
        )
    return computation_call


def generate_pattern_with_output_quant(computation_call, with_dtype_convert=False):
    # [FUNCTION DOC] Build a pattern for a quantize_per_tensor applied to the
    # computation call (optionally through a dtype convert), with keywords
    # o_inv_scale, o_zp, o_qmin, o_qmax and o_dtype.
    quantized_op_output_pattern_pt2e = CallFunction(
        quantized_decomposed.quantize_per_tensor.default,
        _may_generate_pattern_with_dtype_convert(
            computation_call,
            Arg(),
            with_dtype_convert,
        ),
        KeywordArg("o_inv_scale"),
        KeywordArg("o_zp"),
        KeywordArg("o_qmin"),
        KeywordArg("o_qmax"),
        KeywordArg("o_dtype"),
    )
    return quantized_op_output_pattern_pt2e


def _check_node_kwarg_arg_value(check_node, kwarg_name, args_index, expected_value):
    # [FUNCTION DOC] HELPER: Compare a node's argument against an expected value,
    # looking it up by keyword name if present in kwargs, otherwise by positional
    # index. Returns the equality result. Raises AssertionError if the positional
    # argument is missing.
    if kwarg_name in check_node.kwargs:
        actual_value = check_node.kwargs[kwarg_name]
        return actual_value == expected_value
    else:
        if len(check_node.args) < (args_index + 1):
            raise AssertionError(
                f"expected at least {args_index + 1} args, got {len(check_node.args)}"
            )
        actual_value = check_node.args[args_index]
        return actual_value == expected_value


def _is_valid_quantized_conv_optimization_pattern():
    # [FUNCTION DOC] HELPER: Return a match filter for quantized conv patterns. For
    # float32 or bfloat16 outputs it requires the qconv node's output_dtype argument
    # (positional index 13) to equal the pattern output dtype; for other dtypes it
    # accepts.
    def fn(match):
        # [FUNCTION DOC] HELPER: Match filter: see enclosing function. Returns True
        # or False for a match.
        output_dtype = _get_pattern_output_dtype(match)
        if output_dtype in [torch.float32, torch.bfloat16]:
            # Only keep matched pattern with same output_dtype
            qconv_node_after_weight_prepack = filter_nodes(
                match.nodes, torch.ops.onednn.qconv_pointwise
            )[0]
            return _check_node_kwarg_arg_value(
                qconv_node_after_weight_prepack, "output_dtype", 13, output_dtype
            )
        return True

    return fn


def _is_valid_qconv_post_op_fusion_pattern(has_binary_post_op=False):
    # [FUNCTION DOC] HELPER: Select the match filter for qconv post-op fusion: the
    # binary variant if has_binary_post_op, else the unary variant.
    return (
        _is_valid_qconv_binary_optimization_pattern()
        if has_binary_post_op
        else _is_valid_quantized_conv_optimization_pattern()
    )


def _is_valid_qconv_lowering_pattern():
    # [FUNCTION DOC] HELPER: Return a match filter accepting only single-node
    # matches whose target is a qconv_pointwise or qconv2d_pointwise binary op
    # overload.
    def fn(match):
        # [FUNCTION DOC] HELPER: Match filter: True only for a one-node match with a
        # qconv target.
        if len(match.nodes) != 1:
            return False
        return match.nodes[0].target in (
            torch.ops.onednn.qconv_pointwise.default,
            torch.ops.onednn.qconv_pointwise.tensor,
            torch.ops.onednn.qconv2d_pointwise.binary,
            torch.ops.onednn.qconv2d_pointwise.binary_tensor,
        )

    return fn


def _register_quantized_conv_lowering(
    pattern,
    pass_number,
    computation_op,
):
    # [FUNCTION DOC] HELPER: Register a lowering for a qconv pattern at the given
    # pass number, which replaces the matched qconv node with the lowering of
    # computation_op applied to the matched arguments. Registers a lowering pattern
    # (via register_lowering_pattern) and returns the inner lowering function.
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qconv_lowering_pattern(),
        pass_number=pass_number,
    )
    def qconv(match: Match, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Lowering callback: gathers activation, weight, conv
        # and post-op arguments from kwargs, validates the pattern output dtype,
        # bumps qconv_unary_lower counters and returns the Inductor lowering of
        # computation_op.
        # KNOWN ISSUE: output_dtype is first computed from the match and validated,
        # then overwritten with the kwarg output_dtype, so the validated value is
        # not the one passed on.
        # Activation QParams
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        # Weight QParams
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        # Conv Params
        b, stride, padding, dilation, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["groups"],
        )
        output_dtype = _get_pattern_output_dtype(match)
        if output_dtype not in [
            torch.int8,
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float32,
            torch.bfloat16,
        ]:
            raise AssertionError(f"unexpected output dtype {output_dtype}")
        # Output QParams
        o_inv_scale = kwargs["output_scale"]
        o_zero_point = kwargs["output_zero_point"]
        output_dtype = kwargs["output_dtype"]
        # post op
        postop_name = kwargs["postop_name"]
        postop_args = kwargs["postop_args"]
        postop_algorithm = kwargs["postop_algorithm"]

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            postop_name,
            postop_args,
            postop_algorithm,
        )
        counters["inductor"]["qconv_unary_lower_count"] += 1
        counters["inductor"]["qconv_unary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qconv


def _is_valid_quantized_linear_optimization_pattern():
    # [FUNCTION DOC] HELPER: Return a match filter for quantized linear patterns.
    # For float32 or bfloat16 outputs it requires the qlinear node's output_dtype
    # argument (positional index 9) to equal the pattern output dtype; otherwise
    # accepts.
    def fn(match):
        # [FUNCTION DOC] HELPER: Match filter: see enclosing function.
        output_dtype = _get_pattern_output_dtype(match)
        if output_dtype in [torch.float32, torch.bfloat16]:
            # Only keep matched pattern with same output_dtype
            qlinear_node_after_weight_prepack = filter_nodes(
                match.nodes, torch.ops.onednn.qlinear_pointwise
            )[0]
            return _check_node_kwarg_arg_value(
                qlinear_node_after_weight_prepack, "output_dtype", 9, output_dtype
            )
        return True

    return fn


def _is_valid_qlinear_post_op_fusion_pattern(has_binary_post_op=False):
    # [FUNCTION DOC] HELPER: Select the match filter for qlinear post-op fusion: the
    # binary variant if has_binary_post_op, else the unary variant.
    return (
        _is_valid_qlinear_binary_optimization_pattern()
        if has_binary_post_op
        else _is_valid_quantized_linear_optimization_pattern()
    )


def _is_valid_qlinear_lowering_pattern():
    # [FUNCTION DOC] HELPER: Return a match filter accepting only single-node
    # matches whose target is a qlinear_pointwise op overload (default, tensor,
    # binary or binary_tensor).
    def fn(match):
        # [FUNCTION DOC] HELPER: Match filter: True only for a one-node match with a
        # qlinear target.
        if len(match.nodes) != 1:
            return False
        return match.nodes[0].target in (
            torch.ops.onednn.qlinear_pointwise.default,
            torch.ops.onednn.qlinear_pointwise.tensor,
            torch.ops.onednn.qlinear_pointwise.binary,
            torch.ops.onednn.qlinear_pointwise.binary_tensor,
        )

    return fn


def _register_quantized_linear_unary_lowering(
    pattern,
    pass_number,
    computation_op,
):
    # [FUNCTION DOC] HELPER: Register a lowering for a unary-post-op qlinear pattern
    # at the given pass number using computation_op. Registers a lowering pattern
    # (via register_lowering_pattern) and returns the inner lowering function.
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qlinear_lowering_pattern(),
        pass_number=pass_number,
    )
    def qlinear(match: Match, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Lowering callback: gathers activation, weight,
        # optional bias, output qparams and post-op kwargs, bumps
        # qlinear_unary_lower counters and returns the lowering of computation_op.
        output_dtype = _get_pattern_output_dtype(match)
        # Activation QParams
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        # Weight QParams
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )

        # bias
        b = kwargs.get("b")

        # Output QParams
        o_inv_scale = kwargs["output_scale"]
        o_zero_point = kwargs["output_zero_point"]

        # post op
        postop_name = kwargs["postop_name"]
        postop_args = kwargs["postop_args"]
        postop_algorithm = kwargs["postop_algorithm"]

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            postop_name,
            postop_args,
            postop_algorithm,
        )
        counters["inductor"]["qlinear_unary_lower_count"] += 1
        counters["inductor"]["qlinear_unary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qlinear


def _register_quantized_linear_binary_lowering(
    pattern,
    pass_number,
    computation_op,
):
    # [FUNCTION DOC] HELPER: Register a lowering for a binary-post-op qlinear
    # pattern at the given pass number using computation_op. Registers a lowering
    # pattern (via register_lowering_pattern) and returns the inner lowering
    # function.
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qlinear_lowering_pattern(),
        pass_number=pass_number,
    )
    def qlinear_binary(match: Match, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Lowering callback: gathers qparams, the second
        # input x2 and its qparams, bias and post-op kwargs; realizes x2; changes a
        # sum binary op to add when x2 cannot be updated in place; bumps
        # qlinear_binary_lower counters and returns the lowering of computation_op.
        # Raises AssertionError if the output dtype is None.
        output_dtype = _get_pattern_output_dtype(match)
        if output_dtype is None:
            raise AssertionError("expected non-None output dtype")
        # Activation QParams
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        x2 = kwargs["x_2"]
        x2_scale = kwargs["x2_scale"]
        x2_zp = kwargs["x2_zp"]
        # Weight QParams
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        # bias
        b = kwargs.get("b")
        # Output QParams
        o_inv_scale = kwargs["output_scale"]
        o_zero_point = kwargs["output_zero_point"]

        x2.realize()
        from .mkldnn_fusion import _qlinear_binary_can_be_inplace

        binary_op_name = kwargs["binary_op_name"]
        alpha = kwargs["alpha"]
        unary_op_name = kwargs["unary_op_name"]
        unary_op_args = kwargs["unary_op_args"]
        unary_op_algorithm = kwargs["unary_op_algorithm"]
        if (
            # TODO Ensure sum is safe and remove such check, i.e.,
            # x2 is not used by other operations
            # or current qlinear sum is the last user of x2.
            # This needs to be ensured when registering
            # the lowering pattern of quantized_linear_binary.
            binary_op_name == "sum" and (not _qlinear_binary_can_be_inplace(x2))
        ):
            binary_op_name = "add"

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            x2,
            b,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            x2_scale,
            x2_zp,
            binary_op_name,
            alpha,
            unary_op_name,
            unary_op_args,
            unary_op_algorithm,
        )
        counters["inductor"]["qlinear_binary_lower_count"] += 1
        counters["inductor"]["qlinear_binary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qlinear_binary


def _is_valid_qconv_binary_optimization_pattern():
    # [FUNCTION DOC] HELPER: Return the binary-pattern validity filter for qconv,
    # built with the generic helper for the qconv_pointwise op.
    return _is_valid_quantized_op_binary_optimization_pattern(
        torch.ops.onednn.qconv_pointwise
    )


def _is_valid_qlinear_binary_optimization_pattern():
    # [FUNCTION DOC] HELPER: Return the binary-pattern validity filter for qlinear
    # (qlinear_pointwise op); the extra input is not required to come from a dequant
    # because no q-dq is inserted for it due to accuracy concerns.
    return _is_valid_quantized_op_binary_optimization_pattern(
        torch.ops.onednn.qlinear_pointwise,
        # we don't insert q-dq for extra input due to accuracy issues
        extra_input_from_dequant=False,
    )


def _is_valid_quantized_op_binary_optimization_pattern(
    qop, extra_input_from_dequant=True
):
    # [FUNCTION DOC] HELPER: Return a match filter validating a binary-fusion
    # pattern for qop: the op must have exactly one user (the binary node with 2
    # inputs, else AssertionError); for float outputs the extra input must come from
    # a dequantize_per_tensor when extra_input_from_dequant; both binary inputs must
    # carry tensor values of the same size; and the extra input of the pattern must
    # not have other remaining users beyond the compute node and must not be the
    # compute node's first argument.
    # Check if it's a valid Binary Pattern for qconv2d and qlinear:
    # * qop_pointwise should only has one users
    # * If extra_input_from_dequant is True, extra input of binary node should come from dequant pattern
    # * the two inputs of binary node should have attribute "meta" and should be tensors
    # * the two inputs of binary node should have the same shape
    # * All users of the extra input in this pattern should be
    #   ancestor nodes of the compute node, except for the binary node
    #   connected to the compute node.
    def fn(match):
        # [FUNCTION DOC] HELPER: Match filter implementing the checks described in
        # the enclosing function; returns True if the pattern may be fused.
        output_dtype = _get_pattern_output_dtype(match)
        compute_node = filter_nodes(match.nodes, qop)[0]
        # qop_pointwise should only have one user
        if len(compute_node.users) != 1:
            return False
        binary_node_inputs = next(iter(compute_node.users)).args
        if len(binary_node_inputs) != 2:
            raise AssertionError("Expects binary node with 2 inputs")
        if output_dtype in [torch.float32, torch.bfloat16]:
            extra_input_of_binary_node = None
            for arg in binary_node_inputs:
                if arg != compute_node:
                    extra_input_of_binary_node = arg
                    break
            if extra_input_of_binary_node is None:
                raise AssertionError("expected non-None extra input of binary node")
            # Extra input of binary node comes from dequant pattern
            if extra_input_from_dequant and (
                (not isinstance(extra_input_of_binary_node, torch.fx.Node))
                or (
                    extra_input_of_binary_node.target
                    != quantized_decomposed.dequantize_per_tensor.default
                )
            ):
                return False

        # the two inputs of binary node should have attribute "meta" and should be tensors
        if not (
            hasattr(binary_node_inputs[0], "meta")
            and isinstance(binary_node_inputs[0].meta.get("val", None), torch.Tensor)  # type: ignore[union-attr]
        ) or not (
            hasattr(binary_node_inputs[1], "meta")
            and isinstance(binary_node_inputs[1].meta.get("val", None), torch.Tensor)  # type: ignore[union-attr]
        ):
            return False
        # the two inputs of binary node should have the same shape
        if (
            binary_node_inputs[0].meta["val"].size()  # type: ignore[union-attr]
            != binary_node_inputs[1].meta["val"].size()  # type: ignore[union-attr]
        ):
            return False

        # All users of the extra input in this pattern should be
        # ancestor nodes of the compute node, except for the binary node
        # connected to the compute node.

        from .mkldnn_fusion import _get_remaining_users

        extra_input_of_pattern = (
            match.kwargs["other"]
            if "other" in match.kwargs
            else (
                match.kwargs["accum"]
                if (output_dtype in [torch.uint8, torch.int8])
                or (not extra_input_from_dequant)
                else match.kwargs["accum_after_dequant"]
            )
        )
        if (
            len(_get_remaining_users(extra_input_of_pattern, compute_node)) > 1
            or extra_input_of_pattern == compute_node.args[0]
        ):
            return False
        return True

    return fn


def _register_quantized_conv_binary_lowering(
    pattern,
    pass_number,
    computation_op,
):
    # [FUNCTION DOC] HELPER: Register a lowering for a qconv binary pattern at the
    # given pass number using computation_op. Registers a lowering pattern (via
    # register_lowering_pattern) and returns the inner lowering function.
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qconv_lowering_pattern(),
        pass_number=pass_number,
    )
    def qconv_binary(match: Match, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Lowering callback: gathers qparams, the
        # accumulation input, conv params and post-op kwargs, realizes accum,
        # requires it to be safely updatable in place (AssertionError otherwise,
        # also if the output dtype is None), bumps qconv2d_binary_lower counters and
        # returns the lowering of computation_op.
        output_dtype = _get_pattern_output_dtype(match)
        if output_dtype is None:
            raise AssertionError("expected non-None output dtype")
        x, x_scale, x_zp = kwargs["x"], kwargs["x_scale"], kwargs["x_zp"]
        accum = kwargs["accum"]
        accum_scale = kwargs["accum_scale"]
        accum_zp = kwargs["accum_zero_point"]
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        b, stride, padding, dilation, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["groups"],
        )
        # Output QParams
        output_scale = kwargs["output_scale"]
        output_zero_point = kwargs["output_zero_point"]

        # post ops
        binary_op_name = kwargs["binary_op_name"]
        alpha = kwargs["alpha"]
        unary_op_name = kwargs["unary_op_name"]
        unary_op_args = kwargs["unary_op_args"]
        unary_op_algorithm = kwargs["unary_op_algorithm"]

        accum.realize()
        from .mkldnn_fusion import _can_be_inplace

        if not _can_be_inplace(accum):
            raise AssertionError(
                "QConv Binary Inplace Fusion requires accum is not an alias or mutation."
            )

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            accum,
            b,
            stride,
            padding,
            dilation,
            groups,
            output_scale,
            output_zero_point,
            output_dtype,
            accum_scale,
            accum_zp,
            binary_op_name,
            alpha,
            unary_op_name,
            unary_op_args,
            unary_op_algorithm,
        )
        counters["inductor"]["qconv2d_binary_lower_count"] += 1
        counters["inductor"]["qconv2d_binary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qconv_binary


def _register_quantization_unary_lowering():
    # [FUNCTION DOC] HELPER: Register unary lowerings: qconv patterns for every
    # combination of tensor-or-scalar qparams and user counts 1 and 2, and qlinear
    # patterns for tensor-or-scalar qparams, all at pass number 2.
    # QConv2d
    for x_scale_zp_are_tensors, users in itertools.product([False, True], [1, 2]):
        qconv_pattern = get_qconv_pt2e_pattern(x_scale_zp_are_tensors, users)
        computation_op = (
            torch.ops.onednn.qconv_pointwise.tensor
            if x_scale_zp_are_tensors
            else torch.ops.onednn.qconv_pointwise.default
        )
        _register_quantized_conv_lowering(
            qconv_pattern,
            2,  # pass_number
            computation_op,
        )

    # QLinear
    for x_scale_zp_are_tensors in (False, True):
        qlinear_pattern = get_qlinear_pt2e_pattern(x_scale_zp_are_tensors)
        computation_op = (
            torch.ops.onednn.qlinear_pointwise.tensor
            if x_scale_zp_are_tensors
            else torch.ops.onednn.qlinear_pointwise.default
        )
        _register_quantized_linear_unary_lowering(
            qlinear_pattern,
            2,  # pass_number
            computation_op,
        )


def _register_quantization_binary_lowering():
    # [FUNCTION DOC] HELPER: Register binary lowerings: qconv2d binary patterns for
    # tensor-or-scalar qparams and user counts 1 and 2, and qlinear binary patterns
    # for tensor-or-scalar qparams, all at pass number 2.
    # QConv2d
    for x_scale_zp_are_tensors, users in itertools.product([False, True], [1, 2]):
        qconv_pattern = get_qconv2d_binary_pt2e_pattern(x_scale_zp_are_tensors, users)
        computation_op = (
            torch.ops.onednn.qconv2d_pointwise.binary_tensor
            if x_scale_zp_are_tensors
            else torch.ops.onednn.qconv2d_pointwise.binary
        )
        _register_quantized_conv_binary_lowering(
            qconv_pattern,
            2,  # pass_number
            computation_op,
        )

    # QLinear
    for x_scale_zp_are_tensors in (False, True):
        qlinear_pattern = get_qlinear_binary_pt2e_pattern(x_scale_zp_are_tensors)
        computation_op = (
            torch.ops.onednn.qlinear_pointwise.binary_tensor
            if x_scale_zp_are_tensors
            else torch.ops.onednn.qlinear_pointwise.binary
        )
        _register_quantized_linear_binary_lowering(
            qlinear_pattern,
            2,  # pass_number
            computation_op,
        )


def _is_valid_quantized_maxpool2d_optimization_pattern():
    # [FUNCTION DOC] HELPER: Return a match filter for the max_pool2d pattern that
    # only accepts when the getitem selects the values output (index 0), not the
    # indices.
    def fn(match):
        # [FUNCTION DOC] HELPER: Match filter: True when the getitem node's index
        # equals 0.
        # Only match the pattern which max_pool2d_with_indices returns value
        # instead of indices.
        get_item_node = filter_nodes(match.nodes, operator.getitem)[0]
        return get_item_node.args[1] == 0

    return fn


def _register_quantized_maxpool2d_lowering(
    pattern,
    computation_op,
):
    # [FUNCTION DOC] HELPER: Register a lowering for a quantized max_pool2d pattern
    # with the given computation_op and an extra check for values-only output.
    # Registers a lowering pattern (via register_lowering_pattern) and returns the
    # inner lowering function.
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_quantized_maxpool2d_optimization_pattern(),
    )
    def qmaxpool2d(match: Match, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Lowering callback: reads kernel_size and optional
        # stride, padding, dilation, ceil_mode; fills defaults (stride defaults to
        # kernel_size), pads each to length 2, validates lengths (AssertionError),
        # requires channels-last input, bumps qmaxpool2d matcher counters and
        # returns the lowering of computation_op.
        x = kwargs["x"]
        kernel_size = kwargs["kernel_size"]
        stride = kwargs.get("stride")
        padding = kwargs.get("padding", 0)
        dilation = kwargs.get("dilation", 1)
        ceil_mode = kwargs.get("ceil_mode", False)

        if padding == 0:
            padding = [0, 0]
        if dilation == 1:
            dilation = [1, 1]
        if not stride:
            stride = kernel_size
        kernel_size = pad_listlike(kernel_size, 2)
        stride = pad_listlike(stride, 2)
        padding = pad_listlike(padding, 2)
        dilation = pad_listlike(dilation, 2)

        if len(kernel_size) != 2:
            raise AssertionError(
                f"expected kernel_size of length 2, got {len(kernel_size)}"
            )
        if len(stride) != 2:
            raise AssertionError(f"expected stride of length 2, got {len(stride)}")
        if len(padding) != 2:
            raise AssertionError(f"expected padding of length 2, got {len(padding)}")
        if len(dilation) != 2:
            raise AssertionError(f"expected dilation of length 2, got {len(dilation)}")

        computation_args = (
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        computation_args, _ = require_channels_last(computation_op, *computation_args)
        counters["inductor"]["qmaxpool2d_matcher_count"] += 1
        counters["inductor"]["qmaxpool2d_matcher_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qmaxpool2d


def _register_quantization_maxpool2d():
    # [FUNCTION DOC] HELPER: Register quantized max_pool2d lowerings for the four
    # argument-count variants (stride, padding, dilation, ceil_mode keywords present
    # or not), for both the standard and low-memory max pool ops, each wrapped in
    # getitem and output quantization patterns, lowering to quantized.max_pool2d.
    # Currently, the default parameters are not in FX Graph generated by Dynamo export.
    # So, if user defines nn.MaxPool2d with different assignment of default parameter,
    # it will generate graph with different number of input nodes and hence
    # different pattern to be matched.
    # Refer to the issue: https://github.com/pytorch/pytorch/issues/105901
    max_pool2d_args_list = [
        [
            KeywordArg("stride"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
            KeywordArg("dilation"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
            KeywordArg("dilation"),
            KeywordArg("ceil_mode"),
        ],
    ]
    for max_pool2d_args in max_pool2d_args_list:
        dequantize_maxpool2d_pattern = CallFunction(
            aten.max_pool2d_with_indices.default,
            get_dequantize_per_tensor_activation_pattern(),
            KeywordArg("kernel_size"),
            *max_pool2d_args,
        )
        dequantize_lowmem_maxpool2d_pattern = CallFunction(
            prims._low_memory_max_pool_with_offsets.default,
            get_dequantize_per_tensor_activation_pattern(),
            KeywordArg("kernel_size"),
            *max_pool2d_args,
            KeywordArg("offset_dtype"),
        )
        dequantize_maxpool2d_get_item_pattern = CallFunction(
            operator.getitem,
            dequantize_maxpool2d_pattern,
            Arg(),
        )
        dequantize_lowmem_maxpool2d_get_item_pattern = CallFunction(
            operator.getitem,
            dequantize_lowmem_maxpool2d_pattern,
            Arg(),
        )
        _register_quantized_maxpool2d_lowering(
            generate_pattern_with_output_quant(dequantize_maxpool2d_get_item_pattern),
            quantized.max_pool2d.default,
        )
        _register_quantized_maxpool2d_lowering(
            generate_pattern_with_output_quant(
                dequantize_lowmem_maxpool2d_get_item_pattern
            ),
            quantized.max_pool2d.default,
        )


def _is_input_output_same_scale_zp(check_node):
    def fn(match):
    # [FUNCTION DOC] HELPER: Return a match filter that requires all dequant input
    # zero points and the single quant output zero point to be equal and all scales
    # to be close (relative tolerance 1e-5). Raises AssertionError if the match does
    # not have exactly one quantize node. The check_node argument is unused.
    # KNOWN ISSUE: the parameter check_node is accepted but never used by the inner
    # filter.
        # Ensure all the inputs and output has same scale and zero point
        # [FUNCTION DOC] HELPER: Match filter implementing the same-scale and same-
        # zero-point check; returns True or False.
        # Step 1: Check inputs/output zero point
        # Get dequant nodes at input
        dequant_nodes = filter_nodes(
            match.nodes, quantized_decomposed.dequantize_per_tensor.default
        )
        zero_points = [node.args[2] for node in dequant_nodes]
        # Get quant nodes at output
        quant_nodes = filter_nodes(
            match.nodes, quantized_decomposed.quantize_per_tensor.default
        )
        if len(quant_nodes) != 1:
            raise AssertionError("expect only 1 add node at output quant pattern")
        zero_points.append(quant_nodes[0].args[2])
        if not all(zero_point == zero_points[0] for zero_point in zero_points):
            return False

        # Step 2: Check inputs/output scale
        scales = [node.args[1] for node in dequant_nodes]
        scales.append(quant_nodes[0].args[1])
        if not all(math.isclose(scale, scales[0], rel_tol=1e-5) for scale in scales):  # type: ignore[arg-type]
            return False

        return True

    return fn


def _register_quantized_cat_lowering(
    pattern,
    computation_op,
):
    # [FUNCTION DOC] HELPER: Register a lowering for a quantized cat pattern with an
    # input/output same-qparams check. Registers a lowering pattern (via
    # register_lowering_pattern) and returns the inner lowering function.
    @register_lowering_pattern(
        pattern,
        extra_check=_is_input_output_same_scale_zp(aten.cat.default),
    )
    def qcat(match: Match, inputs, dim, **kwargs):
        # [FUNCTION DOC] HELPER: Lowering callback: takes the list of matched inputs
        # (each a list whose first entry is the quantized tensor), bumps qcat
        # matcher counters and returns the lowering of computation_op on the
        # quantized tensors and dim.
        # inputs is with format: [[x1, x1_dq_dtype, x1_zp, x1_scale], ...]
        uint8_inputs = [input[0] for input in inputs]
        counters["inductor"]["qcat_matcher_count"] += 1
        counters["inductor"]["qcat_matcher_nodes"] += len(match.nodes)
        return L[computation_op](uint8_inputs, dim)

    return qcat


_raw_dequantize_per_tensor_activation_pattern = CallFunction(
    quantized_decomposed.dequantize_per_tensor.default,
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
)


def _register_quantization_cat():
    # [FUNCTION DOC] HELPER: Register the quantized cat lowering: dequantize each
    # input, cat along keyword dim, then quantize the output, lowered to cat on the
    # quantized tensors.
    dequantize_cat_pattern = CallFunction(
        aten.cat.default,
        ListOf(_raw_dequantize_per_tensor_activation_pattern),
        KeywordArg("dim"),
    )
    _register_quantized_cat_lowering(
        generate_pattern_with_output_quant(dequantize_cat_pattern),
        aten.cat,
    )


def _register_quantized_reshape_lowering(
    pattern,
    computation_op,
):
    # [FUNCTION DOC] HELPER: Register a lowering for a quantized reshape pattern
    # with an input/output same-qparams check; output metadata is computed as a
    # plain reshape of the input. Registers a lowering pattern (via
    # register_lowering_pattern) and returns the inner lowering function.
    def qreshape_output_metadata(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Compute output metadata for the matched reshape by
        # applying aten reshape to the x and shape keyword arguments.
        return aten.reshape.default(kwargs["x"], kwargs["shape"])

    @register_lowering_pattern(
        pattern,
        extra_check=_is_input_output_same_scale_zp(aten.reshape.default),
        output_metadata_ignores_input_storage=False,
        output_metadata_fn=qreshape_output_metadata,
    )
    def qreshape(match: Match, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Lowering callback: reads quantized input x and
        # shape, bumps qreshape matcher counters and returns the lowering of
        # computation_op.
        qx = kwargs["x"]
        shape = kwargs["shape"]
        counters["inductor"]["qreshape_matcher_count"] += 1
        counters["inductor"]["qreshape_matcher_nodes"] += len(match.nodes)
        return L[computation_op](qx, shape)

    return qreshape


def _register_quantization_reshape():
    # [FUNCTION DOC] HELPER: Register the quantized reshape lowering: dequantize,
    # reshape to keyword shape, quantize output, lowered to reshape on the quantized
    # tensor.
    dequantize_reshape_pattern = CallFunction(
        torch.ops.aten.reshape.default,
        get_dequantize_per_tensor_activation_pattern(),
        KeywordArg("shape"),
    )
    _register_quantized_reshape_lowering(
        generate_pattern_with_output_quant(dequantize_reshape_pattern),
        aten.reshape,
    )


def _is_valid_concat_linear_int8_woq_optimization_pattern():
    # [FUNCTION DOC] HELPER: Return a match filter for int8 WOQ concat-linear
    # patterns. Returns False if config.cpp.enable_concat_linear is off, if the
    # needed matched nodes lack metadata, or if scales is more than 1-D. Accepts
    # only when x and scales are bfloat16, the three weights are int8, x is on cpu
    # or cuda with all tensors on the same device, and the scale count equals the
    # total output rows of the three weights. Raises AssertionError if required
    # kwargs are missing.
    def fn(match):
        # [FUNCTION DOC] HELPER: Match filter implementing the checks described in
        # the enclosing function.
        if not config.cpp.enable_concat_linear:
            return False
        if not all(k in match.kwargs for k in ("x", "w1", "w2", "w3", "scales")):
            raise AssertionError("expected x, w1, w2, w3, scales in match kwargs")
        if not all(
            hasattr(match.kwargs[key], "meta")
            for key in ["x", "w1", "w2", "w3", "scales"]
        ):
            return False
        x = match.kwargs["x"].meta["val"]
        w1 = match.kwargs["w1"].meta["val"]
        w2 = match.kwargs["w2"].meta["val"]
        w3 = match.kwargs["w3"].meta["val"]
        scales = match.kwargs["scales"].meta["val"]
        if len(match.kwargs["scales"].meta["val"].size()) > 1:
            return False
        num_scales = match.kwargs["scales"].meta["val"].numel()
        w1_cols = match.kwargs["w1"].meta["val"].size()[0]
        w2_cols = match.kwargs["w2"].meta["val"].size()[0]
        w3_cols = match.kwargs["w3"].meta["val"].size()[0]
        return (
            # For now, we only support woq mm kernels
            # with x.type=bfloat16 and w.type=int8
            x.dtype == torch.bfloat16
            and w1.dtype == torch.int8
            and w2.dtype == torch.int8
            and w3.dtype == torch.int8
            and scales.dtype == torch.bfloat16
            and x.device.type in ("cpu", "cuda")
            and x.device == w1.device
            and w1.device == w2.device
            and w2.device == w3.device
            and x.device == scales.device
            and num_scales == w1_cols + w2_cols + w3_cols
        )

    return fn


def _is_valid_woq_optimization_pattern():
    # [FUNCTION DOC] HELPER: Return a match filter for int8 WOQ mm patterns:
    # requires x and scales in bfloat16, weight int8, device in cpu, cuda or xpu,
    # all on the same device; returns False if metadata is missing. Raises
    # AssertionError if x, weight or scales kwargs are absent.
    def fn(match):
        # [FUNCTION DOC] HELPER: Match filter implementing the checks described in
        # the enclosing function.
        if not all(k in match.kwargs for k in ("x", "weight", "scales")):
            raise AssertionError("expected x, weight, scales in match kwargs")
        if not all(
            hasattr(match.kwargs[key], "meta") for key in ["x", "weight", "scales"]
        ):
            return False
        x = match.kwargs["x"].meta["val"]
        weight = match.kwargs["weight"].meta["val"]
        scales = match.kwargs["scales"].meta["val"]
        return (
            # For now, we only support woq mm kernels
            # with x.type=bfloat16 and w.type=int8
            x.dtype == torch.bfloat16
            and weight.dtype == torch.int8
            and scales.dtype == torch.bfloat16
            and x.device.type in ("cpu", "cuda", "xpu")
            and x.device == weight.device
            and x.device == scales.device
        )

    return fn


def _register_concat_linear_int8_woq_lowering(
    pattern, computation_woq, computation_reshape
):
    # [FUNCTION DOC] HELPER: Register a freezing-graph pattern (pass number 4) that
    # fuses three WOQ int8 linears sharing an input into one concatenated WOQ call.
    # Registers a lowering pattern (via register_lowering_pattern) and returns the
    # inner lowering function.
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_concat_linear_int8_woq_optimization_pattern(),
        pass_number=4,
    )
    def woq_int8(match: Match, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Replacement callback: finds the mm node using x
        # with a cat weight, concatenates the three weights, reshapes x to 2-D,
        # calls the fused WOQ op with the scales, reshapes the result to the
        # original leading dims with the summed output features, redirects uses of
        # the scaling node and erases the old scaling, mm and cat nodes. Sorts the
        # graph topologically first as a workaround and lints afterwards. Mutates
        # the graph and bumps woq matcher counters. Raises AssertionError if the mm
        # node is not found.
        x = kwargs["x"]
        w1 = kwargs["w1"]
        w2 = kwargs["w2"]
        w3 = kwargs["w3"]
        scales = kwargs["scales"]
        counters["inductor"]["woq_matcher_count"] += 1
        counters["inductor"]["woq_matcher_nodes"] += len(match.nodes)
        out_features = (
            w1.meta["val"].size()[0]
            + w2.meta["val"].size()[0]
            + w3.meta["val"].size()[0]
        )
        origin_x_size = tuple(x.meta["val"].size())
        x_shape = [-1, origin_x_size[-1]]
        out_shape = list(origin_x_size[:-1] + (out_features,))
        mm_node_of_x = None
        for candidate in iter(x.users.keys()):
            if (
                candidate.target is aten.mm.default
                and list(candidate._input_nodes)[1].target is aten.cat.default
            ):
                mm_node_of_x = candidate
                break
        if mm_node_of_x is None:
            raise AssertionError("unable to find mm node")
        _, cat_wgt_node = mm_node_of_x._input_nodes
        scaling_node = next(iter(mm_node_of_x.users.keys()))
        user_of_scaling_node = next(iter(scaling_node.users.keys()))
        # Some other pass is making some changes that entails
        # adding a node before it's used, but it can only be found when
        # lint is run. stable_topological_sort() is being run before lint,
        # so that error was not being discovered.
        # We call stable_topological_sort here as a workaround.
        stable_topological_sort(match.graph)
        with match.graph.inserting_before(user_of_scaling_node):
            new_cat_node = match.graph.call_function(
                aten.cat.default,
                args=([w1, w2, w3], 0),
            )
            x_reshape_node = match.graph.call_function(
                computation_reshape, args=(x, x_shape)
            )
            new_woq_node = match.graph.call_function(
                computation_woq,
                args=(x_reshape_node, new_cat_node, scales),
            )
            new_woq_node.meta = copy.copy(x.meta)
            output_reshape_node = match.graph.call_function(
                computation_reshape, args=(new_woq_node, out_shape)
            )
            scaling_node.replace_all_uses_with(output_reshape_node)
            match.graph.erase_node(scaling_node)
            match.graph.erase_node(mm_node_of_x)
            match.graph.erase_node(cat_wgt_node)
            match.graph.lint()

    return woq_int8


def _register_woq_lowering(pattern, computation_woq, computation_reshape):
    # [FUNCTION DOC] HELPER: Register a lowering pattern for WOQ int8 mm using
    # computation_woq and computation_reshape and a validity filter. Registers a
    # lowering pattern (via register_lowering_pattern) and returns the inner
    # lowering function.
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_woq_optimization_pattern(),
    )
    def woq_int8(match: Match, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Lowering callback: flattens x to 2-D with reshape,
        # calls the WOQ int8 op with weight and scales, reshapes back to the
        # original leading dims plus out_features, and bumps woq matcher counters.
        # Returns the lowered result.
        x = kwargs["x"]
        weight = kwargs["weight"]
        scales = kwargs["scales"]
        counters["inductor"]["woq_matcher_count"] += 1
        counters["inductor"]["woq_matcher_nodes"] += len(match.nodes)
        out_features = weight.get_size()[0]
        origin_x_size = x.get_size()
        x_shape = [-1, origin_x_size[-1]]
        out_shape = origin_x_size[:-1] + [
            out_features,
        ]
        func1 = L[computation_reshape](x, x_shape)
        func2 = L[computation_woq](func1, weight, scales)
        return L[computation_reshape](func2, out_shape)

    return woq_int8


def _register_woq_mm_int8_pattern1():
    # [FUNCTION DOC] HELPER: Register the WOQ int8 pattern: multiply of reshape of
    # mm (with reshaped x and permuted, dtype-converted weight) by scales, lowered
    # to the int8 pack mm op.
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to mm, with x reshape
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.mm.default,
                CallFunction(aten.reshape.default, KeywordArg("x"), Arg()),
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
            ),
            Arg(),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern2():
    # [FUNCTION DOC] HELPER: Register the WOQ int8 pattern for mm without an x
    # reshape: reshape of mm of x and the permuted converted weight, multiplied by
    # scales.
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to mm, w/o x reshape
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.mm.default,
                KeywordArg("x"),
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
            ),
            Arg(),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern3():
    # [FUNCTION DOC] HELPER: Register the WOQ int8 pattern for the bmm case: bmm of
    # an expanded x and an expanded permuted converted weight, multiplied by scales.
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to bmm
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.bmm.default,
            CallFunction(aten.expand.default, KeywordArg("x"), Arg()),
            CallFunction(
                aten.expand.default,
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
                Arg(),
            ),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern4():
    # [FUNCTION DOC] HELPER: Register the WOQ int8 pattern where mm of x with a
    # dtype-converted permuted weight is multiplied by scales.
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.mm.default,
            KeywordArg("x"),
            CallFunction(
                prims.convert_element_type.default,
                CallFunction(
                    aten.permute.default,
                    KeywordArg("weight"),
                    Arg(),
                ),
                Arg(),
            ),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_int8_woq_concat_linear_pattern():
    # [FUNCTION DOC] HELPER: Register the concat-linear WOQ int8 pattern: mm of x
    # with a concatenation (dim 1) of three permuted, dtype-converted weights w1,
    # w2, w3, multiplied by scales.
    def _create_wgt_node(wgt_node_name: str):
        # [FUNCTION DOC] HELPER: Build a pattern for one weight (keyword given by
        # wgt_node_name): a dtype convert of a permute.
        return CallFunction(
            prims.convert_element_type.default,
            CallFunction(
                aten.permute.default,
                KeywordArg(wgt_node_name),
                Arg(),
            ),
            Arg(),
        )

    cat_wgt = CallFunction(
        aten.cat.default, [_create_wgt_node(wgt) for wgt in ["w1", "w2", "w3"]], 1
    )

    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(aten.mm.default, KeywordArg("x"), cat_wgt),
        KeywordArg("scales"),
    )
    _register_concat_linear_int8_woq_lowering(
        _woq_pattern, aten._weight_int8pack_mm.default, aten.reshape
    )


def _register_quantization_lowerings():
    # [FUNCTION DOC] Register all quantization lowerings: unary, binary, max_pool2d,
    # cat and reshape.
    _register_quantization_unary_lowering()
    _register_quantization_binary_lowering()
    _register_quantization_maxpool2d()
    _register_quantization_cat()
    _register_quantization_reshape()


def _register_woq_lowerings():
    # [FUNCTION DOC] Register all weight-only-quantization int8 mm lowerings
    # (patterns 1 to 4).
    _register_woq_mm_int8_pattern1()
    _register_woq_mm_int8_pattern2()
    _register_woq_mm_int8_pattern3()
    _register_woq_mm_int8_pattern4()


def _is_valid_concat_linear_woq_int4_fusion(computation_nodes):
    # [FUNCTION DOC] HELPER: Check that a list of int4 WOQ mm nodes can be fused: at
    # least two nodes, all with the same target, the same activation, the same input
    # feature size and group size, all weights constants (get_attr) and distinct
    # from the first weight except at the first index.
    computation_op = torch.ops.aten._weight_int4pack_mm_for_cpu.default
    act = computation_nodes[0].args[0]
    wgt = computation_nodes[0].args[1]
    in_feature_size = wgt.meta.get("val").size(1)  # type: ignore[union-attr]
    group_size = computation_nodes[0].args[2]
    return len(computation_nodes) >= 2 and all(
        (
            node.target == computation_op
            and node.args[0] == act  # share same activation
            and (
                node.args[1].meta.get("val").size(1) == in_feature_size
            )  # same in feature size
            and (node.args[1] != wgt or gemm_idx == 0)
            and node.args[1].op == "get_attr"  # wgt are all constants
            and node.args[2] == group_size  # same group size
        )
        for gemm_idx, node in enumerate(computation_nodes)
    )


def concat_linear_woq_int4(gm: torch.fx.GraphModule):
    # [FUNCTION DOC] Graph pass that fuses several int4 WOQ CPU linears sharing the
    # same activation into a single call on concatenated weights and scale/zero-
    # point buffers, followed by a split and per-output clone nodes. Mutates the
    # graph module: registers new concatenated buffers, creates nodes, replaces and
    # erases the original gemm nodes. Raises AssertionError on inconsistent users.
    # KNOWN ISSUE: each new concatenated buffer is registered as a buffer and then
    # set again as a plain attribute, which is redundant.
    """
    Concat Linear optimization pass for WOQ int4
    This pass fuses the original pattern:
    def ...
        return (woq_int4(x, w1, group_size, scale_zp1), woq_int4(x, w2, group_size, scale_zp1) ...)
    into a single operation:
    def ...
        concat_res = woq_int4(x, concat_w, group_size, concat_scale_zp)
        return split(concat_res, split_size_list)
    """

    def concat_wgt(packed_wgts, scale_zps, group_size, act_dtype):
        # [FUNCTION DOC] HELPER: Unpack each int4 packed weight by multiplying an
        # identity matrix through the int4 mm op with fixed scales and zeros,
        # transpose and cast to int32, concatenate along rows, and repack with the
        # int4 conversion op; also concatenate the scale/zero-point tensors along
        # dim 1. Returns the repacked weight and concatenated scale/zero-point.
        # Concat the wgts and scale_zps, and repack the wgt
        unpacked_wgts = []
        for packed_wgt in packed_wgts:
            # Get the unpacked weight list
            # Same as https://github.com/pytorch/pytorch/pull/156174
            K = packed_wgt.size(1) * 2
            N = packed_wgt.size(0)
            x = torch.eye(K).to(dtype=act_dtype)
            qscales_and_zeros = (
                torch.tensor([1.0, 8.0])
                .to(dtype=act_dtype)
                .expand(K // group_size, N, 2)
                .contiguous()
            )
            unpacked_wgts.append(
                torch.ops.aten._weight_int4pack_mm_for_cpu(
                    x,
                    packed_wgt,
                    group_size,
                    qscales_and_zeros,
                )
                .t()
                .contiguous()
                .to(torch.int32)  # N, K
            )
        concat_unpacked_wgt = torch.cat(unpacked_wgts, dim=0)
        repack_w = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            concat_unpacked_wgt, 1
        )
        concat_scale_zp = torch.cat(scale_zps, dim=1).contiguous()
        return repack_w, concat_scale_zp

    graph = gm.graph
    computation_op = torch.ops.aten._weight_int4pack_mm_for_cpu.default
    for node in graph.find_nodes(op="call_function", target=computation_op):
        if (
            not node._erased
            and isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].device.type == "cpu"
        ):
            act = node.args[0]
            users = list(act.users)
            if _is_valid_concat_linear_woq_int4_fusion(users):
                with graph.inserting_before(node):
                    if not all(user.args[1].op == "get_attr" for user in users):
                        raise AssertionError(
                            "expected all users to have get_attr weight"
                        )
                    computation_node_0 = users[0]
                    packed_wgts = [getattr(gm, user.args[1].target) for user in users]
                    group_size = computation_node_0.args[2]
                    scale_zps = [getattr(gm, user.args[3].target) for user in users]
                    out_feature_size_list = [
                        packed_wgt.size(0) for packed_wgt in packed_wgts
                    ]
                    repack_w, concat_scale_zp = concat_wgt(
                        packed_wgts, scale_zps, group_size, act.meta.get("val").dtype
                    )
                    repack_w_node_name = computation_node_0.args[1].target + "_concat"
                    concat_scale_zp_node_name = (
                        computation_node_0.args[3].target + "_concat"
                    )
                    gm.register_buffer(repack_w_node_name, repack_w)
                    setattr(gm, repack_w_node_name, repack_w)
                    gm.register_buffer(concat_scale_zp_node_name, concat_scale_zp)
                    setattr(gm, concat_scale_zp_node_name, concat_scale_zp)

                    repack_w_node = graph.create_node(
                        "get_attr", repack_w_node_name, (), {}
                    )
                    with graph.inserting_after(repack_w_node):
                        concat_scale_zp_node = graph.create_node(
                            "get_attr", concat_scale_zp_node_name, (), {}
                        )

                    with graph.inserting_after(concat_scale_zp_node):
                        concat_int4_gemm_node = graph.create_node(
                            "call_function",
                            computation_op,
                            (
                                act,
                                repack_w_node,
                                group_size,
                                concat_scale_zp_node,
                            ),
                        )
                    with graph.inserting_after(concat_int4_gemm_node):
                        split_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.split_with_sizes.default,
                            (
                                concat_int4_gemm_node,
                                out_feature_size_list,
                                1,  # split dim
                            ),
                        )
                    with graph.inserting_after(split_node):
                        for gemm_idx, user in enumerate(users):
                            if user.target != computation_op:
                                raise AssertionError(
                                    f"expected target {computation_op}, got {user.target}"
                                )
                            get_item = graph.create_node(
                                "call_function",
                                operator.getitem,
                                (
                                    split_node,
                                    gemm_idx,
                                ),
                            )
                            with graph.inserting_after(get_item):
                                clone_node = graph.create_node(
                                    "call_function",
                                    torch.ops.aten.clone.default,
                                    (get_item,),
                                    {"memory_format": torch.contiguous_format},
                                )
                                user.replace_all_uses_with(clone_node)
                                graph.erase_node(user)


def quant_lift_up(graph_module: torch.fx.GraphModule):
    # [FUNCTION DOC] Graph pass that moves a per-tensor quantize node up before a
    # chain of view-like ops (transpose, permute, view, reshape) whose intermediate
    # nodes each have exactly one user, so a dequant-linear-quant pattern forms.
    # Creates a copy of the quant node on the chain's input, rewires users, erases
    # the original, bumps quant_lift_up_count, then lints and recompiles the graph
    # module. Mutates the module in place.
    """
    Lift up the quant node before view like nodes. It can benefit performance
    of Attention like block. For example, we have the pattern as:

             DQ
    DQ       LINEAR
    LINEAR   VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    Q        Q
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    We want to lift up the quant nodes from matmul before view like nodes
    as the output of Linear node.

             DQ
    DQ       LINEAR
    LINEAR   Q
    Q        VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    It produces a DQ->LINEAR->Q pattern which can be fused by backend.
    """

    def is_view_op(node):
        # [FUNCTION DOC] HELPER: Return True if the node is a call_function whose
        # target is a view-like op.
        return node.op == "call_function" and node.target in _VIEW_OPS

    for node in graph_module.graph.nodes:
        # <TODO> Leslie: Here we verify that the quant node has exactly
        # one input FX node, with constant scalar value for scale and zero point.
        # For the case input of quant node has more than one input FX nodes,
        # extend the implementation to lift up all the connected nodes
        # before the view nodes to keep the topological order.
        if (
            node.op == "call_function"
            and node.target in _PER_TENSOR_QUANTIZE_OPS
            and len(node.all_input_nodes) == 1
            and is_view_op(node.all_input_nodes[0])
        ):
            quant_node = node
            input_node_of_quant = quant_node.args[0]

            # Check the nodes along lift up path has only 1 user node
            # Propagate view like node to find where to insert the new quant node
            could_lift_up = True
            current_node = quant_node
            input_node = current_node.args[0]
            while is_view_op(input_node):
                if len(input_node.users) != 1:
                    could_lift_up = False
                    break
                current_node = input_node
                input_node = current_node.args[0]

            # Further check the input node of the first view node has only 1 user node
            if could_lift_up and len(input_node.users) == 1:
                counters["inductor"]["quant_lift_up_count"] += 1
                # Replace dequant's input from quant to quant's input
                quant_node.replace_all_uses_with(input_node_of_quant)
                # Insert the new quant node
                with graph_module.graph.inserting_before(current_node):
                    new_quant_node = graph_module.graph.node_copy(quant_node)
                    input_node.replace_all_uses_with(new_quant_node)

                    # Update inputs of new_quant_node
                    def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
                        # [FUNCTION DOC] HELPER: Return input_node in place of the
                        # original quant input node, other nodes unchanged; used to
                        # remap the copied quant node's arguments.
                        if n == input_node_of_quant:
                            return input_node
                        else:
                            return n

                    new_args = map_arg(new_quant_node.args, maybe_replace_node)
                    new_kwargs = map_arg(new_quant_node.kwargs, maybe_replace_node)
                    new_quant_node.args = new_args  # type: ignore[assignment]
                    new_quant_node.kwargs = new_kwargs  # type: ignore[assignment]
                    graph_module.graph.erase_node(quant_node)

    graph_module.graph.lint()
    graph_module.recompile()
