# mypy: allow-untyped-defs
# [MODULE DOC] Inductor fx passes for oneDNN (MKLDNN) fusion. When the PyTorch build has mkldnn, defines patterns and lowerings that fuse
# conv/linear with unary and binary post-ops, in-place binary fusion, grouped GEMM, weight prepacking for CPU and XPU, and linear recovery,
# plus cached init functions that register them. Import-time side effects: everything is defined only if mkldnn is compiled in; no patterns are
# registered until the init functions run.
import functools
import operator
from functools import reduce
from typing import Any, TYPE_CHECKING

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols, optimization_hint
from torch.utils._ordered_set import OrderedSet

from .. import ir, mkldnn_ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
    Arg,
    CallFunction,
    filter_nodes,
    get_arg_value,
    KeywordArg,
    MULTIPLE,
)
from ..utils import (
    is_mkldnn_bf16_supported,
    is_mkldnn_fp16_supported,
    SUPPORTED_MKLDNN_DEVICES,
)
from ..virtualized import ops, V
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
    _register_int8_woq_concat_linear_pattern,
    _register_quantization_lowerings,
    _register_woq_lowerings,
)


if TYPE_CHECKING:
    from collections.abc import Callable


if torch._C._has_mkldnn:
    aten = torch.ops.aten
    mkldnn = torch.ops.mkldnn
    prims = torch.ops.prims

    _conv_args = [Arg() for _ in range(10)]
    _linear_args = [Arg() for _ in range(6)]
    _conv_transpose_args = [Arg() for _ in range(11)]

    class MkldnnDeviceOpBase:
        # [CLASS DOC] Abstract per-device strategy for packing (pre-reordering) convolution and linear weights for MKLDNN.
        # Subclasses implement the device-specific graph node creation; every base method raises not-implemented.
        def get_linear_transpose_weight(self, weight_node):
            # [FUNCTION DOC] Returns the node holding the transposed weight for a packed linear weight node. Base: raises NotImplementedError.
            raise NotImplementedError

        def pack_conv_weight(
            self,
            graph,
            is_transposed,
            weight,
            constant_args,
            input_size,
        ):
            # [FUNCTION DOC] Creates the graph node that packs a convolution weight. Args: graph, is_transposed, weight, constant args, input size. Base: raises NotImplementedError.
            raise NotImplementedError

        def pack_linear_weight(
            self, graph, is_lp_weight, transpose_weight_node, batch_size
        ):
            # [FUNCTION DOC] Creates the graph node that packs a linear weight. Args: graph, low-precision flag, transposed weight node, batch size. Base: raises NotImplementedError.
            raise NotImplementedError

        def pack_linear(
            self, graph, is_lp_weight, batch_size, input, packed_weight_node, bias
        ):
            # [FUNCTION DOC] Creates the graph node performing the packed linear. Args: graph, low-precision flag, batch size, input, packed weight node, bias. Base: raises NotImplementedError.
            raise NotImplementedError

    class CpuMkldnnDeviceOp(MkldnnDeviceOpBase):
        # [CLASS DOC] CPU implementation of the MKLDNN device strategy. Packs conv weights with mkldnn reorder ops and linear weights with either
        # the mkldnn reorder (low precision, ACL, AOT or symbolic batch size) or the MKL reorder; no instance state.
        def get_linear_transpose_weight(self, weight_node):
            # [FUNCTION DOC] Given a packed linear weight node, returns the transpose (permute) node it wraps. Raises an error if the node is not an mkldnn linear
            # weight reorder or its input is not a permute.
            packed_weight_node = weight_node
            if packed_weight_node.target != mkldnn._reorder_linear_weight:
                raise AssertionError(
                    "expected packed_weight_node.target to be "
                    "mkldnn._reorder_linear_weight, got "
                    f"{packed_weight_node.target}"
                )
            transpose_weight_node = packed_weight_node.args[0]
            if transpose_weight_node.target is not aten.permute.default:
                raise AssertionError(
                    "expected transpose_weight_node.target to be "
                    f"aten.permute.default, got {transpose_weight_node.target}"
                )
            return transpose_weight_node

        def pack_conv_weight(
            self,
            graph,
            is_transposed,
            weight,
            constant_args,
            input_size,
        ):
            # [FUNCTION DOC] Creates a call_function node reordering the convolution weight (transposed variant when requested) with the constant args and input size.
            # Returns the new node; inserts into the graph at the current insertion point.
            packed_weight_op = mkldnn._reorder_convolution_weight
            if is_transposed:
                packed_weight_op = mkldnn._reorder_convolution_transpose_weight

            # mkldnn_reorder_conv_weight(self, padding, stride, dilation, groups, input_size)
            packed_weight_inputs = (weight,) + tuple(constant_args) + (input_size,)
            return graph.create_node(
                "call_function", packed_weight_op, args=packed_weight_inputs
            )

        def pack_linear_weight(
            self, graph, is_lp_weight, transpose_weight_node, batch_size
        ):
            # [FUNCTION DOC] Creates a call_function node reordering a linear weight. Uses the mkldnn reorder (with batch size hint only when it has free symbols)
            # if the weight is low precision, ACL is supported, AOT compilation is active, or batch size is symbolic; otherwise the MKL reorder.
            # Returns the new node.
            # For bfloat16 dynamic shape path, using input size hint to pack weight for a better performance.
            packed_weight_inputs = (
                transpose_weight_node,
                optimization_hint(batch_size)
                if has_free_symbols(batch_size)
                else batch_size,
            )

            # MKL packed matrix can't be copied to a different address because the internal implementation
            # depends on the alignment of internally-stored metadata.
            # In aot mode, we need to firstly save the packed weight, when loading it,
            # it will be in a different address which doesn't work.
            # Disable MKL prepack linear in AOT mode.
            # Disable MKL prepack linear when batch_size has free symbols.
            packed_weight_op = (
                mkldnn._reorder_linear_weight
                if (
                    is_lp_weight
                    or mkldnn._is_mkldnn_acl_supported()
                    or V.aot_compilation
                    or has_free_symbols(batch_size)
                )
                else torch.ops.mkl._mkl_reorder_linear_weight
            )
            return graph.create_node(
                "call_function", packed_weight_op, args=packed_weight_inputs
            )

        def pack_linear(
            self, graph, is_lp_weight, batch_size, input, packed_weight_node, bias
        ):
            # [FUNCTION DOC] Creates the packed linear node. For the mkldnn path (same condition as weight packing) uses the pointwise linear op with no fused activation;
            # otherwise the MKL linear op taking the transposed weight node, bias and batch size. Returns the new node.
            packed_linear_inputs: tuple[Any, ...] = (input, packed_weight_node)
            transpose_weight_node = packed_weight_node.args[0]
            if (
                is_lp_weight
                or mkldnn._is_mkldnn_acl_supported()
                or V.aot_compilation
                or has_free_symbols(batch_size)
            ):
                packed_linear_inputs += (bias, "none", [], "")
                packed_linear_op: Callable[..., Any] = mkldnn._linear_pointwise.default
            else:
                packed_linear_inputs += (transpose_weight_node, bias, batch_size)
                packed_linear_op = torch.ops.mkl._mkl_linear

            return graph.create_node(
                "call_function", packed_linear_op, packed_linear_inputs
            )

    class XpuMkldnnDeviceOp(MkldnnDeviceOpBase):
        # [CLASS DOC] XPU implementation of the MKLDNN device strategy; only supports convolution weight packing, which is a pass-through.
        def pack_conv_weight(
            self,
            graph,
            is_transposed,
            weight,
            constant_args,
            input_size,
        ):
            # [FUNCTION DOC] Returns the weight unchanged (no prepacking on XPU). Raises an error if a transposed convolution is requested.
            if is_transposed:
                raise AssertionError(
                    "'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device."
                )
            return weight

    def _get_mkldnn_device_op(device_type: str) -> MkldnnDeviceOpBase:
        # [FUNCTION DOC] Returns a CPU or XPU device strategy instance for the device type string; raises RuntimeError for other device types.
        """
        Returns the MKLDNN device operation class based on the current device type.
        """
        if device_type == "cpu":
            return CpuMkldnnDeviceOp()
        elif device_type == "xpu":
            return XpuMkldnnDeviceOp()
        else:
            raise RuntimeError(f"MKLDNN is not supported on {device_type} device.")

    def _is_valid_grouped_gemm_fusion(computation_nodes):
        # [FUNCTION DOC] HELPER: Returns True if at least two linear pointwise nodes share the same activation, all have the same weight shape, and weights are
        # distinct nodes (the first may repeat only at index 0). Used to decide whether to group the GEMMs.
        """
        Here we check:
        1. More than 1 GEMM nodes has been found.
        2. All the GEMM nodes share the same activation.
        3. All the GEMM nodes have same weight size but different wgt node.
        """
        computation_op = mkldnn._linear_pointwise.default
        act = computation_nodes[0].args[0]
        wgt = computation_nodes[0].args[1]
        wgt_size = wgt.meta.get("val").size()  # type: ignore[union-attr]
        return len(computation_nodes) >= 2 and all(
            (
                node.target == computation_op
                and node.args[0] == act
                and (node.args[1].meta.get("val").size() == wgt_size)
                and (node.args[1] != wgt or gemm_idx == 0)
            )
            for gemm_idx, node in enumerate(computation_nodes)
        )

    def grouped_gemm_pass(graph: torch.fx.Graph):
        # [FUNCTION DOC] Graph pass that fuses sibling mkldnn linear pointwise nodes on CPU that share one activation into one grouped GEMM call plus item extraction
        # nodes per output, replacing and erasing the originals. Mutates the graph in place; returns nothing. Raises an error if a user is not a linear node.
        # KNOWN ISSUE: validity is checked on all users of the activation, not just linear ones, so any other user type prevents grouping.
        """
        Group GEMM has multi output nodes which is complicated to define a Pattern.
        Use below way to connect the pattern to the lowering.
        TODO: Use MultiOutputPattern, current limitation is the pattern requires
        fixed number of output nodes. Extend to support Group GEMM for pattern matcher.
        """
        computation_op = mkldnn._linear_pointwise.default
        from ..mkldnn_lowerings import grouped_gemm_lowering

        for node in graph.find_nodes(op="call_function", target=computation_op):
            if (
                not node._erased
                and isinstance(node.meta.get("val"), torch.Tensor)
                and node.meta["val"].device.type == "cpu"
            ):
                act = node.args[0]
                users = list(act.users)
                if _is_valid_grouped_gemm_fusion(users):
                    with graph.inserting_before(node):
                        grouped_gemm_node = graph.create_node(
                            "call_function",
                            grouped_gemm_lowering,
                            (
                                act,
                                [user.args[1] for user in users],
                                [user.args[2] for user in users],
                            ),
                        )
                        grouped_gemm_node.meta["val"] = [
                            user.meta["val"] for user in users
                        ]
                        with graph.inserting_after(grouped_gemm_node):
                            for gemm_idx, user in enumerate(users):
                                if user.target != computation_op:
                                    raise AssertionError(
                                        "expected user.target to be "
                                        f"{computation_op}, got {user.target}"
                                    )
                                get_item = graph.create_node(
                                    "call_function",
                                    operator.getitem,
                                    (
                                        grouped_gemm_node,
                                        gemm_idx,
                                    ),
                                )
                                user.replace_all_uses_with(get_item)
                                graph.erase_node(user)
        return

    def _conv_call(users=1):
        # [FUNCTION DOC] HELPER: Pattern for a pointwise convolution call with the given user count.
        return CallFunction(
            mkldnn._convolution_pointwise.default, *_conv_args, _users=users
        )

    def _linear_call(users=1):
        # [FUNCTION DOC] HELPER: Pattern for a pointwise linear call with the given user count.
        return CallFunction(
            mkldnn._linear_pointwise.default, *_linear_args, _users=users
        )

    def _conv_transpose_call(users=1):
        # [FUNCTION DOC] HELPER: Pattern for a pointwise transposed-convolution call with the given user count.
        return CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=users,
        )

    def _to_float(input_call, users=1):
        # [FUNCTION DOC] HELPER: Pattern wrapping a call in a dtype conversion to float, capturing the target dtype as keyword argument to_float.
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_float"),
            _users=users,
        )

    def _to_bf16(input_call):
        # [FUNCTION DOC] HELPER: Pattern wrapping a call in a single-user dtype conversion capturing the target as keyword argument to_bf16.
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_bf16"),
            _users=1,
        )

    def _to_fp16(input_call):
        # [FUNCTION DOC] HELPER: Pattern wrapping a call in a single-user dtype conversion capturing the target as keyword argument to_fp16.
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_fp16"),
            _users=1,
        )

    def _unary_fusion_pattern(unary_fusion, call_fn, users, lowp_dtype):
        # [FUNCTION DOC] HELPER: Builds the full pattern for computation followed by a unary activation. For low precision, the computation is wrapped in a
        # convert-to-float, and the result converted back to bf16 or fp16. Args: activation pattern builder, computation call builder, user count, low precision dtype.
        # only insert to_dtype if lowp_dtype is True
        computation_call = (
            _to_float(call_fn(), users=users) if lowp_dtype else call_fn(users=users)
        )
        out = unary_fusion(computation_call)
        if lowp_dtype == torch.bfloat16:
            return _to_bf16(out)
        elif lowp_dtype == torch.float16:
            return _to_fp16(out)
        else:
            return out

    def _gelu_fusion_1(computation_call):
        # [FUNCTION DOC] HELPER: Pattern of exact gelu (erf form) applied to the computation call.
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.erf,
                    CallFunction(aten.mul, computation_call, 0.7071067811865476),
                ),
                1,
            ),
        )

    def _gelu_fusion_2(computation_call):
        # [FUNCTION DOC] HELPER: Pattern of tanh-approximated gelu applied to the computation call.
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.tanh,
                    CallFunction(
                        aten.mul,
                        CallFunction(
                            aten.add,
                            computation_call,
                            CallFunction(
                                aten.mul,
                                CallFunction(
                                    aten.mul,
                                    CallFunction(
                                        aten.mul, computation_call, computation_call
                                    ),
                                    computation_call,
                                ),
                                0.044715,
                            ),
                        ),
                        0.7978845608028654,
                    ),
                ),
                1,
            ),
        )

    def _hardswish_fusion(computation_call):
        # [FUNCTION DOC] HELPER: Pattern of hardswish expressed as multiply, add, clamp and divide primitives.
        return CallFunction(
            aten.div,
            CallFunction(
                aten.mul,
                computation_call,
                CallFunction(
                    aten.clamp_max,
                    CallFunction(
                        aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                    ),
                    6,
                ),
            ),
            6,
        )

    def _silu_fusion(computation_call):
        # [FUNCTION DOC] HELPER: Pattern of silu (swish) expressed as x divided by (exp of negative x plus one).
        # Match: x / (exp(-x) + 1) pattern used by inductor's silu decomposition
        return CallFunction(
            aten.div,
            computation_call,
            CallFunction(
                aten.add,
                CallFunction(aten.exp, CallFunction(aten.neg, computation_call)),
                1,
            ),
        )

    def _hardsigmoid_fusion(computation_call):
        # [FUNCTION DOC] HELPER: Pattern of hardsigmoid expressed as add, clamp and divide primitives.
        return CallFunction(
            aten.div,
            CallFunction(
                aten.clamp_max,
                CallFunction(
                    aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                ),
                6,
            ),
            6,
        )

    def _leaky_relu_fusion(computation_call):
        # [FUNCTION DOC] HELPER: Pattern of leaky relu expressed with a greater-than test, where and multiply with captured negative_slope.
        return CallFunction(
            aten.where,
            CallFunction(aten.gt, computation_call, 0),
            computation_call,
            CallFunction(aten.mul, computation_call, KeywordArg("negative_slope")),
        )

    def _hardtanh_fusion(computation_call):
        # [FUNCTION DOC] HELPER: Pattern of hardtanh as clamp min then clamp max with captured min_value and max_value.
        return CallFunction(
            aten.clamp_max,
            CallFunction(aten.clamp_min, computation_call, KeywordArg("min_value")),
            KeywordArg("max_value"),
        )

    def _combined_fusion(computation_call, elementwise_op):
        # [FUNCTION DOC] HELPER: Pattern applying the given elementwise op to the computation call.
        return CallFunction(elementwise_op, computation_call)

    # binary_op(other, computation_op)
    def _binary_fusion_v1(computation_call, binary_fn):
        # [FUNCTION DOC] HELPER: Pattern of a binary op with the captured 'other' as first operand and the computation as second.
        return CallFunction(binary_fn, KeywordArg("other"), computation_call)

    # binary_op(computation_op, other)
    def _binary_fusion_v2(computation_call, binary_fn):
        # [FUNCTION DOC] HELPER: Pattern of a binary op with the computation first and the captured 'other' second.
        return CallFunction(binary_fn, computation_call, KeywordArg("other"))

    def _is_single_computation_op(computation_op, lowp_dtype=None):
        # [FUNCTION DOC] HELPER: Builds a match check requiring at least one computation node, no already fused post-op (the third-from-last argument is none),
        # and, if a low precision dtype is given, that the match output has that dtype.
        def fn(match):
            # [FUNCTION DOC] HELPER: The check described above; returns True or False for a match.
            computation_nodes = filter_nodes(match.nodes, computation_op)

            if lowp_dtype:
                output_node_meta = match.output_node().meta.get("val")
                if output_node_meta.dtype != lowp_dtype:
                    return False

            if len(computation_nodes) < 1:
                return False
            if any(n.args[-3] != "none" for n in computation_nodes):
                return False
            return True

        return fn

    def _is_valid_computation_unary_fusion(computation_op, lowp_dtype=None):
        # [FUNCTION DOC] HELPER: Builds a match check for unary fusion: the single-computation check plus, for low precision, exactly two dtype conversion nodes
        # that go to float32 after the computation and to the low precision dtype at the end.
        def fn(match):
            # [FUNCTION DOC] HELPER: The check described above; returns True or False for a match.
            matched = _is_single_computation_op(computation_op, lowp_dtype)(match)
            computation_node = filter_nodes(match.nodes, computation_op)[0]
            if lowp_dtype:
                conversion_dtype_nodes = filter_nodes(
                    match.nodes, prims.convert_element_type.default
                )
                if len(conversion_dtype_nodes) != 2:
                    return False
                # fusion pattern is always in the form of computation_op + to_float32 + unary_op + to_bfloat16
                if computation_node == conversion_dtype_nodes[0].args[0]:
                    to_float = conversion_dtype_nodes[0].args[1]
                    to_lp = conversion_dtype_nodes[1].args[1]
                else:
                    to_float = conversion_dtype_nodes[1].args[1]
                    to_lp = conversion_dtype_nodes[0].args[1]
                matched = matched and to_float == torch.float and to_lp == lowp_dtype
            return matched

        return fn

    def _register_unary_fusion_lowering(
        pattern, unary_attr, computation_op, lowp_dtype=None
    ):
        # [FUNCTION DOC] HELPER: Registers a lowering pattern replacing computation plus unary activation with the computation op carrying the activation name,
        # scalars and algorithm as its last three arguments. Increments the unary fusion counters. Returns the registered replacement function.
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_unary_fusion(computation_op, lowp_dtype),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement handler: lowers the computation op with the unary attributes substituted for the final three arguments and updates counters.
            computation_args = list(args)[:-3] + [
                unary_attr.op_name,
                unary_attr.scalars_attr,
                unary_attr.algorithm_attr,
            ]
            counters["inductor"]["mkldnn_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"] += len(
                match.nodes
            )
            return L[computation_op](*computation_args)

        return fn

    def _register_leaky_relu_fusion_lowering(pattern, computation_op, lowp_dtype=None):
        # [FUNCTION DOC] HELPER: Registers a leaky relu fusion lowering for the computation op. Returns the registered handler.
        @register_lowering_pattern(
            pattern,
            extra_check=_is_single_computation_op(computation_op, lowp_dtype),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement handler. If the negative slope is a number (and low-precision dtypes match), lowers the computation with leaky relu attributes;
            # otherwise lowers the plain computation and applies the leaky relu with generic lowerings, converting dtypes around it for low precision.
            # KNOWN ISSUE: the fusion counters are incremented even when the unfused fallback is taken.
            negative_slope = kwargs.get("negative_slope")
            if isinstance(negative_slope, ir.TensorBox):
                matched = False
            else:  # inp is a Number
                matched = True
            if lowp_dtype:
                dtype1 = kwargs.get("to_float")
                dtype2 = (
                    kwargs.get("to_bf16")
                    if lowp_dtype == torch.bfloat16
                    else kwargs.get("to_fp16")
                )
                matched = matched and dtype1 == torch.float and dtype2 == lowp_dtype
            computation_args = list(args)
            counters["inductor"]["mkldnn_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"] += len(
                match.nodes
            )
            if matched:
                computation_args = computation_args[:-3] + [
                    "leaky_relu",
                    [negative_slope],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                # computation_args += ["none", [], ""]
                out = L[computation_op](*computation_args)
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                out = L[aten.where](
                    L[aten.gt](out, 0),
                    out,
                    L[aten.mul](out, negative_slope),
                )
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=dtype2)  # type: ignore[possibly-undefined]
                return out

        return fn

    def _register_hardtanh_fusion_lowering(pattern, computation_op, lowp_dtype=None):
        # [FUNCTION DOC] HELPER: Registers a hardtanh fusion lowering for the computation op. Returns the registered handler.
        @register_lowering_pattern(
            pattern,
            extra_check=_is_single_computation_op(computation_op, lowp_dtype),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement handler. Fuses hardtanh when min and max are numbers with min not above max (and low-precision dtypes match);
            # otherwise lowers the plain computation then applies clamps with generic lowerings, converting dtypes for low precision. Raises an error if max is missing.
            # KNOWN ISSUE: the fusion counters are incremented even when the unfused fallback is taken.
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")
            if isinstance(min_value, ir.TensorBox) or isinstance(
                max_value, ir.TensorBox
            ):
                matched = False
            else:  # inp is a Number
                if max_value is None:
                    raise AssertionError("expected max_value to not be None")
                matched = min_value <= max_value
            if lowp_dtype:
                dtype1 = kwargs.get("to_float")
                dtype2 = (
                    kwargs.get("to_bf16")
                    if lowp_dtype == torch.bfloat16
                    else kwargs.get("to_fp16")
                )
                matched = matched and dtype1 == torch.float and dtype2 == lowp_dtype
            computation_args = list(args)
            counters["inductor"]["mkldnn_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"] += len(
                match.nodes
            )
            if matched:
                computation_args = computation_args[:-3] + [
                    "hardtanh",
                    [min_value, max_value],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                out = L[computation_op](*computation_args)
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                out = L[aten.clamp_max](L[aten.clamp_min](out, min_value), max_value)
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=dtype2)  # type: ignore[possibly-undefined]
                return out

        return fn

    _binary_attr = {
        aten.add: "add",
        ops.add: "add",
        aten.sub: "sub",
        ops.sub: "sub",
    }

    def _is_valid_binary(match, computation_op, binary_op):
        # [FUNCTION DOC] HELPER: Validates a binary add/sub match: all binary nodes have tensor operands with metadata, alpha equal to one or absent, the 'other' operand
        # shape equal to or broadcastable to the computation output (linear: last dim; conv: channel patterns), same device and dtype, and the two operands differ.
        # Returns True or False. Raises an error if the computation node target is unexpected or conv output has fewer than three dims.
        binary_nodes = filter_nodes(match.nodes, binary_op)
        if len(binary_nodes) < 1:
            return False

        def get_meta_value(argument: torch.fx.node.Argument):
            # [FUNCTION DOC] HELPER: Returns the fake tensor metadata of a node argument, or None for non-nodes or missing metadata.
            # Only torch.fx.Node is expected to have meta.
            if isinstance(argument, torch.fx.Node):
                return argument.meta.get("val", None)
            return None

        if any(
            not isinstance(get_meta_value(n.args[0]), torch.Tensor)
            or not isinstance(get_meta_value(n.args[1]), torch.Tensor)
            for n in binary_nodes
        ):
            return False
        # check alpha is one.
        if any(
            get_arg_value(n, 2, kwarg_name="alpha") != 1.0
            and get_arg_value(n, 2, kwarg_name="alpha") is not None
            for n in binary_nodes
        ):
            return False

        def _check_input_sizes(n, computation_op):
            # [FUNCTION DOC] HELPER: Returns True if the shape of the match's 'other' operand equals the computation output shape or one of the allowed broadcast shapes.
            # Raises an error if the computation node does not use the expected op or (for conv) has fewer than three dims.
            # Check if the tensor shape of the 'other' node is the same as or
            # can be broadcasted to the tensor shape of the computation node.
            computation_node = (
                n.args[0] if n.args[1] is match.kwargs["other"] else n.args[1]
            )
            if computation_node.target != computation_op:
                raise AssertionError(
                    "expected computation_node.target to be "
                    f"{computation_op}, got {computation_node.target}"
                )
            computation_node_size = get_meta_value(computation_node).size()
            if computation_op is mkldnn._linear_pointwise.default:
                broadcast_sizes = []
                if len(computation_node_size) >= 2:
                    broadcast_sizes = [
                        torch.Size(
                            [1 for _ in range(len(computation_node_size) - 1)]
                            + [computation_node_size[-1]]
                        ),
                    ]
            else:
                if not len(computation_node_size) > 2:
                    raise AssertionError(
                        "expected len(computation_node_size) > 2, got "
                        f"{len(computation_node_size)}"
                    )
                broadcast_sizes = [
                    torch.Size(
                        [computation_node_size[0], computation_node_size[1]]
                        + [1 for _ in range(len(computation_node_size) - 2)]
                    ),
                    torch.Size(
                        [1, computation_node_size[1]]
                        + [1 for _ in range(len(computation_node_size) - 2)]
                    ),
                    torch.Size([1 for _ in range(len(computation_node_size))]),
                ]
            return (
                get_meta_value(match.kwargs["other"]).size()
                in [
                    computation_node_size,
                ]
                + broadcast_sizes
            )

        if any(
            not _check_input_sizes(n, computation_op)
            or get_meta_value(n.args[0]).device != get_meta_value(n.args[1]).device
            or get_meta_value(n.args[0]).dtype != get_meta_value(n.args[1]).dtype
            for n in binary_nodes
        ):
            return False
        # check args[0] and args[1] is not same
        if any(n.args[0] == n.args[1] for n in binary_nodes):
            return False
        return True

    def _is_valid_computation_binary(computation_op, binary_op, other_index=None):
        # [FUNCTION DOC] HELPER: Builds the match check for binary fusion: single computation op plus a valid binary.
        # KNOWN ISSUE: the other_index argument is unused.
        def fn(match):
            # [FUNCTION DOC] HELPER: The check described above; returns True or False.
            if not _is_single_computation_op(computation_op)(match):
                return False
            if not _is_valid_binary(match, computation_op, binary_op):
                return False
            return True

        return fn

    def _get_remaining_users(extra_input_node, compute_node):
        # [FUNCTION DOC] HELPER: Returns the users of the extra input node that are not ancestors of the compute node. Users that are ancestors have already finished
        # using the buffer, so only the remaining users matter when deciding whether an in-place update is safe.
        # Think about this pattern:
        #      ReLU
        #     /   \
        #  Conv1
        #   /      \
        # Conv2
        #   \      /
        #      Add
        # Although, the extra input node (ReLU) has more than 1 users: Conv1 and Add.
        # The Conv1 is the ancestor node of the current compute node (Conv2).
        # This indicates that the buffer of ReLU has completed all its usage,
        # So we can safely make changes to it now by doing Conv2->Add inplace fusion.
        # Take above case as example:
        # * extra_input_node: ReLU
        # * compute_node: Conv2
        # _get_remaining_users will return the users of extra_input_node which are not
        # ancestor node of compute_node.
        def _is_ancestor_node(_current_node, _ancestor_node):
            # [FUNCTION DOC] HELPER: Breadth-first search backward from the current node (through input nodes, stopping at placeholders, outputs and get_attr) returning True if the ancestor is found.
            # Check whether _ancestor_node is the ancestor node of _current_node
            _node_list = [_current_node]
            _visited_nodes = OrderedSet[torch.fx.Node]()
            while len(_node_list) != 0:
                _current_node = _node_list.pop(0)
                if _current_node not in _visited_nodes:
                    _visited_nodes.add(_current_node)
                    if _current_node == _ancestor_node:
                        return True
                    elif isinstance(
                        _current_node, torch.fx.Node
                    ) and _current_node.op not in ["placeholder", "output", "get_attr"]:
                        for input in _current_node.all_input_nodes:
                            _node_list.append(input)  # noqa: PERF402
            return False

        return [
            user
            for user in list(extra_input_node.users)
            if not _is_ancestor_node(compute_node, user)
        ]

    def _is_valid_computation_binary_inplace(computation_op, binary_op, other_index):
        # [FUNCTION DOC] HELPER: Builds the match check for in-place binary fusion: the normal binary check, and for every binary node the other operand has no remaining
        # user besides this one, is not the computation's own input, and is not a placeholder or output node.
        def fn(match):
            # [FUNCTION DOC] HELPER: The check described above; returns True or False.
            if not _is_valid_computation_binary(computation_op, binary_op)(match):
                return False
            binary_nodes = filter_nodes(match.nodes, binary_op)

            def _get_compute_node(_binary_node, _other_index):
                # [FUNCTION DOC] HELPER: Returns the compute operand (the one that is not at the other index) of a binary node; raises an error unless it has exactly two input nodes.
                if len(_binary_node.all_input_nodes) != 2:
                    raise AssertionError("Binary node should have 2 input nodes.")
                _compute_index = 1 if (_other_index == 0) else 0
                return _binary_node.args[_compute_index]

            def _other_input_not_inplaceable(_binary_node, _other_index):
                # [FUNCTION DOC] HELPER: Returns True if the other operand cannot be overwritten in place (more than one remaining user or it is the computation's input).
                _compute_node = _get_compute_node(_binary_node, _other_index)
                return (
                    len(
                        _get_remaining_users(
                            _binary_node.args[_other_index], _compute_node
                        )
                    )
                    > 1
                    or _binary_node.args[_other_index] == _compute_node.args[0]
                )

            if any(_other_input_not_inplaceable(n, other_index) for n in binary_nodes):
                return False
            if any(
                n.args[other_index].op in ["placeholder", "output"]
                for n in binary_nodes
            ):
                return False
            return True

        return fn

    def _register_binary_unary_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        fusion_op,
        unary_attr=None,
    ):
        # [FUNCTION DOC] HELPER: Registers a lowering pattern replacing computation plus binary op (plus optional unary activation) with the fused binary op.
        # Returns the registered handler.
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_binary(computation_op, binary_op),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement handler: requires the other operand to be a tensor box, reorders arguments to put it second, appends the binary attribute and
            # (for the long form) the unary attributes or defaults, updates counters and lowers the fusion op. Raises an error if other is not a tensor box.
            other = kwargs.get("other")
            if not isinstance(other, ir.TensorBox):
                raise AssertionError(
                    f"expected other to be ir.TensorBox, got {type(other)}"
                )
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_nodes"] += (
                len(match.nodes)
            )
            return L[fusion_op](*computation_args)

        return fn

    def _can_be_inplace(_other):
        # [FUNCTION DOC] HELPER: Returns True if the tensor box is not a view and does not alias an input of its producer, so it may be overwritten in place.
        return not (
            isinstance(_other.data, ir.BaseView)
            or len(_other.get_inputs_that_alias_output()) > 0
        )

    def _qlinear_binary_can_be_inplace(_other):
        # [FUNCTION DOC] HELPER: Returns True if a quantized-linear binary operand may be overwritten in place: for views, allowed when the base is a C++ template buffer or
        # the output of an earlier quantized linear binary op in the current graph operations; for non-views, when it aliases no input.
        if isinstance(_other.data, ir.BaseView):

            def unwrap_buffer(data):
                # [FUNCTION DOC] HELPER: Returns the underlying data of a storage box, or the input itself otherwise.
                if isinstance(data, ir.StorageBox):
                    return data.data
                return data

            data = _other.data.unwrap_view()
            if isinstance(unwrap_buffer(data), ir.CppTemplateBuffer):
                # It can be inplaced when _other is the 2D to 3D view of
                # a CppTemplateBuffer because if there is a view of CppTemplateBuffer,
                # CppTemplateBuffer will not be used directly but the view.
                return True
            else:
                # The case of QLinearPointwiseBinaryPT2E(sum) -> QLinearPointwiseBinaryPT2E(sum)
                # is similar to CppTemplateBuffer above.
                # The output of previous QLinearPointwiseBinaryPT2E is
                # the input x2 of current QLinearPointwiseBinaryPT2E.
                # Use V.graph.operations to check if _other is a view of the output
                # of previous QLinearPointwiseBinaryPT2E (the inputs[6]).
                for op in V.graph.operations:
                    if (
                        isinstance(op, mkldnn_ir.QLinearPointwiseBinaryPT2E)
                        and unwrap_buffer(data) == op.inputs[6]  # type: ignore[attr-defined]
                    ):
                        return True
            return False
        elif len(_other.get_inputs_that_alias_output()) > 0:
            return False
        else:
            return True

    def _register_binary_unary_maybe_inplace_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        inplace_fusion_op,
        outplace_fusion_op,
        unary_attr=None,
        other_index=None,
    ):
        # [FUNCTION DOC] HELPER: Registers a lowering for convolution plus add (optionally plus relu) that chooses the in-place or out-of-place fused op at lowering time.
        # Returns the registered handler.
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_binary_inplace(
                computation_op, binary_op, other_index
            ),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement handler: builds the fused arguments as in the plain binary fusion, realizes the other operand, and uses the out-of-place op if
            # it is a view, aliases something, or has a different shape than the output; otherwise uses the in-place op. Raises an error if other is not a tensor box.
            other = kwargs.get("other")
            if not isinstance(other, ir.TensorBox):
                raise AssertionError(
                    f"expected other to be ir.TensorBox, got {type(other)}"
                )
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_nodes"] += (
                len(match.nodes)
            )
            # Make sure the other is not an alias or mutation (fx side doesn't have such info).
            other.realize()
            if not _can_be_inplace(other) or other.data.shape != list(
                match.nodes[0].meta["val"].size()
            ):
                return L[outplace_fusion_op](*computation_args)
            return L[inplace_fusion_op](*computation_args)

        return fn

    computation_ops = [
        mkldnn._convolution_pointwise.default,
        mkldnn._linear_pointwise.default,
        mkldnn._convolution_transpose_pointwise.default,
    ]

    class UnaryAttr:
        # [CLASS DOC] HELPER: Holds the attributes of a fused unary post-op: op_name, scalars_attr (list, empty by default) and algorithm_attr (string, empty by default).
        def __init__(
            self, op_name: str, scalars_attr=None, algorithm_attr=None
        ) -> None:
            # [FUNCTION DOC] HELPER: Stores the op name and normalizes missing scalars and algorithm to an empty list and an empty string.
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""

    def _register_unary_fusion():
        # [FUNCTION DOC] HELPER: Registers pattern-based lowerings fusing conv, linear and transposed conv with unary activations (gelu, hardswish, hardsigmoid, swish,
        # relu, sigmoid, tanh, leaky relu, hardtanh) for bf16, fp16 and full precision variants. Side effect: registers patterns globally.
        computation_call_fns = [_conv_call, _linear_call, _conv_transpose_call]

        def _unary_fusion_patterns(lowp_dtype):
            # [FUNCTION DOC] HELPER: Returns a mapping from unary attribute objects to a list of three patterns (conv, linear, transposed conv) for the given low precision dtype;
            # relu, sigmoid and tanh are only included for the non-low-precision case.
            replacement_unary_fusion_patterns = {
                UnaryAttr("gelu", algorithm_attr="tanh"): [
                    _unary_fusion_pattern(_gelu_fusion_2, call_fn, 4, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("gelu", algorithm_attr="none"): [
                    _unary_fusion_pattern(_gelu_fusion_1, call_fn, 2, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("hardswish"): [
                    _unary_fusion_pattern(_hardswish_fusion, call_fn, 2, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("hardsigmoid"): [
                    _unary_fusion_pattern(_hardsigmoid_fusion, call_fn, 1, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("swish"): [
                    _unary_fusion_pattern(_silu_fusion, call_fn, 2, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
            }
            if not lowp_dtype:
                call_user1 = [call_fn(users=1) for call_fn in computation_call_fns]
                replacement_unary_fusion_patterns.update(
                    {
                        UnaryAttr("relu"): [
                            _combined_fusion(u, aten.relu) for u in call_user1
                        ],
                        UnaryAttr("sigmoid"): [
                            _combined_fusion(u, aten.sigmoid) for u in call_user1
                        ],
                        UnaryAttr("tanh"): [
                            _combined_fusion(u, aten.tanh) for u in call_user1
                        ],
                    }
                )

            return replacement_unary_fusion_patterns

        for lowp_dtype in [torch.bfloat16, torch.float16, None]:
            replace_patterns = _unary_fusion_patterns(lowp_dtype)
            for unary_attr, patterns in replace_patterns.items():
                _register_unary_fusion_lowering(
                    patterns[0], unary_attr, computation_ops[0], lowp_dtype
                )
                _register_unary_fusion_lowering(
                    patterns[1], unary_attr, computation_ops[1], lowp_dtype
                )
                _register_unary_fusion_lowering(
                    patterns[2], unary_attr, computation_ops[2], lowp_dtype
                )
            _leaky_relu_patterns = [
                _unary_fusion_pattern(_leaky_relu_fusion, call_fn, 3, lowp_dtype)
                for call_fn in computation_call_fns
            ]
            for pattern, computation_op in zip(_leaky_relu_patterns, computation_ops):
                _register_leaky_relu_fusion_lowering(
                    pattern, computation_op, lowp_dtype
                )
            hardtanh_patterns = [
                _unary_fusion_pattern(_hardtanh_fusion, call_fn, 1, lowp_dtype)
                for call_fn in computation_call_fns
            ]
            for pattern, computation_op in zip(hardtanh_patterns, computation_ops):
                _register_hardtanh_fusion_lowering(pattern, computation_op, lowp_dtype)

    def _register_inplace_fusion():
        # [FUNCTION DOC] HELPER: Registers lowerings for convolution plus add (both operand orders) with optional relu, using the in-place-capable fusion. Side effect: registers patterns globally.
        binary_ops = [aten.add, ops.add]
        inplace_fusion_op = mkldnn._convolution_pointwise_.binary
        outplace_fusion_op = mkldnn._convolution_pointwise.binary
        conv_call = _conv_call(users=1)
        conv_op = computation_ops[0]
        for binary_op in binary_ops:
            binary_v1 = _binary_fusion_v1(conv_call, binary_op)
            binary_unary_v1 = _combined_fusion(binary_v1, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
            )
            binary_v2 = _binary_fusion_v2(conv_call, binary_op)
            binary_unary_v2 = _combined_fusion(binary_v2, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
            )

    def _register_binary_fusion():
        # [FUNCTION DOC] HELPER: Registers lowerings fusing conv and linear with add or sub (and add with swapped operands) into the pointwise binary op. Side effect: registers patterns.
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [
            mkldnn._convolution_pointwise.binary,
            mkldnn._linear_pointwise.binary,
        ]
        _computation_user_1 = [_conv_call(users=1), _linear_call(users=1)]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern = _binary_fusion_v2(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

            for binary_op in [aten.add, ops.add]:
                pattern = _binary_fusion_v1(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

    def _register_binary_unary_fusion():
        # [FUNCTION DOC] HELPER: Registers lowerings fusing conv with add or sub followed by relu into the pointwise binary op with relu. Side effect: registers patterns.
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [mkldnn._convolution_pointwise.binary]
        _computation_user_1 = [_conv_call(users=1)]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern_v1 = _combined_fusion(
                    _binary_fusion_v2(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v1,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )
            for binary_op in [aten.add, ops.add]:
                pattern_v2 = _combined_fusion(
                    _binary_fusion_v1(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v2,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )

    def _recover_linear():
        # [FUNCTION DOC] HELPER: Registers two freezing-graph patterns (pass number 2): collapse reshape + linear + reshape into one linear when the reshapes are
        # redundant, and fold an added bias vector into the linear's bias argument. Side effect: registers patterns globally.
        # convert reshape+linear+reshape to a single linear for applying fusion path.
        # concat_linear (pass_number=0) -> mkldnn_linear_pack (pass_number=1) -> _recover_linear(pass_number=2)
        @register_freezing_graph_pattern(
            CallFunction(
                aten.reshape.default,
                CallFunction(
                    mkldnn._linear_pointwise.default,
                    CallFunction(
                        aten.reshape.default,
                        Arg(),
                        KeywordArg("reshape_1"),
                        _users=MULTIPLE,
                    ),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                ),
                KeywordArg("reshape_2"),
            ),
            pass_number=2,
        )
        def reshape_linear_reshape_pattern(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Pattern handler that replaces reshape, linear, reshape by a single linear call if the leading dims match the second reshape target and
            # their product equals the first reshape's first dim; erases the old nodes (and the first reshape if unused). Raises an error if reshape targets are not lists
            # or the first has other than two entries. Updates counters.
            def get_val(val):
                # [FUNCTION DOC] HELPER: Returns an int as is, or the metadata value of a node.
                return val if isinstance(val, int) else val.meta.get("val")

            reshape_1 = kwargs.get("reshape_1")
            reshape_2 = kwargs.get("reshape_2")
            if not isinstance(reshape_1, list):
                raise AssertionError(
                    f"expected reshape_1 to be list, got {type(reshape_1)}"
                )
            if not isinstance(reshape_2, list):
                raise AssertionError(
                    f"expected reshape_2 to be list, got {type(reshape_2)}"
                )
            if len(reshape_1) != 2:
                raise AssertionError(
                    f"expected len(reshape_1) == 2, got {len(reshape_1)}"
                )

            graph = match.graph
            reshape_2_node = match.output_node()
            linear_input_node = reshape_2_node.args[0].args[0].args[0]
            # check linear's input's shape[:-1] == reshape_2[:-1]
            # and check product(reshape_2[:-1]) == reshape_1[0]
            can_remove_reshape = linear_input_node.meta.get("val").shape[
                :-1
            ] == torch.Size([get_val(val) for val in reshape_2[:-1]])
            can_remove_reshape = can_remove_reshape and (
                reduce(
                    operator.mul,
                    [get_val(val) for val in reshape_2[:-1]],
                )
                == get_val(reshape_1[0])
            )

            if can_remove_reshape:
                repl = graph.call_function(mkldnn._linear_pointwise.default, args)
                repl.meta.update(reshape_2_node.meta)
                reshape_2_node.replace_all_uses_with(repl)
                old_linear_node = reshape_2_node.args[0]
                reshape_1_node = old_linear_node.args[0]
                graph.erase_node(reshape_2_node)
                graph.erase_node(old_linear_node)
                if len(reshape_1_node.users) == 0:
                    graph.erase_node(reshape_1_node)
            counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_count"] += 1
            counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"] += len(
                match.nodes
            )

        def is_linear_add_bias(match):
            # [FUNCTION DOC] HELPER: Match check for linear plus add bias: bias must be a node with metadata, same dtype as the weight, one-dimensional with length equal to the
            # weight output features, and the linear must have no bias already.
            add_node = match.output_node()
            linear_node = add_node.args[0]
            device_type = add_node.meta.get("val").device.type
            mkldnn_device_op = _get_mkldnn_device_op(device_type)
            transpose_weight_node = mkldnn_device_op.get_linear_transpose_weight(
                linear_node.args[1]
            )
            weight_meta = transpose_weight_node.args[0].meta.get("val")
            bias_node = add_node.args[1]
            if not isinstance(bias_node, torch.fx.Node):
                return False
            bias_meta = bias_node.meta.get("val")
            if weight_meta is None or bias_meta is None:
                return False

            if bias_meta.dtype != weight_meta.dtype:
                return False
            return (
                linear_node.args[2] is None
                and bias_meta.dim() == 1
                and bias_meta.size(0) == weight_meta.size(1)
            )

        # convert linear+bias to a single linear for applying fusion path.
        @register_freezing_graph_pattern(
            CallFunction(
                aten.add.Tensor,
                CallFunction(mkldnn._linear_pointwise.default, *_linear_args),
                Arg(),
            ),
            pass_number=2,
            extra_check=is_linear_add_bias,
        )
        def linear_bias_pattern(match, *args):
            # [FUNCTION DOC] HELPER: Pattern handler that creates a linear with the added tensor as bias, replaces the add, erases the matched nodes and updates counters.
            graph = match.graph
            add_node = match.output_node()
            linear_node = add_node.args[0]
            new_args = list(linear_node.args)
            new_args[2] = add_node.args[1]
            repl = graph.call_function(
                mkldnn._linear_pointwise.default, tuple(new_args)
            )
            repl.meta.update(add_node.meta)
            add_node.replace_all_uses_with(repl)
            match.erase_nodes()
            counters["inductor"]["mkldnn_linear_bias_matcher_count"] += 1
            counters["inductor"]["mkldnn_linear_bias_matcher_nodes"] += len(match.nodes)

    def _is_packable_mkldnn_rnn_layer(match):
        # [FUNCTION DOC] HELPER: Match check for packing an mkldnn rnn layer: weights are constants, metadata exists, all on CPU, and bf16/fp16 supported by hardware when used.
        lstm_node = match.output_node()
        POS_WEIGHTS = [1, 2]
        POS_INPUTS = [0, 5, 6]
        POS_ARGS = POS_WEIGHTS + POS_INPUTS
        # Weights should be Constant
        if any(
            lstm_node.args[POS_WEIGHT].op != "get_attr" for POS_WEIGHT in POS_WEIGHTS
        ):
            return False

        # Meta info for weights and inputs should be available
        if any(lstm_node.args[POS_ARG].meta.get("val") is None for POS_ARG in POS_ARGS):
            return False

        # Check device
        if any(
            lstm_node.args[POS_ARG].meta.get("val").device.type != "cpu"
            for POS_ARG in POS_ARGS
        ):
            return False

        # Check dtype
        if any(
            lstm_node.args[POS_ARG].meta.get("val").dtype == torch.bfloat16
            and not is_mkldnn_bf16_supported("cpu")
            for POS_ARG in POS_ARGS
        ):
            return False
        if any(
            lstm_node.args[POS_ARG].meta.get("val").dtype == torch.float16
            and not is_mkldnn_fp16_supported("cpu")
            for POS_ARG in POS_ARGS
        ):
            return False

        return True

    def _is_packable_convolution(match):
        # [FUNCTION DOC] HELPER: Match check that a convolution can be packed for mkldnn: supported device and rank (4 or 5 dims), constant weight, supported low precision,
        # and for transposed convs a static shape, no depthwise groups and output padding smaller than stride. Returns True or False.
        """
        Check if the node is supported for MKLDNN convolution.
        """
        conv_node = match.output_node()
        device_type = conv_node.meta.get("val").device.type
        # The operator 'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device.
        if match.kwargs["is_transposed"] and device_type == "xpu":
            return False

        input_meta_value = conv_node.args[0].meta.get("val")
        weight_meta_value = conv_node.args[1].meta.get("val")
        if input_meta_value is None or weight_meta_value is None:
            return False
        input_size = input_meta_value.shape
        if conv_node.args[1].op != "get_attr":
            return False
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type not in SUPPORTED_MKLDNN_DEVICES
                or (meta_value.dim() != 4 and meta_value.dim() != 5)
            ):
                return False

        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not is_mkldnn_bf16_supported(device_type):
                return False
        if (
            input_meta_value.dtype == torch.float16
            or weight_meta_value.dtype == torch.float16
        ):
            if not is_mkldnn_fp16_supported(device_type):
                return False
        is_transposed = conv_node.args[-3]
        if is_transposed:
            # TODO: Support dynamic shape case for MKLDNN conv transpose.
            if has_free_symbols(input_size):
                return False
            groups = conv_node.args[-1]
            in_channels = weight_meta_value.size(0)
            # doesn't support group_depthwise_conv_transpose.
            if groups > 1 and groups == in_channels:
                return False
            # Port from: aten/src/ATen/native/Convolution.cpp:is_output_padding_big
            output_paddings = conv_node.args[-2]
            strides = conv_node.args[3]
            if any(
                output_padding >= stride
                for output_padding, stride in zip(output_paddings, strides)
            ):
                return False
        return True

    def _is_packable_linear(match):
        # [FUNCTION DOC] HELPER: Match check that an mm or addmm can be packed for mkldnn linear: alpha 1 and beta 0 or 1, constant (or concatenated constant) weight, 2D CPU tensors,
        # not float64, a bias that is 1D and matching, MKL or ACL available for full precision, and supported low precision. Returns True or False.
        # KNOWN ISSUE: the bias device check reads the device of the loop variable left from the input/weight loop, not of the bias.
        """
        Check if the node is supported for MKLDNN linear.
        """

        def is_const_or_cat_by_const(weight):
            # [FUNCTION DOC] HELPER: Returns True if the weight node is a constant or a concatenation of constants.
            if weight.op == "get_attr":
                return True
            if weight.target != aten.cat.default:
                return False
            return all(arg.op == "get_attr" for arg in weight.args[0])

        linear_node = match.output_node()
        # mkldnn linear only supports beta=1 or 0 and alpha=1
        if linear_node.target is aten.addmm.default:
            alpha = linear_node.kwargs.get("alpha", 1.0)
            beta = linear_node.kwargs.get("beta", 1.0)
            if (beta != 0.0 and beta != 1.0) or alpha != 1.0:
                return False
        # weight_idx is 1 for aten.mm and is 2 for aten.addmm
        weight_idx = 2 if linear_node.target is aten.addmm.default else 1
        if not is_const_or_cat_by_const(linear_node.args[weight_idx]):
            return False
        input_meta_value = linear_node.args[weight_idx - 1].meta.get("val")
        weight_meta_value = linear_node.args[weight_idx].meta.get("val")
        if input_meta_value is None or weight_meta_value is None:
            return False
        if (
            input_meta_value.dtype == torch.float64
            or weight_meta_value.dtype == torch.float64
        ):
            return False
        is_lp_weight = weight_meta_value.dtype in (
            torch.bfloat16,
            torch.float16,
        )
        reduced_f32_matmul_enabled = torch.backends.mkldnn.matmul.fp32_precision in [  # type: ignore[attr-defined]
            "bf16",
            "tf32",
        ]
        use_reduced_f32_for_fp32_weight = (
            reduced_f32_matmul_enabled and weight_meta_value.dtype == torch.float32
        )
        compute_with_lp = is_lp_weight or use_reduced_f32_for_fp32_weight
        # on x86, for fp32, mkl should be enabled.
        # on aarch64, use mkldnn op for fp32 as well if acl is enabled
        if (
            not compute_with_lp
            and not mkldnn._is_mkldnn_acl_supported()
            and not torch._C.has_mkl
        ):
            return False
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type != "cpu"
                or meta_value.dim() != 2
            ):
                return False
        if weight_idx == 2:
            bias_meta_value = linear_node.args[0].meta.get("val")
            if (
                bias_meta_value is None
                or meta_value.device.type != "cpu"
                or bias_meta_value.dim() != 1
                or bias_meta_value.size(0) != weight_meta_value.size(1)
            ):
                return False

        device_type = input_meta_value.device.type
        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not is_mkldnn_bf16_supported(device_type):
                return False
        if (
            input_meta_value.dtype == torch.float16
            or weight_meta_value.dtype == torch.float16
        ):
            if not is_mkldnn_fp16_supported(device_type):
                return False
        return True

    _aten_conv_args = (
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        KeywordArg("is_transposed"),
        Arg(),
        Arg(),
    )

    _aten_mkldnn_rnn_layer_args = (
        Arg(),  # input
        Arg(),  # weight0
        Arg(),  # weight1
        Arg(),  # weight2
        Arg(),  # weight3
        Arg(),  # hx_
        Arg(),  # cx_
        KeywordArg("reverse"),  # reverse
        Arg(),  # batch_sizes
        Arg(),  # mode
        Arg(),  # hidden_size
        Arg(),  # num_layers
        Arg(),  # has_biases
        Arg(),  # bidirectional
        Arg(),  # batch_first
        Arg(),  # train
    )

    def _register_weight_pack_pass():
        # [FUNCTION DOC] HELPER: Registers freezing-graph patterns that replace convolution (pass 0 default), mkldnn rnn layer and mm/addmm (pass 1) with packed-weight
        # versions using the device strategy. Side effect: registers patterns globally.
        @register_freezing_graph_pattern(
            CallFunction(aten.convolution.default, *_aten_conv_args),
            extra_check=_is_packable_convolution,
        )
        def convolution(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Pattern handler: packs the convolution weight (at runtime for symbolic shapes, which must be non-transposed) and replaces the node with the pointwise convolution op
            # (no fused post-op); updates counters. Raises an error if the transposed flag is not a bool.
            is_transposed = kwargs.get("is_transposed")
            if not isinstance(is_transposed, bool):
                raise AssertionError(
                    f"expected is_transposed to be bool, got {type(is_transposed)}"
                )
            graph = match.graph
            conv_node = match.output_node()
            device_type = conv_node.args[0].meta.get("val").device.type
            mkldnn_device_op = _get_mkldnn_device_op(device_type)
            input_size = conv_node.args[0].meta.get("val").shape
            with graph.inserting_before(conv_node):
                constant_args = [args[4], args[3], args[5], args[-1]]
                packed_conv_op = mkldnn._convolution_pointwise.default
                if is_transposed:
                    constant_args.insert(1, args[-2])  # output_padding
                    packed_conv_op = mkldnn._convolution_transpose_pointwise.default

                if not has_free_symbols(input_size):
                    packed_weight_node = mkldnn_device_op.pack_conv_weight(
                        graph,
                        is_transposed,
                        args[1],
                        constant_args,
                        input_size,
                    )
                else:
                    if is_transposed:
                        raise AssertionError("expected is_transposed to be False")
                    # For dynamic shape case, we need to pack weight in runtime.
                    packed_weight_node = args[1]

                packed_conv_inputs = (
                    (args[0], packed_weight_node, args[2])
                    + tuple(constant_args)
                    + ("none", [], "")
                )
                packed_conv_node = graph.create_node(
                    "call_function", packed_conv_op, tuple(packed_conv_inputs)
                )
                conv_node.replace_all_uses_with(packed_conv_node)
                packed_conv_node.meta.update(conv_node.meta)
                graph.erase_node(conv_node)
            counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"] += 1
            counters["inductor"]["mkldnn_conv_weight_pack_matcher_nodes"] += len(
                match.nodes
            )

        @register_freezing_graph_pattern(
            CallFunction(aten.mkldnn_rnn_layer.default, *_aten_mkldnn_rnn_layer_args),
            extra_check=_is_packable_mkldnn_rnn_layer,
        )
        def mkldnn_rnn_layer(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Pattern handler: inserts a weight reorder node for the two rnn weights, unpacks its two outputs, and replaces the rnn layer with a call using them; updates counters.
            def get_item(graph, node, index):
                # [FUNCTION DOC] HELPER: Creates a node extracting one item of a multi-output node.
                return graph.call_function(operator.getitem, (node, index))

            graph = match.graph
            lstm_node = match.output_node()
            weight0, weight1 = args[1:3]
            reverse = kwargs.get("reverse")
            packed_lstm_op = aten.mkldnn_rnn_layer.default
            hidden_size = args[9]
            has_biases = args[11]
            batch_first = args[13]
            with graph.inserting_before(lstm_node):
                packed_weight_op = mkldnn._reorder_mkldnn_rnn_layer_weight.default
                packed_weight_inputs = (
                    weight0,
                    weight1,
                    hidden_size,
                    reverse,
                    has_biases,
                    batch_first,
                )
                packed_weight_node = graph.create_node(
                    "call_function", packed_weight_op, packed_weight_inputs, {}, "name"
                )
                packed_weight_items = [
                    get_item(graph, packed_weight_node, i) for i in range(2)
                ]
                pack_lstm_inputs = (
                    args[0],
                    *packed_weight_items,
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                    reverse,
                    *args[7:],
                )

                packed_lstm_node = graph.create_node(
                    "call_function", packed_lstm_op, args=pack_lstm_inputs
                )
                lstm_node.replace_all_uses_with(packed_lstm_node)
                packed_lstm_node.meta.update(lstm_node.meta)
                graph.erase_node(lstm_node)
            counters["inductor"]["mkldnn_rnn_weight_pack_matcher_count"] += 1
            counters["inductor"]["mkldnn_rnn_weight_pack_matcher_nodes"] += len(
                match.nodes
            )

        @register_freezing_graph_pattern(
            CallFunction(
                aten.addmm.default,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            extra_check=_is_packable_linear,
            pass_number=1,
        )
        @register_freezing_graph_pattern(
            CallFunction(aten.mm.default, Arg(), Arg()),
            extra_check=_is_packable_linear,
            pass_number=1,
        )
        def linear(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Pattern handler: replaces mm or addmm with permute of the weight, packed weight and packed linear nodes created by the device strategy (low precision
            # also when reduced fp32 matmul precision is enabled); bias is dropped when the target is mm or beta is zero. Updates counters.
            graph = match.graph
            linear_node = match.output_node()
            input = args[0] if linear_node.target is aten.mm.default else args[1]
            bias = (
                None
                if linear_node.target is aten.mm.default
                or (
                    linear_node.target is aten.addmm.default
                    and linear_node.kwargs.get("beta", 1.0) == 0.0
                )
                else args[0]
            )
            weight = args[1] if linear_node.target is aten.mm.default else args[2]
            device_type = input.meta.get("val").device.type
            mkldnn_device_op = _get_mkldnn_device_op(device_type)
            with graph.inserting_before(linear_node):
                transpose_weight_node = graph.create_node(
                    "call_function", aten.permute.default, (weight, (1, 0))
                )
                weight_dtype = weight.meta.get("val").dtype
                is_lp_weight = weight_dtype in (
                    torch.bfloat16,
                    torch.float16,
                )
                reduced_f32_matmul_enabled = (
                    torch.backends.mkldnn.matmul.fp32_precision in ["bf16", "tf32"]  # type: ignore[attr-defined]
                )
                use_reduced_f32_for_fp32_weight = (
                    reduced_f32_matmul_enabled and weight_dtype == torch.float32
                )
                compute_with_lp = is_lp_weight or use_reduced_f32_for_fp32_weight
                batch_size = input.meta.get("val").shape[0]
                packed_weight_node = mkldnn_device_op.pack_linear_weight(
                    graph, compute_with_lp, transpose_weight_node, batch_size
                )
                packed_linear_node = mkldnn_device_op.pack_linear(
                    graph, compute_with_lp, batch_size, input, packed_weight_node, bias
                )

                linear_node.replace_all_uses_with(packed_linear_node)
                packed_linear_node.meta.update(linear_node.meta)
                graph.erase_node(linear_node)
            counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"] += 1
            counters["inductor"]["mkldnn_linear_weight_pack_matcher_nodes"] += len(
                match.nodes
            )

    def _eliminate_duplicate_packed_nodes(gm):
        # [FUNCTION DOC] HELPER: Graph pass that merges identical packed-weight reorder nodes sharing the same input and arguments, redirecting uses to the first and erasing duplicates.
        # Returns the graph module unchanged if mkldnn is disabled. Mutates the graph in place.
        """
        Combine packed weight nodes with the same inputs to reduce memory usage.
        for example:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(32, 32, bias=True)

            def forward(self, x):
                return self.linear(self.linear(x))

        the above's packed weight nodes are duplicate if two linear calls have same input size.
        """
        if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
            return gm

        packed_weight_ops = [
            torch._C._nn.mkldnn_reorder_conv2d_weight,
            torch._C._nn.mkldnn_reorder_conv3d_weight,
            mkldnn._reorder_convolution_transpose_weight,
            mkldnn._reorder_linear_weight,
            mkldnn._reorder_mkldnn_rnn_layer_weight,
        ]
        if torch._C.has_mkl:
            packed_weight_ops.append(torch.ops.mkl._mkl_reorder_linear_weight)

        for node in gm.graph.nodes:
            if node.target in packed_weight_ops and len(node.args[0].users) > 1:
                for user_node in list(node.args[0].users.keys()):
                    if (
                        user_node.target == node.target
                        and user_node != node
                        and user_node.args == node.args
                    ):
                        user_node.replace_all_uses_with(node)
                        gm.graph.erase_node(user_node)

    @functools.cache
    def _mkldnn_fusion_init():
        # [FUNCTION DOC] HELPER: Cached one-time initializer registering unary, inplace, binary and quantization lowerings (skipped when ACL is supported) and weight-only-quantization
        # lowerings. Does nothing if mkldnn is disabled. Side effect: global pattern registration.
        # TODO: aarch64: enable op fusion for acl once it supports fused operators. Disabling it for now.
        # Otherwise even the matmul or innerproduct can not be accelerated with acl
        if (
            not torch.backends.mkldnn.enabled
            or not torch.backends.mkldnn.is_available()
        ):
            return

        if not torch.ops.mkldnn._is_mkldnn_acl_supported():
            _register_unary_fusion()
            _register_inplace_fusion()
            _register_binary_unary_fusion()
            _register_binary_fusion()
            _register_quantization_lowerings()

        _register_woq_lowerings()

    @functools.cache
    def _mkldnn_weight_pack_init():
        # [FUNCTION DOC] HELPER: Cached one-time initializer registering weight packing patterns, linear recovery patterns and int8 weight-only concat linear patterns when mkldnn is available.
        if torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
            _register_weight_pack_pass()
            _recover_linear()
            _register_int8_woq_concat_linear_pattern()
