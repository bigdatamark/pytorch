# [MODULE DOC] Weight extraction helpers for the FX numeric suite. Contains small functions that
# pull a weight tensor out of a module (float, QAT, quantized, dynamic, LSTM) or out of a
# function-call graph node (float or quantized conv/linear), a factory building the table of
# node op -> module/function type -> extraction function, and extract_weight_from_node which
# packages the result in the numeric suite single-result record format.
# Import-time side effect: binds toq to the quantized ops namespace.
from collections.abc import Callable

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule
from torch.fx.graph import Node

from .ns_types import NSSingleResultType, NSSingleResultValuesType
from .utils import get_target_type_str, getattr_from_fqn, return_first_non_observer_node


toq = torch.ops.quantized


def mod_weight_detach(mod: nn.Module) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Return the detached weight attribute of a module.
    return mod.weight.detach()  # type: ignore[operator]


def mod_0_weight_detach(mod: nn.Module) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Return the detached weight of the first submodule (index 0) of a fused module.
    return mod[0].weight.detach()  # type: ignore[index]


def mod_weight_bias_0(mod: nn.Module) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Return the first element (the weight) of a quantized module's weight-and-bias pair.
    return mod._weight_bias()[0]  # type: ignore[operator]


def get_lstm_weight(mod: nn.Module) -> list[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Return a list of detached input-hidden and hidden-hidden weight tensors of a
    # float LSTM, selected by parameter name from its flat weights, in storage order.
    res = []
    for idx, param_name in enumerate(mod._flat_weights_names):  # type: ignore[arg-type]
        if "weight_ih_l" in param_name or "weight_hh_l" in param_name:
            param_value = mod._flat_weights[idx].detach()  # type: ignore[index,union-attr]
            res.append(param_value)
    return res


def get_qlstm_weight(mod: nn.Module) -> list[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Return the weights of a dynamic quantized LSTM as a list: for each layer the
    # two unpacked weight tensors, obtained by digging into the packed parameter state.
    res = []
    for weight_value in mod._all_weight_values:  # type: ignore[union-attr]
        res.append(weight_value.param.__getstate__()[0][4][0].__getstate__()[0][0])
        res.append(weight_value.param.__getstate__()[0][4][1].__getstate__()[0][0])
    return res


def get_conv_mod_weight(mod: nn.Module) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Weight of a conv module: detached weight for float convs, detached weight of
    # the first child for conv-relu fused modules, otherwise the first item of the quantized weight-and-bias pair.
    if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return mod.weight.detach()
    elif isinstance(mod, (nni.ConvReLU1d, nni.ConvReLU2d, nni.ConvReLU3d)):
        return mod[0].weight.detach()  # type: ignore[operator]
    else:
        return mod._weight_bias()[0]  # type: ignore[operator]


def get_linear_mod_weight(mod: nn.Module) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Weight of a linear module: detached weight for a float linear, of the first
    # child for a linear-relu fused module, otherwise the first item of the quantized weight-and-bias pair.
    if isinstance(mod, nn.Linear):
        return mod.weight.detach()
    elif isinstance(mod, nni.LinearReLU):
        return mod[0].weight.detach()  # type: ignore[operator]
    else:
        return mod._weight_bias()[0]  # type: ignore[operator]


def get_lstm_mod_weights(mod: nn.Module) -> list[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Return list of LSTM weights for a float LSTM (detached ih/hh weights) or a
    # dynamic quantized LSTM (unpacked weights). Raises an error for any other module type.
    # TODO(future PR): make more generic, handle everything
    if isinstance(mod, nn.LSTM):
        res = []
        for idx, param_name in enumerate(mod._flat_weights_names):
            if "weight_ih_l" in param_name or "weight_hh_l" in param_name:
                param_value = mod._flat_weights[idx].detach()  # type: ignore[index,union-attr]
                res.append(param_value)
        return res
    else:
        if not isinstance(mod, nnqd.LSTM):
            raise AssertionError(f"type {type(mod)} not handled yet")
        res = []
        for weight_value in mod._all_weight_values:
            res.append(
                weight_value.param.__getstate__()[0][4][0].__getstate__()[0][0]  # type: ignore[index]
            )
            res.append(
                weight_value.param.__getstate__()[0][4][1].__getstate__()[0][0]  # type: ignore[index]
            )
        return res


def get_conv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Weight of a functional conv node: follows the second arg back through
    # observers to a get_attr node, looks the attribute up on gm and returns it detached.
    # Raises an error if the structure is not as expected.
    # traverse backwards from the weight arg, accounting for any observers
    weight_arg_node = node.args[1]
    if not isinstance(weight_arg_node, Node):
        raise AssertionError(f"Expected Node, got {type(weight_arg_node)}")
    weight_node = return_first_non_observer_node(weight_arg_node, gm)
    if not isinstance(weight_node, Node):
        raise AssertionError(f"Expected Node, got {type(weight_node)}")
    if weight_node.op != "get_attr":
        raise AssertionError(f"Expected get_attr, got {weight_node.op}")
    weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
    return weight.detach()


def get_qconv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Weight of a quantized functional conv node: the second arg must be a get_attr
    # node holding a packed conv state object; returns the result of calling its weight method.
    # qconv state is arg 1
    qconv_state_node = node.args[1]
    if not isinstance(qconv_state_node, Node):
        raise AssertionError(f"Expected Node, got {type(qconv_state_node)}")
    if qconv_state_node.op != "get_attr":
        raise AssertionError(f"Expected get_attr, got {qconv_state_node.op}")
    qconv_state_obj = getattr_from_fqn(gm, qconv_state_node.target)  # type: ignore[arg-type]
    return qconv_state_obj.weight()


def get_linear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Weight of a functional linear node. Supported weight patterns: via an
    # observer module call (reads the observer's input attribute), via cast to a dtype then dequantize
    # (returns the attribute detached and cast to that dtype), or a direct get_attr. Raises an error otherwise.
    # traverse backwards from the weight arg, accounting for any observers
    # supported patterns:
    # weight -> obs -> linear
    # weight -> to(torch.float16) -> dequantize -> linear
    linear_second_arg = node.args[1]
    if not isinstance(linear_second_arg, Node):
        raise AssertionError(f"Expected Node, got {type(linear_second_arg)}")

    if linear_second_arg.op == "call_module":
        # weight -> obs -> linear
        weight_arg_node = node.args[1]
        if not isinstance(weight_arg_node, Node):
            raise AssertionError(f"Expected Node, got {type(weight_arg_node)}")
        weight_node = weight_arg_node.args[0]
        if not isinstance(weight_node, Node):
            raise AssertionError(f"Expected Node, got {type(weight_node)}")
        if weight_node.op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {weight_node.op}")
        weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
        return weight.detach()
    elif linear_second_arg.op == "call_method":
        # weight -> to(torch.float16) -> dequantize -> linear
        if linear_second_arg.op != "call_method":
            raise AssertionError(f"Expected call_method, got {linear_second_arg.op}")
        dequant_node = node.args[1]
        if not isinstance(dequant_node, Node):
            raise AssertionError(f"Expected Node, got {type(dequant_node)}")
        to_fp16_node = dequant_node.args[0]
        if not isinstance(to_fp16_node, Node):
            raise AssertionError(f"Expected Node, got {type(to_fp16_node)}")
        # extract the dtype, so we can cast to it before returning
        target_dtype = to_fp16_node.args[1]
        weight_node = to_fp16_node.args[0]
        if not isinstance(weight_node, Node):
            raise AssertionError(f"Expected Node, got {type(weight_node)}")
        if weight_node.op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {weight_node.op}")
        weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
        # return the weight with fp16 cast
        return weight.detach().to(target_dtype)
    else:
        if linear_second_arg.op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {linear_second_arg.op}")
        weight = getattr_from_fqn(gm, linear_second_arg.target)  # type: ignore[arg-type]
        return weight.detach()


def get_qlinear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Weight of a quantized functional linear node: the second arg must be a
    # get_attr node holding the packed params; the weight is read from its serialized state.
    # packed weight is arg 1
    packed_weight_node = node.args[1]
    if not isinstance(packed_weight_node, Node):
        raise AssertionError(f"Expected Node, got {type(packed_weight_node)}")
    if packed_weight_node.op != "get_attr":
        raise AssertionError(f"Expected get_attr, got {packed_weight_node.op}")
    packed_weight = getattr_from_fqn(gm, packed_weight_node.target)  # type: ignore[arg-type]
    # TODO(future PR): why does packed_weight.unpack() not work?
    (weight, _bias), _name = packed_weight.__getstate__()
    return weight


def get_op_to_type_to_weight_extraction_fn() -> dict[str, dict[Callable, Callable]]:
    # [FUNCTION DOC] Build and return a fresh nested mapping: node op ("call_module" or "call_function")
    # to a mapping from module class / function to the weight extraction function for it. Covers
    # conv 1d/2d/3d and linear in float, fused, QAT, quantized and dynamic variants, plus LSTM.
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]] = {
        "call_module": {
            # Conv1d
            nn.Conv1d: mod_weight_detach,
            nni.ConvReLU1d: mod_0_weight_detach,
            nnq.Conv1d: mod_weight_bias_0,
            nnqat.Conv1d: mod_weight_detach,
            nniqat.ConvBn1d: mod_weight_detach,
            nniqat.ConvBnReLU1d: mod_weight_detach,
            nniqat.ConvReLU1d: mod_weight_detach,
            nniq.ConvReLU1d: mod_weight_bias_0,
            # Conv2d
            nn.Conv2d: mod_weight_detach,
            nni.ConvReLU2d: mod_0_weight_detach,
            nnq.Conv2d: mod_weight_bias_0,
            nnqat.Conv2d: mod_weight_detach,
            nniqat.ConvBn2d: mod_weight_detach,
            nniqat.ConvBnReLU2d: mod_weight_detach,
            nniqat.ConvReLU2d: mod_weight_detach,
            nniq.ConvReLU2d: mod_weight_bias_0,
            # Conv3d
            nn.Conv3d: mod_weight_detach,
            nni.ConvReLU3d: mod_0_weight_detach,
            nnq.Conv3d: mod_weight_bias_0,
            nnqat.Conv3d: mod_weight_detach,
            nniqat.ConvBn3d: mod_weight_detach,
            nniqat.ConvBnReLU3d: mod_weight_detach,
            nniqat.ConvReLU3d: mod_weight_detach,
            nniq.ConvReLU3d: mod_weight_bias_0,
            # Linear
            nn.Linear: mod_weight_detach,
            nnq.Linear: mod_weight_bias_0,
            nni.LinearReLU: mod_0_weight_detach,
            nniq.LinearReLU: mod_weight_bias_0,
            nnqat.Linear: mod_weight_detach,
            nnqd.Linear: mod_weight_bias_0,
            nniqat.LinearReLU: mod_weight_detach,
            nniqat.LinearBn1d: mod_weight_detach,
            nn.modules.linear.NonDynamicallyQuantizableLinear: mod_weight_detach,
            # LSTM
            nn.LSTM: get_lstm_weight,
            nnqd.LSTM: get_qlstm_weight,
        },
        "call_function": {
            # Conv
            F.conv1d: get_conv_fun_weight,
            F.conv2d: get_conv_fun_weight,
            F.conv3d: get_conv_fun_weight,
            toq.conv1d: get_qconv_fun_weight,
            toq.conv2d: get_qconv_fun_weight,
            toq.conv3d: get_qconv_fun_weight,
            toq.conv1d_relu: get_qconv_fun_weight,
            toq.conv2d_relu: get_qconv_fun_weight,
            toq.conv3d_relu: get_qconv_fun_weight,
            # Linear
            F.linear: get_linear_fun_weight,
            toq.linear: get_qlinear_fun_weight,
            toq.linear_relu: get_qlinear_fun_weight,
        },
    }

    return op_to_type_to_weight_extraction_fn


def extract_weight_from_node(
    node: Node,
    gm: GraphModule,
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]]
    | None = None,
) -> NSSingleResultType | None:
    # [FUNCTION DOC] Extract the weight of a call_function or call_module node as a single-result record.
    # Uses the given extraction table or builds the default one. Function targets are matched by
    # equality, module types by exact type. Returns a mapping with the weight type tag, values list
    # holding the weight, prev/ref node name and target type string (same node), arg indices of 0,
    # and fqn (from gm's node scope table if present, else None). Returns None if no rule matches.
    # Raises an error if a module target is not a string.
    res_type = NSSingleResultValuesType.WEIGHT.value

    # Not all graphmodules have _node_name_to_scope, so only fill it
    # out if it exists.
    fqn = None
    if hasattr(gm, "_node_name_to_scope"):
        fqn = gm._node_name_to_scope[node.name][0]  # type: ignore[index]

    if op_to_type_to_weight_extraction_fn is None:
        op_to_type_to_weight_extraction_fn = get_op_to_type_to_weight_extraction_fn()

    ref_node_type = get_target_type_str(node, gm)
    # for extracting weights, these are always the same
    prev_node_type = ref_node_type

    if node.op == "call_function":
        function_mapping = op_to_type_to_weight_extraction_fn["call_function"]
        for target_fn_type, weight_extraction_fn in function_mapping.items():
            if node.target == target_fn_type:
                weight = weight_extraction_fn(node, gm)
                return {
                    "type": res_type,
                    "values": [weight],
                    "prev_node_name": node.name,
                    "prev_node_target_type": prev_node_type,
                    "ref_node_name": node.name,
                    "ref_node_target_type": ref_node_type,
                    "index_within_arg": 0,
                    "index_of_arg": 0,
                    "fqn": fqn,
                }

    elif node.op == "call_module":
        # for call_module, we need to look up the modules to do the type check
        if not isinstance(node.target, str):
            raise AssertionError(f"Expected str, got {type(node.target)}")
        mod = getattr_from_fqn(gm, node.target)
        module_mapping = op_to_type_to_weight_extraction_fn["call_module"]
        for target_mod_type, weight_extraction_fn in module_mapping.items():
            if type(mod) is target_mod_type:
                weight = weight_extraction_fn(mod)
                return {
                    "type": res_type,
                    "values": [weight],
                    "prev_node_name": node.name,
                    "prev_node_target_type": prev_node_type,
                    "ref_node_name": node.name,
                    "ref_node_target_type": ref_node_type,
                    "index_within_arg": 0,
                    "index_of_arg": 0,
                    "fqn": fqn,
                }

    return None
