# mypy: allow-untyped-defs
# [MODULE DOC] Backend config for QNNPACK in the PyTorch 2.0 export (aten-level graph) flow. Defines a shared
# weighted-op quint8 dtype config at import time and getters for linear, conv, pooling, relu and add
# pattern configs, assembled by get_qnnpack_pt2e_backend_config.
import operator

import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)


weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)


def get_linear_configs():
    # [FUNCTION DOC] Returns configs for aten addmm (weight index 2, bias index 0) and mm (weight index 1), with the
    # different-observer observation type. A commented-out complex-pattern alternative is left in place.
    linear_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    # TODO: need to fix the way we insert observers for this pattern
    # should be solved in the new fusion API
    # reason that this doesn't work: the pattern is a bit complicated and we don't
    # have a way to specify which input of the pattern we would like to observe
    # pattern:
    # bias input weight
    # \     |    /
    #  \    |   t
    #   \   |  /
    #    addmm
    # we want to observe "weight" as weight, but there is not way to convey this
    # information with current pattern language
    #
    # right now:
    # original:
    #         weight - t \
    #         input  - addmm
    # observed (no hack):
    #      weight - t - observer \
    #       input - observer - addmm
    # target:
    #      weight - observer - t \
    #        input - observer - addmm

    # def root_node_getter(node_pattern):
    #     addmm, bias, act, weight = node_pattern
    #     return addmm

    # linear_configs.append(
    #     BackendPatternConfig((torch.ops.aten.addmm.default, MatchAllNode, MatchAllNode, torch.ops.aten.t.default))
    #     .set_observation_type(observation_type)
    #     .set_dtype_configs(dtype_configs)
    #     ._set_root_node_getter(root_node_getter))

    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.addmm.default)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 0})
    )
    # linear is decomposed to `t - mm` if bias is not present
    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.mm.default)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1})
    )
    return linear_configs


def get_conv_configs():
    # [FUNCTION DOC] Returns configs for aten convolution alone and followed by relu or in-place relu (weight index 1,
    # bias index 2), using the quint8 weighted dtype config.
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    conv_configs.append(
        BackendPatternConfig(torch.ops.aten.convolution.default)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    conv_configs.append(
        BackendPatternConfig(
            (torch.ops.aten.convolution.default, torch.ops.aten.relu.default)
        )
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    # TODO: remove when functionalization is supported in PT2 mode
    conv_configs.append(
        BackendPatternConfig(
            (torch.ops.aten.convolution.default, torch.ops.aten.relu_.default)
        )
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    return conv_configs


def get_pooling_configs():
    # [FUNCTION DOC] Returns a config for the max_pool2d_with_indices then getitem pattern (complex format) with
    # shared observer; a nested root node getter picks the max pool node.
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    def root_node_getter(node_pattern):
        # [FUNCTION DOC] HELPER: Given the matched node pattern (getitem, maxpool, index), returns the maxpool node.
        _getitem, maxpool, _index = node_pattern
        return maxpool

    backend_pattern_configs.append(
        BackendPatternConfig()
        ._set_pattern_complex_format(
            (operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0)
        )
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_root_node_getter(root_node_getter)
    )

    return backend_pattern_configs


def get_relu_configs():
    # [FUNCTION DOC] Returns a config for aten relu that shares the observer with its input.
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    backend_pattern_configs.append(
        BackendPatternConfig(torch.ops.aten.relu.default)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
    )
    return backend_pattern_configs


def get_binary_op_configs():
    # [FUNCTION DOC] Returns configs for aten add.Tensor and add_.Tensor, alone and followed by relu or in-place relu,
    # with observation type chosen by number of tensor args.
    binary_op_configs: list[BackendPatternConfig] = []
    dtype_configs = [weighted_op_quint8_dtype_config]
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    for op_with_quantized_bop_scalar_variant in [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
    ]:
        bop_patterns = [
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu.default),
            op_with_quantized_bop_scalar_variant,
            # TODO: remove when functionalization is supported in pt2_mode
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu_.default),
        ]
        binary_op_configs.extend(
            BackendPatternConfig(bop_pattern)
            .set_dtype_configs(dtype_configs)
            ._set_num_tensor_args_to_observation_type(
                num_tensor_args_to_observation_type_mapping
            )
            for bop_pattern in bop_patterns
        )

    return binary_op_configs


def get_qnnpack_pt2e_backend_config():
    # [FUNCTION DOC] Returns the BackendConfig named qnnpack_pytorch_2.0_export combining linear, binary op, conv,
    # pooling and relu pattern configs.
    return (
        BackendConfig("qnnpack_pytorch_2.0_export")
        .set_backend_pattern_configs(get_linear_configs())
        .set_backend_pattern_configs(get_binary_op_configs())
        .set_backend_pattern_configs(get_conv_configs())
        .set_backend_pattern_configs(get_pooling_configs())
        .set_backend_pattern_configs(get_relu_configs())
    )
