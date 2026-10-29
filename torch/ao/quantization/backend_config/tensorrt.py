# mypy: allow-untyped-defs
# [MODULE DOC] Backend configuration for the TensorRT backend (experimental, API expected to change).
# Exposes get_tensorrt_backend_config and get_tensorrt_backend_config_dict. No import-time side effects beyond imports.
import torch

from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_conv_configs,
    _get_linear_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)


__all__ = [
    "get_tensorrt_backend_config",
    "get_tensorrt_backend_config_dict",
]


def get_tensorrt_backend_config() -> BackendConfig:
    # [FUNCTION DOC] Build and return the BackendConfig named "tensorrt". Takes no arguments.
    # Uses a single qint8 dtype config (weighted: input/output/weight qint8 with float bias; non-weighted: input/output qint8).
    # Registers explicit pattern configs for addmm (different observer for output, with input indices for bias, input, weight)
    # and cat (output shares observer with input), plus conv, linear, binary op, share-qparams and tensor-info op configs
    # from the shared helpers.
    """
    Return the `BackendConfig` for the TensorRT backend.
    NOTE: Current api will change in the future, it's just to unblock experimentation for
    new backends, please don't use it right now.
    TODO: add a README when it's more stable
    """
    # dtype configs
    weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
    non_weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
    )

    addmm_config = (
        BackendPatternConfig(torch.addmm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .add_dtype_config(weighted_op_qint8_dtype_config)
        ._set_input_type_to_index(
            {
                "bias": 0,
                "input": 1,
                "weight": 2,
            }
        )
    )
    cat_config = (
        BackendPatternConfig(torch.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .add_dtype_config(non_weighted_op_qint8_dtype_config)
    )
    conv_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    binary_op_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    tensor_info_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    # there might be things not supported in fx2trt, but it will error out
    # during fx2trt conversion and can support them after that
    return (
        BackendConfig("tensorrt")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_config(addmm_config)
        .set_backend_pattern_config(cat_config)
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_tensor_info_op_configs(tensor_info_op_dtype_configs)
        )
    )


def get_tensorrt_backend_config_dict():
    # [FUNCTION DOC] Return the TensorRT BackendConfig converted to its dictionary form via the to_dict method of the config.
    """
    Return the `BackendConfig` for the TensorRT backend in dictionary form.
    """
    return get_tensorrt_backend_config().to_dict()
