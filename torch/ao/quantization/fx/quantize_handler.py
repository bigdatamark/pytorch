# mypy: allow-untyped-defs
# [MODULE DOC] Quantize handler classes for the FX quantization flow. QuantizeHandler records a
# matched node pattern, its root node and how many of the root node's args are tensors;
# helpers build a configurable handler class per backend pattern config. The many
# named subclasses (BinaryOp, Cat, ConvRelu, ...) are empty legacy placeholders kept
# only for backward compatible imports from torch.ao.quantization.
from abc import ABC
from collections.abc import Callable

import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeConfig,
    ObservationType,
)
from torch.ao.quantization.utils import NodePattern, Pattern, QuantizerCls
from torch.fx.graph import Node

from .utils import all_node_args_have_no_tensors


__all__ = [
    "QuantizeHandler",
    "BinaryOpQuantizeHandler",
    "CatQuantizeHandler",
    "ConvReluQuantizeHandler",
    "LinearReLUQuantizeHandler",
    "BatchNormQuantizeHandler",
    "EmbeddingQuantizeHandler",
    "RNNDynamicQuantizeHandler",
    "DefaultNodeQuantizeHandler",
    "FixedQParamsOpQuantizeHandler",
    "CopyNodeQuantizeHandler",
    "GeneralTensorShapeOpQuantizeHandler",
    "CustomModuleQuantizeHandler",
    "StandaloneModuleQuantizeHandler",
]


def _default_root_node_getter(node_pattern):
    # [FUNCTION DOC] HELPER: Returns the root Node of a (possibly nested tuple) pattern by repeatedly taking
    # the last element until a Node is reached. Returns None unchanged if the pattern is None.
    if node_pattern is None:
        return node_pattern
    while not isinstance(node_pattern, Node):
        node_pattern = node_pattern[-1]
    return node_pattern


# Base Pattern Handler
class QuantizeHandler(ABC):  # noqa: B024
    # [CLASS DOC] Base holder for a matched quantization pattern. State: node_pattern, modules mapping,
    # root_node, custom/standalone module flags and num_tensor_args (count of root node
    # args that carry tensors). Subclasses override is_general_tensor_value_op.
    """Base handler class for the quantizer patterns"""

    def __init__(
        self,
        node_pattern: NodePattern,
        modules: dict[str, torch.nn.Module],
        root_node_getter: Callable | None = None,
        is_custom_module=False,
        is_standalone_module=False,
    ):
        # [FUNCTION DOC] Stores the pattern and modules, computes root_node with the given getter (default:
        # last-element descent), stores the custom/standalone flags, and counts the root
        # node's args that are Nodes not made only of non-tensor values (num_tensor_args).
        """Records pattern information in __init__, which will be used
        in convert
        """
        self.node_pattern = node_pattern
        self.modules = modules
        if root_node_getter is None:
            root_node_getter = _default_root_node_getter
        self.root_node = root_node_getter(node_pattern)
        self.is_custom_module_ = is_custom_module
        self.is_standalone_module_ = is_standalone_module
        self.num_tensor_args = 0
        # determine how many of the first two args are Tensors (versus scalars)
        # this distinguishes things like "x + y" from "x + 2" or "2 + x"
        if isinstance(self.root_node, Node):
            cache_for_no_tensor_check: dict[Node, bool] = {}
            for arg_idx in range(len(self.root_node.args)):
                arg = self.root_node.args[arg_idx]
                if isinstance(arg, Node) and (
                    not all_node_args_have_no_tensors(
                        arg, self.modules, cache_for_no_tensor_check
                    )
                ):
                    self.num_tensor_args += 1

    def is_general_tensor_value_op(self) -> bool:
        # [FUNCTION DOC] Base implementation always returns False (output does not share the input observer).
        """
        Returns True if the operator works for both floating point and
        quantized input, and does some computation based on the input Tensor,
        or the ops that only re-arranges the Tensor values or query some metadata
        about the Tensor
        so we need to insert observer/fake_quant for the output of the
        operator (same observer instance as input)
        since the distribution of values is different for input and output
        Tensors (for HistogramObserver) while they share the same quantization
        parameters
        Example operator: avgpool2d, reshape, transpose, maxpool2d
        Example observed operator:
        observer_0 - avgpool2d - observer_0 (same observer instance as input)
        """
        return False

    def is_custom_module(self):
        # [FUNCTION DOC] Returns the stored custom-module flag.
        return self.is_custom_module_

    def is_standalone_module(self):
        # [FUNCTION DOC] Returns the stored standalone-module flag.
        return self.is_standalone_module_


def _get_quantize_handler_cls(
    observation_type: ObservationType,
    dtype_configs: list[DTypeConfig],
    num_tensor_args_to_observation_type: dict[int, ObservationType],
) -> type[QuantizeHandler]:
    # [FUNCTION DOC] HELPER: Builds and returns a QuantizeHandler subclass closed over the backend's
    # observation type, dtype configs and a mapping from tensor-arg count to observation type.
    """
    Return a configurable QuantizeHandler that matches the given specifications from the backend.
    """

    class ConfigurableQuantizeHandler(QuantizeHandler):
        # [CLASS DOC] HELPER: Handler that sets observation_type and dtype_configs from the enclosing
        # backend specification.
        def __init__(
            self,
            node_pattern: NodePattern,
            modules: dict[str, torch.nn.Module],
            root_node_getter: Callable | None = None,
        ):
            # [FUNCTION DOC] HELPER: Runs the base init, then picks observation_type: if a per-tensor-count mapping
            # exists, looks up num_tensor_args (raises AssertionError if missing), else uses
            # the default observation type. Stores dtype_configs.
            super().__init__(node_pattern, modules, root_node_getter)
            if num_tensor_args_to_observation_type:
                if self.num_tensor_args not in num_tensor_args_to_observation_type:
                    raise AssertionError(
                        f"Must provide observation_type config for tensor number {self.num_tensor_args}"
                        f" in num_tensor_args_to_observation_type for {node_pattern}"
                    )
                self.observation_type = num_tensor_args_to_observation_type[
                    self.num_tensor_args
                ]
            else:
                self.observation_type = observation_type
            self.dtype_configs = dtype_configs

        def is_general_tensor_value_op(self) -> bool:
            # [FUNCTION DOC] HELPER: True when observation_type is OUTPUT_SHARE_OBSERVER_WITH_INPUT.
            return (
                self.observation_type
                == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
            )

    return ConfigurableQuantizeHandler


def _get_pattern_to_quantize_handlers(
    backend_config: BackendConfig,
) -> dict[Pattern, QuantizerCls]:
    # [FUNCTION DOC] HELPER: Returns a mapping from each pattern in the backend config (complex format) to a
    # configurable handler class built from that pattern's observation type, dtype
    # configs and per-tensor-count observation types.
    """
    Note: Quantize handler is just a holder for some check methods like
    (should_insert_observer_for_output), maybe this can be a enum as well,
    we can refactor this after we convert the path for fbgemm/qnnpack fully to the
    new path, this is not exposed to backend developers
    """
    pattern_to_quantize_handlers = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        observation_type = config.observation_type
        dtype_configs = config.dtype_configs
        num_tensor_args_to_observation_type = (
            config._num_tensor_args_to_observation_type
        )
        pattern_to_quantize_handlers[pattern] = _get_quantize_handler_cls(
            observation_type, dtype_configs, num_tensor_args_to_observation_type
        )
    return pattern_to_quantize_handlers


# TODO: remove this class, this is still exposed in torch.ao.quantization
# but we should be able to break bc
class BinaryOpQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


class CatQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: remove this class
class ConvReluQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: remove this class
class LinearReLUQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: remove this class
class BatchNormQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: remove this class
class EmbeddingQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: remove this class
class RNNDynamicQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: remove this class
class DefaultNodeQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    """Common quantized op, first input and first output will be quantized"""


# TODO: remove this class
class FixedQParamsOpQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: remove
class CopyNodeQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: remove
class GeneralTensorShapeOpQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: not used, can be removed after torch.ao.quantization namespace is deprecated
class CustomModuleQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass


# TODO: not used, can be removed after torch.ao.quantization namespace is deprecated
class StandaloneModuleQuantizeHandler(QuantizeHandler):
    # [CLASS DOC] HELPER: Empty legacy subclass kept for backward compatible imports.
    pass
