# mypy: allow-untyped-defs
# [MODULE DOC] Global registries of default fusion patterns, quantization patterns and fixed-qparams
# output activation post-process maps (observer and fake-quant), with registration decorators,
# copy-returning getters and a helper that sorts patterns longest-first. Import-time side effect:
# creates the empty module-level registries. Exports the three get_default_* functions.
import copy
from collections import OrderedDict
from typing import Any

from torch.ao.quantization.fake_quantize import FixedQParamsFakeQuantize
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.utils import Pattern


__all__ = [
    "get_default_fusion_patterns",
    "get_default_quant_patterns",
    "get_default_output_activation_post_process_map",
]

# TODO(future PR): fix the typing on QuantizeHandler (currently a circular dependency)
QuantizeHandler = Any

# pattern for conv bn fusion
_DEFAULT_FUSION_PATTERNS: dict[Pattern, QuantizeHandler] = OrderedDict()


def _register_fusion_pattern(pattern):
    # [FUNCTION DOC] HELPER: returns a decorator that registers the decorated object in the default
    # fusion pattern registry under the given pattern and returns the object unchanged.
    def insert(fn):
        # [FUNCTION DOC] HELPER: registers fn for the captured pattern (mutates registry); returns fn.
        _DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn

    return insert


def get_default_fusion_patterns() -> dict[Pattern, QuantizeHandler]:
    # [FUNCTION DOC] Return a shallow copy of the registry mapping fusion patterns to handlers.
    return copy.copy(_DEFAULT_FUSION_PATTERNS)


_DEFAULT_QUANTIZATION_PATTERNS: dict[Pattern, QuantizeHandler] = OrderedDict()

# Mapping from pattern to activation_post_process(observer/fake_quant) constructor for output activation
# e.g. pattern: torch.sigmoid,
#      output_activation_post_process: default_fixed_qparams_range_0to1_fake_quant
_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP: dict[Pattern, QuantizeHandler] = {}
_DEFAULT_OUTPUT_OBSERVER_MAP: dict[Pattern, QuantizeHandler] = {}


# Register pattern for both static quantization and qat
def _register_quant_pattern(pattern, fixed_qparams_observer=None):
    # [FUNCTION DOC] HELPER: returns a decorator registering the decorated handler for pattern in the
    # default quantization pattern registry, and returns it unchanged. If fixed_qparams_observer is
    # given, also records a fixed-qparams fake-quantize constructor (built from that observer) and the
    # observer itself for the pattern in the output activation maps.
    def insert(fn):
        # [FUNCTION DOC] HELPER: performs the registration described above (mutates global registries); returns fn.
        _DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        if fixed_qparams_observer is not None:
            _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP[pattern] = (
                FixedQParamsFakeQuantize.with_args(observer=fixed_qparams_observer)
            )
            _DEFAULT_OUTPUT_OBSERVER_MAP[pattern] = fixed_qparams_observer
        return fn

    return insert


# Get patterns for both static quantization and qat
def get_default_quant_patterns() -> dict[Pattern, QuantizeHandler]:
    # [FUNCTION DOC] Return a shallow copy of the registry mapping quantization patterns to handlers.
    return copy.copy(_DEFAULT_QUANTIZATION_PATTERNS)


# a map from pattern to output activation post process constructor
# e.g. torch.sigmoid -> default_affine_fixed_qparam_fake_quant
def get_default_output_activation_post_process_map(
    is_training,
) -> dict[Pattern, ObserverBase]:
    # [FUNCTION DOC] Return a shallow copy of the pattern to output activation post-process map:
    # the fake-quantize map when is_training is true, otherwise the observer map.
    if is_training:
        return copy.copy(_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP)
    else:
        return copy.copy(_DEFAULT_OUTPUT_OBSERVER_MAP)


# Example use of register pattern function:
# @_register_fusion_pattern(torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
# class ConvOrLinearBNReLUFusion():
#     def __init__(...):
#         ...
#


def _sorted_patterns_dict(
    patterns_dict: dict[Pattern, QuantizeHandler],
) -> dict[Pattern, QuantizeHandler]:
    # [FUNCTION DOC] HELPER: returns an ordered mapping with tuple patterns first, longest first
    # (length counts all nested leaf entries), so longer patterns are tried before shorter ones.
    # KNOWN ISSUE: non-tuple patterns all get the same key (1) while tuples get a negative key, so
    # they sort after tuples; relative order among equal keys is preserved (stable sort).
    """
    Return a sorted version of the patterns dictionary such that longer patterns are matched first,
    e.g. match (F.relu, F.linear) before F.relu.
    This works for current use cases, but we may need to have a more clever way to sort
    things to address more complex patterns
    """

    def get_len(pattern):
        """this will calculate the length of the pattern by counting all the entries
        in the pattern.
        this will make sure (nn.ReLU, (nn.BatchNorm, nn.Conv2d)) comes before
        (nn.BatchNorm, nn.Conv2d) so that we can match the former first
        """
        # [FUNCTION DOC] HELPER: count leaf entries of a possibly nested tuple pattern (a non-tuple counts as 1).
        len = 0
        if isinstance(pattern, tuple):
            for item in pattern:
                len += get_len(item)
        else:
            len += 1
        return len

    return OrderedDict(
        sorted(
            patterns_dict.items(),
            key=lambda kv: -get_len(kv[0]) if isinstance(kv[0], tuple) else 1,
        )
    )
