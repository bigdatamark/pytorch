# mypy: allow-untyped-defs
# [MODULE DOC] Defines QConfig (a namedtuple of activation and weight observer/fake-quant constructors),
# the deprecated QConfigDynamic, and a large set of predefined default qconfigs (static, dynamic, QAT,
# weight-only, qnnpack symmetric, embedding). Also provides backend-specific factories
# (get_default_qconfig, get_default_qat_qconfig), deprecated dict variants, and helpers for validating,
# comparing and device-adapting qconfigs. Import-time side effect: builds all module-level default
# qconfig constants.
import copy
import warnings
from collections import namedtuple
from typing import Any
from typing_extensions import deprecated, TypeAliasType

import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
    default_dynamic_fake_quant,
    default_embedding_fake_quant,
    default_embedding_fake_quant_4bit,
    default_fake_quant,
    default_fused_act_fake_quant,
    default_fused_per_channel_wt_fake_quant,
    default_fused_wt_fake_quant,
    default_per_channel_weight_fake_quant,
    default_weight_fake_quant,
    FakeQuantize,
    FakeQuantizeBase,
    fused_per_channel_wt_fake_quant_range_neg_127_to_127,
    fused_wt_fake_quant_range_neg_127_to_127,
    FusedMovingAvgObsFakeQuantize,
)

from .observer import (
    _PartialWrapper,
    default_debug_observer,
    default_dynamic_quant_observer,
    default_float_qparams_observer,
    default_float_qparams_observer_4bit,
    default_observer,
    default_per_channel_weight_observer,
    default_placeholder_observer,
    default_reuse_input_observer,
    default_weight_observer,
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    NoopObserver,
    ObserverBase,
    per_channel_weight_observer_range_neg_127_to_127,
    PlaceholderObserver,
    ReuseInputObserver,
    weight_observer_range_neg_127_to_127,
)


__all__ = [
    "QConfig",
    # TODO: deprecated, remove
    "QConfigDynamic",
    "default_qconfig",
    "default_debug_qconfig",
    "default_per_channel_qconfig",
    "default_dynamic_qconfig",
    "float16_dynamic_qconfig",
    "float16_static_qconfig",
    "per_channel_dynamic_qconfig",
    "float_qparams_weight_only_qconfig",
    "float_qparams_weight_only_qconfig_4bit",
    "default_quint8_weight_qconfig",
    "default_qat_qconfig",
    "default_dynamic_qat_qconfig",
    "default_weight_only_qconfig",
    "default_activation_only_qconfig",
    "default_qat_qconfig_v2",
    "default_reuse_input_qconfig",
    "default_symmetric_qnnpack_qconfig",
    "default_per_channel_symmetric_qnnpack_qconfig",
    "default_symmetric_qnnpack_qat_qconfig",
    "default_per_channel_symmetric_qnnpack_qat_qconfig",
    "default_embedding_qat_qconfig",
    "default_embedding_qat_qconfig_4bit",
    "get_default_qconfig",
    "get_default_qat_qconfig",
    "get_default_qconfig_dict",
    "get_default_qat_qconfig_dict",
    "QConfigAny",
    "qconfig_equals",
]


# pyrefly: ignore [invalid-inheritance]
class QConfig(namedtuple("QConfig", ["activation", "weight"])):
    # [CLASS DOC] Immutable namedtuple with fields activation and weight, each an observer or fake-quantize
    # constructor (class or partial wrapper), never an instance. Describes how a layer is quantized.
    # The constructor raises ValueError if either argument is an nn.Module instance (common mistake).
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfig(
          activation=MinMaxObserver.with_args(dtype=torch.qint8),
          weight=default_observer.with_args(dtype=torch.qint8),
      )

    """

    __slots__ = ()

    def __new__(cls, activation, weight):
        # [FUNCTION DOC] Creates a QConfig after rejecting module instances for activation or weight.
        # Raises ValueError if either is an nn.Module; otherwise returns the namedtuple.
        # catch common mistakes
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError(
                "QConfig received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        return super().__new__(cls, activation, weight)


@deprecated(
    "`QConfigDynamic` is going to be deprecated in PyTorch 1.12, please use `QConfig` instead",
    category=FutureWarning,
)
# pyrefly: ignore [invalid-inheritance]
class QConfigDynamic(namedtuple("QConfigDynamic", ["activation", "weight"])):
    # [CLASS DOC] Deprecated (emits FutureWarning on use) namedtuple like QConfig for dynamic quantization;
    # both fields default to nn.Identity. Superseded by QConfig.
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classes) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """

    __slots__ = ()

    def __new__(cls, activation=torch.nn.Identity, weight=torch.nn.Identity):
        # [FUNCTION DOC] Creates a QConfigDynamic; raises ValueError if weight is an nn.Module instance.
        # Note only weight is checked, activation is not.
        # catch common mistakes
        if isinstance(weight, nn.Module):
            raise ValueError(
                "QConfigDynamic received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        return super().__new__(cls, activation, weight)


default_qconfig = QConfig(activation=default_observer, weight=default_weight_observer)
"""
Default qconfig configuration.
"""

default_debug_qconfig = QConfig(
    weight=default_weight_observer, activation=default_debug_observer
)
"""
Default qconfig configuration for debugging.
"""

default_per_channel_qconfig = QConfig(
    activation=default_observer, weight=default_per_channel_weight_observer
)
"""
Default qconfig configuration for per channel weight quantization.
"""

default_dynamic_qconfig = QConfig(
    activation=default_dynamic_quant_observer, weight=default_weight_observer
)
"""
Default dynamic qconfig.
"""

float16_dynamic_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.float16, is_dynamic=True),
    weight=PlaceholderObserver.with_args(dtype=torch.float16),
)
"""
Dynamic qconfig with weights quantized to `torch.float16`.
"""

float16_static_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.float16),
    weight=PlaceholderObserver.with_args(dtype=torch.float16),
)
"""
Dynamic qconfig with both activations and weights quantized to `torch.float16`.
"""

per_channel_dynamic_qconfig = QConfig(
    activation=default_dynamic_quant_observer,
    weight=default_per_channel_weight_observer,
)
"""
Dynamic qconfig with weights quantized per channel.
"""

float_qparams_weight_only_qconfig = QConfig(
    activation=default_placeholder_observer, weight=default_float_qparams_observer
)
"""
Dynamic qconfig with weights quantized with a floating point zero_point.
"""

float_qparams_weight_only_qconfig_4bit = QConfig(
    activation=default_placeholder_observer, weight=default_float_qparams_observer_4bit
)

default_qat_qconfig = QConfig(
    activation=default_fake_quant, weight=default_weight_fake_quant
)
"""
Default qconfig for QAT.
"""

default_dynamic_qat_qconfig = QConfig(
    activation=default_dynamic_fake_quant, weight=default_weight_fake_quant
)
"""
Default qconfig for dynamic QAT.
"""

default_weight_only_qconfig = QConfig(
    activation=torch.nn.Identity, weight=default_weight_fake_quant
)
"""
Default qconfig for quantizing weights only.
"""

default_activation_only_qconfig = QConfig(
    activation=default_fake_quant, weight=torch.nn.Identity
)
"""
Default qconfig for quantizing activations only.
"""

# QAT config that uses a fused observer + fake quant modules for optimized training performance.
# to modify the activation/weight observers, the default entries in fake_quantize.py can be modified.
default_qat_qconfig_v2 = QConfig(
    activation=default_fused_act_fake_quant, weight=default_fused_wt_fake_quant
)
"""
Fused version of `default_qat_config`, has performance benefits.
"""

default_reuse_input_qconfig = QConfig(
    activation=default_reuse_input_observer, weight=NoopObserver
)
"""
Default qconfig for operators that reuse the observers from input Tensor, e.g. reshape
"""


def get_default_qconfig(backend="x86", version=0):
    # [FUNCTION DOC] Returns the default post-training-quantization QConfig for a backend.
    # Args: backend, one of fbgemm, x86, qnnpack, onednn; version, must be 0.
    # fbgemm and x86 use histogram activation with reduced range and per-channel weights; qnnpack uses
    # histogram activation without reduced range and per-tensor weights; onednn uses per-channel weights
    # without reduced range and warns if the CPU lacks VNNI support.
    # Raises AssertionError for an unsupported backend or version.
    """
    Returns the default PTQ qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.

    Return:
        qconfig
    """
    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    if backend not in supported_backends:
        raise AssertionError(
            "backend: "
            + str(backend)
            + f" not supported. backend must be one of {supported_backends}"
        )

    if version == 0:
        if backend == "fbgemm":
            qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=True),
                weight=default_per_channel_weight_observer,
            )
        elif backend == "qnnpack":
            # TODO: make this compatible with xnnpack constraints
            qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=False),
                weight=default_weight_observer,
            )
        elif backend == "onednn":
            if not torch.cpu._is_vnni_supported():
                warnings.warn(
                    "Default qconfig of oneDNN backend with reduce_range of false may have accuracy issues "
                    "on CPU without Vector Neural Network Instruction support.",
                    stacklevel=2,
                )
            qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=False),
                weight=default_per_channel_weight_observer,
            )
        elif backend == "x86":
            qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=True),
                weight=default_per_channel_weight_observer,
            )
        else:
            # won't reach
            qconfig = default_qconfig
    else:
        raise AssertionError(
            "Version number: "
            + str(version)
            + " in get_default_qconfig is not supported. Version number must be 0"
        )

    return qconfig


"""
Default, symmetric PTQ qconfig for the specified backend. And a per_channel
variant of the same.

Symmetric here applies to signed weights with zero point = 0, and additional
value restrictions. The activations are also signed 8-bit integers with this
qconfig.

    * Once this change is merged [as of 3/17/22], with backend or qengine =
    'qnnpack', some quantized operators with this symmetric qconfig may use
    operators from xnnpack library.

        ** Support to use xnnpack ops with `qnnpack` backed for asymmetric
        qconfig (returned by get_default_qconfig()) is not available yet.

    * This qconfig uses signed activations and weights. Weights have added
    restrictions such as zero point is forced to be 0, making the weights
    symmetric, hence the name. And the 8-bit quantized values are
    restricting to [-127, +127], excluding -128.

    * xnnpack has a requantization scale value restriction, 0x1p-32 <=
    requantization_scale < 256.0 where, `requantization_scale = (input_scale
    * kernel_scale) / (output_scale)`. Using this eps (w/ assumed max value
    of 256) is to prevent requantization_scale to go below xnnpack lower
    threshold.
"""
default_symmetric_qnnpack_qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=torch.qint8, reduce_range=False, eps=2**-12
    ),
    weight=weight_observer_range_neg_127_to_127,
)

default_per_channel_symmetric_qnnpack_qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=torch.qint8, reduce_range=False, eps=2**-12
    ),
    weight=per_channel_weight_observer_range_neg_127_to_127,
)

default_embedding_qat_qconfig = QConfig(
    activation=NoopObserver.with_args(dtype=torch.float32),
    weight=default_embedding_fake_quant,
)

default_embedding_qat_qconfig_4bit = QConfig(
    activation=NoopObserver.with_args(dtype=torch.float32),
    weight=default_embedding_fake_quant_4bit,
)

default_quint8_weight_qconfig = QConfig(
    activation=HistogramObserver, weight=MinMaxObserver
)


def get_default_qat_qconfig(backend="x86", version=1):
    # [FUNCTION DOC] Returns the default quantization-aware-training QConfig for a backend.
    # Args: backend, one of fbgemm, x86, qnnpack, onednn; version, 0 (separate observer plus FakeQuantize
    # with moving-average min/max) or 1 (fused moving-average observer and fake-quant, default).
    # Reduce-range and per-channel weight choices mirror the PTQ defaults per backend.
    # Raises AssertionError for an unsupported backend or version.
    # KNOWN ISSUE: the version error message lacks a space before "in get_default_qat_qconfig" and the
    # docstring says version may be None, but None raises.
    """
    Returns the default QAT qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.
      * `version`: version, for backwards compatibility. Can be `None` or `1`.

    Return:
        qconfig
    """
    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    if backend not in supported_backends:
        raise AssertionError(
            "backend: "
            + str(backend)
            + f" not supported. backend must be one of {supported_backends}"
        )

    # Histogram observer is too slow for quantization aware training
    if version == 0:
        if backend == "fbgemm":
            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=True,
                ),
                weight=default_per_channel_weight_fake_quant,
            )
        elif backend == "qnnpack":
            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=False,
                ),
                weight=default_weight_fake_quant,
            )
        elif backend == "onednn":
            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255
                ),
                weight=default_per_channel_weight_fake_quant,
            )
        elif backend == "x86":
            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=True,
                ),
                weight=default_per_channel_weight_fake_quant,
            )
        else:
            qconfig = default_qat_qconfig
    # Use the fused observe + fake_quant modules for doing QAT.
    elif version == 1:
        if backend == "fbgemm":
            qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=True,
                ),
                weight=default_fused_per_channel_wt_fake_quant,
            )
        elif backend == "qnnpack":
            # TODO: make this compatible with xnnpack constraints
            qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=False,
                ),
                weight=default_fused_wt_fake_quant,
            )
        elif backend == "onednn":
            qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255
                ),
                weight=default_fused_per_channel_wt_fake_quant,
            )
        elif backend == "x86":
            qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=True,
                ),
                weight=default_fused_per_channel_wt_fake_quant,
            )
        else:
            qconfig = default_qat_qconfig_v2
    else:
        raise AssertionError(
            "Version number: "
            + str(version)
            + "in get_default_qat_qconfig is not supported. Version number must be 0 or 1"
        )

    return qconfig


"""
Default symmetric QAT qconfig for qnnpack. And its per channel weight variant.
"""
default_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        reduce_range=False,
        eps=2**-12,
    ),
    weight=fused_wt_fake_quant_range_neg_127_to_127,
)

default_per_channel_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        reduce_range=False,
        eps=2**-12,
    ),
    weight=fused_per_channel_wt_fake_quant_range_neg_127_to_127,
)

_default_fp32_placeholder_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.float32),
    weight=PlaceholderObserver.with_args(dtype=torch.float32),
)

_default_quint8_placeholder_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.quint8),
    # operators using this qconfig doesn't have weights
    weight=None,
)


@deprecated(
    "`torch.ao.quantization.get_default_qconfig_dict` is deprecated and will be removed in "
    "a future version. Please use `torch.ao.quantization.get_default_qconfig_mapping` instead.",
    category=FutureWarning,
)
def get_default_qconfig_dict(backend="x86", version=0):
    # [FUNCTION DOC] Deprecated: builds the default qconfig mapping for the backend and version and returns
    # it converted to a dictionary.
    return torch.ao.quantization.get_default_qconfig_mapping(backend, version).to_dict()


@deprecated(
    "`torch.ao.quantization.get_default_qat_qconfig_dict` is deprecated and will be removed in "
    "a future version. Please use `torch.ao.quantization.get_default_qat_qconfig_mapping` instead.",
    category=FutureWarning,
)
def get_default_qat_qconfig_dict(backend="x86", version=1):
    # [FUNCTION DOC] Deprecated: builds the default QAT qconfig mapping for the backend and version and
    # returns it converted to a dictionary.
    return torch.ao.quantization.get_default_qat_qconfig_mapping(
        backend, version
    ).to_dict()


def _assert_valid_qconfig(qconfig: QConfig | None, mod: torch.nn.Module) -> None:
    # [FUNCTION DOC] HELPER: Validates a qconfig for a module. Returns silently for None. For ConvTranspose
    # 1d/2d/3d modules with a weight constructor, instantiates the weight observer and raises
    # AssertionError if it is a per-channel observer (unsupported). Other modules are not checked.
    """
    Verifies that this `qconfig` is valid.
    """
    if qconfig is None:
        return
    is_conv_transpose_mod = isinstance(
        mod,
        (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d),
    )
    if is_conv_transpose_mod:
        if qconfig.weight is None:
            # for now, we assume that any qconfig for ConvTranspose without a weight is valid
            return
        example_observer = qconfig.weight()
        is_per_channel = isinstance(
            example_observer,
            (
                torch.ao.quantization.PerChannelMinMaxObserver,
                torch.ao.quantization.MovingAveragePerChannelMinMaxObserver,
            ),
        )
        if is_per_channel:
            raise AssertionError(
                "Per channel weight observer is not supported yet for ConvTranspose{n}d."
            )


QConfigAny = TypeAliasType("QConfigAny", QConfig | None)


def _add_module_to_qconfig_obs_ctr(
    qconfig: QConfigAny, module: nn.Module | None
) -> Any:
    # [FUNCTION DOC] HELPER: Returns a new QConfig whose activation and weight constructors create observers
    # on the same device as the given module. Returns the input unchanged if module or qconfig is None or
    # the qconfig lacks exactly the fields activation and weight. Constructors that do not accept a
    # factory_kwargs argument are left unchanged.
    r"""This is a helper function for use in quantization prepare that updates a qconfig so that
    the constructors stored in the qconfig will create observers on the same device that
    'module' is on. This is intended to be used when the qconfigs are propagated to each
    module in order to avoid potential device alignment issues.

    Args:
        qconfig: QConfig with obs constructors stored in activation and weight
        module: module which the qconfig is related to

    Return:
        qconfig: configured so that obs constructors set to construct on the same device as module
    """

    if module is None or qconfig is None or qconfig._fields != ("activation", "weight"):
        return qconfig

    def get_factory_kwargs_based_on_module_device():
        # [FUNCTION DOC] HELPER: Collects devices of the module's parameters and buffers and returns a mapping with
        # the first device, or None if the module has none. Raises AssertionError if module is not an nn.Module.
        if not isinstance(module, torch.nn.Module):
            raise AssertionError("module must be an instance of torch.nn.Module")
        devices = {p.device for p in module.parameters()} | {
            p.device for p in module.buffers()
        }
        device = next(iter(devices)) if len(devices) > 0 else None
        return None if device is None else {"device": device}

    def configure_constructor_to_put_obs_on_module_device(original_constructor):
        # [FUNCTION DOC] HELPER: Probes whether the constructor accepts factory_kwargs by instantiating it; if so
        # returns a version binding factory_kwargs lazily to the module device callable, otherwise (attribute or
        # type error) returns the original constructor. Side effect: constructs one throwaway observer.
        try:
            # check if constructor can accept factory_kwargs
            check = original_constructor.with_args(factory_kwargs=None)
            check()
            return original_constructor.with_callable_args(
                factory_kwargs=get_factory_kwargs_based_on_module_device
            )
        except AttributeError:  # qconfig doesn't have activation or weight
            return original_constructor
        except TypeError:  # the class doesn't accept factory_kwargs argument
            return original_constructor

    activation = configure_constructor_to_put_obs_on_module_device(qconfig.activation)
    weight = configure_constructor_to_put_obs_on_module_device(qconfig.weight)

    return QConfig(activation, weight)


_ObserverOrFakeQuantizeConstructor = (
    _PartialWrapper | type[ObserverBase] | type[FakeQuantizeBase]
)


def _obs_or_fq_ctr_equals(
    obs_or_fq1: _ObserverOrFakeQuantizeConstructor,
    obs_or_fq2: _ObserverOrFakeQuantizeConstructor,
):
    # [FUNCTION DOC] HELPER: Equality of two observer/fake-quant constructors; compares by partial-wrapper
    # contents when both are partial wrappers, otherwise by plain equality. Returns a bool.
    if isinstance(obs_or_fq1, _PartialWrapper) and isinstance(
        obs_or_fq2, _PartialWrapper
    ):
        return _partial_wrapper_equals(obs_or_fq1, obs_or_fq2)
    return obs_or_fq1 == obs_or_fq2


def _partial_wrapper_equals(obs_or_fq1: _PartialWrapper, obs_or_fq2: _PartialWrapper):
    # [FUNCTION DOC] HELPER: Returns True if two partial wrappers have equal wrapped function, positional args
    # and keyword args. The nested observer keyword is compared recursively and removed from copies first,
    # since partial objects have no equality. Does not mutate the inputs.
    """
    Return whether the two partial wrappers are equal,
    """
    # functools.partial has no __eq__ operator defined so '==' defaults to 'is'
    obs_or_fq1_keywords = copy.copy(obs_or_fq1.p.keywords)
    obs_or_fq2_keywords = copy.copy(obs_or_fq2.p.keywords)
    keywords_equal = True
    # compare observer constructor with _obs_or_fq_ctr_equals since direct compare would fail
    if "observer" in obs_or_fq1_keywords and "observer" in obs_or_fq2_keywords:
        keywords_equal = keywords_equal and _obs_or_fq_ctr_equals(
            obs_or_fq1_keywords["observer"], obs_or_fq2_keywords["observer"]
        )
        obs_or_fq1_keywords.pop("observer")
        obs_or_fq2_keywords.pop("observer")
    keywords_equal = keywords_equal and obs_or_fq1_keywords == obs_or_fq2_keywords
    return (
        obs_or_fq1.p.func == obs_or_fq2.p.func
        and obs_or_fq1.p.args == obs_or_fq2.p.args
        and keywords_equal
    )


def qconfig_equals(q1: QConfigAny, q2: QConfigAny):
    # [FUNCTION DOC] Returns True if two qconfigs are equal. If either is None, compares them directly.
    # Otherwise compares activation and weight constructors (partial-aware); on attribute errors falls back
    # to plain equality.
    """
    Returns `True` if `q1` equals `q2`, and `False` otherwise.
    """
    if q1 is None or q2 is None:
        return q1 == q2
    else:
        if q1 is None or q2 is None:
            raise AssertionError(
                "Both q1 and q2 must be non-None for qconfig comparison"
            )
        try:
            # Qconfig weight and activation can be either a partial wrapper,
            # or an observer class. Special handling is required (above) for
            # comparing partial wrappers.
            activation_same = _obs_or_fq_ctr_equals(q1.activation, q2.activation)
            weight_same = _obs_or_fq_ctr_equals(q1.weight, q2.weight)
            return activation_same and weight_same
        except AttributeError:
            return q1 == q2


def _activation_is_memoryless(qconfig: QConfig):
    # [FUNCTION DOC] HELPER: Returns True if the activation observer of the qconfig is memoryless, i.e. it has
    # an averaging constant equal to 1. Instantiates the activation constructor; for fake-quant modules
    # inspects the inner activation_post_process observer.
    """
    Return whether the observer for activations defined in the given QConfig is memoryless.
    This means a MovingAverage observer with averaging constant equal to 1.
    """

    def _is_memoryless(observer):
        # [FUNCTION DOC] HELPER: True if the observer has an averaging_constant attribute equal to 1.
        return (
            hasattr(observer, "averaging_constant") and observer.averaging_constant == 1
        )

    act = qconfig.activation()
    if isinstance(act, FakeQuantizeBase) and hasattr(act, "activation_post_process"):
        return _is_memoryless(act.activation_post_process)
    else:
        return _is_memoryless(act)


def _is_reuse_input_qconfig(qconfig: QConfig | None):
    # [FUNCTION DOC] HELPER: True if qconfig is not None, its activation builds a ReuseInputObserver and its
    # weight builds a NoopObserver. Instantiates both constructors.
    return (
        qconfig is not None
        and isinstance(qconfig.activation(), ReuseInputObserver)
        and isinstance(qconfig.weight(), NoopObserver)
    )
