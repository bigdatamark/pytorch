# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# [MODULE DOC] Fake quantization modules for quantization aware training and calibration: the abstract FakeQuantizeBase,
# FakeQuantize (observer driven quantize/dequantize simulation), FixedQParamsFakeQuantize, and FusedMovingAvgObsFakeQuantize
# (single fused kernel), qscheme predicate helpers, module-level helper functions to enable/disable observers and fake quant
# on a module (for use with Module.apply), and a set of default factory objects (default_fake_quant, default_weight_fake_quant,
# per-channel, embedding, histogram, fused variants) created with with_args at import time. Import has no other side effects.
"""Implements modules  used to perform fake quantization."""

import re
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.ao.quantization.observer import (
    _with_args,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    FixedQParamsObserver,
    HistogramObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.nn import Module


__all__ = [
    "FakeQuantizeBase",
    "FakeQuantize",
    "FixedQParamsFakeQuantize",
    "FusedMovingAvgObsFakeQuantize",
    "disable_fake_quant",
    "disable_observer",
    "enable_fake_quant",
    "enable_observer",
    "default_fake_quant",
    "default_weight_fake_quant",
    "default_dynamic_fake_quant",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_symmetric_fixed_qparams_fake_quant",
    "default_affine_fixed_qparams_fake_quant",
    "default_per_channel_weight_fake_quant",
    "default_embedding_fake_quant",
    "default_embedding_fake_quant_4bit",
    "default_histogram_fake_quant",
    "default_fused_act_fake_quant",
    "default_fused_wt_fake_quant",
    "default_fused_per_channel_wt_fake_quant",
    "fused_wt_fake_quant_range_neg_127_to_127",
    "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
]


def _is_per_channel(qscheme: "torch.qscheme") -> bool:
    # [FUNCTION DOC] HELPER: True if the qscheme is one of the per-channel schemes (symmetric, affine, affine with float qparams).
    return qscheme in [
        torch.per_channel_symmetric,
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]


def _is_per_tensor(qscheme: "torch.qscheme") -> bool:
    # [FUNCTION DOC] HELPER: True if the qscheme is per-tensor symmetric or per-tensor affine.
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]


def _is_symmetric_quant(qscheme: "torch.qscheme") -> bool:
    # [FUNCTION DOC] HELPER: True if the qscheme is symmetric (per-tensor or per-channel).
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]


def _is_float_qparams(qscheme: "torch.qscheme") -> bool:
    # [FUNCTION DOC] HELPER: True only for the per-channel affine scheme with float qparams.
    return qscheme == torch.per_channel_affine_float_qparams


class FakeQuantizeBase(ABC, Module):
    # [CLASS DOC] Abstract base for fake quantize modules (an ABC and a torch Module). Holds two uint8 one-element buffers,
    # fake_quant_enabled and observer_enabled (buffers so DDP replicates them). Subclasses implement forward and calculate_qparams.
    # Provides enable/disable helpers and a with_args class method that builds a partial constructor.
    r"""Base fake quantize module.

    Base fake quantize module
    Any fake quantize implementation should derive from this class.

    Concrete fake quantize module should follow the same API. In forward, they will update
    the statistics of the observed Tensor and fake quantize the input. They should also provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    """

    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor

    def __init__(self) -> None:
        # [FUNCTION DOC] Initialise the Module and register the buffers fake_quant_enabled and observer_enabled, both set to 1 (uint8 because NCCL lacks bool support).
        """Set fake_quant_enabled and observer_enabled."""
        super().__init__()
        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because NCCL does not support
        # bool tensors.
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.uint8))

    @abstractmethod
    def forward(self, x):
        # [FUNCTION DOC] Abstract forward; subclasses must implement. The base body does nothing.
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        # [FUNCTION DOC] Abstract method returning quantization parameters (scale, zero point); subclasses must implement. Base body does nothing.
        pass

    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        # [FUNCTION DOC] Set the fake quant flag buffer to 1 or 0 according to the enabled argument (default True). Mutates the buffer in place.
        self.fake_quant_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_fake_quant(self):
        # [FUNCTION DOC] Turn fake quantization off by calling enable_fake_quant with False.
        self.enable_fake_quant(False)

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        # [FUNCTION DOC] Set the observer flag buffer to 1 or 0 according to the enabled argument (default True). Mutates the buffer in place.
        self.observer_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_observer(self):
        # [FUNCTION DOC] Turn observation off by calling enable_observer with False.
        self.enable_observer(False)

    @classmethod
    def with_args(cls, **kwargs):
        # [FUNCTION DOC] Class method returning a constructor factory with the given keyword arguments pre-bound (via _with_args).
        # The returned factory's __module__ is set to torch.ao.quantization.fake_quantize for public/private API checks.
        fake_quant_constructor = _with_args(cls, **kwargs)
        # need to assign the correct module to fake_quantize
        # constructors to satisfy public v private requirements
        fake_quant_constructor.__module__ = "torch.ao.quantization.fake_quantize"
        return fake_quant_constructor


class FakeQuantize(FakeQuantizeBase):
    # [CLASS DOC] Simulates quantize followed by dequantize at training time using an observer (activation_post_process) to compute scale
    # and zero_point. State: buffers scale (shape 1, float) and zero_point (int, or float for float-qparams schemes), plus quant_min,
    # quant_max, is_dynamic, dtype, qscheme, ch_axis (-1 if the observer has none) and is_per_channel copied from the observer.
    # Observer and fake quant are independently switchable. Only per-tensor and per-channel schemes are supported.
    r"""Simulate the quantize and dequantize operations in training time.

    The output of this module is given by::

        x_out = (
            clamp(round(x / scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale

    * :attr:`is_dynamic` indicates whether the fake quantize is a placeholder for dynamic quantization
      operators (choose_qparams -> q -> dq) or static quantization operators (q -> dq)

    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`fake_quant_enabled` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enabled` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
        allowable values are torch.qint8 and torch.quint8.

    Args:

        observer (module): Module for observing statistics on input tensors and calculating scale
          and zero-point.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        activation_post_process (Module): User provided module that collects statistics on the input tensor and
          provides a method to calculate scale and zero-point.

    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        observer=MovingAverageMinMaxObserver,
        quant_min=None,
        quant_max=None,
        is_dynamic=False,
        **observer_kwargs,
    ):
        # [FUNCTION DOC] Build the observer from the observer class (or partial) plus is_dynamic and optional quant_min/quant_max, then register scale and zero_point buffers.
        # Raises AssertionError if quant_min exceeds quant_max, if either bound lies outside the integer range of the dtype
        # (dtype from observer kwargs, or from a partially applied observer, default quint8), or if the qscheme is neither per-channel nor per-tensor.
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            if quant_min > quant_max:
                raise AssertionError(
                    "quant_min must be less than or equal to quant_max"
                )
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # In case observer is _PartialWrapper, dtype can be stored in
                # observer.p.keywords["dtype"]
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )

            if torch.iinfo(dtype).min > quant_min:
                raise AssertionError("quant_min out of bound")

            if quant_max > torch.iinfo(dtype).max:
                raise AssertionError("quant_max out of bound")
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        observer_kwargs["is_dynamic"] = is_dynamic
        self.activation_post_process = observer(**observer_kwargs)
        # TODO: keeping self.quant_min/max for BC; remove after a couple releases
        # Users should use self.activation_post_process.quant_min
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.is_dynamic = self.activation_post_process.is_dynamic
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )
        if not (_is_per_channel(self.qscheme) or _is_per_tensor(self.qscheme)):
            raise AssertionError(
                "Only per channel and per tensor quantization are supported in fake quantize"
                + " got qscheme: "
                + str(self.qscheme)
            )
        self.is_per_channel = _is_per_channel(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        # [FUNCTION DOC] Return the (scale, zero_point) computed by the observer from its statistics.
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        # [FUNCTION DOC] Fake quantize X. If the observer is enabled: observe a detached X, compute qparams, move them to the buffers' devices,
        # resize the scale/zero_point buffers if the shape changed, and copy the values in (mutating buffers). If fake quant is enabled:
        # apply the per-channel or per-tensor fake quantize op using the observer's quant range. Returns the result (X unchanged if disabled).
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = (
                _scale.to(self.scale.device),
                _zero_point.to(self.zero_point.device),
            )
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.ch_axis,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,
                )
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,
                )
        return X

    @torch.jit.export
    def extra_repr(self):
        # [FUNCTION DOC] Return a string describing the enabled flags, quant range, dtype, qscheme, ch_axis, scale and zero_point.
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"scale={self.scale}, zero_point={self.zero_point}"
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # [FUNCTION DOC] HELPER: Serialize like the parent and additionally store scale and zero_point explicitly under the prefixed keys (scalar-like values are not registered as plain buffers here).
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = self.scale
        destination[prefix + "zero_point"] = self.zero_point

    def _load_from_state_dict(
        # [FUNCTION DOC] HELPER: Load state, first resizing the scale and zero_point buffers to the saved shapes (they may start empty/different)
        # so the parent loader can copy values. A missing key is added to missing_keys when strict. Under TorchScript the values are also copied directly.
        # Raises AssertionError if the key name handling sees an unexpected name. Mutates buffers.
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ["scale", "zero_point"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "scale":
                    self.scale.resize_(val.shape)
                else:
                    if name != "zero_point":
                        raise AssertionError(
                            "Expected 'zero_point' but got different state key"
                        )
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == "scale":
                        self.scale.copy_(val)
                    else:
                        if name != "zero_point":
                            raise AssertionError(
                                "Expected 'zero_point' but got different state key"
                            )
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class FixedQParamsFakeQuantize(FakeQuantize):
    # [CLASS DOC] Fake quantize with fixed (non-learned) scale and zero_point taken from a FixedQParamsObserver. Only per-tensor schemes are allowed.
    # Keeps the observer constructor in _observer_ctr; qparams never change after construction.
    """Simulate quantize and dequantize in training time.

    Simulate quantize and dequantize with fixed quantization
    parameters in training time. Only per tensor quantization
    is supported.
    """

    # TODO: rename observer to observer_ctr
    def __init__(self, observer):
        # [FUNCTION DOC] Initialise via FakeQuantize with the given observer. Raises AssertionError if the built observer is not exactly a FixedQParamsObserver
        # or if the qscheme is not per-tensor. Saves the observer constructor and takes scale and zero_point from the observer.
        super().__init__(observer=observer)
        if type(self.activation_post_process) is not FixedQParamsObserver:
            raise AssertionError(
                f"{self.__class__.__name__}'s observer must be a {FixedQParamsObserver.__name__}"
            )
        self._observer_ctr = observer
        self.scale = self.activation_post_process.scale
        self.zero_point = self.activation_post_process.zero_point
        if not _is_per_tensor(self.qscheme):
            raise AssertionError(
                "Only per tensor quantization is supported"
                + " FixedQParamsFakeQuantize module, got qscheme:"
                + str(self.qscheme)
            )

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        # [FUNCTION DOC] Return the fixed (scale, zero_point) stored on this module.
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self):
        # [FUNCTION DOC] Return a string describing flags, scale, zero_point, dtype, quant range and qscheme.
        """Define a string representation of the object's attributes."""
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"scale={self.scale}, zero_point={self.zero_point}, "
            f"dtype={self.dtype}, quant_min={self.activation_post_process.quant_min}, "
            f"quant_max={self.activation_post_process.quant_max}, qscheme={self.qscheme}"
        )


class FusedMovingAvgObsFakeQuantize(FakeQuantize):
    # [CLASS DOC] FakeQuantize variant that observes (moving average min/max), computes qparams and fake quantizes in one fused op.
    # Requires a MovingAverageMinMaxObserver or per-channel version. Flag buffers are long dtype. Stores is_symmetric_quant.
    r"""Define a fused module to observe the tensor.

    Fused module that is used to observe the input tensor (compute min/max), compute
    scale/zero_point and fake_quantize the tensor.
    This module uses calculation similar MovingAverageMinMaxObserver for the inputs,
    to compute the min/max values in order to compute the scale/zero_point.
    The qscheme input in the observer is used to differentiate between symmetric/affine
    quantization scheme.

    The output of this module is given by
    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale

    Similar to :class:`~torch.ao.quantization.FakeQuantize`, and accepts the same attributes as the
    base class.

    """

    def __init__(
        self,
        observer: Any = MovingAverageMinMaxObserver,
        quant_min: int = 0,
        quant_max: int = 255,
        **observer_kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Initialise via FakeQuantize (default quant range 0 to 255). Raises AssertionError if the observer is not a moving average min/max
        # observer (per-tensor or per-channel). Re-registers fake_quant_enabled and observer_enabled as long buffers set to 1
        # and records whether the qscheme is symmetric.
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        if not isinstance(
            self.activation_post_process,
            (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver),
        ):
            raise AssertionError(
                "Fused observer+fake_quant module only works with MovingAverageMinMaxObserver"
            )
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.long))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.long))
        self.is_symmetric_quant = _is_symmetric_quant(
            self.activation_post_process.qscheme
        )

    @torch.jit.export
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        # [FUNCTION DOC] Return (scale, zero_point) computed by the observer.
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return a string describing flags, scale, zero_point, dtype, quant range, qscheme and observer reduce_range.
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"qscheme={self.qscheme}, reduce_range={self.activation_post_process.reduce_range}"
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Call the fused moving-average observer + fake quant kernel with the flags, the observer's min/max buffers, scale, zero_point,
        # averaging constant, quant range, channel axis and per-channel/symmetric options. Returns the fake quantized tensor and updates observer state in place inside the kernel.
        return torch.fused_moving_avg_obs_fake_quant(
            X,
            self.observer_enabled,
            self.fake_quant_enabled,
            self.activation_post_process.min_val,
            self.activation_post_process.max_val,
            self.scale,
            self.zero_point,
            self.activation_post_process.averaging_constant,
            self.activation_post_process.quant_min,
            self.activation_post_process.quant_max,
            self.ch_axis,
            self.is_per_channel,
            self.is_symmetric_quant,
        )


default_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=True,
)
"""
Default fake_quant for activations.
"""

default_weight_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    reduce_range=False,
)
"""
Default fake_quant for weights.
Observer is memoryless since averaging_constant is 1.
"""

default_dynamic_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    is_dynamic=True,
    dtype=torch.quint8,
    averaging_constant=1,
)
"""
Default dynamic fake_quant for activations.
"""

default_fixed_qparams_range_neg1to1_fake_quant = FixedQParamsFakeQuantize.with_args(
    observer=default_fixed_qparams_range_neg1to1_observer
)
default_fixed_qparams_range_0to1_fake_quant = FixedQParamsFakeQuantize.with_args(
    observer=default_fixed_qparams_range_0to1_observer
)
# TODO: the following 2 variables are kept for backwards compatibility; remove after a few releases
default_symmetric_fixed_qparams_fake_quant = (
    default_fixed_qparams_range_neg1to1_fake_quant
)
default_affine_fixed_qparams_fake_quant = default_fixed_qparams_range_0to1_fake_quant

default_per_channel_weight_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    reduce_range=False,
    ch_axis=0,
)
"""
Default fake_quant for per-channel weights.
Observer is memoryless since averaging_constant is 1.
"""
default_embedding_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    qscheme=torch.per_channel_affine_float_qparams,
    dtype=torch.quint8,
    quant_min=0,
    quant_max=255,
    ch_axis=0,
    averaging_constant=1,
)
"""
Default fake_quant for embeddings.
Observer is memoryless since averaging_constant is 1.
"""

default_embedding_fake_quant_4bit = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    qscheme=torch.per_channel_affine_float_qparams,
    ch_axis=0,
    dtype=torch.quint4x2,
    averaging_constant=1,
)

default_histogram_fake_quant = FakeQuantize.with_args(
    observer=HistogramObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=True,
)
"""
Fake_quant for activations using a histogram..
"""


default_fused_act_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
)

"""
Fused version of `default_fake_quant`, with improved performance.
"""


default_fused_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
)
"""
Fused version of `default_weight_fake_quant`, with improved performance.
"""

default_fused_per_channel_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
)
"""
Fused version of `default_per_channel_weight_fake_quant`, with improved performance.
"""

fused_wt_fake_quant_range_neg_127_to_127 = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-127,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    eps=2**-12,
)
"""
Fused version of `default_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""

fused_per_channel_wt_fake_quant_range_neg_127_to_127 = (
    FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        eps=2**-12,
    )
)

"""
Fused version of `default_per_channel_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""


def _is_fake_quant_script_module(mod):
    # [FUNCTION DOC] HELPER: True if mod is a TorchScript RecursiveScriptModule whose qualified name (with mangling suffix removed) is FakeQuantize or FusedMovingAvgObsFakeQuantize; False otherwise.
    """Return true if given mod is an instance of FakeQuantize script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.ao.quantization.fake_quantize.___torch_mangle_2.FakeQuantize'
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return (
            name == "torch.ao.quantization.fake_quantize.FakeQuantize"
            or name
            == "torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize"
        )
    return False


def disable_fake_quant(mod):
    # [FUNCTION DOC] Disable fake quantization if mod is a FakeQuantizeBase or scripted fake quant module; otherwise do nothing. Intended for Module.apply.
    """Disable fake quantization for the module.

    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_fake_quant()


def enable_fake_quant(mod):
    # [FUNCTION DOC] Enable fake quantization if mod is a FakeQuantizeBase or scripted fake quant module; otherwise do nothing. Intended for Module.apply.
    """Enable fake quantization for the module.

    Enable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_fake_quant)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_fake_quant()


def disable_observer(mod):
    # [FUNCTION DOC] Disable observation if mod is a FakeQuantizeBase or scripted fake quant module; otherwise do nothing. Intended for Module.apply.
    """Disable observation for this module.

    Disable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_observer)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_observer()


def enable_observer(mod):
    # [FUNCTION DOC] Enable observation if mod is a FakeQuantizeBase or scripted fake quant module; otherwise do nothing. Intended for Module.apply.
    """Enable observation for this module.

    Enable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_observer)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_observer()
