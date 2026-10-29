# mypy: allow-untyped-decorators
# [MODULE DOC] AdaroundFakeQuantizer: a FakeQuantize subclass implementing adaptive rounding (Adaround, arXiv 2004.10568)
# for weights with symmetric quantization. Rounding is learned through a trainable parameter V passed through a rectified sigmoid.
from __future__ import annotations

from typing import Any

import torch
from torch.ao.quantization.fake_quantize import _is_symmetric_quant
from torch.ao.quantization.utils import is_per_tensor
from torch.quantization import FakeQuantize
from torch.quantization.observer import MinMaxObserver


class AdaroundFakeQuantizer(FakeQuantize):
    # [CLASS DOC] Fake quantizer with learnable rounding offsets. State: scale and zero_point tensors (empty until the observer
    # runs), V (trainable parameter of the same shape as the weight, initialised from the fractional part of the scaled weight),
    # fixed stretch constants zeta and gamma, a sigmoid module, and use_soft_rounding (soft sigmoid vs hard binary rounding).
    # Only symmetric qschemes are accepted. Observed qparams are copied into scale/zero_point; only V is meant to be trained.
    """
    This is a FakeQuantizer that enables an adaptive rounding fake quantizer.
    Adaround is a technique to adaptively round weights, derived from the paper https://arxiv.org/pdf/2004.10568.pdf
    For HTP compatibility, we are targeting to use symmetric quantization
    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    V: torch.nn.Parameter

    def __init__(
        self,
        observer: type = MinMaxObserver,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,  # not used, but needed for fakequant
        quant_min: int = -128,
        quant_max: int = 127,
        ch_axis: int = 0,
        **observer_kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Construct the fake quantizer via the FakeQuantize base with a static (non-dynamic) observer.
        # Args: observer class, qscheme (must be symmetric), quant_min/quant_max, ch_axis, extra observer kwargs.
        # Raises AssertionError if quant_min exceeds quant_max or the qscheme is not symmetric.
        # Creates empty scale/zero_point, an empty trainable V, constants zeta=1.1 and gamma=-0.1, and enables soft rounding.
        super().__init__(
            observer=observer,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_dynamic=False,
            **observer_kwargs,
        )
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            if quant_min > quant_max:
                raise AssertionError(
                    "quant_min must be less than or equal to quant_max, "
                    f"got quant_min:{quant_min}, quant_max:{quant_max}"
                )
        self.qscheme: torch.qscheme = qscheme
        self.is_per_tensor: bool = is_per_tensor(qscheme)
        self.is_symmetric: bool = _is_symmetric_quant(qscheme)
        if not self.is_symmetric:
            raise AssertionError("Only symmetric quantization is supported")
        self.ch_axis: int = ch_axis

        self.scale = torch.tensor([], requires_grad=False)
        self.zero_point = torch.tensor([], requires_grad=False)
        self.V = torch.nn.Parameter(torch.tensor([]), requires_grad=True)
        # Fixed Stretch parameters
        self.zeta: torch.Tensor = torch.tensor(1.1, requires_grad=False)
        self.gamma: torch.Tensor = torch.tensor(-0.1, requires_grad=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.use_soft_rounding = True

    @torch.jit.export
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        # [FUNCTION DOC] Return the stored (scale, zero_point) tensors rather than querying the observer.
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return a string describing fake quant/observer flags, quant range, dtype, qscheme, ch_axis, scale, zero_point
        # and the count of non-negative entries of V.
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"scale={self.scale}, zero_point={self.zero_point}, (self.V >= 0).int().sum()={(self.V >= 0).int().sum()}"
        )

    def enable_weight_fake_quant(self) -> None:
        # [FUNCTION DOC] Turn on fake quantization by setting the first element of the fake_quant_enabled flag tensor to 1.
        self.fake_quant_enabled[0] = 1

    def get_rectified_sigmoid_func(self) -> torch.Tensor:
        # [FUNCTION DOC] Return the rounding offset h(V): with soft rounding, the sigmoid of V stretched by zeta/gamma and clamped to [0, 1];
        # otherwise a hard binary tensor that is 1 where V is non-negative and 0 elsewhere.
        if self.use_soft_rounding:
            return torch.clamp(
                self.sigmoid(self.V) * (self.zeta - self.gamma) + self.gamma,
                min=0,
                max=1,
            )
        else:
            # This will dump a binary solution
            return (self.V >= 0).int()

    @torch.jit.ignore
    def update_scale(
        self, X: torch.Tensor, _scale: torch.Tensor, _zero_point: torch.Tensor
    ) -> None:
        # [FUNCTION DOC] Update scale/zero_point from newly observed values and re-initialise V.
        # Args: X weight tensor, _scale and _zero_point from the observer. On first call both are moved to X's device;
        # later the scale data is replaced and zero_point is set (zeros for symmetric) and unsqueezed along every non-channel dimension.
        # Then V is set so the rectified sigmoid reproduces the fractional part of X divided by scale.
        # Raises AssertionError if the fractional part is negative. Mutates scale, zero_point and V.
        if self.scale.numel() == 0:
            self.scale.data = _scale.to(X.device)
            self.zero_point = _zero_point.to(X.device)
        else:
            self.scale.data = _scale
            if not self.is_symmetric:
                self.zero_point = _zero_point
            else:
                self.zero_point = torch.zeros_like(_zero_point)
            for i in range(X.dim()):
                if i == self.ch_axis:
                    continue
                self.zero_point = self.zero_point.unsqueeze(i)
        X_q = X / self.scale
        X_q_floor = torch.floor(X_q)
        residual = X_q - X_q_floor  # [0,1)
        if not torch.all(torch.ge(residual, 0)):
            raise AssertionError("residual should be non-negative in [0, 1)")
        V_init = -torch.log((self.zeta - self.gamma) / (residual - self.gamma) - 1)
        self.V.data = V_init

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Forward pass. If the observer is enabled: observe a detached copy of X, fetch its qparams, move them to the
        # stored devices, reshape per-channel qparams to broadcast, and call update_scale. If fake quant is enabled: compute the
        # floor of X over scale plus zero_point, add the learned rounding offset, clamp to the quant range, then dequantize,
        # so gradients flow to V; otherwise return X unchanged.
        # KNOWN ISSUE: the local variable dims is computed (and a per-channel branch repeated) but never used.
        if self.observer_enabled[0] == 1:
            X_detached = X.detach()
            self.activation_post_process(X_detached)
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale, _zero_point = (
                _scale.to(self.scale.device),
                _zero_point.to(self.zero_point.device),
            )
            dims = list(range(X.dim()))
            if not self.is_per_tensor:
                dims.remove(self.ch_axis)
            if not self.is_per_tensor:
                for i in range(X.dim()):
                    if i == self.ch_axis:
                        continue
                    _scale = _scale.unsqueeze(i)
                    _zero_point = _zero_point.unsqueeze(i)
            self.update_scale(X_detached, _scale, _zero_point)

        if self.fake_quant_enabled[0] == 1:
            # Perform soft quantization
            # See the equation (23) in Adaround paper
            h_v = self.get_rectified_sigmoid_func()
            X_q = X / self.scale
            # Straight-Through Estimator for floor function
            X_q_floor = torch.floor(X_q) + self.zero_point
            # Regardless of rounding, gradient should be able to flow back to self.V from X_q_dq.
            # With adaround, we don't train weight, but train V only.
            X_q_dq = (
                torch.clamp(X_q_floor + h_v, min=self.quant_min, max=self.quant_max)
                - self.zero_point
            ) * self.scale
            return X_q_dq
        else:
            return X
