# [MODULE DOC] Loss functions for Adaround: AdaptiveRoundingLoss combines a rounding regularizer and an MSE reconstruction loss.
# Defines constants ADAROUND_ZETA (1.1) and ADAROUND_GAMMA (-0.1) used by the rectified sigmoid.
import numpy as np

import torch
from torch.nn import functional as F


ADAROUND_ZETA: float = 1.1
ADAROUND_GAMMA: float = -0.1


class AdaptiveRoundingLoss(torch.nn.Module):
    # [CLASS DOC] torch Module computing the Adaround regularization and reconstruction terms (paper equations 24 and 25).
    # State: max_iter, warm_start fraction, beta_range (start, end) annealed over iterations, and reg_param weight.
    """
    Adaptive Rounding Loss functions described in https://arxiv.org/pdf/2004.10568.pdf
    rounding regularization is eq [24]
    reconstruction loss is eq [25] except regularization term
    """

    def __init__(
        self,
        max_iter: int,
        warm_start: float = 0.2,
        beta_range: tuple[int, int] = (20, 2),
        reg_param: float = 0.001,
    ) -> None:
        # [FUNCTION DOC] Store max_iter, warm_start, beta_range and reg_param on the module.
        super().__init__()
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.beta_range = beta_range
        self.reg_param = reg_param

    def rounding_regularization(
        self,
        V: torch.Tensor,
        curr_iter: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Compute the weighted rounding regularization for the rounding variable V at iteration curr_iter.
        # Raises AssertionError if curr_iter is not less than max_iter. During the warm start fraction of iterations returns a zero tensor.
        # After that, beta is annealed by a cosine schedule from the start to the end value, the rectified sigmoid of V is computed,
        # and the result is the sum of one minus the absolute value of (2h-1) raised to beta, scaled by reg_param.
        """
        Major logics copied from official Adaround Implementation.
        Apply rounding regularization to the input tensor V.
        """
        if curr_iter >= self.max_iter:
            raise AssertionError("Current iteration strictly les sthan max iteration")
        if curr_iter < self.warm_start * self.max_iter:
            return torch.tensor(0.0)
        else:
            start_beta, end_beta = self.beta_range
            warm_start_end_iter = self.warm_start * self.max_iter

            # compute relative iteration of current iteration
            rel_iter = (curr_iter - warm_start_end_iter) / (
                self.max_iter - warm_start_end_iter
            )
            beta = end_beta + 0.5 * (start_beta - end_beta) * (
                1 + np.cos(rel_iter * np.pi)
            )

            # A rectified sigmoid for soft-quantization as formulated [23] in https://arxiv.org/pdf/2004.10568.pdf
            h_alpha = torch.clamp(
                torch.sigmoid(V) * (ADAROUND_ZETA - ADAROUND_GAMMA) + ADAROUND_GAMMA,
                min=0,
                max=1,
            )

            # Apply rounding regularization
            # This regularization term helps out term to converge into binary solution either 0 or 1 at the end of optimization.
            inner_term = torch.add(2 * h_alpha, -1).abs().pow(beta)
            regularization_term = torch.add(1, -inner_term).sum()
            return regularization_term * self.reg_param

    def reconstruction_loss(
        self,
        soft_quantized_output: torch.Tensor,
        original_output: torch.Tensor,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Return the mean squared error between the soft quantized output and the original output.
        """
        Compute the reconstruction loss between the soft quantized output and the original output.
        """
        return F.mse_loss(
            soft_quantized_output, original_output, reduction="none"
        ).mean()

    def forward(
        self,
        soft_quantized_output: torch.Tensor,
        original_output: torch.Tensor,
        V: torch.Tensor,
        curr_iter: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] Return the pair (regularization term, reconstruction term) for the given outputs, V and iteration.
        # Note it does not sum them; the caller combines the two terms.
        # KNOWN ISSUE: docstring mentions only the reconstruction formulation but a tuple of two terms is returned.
        """
        Compute the asymmetric reconstruction formulation as eq [25]
        """
        regularization_term = self.rounding_regularization(V, curr_iter)
        reconstruction_term = self.reconstruction_loss(
            soft_quantized_output, original_output
        )
        return regularization_term, reconstruction_term
