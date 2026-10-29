# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Dirichlet distribution (exponential family) over the simplex, with
# reparameterized sampling via a custom autograd Function whose backward uses the
# implicit-gradient kernel torch._dirichlet_grad. No import-time side effects.
# Exports: Dirichlet.

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.types import _size


__all__ = ["Dirichlet"]


# This helper is exposed for testing.
def _Dirichlet_backward(x, concentration, grad_output):
    # [FUNCTION DOC] HELPER: Computes the gradient of Dirichlet samples with respect to concentration: the implicit-gradient kernel evaluated at samples x, the concentration and its last-dim total, multiplied by grad_output minus the x-weighted sum of grad_output over the last dim. Returns a tensor shaped like concentration.
    total = concentration.sum(-1, True).expand_as(concentration)
    grad = torch._dirichlet_grad(x, concentration, total)
    return grad * (grad_output - (x * grad_output).sum(-1, True))


class _Dirichlet(Function):
    # [CLASS DOC] HELPER: Autograd Function that draws Dirichlet samples from a concentration tensor and provides pathwise (implicit reparameterization) gradients with respect to concentration.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, concentration):
        # [FUNCTION DOC] HELPER: Samples from the Dirichlet distribution with the given concentration (consumes random state), saves samples and concentration for backward, and returns the samples.
        x = torch._sample_dirichlet(concentration)
        ctx.save_for_backward(x, concentration)
        return x

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Restores saved samples and concentration and returns the gradient with respect to concentration (once differentiable only: not double-backward capable).
        x, concentration = ctx.saved_tensors
        return _Dirichlet_backward(x, concentration, grad_output)


class Dirichlet(ExponentialFamily):
    # [CLASS DOC]
    # Dirichlet distribution over vectors on the unit simplex. State: concentration
    # tensor whose last dimension is the event dimension (batch shape is all leading
    # dims). Support is the simplex; sampling is reparameterized.
    r"""
    Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentration [0.5, 0.5]
        tensor([ 0.1046,  0.8954])

    Args:
        concentration (Tensor): concentration parameter of the distribution
            (often referred to as alpha)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1)
    }
    support = constraints.simplex
    has_rsample = True

    def __init__(
        self,
        concentration: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores concentration; batch shape is its leading dims and event shape its last dim. Raises ValueError if concentration has no dimensions. Argument validation may raise via the base class.
        if concentration.dim() < 1:
            raise ValueError(
                "`concentration` parameter must be at least one-dimensional."
            )
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new Dirichlet whose concentration is expanded to batch_shape plus the event shape, preserving the validation setting; self is unchanged.
        new = self._get_checked_instance(Dirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape + self.event_shape)
        super(Dirichlet, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = ()) -> Tensor:
        # [FUNCTION DOC] Draws reparameterized samples of shape sample_shape plus batch and event shape by expanding concentration and applying the custom autograd sampler. Consumes random state.
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        return _Dirichlet.apply(concentration)

    def log_prob(self, value):
        # [FUNCTION DOC] Log density at value on the simplex: sum over the last dim of (concentration minus one) times log value (zero where both are zero), plus log-gamma of the concentration total minus the sum of log-gammas. Validates value if validation is on (ValueError).
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration - 1.0, value).sum(-1)
            + torch.lgamma(self.concentration.sum(-1))
            - torch.lgamma(self.concentration).sum(-1)
        )

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Mean: concentration divided by its sum over the last dim.
        return self.concentration / self.concentration.sum(-1, True)

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC]
        # Mode: concentration minus one clamped at zero then normalized over the last dim.
        # For batch entries whose concentrations are all below one, the result is replaced
        # by a one-hot vector at the argmax of the normalized values.
        # KNOWN ISSUE: in the all-below-one case the clamped values are all zero, so the
        # normalization is 0/0 (NaN) and the argmax picks an arbitrary index rather than
        # the largest concentration; the true density has no interior mode there.
        concentrationm1 = (self.concentration - 1).clamp(min=0.0)
        mode = concentrationm1 / concentrationm1.sum(-1, True)
        mask = (self.concentration < 1).all(dim=-1)
        mode[mask] = torch.nn.functional.one_hot(
            mode[mask].argmax(dim=-1), concentrationm1.shape[-1]
        ).to(mode)
        return mode

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Per-component variance: c_i times (total minus c_i) divided by total squared times (total plus one), where total is the concentration sum.
        con0 = self.concentration.sum(-1, True)
        return (
            self.concentration
            * (con0 - self.concentration)
            / (con0.pow(2) * (con0 + 1))
        )

    def entropy(self):
        # [FUNCTION DOC] Differential entropy from log-gamma and digamma of the concentration and its total, summed over the last dim. Returns a tensor of batch shape.
        k = self.concentration.size(-1)
        a0 = self.concentration.sum(-1)
        return (
            torch.lgamma(self.concentration).sum(-1)
            - torch.lgamma(a0)
            - (k - a0) * torch.digamma(a0)
            - ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1)
        )

    @property
    def _natural_params(self) -> tuple[Tensor]:
        # [FUNCTION DOC] HELPER: Natural parameters as a one-element tuple holding the concentration.
        return (self.concentration,)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        # [FUNCTION DOC] HELPER: Log normalizer at natural parameter x: sum of log-gamma of x over the last dim minus log-gamma of the sum of x.
        return x.lgamma().sum(-1) - torch.lgamma(x.sum(-1))
