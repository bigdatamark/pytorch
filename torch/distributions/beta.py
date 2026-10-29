# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Beta distribution on the unit interval, implemented as a
# two-component Dirichlet wrapper. Exports Beta via __all__. No import-time
# side effects.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Beta"]


class Beta(ExponentialFamily):
    # [CLASS DOC]
    # Beta distribution with positive parameters concentration1 and
    # concentration0, supported on the unit interval, with reparameterized
    # sampling. State is held in a private Dirichlet over two components whose
    # last dimension stacks (concentration1, concentration0); all sampling,
    # density and entropy computations delegate to it. Is an exponential family.
    r"""
    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
        tensor([ 0.1046])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Tensor | float,
        concentration0: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the wrapped Dirichlet. For two plain numbers, makes a length-two
        # float tensor (scalar batch). Otherwise broadcasts both parameters to a
        # common shape and stacks them along a new last dimension. Batch shape
        # is taken from the Dirichlet. validate_args is forwarded.
        if isinstance(concentration1, _Number) and isinstance(concentration0, _Number):
            concentration1_concentration0 = torch.tensor(
                [float(concentration1), float(concentration0)]
            )
        else:
            concentration1, concentration0 = broadcast_all(
                concentration1, concentration0
            )
            concentration1_concentration0 = torch.stack(
                [concentration1, concentration0], -1
            )
        self._dirichlet = Dirichlet(
            concentration1_concentration0, validate_args=validate_args
        )
        super().__init__(self._dirichlet._batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC]
        # Returns a new Beta whose wrapped Dirichlet is expanded to batch_shape;
        # reuses _instance if given. Preserves the validate_args setting.
        # Does not mutate self.
        new = self._get_checked_instance(Beta, _instance)
        batch_shape = torch.Size(batch_shape)
        new._dirichlet = self._dirichlet.expand(batch_shape)
        super(Beta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: concentration1 divided by the sum of both concentrations.
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: the first component of the wrapped Dirichlet's mode.
        return self._dirichlet.mode[..., 0]

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: product of the concentrations divided by the squared total
        # times the total plus one.
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total.pow(2) * (total + 1))

    def rsample(self, sample_shape: _size = ()) -> Tensor:
        # [FUNCTION DOC]
        # Draws a reparameterized sample by sampling the wrapped Dirichlet and
        # selecting its first component along the last dimension. Result shape
        # is sample_shape plus batch shape. Consumes global RNG state.
        return self._dirichlet.rsample(sample_shape).select(-1, 0)

    def log_prob(self, value):
        # [FUNCTION DOC]
        # Returns the log density at value: validates value against the support
        # when validation is enabled, stacks value with one minus value as a
        # two-component simplex point, and evaluates the Dirichlet log density.
        if self._validate_args:
            self._validate_sample(value)
        heads_tails = torch.stack([value, 1.0 - value], -1)
        return self._dirichlet.log_prob(heads_tails)

    def entropy(self):
        # [FUNCTION DOC]
        # Returns the entropy, delegated to the wrapped Dirichlet.
        return self._dirichlet.entropy()

    @property
    def concentration1(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: first Dirichlet concentration component (last-dim index 0).
        # KNOWN ISSUE: the number check on the indexed tensor can never be true
        # for a tensor, so the wrap-in-tensor branch is effectively dead code.
        result = self._dirichlet.concentration[..., 0]
        if isinstance(result, _Number):
            return torch.tensor([result])
        else:
            return result

    @property
    def concentration0(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: second Dirichlet concentration component (last-dim index 1).
        # KNOWN ISSUE: same dead number-check branch as concentration1.
        result = self._dirichlet.concentration[..., 1]
        if isinstance(result, _Number):
            return torch.tensor([result])
        else:
            return result

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        # [FUNCTION DOC] HELPER: exponential-family natural parameters, the pair
        # (concentration1, concentration0).
        return (self.concentration1, self.concentration0)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        # [FUNCTION DOC] HELPER: log normalizer of the exponential family, the
        # log-gamma of x plus log-gamma of y minus log-gamma of x plus y.
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
