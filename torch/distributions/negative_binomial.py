# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the NegativeBinomial distribution (count of successes before a given
# number of failures), parameterized by total_count and either probs or logits;
# sampling uses a Gamma-Poisson mixture. Exports NegativeBinomial. No import-time
# side effects.

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)


__all__ = ["NegativeBinomial"]


class NegativeBinomial(Distribution):
    # [CLASS DOC] Negative binomial distribution over non-negative integers with
    # non-negative (possibly real) total_count and success probability given as
    # probs or logits (only one stored eagerly, the other computed lazily). All
    # parameters are broadcast to one shape which is the batch shape. State:
    # total_count, probs/logits, and _param (whichever was supplied). Not reparameterized.
    r"""
    Creates a Negative Binomial distribution, i.e. distribution
    of the number of successful independent and identical Bernoulli trials
    before :attr:`total_count` failures are achieved. The probability
    of success of each Bernoulli trial is :attr:`probs`.

    Args:
        total_count (float or Tensor): non-negative number of negative Bernoulli
            trials to stop, although the distribution is still valid for real
            valued count
        probs (Tensor): Event probabilities of success in the half open interval [0, 1)
        logits (Tensor): Event log-odds for probabilities of success
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "total_count": constraints.greater_than_eq(0),
        "probs": constraints.half_open_interval(0.0, 1.0),
        "logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Tensor | float,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Args: total_count (float or tensor), and exactly one of
        # probs or logits. Raises ValueError if both or neither are given. Broadcasts
        # total_count with the chosen parameter, casts total_count to its dtype,
        # remembers the chosen parameter as _param and uses its shape as batch shape.
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            (
                self.total_count,
                # pyrefly: ignore [read-only]
                self.probs,
            ) = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.probs)
        else:
            if logits is None:
                raise AssertionError("logits is unexpectedly None")
            (
                self.total_count,
                # pyrefly: ignore [read-only]
                self.logits,
            ) = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)

        self._param = self.probs if probs is not None else self.logits
        batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new NegativeBinomial (or _instance) with
        # total_count and whichever of probs/logits were already materialized
        # expanded to batch_shape; _param points at the last one set. Copies the validation flag.
        new = self._get_checked_instance(NegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(NegativeBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Creates a new tensor with the dtype/device of _param.
        return self._param.new(*args, **kwargs)

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Property: total_count times the odds of success (exponential of logits).
        return self.total_count * torch.exp(self.logits)

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Property: floor of (total_count - 1) times the odds,
        # clamped to be at least zero.
        return ((self.total_count - 1) * self.logits.exp()).floor().clamp(min=0.0)

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Property: mean divided by the failure probability (sigmoid of minus logits).
        return self.mean / torch.sigmoid(-self.logits)

    @lazy_property
    def logits(self) -> Tensor:
        # [FUNCTION DOC] Lazy property (computed once, then cached): logits
        # derived from probs using the binary conversion.
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        # [FUNCTION DOC] Lazy property (computed once, then cached): probabilities
        # derived from logits using the binary conversion.
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self) -> torch.Size:
        # [FUNCTION DOC] Property: size of the stored parameter tensor.
        return self._param.size()

    @lazy_property
    def _gamma(self) -> Gamma:
        # [FUNCTION DOC] HELPER: Lazy cached Gamma mixing distribution with
        # concentration total_count and rate exp(-logits); argument validation is
        # disabled because total_count may be zero.
        # Note we avoid validating because self.total_count can be zero.
        return Gamma(
            concentration=self.total_count,
            rate=torch.exp(-self.logits),
            validate_args=False,
        )

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC] Draws a gamma rate for each element, then a Poisson count
        # with that rate (gamma-Poisson mixture). Runs without gradient tracking and
        # consumes RNG state. Returns a tensor of sample_shape + batch shape.
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return torch.poisson(rate)

    def log_prob(self, value):
        # [FUNCTION DOC] Returns the log probability mass at value: log-sigmoid
        # terms for failures and successes minus a log-gamma based normalization.
        # The case of zero total_count and zero value is masked to a zero
        # normalization to avoid infinite log-gamma, without in-place mutation.
        # Validates value when enabled (may raise).
        if self._validate_args:
            self._validate_sample(value)

        log_unnormalized_prob = self.total_count * F.logsigmoid(
            -self.logits
        ) + value * F.logsigmoid(self.logits)

        log_normalization = (
            -torch.lgamma(self.total_count + value)
            + torch.lgamma(1.0 + value)
            + torch.lgamma(self.total_count)
        )
        # The case self.total_count == 0 and value == 0 has probability 1 but
        # lgamma(0) is infinite. Handle this case separately using a function
        # that does not modify tensors in place to allow Jit compilation.
        log_normalization = log_normalization.masked_fill(
            self.total_count + value == 0.0, 0.0
        )

        return log_unnormalized_prob - log_normalization
