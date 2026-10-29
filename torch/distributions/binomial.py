# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Binomial distribution (number of successes in total_count trials),
# parameterized by probs or logits, plus a small clamp helper. Exports Binomial
# via __all__. No import-time side effects.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)


__all__ = ["Binomial"]


def _clamp_by_zero(x):
    # [FUNCTION DOC] HELPER: elementwise maximum of x and zero computed as the
    # average of clamp-min-zero plus x minus clamp-max-zero, so that the gradient
    # at exactly zero is one half. Returns a tensor shaped like x.
    # works like clamp(x, min=0) but has grad at 0 is 0.5
    return (x.clamp(min=0) + x - x.clamp(max=0)) / 2


class Binomial(Distribution):
    # [CLASS DOC]
    # Binomial distribution with tensor attributes total_count (nonnegative
    # integer valued, stored in the dtype of the probability parameter) and one
    # of probs/logits (the other computed lazily and cached). _param tracks the
    # supplied one and fixes the batch shape. Support depends on total_count;
    # discrete and enumerable (only for homogeneous total_count), no rsample.
    r"""
    Creates a Binomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be
    broadcastable with :attr:`probs`/:attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
        tensor([   0.,   22.,   71.,  100.])

        >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
        >>> x = m.sample()
        tensor([[ 4.,  5.],
                [ 7.,  6.]])

    Args:
        total_count (int or Tensor): number of Bernoulli trials
        probs (Tensor): Event probabilities
        logits (Tensor): Event log-odds
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "probs": constraints.unit_interval,
        "logits": constraints.real,
    }
    has_enumerate_support = True

    def __init__(
        self,
        total_count: Tensor | int = 1,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Broadcasts total_count with the supplied probs or logits, casts
        # total_count to the parameter dtype, and sets the batch shape to the
        # broadcast parameter shape. Raises ValueError if both or neither of
        # probs and logits are provided.
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
        # [FUNCTION DOC]
        # Returns a new Binomial with total_count and whichever of probs/logits
        # is materialized expanded to batch_shape; reuses _instance if given.
        # Preserves the validate_args setting. Does not mutate self.
        new = self._get_checked_instance(Binomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Binomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: creates a new tensor of the same type and device
        # as _param, forwarding all arguments to its new method.
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    # pyrefly: ignore [bad-override]
    def support(self):
        # [FUNCTION DOC]
        # Dependent property: integer interval from zero to total_count inclusive.
        return constraints.integer_interval(0, self.total_count)

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: total_count times probs.
        return self.total_count * self.probs

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: floor of (total_count plus one) times probs, clamped to at
        # most total_count.
        return ((self.total_count + 1) * self.probs).floor().clamp(max=self.total_count)

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: total_count times probs times one minus probs.
        return self.total_count * self.probs * (1 - self.probs)

    @lazy_property
    def logits(self) -> Tensor:
        # [FUNCTION DOC]
        # Lazy cached property: converts probs to binary logits when logits were
        # not supplied at construction.
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        # [FUNCTION DOC]
        # Lazy cached property: converts logits to binary probabilities when
        # probs were not supplied at construction.
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self) -> torch.Size:
        # [FUNCTION DOC]
        # Property: shape of the stored parameter tensor.
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC]
        # Draws samples of shape sample_shape plus batch shape using the
        # binomial sampler on total_count and probs expanded to that shape,
        # without gradient tracking. Consumes global RNG state.
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.binomial(
                self.total_count.expand(shape), self.probs.expand(shape)
            )

    def log_prob(self, value):
        # [FUNCTION DOC]
        # Returns the elementwise log probability mass at value, combining the
        # log binomial coefficient (via log-gamma) with a numerically stable
        # logit-based form of the success/failure terms. Validates value when
        # validation is enabled (raises ValueError).
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        # k * log(p) + (n - k) * log(1 - p) = k * (log(p) - log(1 - p)) + n * log(1 - p)
        #     (case logit < 0)              = k * logit - n * log1p(e^logit)
        #     (case logit > 0)              = k * logit - n * (log(p) - log(1 - p)) + n * log(p)
        #                                   = k * logit - n * logit - n * log1p(e^-logit)
        #     (merge two cases)             = k * logit - n * max(logit, 0) - n * log1p(e^-|logit|)
        normalize_term = (
            self.total_count * _clamp_by_zero(self.logits)
            + self.total_count * torch.log1p(torch.exp(-torch.abs(self.logits)))
            - log_factorial_n
        )
        return (
            value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term
        )

    def entropy(self):
        # [FUNCTION DOC]
        # Returns the entropy by summing probability times log probability over
        # the full enumerated support. Raises NotImplementedError if total_count
        # is not the same everywhere in the batch.
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `entropy`."
            )

        log_prob = self.log_prob(self.enumerate_support(False))
        return -(torch.exp(log_prob) * log_prob).sum(0)

    def enumerate_support(self, expand=True):
        # [FUNCTION DOC]
        # Returns the integers zero through total_count as a tensor of parameter
        # dtype/device, with a leading support dimension and singleton batch
        # dims; expanded to the full batch shape when expand is true. Raises
        # NotImplementedError if total_count differs across the batch.
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `enumerate_support`."
            )
        values = torch.arange(
            1 + total_count, dtype=self._param.dtype, device=self._param.device
        )
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values
