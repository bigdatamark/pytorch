# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Concrete/relaxed Bernoulli family: LogitRelaxedBernoulli (samples
# are logits of relaxed values) and RelaxedBernoulli (sigmoid-transformed, values
# in (0, 1), reparameterized). No import-time side effects.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.types import _Number, _size, Number


__all__ = ["LogitRelaxedBernoulli", "RelaxedBernoulli"]


class LogitRelaxedBernoulli(Distribution):
    # [CLASS DOC] Distribution of the logit of a relaxed Bernoulli variable, with
    # a relaxation temperature tensor and success probs or logits (one stored
    # eagerly, the other computed lazily). Batch shape is the parameter shape
    # (empty for a plain number). Supports rsample; real-valued support.
    r"""
    Creates a LogitRelaxedBernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both), which is the logit of a RelaxedBernoulli
    distribution.

    Samples are logits of values in (0, 1). See [1] for more details.

    Args:
        temperature (Tensor): relaxation temperature
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random
    Variables (Maddison et al., 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al., 2017)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.real

    def __init__(
        self,
        temperature: Tensor,
        probs: Tensor | Number | None = None,
        logits: Tensor | Number | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Args: temperature (tensor, stored as is, not broadcast
        # with the parameters), and exactly one of probs or logits (number or
        # tensor). Raises ValueError if both or neither are given. Sets _param to the
        # supplied one and batch shape to empty for numbers, else its size.
        self.temperature = temperature
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, _Number)
            # pyrefly: ignore [read-only]
            (self.probs,) = broadcast_all(probs)
        else:
            if logits is None:
                raise AssertionError("logits is unexpectedly None")
            is_scalar = isinstance(logits, _Number)
            # pyrefly: ignore [read-only]
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new LogitRelaxedBernoulli (or _instance) sharing
        # temperature, with whichever of probs/logits are materialized expanded to
        # batch_shape; copies the validation flag.
        new = self._get_checked_instance(LogitRelaxedBernoulli, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(LogitRelaxedBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Creates a new tensor with the dtype/device of _param.
        return self._param.new(*args, **kwargs)

    @lazy_property
    def logits(self) -> Tensor:
        # [FUNCTION DOC] Lazy property (cached): logits derived from probs (binary conversion).
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        # [FUNCTION DOC] Lazy property (cached): probs derived from logits (binary conversion).
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self) -> torch.Size:
        # [FUNCTION DOC] Property: size of the stored parameter tensor.
        return self._param.size()

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Draws reparameterized samples of shape sample_shape +
        # batch: clamps probs and a uniform draw away from 0 and 1, computes the
        # sum of the logistic noise logit and the probs logit, divided by temperature.
        # Consumes RNG state.
        shape = self._extended_shape(sample_shape)
        probs = clamp_probs(self.probs.expand(shape))
        uniforms = clamp_probs(
            torch.rand(shape, dtype=probs.dtype, device=probs.device)
        )
        return (
            uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()
        ) / self.temperature

    def log_prob(self, value):
        # [FUNCTION DOC] Returns the log density of logit-space value: log
        # temperature plus the difference between logits and temperature times
        # value, minus twice the softplus of that difference. Validates value when
        # enabled (may raise).
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)
        return self.temperature.log() + diff - 2 * diff.exp().log1p()


class RelaxedBernoulli(TransformedDistribution):
    # [CLASS DOC] Relaxed Bernoulli (Concrete) distribution on (0, 1): a
    # LogitRelaxedBernoulli base passed through a sigmoid transform. Reparameterized.
    # temperature, probs and logits are read-only views of the base distribution.
    r"""
    Creates a RelaxedBernoulli distribution, parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`
    (but not both). This is a relaxed version of the `Bernoulli` distribution,
    so the values are in (0, 1), and has reparametrizable samples.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = RelaxedBernoulli(torch.tensor([2.2]),
        ...                      torch.tensor([0.1, 0.2, 0.3, 0.99]))
        >>> m.sample()
        tensor([ 0.2951,  0.3442,  0.8918,  0.9021])

    Args:
        temperature (Tensor): relaxation temperature
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
    """

    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    # pyrefly: ignore [bad-override]
    support = constraints.unit_interval
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: LogitRelaxedBernoulli

    def __init__(
        self,
        temperature: Tensor,
        probs: Tensor | Number | None = None,
        logits: Tensor | Number | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds the LogitRelaxedBernoulli base from temperature
        # and exactly one of probs/logits and initializes the parent with a sigmoid transform.
        # KNOWN ISSUE: validate_args is not forwarded to the base distribution.
        base_dist = LogitRelaxedBernoulli(temperature, probs, logits)
        super().__init__(base_dist, SigmoidTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns an expanded RelaxedBernoulli of batch_shape via the parent expand.
        new = self._get_checked_instance(RelaxedBernoulli, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def temperature(self) -> Tensor:
        # [FUNCTION DOC] Property: relaxation temperature of the base distribution.
        return self.base_dist.temperature

    @property
    def logits(self) -> Tensor:
        # [FUNCTION DOC] Property: logits of the base distribution.
        return self.base_dist.logits

    @property
    def probs(self) -> Tensor:
        # [FUNCTION DOC] Property: probs of the base distribution.
        return self.base_dist.probs
