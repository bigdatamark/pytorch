# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Gumbel-softmax (Concrete) family: ExpRelaxedCategorical (log of a
# relaxed one-hot point on the simplex) and RelaxedOneHotCategorical (exponential
# transform of it, values on the simplex, reparameterized). No import-time side effects.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform
from torch.distributions.utils import broadcast_all, clamp_probs
from torch.types import _size


__all__ = ["ExpRelaxedCategorical", "RelaxedOneHotCategorical"]


class ExpRelaxedCategorical(Distribution):
    # [CLASS DOC] Distribution over log-coordinates of relaxed one-hot vectors.
    # State: temperature tensor and an internal Categorical (owns probs/logits;
    # defines batch shape and event shape = category count). Reparameterized via
    # Gumbel noise. Declared support is the whole real vector space although the
    # true support is a submanifold.
    r"""
    Creates a ExpRelaxedCategorical parameterized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits` (but not both).
    Returns the log of a point in the simplex. Based on the interface to
    :class:`OneHotCategorical`.

    Implementation based on [1].

    See also: :func:`torch.distributions.OneHotCategorical`

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al., 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al., 2017)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = (
        constraints.real_vector
    )  # The true support is actually a submanifold of this.
    has_rsample = True

    def __init__(
        self,
        temperature: Tensor,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Args: temperature (tensor), and exactly one of probs or
        # logits (last dim indexes categories); the internal Categorical raises
        # if both or neither are given. Sets batch and event shapes from it.
        self._categorical = Categorical(probs, logits)
        self.temperature = temperature
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new ExpRelaxedCategorical (or _instance) sharing
        # temperature, with the internal Categorical expanded to batch_shape.
        new = self._get_checked_instance(ExpRelaxedCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        new._categorical = self._categorical.expand(batch_shape)
        super(ExpRelaxedCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Delegates new-tensor creation to the internal Categorical.
        return self._categorical._new(*args, **kwargs)

    @property
    def param_shape(self) -> torch.Size:
        # [FUNCTION DOC] Property: parameter shape of the internal Categorical.
        return self._categorical.param_shape

    @property
    def logits(self) -> Tensor:
        # [FUNCTION DOC] Property: normalized logits of the internal Categorical.
        return self._categorical.logits

    @property
    def probs(self) -> Tensor:
        # [FUNCTION DOC] Property: normalized probs of the internal Categorical.
        return self._categorical.probs

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Draws reparameterized samples of shape sample_shape +
        # batch + (categories): adds Gumbel noise (from clamped uniforms) to the
        # logits, divides by temperature, and subtracts the log-sum-exp so that
        # exponentiated values sum to one. Consumes RNG state.
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(
            torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        )
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels) / self.temperature
        return scores - scores.logsumexp(dim=-1, keepdim=True)

    def log_prob(self, value):
        # [FUNCTION DOC] Returns the log density of log-simplex points in value:
        # log-gamma of the category count plus (count - 1) times log temperature,
        # plus the sum over categories of normalized scores (logits minus
        # temperature times value, log-softmax over the last dim). Validates value
        # when enabled (may raise). Result has the batch shape.
        K = self._categorical._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        log_scale = torch.full_like(
            self.temperature, float(K)
        ).lgamma() - self.temperature.log().mul(-(K - 1))
        score = logits - value.mul(self.temperature)
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        return score + log_scale


class RelaxedOneHotCategorical(TransformedDistribution):
    # [CLASS DOC] Relaxed one-hot categorical (Gumbel-softmax / Concrete) on the
    # simplex: an ExpRelaxedCategorical base passed through an exponential
    # transform. Reparameterized. temperature, probs, logits are views of the base.
    r"""
    Creates a RelaxedOneHotCategorical distribution parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`.
    This is a relaxed version of the :class:`OneHotCategorical` distribution, so
    its samples are on simplex, and are reparametrizable.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
        ...                              torch.tensor([0.1, 0.2, 0.3, 0.4]))
        >>> m.sample()
        tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event
    """

    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    # pyrefly: ignore [bad-override]
    support = constraints.simplex
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: ExpRelaxedCategorical

    def __init__(
        self,
        temperature: Tensor,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds the ExpRelaxedCategorical base from temperature
        # and exactly one of probs/logits, and initializes the parent with an exponential transform.
        base_dist = ExpRelaxedCategorical(
            temperature, probs, logits, validate_args=validate_args
        )
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns an expanded RelaxedOneHotCategorical of batch_shape via the parent expand.
        new = self._get_checked_instance(RelaxedOneHotCategorical, _instance)
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
