# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Multinomial distribution over category count vectors for a fixed
# number of trials, implemented on top of Categorical (sampling, parameters) and
# Binomial (entropy). Exports Multinomial. No import-time side effects.

import torch
from torch import inf, Tensor
from torch.distributions import Categorical, constraints
from torch.distributions.binomial import Binomial
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["Multinomial"]


class Multinomial(Distribution):
    # [CLASS DOC] Distribution over non-negative integer count vectors summing to
    # total_count (a single Python int), with category probabilities given by
    # probs or logits on the last dimension. State: total_count, an internal
    # Categorical (owns probs/logits) and an internal Binomial (used for entropy).
    # Batch shape is the Categorical batch shape; event shape is the number of
    # categories. Sampling requires one shared total_count. Not reparameterized.
    r"""
    Creates a Multinomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). The innermost dimension of
    :attr:`probs` indexes over categories. All other dimensions index over batches.

    Note that :attr:`total_count` need not be specified if only :meth:`log_prob` is
    called (see example below)

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    -   :meth:`sample` requires a single shared `total_count` for all
        parameters and samples.
    -   :meth:`log_prob` allows different `total_count` for each parameter and
        sample.

    Example::

        >>> # xdoctest: +SKIP("FIXME: found invalid values")
        >>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
        >>> x = m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 21.,  24.,  30.,  25.])

        >>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)
        tensor([-4.1338])

    Args:
        total_count (int): number of trials
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    total_count: int

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Property: normalized probabilities times total_count.
        return self.probs * self.total_count

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Property: per-category variance, total_count times p times (1 - p).
        return self.total_count * self.probs * (1 - self.probs)

    def __init__(
        self,
        total_count: int = 1,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Args: total_count (must be a Python int, default 1), and
        # exactly one of probs or logits (tensor, last dim indexes categories).
        # Raises NotImplementedError if total_count is not an int; the Categorical
        # constructor raises if both or neither of probs/logits are given. Builds the
        # internal Categorical and Binomial and sets batch and event shapes.
        if not isinstance(total_count, int):
            raise NotImplementedError("inhomogeneous total_count is not supported")
        self.total_count = total_count
        self._categorical = Categorical(probs=probs, logits=logits)
        self._binomial = Binomial(total_count=total_count, probs=self.probs)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new Multinomial (or _instance) with the internal
        # Categorical expanded to batch_shape and total_count copied.
        # KNOWN ISSUE: the internal Binomial is not copied or expanded, so entropy
        # on the expanded instance fails because that attribute is missing.
        new = self._get_checked_instance(Multinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count
        new._categorical = self._categorical.expand(batch_shape)
        super(Multinomial, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Delegates creation of a new tensor of matching
        # dtype/device to the internal Categorical.
        return self._categorical._new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    # pyrefly: ignore [bad-override]
    def support(self):
        # [FUNCTION DOC] Property: the multinomial constraint for this total_count
        # (discrete, event dimension one).
        return constraints.multinomial(self.total_count)

    @property
    def logits(self) -> Tensor:
        # [FUNCTION DOC] Property: normalized logits from the internal Categorical.
        return self._categorical.logits

    @property
    def probs(self) -> Tensor:
        # [FUNCTION DOC] Property: normalized probabilities from the internal Categorical.
        return self._categorical.probs

    @property
    def param_shape(self) -> torch.Size:
        # [FUNCTION DOC] Property: shape of the parameter tensor of the internal Categorical.
        return self._categorical.param_shape

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC] Draws total_count independent categorical samples per
        # batch element, moves the trial axis to the end, and tallies them with a
        # scatter-add into a zero count tensor of shape sample_shape + batch +
        # (categories). Returns counts cast to the probs dtype. Consumes RNG state.
        sample_shape = torch.Size(sample_shape)
        samples = self._categorical.sample(
            torch.Size((self.total_count,)) + sample_shape
        )
        # samples.shape is (total_count, sample_shape, batch_shape), need to change it to
        # (sample_shape, batch_shape, total_count)
        shifted_idx = list(range(samples.dim()))
        shifted_idx.append(shifted_idx.pop(0))
        samples = samples.permute(*shifted_idx)
        counts = samples.new(self._extended_shape(sample_shape)).zero_()
        counts.scatter_add_(-1, samples, torch.ones_like(samples))
        return counts.type_as(self.probs)

    def entropy(self):
        # [FUNCTION DOC] Returns the entropy: total_count times the categorical
        # entropy minus log factorial of total_count, plus the expected log
        # factorial of each category count under its binomial marginal (summed over
        # support values 1..total_count and over categories). Result has batch shape.
        # KNOWN ISSUE: total_count is turned into a CPU tensor, which can mismatch
        # the device of the parameters.
        n = torch.tensor(self.total_count)

        cat_entropy = self._categorical.entropy()
        term1 = n * cat_entropy - torch.lgamma(n + 1)

        support = self._binomial.enumerate_support(expand=False)[1:]
        binomial_probs = torch.exp(self._binomial.log_prob(support))
        weights = torch.lgamma(support + 1)
        term2 = (binomial_probs * weights).sum([0, -1])

        return term1 + term2

    def log_prob(self, value):
        # [FUNCTION DOC] Returns the log probability mass of count vectors in
        # value: log factorial of the total, minus the sum of log factorials of
        # counts, plus the sum of count times logit. Logits and value are broadcast;
        # entries with zero count and negative-infinite logit contribute zero (the
        # logits are cloned first, so parameters are not mutated). The total may
        # differ per sample. Validates value when enabled (may raise).
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        logits = logits.clone(memory_format=torch.contiguous_format)
        log_factorial_n = torch.lgamma(value.sum(-1) + 1)
        log_factorial_xs = torch.lgamma(value + 1).sum(-1)
        logits[(value == 0) & (logits == -inf)] = 0
        log_powers = (logits * value).sum(-1)
        return log_factorial_n - log_factorial_xs + log_powers
