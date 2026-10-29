# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines OneHotCategorical (categorical distribution whose samples are one-hot
# vectors, wrapping Categorical) and OneHotCategoricalStraightThrough (adds a
# straight-through gradient estimator for rsample). No import-time side effects.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
from torch.types import _size


__all__ = ["OneHotCategorical", "OneHotCategoricalStraightThrough"]


class OneHotCategorical(Distribution):
    # [CLASS DOC] Distribution over one-hot vectors whose length is the number
    # of categories. State: an internal Categorical that owns probs/logits and
    # defines batch shape (all but last parameter dim) and event shape (the
    # category count). Supports enumerate_support; no reparameterized sampling.
    r"""
    Creates a one-hot categorical distribution parameterized by :attr:`probs` or
    :attr:`logits`.

    Samples are one-hot coded vectors of size ``probs.size(-1)``.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.distributions.Categorical` for specifications of
    :attr:`probs` and :attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 0.,  0.,  0.,  1.])

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = constraints.one_hot
    has_enumerate_support = True

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Args: exactly one of probs or logits (tensor; last dim
        # indexes categories). The internal Categorical raises if both or neither
        # are given. Sets batch shape and event shape from the Categorical.
        self._categorical = Categorical(probs, logits)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new OneHotCategorical (or _instance) whose
        # internal Categorical is expanded to batch_shape; copies the validation flag.
        new = self._get_checked_instance(OneHotCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        new._categorical = self._categorical.expand(batch_shape)
        super(OneHotCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Delegates new-tensor creation to the internal Categorical.
        return self._categorical._new(*args, **kwargs)

    @property
    def _param(self) -> Tensor:
        # [FUNCTION DOC] HELPER: Property returning the parameter tensor stored by the internal Categorical.
        return self._categorical._param

    @property
    def probs(self) -> Tensor:
        # [FUNCTION DOC] Property: normalized probabilities of the internal Categorical.
        return self._categorical.probs

    @property
    def logits(self) -> Tensor:
        # [FUNCTION DOC] Property: normalized logits of the internal Categorical.
        return self._categorical.logits

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Property: the probabilities (expected one-hot vector).
        return self._categorical.probs

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Property: one-hot encoding of the most probable category
        # (first on ties), in the dtype of probs.
        probs = self._categorical.probs
        mode = probs.argmax(dim=-1)
        return torch.nn.functional.one_hot(mode, num_classes=probs.shape[-1]).to(probs)

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Property: per-category variance, p times (1 - p).
        return self._categorical.probs * (1 - self._categorical.probs)

    @property
    def param_shape(self) -> torch.Size:
        # [FUNCTION DOC] Property: parameter shape of the internal Categorical.
        return self._categorical.param_shape

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC] Samples category indices from the internal Categorical
        # and returns their one-hot encodings with probs dtype, shape
        # sample_shape + batch + (categories). Consumes RNG state.
        sample_shape = torch.Size(sample_shape)
        probs = self._categorical.probs
        num_events = self._categorical._num_events
        indices = self._categorical.sample(sample_shape)
        return torch.nn.functional.one_hot(indices, num_events).to(probs)

    def log_prob(self, value):
        # [FUNCTION DOC] Returns the log probability of one-hot vectors in value
        # by taking the index of the maximum along the last dim and scoring it with
        # the internal Categorical. Validates value when enabled (may raise).
        if self._validate_args:
            self._validate_sample(value)
        indices = value.max(-1)[1]
        return self._categorical.log_prob(indices)

    def entropy(self):
        # [FUNCTION DOC] Returns the entropy of the internal Categorical.
        return self._categorical.entropy()

    def enumerate_support(self, expand=True):
        # [FUNCTION DOC] Returns all one-hot vectors as an identity matrix reshaped
        # to (n, 1 per batch dim..., n), where n is the category count. If expand is
        # true, it is broadcast to (n,) + batch shape + (n,) as a view.
        n = self.event_shape[0]
        values = torch.eye(n, dtype=self._param.dtype, device=self._param.device)
        values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
        if expand:
            values = values.expand((n,) + self.batch_shape + (n,))
        return values


class OneHotCategoricalStraightThrough(OneHotCategorical):
    # [CLASS DOC] OneHotCategorical variant with rsample support using the
    # straight-through estimator: forward values are hard one-hot samples while
    # gradients flow to the probabilities.
    r"""
    Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al., 2013)
    """

    has_rsample = True

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Draws a hard one-hot sample and adds the probabilities
        # minus their detached copy, so the value is unchanged but gradients reach
        # probs. Consumes RNG state. Returns the same shape as sample.
        samples = self.sample(sample_shape)
        probs = self._categorical.probs  # cached via @lazy_property
        return samples + (probs - probs.detach())
