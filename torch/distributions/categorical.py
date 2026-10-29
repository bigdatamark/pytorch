# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Categorical distribution over integer classes, parameterized by
# probs or logits along the last dimension. Exports Categorical via __all__.
# No import-time side effects.

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits


__all__ = ["Categorical"]


class Categorical(Distribution):
    # [CLASS DOC]
    # Categorical distribution over the integers zero to K minus one, where K is
    # the size of the last dimension of the parameter. Stores normalized probs
    # or normalized logits (the other is lazily computed and cached), _param
    # (the supplied one) and _num_events (K). Batch shape is all leading
    # dimensions. Discrete, enumerable, no rsample; mean and variance are nan.
    r"""
    Creates a categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    .. note::
        It is equivalent to the distribution that :func:`torch.multinomial`
        samples from.

    Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.multinomial`

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor(3)

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    has_enumerate_support = True

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Normalizes the supplied parameter along the last dimension (probs are
        # divided by their sum, logits have their log-sum-exp subtracted) and
        # derives the event count and batch shape from it. Raises ValueError if
        # both or neither of probs and logits are given, or if the supplied
        # tensor is zero-dimensional.
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            # pyrefly: ignore [read-only]
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits is None:
                raise AssertionError("logits is unexpectedly None")
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            # pyrefly: ignore [read-only]
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = (
            self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        )
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC]
        # Returns a new Categorical whose materialized probs/logits are expanded
        # to batch_shape plus the event dimension; reuses _instance if given.
        # Copies the event count and validate_args setting. Does not mutate self.
        new = self._get_checked_instance(Categorical, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        new._num_events = self._num_events
        super(Categorical, new).__init__(batch_shape, validate_args=False)
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
        # Dependent property: integer interval from zero to the event count
        # minus one, inclusive.
        return constraints.integer_interval(0, self._num_events - 1)

    @lazy_property
    def logits(self) -> Tensor:
        # [FUNCTION DOC]
        # Lazy cached property: converts probs to (normalized) logits when logits
        # were not supplied at construction.
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self) -> Tensor:
        # [FUNCTION DOC]
        # Lazy cached property: converts logits to probabilities when probs were
        # not supplied at construction.
        return logits_to_probs(self.logits)

    @property
    def param_shape(self) -> torch.Size:
        # [FUNCTION DOC]
        # Property: shape of the stored parameter tensor (batch shape plus the
        # event dimension).
        return self._param.size()

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: a tensor of nan with the extended shape (mean is undefined
        # for categorical values), in probs dtype and device.
        return torch.full(
            self._extended_shape(),
            nan,
            dtype=self.probs.dtype,
            device=self.probs.device,
        )

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: index of the largest probability along the last dimension.
        return self.probs.argmax(dim=-1)

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: a tensor of nan with the extended shape (variance is
        # undefined for categorical values), in probs dtype and device.
        return torch.full(
            self._extended_shape(),
            nan,
            dtype=self.probs.dtype,
            device=self.probs.device,
        )

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC]
        # Draws class indices of shape sample_shape plus batch shape: flattens
        # probs to two dimensions, samples with replacement via multinomial,
        # transposes and reshapes. Converts sample_shape to a Size if needed.
        # Consumes global RNG state.
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))

    def log_prob(self, value):
        # [FUNCTION DOC]
        # Returns the log probability of the integer classes in value: casts to
        # long, broadcasts against logits, and gathers the matching entry along
        # the last dimension. Validates value when validation is enabled.
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        # [FUNCTION DOC]
        # Returns the entropy per batch element: negative sum of probs times
        # logits over the last dimension, with logits clamped to the dtype minimum
        # so that zero-probability classes contribute zero rather than nan.
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    def enumerate_support(self, expand=True):
        # [FUNCTION DOC]
        # Returns the long integers zero through K minus one on the parameter
        # device, with a leading support dimension and singleton batch dims;
        # expanded to the full batch shape when expand is true.
        num_events = self._num_events
        values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values
