# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the ContinuousBernoulli distribution on [0, 1] (an exponential family
# member) parameterized by probs or logits, with closed-form mean, variance,
# cdf, icdf, entropy, and a Taylor approximation near probs of one half where
# the closed forms are numerically unstable. No import-time side effects.
# Exports: ContinuousBernoulli.
import math

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.types import _Number, _size, Number


__all__ = ["ContinuousBernoulli"]


class ContinuousBernoulli(ExponentialFamily):
    # [CLASS DOC]
    # Continuous Bernoulli distribution on the unit interval, with density
    # proportional to p^x (1-p)^(1-x). State: probs (clamped away from 0 and 1)
    # or logits (one is lazily derived from the other), the private _param
    # (whichever was given), and _lims, the interval of probs around one half
    # treated as numerically unstable. Supports reparameterized sampling.
    r"""
    Creates a continuous Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    The distribution is supported in [0, 1] and parameterized by 'probs' (in
    (0,1)) or 'logits' (real-valued). Note that, unlike the Bernoulli, 'probs'
    does not correspond to a probability and 'logits' does not correspond to
    log-odds, but the same names are used due to the similarity with the
    Bernoulli. See [1] for more details.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = ContinuousBernoulli(torch.tensor([0.3]))
        >>> m.sample()
        tensor([ 0.2538])

    Args:
        probs (Number, Tensor): (0,1) valued parameters
        logits (Number, Tensor): real valued parameters whose sigmoid matches 'probs'

    [1] The continuous Bernoulli: fixing a pervasive error in variational
    autoencoders, Loaiza-Ganem G and Cunningham JP, NeurIPS 2019.
    https://arxiv.org/abs/1907.06845
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.unit_interval
    _mean_carrier_measure = 0
    has_rsample = True

    def __init__(
        self,
        probs: Tensor | Number | None = None,
        logits: Tensor | Number | None = None,
        lims: tuple[float, float] = (0.499, 0.501),
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the distribution from exactly one of probs or logits (numbers or
        # tensors, broadcast); batch shape is empty for scalars, else the parameter
        # shape. Probs are clamped away from 0 and 1 after optional validation.
        # lims is the unstable probs interval. Raises ValueError if both or neither
        # parameter is given, or if probs has invalid values.
        # KNOWN ISSUE: probs are validated whenever validate_args is not None, so
        # an explicit validate_args of False still validates probs.
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, _Number)
            # pyrefly: ignore [read-only]
            (self.probs,) = broadcast_all(probs)
            # validate 'probs' here if necessary as it is later clamped for numerical stability
            # close to 0 and 1, later on; otherwise the clamped 'probs' would always pass
            if validate_args is not None:
                if not self.arg_constraints["probs"].check(self.probs).all():
                    raise ValueError("The parameter probs has invalid values")
            # pyrefly: ignore [read-only]
            self.probs = clamp_probs(self.probs)
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
        self._lims = lims
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new instance with probs and/or logits expanded to batch_shape (whichever are already materialized), same lims and validation setting; does not mutate self.
        new = self._get_checked_instance(ContinuousBernoulli, _instance)
        new._lims = self._lims
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(ContinuousBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Creates a new tensor of the same type and device as the parameter, forwarding all arguments.
        return self._param.new(*args, **kwargs)

    def _outside_unstable_region(self):
        # [FUNCTION DOC] HELPER: Boolean tensor, true where probs is at or below the lower lim or above the upper lim (closed forms are safe there).
        return torch.max(
            torch.le(self.probs, self._lims[0]), torch.gt(self.probs, self._lims[1])
        )

    def _cut_probs(self):
        # [FUNCTION DOC] HELPER: Returns probs with entries inside the unstable region replaced by the lower lim, so closed-form expressions avoid division by zero; callers then override those entries with Taylor values.
        return torch.where(
            self._outside_unstable_region(),
            self.probs,
            self._lims[0] * torch.ones_like(self.probs),
        )

    def _cont_bern_log_norm(self):
        # [FUNCTION DOC] HELPER: Log of the normalizing constant as a function of probs: closed form outside the unstable region, a Taylor expansion around one half inside it. Returns a tensor of batch shape.
        """computes the log normalizing constant as a function of the 'probs' parameter"""
        cut_probs = self._cut_probs()
        cut_probs_below_half = torch.where(
            torch.le(cut_probs, 0.5), cut_probs, torch.zeros_like(cut_probs)
        )
        cut_probs_above_half = torch.where(
            torch.ge(cut_probs, 0.5), cut_probs, torch.ones_like(cut_probs)
        )
        log_norm = torch.log(
            torch.abs(torch.log1p(-cut_probs) - torch.log(cut_probs))
        ) - torch.where(
            torch.le(cut_probs, 0.5),
            torch.log1p(-2.0 * cut_probs_below_half),
            torch.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = torch.pow(self.probs - 0.5, 2)
        taylor = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        return torch.where(self._outside_unstable_region(), log_norm, taylor)

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Mean of the distribution: closed form in probs outside the unstable region, Taylor approximation near one half.
        cut_probs = self._cut_probs()
        mus = cut_probs / (2.0 * cut_probs - 1.0) + 1.0 / (
            torch.log1p(-cut_probs) - torch.log(cut_probs)
        )
        x = self.probs - 0.5
        taylor = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * torch.pow(x, 2)) * x
        return torch.where(self._outside_unstable_region(), mus, taylor)

    @property
    def stddev(self) -> Tensor:
        # [FUNCTION DOC] Standard deviation, the square root of the variance.
        return torch.sqrt(self.variance)

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Variance: closed form in probs outside the unstable region, Taylor approximation near one half (tends to one twelfth at probs one half).
        cut_probs = self._cut_probs()
        vars = cut_probs * (cut_probs - 1.0) / torch.pow(
            1.0 - 2.0 * cut_probs, 2
        ) + 1.0 / torch.pow(torch.log1p(-cut_probs) - torch.log(cut_probs), 2)
        x = torch.pow(self.probs - 0.5, 2)
        taylor = 1.0 / 12.0 - (1.0 / 15.0 - 128.0 / 945.0 * x) * x
        return torch.where(self._outside_unstable_region(), vars, taylor)

    @lazy_property
    def logits(self) -> Tensor:
        # [FUNCTION DOC] Lazily computed logits derived from probs via the binary log-odds transform; cached on first access.
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        # [FUNCTION DOC] Lazily computed probs derived from logits via the sigmoid and clamped away from 0 and 1; cached on first access.
        return clamp_probs(logits_to_probs(self.logits, is_binary=True))

    @property
    def param_shape(self) -> torch.Size:
        # [FUNCTION DOC] Shape of the stored parameter tensor (probs or logits).
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC] Draws samples of shape sample_shape plus batch shape by inverse-cdf transform of uniform noise, under no-grad so no gradient flows. Consumes global random state.
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        with torch.no_grad():
            return self.icdf(u)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Reparameterized samples of shape sample_shape plus batch shape: uniform noise pushed through the differentiable inverse cdf. Consumes global random state.
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        return self.icdf(u)

    def log_prob(self, value):
        # [FUNCTION DOC] Log density at value: the negative binary cross entropy with logits plus the log normalizing constant. Validates value when argument validation is on (raises ValueError if outside support); broadcasts logits with value.
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return (
            -binary_cross_entropy_with_logits(logits, value, reduction="none")
            + self._cont_bern_log_norm()
        )

    def cdf(self, value):
        # [FUNCTION DOC] Cumulative distribution at value: closed form outside the unstable region, identity (the uniform cdf) inside it; zero at or below 0 and one at or above 1. Validates value when validation is on.
        if self._validate_args:
            self._validate_sample(value)
        cut_probs = self._cut_probs()
        cdfs = (
            torch.pow(cut_probs, value) * torch.pow(1.0 - cut_probs, 1.0 - value)
            + cut_probs
            - 1.0
        ) / (2.0 * cut_probs - 1.0)
        unbounded_cdfs = torch.where(self._outside_unstable_region(), cdfs, value)
        return torch.where(
            torch.le(value, 0.0),
            torch.zeros_like(value),
            torch.where(torch.ge(value, 1.0), torch.ones_like(value), unbounded_cdfs),
        )

    def icdf(self, value):
        # [FUNCTION DOC] Inverse cdf (quantile) of value in [0, 1]: closed form outside the unstable region, identity inside it. No input validation.
        cut_probs = self._cut_probs()
        return torch.where(
            self._outside_unstable_region(),
            (
                torch.log1p(-cut_probs + value * (2.0 * cut_probs - 1.0))
                - torch.log1p(-cut_probs)
            )
            / (torch.log(cut_probs) - torch.log1p(-cut_probs)),
            value,
        )

    def entropy(self):
        # [FUNCTION DOC] Differential entropy computed from the mean, log probabilities and log normalizing constant. Returns a tensor of batch shape.
        log_probs0 = torch.log1p(-self.probs)
        log_probs1 = torch.log(self.probs)
        return (
            self.mean * (log_probs0 - log_probs1)
            - self._cont_bern_log_norm()
            - log_probs0
        )

    @property
    def _natural_params(self) -> tuple[Tensor]:
        # [FUNCTION DOC] HELPER: Natural parameters of the exponential family, a one-element tuple holding logits.
        return (self.logits,)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        # [FUNCTION DOC]
        # HELPER: Log normalizer as a function of the natural parameter x: closed form
        # using the exponential minus one outside the unstable region, Taylor series inside.
        # KNOWN ISSUE: the unstable region is derived by shifting the probs-based lims
        # by one half and applying them to the logit-scale x, which does not match the
        # probs-space region used elsewhere (only coincides near x = 0).
        """computes the log normalizing constant as a function of the natural parameter"""
        out_unst_reg = torch.max(
            torch.le(x, self._lims[0] - 0.5), torch.gt(x, self._lims[1] - 0.5)
        )
        cut_nat_params = torch.where(
            out_unst_reg, x, (self._lims[0] - 0.5) * torch.ones_like(x)
        )
        log_norm = torch.log(
            torch.abs(torch.special.expm1(cut_nat_params))
        ) - torch.log(torch.abs(cut_nat_params))
        taylor = 0.5 * x + torch.pow(x, 2) / 24.0 - torch.pow(x, 4) / 2880.0
        return torch.where(out_unst_reg, log_norm, taylor)
