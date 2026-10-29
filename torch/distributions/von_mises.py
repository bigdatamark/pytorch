# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the circular VonMises distribution plus helpers: polynomial
# approximations of the log modified Bessel functions of order 0 and 1 (module
# coefficient tables) and a rejection sampler (Best and Fisher). Exports
# VonMises. No import-time side effects besides decorating the sampler for
# scripting when tracing.
import math

import torch
import torch.jit
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property


__all__ = ["VonMises"]


def _eval_poly(y, coef):
    # [FUNCTION DOC] HELPER: Evaluates a polynomial with coefficients in
    # ascending order at y using Horner's scheme. Copies coef first, so the
    # input list is not mutated. Returns a value shaped like y.
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


_I0_COEF_SMALL = [
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.360768e-1,
    0.45813e-2,
]
_I0_COEF_LARGE = [
    0.39894228,
    0.1328592e-1,
    0.225319e-2,
    -0.157565e-2,
    0.916281e-2,
    -0.2057706e-1,
    0.2635537e-1,
    -0.1647633e-1,
    0.392377e-2,
]
_I1_COEF_SMALL = [
    0.5,
    0.87890594,
    0.51498869,
    0.15084934,
    0.2658733e-1,
    0.301532e-2,
    0.32411e-3,
]
_I1_COEF_LARGE = [
    0.39894228,
    -0.3988024e-1,
    -0.362018e-2,
    0.163801e-2,
    -0.1031555e-1,
    0.2282967e-1,
    -0.2895312e-1,
    0.1787654e-1,
    -0.420059e-2,
]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0):
    # [FUNCTION DOC] HELPER: Approximates the log of the modified Bessel
    # function of the first kind, order 0 or 1, for positive x. Uses a
    # polynomial in (x/3.75) squared below 3.75 and an asymptotic form above,
    # selected elementwise. Raises AssertionError for other orders.
    """
    Returns ``log(I_order(x))`` for ``x > 0``,
    where `order` is either 0 or 1.
    """
    if order != 0 and order != 1:
        raise AssertionError(f"order must be 0 or 1, got {order}")

    # compute small solution
    y = x / 3.75
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    result = torch.where(x < 3.75, small, large)
    return result


@torch.jit.script_if_tracing
def _rejection_sample(loc, concentration, proposal_r, x):
    # [FUNCTION DOC] HELPER: Best-Fisher rejection sampler. Repeatedly draws
    # three uniform values per element (consuming RNG state), accepts those
    # meeting the acceptance tests, and writes accepted angles into x until all
    # elements are done. Returns the angles shifted by loc and wrapped to the
    # range from minus pi to pi. May loop for a long time for tiny
    # concentrations in low precision.
    done = torch.zeros(x.shape, dtype=torch.bool, device=loc.device)
    # pyrefly: ignore [bad-assignment, missing-attribute]
    while not done.all():
        u = torch.rand((3,) + x.shape, dtype=loc.dtype, device=loc.device)
        u1, u2, u3 = u.unbind()
        z = torch.cos(math.pi * u1)
        f = (1 + proposal_r * z) / (proposal_r + z)
        c = concentration * (proposal_r - f)
        accept = ((c * (2 - c) - u2) > 0) | ((c / u2).log() + 1 - c >= 0)
        if accept.any():
            # pyrefly: ignore [no-matching-overload]
            x = torch.where(accept, (u3 - 0.5).sign() * f.acos(), x)
            done = done | accept
    return (x + math.pi + loc) % (2 * math.pi) - math.pi


class VonMises(Distribution):
    # [CLASS DOC]
    # Circular von Mises distribution with real loc (angle) and positive
    # concentration, broadcast together; scalar events. No reparameterized
    # sampling. Samples via rejection sampling in double precision with cached
    # lazy double-precision parameters. Mean and mode equal loc; variance is the
    # circular variance.
    """
    A circular von Mises distribution.

    This implementation uses polar coordinates. The ``loc`` and ``value`` args
    can be any real number (to facilitate unconstrained optimization), but are
    interpreted as angles modulo 2 pi.

    Example::
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = VonMises(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # von Mises distributed with loc=1 and concentration=1
        tensor([1.9777])

    :param torch.Tensor loc: an angle in radians.
    :param torch.Tensor concentration: concentration parameter
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "concentration": constraints.positive}
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        loc: Tensor,
        concentration: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Broadcasts loc and concentration into tensors stored on the instance;
        # batch shape is the broadcast shape and event shape is empty.
        # Validation errors come from the base class.
        self.loc, self.concentration = broadcast_all(loc, concentration)
        batch_shape = self.loc.shape
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        # [FUNCTION DOC]
        # Log density: concentration times cosine of (value - loc), minus log
        # of two pi, minus the log modified Bessel function of order 0 at the
        # concentration. Optionally validates value.
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.concentration * torch.cos(value - self.loc)
        log_prob = (
            log_prob
            - math.log(2 * math.pi)
            - _log_modified_bessel_fn(self.concentration, order=0)
        )
        return log_prob

    @lazy_property
    def _loc(self) -> Tensor:
        # [FUNCTION DOC] HELPER: Cached double-precision copy of loc.
        return self.loc.to(torch.double)

    @lazy_property
    def _concentration(self) -> Tensor:
        # [FUNCTION DOC] HELPER: Cached double-precision copy of concentration.
        return self.concentration.to(torch.double)

    @lazy_property
    def _proposal_r(self) -> Tensor:
        # [FUNCTION DOC] HELPER: Cached proposal parameter of the Best-Fisher
        # sampler computed from the double-precision concentration; uses a
        # second order Taylor expansion where concentration is below 1e-5.
        kappa = self._concentration
        tau = 1 + (1 + 4 * kappa**2).sqrt()
        rho = (tau - (2 * tau).sqrt()) / (2 * kappa)
        _proposal_r = (1 + rho**2) / (2 * rho)
        # second order Taylor expansion around 0 for small kappa
        _proposal_r_taylor = 1 / kappa + kappa
        return torch.where(kappa < 1e-5, _proposal_r_taylor, _proposal_r)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC]
        # Draws samples (no gradients) of the extended shape using the rejection
        # sampler in double precision, then casts back to the dtype of loc.
        # Consumes RNG state. Results lie in the range from minus pi to pi.
        """
        The sampling algorithm for the von Mises distribution is based on the
        following paper: D.J. Best and N.I. Fisher, "Efficient simulation of the
        von Mises distribution." Applied Statistics (1979): 152-157.

        Sampling is always done in double precision internally to avoid a hang
        in _rejection_sample() for small values of the concentration, which
        starts to happen for single precision around 1e-4 (see issue #88443).
        """
        shape = self._extended_shape(sample_shape)
        x = torch.empty(shape, dtype=self._loc.dtype, device=self.loc.device)
        return _rejection_sample(
            self._loc, self._concentration, self._proposal_r, x
        ).to(self.loc.dtype)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC]
        # Tries the base class expand; if it raises NotImplementedError, builds
        # a new VonMises from expanded loc and concentration, preserving the
        # validate_args setting.
        # KNOWN ISSUE: _instance is ignored in both paths.
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get("_validate_args")
            loc = self.loc.expand(batch_shape)
            concentration = self.concentration.expand(batch_shape)
            return type(self)(loc, concentration, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC]
        # Circular mean, which is loc.
        """
        The provided mean is the circular one.
        """
        return self.loc

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC]
        # Mode, which is loc.
        return self.loc

    @lazy_property
    def variance(self) -> Tensor:  # type: ignore[override]
        # [FUNCTION DOC]
        # Cached circular variance: one minus the ratio of Bessel functions of
        # order 1 and order 0 at the concentration (via log differences).
        """
        The provided variance is the circular one.
        """
        return (
            1
            - (
                _log_modified_bessel_fn(self.concentration, order=1)
                - _log_modified_bessel_fn(self.concentration, order=0)
            ).exp()
        )
