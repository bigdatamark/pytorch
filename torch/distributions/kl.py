# mypy: allow-untyped-defs
# [MODULE DOC]
# Registry and dispatch for Kullback-Leibler divergence between distributions.
# Main contents: the register_kl decorator, the kl_divergence entry point that
# finds the most specific registered function for a pair of distribution types
# (with a memoization cache and an ambiguity warning), small numeric helpers, and
# many closed-form KL implementations registered for same-family pairs (for
# example normal-normal, or the generic exponential-family Bregman version) and
# for cross-family pairs (many of which return infinity). Exports register_kl and
# kl_divergence. Import-time side effects: populates the module-level registry by
# running every register_kl decorator. A helper to append the supported-pairs list
# to the kl_divergence docstring is defined here but not called in this module.
import math
import warnings
from collections.abc import Callable
from functools import total_ordering

import torch
from torch import inf, Tensor

from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .gamma import Gamma
from .geometric import Geometric
from .gumbel import Gumbel
from .half_normal import HalfNormal
from .independent import Independent
from .laplace import Laplace
from .lowrank_multivariate_normal import (
    _batch_lowrank_logdet,
    _batch_lowrank_mahalanobis,
    LowRankMultivariateNormal,
)
from .multivariate_normal import _batch_mahalanobis, MultivariateNormal
from .normal import Normal
from .one_hot_categorical import OneHotCategorical
from .pareto import Pareto
from .poisson import Poisson
from .transformed_distribution import TransformedDistribution
from .uniform import Uniform
from .utils import _sum_rightmost, euler_constant as _euler_gamma


_KL_REGISTRY: dict[
    tuple[type, type], Callable
] = {}  # Source of truth mapping a few general (type, type) pairs to functions.
_KL_MEMOIZE: dict[
    tuple[type, type], Callable
] = {}  # Memoized version mapping many specific (type, type) pairs to functions.

__all__ = ["register_kl", "kl_divergence"]


def register_kl(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    # [FUNCTION DOC] Decorator factory. Args: type_p and type_q, the distribution
    # classes of the first and second KL argument. Returns a decorator that stores
    # the decorated function in the registry under the pair (replacing any previous
    # entry), clears the lookup cache, and returns the function unchanged.
    # KNOWN ISSUE: the type checks are logically wrong. They test that the value is
    # not a class and is also a Distribution subclass, so a non-Distribution class
    # passes silently and a non-class value makes the subclass test itself raise
    # TypeError rather than the intended message.
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError(
            f"Expected type_p to be a Distribution subclass but got {type_p}"
        )
    if not isinstance(type_q, type) and issubclass(type_q, Distribution):
        raise TypeError(
            f"Expected type_q to be a Distribution subclass but got {type_q}"
        )

    def decorator(fun):
        # [FUNCTION DOC] HELPER: Registers fun for the enclosing type pair in the
        # registry, invalidates the memoized lookup cache, and returns fun.
        _KL_REGISTRY[type_p, type_q] = fun
        _KL_MEMOIZE.clear()  # reset since lookup order may have changed
        return fun

    return decorator


@total_ordering
class _Match:
    # [CLASS DOC] HELPER: Wrapper around a tuple of registered classes used to
    # order registry matches by specificity (subclass is smaller). Only equality
    # and less-or-equal are defined; the rest come from total ordering.
    __slots__ = ["types"]

    def __init__(self, *types):
        # [FUNCTION DOC] HELPER: Stores the tuple of classes being compared.
        self.types = types

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: True when both wrappers hold the same tuple of classes.
        return self.types == other.types

    def __le__(self, other):
        # [FUNCTION DOC] HELPER: Walks the two class tuples in order; returns false
        # if a class here is not a subclass of its counterpart, stops at the first
        # position where the classes differ, and returns true otherwise.
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


def _dispatch_kl(type_p, type_q):
    # [FUNCTION DOC] HELPER: Finds the registered KL function for the given pair
    # of classes. Collects all registry pairs the classes are subclasses of;
    # returns the not-implemented sentinel if none. Picks the most specific match
    # ordering by first then second class, and again by second then first class;
    # if the two choices are different functions it emits a RuntimeWarning asking
    # for an explicit registration. Returns the first-then-second choice.
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [
        (super_p, super_q)
        for super_p, super_q in _KL_REGISTRY
        if issubclass(type_p, super_p) and issubclass(type_q, super_q)
    ]
    if not matches:
        return NotImplemented
    # Check that the left- and right- lexicographic orders agree.
    # mypy isn't smart enough to know that _Match implements __lt__
    # see: https://github.com/python/typing/issues/760#issuecomment-710670503
    left_p, left_q = min(_Match(*m) for m in matches).types  # type: ignore[type-var]
    right_q, right_p = min(_Match(*reversed(m)) for m in matches).types  # type: ignore[type-var]
    left_fun = _KL_REGISTRY[left_p, left_q]
    right_fun = _KL_REGISTRY[right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn(
            f"Ambiguous kl_divergence({type_p.__name__}, {type_q.__name__}). "
            f"Please register_kl({left_p.__name__}, {right_q.__name__})",
            RuntimeWarning,
            stacklevel=2,
        )
    return left_fun


def _infinite_like(tensor):
    # [FUNCTION DOC] HELPER: Returns a tensor shaped like the input filled with positive infinity.
    """
    Helper function for obtaining infinite KL Divergence throughout
    """
    return torch.full_like(tensor, inf)


def _x_log_x(tensor):
    # [FUNCTION DOC] HELPER: Elementwise x times log x, defined as zero at x equal to zero.
    """
    Utility function for calculating x log x
    """
    return torch.special.xlogy(tensor, tensor)  # produces correct result for x=0


def _batch_trace_XXT(bmat):
    # [FUNCTION DOC] HELPER: For a batch of matrices, returns the trace of the
    # matrix times its transpose (sum of squares of all entries per matrix),
    # shaped like the leading batch dims.
    """
    Utility function for calculating the trace of XX^{T} with X having arbitrary trailing batch dimensions
    """
    n = bmat.size(-1)
    m = bmat.size(-2)
    flat_trace = bmat.reshape(-1, m * n).pow(2).sum(-1)
    return flat_trace.reshape(bmat.shape[:-2])


def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    # [FUNCTION DOC] Public entry point. Args: p and q, distributions. Looks up
    # the KL function for the pair of concrete types in the cache, falling back to
    # the registry dispatch (and caching the result, including the not-implemented
    # sentinel). Raises NotImplementedError naming both types when no function
    # applies; otherwise returns the tensor of divergences with the batch shape.
    # May emit a RuntimeWarning for ambiguous registrations.
    try:
        fun = _KL_MEMOIZE[type(p), type(q)]
    except KeyError:
        fun = _dispatch_kl(type(p), type(q))
        _KL_MEMOIZE[type(p), type(q)] = fun
    if fun is NotImplemented:
        raise NotImplementedError(
            f"No KL(p || q) is implemented for p type {p.__class__.__name__} and q type {q.__class__.__name__}"
        )
    return fun(p, q)


################################################################################
# KL Divergence Implementations
################################################################################

# Same distributions


@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(p, q):
    # [FUNCTION DOC] HELPER: KL between two Bernoulli distributions from softplus
    # of logits, split into success and failure terms; a term is infinite where q
    # has zero (or one) probability of that outcome and zero where p does.
    t1 = p.probs * (
        torch.nn.functional.softplus(-q.logits)
        - torch.nn.functional.softplus(-p.logits)
    )
    t1[q.probs == 0] = inf
    t1[p.probs == 0] = 0
    t2 = (1 - p.probs) * (
        torch.nn.functional.softplus(q.logits) - torch.nn.functional.softplus(p.logits)
    )
    t2[q.probs == 1] = inf
    t2[p.probs == 1] = 0
    return t1 + t2


@register_kl(Beta, Beta)
def _kl_beta_beta(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between two Beta distributions using
    # log-gamma and digamma of the concentrations and their sums.
    sum_params_p = p.concentration1 + p.concentration0
    sum_params_q = q.concentration1 + q.concentration0
    t1 = q.concentration1.lgamma() + q.concentration0.lgamma() + (sum_params_p).lgamma()
    t2 = p.concentration1.lgamma() + p.concentration0.lgamma() + (sum_params_q).lgamma()
    t3 = (p.concentration1 - q.concentration1) * torch.digamma(p.concentration1)
    t4 = (p.concentration0 - q.concentration0) * torch.digamma(p.concentration0)
    t5 = (sum_params_q - sum_params_p) * torch.digamma(sum_params_p)
    return t1 - t2 + t3 + t4 + t5


@register_kl(Binomial, Binomial)
def _kl_binomial_binomial(p, q):
    # [FUNCTION DOC] HELPER: KL between Binomials. Raises NotImplementedError if
    # any p total_count is below the q total_count (the message wording is
    # reversed from the check). Uses the closed form for equal counts and sets
    # entries where p has the larger count to infinity (in-place on a local tensor).
    # from https://math.stackexchange.com/questions/2214993/
    # kullback-leibler-divergence-for-binomial-distributions-p-and-q
    if (p.total_count < q.total_count).any():
        raise NotImplementedError(
            "KL between Binomials where q.total_count > p.total_count is not implemented"
        )
    kl = p.total_count * (
        p.probs * (p.logits - q.logits) + (-p.probs).log1p() - (-q.probs).log1p()
    )
    inf_idxs = p.total_count > q.total_count
    kl[inf_idxs] = _infinite_like(kl[inf_idxs])
    return kl


@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    # [FUNCTION DOC] HELPER: KL between Categoricals as the sum over categories of
    # p probability times the logit difference; terms are infinite where q has
    # zero probability and zero where p does (p taking precedence), then summed on the last dim.
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)


@register_kl(ContinuousBernoulli, ContinuousBernoulli)
def _kl_continuous_bernoulli_continuous_bernoulli(p, q):
    # [FUNCTION DOC] HELPER: KL between ContinuousBernoulli distributions from the
    # mean times logit difference plus the difference of log normalizers and log1p terms.
    t1 = p.mean * (p.logits - q.logits)
    t2 = p._cont_bern_log_norm() + torch.log1p(-p.probs)
    t3 = -q._cont_bern_log_norm() - torch.log1p(-q.probs)
    return t1 + t2 + t3


@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between Dirichlet distributions using
    # log-gamma of concentration sums and entries and digamma terms; reduces over the last dim.
    # From https://statproofbook.github.io/P/dir-kl.html
    sum_p_concentration = p.concentration.sum(-1)
    sum_q_concentration = q.concentration.sum(-1)
    t1 = sum_p_concentration.lgamma() - sum_q_concentration.lgamma()
    t2 = (p.concentration.lgamma() - q.concentration.lgamma()).sum(-1)
    t3 = p.concentration - q.concentration
    t4 = p.concentration.digamma() - sum_p_concentration.digamma().unsqueeze(-1)
    return t1 - t2 + (t3 * t4).sum(-1)


@register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p, q):
    # [FUNCTION DOC] HELPER: KL between Exponentials from the rate ratio: ratio minus one minus its log.
    rate_ratio = q.rate / p.rate
    t1 = -rate_ratio.log()
    return t1 + rate_ratio - 1


@register_kl(ExponentialFamily, ExponentialFamily)
def _kl_expfamily_expfamily(p, q):
    # [FUNCTION DOC] HELPER: Generic KL for two exponential-family distributions
    # of the same class, as a Bregman divergence of the log normalizer. Raises
    # NotImplementedError if the two types differ. Detaches p's natural
    # parameters, marks them as requiring gradients, and uses autograd (with graph
    # creation) to get the gradient of the log normalizer; then subtracts the
    # inner products with parameter differences, summed over event dims.
    if type(p) is not type(q):
        raise NotImplementedError(
            "The cross KL-divergence between different exponential families cannot \
                            be computed using Bregman divergences"
        )
    p_nparams = [np.detach().requires_grad_() for np in p._natural_params]
    q_nparams = q._natural_params
    lg_normal = p._log_normalizer(*p_nparams)
    gradients = torch.autograd.grad(lg_normal.sum(), p_nparams, create_graph=True)
    result = q._log_normalizer(*q_nparams) - lg_normal
    for pnp, qnp, g in zip(p_nparams, q_nparams, gradients):
        term = (qnp - pnp) * g
        result -= _sum_rightmost(term, len(q.event_shape))
    return result


@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between Gamma distributions (concentration/rate).
    t1 = q.concentration * (p.rate / q.rate).log()
    t2 = torch.lgamma(q.concentration) - torch.lgamma(p.concentration)
    t3 = (p.concentration - q.concentration) * torch.digamma(p.concentration)
    t4 = (q.rate - p.rate) * (p.concentration / p.rate)
    return t1 + t2 + t3 + t4


@register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between Gumbel distributions using
    # the Euler constant and a log-gamma term of the scale ratio.
    ct1 = p.scale / q.scale
    ct2 = q.loc / q.scale
    ct3 = p.loc / q.scale
    t1 = -ct1.log() - ct2 + ct3
    t2 = ct1 * _euler_gamma
    t3 = torch.exp(ct2 + (1 + ct1).lgamma() - ct3)
    return t1 + t2 + t3 - (1 + _euler_gamma)


@register_kl(Geometric, Geometric)
def _kl_geometric_geometric(p, q):
    # [FUNCTION DOC] HELPER: KL between Geometric distributions from the negative
    # entropy of p, the log of one minus q probs over p probs, and q logits.
    return -p.entropy() - torch.log1p(-q.probs) / p.probs - q.logits


@register_kl(HalfNormal, HalfNormal)
def _kl_halfnormal_halfnormal(p, q):
    # [FUNCTION DOC] HELPER: KL between HalfNormals, equal to the KL between their underlying Normal base distributions.
    return _kl_normal_normal(p.base_dist, q.base_dist)


@register_kl(Laplace, Laplace)
def _kl_laplace_laplace(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between Laplace distributions from the
    # scale ratio and the absolute location difference.
    # From http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf
    scale_ratio = p.scale / q.scale
    loc_abs_diff = (p.loc - q.loc).abs()
    t1 = -scale_ratio.log()
    t2 = loc_abs_diff / q.scale
    t3 = scale_ratio * torch.exp(-loc_abs_diff / p.scale)
    return t1 + t2 + t3 - 1


@register_kl(LowRankMultivariateNormal, LowRankMultivariateNormal)
def _kl_lowrankmultivariatenormal_lowrankmultivariatenormal(p, q):
    # [FUNCTION DOC] HELPER: KL between two low-rank multivariate normals. Raises
    # ValueError if event shapes differ. Combines the log determinant difference,
    # a trace term expanded with the Woodbury identity (four pieces built from
    # diagonals, factors and a triangular solve), and the Mahalanobis distance
    # between means, minus the dimension, all halved.
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two Low Rank Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    term1 = _batch_lowrank_logdet(
        q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q._capacitance_tril
    ) - _batch_lowrank_logdet(
        p._unbroadcasted_cov_factor, p._unbroadcasted_cov_diag, p._capacitance_tril
    )
    term3 = _batch_lowrank_mahalanobis(
        q._unbroadcasted_cov_factor,
        q._unbroadcasted_cov_diag,
        q.loc - p.loc,
        q._capacitance_tril,
    )
    # Expands term2 according to
    # inv(qcov) @ pcov = [inv(qD) - inv(qD) @ qW @ inv(qC) @ qW.T @ inv(qD)] @ (pW @ pW.T + pD)
    #                  = [inv(qD) - A.T @ A] @ (pD + pW @ pW.T)
    qWt_qDinv = q._unbroadcasted_cov_factor.mT / q._unbroadcasted_cov_diag.unsqueeze(-2)
    A = torch.linalg.solve_triangular(q._capacitance_tril, qWt_qDinv, upper=False)
    term21 = (p._unbroadcasted_cov_diag / q._unbroadcasted_cov_diag).sum(-1)
    term22 = _batch_trace_XXT(
        p._unbroadcasted_cov_factor * q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1)
    )
    term23 = _batch_trace_XXT(A * p._unbroadcasted_cov_diag.sqrt().unsqueeze(-2))
    term24 = _batch_trace_XXT(A.matmul(p._unbroadcasted_cov_factor))
    term2 = term21 + term22 - term23 - term24
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


@register_kl(MultivariateNormal, LowRankMultivariateNormal)
def _kl_multivariatenormal_lowrankmultivariatenormal(p, q):
    # [FUNCTION DOC] HELPER: KL from a full-covariance multivariate normal p to a
    # low-rank multivariate normal q. Raises ValueError if event shapes differ.
    # Uses the low-rank log determinant of q minus p's log determinant (from its
    # Cholesky diagonal), a Woodbury-expanded trace term, and the Mahalanobis
    # distance of the mean difference under q, minus the dimension, all halved.
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two (Low Rank) Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    term1 = _batch_lowrank_logdet(
        q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q._capacitance_tril
    ) - 2 * p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    term3 = _batch_lowrank_mahalanobis(
        q._unbroadcasted_cov_factor,
        q._unbroadcasted_cov_diag,
        q.loc - p.loc,
        q._capacitance_tril,
    )
    # Expands term2 according to
    # inv(qcov) @ pcov = [inv(qD) - inv(qD) @ qW @ inv(qC) @ qW.T @ inv(qD)] @ p_tril @ p_tril.T
    #                  = [inv(qD) - A.T @ A] @ p_tril @ p_tril.T
    qWt_qDinv = q._unbroadcasted_cov_factor.mT / q._unbroadcasted_cov_diag.unsqueeze(-2)
    A = torch.linalg.solve_triangular(q._capacitance_tril, qWt_qDinv, upper=False)
    term21 = _batch_trace_XXT(
        p._unbroadcasted_scale_tril * q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1)
    )
    term22 = _batch_trace_XXT(A.matmul(p._unbroadcasted_scale_tril))
    term2 = term21 - term22
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


@register_kl(LowRankMultivariateNormal, MultivariateNormal)
def _kl_lowrankmultivariatenormal_multivariatenormal(p, q):
    # [FUNCTION DOC] HELPER: KL from a low-rank multivariate normal p to a
    # full-covariance multivariate normal q. Raises ValueError if event shapes
    # differ. Broadcasts the batch shapes of q's Cholesky factor and p's factor,
    # solves triangular systems against p's factor and diagonal for the trace term,
    # and adds the log determinant difference and mean Mahalanobis distance.
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two (Low Rank) Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    term1 = 2 * q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(
        -1
    ) - _batch_lowrank_logdet(
        p._unbroadcasted_cov_factor, p._unbroadcasted_cov_diag, p._capacitance_tril
    )
    term3 = _batch_mahalanobis(q._unbroadcasted_scale_tril, (q.loc - p.loc))
    # Expands term2 according to
    # inv(qcov) @ pcov = inv(q_tril @ q_tril.T) @ (pW @ pW.T + pD)
    combined_batch_shape = torch._C._infer_size(
        q._unbroadcasted_scale_tril.shape[:-2], p._unbroadcasted_cov_factor.shape[:-2]
    )
    n = p.event_shape[0]
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_cov_factor = p._unbroadcasted_cov_factor.expand(
        combined_batch_shape + (n, p.cov_factor.size(-1))
    )
    p_cov_diag = torch.diag_embed(p._unbroadcasted_cov_diag.sqrt()).expand(
        combined_batch_shape + (n, n)
    )
    term21 = _batch_trace_XXT(
        torch.linalg.solve_triangular(q_scale_tril, p_cov_factor, upper=False)
    )
    term22 = _batch_trace_XXT(
        torch.linalg.solve_triangular(q_scale_tril, p_cov_diag, upper=False)
    )
    term2 = term21 + term22
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


@register_kl(MultivariateNormal, MultivariateNormal)
def _kl_multivariatenormal_multivariatenormal(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between multivariate normals from
    # Cholesky factors: log determinant ratio, trace term via a triangular solve of
    # p's factor against q's (batch shapes broadcast), and the Mahalanobis distance
    # of the mean difference. Raises ValueError if event shapes differ.
    # From https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    half_term1 = q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(
        -1
    ) - p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    combined_batch_shape = torch._C._infer_size(
        q._unbroadcasted_scale_tril.shape[:-2], p._unbroadcasted_scale_tril.shape[:-2]
    )
    n = p.event_shape[0]
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_scale_tril = p._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    term2 = _batch_trace_XXT(
        torch.linalg.solve_triangular(q_scale_tril, p_scale_tril, upper=False)
    )
    term3 = _batch_mahalanobis(q._unbroadcasted_scale_tril, (q.loc - p.loc))
    return half_term1 + 0.5 * (term2 + term3 - n)


@register_kl(Normal, Normal)
def _kl_normal_normal(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between Normals from the variance ratio and standardized mean difference.
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


@register_kl(OneHotCategorical, OneHotCategorical)
def _kl_onehotcategorical_onehotcategorical(p, q):
    # [FUNCTION DOC] HELPER: KL between OneHotCategoricals, computed on their internal Categoricals.
    return _kl_categorical_categorical(p._categorical, q._categorical)


@register_kl(Pareto, Pareto)
def _kl_pareto_pareto(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between Pareto distributions from the
    # scale ratio and alpha ratio; entries where p's lower bound is below q's are set to infinity.
    # From http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf
    scale_ratio = p.scale / q.scale
    alpha_ratio = q.alpha / p.alpha
    t1 = q.alpha * scale_ratio.log()
    t2 = -alpha_ratio.log()
    result = t1 + t2 + alpha_ratio - 1
    result[p.support.lower_bound < q.support.lower_bound] = inf
    return result


@register_kl(Poisson, Poisson)
def _kl_poisson_poisson(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between Poissons from the rates.
    return p.rate * (p.rate.log() - q.rate.log()) - (p.rate - q.rate)


@register_kl(TransformedDistribution, TransformedDistribution)
def _kl_transformed_transformed(p, q):
    # [FUNCTION DOC] HELPER: KL between transformed distributions sharing the same
    # transforms and event shape, equal to the KL of their base distributions.
    # Raises NotImplementedError (no message) if transforms or event shapes differ.
    if p.transforms != q.transforms:
        raise NotImplementedError
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    return kl_divergence(p.base_dist, q.base_dist)


@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p, q):
    # [FUNCTION DOC] HELPER: KL between Uniforms: log ratio of widths, infinite
    # where q's interval does not contain p's.
    result = ((q.high - q.low) / (p.high - p.low)).log()
    result[(q.low > p.low) | (q.high < p.high)] = inf
    return result


# Different distributions
@register_kl(Bernoulli, Poisson)
def _kl_bernoulli_poisson(p, q):
    # [FUNCTION DOC] HELPER: KL from a Bernoulli to a Poisson: negative entropy of p
    # minus the expected Poisson log mass under p (success prob times log rate minus rate).
    return -p.entropy() - (p.probs * q.rate.log() - q.rate)


@register_kl(Beta, ContinuousBernoulli)
def _kl_beta_continuous_bernoulli(p, q):
    # [FUNCTION DOC] HELPER: KL from a Beta to a ContinuousBernoulli: negative
    # entropy of p minus the expected log density of q using the Beta mean.
    return (
        -p.entropy()
        - p.mean * q.logits
        - torch.log1p(-q.probs)
        - q._cont_bern_log_norm()
    )


@register_kl(Beta, Pareto)
def _kl_beta_infinity(p, q):
    # [FUNCTION DOC] HELPER: KL from a Beta to a Pareto: infinity, shaped like the concentration.
    return _infinite_like(p.concentration1)


@register_kl(Beta, Exponential)
def _kl_beta_exponential(p, q):
    # [FUNCTION DOC] HELPER: KL from a Beta to an Exponential: negative entropy of p
    # minus log rate plus rate times the Beta mean.
    return (
        -p.entropy()
        - q.rate.log()
        + q.rate * (p.concentration1 / (p.concentration1 + p.concentration0))
    )


@register_kl(Beta, Gamma)
def _kl_beta_gamma(p, q):
    # [FUNCTION DOC] HELPER: KL from a Beta to a Gamma using the negative entropy
    # of p and the expected log density of q (digamma terms and the Beta mean).
    t1 = -p.entropy()
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (q.concentration - 1) * (
        p.concentration1.digamma() - (p.concentration1 + p.concentration0).digamma()
    )
    t4 = q.rate * p.concentration1 / (p.concentration1 + p.concentration0)
    return t1 + t2 - t3 + t4


# TODO: Add Beta-Laplace KL Divergence


@register_kl(Beta, Normal)
def _kl_beta_normal(p, q):
    # [FUNCTION DOC] HELPER: KL from a Beta to a Normal using the negative entropy of
    # p and the expected normal log density from the Beta mean and second moment.
    E_beta = p.concentration1 / (p.concentration1 + p.concentration0)
    var_normal = q.scale.pow(2)
    t1 = -p.entropy()
    t2 = 0.5 * (var_normal * 2 * math.pi).log()
    t3 = (
        E_beta * (1 - E_beta) / (p.concentration1 + p.concentration0 + 1)
        + E_beta.pow(2)
    ) * 0.5
    t4 = q.loc * E_beta
    t5 = q.loc.pow(2) * 0.5
    return t1 + t2 + (t3 - t4 + t5) / var_normal


@register_kl(Beta, Uniform)
def _kl_beta_uniform(p, q):
    # [FUNCTION DOC] HELPER: KL from a Beta to a Uniform: negative entropy of p plus
    # log width of q; infinite where q's interval does not cover the unit interval.
    result = -p.entropy() + (q.high - q.low).log()
    result[(q.low > p.support.lower_bound) | (q.high < p.support.upper_bound)] = inf
    return result


# Note that the KL between a ContinuousBernoulli and Beta has no closed form


@register_kl(ContinuousBernoulli, Pareto)
def _kl_continuous_bernoulli_infinity(p, q):
    # [FUNCTION DOC] HELPER: KL from a ContinuousBernoulli to a Pareto: infinity, shaped like probs.
    return _infinite_like(p.probs)


@register_kl(ContinuousBernoulli, Exponential)
def _kl_continuous_bernoulli_exponential(p, q):
    # [FUNCTION DOC] HELPER: KL from a ContinuousBernoulli to an Exponential: negative
    # entropy of p minus log rate plus rate times the mean of p.
    return -p.entropy() - torch.log(q.rate) + q.rate * p.mean


# Note that the KL between a ContinuousBernoulli and Gamma has no closed form
# TODO: Add ContinuousBernoulli-Laplace KL Divergence


@register_kl(ContinuousBernoulli, Normal)
def _kl_continuous_bernoulli_normal(p, q):
    # [FUNCTION DOC] HELPER: KL from a ContinuousBernoulli to a Normal using the
    # negative entropy of p and the expected normal log density from p's mean and variance.
    t1 = -p.entropy()
    t2 = 0.5 * (math.log(2.0 * math.pi) + torch.square(q.loc / q.scale)) + torch.log(
        q.scale
    )
    t3 = (p.variance + torch.square(p.mean) - 2.0 * q.loc * p.mean) / (
        2.0 * torch.square(q.scale)
    )
    return t1 + t2 + t3


@register_kl(ContinuousBernoulli, Uniform)
def _kl_continuous_bernoulli_uniform(p, q):
    # [FUNCTION DOC] HELPER: KL from a ContinuousBernoulli to a Uniform: negative
    # entropy of p plus log width of q, with infinity selected by a support test.
    # KNOWN ISSUE: the support test is inverted/off; it returns infinity when q.low
    # is at least p's lower bound or q.high is at most p's upper bound, which flags
    # the valid case of q covering the unit interval (the Beta-Uniform version uses
    # strict opposite comparisons).
    result = -p.entropy() + (q.high - q.low).log()
    return torch.where(
        torch.max(
            torch.ge(q.low, p.support.lower_bound),
            torch.le(q.high, p.support.upper_bound),
        ),
        torch.ones_like(result) * inf,
        result,
    )


@register_kl(Exponential, Beta)
@register_kl(Exponential, ContinuousBernoulli)
@register_kl(Exponential, Pareto)
@register_kl(Exponential, Uniform)
def _kl_exponential_infinity(p, q):
    # [FUNCTION DOC] HELPER: KL from an Exponential to Beta, ContinuousBernoulli, Pareto or Uniform: infinity, shaped like rate.
    return _infinite_like(p.rate)


@register_kl(Exponential, Gamma)
def _kl_exponential_gamma(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from an Exponential to a Gamma using the rate ratio, log-gamma and the Euler constant.
    ratio = q.rate / p.rate
    t1 = -q.concentration * torch.log(ratio)
    return (
        t1
        + ratio
        + q.concentration.lgamma()
        + q.concentration * _euler_gamma
        - (1 + _euler_gamma)
    )


@register_kl(Exponential, Gumbel)
def _kl_exponential_gumbel(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from an Exponential to a Gumbel using
    # the product of p's rate and q's scale and the ratio of q's location to scale.
    scale_rate_prod = p.rate * q.scale
    loc_scale_ratio = q.loc / q.scale
    t1 = scale_rate_prod.log() - 1
    t2 = torch.exp(loc_scale_ratio) * scale_rate_prod / (scale_rate_prod + 1)
    t3 = scale_rate_prod.reciprocal()
    return t1 - loc_scale_ratio + t2 + t3


# TODO: Add Exponential-Laplace KL Divergence


@register_kl(Exponential, Normal)
def _kl_exponential_normal(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from an Exponential to a Normal using rate and q's location and variance.
    var_normal = q.scale.pow(2)
    rate_sqr = p.rate.pow(2)
    t1 = 0.5 * torch.log(rate_sqr * var_normal * 2 * math.pi)
    t2 = rate_sqr.reciprocal()
    t3 = q.loc / p.rate
    t4 = q.loc.pow(2) * 0.5
    return t1 - 1 + (t2 - t3 + t4) / var_normal


@register_kl(Gamma, Beta)
@register_kl(Gamma, ContinuousBernoulli)
@register_kl(Gamma, Pareto)
@register_kl(Gamma, Uniform)
def _kl_gamma_infinity(p, q):
    # [FUNCTION DOC] HELPER: KL from a Gamma to Beta, ContinuousBernoulli, Pareto or Uniform: infinity, shaped like the concentration.
    return _infinite_like(p.concentration)


@register_kl(Gamma, Exponential)
def _kl_gamma_exponential(p, q):
    # [FUNCTION DOC] HELPER: KL from a Gamma to an Exponential: negative entropy of
    # p minus log rate of q plus q's rate times the Gamma mean.
    return -p.entropy() - q.rate.log() + q.rate * p.concentration / p.rate


@register_kl(Gamma, Gumbel)
def _kl_gamma_gumbel(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Gamma to a Gumbel using digamma,
    # log-gamma and the product of p's rate and q's scale.
    beta_scale_prod = p.rate * q.scale
    loc_scale_ratio = q.loc / q.scale
    t1 = (
        (p.concentration - 1) * p.concentration.digamma()
        - p.concentration.lgamma()
        - p.concentration
    )
    t2 = beta_scale_prod.log() + p.concentration / beta_scale_prod
    t3 = (
        torch.exp(loc_scale_ratio)
        * (1 + beta_scale_prod.reciprocal()).pow(-p.concentration)
        - loc_scale_ratio
    )
    return t1 + t2 + t3


# TODO: Add Gamma-Laplace KL Divergence


@register_kl(Gamma, Normal)
def _kl_gamma_normal(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Gamma to a Normal using digamma,
    # log-gamma and the first two moments of p.
    var_normal = q.scale.pow(2)
    beta_sqr = p.rate.pow(2)
    t1 = (
        0.5 * torch.log(beta_sqr * var_normal * 2 * math.pi)
        - p.concentration
        - p.concentration.lgamma()
    )
    t2 = 0.5 * (p.concentration.pow(2) + p.concentration) / beta_sqr
    t3 = q.loc * p.concentration / p.rate
    t4 = 0.5 * q.loc.pow(2)
    return (
        t1
        + (p.concentration - 1) * p.concentration.digamma()
        + (t2 - t3 + t4) / var_normal
    )


@register_kl(Gumbel, Beta)
@register_kl(Gumbel, ContinuousBernoulli)
@register_kl(Gumbel, Exponential)
@register_kl(Gumbel, Gamma)
@register_kl(Gumbel, Pareto)
@register_kl(Gumbel, Uniform)
def _kl_gumbel_infinity(p, q):
    # [FUNCTION DOC] HELPER: KL from a Gumbel to Beta, ContinuousBernoulli, Exponential, Gamma, Pareto or Uniform: infinity, shaped like loc.
    return _infinite_like(p.loc)


# TODO: Add Gumbel-Laplace KL Divergence


@register_kl(Gumbel, Normal)
def _kl_gumbel_normal(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Gumbel to a Normal using the
    # Gumbel mean (with the Euler constant) and variance (pi squared over six scale squared).
    param_ratio = p.scale / q.scale
    t1 = (param_ratio / math.sqrt(2 * math.pi)).log()
    t2 = (math.pi * param_ratio * 0.5).pow(2) / 3
    t3 = ((p.loc + p.scale * _euler_gamma - q.loc) / q.scale).pow(2) * 0.5
    return -t1 + t2 + t3 - (_euler_gamma + 1)


@register_kl(Laplace, Beta)
@register_kl(Laplace, ContinuousBernoulli)
@register_kl(Laplace, Exponential)
@register_kl(Laplace, Gamma)
@register_kl(Laplace, Pareto)
@register_kl(Laplace, Uniform)
def _kl_laplace_infinity(p, q):
    # [FUNCTION DOC] HELPER: KL from a Laplace to Beta, ContinuousBernoulli, Exponential, Gamma, Pareto or Uniform: infinity, shaped like loc.
    return _infinite_like(p.loc)


@register_kl(Laplace, Normal)
def _kl_laplace_normal(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Laplace to a Normal using the
    # Laplace variance and the location differences scaled by q's variance.
    var_normal = q.scale.pow(2)
    scale_sqr_var_ratio = p.scale.pow(2) / var_normal
    t1 = 0.5 * torch.log(2 * scale_sqr_var_ratio / math.pi)
    t2 = 0.5 * p.loc.pow(2)
    t3 = p.loc * q.loc
    t4 = 0.5 * q.loc.pow(2)
    return -t1 + scale_sqr_var_ratio + (t2 - t3 + t4) / var_normal - 1


@register_kl(Normal, Beta)
@register_kl(Normal, ContinuousBernoulli)
@register_kl(Normal, Exponential)
@register_kl(Normal, Gamma)
@register_kl(Normal, Pareto)
@register_kl(Normal, Uniform)
def _kl_normal_infinity(p, q):
    # [FUNCTION DOC] HELPER: KL from a Normal to Beta, ContinuousBernoulli, Exponential, Gamma, Pareto or Uniform: infinity, shaped like loc.
    return _infinite_like(p.loc)


@register_kl(Normal, Gumbel)
def _kl_normal_gumbel(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Normal to a Gumbel using the
    # mean/scale ratios and an exponential moment of the normal.
    mean_scale_ratio = p.loc / q.scale
    var_scale_sqr_ratio = (p.scale / q.scale).pow(2)
    loc_scale_ratio = q.loc / q.scale
    t1 = var_scale_sqr_ratio.log() * 0.5
    t2 = mean_scale_ratio - loc_scale_ratio
    t3 = torch.exp(-mean_scale_ratio + 0.5 * var_scale_sqr_ratio + loc_scale_ratio)
    return -t1 + t2 + t3 - (0.5 * (1 + math.log(2 * math.pi)))


@register_kl(Normal, Laplace)
def _kl_normal_laplace(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Normal to a Laplace using the
    # expected absolute deviation (error function and Gaussian terms) and the scale ratio.
    loc_diff = p.loc - q.loc
    scale_ratio = p.scale / q.scale
    loc_diff_scale_ratio = loc_diff / p.scale
    t1 = torch.log(scale_ratio)
    t2 = (
        math.sqrt(2 / math.pi) * p.scale * torch.exp(-0.5 * loc_diff_scale_ratio.pow(2))
    )
    t3 = loc_diff * torch.erf(math.sqrt(0.5) * loc_diff_scale_ratio)
    return -t1 + (t2 + t3) / q.scale - (0.5 * (1 + math.log(0.5 * math.pi)))


@register_kl(Pareto, Beta)
@register_kl(Pareto, ContinuousBernoulli)
@register_kl(Pareto, Uniform)
def _kl_pareto_infinity(p, q):
    # [FUNCTION DOC] HELPER: KL from a Pareto to Beta, ContinuousBernoulli or Uniform: infinity, shaped like scale.
    return _infinite_like(p.scale)


@register_kl(Pareto, Exponential)
def _kl_pareto_exponential(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Pareto to an Exponential; infinity
    # where p's alpha is at most 1 (infinite mean).
    scale_rate_prod = p.scale * q.rate
    t1 = (p.alpha / scale_rate_prod).log()
    t2 = p.alpha.reciprocal()
    t3 = p.alpha * scale_rate_prod / (p.alpha - 1)
    result = t1 - t2 + t3 - 1
    result[p.alpha <= 1] = inf
    return result


@register_kl(Pareto, Gamma)
def _kl_pareto_gamma(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Pareto to a Gamma; infinity where
    # p's alpha is at most 1 (infinite mean).
    common_term = p.scale.log() + p.alpha.reciprocal()
    t1 = p.alpha.log() - common_term
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (1 - q.concentration) * common_term
    t4 = q.rate * p.alpha * p.scale / (p.alpha - 1)
    result = t1 + t2 + t3 + t4 - 1
    result[p.alpha <= 1] = inf
    return result


# TODO: Add Pareto-Laplace KL Divergence


@register_kl(Pareto, Normal)
def _kl_pareto_normal(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Pareto to a Normal using the
    # first two Pareto moments; infinity where p's alpha is at most 2 (infinite variance).
    var_normal = 2 * q.scale.pow(2)
    common_term = p.scale / (p.alpha - 1)
    t1 = (math.sqrt(2 * math.pi) * q.scale * p.alpha / p.scale).log()
    t2 = p.alpha.reciprocal()
    t3 = p.alpha * common_term.pow(2) / (p.alpha - 2)
    t4 = (p.alpha * common_term - q.loc).pow(2)
    result = t1 - t2 + (t3 + t4) / var_normal - 1
    result[p.alpha <= 2] = inf
    return result


@register_kl(Poisson, Bernoulli)
@register_kl(Poisson, Binomial)
def _kl_poisson_infinity(p, q):
    # [FUNCTION DOC] HELPER: KL from a Poisson to Bernoulli or Binomial: infinity, shaped like rate.
    return _infinite_like(p.rate)


@register_kl(Uniform, Beta)
def _kl_uniform_beta(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Uniform to a Beta using x log x
    # terms at the interval endpoints and log-gamma of q's concentrations; infinity
    # where p's interval extends outside the unit interval.
    common_term = p.high - p.low
    t1 = torch.log(common_term)
    t2 = (
        (q.concentration1 - 1)
        * (_x_log_x(p.high) - _x_log_x(p.low) - common_term)
        / common_term
    )
    t3 = (
        (q.concentration0 - 1)
        * (_x_log_x(1 - p.high) - _x_log_x(1 - p.low) + common_term)
        / common_term
    )
    t4 = (
        q.concentration1.lgamma()
        + q.concentration0.lgamma()
        - (q.concentration1 + q.concentration0).lgamma()
    )
    result = t3 + t4 - t1 - t2
    result[(p.high > q.support.upper_bound) | (p.low < q.support.lower_bound)] = inf
    return result


@register_kl(Uniform, ContinuousBernoulli)
def _kl_uniform_continuous_bernoulli(p, q):
    # [FUNCTION DOC] HELPER: KL from a Uniform to a ContinuousBernoulli: negative
    # entropy of p minus the expected log density of q (using p's mean), with
    # infinity selected by a support test.
    # KNOWN ISSUE: the support test uses inclusive comparisons (high at or above
    # the upper bound, low at or below the lower bound), so a Uniform spanning
    # exactly the unit interval wrongly yields infinity; it should be strict
    # (outside the support).
    result = (
        -p.entropy()
        - p.mean * q.logits
        - torch.log1p(-q.probs)
        - q._cont_bern_log_norm()
    )
    return torch.where(
        torch.max(
            torch.ge(p.high, q.support.upper_bound),
            torch.le(p.low, q.support.lower_bound),
        ),
        torch.ones_like(result) * inf,
        result,
    )


@register_kl(Uniform, Exponential)
def _kl_uniform_exponetial(p, q):
    # [FUNCTION DOC] HELPER: KL from a Uniform to an Exponential (function name is
    # misspelled in the source); infinity where p's low is below zero.
    result = q.rate * (p.high + p.low) / 2 - ((p.high - p.low) * q.rate).log()
    result[p.low < q.support.lower_bound] = inf
    return result


@register_kl(Uniform, Gamma)
def _kl_uniform_gamma(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Uniform to a Gamma using x log x
    # terms at the endpoints; infinity where p's low is below zero.
    common_term = p.high - p.low
    t1 = common_term.log()
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (
        (1 - q.concentration)
        * (_x_log_x(p.high) - _x_log_x(p.low) - common_term)
        / common_term
    )
    t4 = q.rate * (p.high + p.low) / 2
    result = -t1 + t2 + t3 + t4
    result[p.low < q.support.lower_bound] = inf
    return result


@register_kl(Uniform, Gumbel)
def _kl_uniform_gumbel(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Uniform to a Gumbel using the standardized interval endpoints.
    common_term = q.scale / (p.high - p.low)
    high_loc_diff = (p.high - q.loc) / q.scale
    low_loc_diff = (p.low - q.loc) / q.scale
    t1 = common_term.log() + 0.5 * (high_loc_diff + low_loc_diff)
    t2 = common_term * (torch.exp(-high_loc_diff) - torch.exp(-low_loc_diff))
    return t1 - t2


# TODO: Uniform-Laplace KL Divergence


@register_kl(Uniform, Normal)
def _kl_uniform_normal(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Uniform to a Normal using the
    # uniform width, variance and midpoint offset from q's location.
    common_term = p.high - p.low
    t1 = (math.sqrt(math.pi * 2) * q.scale / common_term).log()
    t2 = (common_term).pow(2) / 12
    t3 = ((p.high + p.low - 2 * q.loc) / 2).pow(2)
    return t1 + 0.5 * (t2 + t3) / q.scale.pow(2)


@register_kl(Uniform, Pareto)
def _kl_uniform_pareto(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL from a Uniform to a Pareto using x log x
    # terms at the endpoints; infinity where p's low is below q's scale.
    support_uniform = p.high - p.low
    t1 = (q.alpha * q.scale.pow(q.alpha) * (support_uniform)).log()
    t2 = (_x_log_x(p.high) - _x_log_x(p.low) - support_uniform) / support_uniform
    result = t2 * (q.alpha + 1) - t1
    result[p.low < q.support.lower_bound] = inf
    return result


@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    # [FUNCTION DOC] HELPER: KL between Independent wrappers: raises
    # NotImplementedError if the reinterpreted batch dim counts differ; otherwise
    # takes the KL of the base distributions and sums over the reinterpreted dims.
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)


@register_kl(Cauchy, Cauchy)
def _kl_cauchy_cauchy(p, q):
    # [FUNCTION DOC] HELPER: Closed-form KL between Cauchy distributions: log of
    # (sum of scales squared plus location difference squared) over four times the product of scales.
    # From https://arxiv.org/abs/1905.10965
    t1 = ((p.scale + q.scale).pow(2) + (p.loc - q.loc).pow(2)).log()
    t2 = (4 * p.scale * q.scale).log()
    return t1 - t2


def _add_kl_info():
    # [FUNCTION DOC] HELPER: Builds a bullet list of all registered (p, q) type
    # pairs sorted by class names and appends it to the kl_divergence docstring (in
    # place, if that docstring exists). Intended to be called once by the package
    # at import time; it is not called inside this module.
    """Appends a list of implemented KL functions to the doc for kl_divergence."""
    rows = [
        "KL divergence is currently implemented for the following distribution pairs:"
    ]
    for p, q in sorted(
        _KL_REGISTRY, key=lambda p_q: (p_q[0].__name__, p_q[1].__name__)
    ):
        rows.append(
            f"* :class:`~torch.distributions.{p.__name__}` and :class:`~torch.distributions.{q.__name__}`"
        )
    kl_info = "\n\t".join(rows)
    if kl_divergence.__doc__:
        # pyrefly: ignore [missing-attribute]
        kl_divergence.__doc__ += kl_info
