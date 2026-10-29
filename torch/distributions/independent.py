# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines Independent, a generic wrapper that reinterprets the rightmost batch
# dims of a base distribution as event dims, so densities and entropy are summed
# over them. Defines the type variable D bound to Distribution. No import-time
# side effects. Exports: Independent.
from typing import Generic, TypeVar

import torch
from torch import Size, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _sum_rightmost
from torch.types import _size


__all__ = ["Independent"]


D = TypeVar("D", bound=Distribution)


class Independent(Distribution, Generic[D]):
    # [CLASS DOC]
    # Wrapper distribution. State: base_dist (the wrapped distribution) and
    # reinterpreted_batch_ndims. Invariant: the count never exceeds the base batch
    # rank; batch shape is the base batch shape minus the rightmost reinterpreted
    # dims, event shape is those dims followed by the base event shape. Sampling
    # is delegated to the base; log_prob and entropy are summed over the dims.
    # Has no parameters of its own (arg_constraints is empty).
    r"""
    Reinterprets some of the batch dims of a distribution as event dims.

    This is mainly useful for changing the shape of the result of
    :meth:`log_prob`. For example to create a diagonal Normal distribution with
    the same shape as a Multivariate Normal distribution (so they are
    interchangeable), you can::

        >>> from torch.distributions.multivariate_normal import MultivariateNormal
        >>> from torch.distributions.normal import Normal
        >>> loc = torch.zeros(3)
        >>> scale = torch.ones(3)
        >>> mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        >>> [mvn.batch_shape, mvn.event_shape]
        [torch.Size([]), torch.Size([3])]
        >>> normal = Normal(loc, scale)
        >>> [normal.batch_shape, normal.event_shape]
        [torch.Size([3]), torch.Size([])]
        >>> diagn = Independent(normal, 1)
        >>> [diagn.batch_shape, diagn.event_shape]
        [torch.Size([]), torch.Size([3])]

    Args:
        base_distribution (torch.distributions.distribution.Distribution): a
            base distribution
        reinterpreted_batch_ndims (int): the number of batch dims to
            reinterpret as event dims
    """

    arg_constraints: dict[str, constraints.Constraint] = {}
    base_dist: D

    def __init__(
        self,
        base_distribution: D,
        reinterpreted_batch_ndims: int,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the base distribution and count, and computes the new batch and event shapes by moving the rightmost reinterpreted batch dims into the event shape. Raises ValueError if the count exceeds the length of the base batch shape.
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError(
                "Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
                f"actual {reinterpreted_batch_ndims} vs {len(base_distribution.batch_shape)}"
            )
        shape: Size = base_distribution.batch_shape + base_distribution.event_shape
        event_dim: int = reinterpreted_batch_ndims + len(base_distribution.event_shape)
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new Independent whose base distribution is expanded to batch_shape followed by the reinterpreted dims, with the same count and validation setting; self is unchanged.
        new = self._get_checked_instance(Independent, _instance)
        batch_shape = torch.Size(batch_shape)
        new.base_dist = self.base_dist.expand(
            batch_shape + self.event_shape[: self.reinterpreted_batch_ndims]
        )
        new.reinterpreted_batch_ndims = self.reinterpreted_batch_ndims
        super(Independent, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC] Whether the base distribution supports reparameterized sampling.
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC] False if any batch dims are reinterpreted (a Cartesian product is not supported), else the base distribution's flag.
        if self.reinterpreted_batch_ndims > 0:
            return False
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def support(self):
        # [FUNCTION DOC] Support constraint: the base support, wrapped in an independent constraint over the reinterpreted dims when the count is nonzero.
        result = self.base_dist.support
        if self.reinterpreted_batch_ndims:
            result = constraints.independent(result, self.reinterpreted_batch_ndims)
        return result

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Mean of the base distribution (unchanged shape).
        return self.base_dist.mean

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Mode of the base distribution (unchanged shape).
        return self.base_dist.mode

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Variance of the base distribution (unchanged shape).
        return self.base_dist.variance

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        # [FUNCTION DOC] Draws samples by delegating to the base distribution's sample.
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Draws reparameterized samples by delegating to the base distribution's rsample (raises if the base does not support it).
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        # [FUNCTION DOC] Log density of value: the base log density summed over the rightmost reinterpreted dims. Validation is left to the base distribution; no validation is done here.
        log_prob = self.base_dist.log_prob(value)
        return _sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

    def entropy(self):
        # [FUNCTION DOC] Entropy: the base entropy summed over the rightmost reinterpreted dims.
        entropy = self.base_dist.entropy()
        return _sum_rightmost(entropy, self.reinterpreted_batch_ndims)

    def enumerate_support(self, expand=True):
        # [FUNCTION DOC] Delegates to the base distribution's enumerate_support when no dims are reinterpreted; raises NotImplementedError otherwise (Cartesian product not implemented).
        if self.reinterpreted_batch_ndims > 0:
            raise NotImplementedError(
                "Enumeration over cartesian product is not implemented"
            )
        return self.base_dist.enumerate_support(expand=expand)

    def __repr__(self):
        # [FUNCTION DOC] Returns the class name with the base distribution representation and the reinterpreted dim count in parentheses.
        return (
            self.__class__.__name__
            + f"({self.base_dist}, {self.reinterpreted_batch_ndims})"
        )
