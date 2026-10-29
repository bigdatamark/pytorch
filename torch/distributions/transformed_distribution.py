# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines TransformedDistribution, a Distribution built from a base
# distribution plus a sequence of invertible Transform objects. Exports only
# TransformedDistribution. Importing has no side effects beyond imports.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent
from torch.distributions.transforms import ComposeTransform, Transform
from torch.distributions.utils import _sum_rightmost
from torch.types import _size


__all__ = ["TransformedDistribution"]


class TransformedDistribution(Distribution):
    # [CLASS DOC]
    # Distribution of the output of applying a list of Transforms in order to
    # samples of a base distribution. State: transforms (list of Transform) and
    # base_dist (the base distribution, expanded and/or wrapped in Independent
    # so its event dims match the composed transform domain). Batch and event
    # shapes are derived from the forward-mapped base shape; the event rank is
    # the larger of the transform codomain rank and the shifted base event rank.
    # Sampling goes forward; log_prob, cdf and icdf invert the transforms.
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
        log p(Y) = log p(X) + log |det (dX/dY)|

    Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
    maximum shape of its base distribution and its transforms, since transforms
    can introduce correlations among events.

    An example for the usage of :class:`TransformedDistribution` would be::

        # Building a Logistic Distribution
        # X ~ Uniform(0, 1)
        # f = a + b * logit(X)
        # Y ~ f(X) ~ Logistic(a, b)
        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
        logistic = TransformedDistribution(base_distribution, transforms)

    For more examples, please look at the implementations of
    :class:`~torch.distributions.gumbel.Gumbel`,
    :class:`~torch.distributions.half_cauchy.HalfCauchy`,
    :class:`~torch.distributions.half_normal.HalfNormal`,
    :class:`~torch.distributions.log_normal.LogNormal`,
    :class:`~torch.distributions.pareto.Pareto`,
    :class:`~torch.distributions.weibull.Weibull`,
    :class:`~torch.distributions.relaxed_bernoulli.RelaxedBernoulli` and
    :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`
    """

    arg_constraints: dict[str, constraints.Constraint] = {}

    def __init__(
        self,
        base_distribution: Distribution,
        transforms: Transform | list[Transform],
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the distribution from a base distribution and a single
        # Transform or a list of Transforms. Composes them, checks the base
        # shape is at least the domain event rank, expands the base to the
        # shape implied by the inverse of the forward shape, and wraps it in
        # Independent if the transform domain has more event dims. Then sets
        # batch and event shapes. Raises ValueError for invalid transforms or
        # too-small base shape. Sets self.transforms and self.base_dist.
        if isinstance(transforms, Transform):
            self.transforms = [
                transforms,
            ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError(
                    "transforms must be a Transform or a list of Transforms"
                )
            self.transforms = transforms
        else:
            raise ValueError(
                f"transforms must be a Transform or list, but was {transforms}"
            )

        # Reshape base_distribution according to transforms.
        base_shape = base_distribution.batch_shape + base_distribution.event_shape
        base_event_dim = len(base_distribution.event_shape)
        transform = ComposeTransform(self.transforms)
        if len(base_shape) < transform.domain.event_dim:
            raise ValueError(
                f"base_distribution needs to have shape with size at least {transform.domain.event_dim}, but got {base_shape}."
            )
        forward_shape = transform.forward_shape(base_shape)
        expanded_base_shape = transform.inverse_shape(forward_shape)
        if base_shape != expanded_base_shape:
            base_batch_shape = expanded_base_shape[
                : len(expanded_base_shape) - base_event_dim
            ]
            base_distribution = base_distribution.expand(base_batch_shape)
        reinterpreted_batch_ndims = transform.domain.event_dim - base_event_dim
        if reinterpreted_batch_ndims > 0:
            base_distribution = Independent(
                base_distribution, reinterpreted_batch_ndims
            )
        self.base_dist = base_distribution

        # Compute shapes.
        transform_change_in_event_dim = (
            transform.codomain.event_dim - transform.domain.event_dim
        )
        event_dim = max(
            transform.codomain.event_dim,  # the transform is coupled
            base_event_dim + transform_change_in_event_dim,  # the base dist is coupled
        )
        if len(forward_shape) < event_dim:
            raise AssertionError(
                f"forward_shape length {len(forward_shape)} must be >= event_dim {event_dim}"
            )
        cut = len(forward_shape) - event_dim
        batch_shape = forward_shape[:cut]
        event_shape = forward_shape[cut:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC]
        # Returns a new instance with the given batch shape. The base batch
        # shape is found by pushing the full shape back through the inverse
        # shape of each transform in reverse order; the base distribution is
        # expanded to it and the transform list is shared (not copied).
        # Validation is carried over from self.
        new = self._get_checked_instance(TransformedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        shape = batch_shape + self.event_shape
        for t in reversed(self.transforms):
            shape = t.inverse_shape(shape)
        base_batch_shape = shape[: len(shape) - len(self.base_dist.event_shape)]
        new.base_dist = self.base_dist.expand(base_batch_shape)
        new.transforms = self.transforms
        super(TransformedDistribution, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def support(self):
        # [FUNCTION DOC]
        # Support constraint: the base support when there are no transforms,
        # otherwise the codomain of the last transform, wrapped as an
        # independent constraint when the event rank exceeds the codomain's.
        if not self.transforms:
            return self.base_dist.support
        support = self.transforms[-1].codomain
        if len(self.event_shape) > support.event_dim:
            support = constraints.independent(
                support, len(self.event_shape) - support.event_dim
            )
        return support

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC]
        # True exactly when the base distribution supports reparameterized
        # sampling.
        return self.base_dist.has_rsample

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC]
        # Draws a non-differentiable sample (under no-grad) from the base
        # distribution and applies each transform forward in order. Returns a
        # tensor of shape sample_shape plus batch and event shape.
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC]
        # Draws a reparameterized sample from the base distribution (gradients
        # flow) and applies each transform forward in order. Requires the base
        # to implement rsample, otherwise its error propagates.
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        # [FUNCTION DOC]
        # Log density of value. Optionally validates the sample. Walks the
        # transforms in reverse, inverting each and subtracting its log abs
        # det jacobian summed over the extra rightmost event dims, tracking the
        # running event rank; finally adds the base log_prob summed over any
        # remaining event dims. Returns a tensor over the batch shape.
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob: Tensor | float = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.domain.event_dim,
            )
            y = x

        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(y), event_dim - len(self.base_dist.event_shape)
        )
        return log_prob

    def _monotonize_cdf(self, value):
        # [FUNCTION DOC] HELPER: Multiplies the signs of all transforms; if the
        # product is the plain integer one, returns value unchanged, otherwise
        # reflects value about one half scaled by the sign (flipping for
        # decreasing transforms) so cdf/icdf are monotone increasing.
        """
        This conditionally flips ``value -> 1-value`` to ensure :meth:`cdf` is
        monotone increasing.
        """
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        if isinstance(sign, int) and sign == 1:
            return value
        return sign * (value - 0.5) + 0.5

    def cdf(self, value):
        # [FUNCTION DOC]
        # Cumulative distribution: inverts all transforms (last to first),
        # optionally validates against the base support, evaluates the base
        # cdf, then applies the monotonic sign correction. Returns a tensor.
        """
        Computes the cumulative distribution function by inverting the
        transform(s) and computing the score of the base distribution.
        """
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.cdf(value)
        value = self._monotonize_cdf(value)
        return value

    def icdf(self, value):
        # [FUNCTION DOC]
        # Inverse cdf: applies the monotonic sign correction to the
        # probability, evaluates the base icdf, then applies each transform
        # forward in order. Returns a tensor.
        """
        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        """
        value = self._monotonize_cdf(value)
        value = self.base_dist.icdf(value)
        for transform in self.transforms:
            value = transform(value)
        return value
