# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines ConstraintRegistry, a mapping from Constraint classes to factory
# callables that build Transform objects (unconstrained real space -> constrained
# space), and the two global registries biject_to (bijective transforms) and
# transform_to (not necessarily bijective, cheaper transforms).
# Import-time side effect: populates both registries with the standard
# constraint/transform pairs (real, independent, positive, greater_than, less_than,
# interval, simplex, lower_cholesky, positive_definite, corr_cholesky, cat, stack).
# Exports: ConstraintRegistry, biject_to, transform_to.
r"""
PyTorch provides two global :class:`ConstraintRegistry` objects that link
:class:`~torch.distributions.constraints.Constraint` objects to
:class:`~torch.distributions.transforms.Transform` objects. These objects both
input constraints and return transforms, but they have different guarantees on
bijectivity.

1. ``biject_to(constraint)`` looks up a bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is guaranteed to have
   ``.bijective = True`` and should implement ``.log_abs_det_jacobian()``.
2. ``transform_to(constraint)`` looks up a not-necessarily bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is not guaranteed to
   implement ``.log_abs_det_jacobian()``.

The ``transform_to()`` registry is useful for performing unconstrained
optimization on constrained parameters of probability distributions, which are
indicated by each distribution's ``.arg_constraints`` dict. These transforms often
overparameterize a space in order to avoid rotation; they are thus more
suitable for coordinate-wise optimization algorithms like Adam::

    loc = torch.zeros(100, requires_grad=True)
    unconstrained = torch.zeros(100, requires_grad=True)
    scale = transform_to(Normal.arg_constraints["scale"])(unconstrained)
    loss = -Normal(loc, scale).log_prob(data).sum()

The ``biject_to()`` registry is useful for Hamiltonian Monte Carlo, where
samples from a probability distribution with constrained ``.support`` are
propagated in an unconstrained space, and algorithms are typically rotation
invariant.::

    dist = Exponential(rate)
    unconstrained = torch.zeros(100, requires_grad=True)
    sample = biject_to(dist.support)(unconstrained)
    potential_energy = -dist.log_prob(sample).sum()

.. note::

    An example where ``transform_to`` and ``biject_to`` differ is
    ``constraints.simplex``: ``transform_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.SoftmaxTransform` that simply
    exponentiates and normalizes its inputs; this is a cheap and mostly
    coordinate-wise operation appropriate for algorithms like SVI. In
    contrast, ``biject_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.StickBreakingTransform` that
    bijects its input down to a one-fewer-dimensional space; this a more
    expensive less numerically stable transform but is needed for algorithms
    like HMC.

The ``biject_to`` and ``transform_to`` objects can be extended by user-defined
constraints and transforms using their ``.register()`` method either as a
function on singleton constraints::

    transform_to.register(my_constraint, my_transform)

or as a decorator on parameterized constraints::

    @transform_to.register(MyConstraintClass)
    def my_factory(constraint):
        assert isinstance(constraint, MyConstraintClass)
        return MyTransform(constraint.param1, constraint.param2)

You can create your own registry by creating a new :class:`ConstraintRegistry`
object.
"""

from torch.distributions import constraints, transforms
from torch.types import _Number


__all__ = [
    "ConstraintRegistry",
    "biject_to",
    "transform_to",
]


class ConstraintRegistry:
    # [CLASS DOC]
    # Registry mapping Constraint subclasses to factory callables. State: the
    # private mapping _registry from constraint class to factory. Calling the
    # registry on a constraint instance dispatches on its exact type (no
    # inheritance lookup) and returns the Transform built by the factory.
    """
    Registry to link constraints to transforms.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Creates an empty registry (initializes the private mapping to empty).
        self._registry = {}
        super().__init__()

    def register(self, constraint, factory=None):
        # [FUNCTION DOC]
        # Associates a Constraint subclass (or the class of a given Constraint
        # instance) with a factory callable. If factory is omitted, returns a
        # decorator that registers the decorated function. Returns the factory.
        # Mutates the registry; a later registration overrides an earlier one.
        # Raises TypeError if constraint is neither a Constraint subclass nor instance.
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (Callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        """
        # Support use as decorator.
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        # Support calling on singleton instances.
        if isinstance(constraint, constraints.Constraint):
            constraint = type(constraint)

        if not isinstance(constraint, type) or not issubclass(
            constraint, constraints.Constraint
        ):
            raise TypeError(
                f"Expected constraint to be either a Constraint subclass or instance, but got {constraint}"
            )

        self._registry[constraint] = factory
        return factory

    def __call__(self, constraint):
        # [FUNCTION DOC]
        # Looks up the factory registered for the exact type of the constraint
        # object and returns the Transform it builds from that constraint.
        # Raises NotImplementedError if no factory is registered for that type.
        """
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.arg_constraints["scale"]
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)  # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        """
        # Look up by Constraint subclass.
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                f"Cannot transform {type(constraint).__name__} constraints"
            ) from None
        return factory(constraint)


biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()


################################################################################
# Registration Table
################################################################################


@biject_to.register(constraints.real)
@transform_to.register(constraints.real)
def _transform_to_real(constraint):
    # [FUNCTION DOC] HELPER: Factory for the real constraint; returns the shared identity transform.
    return transforms.identity_transform


@biject_to.register(constraints.independent)
def _biject_to_independent(constraint):
    # [FUNCTION DOC] HELPER: Factory for the independent constraint; wraps the bijection of the base constraint in an IndependentTransform with the same number of reinterpreted batch dims.
    base_transform = biject_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )


@transform_to.register(constraints.independent)
def _transform_to_independent(constraint):
    # [FUNCTION DOC] HELPER: Same as the bijective independent factory but uses the transform_to registry for the base constraint.
    base_transform = transform_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )


@biject_to.register(constraints.positive)
@biject_to.register(constraints.nonnegative)
@transform_to.register(constraints.positive)
@transform_to.register(constraints.nonnegative)
def _transform_to_positive(constraint):
    # [FUNCTION DOC] HELPER: Factory for positive and nonnegative constraints in both registries; returns an ExpTransform.
    return transforms.ExpTransform()


@biject_to.register(constraints.greater_than)
@biject_to.register(constraints.greater_than_eq)
@transform_to.register(constraints.greater_than)
@transform_to.register(constraints.greater_than_eq)
def _transform_to_greater_than(constraint):
    # [FUNCTION DOC] HELPER: Factory for greater_than(_eq) constraints; composes exp with an affine shift by the lower bound (scale 1).
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.lower_bound, 1),
        ]
    )


@biject_to.register(constraints.less_than)
@transform_to.register(constraints.less_than)
def _transform_to_less_than(constraint):
    # [FUNCTION DOC] HELPER: Factory for less_than constraints; composes exp with an affine map of loc upper bound and scale -1, so outputs lie below the upper bound.
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.upper_bound, -1),
        ]
    )


@biject_to.register(constraints.interval)
@biject_to.register(constraints.half_open_interval)
@transform_to.register(constraints.interval)
@transform_to.register(constraints.half_open_interval)
def _transform_to_interval(constraint):
    # [FUNCTION DOC] HELPER: Factory for interval and half_open_interval constraints. If the bounds are plain numbers 0 and 1, returns a bare SigmoidTransform; otherwise composes sigmoid with an affine map (loc = lower bound, scale = upper minus lower).
    # Handle the special case of the unit interval.
    lower_is_0 = (
        isinstance(constraint.lower_bound, _Number) and constraint.lower_bound == 0
    )
    upper_is_1 = (
        isinstance(constraint.upper_bound, _Number) and constraint.upper_bound == 1
    )
    if lower_is_0 and upper_is_1:
        return transforms.SigmoidTransform()

    loc = constraint.lower_bound
    scale = constraint.upper_bound - constraint.lower_bound
    return transforms.ComposeTransform(
        [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
    )


@biject_to.register(constraints.simplex)
def _biject_to_simplex(constraint):
    # [FUNCTION DOC] HELPER: Factory for simplex in biject_to; returns a StickBreakingTransform.
    return transforms.StickBreakingTransform()


@transform_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    # [FUNCTION DOC] HELPER: Factory for simplex in transform_to; returns a SoftmaxTransform (not bijective).
    return transforms.SoftmaxTransform()


# TODO define a bijection for LowerCholeskyTransform
@transform_to.register(constraints.lower_cholesky)
def _transform_to_lower_cholesky(constraint):
    # [FUNCTION DOC] HELPER: Factory for lower_cholesky; returns a LowerCholeskyTransform (registered only in transform_to).
    return transforms.LowerCholeskyTransform()


@transform_to.register(constraints.positive_definite)
@transform_to.register(constraints.positive_semidefinite)
def _transform_to_positive_definite(constraint):
    # [FUNCTION DOC] HELPER: Factory for positive_definite and positive_semidefinite (transform_to only); returns a PositiveDefiniteTransform.
    return transforms.PositiveDefiniteTransform()


@biject_to.register(constraints.corr_cholesky)
@transform_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    # [FUNCTION DOC] HELPER: Factory for corr_cholesky in both registries; returns a CorrCholeskyTransform.
    return transforms.CorrCholeskyTransform()


@biject_to.register(constraints.cat)
def _biject_to_cat(constraint):
    # [FUNCTION DOC] HELPER: Factory for cat constraints; builds a CatTransform from the bijections of each component constraint, reusing the constraint's dim and lengths.
    return transforms.CatTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


@transform_to.register(constraints.cat)
def _transform_to_cat(constraint):
    # [FUNCTION DOC] HELPER: Same as the bijective cat factory but uses transform_to for each component constraint.
    return transforms.CatTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


@biject_to.register(constraints.stack)
def _biject_to_stack(constraint):
    # [FUNCTION DOC] HELPER: Factory for stack constraints; builds a StackTransform from the bijections of each component constraint along the constraint's dim.
    return transforms.StackTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim
    )


@transform_to.register(constraints.stack)
def _transform_to_stack(constraint):
    # [FUNCTION DOC] HELPER: Same as the bijective stack factory but uses transform_to for each component constraint.
    return transforms.StackTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim
    )
