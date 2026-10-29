# mypy: allow-untyped-defs
# [MODULE DOC]
# Library of invertible transforms with computable log abs det jacobians, used
# mainly by TransformedDistribution. Defines the abstract Transform base class
# (optional single-entry caching, lazy inverse object), an internal inverse
# wrapper, combinators (Compose, Independent, Cat, Stack), shape transforms
# (Reshape) and concrete maps (Exp, Power, Sigmoid, Softplus, Tanh, Abs, Affine,
# CorrCholesky, Softmax, StickBreaking, LowerCholesky, PositiveDefinite,
# CumulativeDistribution). Also defines the module-level identity_transform (an
# empty ComposeTransform). Import-time side effects: creates identity_transform.
import functools
import math
import operator
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import (
    _sum_rightmost,
    broadcast_all,
    lazy_property,
    tril_matrix_to_vec,
    vec_to_tril_matrix,
)
from torch.nn.functional import pad, softplus
from torch.types import _Number


__all__ = [
    "AbsTransform",
    "AffineTransform",
    "CatTransform",
    "ComposeTransform",
    "CorrCholeskyTransform",
    "CumulativeDistributionTransform",
    "ExpTransform",
    "IndependentTransform",
    "LowerCholeskyTransform",
    "PositiveDefiniteTransform",
    "PowerTransform",
    "ReshapeTransform",
    "SigmoidTransform",
    "SoftplusTransform",
    "TanhTransform",
    "SoftmaxTransform",
    "StackTransform",
    "StickBreakingTransform",
    "Transform",
    "identity_transform",
]


class Transform:
    # [CLASS DOC]
    # Abstract base for invertible maps. Subclasses set domain, codomain,
    # bijective and sign and implement _call and/or _inverse and
    # log_abs_det_jacobian. State: _cache_size (0 or 1), _inv (lazily created
    # inverse transform) and, when caching, _cached_x_y holding the latest
    # input/output pair (identity-compared). Calling the object applies the
    # forward map (with cache); the inv property gives the inverse transform.
    # Equality defaults to identity.
    """
    Abstract class for invertible transformations with computable log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.

    Caching is useful for transforms whose inverses are either expensive or
    numerically unstable. Note that care must be taken with memoized values
    since the autograd graph may be reversed. For example while the following
    works with or without caching::

        y = t(x)
        t.log_abs_det_jacobian(x, y).backward()  # x will receive gradients.

    However the following will error when caching due to dependency reversal::

        y = t(x)
        z = t.inv(y)
        grad(z.sum(), [y])  # error because z is x

    Derived classes should implement one or both of :meth:`_call` or
    :meth:`_inverse`. Derived classes that set `bijective=True` should also
    implement :meth:`log_abs_det_jacobian`.

    Args:
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.

    Attributes:
        domain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid inputs to this transform.
        codomain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid outputs to this transform
            which are inputs to the inverse transform.
        bijective (bool): Whether this transform is bijective. A transform
            ``t`` is bijective iff ``t.inv(t(x)) == x`` and
            ``t(t.inv(y)) == y`` for every ``x`` in the domain and ``y`` in
            the codomain. Transforms that are not bijective should at least
            maintain the weaker pseudoinverse properties
            ``t(t.inv(t(x)) == t(x)`` and ``t.inv(t(t.inv(y))) == t.inv(y)``.
        sign (int or Tensor): For bijective univariate transforms, this
            should be +1 or -1 depending on whether transform is monotone
            increasing or decreasing.
    """

    bijective = False
    domain: constraints.Constraint
    codomain: constraints.Constraint

    def __init__(self, cache_size: int = 0) -> None:
        # [FUNCTION DOC]
        # Stores the cache size and an empty inverse slot; for cache size one
        # initializes the cached pair to empty. Raises ValueError for any cache
        # size other than 0 or 1.
        self._cache_size = cache_size
        self._inv: Transform | None = None
        if cache_size == 0:
            pass  # default behavior
        elif cache_size == 1:
            self._cached_x_y = None, None
        else:
            raise ValueError("cache_size must be 0 or 1")
        super().__init__()

    def __getstate__(self):
        # [FUNCTION DOC]
        # Pickling support: returns a copy of the instance state with the
        # cached inverse link cleared, to avoid circular references.
        state = self.__dict__.copy()
        state["_inv"] = None
        return state

    @property
    def event_dim(self) -> int:
        # [FUNCTION DOC]
        # Event rank of the transform: the shared event rank of domain and
        # codomain. Raises ValueError if they differ.
        if self.domain.event_dim == self.codomain.event_dim:
            return self.domain.event_dim
        raise ValueError("Please use either .domain.event_dim or .codomain.event_dim")

    @property
    def inv(self) -> "Transform":
        # [FUNCTION DOC]
        # Returns the inverse transform, creating and remembering an inverse
        # wrapper on first access (side effect: sets _inv).
        """
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        """
        inv = self._inv
        if inv is None:
            inv = _InverseTransform(self)
            self._inv = inv
        return inv

    @property
    def sign(self) -> int:
        # [FUNCTION DOC]
        # Abstract: subclasses give the sign of the jacobian determinant (plus or
        # minus one for monotone univariate maps). Base raises NotImplementedError.
        """
        Returns the sign of the determinant of the Jacobian, if applicable.
        In general this only makes sense for bijective transforms.
        """
        raise NotImplementedError

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC]
        # Returns this transform if the cache size already matches, otherwise a
        # new instance of the same class with the requested cache size. Only
        # works for subclasses that do not override the constructor; otherwise
        # raises NotImplementedError.
        if self._cache_size == cache_size:
            return self
        if type(self).__init__ is Transform.__init__:
            return type(self)(cache_size=cache_size)
        raise NotImplementedError(f"{type(self)}.with_cache is not implemented")

    def __eq__(self, other):
        # [FUNCTION DOC] Default equality is object identity.
        return self is other

    def __ne__(self, other):
        # [FUNCTION DOC] Negation of the equality method.
        # Necessary for Python2
        return not self.__eq__(other)

    def __call__(self, x):
        # [FUNCTION DOC]
        # Applies the forward map. Without caching calls _call directly. With
        # caching returns the cached output if x is the very same object as the
        # cached input, else computes, stores the (x, y) pair and returns y.
        """
        Computes the transform `x => y`.
        """
        if self._cache_size == 0:
            return self._call(x)
        x_old, y_old = self._cached_x_y
        if x is x_old:
            return y_old
        y = self._call(x)
        self._cached_x_y = x, y
        return y

    def _inv_call(self, y):
        # [FUNCTION DOC] HELPER: Applies the inverse map with the same single
        # entry cache: returns the cached input if y is the cached output object,
        # else calls _inverse and stores the pair.
        """
        Inverts the transform `y => x`.
        """
        if self._cache_size == 0:
            return self._inverse(y)
        x_old, y_old = self._cached_x_y
        if y is y_old:
            return x_old
        x = self._inverse(y)
        self._cached_x_y = x, y
        return x

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Abstract forward computation; raises
        # NotImplementedError in the base class.
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Abstract inverse computation; raises
        # NotImplementedError in the base class.
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC]
        # Abstract: log of the absolute jacobian determinant of the forward map
        # given input x and output y, reduced over the event dims. Base raises
        # NotImplementedError.
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        raise NotImplementedError

    def __repr__(self):
        # [FUNCTION DOC] String form: the class name followed by empty parentheses.
        return self.__class__.__name__ + "()"

    def forward_shape(self, shape):
        # [FUNCTION DOC] Output shape of the forward map for an input shape;
        # default preserves the shape.
        """
        Infers the shape of the forward computation, given the input shape.
        Defaults to preserving shape.
        """
        return shape

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Output shape of the inverse map for an output shape;
        # default preserves the shape.
        """
        Infers the shapes of the inverse computation, given the output shape.
        Defaults to preserving shape.
        """
        return shape


class _InverseTransform(Transform):
    # [CLASS DOC] HELPER: Wrapper presenting the inverse of one Transform. Its
    # _inv slot holds the original transform; domain/codomain are swapped, calls
    # delegate to the original's inverse call, and the jacobian is negated.
    """
    Inverts a single :class:`Transform`.
    This class is private; please instead use the ``Transform.inv`` property.
    """

    def __init__(self, transform: Transform) -> None:
        # [FUNCTION DOC] HELPER: Wraps transform, copying its cache size and
        # linking it as the inverse.
        super().__init__(cache_size=transform._cache_size)
        self._inv: Transform | None = transform

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def domain(self):
        # [FUNCTION DOC] Domain is the wrapped transform's codomain. Raises
        # AssertionError if the link is missing.
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv.codomain

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def codomain(self):
        # [FUNCTION DOC] Codomain is the wrapped transform's domain. Raises
        # AssertionError if the link is missing.
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv.domain

    @property
    def bijective(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC] Same bijectivity as the wrapped transform.
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv.bijective

    @property
    def sign(self) -> int:
        # [FUNCTION DOC] Same sign as the wrapped transform (the inverse of a
        # monotone map has the same monotonicity direction).
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv.sign

    @property
    def inv(self) -> Transform:
        # [FUNCTION DOC] The inverse of the inverse is the original transform.
        return self._inv  # pyrefly: ignore[bad-return]

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC] Returns the inverse of the original transform rebuilt
        # with the requested cache size.
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self.inv.with_cache(cache_size).inv

    def __eq__(self, other):
        # [FUNCTION DOC] Equal when other is also an inverse wrapper around an
        # equal transform.
        if not isinstance(other, _InverseTransform):
            return False
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv == other._inv

    def __repr__(self):
        # [FUNCTION DOC] Class name wrapping the repr of the original transform.
        return f"{self.__class__.__name__}({repr(self._inv)})"

    def __call__(self, x):
        # [FUNCTION DOC] Applies the original transform's cached inverse to x.
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv._inv_call(x)

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Negative of the original transform's log abs det
        # jacobian evaluated with the roles of input and output swapped.
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return -self._inv.log_abs_det_jacobian(y, x)

    def forward_shape(self, shape):
        # [FUNCTION DOC] Forward shape is the original's inverse shape.
        return self._inv.inverse_shape(shape)  # pyrefly: ignore[missing-attribute]

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Inverse shape is the original's forward shape.
        return self._inv.forward_shape(shape)  # pyrefly: ignore[missing-attribute]


class ComposeTransform(Transform):
    # [CLASS DOC]
    # Chain of transforms applied in order. State: parts (list of Transform;
    # rebuilt with caching when a cache size is given, and parts do the caching).
    # Domain/codomain come from the first/last part with event rank adjusted to
    # the maximum along the chain. An empty chain is the identity.
    """
    Composes multiple transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform`): A list of transforms to compose.
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.
    """

    def __init__(self, parts: list[Transform], cache_size: int = 0) -> None:
        # [FUNCTION DOC] Stores parts; if cache_size is nonzero, replaces each
        # part with a cached version first.
        if cache_size:
            parts = [part.with_cache(cache_size) for part in parts]
        super().__init__(cache_size=cache_size)
        self.parts = parts

    def __eq__(self, other):
        # [FUNCTION DOC] Equal when other is a ComposeTransform with equal parts.
        if not isinstance(other, ComposeTransform):
            return False
        return self.parts == other.parts

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def domain(self):
        # [FUNCTION DOC] Real line if empty; else the first part's domain,
        # wrapped as independent when the chain needs a higher event rank (found
        # by walking parts backward). Raises AssertionError if inconsistent.
        if not self.parts:
            return constraints.real
        domain = self.parts[0].domain
        # Adjust event_dim to be maximum among all parts.
        event_dim = self.parts[-1].codomain.event_dim
        for part in reversed(self.parts):
            event_dim += part.domain.event_dim - part.codomain.event_dim
            event_dim = max(event_dim, part.domain.event_dim)
        if event_dim < domain.event_dim:
            raise AssertionError(
                f"event_dim {event_dim} must be >= domain.event_dim {domain.event_dim}"
            )
        if event_dim > domain.event_dim:
            domain = constraints.independent(domain, event_dim - domain.event_dim)
        return domain

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def codomain(self):
        # [FUNCTION DOC] Real line if empty; else the last part's codomain,
        # wrapped as independent when the chain needs a higher event rank (found
        # by walking parts forward). Raises AssertionError if inconsistent.
        if not self.parts:
            return constraints.real
        codomain = self.parts[-1].codomain
        # Adjust event_dim to be maximum among all parts.
        event_dim = self.parts[0].domain.event_dim
        for part in self.parts:
            event_dim += part.codomain.event_dim - part.domain.event_dim
            event_dim = max(event_dim, part.codomain.event_dim)
        if event_dim < codomain.event_dim:
            raise AssertionError(
                f"event_dim {event_dim} must be >= codomain.event_dim {codomain.event_dim}"
            )
        if event_dim > codomain.event_dim:
            codomain = constraints.independent(codomain, event_dim - codomain.event_dim)
        return codomain

    @lazy_property
    def bijective(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC] Cached: true when every part is bijective.
        return all(p.bijective for p in self.parts)

    @lazy_property
    def sign(self) -> int:  # type: ignore[override]
        # [FUNCTION DOC] Cached: product of the signs of all parts.
        sign = 1
        for p in self.parts:
            sign = sign * p.sign
        return sign

    @property
    def inv(self) -> Transform:
        # [FUNCTION DOC] Lazily builds and remembers a ComposeTransform of the
        # inverses of the parts in reverse order, linking it back to this one.
        inv = self._inv
        if inv is None:
            inv = ComposeTransform([p.inv for p in reversed(self.parts)])
            self._inv = inv
            inv._inv = self
        return inv

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC] Same object if cache size matches, else a new
        # ComposeTransform over the same parts with the given cache size.
        if self._cache_size == cache_size:
            return self
        return ComposeTransform(self.parts, cache_size=cache_size)

    def __call__(self, x):
        # [FUNCTION DOC] Applies each part in order to x and returns the result
        # (caching, if any, is done inside the parts).
        for part in self.parts:
            x = part(x)
        return x

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC]
        # Zeros shaped like x for an empty chain. Otherwise recomputes the
        # intermediate values by applying the parts (free when cached), sums each
        # part's log abs det jacobian over the extra rightmost event dims so all
        # terms share the chain's event rank, and adds the terms.
        if not self.parts:
            return torch.zeros_like(x)

        # Compute intermediates. This will be free if parts[:-1] are all cached.
        xs = [x]
        for part in self.parts[:-1]:
            xs.append(part(xs[-1]))
        xs.append(y)

        terms = []
        event_dim = self.domain.event_dim
        for part, x, y in zip(self.parts, xs[:-1], xs[1:]):
            terms.append(
                _sum_rightmost(
                    part.log_abs_det_jacobian(x, y), event_dim - part.domain.event_dim
                )
            )
            event_dim += part.codomain.event_dim - part.domain.event_dim
        return functools.reduce(operator.add, terms)

    def forward_shape(self, shape):
        # [FUNCTION DOC] Pipes the shape through each part's forward shape.
        for part in self.parts:
            shape = part.forward_shape(shape)
        return shape

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Pipes the shape through each part's inverse shape in
        # reverse order.
        for part in reversed(self.parts):
            shape = part.inverse_shape(shape)
        return shape

    def __repr__(self):
        # [FUNCTION DOC] Multi-line string listing the repr of every part.
        fmt_string = self.__class__.__name__ + "(\n    "
        fmt_string += ",\n    ".join([p.__repr__() for p in self.parts])
        fmt_string += "\n)"
        return fmt_string


identity_transform = ComposeTransform([])


class IndependentTransform(Transform):
    # [CLASS DOC]
    # Wraps a base transform and declares extra rightmost batch dims as event
    # dims. State: base_transform (cached to the same size) and
    # reinterpreted_batch_ndims. Forward and inverse are unchanged; only the log
    # abs det jacobian is summed over the extra dims; domain/codomain become
    # independent constraints.
    """
    Wrapper around another transform to treat
    ``reinterpreted_batch_ndims``-many extra of the right most dimensions as
    dependent. This has no effect on the forward or backward transforms, but
    does sum out ``reinterpreted_batch_ndims``-many of the rightmost dimensions
    in :meth:`log_abs_det_jacobian`.

    Args:
        base_transform (:class:`Transform`): A base transform.
        reinterpreted_batch_ndims (int): The number of extra rightmost
            dimensions to treat as dependent.
    """

    def __init__(
        self,
        base_transform: Transform,
        reinterpreted_batch_ndims: int,
        cache_size: int = 0,
    ) -> None:
        # [FUNCTION DOC] Stores the base transform (with the given cache size)
        # and the number of reinterpreted batch dims.
        super().__init__(cache_size=cache_size)
        self.base_transform = base_transform.with_cache(cache_size)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC] Same object if cache size matches, else a new wrapper
        # with the given cache size.
        if self._cache_size == cache_size:
            return self
        return IndependentTransform(
            self.base_transform, self.reinterpreted_batch_ndims, cache_size=cache_size
        )

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def domain(self):
        # [FUNCTION DOC] Base domain reinterpreted with extra event dims.
        return constraints.independent(
            self.base_transform.domain, self.reinterpreted_batch_ndims
        )

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def codomain(self):
        # [FUNCTION DOC] Base codomain reinterpreted with extra event dims.
        return constraints.independent(
            self.base_transform.codomain, self.reinterpreted_batch_ndims
        )

    @property
    def bijective(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC] Same as the base transform.
        return self.base_transform.bijective

    @property
    def sign(self) -> int:
        # [FUNCTION DOC] Same as the base transform.
        return self.base_transform.sign

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Applies the base transform; raises ValueError if
        # x has fewer dims than the domain event rank.
        if x.dim() < self.domain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform(x)

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Applies the base inverse; raises ValueError if y
        # has fewer dims than the codomain event rank.
        if y.dim() < self.codomain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform.inv(y)

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Base log abs det jacobian summed over the
        # reinterpreted rightmost dims.
        result = self.base_transform.log_abs_det_jacobian(x, y)
        result = _sum_rightmost(result, self.reinterpreted_batch_ndims)
        return result

    def __repr__(self):
        # [FUNCTION DOC] Class name with the base transform repr and dim count.
        return f"{self.__class__.__name__}({repr(self.base_transform)}, {self.reinterpreted_batch_ndims})"

    def forward_shape(self, shape):
        # [FUNCTION DOC] Delegates to the base transform.
        return self.base_transform.forward_shape(shape)

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Delegates to the base transform.
        return self.base_transform.inverse_shape(shape)


class ReshapeTransform(Transform):
    # [CLASS DOC]
    # Bijective unit-jacobian transform that reshapes the rightmost event part of
    # a tensor from in_shape to out_shape (equal element counts). State: in_shape
    # and out_shape (torch.Size). Event ranks are the lengths of these shapes.
    """
    Unit Jacobian transform to reshape the rightmost part of a tensor.

    Note that ``in_shape`` and ``out_shape`` must have the same number of
    elements, just as for :meth:`torch.Tensor.reshape`.

    Arguments:
        in_shape (torch.Size): The input event shape.
        out_shape (torch.Size): The output event shape.
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported. (Default 0.)
    """

    bijective = True

    def __init__(
        self,
        in_shape: torch.Size,
        out_shape: torch.Size,
        cache_size: int = 0,
    ) -> None:
        # [FUNCTION DOC] Stores both shapes as torch.Size. Raises ValueError if
        # their element counts differ.
        self.in_shape = torch.Size(in_shape)
        self.out_shape = torch.Size(out_shape)
        if self.in_shape.numel() != self.out_shape.numel():
            raise ValueError("in_shape, out_shape have different numbers of elements")
        super().__init__(cache_size=cache_size)

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def domain(self):
        # [FUNCTION DOC] Real values with event rank equal to len of in_shape.
        return constraints.independent(constraints.real, len(self.in_shape))

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def codomain(self):
        # [FUNCTION DOC] Real values with event rank equal to len of out_shape.
        return constraints.independent(constraints.real, len(self.out_shape))

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC] Same object if cache size matches, else a new
        # ReshapeTransform with the same shapes and the given cache size.
        if self._cache_size == cache_size:
            return self
        return ReshapeTransform(self.in_shape, self.out_shape, cache_size=cache_size)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Reshapes the trailing in_shape dims of x to
        # out_shape, keeping leading batch dims.
        batch_shape = x.shape[: x.dim() - len(self.in_shape)]
        return x.reshape(batch_shape + self.out_shape)

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Reshapes the trailing out_shape dims of y back
        # to in_shape, keeping leading batch dims.
        batch_shape = y.shape[: y.dim() - len(self.out_shape)]
        return y.reshape(batch_shape + self.in_shape)

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Zeros shaped like the batch part of x (unit jacobian).
        batch_shape = x.shape[: x.dim() - len(self.in_shape)]
        return x.new_zeros(batch_shape)

    def forward_shape(self, shape):
        # [FUNCTION DOC] Replaces the trailing in_shape part of shape with
        # out_shape. Raises ValueError if shape is too short or its trailing part
        # differs from in_shape.
        # KNOWN ISSUE: the mismatch message swaps "expected" and "got" (it prints
        # the actual trailing shape as expected).
        if len(shape) < len(self.in_shape):
            raise ValueError("Too few dimensions on input")
        cut = len(shape) - len(self.in_shape)
        if shape[cut:] != self.in_shape:
            raise ValueError(
                f"Shape mismatch: expected {shape[cut:]} but got {self.in_shape}"
            )
        return shape[:cut] + self.out_shape

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Replaces the trailing out_shape part of shape with
        # in_shape. Raises ValueError if shape is too short or its trailing part
        # differs from out_shape (message has the same swapped wording).
        if len(shape) < len(self.out_shape):
            raise ValueError("Too few dimensions on input")
        cut = len(shape) - len(self.out_shape)
        if shape[cut:] != self.out_shape:
            raise ValueError(
                f"Shape mismatch: expected {shape[cut:]} but got {self.out_shape}"
            )
        return shape[:cut] + self.in_shape


class ExpTransform(Transform):
    # [CLASS DOC]
    # Bijective increasing map from the reals to the positive reals via the
    # exponential; inverse is the natural log.
    r"""
    Transform via the mapping :math:`y = \exp(x)`.
    """

    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to any ExpTransform instance.
        return isinstance(other, ExpTransform)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Elementwise exponential.
        return x.exp()

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Elementwise natural log.
        return y.log()

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Elementwise log derivative of exp, which is x itself.
        return x


class PowerTransform(Transform):
    # [CLASS DOC]
    # Bijective map on the positive reals raising x to a (tensor) exponent.
    # State: exponent (broadcast tensor). Inverse raises to the reciprocal.
    r"""
    Transform via the mapping :math:`y = x^{\text{exponent}}`.
    """

    domain = constraints.positive
    codomain = constraints.positive
    bijective = True

    def __init__(self, exponent: Tensor, cache_size: int = 0) -> None:
        # [FUNCTION DOC] Stores the exponent converted to a tensor.
        super().__init__(cache_size=cache_size)
        (self.exponent,) = broadcast_all(exponent)

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC] Same object if cache size matches, else a new
        # PowerTransform with the same exponent and the given cache size.
        if self._cache_size == cache_size:
            return self
        return PowerTransform(self.exponent, cache_size=cache_size)

    @lazy_property
    def sign(self) -> int:  # type: ignore[override]
        # [FUNCTION DOC] Cached elementwise sign of the exponent (a tensor, not
        # an int, despite the annotation).
        return self.exponent.sign()  # type: ignore[return-value]

    def __eq__(self, other):
        # [FUNCTION DOC] Equal when other is a PowerTransform whose exponent equals
        # this one at every element.
        if not isinstance(other, PowerTransform):
            return False
        return self.exponent.eq(other.exponent).all().item()

    def _call(self, x):
        # [FUNCTION DOC] HELPER: x raised to the exponent.
        return x.pow(self.exponent)

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: y raised to the reciprocal of the exponent.
        return y.pow(1 / self.exponent)

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Elementwise log of the absolute value of exponent times
        # y over x (the derivative of the power map).
        return (self.exponent * y / x).abs().log()

    def forward_shape(self, shape):
        # [FUNCTION DOC] Broadcast of shape with the exponent shape.
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Broadcast of shape with the exponent shape.
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))


def _clipped_sigmoid(x):
    # [FUNCTION DOC] HELPER: Sigmoid of x clamped between the dtype's smallest
    # normal number and one minus epsilon, so outputs stay strictly inside (0, 1).
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)


class SigmoidTransform(Transform):
    # [CLASS DOC]
    # Bijective increasing map from the reals to the unit interval via a clipped
    # sigmoid; inverse is the logit with clamped input.
    r"""
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    """

    domain = constraints.real
    codomain = constraints.unit_interval
    bijective = True
    sign = +1

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to any SigmoidTransform instance.
        return isinstance(other, SigmoidTransform)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Clipped sigmoid of x.
        return _clipped_sigmoid(x)

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Clamps y away from zero and one, then returns the
        # logit (log y minus log of one minus y).
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Stable log derivative of the sigmoid: minus softplus of
        # -x minus softplus of x.
        return -F.softplus(-x) - F.softplus(x)


class SoftplusTransform(Transform):
    # [CLASS DOC]
    # Bijective increasing map from the reals to the positive reals via softplus
    # (linear for large inputs); inverse is computed with expm1 for stability.
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    The implementation reverts to the linear function when :math:`x > 20`.
    """

    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to any SoftplusTransform instance.
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Softplus of x.
        return softplus(x)

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Inverse softplus: log of (1 - exp(-y)), plus y.
        return (-y).expm1().neg().log() + y

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Log derivative of softplus, which is the log sigmoid:
        # minus softplus of -x.
        return -softplus(-x)


class TanhTransform(Transform):
    # [CLASS DOC]
    # Bijective increasing map from the reals to the open interval (-1, 1) via
    # hyperbolic tangent. The inverse does not clamp, so values at the boundary
    # give infinities (caching is advised).
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.

    It is equivalent to

    .. code-block:: python

        ComposeTransform(
            [
                AffineTransform(0.0, 2.0),
                SigmoidTransform(),
                AffineTransform(-1.0, 2.0),
            ]
        )

    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.

    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.

    """

    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to any TanhTransform instance.
        return isinstance(other, TanhTransform)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Hyperbolic tangent of x.
        return x.tanh()

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Inverse hyperbolic tangent of y, unclamped.
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Numerically stable log derivative of tanh: twice
        # (log 2 minus x minus softplus of -2x).
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2.0 * (math.log(2.0) - x - softplus(-2.0 * x))


class AbsTransform(Transform):
    # [CLASS DOC]
    # Non-bijective absolute value map from the reals to the positive reals; the
    # pseudo-inverse is the identity. No jacobian or sign is defined.
    r"""Transform via the mapping :math:`y = |x|`."""

    domain = constraints.real
    codomain = constraints.positive

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to any AbsTransform instance.
        return isinstance(other, AbsTransform)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Elementwise absolute value.
        return x.abs()

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Pseudo-inverse: returns y unchanged (the
        # positive preimage).
        return y


class AffineTransform(Transform):
    # [CLASS DOC]
    # Bijective pointwise map y = loc + scale * x, with loc and scale numbers or
    # tensors. State: loc, scale, _event_dim (number of rightmost event dims; the
    # jacobian is summed over them). Sign follows the scale sign.
    r"""
    Transform via the pointwise affine mapping :math:`y = \text{loc} + \text{scale} \times x`.

    Args:
        loc (Tensor or float): Location parameter.
        scale (Tensor or float): Scale parameter.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """

    bijective = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        event_dim: int = 0,
        cache_size: int = 0,
    ) -> None:
        # [FUNCTION DOC] Stores loc, scale and event_dim (no validation of scale).
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale = scale
        self._event_dim = event_dim

    @property
    def event_dim(self) -> int:
        # [FUNCTION DOC] The configured event rank.
        return self._event_dim

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def domain(self):
        # [FUNCTION DOC] Real line, wrapped as independent when event rank is
        # positive.
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def codomain(self):
        # [FUNCTION DOC] Real line, wrapped as independent when event rank is
        # positive.
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC] Same object if cache size matches, else a new
        # AffineTransform with the same parameters and the given cache size.
        if self._cache_size == cache_size:
            return self
        return AffineTransform(
            self.loc, self.scale, self.event_dim, cache_size=cache_size
        )

    def __eq__(self, other):
        # [FUNCTION DOC] Equal when other is an AffineTransform whose loc and
        # scale match (numbers by value, tensors elementwise-all).
        # KNOWN ISSUE: event_dim and cache size are not compared.
        if not isinstance(other, AffineTransform):
            return False

        if isinstance(self.loc, _Number) and isinstance(other.loc, _Number):
            if self.loc != other.loc:
                return False
        else:
            if not (self.loc == other.loc).all().item():  # type: ignore[union-attr]
                return False

        if isinstance(self.scale, _Number) and isinstance(other.scale, _Number):
            if self.scale != other.scale:
                return False
        else:
            if not (self.scale == other.scale).all().item():  # type: ignore[union-attr]
                return False

        return True

    @property
    def sign(self) -> Tensor | int:  # type: ignore[override]
        # [FUNCTION DOC] For a numeric scale: one, minus one or zero by its sign;
        # for a tensor scale: its elementwise sign.
        if isinstance(self.scale, _Number):
            return 1 if float(self.scale) > 0 else -1 if float(self.scale) < 0 else 0
        return self.scale.sign()

    def _call(self, x):
        # [FUNCTION DOC] HELPER: loc plus scale times x.
        return self.loc + self.scale * x

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: (y minus loc) divided by scale.
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Log of the absolute scale (a full tensor like x for a
        # numeric scale), summed over the rightmost event dims when event rank is
        # positive, then expanded to the batch part of x's shape.
        shape = x.shape
        scale = self.scale
        if isinstance(scale, _Number):
            result = torch.full_like(x, math.log(abs(scale)))
        else:
            result = torch.abs(scale).log()
        if self.event_dim:
            result_size = result.size()[: -self.event_dim] + (-1,)
            result = result.view(result_size).sum(-1)
            shape = shape[: -self.event_dim]
        return result.expand(shape)

    def forward_shape(self, shape):
        # [FUNCTION DOC] Broadcast of shape with the shapes of loc and scale.
        return torch.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Broadcast of shape with the shapes of loc and scale.
        return torch.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )


class CorrCholeskyTransform(Transform):
    # [CLASS DOC]
    # Bijective map from an unconstrained real vector of length D(D-1)/2 to the
    # Cholesky factor of a D by D correlation matrix, using a signed
    # stick-breaking construction on tanh-squashed values. Domain is real
    # vectors; codomain is correlation Cholesky matrices.
    r"""
    Transforms an unconstrained real vector :math:`x` with length :math:`D*(D-1)/2` into the
    Cholesky factor of a D-dimension correlation matrix. This Cholesky factor is a lower
    triangular matrix with positive diagonals and unit Euclidean norm for each row.
    The transform is processed as follows:

        1. First we convert x into a lower triangular matrix in row order.
        2. For each row :math:`X_i` of the lower triangular part, we apply a *signed* version of
           class :class:`StickBreakingTransform` to transform :math:`X_i` into a
           unit Euclidean length vector using the following steps:
           - Scales into the interval :math:`(-1, 1)` domain: :math:`r_i = \tanh(X_i)`.
           - Transforms into an unsigned domain: :math:`z_i = r_i^2`.
           - Applies :math:`s_i = StickBreakingTransform(z_i)`.
           - Transforms back into signed domain: :math:`y_i = sign(r_i) * \sqrt{s_i}`.
    """

    domain = constraints.real_vector
    codomain = constraints.corr_cholesky
    bijective = True

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Applies tanh and clamps just inside (-1, 1),
        # arranges the values into a strictly lower-triangular matrix, adds the
        # identity diagonal and scales each row by the cumulative product of the
        # square roots of one minus squared entries (shifted right by one).
        # Returns a batch of lower-triangular matrices with unit-norm rows.
        x = torch.tanh(x)
        eps = torch.finfo(x.dtype).eps
        x = x.clamp(min=-1 + eps, max=1 - eps)
        r = vec_to_tril_matrix(x, diag=-1)
        # apply stick-breaking on the squared values
        # Note that y = sign(r) * sqrt(z * z1m_cumprod)
        #             = (sign(r) * sqrt(z)) * sqrt(z1m_cumprod) = r * sqrt(z1m_cumprod)
        z = r**2
        z1m_cumprod_sqrt = (1 - z).sqrt().cumprod(-1)
        # Diagonal elements must be 1.
        r = r + torch.eye(r.shape[-1], dtype=r.dtype, device=r.device)
        y = r * pad(z1m_cumprod_sqrt[..., :-1], [1, 0], value=1)
        return y

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Inverts the construction: divides the strictly
        # lower entries of y by the square root of the remaining squared norm of
        # each row prefix, then applies inverse tanh. Returns the flattened
        # vector of length D(D-1)/2.
        # inverse stick-breaking
        # See: https://mc-stan.org/docs/2_18/reference-manual/cholesky-factors-of-correlation-matrices-1.html
        y_cumsum = 1 - torch.cumsum(y * y, dim=-1)
        y_cumsum_shifted = pad(y_cumsum[..., :-1], [1, 0], value=1)
        y_vec = tril_matrix_to_vec(y, diag=-1)
        y_cumsum_vec = tril_matrix_to_vec(y_cumsum_shifted, diag=-1)
        t = y_vec / (y_cumsum_vec).sqrt()
        # inverse of tanh
        x = (t.log1p() - t.neg().log1p()) / 2
        return x

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # [FUNCTION DOC]
        # Log jacobian between x and the flattened strictly-lower part of y:
        # half the sum of logs of the remaining row-norm terms (stick-breaking
        # part) plus the tanh part. Returns a tensor over batch dims. The
        # intermediates argument is accepted but unused.
        # Because domain and codomain are two spaces with different dimensions, determinant of
        # Jacobian is not well-defined. We return `log_abs_det_jacobian` of `x` and the
        # flattened lower triangular part of `y`.

        # See: https://mc-stan.org/docs/2_18/reference-manual/cholesky-factors-of-correlation-matrices-1.html
        y1m_cumsum = 1 - (y * y).cumsum(dim=-1)
        # by taking diagonal=-2, we don't need to shift z_cumprod to the right
        # also works for 2 x 2 matrix
        y1m_cumsum_tril = tril_matrix_to_vec(y1m_cumsum, diag=-2)
        stick_breaking_logdet = 0.5 * (y1m_cumsum_tril).log().sum(-1)
        tanh_logdet = -2 * (x + softplus(-2 * x) - math.log(2.0)).sum(dim=-1)
        return stick_breaking_logdet + tanh_logdet

    def forward_shape(self, shape):
        # [FUNCTION DOC] Maps a vector shape (batch, N) to (batch, D, D) where N is
        # D(D-1)/2. Raises ValueError for empty shape or if N is not such a number.
        # Reshape from (..., N) to (..., D, D).
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        N = shape[-1]
        D = round((0.25 + 2 * N) ** 0.5 + 0.5)
        if D * (D - 1) // 2 != N:
            raise ValueError("Input is not a flattened lower-diagonal number")
        return shape[:-1] + (D, D)

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Maps a matrix shape (batch, D, D) to (batch, N) with N =
        # D(D-1)/2. Raises ValueError if fewer than two dims or not square.
        # Reshape from (..., D, D) to (..., N).
        if len(shape) < 2:
            raise ValueError("Too few dimensions on input")
        if shape[-2] != shape[-1]:
            raise ValueError("Input is not square")
        D = shape[-1]
        N = D * (D - 1) // 2
        return shape[:-2] + (N,)


class SoftmaxTransform(Transform):
    # [CLASS DOC]
    # Non-bijective map from real vectors to the simplex of the same dimension
    # via softmax; the pseudo-inverse is the log. Acts mostly coordinate-wise.
    r"""
    Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """

    domain = constraints.real_vector
    codomain = constraints.simplex

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to any SoftmaxTransform instance.
        return isinstance(other, SoftmaxTransform)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Softmax along the last dim (max-subtracted for
        # stability).
        logprobs = x
        probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        return probs / probs.sum(-1, True)

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Pseudo-inverse: elementwise log of the
        # probabilities.
        probs = y
        return probs.log()

    def forward_shape(self, shape):
        # [FUNCTION DOC] Shape is unchanged; raises ValueError if shape is empty.
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Shape is unchanged; raises ValueError if shape is empty.
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape


class StickBreakingTransform(Transform):
    # [CLASS DOC]
    # Bijective map from real vectors of length n to the simplex of dimension
    # n + 1 through an iterated sigmoid stick-breaking process, with an offset
    # per coordinate so zero maps to the uniform point. Suitable for HMC.
    """
    Transform from unconstrained space to the simplex of one additional
    dimension via a stick-breaking process.

    This transform arises as an iterated sigmoid transform in a stick-breaking
    construction of the `Dirichlet` distribution: the first logit is
    transformed via sigmoid to the first probability and the probability of
    everything else, and then the process recurses.

    This is bijective and appropriate for use in HMC; however it mixes
    coordinates together and is less appropriate for optimization.
    """

    domain = constraints.real_vector
    codomain = constraints.simplex
    bijective = True

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to any StickBreakingTransform instance.
        return isinstance(other, StickBreakingTransform)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Shifts x by the log of a descending offset,
        # applies a clipped sigmoid to get break fractions, and builds the simplex
        # point as each fraction times the remaining stick length, with the final
        # coordinate the leftover.
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        z = _clipped_sigmoid(x - offset.log())
        z_cumprod = (1 - z).cumprod(-1)
        y = pad(z, [0, 1], value=1) * pad(z_cumprod, [1, 0], value=1)
        return y

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Recovers x from simplex point y: logs of the
        # cropped coordinates minus logs of the remaining stick (clamped to stay
        # positive) plus the log offset.
        y_crop = y[..., :-1]
        offset = y.shape[-1] - y.new_ones(y_crop.shape[-1]).cumsum(-1)
        sf = 1 - y_crop.cumsum(-1)
        # we clamp to make sure that sf is positive which sometimes does not
        # happen when y[-1] ~ 0 or y[:-1].sum() ~ 1
        sf = torch.clamp(sf, min=torch.finfo(y.dtype).tiny)
        x = y_crop.log() - sf.log() + offset.log()
        return x

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Sum over the last dim of the stable log jacobian terms
        # built from the shifted x, a log sigmoid, and the log of the cropped y.
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        x = x - offset.log()
        # use the identity 1 - sigmoid(x) = exp(-x) * sigmoid(x)
        detJ = (-x + F.logsigmoid(x) + y[..., :-1].log()).sum(-1)
        return detJ

    def forward_shape(self, shape):
        # [FUNCTION DOC] Last dim grows by one. Raises ValueError if shape is empty.
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] + 1,)

    def inverse_shape(self, shape):
        # [FUNCTION DOC] Last dim shrinks by one. Raises ValueError if shape is empty.
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] - 1,)


class LowerCholeskyTransform(Transform):
    # [CLASS DOC]
    # Maps unconstrained square matrices to lower-triangular matrices with
    # positive diagonal (strict lower part kept, diagonal exponentiated, upper
    # part dropped). Not marked bijective; the inverse takes the log of the
    # diagonal. No jacobian is defined.
    """
    Transform from unconstrained matrices to lower-triangular matrices with
    nonnegative diagonal entries.

    This is useful for parameterizing positive definite matrices in terms of
    their Cholesky factorization.
    """

    domain = constraints.independent(constraints.real, 2)
    codomain = constraints.lower_cholesky

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to any LowerCholeskyTransform instance.
        return isinstance(other, LowerCholeskyTransform)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Strictly lower part of x plus a diagonal matrix of
        # the exponentiated diagonal of x.
        return x.tril(-1) + x.diagonal(dim1=-2, dim2=-1).exp().diag_embed()

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Strictly lower part of y plus a diagonal matrix
        # of the log of the diagonal of y.
        return y.tril(-1) + y.diagonal(dim1=-2, dim2=-1).log().diag_embed()


class PositiveDefiniteTransform(Transform):
    # [CLASS DOC]
    # Maps unconstrained matrices to positive-definite matrices: first through
    # LowerCholeskyTransform, then multiplying the factor by its transpose. The
    # inverse takes a Cholesky decomposition then the inverse lower transform.
    """
    Transform from unconstrained matrices to positive-definite matrices.
    """

    domain = constraints.independent(constraints.real, 2)
    codomain = constraints.positive_definite

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to any PositiveDefiniteTransform instance.
        return isinstance(other, PositiveDefiniteTransform)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Lower Cholesky factor of x times its transpose.
        x = LowerCholeskyTransform()(x)
        return x @ x.mT

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Cholesky-factorizes y (may raise a linear algebra
        # error if not positive definite) and applies the inverse lower-Cholesky map.
        y = torch.linalg.cholesky(y)
        return LowerCholeskyTransform().inv(y)


class CatTransform(Transform):
    # [CLASS DOC]
    # Applies a sequence of transforms to consecutive slices of given lengths
    # along dim and concatenates the results (like torch.cat). State:
    # transforms, lengths (default one per transform), dim. Jacobian terms are
    # concatenated along the dim, or summed when the dim falls inside the event
    # dims.
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`, of length `lengths[dim]`,
    in a way compatible with :func:`torch.cat`.

    Example::

       x0 = torch.cat([torch.range(1, 10), torch.range(1, 10)], dim=0)
       x = torch.cat([x0, x0], dim=0)
       t0 = CatTransform([ExpTransform(), identity_transform], dim=0, lengths=[10, 10])
       t = CatTransform([t0, t0], dim=0, lengths=[20, 20])
       y = t(x)
    """

    transforms: list[Transform]

    def __init__(
        self,
        tseq: Sequence[Transform],
        dim: int = 0,
        lengths: Sequence[int] | None = None,
        cache_size: int = 0,
    ) -> None:
        # [FUNCTION DOC] Stores the transforms (cached if requested), lengths
        # and dim. Raises AssertionError if any element is not a Transform or the
        # number of lengths differs from the number of transforms.
        if not all(isinstance(t, Transform) for t in tseq):
            raise AssertionError("All elements of tseq must be Transform instances")
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super().__init__(cache_size=cache_size)
        self.transforms = list(tseq)
        if lengths is None:
            lengths = [1] * len(self.transforms)
        self.lengths = list(lengths)
        if len(self.lengths) != len(self.transforms):
            raise AssertionError(
                f"lengths ({len(self.lengths)}) must match transforms ({len(self.transforms)})"
            )
        self.dim = dim

    @lazy_property
    def event_dim(self) -> int:  # type: ignore[override]
        # [FUNCTION DOC] Cached maximum event rank among the transforms.
        return max(t.event_dim for t in self.transforms)

    @lazy_property
    def length(self) -> int:
        # [FUNCTION DOC] Cached total length (sum of lengths) expected along dim.
        return sum(self.lengths)

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC] Same object if cache size matches, else a new
        # CatTransform with the same configuration and the given cache size.
        if self._cache_size == cache_size:
            return self
        return CatTransform(self.transforms, self.dim, self.lengths, cache_size)

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Splits x along dim into the configured lengths,
        # applies each transform to its slice and concatenates. Raises
        # AssertionError if dim is out of range or x's size along dim is not the
        # total length.
        if not (-x.dim() <= self.dim < x.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for tensor with {x.dim()} dimensions"
            )
        if x.size(self.dim) != self.length:
            raise AssertionError(
                f"x.size({self.dim}) = {x.size(self.dim)} must equal length {self.length}"
            )
        yslices = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            yslices.append(trans(xslice))
            start = start + length  # avoid += for jit compat
        return torch.cat(yslices, dim=self.dim)

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Splits y along dim, applies each transform's
        # inverse to its slice and concatenates. Raises AssertionError on a bad
        # dim or size.
        if not (-y.dim() <= self.dim < y.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for tensor with {y.dim()} dimensions"
            )
        if y.size(self.dim) != self.length:
            raise AssertionError(
                f"y.size({self.dim}) = {y.size(self.dim)} must equal length {self.length}"
            )
        xslices = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            yslice = y.narrow(self.dim, start, length)
            xslices.append(trans.inv(yslice))
            start = start + length  # avoid += for jit compat
        return torch.cat(xslices, dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Computes each transform's log jacobian on its slices (summed
        # over extra event dims to a common rank), then concatenates them along the
        # dim if it lies in the batch part, otherwise sums them. Raises
        # AssertionError on a bad dim or size of x or y.
        if not (-x.dim() <= self.dim < x.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for x with {x.dim()} dimensions"
            )
        if x.size(self.dim) != self.length:
            raise AssertionError(
                f"x.size({self.dim}) = {x.size(self.dim)} must equal length {self.length}"
            )
        if not (-y.dim() <= self.dim < y.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for y with {y.dim()} dimensions"
            )
        if y.size(self.dim) != self.length:
            raise AssertionError(
                f"y.size({self.dim}) = {y.size(self.dim)} must equal length {self.length}"
            )
        logdetjacs = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            yslice = y.narrow(self.dim, start, length)
            logdetjac = trans.log_abs_det_jacobian(xslice, yslice)
            if trans.event_dim < self.event_dim:
                logdetjac = _sum_rightmost(logdetjac, self.event_dim - trans.event_dim)
            logdetjacs.append(logdetjac)
            start = start + length  # avoid += for jit compat
        # Decide whether to concatenate or sum.
        dim = self.dim
        if dim >= 0:
            dim = dim - x.dim()
        dim = dim + self.event_dim
        if dim < 0:
            return torch.cat(logdetjacs, dim=dim)
        else:
            return sum(logdetjacs)

    @property
    def bijective(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC] True when every transform is bijective.
        return all(t.bijective for t in self.transforms)

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def domain(self):
        # [FUNCTION DOC] Concatenation constraint of the transforms' domains.
        return constraints.cat(
            [t.domain for t in self.transforms], self.dim, self.lengths
        )

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def codomain(self):
        # [FUNCTION DOC] Concatenation constraint of the transforms' codomains.
        return constraints.cat(
            [t.codomain for t in self.transforms], self.dim, self.lengths
        )


class StackTransform(Transform):
    # [CLASS DOC]
    # Applies one transform per index along dim of the input (like torch.stack):
    # slices are selected, transformed individually and stacked back. State:
    # transforms and dim. The size along dim must equal the number of transforms.
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`
    in a way compatible with :func:`torch.stack`.

    Example::

       x = torch.stack([torch.range(1, 10), torch.range(1, 10)], dim=1)
       t = StackTransform([ExpTransform(), identity_transform], dim=1)
       y = t(x)
    """

    transforms: list[Transform]

    def __init__(
        self, tseq: Sequence[Transform], dim: int = 0, cache_size: int = 0
    ) -> None:
        # [FUNCTION DOC] Stores the transforms (cached if requested) and dim.
        # Raises AssertionError if any element is not a Transform.
        if not all(isinstance(t, Transform) for t in tseq):
            raise AssertionError("All elements of tseq must be Transform instances")
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super().__init__(cache_size=cache_size)
        self.transforms = list(tseq)
        self.dim = dim

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC] Same object if cache size matches, else a new
        # StackTransform with the same configuration and the given cache size.
        if self._cache_size == cache_size:
            return self
        return StackTransform(self.transforms, self.dim, cache_size)

    def _slice(self, z):
        # [FUNCTION DOC] HELPER: Returns the list of sub-tensors of z selected at
        # each index along dim.
        return [z.select(self.dim, i) for i in range(z.size(self.dim))]

    def _call(self, x):
        # [FUNCTION DOC] HELPER: Applies the i-th transform to the i-th slice of x
        # along dim and stacks the results. Raises AssertionError on a bad dim or
        # a size different from the number of transforms.
        if not (-x.dim() <= self.dim < x.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for tensor with {x.dim()} dimensions"
            )
        if x.size(self.dim) != len(self.transforms):
            raise AssertionError(
                f"x.size({self.dim}) = {x.size(self.dim)} must equal len(transforms) {len(self.transforms)}"
            )
        yslices = []
        for xslice, trans in zip(self._slice(x), self.transforms):
            yslices.append(trans(xslice))
        return torch.stack(yslices, dim=self.dim)

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: Applies the i-th transform's inverse to the i-th
        # slice of y along dim and stacks the results. Raises AssertionError on a
        # bad dim or size.
        if not (-y.dim() <= self.dim < y.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for tensor with {y.dim()} dimensions"
            )
        if y.size(self.dim) != len(self.transforms):
            raise AssertionError(
                f"y.size({self.dim}) = {y.size(self.dim)} must equal len(transforms) {len(self.transforms)}"
            )
        xslices = []
        for yslice, trans in zip(self._slice(y), self.transforms):
            xslices.append(trans.inv(yslice))
        return torch.stack(xslices, dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] Stacks, along dim, each transform's log jacobian on its
        # slice pair. Raises AssertionError on a bad dim or size of x or y.
        if not (-x.dim() <= self.dim < x.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for x with {x.dim()} dimensions"
            )
        if x.size(self.dim) != len(self.transforms):
            raise AssertionError(
                f"x.size({self.dim}) = {x.size(self.dim)} must equal len(transforms) {len(self.transforms)}"
            )
        if not (-y.dim() <= self.dim < y.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for y with {y.dim()} dimensions"
            )
        if y.size(self.dim) != len(self.transforms):
            raise AssertionError(
                f"y.size({self.dim}) = {y.size(self.dim)} must equal len(transforms) {len(self.transforms)}"
            )
        logdetjacs = []
        yslices = self._slice(y)
        xslices = self._slice(x)
        for xslice, yslice, trans in zip(xslices, yslices, self.transforms):
            logdetjacs.append(trans.log_abs_det_jacobian(xslice, yslice))
        return torch.stack(logdetjacs, dim=self.dim)

    @property
    def bijective(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC] True when every transform is bijective.
        return all(t.bijective for t in self.transforms)

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def domain(self):
        # [FUNCTION DOC] Stack constraint of the transforms' domains along dim.
        return constraints.stack([t.domain for t in self.transforms], self.dim)

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def codomain(self):
        # [FUNCTION DOC] Stack constraint of the transforms' codomains along dim.
        return constraints.stack([t.codomain for t in self.transforms], self.dim)


class CumulativeDistributionTransform(Transform):
    # [CLASS DOC]
    # Bijective increasing map from a distribution's support to the unit
    # interval via its cdf; inverse uses the icdf. State: distribution. The
    # log jacobian is the distribution's log density at the input.
    """
    Transform via the cumulative distribution function of a probability distribution.

    Args:
        distribution (Distribution): Distribution whose cumulative distribution function to use for
            the transformation.

    Example::

        # Construct a Gaussian copula from a multivariate normal.
        base_dist = MultivariateNormal(
            loc=torch.zeros(2),
            scale_tril=LKJCholesky(2).sample(),
        )
        transform = CumulativeDistributionTransform(Normal(0, 1))
        copula = TransformedDistribution(base_dist, [transform])
    """

    bijective = True
    codomain = constraints.unit_interval
    sign = +1

    def __init__(self, distribution: Distribution, cache_size: int = 0) -> None:
        # [FUNCTION DOC] Stores the distribution whose cdf defines the map.
        super().__init__(cache_size=cache_size)
        self.distribution = distribution

    @property
    def domain(self) -> constraints.Constraint | None:  # type: ignore[override]
        # [FUNCTION DOC] The distribution's support constraint.
        return self.distribution.support

    def _call(self, x):
        # [FUNCTION DOC] HELPER: The distribution's cdf at x.
        return self.distribution.cdf(x)

    def _inverse(self, y):
        # [FUNCTION DOC] HELPER: The distribution's inverse cdf at y.
        return self.distribution.icdf(y)

    def log_abs_det_jacobian(self, x, y):
        # [FUNCTION DOC] The distribution's log density at x (the cdf derivative).
        return self.distribution.log_prob(x)

    def with_cache(self, cache_size=1):
        # [FUNCTION DOC] Same object if cache size matches, else a new transform
        # over the same distribution with the given cache size.
        if self._cache_size == cache_size:
            return self
        return CumulativeDistributionTransform(self.distribution, cache_size=cache_size)
