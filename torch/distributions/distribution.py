# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines Distribution, the abstract base class of all probability distributions:
# shape bookkeeping (batch_shape, event_shape), constructor-time parameter
# validation against arg_constraints, sample-value validation against support,
# and the abstract interface (sample, rsample, log_prob, cdf, icdf, entropy, ...).
# Import-time side effect: the class attribute _validate_args is initialized from
# the interpreter's debug flag (validation off under optimized mode).
# Exports: Distribution.
import warnings
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.utils import lazy_property
from torch.types import _size


__all__ = ["Distribution"]


class Distribution:
    # [CLASS DOC]
    # Abstract base class for distributions. Instance state: _batch_shape and
    # _event_shape (set at construction, fixed afterwards) and a possibly
    # per-instance _validate_args flag shadowing the class-level default.
    # Class flags: has_rsample and has_enumerate_support (False by default).
    # Subclasses define arg_constraints, support and the sampling/density methods;
    # unimplemented members raise NotImplementedError.
    r"""
    Distribution is the abstract base class for probability distributions.

    Args:
        batch_shape (torch.Size): The shape over which parameters are batched.
        event_shape (torch.Size): The shape of a single sample (without batching).
        validate_args (bool, optional): Whether to validate arguments. Default: None.
    """

    has_rsample = False
    has_enumerate_support = False
    _validate_args = __debug__

    @staticmethod
    def set_default_validate_args(value: bool) -> None:
        # [FUNCTION DOC]
        # Sets the class-wide default for argument validation, affecting every
        # distribution that does not pass validate_args explicitly (global state).
        # Raises ValueError (with no message) if value is not equal to True or False.
        # Note that the membership test also accepts 0 and 1.
        """
        Sets whether validation is enabled or disabled.

        The default behavior mimics Python's ``assert`` statement: validation
        is on by default, but is disabled if Python is run in optimized mode
        (via ``python -O``). Validation may be expensive, so you may want to
        disable it once a model is working.

        Args:
            value (bool): Whether to enable validation.
        """
        if value not in [True, False]:
            raise ValueError
        Distribution._validate_args = value

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size(),
        event_shape: torch.Size = torch.Size(),
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Records batch_shape and event_shape. If validate_args is given, stores it on
        # the instance. When validation is enabled, checks each parameter named in
        # arg_constraints against its constraint: skips dependent constraints and
        # lazily computed parameters not yet materialized; warns (not errors) if the
        # subclass defines no arg_constraints. Raises ValueError listing the offending
        # parameter if any value violates its constraint.
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        if validate_args is not None:
            self._validate_args = validate_args
        if self._validate_args:
            try:
                arg_constraints = self.arg_constraints
            except NotImplementedError:
                arg_constraints = {}
                warnings.warn(
                    f"{self.__class__} does not define `arg_constraints`. "
                    + "Please set `arg_constraints = {}` or initialize the distribution "
                    + "with `validate_args=False` to turn off validation.",
                    stacklevel=2,
                )
            for param, constraint in arg_constraints.items():
                if constraints.is_dependent(constraint):
                    continue  # skip constraints that cannot be checked
                if param not in self.__dict__ and isinstance(
                    getattr(type(self), param), lazy_property
                ):
                    continue  # skip checking lazily-constructed args
                value = getattr(self, param)
                valid = constraint.check(value)
                if not torch._is_all_true(valid):
                    raise ValueError(
                        f"Expected parameter {param} "
                        f"({type(value).__name__} of shape {tuple(value.shape)}) "
                        f"of distribution {repr(self)} "
                        f"to satisfy the constraint {repr(constraint)}, "
                        f"but found invalid values:\n{value}"
                    )
        super().__init__()

    def expand(self, batch_shape: _size, _instance=None):
        # [FUNCTION DOC] Abstract: subclasses return a new instance with parameters expanded to batch_shape without validation. The base version raises NotImplementedError.
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_shape`.
        """
        raise NotImplementedError

    @property
    def batch_shape(self) -> torch.Size:
        # [FUNCTION DOC] Returns the shape over which parameters are batched.
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape

    @property
    def event_shape(self) -> torch.Size:
        # [FUNCTION DOC] Returns the shape of a single sample without batching.
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        # [FUNCTION DOC] Abstract: subclasses return a mapping from parameter name to its Constraint. The base version raises NotImplementedError.
        """
        Returns a dictionary from argument names to
        :class:`~torch.distributions.constraints.Constraint` objects that
        should be satisfied by each argument of this distribution. Args that
        are not tensors need not appear in this dict.
        """
        raise NotImplementedError

    @property
    def support(self) -> constraints.Constraint | None:
        # [FUNCTION DOC] Abstract: subclasses return the Constraint describing the support. The base version raises NotImplementedError.
        """
        Returns a :class:`~torch.distributions.constraints.Constraint` object
        representing this distribution's support.
        """
        raise NotImplementedError

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Abstract: subclasses return the mean. The base version raises NotImplementedError.
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Abstract: subclasses return the mode. The base version raises NotImplementedError naming the class.
        """
        Returns the mode of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement mode")

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Abstract: subclasses return the variance. The base version raises NotImplementedError.
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @property
    def stddev(self) -> Tensor:
        # [FUNCTION DOC] Standard deviation, the square root of variance (so it raises if variance is unimplemented).
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Default sampler: calls rsample with gradient tracking disabled and returns the result. Subclasses without rsample must override it; otherwise rsample raises NotImplementedError.
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Abstract: subclasses return reparameterized samples of shape sample_shape plus batch and event shape. The base version raises NotImplementedError.
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError

    @deprecated(
        "`sample_n(n)` will be deprecated. Use `sample((n,))` instead.",
        category=FutureWarning,
    )
    def sample_n(self, n: int) -> Tensor:
        # [FUNCTION DOC] Deprecated (emits a FutureWarning): draws n samples by calling sample with a one-element shape of n.
        """
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        """
        return self.sample(torch.Size((n,)))

    def log_prob(self, value: Tensor) -> Tensor:
        # [FUNCTION DOC] Abstract: subclasses return the log density or mass at value. The base version raises NotImplementedError.
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def cdf(self, value: Tensor) -> Tensor:
        # [FUNCTION DOC] Abstract: subclasses return the cumulative distribution at value. The base version raises NotImplementedError.
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def icdf(self, value: Tensor) -> Tensor:
        # [FUNCTION DOC] Abstract: subclasses return the inverse cumulative distribution at value. The base version raises NotImplementedError.
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def enumerate_support(self, expand: bool = True) -> Tensor:
        # [FUNCTION DOC] Abstract: discrete subclasses return all support values enumerated along dimension 0, optionally expanded over batch dims. The base version raises NotImplementedError.
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Returns:
            Tensor iterating over dimension 0.
        """
        raise NotImplementedError

    def entropy(self) -> Tensor:
        # [FUNCTION DOC] Abstract: subclasses return the entropy as a tensor of batch shape. The base version raises NotImplementedError.
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        raise NotImplementedError

    def perplexity(self) -> Tensor:
        # [FUNCTION DOC] Returns the exponential of the entropy, a tensor of batch shape (raises if entropy is unimplemented).
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        return torch.exp(self.entropy())

    def _extended_shape(self, sample_shape: _size = torch.Size()) -> torch.Size:
        # [FUNCTION DOC]
        # HELPER: Returns sample_shape (converted to a size object if needed) followed by
        # batch shape and event shape.
        # KNOWN ISSUE: the docstring says an empty result is upcast to (1,), but the code
        # performs no such upcast.
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (torch.Size): the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return torch.Size(sample_shape + self._batch_shape + self._event_shape)

    def _validate_sample(self, value: Tensor) -> None:
        # [FUNCTION DOC]
        # HELPER: Validates a value passed to log_prob, cdf or icdf. Checks it is a
        # tensor, that its trailing dims equal the event shape, that it broadcasts with
        # batch plus event shape, and that every element lies in the support. If the
        # subclass defines no support, only warns and returns. Raises ValueError on any
        # violation. Returns nothing.
        # KNOWN ISSUE: the tensor-type error message always mentions log_prob even when
        # called from cdf or icdf.
        """
        Argument validation for distribution methods such as `log_prob`,
        `cdf` and `icdf`. The rightmost dimensions of a value to be
        scored via these methods must agree with the distribution's batch
        and event shapes.

        Args:
            value (Tensor): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
        if not isinstance(value, torch.Tensor):
            raise ValueError("The value argument to log_prob must be a Tensor")

        event_dim_start = len(value.size()) - len(self._event_shape)
        if value.size()[event_dim_start:] != self._event_shape:
            raise ValueError(
                f"The right-most size of value must match event_shape: {value.size()} vs {self._event_shape}."
            )

        actual_shape = value.size()
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError(
                    f"Value is not broadcastable with batch_shape+event_shape: {actual_shape} vs {expected_shape}."
                )
        try:
            support = self.support
        except NotImplementedError:
            warnings.warn(
                f"{self.__class__} does not define `support` to enable "
                + "sample validation. Please initialize the distribution with "
                + "`validate_args=False` to turn off validation.",
                stacklevel=2,
            )
            return
        if support is None:
            raise AssertionError("support is unexpectedly None")
        valid = support.check(value)
        if not torch._is_all_true(valid):
            raise ValueError(
                "Expected value argument "
                f"({type(value).__name__} of shape {tuple(value.shape)}) "
                f"to be within the support ({repr(support)}) "
                f"of the distribution {repr(self)}, "
                f"but found invalid values:\n{value}"
            )

    def _get_checked_instance(self, cls, _instance=None):
        # [FUNCTION DOC] HELPER: Used by expand overrides. If no instance is supplied, returns a fresh uninitialized object of this instance's type; raises NotImplementedError if this type defines its own initializer different from cls's (it must then override expand itself). If an instance is supplied, returns it unchanged.
        if _instance is None and type(self).__init__ != cls.__init__:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} of {cls.__name__} that defines a custom __init__ method "
                "must also define a custom .expand() method."
            )
        return self.__new__(type(self)) if _instance is None else _instance

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the class name followed by parameters listed in arg_constraints that are present in the instance attributes; each shown by value if single-element, else by size. Assumes those attributes are tensors; raises NotImplementedError if arg_constraints is undefined.
        param_names = [k for k in self.arg_constraints if k in self.__dict__]
        args_string = ", ".join(
            [
                f"{p}: {self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].size()}"
                for p in param_names
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"
