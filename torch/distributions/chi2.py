# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Chi2 distribution, a thin subclass of Gamma parameterized by
# degrees of freedom. Exports Chi2 via __all__. No import-time side effects.

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.gamma import Gamma


__all__ = ["Chi2"]


class Chi2(Gamma):
    # [CLASS DOC]
    # Chi-squared distribution implemented as a Gamma with concentration equal to
    # half the degrees of freedom and rate fixed at one half. All sampling and
    # density behavior is inherited from Gamma; only the parameter constraint
    # (df must be positive) and the df accessor are specific to this class.
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
    This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Chi2(torch.tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
        tensor([ 0.1046])

    Args:
        df (float or Tensor): shape parameter of the distribution
    """

    arg_constraints = {"df": constraints.positive}

    def __init__(
        self,
        df: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the distribution from df (float or Tensor) by initializing the
        # Gamma parent with concentration half of df and rate one half.
        # validate_args optionally toggles argument validation. Returns nothing.
        super().__init__(0.5 * df, 0.5, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC]
        # Returns a new Chi2 whose batch shape is expanded to batch_shape. Gets a
        # checked instance of Chi2 (or reuses _instance) and delegates the actual
        # parameter expansion to Gamma.expand. Does not mutate self.
        new = self._get_checked_instance(Chi2, _instance)
        return super().expand(batch_shape, new)

    @property
    def df(self) -> Tensor:
        # [FUNCTION DOC]
        # Property returning the degrees of freedom as a Tensor, recovered as
        # twice the underlying Gamma concentration.
        return self.concentration * 2
