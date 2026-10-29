# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Fisher-Snedecor (F) distribution parameterized by two positive
# degrees of freedom, sampled as a ratio of two Gamma variables. Imports Gamma.
# No import-time side effects. Exports: FisherSnedecor.

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["FisherSnedecor"]


class FisherSnedecor(Distribution):
    # [CLASS DOC]
    # F distribution on positive reals. State: df1, df2 (broadcast positive
    # tensors) and two private Gamma distributions used for reparameterized
    # sampling. Moments are NaN where undefined.
    r"""
    Creates a Fisher-Snedecor distribution parameterized by :attr:`df1` and :attr:`df2`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
        tensor([ 0.2453])

    Args:
        df1 (float or Tensor): degrees of freedom parameter 1
        df2 (float or Tensor): degrees of freedom parameter 2
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"df1": constraints.positive, "df2": constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(
        self,
        df1: Tensor | float,
        df2: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Broadcasts df1 and df2 into tensors and builds two Gamma helpers (shape df/2, rate df). Batch shape is empty if both inputs are Python numbers, else the broadcast shape. Gamma construction or base validation may raise ValueError for nonpositive values.
        self.df1, self.df2 = broadcast_all(df1, df2)
        self._gamma1 = Gamma(self.df1 * 0.5, self.df1)
        self._gamma2 = Gamma(self.df2 * 0.5, self.df2)

        if isinstance(df1, _Number) and isinstance(df2, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.df1.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new FisherSnedecor with df1, df2 and both Gamma helpers expanded to batch_shape, preserving the validation setting; self is unchanged.
        new = self._get_checked_instance(FisherSnedecor, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df1 = self.df1.expand(batch_shape)
        new.df2 = self.df2.expand(batch_shape)
        new._gamma1 = self._gamma1.expand(batch_shape)
        new._gamma2 = self._gamma2.expand(batch_shape)
        super(FisherSnedecor, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Mean: df2 divided by df2 minus two; NaN where df2 is at most 2 (undefined).
        df2 = self.df2.clone(memory_format=torch.contiguous_format)
        df2[df2 <= 2] = nan
        return df2 / (df2 - 2)

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Mode: (df1 minus 2) over df1 times df2 over (df2 plus 2); NaN where df1 is at most 2.
        mode = (self.df1 - 2) / self.df1 * self.df2 / (self.df2 + 2)
        mode[self.df1 <= 2] = nan
        return mode

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Variance by the standard F formula; NaN where df2 is at most 4 (undefined).
        df2 = self.df2.clone(memory_format=torch.contiguous_format)
        df2[df2 <= 4] = nan
        return (
            2
            * df2.pow(2)
            * (self.df1 + df2 - 2)
            / (self.df1 * (df2 - 2).pow(2) * (df2 - 4))
        )

    def rsample(self, sample_shape: _size = torch.Size(())) -> Tensor:
        # [FUNCTION DOC] Draws reparameterized samples as the ratio of two independent Gamma samples (consumes random state). The denominator and the result are clamped from below at the smallest normal float of the dtype to avoid zero or division blowups; results are reshaped to the extended shape.
        shape = self._extended_shape(sample_shape)
        #   X1 ~ Gamma(df1 / 2, 1 / df1), X2 ~ Gamma(df2 / 2, 1 / df2)
        #   Y = df2 * df1 * X1 / (df1 * df2 * X2) = X1 / X2 ~ F(df1, df2)
        X1 = self._gamma1.rsample(sample_shape).view(shape)
        X2 = self._gamma2.rsample(sample_shape).view(shape)
        tiny = torch.finfo(X2.dtype).tiny
        X2.clamp_(min=tiny)
        Y = X1 / X2
        Y.clamp_(min=tiny)
        return Y

    def log_prob(self, value):
        # [FUNCTION DOC] Log density of the F distribution at value, via log-gamma terms of df1/2 and df2/2 and the ratio df1/df2. Validates value if validation is on (ValueError if not positive).
        if self._validate_args:
            self._validate_sample(value)
        ct1 = self.df1 * 0.5
        ct2 = self.df2 * 0.5
        ct3 = self.df1 / self.df2
        t1 = (ct1 + ct2).lgamma() - ct1.lgamma() - ct2.lgamma()
        t2 = ct1 * ct3.log() + (ct1 - 1) * torch.log(value)
        t3 = (ct1 + ct2) * torch.log1p(ct3 * value)
        return t1 + t2 - t3
