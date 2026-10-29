# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines ExponentialFamily, an abstract Distribution subclass for exponential
# family members. It declares the hooks natural parameters, log normalizer and
# mean carrier measure, and provides a generic entropy computed with autograd via
# a Bregman-divergence formula. No import-time side effects.
# Exports: ExponentialFamily.

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution


__all__ = ["ExponentialFamily"]


class ExponentialFamily(Distribution):
    # [CLASS DOC]
    # Abstract intermediate base for exponential-family distributions. Subclasses
    # supply _natural_params (tuple of tensors), _log_normalizer (function of the
    # natural parameters) and _mean_carrier_measure (a number or property); this
    # class then offers a generic entropy and is used for analytic KL divergences.
    r"""
    ExponentialFamily is the abstract base class for probability distributions belonging to an
    exponential family, whose probability mass/density function is defined below

    .. math::

        p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))

    where :math:`\theta` denotes the natural parameters, :math:`t(x)` denotes the sufficient statistic,
    :math:`F(\theta)` is the log normalizer function for a given family and :math:`k(x)` is the carrier
    measure.

    Note:
        This class is an intermediary between the `Distribution` class and distributions which belong
        to an exponential family mainly to check the correctness of the `.entropy()` and analytic KL
        divergence methods. We use this class to compute the entropy and KL divergence using the AD
        framework and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and
        Cross-entropies of Exponential Families).
    """

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        # [FUNCTION DOC] HELPER: Abstract: subclasses return the tuple of natural parameter tensors. The base version raises NotImplementedError.
        """
        Abstract method for natural parameters. Returns a tuple of Tensors based
        on the distribution
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        # [FUNCTION DOC] HELPER: Abstract: subclasses return the log normalizer evaluated at the given natural parameter tensors. The base version raises NotImplementedError.
        """
        Abstract method for log normalizer function. Returns a log normalizer based on
        the distribution and input
        """
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self) -> float:
        # [FUNCTION DOC] HELPER: Abstract: subclasses provide the expected carrier measure (often as a plain class attribute overriding this property). The base version raises NotImplementedError.
        """
        Abstract method for expected carrier measure, which is required for computing
        entropy.
        """
        raise NotImplementedError

    def entropy(self):
        # [FUNCTION DOC] Generic entropy: starts from the negative mean carrier measure, adds the log normalizer, and subtracts the inner product of each detached natural parameter with the gradient of the summed log normalizer (obtained by autograd with graph creation), summed over non-batch dims. Returns a tensor of batch shape; does not modify the stored parameters.
        """
        Method to compute the entropy using Bregman divergence of the log normalizer.
        """
        result: Tensor | float = -self._mean_carrier_measure
        nparams = [p.detach().requires_grad_() for p in self._natural_params]
        lg_normal = self._log_normalizer(*nparams)
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        result += lg_normal
        for np, g in zip(nparams, gradients):
            result -= (np * g).reshape(self._batch_shape + (-1,)).sum(-1)
        return result
