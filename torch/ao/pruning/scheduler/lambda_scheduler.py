# [MODULE DOC] Defines LambdaSL, a scheduler that sets each group's sparsity level to
# its base level times a user-supplied function of the epoch. Exports LambdaSL.
import warnings
from collections.abc import Callable

from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from .base_scheduler import BaseScheduler


__all__ = ["LambdaSL"]


class LambdaSL(BaseScheduler):
    # [CLASS DOC] Scheduler holding one multiplier function per sparsifier group
    # (sl_lambdas); each step sets a group's level to its base level times its function
    # of the current epoch.
    """Sets the sparsity level of each parameter group to the final sl
    times a given function. When last_epoch=-1, sets initial sl as zero.
    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        sl_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in sparsifier.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming sparsifier has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95**epoch
        >>> # xdoctest: +SKIP
        >>> scheduler = LambdaSL(sparsifier, sl_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        sparsifier: BaseSparsifier,
        sl_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        # [FUNCTION DOC] Stores the sparsifier and builds sl_lambdas: a single callable is
        # repeated for every group; a list or tuple must match the group count (ValueError
        # otherwise). Then runs the base constructor, which performs the first step.
        self.sparsifier = sparsifier

        if not isinstance(sl_lambda, list) and not isinstance(sl_lambda, tuple):
            self.sl_lambdas = [sl_lambda] * len(sparsifier.groups)
        else:
            if len(sl_lambda) != len(sparsifier.groups):
                raise ValueError(
                    f"Expected {len(sparsifier.groups)} lr_lambdas, but got {len(sl_lambda)}"
                )
            self.sl_lambdas = list(sl_lambda)
        super().__init__(sparsifier, last_epoch, verbose)  # type: ignore[no-untyped-call]

    def get_sl(self) -> list[float]:
        # [FUNCTION DOC] Returns for each group its base level multiplied by its lambda
        # evaluated at the current epoch. Warns if called outside of step.
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.",
                stacklevel=2,
            )
        return [
            base_sl * lmbda(self.last_epoch)
            for lmbda, base_sl in zip(self.sl_lambdas, self.base_sl)
        ]
