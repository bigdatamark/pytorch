# [MODULE DOC] Utilities that strip running-statistics tracking from batch norm modules so they
# behave functionally under torch.func transforms (e.g. vmap). Exposes
# replace_all_batch_norm_modules_ in torch.func. No import-time side effects beyond decorator registration.
import torch.nn as nn
from torch._functorch.utils import exposed_in


def batch_norm_without_running_stats(module: nn.Module) -> None:
    # [FUNCTION DOC] If module is a batch norm module that tracks running stats, mutate it in place:
    # set running_mean, running_var and num_batches_tracked to None and track_running_stats to False.
    # Other modules (or batch norms not tracking stats) are left untouched. Returns None.
    if (
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ):
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None
        module.track_running_stats = False


@exposed_in("torch.func")
def replace_all_batch_norm_modules_(root: nn.Module) -> nn.Module:
    # [FUNCTION DOC] Public torch.func API. Applies the running-stats removal to root and every
    # submodule (root.modules() already includes root, so the explicit root call is redundant).
    # Mutates modules in place and returns root. Never raises on non-batch-norm modules.
    """
    In place updates :attr:`root` by setting the ``running_mean`` and ``running_var`` to be None and
    setting track_running_stats to be False for any nn.BatchNorm module in :attr:`root`
    """
    # base case
    batch_norm_without_running_stats(root)

    for obj in root.modules():
        batch_norm_without_running_stats(obj)
    return root
