# [MODULE DOC] HELPER: Provides a single function that emits the deprecation warning
# for TorchScript support of functional optimizers.
import warnings

import torch


@torch.jit.ignore  # type: ignore[misc]
def _scripted_functional_optimizer_deprecation_warning(stacklevel: int = 0) -> None:
    # [FUNCTION DOC] HELPER: Emits a DeprecationWarning saying TorchScript support for
    # functional optimizers is deprecated, forcing the warning to
    # always show (filters are reset locally). The reported stack level
    # is the given stacklevel plus two. Ignored by TorchScript
    # compilation. Returns nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            "`TorchScript` support for functional optimizers is deprecated "
            "and will be removed in a future PyTorch release. "
            "Consider using the `torch.compile` optimizer instead.",
            DeprecationWarning,
            stacklevel=stacklevel + 2,
        )
