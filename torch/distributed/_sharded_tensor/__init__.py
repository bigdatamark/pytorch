# [PACKAGE DOC] Deprecated backward-compat alias for
# torch.distributed._shard.sharded_tensor: re-exports its public names, emits a
# DeprecationWarning at import, and replaces its own sys.modules entry with the
# new package.
# Keep old package for BC purposes, this file should be removed once
# everything moves to the `torch.distributed._shard` package.
import sys
import warnings

import torch
from torch.distributed._shard.sharded_tensor import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`torch.distributed._sharded_tensor` will be deprecated, "
        "use `torch.distributed._shard.sharded_tensor` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules["torch.distributed._sharded_tensor"] = (
    torch.distributed._shard.sharded_tensor
)
