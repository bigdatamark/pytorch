# [PACKAGE DOC] Deprecated backward-compat alias for
# torch.distributed._shard.sharding_spec: re-exports its public names, emits a
# DeprecationWarning at import, and replaces its own sys.modules entry with the
# new package.
# Keep old package for BC purposes, this file should be removed once
# everything moves to the `torch.distributed._shard` package.
import sys
import warnings

import torch
from torch.distributed._shard.sharding_spec import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`torch.distributed._sharding_spec` will be deprecated, "
        "use `torch.distributed._shard.sharding_spec` instead",
        DeprecationWarning,
        stacklevel=2,
    )

import torch.distributed._shard.sharding_spec as _sharding_spec


sys.modules["torch.distributed._sharding_spec"] = _sharding_spec
