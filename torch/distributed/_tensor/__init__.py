# [PACKAGE DOC] Backward-compatibility shim for the old torch.distributed._tensor
# path, which moved to torch.distributed.tensor. At import time it imports the
# submodules _shards_wrapper, _utils, experimental and device_mesh from the new
# location and registers them in the module table under the old names, then re-exports
# the public DTensor API (DeviceMesh, DTensor, Partial, Placement, Replicate, Shard,
# distribute_module, distribute_tensor, init_device_mesh, empty, full, ones, rand,
# randn, zeros).
"""
NOTICE: DTensor has moved to torch.distributed.tensor

This file is a shim to redirect to the new location, and
we keep the old import path starts with `_tensor` for
backward compatibility. We will remove this folder once
we resolve all the BC issues.
"""

import sys
from importlib import import_module


submodules = [
    # TODO: _shards_wrapper/_utils here mainly for checkpoint BC, remove them
    "_shards_wrapper",
    "_utils",
    "experimental",
    "device_mesh",
]

# Redirect imports
for submodule in submodules:
    full_module_name = f"torch.distributed.tensor.{submodule}"
    sys.modules[f"torch.distributed._tensor.{submodule}"] = import_module(
        full_module_name
    )

from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    empty,
    full,
    init_device_mesh,
    ones,
    Partial,
    Placement,
    rand,
    randn,
    Replicate,
    Shard,
    zeros,
)
