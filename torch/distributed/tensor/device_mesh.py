# [MODULE DOC] Compatibility re-export of the device mesh API under the DTensor package.
# Re-exports DeviceMesh and init_device_mesh (the public names listed in the export list)
# plus the private _get_device_handle and _mesh_resources from torch.distributed.device_mesh.
# No import-time side effects beyond those imports.
from torch.distributed.device_mesh import (  # noqa: F401
    _get_device_handle,
    _mesh_resources,
    DeviceMesh,
    init_device_mesh,
)


__all__ = ["init_device_mesh", "DeviceMesh"]
