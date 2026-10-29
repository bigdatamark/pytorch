# [MODULE DOC] Test for DTensorTestBase utilities. Defines a CPU/gloo test class that builds a
# 3-D device mesh (data=2, fsdp=3, tensor=5, so 30 ranks) and checks that process-group setup and
# teardown via with_comms completes. No import-time side effects; run_tests when run as a script.
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import numpy as np

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class DTensorTestBaseUtilCPUTest(DTensorTestBase):
    # [CLASS DOC] CPU test of DTensorTestBase: overrides backend (gloo), device_type (cpu),
    # world_size (product of mesh_dim_sizes) and mesh_dim_sizes, and provides build_device_mesh.
    """
    This class tests if the basic functionalities of DTensorTestBase are
    working as expected on CPU, regardless of the presence of CUDA devices.
    """

    @property
    def backend(self):
        # [FUNCTION DOC] HELPER: Returns the process group backend name, always gloo.
        return "gloo"

    @property
    def device_type(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the device type string, always cpu.
        return "cpu"

    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Returns the total number of ranks, the product of all mesh
        # dimension sizes (30), as a Python int.
        return np.prod(list(self.mesh_dim_sizes.values())).item()

    @property
    def mesh_dim_sizes(self) -> dict[str, int]:
        # [FUNCTION DOC] HELPER: Returns the ordered mapping of mesh dimension name to size:
        # data 2, fsdp 3, tensor 5.
        """Mapping from mesh dimension names to sizes."""
        return {"data": 2, "fsdp": 3, "tensor": 5}

    def build_device_mesh(self) -> DeviceMesh:
        # [FUNCTION DOC] HELPER: Creates and returns an N-D DeviceMesh on the test device type,
        # with shape and dimension names taken from mesh_dim_sizes. Requires an initialized group.
        return init_device_mesh(
            self.device_type,
            mesh_shape=tuple(self.mesh_dim_sizes.values()),
            mesh_dim_names=tuple(self.mesh_dim_sizes.keys()),
        )

    @with_comms
    def test_dtensor_testbase_destroy_pg(self):
        # [FUNCTION DOC] Builds the device mesh under with_comms and returns; passes if the
        # process group init and the destroy at the end of the test finish without error.
        # This tests destroy_pg() correctly finishes.
        device_mesh = self.build_device_mesh()  # noqa: F841


if __name__ == "__main__":
    run_tests()
