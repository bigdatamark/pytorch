# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Device-generic tests of the default backend mapping and basic process
# group creation in torch.distributed. Instantiates TestMiscCollectiveUtils for cpu,
# cuda, mps and xpu at import time via instantiate_device_type_tests.

import os

import torch.distributed as dist
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


"""
common backend API tests
"""


class TestMiscCollectiveUtils(TestCase):
    # [CLASS DOC] Device-parametrized tests for dist.get_default_backend_for_device and
    # creating a single-rank default process group on that backend.
    def test_device_to_backend_mapping(self, device) -> None:
        # [FUNCTION DOC] Checks the default backend per device: nccl for cuda, gloo for cpu and
        # mps, xccl for xpu; any other device must raise ValueError.
        """
        Test device to backend mapping
        """
        if "cuda" in device:
            if dist.get_default_backend_for_device(device) != "nccl":
                raise AssertionError(
                    f"Expected nccl, got {dist.get_default_backend_for_device(device)}"
                )
        elif "cpu" in device:
            if dist.get_default_backend_for_device(device) != "gloo":
                raise AssertionError(
                    f"Expected gloo, got {dist.get_default_backend_for_device(device)}"
                )
        elif "mps" in device:
            if dist.get_default_backend_for_device(device) != "gloo":
                raise AssertionError(
                    f"Expected gloo, got {dist.get_default_backend_for_device(device)}"
                )
        elif "xpu" in device:
            if dist.get_default_backend_for_device(device) != "xccl":
                raise AssertionError(
                    f"Expected xccl, got {dist.get_default_backend_for_device(device)}"
                )
        else:
            with self.assertRaises(ValueError):
                dist.get_default_backend_for_device(device)

    def test_create_pg(self, device) -> None:
        # [FUNCTION DOC] Sets MASTER_ADDR and MASTER_PORT (29500) in os.environ, initializes a
        # world-size-1 group via env:// with the default backend, checks the default group's
        # backend name matches, then destroys the process group.
        """
        Test create process group
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        backend = dist.get_default_backend_for_device(device)
        dist.init_process_group(
            backend=backend, rank=0, world_size=1, init_method="env://"
        )
        pg = dist.distributed_c10d._get_default_group()
        backend_pg = pg._get_backend_name()
        if backend_pg != backend:
            raise AssertionError(f"Expected {backend}, got {backend_pg}")
        dist.destroy_process_group()


devices = ["cpu", "cuda", "mps", "xpu"]
instantiate_device_type_tests(
    TestMiscCollectiveUtils, globals(), only_for=devices, allow_mps=True, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
