# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for DTensor integration with torch_xla (SPMD sharding on XLA devices).
# Defines a with_xla decorator that enables XLA SPMD and skips tests when torch_xla is missing,
# and DTensorXLAIntegrationTest, which checks that distribute_tensor on an "xla" DeviceMesh
# returns an XLAShardedTensor with the expected global and local shard sizes, plus a
# distribute_module sharding check. No import-time side effects beyond imports;
# run_tests is invoked only when executed as a script.

import os
import unittest
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

import torch
from torch import nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# wrapper to check xla test requirements
def with_xla(func: Callable) -> Callable:
    # [FUNCTION DOC] Decorator factory for test methods that need torch_xla. Raises
    # AssertionError if func is None. Returns a wrapper that sets the XLA_USE_SPMD environment
    # variable to "1", tries to import torch_xla (raising unittest.SkipTest if missing), sets
    # the test instance's device_type to "xla", runs the test, then resets XLA_USE_SPMD to "0".
    # KNOWN ISSUE: the reset to "0" is not in a finally block, so it is skipped if the test
    # raises or the import fails, leaving SPMD enabled in the process environment.
    if func is None:
        raise AssertionError("Expected func to not be None")

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self,
        *args: tuple[object],
        **kwargs: dict[str, Any],  # type: ignore[misc]
    ) -> None:
        # [FUNCTION DOC] HELPER: Wrapper produced by with_xla. Enables XLA SPMD through the
        # environment, skips the test if torch_xla cannot be imported, marks self.device_type
        # as "xla", calls the wrapped test with the original arguments, then disables SPMD.
        # TODO(yeounoh) replace this with xr.use_spmd() when we deprecate the flag.
        os.environ["XLA_USE_SPMD"] = "1"
        try:
            import torch_xla  # type:ignore[import]  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("torch_xla is not installed.") from exc
        self.device_type = "xla"
        func(self, *args, **kwargs)  # type: ignore[misc]
        os.environ["XLA_USE_SPMD"] = "0"

    return wrapper


class DTensorXLAIntegrationTest(TestCase):
    # [CLASS DOC] Test case for DTensor APIs on XLA devices. Each test is wrapped by with_xla,
    # so it is skipped when torch_xla is absent. Holds a nested SimpleLinear model fixture
    # used by the distribute_module test. Has no instance state of its own.
    class SimpleLinear(nn.Module):
        # [CLASS DOC] HELPER: Small two-layer MLP fixture (Linear 128 to 64, ReLU, Linear
        # 64 to 1) with attributes fc1, relu and fc2.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Builds the layers fc1 (128 to 64), relu and fc2 (64 to 1).
            super(DTensorXLAIntegrationTest.SimpleLinear, self).__init__()
            self.fc1 = nn.Linear(128, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies fc1, ReLU, then fc2 to x and returns the result.
            y = self.relu(self.fc1(x))
            z = self.fc2(y)
            return z

    @with_xla
    def test_xla_distribute_tensor_1d_shard(self):
        # [FUNCTION DOC] Builds a 1D xla DeviceMesh over all XLA devices and shards a random
        # tensor on dim 0, with and without requires_grad. Does nothing when only one device
        # exists. Asserts the result is an XLAShardedTensor, its global_tensor has the full
        # shape, the first local shard has 3 rows, and with requires_grad the global tensor
        # requires grad and the result is a leaf.
        import torch_xla.runtime as xr  # type:ignore[import]

        device_count = xr.global_runtime_device_count()
        if device_count > 1:
            device_mesh = DeviceMesh("xla", list(range(device_count)))
            shard_spec = [Shard(0)]

            for requires_grad in [True, False]:
                tensor_to_shard = torch.randn(
                    3 * device_count, 3, requires_grad=requires_grad
                )
                dist_tensor = distribute_tensor(
                    tensor_to_shard, device_mesh, shard_spec
                )
                # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
                if type(dist_tensor).__name__ != "XLAShardedTensor":
                    raise AssertionError(
                        f"Expected XLAShardedTensor, got {type(dist_tensor).__name__}"
                    )
                global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
                self.assertEqual(
                    global_tensor.size(), torch.Size([3 * device_count, 3])
                )
                local_tensor = dist_tensor.local_shards[0].data
                self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
                if requires_grad:
                    self.assertTrue(dist_tensor.global_tensor.requires_grad)
                    self.assertTrue(dist_tensor.is_leaf)

    @with_xla
    def test_xla_distribute_tensor_1d_replicate(self):
        # [FUNCTION DOC] Distributes a random tensor with a Replicate placement on a 1D xla
        # mesh (runs even with one device), with and without requires_grad. Asserts an
        # XLAShardedTensor result whose global tensor and first local shard both have the full
        # shape, and the requires_grad and leaf properties when gradients are requested.
        import torch_xla.runtime as xr  # type:ignore[import]

        device_count = xr.global_runtime_device_count()
        device_mesh = DeviceMesh("xla", list(range(device_count)))
        shard_spec = [Replicate()]

        for requires_grad in [True, False]:
            tensor_to_shard = torch.randn(
                3 * device_count, 3, requires_grad=requires_grad
            )
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
            if type(dist_tensor).__name__ != "XLAShardedTensor":
                raise AssertionError(
                    f"Expected XLAShardedTensor, got {type(dist_tensor).__name__}"
                )
            global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
            self.assertEqual(global_tensor.size(), torch.Size([3 * device_count, 3]))
            local_tensor = dist_tensor.local_shards[0].data
            self.assertEqual(local_tensor.size(), torch.Size([3 * device_count, 3]))
            if requires_grad:
                self.assertTrue(dist_tensor.global_tensor.requires_grad)
                self.assertTrue(dist_tensor.is_leaf)

    @with_xla
    def test_xla_distribute_tensor_2d(self):
        # [FUNCTION DOC] Uses a 2 by (device_count/2) xla mesh with placements Replicate and
        # Shard(0). Skipped silently with one device. Asserts an XLAShardedTensor whose global
        # tensor has 3*device_count/2 rows and whose first local shard is 3 by 3, plus
        # requires_grad and leaf checks for the requires_grad case.
        import torch_xla.runtime as xr  # type:ignore[import]

        device_count = xr.global_runtime_device_count()
        if device_count > 1:
            device_mesh = DeviceMesh(
                "xla", np.array(range(device_count)).reshape(2, device_count // 2)
            )
            shard_spec = [Replicate(), Shard(0)]

            for requires_grad in [True, False]:
                tensor_to_shard = torch.randn(
                    3 * device_count // 2, 3, requires_grad=requires_grad
                )
                dist_tensor = distribute_tensor(
                    tensor_to_shard, device_mesh, shard_spec
                )
                # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
                if type(dist_tensor).__name__ != "XLAShardedTensor":
                    raise AssertionError(
                        f"Expected XLAShardedTensor, got {type(dist_tensor).__name__}"
                    )
                global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
                self.assertEqual(
                    global_tensor.size(), torch.Size([3 * device_count // 2, 3])
                )
                local_tensor = dist_tensor.local_shards[0].data
                self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
                if requires_grad:
                    self.assertTrue(dist_tensor.global_tensor.requires_grad)
                    self.assertTrue(dist_tensor.is_leaf)

    @with_xla
    def text_xla_distribute_module(self):
        # [FUNCTION DOC] Moves a SimpleLinear model to the XLA device and calls
        # distribute_module with a partition function that shards every Linear parameter on
        # dim 0, then asserts fc1.weight and fc2.weight have a non-empty XLA sharding spec.
        # KNOWN ISSUE: the name starts with "text" rather than "test", so the test runner never
        # collects it. Also the partition function discards distribute_tensor results, relying
        # on XLA annotating the parameters in place.
        import torch_xla  # type:ignore[import]
        import torch_xla.core.xla_model as xm  # type:ignore[import]
        import torch_xla.runtime as xr  # type:ignore[import]

        model = self.SimpleLinear().to(xm.xla_device())

        device_count = xr.global_runtime_device_count()
        device_mesh = DeviceMesh("xla", list(range(device_count)))

        def shard_params(mod_name, mod, mesh):
            # [FUNCTION DOC] HELPER: Partition function for distribute_module. For Linear
            # modules, calls distribute_tensor with Shard(0) on each parameter for its XLA
            # sharding annotation side effect; other modules are left untouched.
            shard_spec = [Shard(0)]
            # annotate fc1 and fc2
            if isinstance(mod, nn.Linear):
                for _, param in mod.named_parameters():
                    # annotate the parameter tensors directly
                    distribute_tensor(param, mesh, shard_spec)

        sharded_model = distribute_module(model, device_mesh, shard_params)
        self.assertTrue(
            torch_xla._XLAC._get_xla_sharding_spec(sharded_model.fc1.weight) != ""
        )
        self.assertTrue(
            torch_xla._XLAC._get_xla_sharding_spec(sharded_model.fc2.weight) != ""
        )


if __name__ == "__main__":
    run_tests()
