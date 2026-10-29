# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tutorial example 6: multi-dimensional device meshes under LocalTensorMode
# (2D data/tensor parallel, hybrid parallel matmul, 3D mesh). Each function returns actual
# and expected values. As a script it initializes a fake group of world size 24.

"""
Example 6: Multi-Dimensional Meshes
====================================

This module contains core patterns for multi-dimensional meshes.
The functions below are designed to be:
1. Included in the tutorial via literalinclude
2. Directly tested to ensure tutorial correctness

Each function returns a tuple of (actual, expected) for testing.
"""

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.tensor import (
    distribute_tensor,
    init_device_mesh,
    Replicate,
    Shard,
)


# [core_2d_mesh]
def create_2d_mesh():
    # [FUNCTION DOC] Creates a 4x2 CPU mesh named dp and tp over 8 simulated ranks. Returns
    # ((mesh shape, mesh dim names, mesh size), ((4, 2), (dp, tp), 8)).
    """Create a 2D mesh for hybrid parallelism.

    Returns: ((shape, dim_names, total_size), (expected_shape, expected_names, expected_size))
    """
    world_size = 8
    dp_size, tp_size = 4, 2

    with LocalTensorMode(world_size):
        mesh = init_device_mesh("cpu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        shape = mesh.shape
        dim_names = mesh.mesh_dim_names
        total_size = mesh.size()

    expected = ((dp_size, tp_size), ("dp", "tp"), world_size)
    return (shape, dim_names, total_size), expected


# [end_core_2d_mesh]


# [core_hybrid_parallel]
def hybrid_parallelism():
    # [FUNCTION DOC] On a 4x2 mesh, shards a 16x8 matrix on rows over dp and replicates over tp,
    # shards an 8x12 matrix on columns over tp and replicates over dp, multiplies them, and
    # gathers the result. Returns (gathered result, plain matmul). Random inputs.
    """Combine data parallel and tensor parallel.

    Returns: (actual, expected)
    """
    world_size = 8
    dp_size, tp_size = 4, 2

    with LocalTensorMode(world_size):
        mesh = init_device_mesh("cpu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

        x = torch.randn(16, 8)
        dx = distribute_tensor(x, mesh, [Shard(0), Replicate()])

        w = torch.randn(8, 12)
        dw = distribute_tensor(w, mesh, [Replicate(), Shard(1)])

        dy = dx @ dw

        expected = x @ w
        actual = dy.full_tensor().reconcile()

    return actual, expected


# [end_core_hybrid_parallel]


# [core_3d_mesh]
def create_3d_mesh():
    # [FUNCTION DOC] On a 2x3x4 mesh (pp, dp, tp; 24 ranks) distributes an 8x16x32 random tensor
    # replicated over pp, sharded on dim 0 over dp and on dim 2 over tp, then gathers it. Returns
    # (gathered result, original tensor).
    """Create a 3D mesh for DP + TP + PP.

    Returns: (actual, expected)
    """
    world_size = 24
    pp_size, dp_size, tp_size = 2, 3, 4

    with LocalTensorMode(world_size):
        mesh = init_device_mesh(
            "cpu",
            (pp_size, dp_size, tp_size),
            mesh_dim_names=("pp", "dp", "tp"),
        )

        tensor = torch.randn(8, 16, 32)
        dt = distribute_tensor(tensor, mesh, [Replicate(), Shard(0), Shard(2)])

        actual = dt.full_tensor().reconcile()

    return actual, tensor


# [end_core_3d_mesh]


if __name__ == "__main__":
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group("fake", rank=0, world_size=24)

    try:
        print("=== create_2d_mesh ===")
        (shape, names, size), (exp_shape, exp_names, exp_size) = create_2d_mesh()
        print(
            f"Shape: {shape}={exp_shape}, Names: {names}={exp_names}, Size: {size}={exp_size}"
        )

        print("\n=== hybrid_parallelism ===")
        actual, expected = hybrid_parallelism()
        print(f"Matches: {torch.allclose(actual, expected, atol=1e-5)}")

        print("\n=== create_3d_mesh ===")
        actual, expected = create_3d_mesh()
        print(f"Matches: {torch.equal(actual, expected)}")
    finally:
        dist.destroy_process_group()
