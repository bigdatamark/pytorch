# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tutorial example 5: computations that differ per rank, using rank_map,
# tensor_map, mode.disable, maybe_disable_local_tensor_mode and the maybe_run_for_local_tensor
# decorator. Each function returns actual and expected values; as a script it prints them.
# No import-time side effects.

"""
Example 5: Rank-Specific Computations
======================================

This module contains core patterns for rank-specific operations.
The functions below are designed to be:
1. Included in the tutorial via literalinclude
2. Directly tested to ensure tutorial correctness

Each function returns a tuple of (actual, expected) for testing.
"""

import torch
from torch.distributed._local_tensor import (
    LocalTensorMode,
    maybe_disable_local_tensor_mode,
    maybe_run_for_local_tensor,
    rank_map,
    tensor_map,
)


# [core_rank_map]
def use_rank_map(world_size: int = 4):
    # [FUNCTION DOC] Uses rank_map inside LocalTensorMode to build a LocalTensor where each rank
    # holds a 2x3 tensor filled with its rank number. Returns (mapping of rank to the first
    # element, mapping of rank to its float rank).
    """Create LocalTensors with per-rank values using rank_map.

    Returns: (values_dict, expected_dict)
    """
    with LocalTensorMode(world_size):
        lt = rank_map(lambda rank: torch.full((2, 3), float(rank)))
        values = {
            rank: lt._local_tensors[rank][0, 0].item() for rank in range(world_size)
        }

    expected = {rank: float(rank) for rank in range(world_size)}
    return values, expected


# [end_core_rank_map]


# [core_tensor_map]
def use_tensor_map(world_size: int = 4):
    # [FUNCTION DOC] Builds per-rank tensors of value rank+1, then applies tensor_map with a
    # function scaling each shard by rank+1. Returns (rank to first element, rank to the square
    # of rank+1).
    """Transform each shard differently using tensor_map.

    Returns: (values_dict, expected_dict)
    """
    with LocalTensorMode(world_size):
        lt = rank_map(lambda rank: torch.ones(2, 2) * (rank + 1))

        def scale_by_rank(rank: int, tensor: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: returns the shard multiplied by rank plus one.
            return tensor * (rank + 1)

        scaled = tensor_map(lt, scale_by_rank)
        values = {
            rank: scaled._local_tensors[rank][0, 0].item() for rank in range(world_size)
        }

    # (rank + 1) * (rank + 1) = (rank + 1)^2
    expected = {rank: float((rank + 1) ** 2) for rank in range(world_size)}
    return values, expected


# [end_core_tensor_map]


# [core_disable_mode]
def disable_mode_temporarily(world_size: int = 4):
    # [FUNCTION DOC] Records the type name of a ones tensor created inside LocalTensorMode, then
    # the type name of one created within mode.disable(). Returns ((inside, disabled),
    # (LocalTensor, Tensor)).
    """Temporarily exit LocalTensorMode for regular tensor ops.

    Returns: ((inside_type, disabled_type), (expected_inside, expected_disabled))
    """
    with LocalTensorMode(world_size) as mode:
        lt = torch.ones(2, 2)
        inside_type = type(lt).__name__

        with mode.disable():
            regular = torch.ones(2, 2)
            disabled_type = type(regular).__name__

    return (inside_type, disabled_type), ("LocalTensor", "Tensor")


# [end_core_disable_mode]


# [core_maybe_disable]
def use_maybe_disable():
    # [FUNCTION DOC] Shows maybe_disable_local_tensor_mode makes tensor creation yield a regular
    # tensor both outside and inside LocalTensorMode (world size 4). Returns ((outside type name,
    # inside type name), (Tensor, Tensor)).
    """Use maybe_disable_local_tensor_mode() for portable code.

    Returns: ((outside_type, inside_type), (expected_outside, expected_inside))
    """

    def create_tensor():
        # [FUNCTION DOC] HELPER: creates a three element float tensor with local tensor mode
        # disabled when active, so the result is a plain tensor.
        with maybe_disable_local_tensor_mode():
            return torch.tensor([1.0, 2.0, 3.0])

    t1 = create_tensor()
    outside_type = type(t1).__name__

    with LocalTensorMode(4):
        t2 = create_tensor()
        inside_type = type(t2).__name__

    return (outside_type, inside_type), ("Tensor", "Tensor")


# [end_core_maybe_disable]


# [core_maybe_run_decorator]
def use_maybe_run_decorator(world_size: int = 4):
    # [FUNCTION DOC] Demonstrates the maybe_run_for_local_tensor decorator: a function taking the
    # data and a rank is run once per rank on a LocalTensor of ranks, assembling per rank
    # outputs. Splits 16 floats into 4 chunks. Returns (mapping rank to list, expected mapping).
    """Use @maybe_run_for_local_tensor for rank-specific computations.

    The decorator runs a function once per rank when inside LocalTensorMode,
    automatically handling LocalTensor inputs and collecting per-rank outputs.

    Returns: (result_values, expected_values)
    """

    @maybe_run_for_local_tensor
    def compute_rank_offset(data: torch.Tensor, rank: int) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: returns the slice of data for the given rank, of length a
        # quarter of the first dimension (assumes exactly 4 ranks), starting at rank times that
        # length.
        """Compute offset into data based on rank - non-SPMD behavior."""
        chunk_size = data.shape[0] // 4  # Assume 4 ranks
        start = rank * chunk_size
        return data[start : start + chunk_size]

    full_data = torch.arange(16).float()

    with LocalTensorMode(world_size):
        ranks = rank_map(lambda r: torch.tensor(r))
        result = compute_rank_offset(full_data, ranks)

        values = {
            rank: result._local_tensors[rank].tolist() for rank in range(world_size)
        }

    expected = {
        0: [0.0, 1.0, 2.0, 3.0],
        1: [4.0, 5.0, 6.0, 7.0],
        2: [8.0, 9.0, 10.0, 11.0],
        3: [12.0, 13.0, 14.0, 15.0],
    }
    return values, expected


# [end_core_maybe_run_decorator]


if __name__ == "__main__":
    print("=== use_rank_map ===")
    values, expected = use_rank_map()
    print(f"Values: {values}={expected}")

    print("\n=== use_tensor_map ===")
    values, expected = use_tensor_map()
    print(f"Values: {values}={expected}")

    print("\n=== disable_mode_temporarily ===")
    (inside, disabled), (exp_in, exp_dis) = disable_mode_temporarily()
    print(f"Inside: {inside}={exp_in}, Disabled: {disabled}={exp_dis}")

    print("\n=== use_maybe_disable ===")
    (outside, inside), (exp_out, exp_in) = use_maybe_disable()
    print(f"Outside: {outside}={exp_out}, Inside: {inside}={exp_in}")

    print("\n=== use_maybe_run_decorator ===")
    values, expected = use_maybe_run_decorator()
    print(f"Values match: {values == expected}")
