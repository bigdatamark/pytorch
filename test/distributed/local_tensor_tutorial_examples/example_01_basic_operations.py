# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tutorial example 1: creating and using LocalTensor, which simulates several
# ranks inside one process. Functions are delimited by marker comments so documentation can
# include them, and each returns actual values together with expected values for tests.
# Running as a script prints a short summary of each case. No import-time side effects.

"""
Example 1: Basic LocalTensor Creation and Operations
=====================================================

This module contains core patterns for LocalTensor basics.
The functions below are designed to be:
1. Included in the tutorial via literalinclude
2. Directly tested to ensure tutorial correctness

Each function returns a tuple of (actual, expected) for testing.
"""

import torch
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode


# [core_create_local_tensor]
def create_local_tensor():
    # [FUNCTION DOC] Builds a LocalTensor from a mapping of two ranks (0 and 1) to 2x2 float
    # tensors. Returns the LocalTensor and a tuple of expected shape, expected rank set (a
    # frozenset of 0 and 1) and the two input tensors.
    """Create a LocalTensor from per-rank tensors.

    Returns: (local_tensor, (expected_shape, expected_ranks, expected_rank_0, expected_rank_1))
    """
    rank_0_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rank_1_tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    local_tensor = LocalTensor({0: rank_0_tensor, 1: rank_1_tensor})

    expected = (torch.Size([2, 2]), frozenset({0, 1}), rank_0_tensor, rank_1_tensor)
    return local_tensor, expected


# [end_core_create_local_tensor]


# [core_arithmetic_operations]
def arithmetic_operations():
    # [FUNCTION DOC] Builds a two rank LocalTensor, multiplies it by 2 and adds 10 (each rank is
    # transformed independently). Returns the pair (doubled, added) and the expected per rank
    # results: doubled rank 0, doubled rank 1 and added rank 0.
    """Demonstrate arithmetic on LocalTensor.

    Returns: ((doubled, added), (expected_doubled_0, expected_doubled_1, expected_added_0))
    """
    input_0 = torch.tensor([1.0, 2.0, 3.0])
    input_1 = torch.tensor([4.0, 5.0, 6.0])

    lt = LocalTensor({0: input_0, 1: input_1})

    doubled = lt * 2
    added = lt + 10

    expected = (input_0 * 2, input_1 * 2, input_0 + 10)
    return (doubled, added), expected


# [end_core_arithmetic_operations]


# [core_reconcile]
def reconcile_identical_shards():
    # [FUNCTION DOC] Creates a three rank LocalTensor whose shards are identical clones and calls
    # reconcile, which collapses it to one plain tensor. Returns (result, original tensor).
    """Extract a single tensor when all shards are identical.

    Returns: (result, expected)
    """
    value = torch.tensor([1.0, 2.0, 3.0])
    lt = LocalTensor({0: value.clone(), 1: value.clone(), 2: value.clone()})

    result = lt.reconcile()
    return result, value


# [end_core_reconcile]


# [core_local_tensor_mode]
def use_local_tensor_mode(world_size: int = 4):
    # [FUNCTION DOC] Inside a LocalTensorMode of the given world size, creates a ones tensor and
    # records whether it is a LocalTensor and how many ranks it spans. Returns ((is_local,
    # num_ranks), (True, world_size)).
    """Use LocalTensorMode to auto-create LocalTensors.

    Returns: ((is_local, num_ranks), (expected_is_local, expected_num_ranks))
    """
    with LocalTensorMode(world_size):
        x = torch.ones(2, 3)
        is_local = isinstance(x, LocalTensor)
        num_ranks = len(x._ranks)

    return (is_local, num_ranks), (True, world_size)


# [end_core_local_tensor_mode]


# [core_access_shards]
def access_individual_shards():
    # [FUNCTION DOC] Builds a three rank LocalTensor and reads the shard of rank 0 through the
    # mapping of local tensors and the shard of rank 1 through the per rank attribute. Returns
    # ((shard_0, shard_1), (input_0, input_1)).
    """Access shards for debugging.

    Returns: ((shard_0, shard_1), (expected_0, expected_1))
    """
    input_0 = torch.tensor([1.0, 2.0])
    input_1 = torch.tensor([3.0, 4.0])

    lt = LocalTensor({0: input_0, 1: input_1, 2: torch.tensor([5.0, 6.0])})

    shard_0 = lt._local_tensors[0]
    shard_1 = lt._local_tensor_1

    return (shard_0, shard_1), (input_0, input_1)


# [end_core_access_shards]


if __name__ == "__main__":
    print("=== create_local_tensor ===")
    lt, (exp_shape, exp_ranks, _, _) = create_local_tensor()
    print(f"Shape: {lt.shape}, Expected: {exp_shape}")

    print("\n=== arithmetic_operations ===")
    (doubled, _), (exp_d0, _, _) = arithmetic_operations()
    print(f"Doubled rank 0: {doubled._local_tensors[0]}, Expected: {exp_d0}")

    print("\n=== reconcile_identical_shards ===")
    result, expected = reconcile_identical_shards()
    print(f"Reconciled: {result}, Expected: {expected}")

    print("\n=== use_local_tensor_mode ===")
    (is_local, num_ranks), (exp_local, exp_ranks) = use_local_tensor_mode()
    print(f"Is LocalTensor: {is_local}={exp_local}, Ranks: {num_ranks}={exp_ranks}")

    print("\n=== access_individual_shards ===")
    (s0, s1), (e0, e1) = access_individual_shards()
    print(f"Shard 0: {s0}, Expected: {e0}")
