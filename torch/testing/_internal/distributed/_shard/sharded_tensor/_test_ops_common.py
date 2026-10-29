# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Utilities for sharded tensor op tests: generators of chunk and enumerable
# sharding specs for four ranks, a helper computing a rank's shard start and size for a local
# weight, a parameter cloner, and a generator of binary operator functions. Import-time side
# effect: queries the current accelerator to set the module-level device_type (cpu if none).

import builtins

import torch
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec._internals import (
    get_chunked_dim_size,
    get_split_size,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def generate_chunk_sharding_specs_for_test(sharding_dim):
    # [FUNCTION DOC] HELPER: Returns a list of three chunk sharding specs over the given
    # dimension for four ranks on the current device type: the natural rank order and two
    # rotated placement orders.
    return [
        ChunkShardingSpec(
            dim=sharding_dim,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        ),
        # Test different ordering. (Case 1)
        ChunkShardingSpec(
            dim=sharding_dim,
            placements=[
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
            ],
        ),
        # Test different ordering. (Case 2)
        ChunkShardingSpec(
            dim=sharding_dim,
            placements=[
                f"rank:3/{device_type}:3",
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
            ],
        ),
    ]


def generate_enumerable_sharding_specs_for_test():
    # [FUNCTION DOC] HELPER: Returns a list with one enumerable sharding spec that splits a 10 by
    # 10 tensor into four 5 by 5 quadrants, one per rank (rank 0 top-left, 1 bottom-left, 2
    # top-right, 3 bottom-right) on the current device type.
    return [
        EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement=f"rank:0/{device_type}:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement=f"rank:1/{device_type}:1",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement=f"rank:2/{device_type}:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement=f"rank:3/{device_type}:3",
                ),
            ]
        )
    ]


def generate_local_weight_sharding_params_for_test(
    local_weight, sharded_dim, gpu_num, spec, rank
):
    # [FUNCTION DOC] HELPER: Computes where the given rank's shard of local_weight begins along
    # sharded_dim and how large the chunk is, by splitting the dimension evenly across gpu_num
    # ranks and walking the spec's placements in order until the one owned by rank is found.
    # Returns (start_pos, chunk_size).
    # KNOWN ISSUE: if rank matches no placement, the start stays at zero and the chunk size is
    # that of the last placement; if the placements are empty, chunk_size is undefined and an
    # error is raised.
    """
    Shard the local weight based the given spec, so we can compare against
    the one from sharded tensor.

    Args:
        local_weight: weight matrix to be sharded.
        sharded_dim: The dimension which we shard on.
        gpu_num: number of ranks.
        spec: sharding spec.
        rank: # of cuda process.

    Returns:
        start_pos: start position of sharded weight on the given rank.
        chunk_size: chunk size of sharded weight on the given rank.
    """
    sharding_dim_size = local_weight.size(sharded_dim)
    split_size = get_split_size(sharding_dim_size, gpu_num)
    current_offsets = 0
    start_pos = current_offsets
    for idx, placement in enumerate(spec.placements):
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        if rank == placement.rank():
            start_pos = current_offsets
            break
        current_offsets += chunk_size
    return start_pos, chunk_size


def clone_module_parameter(module, param_name):
    # [FUNCTION DOC] HELPER: Returns a new Parameter holding a detached copy of the named
    # parameter of module; the original is untouched. Raises AttributeError if the name is absent.
    """
    Clone a parameter from a given existing module.

    Args:
        module (:class:`torch.nn.Module`): Module whose parameter needs to be cloned.
        param_name (str): Name of the parameter of ``module`` that needs to be cloned.

    Returns: cloned tensor as :class:`torch.nn.Parameter`.
    """
    tensor = getattr(module, param_name)
    return torch.nn.Parameter(tensor.detach().clone())


def gen_binary_op_func(python_op, inplace=False):
    # [FUNCTION DOC] HELPER: Builds, by generating and executing source text, a two-argument
    # function (lhs, rhs) applying python_op. If the op string mentions torch, it is treated as a
    # function name and called with both arguments; else if inplace is true, the augmented
    # assignment form of the operator is applied to lhs and lhs is returned; else the plain
    # infix operator result is returned. Only torch is available in the generated code's globals.
    # Side effect: dynamic code execution; python_op must be trusted test input.
    src_lines = ["def f(lhs, rhs):"]
    if "torch" in python_op:
        src_lines.append(f"  return {python_op}(lhs, rhs)\n")
    elif inplace:
        src_lines.append(f"  lhs {python_op}= rhs\n  return lhs\n")
    else:
        src_lines.append(f"  return lhs {python_op} rhs\n")

    code_str = "\n".join(src_lines)
    g = {"torch": torch}
    builtins.exec(code_str, g)
    return g["f"]
