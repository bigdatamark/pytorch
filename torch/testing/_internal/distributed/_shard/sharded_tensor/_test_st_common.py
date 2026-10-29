# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Shared fixtures for sharded tensor tests: the module-level list of four
# rank/device placement strings, a generator of randomly ordered chunk sharding specs, and two
# nested sample models holding sharded tensors. Import-time side effect: queries the current
# accelerator to set device_type (cpu if none).

import copy
import random

import torch
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


PLACEMENTS = [
    f"rank:0/{device_type}:0",
    f"rank:1/{device_type}:1",
    f"rank:2/{device_type}:2",
    f"rank:3/{device_type}:3",
]

DEFAULT_GPU_NUM = 4


def _chunk_sharding_specs_list_for_test(sharding_dims, seed=0):
    # [FUNCTION DOC] HELPER: Returns one chunk sharding spec per entry of sharding_dims, each with
    # the four placements in a pseudo-random order seeded by seed plus the entry index.
    # Side effect: shuffles the module-level PLACEMENTS list in place (specs get deep copies).
    spec_list = []
    for i in range(len(sharding_dims)):
        random.Random(seed + i).shuffle(PLACEMENTS)
        spec_list.append(
            ChunkShardingSpec(
                dim=sharding_dims[i],
                placements=copy.deepcopy(PLACEMENTS),
            )
        )
    return spec_list


class MyShardedModel2(torch.nn.Module):
    # [CLASS DOC] HELPER: Leaf sample model with attributes sharded_tensor2 (a random 10 by 20
    # sharded tensor, or None when no spec is given) and random_tensor2 (a 2 by 2 parameter).
    def __init__(self, spec=None, group=None, init_rrefs=True) -> None:
        # [FUNCTION DOC] HELPER: Creates the sharded tensor from spec on the given process group
        # (optionally with remote references) when spec is not None, else sets it to None, and
        # creates a random 2 by 2 parameter.
        super().__init__()
        if spec is not None:
            self.sharded_tensor2 = sharded_tensor.rand(
                spec, 10, 20, process_group=group, init_rrefs=init_rrefs
            )
        else:
            self.sharded_tensor2 = None
        self.random_tensor2 = torch.nn.Parameter(torch.rand(2, 2))


class MyShardedModel1(torch.nn.Module):
    # [CLASS DOC] HELPER: Sample model with attributes sharded_tensor1 (random 10 by 20 sharded
    # tensor, or None without a spec), random_tensor1 (2 by 2 parameter) and a nested
    # MyShardedModel2 submodule, for state-dict and nested-module sharding tests.
    def __init__(self, spec=None, group=None, init_rrefs=True) -> None:
        # [FUNCTION DOC] HELPER: Like MyShardedModel2 construction, plus builds the nested
        # submodule with the same spec, group and init_rrefs arguments.
        super().__init__()
        if spec is not None:
            self.sharded_tensor1 = sharded_tensor.rand(
                spec, 10, 20, process_group=group, init_rrefs=init_rrefs
            )
        else:
            self.sharded_tensor1 = None
        self.random_tensor1 = torch.nn.Parameter(torch.rand(2, 2))
        self.submodule = MyShardedModel2(spec, group, init_rrefs)
