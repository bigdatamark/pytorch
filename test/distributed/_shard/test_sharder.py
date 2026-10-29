# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-device tests for custom Sharder objects used inside a ShardingPlan with
# shard_module. Defines a toy embedding-bag collection, a sharded counterpart that applies
# row-wise sharding to the first bags and column-wise to the rest, and a Sharder that builds it.
# Import-time effects: exits the process with status 0 under dev-asan builds; computes the module
# globals device_type (current accelerator type or cpu) and BACKEND (default distributed backend
# for that device).
import copy
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard import shard_module
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharder import Sharder
from torch.distributed._shard.sharding_plan import ShardingPlan
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)
BACKEND = dist.Backend.default_device_backend_map[device_type]


# a simple collection of embedding bag implementation
class CustomEmbeddingBagCollection(nn.Module):
    # [CLASS DOC] HELPER: Toy module holding num_bags EmbeddingBag layers (sum mode) in a
    # ModuleDict keyed by embedding_bag_ plus index; forward concatenates every bag's output for
    # the same input along dim 0.
    def __init__(self, num_bags, num_embeddings_per_bag, num_dims):
        # [FUNCTION DOC] HELPER: Records num_bags and creates that many sum-mode EmbeddingBag
        # layers, each with num_embeddings_per_bag rows and num_dims columns.
        super().__init__()
        self.num_bags = num_bags
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()

        for i in range(num_bags):
            self.embedding_bags[f"embedding_bag_{i}"] = nn.EmbeddingBag(
                num_embeddings_per_bag, num_dims, mode="sum"
            )

    def forward(self, inputs):
        # [FUNCTION DOC] HELPER: Applies every bag to the inputs in insertion order and returns
        # the outputs concatenated along the first dimension.
        outputs = []
        for bag in self.embedding_bags.values():
            outputs.append(bag(inputs))
        return torch.cat(outputs)


# a simple sharded version of EBC
class CustomShardedEBC(nn.Module):
    # [CLASS DOC] HELPER: Sharded replacement for CustomEmbeddingBagCollection. The constructor
    # shards the source module's bag weights in place and re-registers the bags: bags with index
    # below split_idx get the row spec, the rest get the column spec.
    def __init__(self, ebc, split_idx, specs):
        # [FUNCTION DOC] HELPER: Unpacks specs into a row spec and a column spec, requires
        # split_idx to be smaller than the source's bag count (raises AssertionError otherwise),
        # then for each bag calls shard_module on the source with a one-entry plan for that bag's
        # weight and stores the bag in its own ModuleDict. Mutates the source module's
        # parameters.
        super().__init__()
        self.split_idx = split_idx
        row_spec, col_spec = specs

        # create embedding bags base on the spec
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()

        if not (self.split_idx < ebc.num_bags):
            raise AssertionError(
                f"Expected split_idx < num_bags, got {self.split_idx} vs {ebc.num_bags}"
            )
        for i in range(ebc.num_bags):
            bag_key = f"embedding_bag_{i}"
            if i < self.split_idx:
                shard_module(
                    ebc,
                    plan=ShardingPlan(
                        plan={f"embedding_bags.{bag_key}.weight": row_spec}
                    ),
                )
            else:
                shard_module(
                    ebc,
                    plan=ShardingPlan(
                        plan={f"embedding_bags.{bag_key}.weight": col_spec}
                    ),
                )

            self.embedding_bags[bag_key] = ebc.embedding_bags[bag_key]


class CustomSharder(Sharder):
    # [CLASS DOC] HELPER: Sharder for CustomEmbeddingBagCollection. Holds the device placements,
    # the split index and prebuilt row-wise (dim 0) and column-wise (dim 1) ChunkShardingSpecs
    # over those placements.
    def __init__(self, devices, split_sharding_idx):
        # [FUNCTION DOC] HELPER: Stores devices and split_sharding_idx and creates the row-wise
        # and column-wise chunk sharding specs over the given placements.
        self.devices = devices
        self.split_sharding_idx = split_sharding_idx
        self.rowwise_spec = ChunkShardingSpec(dim=0, placements=devices)
        self.colwise_spec = ChunkShardingSpec(dim=1, placements=devices)

    def shard(self, ebc: nn.Module) -> nn.Module:
        # [FUNCTION DOC] HELPER: Returns a CustomShardedEBC built from the given module and the
        # stored split index and specs. Raises RuntimeError if the module is not a
        # CustomEmbeddingBagCollection.
        if not isinstance(ebc, CustomEmbeddingBagCollection):
            raise RuntimeError(
                "The custom sharder only supports CustomEmbeddingBagCollection"
            )

        return CustomShardedEBC(
            ebc, self.split_sharding_idx, (self.rowwise_spec, self.colwise_spec)
        )


class TestCustomSharder(ShardedTensorTestBase):
    # [CLASS DOC] Multi-device test case for plans that contain a custom Sharder.
    @with_comms(init_rpc=False, backend=BACKEND)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_custom_sharder(self):
        # [FUNCTION DOC] Shards a module whose submodule ebc is covered by a plan mapping to
        # CustomSharder, on a deep copy of the local model. Asserts first bag weights are
        # ShardedTensors with the row spec and the last bag with the column spec, then runs both
        # models on a small integer input and asserts the outputs are equal.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Wrapper module with a single CustomEmbeddingBagCollection of 10
            # bags, 10 embeddings and 8 dims.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ebc submodule.
                super().__init__()
                self.ebc = CustomEmbeddingBagCollection(10, 10, 8)

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns the output of the ebc submodule for the inputs.
                return self.ebc(inputs)

        custom_sharder = CustomSharder(
            devices=[f"rank:{i}/{device_type}:{i}" for i in range(TEST_GPU_NUM)],
            split_sharding_idx=TEST_GPU_NUM // 2,
        )

        sharding_plan = ShardingPlan(
            plan={
                "ebc": custom_sharder,
            }
        )

        local_model = MyModule().to(f"{device_type}:{self.rank}")
        sharded_model = copy.deepcopy(local_model)

        # shard the module with the provided sharding plan
        shard_module(sharded_model, sharding_plan)

        # check to make sure the module already been sharded
        emb_bags = sharded_model.ebc.embedding_bags
        self.assertTrue(isinstance(emb_bags["embedding_bag_0"].weight, ShardedTensor))
        self.assertTrue(isinstance(emb_bags["embedding_bag_9"].weight, ShardedTensor))
        self.assertEqual(
            emb_bags["embedding_bag_0"].weight.sharding_spec(),
            custom_sharder.rowwise_spec,
        )
        self.assertEqual(
            emb_bags["embedding_bag_9"].weight.sharding_spec(),
            custom_sharder.colwise_spec,
        )

        # make sure we can run sharded computation and compare outputs
        # with the local model version
        input = torch.arange(8).reshape((2, 4)).to(f"{device_type}:{self.rank}")
        local_output = local_model(input)
        sharded_output = sharded_model(input)

        self.assertEqual(local_output, sharded_output)

    @with_comms(init_rpc=False, backend=BACKEND)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_custom_sharder_errors(self):
        # [FUNCTION DOC] Checks two shard_module failures with a custom sharder: a plan keyed by
        # the empty path raises KeyError (path must not be empty), and a plan with both a weight
        # spec and a sharder on its parent submodule raises RuntimeError (conflict with the
        # submodule tree).
        custom_sharder = CustomSharder(
            devices=[f"rank:{i}/{device_type}:{i}" for i in range(TEST_GPU_NUM)],
            split_sharding_idx=TEST_GPU_NUM // 2,
        )

        sharding_plan = ShardingPlan(
            plan={
                "": custom_sharder,
            }
        )

        sharded_model = CustomEmbeddingBagCollection(10, 10, 8).to(
            f"{device_type}:{self.rank}"
        )

        with self.assertRaisesRegex(
            KeyError, "path must not be empty for custom sharder!"
        ):
            # shard the module with the provided sharding plan
            shard_module(sharded_model, sharding_plan)

        # test conflicted sharding plan
        spec = ChunkShardingSpec(
            dim=0, placements=[f"rank:0/{device_type}:0", f"rank:1/{device_type}:1"]
        )
        sharding_plan = ShardingPlan(
            plan={
                "embedding_bags.embedding_bag_0.weight": spec,
                "embedding_bags": custom_sharder,
            }
        )

        with self.assertRaisesRegex(
            RuntimeError, "should not conflict with the submodule tree"
        ):
            # shard the module with the provided sharding plan
            shard_module(sharded_model, sharding_plan)


if __name__ == "__main__":
    run_tests()
