# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU tests for ShardedOptimizer, which applies a regular optimizer to
# modules holding ShardedTensor parameters. Defines two small models and a test case needing
# four accelerators. At import it computes the module-level device_type (current accelerator
# type or cpu) and the default distributed backend for it.

from copy import deepcopy

import torch
import torch.optim as optim
from torch.distributed._shard import shard_parameter, sharded_tensor
from torch.distributed._shard.sharded_optim import ShardedOptimizer
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)
backend = torch.distributed.get_default_backend_for_device(device_type)


class MyShardedModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Tiny module with a regular 5x10 parameter and a second parameter
    # named sharded_param. With a sharding spec the latter is a 20x10 ShardedTensor
    # parameter; without one it is a plain 5x10 random parameter. Forward adds the inputs
    # elementwise to both parameters (using the first local shard when sharded).
    def __init__(self, spec=None, group=None):
        # [FUNCTION DOC] HELPER: Seeds the global RNG with 0 so every rank/model gets the
        # same values, then creates the parameters. spec is an optional sharding spec and
        # group the process group used to build the sharded tensor. Mutates global RNG state.
        super().__init__()
        # Use same seed.
        torch.manual_seed(0)
        self.param = torch.nn.Parameter(torch.rand(5, 10))
        if spec is not None:
            self.sharded_param = torch.nn.Parameter(
                sharded_tensor.rand(
                    spec, 20, 10, requires_grad=True, process_group=group
                )
            )
        else:
            self.sharded_param = torch.nn.Parameter(torch.rand(5, 10))

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Returns the sum of the regular parameter, the sharded
        # parameter (its first local shard tensor if it is a ShardedTensor, else the plain
        # tensor) and the input. Shapes must broadcast with each other.
        if isinstance(self.sharded_param, sharded_tensor.ShardedTensor):
            return self.param + self.sharded_param.local_shards()[0].tensor + input
        else:
            return self.sharded_param + self.param + input


class MyShardedLinear(torch.nn.Module):
    # [CLASS DOC] HELPER: Two-layer MLP (17 -> 12 -> 29 linear layers with GELU between)
    # whose weights can be sharded across four ranks: first layer row-wise, second layer
    # column-wise, via shard_parameter.
    def __init__(self, rank=None):
        # [FUNCTION DOC] HELPER: Seeds the global RNG with 0, builds the layers and, if rank
        # is truthy, moves both linear layers to that device (rank 0 is falsy, so it is not
        # moved). Mutates global RNG state.
        super().__init__()
        # Use same seed.
        torch.manual_seed(0)
        self.linear1 = torch.nn.Linear(17, 12)
        self.linear2 = torch.nn.Linear(12, 29)
        self.gelu = torch.nn.GELU()

        if rank:
            self.linear1.to(rank)
            self.linear2.to(rank)

    def shard_parameter(self):
        # [FUNCTION DOC] HELPER: Replaces linear1.weight by a tensor sharded along dim 0
        # and linear2.weight by one sharded along dim 1, each over ranks 0-3 placed on the
        # matching device index. Modifies the module in place; needs an initialized group.
        rowwise_sharding_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )

        colwise_sharding_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )

        shard_parameter(self.linear1, "weight", rowwise_sharding_spec)
        shard_parameter(self.linear2, "weight", colwise_sharding_spec)

    def forward(self, inp):
        # [FUNCTION DOC] HELPER: Applies linear1, GELU, then linear2 to the input.
        return self.linear2(self.gelu(self.linear1(inp)))


class TestShardedOptimizer(ShardedTensorTestBase):
    # [CLASS DOC] Multi-process test case (4 ranks, accelerator backend) for
    # ShardedOptimizer; process group setup is done by the with_comms decorator.
    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend(["nccl", "xccl", "privateuse1"])
    def test_sharded_optim(self):
        # [FUNCTION DOC] Builds a plain model and an identically seeded model whose second
        # parameter is row-sharded over 4 ranks, copies the plain values into the local
        # shard, and trains both with SGD (lr 0.1; ShardedOptimizer for the sharded one) for
        # one forward/backward/step. Asserts each parameter changed and that the updated
        # sharded shard and regular parameter equal the local model's updated parameters.
        rowwise_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )
        local_model = MyShardedModel().to(device_type)
        sharded_model = MyShardedModel(spec=rowwise_spec).to(device_type)

        # copy the parameters from local model
        sharded_model.sharded_param.local_shards()[0].tensor = (
            local_model.sharded_param.detach().clone().requires_grad_()
        )

        local_optim = optim.SGD(local_model.parameters(), lr=0.1)
        sharded_model_params = dict(sharded_model.named_parameters())
        sharded_optim = ShardedOptimizer(sharded_model_params, optim.SGD, lr=0.1)

        local_optim.zero_grad()
        sharded_optim.zero_grad()

        before_update = deepcopy(sharded_optim.named_params)

        inp = torch.rand(
            [5, 10], device=torch.device(device_type, self.rank), requires_grad=True
        )

        # run forward
        local_output = local_model(inp)
        sharded_output = sharded_model(inp)
        # backward
        local_output.sum().backward()
        sharded_output.sum().backward()

        # optimizer update
        local_optim.step()
        sharded_optim.step()

        # make sure the parameters (including sharded param)
        # get updated by the optimizer, and the updated
        # local params are the same as the sharded params
        for key, val in before_update.items():
            new_val = sharded_optim.named_params[key]
            if isinstance(val, sharded_tensor.ShardedTensor):
                self.assertNotEqual(
                    val.local_shards()[0].tensor, new_val.local_shards()[0].tensor
                )
                self.assertEqual(
                    new_val.local_shards()[0].tensor, local_model.sharded_param
                )
            else:
                self.assertNotEqual(val, new_val)
                self.assertEqual(new_val, local_model.param)

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend(["nccl", "xccl", "privateuse1"])
    def test_named_params_with_sharded_tensor(self):
        # [FUNCTION DOC] Checks named_parameters of a model with a sharded parameter lists
        # the expected names (param, sharded_param), and likewise for the sharded two-layer
        # MLP (both layers' weights and biases, no bare bias).
        rowwise_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )
        sharded_model = MyShardedModel(spec=rowwise_spec).to(device_type)
        sharded_model_params = dict(sharded_model.named_parameters())
        param_keys = list(sharded_model_params.keys())
        self.assertEqual(len(param_keys), 2)
        self.assertTrue("param" in param_keys)
        self.assertTrue("sharded_param" in param_keys)

        sharded_linear = MyShardedLinear(rank=self.rank).to(device_type)
        sharded_linear.shard_parameter()
        sharded_linear_params = dict(sharded_linear.named_parameters())
        param_keys = list(sharded_linear_params.keys())
        self.assertEqual(len(param_keys), 4)
        self.assertTrue("linear1.bias" in param_keys)
        self.assertTrue("linear2.bias" in param_keys)
        self.assertTrue("linear1.weight" in param_keys)
        self.assertTrue("linear2.weight" in param_keys)
        self.assertFalse("bias" in param_keys)


if __name__ == "__main__":
    run_tests()
