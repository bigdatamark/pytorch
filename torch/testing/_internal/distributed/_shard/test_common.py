# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Shared test model for sharding tests: SimpleMegatronLM, a two-layer
# feed-forward block (linear, GELU, linear) in the style of tensor-parallel Megatron MLPs, with
# accessors that unwrap sharded weights to local tensors. No import-time side effects.

import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor


class SimpleMegatronLM(nn.Module):
    # [CLASS DOC] HELPER: Test module with attributes fc1, gelu and fc2. Tests may replace the
    # fc1/fc2 weights with ShardedTensor parameters; the get_* accessors return weights (local
    # shard if sharded), biases and gradients of both layers as 2-tuples.
    def __init__(self, linear_size, rank=None, dtype=torch.float32):
        # [FUNCTION DOC] HELPER: Builds two linear layers whose (in, out) feature sizes come from
        # the two entries of linear_size, with the given dtype, and a GELU. If rank is not None,
        # both layers are moved to the device for that rank.
        super().__init__()
        self.fc1 = nn.Linear(*linear_size[0], dtype=dtype)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(*linear_size[1], dtype=dtype)
        if rank is not None:
            self.fc1.to(torch.device(rank))
            self.fc2.to(torch.device(rank))

    def forward(self, inp):
        # [FUNCTION DOC] HELPER: Applies fc1, then GELU, then fc2 to inp and returns the result.
        return self.fc2(self.gelu(self.fc1(inp)))

    def get_weights(self):
        # [FUNCTION DOC] HELPER: Returns a tuple of the fc1 and fc2 weights; a weight that is a
        # ShardedTensor is replaced by its local shard tensor, other weights are returned as is.
        if isinstance(self.fc1.weight, ShardedTensor):
            weight1 = self.fc1.weight.local_tensor()
        else:
            weight1 = self.fc1.weight

        if isinstance(self.fc2.weight, ShardedTensor):
            weight2 = self.fc2.weight.local_tensor()
        else:
            weight2 = self.fc2.weight

        return (weight1, weight2)

    def get_biases(self):
        # [FUNCTION DOC] HELPER: Returns a tuple of the fc1 and fc2 bias parameters.
        return (self.fc1.bias, self.fc2.bias)

    def get_weight_grads(self):
        # [FUNCTION DOC] HELPER: Returns a tuple of the gradients of the fc1 and fc2 weights
        # (may be None before backward; sharded weights are not unwrapped here).
        return (self.fc1.weight.grad, self.fc2.weight.grad)

    def get_bias_grads(self):
        # [FUNCTION DOC] HELPER: Returns a tuple of the gradients of the fc1 and fc2 biases
        # (None before backward).
        return (self.fc1.bias.grad, self.fc2.bias.grad)
