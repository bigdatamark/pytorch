# mypy: allow-untyped-defs

# Copyright (c) Meta Platforms, Inc. and affiliates

# [MODULE DOC] Shared test infrastructure for DTensor and distributed tests: device and world-size
# constants chosen from the available accelerators, small test models (MLP, a toy Transformer with
# optional mixture-of-experts and tensor/expert parallel plans), test base classes for multi-
# process, multi-thread, continuous and local-tensor (fake process group) modes, the with_comms
# decorator, a converter that enumerates DTensor placements for an op's arguments, shard-order
# redistribution wrappers, sharding-rule validators and an op-strategy context manager. Import-
# time side effects: probes accelerators and sets the module constants DEVICE_TYPE, DEVICE_COUNT,
# PG_BACKEND and NUM_DEVICES (at most four, fewer on ROCm or when few devices exist).

import contextlib
import copy
import functools
import itertools
import sys
import threading
import types
import unittest
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, cast, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import (
    all_gather_single_autograd,
    reduce_scatter_single_autograd,
)
from torch.distributed._local_tensor import (
    LocalIntNode,
    LocalTensor,
    LocalTensorMode,
    maybe_disable_local_tensor_mode,
    maybe_run_for_local_tensor,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import ShardOrderEntry
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_distributed import (
    ACCELERATOR_DIST_BACKENDS,
    MultiProcContinuousTest,
    MultiProcessTestCase,
    MultiThreadedTestCase,
    run_subtests,
    skip_if_lt_x_gpu,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    TEST_CUDA,
    TEST_HPU,
    TEST_PRIVATEUSE1,
    TEST_WITH_ROCM,
    TEST_XPU,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec


DEVICE_COUNT: int

if TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1:
    DEVICE_TYPE = torch.accelerator.current_accelerator().type
    DEVICE_COUNT = torch.accelerator.device_count()
    PG_BACKEND = dist.Backend.default_device_backend_map[DEVICE_TYPE]
else:
    DEVICE_TYPE = "cpu"
    PG_BACKEND = "gloo"

if TEST_WITH_ROCM:
    NUM_DEVICES = min(4, max(2, torch.cuda.device_count()))
else:
    NUM_DEVICES = 4

# We use this as a proxy for "multiple GPUs exist"
if (TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1) and DEVICE_COUNT > 1:
    # when we actually have multiple GPUs, relax the requirement to smaller counts.
    NUM_DEVICES = min(NUM_DEVICES, DEVICE_COUNT)

T = TypeVar("T")


# simple RMSNorm layer for testing
class RMSNormPython(torch.nn.Module):
    # [CLASS DOC] Simple RMS normalization layer: scales the input by the inverse root-mean-square
    # over the last dimension and multiplies by a learned per-dimension weight (initialized to
    # ones). Attributes: eps and weight.
    def __init__(self, dim: int, eps: float = 1e-6):
        # [FUNCTION DOC] Stores eps and creates a ones-initialized weight parameter of size dim.
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # [FUNCTION DOC] HELPER: Returns the input divided by the root mean square of its last
        # dimension plus eps.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # [FUNCTION DOC] Normalizes the input and multiplies by the weight; returns the result.
        output = self._norm(x)
        return output * self.weight


class MLPModule(nn.Module):
    # [CLASS DOC] Two-layer perceptron test model: linear 10 to 16, ReLU, linear 16 to 10, created
    # on the given device with optional bias.
    def __init__(self, device, bias: bool = True):
        # [FUNCTION DOC] Seeds the global random generator with 5 (side effect) so initialization
        # is deterministic, then builds the two linear layers and the ReLU on the device.
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, bias=bias, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 10, bias=bias, device=device)

    def forward(self, x):
        # [FUNCTION DOC] Applies the first linear layer, the ReLU and the second linear layer;
        # returns the output.
        return self.net2(self.relu(self.net1(x)))

    def reset_parameters(self):
        # [FUNCTION DOC] Re-initializes the parameters of both linear layers.
        self.net1.reset_parameters()
        self.net2.reset_parameters()


class MLPStacked(nn.Module):
    # [CLASS DOC] Stack of MLPModule layers held in a module list, applied in sequence.
    def __init__(self, device, n_layers: int = 2):
        # [FUNCTION DOC] Creates n_layers MLPModule instances on the given device.
        super().__init__()
        self.layers = nn.ModuleList([MLPModule(device) for i in range(n_layers)])

    def forward(self, x):
        # [FUNCTION DOC] Feeds the input through every layer in order and returns the result.
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class ModelArgs:
    # [CLASS DOC] Dataclass of hyperparameters for the toy Transformer: layer count, vocabulary
    # size, maximum sequence length, model dimension, head count, dropout probability, attention
    # flag, weight tying, activation checkpointing and number of experts (zero means a dense feed-
    # forward).
    n_layers: int = 2
    vocab_size: int = 8
    max_seq_len: int = 16
    dim: int = 16
    n_heads: int = 4
    dropout_p: float = 0.1
    use_attn_mask: bool = True
    weight_tying: bool = True
    checkpoint_activations: bool = False
    num_experts: int = 0


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        # [FUNCTION DOC] Builds the four bias-free projections and dropout from the model args and
        # stores head count, head dimension and flags. Raises AssertionError if the dimension is
        # not divisible by the head count.
        super().__init__()
        if args.dim % args.n_heads != 0:
            raise AssertionError(
                f"Expected args.dim % args.n_heads == 0, got {args.dim} % {args.n_heads}"
            )
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = nn.Dropout(args.dropout_p)
        self.use_attn_mask = args.use_attn_mask

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        # [FUNCTION DOC] Multi-head self-attention over a batch of sequences shaped batch,
        # sequence, dim: projects to queries, keys and values, splits heads, runs scaled dot
        # product attention (dropout only in training), merges heads, applies the output
        # projection and residual dropout. Returns a tensor of the input shape.
        # KNOWN ISSUE: the use_attn_mask flag is passed in the position of the causal flag of the
        # attention call, so it enables causal masking rather than supplying a mask.
        bsz, seq_len, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)

        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            None,
            self.dropout_p if self.training else 0,
            self.use_attn_mask,
        )
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        # [FUNCTION DOC] Builds a two-layer feed-forward block (linear, GELU, linear) with output
        # dropout.
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim)
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # [FUNCTION DOC] Applies the first linear, GELU, second linear and dropout; returns the
        # result.
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))


class Experts(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int):
        # [FUNCTION DOC] Creates expert weight parameters of shape (num_experts, hidden_dim, dim)
        # and (num_experts, dim, hidden_dim), initializes them with a normal distribution, and
        # creates a GELU.
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.reset_parameters()
        self.gelu = nn.GELU()

    def reset_parameters(self):
        # [FUNCTION DOC] Re-initializes both expert weights with a normal distribution of standard
        # deviation 0.02.
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Runs every expert on all input tokens with batched matrix multiplies and
        # sums the expert outputs. Uses the local shards if the weights are DTensors (the input is
        # a plain tensor). Returns a tensor of tokens by dim.
        # Weights are DTensors (sharded by EP/TP) but x is a plain tensor
        # (dispatched by EP hooks), so extract local shards for bmm.
        if isinstance(self.w1, DTensor):
            w1, w2 = self.w1.to_local(), self.w2.to_local()
        else:
            w1, w2 = self.w1, self.w2
        E = w1.shape[0]
        x_exp = x.unsqueeze(0).expand(E, -1, -1)
        h = self.gelu(torch.bmm(x_exp, w1.transpose(-2, -1)))
        out = torch.bmm(h, w2.transpose(-2, -1))
        return out.sum(dim=0)


class ExpertLayer(nn.Module):
    def __init__(self, num_experts: int, dim: int, hidden_dim: int):
        # [FUNCTION DOC] Stores the expert count and creates the Experts module.
        super().__init__()
        self.num_experts = num_experts
        self.experts = Experts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Flattens batch and sequence, runs the experts, divides by the number of
        # experts and restores the original shape; returns the result.
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)
        expert_out = self.experts(x_flat)
        expert_out = expert_out / self.num_experts
        return expert_out.view(bs, slen, dim)


class TensorParallelForExpert(ParallelStyle):
    # [CLASS DOC] Parallel style for tensor parallelism of an Experts module: shards the first
    # weight along dim 1 and the second along dim 2. Optional input layouts (for sequence
    # parallel) cause tokens to be gathered before compute; outputs are partial sums redistributed
    # to the output layouts (default replicated).
    """TP for Experts: shard w1 colwise (Shard(1)), w2 rowwise (Shard(2)).

    For seq parallel, set input_layouts=Shard(0) and output_layouts=Shard(0)
    to all-gather tokens before computation and reduce-scatter after.
    """

    def __init__(self, *, input_layouts=None, output_layouts=None):
        # [FUNCTION DOC] Stores the optional input layouts and the output layouts (replicated if
        # not given).
        super().__init__()
        self.input_layouts = input_layouts
        self.output_layouts = output_layouts or Replicate()

    def _partition_fn(self, name, mod, device_mesh):
        # [FUNCTION DOC] HELPER: Partition function: replaces the w1 and w2 parameters with
        # DTensor parameters sharded on dim 1 and dim 2 respectively; other parameters are left
        # alone.
        for pn, p in mod.named_parameters(recurse=False):
            if pn == "w1":
                placement = [Shard(1)]
            elif pn == "w2":
                placement = [Shard(2)]
            else:
                continue
            mod.register_parameter(
                pn, nn.Parameter(distribute_tensor(p, device_mesh, placement))
            )

    def _input_fn(self, mod, inputs, device_mesh):
        # [FUNCTION DOC] HELPER: Input hook: if input layouts were given, converts the first input
        # into a DTensor with those layouts and redistributes it to replicated, returning its
        # local tensor in a tuple; otherwise passes inputs through unchanged.
        if self.input_layouts is not None:
            x = inputs[0]
            x = (
                DTensor.from_local(x, device_mesh, [self.input_layouts])
                .redistribute(device_mesh, [Replicate()])
                .to_local()
            )
            return (x,)
        return inputs

    def _output_fn(self, mod, outputs, device_mesh):
        # [FUNCTION DOC] HELPER: Output hook: treats the local output as a partial sum over the
        # mesh, redistributes it to the output layouts and returns the local tensor.
        return (
            DTensor.from_local(outputs, device_mesh, [Partial()])
            .redistribute(device_mesh, [self.output_layouts])
            .to_local()
        )

    def _apply(self, module, device_mesh):
        # [FUNCTION DOC] HELPER: Applies the style to a module by distributing it with the
        # partition, input and output hooks. Returns the module.
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._input_fn,
            output_fn=self._output_fn,
        )


class ExpertParallel(ParallelStyle):
    # [CLASS DOC] Parallel style for expert parallelism: shards all expert parameters on dim 0
    # (the expert dimension), all-gathers tokens before compute and reduce-scatters (sum) outputs
    # afterwards.
    """Distributes experts across ranks with Shard(0) on the expert dimension.

    Dispatch: all-gather tokens so every rank sees all tokens.
    Combine: reduce-scatter (sum) expert outputs back to token owners.
    """

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        # [FUNCTION DOC] HELPER: Partition function: replaces each direct parameter with a DTensor
        # parameter sharded on dim 0.
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Input hook: all-gathers the single input tensor along dim 0
        # across the mesh group with autograd support and waits for completion. Returns it in a
        # tuple.
        (x,) = inputs
        x_gathered = all_gather_single_autograd(
            x,
            gather_dim=0,
            group=device_mesh.get_group(),
        )
        x_gathered = torch.ops._c10d_functional.wait_tensor(x_gathered)
        return (x_gathered,)

    def _token_combine(
        self, mod: nn.Module, output: torch.Tensor, device_mesh: DeviceMesh
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Output hook: reduce-scatters the output with sum along dim 0 over
        # the mesh group (autograd-aware), waits for completion and returns the local result.
        result = reduce_scatter_single_autograd(
            output,
            "sum",
            scatter_dim=0,
            group=device_mesh.get_group(),
        )
        result = torch.ops._c10d_functional.wait_tensor(result)
        return result

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # [FUNCTION DOC] HELPER: Applies the style by distributing the module with the partition,
        # dispatch and combine hooks; returns the module.
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


class ExpertParallelWithTP(ParallelStyle):
    # [CLASS DOC] Combined expert and tensor parallel style applied to an ExpertLayer. Expert
    # weights are sharded on a concatenated 2D (ep, tp) mesh. Forward order: expert-parallel
    # dispatch, tensor-parallel input handling, compute, tensor-parallel reduction, expert-
    # parallel combine. Attributes: ep_mesh, tp_mesh, ep_tp_mesh.
    """Combined EP + TP for experts.

    Applied to ExpertLayer. Distributes expert params on a 2D (ep, tp) mesh
    with [Shard(0), Shard(1/2)]. Token dispatch/combine hooks are registered
    on ExpertLayer (outer), TP reduction hook on Experts (inner), so forward
    execution is: EP dispatch -> TP input -> forward -> TP reduce -> EP combine.
    """

    def __init__(
        self,
        ep_mesh: DeviceMesh,
        tp_mesh: DeviceMesh,
    ):
        # [FUNCTION DOC] Stores both meshes and builds the concatenated 2D mesh.
        super().__init__()
        self.ep_mesh = ep_mesh
        self.tp_mesh = tp_mesh
        self.ep_tp_mesh = DeviceMesh._concatenate([ep_mesh, tp_mesh])

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # [FUNCTION DOC] HELPER: Applies the style: shards the experts' w1 on (0, 1) and w2 on (0,
        # 2) placements over the 2D mesh, registers EP dispatch (all-gather) and combine (reduce-
        # scatter) hooks on the outer layer, and TP input-gradient all-reduce and output-reduction
        # hooks on the inner experts. Mutates the module and returns it.
        experts = module.experts  # type: ignore[attr-defined]

        # Partition expert weights on 2D (ep, tp) mesh
        for pn, p in experts.named_parameters(recurse=False):
            if pn == "w1":
                placements = [Shard(0), Shard(1)]
            elif pn == "w2":
                placements = [Shard(0), Shard(2)]
            else:
                continue
            experts.register_parameter(
                pn, nn.Parameter(distribute_tensor(p, self.ep_tp_mesh, placements))
            )

        # EP dispatch/combine hooks on ExpertLayer (outer module)
        ep_mesh = self.ep_mesh

        def ep_dispatch(mod, inputs):
            # [FUNCTION DOC] HELPER: Pre-forward hook: all-gathers the input along dim 0 over the
            # expert-parallel group and returns it in a tuple.
            (x,) = inputs
            x = all_gather_single_autograd(x, gather_dim=0, group=ep_mesh.get_group())
            return (torch.ops._c10d_functional.wait_tensor(x),)

        def ep_combine(mod, inputs, output):
            # [FUNCTION DOC] HELPER: Forward hook: reduce-scatters the output with sum over the
            # expert-parallel group and returns the result.
            out = reduce_scatter_single_autograd(
                output, "sum", scatter_dim=0, group=ep_mesh.get_group()
            )
            return torch.ops._c10d_functional.wait_tensor(out)

        module.register_forward_pre_hook(ep_dispatch)
        module.register_forward_hook(ep_combine)

        # TP reduction hook on Experts (inner module)
        tp_mesh = self.tp_mesh

        def tp_allreduce_input_grad(mod, inputs):
            # [FUNCTION DOC] HELPER: Pre-forward hook: marks the input as replicated across the
            # tensor-parallel mesh so its gradient is partial (summed) in backward; returns it in
            # a tuple.
            (x,) = inputs
            return (
                DTensor.from_local(x, tp_mesh, [Replicate()]).to_local(
                    grad_placements=[Partial()]
                ),
            )

        experts.register_forward_pre_hook(tp_allreduce_input_grad)

        def tp_reduce(mod, inputs, output):
            # [FUNCTION DOC] HELPER: Forward hook: reduces the partial outputs across the tensor-
            # parallel mesh to a replicated local tensor.
            return (
                DTensor.from_local(output, tp_mesh, [Partial()])
                .redistribute(tp_mesh, [Replicate()])
                .to_local()
            )

        experts.register_forward_hook(tp_reduce)
        return module


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        # [FUNCTION DOC] Builds the attention norm and attention, the feed-forward norm, and
        # either an expert layer (if experts are configured) or a dense feed-forward block,
        # recording whether experts are used.
        super().__init__()
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)

        self.has_experts = args.num_experts > 0
        if self.has_experts:
            self.expert_layer = ExpertLayer(
                args.num_experts,
                dim=args.dim,
                hidden_dim=4 * args.dim,
            )
        else:
            self.feed_forward = FeedForward(
                args.dim,
                hidden_dim=4 * args.dim,
                dropout_p=args.dropout_p,
            )

    def forward(self, x):
        # [FUNCTION DOC] Pre-norm residual block: adds attention of the normalized input, then
        # adds the expert or feed-forward output of the normalized result; returns it.
        h = x + self.attention(self.attention_norm(x))
        if self.has_experts:
            out = h + self.expert_layer(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out


# A toy transformer model, partly inspired by the nanoGPT model:
# https://github.com/karpathy/nanoGPT.
class Transformer(nn.Module):
    # [CLASS DOC] Toy decoder-style transformer: token and position embeddings, dropout, a list of
    # TransformerBlock layers, final layer norm and an output projection (optionally tied to the
    # token embedding). Provides a static parallelize helper that applies tensor and/or expert
    # parallel plans.
    def __init__(self, args: ModelArgs):
        # [FUNCTION DOC] Builds all submodules from the model args and optionally ties output and
        # embedding weights. Raises AssertionError if vocab size or max sequence length is None.
        super().__init__()
        if args.vocab_size is None:
            raise AssertionError("Expected args.vocab_size to not be None")
        if args.max_seq_len is None:
            raise AssertionError("Expected args.max_seq_len to not be None")
        self.model_args = args
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = nn.Dropout(args.dropout_p)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight
        self.checkpoint_activations = args.checkpoint_activations

    def forward(self, tokens):
        # [FUNCTION DOC] Embeds tokens, adds position embeddings, applies dropout, runs the layers
        # (with activation checkpointing when enabled), normalizes and projects to logits in
        # float. Raises AssertionError if the sequence is longer than the maximum.
        _bsz, seq_len = tokens.size()
        if seq_len > self.max_seq_len:
            raise AssertionError(
                f"Expected seq_len <= max_seq_len, got {seq_len} > {self.max_seq_len}"
            )
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)  # positional embeddings of shape (seq_len, dim)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            if self.checkpoint_activations:
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    @staticmethod
    def parallelize(
        module: "Transformer",
        tp_mesh: DeviceMesh | None,
        use_seq_parallel: bool,
        local_output_for_attn: bool = False,
        ep_mesh: DeviceMesh | None = None,
    ) -> nn.Module:
        # [FUNCTION DOC] Applies tensor parallel and/or expert parallel plans to a Transformer in
        # place and returns it. Root embeddings and norm get row-wise or sequence-parallel plans,
        # attention and feed-forward projections get column/row-wise plans, expert layers get TP,
        # EP or EP plus TP styles, and the output layer is made column-wise (re-tying weights if
        # configured). With local_output_for_attn the attention head count is divided by the
        # tensor-parallel size. Raises AssertionError if the module is not a Transformer and
        # ValueError if neither mesh is given.
        if not isinstance(module, Transformer):
            raise AssertionError(f"Requires Transformer but got {module}")
        if tp_mesh is None and ep_mesh is None:
            raise ValueError("At least one of tp_mesh or ep_mesh must be provided")

        # Parallelize the root submodules with TP.
        if tp_mesh is not None:
            if use_seq_parallel:
                root_plan = {
                    "tok_embeddings": RowwiseParallel(
                        input_layouts=Replicate(), output_layouts=Shard(1)
                    ),
                    "pos_embeddings": RowwiseParallel(
                        input_layouts=Replicate(), output_layouts=Shard(0)
                    ),
                    "norm": SequenceParallel(),
                }
            else:
                root_plan = {
                    "tok_embeddings": RowwiseParallel(
                        input_layouts=Replicate(), output_layouts=Replicate()
                    ),
                    "pos_embeddings": RowwiseParallel(
                        input_layouts=Replicate(), output_layouts=Replicate()
                    ),
                }
            parallelize_module(module, tp_mesh, root_plan)

        # Parallelize the attention and feed forward submodules.
        for layer in module.layers:
            if tp_mesh is not None:
                layer_parallelize_plan = {}
                if use_seq_parallel:
                    layer_parallelize_plan["attention"] = PrepareModuleInput(
                        input_layouts=Shard(1),
                        desired_input_layouts=Replicate(),
                    )
                    # shard the RMSNorms
                    layer_parallelize_plan["attention_norm"] = SequenceParallel()
                    layer_parallelize_plan["ffn_norm"] = SequenceParallel()
                layer_parallelize_plan["attention.wq"] = ColwiseParallel(
                    use_local_output=local_output_for_attn
                )
                layer_parallelize_plan["attention.wk"] = ColwiseParallel(
                    use_local_output=local_output_for_attn
                )
                layer_parallelize_plan["attention.wv"] = ColwiseParallel(
                    use_local_output=local_output_for_attn
                )
                layer_parallelize_plan["attention.wo"] = (
                    RowwiseParallel(output_layouts=Shard(1))
                    if use_seq_parallel
                    else RowwiseParallel()
                )

                if not layer.has_experts:
                    layer_parallelize_plan["feed_forward.w1"] = (
                        ColwiseParallel(input_layouts=Shard(1))
                        if use_seq_parallel
                        else ColwiseParallel()
                    )
                    layer_parallelize_plan["feed_forward.w2"] = (
                        RowwiseParallel(output_layouts=Shard(1))
                        if use_seq_parallel
                        else RowwiseParallel()
                    )
                elif ep_mesh is None:
                    # No EP mesh provided, use TP for experts
                    layer_parallelize_plan["expert_layer.experts"] = (
                        TensorParallelForExpert(
                            input_layouts=Shard(0),
                            output_layouts=Shard(0),
                        )
                        if use_seq_parallel
                        else TensorParallelForExpert()
                    )

                parallelize_module(layer, tp_mesh, layer_parallelize_plan)

            # EP (+ optional TP) for experts
            if ep_mesh is not None and layer.has_experts:
                if tp_mesh is not None:
                    parallelize_module(
                        layer.expert_layer,
                        ep_mesh,
                        ExpertParallelWithTP(ep_mesh, tp_mesh),
                    )
                else:
                    parallelize_module(
                        layer.expert_layer.experts, ep_mesh, ExpertParallel()
                    )

        if tp_mesh is not None:
            # Parallelize the output submodule. If weight tying is enabled,
            # we need to make sure output.weight is sharded consistently as
            # tok_embeddings.weight, at the cost of the all_reduce operation
            # using RowwiseParallel.
            output_parallelize_plan = (
                ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate(),
                )
                if use_seq_parallel
                else ColwiseParallel(output_layouts=Replicate())
            )
            parallelize_module(module.output, tp_mesh, output_parallelize_plan)

            if local_output_for_attn:
                for layer in module.layers:
                    layer.attention.n_heads = (
                        module.model_args.n_heads // tp_mesh.size()
                    )

            # Manually set output.weight so that parameters and gradients
            # are shared.
            if module.model_args.weight_tying:
                module.output.weight = module.tok_embeddings.weight

        return module


def skip_unless_torch_gpu(method: T) -> T:
    # [FUNCTION DOC] Decorator that skips a test unless at least NUM_DEVICES GPUs are available;
    # returns the wrapped method.
    """
    Test decorator which skips the test unless there's a GPU available to torch.

    >>> # xdoctest: +SKIP
    >>> @skip_unless_torch_gpu
    >>> def test_some_method(self) -> None:
    >>>   ...
    """
    # The builtin @skip_if_no_gpu relies on os.environ['WORLD_SIZE'] being set.
    return cast(T, skip_if_lt_x_gpu(NUM_DEVICES)(method))


class DTensorTestMixin:
    # [CLASS DOC] Mixin of helpers shared by the multi-process and continuous DTensor test bases:
    # device type selection, device mesh construction, per-rank seeding and op-versus-full-tensor
    # equality checks. Expects the host to provide world_size, rank and assertion methods.
    """Shared test helpers for DTensorTestBase and DTensorContinuousTestBase."""

    @property
    def is_local_tensor_enabled(self) -> bool:
        # [FUNCTION DOC] Reports whether local tensor mode is used; always False here.
        return False

    @property
    def device_type(self) -> str:
        # [FUNCTION DOC] Returns cpu when no accelerator is present or fewer devices than the
        # world size exist, otherwise the detected accelerator type.
        if (
            not (TEST_CUDA or TEST_XPU or TEST_HPU or TEST_PRIVATEUSE1)
            or DEVICE_COUNT < self.world_size
        ):
            return "cpu"
        else:
            return DEVICE_TYPE

    def build_device_mesh(self) -> DeviceMesh:
        # [FUNCTION DOC] Builds a one-dimensional device mesh over the world size on the test
        # device type.
        return init_device_mesh(self.device_type, (self.world_size,))

    def init_manual_seed_for_rank(self) -> None:
        # [FUNCTION DOC] Seeds the global random generator with this rank.
        torch.manual_seed(self.rank)

    def _test_op_on_dtensor(self, op_call, *args, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Checks that running an op on DTensor arguments, then taking full
        # tensors, equals running the op on the full-tensor versions of the arguments. Flattens
        # args, converts any DTensor to detached full tensors, compares outputs with assertEqual.
        # Fails the test on mismatch.
        """
        Checks ``op_call(dtensor).full_tensor() == op_call(dtensor.full_tensor())``.
        Unlike _test_op where the DTensor sharding is generated by DTensorConverter,
        this function takes in DTensor object directly as argument and test the equality
        of calling op on full_tensor() and DTensor.
        """
        args_flattened, args_spec = tree_flatten(args)
        full_tensor_args_flattened = tuple(
            arg.full_tensor().detach().clone() if isinstance(arg, DTensor) else arg
            for arg in args_flattened
        )
        full_tensor_args = tree_unflatten(full_tensor_args_flattened, args_spec)
        full_tensor_kwargs = {
            k: v.full_tensor() if isinstance(v, DTensor) else v
            for k, v in kwargs.items()
        }

        out_flattened, _ = tree_flatten(
            op_call(*full_tensor_args, **full_tensor_kwargs)
        )
        d_out_flattened, _ = tree_flatten(op_call(*args, **kwargs))
        d_out_full_tensor_flattened = [dt.full_tensor() for dt in d_out_flattened]
        self.assertEqual(out_flattened, d_out_full_tensor_flattened)

    # pyre-ignore[2]:
    def _test_op(self, mesh: DeviceMesh, op_call, *args, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: For every placement combination generated by DTensorConverter for
        # the args, runs the op and asserts the converter succeeded and the full tensor of the
        # result equals the plain result.
        out = op_call(*args, **kwargs)
        dtc = DTensorConverter(mesh, args, kwargs)
        for d_args, d_kwargs in dtc:
            # pyre can't find assertTrue anymore?
            self.assertEqual(dtc.successful(), True)
            d_out = op_call(*d_args, **d_kwargs)
            self.assertEqual(d_out.full_tensor(), out)

    def run_subtests(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the shared subtests helper with this test case and the given
        # arguments; returns its result.
        return run_subtests(self, *args, **kwargs)


class DTensorContinuousTestBase(DTensorTestMixin, MultiProcContinuousTest):
    # [CLASS DOC] Base for continuous multi-process DTensor tests (processes persist across
    # tests). Adds the DTensorTestMixin helpers and picks the backend for the default device type.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] Returns the default backend name for the module device type.
        backend = dist.get_default_backend_for_device(DEVICE_TYPE)
        return backend

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # [FUNCTION DOC] HELPER: Process group setup for a worker process: if an accelerator
        # exists, exits the process with the skip code when world size exceeds the device count,
        # otherwise binds the rank's device; then calls the parent initialization. Side effects:
        # sets the current device and may exit the process.
        # Set device before initializing process group to ensure
        # each rank is bound to the correct GPU. However, if world_size > device_count,
        # we skip the test.
        if torch.accelerator.is_available():
            if world_size > torch.accelerator.device_count():
                sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)
            else:
                torch.accelerator.set_device_index(rank)

        # Call parent's _init_pg to do the actual process group initialization
        super()._init_pg(rank, world_size, rdvz_file)


class LocalDTensorContinuousTestBase(DTensorContinuousTestBase):
    # [CLASS DOC] Variant of the continuous base that runs all ranks in a single process using
    # local tensors and a fake process group; no real processes are spawned. The rank property
    # returns a symbolic int that differs per simulated rank.
    @property
    def is_local_tensor_enabled(self) -> bool:
        # [FUNCTION DOC] Reports that local tensor mode is enabled (True).
        return True

    def _handle_test_skip(self, msg: str) -> None:
        # [FUNCTION DOC] HELPER: Skips the current test with the given message.
        self.skipTest(msg)

    def _get_local_tensor_mode(self):
        # [FUNCTION DOC] HELPER: Returns a LocalTensorMode over all ranks of the world size.
        return LocalTensorMode(frozenset(range(self.world_size)))

    @classmethod
    def _ensure_processes_spawned(cls):
        # [FUNCTION DOC] HELPER: Class-level setup that replaces process spawning: if not yet
        # done, resolves a pending world size from NUM_DEVICES, initializes a fake process group
        # on rank 0 with a fake store, and resets the process and queue lists. Idempotent. Side
        # effect: initializes global distributed state.
        if cls._processes_spawned:
            return
        if cls.world_size == -2:
            cls.world_size = NUM_DEVICES
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            world_size=cls.world_size,
            rank=0,
            store=store,
        )
        cls.processes = []
        cls.task_queues = []
        cls.completion_queues = []
        cls._processes_spawned = True

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Destroys the fake process group if it was created and marks processes as
        # not spawned, then calls the base unittest class teardown.
        if cls._processes_spawned:
            dist.destroy_process_group()
            cls._processes_spawned = False
        unittest.TestCase.tearDownClass()

    def setUp(self):
        # [FUNCTION DOC] Runs the unittest setup, ensures the fake group exists and disables
        # autograd record function.
        unittest.TestCase.setUp(self)
        self.__class__._ensure_processes_spawned()
        torch.autograd._enable_record_function(False)

    def tearDown(self):
        # [FUNCTION DOC] Clears the DTensor random number generator tracker, runs unittest
        # teardown and re-enables autograd record function.
        from torch.distributed.tensor import _random as random

        random._rng_tracker = None
        unittest.TestCase.tearDown(self)
        torch.autograd._enable_record_function(True)

    def __init__(self, method_name="runTest", methodName="runTest"):
        # [FUNCTION DOC] Initializes the unittest case; accepts either method_name or methodName,
        # the latter taking precedence when it is not the default.
        if methodName != "runTest":
            method_name = methodName
        unittest.TestCase.__init__(self, method_name)

    @property
    def rank(self):
        # [FUNCTION DOC] Returns a symbolic int whose per-rank local values are the rank numbers.
        return torch.SymInt(LocalIntNode({r: r for r in range(self.world_size)}))

    @rank.setter
    def rank(self, rank):
        # [FUNCTION DOC] Setter that ignores the assigned value.
        pass

    def build_device_mesh(self) -> DeviceMesh:
        # [FUNCTION DOC] Builds the device mesh with local tensor mode temporarily disabled.
        with maybe_disable_local_tensor_mode():
            return super().build_device_mesh()

    def init_manual_seed_for_rank(self) -> None:
        # [FUNCTION DOC] Seeds the global random generator with 0 for all simulated ranks.
        torch.manual_seed(0)


class DTensorTestBase(DTensorTestMixin, MultiProcessTestCase):
    # [CLASS DOC] Base class for multi-process DTensor tests. Each test runs in spawned worker
    # processes (one per rank, world size NUM_DEVICES); provides process group init/teardown and
    # the DTensorTestMixin helpers.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of ranks, NUM_DEVICES.
        return NUM_DEVICES

    @property
    def backend(self) -> str:
        # [FUNCTION DOC] Returns the default backend name for this test's device type.
        backend = dist.get_default_backend_for_device(self.device_type)
        return backend

    def init_pg(self, eager_init, backend: str | None = None) -> None:
        # [FUNCTION DOC] Initializes the default process group for this rank. Exits with the skip
        # code if an accelerator backend is needed but fewer devices than ranks exist; raises
        # RuntimeError for unsupported backend strings. For accelerator backends sets the current
        # device to the rank and, with eager_init, passes a device id so the communicator forms
        # immediately. Uses a file-based init method on the test file name. Side effect: global
        # process group.
        if backend is None:
            backend = self.backend

        requires_gpu = any(
            gpu_backend in backend for gpu_backend in ACCELERATOR_DIST_BACKENDS
        )
        if requires_gpu and torch.accelerator.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        curr_backend = dist.get_default_backend_for_device(self.device_type)

        if backend not in [
            "nccl",
            "gloo",
            "mpi",
            f"cpu:gloo,{self.device_type}:{curr_backend}",
            "cpu:gloo,cuda:ncclx",
            "cuda:ncclx",
            "hccl",
            "xccl",
            "fake",
            "cpu:gloo,xpu:xccl",
        ]:
            raise RuntimeError(f"Backend {backend} not supported!")

        device_id = None
        if requires_gpu:
            # set device for accelerator pg for collectives (nccl/xccl/hccl)
            # TODO: if users want to enable testing across hosts, we may need
            # to change this part.
            torch.accelerator.set_device_index(self.rank)
            # we only need to set device_id for nccl backend with eager init
            device_id = (
                torch.device(f"{self.device_type}:{self.rank}") if eager_init else None
            )

        # For nccl backend, bind the device to the process if device_id is not None
        # so the nccl communicator is immediately formed and we can use `ncclCommSplit`
        # for form subgroup to avoid unnecessary overhead.
        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
            device_id=device_id,
        )

    def destroy_pg(self, device_id: int | None = None) -> None:
        # [FUNCTION DOC] Destroys the process group after a barrier. If no device id is given,
        # uses the current CUDA device for cuda or the rank otherwise; on CPU uses a plain
        # barrier, else a barrier on that device.
        # Wait for all ranks to reach here before starting shutdown.
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if TEST_CUDA else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        if device_id is None:
            device_id = (
                torch.cuda.current_device() if self.device_type == "cuda" else self.rank
            )

        if self.device_type == "cpu":
            # NOTE: when `device_id` is not None, barrier() will choose the accelerator
            # of the most priority, which means if the test specifies to use CPU for
            # testing while CUDA is available on the host, the barrier() will use CUDA.
            # To avoid this and better respect `self.device_type`, we add this branch to
            # enforce barrier() to use CPU when `self.device_type` is CPU and other
            # accelerator is also available.
            dist.barrier()
        else:
            dist.barrier(device_ids=[device_id])

        dist.destroy_process_group()

    def setUp(self) -> None:
        # [FUNCTION DOC] Runs the base setup and spawns the worker processes.
        super().setUp()
        self._spawn_processes()


TestFunc = Callable[[...], object]


# wrapper to initialize comms (processgroup)
def with_comms(
    eager_init: TestFunc | bool = False,
    backend: str | None = None,
) -> TestFunc:
    # [FUNCTION DOC] Decorator factory for distributed tests. Used with or without arguments:
    # given a function directly, returns a wrapped function; given eager_init and backend options,
    # returns a decorator. The wrapper, if the test object has an init_pg method, initializes the
    # process group, runs the test, destroys the group on success, and on exception destroys the
    # process group directly and re-raises. Otherwise just runs the test.
    def decorator(func, eager_init: bool = False, backend: str | None = None):
        # [FUNCTION DOC] HELPER: Builds the wrapper for a function with the given eager init and
        # backend options.
        @wraps(func)  # pyre-ignore[6]
        def wrapper(
            self,
            *args: tuple[object],
            **kwargs: dict[str, Any],  # type: ignore[misc]
        ) -> None:
            # [FUNCTION DOC] HELPER: Test wrapper: initializes comms if supported, runs the test,
            # tears down comms (also on error). Returns None.
            # just passthrough if harness doesn't
            # support init_pg e.g., DTensorOpTestBase
            if not hasattr(self, "init_pg"):
                func(self, *args, **kwargs)
                return

            self.init_pg(eager_init, backend)

            try:
                func(self, *args, **kwargs)  # type: ignore[misc]
            except Exception as e:
                dist.destroy_process_group()
                raise e

            self.destroy_pg()

        return wrapper

    return (
        decorator(func=eager_init)
        if callable(eager_init)
        else partial(decorator, eager_init=eager_init, backend=backend)
    )


class DTensorOpTestBase(MultiThreadedTestCase):
    # [CLASS DOC] Base class for multi-threaded DTensor op tests (one thread per rank, world size
    # NUM_DEVICES). Temporarily replaces the sharding propagator's lock with a real lock during
    # each test.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of ranks, NUM_DEVICES.
        return NUM_DEVICES

    @property
    def device_type(self) -> str:
        # [FUNCTION DOC] Returns the module-level device type.
        return DEVICE_TYPE

    def build_device_mesh(self):
        # [FUNCTION DOC] Builds a one-dimensional device mesh over the world size on the device
        # type.
        return init_device_mesh(self.device_type, (self.world_size,))

    def setUp(self) -> None:
        # [FUNCTION DOC] Runs base setup, saves the sharding propagator's fake mode lock, installs
        # a real threading lock for thread safety (global side effect), and spawns threads.
        super().setUp()
        # Enable thread-safe lock for ShardingPropagator since we run
        # multi-threaded tests.
        from torch.distributed.tensor._sharding_prop import ShardingPropagator

        self._orig_fake_mode_lock = ShardingPropagator._fake_mode_lock
        ShardingPropagator._fake_mode_lock = threading.Lock()
        self._spawn_threads()

    def tearDown(self) -> None:
        # [FUNCTION DOC] Restores the saved sharding propagator lock and runs base teardown.
        # Restore the original (no-op) lock
        from torch.distributed.tensor._sharding_prop import ShardingPropagator

        ShardingPropagator._fake_mode_lock = self._orig_fake_mode_lock
        super().tearDown()


# This is a class for converting args/kwargs of an op into distributed args/kwargs
class DTensorConverter:
    # [CLASS DOC] Iterator that converts the tensor arguments of an op call into DTensors,
    # yielding one converted (args, kwargs) pair for each combination of sharding choices
    # (replicate or even shard of each dimension). Attributes: hit and miss counters of converted
    # and skipped tensors, the mesh, flattened args and kwargs with their tree specs, and the
    # combination iterator. Use successful to check that something was converted and nothing
    # skipped.
    def __init__(
        self,
        mesh: DeviceMesh,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        replicate_only: bool = False,
    ) -> None:
        # [FUNCTION DOC] Flattens args and kwargs, and builds the cartesian product of per-tensor
        # sharding choices (replicate only if requested). Initializes counters to zero.
        self.hit = 0
        self.miss = 0
        self.mesh = mesh
        self.args = args
        self.kwargs = kwargs
        self.replicate_only = replicate_only
        flatten_args, flatten_args_spec = tree_flatten(args)
        flatten_kwargs, flatten_kwargs_spec = tree_flatten(kwargs)

        self.flatten_args: list[object] = flatten_args
        self.flatten_args_spec: TreeSpec = flatten_args_spec
        self.flatten_kwargs: list[object] = flatten_kwargs
        self.flatten_kwargs_spec: TreeSpec = flatten_kwargs_spec

        choices_for_args = [
            self.gen_sharding_choices_for_arg(arg)
            for arg in self.flatten_args
            if isinstance(arg, torch.Tensor)
        ]

        choices_for_args.extend(
            self.gen_sharding_choices_for_arg(arg)
            for arg in self.flatten_kwargs
            if isinstance(arg, torch.Tensor)
        )

        self.sharding_combs: Iterator[Sequence[Placement]] = iter(
            itertools.product(*choices_for_args)
        )

    def successful(self) -> bool:
        # [FUNCTION DOC] Returns True if at least one tensor was converted and none was skipped.
        return self.hit > 0 and self.miss == 0

    def is_supported_tensor(self, t: torch.Tensor) -> bool:
        # [FUNCTION DOC] Returns whether a tensor can be wrapped as a DTensor: False for sparse,
        # mkldnn, quantized, nested, functional, negative or conjugate views, and lazy or meta
        # device tensors.
        # TODO: dist tensor need to support quantized and sparse
        # tensors, quantized tensor might be relatively easy, but
        # sparse tensor have special layouts that we need to possibly
        # deal with, until we are clear about them, we don't officially
        # support them.
        return not any(
            [
                t.is_sparse_csr,
                t.is_sparse,
                t.is_mkldnn,
                t.is_quantized,
                t.is_nested,
                torch._is_functional_tensor(t),
                t.is_neg(),
                t.is_conj(),
                t.device.type in ("lazy", "meta"),
                # We need a way to test if a tensor is batched but there
                # is no official APi to do it
                # torch._C._is_batched(t),
            ]
        )

    def gen_sharding_choices_for_arg(self, arg: torch.Tensor) -> Sequence[Placement]:
        # [FUNCTION DOC] Returns the candidate placements for a tensor: replicate, plus shard on
        # every dimension whose size is greater than one and divisible by the mesh size (none for
        # bool tensors, which are treated as replicated). Only replicate if replicate_only is set.
        # If replicate_only is set, only use Replicate placement
        if self.replicate_only:
            return [Replicate()]

        mesh_size = self.mesh.size()
        sharding_choices: list[Placement] = [Replicate()]
        # c10d collective does not support bool tensor
        # for bool tensor we treat it as replicated
        if arg.dtype != torch.bool:
            # only generating choices with: replicate, or sharding
            # evenly on a dimension that could be sharded
            sharding_choices = sharding_choices + [
                Shard(i)
                for i, s in enumerate(arg.shape)
                if s > 1 and s % mesh_size == 0
            ]
        # TODO: add multi mesh choices
        # all_choices = itertools.product(
        #     *(self.mesh.ndim * [sharding_choices])
        # )
        return sharding_choices

    def __iter__(self) -> "DTensorConverter":
        # [FUNCTION DOC] Returns the converter itself as the iterator.
        return self

    def __next__(self) -> tuple[tuple[object, ...], dict[str, object]]:
        # [FUNCTION DOC] Takes the next sharding combination and returns converted args and kwargs
        # rebuilt with their original structure; each tensor is distributed through
        # to_dist_tensor. Raises StopIteration when combinations are exhausted.
        try:
            next_sharding_choices = next(self.sharding_combs)
            idx = 0

            new_args: list[object] = []
            for arg in self.flatten_args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(
                        self.to_dist_tensor(
                            arg, self.mesh, [next_sharding_choices[idx]]
                        )
                    )
                    idx += 1
                else:
                    new_args.append(arg)

            new_kwargs: list[object] = []
            for arg in self.flatten_kwargs:
                if isinstance(arg, torch.Tensor):
                    new_kwargs.append(
                        self.to_dist_tensor(
                            arg, self.mesh, [next_sharding_choices[idx]]
                        )
                    )
                    idx += 1
                else:
                    new_kwargs.append(arg)

            return (
                tree_unflatten(new_args, self.flatten_args_spec),
                tree_unflatten(new_kwargs, self.flatten_kwargs_spec),
            )
        except StopIteration as e:
            raise StopIteration from e

    def to_dist_tensor(
        self, t: torch.Tensor, mesh: DeviceMesh, placements: list[Placement]
    ) -> torch.Tensor:
        # [FUNCTION DOC] Converts a tensor to a DTensor. Plain tensors, parameters and local
        # tensors that are supported are distributed (scalars replicated on all mesh dims,
        # parameters re-wrapped) and counted as hit; unsupported ones and other tensor-like
        # subclasses are returned unchanged and counted as miss. Raises RuntimeError for non-
        # tensor inputs.
        if type(t) is torch.Tensor or type(t) is nn.Parameter or type(t) is LocalTensor:
            if self.is_supported_tensor(t):
                self.hit += 1
                if t.ndim == 0:
                    # scalar tensor by default will be replicated
                    r = distribute_tensor(t, mesh, [Replicate()] * mesh.ndim)
                else:
                    # distribute non-scalar tensors
                    r = distribute_tensor(t, mesh, placements)
                if isinstance(t, nn.Parameter):
                    r = nn.Parameter(  # type: ignore[assignment]
                        r, requires_grad=r.requires_grad
                    )
                return r
            else:
                self.miss += 1
                return t
        elif torch.overrides.is_tensor_like(t):
            # Blindly converting tensor subclasses to dist tensor can cause
            # unpredictable problems, we explicitly disable this conversion
            # for now (i.e. we don't support DTensor holding tensor subclass
            # until there's a strong reason later).
            self.miss += 1
            return t
        else:
            raise RuntimeError(f"Trying to convert to DTensor, but got {type(t)}")


class LocalDTensorOpTestBase(DTensorOpTestBase):
    # [CLASS DOC] Op test base that runs all ranks in one thread using local tensors and a fake
    # process group; thread spawning and process spawning are no-ops. The rank property returns a
    # symbolic per-rank int.
    @property
    def is_local_tensor_enabled(self) -> bool:
        # [FUNCTION DOC] Reports that local tensor mode is enabled (True).
        return True

    def _handle_test_skip(self, msg: str) -> None:
        # [FUNCTION DOC] HELPER: Skips the current test with the given message.
        self.skipTest(msg)

    def _get_local_tensor_mode(self):
        # [FUNCTION DOC] HELPER: Returns a LocalTensorMode over all ranks of the world size.
        return LocalTensorMode(frozenset(range(self.world_size)))

    def setUp(self) -> None:
        # [FUNCTION DOC] Runs base setup and disables autograd record function.
        super().setUp()
        torch.autograd._enable_record_function(False)

    def tearDown(self) -> None:
        # [FUNCTION DOC] Clears the DTensor random number generator tracker, runs base teardown
        # and re-enables autograd record function.
        from torch.distributed.tensor import _random as random

        random._rng_tracker = None
        super().tearDown()
        torch.autograd._enable_record_function(True)

    @property
    def rank(self):
        # [FUNCTION DOC] Returns a symbolic int whose per-rank local values are the rank numbers.
        return torch.SymInt(LocalIntNode({r: r for r in range(self.world_size)}))

    @rank.setter
    def rank(self, rank):
        # [FUNCTION DOC] Setter that ignores the assigned value.
        pass

    def join_or_run(self, fn):
        # [FUNCTION DOC] Wraps a zero-argument test function as a bound method of this test;
        # returns it.
        @wraps(fn)
        def wrapper(self):
            # [FUNCTION DOC] HELPER: Calls the wrapped function ignoring the instance.
            fn()

        return types.MethodType(wrapper, self)

    def build_device_mesh(self) -> DeviceMesh:
        # [FUNCTION DOC] Builds the device mesh with local tensor mode temporarily disabled.
        with maybe_disable_local_tensor_mode():
            return super().build_device_mesh()

    def init_pg(self, eager_init, backend: str | None = None) -> None:
        # [FUNCTION DOC] Initializes a fake process group with rank 0 and the world size and
        # stores the default group (arguments eager_init and backend are ignored).
        dist.init_process_group("fake", rank=0, world_size=self.world_size)
        self._pg = dist.distributed_c10d._get_default_group()

    def destroy_pg(self, device_id: int | None = None) -> None:
        # [FUNCTION DOC] Destroys the stored process group and clears the reference.
        dist.destroy_process_group(self._pg)
        self._pg = None

    def _spawn_processes(self) -> None:
        # [FUNCTION DOC] No-op: no processes are spawned.
        pass

    def _spawn_threads(self) -> None:
        # [FUNCTION DOC] No-op: no threads are spawned.
        pass

    def run_test(self, test_name: str, parent_pipe) -> None:
        # [FUNCTION DOC] Runs the named test method directly in this process; the pipe is unused.
        getattr(self, test_name)()

    def init_manual_seed_for_rank(self) -> None:
        # [FUNCTION DOC] Seeds the global random generator with 0 for all simulated ranks.
        torch.manual_seed(0)


class LocalDTensorTestBase(DTensorTestBase):
    # [CLASS DOC] Process-based DTensor test base run in local tensor mode: a single process with
    # a fake process group simulates all ranks, with no process spawning. Rank is a symbolic per-
    # rank int.
    @property
    def is_local_tensor_enabled(self) -> bool:
        # [FUNCTION DOC] Reports that local tensor mode is enabled (True).
        return True

    def _handle_test_skip(self, msg: str) -> None:
        # [FUNCTION DOC] HELPER: Skips the current test with the given message.
        self.skipTest(msg)

    def _get_local_tensor_mode(self):
        # [FUNCTION DOC] HELPER: Returns a LocalTensorMode over all ranks of the world size.
        return LocalTensorMode(frozenset(range(self.world_size)))

    def setUp(self) -> None:
        # [FUNCTION DOC] Runs base setup and disables autograd record function.
        super().setUp()
        torch.autograd._enable_record_function(False)

    def tearDown(self) -> None:
        # [FUNCTION DOC] Clears the DTensor random number generator tracker, runs base teardown
        # and re-enables autograd record function.
        from torch.distributed.tensor import _random as random

        random._rng_tracker = None
        super().tearDown()
        torch.autograd._enable_record_function(True)

    @property
    def rank(self):
        # [FUNCTION DOC] Returns a symbolic int whose per-rank local values are the rank numbers.
        return torch.SymInt(LocalIntNode({r: r for r in range(self.world_size)}))

    @rank.setter
    def rank(self, rank):
        # [FUNCTION DOC] Setter that ignores the assigned value.
        pass

    def join_or_run(self, fn):
        # [FUNCTION DOC] Wraps a zero-argument test function as a bound method of this test;
        # returns it.
        @wraps(fn)
        def wrapper(self):
            # [FUNCTION DOC] HELPER: Calls the wrapped function ignoring the instance.
            fn()

        return types.MethodType(wrapper, self)

    def build_device_mesh(self) -> DeviceMesh:
        # [FUNCTION DOC] Builds the device mesh with local tensor mode temporarily disabled.
        with maybe_disable_local_tensor_mode():
            return super().build_device_mesh()

    def init_pg(self, eager_init, backend: str | None = None) -> None:
        # [FUNCTION DOC] Initializes a fake process group with rank 0 and the world size and
        # stores the default group (arguments eager_init and backend are ignored).
        dist.init_process_group("fake", rank=0, world_size=self.world_size)
        self._pg = dist.distributed_c10d._get_default_group()

    def destroy_pg(self, device_id: int | None = None) -> None:
        # [FUNCTION DOC] Destroys the stored process group and clears the reference.
        dist.destroy_process_group(self._pg)
        self._pg = None

    def _spawn_processes(self) -> None:
        # [FUNCTION DOC] No-op: no processes are spawned.
        pass

    def run_test(self, test_name: str, parent_pipe) -> None:
        # [FUNCTION DOC] Runs the named test method directly in this process; the pipe is unused.
        getattr(self, test_name)()

    def init_manual_seed_for_rank(self) -> None:
        # [FUNCTION DOC] Seeds the global random generator with 0 for all simulated ranks.
        torch.manual_seed(0)


def make_wrapped(fn, ctxs):
    # [FUNCTION DOC] HELPER: Wraps a test function so that it resets dynamo, enters each given
    # context (calling it with the test instance if it is callable, else using it directly) on an
    # exit stack, runs the test, and always closes the stack. Returns the wrapper.
    @functools.wraps(fn)
    def wrapped(self):
        # [FUNCTION DOC] HELPER: Wrapper body described above; returns the test function's result.
        torch._dynamo.reset()
        stack = contextlib.ExitStack()
        for ctx in ctxs:
            if callable(ctx):
                stack.enter_context(ctx(self))
            else:
                stack.enter_context(ctx)
        try:
            out = fn(self)
        finally:
            stack.close()
        return out

    return wrapped


def create_local_tensor_test_class(
    orig_cls, skipped_tests=None, base_class=LocalDTensorTestBase
):
    # [FUNCTION DOC] Creates a subclass of a test class that runs its tests in local tensor mode.
    # Copies the original class namespace; names in skipped_tests are replaced by functions that
    # skip, test methods are wrapped to run inside a LocalTensorMode, and the new class (named
    # with a WithLocalTensor suffix, deriving from base_class and the original bases) has its file
    # attribute set to this module's. Returns the new class.
    if skipped_tests is None:
        skipped_tests = []

    dct = orig_cls.__dict__.copy()
    for name in list(dct.keys()):
        fn = dct[name]
        if not callable(fn):
            continue
        elif name in skipped_tests:
            dct[name] = lambda self: self.skipTest("Skipped test")
        elif name.startswith("test_"):
            ctxs = [
                lambda test: test._get_local_tensor_mode(),
            ]
            dct[name] = make_wrapped(fn, ctxs)

    cls = type(
        orig_cls.__name__ + "WithLocalTensor",
        (base_class,) + orig_cls.__bases__,
        dct,
    )
    cls.__file__ = __file__
    return cls


@maybe_run_for_local_tensor
def map_local_tensor_for_rank(tensor, rank, func):
    # [FUNCTION DOC] Applies func to a tensor and a rank (run once per rank under local tensor
    # mode via the decorator) and returns the result.
    return func(tensor, rank)


@maybe_run_for_local_tensor
def map_local_for_rank(rank, func):
    # [FUNCTION DOC] Applies func to a rank (run once per rank under local tensor mode via the
    # decorator) and returns the result.
    return func(rank)


def reduce_local_int(val, func):
    # [FUNCTION DOC] Applies func to the mapping of per-rank local integer values of a symbolic
    # int and returns the result.
    return func(val.node._local_ints)


def _convert_shard_order_dict_to_ShardOrder(shard_order):
    # [FUNCTION DOC] HELPER: Converts a mapping from tensor dimension to mesh dimensions into a
    # tuple of ShardOrderEntry objects, in mapping order.
    """Convert shard_order dict to ShardOrder"""
    return tuple(
        ShardOrderEntry(tensor_dim=tensor_dim, mesh_dims=tuple(mesh_dims))
        for tensor_dim, mesh_dims in shard_order.items()
    )


# TODO(zpcore): remove once the native redistribute supports shard_order arg
def redistribute(
    dtensor_input,
    device_mesh,
    placements,
    shard_order,
    use_graph_based_transform=True,
):
    # [FUNCTION DOC] Test-only redistribution that supports an explicit shard order, forward only.
    # Derives placements from shard_order if placements is None, copies the DTensor spec with the
    # new placements and shard order (empty if None) and returns the input unchanged if the spec
    # is equal. Otherwise redistributes the local tensor between the specs, wraps it as a DTensor
    # on the mesh, sets its spec and returns it.
    """
    wrapper function to support shard_order for redistribution
    This is a simpler version of Redistribute, only considers the forward.
    """
    if placements is None:
        placements = shard_order_to_placement(shard_order, device_mesh)
    placements = tuple(placements)
    old_spec = dtensor_input._spec
    new_spec = copy.deepcopy(old_spec)
    new_spec.placements = placements
    if shard_order is not None:
        new_spec.shard_order = shard_order
    else:
        new_spec.shard_order = ()
    if old_spec == new_spec:
        return dtensor_input
    dtensor_input = DTensor.from_local(
        redistribute_local_tensor(
            dtensor_input.to_local(),
            old_spec,
            new_spec,
            use_graph_based_transform=use_graph_based_transform,
        ),
        device_mesh,
    )
    dtensor_input._spec = copy.deepcopy(new_spec)
    return dtensor_input  # returns DTensor


# TODO(zpcore): remove once the native distribute_tensor supports
# shard_order arg
def patched_distribute_tensor(
    input_tensor,
    device_mesh,
    placements,
    shard_order,
    use_graph_based_transform=True,
    src_data_rank: int | None = 0,
):
    # [FUNCTION DOC] Test-only version of distribute_tensor supporting shard_order. Derives
    # placements from shard_order if needed, distributes the tensor, disables strided shard as
    # shard order in its spec and re-runs the spec post-init, then redistributes to the requested
    # shard order. Returns the DTensor.
    """wrapper function to support shard_order for tensor distribution"""
    if placements is None:
        placements = shard_order_to_placement(shard_order, device_mesh)
    placements = tuple(placements)
    tensor_dt = distribute_tensor(
        input_tensor, device_mesh, placements, src_data_rank=src_data_rank
    )
    # Do not consider _StridedShard to express shard order
    tensor_dt._spec.use_strided_shard_as_shard_order = False
    tensor_dt._spec.__post_init__()
    # fix the shard order
    return redistribute(
        tensor_dt, device_mesh, placements, shard_order, use_graph_based_transform
    )


# TODO(zpcore): remove once the native redistribute supports shard_order arg
def make_full_tensor(dtensor_input):
    # [FUNCTION DOC] Returns the full (replicated) local tensor of a DTensor by redistributing to
    # replicated placements with an empty shard order.
    """wrapper function to support DTensor.full_tensor"""
    return redistribute(
        dtensor_input, dtensor_input.device_mesh, placements=None, shard_order=()
    ).to_local()


def shard_order_to_placement(shard_order, mesh):
    # [FUNCTION DOC] Converts a shard order to a tuple of placements with one entry per mesh
    # dimension: Shard of the tensor dimension for mesh dimensions listed, Replicate otherwise
    # (all Replicate if shard_order is None).
    """convert shard_order to placement with only Replicate() and Shard()"""
    placements: list[Any] = [Replicate() for _ in range(mesh.ndim)]
    if shard_order is not None:
        for entry in shard_order:
            tensor_dim = entry.tensor_dim
            mesh_dims = entry.mesh_dims
            for mesh_dim in mesh_dims:
                placements[mesh_dim] = Shard(tensor_dim)
    return tuple(placements)


def generate_shard_orders(mesh, tensor_rank):
    # [FUNCTION DOC] Generator over all possible shard orders of a tensor of the given rank on a
    # mesh: for each permutation of mesh dimensions, each way to split it into consecutive non-
    # empty segments, and each choice of tensor dimensions for the segments, yields a tuple of
    # ShardOrderEntry objects. Raises AssertionError if internal counts disagree.
    # Generate all possible sharding placement of tensor with rank
    # `tensor_rank` over mesh.
    def _split_list(lst: list, N: int):
        # [FUNCTION DOC] HELPER: Generator that yields every way of splitting a list into N
        # consecutive non-empty parts.
        def compositions(n: int, k: int):
            # [FUNCTION DOC] HELPER: Generator that yields lists of k positive integers summing to
            # n.
            # yields lists of length k, positive ints summing to n
            for cuts in itertools.combinations(range(1, n), k - 1):
                # add 0 and n as sentinels, then take consecutive differences
                yield [b - a for a, b in itertools.pairwise((0, *cuts, n))]

        length = len(lst)
        for comp in compositions(length, N):
            result = []
            start = 0
            for size in comp:
                result.append(lst[start : start + size])
                start += size
            yield result

    all_mesh = list(range(mesh.ndim))
    all_device_order = list(itertools.permutations(all_mesh))
    for device_order in all_device_order:
        # split on device orders, and assign each device order segment to a tensor dim
        for num_split in range(1, mesh.ndim + 1):
            for splitted_list in _split_list(list(range(mesh.ndim)), num_split):
                for tensor_dims in itertools.combinations(
                    range(tensor_rank), len(splitted_list)
                ):
                    shard_order = {}
                    if len(tensor_dims) != len(splitted_list):
                        raise AssertionError(
                            f"Expected len(tensor_dims) == len(splitted_list), "
                            f"got {len(tensor_dims)} != {len(splitted_list)}"
                        )
                    for tensor_dim, mesh_dims in zip(tensor_dims, splitted_list):
                        shard_order[tensor_dim] = device_order[
                            mesh_dims[0] : mesh_dims[-1] + 1
                        ]
                    yield _convert_shard_order_dict_to_ShardOrder(shard_order)


def validate_sharding_rule_sample(
    op, full_args, full_kwargs, input_placements, output_placements, device_mesh
):
    # [FUNCTION DOC] Checks one sharding rule sample: distributes the input tensors with the given
    # per-tensor placements, runs the op on full arguments and on the local shards, wraps each
    # local output with the given output placements, redistributes to replicated and compares with
    # the reference (shape and allclose with NaN equality). Returns True if all outputs match,
    # otherwise False.
    from torch.utils import _pytree as pytree

    # Extract tensors from args in order, pair with placements
    full_tensors = [
        a for a in pytree.tree_leaves(full_args) if isinstance(a, torch.Tensor)
    ]
    full_tensors += [
        a for a in pytree.tree_leaves(full_kwargs) if isinstance(a, torch.Tensor)
    ]

    dtensors = [
        distribute_tensor(t, device_mesh, (p,))
        for t, p in zip(full_tensors, input_placements)
    ]

    # Build sharded args by replacing tensors with their sharded local versions
    dtensor_idx = 0

    def _to_local_shard(a):
        # [FUNCTION DOC] HELPER: Replaces the next tensor argument with the local shard of the
        # matching distributed tensor (tracking an index in the enclosing scope); non-tensor
        # values are returned unchanged.
        nonlocal dtensor_idx
        if isinstance(a, torch.Tensor):
            local = dtensors[dtensor_idx].to_local()
            dtensor_idx += 1
            return local
        return a

    local_args, local_kwargs = pytree.tree_map(
        _to_local_shard, (full_args, full_kwargs)
    )

    # run and compare
    ref_output = op(*full_args, **full_kwargs)
    local_output = op(*local_args, **local_kwargs)

    ref_tensors = [
        t for t in pytree.tree_leaves(ref_output) if isinstance(t, torch.Tensor)
    ]
    local_tensors = [
        t for t in pytree.tree_leaves(local_output) if isinstance(t, torch.Tensor)
    ]

    for ref, local, plc in zip(ref_tensors, local_tensors, output_placements):
        dt = DTensor.from_local(local, device_mesh, (plc,))
        full = dt.redistribute(device_mesh, (Replicate(),)).to_local()
        if ref.shape != full.shape or not torch.allclose(
            ref, full, atol=1e-5, rtol=1e-5, equal_nan=True
        ):
            return False
    return True


def validate_sharding_rule_sample_backward(
    op, full_args, full_kwargs, input_placements, device_mesh
):
    # [FUNCTION DOC] Verifies backward gradients for a sharding strategy. Returns None (skip) if
    # there are no floating tensors, the op produces no differentiable output, the reference
    # backward raises RuntimeError, or the DTensor path reports an unregistered strategy or mixed
    # tensor and DTensor inputs; other exceptions propagate. Otherwise runs backward on the sum of
    # outputs for plain and distributed clones and returns True if all gradients match in shape
    # and values, False otherwise. Raises AssertionError if the placement count differs from the
    # tensor count.
    """Verify backward gradient correctness for a sharding strategy.

    Returns None if the op is not differentiable or lacks backward DTensor
    support (skip), True if gradients match, False if they differ.
    """
    from torch.utils import _pytree as pytree

    full_tensors = [
        a for a in pytree.tree_leaves(full_args) if isinstance(a, torch.Tensor)
    ]
    full_tensors += [
        a for a in pytree.tree_leaves(full_kwargs) if isinstance(a, torch.Tensor)
    ]

    if not any(t.is_floating_point() for t in full_tensors):
        return None

    def _clone_with_grad(tensors):
        # [FUNCTION DOC] HELPER: Returns detached clones of the tensors, with grad enabled for
        # floating-point ones.
        out = []
        for t in tensors:
            c = t.detach().clone()
            if c.is_floating_point():
                c.requires_grad_(True)
            out.append(c)
        return out

    def _replace_tensors(args, kwargs, replacements):
        # [FUNCTION DOC] HELPER: Returns args and kwargs with every tensor replaced, in order, by
        # the replacement list entries.
        idx = 0

        def _replace(a):
            # [FUNCTION DOC] HELPER: Replaces one tensor with the next replacement, advancing a
            # counter in the enclosing scope; other values unchanged.
            nonlocal idx
            if isinstance(a, torch.Tensor):
                r = replacements[idx]
                idx += 1
                return r
            return a

        return pytree.tree_map(_replace, (args, kwargs))

    def _run_backward(output):
        # [FUNCTION DOC] HELPER: Runs backward on the sum of the floating-point outputs that
        # require grad. Returns False if there are none, else True.
        out_tensors = [
            t
            for t in pytree.tree_leaves(output)
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.requires_grad
        ]
        if not out_tensors:
            return False
        sum(t.sum() for t in out_tensors).backward()
        return True

    # Reference backward
    ref_tensors = _clone_with_grad(full_tensors)
    ref_args, ref_kwargs = _replace_tensors(full_args, full_kwargs, ref_tensors)
    try:
        if not _run_backward(op(*ref_args, **ref_kwargs)):
            return None
    except RuntimeError:
        return None

    # DTensor backward
    if len(input_placements) != len(full_tensors):
        raise AssertionError(
            f"placement/tensor count mismatch: {len(input_placements)} vs {len(full_tensors)}"
        )
    dt_tensors = [
        distribute_tensor(c, device_mesh, (p,))
        for c, p in zip(_clone_with_grad(full_tensors), input_placements, strict=True)
    ]
    dt_args, dt_kwargs = _replace_tensors(full_args, full_kwargs, dt_tensors)
    try:
        if not _run_backward(op(*dt_args, **dt_kwargs)):
            return None
    except Exception as e:
        msg = str(e)
        if (
            "does not have a sharding strategy registered" in msg
            or "got mixed torch.Tensor and DTensor" in msg
        ):
            return None
        raise

    # Compare gradients
    for ref_t, dt_t in zip(ref_tensors, dt_tensors, strict=True):
        if not ref_t.is_floating_point() or ref_t.grad is None:
            continue
        if dt_t.grad is None:
            return False
        dt_grad = (
            dt_t.grad.full_tensor() if isinstance(dt_t.grad, DTensor) else dt_t.grad
        )
        if ref_t.grad.shape != dt_grad.shape:
            return False
        if not torch.allclose(
            ref_t.grad, dt_grad, atol=1e-5, rtol=1.3e-6, equal_nan=True
        ):
            return False
    return True


@contextlib.contextmanager
def op_strategy_context(op_overload, strategy_func, schema_info=None):
    # [FUNCTION DOC] Context manager that temporarily registers a sharding strategy function for
    # an operator overload on the DTensor dispatcher's sharding propagator. Saves and removes any
    # existing strategy and schema info, registers the new strategy, yields, then in a finally
    # clause restores the original entries (or removes the new ones) and clears the sharding
    # propagation cache. Yields None.
    """
    Context manager for setting and clearing op strategies.
    Args:
        op_overload: The operator overload to set or clear the strategy for.
        strategy_func: The strategy function to set for the operator overload.
        schema_info: Optional schema information for the operator overload.
    Yields:
        None
    """
    from torch.distributed.tensor._ops.utils import register_op_strategy
    from torch.distributed.tensor.debug import _clear_sharding_prop_cache

    propagator = DTensor._op_dispatcher.sharding_propagator
    _origin_op_strategy_funcs = None
    _origin_op_strategy_schema = None
    try:
        # register the op strategy
        if op_overload in propagator.op_strategy_funcs:
            _origin_op_strategy_funcs = propagator.op_strategy_funcs[op_overload]
            del propagator.op_strategy_funcs[op_overload]
        if op_overload in propagator.op_to_schema_info:
            _origin_op_strategy_schema = propagator.op_to_schema_info[op_overload]
            del propagator.op_to_schema_info[op_overload]
        register_op_strategy(op_overload, schema_info=schema_info)(strategy_func)
        yield
    finally:
        # clear this op strategy cache
        if _origin_op_strategy_funcs is None:
            if op_overload in propagator.op_strategy_funcs:
                del propagator.op_strategy_funcs[op_overload]
        else:
            propagator.op_strategy_funcs[op_overload] = _origin_op_strategy_funcs
        if _origin_op_strategy_schema is None:
            if op_overload in propagator.op_to_schema_info:
                del propagator.op_to_schema_info[op_overload]
        else:
            propagator.op_to_schema_info[op_overload] = _origin_op_strategy_schema
        _clear_sharding_prop_cache()
