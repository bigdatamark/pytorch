# [MODULE DOC] Helpers for the distributed (DDP/FSDP) dynamo benchmark driver. Provides
# process-group setup/teardown, toy models, a torchbench model loader, a train-step function,
# and FSDP wrapping with optional activation checkpointing. Imports torchbench helper with a
# relative-then-absolute fallback; no other import-time side effects.
import argparse
import functools
import importlib
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.testing import reduce_to_scalar_loss
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


try:
    from .torchbench import setup_torchbench_cwd
except ImportError:
    from torchbench import setup_torchbench_cwd


def setup(rank, world_size):
    # [FUNCTION DOC] Initialize the NCCL process group. Fills MASTER_ADDR, MASTER_PORT, RANK
    # and WORLD_SIZE in the environment with defaults (localhost, 12355, 0, 1) when unset.
    # The rank and world_size arguments are ignored; values come only from the environment.
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    os.environ["RANK"] = os.getenv("RANK", "0")
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
    dist.init_process_group("nccl")


def cleanup():
    # [FUNCTION DOC] Destroy the default distributed process group.
    dist.destroy_process_group()


class CustomLinear(torch.nn.Module):
    # [CLASS DOC] Module holding a single learnable matrix parameter of shape (a, b), applied
    # to the input by matrix multiplication. Not used elsewhere in this file.
    def __init__(self, a, b):
        # [FUNCTION DOC] Create a randomly initialized weight parameter with a rows and b columns.
        super().__init__()
        self.weight = nn.Parameter(torch.randn(a, b))

    def forward(self, x):
        # [FUNCTION DOC] Return the matrix product of input x with the weight parameter.
        return torch.mm(x, self.weight)


class MyModule(torch.nn.Module):
    # [CLASS DOC] Block of a linear layer (a to b features) followed by ReLU; used as the FSDP
    # wrapping unit for the toy model.
    def __init__(self, a, b):
        # [FUNCTION DOC] Build the sequential linear + ReLU network for sizes a and b.
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
        )

    def forward(self, x):
        # [FUNCTION DOC] Apply the linear + ReLU network to x and return the result.
        return self.net(x)


class ToyModel(nn.Module):
    # [CLASS DOC] Large toy network: a stack of wide linear/ReLU layers and MyModule blocks
    # (widths 10 -> 10000 -> 1000), ending in a 5-way linear. Takes inputs with 10 features.
    def __init__(self):
        # [FUNCTION DOC] Build the fixed sequential stack of layers; no arguments.
        super().__init__()
        self.net = nn.Sequential(
            *[nn.Linear(10, 10000), nn.ReLU()]
            + [nn.Linear(10000, 10000), nn.ReLU()]
            + [MyModule(10000, 10000)]
            + [MyModule(10000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [nn.Linear(1000, 5)]
        )

    def forward(self, x):
        # [FUNCTION DOC] Run x through the sequential stack and return the output.
        return self.net(x)


def model_iter_fn(model, example_inputs, collect_outputs=False):
    # [FUNCTION DOC] One training iteration: forward pass, reduce outputs to a scalar loss,
    # and backward pass. Returns the outputs only if collect_outputs is true, else None.
    outputs = model(*example_inputs)
    loss = reduce_to_scalar_loss(outputs)
    loss.backward()
    if collect_outputs:
        return outputs


def get_model(args):
    # [FUNCTION DOC] Build the (model, example_inputs) pair from parsed args. With a torchbench
    # model name, changes to the torchbench directory, imports that model, and instantiates it
    # in train mode with the given device and batch size. With the toy flag, builds the toy model
    # with a random input of shape 20 by 10. Otherwise raises argparse.ArgumentError.
    if args.torchbench_model:
        setup_torchbench_cwd()
        module = importlib.import_module(
            f"torchbenchmark.models.{args.torchbench_model}"
        )
        benchmark_cls = getattr(module, "Model", None)
        bm = benchmark_cls(test="train", device=args.device, batch_size=args.batch_size)
        model, inputs = bm.get_module()
    elif args.toy_model:
        model = ToyModel()
        inputs = (torch.randn(20, 10),)
    else:
        raise argparse.ArgumentError(
            args.torchbench_model, message="Must specify a model"
        )

    return model, inputs


def fsdp_checkpointing_base(model, blocks):
    # [FUNCTION DOC] Apply non-reentrant activation checkpointing (no CPU offload) in place to
    # every submodule that is an instance of the given block types. Returns None.
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_fn(submodule):
        # [FUNCTION DOC] HELPER: Predicate for checkpointing: true if submodule is one of blocks.
        return isinstance(submodule, blocks)

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


MODEL_FSDP_WRAP = {
    "toy_model": (MyModule,),
}


def apply_fsdp(args, model, use_checkpointing=False, use_wrap_policy=True):
    # [FUNCTION DOC] Wrap the model in FSDP with original parameters kept. Block types come from
    # the wrap table, keyed by toy model if the model is the toy class, else by the torchbench
    # model name in args (raises KeyError if absent). Uses a module wrap policy over those
    # blocks when use_wrap_policy is true. Optionally applies activation checkpointing.
    # Returns the wrapped model.
    wrap_policy = None
    blocks = MODEL_FSDP_WRAP[
        "toy_model" if model.__class__ is ToyModel else args.torchbench_model
    ]
    if use_wrap_policy:
        wrap_policy = ModuleWrapPolicy(blocks)

    model = FSDP(model, auto_wrap_policy=wrap_policy, use_orig_params=True)
    if use_checkpointing:
        fsdp_checkpointing_base(model, blocks)
    return model
