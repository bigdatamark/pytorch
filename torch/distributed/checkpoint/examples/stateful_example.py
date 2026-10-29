# mypy: allow-untyped-defs
# [MODULE DOC] Runnable example of checkpointing FSDP model and optimizer through the stateful interface.
# Trains briefly, saves with dcp.save, rebuilds model/optimizer, loads with dcp.load, trains again. Run as a
# script: it deletes CHECKPOINT_DIR and spawns one process per CUDA device.
# Owner(s): ["oncall: distributed"]

# pyre-unsafe


import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


CHECKPOINT_DIR = f"~/{os.environ.get('LOGNAME', '')}/checkpoint"


class Model(torch.nn.Module):
    # [CLASS DOC] Small four-stage MLP (linear/relu blocks) used by the example; also provides random CUDA inputs.
    def __init__(self) -> None:
        # [FUNCTION DOC] Seeds the RNG and builds the four stages.
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        # [FUNCTION DOC] Runs the input through the four stages in order and returns the output.
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        # [FUNCTION DOC] Returns a random 8x8 CUDA tensor.
        return torch.rand(8, 8, device="cuda")


def _make_stateful(model, optim):
    # [FUNCTION DOC] HELPER: Patches model and optimizer in place so they expose state-dict methods usable by checkpointing.
    _patch_model_state_dict(model)
    _patch_optimizer_state_dict(model, optimizers=optim)


def _train(model, optim, train_steps=1):
    # [FUNCTION DOC] HELPER: Runs train_steps steps with a fixed seed (sum-of-output loss, backward, Adam step, zero grad).
    # Returns the last loss, or None if no steps ran.
    torch.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss


def _init_model(device, world_size):
    # [FUNCTION DOC] HELPER: Builds a device mesh, an FSDP-wrapped Model on CUDA, an Adam optimizer, and makes them stateful.
    # Returns (model, optimizer).
    device_mesh = init_device_mesh(device, (world_size,))
    model = Model().cuda()
    model = FSDP(
        model,
        device_mesh=device_mesh,
        use_orig_params=True,
    )
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    _make_stateful(model, optim)

    return model, optim


def run(rank, world_size, device="cuda"):
    # [FUNCTION DOC] Per-process entry point. Sets rendezvous env vars, initializes the process group, trains two steps,
    # saves model and optimizer to CHECKPOINT_DIR, rebuilds fresh ones, loads the checkpoint into them, and trains two more steps.
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model, optim = _init_model(device, world_size)
    _train(model, optim, train_steps=2)

    dcp.save(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )

    # presumably do something else
    model, optim = _init_model(device, world_size)
    dcp.load(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )
    _train(model, optim, train_steps=2)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running stateful checkpoint example on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
