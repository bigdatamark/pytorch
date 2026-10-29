# mypy: allow-untyped-defs
# [MODULE DOC] Runnable example of asynchronous distributed checkpointing with FSDP.
# Trains a small model on every GPU, saves with async_save every SAVE_PERIOD epochs, injects a fault
# every FAULT_PERIOD epochs, and reloads from the last checkpoint. Run as a script: it deletes
# CHECKPOINT_DIR and spawns one process per CUDA device. Import has no side effects beyond constants.
# Owner(s): ["oncall: distributed"]

import os
import shutil
import traceback
from concurrent.futures import Future

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.device_mesh import init_device_mesh


DEVICE = "cuda"
NUM_EPOCHS = 1000
SAVE_PERIOD = 10
FAULT_PERIOD = 25
CHECKPOINT_DIR = f"~/{os.environ.get('LOGNAME', '')}/checkpoint"


class InjectedException(Exception):
    pass
    # [CLASS DOC] Exception raised deliberately to simulate a trainer fault.


class Model(torch.nn.Module):
    # [CLASS DOC] Small five-layer linear network with relu activations and a sigmoid output.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the five linear layers (8->32->128->64->8->1).
        super().__init__()
        self.net1 = nn.Linear(8, 32)
        self.net2 = nn.Linear(32, 128)
        self.net3 = nn.Linear(128, 64)
        self.net4 = nn.Linear(64, 8)
        self.net5 = nn.Linear(8, 1)

    def forward(self, x):
        # [FUNCTION DOC] Applies relu after the first four layers and sigmoid after the last; returns the output.
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        x = F.sigmoid(self.net5(x))
        return x


def _init_model(rank, world_size):
    # [FUNCTION DOC] HELPER: Builds a device mesh, wraps a CUDA model in FSDP (original params), creates an Adam
    # optimizer, and patches model/optimizer to be stateful for checkpointing. Returns (model, optimizer).
    device_mesh = init_device_mesh(DEVICE, (world_size,))

    # Create a dummy model and wrap it in FSDP
    model = Model().cuda()
    device_mesh = init_device_mesh(DEVICE, (world_size,))
    model = FSDP(model, device_mesh=device_mesh, use_orig_params=True)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    _patch_model_state_dict(model)
    # pyrefly: ignore [bad-argument-type]
    _patch_optimizer_state_dict(model, optimizers=optim)

    return model, optim


def _print(msg):
    # [FUNCTION DOC] HELPER: Prints msg only on rank 0.
    if dist.get_rank() == 0:
        print(msg)


def _input():
    # [FUNCTION DOC] HELPER: Returns random CUDA inputs (128x8) and labels that are 1 where the row sum is at least 4, else 0.
    x = torch.rand(128, 8, device="cuda")
    y = torch.zeros(128, 1, device="cuda")

    y[torch.sum(x, dim=1) >= 4] = 1.0

    return x, y


def run(rank, world_size):
    # [FUNCTION DOC] Per-process entry point. Sets rendezvous env vars, initializes the process group, builds the
    # model, then loops over epochs: train step, async checkpoint save every SAVE_PERIOD epochs (waiting on the
    # previous save future first; raises AssertionError if it is not a Future), and a fault every FAULT_PERIOD epochs.
    # On the injected fault it syncs ranks, prints the traceback, waits for pending save, and reloads state.
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model, optim = _init_model(rank, world_size)
    state_dict = {"model": model, "optim": optim}
    loss_calc = torch.nn.BCELoss()

    f = None
    # pyrefly: ignore [bad-assignment]
    for epoch in range(NUM_EPOCHS):
        try:
            torch.manual_seed(epoch)
            x, y = _input()

            loss = loss_calc(model(x), y)

            _print(f"{epoch=} {loss=}")

            loss.backward()
            optim.step()
            optim.zero_grad()

            if epoch % SAVE_PERIOD == 0:
                if f is not None:
                    if not isinstance(f, Future):
                        raise AssertionError("f should be a Future instance")
                    f.result()
                f = dcp.state_dict_saver.async_save(
                    state_dict, checkpoint_id=CHECKPOINT_DIR
                )

            if FAULT_PERIOD > 0 and epoch % FAULT_PERIOD == 0:
                raise InjectedException("Fault injection!")

        except InjectedException as e:
            dist.barrier()

            _print("Trainer encountered exception:")
            traceback.print_tb(e.__traceback__)

            _print("Reloading model from last checkpoint!")
            if f is not None:
                if not isinstance(f, Future):
                    raise AssertionError("f should be a Future instance") from None
                f.result()
            dcp.load(state_dict)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running an example of Async Checkpointing on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
