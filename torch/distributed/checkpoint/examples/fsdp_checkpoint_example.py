# mypy: allow-untyped-defs
# [MODULE DOC] Runnable example that saves and loads an FSDP model and optimizer with distributed checkpoint.
# Saves sharded model and optimizer state, then loads them into a second model and prints parameters before and
# after. Run as a script: it deletes CHECKPOINT_DIR and spawns one process per CUDA device.
# Copyright (c) Meta Platforms, Inc. and affiliates

"""
The following example demonstrates how to use PyTorch Distributed Checkpoint to save a FSDP model.

This is the current recommended way to checkpoint FSDP.
torch.save() and torch.load() is not recommended when checkpointing sharded models.
"""

import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


CHECKPOINT_DIR = f"/scratch/{os.environ.get('LOGNAME', '')}/checkpoint"


def opt_at(opt, idx):
    # [FUNCTION DOC] HELPER: Returns the idx-th per-parameter optimizer state entry (in insertion order).
    return list(opt.state.values())[idx]


def init_model():
    # [FUNCTION DOC] HELPER: Builds an FSDP-wrapped linear layer on the current rank's GPU, runs one forward/backward and
    # one Adam step so optimizer state exists. Returns (model, optimizer).
    model = FSDP(torch.nn.Linear(4, 4).cuda(dist.get_rank()))
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    model(torch.rand(4, 4)).sum().backward()
    optim.step()

    return model, optim


def print_params(stage, model_1, model_2, optim_1, optim_2):
    # [FUNCTION DOC] HELPER: Prints, for this rank, the full parameters of both models and the Adam moment state of both
    # optimizers, labelled with the stage string.
    with FSDP.summon_full_params(model_1), FSDP.summon_full_params(model_2):
        print(
            f"{stage} --- rank: {dist.get_rank()}\n"
            f"model.weight: {model_1.weight}\n"
            f"model_2.weight:{model_2.weight}\n"
            f"model.bias: {model_1.bias}\n"
            f"model_2.bias: {model_2.bias}\n"
        )

    print(
        f"{stage} --- rank: {dist.get_rank()}\n"
        f"optim exp_avg:{opt_at(optim_1, 0)['exp_avg']}\n"
        f"optim_2 exp_avg:{opt_at(optim_2, 0)['exp_avg']}\n"
        f"optim exp_avg_sq:{opt_at(optim_1, 0)['exp_avg_sq']}\n"
        f"optim_2 exp_avg_sq:{opt_at(optim_2, 0)['exp_avg_sq']}\n"
    )


def run_fsdp_checkpoint_example(rank, world_size):
    # [FUNCTION DOC] Per-process entry point. Initializes the process group, saves sharded model and optimizer state of
    # model 1, creates model 2, loads the sharded model state into it, loads the sharded optimizer state, prints parameters
    # before and after loading, and destroys the process group. Side effects: env vars, files in CHECKPOINT_DIR.
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Create a model
    model_1, optim_1 = init_model()

    # Save the model to CHECKPOINT_DIR
    with FSDP.state_dict_type(model_1, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model_1.state_dict(),
            "optim": FSDP.optim_state_dict(model_1, optim_1),
        }

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        )

    # Create a second model
    model_2, optim_2 = init_model()

    # Print the model parameters for both models.
    # Before loading, the parameters should be different.
    print_params("Before loading", model_1, model_2, optim_1, optim_2)

    # Load model_2 with parameters saved in CHECKPOINT_DIR
    with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model_2.state_dict(),
            # cannot load the optimizer state_dict together with the model state_dict
        }

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
        model_2.load_state_dict(state_dict["model"])

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )

        flattened_osd = FSDP.optim_state_dict_to_load(
            model_2, optim_2, optim_state["optim"]
        )
        optim_2.load_state_dict(flattened_osd)

    # Print the model parameters for both models.
    # After loading, the parameters should be the same.
    print_params("After loading", model_1, model_2, optim_1, optim_2)

    # Shut down world pg
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(
        run_fsdp_checkpoint_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
