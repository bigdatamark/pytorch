# Owner(s): ["oncall: distributed"]

# [MODULE DOC] End-to-end multi-GPU test of a pretrain then fine-tune checkpoint workflow
# with FSDP: a pretrained model is saved, then a fine-tuning model embedding it (frozen) loads
# the pretrained submodule and any earlier fine-tune checkpoint via the DCP state dict APIs,
# trains, and saves only non-frozen params. DIM is the Linear layer width. Import-time side
# effect: exits the process (status 0) if distributed is unavailable or under dev-asan.

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


DIM = 500


class PreTrainedModel(nn.Module):
    # [CLASS DOC] HELPER: Stand-in pretrained network of three Linear layers, a Sequential
    # and a ModuleList (each of Linear plus ReLU), and a ReLU; all widths are DIM.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers listed in the class description.
        super().__init__()
        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.sequential = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())
        self.module_list = nn.ModuleList([nn.Linear(DIM, DIM), nn.ReLU()])
        self.relu = nn.ReLU()

    def forward(self, batch):
        # [FUNCTION DOC] HELPER: Applies the three Linear+ReLU layers, then the Sequential,
        # then the two ModuleList members in order; returns the output tensor.
        x = self.relu(self.layer1(batch))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sequential(x)
        x = self.module_list[1](self.module_list[0](x))
        return x


class FineTuningModel(nn.Module):
    # [CLASS DOC] HELPER: Fine-tuning network: a PreTrainedModel child named pretrain whose
    # parameters are frozen, followed by three trainable Linear layers with ReLU.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the pretrained child, freezes its parameters by
        # clearing requires_grad, and creates the trainable layers.
        super().__init__()
        self.pretrain = PreTrainedModel()
        for p in self.pretrain.parameters():
            p.requires_grad = False

        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.relu = nn.ReLU()

    def forward(self, batch):
        # [FUNCTION DOC] HELPER: Runs the pretrained child, then the three trainable layers,
        # each followed by ReLU; returns the output tensor.
        x = self.relu(self.pretrain(batch))
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x


class TestFineTuning(DTensorTestBase):
    # [CLASS DOC] Multi-process test case (up to 4 ranks) driving the pretrain and fine-tune
    # checkpoint flow with FSDP on a one-dimensional device mesh.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Number of processes: the accelerator count capped at 4.
        return min(4, torch.accelerator.device_count())

    @property
    def backend(self):
        # [FUNCTION DOC] Returns a combined backend string using gloo for CPU and the
        # default backend of the test device type for the device.
        curr_backend = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:gloo,{self.device_type}:{curr_backend}"

    def pretrain(self, pretrain_dir: str) -> None:
        # [FUNCTION DOC] HELPER: Wraps a PreTrainedModel in FSDP, runs 3 Adam steps on random
        # batches, and saves the model and optimizer state dicts under keys model and optim
        # with a FileSystemWriter at pretrain_dir. Collective: all ranks must call it.
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        model = PreTrainedModel().to(self.device_type)
        model = FSDP(model, device_mesh=device_mesh)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training
        for _ in range(3):
            batch = torch.rand(32, DIM, device=self.device_type)
            loss = model(batch).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()

        # Save state_dict
        model_state_dict, optim_state_dict = get_state_dict(model, optimizers=optim)
        saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
        dist_cp.save(
            state_dict=saved_state_dict,
            storage_writer=dist_cp.FileSystemWriter(pretrain_dir),
        )

    def finetune(self, pretrain_dir: str, finetune_dir: str) -> None:
        # [FUNCTION DOC] HELPER: Runs two rounds with an FSDP-wrapped FineTuningModel. Each
        # round loads the pretrained submodule weights from pretrain_dir and applies them
        # non-strictly, then tries to load model and optimizer state (frozen params ignored)
        # and applies it; a KeyError is tolerated only in round 0 (nothing saved yet).
        # Then trains 3 steps and saves the non-frozen state to finetune_dir.
        # KNOWN ISSUE: the restart load reads pretrain_dir, not finetune_dir, so the
        # fine-tune checkpoint written by the previous round is never read back.
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        model = FineTuningModel().to(self.device_type)
        # TODO: make the parallelism more complicated, e.g., using 2D + DDP.
        model = FSDP(model, use_orig_params=True, device_mesh=device_mesh)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Simulate that the fine tuning restart after 3 iterations
        for i in range(2):
            # Load pretrain submodules checkpoint
            pretrain_state_dict = get_model_state_dict(
                model,
                submodules={model.pretrain},
                options=StateDictOptions(keep_submodule_prefixes=False),
            )
            dist_cp.load(
                {"model": pretrain_state_dict},
                storage_reader=dist_cp.FileSystemReader(pretrain_dir),
            )
            set_model_state_dict(
                model,
                model_state_dict={model.pretrain: pretrain_state_dict},
                options=StateDictOptions(strict=False),
            )

            try:
                # Load training submodules checkpoint
                model_state_dict, optim_state_dict = get_state_dict(
                    model,
                    optimizers=optim,
                    options=StateDictOptions(ignore_frozen_params=True),
                )
                dist_cp.load_state_dict(
                    {"model": model_state_dict, "optim": optim_state_dict},
                    storage_reader=dist_cp.FileSystemReader(pretrain_dir),
                )
                set_state_dict(
                    model,
                    optimizers=optim,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                    options=StateDictOptions(strict=False),
                )
            except KeyError:
                # If this is the first round of the fine tuning, then nothing is saved.
                # If this is the restart of the fine tuning, then checkpoint should exit.
                self.assertEqual(i, 0)

            # Training
            for _ in range(3):
                batch = torch.rand(32, DIM, device=self.device_type)
                loss = model(batch).sum()
                loss.backward()
                optim.step()
                optim.zero_grad()

            # Save state_dict
            model_state_dict, optim_state_dict = get_state_dict(
                model,
                optimizers=optim,
                options=StateDictOptions(ignore_frozen_params=True),
            )
            saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
            dist_cp.save(
                state_dict=saved_state_dict,
                storage_writer=dist_cp.FileSystemWriter(finetune_dir),
            )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    def test_fine_tuning(self) -> None:
        # [FUNCTION DOC] Creates pretrain and finetune subdirectories in the temp dir (rank 0
        # only, then a barrier and filesystem sync), checks both exist, then runs pretrain
        # followed by finetune. Needs at least 4 GPUs. Prints the two paths.
        self.assertTrue(os.path.exists(self.temp_dir))
        pretrain_dir = os.path.join(self.temp_dir, "pretrain")
        finetune_dir = os.path.join(self.temp_dir, "finetune")
        print(pretrain_dir, finetune_dir)
        if self.rank == 0:
            os.mkdir(pretrain_dir)
            os.mkdir(finetune_dir)
        dist.barrier()
        os.sync()
        self.assertTrue(os.path.exists(pretrain_dir))
        self.assertTrue(os.path.exists(finetune_dir))

        self.pretrain(pretrain_dir)
        self.finetune(pretrain_dir, finetune_dir)


if __name__ == "__main__":
    run_tests()
