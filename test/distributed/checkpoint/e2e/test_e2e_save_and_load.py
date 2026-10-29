# Owner(s): ["oncall: distributed"]

# [MODULE DOC] End-to-end multi-GPU tests of DCP save and load: a small MLP trained under
# FSDP, HSDP or DDP (optionally compiled) is saved (sync, or async with thread or process
# checkpointer, cached or uncached staging) and loaded back, then compared against an
# unparallelized reference trained identically. Also covers Stateful in-place loading, key
# order independence, empty state dicts, partial loads by key, overwrite behavior, the CPU
# backend requirement for async save, and initializing state dicts in place. Import-time
# side effect: computes the module global device_type from the current accelerator.

import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import auto, Enum
from functools import partial
from io import BytesIO
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.distributed.checkpoint.state_dict_saver as saver
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.utils import CheckpointException
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


# Simple and boring model
class TestDummyModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Four-layer MLP (8 to 16 to 32 to 64 to 8 Linear layers with ReLU
    # after each) used as the model under test; can also generate a random input batch.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Seeds the global RNG with 0 (side effect) so every instance
        # is initialized identically, then creates the four Linear layers.
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(8, 16)
        self.net2 = nn.Linear(16, 32)
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Linear(64, 8)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the four layers in order, each followed by ReLU.
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return x

    def get_input(self):
        # [FUNCTION DOC] HELPER: Returns a random 8x8 tensor on the test device type.
        return torch.rand(8, 8, device=device_type)


class TestStatefulObj:
    # [CLASS DOC] HELPER: Plain (non-Stateful base) object holding one random 10x10 tensor in
    # data, exposing the state dict protocol so DCP saves and loads it in place; equality
    # compares the tensors exactly.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the random tensor on the test device type.
        self.data = torch.rand(10, 10, device=device_type)

    def state_dict(self):
        # [FUNCTION DOC] HELPER: Returns a mapping with the single key data.
        return {"data": self.data}

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC] HELPER: Replaces data with the loaded tensor.
        self.data = state_dict["data"]

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: True when the other object's tensor is exactly equal.
        return torch.equal(self.data, other.data)


class ModelType(Enum):
    # [CLASS DOC] HELPER: Parallelization choices for the model under test: FSDP, HSDP,
    # FSDP combined with tensor parallel, DDP, or NONE (plain model).
    FSDP = auto()
    HSDP = auto()
    FSDP_TP = auto()
    DDP = auto()
    NONE = auto()  # no parallelization


@dataclass
class TestTrainState:
    # [CLASS DOC] HELPER: Dataclass with the training progress step count, latest loss and list
    # of losses, exposing the state dict protocol; losses are pickled into a byte stream so
    # DCP stores them as bytes. Equality compares all three fields.
    step: int = 0
    current_loss: float = -1
    losses: list[float] = field(default_factory=list)

    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Returns a mapping with step as an int32 tensor, current_loss
        # as a float32 tensor, and losses serialized into an in-memory byte buffer.
        loss_bytes = BytesIO()
        torch.save(self.losses, loss_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "current_loss": torch.tensor(self.current_loss, dtype=torch.float32),
            "losses": loss_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        # [FUNCTION DOC] HELPER: Restores step and current_loss from the tensors and the loss
        # list by rewinding and deserializing the byte buffer.
        self.step = state_dict["step"].item()
        self.current_loss = state_dict["current_loss"].item()
        state_dict["losses"].seek(0)
        self.losses = torch.load(state_dict["losses"])

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: True when step, current_loss and losses all match exactly.
        return (
            self.step == other.step
            and self.current_loss == other.current_loss
            and self.losses == other.losses
        )


def _train(model, optim, train_steps=1):
    # [FUNCTION DOC] HELPER: Seeds the global RNG with 0, then runs train_steps iterations of
    # forward on the model's own input, sum-loss, backward, optimizer step and zero-grad,
    # recording a random fake loss value and step count in a new TestTrainState. Returns a
    # tuple of the last real loss tensor (None if no steps) and the train state.
    torch.manual_seed(0)
    loss = None

    train_state = TestTrainState()

    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()

        # We usually sync the loss across dp ranks in real training.
        # This is just simulating for testing purpose.
        train_state.step += 1
        train_state.current_loss = torch.rand(1).item()
        train_state.losses.append(train_state.current_loss)

        optim.step()
        optim.zero_grad()

    return loss, train_state


class TestE2ESaveAndLoad(DTensorTestBase, VerifyStateDictMixin):
    # [CLASS DOC] Multi-process test case for DCP round trips of parallelized models and
    # optimizers, plus several smaller save/load behavior checks.
    @property
    def backend(self):
        # [FUNCTION DOC] Returns a combined backend string: gloo for CPU and the default
        # backend of the test device type for the device.
        curr_backend = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:gloo,{self.device_type}:{curr_backend}"

    def _create_model(self, compile, model_type, state_dict_options=None):
        # [FUNCTION DOC] HELPER: Builds a TestDummyModel on the test device and wraps it per
        # model_type: FSDP on a 1D mesh, HSDP on a 2D mesh, FSDP over a data-parallel mesh after
        # tensor parallel on net1 and net2, DDP, or unwrapped. Raises AssertionError for
        # unsupported types. Optionally compiles with static shapes, creates an Adam
        # optimizer, and for parallelized models patches the model and optimizer state dict
        # methods with state_dict_options. Returns the model and optimizer.
        dummy_model = TestDummyModel().to(self.device_type)

        if model_type not in ModelType:
            raise AssertionError(f"{model_type} is not supported.")
        if model_type == ModelType.FSDP:
            device_mesh = init_device_mesh(self.device_type, (self.world_size,))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
            )
        elif model_type == ModelType.HSDP:
            device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        elif model_type == ModelType.FSDP_TP:
            mesh_2d = init_device_mesh(
                self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
            )
            tp_mesh = mesh_2d["tp"]
            dp_mesh = mesh_2d["dp"]
            parallelize_plan = {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
            }
            model = parallelize_module(dummy_model, tp_mesh, parallelize_plan)
            model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
        elif model_type == ModelType.DDP:
            model = DistributedDataParallel(dummy_model)
            model.get_input = partial(TestDummyModel.get_input, model)
        else:
            model = dummy_model

        if compile:
            # TODO: enable dynamic=True when dynamic shape support is enabled.
            # model = torch.compile(model)
            model = torch.compile(model, dynamic=False)

        optim = self._optim(model)
        if model_type is not ModelType.NONE:
            _patch_model_state_dict(model, options=state_dict_options)
            _patch_optimizer_state_dict(
                model, optimizers=optim, options=state_dict_options
            )

        return model, optim

    def _optim(self, model):
        # [FUNCTION DOC] HELPER: Returns an Adam optimizer with learning rate 0.1.
        return torch.optim.Adam(model.parameters(), lr=0.1)

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    @parametrize("compile", [True, False])
    # TODO: Previously PairwiseParallel does not shard properly, passing ModelType.FSDP_TP test where it
    # should have failed. Disabling the failed test temporarily to unblock the deprecation of PairwiseParallel.
    @parametrize("model_type", [ModelType.FSDP, ModelType.HSDP, ModelType.DDP])
    def test_e2e(self, compile, model_type):
        # [FUNCTION DOC] Parametrized over compile on/off and FSDP, HSDP or DDP; runs the
        # synchronous save and load round trip. Needs at least 4 GPUs.
        self._run_e2e_test(compile, model_type)

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    @parametrize(
        "cache_staged_state_dict, async_checkpointer_type, zoc",
        [
            (False, AsyncCheckpointerType.THREAD, False),
            (True, AsyncCheckpointerType.THREAD, False),
            (False, AsyncCheckpointerType.PROCESS, False),
            (True, AsyncCheckpointerType.PROCESS, False),
            (False, AsyncCheckpointerType.PROCESS, True),
            (False, AsyncCheckpointerType.THREAD, True),
        ],
    )
    def test_e2e_async_cached(
        self, cache_staged_state_dict, async_checkpointer_type, zoc
    ):
        # [FUNCTION DOC] Parametrized over staging cache, thread or process checkpointer and
        # zero-overhead-checkpointing mode; runs the FSDP round trip using async save.
        # Needs at least 4 GPUs.
        self._run_e2e_test(
            compile=False,
            model_type=ModelType.FSDP,
            async_op=True,
            cache_staged_state_dict=cache_staged_state_dict,
            async_checkpointer_type=async_checkpointer_type,
            zoc=zoc,
        )

    def _run_e2e_test(
        self,
        compile,
        model_type,
        async_op=False,
        cache_staged_state_dict=False,
        async_checkpointer_type=None,
        zoc=False,
    ):
        # [FUNCTION DOC] HELPER: Trains an unparallelized reference and a parallelized model
        # two steps each, saves a state dict of model, optimizer, a stateful object and a
        # train state to the temp dir (sync, or async with an optional stager built from
        # zoc and checkpointer type and polled until the upload future is done), loads
        # into a fresh model set, and checks stateful object and train state equality. Then
        # trains one more step on both, requires equal losses, and compares the model
        # and optimizer state dicts with the reference.
        model, optim = self._create_model(compile, ModelType.NONE)
        _train(model, optim, train_steps=2)

        dist_model, dist_optim = self._create_model(compile, model_type)
        _, original_train_state = _train(dist_model, dist_optim, train_steps=2)

        original_stateful_obj = TestStatefulObj()  # tests arbitrary saving/loading
        sd = {
            "model": dist_model,
            "optimizer": dist_optim,
            "s": original_stateful_obj,
            "train_state": original_train_state,
        }

        if async_op:
            writer = DCP.FileSystemWriter(
                self.temp_dir, cache_staged_state_dict=cache_staged_state_dict
            )
            stager = None
            if not cache_staged_state_dict:
                use_shared_memory = (
                    async_checkpointer_type == AsyncCheckpointerType.PROCESS
                )
                staging_options = StagingOptions(
                    use_async_staging=zoc,
                    use_shared_memory=use_shared_memory,
                    use_pinned_memory=zoc,
                    use_non_blocking_copy=zoc,
                )
                stager = DefaultStager(staging_options)
            async_save_response_or_future = saver.async_save(
                sd,
                storage_writer=writer,
                async_checkpointer_type=(
                    async_checkpointer_type
                    if async_checkpointer_type
                    else AsyncCheckpointerType.THREAD
                ),
                async_stager=stager,
            )
            if isinstance(async_save_response_or_future, Future):
                save_future = async_save_response_or_future
            else:
                if not isinstance(async_save_response_or_future, AsyncSaveResponse):
                    raise AssertionError(
                        f"Expected AsyncSaveResponse, got {type(async_save_response_or_future)}"
                    )
                save_future = async_save_response_or_future.upload_completion
            # wait for the future to complete
            t = time.monotonic()
            while not save_future.done():
                time.sleep(1)
                print(f"still waiting... {time.monotonic() - t}")

            save_future.result()
        else:
            DCP.save(sd, checkpoint_id=self.temp_dir)

        loaded_stateful_obj = TestStatefulObj()
        loaded_train_state = TestTrainState()
        dist_model, dist_optim = self._create_model(compile, model_type)

        DCP.load(
            state_dict={
                "model": dist_model,
                "optimizer": dist_optim,
                "s": loaded_stateful_obj,
                "train_state": loaded_train_state,
            },
            checkpoint_id=self.temp_dir,
        )

        self.assertEqual(original_stateful_obj, loaded_stateful_obj)
        self.assertEqual(original_train_state, loaded_train_state)

        # train one more step on both models
        loss, _ = _train(model, optim, train_steps=1)
        dist_loss, _ = _train(dist_model, dist_optim, train_steps=1)
        self.assertEqual(loss, dist_loss)

        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        model_sd, _ = get_state_dict(model, optimizers=optim)

        self._verify_msd(model_sd, dist_msd)
        self._verify_osd_by_load(model, optim, self._optim(model), dist_osd)

    @with_temp_dir
    def test_stateful_and_non_stateful_loads(self) -> None:
        # [FUNCTION DOC] Verifies that loading into a state dict calls load_state_dict on a
        # Stateful value in place without reassigning the key, while a plain nested dict
        # value is replaced by assignment. Uses a dict subclass that records writes.
        class StateDict(dict):
            # [CLASS DOC] HELPER: Dict recording whether any item was assigned after reset.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Starts with the assignment flag cleared.
                self.set_sd_item_called = False

            def __setitem__(self, item, value):
                # [FUNCTION DOC] HELPER: Sets the assignment flag and stores the item.
                self.set_sd_item_called = True
                super().__setitem__(item, value)

        class Foo(Stateful):
            # [CLASS DOC] HELPER: Stateful with an empty state that records load calls.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Starts with the load flag cleared.
                self.load_state_dict_called = False

            def state_dict(self):
                # [FUNCTION DOC] HELPER: Returns an empty mapping.
                return {}

            def load_state_dict(self, state_dict):
                # [FUNCTION DOC] HELPER: Sets the load flag; ignores the state.
                self.load_state_dict_called = True

        stateful_foo = Foo()
        sd = StateDict()
        sd["foo"] = stateful_foo
        sd.set_sd_item_called = False

        DCP.save(sd, checkpoint_id=self.temp_dir)
        DCP.load(sd, checkpoint_id=self.temp_dir)

        # Validate that the stateful object was loaded in-place
        self.assertTrue(stateful_foo.load_state_dict_called)
        # Validate that the stateful object was NOT replaced in the state dict
        self.assertFalse(sd.set_sd_item_called)

        sd = StateDict()
        sd["foo"] = {"replicated": torch.rand(10, 10), "bytes": [1, 2, 3, 4]}
        sd.set_sd_item_called = False

        DCP.save(sd, checkpoint_id=self.temp_dir)
        DCP.load(sd, checkpoint_id=self.temp_dir)

        # Validate that the non-stateful state dict was replaced with the loaded state dict
        self.assertTrue(sd.set_sd_item_called)

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    def test_different_ordered_state_dict_keys(self):
        # [FUNCTION DOC] Saves and loads state dicts whose keys A and B are inserted in opposite
        # order on rank 0 versus other ranks. Their load hooks perform collectives (all
        # gather and all reduce), so the test would deadlock if load did not process keys
        # in a rank-consistent order. Needs 4 GPUs.
        """Tests that the order of keys in the state dict does not matter when loading
        If order was not accounted for, the following test would cause a deadlock.
        """

        world_size = self.world_size

        class Foo:
            # [CLASS DOC] HELPER: Empty-state object whose load issues an all-gather.
            def state_dict(self):
                # [FUNCTION DOC] HELPER: Returns an empty mapping.
                return {}

            def load_state_dict(self, state_dict):
                # [FUNCTION DOC] HELPER: Performs a collective all-gather of a small tensor
                # across all ranks; ignores the loaded state.
                tl = [
                    torch.ones(2, dtype=torch.int64, device=device_type)
                    for _ in range(world_size)
                ]
                t = (
                    torch.arange(2, dtype=torch.int64, device=device_type)
                    + 1
                    + 2 * dist.get_rank()
                )
                dist.all_gather(tl, t, async_op=False)

        class Bar:
            # [CLASS DOC] HELPER: Empty-state object whose load issues an all-reduce.
            def state_dict(self):
                # [FUNCTION DOC] HELPER: Returns an empty mapping.
                return {}

            def load_state_dict(self, state_dict):
                # [FUNCTION DOC] HELPER: Performs a collective sum all-reduce of a small tensor
                # across all ranks; ignores the loaded state.
                tensor = (
                    torch.arange(2, dtype=torch.int64, device=device_type)
                    + 1
                    + 2 * dist.get_rank()
                )
                dist.all_reduce(tensor, op=ReduceOp.SUM)

        if self.rank == 0:
            sd = {
                "A": Foo(),
                "B": Bar(),
            }
        else:
            sd = {
                "B": Bar(),
                "A": Foo(),
            }

        DCP.save(sd, checkpoint_id=self.temp_dir)
        DCP.load(sd, checkpoint_id=self.temp_dir)

    @with_temp_dir
    def test_no_dist(self):
        # [FUNCTION DOC] Saves and loads an empty state dict without initializing process
        # groups; checks only that neither call raises.
        # since comm's are not initialized in this method, `no_dist`
        # is assumed False
        DCP.save({}, checkpoint_id=self.temp_dir)
        DCP.load({}, checkpoint_id=self.temp_dir)

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    def test_partial_load(self):
        # [FUNCTION DOC] Saves a trained FSDP model and optimizer, then loads only the model
        # into a fresh FSDP model and compares it with the reference. Also reads parts of the
        # checkpoint by key prefix: the model keys, and the optimizer state keys (checking
        # no param_groups entry is returned), comparing exp_avg, exp_avg_sq and step values
        # against the live optimizer state. Needs 4 GPUs.
        model, optim = self._create_model(compile=False, model_type=ModelType.NONE)
        _train(model, optim, train_steps=2)

        dist_model, dist_optim = self._create_model(
            compile=False, model_type=ModelType.FSDP
        )
        _train(dist_model, dist_optim, train_steps=2)

        DCP.save(
            {"model": dist_model, "optimizer": dist_optim}, checkpoint_id=self.temp_dir
        )

        dist_model, _ = self._create_model(compile=False, model_type=ModelType.FSDP)
        DCP.load({"model": dist_model}, checkpoint_id=self.temp_dir)

        dist_msd = get_model_state_dict(dist_model)
        model_sd = get_model_state_dict(model)
        self._verify_msd(model_sd, dist_msd)

        # another way
        loaded_model_sd = _load_state_dict_from_keys(
            "model", checkpoint_id=self.temp_dir
        )["model"]
        self._verify_msd(model_sd, loaded_model_sd, offload_to_cpu=True)

        loaded_optim_state = _load_state_dict_from_keys(
            "optimizer.state", checkpoint_id=self.temp_dir
        )["optimizer"]["state"]
        self.assertNotIn("param_groups", loaded_optim_state)
        for k, v in dist_optim.state_dict()["state"].items():
            for optim_key in ["exp_avg", "exp_avg_sq", "step"]:
                self._compare_tensor(
                    loaded_optim_state[k][optim_key], v[optim_key], offload_to_cpu=True
                )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    def test_overwrite(self):
        # [FUNCTION DOC] Saves one random tensor, saves a second to the same directory with
        # overwrite enabled, loads and checks the second values win, then checks a save with
        # overwrite disabled raises CheckpointException saying the checkpoint already exists.
        t1, t2 = torch.randn(10), torch.randn(10)
        DCP.save({"random": t1}, checkpoint_id=self.temp_dir)
        DCP.save(
            {"random": t2},
            storage_writer=DCP.FileSystemWriter(self.temp_dir, overwrite=True),
        )

        sd = {"random": torch.zeros(10)}
        DCP.load(sd, checkpoint_id=self.temp_dir)

        self.assertTrue(torch.allclose(sd["random"], t2))

        with self.assertRaisesRegex(
            CheckpointException, ".*Checkpoint already exists.*"
        ):
            DCP.save(
                {"random": t2},
                storage_writer=DCP.FileSystemWriter(self.temp_dir, overwrite=False),
            )


class TestNoCPU(DTensorTestBase):
    # [CLASS DOC] Test case that uses an NCCL-only process group to check async save rejects
    # a configuration without a CPU backend.
    @property
    def backend(self):
        # [FUNCTION DOC] Always the nccl backend, with no CPU backend.
        return "nccl"

    @with_comms
    def test_no_cpu(self):
        # [FUNCTION DOC] Expects async_save to raise AssertionError saying a CPU backend must
        # be enabled for async save.
        with self.assertRaisesRegex(
            AssertionError, r"A CPU backend must be enabled for async save;.*?"
        ):
            f = saver.async_save({})
            f.result()


class TestInitStateDict(DTensorTestBase):
    # [CLASS DOC] Test case for in-place loading into model and optimizer state dicts.
    @with_temp_dir
    def test_init_state_dict(self):
        # [FUNCTION DOC] Saves model and optimizer state dicts, builds a second model and an
        # optimizer with a different learning rate, loads into their state dicts, and checks
        # the very same dict objects were filled in place. Then applies them with
        # set_state_dict and checks the state matches and the learning rate is back to 0.1.
        temp_dir = self.temp_dir
        model = TestDummyModel()
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        state_dict_to_save = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optim),
        }
        DCP.save(state_dict_to_save, checkpoint_id=temp_dir)

        torch.manual_seed(0)
        model_2 = TestDummyModel()
        # Changing the learning rate for optimizer, which is not a tensor.
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.2)

        msd = get_model_state_dict(model_2)
        osd = get_optimizer_state_dict(model_2, optim_2)

        state_dict_to_load = {"model": msd, "optimizer": osd}
        DCP.load(state_dict_to_load, checkpoint_id=temp_dir)

        # We need to check that the two variables point to the same object in memory,
        # since we claim DCP is in-place loading.
        self.assertTrue(msd is state_dict_to_load["model"])
        self.assertTrue(osd is state_dict_to_load["optimizer"])

        # set_state_dict calls load_state_dict for model and optimizer.
        # so we should see the optim_2.param_groups learning rate is 0.1 instead of 0.2 now.
        set_state_dict(
            model_2,
            optim_2,
            model_state_dict=state_dict_to_load["model"],
            optim_state_dict=state_dict_to_load["optimizer"],
        )
        self.assertEqual(msd, get_model_state_dict(model_2))
        self.assertEqual(osd, get_optimizer_state_dict(model_2, optim_2))
        self.assertEqual(optim_2.param_groups[0]["lr"], 0.1)


instantiate_parametrized_tests(TestE2ESaveAndLoad)
if __name__ == "__main__":
    run_tests()
