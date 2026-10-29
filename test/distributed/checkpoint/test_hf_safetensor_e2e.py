# Owner(s): ["oncall: distributed checkpointing"]
# [MODULE DOC] End-to-end tests for the HuggingFace safetensors checkpoint storage: single
# rank save/load (including multithreaded and empty-dict loads), loading a hand-made
# block-quantized checkpoint, distributed save with consolidation into one file, and DTensor
# resharding across placements and meshes. Most tests skip when safetensors is missing.
# Defines module-level placement lists (including all distinct 2D pairs).

import importlib
import json
import os

import torch
import torch.distributed.checkpoint as dist_cp
from torch import distributed as dist
from torch.distributed.checkpoint.quantized_hf_storage import (
    QuantizedHuggingFaceStorageReader,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard, zeros
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


CHECKPOINT_DIR = "checkpoint"


class MyTestModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module (two Linear layers, one EmbeddingBag) whose state
    # dict supplies tensors for the safetensors tests.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear_1 (5->5), linear_2 (5->1), emb (5x10).
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


class TestSingleRankSaveLoad(TestCase):
    # [CLASS DOC] Single-process tests of HuggingFace storage writer/reader round trips.
    @with_temp_dir
    def test_save(self) -> None:
        # [FUNCTION DOC] Saves a module state dict with HuggingFaceStorageWriter, reads the
        # resulting single safetensors file with the safetensors library, and asserts keys
        # and tensor values match. Prints a message and returns if safetensors is missing.
        try:
            from safetensors.torch import load_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.HuggingFaceStorageWriter(path=CHECKPOINT_DIR),
        )

        state_dict_loaded = load_file(
            CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )
        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_loaded.keys())
        )
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )

    @with_temp_dir
    def test_load(self) -> None:
        # [FUNCTION DOC] Writes a state dict to a safetensors file with the safetensors
        # library, loads it into another module's state dict with HuggingFaceStorageReader,
        # and asserts values match. Returns early if safetensors is missing.
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        state_dict_to_load = MyTestModule().state_dict()
        save_file(
            state_dict_to_save, CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )

        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp.HuggingFaceStorageReader(path=CHECKPOINT_DIR),
        )

        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_to_load.keys())
        )
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_to_load[key])
            )

    @with_temp_dir
    def test_load_into_empty_dict(self) -> None:
        # [FUNCTION DOC] Writes a safetensors file, then reconstructs a state dict using only
        # the reader (no target dict) via the key-based loader, and asserts keys and values
        # match. Returns early if safetensors is missing.
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        save_file(
            state_dict_to_save, CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )

        state_dict_loaded = _load_state_dict_from_keys(
            storage_reader=dist_cp.HuggingFaceStorageReader(path=CHECKPOINT_DIR),
        )

        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_loaded.keys())
        )
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )

    @with_temp_dir
    def test_load_with_multiple_threads(self) -> None:
        # [FUNCTION DOC] Saves a state dict split across two safetensors files by an
        # fqn-to-file-index mapping, loads it with a reader using two threads, and asserts
        # keys and values match. Returns early if safetensors is not installed.
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        state_dict_to_load = MyTestModule().state_dict()

        # Create a mapping to split tensors across multiple files
        # This will force multiple files to be created, enabling multi-threading
        fqn_to_index_mapping = {}
        for i, fqn in enumerate(state_dict_to_save.keys()):
            fqn_to_index_mapping[fqn] = (i % 2) + 1  # Split across 2 files

        # Save using HuggingFaceStorageWriter with multiple files
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=CHECKPOINT_DIR, fqn_to_index_mapping=fqn_to_index_mapping
            ),
        )

        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp.HuggingFaceStorageReader(
                path=CHECKPOINT_DIR, thread_count=2
            ),
        )

        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_to_load.keys())
        )
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_to_load[key])
            )

    @with_temp_dir
    def test_quantized_checkpoint_loading(self) -> None:
        # [FUNCTION DOC] Builds three float tensors, halves each as a fake quantized weight
        # with a companion inverse-scale tensor of 2.0 per 128x128 block, writes them to a
        # safetensors file with an index json, and loads with the quantized reader (float32
        # target, block size 128, two threads). Asserts shapes, dtypes and values match the
        # originals within tolerance. Returns early if safetensors is missing.
        """Test end-to-end saving a quantizaed checkpoint and loading it."""
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        # Create original (unquantized) tensors to validate against
        original_tensors = {
            "linear1.weight": torch.randn(256, 128, dtype=torch.float32) * 2.0,
            "linear2.weight": torch.randn(128, 64, dtype=torch.float32) * 1.5,
            "embedding.weight": torch.randn(512, 256, dtype=torch.float32) * 3.0,
        }

        # Create quantized tensors and scale tensors
        quantized_checkpoint = {}
        block_size = 128

        for tensor_name, original_tensor in original_tensors.items():
            # Simulate quantization: scale down the tensor for quantization
            # This is a simplified quantization - in real scenarios it would be more complex
            rows, cols = original_tensor.shape

            # Create scale tensor for block-wise dequantization
            block_rows = (rows + block_size - 1) // block_size
            block_cols = (cols + block_size - 1) // block_size

            # Create scale inverse tensor (used for dequantization)
            scale_inv = torch.ones(block_rows, block_cols, dtype=torch.float32) * 2.0

            # Create quantized version (divide by scale for quantization)
            quantized_tensor = original_tensor / 2.0  # Simplified quantization

            # Store quantized tensor and its scale
            quantized_checkpoint[tensor_name] = quantized_tensor
            quantized_checkpoint[f"{tensor_name}_scale_inv"] = scale_inv

        # Save quantized checkpoint to safetensors file
        safetensors_file = os.path.join(CHECKPOINT_DIR, "model.safetensors")
        save_file(quantized_checkpoint, safetensors_file)

        # Create model.safetensors.index.json with weight mapping
        weight_map = {}
        for key in quantized_checkpoint:
            weight_map[key] = "model.safetensors"

        index_data = {
            "metadata": {
                "total_size": sum(
                    t.numel() * t.element_size() for t in quantized_checkpoint.values()
                )
            },
            "weight_map": weight_map,
        }

        index_file = os.path.join(CHECKPOINT_DIR, "model.safetensors.index.json")
        with open(index_file, "w") as f:
            json.dump(index_data, f, indent=2)

        # Prepare state dict to load into
        state_dict_to_load = {}
        for tensor_name, original_tensor in original_tensors.items():
            state_dict_to_load[tensor_name] = torch.zeros_like(original_tensor)

        # Load using QuantizedHuggingFaceStorageReader
        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=QuantizedHuggingFaceStorageReader(
                path=CHECKPOINT_DIR,
                target_dtype=torch.float32,
                block_size=block_size,
                thread_count=2,
            ),
        )

        # Validate that loaded tensors match original tensors
        self.assertEqual(
            sorted(original_tensors.keys()), sorted(state_dict_to_load.keys())
        )

        for tensor_name in original_tensors:
            original = original_tensors[tensor_name]
            loaded = state_dict_to_load[tensor_name]

            # Verify shapes match
            self.assertEqual(
                original.shape,
                loaded.shape,
                lambda msg: f"{msg}\nShape mismatch for {tensor_name}: {original.shape} vs {loaded.shape}",
            )

            # Verify dtypes match
            self.assertEqual(
                original.dtype,
                loaded.dtype,
                lambda msg: f"{msg}\nDtype mismatch for {tensor_name}: {original.dtype} vs {loaded.dtype}",
            )

            # Verify dequantized values match original values
            # We expect exact match since we used simple 2x scaling
            torch.testing.assert_close(
                loaded,
                original,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Value mismatch for tensor {tensor_name}",
            )


class TestDistributedHFSafetensorsConsolidation(DTensorTestBase):
    # [CLASS DOC] Multi-GPU test of distributed safetensors save with consolidation.
    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_consolidate_to_one_file(self) -> None:
        # [FUNCTION DOC] Builds a row-sharded 4x4 DTensor from per-rank slices, saves it with
        # HuggingFaceStorageWriter in distributed mode with consolidation, and on rank 0 reads
        # the single consolidated safetensors file and asserts it holds exactly the dtensor
        # key with the full global tensor. Skips if safetensors is missing; needs 2+ GPUs.
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        import safetensors

        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
        mesh_shape = (self.world_size,)
        mesh_1d = init_device_mesh(self.device_type, mesh_shape)

        # Create local tensor with row-wise sharding
        rows_per_rank = global_tensor.shape[0] // self.world_size
        start_row = self.rank * rows_per_rank
        end_row = start_row + rows_per_rank
        local_tensor = global_tensor[start_row:end_row].clone()

        # Create DTensor with row-wise sharding
        dtensor = DTensor.from_local(
            local_tensor,
            device_mesh=mesh_1d,
            placements=[Shard(0)],
            shape=global_tensor.shape,
            stride=(4, 1),
        )

        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)

        checkpoint_dir = self.temp_dir

        state_dict_to_save = {"dtensor": dtensor}
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=checkpoint_dir,
                save_distributed=True,
                enable_consolidation=True,
            ),
        )
        dist.barrier()

        if self.rank == 0:
            file_path = os.path.join(checkpoint_dir, "model-00001-of-00001.safetensors")
            loaded_dict = safetensors.torch.load_file(file_path)
            self.assertEqual(loaded_dict.keys(), {"dtensor"})
            self.assertTrue(torch.equal(loaded_dict["dtensor"], global_tensor))

        dist.barrier()


ONE_D_PLACEMENTS = [
    [Shard(0)],
    [Replicate()],
]
ONE_D_TO_ONE_D_PLACEMENTS = [
    ([Replicate()], [Shard(0)]),
    ([Shard(0)], [Replicate()]),
]

TWO_D_PLACEMENTS = [
    [Replicate(), Replicate()],
    [Replicate(), Shard(0)],
    [Shard(0), Replicate()],
    [Shard(0), Shard(0)],
]
TWO_D_TO_TWO_D_PLACEMENTS = []
for p1 in TWO_D_PLACEMENTS:
    for p2 in TWO_D_PLACEMENTS:
        if p1 != p2:
            TWO_D_TO_TWO_D_PLACEMENTS.append((p1, p2))


@instantiate_parametrized_tests
class TestDTensorReshardPlacementChange(DTensorTestBase):
    # [CLASS DOC] Tests saving DTensors to HuggingFace storage and loading them with a
    # different placement on the same mesh shape.
    """
    Test DCP reshard for DTensor with placements changes and without world_size change and mesh_tensor change.
    """

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_1d_to_1d_reshard_placement_change(self) -> None:
        # [FUNCTION DOC] For each Replicate<->Shard(0) pair on a 1D mesh, saves a 4x4 DTensor
        # with a distributed HuggingFaceStorageWriter, loads into a zero DTensor with the new
        # placement via HuggingFaceStorageReader, and checks the full tensor and the
        # round trip to the original placement. Skips without safetensors; needs 2+ GPUs.
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        for one_d_to_one_d_placements in ONE_D_TO_ONE_D_PLACEMENTS:
            original_placement, new_placement = one_d_to_one_d_placements

            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (self.world_size,)
            device_mesh = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor, device_mesh, placements=original_placement
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.HuggingFaceStorageWriter(
                    path=CHECKPOINT_DIR,
                    save_distributed=True,
                ),
            )

            zero_dtensor = zeros(
                [4, 4], device_mesh=device_mesh, placements=new_placement
            )
            state_dict_to_load = {"dtensor": zero_dtensor}

            dist_cp.load(
                state_dict=state_dict_to_load,
                storage_reader=dist_cp.HuggingFaceStorageReader(
                    CHECKPOINT_DIR,
                ),
            )

            # materialize the whole tensor to compare with the original global_tensor
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                device_mesh,
                placements=[Replicate()],
            )
            self.assertEqual(global_tensor, state_dict_to_load["dtensor"].to_local())

            # redistribute the tensor back to its original placement for comparison.
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                device_mesh,
                placements=original_placement,
            )
            self.assertEqual(
                state_dict_to_save["dtensor"].to_local(),
                state_dict_to_load["dtensor"].to_local(),
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_2d_to_2d_reshard_placement_change(self) -> None:
        # [FUNCTION DOC] Same as the 1D test on a (2, world_size/2) mesh across every ordered
        # pair of distinct 2D placements, using the distributed HuggingFace writer and the
        # default save planner. Skips without safetensors; needs 4+ GPUs.
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir
        for two_d_to_two_d_placements in TWO_D_TO_TWO_D_PLACEMENTS:
            original_placement, new_placement = two_d_to_two_d_placements

            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor,
                mesh_2d,
                placements=original_placement,
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.HuggingFaceStorageWriter(
                    path=CHECKPOINT_DIR, save_distributed=True
                ),
                planner=dist_cp.DefaultSavePlanner(),
            )

            zero_dtensor = zeros([4, 4], device_mesh=mesh_2d, placements=new_placement)
            state_dict_to_load = {"dtensor": zero_dtensor}

            dist_cp.load(
                state_dict=state_dict_to_load,
                storage_reader=dist_cp.HuggingFaceStorageReader(CHECKPOINT_DIR),
            )

            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                mesh_2d,
                placements=[Replicate(), Replicate()],
            )
            self.assertEqual(global_tensor, state_dict_to_load["dtensor"].to_local())

            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                mesh_2d,
                placements=original_placement,
            )
            self.assertEqual(
                state_dict_to_save["dtensor"].to_local(),
                state_dict_to_load["dtensor"].to_local(),
            )


class TestDTensorReshardMeshChange(DTensorTestBase):
    # [CLASS DOC] Tests HuggingFace storage loads into DTensors whose mesh dimensionality
    # differs from the save-time mesh, plus an empty-shard case.
    """
    Test DCP reshard for DTensor with placements changes and mesh_tensor change.
    """

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_1d_to_2d_reshard_mesh_change(self) -> None:
        # [FUNCTION DOC] Saves a 4x4 DTensor on a 1D mesh (Shard(0) or Replicate), then for
        # each 2D placement loads into a zero DTensor on a 2D mesh and compares the fully
        # replicated result with the global tensor. Skips without safetensors; 2+ GPUs.
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir
        for placements_1d in ONE_D_PLACEMENTS:
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (self.world_size,)
            mesh_1d = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor, mesh_1d, placements=placements_1d
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.HuggingFaceStorageWriter(
                    path=CHECKPOINT_DIR, save_distributed=True
                ),
            )

            for placements_2d in TWO_D_PLACEMENTS:
                mesh_shape = (2, self.world_size // 2)
                mesh_2d = init_device_mesh(self.device_type, mesh_shape)

                zero_dtensor = zeros(
                    [4, 4], device_mesh=mesh_2d, placements=placements_2d
                )
                state_dict_to_load = {"dtensor": zero_dtensor}

                dist_cp.load(
                    state_dict=state_dict_to_load,
                    storage_reader=dist_cp.HuggingFaceStorageReader(CHECKPOINT_DIR),
                    planner=dist_cp.DefaultLoadPlanner(),
                )

                # materialzie the whole tensor to compare with the original global_tensor
                state_dict_to_load["dtensor"] = state_dict_to_load[
                    "dtensor"
                ].redistribute(
                    mesh_2d,
                    placements=[Replicate(), Replicate()],
                )
                self.assertEqual(
                    global_tensor, state_dict_to_load["dtensor"].to_local()
                )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_2d_to_1d_reshard_mesh_change(self) -> None:
        # [FUNCTION DOC] Saves on a 2D mesh with each 2D placement and loads into 1D-mesh
        # DTensors (Shard(0) or Replicate), comparing the replicated result with the global
        # tensor. Skips without safetensors; needs 4+ GPUs.
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir
        for placements_2d in TWO_D_PLACEMENTS:
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor, mesh_2d, placements=placements_2d
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.HuggingFaceStorageWriter(
                    path=CHECKPOINT_DIR, save_distributed=True
                ),
                planner=dist_cp.DefaultSavePlanner(),
            )

            for placements_1d in ONE_D_PLACEMENTS:
                mesh_shape = (self.world_size,)
                mesh_1d = init_device_mesh(self.device_type, mesh_shape)

                zero_dtensor = zeros(
                    [4, 4], device_mesh=mesh_1d, placements=placements_1d
                )
                state_dict_to_load = {"dtensor": zero_dtensor}

                dist_cp.load(
                    state_dict=state_dict_to_load,
                    storage_reader=dist_cp.HuggingFaceStorageReader(CHECKPOINT_DIR),
                    planner=dist_cp.DefaultLoadPlanner(),
                )

                # materialzie the whole tensor to compare with the original global_tensor
                state_dict_to_load["dtensor"] = state_dict_to_load[
                    "dtensor"
                ].redistribute(
                    mesh_1d,
                    placements=[Replicate()],
                )
                self.assertEqual(
                    global_tensor, state_dict_to_load["dtensor"].to_local()
                )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_dtensor_checkpoint_resharding_with_empty_shard(self):
        # [FUNCTION DOC] Saves a one-element tensor sharded across all ranks (most shards
        # empty) with the distributed HuggingFace writer, then loads it into a 2D mesh with
        # Shard(0) on both dims. Passes if no error is raised. Skips without safetensors.
        """
        Test dtensor checkpoint resharding with dtensor containing empty shards.
        """
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        tensor = torch.rand(1).to(self.device_type)
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dtensor = distribute_tensor(tensor, mesh, [Shard(0)])
        ref_state_dict = {"dtensor": dtensor}

        dist_cp.save(
            state_dict=ref_state_dict,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=self.temp_dir, save_distributed=True
            ),
        )

        tensor = torch.rand(1).to(self.device_type)
        mesh_2 = init_device_mesh(self.device_type, (2, self.world_size // 2))
        dtensor = distribute_tensor(tensor, mesh_2, [Shard(0), Shard(0)])
        state_dict = {"dtensor": dtensor}
        dist_cp.load(
            state_dict=state_dict,
            storage_reader=dist_cp.HuggingFaceStorageReader(self.temp_dir),
        )


if __name__ == "__main__":
    run_tests()
