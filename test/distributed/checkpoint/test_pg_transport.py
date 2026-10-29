# Owner(s): ["oncall: distributed"]
# [MODULE DOC]
# Tests for PGTransport, the process-group based checkpoint transport used to send and
# receive state dicts between ranks, and its helpers _cast_tensor, _prepare_tensor and
# _prepare_state_dict. Contains: multi-process ring exchange tests on CPU (gloo, 8 ranks)
# and accelerators (2 ranks) over plain tensors, DTensor/mixed content and ShardedTensor;
# single-process tests of the helpers; and mocked-process-group tests of send/recv call
# counts. Import-time side effects: defines module-level device_type (current accelerator
# type or cpu) and a logger.

import logging
import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard as ShardedTensorShard,
    ShardMetadata,
)
from torch.distributed.checkpoint._pg_transport import (
    _cast_tensor,
    _prepare_state_dict,
    _prepare_tensor,
    _StateDictMeta,
    _TensorMeta,
    PGTransport,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_distributed import (
    at_least_x_gpu,
    HAS_ACCELERATOR,
    MultiProcContinuousTest,
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

logger = logging.getLogger(__name__)


def _create_sharded_tensor_state_dict(
    rank: int, world_size: int, device: torch.device
) -> dict:
    # [FUNCTION DOC] HELPER: builds a state dict with two entries: "sharded_tensor", a
    # ShardedTensor with a global 64x8 shape of which this rank holds one row block (64
    # divided by world_size rows, values counting up from the block start), and
    # "rank_scalar", a scalar tensor holding the rank. Tensors are created on the device.
    # Assumes 64 divides evenly by world_size.
    """
    Create state_dict with ShardedTensor for deterministic testing.
    Args:
        rank: Current rank
        world_size: Total world size
        device: Device to create tensors on
    Returns:
        dict: State dictionary with ShardedTensor
    """
    # Create deterministic local shard for this rank
    global_size = 64
    shard_size = global_size // world_size
    start_idx = rank * shard_size
    end_idx = (rank + 1) * shard_size

    # Create local tensor with deterministic values
    local_tensor = torch.arange(
        start_idx * 8, end_idx * 8, dtype=torch.float32, device=device
    ).reshape(shard_size, 8)

    # Create ShardedTensor using init_from_local_shards
    sharded_tensor = init_from_local_shards(
        [
            ShardedTensorShard(
                tensor=local_tensor,
                metadata=ShardMetadata(
                    shard_offsets=[start_idx, 0],
                    shard_sizes=[shard_size, 8],
                    placement=f"rank:{rank}/{device}",
                ),
            )
        ],
        global_size,
        8,
    )

    return {
        "sharded_tensor": sharded_tensor,
        "rank_scalar": torch.tensor(float(rank), device=device),
    }


class SimpleModel(nn.Module):
    # [CLASS DOC] HELPER: two linear layers (10 -> 10) with a ReLU between, deterministic
    # initialization via a seed; its state dict is the payload for transport tests.
    def __init__(self, seed: int = 42):
        # [FUNCTION DOC] HELPER: seeds the global random generator with seed (side effect)
        # and creates net1, relu and net2.
        super().__init__()
        # Set seed for deterministic initialization
        torch.manual_seed(seed)
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: net1, ReLU, then net2 applied to x; returns the output.
        return self.net2(self.relu(self.net1(x)))


def ring_send_recv_checkpoint(
    transport: PGTransport, state_dict, rank, world_size, step=0
):
    # [FUNCTION DOC] HELPER: ring exchange. Every rank sends state_dict to the next rank
    # (wrapping around) and receives the previous rank's state dict, which it returns. Rank
    # 0 sends first then receives; all other ranks receive first then send, which avoids
    # deadlock on blocking transfers. The step argument is unused.
    """
    Use the transport to send to rank + 1 and receive from rank - 1.
    Each rank exchanges its own state_dict with the previous rank.
    """
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    if rank == 0:
        transport.send_checkpoint([next_rank], state_dict)
        received_checkpoint = transport.recv_checkpoint(prev_rank)
    else:
        received_checkpoint = transport.recv_checkpoint(prev_rank)
        transport.send_checkpoint([next_rank], state_dict)
    return received_checkpoint


def _test_pg_transport(self, device) -> None:
    # [FUNCTION DOC] HELPER: shared test body (self is the test case). Builds a model on
    # device, ring-exchanges its state dict through a PGTransport on the default group with
    # a 10 second timeout, and asserts the received dict equals the sent one (all ranks
    # use the same seed so states are identical).
    model = SimpleModel().to(device)
    transport = PGTransport(_get_default_group(), timedelta(seconds=10), device)
    original_state_dict = model.state_dict()
    received_checkpoint = ring_send_recv_checkpoint(
        transport=transport,
        state_dict=original_state_dict,
        rank=self.rank,
        world_size=self.world_size,
    )
    self.assertEqual(original_state_dict, received_checkpoint)


def _test_pg_transport_with_mixed_content(self, device) -> None:
    # [FUNCTION DOC] HELPER: shared test body. Builds a state dict mixing plain tensors, a
    # DTensor on a 1D device mesh, a string, a nested dict with a tensor and an integer, and
    # a list of ints; ring-exchanges it with PGTransport and asserts equality with the sent
    # dict.
    # Create a device mesh for DTensor
    device_mesh = init_device_mesh(device.type, (self.world_size,))

    # Create a DTensor
    local_tensor = torch.randn(10, 10, device=device)
    dtensor = DTensor.from_local(local_tensor, device_mesh)

    # Include mixed content in the state dict
    # Dtensor, Tensor, and non-tensor
    model = SimpleModel().to(device)
    state_dict = {
        "net1.weight": model.net1.weight.data,
        "net1.bias": model.net1.bias.data,
        "net2.weight": model.net2.weight.data,
        "net2.bias": model.net2.bias.data,
        "dtensor": dtensor,
        "non-tensor": "some string",
        "nested": {"tensor": torch.randn(1, 2), "value": 42},
        "list": [1, 2, 3],
    }

    transport = PGTransport(_get_default_group(), timedelta(seconds=10), device)
    received_checkpoint = ring_send_recv_checkpoint(
        transport=transport,
        state_dict=state_dict,
        rank=self.rank,
        world_size=self.world_size,
    )
    self.assertEqual(state_dict, received_checkpoint)


def _test_pg_transport_with_sharded_tensor(self, device) -> None:
    # [FUNCTION DOC] HELPER: shared test body. Sets the current accelerator index for non-cpu
    # devices, ring-exchanges the sharded tensor state dict, and checks that the received
    # scalar equals the previous rank's value and the received local shard equals the
    # previous rank's block. Expected values are on cpu because the transport returns cpu
    # tensors when no state dict callback is given. Prints the dicts (debug output).
    if device.type != "cpu":
        torch.accelerator.set_device_index(device)

    state_dict = _create_sharded_tensor_state_dict(self.rank, self.world_size, device)
    transport = PGTransport(_get_default_group(), timedelta(seconds=10), device)
    print(state_dict)
    received_checkpoint = ring_send_recv_checkpoint(
        transport=transport,
        state_dict=state_dict,
        rank=self.rank,
        world_size=self.world_size,
    )
    print("finished comms")
    print(received_checkpoint)

    # Validate that received checkpoint matches what we expect from rank - 1
    prev_rank = (self.rank - 1) % self.world_size

    # Compare rank_scalar (should be from previous rank)
    # Note: PGTransport moves received tensors to CPU when no state_dict callback is provided
    expected_rank_scalar = torch.tensor(float(prev_rank), device="cpu")
    received_rank_scalar = received_checkpoint["rank_scalar"]  # type: ignore[index]
    print(f"{expected_rank_scalar=} {received_rank_scalar=}")
    torch.testing.assert_close(expected_rank_scalar, received_rank_scalar)

    # For ShardedTensor, validate the local shard data matches what prev_rank would have
    received_st = received_checkpoint["sharded_tensor"]  # type: ignore[index]
    global_size = 64
    shard_size = global_size // self.world_size
    prev_start_idx = prev_rank * shard_size
    prev_end_idx = (prev_rank + 1) * shard_size
    expected_local_tensor = torch.arange(
        prev_start_idx * 8, prev_end_idx * 8, dtype=torch.float32, device="cpu"
    ).reshape(shard_size, 8)

    # Compare the actual tensor data
    received_local_tensor = received_st.local_shards()[0].tensor
    torch.testing.assert_close(expected_local_tensor, received_local_tensor)


class PgTransportCPU(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process test case: 8 ranks, gloo backend, cpu device, 20 second
    # timeout. Each test delegates to a shared ring-exchange test body.
    world_size = 8
    timeout: timedelta = timedelta(seconds=20)

    @classmethod
    def backend_str(cls) -> str | None:
        # [FUNCTION DOC] Returns the process group backend name, always gloo.
        return "gloo"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] Returns the device type string, always cpu.
        return "cpu"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] Returns the cpu torch device for the tests.
        return torch.device(self.device_type())

    def test_pg_transport(self) -> None:
        # [FUNCTION DOC] Ring exchange of a model state dict on cpu; see shared test body.
        _test_pg_transport(self, self.device)

    def test_pg_transport_with_mixed_content(self) -> None:
        # [FUNCTION DOC] Ring exchange of a state dict with DTensor and non-tensor content on cpu.
        _test_pg_transport_with_mixed_content(self, self.device)

    def test_pg_transport_with_sharded_tensor(self) -> None:
        # [FUNCTION DOC] Ring exchange of a state dict containing a ShardedTensor on cpu.
        _test_pg_transport_with_sharded_tensor(self, self.device)


@skip_but_pass_in_sandcastle_if(not at_least_x_gpu(2), "test requires 2+ accelerators")
class PgTransportGPU(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process test case: 2 ranks on accelerators using the default backend
    # for the accelerator, 20 second timeout. Skipped (but passing in sandcastle) when fewer
    # than 2 accelerators exist. Each rank uses the device whose index equals its rank.
    world_size = 2
    timeout: timedelta = timedelta(seconds=20)

    @classmethod
    def backend_str(cls) -> str | None:
        # [FUNCTION DOC] Returns the default distributed backend for the class's device type.
        return dist.get_default_backend_for_device(cls.device_type())

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] Returns the accelerator device indexed by this rank.
        return torch.device(f"{self.device_type()}:{self.rank}")

    @requires_accelerator_dist_backend()
    @skip_but_pass_in_sandcastle_if(
        not at_least_x_gpu(2), "test requires 2+ accelerators"
    )
    def test_pg_transport(self) -> None:
        # [FUNCTION DOC] Ring exchange of a model state dict on accelerators.
        _test_pg_transport(self, self.device)

    @requires_accelerator_dist_backend()
    @skip_but_pass_in_sandcastle_if(
        not at_least_x_gpu(2), "test requires 2+ accelerators"
    )
    def test_pg_transport_with_mixed_content(self) -> None:
        # [FUNCTION DOC] Ring exchange of a state dict with DTensor and non-tensor content on
        # accelerators.
        _test_pg_transport_with_mixed_content(self, self.device)

    @requires_accelerator_dist_backend()
    @skip_but_pass_in_sandcastle_if(
        not at_least_x_gpu(2), "test requires 2+ accelerators"
    )
    def test_pg_transport_with_sharded_tensor(self) -> None:
        # [FUNCTION DOC] Ring exchange of a state dict containing a ShardedTensor on
        # accelerators.
        _test_pg_transport_with_sharded_tensor(self, self.device)


class TestCastTensor(TestCase):
    # [CLASS DOC] Single-process tests for _cast_tensor, which reinterprets a tensor's
    # underlying storage as a flat tensor of another dtype without copying.
    def test_cast_tensor_different_dtypes(self):
        # [FUNCTION DOC] For several source dtypes, casts to uint8 and asserts the result
        # shares the same untyped storage object and has one element per storage byte.
        """Test casting tensors of different dtypes."""
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]

        for dtype in dtypes:
            original = torch.tensor([1, 2, 3], dtype=dtype)
            casted = _cast_tensor(original, torch.uint8)

            # Check that the storage is the same
            self.assertIs(original.untyped_storage(), casted.untyped_storage())

            # Check that the size is correct
            self.assertEqual(casted.numel(), original.untyped_storage().nbytes())

    def test_cast_tensor_with_stride(self):
        # [FUNCTION DOC] Casts a transposed (non-contiguous) tensor to uint8 and asserts the
        # same storage is shared and the element count equals the storage byte size.
        """Test casting tensors with non-standard strides."""
        # Create a tensor with non-standard stride
        original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        transposed = original.t()  # Transpose to get non-standard stride

        casted = _cast_tensor(transposed, torch.uint8)

        # Check that the storage is the same
        self.assertIs(transposed.untyped_storage(), casted.untyped_storage())

        # Check that the size is correct
        self.assertEqual(casted.numel(), transposed.untyped_storage().nbytes())

    def test_cast_tensor_with_offset(self):
        # [FUNCTION DOC] Casts a sliced tensor (nonzero storage offset) to uint8 and asserts
        # the cast covers the whole underlying storage, not just the slice.
        """Test casting tensors with storage offset."""
        # Create a tensor with storage offset
        original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        sliced = original[2:]  # This creates a tensor with storage offset

        casted = _cast_tensor(sliced, torch.uint8)

        # Check that the storage is the same
        self.assertIs(sliced.untyped_storage(), casted.untyped_storage())

        # Check that the size is correct
        self.assertEqual(casted.numel(), sliced.untyped_storage().nbytes())


class TestPrepareTensor(TestCase):
    # [CLASS DOC] Single-process tests for _prepare_tensor, which returns a uint8 view of
    # a tensor's storage plus a _TensorMeta (shape, dtype, storage offset, stride, nbytes).
    def test_prepare_tensor_basic(self):
        # [FUNCTION DOC] Prepares a small float tensor; asserts every metadata field matches
        # the tensor (nbytes is the storage size) and the prepared tensor is uint8 with one
        # element per storage byte.
        """Test basic tensor preparation."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        prepared_tensor, meta = _prepare_tensor(tensor)

        # Check metadata
        self.assertEqual(meta.shape, tensor.shape)
        self.assertEqual(meta.dtype, tensor.dtype)
        self.assertEqual(meta.storage_offset, tensor.storage_offset())
        self.assertEqual(meta.stride, tensor.stride())
        self.assertEqual(meta.nbytes, tensor.untyped_storage().nbytes())

        # Check prepared tensor
        self.assertEqual(prepared_tensor.dtype, torch.uint8)
        self.assertEqual(prepared_tensor.numel(), tensor.untyped_storage().nbytes())

    def test_prepare_tensor_different_shapes(self):
        # [FUNCTION DOC] Repeats the metadata checks for 1D, 2D and 3D random tensors.
        """Test preparing tensors with different shapes."""
        shapes = [(3,), (2, 3), (2, 3, 4)]

        for shape in shapes:
            tensor = torch.randn(shape)
            prepared_tensor, meta = _prepare_tensor(tensor)

            # Check metadata
            self.assertEqual(meta.shape, tensor.shape)
            self.assertEqual(meta.dtype, tensor.dtype)
            self.assertEqual(meta.storage_offset, tensor.storage_offset())
            self.assertEqual(meta.stride, tensor.stride())
            self.assertEqual(meta.nbytes, tensor.untyped_storage().nbytes())

    def test_prepare_tensor_with_stride(self):
        # [FUNCTION DOC] Prepares a transposed tensor and asserts the metadata records its
        # non-standard stride along with shape, dtype, offset and storage byte size.
        """Test preparing tensors with non-standard strides."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        transposed = tensor.t()  # Transpose to get non-standard stride

        prepared_tensor, meta = _prepare_tensor(transposed)

        # Check metadata
        self.assertEqual(meta.shape, transposed.shape)
        self.assertEqual(meta.dtype, transposed.dtype)
        self.assertEqual(meta.storage_offset, transposed.storage_offset())
        self.assertEqual(meta.stride, transposed.stride())
        self.assertEqual(meta.nbytes, transposed.untyped_storage().nbytes())


class TestPrepareStateDict(TestCase):
    # [CLASS DOC] Single-process tests for _prepare_state_dict, which flattens a state dict
    # into a _StateDictMeta (tree spec, key paths, per-leaf metadata or raw non-tensor value)
    # and a list of prepared tensors.
    def test_prepare_state_dict_basic(self):
        # [FUNCTION DOC] Two-tensor dict on cpu: asserts two paths, two leaves, two prepared
        # tensors, and that every leaf is a _TensorMeta.
        """Test basic state dict preparation."""
        state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
        device = torch.device("cpu")

        meta, tensors = _prepare_state_dict(state_dict, device)

        # Check metadata
        self.assertEqual(len(meta.paths), 2)
        self.assertEqual(len(meta.non_tensor_leaves), 2)
        self.assertEqual(len(tensors), 2)

        # Check that all non_tensor_leaves are _TensorMeta instances
        for leaf in meta.non_tensor_leaves:
            self.assertIsInstance(leaf, _TensorMeta)

    def test_prepare_state_dict_nested(self):
        # [FUNCTION DOC] Two nested dicts of two tensors each: asserts four paths, four leaves
        # and four prepared tensors, i.e. nesting is flattened.
        """Test preparing nested state dict."""
        state_dict = {
            "layer1": {"weight": torch.randn(3, 4), "bias": torch.randn(4)},
            "layer2": {"weight": torch.randn(4, 5), "bias": torch.randn(5)},
        }
        device = torch.device("cpu")

        meta, tensors = _prepare_state_dict(state_dict, device)

        # Check metadata
        self.assertEqual(len(meta.paths), 4)
        self.assertEqual(len(meta.non_tensor_leaves), 4)
        self.assertEqual(len(tensors), 4)

    def test_prepare_state_dict_with_non_tensor_values(self):
        # [FUNCTION DOC] Dict with two tensors, a nested config dict of two floats and an
        # int: asserts leaves and paths have equal counts, only two tensors are prepared, and
        # the three non-tensor values are kept as plain (non-_TensorMeta) leaves.
        """Test preparing state dict with non-tensor values."""
        state_dict = {
            "weight": torch.randn(3, 4),
            "bias": torch.randn(4),
            "config": {"lr": 0.01, "momentum": 0.9},
            "step": 42,
        }
        device = torch.device("cpu")

        meta, tensors = _prepare_state_dict(state_dict, device)

        # Check metadata - the actual number of paths depends on how the pytree flattens the dict
        # The nested config dict might be flattened differently
        self.assertEqual(len(meta.non_tensor_leaves), len(meta.paths))
        self.assertEqual(len(tensors), 2)

        # Check that non-tensor values are preserved
        non_tensor_values = [
            leaf for leaf in meta.non_tensor_leaves if not isinstance(leaf, _TensorMeta)
        ]
        self.assertEqual(len(non_tensor_values), 3)  # config (2) and step


class TestPGTransportMocked(TestCase):
    # [CLASS DOC] Single-process tests of PGTransport using a mock process group whose send
    # and recv return a mock work handle. State: cpu device, 10 second timeout, pg, and
    # mock_work (records wait calls).
    def setUp(self):
        # [FUNCTION DOC] Creates the cpu device, mock process group, timeout and mock work
        # handle; wires pg send and recv to return the work handle.
        super().setUp()
        self.device = torch.device("cpu")
        self.pg = MagicMock()
        self.timeout = timedelta(seconds=10)

        # Mock Work object
        self.mock_work = MagicMock()
        self.mock_work.wait = MagicMock()

        # Setup process group mock to return mock_work
        self.pg.send = MagicMock(return_value=self.mock_work)
        self.pg.recv = MagicMock(return_value=self.mock_work)

    def test_send_checkpoint_basic(self):
        # [FUNCTION DOC] Sends a two-tensor state dict to two destination ranks; asserts the
        # process group send count and the work wait count both equal, per destination, one
        # metadata length send, one metadata send and one send per tensor.
        """Test basic send_checkpoint functionality with mocked process group."""
        transport = PGTransport(self.pg, self.timeout, self.device)
        state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
        dst_ranks = [1, 2]

        transport.send_checkpoint(dst_ranks, state_dict)

        # Check that send was called with correct parameters
        # First for metadata length, then for metadata, then for each tensor
        expected_calls = len(dst_ranks) * (2 + len(state_dict))
        self.assertEqual(self.pg.send.call_count, expected_calls)

        # Check that wait was called on all work objects
        self.assertEqual(self.mock_work.wait.call_count, expected_calls)

    def test_recv_checkpoint_basic(self):
        # [FUNCTION DOC] Patches pickle loading to return a prebuilt _StateDictMeta for a
        # two-tensor dict, and makes the mock recv fill the single-element length buffer with
        # 100. Calls recv_checkpoint from rank 0 and asserts at least two recv calls (length
        # and metadata buffer) and at least two work waits.
        """Test basic recv_checkpoint functionality with mocked process group."""
        # Setup mock for pickle.loads to return a valid _StateDictMeta
        with patch("pickle.loads") as mock_loads:
            # Create a mock state dict metadata
            from torch.utils._pytree import tree_flatten_with_path

            state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
            leaves, treespec = tree_flatten_with_path(state_dict)
            paths = [path for path, _ in leaves]

            # Create mock tensor metadata
            tensor_metas = []
            for _, v in leaves:
                tensor_metas.append(
                    _TensorMeta(
                        shape=v.shape,
                        dtype=v.dtype,
                        storage_offset=v.storage_offset(),
                        stride=v.stride(),
                        nbytes=v.untyped_storage().nbytes(),
                    )
                )

            mock_meta = _StateDictMeta(
                treespec=treespec, paths=paths, non_tensor_leaves=tensor_metas
            )
            mock_loads.return_value = mock_meta

            # Setup len_t and buf tensors for the mock recv
            def side_effect(tensor_list, *args, **kwargs):
                # [FUNCTION DOC] HELPER: fake recv; if the first buffer has one element
                # (the length buffer) fills it with 100 in place; returns the mock work.
                if tensor_list[0].numel() == 1:  # This is len_t
                    tensor_list[0].fill_(100)  # Some arbitrary length
                return self.mock_work

            self.pg.recv.side_effect = side_effect

            # Create transport and call recv_checkpoint
            transport = PGTransport(self.pg, self.timeout, self.device)
            transport.recv_checkpoint(src_rank=0)

            # Check that recv was called
            self.assertGreaterEqual(
                self.pg.recv.call_count, 2
            )  # At least for len_t and buf

            # Check that wait was called
            self.assertGreaterEqual(self.mock_work.wait.call_count, 2)

    def test_send_checkpoint_empty_state_dict(self):
        # [FUNCTION DOC] Sends an empty state dict to one rank; asserts only the two metadata
        # sends happen (length and metadata buffer), each waited on once.
        """Test send_checkpoint with empty state dict."""
        transport = PGTransport(self.pg, self.timeout, self.device)
        state_dict = {}
        dst_ranks = [1]

        transport.send_checkpoint(dst_ranks, state_dict)

        # Check that send was called only for metadata
        self.assertEqual(self.pg.send.call_count, 2)  # len_t and buf_t

        # Check that wait was called
        self.assertEqual(self.mock_work.wait.call_count, 2)

    def test_send_checkpoint_with_non_tensor_values(self):
        # [FUNCTION DOC] Sends a dict with one tensor and a nested non-tensor config to one
        # rank; asserts three sends and three waits (two metadata, one tensor), i.e. non-tensor
        # values travel inside the metadata only.
        """Test send_checkpoint with non-tensor values in state dict."""
        transport = PGTransport(self.pg, self.timeout, self.device)
        state_dict = {"weight": torch.randn(3, 4), "config": {"lr": 0.01}}
        dst_ranks = [1]

        transport.send_checkpoint(dst_ranks, state_dict)

        # Check that send was called for metadata and one tensor
        self.assertEqual(self.pg.send.call_count, 3)  # len_t, buf_t, and one tensor

        # Check that wait was called
        self.assertEqual(self.mock_work.wait.call_count, 3)

    def test_recv_checkpoint_with_state_dict_callback(self):
        # [FUNCTION DOC] Same mocked receive setup as the basic recv test, but the transport
        # is built with a state dict callback returning destination tensors; asserts the
        # callback is called exactly once during recv_checkpoint.
        """Test recv_checkpoint with state_dict callback."""
        # Setup mock for pickle.loads to return a valid _StateDictMeta
        with patch("pickle.loads") as mock_loads:
            # Create a mock state dict metadata
            from torch.utils._pytree import tree_flatten_with_path

            state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
            leaves, treespec = tree_flatten_with_path(state_dict)
            paths = [path for path, _ in leaves]

            # Create mock tensor metadata
            tensor_metas = []
            for _, v in leaves:
                tensor_metas.append(
                    _TensorMeta(
                        shape=v.shape,
                        dtype=v.dtype,
                        storage_offset=v.storage_offset(),
                        stride=v.stride(),
                        nbytes=v.untyped_storage().nbytes(),
                    )
                )

            mock_meta = _StateDictMeta(
                treespec=treespec, paths=paths, non_tensor_leaves=tensor_metas
            )
            mock_loads.return_value = mock_meta

            # Setup len_t and buf tensors for the mock recv
            def side_effect(tensor_list, *args, **kwargs):
                # [FUNCTION DOC] HELPER: fake recv; if the first buffer has one element
                # (the length buffer) fills it with 100 in place; returns the mock work.
                if tensor_list[0].numel() == 1:  # This is len_t
                    tensor_list[0].fill_(100)  # Some arbitrary length
                return self.mock_work

            self.pg.recv.side_effect = side_effect

            # Create a state_dict callback
            callback_state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
            state_dict_callback = MagicMock(return_value=callback_state_dict)

            # Create transport with state_dict callback and call recv_checkpoint
            transport = PGTransport(
                self.pg, self.timeout, self.device, state_dict=state_dict_callback
            )
            transport.recv_checkpoint(src_rank=0)

            # Check that state_dict callback was called
            state_dict_callback.assert_called_once()


class TestPGTransportEdgeCases(TestCase):
    # [CLASS DOC] Single-process edge case tests of PGTransport with a mock process group;
    # same fixture state as the mocked test case (device, pg, timeout, mock_work).
    def setUp(self):
        # [FUNCTION DOC] Creates the cpu device, mock process group, timeout and mock work
        # handle; wires pg send and recv to return the work handle.
        super().setUp()
        self.device = torch.device("cpu")
        self.pg = MagicMock()
        self.timeout = timedelta(seconds=10)

        # Mock Work object
        self.mock_work = MagicMock()
        self.mock_work.wait = MagicMock()

        # Setup process group mock to return mock_work
        self.pg.send = MagicMock(return_value=self.mock_work)
        self.pg.recv = MagicMock(return_value=self.mock_work)

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_send_checkpoint_with_cpu_tensors(self):
        # [FUNCTION DOC] Only runs when an accelerator exists. Builds a transport targeting
        # accelerator device 0 and sends a state dict of two cpu tensors; asserts at least four
        # sends and four waits (two metadata, two tensors).
        """Test send_checkpoint with CPU tensors when device is accelerator."""
        device = torch.device(f"{device_type}:0")

        # Create a state dict with CPU tensors
        state_dict = {
            "cpu_tensor1": torch.randn(2, 3),
            "cpu_tensor2": torch.randn(3, 4),
        }

        # Create transport with accelerator device
        transport = PGTransport(self.pg, self.timeout, device)

        # Call send_checkpoint
        transport.send_checkpoint([1], state_dict)

        # Check that send was called
        self.assertGreaterEqual(
            self.pg.send.call_count, 4
        )  # len_t, buf_t, and 2 tensors

        # Check that wait was called
        self.assertGreaterEqual(self.mock_work.wait.call_count, 4)


if __name__ == "__main__":
    run_tests()
