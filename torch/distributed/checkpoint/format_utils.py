# mypy: allow-untyped-defs
# [MODULE DOC] Conversion between DCP checkpoints and single-file torch.save checkpoints.
# Provides dcp_to_torch_save, torch_save_to_dcp, a storage reader that loads a torch.save file on the
# coordinator and broadcasts tensors to all ranks (BroadcastingTorchSaveReader), and a load planner that
# builds metadata from the given state dict (DynamicMetaLoadPlanner). Running the module as a script
# exposes a command line converter taking mode, source path and destination path.
import argparse
import os
from enum import Enum
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner,
    DefaultLoadPlanner,
)
from torch.distributed.checkpoint.metadata import (
    Metadata,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner
from torch.distributed.checkpoint.planner_helpers import _create_chunk_list
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
from torch.distributed.checkpoint.storage import StorageReader
from torch.futures import Future


__all__ = [
    "dcp_to_torch_save",
    "torch_save_to_dcp",
    "BroadcastingTorchSaveReader",
    "DynamicMetaLoadPlanner",
]


class BroadcastingTorchSaveReader(StorageReader):
    """
    # [CLASS DOC] StorageReader for a torch.save file. The coordinator rank loads the whole file on CPU; every
    # rank then receives each requested tensor by broadcast and copies its slice into the target. Only tensors are
    # supported. State: checkpoint_id (file path), coordinator_rank, and is_coordinator (set in set_up_storage_reader).
    # Meant to be paired with DynamicMetaLoadPlanner.
    StorageReader for reading a Torch Save file. This reader will read the entire checkpoint
    on the coordinator rank, and then broadcast and shard each tensor to all ranks.

    . N.B. Intended to be used with DynamicMetaLoadPlanner

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    """

    def __init__(
        self,
        checkpoint_id: str | os.PathLike | None = None,
        coordinator_rank: int = 0,
    ) -> None:
        # [FUNCTION DOC] Stores checkpoint_id and coordinator_rank. is_coordinator is not set until set_up_storage_reader.
        self.checkpoint_id = checkpoint_id
        self.coordinator_rank = coordinator_rank

    # pyrefly: ignore [bad-override]
    def read_metadata(self) -> Metadata:
        # [FUNCTION DOC] Returns an empty Metadata; real metadata is built by the planner since the file has none.
        """Extends the default StorageReader to support building the metadata file"""
        # Metadata is built in planner.set_up_planner, since we are not actually reading metadata from
        # the disk
        return Metadata(state_dict_metadata={})

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # [FUNCTION DOC] Executes the load plan. Coordinator loads the file (flattening it if the planner flattens state
        # dicts); for each item, the coordinator moves its tensor to the default process-group device, other ranks allocate
        # an empty tensor like the planner's entry, then the tensor is broadcast from the coordinator, narrowed to the item's
        # offsets and lengths, and copied into the planner's target tensor and committed. Raises RuntimeError for byte
        # (non-tensor) items and AssertionError on size mismatch or missing checkpoint_id. Returns an already-completed Future.
        # Side effects: collective broadcasts (all ranks must participate), in-place tensor writes.
        """
        Reads torch save data on the coordinator rank, and broadcast afterwards
        this incurrs a communication cost, but avoids having to load
        the entire checkpoint on each rank, hopefully preventing OOM issues
        """
        planner = cast(DefaultLoadPlanner, planner)

        # data is read in on the coordinator rank, and broadcast afterwards
        # this incurs a communication cost, but it avoids having to load
        # the entire checkpoint on each rank, hopefully preventing OOM issues
        # TODO: read on each host, instead of only the coordinator
        if self.is_coordinator:
            if self.checkpoint_id is None:
                raise AssertionError("checkpoint_id must be set before reading data")
            torch_state_dict = torch.load(
                self.checkpoint_id, map_location="cpu", weights_only=False
            )
            if planner.flatten_state_dict:
                torch_state_dict, _ = flatten_state_dict(torch_state_dict)
        else:
            torch_state_dict = None

        for req in plan.items:
            if req.type == LoadItemType.BYTE_IO:
                raise RuntimeError(
                    f"Non-tensor value identified at {req.storage_index.fqn}. "
                    f"At this time {type(self).__name__} only supports loading Tensors."
                )

            #  Broadcast the tensor from the coordinator rank
            if self.is_coordinator:
                pg_device = dist.distributed_c10d._get_pg_default_device()
                # pyrefly: ignore [unsupported-operation]
                tensor = torch_state_dict[req.storage_index.fqn].to(pg_device)
            else:
                tensor = torch.empty_like(planner.state_dict[req.storage_index.fqn])

            dist.broadcast(tensor, src=self.coordinator_rank, async_op=False)

            tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
            target_tensor = planner.resolve_tensor(req).detach()
            if not target_tensor.size() == tensor.size():
                raise AssertionError(
                    f"req {req.storage_index} mismatch sizes, "
                    f"{target_tensor.size()} vs {tensor.size()}"
                )
            target_tensor.copy_(tensor)
            planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # pyrefly: ignore [bad-override]
    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        # [FUNCTION DOC] Records whether this rank is the coordinator. Raises AssertionError if the coordinator flag is set
        # but the rank differs from coordinator_rank, or if checkpoint_id is unset. metadata is unused.
        """Implementation of the StorageReader method"""
        self.is_coordinator = is_coordinator
        if self.is_coordinator:
            if not dist.get_rank() == self.coordinator_rank:
                raise AssertionError(
                    f"Coordinator rank mismatch: expected {self.coordinator_rank}, "
                    f"got {dist.get_rank()}"
                )

        if self.checkpoint_id is None:
            raise AssertionError(
                "checkpoint_id must be set before setting up storage reader"
            )

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        # [FUNCTION DOC] Returns the plan unchanged.
        """Implementation of the StorageReader method"""
        return plan

    def prepare_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        # [FUNCTION DOC] Returns the global plan unchanged.
        """Implementation of the StorageReader method"""
        return global_plan

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        # [FUNCTION DOC] Resets the reader to a new checkpoint_id.
        """Implementation of the StorageReader method"""
        self.checkpoint_id = checkpoint_id

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        # [FUNCTION DOC] Returns True if the given id is an existing regular file.
        """Implementation of the StorageReader method"""
        return os.path.isfile(checkpoint_id)


class DynamicMetaLoadPlanner(DefaultLoadPlanner):
    # [CLASS DOC] DefaultLoadPlanner variant that builds Metadata from the state dict being loaded instead of reading
    # it from storage, so formats without a metadata file (such as torch.save) can be loaded. Tensors only.
    """
    Extension of DefaultLoadPlanner, which creates a new Metadata object based on the passed in state dict,
    avoiding the need to read metadata from disk. This is useful when reading formats which don't have a
    metadata file, like Torch Save files.

    . N.B. Intended to be used with BroadcastingTorchSaveReader

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        # [FUNCTION DOC] Runs the default setup, then builds tensor storage metadata (dtype, size, chunk list) for every
        # entry of the state dict and stores it as the planner's metadata. Raises RuntimeError if any value is not a tensor.
        """Setups of the planner, extnding default behavior by creating the Metadata object from the state dict"""
        super().set_up_planner(state_dict, metadata, is_coordinator)

        state_dict_metadata: dict[str, STORAGE_TYPES] = {}
        for key, tensor in self.state_dict.items():
            if not torch.is_tensor(tensor):
                raise RuntimeError(
                    f"Non-tensor value identified at {key}. "
                    f"At this time {type(self).__name__} only supports loading Tensors."
                )

            state_dict_metadata[key] = TensorStorageMetadata(
                TensorProperties(dtype=tensor.dtype),
                tensor.size(),
                _create_chunk_list(tensor),
            )
        self.metadata = Metadata(state_dict_metadata=state_dict_metadata)


def dcp_to_torch_save(
    dcp_checkpoint_dir: str | os.PathLike,
    torch_save_path: str | os.PathLike,
):
    # [FUNCTION DOC] Converts a DCP checkpoint directory to one torch.save file: loads the whole checkpoint into a plain
    # state dict without distributed coordination (no process group needed) and saves it to torch_save_path. Returns None.
    # Holds the whole checkpoint in memory.
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch save file.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        torch_save_path: Filename to store the converted Torch save file.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    torch.save(sd, torch_save_path)


def torch_save_to_dcp(
    torch_save_path: str | os.PathLike,
    dcp_checkpoint_dir: str | os.PathLike,
):
    # [FUNCTION DOC] Converts a torch.save file to a DCP checkpoint directory: loads the file (unrestricted unpickling)
    # and writes it with a filesystem writer without distributed coordination. Returns None.
    """
    Given the location of a torch save file, converts it into a DCP checkpoint.

    Args:
        torch_save_path: Filename of the Torch save file.
        dcp_checkpoint_dir: Directory to store the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """

    state_dict = torch.load(torch_save_path, weights_only=False)
    # we don't need stateful behavior here because the expectation is anything loaded by
    # torch.load would not contain stateful objects.
    _save_state_dict(
        state_dict, storage_writer=FileSystemWriter(dcp_checkpoint_dir), no_dist=True
    )


if __name__ == "__main__":

    class FormatMode(Enum):
        # [CLASS DOC] HELPER: Enum of the two command line conversion modes.
        TORCH_TO_DCP = "torch_to_dcp"
        DCP_TO_TORCH = "dcp_to_torch"

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        help="Conversion mode",
        choices=[m.value for m in FormatMode],
        default=FormatMode.TORCH_TO_DCP,
    )
    parser.add_argument("src", type=str, help="Path to the source model")
    parser.add_argument("dst", type=str, help="Path to the destination model")
    args = parser.parse_args()

    print(
        f"Converting checkpoint from {args.src} to {args.dst} using method: '{args.mode}'"
    )
    checkpoint_missing_warning = (
        f"No checkpoint found at {args.src}. Skipping conversion."
    )
    if args.mode == FormatMode.TORCH_TO_DCP.value:
        if os.path.isfile(args.src):
            torch_save_to_dcp(args.src, args.dst)
        else:
            print(checkpoint_missing_warning)
    elif args.mode == FormatMode.DCP_TO_TORCH.value:
        if os.path.isdir(args.src):
            dcp_to_torch_save(args.src, args.dst)
        else:
            print(checkpoint_missing_warning)
    else:
        raise ValueError(f"Unknown conversion mode: {args.mode}")
