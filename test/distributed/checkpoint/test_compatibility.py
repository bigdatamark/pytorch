# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Backward-compatibility tests for distributed checkpoint (DCP): metadata
# dataclasses must be constructible with legacy arguments, checkpoints written with the
# legacy sharded-tensor TensorProperties and with the version 2_3 behavior must still load,
# and storage metadata must carry the checkpoint id and save/load ids. Some tests write to
# a fixed directory under /tmp; no import-time side effects.

from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class TestDCPCompatbility(TestCase):
    # [CLASS DOC] Single-process TestCase guarding DCP metadata and on-disk format BC.
    def test_metadata(self) -> None:
        # [FUNCTION DOC] Constructs chunk, tensor, bytes, Metadata and MetadataIndex objects
        # using only the minimal original fields; any exception is re-raised as a
        # RuntimeError saying the change may break DCP backward compatibility, which
        # enforces that newly added metadata fields keep default values.
        # Ensure that all the new fields of all the metadata have the default
        # values so that we can always deserialize from a legacy metadata.
        try:
            tensor = torch.zeros(4, 4)
            chunk_meta = ChunkStorageMetadata(
                torch.Size((1, 1)),
                torch.Size((1, 1)),
            )
            tensor_meta = TensorStorageMetadata(
                properties=TensorProperties.create_from_tensor(tensor),
                size=tensor.size(),
                chunks=[chunk_meta],
            )
            b_meta = BytesStorageMetadata()
            _ = Metadata(state_dict_metadata={"a": tensor_meta, "b": b_meta})

            _ = MetadataIndex(fqn="a.b.c")
        except Exception as e:
            raise RuntimeError(
                "The change may break the BC of distributed checkpoint."
            ) from e

    def test_sharded_tensor_dependency(self) -> None:
        # [FUNCTION DOC] Saves a state dict while the checkpoint metadata module's
        # TensorProperties is patched to the legacy sharded-tensor TensorProperties class,
        # then loads it back with a fresh reader. Passes if load does not raise.
        # Side effect: writes to the fixed directory /tmp/dcp_testing (not cleaned up).
        # KNOWN ISSUE: the load result is never compared with the saved values.
        # Ensure that we can load the existing DCP checkpoints back even if the
        # metadata contain # _shard.sharded_tensor.metadata.
        from torch.distributed._shard.sharded_tensor.metadata import (
            TensorProperties as stp,
        )

        with patch("torch.distributed.checkpoint.metadata.TensorProperties", stp):
            dcp.save(
                {"a": torch.zeros(4, 4)},
                dcp.FileSystemWriter("/tmp/dcp_testing"),
            )

        dcp.load(
            {"a": torch.zeros(4, 4)},
            dcp.FileSystemReader("/tmp/dcp_testing"),
        )

    @with_temp_dir
    def test_storage_meta(self) -> None:
        # [FUNCTION DOC] Saves a state dict to a temp dir with a FileSystemWriter, reads the
        # metadata back with a FileSystemReader, and checks storage_meta is present and its
        # checkpoint_id, save_id and load_id equal the temp dir, the writer's save_id and
        # the reader's load_id respectively.
        writer = dcp.FileSystemWriter(self.temp_dir)
        dcp.save({"a": torch.zeros(4, 4)}, storage_writer=writer)

        reader = dcp.FileSystemReader(self.temp_dir)
        storage_meta = reader.read_metadata().storage_meta
        self.assertNotEqual(storage_meta, None)
        self.assertEqual(str(storage_meta.checkpoint_id), self.temp_dir)
        self.assertEqual(storage_meta.save_id, writer.save_id)
        self.assertEqual(storage_meta.load_id, reader.load_id)

    @with_temp_dir
    def test_with_v_2_3(self) -> None:
        # [FUNCTION DOC] Saves a nested state dict (tensor, dicts, list) while the module
        # global _act_like_version in dcp._version is set to 2_3, resets it to None, then
        # loads into a differently valued state dict and asserts it now equals the saved one.
        # Side effect: temporarily mutates that global; it is not restored if save raises.
        sd = {
            "a": torch.zeros(4, 4),
            "dict": {
                "dict_a": {"dict_a_1": 1, "dict_a_2": 2},
                "dict_b": {"dict_b_1": 1, "dict_b_2": 2},
            },
            "list": [0, 1, 2, 3, 4, 5],
        }
        load_sd = {
            "a": torch.ones(4, 4),
            "dict": {
                "dict_a": {"dict_a_1": 2, "dict_a_2": 4},
                "dict_b": {"dict_b_1": 2, "dict_b_2": 4},
            },
            "list": [10, 11, 12, 13, 14, 15],
        }

        dcp._version._act_like_version = "2_3"
        dcp.save(sd, checkpoint_id=self.temp_dir)
        dcp._version._act_like_version = None
        dcp.load(load_sd, checkpoint_id=self.temp_dir)
        self.assertEqual(sd, load_sd)


if __name__ == "__main__":
    run_tests()
