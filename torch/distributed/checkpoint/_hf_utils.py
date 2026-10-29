# [MODULE DOC] HELPER: Constants and helpers for the HuggingFace safetensors checkpoint format used by DCP:
# file name templates, metadata keys, a dtype-string to torch dtype map, DCP version markers, and functions
# to generate file names, parse a safetensors header, map dtype strings and extract DCP custom metadata.
# No import-time side effects beyond defining constants.
import io
import json
import struct
from dataclasses import dataclass
from typing import Any

import torch


_metadata_fn: str = "model.safetensors.index.json"

FILE_NAME = "model-{cpt_idx}-of-{num_files}"
SHARDED_FILE_NAME = "shard-{shard_idx}-model-{cpt_idx}-of-{num_files}"
SUFFIX = ".safetensors"

# metadata keys
CUSTOM_METADATA_KEY = "DCP_SHARDING_INFO"
DEFAULT_EXTRA_METADATA_KEY = "__metadata__"
SAVED_OFFSETS_KEY = "saved_offsets"
SHAPE_KEY = "shape"
DATA_KEY = "data"
DTYPE_KEY = "dtype"
DATA_OFFSETS_KEY = "data_offsets"

DTYPE_MAP = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BF16": torch.bfloat16,
}

HF_DCP_VERSION: float = 1.0
DCP_VERSION_KEY = "DCP_VERSION"
DCP_SHARDING_INFO_KEY = "DCP_SHARDING_INFO"

FORMAT_KEY = "format"
FORMAT_VALUE = "pt"

NUM_BYTES_FOR_HEADER_LEN = 8

SHARDED_DIR_NAME = "sharded"


@dataclass
class _HFStorageInfo:
    # [CLASS DOC] HELPER: Per-tensor storage info: relative file path, tensor shape and dtype.
    """This is the per entry storage info."""

    relative_path: str
    shape: torch.Size
    dtype: torch.dtype


def _gen_file_name(
    index: int, largest_index: int, shard_index: int | None = None
) -> str:
    # [FUNCTION DOC] HELPER: Builds a safetensors file name from a 1-based index and total file count (both zero-padded
    # to five digits), using the sharded name template when shard_index is given, and appends the suffix.
    if shard_index is not None:
        return (
            SHARDED_FILE_NAME.format(
                shard_idx=f"{shard_index}".zfill(5),
                cpt_idx=f"{index}".zfill(5),
                num_files=f"{largest_index}".zfill(5),
            )
            + SUFFIX
        )
    else:
        return (
            FILE_NAME.format(
                cpt_idx=f"{index}".zfill(5), num_files=f"{largest_index}".zfill(5)
            )
            + SUFFIX
        )


def _get_safetensors_file_metadata(file_bytes: io.IOBase) -> tuple[Any, int]:
    # [FUNCTION DOC] HELPER: Reads a safetensors header from an open binary stream positioned at the start: an 8-byte
    # little-endian length followed by that many bytes of JSON. Returns a tuple of the parsed metadata and the
    # total header size (JSON length plus 8), i.e. the data start offset.
    # this uses the same logic that's done in HF code base
    # https://github.com/2404589803/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L5308
    # and follows their documentation on how their files are serialized
    # https://huggingface.co/docs/safetensors/index#format

    header_len_bytes = file_bytes.read(NUM_BYTES_FOR_HEADER_LEN)
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_json = file_bytes.read(header_len)
    metadata = json.loads(header_json)
    return (metadata, header_len + NUM_BYTES_FOR_HEADER_LEN)


def _get_dtype(dtype_str: str) -> torch.dtype:
    # [FUNCTION DOC] HELPER: Maps a safetensors dtype string to a torch dtype. Unknown strings silently fall back to the current default dtype.
    try:
        dtype = DTYPE_MAP[dtype_str]
    except KeyError:
        dtype = torch.get_default_dtype()

    return dtype


def _get_dcp_custom_metadata(metadata: Any) -> Any | None:
    # [FUNCTION DOC] HELPER: Returns the decoded DCP sharding info found inside the extra-metadata entry of a
    # safetensors header, or None if the entry or the DCP key is missing.
    if DEFAULT_EXTRA_METADATA_KEY in metadata:
        custom_metadata = metadata[DEFAULT_EXTRA_METADATA_KEY]
        if CUSTOM_METADATA_KEY in custom_metadata:
            return json.loads(custom_metadata[CUSTOM_METADATA_KEY])
    return None
