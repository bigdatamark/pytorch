# mypy: allow-untyped-defs
# [MODULE DOC] Metadata dataclasses for ShardedTensor: an encoding enum for memory formats (so they can be
# pickled), TensorProperties (dtype, layout, requires_grad, memory format, pin_memory) and
# ShardedTensorMetadata (per-shard metadata list, global size, tensor properties). No import-time side effects.
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch.distributed._shard.metadata import ShardMetadata


class MEM_FORMAT_ENCODING(Enum):
    # [CLASS DOC] Enum of integer codes for torch memory formats (contiguous, channels last, preserve). Used only
    # to make TensorProperties picklable because torch.memory_format objects cannot be pickled.
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


@dataclass
class TensorProperties:
    # [CLASS DOC] Dataclass of tensor creation properties: dtype (defaults to the default dtype at import time),
    # layout, requires_grad, memory_format, pin_memory. Supports pickling via custom state methods that
    # encode the memory format as MEM_FORMAT_ENCODING.
    """Properties used to create :class:`Tensor`"""

    # Regular tensor fields
    dtype: torch.dtype = field(default=torch.get_default_dtype())
    layout: torch.layout = field(default=torch.strided)
    requires_grad: bool = False
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    pin_memory: bool = False

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support: returns a tuple of (dtype, layout, requires_grad, encoded memory format,
        # pin_memory). Raises RuntimeError if the memory format is not contiguous, channels_last or preserve.
        # Since torch.memory_format cannot be pickled!
        memory_format = self.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f"Invalid torch.memory_format: {memory_format}")

        return (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        # [FUNCTION DOC] Pickle support: restores fields from the state tuple produced by the state getter and decodes
        # the memory format encoding back to a torch memory format. Raises RuntimeError for an unknown encoding.
        (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        ) = state

        if mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format
        else:
            raise RuntimeError(
                f"Invalid torch.memory_format encoding: {mem_format_encoding}"
            )

        self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> "TensorProperties":
        # [FUNCTION DOC] Static factory: builds TensorProperties from a tensor, copying dtype, layout, requires_grad and
        # pinned status; memory format is always set to contiguous regardless of the tensor's actual strides.
        return TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )


@dataclass
class ShardedTensorMetadata:
    """
    # [CLASS DOC] Dataclass holding global metadata of a ShardedTensor: shards_metadata (list of ShardMetadata, one
    # per shard across all ranks), size (overall torch.Size, default empty) and tensor_properties.
    # KNOWN ISSUE: the size default is a single shared torch.Size instance (immutable, so harmless).
    Represents metadata for :class:`ShardedTensor`
    """

    # Metadata about each shard of the Tensor
    shards_metadata: list[ShardMetadata] = field(default_factory=list)

    # Size of each dim of the overall Tensor.
    size: torch.Size = field(default=torch.Size([]))

    tensor_properties: TensorProperties = field(default_factory=TensorProperties)
