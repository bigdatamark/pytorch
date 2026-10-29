# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC] Defines the _Checkpointable runtime-checkable Protocol that arbitrary objects
# or tensor subclasses can implement to plug into distributed checkpointing (DCP).
from typing_extensions import Protocol, runtime_checkable

import torch


@runtime_checkable
class _Checkpointable(Protocol):  # noqa: PYI046
    # [CLASS DOC] Structural interface (no inheritance needed) with three hooks:
    # create write items for an object, list chunk storage metadata, and fetch a tensor
    # shard by metadata index. Default bodies raise NotImplementedError.
    """
    Interface for checkpointable objects.
    Implemented as a protocol, implicit subtyping is supported so subclasses do not need to inherit this explicitly.
    This is to allow arbitrary objects/tensor subclasses to hook into DCP seamlessly through implementing the interface.
    """

    def __create_write_items__(self, fqn: str, object: object) -> list[object]:
        # [FUNCTION DOC] Should return the list of write items describing object under the
        # fully qualified name fqn. Default implementation raises NotImplementedError.
        """
        Return a list of WriteItems based on object's contents.
        """
        raise NotImplementedError(
            "_Checkpointable._create_write_items is not implemented"
        )

    def __create_chunk_list__(self) -> list[object]:
        # [FUNCTION DOC] Should return the list of chunk storage metadata describing this
        # object's local contents. Default raises NotImplementedError.
        """
        Return a list of `ChunkStorageMetadata` based on object's contents.
        """
        raise NotImplementedError(
            "_Checkpointable._create_chunk_list is not implemented"
        )

    def __get_tensor_shard__(self, index: int) -> torch.Tensor:
        # [FUNCTION DOC] Should return the tensor shard identified by the given metadata
        # index. Default raises NotImplementedError.
        # KNOWN ISSUE: the annotation says int although the docstring says MetadataIndex.
        """
        Return a 'torch.Tensor' shard based on 'MetadataIndex'.
        """
        raise NotImplementedError(
            "_Checkpointable._get_tensor_shard is not implemented"
        )
