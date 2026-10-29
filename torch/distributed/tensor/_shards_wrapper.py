# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Defines LocalShardsWrapper, a tensor subclass that holds several local shards of
# a DTensor on one rank together with checkpoint (DCP) storage metadata, so that checkpointing
# can treat it as a sharded tensor. Import-time side effect: only defines the aten alias.
from typing import Any

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)


aten = torch.ops.aten


class LocalShardsWrapper(torch.Tensor):
    # [CLASS DOC] Wrapper tensor subclass (no real storage) that owns a list of local shard
    # tensors and a TensorStorageMetadata describing their offsets, sizes and properties.
    # Its reported shape is the concatenation of shards along dim 1 (2D) or dim 0 (1D); empty
    # shard lists give a 0 by 0 tensor. Attributes: _local_shards (list of tensors) and
    # _storage_meta. Only a fixed set of ops is supported via dispatch; others raise
    # NotImplementedError. Provides accessors and hooks used by distributed checkpointing.
    """
    A wrapper class to hold local shards of a DTensor.
    This class is used largely for checkpointing purposes and implicitly subtypes
    the _Checkpointable protocol.
    """

    __slots__ = ["_local_shards", "_storage_meta"]
    _local_shards: list[torch.Tensor]
    _storage_meta: TensorStorageMetadata

    @staticmethod
    def __new__(
        cls, local_shards: list[torch.Tensor], local_offsets: list[tuple[int, ...]]
    ) -> "LocalShardsWrapper":
        # [FUNCTION DOC] Builds the wrapper from a list of shard tensors and matching offsets.
        # Raises AssertionError if shards are on different devices. An empty list produces a
        # 0 by 0 wrapper with a single zero-sized chunk. Otherwise computes the wrapper shape
        # by summing shard sizes along dim 1 for 2D shards or dim 0 for 1D shards, and records
        # a chunk per shard (offset, size) plus properties taken from the first shard.
        # KNOWN ISSUE: shards with other ndim and more than one shard keep the first shard's
        # shape; extra offsets or shards are silently dropped by pairwise zipping.
        if not all(
            tensor.device == local_shards[0].device for tensor in local_shards[1:]
        ):
            raise AssertionError

        # if empty shard, we create an empty tensor
        if len(local_shards) == 0:
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                torch.Size([0, 0]),
            )
            r._local_shards = []
            r._storage_meta = TensorStorageMetadata(
                properties=TensorProperties(),
                size=torch.Size([0, 0]),
                chunks=[
                    ChunkStorageMetadata(
                        offsets=torch.Size([0, 0]), sizes=torch.Size([0, 0])
                    )
                ],
            )
            return r

        # we calculate the total tensor size by "concat" on second tensor dimension
        cat_tensor_shape = list(local_shards[0].size())
        if len(local_shards) > 1 and local_shards[0].ndim == 2:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[1] += shard.size()[1]

        # in cases of sharding optimizer rowwise, we calculate total tensor size by "concat" on first tensor dimension
        if len(local_shards) > 1 and local_shards[0].ndim == 1:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[0] += shard.size()[0]

        wrapper_properties = TensorProperties.create_from_tensor(local_shards[0])
        wrapper_shape = torch.Size(cat_tensor_shape)
        chunks_meta = [
            ChunkStorageMetadata(
                offsets=torch.Size(offset),
                sizes=shard.size(),
            )
            for shard, offset in zip(local_shards, local_offsets)
        ]

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            torch.Size(cat_tensor_shape),
        )
        r._local_shards = local_shards
        r._storage_meta = TensorStorageMetadata(
            properties=wrapper_properties,
            size=wrapper_shape,
            chunks=chunks_meta,
        )

        return r

    # necessary for ops dispatching from this subclass to its local shards
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        # [FUNCTION DOC] Dispatch hook: looks up func in a table of supported ops (all-gather,
        # wait_tensor, _to_copy, view, equal, detach, clone, new_empty) and calls the matching
        # handler with the args and kwargs. Raises NotImplementedError for any other op.
        kwargs = kwargs or {}

        dispatcher = {
            torch.ops._c10d_functional.all_gather_into_tensor.default: cls.handle_all_gather_into_tensor,
            torch.ops._c10d_functional.wait_tensor.default: cls.handle_wait_tensor,
            aten._to_copy.default: cls.handle_to_copy,
            aten.view.default: cls.handle_view,
            aten.equal.default: cls.handle_equal,
            aten.detach.default: cls.handle_detach,
            aten.clone.default: cls.handle_clone,
            aten.new_empty.default: cls.handle_new_empty,
        }

        if func in dispatcher:
            return dispatcher[func](args, kwargs)
        else:
            raise NotImplementedError(
                f"{func} is not supported for LocalShardsWrapper!"
            )

    @staticmethod
    def handle_all_gather_into_tensor(args, kwargs) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Flattens and concatenates all local shards of the wrapper
        # (first arg), reshapes to rows of the first shard's column count, and calls the
        # functional all-gather on that tensor with the remaining args. Returns the gathered
        # tensor (communication side effect).
        # KNOWN ISSUE: assumes 2D shards; the column count comes from the first shard only.
        dim = args[0].local_sizes()[0][1]
        cat_tensor = torch.cat(
            [t.view(-1) for t in args[0].local_shards()], dim=0
        ).view(-1, dim)
        return torch.ops._c10d_functional.all_gather_into_tensor.default(
            cat_tensor, *args[1:], **kwargs
        )

    @staticmethod
    def handle_wait_tensor(args, kwargs) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Forwards the first arg to the functional wait_tensor op and
        # returns its result; kwargs are ignored.
        return torch.ops._c10d_functional.wait_tensor(args[0])

    @staticmethod
    def handle_to_copy(args, kwargs) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies the copy/convert op to each local shard with the
        # remaining args and kwargs and returns a new wrapper over the results using the
        # original offsets.
        res_shards_list = [
            aten._to_copy.default(shard, *args[1:], **kwargs)
            for shard in args[0].local_shards()
        ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_view(args, kwargs) -> "LocalShardsWrapper":
        # [FUNCTION DOC] HELPER: View on a wrapper. With multiple shards, requires the requested
        # shape to equal the global storage size (AssertionError otherwise; 2D compares both
        # dims, 1D compares dim 0) and then views each shard as its own shape (a no-op); more
        # than 2D raises NotImplementedError. With a single shard (or none), views each shard to
        # the requested shape. Returns a new wrapper with the original offsets.
        view_shape = args[1]
        res_shards_list = []
        if len(args[0].local_shards()) > 1:
            if args[0].local_shards()[0].ndim == 2:
                if not (
                    args[0].storage_metadata().size[0] == view_shape[0]
                    and args[0].storage_metadata().size[1] == view_shape[1]
                ):
                    raise AssertionError
                # This accounts for a DTensor quirk, when multiple shards are present on a rank, DTensor on
                # init calls view_as() on the global tensor shape
                # will fail because the view shape is not applicable to individual shards.
                res_shards_list = [
                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            elif args[0].local_shards()[0].ndim == 1:
                if args[0].storage_metadata().size[0] != view_shape[0]:
                    raise AssertionError
                # This case is for optimizer sharding as regardless of sharding type, optimizer state is row wise sharded
                res_shards_list = [
                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            else:
                raise NotImplementedError("No support for view on tensors ndim > 2")
        else:
            # view is called per shard
            res_shards_list = [
                aten.view.default(shard, args[1], **kwargs)
                for shard in args[0].local_shards()
            ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_equal(args, kwargs) -> bool:
        # [FUNCTION DOC] HELPER: Compares two wrappers; true only if they have the same number of
        # shards, every shard pair is equal in order, and storage metadata is equal.
        """
        LocalShardsWrapper equal impl also checks for equality of storage metadata
        and the order of shards
        """
        a, b = args[0], args[1]
        if len(a.local_shards()) != len(b.local_shards()):
            return False
        if not all(
            aten.equal.default(x, y) for x, y in zip(a.local_shards(), b.local_shards())
        ):
            return False
        if a.storage_metadata() != b.storage_metadata():
            return False
        return True

    @staticmethod
    def handle_detach(args, kwargs) -> "LocalShardsWrapper":
        # [FUNCTION DOC] HELPER: Detaches every shard and returns the same wrapper object.
        # KNOWN ISSUE: mutates the input wrapper in place (replaces its shards and clears the
        # requires_grad flag in its shared metadata) rather than returning a new wrapper.
        self_ls = args[0]
        deatched_local_shards = [
            aten.detach.default(shard) for shard in self_ls.local_shards()
        ]
        self_ls._local_shards = deatched_local_shards
        self_ls._storage_meta.properties.requires_grad = False
        return self_ls

    @staticmethod
    def handle_clone(args, kwargs) -> "LocalShardsWrapper":
        # [FUNCTION DOC] HELPER: Clones each shard and returns a new wrapper with the same
        # offsets. Raises NotImplementedError if a memory format other than preserve is given.
        self_ls = args[0]
        desired_memory_format = kwargs.get("memory_format", None)
        if desired_memory_format and desired_memory_format != torch.preserve_format:
            raise NotImplementedError(
                f"{desired_memory_format} is not supported for LocalShardsWrapper!"
            )
        cloned_local_shards = [
            shard.clone(memory_format=desired_memory_format)
            for shard in self_ls._local_shards
        ]
        return LocalShardsWrapper(cloned_local_shards, self_ls.local_offsets())

    @staticmethod
    def handle_new_empty(args, kwargs) -> "LocalShardsWrapper":
        # [FUNCTION DOC] HELPER: Returns a new wrapper whose shards are uninitialized tensors
        # shaped like the originals, with the same offsets.
        # KNOWN ISSUE: ignores the requested size and kwargs of the new_empty call.
        self_ls = args[0]
        return LocalShardsWrapper(
            [torch.empty_like(shard) for shard in self_ls._local_shards],
            self_ls.local_offsets(),
        )

    @property
    def device(self) -> torch._C.device:  # type: ignore[override]
        # [FUNCTION DOC] Property: device of the first local shard, or the meta device if there
        # are no shards.
        return (
            self._local_shards[0].device if self._local_shards else torch.device("meta")
        )

    @property
    def is_meta(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC] Property: whether the first shard is a meta tensor; true if no shards.
        return self._local_shards[0].is_meta if self._local_shards else True

    def is_pinned(self) -> bool:  # type: ignore[override]
        # [FUNCTION DOC] Returns the pin_memory flag recorded in the storage metadata properties
        # (not the actual pinned state of the shards).
        return self._storage_meta.properties.pin_memory

    def requires_grad_(self, requires_grad: bool = True) -> "LocalShardsWrapper":
        # [FUNCTION DOC] Sets the requires_grad flag in the metadata and on every local shard in
        # place (mutation). Returns the wrapper itself.
        self._storage_meta.properties.requires_grad = requires_grad
        [shard.requires_grad_(requires_grad) for shard in self._local_shards]
        return self

    def local_shards(self) -> list[torch.Tensor]:
        # [FUNCTION DOC] Returns the internal list of local shard tensors (not a copy).
        """
        Returns a list of :class:`torch.Tensor' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    def local_sizes(self) -> list[torch.Size]:
        # [FUNCTION DOC] Returns the per-chunk sizes from the storage metadata, in shard order.
        # KNOWN ISSUE: for an empty wrapper this returns one zero-sized entry, not an empty list
        # as the docstring says.
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local sizes for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.sizes for chunk in self._storage_meta.chunks]

    def local_offsets(self) -> list[torch.Size]:
        # [FUNCTION DOC] Returns the per-chunk offsets from the storage metadata, in shard order.
        # KNOWN ISSUE: for an empty wrapper this returns one zero entry, not an empty list as
        # the docstring says.
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local offsets for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.offsets for chunk in self._storage_meta.chunks]

    @property
    def local_chunks(self) -> list[ChunkStorageMetadata]:
        # [FUNCTION DOC] Property: the list of chunk metadata (offsets and sizes) per shard.
        """
        Returns a :class:`list[ChunkStorageMetadata]` object corresponding to the
        metadata for each tensor shard
        """
        return self._storage_meta.chunks

    def storage_metadata(self) -> TensorStorageMetadata:
        # [FUNCTION DOC] Returns the stored TensorStorageMetadata object (not a copy).
        """
        Returns a :class:`TensorStorageMetadata` object corresponding to the
        metadata for the local tensor on current rank
        """
        return self._storage_meta

    def is_empty_shard(self) -> bool:
        # [FUNCTION DOC] True if the metadata size is zero in both of its first two dimensions.
        # Indexes dim 1, so a 1D wrapper would raise an index error.
        """
        Returns a :class:`bool` object indicating if the local tensor on current rank
        is an empty tensor
        """
        return self._storage_meta.size[0] == 0 and self._storage_meta.size[1] == 0

    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
        # [FUNCTION DOC] Checkpoint hook: for each shard and chunk pair, creates a SHARD-type
        # WriteItem keyed by the fully qualified name and chunk offsets, carrying the chunk
        # metadata, the wrapper's tensor properties and the size of the given object (the
        # global tensor). Returns the list of write items.
        """
        For compatibility with DCP, we support creation of WriteItems
        such that they can be saved properly.
        """
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunks.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=chunks.offsets,
                        sizes=chunks.sizes,
                    ),
                    properties=self._storage_meta.properties,
                    size=object.size(),
                ),
            )
            for tensor, chunks in zip(self.local_shards(), self.local_chunks)
        ]

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        # [FUNCTION DOC] Checkpoint hook: returns the chunk metadata list of this wrapper.
        """
        For compatibility with DCP, we support creation of chunk lists
        such that they can be saved properly.
        """
        return self._storage_meta.chunks

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        # [FUNCTION DOC] Checkpoint hook: finds the shard for a MetadataIndex. First tries the
        # positional index if its chunk offsets match, then searches chunks by offset, then
        # returns an empty tensor for the empty-shard case. Raises ValueError if none match.
        """
        For compatibility with DCP, we support finding shard based on index
        Return a 'torch.Tensor' shard based on 'MetadataIndex'.
        """
        # Fast lookup path
        if index.index is not None:
            if (
                len(self._local_shards) > index.index
                and self._storage_meta.chunks[index.index].offsets == index.offset
            ):
                return self._local_shards[index.index]

        if index.offset is not None:
            for shard, chunk in zip(self._local_shards, self._storage_meta.chunks):
                if chunk.offsets == index.offset:
                    return shard

        # Empty shard case
        if len(self._local_shards) == 0 and self._storage_meta.chunks[
            0
        ].sizes == torch.Size([0, 0]):
            return torch.empty(0)

        raise ValueError(
            f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'"
        )

    def _get_tensor_size_bytes(self) -> int:
        # [FUNCTION DOC] HELPER: Total bytes of all local shards (element count times element size).
        object_size = 0
        for shard in self.local_shards():
            object_size += shard.nelement() * shard.element_size()
        return object_size

    def __hash__(self) -> int:
        # [FUNCTION DOC] Identity-based hash (object id).
        return id(self)

    def __repr__(self) -> str:  # type: ignore[override]
        # [FUNCTION DOC] String form listing the local shards and the storage metadata.
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

    def __str__(self) -> str:
        # [FUNCTION DOC] Same output as the repr form.
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"
