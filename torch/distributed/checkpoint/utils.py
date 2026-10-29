# mypy: allow-untyped-defs
# [MODULE DOC] Utilities for distributed checkpointing: cross-rank key gathering and consistency checks,
# the _DistWrapper class (exception-robust object collectives that degrade to no-ops without
# distributed), lookup of tensor shards / state dict objects from a MetadataIndex, a bounded
# file-view stream, device-string normalization, an optional cProfile context manager and a
# decorator for backward compatibility of argument order. Public names: find_tensor_shard,
# find_state_dict_object. Module-level flag ENABLE_PROFILE defaults to off. No import-time side effects.
import cProfile
import inspect
import io
import itertools
import os
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import wraps
from pstats import Stats
from typing import Any, cast, TypeVar

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard

from .api import (
    _is_wrapped_exception,
    _wrap_exception,
    CheckpointException,
    WRAPPED_EXCEPTION,
)
from .metadata import MetadataIndex, STATE_DICT_TYPE
from .protocol import _get_checkpointable_tensor_shard, _is_checkpointable_tensor


__all__ = ["find_tensor_shard", "find_state_dict_object"]

T = TypeVar("T")
R = TypeVar("R")


def _get_failure_dict(
    results: list[T | WRAPPED_EXCEPTION],
) -> dict[int, WRAPPED_EXCEPTION]:
    # [FUNCTION DOC] HELPER: from a list of per-rank results, build a mapping from list position (rank) to
    # the entries that are wrapped exceptions; empty when no rank failed.
    return cast(
        dict[int, WRAPPED_EXCEPTION],
        {i: err for i, err in enumerate(results) if _is_wrapped_exception(err)},
    )


def _all_gather_keys(
    local_dict: dict[str, Any], group: dist.ProcessGroup | None = None
) -> set[str]:
    # [FUNCTION DOC] HELPER: collective. Gather the key names of the local dict from every rank in the group
    # via object all-gather and return the union as a set.
    # KNOWN ISSUE: docstring says the keys are returned sorted, but the result is an unordered set.
    """Gathers all keys, and returns them sorted."""
    keys = list(local_dict.keys())
    gathered_keys: list[list[str]] = [None] * dist.get_world_size(group)  # type: ignore[list-item]

    dist.all_gather_object(gathered_keys, keys, group=group)
    return set(itertools.chain.from_iterable(gathered_keys))


def _assert_same_keys(
    state_dict: dict[str, Any], process_group: dist.ProcessGroup | None = None
) -> None:
    # [FUNCTION DOC] HELPER: collective check that this rank's keys cover all keys seen on any rank. Returns
    # immediately for world size one; otherwise gathers all keys and raises AssertionError listing keys
    # present on other ranks but missing locally. Blocks on cross-rank communication.
    # KNOWN ISSUE: only detects keys missing locally, not extra local keys, despite docstring claiming same keys.
    """
    Asserts that all ranks have the same keys in their state dict.
    This is a collective call which requires all ranks in ``process_group`` to
    join. It will also induce cross-rank communication and block CPU.
    """

    if dist.get_world_size(process_group) == 1:
        return

    all_keys = _all_gather_keys(state_dict, process_group)
    my_keys = set(state_dict.keys())
    diff = all_keys - my_keys
    if len(diff) > 0:
        raise AssertionError(
            f"Key(s) present in other ranks but not this one, difference: {diff}"
        )


class _DistWrapper:
    # [CLASS DOC] HELPER: wrapper over a process group providing object collectives (broadcast, gather, all-gather,
    # scatter, reduce-scatter, all-reduce, all-gather with map function, barrier). Works without
    # distributed initialized, where collectives become local no-ops. Function-taking variants are
    # exception robust: failures on any rank are collected and re-raised on all ranks as a
    # CheckpointException. State: group, use_dist flag, coordinator_rank, global_coordinator_rank, rank, is_coordinator.
    """
    This is a wrapper around PG that provides a series of features around object collectives.

    It works without distributed initialized, where most collectives turn into nops.

    All variants that take functions are exception robust, meaning that if one or more
    ranks raise errors, all ranks will observe those.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None,
        use_dist: bool,
        coordinator_rank: int,
    ):
        # [FUNCTION DOC] Store the group, distributed flag and coordinator rank. With distributed, compute the
        # coordinator's global rank (translated from group rank when a group is given), this rank and whether
        # it is the coordinator. Without distributed, rank and coordinator rank are zero and this rank is coordinator.
        self.group = group
        self.use_dist = use_dist
        self.coordinator_rank = coordinator_rank
        if self.use_dist:
            self.global_coordinator_rank = (
                dist.get_global_rank(group, coordinator_rank)
                if group is not None
                else coordinator_rank
            )
            self.rank = dist.get_rank(group)
            self.is_coordinator = self.rank == coordinator_rank
        else:
            self.global_coordinator_rank = 0
            self.rank = 0
            self.is_coordinator = True

    def get_rank(self) -> int:
        # [FUNCTION DOC] Return this rank in the wrapper.
        return self.rank

    def get_world_size(self) -> int:
        # [FUNCTION DOC] Return the group world size, or one when distributed is not used.
        if self.use_dist:
            return dist.get_world_size(self.group)
        return 1

    def broadcast_object(self, object: T | None) -> T:
        # [FUNCTION DOC] Broadcast a single object from the coordinator to all ranks and return it. Without
        # distributed, returns the input unchanged.
        """Implement functionality similar to c10d::broadcast_object_list but without distributed enabled."""
        object_list = [object]
        if self.use_dist:
            dist.broadcast_object_list(
                object_list=object_list,
                group=self.group,
                src=self.global_coordinator_rank,
            )
        return cast(T, object_list[0])

    def gather_object(self, object: T) -> list[T] | None:
        # [FUNCTION DOC] Gather one object per rank onto the coordinator. Returns the list of objects on the
        # coordinator and None on other ranks; without distributed, a list holding just the input.
        """Implement functionality similar to c10d::gather_object but without distributed enabled."""
        if self.use_dist:
            gather_objs = (
                cast(list[T], [None] * dist.get_world_size(self.group))
                if self.is_coordinator
                else None
            )

            dist.gather_object(
                obj=object,
                object_gather_list=gather_objs if self.is_coordinator else None,
                dst=self.global_coordinator_rank,
                group=self.group,
            )
            result = gather_objs
        else:
            result = [object]
        return result

    def all_gather_object(self, object: T) -> list[T]:
        # [FUNCTION DOC] All-gather one object per rank and return the list on every rank; without distributed,
        # a list holding just the input.
        """Implement functionality similar to c10d::all_gather_object but without distributed enabled."""
        if self.use_dist:
            gather_objs = cast(list[T], [None] * dist.get_world_size(self.group))

            dist.all_gather_object(
                object_list=gather_objs, obj=object, group=self.group
            )
        else:
            gather_objs = [object]
        return gather_objs

    def scatter_object(self, object_list: list[T] | None) -> T:
        # [FUNCTION DOC] Scatter one object per rank from the coordinator's list and return this rank's item.
        # Without distributed, returns the first item and raises AssertionError if the list is None.
        """Implement functionality similar to c10d::scatter_object but without distributed enabled."""
        if self.use_dist:
            gather_result = cast(list[T], [None])
            dist.scatter_object_list(
                scatter_object_output_list=gather_result,
                scatter_object_input_list=object_list if self.is_coordinator else None,
                src=self.global_coordinator_rank,
                group=self.group,
            )

            local_reply = gather_result[0]
        else:
            if object_list is None:
                raise AssertionError("object_list is None")
            local_reply = object_list[0]
        return local_reply

    def reduce_scatter(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[list[T]], list[R]],
    ) -> R:
        # [FUNCTION DOC] Run the map function on every rank (capturing exceptions), gather results on the
        # coordinator, run the reduce function there to get one result per rank, and scatter them back.
        # If any rank (or the reduce) fails, every rank receives and raises a CheckpointException for the
        # given step naming the failing ranks. Returns this rank's reduced item. Collective.
        # KNOWN ISSUE: docstring says results are gathered on rank 0, but the coordinator rank is used.
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a scatter.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Scatter to each rank part of the result.
        """
        local_data: WRAPPED_EXCEPTION | T
        try:
            local_data = map_fun()
        except BaseException as e:
            local_data = _wrap_exception(e)

        all_data = self.gather_object(local_data)
        all_results: list[R | CheckpointException] | None = None
        if self.is_coordinator:
            if all_data is None:
                raise AssertionError("all_data is None")
            node_failures = _get_failure_dict(all_data)

            if len(node_failures) == 0:
                try:
                    # N.B. why can't mypy cast List[R] to List[Union[R, WRAPPED_EXCEPTION]]?
                    all_results = cast(
                        list[R | CheckpointException],
                        reduce_fun(cast(list[T], all_data)),
                    )
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)

            if len(node_failures) > 0:
                all_results = [
                    CheckpointException(step, node_failures)
                ] * self.get_world_size()

        result = self.scatter_object(all_results)
        if isinstance(result, CheckpointException):
            raise result
        return result

    def all_reduce(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[list[T]], R],
    ) -> R:
        # [FUNCTION DOC] Run the map function on every rank (capturing exceptions), gather results on the
        # coordinator, run the reduce function there, and broadcast the single reduced value to all ranks.
        # Any failure on any rank or in the reduce makes every rank raise a CheckpointException for the step.
        # Collective.
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a broadcast.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Broadcast the reduced value to all ranks.
        """
        local_data: T | WRAPPED_EXCEPTION
        try:
            local_data = map_fun()
        except BaseException as e:
            local_data = _wrap_exception(e)

        all_data = self.gather_object(local_data)
        result: R | CheckpointException | None = None
        if self.is_coordinator:
            if all_data is None:
                raise AssertionError("all_data is None")
            node_failures = _get_failure_dict(all_data)
            if len(node_failures) == 0:
                try:
                    result = reduce_fun(cast(list[T], all_data))
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)

            if len(node_failures) > 0:
                result = CheckpointException(step, node_failures)

        # pyrefly: ignore [bad-argument-type]
        final_result = self.broadcast_object(result)
        if isinstance(final_result, CheckpointException):
            raise final_result
        # pyrefly: ignore [redundant-cast]
        return cast(R, final_result)

    def all_gather(
        self,
        step: str,
        map_fun: Callable[[], T],
    ) -> list[T]:
        # [FUNCTION DOC] Run the map function on every rank (capturing exceptions), all-gather the results and
        # return the list of per-rank values. If any rank failed, all ranks raise a CheckpointException
        # for the step. Collective.
        # KNOWN ISSUE: docstring refers to a map_cp function; the actual parameter is map_fun.
        """
        Compute a value on each rank, then all_gather them.

        This method operates in the following way:
            Run ``map_cp`` on all ranks
            all_gather the values to all ranks
        """
        result: T | WRAPPED_EXCEPTION
        try:
            result = map_fun()
        except BaseException as e:
            result = _wrap_exception(e)

        all_results = self.all_gather_object(result)

        node_failures = _get_failure_dict(all_results)
        if len(node_failures) > 0:
            raise CheckpointException(step, node_failures)
        return cast(list[T], all_results)

    def broadcast(
        self,
        step: str,
        map_fun: Callable[[], T],
    ) -> T:
        # [FUNCTION DOC] Run the map function only on the coordinator (exceptions captured) and broadcast the
        # result to all ranks. If it failed, all ranks raise a CheckpointException for the step.
        # Returns the value. Collective.
        # KNOWN ISSUE: docstring says rank 0 and map_cp; code uses the coordinator rank and map_fun.
        """
        Compute a value on rank 0 and broadcast it.

        This method operates in the following way:
            Run ``map_cp`` on rank 0
            broadcast the value
        """
        result: T | CheckpointException | None = None
        if self.is_coordinator:
            try:
                result = map_fun()
            except BaseException as e:
                result = CheckpointException(step, {self.rank: _wrap_exception(e)})
        # pyrefly: ignore [bad-argument-type]
        final_result = self.broadcast_object(result)
        if isinstance(final_result, CheckpointException):
            raise final_result
        # pyrefly: ignore [redundant-cast]
        return cast(T, final_result)

    def barrier(self) -> None:
        # [FUNCTION DOC] Barrier over the wrapper's group when distributed is used; otherwise a no-op.
        """
        Add a synchronization point across all processes when using distributed.
        If torch.distributed is initialized, this function will invoke a barrier across the global process group.
        If torch.distributed is not initialized, this function is a no-op.
        """
        if not self.use_dist:
            return
        dist.barrier(group=self.group)


def _find_shard(tensor: ShardedTensor, index: MetadataIndex) -> Shard:
    # [FUNCTION DOC] HELPER: find the local shard of a ShardedTensor matching the offset in the index.
    # Raises ValueError if no offset is given or no shard has that offset. Tries the positional index
    # hint first as a fast path, then scans all local shards.
    if index.offset is None:
        raise ValueError(
            f"Cannot lookup {index.fqn} since its a ShardedTensor and no offset was provided"
        )

    shards = tensor.local_shards()
    # index fast path
    if index.index is not None:
        if (
            len(shards) > index.index
            and torch.Size(shards[index.index].metadata.shard_offsets) == index.offset
        ):
            return shards[index.index]

    for shard in shards:
        if torch.Size(shard.metadata.shard_offsets) == index.offset:
            return shard
    raise ValueError(f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'")


def find_tensor_shard(tensor: torch.Tensor, index: MetadataIndex) -> torch.Tensor:
    # [FUNCTION DOC] Return the tensor (shard) that the MetadataIndex refers to. Order of handling: objects
    # with a tensor-shard hook (such as DTensor), checkpointable tensors, ShardedTensor (via local
    # shard lookup), a plain tensor with an offset (only the all-zero origin is accepted, returns the
    # whole tensor), otherwise the tensor itself. Raises ValueError for a non-zero offset on a plain tensor.
    if hasattr(tensor, "__get_tensor_shard__"):
        # DTensor implements _Checkpointable
        return tensor.__get_tensor_shard__(index)  # type: ignore[attr-defined]
    if _is_checkpointable_tensor(tensor):
        return _get_checkpointable_tensor_shard(tensor, index)
    if isinstance(tensor, ShardedTensor):
        return _find_shard(tensor, index).tensor
    if index.offset is not None:
        # special case looking up a tensor by origin
        if index.offset == torch.Size([0] * len(tensor.size())):
            return tensor
        raise ValueError(
            f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'"
        )
    return tensor


def find_state_dict_object(state_dict: STATE_DICT_TYPE, index: MetadataIndex) -> Any:
    # [FUNCTION DOC] Look up the object in a state dict by the index's fully qualified name. Tensors are
    # resolved to the shard via find_tensor_shard; other objects are returned as is. Raises ValueError
    # if the name is missing or an offset is given for a non-tensor.
    if index.fqn not in state_dict:
        raise ValueError(f"Could not find FQN: '{index.fqn}'")
    obj = state_dict[index.fqn]

    if isinstance(obj, torch.Tensor):
        return find_tensor_shard(obj, index)
    elif index.offset is not None:
        raise ValueError(
            f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'"
        )
    return obj


def _element_wise_add(a: Sequence[int], b: Sequence[int]) -> list[int]:
    # [FUNCTION DOC] HELPER: return a list with the element-wise sum of two integer sequences (truncated to the shorter).
    return [i_a + i_b for i_a, i_b in zip(a, b)]


def _element_wise_sub(a: Sequence[int], b: Sequence[int]) -> list[int]:
    # [FUNCTION DOC] HELPER: return a list with the element-wise difference a minus b of two integer sequences (truncated to the shorter).
    return [i_a - i_b for i_a, i_b in zip(a, b)]


class _ReaderView(io.IOBase):
    # [CLASS DOC] HELPER: read-only stream exposing a window (offset, length) of an underlying stream, with
    # positions reported relative to the window start. State: window offset, window length and base stream.
    # Meant for reading a slice of a file as if it were a standalone file.
    def __init__(self, base_stream: io.IOBase, offset: int, len: int):
        # [FUNCTION DOC] HELPER: store window offset, length and base stream, then seek to the window start.
        super().__init__()
        self.offset = offset
        self.len = len
        self.base_stream = base_stream
        self.seek(0)

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int:
        # [FUNCTION DOC] HELPER: seek within the window. For absolute seeks, shift by the window offset; for
        # end-relative seeks, convert to an absolute position from the window end. Other modes pass through.
        # Returns the new absolute position of the base stream.
        # KNOWN ISSUE: end-relative seek subtracts the given offset from the window end, which reverses the usual sign convention; the returned position is absolute rather than window-relative.
        if whence == os.SEEK_SET:
            offset = self.offset + offset
        elif whence == os.SEEK_END:
            whence = os.SEEK_SET
            offset = (self.offset + self.len) - offset
        return self.base_stream.seek(offset, whence)

    def tell(self) -> int:
        # [FUNCTION DOC] HELPER: return the current position relative to the window start.
        return self.base_stream.tell() - self.offset

    def readable(self) -> bool:
        # [FUNCTION DOC] HELPER: delegate readability to the base stream.
        return self.base_stream.readable()

    def seekable(self) -> bool:
        # [FUNCTION DOC] HELPER: delegate seekability to the base stream.
        return self.base_stream.seekable()

    def readinto(self, b):
        # [FUNCTION DOC] HELPER: read into the given buffer from the base stream, limited to the bytes left in the
        # window (returns 0 at window end). Returns the number of bytes read.
        max_size = self.len - self.tell()
        if max_size == 0:
            return 0
        if len(b) > max_size:
            b = memoryview(b)[:max_size]
        return self.base_stream.readinto(b)  # type: ignore[attr-defined]

    def read(self, size=-1):
        # [FUNCTION DOC] HELPER: read up to size bytes (all remaining when size is -1), clamped to the bytes
        # left in the window, from the base stream and return them.
        max_size = self.len - self.tell()
        if size == -1 or size > max_size:
            size = max_size
        return self.base_stream.read(size)


def _create_file_view(file: io.IOBase, offset: int, length: int) -> io.IOBase:
    # [FUNCTION DOC] HELPER: return a _ReaderView over the file for the given offset and length (not wrapped
    # in a buffered reader because torch.load fails with that).
    # FIXME (kumpera) torch.load fails if we wrap with io.BufferedReader
    return _ReaderView(file, offset, length)


def _normalize_device_info(device_type: str, device_id: int) -> str:
    # [FUNCTION DOC] HELPER: return the string cpu for the cpu device type, otherwise the device type and id
    # joined by a colon.
    """Device info normalization."""
    if device_type == "cpu":
        return "cpu"
    return f"{device_type}:{device_id}"


# TODO: integrate with distributed logging flag
ENABLE_PROFILE = False


@contextmanager
def _profile():
    # [FUNCTION DOC] HELPER: context manager that, only when ENABLE_PROFILE is set and this is rank 0 (or
    # distributed is unavailable), profiles the body with cProfile and prints the top ten entries sorted
    # by time to stdout on exit. Otherwise simply runs the body.
    # Only log the profiling when it is enable and is on rank0  or dist is not
    # available.
    if ENABLE_PROFILE and (not dist.is_available() or dist.get_rank() == 0):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            stats = Stats(profiler)
            stats.sort_stats("time").print_stats(10)
    else:
        yield


def _api_bc_check(func):
    # [FUNCTION DOC] HELPER: decorator for backward compatibility of save/load entry points. Returns a wrapper
    # that, when called with exactly two positional arguments, warns that the argument order changed
    # and moves the second positional argument into the storage_writer or storage_reader keyword
    # (whichever the function declares as keyword-only), then calls the function. Raises AssertionError if
    # that keyword is also given and RuntimeError if neither is declared. Other calls pass through.
    @wraps(func)
    def inner_func(*args, **kwargs) -> Any:
        # [FUNCTION DOC] HELPER: the wrapper described by _api_bc_check; see its documentation for behavior.
        if len(args) == 2:
            warnings.warn(
                f"The argument order of {func.__name__} has been changed. "
                "Please check the document to avoid future breakages.",
                stacklevel=2,
            )
            sig = inspect.signature(func)
            kwonlyargs = [
                p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY
            ]
            if "storage_writer" in kwonlyargs:
                if "storage_writer" in kwargs:
                    raise AssertionError(f"storage_writer in kwargs: {(args, kwargs)}")
                kwargs["storage_writer"] = args[1]
            elif "storage_reader" in kwonlyargs:
                if "storage_reader" in kwargs:
                    raise AssertionError(f"storage_reader in kwargs: {(args, kwargs)}")
                kwargs["storage_reader"] = args[1]
            else:
                raise RuntimeError(f"Unexpected kwonlyargs = {kwonlyargs}")
            return func(args[0], **kwargs)
        else:
            return func(*args, **kwargs)

    return inner_func
