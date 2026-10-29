# mypy: allow-untyped-defs
# [MODULE DOC] Local file system storage for distributed checkpoint.
# Defines FileSystemWriter / FileSystemReader (the default StorageWriter / StorageReader), an abstract file system
# interface and its local implementation (FileSystemBase, FileSystem), serialization format enum (torch save or
# safetensors), storage bookkeeping records, tensor loaders that copy tensors to CPU (serial or overlapping with a
# device stream), stream-transform wrappers for extensions, and the threaded file-writing worker. A checkpoint is a
# directory with data files (.distcp, one per rank or per item) plus a pickled .metadata file (or per-rank metadata
# files when collectives are off). Import-time side effects: none beyond definitions and constants.
import collections
import dataclasses
import io
import json
import operator
import os
import pickle
import queue
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from io import UnsupportedOperation
from pathlib import Path
from typing import Any, cast, Final, IO

# introduced as collections.abc.Buffer in Python 3.12
from typing_extensions import Buffer

import torch
from torch import Tensor
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._extension import (
    ExtensionRegistry,
    StreamTransformExtension,
)
from torch.distributed.checkpoint._hf_utils import (
    CUSTOM_METADATA_KEY,
    DCP_VERSION_KEY,
    FORMAT_KEY,
    FORMAT_VALUE,
    HF_DCP_VERSION,
)
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, StorageMeta
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.staging import BlockingAsyncStager
from torch.distributed.checkpoint.storage import (
    StorageReader,
    StorageWriter,
    WriteResult,
)
from torch.distributed.checkpoint.utils import _create_file_view
from torch.futures import Future


__all__ = [
    "FileSystemWriter",
    "FileSystemReader",
    "FileSystem",
    "FileSystemBase",
    "SerializationFormat",
]

_metadata_fn: str = ".metadata"

CURRENT_DCP_VERSION: Final[str] = "1.0.0"


@dataclass
class _StorageInfo:
    # [CLASS DOC] HELPER: Per-entry storage record: relative file path, byte offset, byte length and optional transform descriptors.
    # Pickling omits fields that are None, keeping files compatible with versions that lack the descriptors.
    """This is the per entry storage info."""

    relative_path: str
    offset: int
    length: int
    transform_descriptors: Sequence[str] | None = None

    def __getstate__(self):
        # [FUNCTION DOC] HELPER: Pickling hook: returns the instance attributes excluding those that are None.
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class _StoragePrefix:
    # [CLASS DOC] HELPER: Planner storage data holding the file name prefix assigned to a rank's plan.
    prefix: str


class SerializationFormat(Enum):
    # [CLASS DOC] Enum of on-disk tensor formats: TORCH_SAVE (pickle based) and SAFETENSORS.
    TORCH_SAVE = "torch_save"
    SAFETENSORS = "safetensors"


DEFAULT_SUFFIX = ".distcp"


def _generate_uuid() -> str:
    # [FUNCTION DOC] HELPER: Returns a new random UUID as a string (used for save and load ids).
    return str(uuid.uuid4())


class _TensorLoader(ABC):
    # [CLASS DOC] HELPER: Abstract interface for collecting tensors (by size and a user object) and yielding CPU tensors with their objects.
    @abstractmethod
    def add(self, size: int, obj: object) -> None:
        # [FUNCTION DOC] HELPER: Abstract: register an object of the given byte size to be loaded.
        pass

    @abstractmethod
    def start_loading(self) -> None:
        # [FUNCTION DOC] HELPER: Abstract: begin loading the registered objects.
        pass

    @abstractmethod
    def values(self) -> Iterator[tuple[torch.Tensor, object]]:
        # [FUNCTION DOC] HELPER: Abstract: iterate over (CPU tensor, object) pairs.
        pass


class _SerialCpuLoader(_TensorLoader):
    # [CLASS DOC] HELPER: Loader that resolves each item one at a time, in insertion order, copying to CPU on demand.
    def __init__(self, resolve_fun: Callable) -> None:
        # [FUNCTION DOC] HELPER: Stores the resolve function and an empty item list.
        self.resolve_fun = resolve_fun
        self.items: list[tuple[int, object]] = []

    def add(self, size: int, obj: object) -> None:
        # [FUNCTION DOC] HELPER: Appends (size, object) to the pending items.
        self.items.append((size, obj))

    def start_loading(self) -> None:
        # [FUNCTION DOC] HELPER: No-op; loading happens lazily during iteration.
        pass

    def values(self) -> Iterator[tuple[torch.Tensor, object]]:
        # [FUNCTION DOC] HELPER: Yields (tensor, object) for each item: resolves, detaches, moves to CPU, and clones if the tensor is a small view
        # of a larger storage so only the needed bytes are kept.
        for _, obj in self.items:
            tensor = self.resolve_fun(obj).detach()
            tensor = tensor.cpu()
            if tensor.untyped_storage().size() != tensor.nbytes:
                # creates a new tensor with minimal storage while preserving memory format.
                tensor = tensor.clone()
            yield (
                tensor,
                obj,
            )


class _OverlappingCpuLoader(_TensorLoader):
    # [CLASS DOC] HELPER: Loader that overlaps device-to-CPU copies with consumption. Items are sorted by size and copied asynchronously on a
    # stream while the number of in-flight bytes stays below a threshold; completed copies are drained after a stream synchronize.
    # State: resolve function, items, in-flight byte count, queue of copied items, cursor, started flag, device type/module, stream.
    def __init__(
        self,
        resolve_fun: Callable,
        stream: torch.Stream | None = None,
        inflight_threshhold: int = 1_000_000,
    ) -> None:
        # [FUNCTION DOC] HELPER: Picks the device type (from the stream or the available accelerator), the stream (given or current), and makes it wait
        # on the current stream if different. Stores the in-flight threshold in bytes.
        self.resolve_fun = resolve_fun
        self.items: list[tuple[int, object]] = []
        self.inflight_threshhold = inflight_threshhold
        self.in_flight_data = 0
        self.current_items: collections.deque = collections.deque()
        self.idx = 0
        self.started = False
        self.device_type = (
            stream.device_type if stream else _get_available_device_type()
        )
        self.device_module = _get_device_module(self.device_type)
        self.stream = cast(
            torch.cuda.Stream, stream or self.device_module.current_stream()
        )
        if self.stream != self.device_module.current_stream():
            self.stream.wait_stream(self.device_module.current_stream())

    @property
    def _done(self) -> bool:
        # [FUNCTION DOC] HELPER: True when every item has been scheduled for copying.
        return self.idx >= len(self.items)

    def _drain(self) -> list[tuple[torch.Tensor, object]]:
        # [FUNCTION DOC] HELPER: If in-flight bytes reach the threshold, synchronizes the stream and pops copied items until below it. Returns the popped
        # (tensor, object) pairs.
        drained = []
        if self.in_flight_data >= self.inflight_threshhold:
            self.stream.synchronize()
        while self.in_flight_data >= self.inflight_threshhold:
            val = self.current_items.popleft()
            self.in_flight_data -= val[0].numel() * val[0].element_size()
            drained.append(val)
        return drained

    def _refill(self) -> None:
        # [FUNCTION DOC] HELPER: On the stream, resolves and detaches items and copies device tensors to CPU without blocking (CPU tensors are cloned if they
        # are views of larger storage) until the threshold is reached or no items remain; tracks in-flight bytes.
        with self.device_module.stream(self.stream):
            while not self._done and self.in_flight_data < self.inflight_threshhold:
                _, obj = self.items[self.idx]
                self.idx += 1
                tensor = self.resolve_fun(obj).detach()
                if tensor.device.type == self.device_type:
                    tensor = tensor.to(device="cpu", non_blocking=True)
                elif tensor.device == torch.device("cpu"):
                    if (
                        tensor.untyped_storage().size()
                        != tensor.numel() * tensor.itemsize
                    ):
                        # creates a new tensor with minimal storage while preserving memory format.
                        tensor = tensor.clone()

                self.current_items.append(
                    (
                        tensor,
                        obj,
                    )
                )
                self.in_flight_data += tensor.numel() * tensor.element_size()

    def _finish(self) -> Iterable[tuple[torch.Tensor, object]]:
        # [FUNCTION DOC] HELPER: Synchronizes the stream if anything is queued and returns the remaining queued items. Raises AssertionError if items remain.
        if not self._done:
            raise AssertionError("_finish called before all items were processed")
        if len(self.current_items) > 0:
            self.stream.synchronize()
        return self.current_items

    def add(self, size: int, obj: object) -> None:
        # [FUNCTION DOC] HELPER: Registers an item. Raises RuntimeError if loading has already started.
        if self.started:
            raise RuntimeError("cannot add items after loading started")
        self.items.append((size, obj))

    def start_loading(self) -> None:
        # [FUNCTION DOC] HELPER: First call sorts items by size ascending and starts the first copies; later calls do nothing.
        if self.started:
            return
        self.started = True
        self.items.sort(key=operator.itemgetter(0))
        self._refill()

    def values(self) -> Iterator[tuple[torch.Tensor, object]]:
        # [FUNCTION DOC] HELPER: Starts loading and yields all (tensor, object) pairs, interleaving drain and refill, then the remainder.
        self.start_loading()
        while not self._done:
            drained = self._drain()
            self._refill()
            yield from drained

        yield from self._finish()


class _StorageWriterTransforms:
    # [CLASS DOC] HELPER: Wraps the stream transform extensions applied when writing; the default is no extensions. Experimental.
    """
    This is experimental, and will likely move elsewhere in the
    future.  It lives here to minimize changes while we are still
    learning and gathering feedback.
    """

    def __init__(
        self, extensions: Sequence[StreamTransformExtension] | None = None
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the extensions; None means the default (empty tuple).
        """
        If the extensions arg is None, this means the implementation
        should provide whatever defaults it chooses.  An empty
        sequence indicates no extensions should be used.  At this
        time, the default extensions sequence is empty.
        """
        self.extensions = () if extensions is None else extensions

    def transform_save_stream(
        self, write_item: WriteItem, raw_stream: io.IOBase
    ) -> tuple[IO[bytes], list[str]]:
        # [FUNCTION DOC] HELPER: Wraps the raw stream so closing it only flushes (so it can be reused for appended items), then layers each extension's
        # transform on top. Returns the outermost writable stream and the list of extension descriptors in reverse order.
        # In order to avoid leaking fds, transformers' close must
        # cascade to wrapped streams, but since this function can
        # append to the raw stream, we can't close the actual stream.
        # So, we use this to put a wrapper around the raw stream's
        # close() to make it a noop, and it gets closed once all files
        # are appended.

        class NoCloseWriter(io.IOBase):
            # [CLASS DOC] HELPER: Writable wrapper over a raw stream whose close flushes but does not close the underlying stream.
            def __init__(self, raw: io.IOBase):
                # [FUNCTION DOC] HELPER: Stores the wrapped stream.
                self.raw = raw

            def writeable(self) -> bool:
                # [FUNCTION DOC] HELPER: Always returns True.
                return True

            def write(self, b: Buffer) -> int:
                # [FUNCTION DOC] HELPER: Forwards the bytes to the wrapped stream and returns the count written.
                return self.raw.write(b)

            def close(self):
                # [FUNCTION DOC] HELPER: Flushes this wrapper and the underlying stream without closing it.
                self.flush()
                self.raw.flush()
                # but not close.

        transform_to = cast(IO[bytes], NoCloseWriter(raw_stream))

        for ex in self.extensions:
            transform_to = ex.transform_to(transform_to)

        return (transform_to, [ex.get_descriptor() for ex in reversed(self.extensions)])


def _item_size(item: WriteItem) -> int:
    # [FUNCTION DOC] HELPER: Returns the byte size of a tensor write item (element count times element size). Raises AssertionError if no tensor data.
    size = 1
    if item.tensor_data is None:
        raise AssertionError("WriteItem tensor_data must not be None")
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.size:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _split_by_size_and_type(bins: int, items: list[WriteItem]) -> list[list[WriteItem]]:
    # [FUNCTION DOC] HELPER: Splits write items into the given number of buckets: one bucket returns all items; otherwise byte-stream items are
    # spread round-robin and tensor items are assigned largest-first to the currently lightest bucket. Returns the list of buckets.
    if bins == 1:
        return [items]

    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]

    buckets: list[list[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    tensor_w.sort(key=_item_size, reverse=True)

    for i, wi in enumerate(bytes_w):
        buckets[i % bins].append(wi)

    for wi in tensor_w:
        # TODO replace with heapq
        idx = min(enumerate(bucket_sizes), key=operator.itemgetter(1))[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)

    return buckets


def _write_item(
    transforms: _StorageWriterTransforms,
    stream: io.IOBase,
    data: io.BytesIO | torch.Tensor,
    write_item: WriteItem,
    storage_key: str,
    serialization_format: SerializationFormat,
) -> WriteResult:
    # [FUNCTION DOC] HELPER: Writes one item to the stream through the extension transforms. Byte items write their buffer; tensors are written with torch
    # save when that format is selected (for safetensors the data is written later by the caller). Records the starting offset and length
    # (stream growth, or raw tensor size for safetensors tensors). Raises AssertionError on a wrong data type or a non-CPU tensor.
    # Returns a WriteResult carrying a _StorageInfo; transform descriptors are omitted when there are none.
    offset = stream.tell()

    (transform_to, transform_descriptors) = transforms.transform_save_stream(
        write_item, stream
    )

    if write_item.type == WriteItemType.BYTE_IO:
        if not isinstance(data, io.BytesIO):
            raise AssertionError("Data must be io.BytesIO for BYTE_IO write items")
        transform_to.write(data.getbuffer())
    else:
        if not isinstance(data, torch.Tensor):
            raise AssertionError(
                "Data must be torch.Tensor for non-BYTE_IO write items"
            )
        if data.device != torch.device("cpu"):
            raise AssertionError("Tensor must be on CPU device")
        if serialization_format == SerializationFormat.TORCH_SAVE:
            torch.save(data, transform_to)

    transform_to.close()

    if serialization_format == SerializationFormat.TORCH_SAVE or isinstance(
        data, io.BytesIO
    ):
        length = stream.tell() - offset
    else:
        length = data.numel() * data.element_size()

    # For consistency with earlier versions, leave this field out of the
    # metadata if there are no extensions.
    info_transform_descriptors = (
        None if len(transform_descriptors) == 0 else transform_descriptors
    )

    return WriteResult(
        index=write_item.index,
        size_in_bytes=length,
        storage_data=_StorageInfo(
            storage_key,
            offset,
            length,
            transform_descriptors=info_transform_descriptors,
        ),
    )


def _write_files_from_queue(
    create_stream: Callable,
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    transforms: _StorageWriterTransforms,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
    serialization_format: SerializationFormat,
) -> None:
    # [FUNCTION DOC] HELPER: Worker for writing files: repeatedly takes (path, name, items) from file_queue until empty. Chooses an overlapping loader only for a
    # single-thread setup with an available accelerator and a positive threshold, else a serial loader. For each file writes byte items, then tensors
    # (written in loader order), and for safetensors serializes all tensors with DCP metadata in one write. Optionally flushes and fsyncs (warns if
    # unsupported), closes the stream and puts the list of WriteResults on result_queue. Raises AssertionError if a tensor is not on CPU.
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader

            custom_backend_name = torch._C._get_privateuse1_backend_name()
            custom_device_mod = getattr(torch, custom_backend_name, None)

            # TODO: Using the OverlappingCpuLoader with multiple threads creates significant
            # performance degradation, observed as being related to cuda stream syncs. We
            # should try to fix this and use _OverlappingCpuLoader for all threaded cases
            if (
                thread_count == 1
                and (
                    torch.cuda.is_available()
                    or (custom_device_mod and custom_device_mod.is_available())
                )
                and inflight_threshhold > 0
            ):
                loader = _OverlappingCpuLoader(
                    planner.resolve_data,
                    inflight_threshhold=inflight_threshhold,
                )
            else:
                loader = _SerialCpuLoader(
                    planner.resolve_data,
                )

            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []

            with create_stream(file_name, "wb") as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(
                            transforms,
                            stream,
                            data,
                            write_item,
                            storage_key,
                            serialization_format,
                        )
                    )

                tensor_dict = {}
                metadata_dict = {}
                for tensor, write_item in loader.values():
                    if not tensor.is_cpu:
                        raise AssertionError("Tensor must be on CPU")
                    write_results.append(
                        _write_item(
                            transforms,
                            stream,
                            tensor,
                            write_item,  # type: ignore[arg-type]
                            storage_key,
                            serialization_format,
                        )
                    )
                    tensor_dict[write_item.index.fqn] = tensor  # type: ignore[attr-defined]
                    metadata_dict[write_item.index.fqn] = {  # type: ignore[attr-defined]
                        "saved_offsets": write_item.tensor_data.chunk.offsets  # type: ignore[attr-defined]
                    }

                if serialization_format == SerializationFormat.SAFETENSORS:
                    from safetensors.torch import save  # type: ignore[import-not-found]

                    stream.write(
                        save(
                            tensor_dict,
                            metadata={
                                CUSTOM_METADATA_KEY: json.dumps(metadata_dict),
                                DCP_VERSION_KEY: str(HF_DCP_VERSION),
                                FORMAT_KEY: FORMAT_VALUE,
                            },
                        )
                    )

                if use_fsync:
                    # Flush Python-level buffers (OS for local files, network for cloud storage) before fsync.
                    stream.flush()
                    try:
                        os.fsync(stream.fileno())
                    except (AttributeError, UnsupportedOperation) as e:
                        warnings.warn(
                            f"fsync not supported for this stream, relying on flush(): {e}"
                        )
                stream.close()
            result_queue.put(write_results)
    except queue.Empty:
        pass


class FileSystemBase(ABC):
    # [CLASS DOC] Abstract interface for the file operations used by checkpointing: create_stream (context manager), concat_path, rename,
    # init_path, mkdir, validate_checkpoint_id (class method), exists and rm_file. All are abstract one-line stubs.
    @contextmanager
    @abstractmethod
    def create_stream(
        self, path: str | os.PathLike, mode: str
    ) -> Generator[io.IOBase, None, None]: ...

    @abstractmethod
    def concat_path(
        self, path: str | os.PathLike, suffix: str
    ) -> str | os.PathLike: ...

    @abstractmethod
    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None: ...

    @abstractmethod
    def init_path(self, path: str | os.PathLike) -> str | os.PathLike: ...

    @abstractmethod
    def mkdir(self, path: str | os.PathLike) -> None: ...

    @classmethod
    @abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool: ...

    @abstractmethod
    def exists(self, path: str | os.PathLike) -> bool: ...

    @abstractmethod
    def rm_file(self, path: str | os.PathLike) -> None: ...


class FileSystem(FileSystemBase):
    # [CLASS DOC] Local file system implementation of FileSystemBase using pathlib; paths are converted to Path objects as needed.
    @contextmanager
    def create_stream(
        self, path: str | os.PathLike, mode: str
    ) -> Generator[io.IOBase, None, None]:
        # [FUNCTION DOC] Context manager that opens the path in the given mode and yields the stream; closes it on exit.
        if not isinstance(path, Path):
            path = Path(path)
        with path.open(mode) as stream:
            yield cast(io.IOBase, stream)

    def concat_path(self, path: str | os.PathLike, suffix: str) -> str | os.PathLike:
        # [FUNCTION DOC] Returns the path joined with the suffix as a Path.
        if not isinstance(path, Path):
            path = Path(path)
        return path / suffix

    def init_path(self, path: str | os.PathLike) -> str | os.PathLike:
        # [FUNCTION DOC] Returns the path converted to a Path.
        if not isinstance(path, Path):
            path = Path(path)
        return path

    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None:
        # [FUNCTION DOC] Renames the path to new_path.
        if not isinstance(path, Path):
            path = Path(path)

        path.rename(cast(Path, new_path))

    def mkdir(self, path: str | os.PathLike) -> None:
        # [FUNCTION DOC] Creates the directory including parents; no error if it exists.
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        # [FUNCTION DOC] Class method. Returns True for Path objects; False for strings containing a URL scheme separator; otherwise True if any existing
        # ancestor directory is writable, else False.
        if isinstance(checkpoint_id, Path):
            return True

        if "://" in str(checkpoint_id):
            return False

        for p in Path(checkpoint_id).parents:
            if p.exists() and os.access(str(p), os.W_OK):
                return True

        return False

    def exists(self, path: str | os.PathLike) -> bool:
        # [FUNCTION DOC] Returns whether the path exists.
        if not isinstance(path, Path):
            path = Path(path)
        return path.exists()

    def rm_file(self, path: str | os.PathLike) -> None:
        # [FUNCTION DOC] Deletes the file; raises if it does not exist.
        if not isinstance(path, Path):
            path = Path(path)
        path.unlink()

    def ls(self, path: str | os.PathLike) -> list[str]:
        # [FUNCTION DOC] Returns the string paths of the entries of a directory.
        if not isinstance(path, Path):
            path = Path(path)
        return [str(p) for p in path.iterdir()]


class _FileSystemWriter(StorageWriter):
    # [CLASS DOC] HELPER: Base StorageWriter that writes plan items to files in a directory and then a pickled metadata file. Assumes the target is
    # an empty or new directory and atomic file creation. State: fs, path, single_file_per_rank, sync_files, thread_count,
    # per_thread_copy_ahead, save_id (fresh uuid per save), overwrite, transforms, serialization_format, rank and use_collectives
    # (set in set_up_storage_writer).
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: str | os.PathLike,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores options, creates the local file system and path, a fresh save id and the stream transforms. Rank is unset and
        # collectives default to enabled. KNOWN ISSUE: the docstring mentions only some of the arguments (extra positional/keyword args are ignored).
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving them. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files
        self.thread_count = thread_count
        self.per_thread_copy_ahead = per_thread_copy_ahead
        self.save_id = _generate_uuid()
        self.overwrite = overwrite
        self.transforms = _StorageWriterTransforms(_extensions)
        self.serialization_format = serialization_format
        self.rank: int | None = None
        self.use_collectives: bool = True

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        # [FUNCTION DOC] Resets for a new save: optionally changes the path and always generates a new save id.
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.save_id = _generate_uuid()

    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Records rank and use_collectives from keyword arguments (the coordinator flag is unused).
        self.rank = kwargs.get("rank")
        self.use_collectives = kwargs.get("use_collectives", True)

    def _metadata_exists(self) -> bool:
        # [FUNCTION DOC] HELPER: Checks whether metadata already exists: the global metadata file with collectives, else the rank 0 file.
        if self.use_collectives:
            # A global checkpoint metadata file
            metadata_path = self._get_metadata_path(rank=None)
        else:
            # A rank 0 specific metadata file if every rank has written its own metadata
            # Just looking for lowest rank metadata file is sufficient
            metadata_path = self._get_metadata_path(rank=0)

        return self.fs.exists(metadata_path)

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        # [FUNCTION DOC] Creates the directory. If a checkpoint exists: warns when overwrite is true, else raises RuntimeError. Without collectives and with a
        # rank, tags the plan with a rank specific file prefix. Returns the plan. Side effect: mkdir.
        self.fs.mkdir(self.path)
        if self._metadata_exists():
            if self.overwrite:
                warnings.warn(
                    f"Detected an existing checkpoint in {self.path}, overwriting since {self.overwrite=}."
                    " Past version 2.5 of PyTorch, `overwrite` will default to False. Set this variable to True to"
                    " maintain this functionality or False to raise when an existing checkpoint is found.",
                    stacklevel=2,
                )
            else:
                raise RuntimeError(f"Checkpoint already exists and {self.overwrite=}.")

        if self.rank is not None and not self.use_collectives:
            plan = dataclasses.replace(
                plan, storage_data=_StoragePrefix(f"__{self.rank}_")
            )

        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        # [FUNCTION DOC] Assigns a distinct file prefix (by rank position) to every plan that has no storage data. Returns the new plans.
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_"))
            if plan.storage_data is None
            else plan
            for i, plan in enumerate(plans)
        ]
        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        # [FUNCTION DOC] Names files from the plan's prefix and a counter, puts one file per size-balanced bucket (single file per rank) or per item into a queue,
        # and delegates to _write_data. Returns a Future of WriteResults.
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file():
            # [FUNCTION DOC] HELPER: Returns the next file name (prefix, running count, distributed suffix) and increments the count.
            nonlocal file_count
            file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        file_queue: queue.Queue = queue.Queue()
        if self.single_file_per_rank:
            for bucket in _split_by_size_and_type(self.thread_count, plan.items):
                file_name = gen_file()
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, bucket))
        else:
            for item in plan.items:
                file_name = gen_file()
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, [item]))

        return self._write_data(planner, file_queue)

    def _write_data(
        self,
        planner: SavePlanner,
        file_queue: queue.Queue,
    ) -> Future[list[WriteResult]]:
        # [FUNCTION DOC] Starts thread_count minus one worker threads plus runs one worker in the calling thread over the file queue, joins the threads,
        # and collects all results into an already completed Future. Side effect: writes files.
        result_queue: queue.Queue = queue.Queue()

        threads = []
        for _ in range(1, self.thread_count):
            t = threading.Thread(
                target=_write_files_from_queue,
                args=(
                    self.fs.create_stream,
                    file_queue,
                    result_queue,
                    planner,
                    self.transforms,
                    self.per_thread_copy_ahead,
                    self.sync_files,
                    self.thread_count,
                    self.serialization_format,
                ),
            )
            t.start()
            threads.append(t)

        _write_files_from_queue(
            create_stream=self.fs.create_stream,
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
            transforms=self.transforms,
            inflight_threshhold=self.per_thread_copy_ahead,
            use_fsync=self.sync_files,
            thread_count=self.thread_count,
            serialization_format=self.serialization_format,
        )

        for t in threads:
            t.join()

        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            fut: Future[list[WriteResult]] = Future()
            fut.set_result(res)
            return fut

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        # [FUNCTION DOC] Completes the save on the coordinator: sets metadata version, collects storage data from all write results, attaches storage meta, writes
        # the pickled metadata to a temporary file (optionally flushed and fsynced), removes any older metadata file, and renames the temporary file
        # into place. Uses a rank specific name when collectives are off.
        metadata.version = CURRENT_DCP_VERSION

        storage_md = {}
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md

        metadata.storage_meta = self.storage_meta()
        tmp_filename = (
            f"__{self.rank}{_metadata_fn}.tmp"
            if not self.use_collectives and self.rank is not None
            else f"{_metadata_fn}.tmp"
        )
        tmp_path = cast(Path, self.fs.concat_path(self.path, tmp_filename))
        with self.fs.create_stream(tmp_path, "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            if self.sync_files:
                # Flush Python-level buffers (OS for local files, network for cloud storage) before fsync.
                metadata_file.flush()
                try:
                    os.fsync(metadata_file.fileno())
                except (AttributeError, UnsupportedOperation) as e:
                    warnings.warn(
                        f"fsync not supported for this stream, relying on flush(): {e}"
                    )

        # delete in-case other checkpoints were present.
        if not self.use_collectives and self.rank is not None:
            metadata_path = self._get_metadata_path(self.rank)
        else:
            metadata_path = self._get_metadata_path()

        if self.fs.exists(metadata_path):
            self.fs.rm_file(metadata_path)

        self.fs.rename(tmp_path, metadata_path)

    def storage_meta(self) -> StorageMeta | None:
        # [FUNCTION DOC] Returns StorageMeta with this writer's checkpoint id and save id.
        return StorageMeta(checkpoint_id=self.checkpoint_id, save_id=self.save_id)

    def _get_metadata_path(self, rank: int | None = None) -> os.PathLike:
        # [FUNCTION DOC] HELPER: Returns the global metadata path, or the rank specific one when a rank is given.
        filename = f"{_metadata_fn}" if rank is None else f"__{rank}{_metadata_fn}"
        return cast(Path, self.fs.concat_path(self.path, filename))

    @property
    def checkpoint_id(self) -> str | os.PathLike:
        # [FUNCTION DOC] Property: the checkpoint directory path.
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        # [FUNCTION DOC] Class method delegating to the local file system check.
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class _StorageReaderTransforms:
    # [CLASS DOC] HELPER: Wraps the extension registry used to build read-side stream transforms. Experimental.
    """
    This is experimental, and will likely move elsewhere in the
    future.  It lives here to minimize changes while we are still
    learning and gathering feedback.
    """

    def __init__(self, extension_registry: ExtensionRegistry | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the registry or creates an empty one.
        self.extension_registry = (
            ExtensionRegistry() if extension_registry is None else extension_registry
        )

    def transform_load_stream(
        self,
        read_item: ReadItem,
        transform_descriptors: Sequence[str],
        raw_stream: IO[bytes],
    ) -> IO[bytes]:
        # [FUNCTION DOC] HELPER: Looks up extensions from the descriptors and wraps the raw stream with each stream transform extension's read transform.
        # Returns the final stream.
        extensions = self.extension_registry.from_descriptor_list(transform_descriptors)
        transform_from = raw_stream
        for ex in extensions:
            if isinstance(ex, StreamTransformExtension):
                transform_from = ex.transform_from(transform_from)
        return transform_from


class FileSystemReader(StorageReader):
    # [CLASS DOC] Default StorageReader for the local file system. State: fs, path, storage_data (index to storage info, set from metadata), load_id
    # (fresh uuid per load), transforms, rank and use_collectives.
    def __init__(
        self,
        path: str | os.PathLike,
        _extension_registry: ExtensionRegistry | None = None,  # EXPERIMENTAL
    ) -> None:
        # [FUNCTION DOC] Creates the file system and path, empty storage data, a new load id, and read transforms from the optional registry.
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.storage_data: dict[Any, Any] = {}
        self.load_id = _generate_uuid()
        self.transforms = _StorageReaderTransforms(_extension_registry)
        self.rank = None
        self.use_collectives = True

    def _slice_file(self, file, sinfo: _StorageInfo) -> IO[bytes]:
        # [FUNCTION DOC] HELPER: Returns a bounded view of a file limited to the entry's offset and length.
        return cast(IO[bytes], _create_file_view(file, sinfo.offset, sinfo.length))

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        # [FUNCTION DOC] Clears cached storage data, optionally changes the path, and generates a new load id.
        self.storage_data = {}
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.load_id = _generate_uuid()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # [FUNCTION DOC] Groups the plan's read items by file, opens each file once, and for every item reads its slice through the extension transforms.
        # Byte items are handed to the planner as a stream; tensor items are loaded (making the data seekable first) with weights-only torch load
        # on CPU, narrowed to the requested range, checked for size equality (AssertionError otherwise), copied into the planner's target tensor
        # and committed. Returns an already completed Future. Side effect: in-place tensor writes.
        # group requests by file
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md: _StorageInfo = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        # This field wasn't present in older
                        # implementations so provide a fallback.
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        if transform_from.seekable():
                            seekable = transform_from
                        else:
                            # torch.load requires a seekable input, so read the transform
                            # stream now and store the output if needed
                            seekable = io.BytesIO(transform_from.read(-1))
                            seekable.seek(0)

                        tensor = cast(
                            Tensor,
                            torch.load(
                                seekable,
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()

                        if target_tensor.size() != tensor.size():
                            raise AssertionError(
                                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                            )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def _get_metadata_path(self, rank: int | None = None) -> os.PathLike:
        # [FUNCTION DOC] HELPER: Returns the global metadata path, or the rank specific one when a rank is given.
        filename = f"{_metadata_fn}" if rank is None else f"__{rank}{_metadata_fn}"
        return cast(Path, self.fs.concat_path(self.path, filename))

    # Implementing the abstract function in StorageReader
    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        # [FUNCTION DOC] Reads and unpickles the metadata file (rank specific if the rank keyword is given), ensures storage meta exists and records
        # this reader's load id. Returns the Metadata.
        rank = kwargs.get("rank")
        path = self._get_metadata_path(rank)
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = pickle.load(metadata_file)

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        return metadata

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Takes storage data from the metadata and records rank and use_collectives from keyword arguments. Raises AssertionError if
        # the metadata has no storage data.
        self.storage_data = metadata.storage_data
        self.rank = kwargs.get("rank")
        self.use_collectives = kwargs.get("use_collectives", True)
        if self.storage_data is None:
            raise AssertionError("storage_data must not be None in metadata")

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        # [FUNCTION DOC] Returns the plan unchanged.
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        # [FUNCTION DOC] Returns the plans unchanged.
        return plans

    @property
    def checkpoint_id(self) -> str | os.PathLike:
        # [FUNCTION DOC] Property: the checkpoint directory path.
        """
        return the checkpoint_id that will be used to load the checkpoint.
        """
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        # [FUNCTION DOC] Class method delegating to the local file system check.
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class FileSystemWriter(_FileSystemWriter, BlockingAsyncStager):
    # [CLASS DOC] Public storage writer using local files; combines the basic file writer with a blocking async stager. Writes one file per rank
    # (or per item) and a metadata file (global with collectives, rank local otherwise). Adds the cache_staged_state_dict option.
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a global `.metadata` file with the serialized metadata if rank coordination is enabled.
    a rank local `__{rank}.metadata` file with the serialized metadata if rank coordination is NOT enabled.

    """

    def __init__(
        self,
        path: str | os.PathLike,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        cache_staged_state_dict: bool = False,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
    ) -> None:
        # [FUNCTION DOC] Initializes the file writer and the blocking stager with the given options.
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving them. Default 10Mb.
            cache_staged_state_dict: Whether to cache the staged state_dict. This option decreases staging latency
                at the cost of increased memory usage. Additionally, if this parameter is set to True, it's the expectation
                that the stager is maintained and reused for multiple dcp.async_save calls. Default to False.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        _FileSystemWriter.__init__(
            self,
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        BlockingAsyncStager.__init__(
            self,
            cache_staged_state_dict=cache_staged_state_dict,
        )

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        # [FUNCTION DOC] Stages the state dict for async saving. Sets the copy-ahead to zero (the state is already on CPU) and returns the stager result.
        """Override of AsyncStager.stage"""
        # in the async case, the state dict is already on CPU, so maintaining this
        # buffer makes no sense
        self.per_thread_copy_ahead = 0
        return super().stage(state_dict)
