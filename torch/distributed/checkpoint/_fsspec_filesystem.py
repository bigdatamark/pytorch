# Mypy will not try inferring the types of any 3rd party libraries installed.
# mypy: ignore-errors
# [MODULE DOC] fsspec-backed checkpoint storage: an internal FileSystem adapter implementing FileSystemBase
# over an fsspec filesystem, plus the public FsspecWriter and FsspecReader (subclasses of the local
# filesystem writer/reader that use it). Requires the fsspec package at import time.
# Exports: FsspecWriter, FsspecReader.

import io
import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fsspec.core import url_to_fs

from torch.distributed.checkpoint._extension import StreamTransformExtension
from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
    FileSystemWriter,
    SerializationFormat,
)


if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


__all__ = [
    "FsspecWriter",
    "FsspecReader",
]


class FileSystem(FileSystemBase):
    # [CLASS DOC] HELPER: Adapter implementing the FileSystemBase operations (streams, paths, rename, mkdir, exists,
    # remove, list) on an fsspec filesystem. State: fs, an fsspec filesystem set by init_path (None until then).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes with no fsspec filesystem bound.
        self.fs: AbstractFileSystem | None = None

    @contextmanager
    def create_stream(
        self, path: str | os.PathLike, mode: str
    ) -> Generator[io.IOBase, None, None]:
        # [FUNCTION DOC] HELPER: Context manager yielding an open fsspec stream for path in the given mode.
        # If an error occurs while the stream is in use and the mode is writable (write, update or append),
        # the file is removed on a best-effort basis (removal errors are swallowed) and the error re-raised.
        # Raises AssertionError if the filesystem has not been initialized.
        if self.fs is None:
            raise AssertionError("fs should not be None")
        path = os.fspath(path)

        # fsspec does not support concurrent transactions, and not all
        # AbstractFileSystem have working rollback implementations, so
        # just manually delete the file if necessary on errors.
        with self.fs.open(path, mode) as stream:
            try:
                yield stream
            except:
                if any(ch in mode for ch in "w+a"):  # cleanup file if not read-only
                    try:
                        self.rm_file(path)
                    except:  # noqa: E722
                        pass
                raise

    def concat_path(self, path: str | os.PathLike, suffix: str) -> str | os.PathLike:
        # [FUNCTION DOC] HELPER: Joins a path and a suffix using the operating system path join.
        return os.path.join(path, suffix)

    def init_path(self, path: str | os.PathLike, **kwargs) -> str | os.PathLike:
        # [FUNCTION DOC] HELPER: Resolves the fsspec filesystem from the path (and kwargs) and stores it; returns the path unchanged.
        self.fs, _ = url_to_fs(path, **kwargs)
        return path

    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None:
        # [FUNCTION DOC] HELPER: Renames a path on the filesystem.
        self.fs.rename(path, new_path)

    def mkdir(self, path: str | os.PathLike) -> None:
        # [FUNCTION DOC] HELPER: Creates the directory and parents, tolerating existing ones.
        self.fs.makedirs(path, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        # [FUNCTION DOC] HELPER: Returns whether the id can be handled by fsspec: false for pathlib Path objects or
        # when resolving the id raises ValueError, true otherwise.
        if isinstance(checkpoint_id, Path):
            return False

        try:
            url_to_fs(checkpoint_id)
        except ValueError:
            return False

        return True

    def exists(self, path: str | os.PathLike) -> bool:
        # [FUNCTION DOC] HELPER: Returns whether the path exists on the filesystem.
        return self.fs.exists(path)

    def rm_file(self, path: str | os.PathLike) -> None:
        # [FUNCTION DOC] HELPER: Removes the file at path.
        self.fs.rm(path)

    def ls(self, path: str | os.PathLike) -> list[str]:
        # [FUNCTION DOC] HELPER: Lists entries at path as a list of name strings (detail disabled).
        # setting detail to False explicitly to keep the list[str] return type,
        # instead of the list[Dict] return type when detail=True
        return self.fs.ls(path, detail=False)


# TODO: add the dcp.async_save mixin
class FsspecWriter(FileSystemWriter):
    # [CLASS DOC] Checkpoint StorageWriter using fsspec. Assumes the target is an empty or non-existent
    # directory and atomic file creation; writes one file per request (or per rank) plus a metadata file.
    # Replaces the filesystem and path of the base FileSystemWriter with fsspec equivalents.
    """
    Basic implementation of StorageWriter using fsspec.

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
        **kwargs,
    ) -> None:
        # [FUNCTION DOC] Forwards the standard writer options (single file per rank, sync files, thread count,
        # copy-ahead bytes, overwrite, extensions, serialization format) to the base class, then creates the
        # fsspec FileSystem and initializes it with path and the extra kwargs, storing the resulting path.
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
        super().__init__(
            path,
            single_file_per_rank,
            sync_files,
            thread_count,
            per_thread_copy_ahead,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        # [FUNCTION DOC] Class method: returns whether the checkpoint id is usable with fsspec (delegates to the FileSystem adapter).
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class FsspecReader(FileSystemReader):
    # [CLASS DOC] Checkpoint StorageReader using fsspec; same as FileSystemReader but with an fsspec filesystem.
    def __init__(self, path: str | os.PathLike, **kwargs) -> None:
        # [FUNCTION DOC] Calls the base reader initializer, then creates the fsspec FileSystem and initializes it with path and extra kwargs.
        super().__init__(path)
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        # [FUNCTION DOC] Class method: returns whether the checkpoint id is usable with fsspec (delegates to the FileSystem adapter).
        return FileSystem.validate_checkpoint_id(checkpoint_id)
