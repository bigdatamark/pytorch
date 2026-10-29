# [MODULE DOC] HELPER: Storage selection helper for DCP: _storage_setup picks or resets a StorageReader/StorageWriter. No import-time side effects.
import os

from .filesystem import FileSystemReader, FileSystemWriter
from .storage import StorageReader, StorageWriter


def _storage_setup(
    storage: StorageReader | StorageWriter | None,
    checkpoint_id: str | os.PathLike | None,
    reader: bool = False,
) -> StorageReader | StorageWriter | None:
    # [FUNCTION DOC] HELPER: Resolves the storage to use. If storage is given, resets it to checkpoint_id when one is
    # provided and returns it. Otherwise requires a checkpoint_id and tries FileSystemReader/Writer, then the fsspec
    # variants if importable, using the first whose validate_checkpoint_id accepts the id; the chosen instance is
    # constructed with the id and reset. Raises RuntimeError if no id is given without storage or no class accepts the id.
    if storage:
        if checkpoint_id is not None:
            storage.reset(checkpoint_id)
        return storage

    if not checkpoint_id:
        raise RuntimeError(
            "`checkpoint_id` must be specified if "
            "storage_reader/storage_writer is None."
        )

    targets: list[type[StorageReader | StorageWriter]] = []
    if reader:
        targets = [
            FileSystemReader,
        ]
    else:
        targets = [
            FileSystemWriter,
        ]
    try:
        from ._fsspec_filesystem import FsspecReader, FsspecWriter

        targets.append(FsspecReader if reader else FsspecWriter)
    except Exception:
        pass

    for target in targets:
        if target.validate_checkpoint_id(checkpoint_id):
            storage = target(checkpoint_id)  # type: ignore[call-arg]
            storage.reset(checkpoint_id)
            return storage

    raise RuntimeError(
        "Cannot detect which StorageReader or StorageWriter to use. "
        "Please specify the storage_reader/storage_writer."
    )
