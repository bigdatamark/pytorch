# mypy: allow-untyped-defs
# [MODULE DOC] Reader that exposes an unzipped package directory through the same interface as
# the internal file reader, so PackageImporter can load from a folder. Exports DirectoryReader;
# also defines a small storage-holder helper and the serialization id record name.
import os.path
from glob import glob
from typing import cast

import torch
from torch.types import Storage


__serialization_id_record_name__ = ".data/serialization_id"


# because get_storage_from_record returns a tensor!?
class _HasStorage:
    # [CLASS DOC] HELPER: Tiny wrapper mimicking the object returned by the file reader's
    # storage-record call; holds one storage and exposes it through a method.
    def __init__(self, storage):
        # [FUNCTION DOC] HELPER: Stores the given storage object.
        self._storage = storage

    def storage(self):
        # [FUNCTION DOC] HELPER: Returns the held storage.
        return self._storage


class DirectoryReader:
    # [CLASS DOC] Read-only view of a directory laid out like an unzipped package archive.
    # State: the root directory path. Methods mirror the file reader: read a record's bytes,
    # map a record to a storage, test for a record, list all records, get the serialization id.
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, directory):
        # [FUNCTION DOC] Remembers the root directory path; performs no I/O.
        self.directory = directory

    def get_record(self, name):
        # [FUNCTION DOC] Reads and returns the raw bytes of the file at the given relative
        # name under the root. Raises the usual file errors if it is missing.
        filename = f"{self.directory}/{name}"
        with open(filename, "rb") as f:
            return f.read()

    def get_storage_from_record(self, name, numel, dtype):
        # [FUNCTION DOC] Maps the record file into an untyped storage of numel elements of the
        # given dtype (byte size from the element size) and returns it wrapped in a holder
        # whose storage method yields it. The file is accessed from disk by the storage.
        filename = f"{self.directory}/{name}"
        nbytes = torch._utils._element_size(dtype) * numel
        storage = cast(Storage, torch.UntypedStorage)
        return _HasStorage(storage.from_file(filename=filename, nbytes=nbytes))

    def has_record(self, path):
        # [FUNCTION DOC] Returns whether a regular file exists at the path under the root.
        full_path = os.path.join(self.directory, path)
        return os.path.isfile(full_path)

    def get_all_records(
        self,
    ):
        # [FUNCTION DOC] Recursively lists every non-directory file under the root and returns
        # their paths relative to the root (root prefix and separator stripped).
        files = [
            filename[len(self.directory) + 1 :]
            for filename in glob(f"{self.directory}/**", recursive=True)
            if not os.path.isdir(filename)
        ]
        return files

    def serialization_id(
        self,
    ):
        # [FUNCTION DOC] Returns the bytes of the serialization id record if present, else an
        # empty string (note the differing types).
        if self.has_record(__serialization_id_record_name__):
            return self.get_record(__serialization_id_record_name__)
        else:
            return ""
