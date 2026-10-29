# [MODULE DOC] Streaming save/load of arbitrary objects (tensors included) to non-seekable file-like objects such as sockets, by faking the zip
# writer/reader interface that torch.save and torch.load use internally. Format: a pickled list of _Entry records followed by the raw bytes of each record;
# no compatibility guarantees across PyTorch versions. Import-time side effect: registers _Entry as a safe global for the weights-only unpickler.
# Exports nothing publicly (empty __all__).
import pickle
from dataclasses import dataclass
from io import BufferedIOBase
from typing import Any

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch.serialization import _load, _save, DEFAULT_PROTOCOL, MAP_LOCATION


__all__: list[str] = []


@dataclass
class _Entry:
    # [CLASS DOC] HELPER: Dataclass header describing one record in the stream: key name, whether it is an untyped storage, and its byte length.
    key: str
    is_storage: bool
    length: int


_weights_only_unpickler._add_safe_globals([_Entry])


class _PseudoZipFile:
    # [CLASS DOC] HELPER: In-memory stand-in for the zip file object expected by the serialization internals. Attribute records maps a key to a pair of
    # (data, length) where data is bytes, str or an UntypedStorage. Provides write_record, write_to, read_from, has_record, get_record, get_storage_from_record, serialization_id.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initialize an empty records mapping.
        self.records: dict[str, tuple[object, int]] = {}

    def write_record(self, key: str, data: object, length: int) -> None:
        # [FUNCTION DOC] HELPER: Store (data, length) under key, replacing any existing record.
        self.records[key] = (data, length)

    def write_to(self, f: BufferedIOBase) -> None:
        # [FUNCTION DOC] HELPER: Write all records to a binary stream: first a pickled list of _Entry headers (default protocol), then each record's payload in order
        # (bytes as is, str as UTF-8, storages via their raw file writer). Raises TypeError for any other payload type.
        entries = []
        for key, (data, length) in self.records.items():
            entries.append(
                _Entry(
                    key=key,
                    is_storage=isinstance(data, torch.UntypedStorage),
                    length=length,
                )
            )

        pickle.dump(entries, f, protocol=DEFAULT_PROTOCOL)

        for data, _ in self.records.values():
            if isinstance(data, bytes):
                f.write(data)
            elif isinstance(data, str):
                f.write(data.encode("utf-8"))
            elif isinstance(data, torch.UntypedStorage):
                data._write_file(f, False, False, 1)
            else:
                raise TypeError(f"unknown type: {type(data)}")

    def read_from(self, f: BufferedIOBase) -> None:
        # [FUNCTION DOC] HELPER: Read the pickled header list with the weights-only unpickler, then read each record's length in bytes from the stream.
        # Storage entries become UntypedStorage objects (empty one for length zero, otherwise a uint8 buffer view's storage); others are kept as raw bytes. Fills records.
        entries = _weights_only_unpickler.load(f)

        for entry in entries:
            data = f.read(entry.length)
            if entry.is_storage:
                if entry.length == 0:
                    storage = torch.UntypedStorage(0)
                else:
                    storage = torch.frombuffer(
                        data,
                        dtype=torch.uint8,
                    ).untyped_storage()

                self.records[entry.key] = (
                    storage,
                    entry.length,
                )
            else:
                self.records[entry.key] = (data, entry.length)

    def has_record(self, key: str) -> bool:
        # [FUNCTION DOC] HELPER: True if a record with this key exists.
        return key in self.records

    def get_record(self, key: str) -> object:
        # [FUNCTION DOC] HELPER: Return the stored data (without length) for key; raises KeyError if missing.
        return self.records[key][0]

    def get_storage_from_record(
        self, key: str, _length: int, _type: int
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Return the record as a uint8 tensor built from the stored data; the length and type arguments are ignored. Raises KeyError if missing.
        return torch.tensor(self.records[key][0], dtype=torch.uint8)

    def serialization_id(self) -> str:
        # [FUNCTION DOC] HELPER: Return the fixed identifier string used for this pseudo file.
        return "torchft"


def _streaming_save(
    obj: object,
    f: BufferedIOBase,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
) -> None:
    # [FUNCTION DOC] HELPER: Save obj to the binary stream f by running the normal torch save machinery into a _PseudoZipFile and then writing it out in streaming form.
    # Takes the pickle module and protocol like torch.save; the loaded form needs no seeking. Mmap is unsupported.
    """
    Save the object to a file-like object in a streaming fashion compatible with
    network sockets.

    This behaves similarly to :func:`torch.save` with a few notable differences:

    * A non-seekable file like object can be used when loading.
    * No forwards/backwards compatibility is provided for the serialization
      format. This is only intended to be used with a single version of PyTorch
      with transient storage (i.e. sockets or temp files).
    * mmap is not supported

    See :func:`torch.save` for more details on specific arguments.
    """

    zip_file = _PseudoZipFile()
    _save(
        obj,
        zip_file=zip_file,
        pickle_module=pickle_module,
        pickle_protocol=pickle_protocol,
        _disable_byteorder_record=False,
    )
    zip_file.write_to(f)


def _streaming_load(
    f: BufferedIOBase,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = True,
    **pickle_load_args: Any,
) -> object:
    # [FUNCTION DOC] HELPER: Load an object saved by _streaming_save from stream f. With weights_only (default) the weights-only unpickler is used and an explicit
    # pickle_module raises RuntimeError; otherwise pickle (or the given module) is used. Defaults the unpickling encoding to utf-8, reads all records, then calls the normal load internals.
    # Extra keyword arguments are forwarded to the loader. Returns the loaded object.
    """
    Load the object from a file-like object in a streaming fashion compatible with
    network sockets.

    See :func:`_streaming_save` for more details about the streaming behavior.

    See :func:`torch.load` for more details on specific arguments.
    """
    if weights_only:
        if pickle_module is not None:
            raise RuntimeError(
                "Can not safely load weights when explicit pickle_module is specified"
            )
        pickle_module = _weights_only_unpickler
    else:
        if pickle_module is None:
            pickle_module = pickle

    if "encoding" not in pickle_load_args:
        pickle_load_args["encoding"] = "utf-8"

    zip_file = _PseudoZipFile()
    zip_file.read_from(f)
    return _load(
        zip_file=zip_file,
        map_location=map_location,
        pickle_module=pickle_module,
        **pickle_load_args,
    )
