# [MODULE DOC] Configuration namespaces for torch.load and torch.save behavior.
# Defines the option holder classes load and save, then registers this module as a config
# module via install_config_module (import-time side effect: the module is replaced by a
# config object supporting patching and serialization of its settings).
import sys
from typing import Optional as _Optional, TYPE_CHECKING as _TYPE_CHECKING


if _TYPE_CHECKING:
    from torch.serialization import LoadEndianness as _LoadEndianess

from torch.utils._config_module import install_config_module as _install_config_module


class load:
    # [CLASS DOC] Option namespace for torch.load; holds only class attributes (no instances).
    # mmap: memory-map the file; endianness: optional byte-order override for loading;
    # mmap_flags: mmap flags (private mapping, None on Windows);
    # calculate_storage_offsets: compute storage offsets while loading.
    mmap: bool = False
    endianness: _Optional["_LoadEndianess"] = None
    # MAP_PRIVATE = 2
    mmap_flags: int | None = None if sys.platform == "win32" else 2
    calculate_storage_offsets: bool = False


class save:
    # [CLASS DOC] Option namespace for torch.save; holds only class attributes (no instances).
    # compute_crc32: write checksums for records; use_pinned_memory_for_d2h: use pinned host
    # memory when copying device tensors to host; storage_alignment: byte alignment of storages.
    compute_crc32: bool = True
    use_pinned_memory_for_d2h: bool = False
    storage_alignment: int = 64


_install_config_module(sys.modules[__name__])
