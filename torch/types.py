# [MODULE DOC] Shared type aliases for PyTorch: underscore aliases of builtins and torch
# types, tensor-collection aliases, size/dispatch-key aliases, int/float/bool-like symbolic
# unions, Number, FileLike, Device, and a Storage protocol class used for typing. Public
# exports are Number, Device, FileLike and Storage. Import has no side effects beyond
# creating the aliases and two tuples (symbolic types, number types).
# In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
import os
from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from collections.abc import Sequence
from typing import Any, IO, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import Self

# `as` imports have better static analysis support than assignment `ExposedType: TypeAlias = HiddenType`
from torch import (  # noqa: F401
    device as _device,
    DispatchKey,
    dtype as _dtype,
    layout as _layout,
    qscheme as _qscheme,
    Size,
    SymBool,
    SymFloat,
    SymInt,
    Tensor,
)


if TYPE_CHECKING:
    from torch.autograd.graph import GradientEdge


__all__ = ["Number", "Device", "FileLike", "Storage"]

# Convenience aliases for common composite types that we need
# to talk about in PyTorch
_TensorOrTensors: TypeAlias = Tensor | Sequence[Tensor]  # noqa: PYI047
_TensorOrOptionalTensors: TypeAlias = Tensor | Sequence[Tensor | None]  # noqa: PYI047
_TensorOrTensorsOrGradEdge: TypeAlias = Union[  # noqa: PYI047
    Tensor,
    Sequence[Tensor],
    "GradientEdge",
    Sequence["GradientEdge"],
]

_size: TypeAlias = Size | list[int] | tuple[int, ...]  # noqa: PYI042,PYI047
_symsize: TypeAlias = Size | Sequence[int | SymInt]  # noqa: PYI042,PYI047
_dispatchkey: TypeAlias = str | DispatchKey  # noqa: PYI042,PYI047

# int or SymInt
IntLikeType: TypeAlias = int | SymInt
# float or SymFloat
FloatLikeType: TypeAlias = float | SymFloat
# bool or SymBool
BoolLikeType: TypeAlias = bool | SymBool

py_sym_types = (SymInt, SymFloat, SymBool)  # left un-annotated intentionally
PySymType: TypeAlias = SymInt | SymFloat | SymBool

# Meta-type for "numeric" things; matches our docs
Number: TypeAlias = int | float | bool
# tuple for isinstance(x, Number) checks.
# FIXME: refactor once python 3.9 support is dropped.
_Number = (int, float, bool)

FileLike: TypeAlias = str | os.PathLike[str] | IO[bytes]

# Meta-type for "device-like" things.  Not to be confused with 'device' (a
# literal device object).  This nomenclature is consistent with PythonArgParser.
# None means use the default device (typically CPU)
Device: TypeAlias = _device | str | int | None


# Storage protocol implemented by ${Type}StorageBase classes
class Storage:
    # [CLASS DOC] Typing-only protocol of attributes and methods every concrete storage
    # class provides: raw handle, device, dtype and an uninitialized-load flag. All methods
    # here are stubs that raise NotImplementedError; real behavior lives in the concrete
    # storage classes.
    _cdata: int
    device: _device
    dtype: _dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        # [FUNCTION DOC] Stub for deep copy using a memo mapping; raises NotImplementedError.
        raise NotImplementedError

    def _new_shared(self, size: int) -> Self:
        # [FUNCTION DOC] HELPER: Stub for creating a shared-memory storage of the given
        # size; raises NotImplementedError.
        raise NotImplementedError

    def _write_file(
        self,
        f: Any,
        is_real_file: bool,
        save_size: bool,
        element_size: int,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stub for writing the storage to a file object (real file
        # flag, save-size flag, element size); raises NotImplementedError.
        raise NotImplementedError

    def element_size(self) -> int:
        # [FUNCTION DOC] Stub for the byte size of one element; raises NotImplementedError.
        raise NotImplementedError

    def is_shared(self) -> bool:
        # [FUNCTION DOC] Stub for whether storage is in shared memory; raises NotImplementedError.
        raise NotImplementedError

    def share_memory_(self) -> Self:
        # [FUNCTION DOC] Stub for moving storage to shared memory in place; raises
        # NotImplementedError.
        raise NotImplementedError

    def nbytes(self) -> int:
        # [FUNCTION DOC] Stub for total size in bytes; raises NotImplementedError.
        raise NotImplementedError

    def cpu(self) -> Self:
        # [FUNCTION DOC] Stub for returning a CPU copy of the storage; raises NotImplementedError.
        raise NotImplementedError

    def data_ptr(self) -> int:
        # [FUNCTION DOC] Stub for the address of the first byte; raises NotImplementedError.
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: bool = False,
        nbytes: int = 0,
    ) -> Self:
        # [FUNCTION DOC] Stub for building a storage from a file (name, shared flag, byte
        # count); raises NotImplementedError. Declared as an instance method here.
        raise NotImplementedError

    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> Self:
        # [FUNCTION DOC] HELPER: Stub for building a storage from an open file object and
        # element size; raises NotImplementedError.
        raise NotImplementedError
