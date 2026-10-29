# mypy: ignore-errors

# [MODULE DOC] Dtype group helpers for test suites. Defines tuple-like collections of torch dtypes
# that mirror the C++ AT_DISPATCH macro groups (floating, complex, integral, float8, MPS,
# passthru), accessor functions with optional extension by extra dtypes, and list-returning
# get_all_* helpers used to parametrize tests.
# Import-time side effects: none beyond building module-level constant tuples of dtypes.


import torch


# Functions and classes for describing the dtypes a function supports
# NOTE: these helpers should correspond to PyTorch's C++ dispatch macros


# Verifies each given dtype is a torch.dtype
def _validate_dtypes(*dtypes):
    # [FUNCTION DOC] HELPER: Checks that every positional argument is a torch dtype and
    # returns them unchanged as a tuple. Raises an error naming the offending type otherwise.
    # No side effects.
    for dtype in dtypes:
        if not isinstance(dtype, torch.dtype):
            raise AssertionError(f"Expected dtype to be torch.dtype, got {type(dtype)}")
    return dtypes


# class for tuples corresponding to a PyTorch dispatch macro
class _dispatch_dtypes(tuple):
    # [CLASS DOC] HELPER: Tuple subclass marking a group of dtypes that corresponds to a C++
    # dispatch macro. It has no per-instance state (empty slots). Its only override is
    # concatenation, which keeps the result of the same class so groups can be chained.
    __slots__ = ()

    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Concatenates this group with another tuple and returns a new
        # dispatch tuple of the same class. Raises an error if the right operand is not a tuple.
        if not isinstance(other, tuple):
            raise AssertionError(f"Expected other to be a tuple, got {type(other)}")
        return _dispatch_dtypes(tuple.__add__(self, other))


_empty_types = _dispatch_dtypes(())


def empty_types():
    # [FUNCTION DOC] Accessor returning the precomputed empty dtype group (a tuple of torch
    # dtypes). Takes no arguments, no side effects.
    return _empty_types


_floating_types = _dispatch_dtypes((torch.float32, torch.float64))


def floating_types():
    # [FUNCTION DOC] Accessor returning the precomputed float32 and float64 group (a tuple of
    # torch dtypes). Takes no arguments, no side effects.
    return _floating_types


_floating_types_and_half = _floating_types + (torch.half,)


def floating_types_and_half():
    # [FUNCTION DOC] Accessor returning the precomputed float32, float64 and half group (a tuple
    # of torch dtypes). Takes no arguments, no side effects.
    return _floating_types_and_half


def floating_types_and(*dtypes):
    # [FUNCTION DOC] Returns the float32 and float64 tuple extended with the extra dtypes given as
    # positional arguments. Raises an error if any argument is not a torch dtype. Pure; returns a
    # new dispatch tuple.
    return _floating_types + _validate_dtypes(*dtypes)


_floating_and_complex_types = _floating_types + (torch.cfloat, torch.cdouble)


def floating_and_complex_types():
    # [FUNCTION DOC] Accessor returning the precomputed floating plus complex64 and complex128
    # group (a tuple of torch dtypes). Takes no arguments, no side effects.
    return _floating_and_complex_types


def floating_and_complex_types_and(*dtypes):
    # [FUNCTION DOC] Returns the floating plus complex tuple extended with the extra dtypes given
    # as positional arguments. Raises an error if any argument is not a torch dtype. Pure; returns
    # a new dispatch tuple.
    return _floating_and_complex_types + _validate_dtypes(*dtypes)


_double_types = _dispatch_dtypes((torch.float64, torch.complex128))


def double_types():
    # [FUNCTION DOC] Accessor returning the precomputed float64 and complex128 group (a tuple of
    # torch dtypes). Takes no arguments, no side effects.
    return _double_types


# NB: Does not contain uint16/uint32/uint64 for BC reasons
_integral_types = _dispatch_dtypes(
    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
)


def integral_types():
    # [FUNCTION DOC] Accessor returning the precomputed integral group (uint8, int8, int16, int32,
    # int64; no wider unsigned types, for backward compatibility) (a tuple of torch dtypes). Takes
    # no arguments, no side effects.
    return _integral_types


def integral_types_and(*dtypes):
    # [FUNCTION DOC] Returns the integral tuple extended with the extra dtypes given as positional
    # arguments. Raises an error if any argument is not a torch dtype. Pure; returns a new
    # dispatch tuple.
    return _integral_types + _validate_dtypes(*dtypes)


_all_types = _floating_types + _integral_types


def all_types():
    # [FUNCTION DOC] Accessor returning the precomputed floating plus integral group (a tuple of
    # torch dtypes). Takes no arguments, no side effects.
    return _all_types


def all_types_and(*dtypes):
    # [FUNCTION DOC] Returns the floating plus integral tuple extended with the extra dtypes given
    # as positional arguments. Raises an error if any argument is not a torch dtype. Pure; returns
    # a new dispatch tuple.
    return _all_types + _validate_dtypes(*dtypes)


_complex_types = _dispatch_dtypes((torch.cfloat, torch.cdouble))


def complex_types():
    # [FUNCTION DOC] Accessor returning the precomputed complex64 and complex128 group (a tuple of
    # torch dtypes). Takes no arguments, no side effects.
    return _complex_types


def complex_types_and(*dtypes):
    # [FUNCTION DOC] Returns the complex tuple extended with the extra dtypes given as positional
    # arguments. Raises an error if any argument is not a torch dtype. Pure; returns a new
    # dispatch tuple.
    return _complex_types + _validate_dtypes(*dtypes)


_all_types_and_complex = _all_types + _complex_types


def all_types_and_complex():
    # [FUNCTION DOC] Accessor returning the precomputed floating, integral and complex group (a
    # tuple of torch dtypes). Takes no arguments, no side effects.
    return _all_types_and_complex


def all_types_and_complex_and(*dtypes):
    # [FUNCTION DOC] Returns the floating, integral and complex tuple extended with the extra
    # dtypes given as positional arguments. Raises an error if any argument is not a torch dtype.
    # Pure; returns a new dispatch tuple.
    return _all_types_and_complex + _validate_dtypes(*dtypes)


_all_types_and_half = _all_types + (torch.half,)


def all_types_and_half():
    # [FUNCTION DOC] Accessor returning the precomputed floating plus integral plus half group (a
    # tuple of torch dtypes). Takes no arguments, no side effects.
    return _all_types_and_half


_all_mps_types = (
    _dispatch_dtypes({torch.float, torch.half, torch.bfloat16}) + _integral_types
)


def all_mps_types():
    # [FUNCTION DOC] Accessor returning the precomputed dtype group supported on MPS (float32,
    # half, bfloat16 plus integral types, no bool or float64) (a tuple of torch dtypes). Takes no
    # arguments, no side effects.
    # KNOWN ISSUE: the floating part of the group is built from a set, so its element order is not
    # deterministic across runs.
    return _all_mps_types


def all_mps_types_and(*dtypes):
    # [FUNCTION DOC] Returns the MPS tuple extended with the extra dtypes given as positional
    # arguments. Raises an error if any argument is not a torch dtype. Pure; returns a new
    # dispatch tuple.
    return _all_mps_types + _validate_dtypes(*dtypes)


_float8_types = _dispatch_dtypes(
    (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    )
)


def float8_types():
    # [FUNCTION DOC] Accessor returning the precomputed float8 group (e4m3fn, e4m3fnuz, e5m2,
    # e5m2fnuz) (a tuple of torch dtypes). Takes no arguments, no side effects.
    return _float8_types


def float8_types_and(*dtypes):
    # [FUNCTION DOC] Returns the float8 tuple extended with the extra dtypes given as positional
    # arguments. Raises an error if any argument is not a torch dtype. Pure; returns a new
    # dispatch tuple.
    return _float8_types + _validate_dtypes(*dtypes)


def all_types_complex_float8_and(*dtypes):
    # [FUNCTION DOC] Returns floating, integral, complex and float8 dtypes concatenated, plus any
    # extra dtypes given as positional arguments. Raises an error if an extra argument is not a
    # torch dtype. Pure.
    return _all_types + _complex_types + _float8_types + _validate_dtypes(*dtypes)


_barebones_unsigned_types = _dispatch_dtypes((torch.uint16, torch.uint32, torch.uint64))


# Passthru ops (copy, fill, index, gather, scatter, flip, take, put, where, eq, ne)
# move or select data without dtype-specific arithmetic, so they support the
# largest common set of dtypes across CPU and CUDA. Mirrors
# AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kBool, kHalf, kBFloat16,
# AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES) on the C++ side. complex32 (chalf) is
# not included because the CPU dispatch for these ops does not list
# kComplexHalf (scatter/gather additionally can't add it trivially because the
# kernel reinterprets the destination through opmath_t = complex<float>, which
# is twice the storage width). Callers that need chalf coverage on CUDA
# should opt in with all_passthru_types_and(torch.chalf) under @dtypesIfCUDA.
_all_passthru_types = (
    _all_types_and_complex
    + _dispatch_dtypes((torch.bool, torch.half, torch.bfloat16))
    + _barebones_unsigned_types
)


def all_passthru_types():
    # [FUNCTION DOC] Accessor returning the precomputed passthru group: all types and complex,
    # bool, half, bfloat16 and the barebones unsigned types uint16, uint32, uint64 (no complex32)
    # (a tuple of torch dtypes). Takes no arguments, no side effects.
    return _all_passthru_types


def all_passthru_types_and(*dtypes):
    # [FUNCTION DOC] Returns the passthru tuple extended with the extra dtypes given as positional
    # arguments. Raises an error if any argument is not a torch dtype. Pure; returns a new
    # dispatch tuple.
    return _all_passthru_types + _validate_dtypes(*dtypes)


def custom_types(*dtypes):
    # [FUNCTION DOC] Builds a dispatch tuple of arbitrary dtypes from the positional arguments.
    # Raises an error if an argument is not a torch dtype. Pure.
    # KNOWN ISSUE: the docstring says it creates a list, but a dispatch tuple is returned.
    """Create a list of arbitrary dtypes"""
    return _empty_types + _validate_dtypes(*dtypes)


# The functions below are used for convenience in our test suite and thus have no corresponding C++ dispatch macro


# See AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS.
def get_all_dtypes(
    include_half=True,
    include_bfloat16=True,
    include_bool=True,
    include_complex=True,
    include_complex32=False,
    include_qint=False,
    include_bcomplex32=False,
) -> list[torch.dtype]:
    # [FUNCTION DOC] Returns a new list of dtypes for tests: the integral dtypes and float dtypes
    # (float32, float64, optionally half and bfloat16), then bool if requested, complex dtypes
    # (optionally complex32 and bcomplex32) and qint dtypes if requested. Flags: include_half,
    # include_bfloat16, include_bool, include_complex (defaults True), include_complex32,
    # include_qint, include_bcomplex32 (defaults False). Pure.
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(
        include_half=include_half, include_bfloat16=include_bfloat16
    )
    if include_bool:
        dtypes.append(torch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes(
            include_complex32=include_complex32, include_bcomplex32=include_bcomplex32
        )
    if include_qint:
        dtypes += get_all_qint_dtypes()
    return dtypes


def get_all_math_dtypes(device) -> list[torch.dtype]:
    # [FUNCTION DOC] Returns a new list of integral, floating and complex dtypes suitable for math
    # tests on the given device. Half is included only when the device string starts with cuda;
    # bfloat16 is never included. Raises an error if the device is not a string.
    return (
        get_all_int_dtypes()
        + get_all_fp_dtypes(
            include_half=device.startswith("cuda"), include_bfloat16=False
        )
        + get_all_complex_dtypes()
    )


def get_all_complex_dtypes(
    *, include_complex32=False, include_bcomplex32=False
) -> list[torch.dtype]:
    # [FUNCTION DOC] Returns a new list with complex64 and complex128. With include_bcomplex32 the
    # bcomplex32 dtype is inserted first, and with include_complex32 the complex32 dtype is
    # inserted at the very front (keyword-only flags, both default False). Pure.
    dtypes = [torch.complex64, torch.complex128]
    if include_bcomplex32:
        dtypes.insert(0, torch.bcomplex32)
    if include_complex32:
        dtypes.insert(0, torch.complex32)
    return dtypes


def get_all_int_dtypes() -> list[torch.dtype]:
    # [FUNCTION DOC] Returns a new list of the integral dtypes uint8, int8, int16, int32,
    # int64. Pure.
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def get_all_fp_dtypes(include_half=True, include_bfloat16=True) -> list[torch.dtype]:
    # [FUNCTION DOC] Returns a new list with float32 and float64, followed by float16 if
    # include_half and bfloat16 if include_bfloat16 (both default True). Pure.
    dtypes = [torch.float32, torch.float64]
    if include_half:
        dtypes.append(torch.float16)
    if include_bfloat16:
        dtypes.append(torch.bfloat16)
    return dtypes


def get_all_qint_dtypes() -> list[torch.dtype]:
    # [FUNCTION DOC] Returns a new list of the quantized dtypes qint8, quint8, qint32, quint4x2
    # and quint2x4. Pure.
    return [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]


def highest_precision_float(device):
    # [FUNCTION DOC] Returns the widest float dtype usable on the device: float32 for MPS
    # devices (which lack float64), float64 otherwise. The device argument is anything
    # torch.device accepts.
    if torch.device(device).type == "mps":
        return torch.float32
    else:
        return torch.float64


def highest_precision_complex(device):
    # [FUNCTION DOC] Returns the widest complex dtype usable on the device: complex64 for MPS
    # devices, complex128 otherwise. The device argument is anything torch.device accepts.
    if torch.device(device).type == "mps":
        return torch.complex64
    else:
        return torch.complex128


float_to_corresponding_complex_type_map = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.bcomplex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
