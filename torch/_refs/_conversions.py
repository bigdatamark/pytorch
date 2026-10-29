# mypy: allow-untyped-defs
# [MODULE DOC]
# Reference implementations of tensor data-type conversion methods (bfloat16,
# bool, byte, cdouble, cfloat, chalf, char, double, float, half, int, long,
# short) plus the complex and polar constructors. Deliberately kept out of
# _refs/__init__ to avoid clashing with Python builtin type names.
# Import-time side effects: builds the conversion functions and registers
# complex and polar as decompositions of the matching aten operators.
import torch
import torch._prims_common as utils

# Utilities should come BEFORE this import
from torch._decomp import register_decomposition
from torch._prims_common import TensorLikeType
from torch._prims_common.wrappers import out_wrapper
from torch._refs import _broadcast_shapes


# Data conversion references.
#
# Note: this module breaks the usual _refs to torch naming scheme where
# _refs.foo.bar is a ref for torch.foo.bar.  The following definitions are not
# part of _refs/__init__.py to avoid name clashes with Python builtin types
# (like int).

__all__ = [
    # dtypes
    "bfloat16",
    "bool",
    "byte",
    "cdouble",
    "cfloat",
    "chalf",
    "char",
    "double",
    "float",
    "half",
    "int",
    "long",
    "short",
    # misc
    "complex",
    "polar",
]


def _make_conversion_method(name: str, dtype: torch.dtype):
    # [FUNCTION DOC] HELPER: Factory returning a function named name that converts
    # a tensor to the fixed dtype via the tensor to method, taking an optional
    # memory_format (default preserve_format) and returning the converted tensor.
    def fn(
        self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
    ) -> TensorLikeType:
        # [FUNCTION DOC] HELPER: Converts self to the captured dtype with the given
        # memory format; returns self unchanged if already that dtype.
        return self.to(dtype, memory_format=memory_format)  # type: ignore[call-overload]

    fn.__name__ = name
    return fn


bfloat16 = _make_conversion_method("bfloat16", torch.bfloat16)

bool = _make_conversion_method("bool", torch.bool)

byte = _make_conversion_method("byte", torch.uint8)

cdouble = _make_conversion_method("cdouble", torch.cdouble)

cfloat = _make_conversion_method("cfloat", torch.cfloat)

chalf = _make_conversion_method("chalf", torch.complex32)

char = _make_conversion_method("char", torch.int8)

double = _make_conversion_method("double", torch.double)

float = _make_conversion_method("float", torch.float)

half = _make_conversion_method("half", torch.half)

int = _make_conversion_method("int", torch.int)

long = _make_conversion_method("long", torch.long)

short = _make_conversion_method("short", torch.short)


@register_decomposition(torch._ops.ops.aten.complex)
# Note: complex has type promotion tests disabled due to different semantics.
# exact_dtype is for compat with complex_check_dtype from core.
@out_wrapper(exact_dtype=True)
def complex(real: TensorLikeType, imag: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Builds a complex tensor from real and imaginary parts,
    # broadcasting their shapes. Both inputs must be half, float or double and
    # share the same dtype, otherwise an error is raised. The result has the
    # corresponding complex dtype, and the real and imag parts are assigned in place
    # into a freshly allocated tensor.
    allowed_dtypes = (torch.float32, torch.float64, torch.float16)
    torch._check(
        real.dtype in allowed_dtypes and imag.dtype in allowed_dtypes,
        lambda: (
            f"Expected both inputs to be Half, Float or Double tensors but got "
            f"{real.dtype} and {imag.dtype}"
        ),
    )
    torch._check(
        real.dtype == imag.dtype,
        lambda: (
            f"Expected object of scalar type {real.dtype} but got "
            f"scalar type {imag.dtype} for second argument"
        ),
    )
    result_dtype = utils.corresponding_complex_dtype(real.dtype)  # type: ignore[arg-type]
    common_shape = _broadcast_shapes(real.shape, imag.shape)
    result = real.new_empty(
        common_shape,
        dtype=result_dtype,
        layout=real.layout,
        device=real.device,
        # pin_memory=real.is_pinned(),  # NYI
    )
    result.real = real
    result.imag = imag
    return result


@register_decomposition(torch._ops.ops.aten.polar)
# Note: polar has type promotion tests disabled due to different semantics.
# exact_dtype is for compat with complex_check_dtype from core.
@out_wrapper(exact_dtype=True)
def polar(abs: TensorLikeType, angle: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Builds a complex tensor from magnitude abs and phase angle:
    # real part is abs times cosine of angle, imaginary part is abs times sine.
    # Creates the result via the complex op, then overwrites its parts in place.
    result = torch.complex(abs, angle)
    result.real = abs * torch.cos(angle)
    result.imag = abs * torch.sin(angle)
    return result
