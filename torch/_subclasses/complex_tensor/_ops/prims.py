# [MODULE DOC] Registers ComplexTensor implementations for a few prims ops (convert_element_type, conj_physical, conj)
# in the shared dispatch table at import time.
import torch

from .._core import ComplexTensor
from .common import (
    complex_to_real_dtype,
    register_complex,
    register_force_test,
    split_complex_tensor,
)


prims = torch.ops.prims
aten = torch.ops.aten


# TODO (hameerabbasi): Not being tested
@register_force_test(prims.convert_element_type)
def convert_element_type_impl(x: ComplexTensor, dtype: torch.dtype) -> ComplexTensor:
    # [FUNCTION DOC] Convert a ComplexTensor to another dtype: map the target complex dtype to its real dtype, convert real and imaginary parts
    # separately, and recombine. Registered as force-tested (note says it is not currently tested).
    dtype = complex_to_real_dtype(dtype)
    u, v = split_complex_tensor(x)
    u_out = prims.convert_element_type(u, dtype)
    v_out = prims.convert_element_type(v, dtype)

    return ComplexTensor(u_out, v_out)


@register_complex(prims.conj_physical)
def conj_physical_impl(self: ComplexTensor) -> ComplexTensor:
    # [FUNCTION DOC] Physical conjugate; delegates to the aten _conj_physical op.
    return aten._conj_physical(self)


@register_complex(prims.conj)
def conj_impl(self: ComplexTensor) -> ComplexTensor:
    # [FUNCTION DOC] Lazy conjugate; delegates to the aten _conj op.
    return aten._conj(self)
