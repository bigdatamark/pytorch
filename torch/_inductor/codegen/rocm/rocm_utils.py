# mypy: allow-untyped-defs
# [MODULE DOC] Defines DTYPE_TO_ROCM_TYPE, a mapping from torch dtypes to C++ type names
# used when generating ROCm kernel source. It starts from the generic C++ dtype mapping and
# overrides half/bfloat16 with a 16-bit unsigned integer and the float8 variants with an
# 8-bit unsigned integer (raw storage types). No import-time side effects beyond building the dict.


import torch

from ..cpp_utils import DTYPE_TO_CPP


DTYPE_TO_ROCM_TYPE = {
    **DTYPE_TO_CPP,
    torch.float16: "uint16_t",
    torch.float8_e4m3fnuz: "uint8_t",
    torch.float8_e5m2fnuz: "uint8_t",
    torch.float8_e4m3fn: "uint8_t",
    torch.float8_e5m2: "uint8_t",
    torch.bfloat16: "uint16_t",
}
