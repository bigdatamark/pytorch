# [MODULE DOC] Base template class for CK-Tile (ck_tile) code generation in the ROCm backend.
# Defines CKTileTemplate with shared header/global emission and dtype mapping. No import-time side effects.
import torch
from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate
from torch._inductor.ir import IRNode
from torch._inductor.utils import IndentedBuffer


class CKTileTemplate(ROCmTemplate):
    # [CLASS DOC] ROCmTemplate subclass with common non-gemm-specific CK-Tile logic: a torch dtype to CK
    # alias-name table, a size table for FP16/BF16, and C++ header/globals emission.
    """
    Base class for generating CK templates, has common, i.e. non-gemm-specific, code generation logic
    """

    _TORCH_DTYPE_TO_CK = {
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int32: "I32",
        torch.int8: "I8",
        torch.float8_e4m3fnuz: "F8",  # gfx94
        torch.float8_e4m3fn: "F8",  # gfx95
        torch.float8_e5m2fnuz: "BF8",  # gfx94
        torch.float8_e5m2: "BF8",  # gfx95
    }

    ck_dtype_to_size = {
        "FP16": 2,
        "BF16": 2,
    }

    def header(self) -> IndentedBuffer:
        # [FUNCTION DOC] Extends the base ROCm header with the ck_tile core include. Returns the IndentedBuffer.
        res = super().header()
        res.splice(
            """
                // CK headers
                #include "ck_tile/core.hpp"

            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        # [FUNCTION DOC] Extends the base globals with C++ aliases for ck_tile fp8, bf8, half, float and bfloat16
        # types. Returns the IndentedBuffer.
        res = super().globals()
        res.splice(
            """
                using F8  = ck_tile::fp8_t;
                using BF8 = ck_tile::bf8_t;
                using F16 = ck_tile::half_t;
                using F32 = float;
                using BF16 = ck_tile::bfloat16_t;
            """
        )
        return res

    def torch_type_to_ck(self, node: IRNode, ptr: str) -> str:
        # [FUNCTION DOC] Returns a C++ cast expression casting pointer text ptr to the CK type pointer matching
        # the dtype of node; returns ptr unchanged when node is None.
        # KNOWN ISSUE: unmapped dtypes produce the text None in the cast; class docstring wrongly says it
        # is the generic CK base (duplicate of CKTemplate).
        if node is None:
            return ptr
        else:
            return f"({self._TORCH_DTYPE_TO_CK.get(node.get_dtype())}*)({ptr})"
