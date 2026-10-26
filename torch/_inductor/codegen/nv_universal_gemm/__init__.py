# [PACKAGE DOC] Package for NVIDIA universal GEMM autotuning choices in Inductor. Re-exports GemmVariant,
# NVUniversalGemmCaller and the add_nv_universal_{gemm,addmm,grouped_gemm,scaled_gemm}_choices helpers from the
# nv_universal_gemm submodule (listed in __all__).
# mypy: allow-untyped-defs
from .nv_universal_gemm import (
    add_nv_universal_addmm_choices,
    add_nv_universal_gemm_choices,
    add_nv_universal_grouped_gemm_choices,
    add_nv_universal_scaled_gemm_choices,
    GemmVariant,
    NVUniversalGemmCaller,
)


__all__ = [
    "GemmVariant",
    "NVUniversalGemmCaller",
    "add_nv_universal_addmm_choices",
    "add_nv_universal_gemm_choices",
    "add_nv_universal_grouped_gemm_choices",
    "add_nv_universal_scaled_gemm_choices",
]
