# [PACKAGE DOC] Template heuristics package: compile-time config generators for template kernels (mm, conv, flex attention, etc.).
# Importing it imports every heuristic submodule (aten, base, contiguous_mm, decompose_k, nv_universal_gemm, registry, tlx, triton)
# so their decorators register classes in the shared registry. Exports only get_template_heuristic.
"""
Template heuristics for pre-written template kernels (mm, conv, flex_attention...).

These heuristics are used at compile time to generate autotuning configs
for template-based kernels (Triton, ATen, CuTe DSL, etc.).
"""

from __future__ import annotations

# expose the entry function
from torch._inductor.heuristics.registry import get_template_heuristic

# NOTE: add new template heuristics here, so they get imported and registered
from . import (
    aten,
    base,
    contiguous_mm,
    decompose_k,
    nv_universal_gemm,
    registry,
    tlx,
    triton,
)


__all__ = [
    "get_template_heuristic",
]
