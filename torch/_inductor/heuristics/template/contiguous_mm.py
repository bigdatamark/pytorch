# [MODULE DOC] Heuristics for the contiguous-mm subgraph templates (mm and addmm) that make the second matrix contiguous before the matmul.
# Import-time side effect: registers a no-op heuristic for all devices and the real one for cuda only on ROCm builds.
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import register_template_heuristic

from ...ir import get_free_symbols
from ...kernel.mm import (
    addmm_contiguous_subgraph_template,
    mm_contiguous_subgraph_template,
)
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...utils import use_contiguous
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(mm_contiguous_subgraph_template.uid, None, op_name="mm")
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid, None, op_name="addmm"
)
class EmptyContiguousMMConfigHeuristics(TemplateConfigHeuristics):
    # [CLASS DOC] Empty heuristic (yields no configs) registered for all devices, which disables the contiguous mm subgraph where the real one is not registered.
    """empty heuristics to skip contiguous mm on not cuda"""


@register_template_heuristic(
    mm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="mm",
)
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="addmm",
)
class ContiguousMMHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    # [CLASS DOC] Max-autotune gated heuristic that yields one empty config when the contiguous decomposition is useful. Registered for cuda only when torch.version.hip is set.
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        # [FUNCTION DOC] Yield one empty config if mat2 is not already contiguous, there are no unbacked symbols in shapes/strides, and use_contiguous accepts the symbolic m, n, k; otherwise yield nothing.
        # Raises AssertionError if kernel_inputs is not MMKernelInputs.
        # KNOWN ISSUE: the docstring talks about valid k_splits, which is copied from decompose_k; this code never produces k_splits.
        """
        Get all the valid k_splits for the given m, n, k.
        """
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        # Check for unbacked symbols - if found, yield nothing
        unbacked_symbols = any(
            len(get_free_symbols(itr, unbacked_only=True)) > 0
            for itr in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        )
        if unbacked_symbols:
            return
        mat2 = kernel_inputs.mat1mat2()[1]
        if mat2.get_layout().is_contiguous():
            # no need for contiguous decomposition
            return
        m, n, k = kernel_inputs.mnk_symbolic()
        if not use_contiguous(m, n, k):
            return
        yield {}
