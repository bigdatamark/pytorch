# [MODULE DOC] Heuristics for the decompose-K subgraph template of mm, which splits the reduction dimension K into chunks.
# Import-time side effect: registers a no-op heuristic for all devices and the real one for cuda (including ROCm) and xpu.
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

from torch._inductor.heuristics.registry import register_template_heuristic

from ...ir import get_free_symbols
from ...kernel.mm import decompose_k_subgraph_template
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...utils import get_k_splits
from ...virtualized import V
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(decompose_k_subgraph_template.uid, None, op_name="mm")
class EmptyDecomposeKConfigHeuristics(TemplateConfigHeuristics):
    # [CLASS DOC] Empty heuristic (no configs) registered for all devices to disable decompose-K where the real one is not registered.
    """empty heuristics to skip decompose k on anything not cuda"""


@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "xpu",
    op_name="mm",
)
# Register on CUDA (both NVIDIA and ROCm/HIP)
# Runtime enablement is controlled by config.triton.num_decompose_k_splits (0 disables)
@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "cuda",
    op_name="mm",
)
# TODO(coconutruben): enable decompose k on other devices (xpu, cpu, mps, mtia)
# by either adding specific register_template_heuristic tags, or setting the
# device to None (enabled on all devices)
class DecomposeKConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    # [CLASS DOC] Max-autotune gated heuristic yielding one config per candidate K split that divides K.
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        # [FUNCTION DOC] Yield a config with key k_split for each candidate from get_k_splits(m, n, k) whose value statically divides K.
        # Yields nothing if any shape/stride has unbacked symbols. Raises AssertionError if kernel_inputs is not MMKernelInputs.
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

        m, n, k = kernel_inputs.mnk_symbolic()
        k_splits = get_k_splits(m, n, k)
        for k_split in k_splits:
            if not V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            ):
                continue
            yield {"k_split": k_split}
