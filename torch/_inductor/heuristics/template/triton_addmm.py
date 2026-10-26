# [MODULE DOC] Template-config heuristics mixin for addmm-like ops (addmm, baddbmm).
# Provides AddMMConfigMixin, which adds the epilogue function (alpha/beta scaling) to template kwargs.
# No import-time side effects beyond imports.
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...kernel.mm_common import addmm_epilogue
from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from ...kernel_inputs import KernelInputs


class AddMMConfigMixin(TemplateConfigHeuristics):
    # [CLASS DOC] Mixin on top of TemplateConfigHeuristics that supplies extra template kwargs for addmm and baddbmm.
    # Stateless; relies on the base class get_extra_kwargs.
    """
    Simple mixin to handle scalars for addmm like operators (addmm, baddbmm)
    """

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        # [FUNCTION DOC] Return the base extra kwargs plus an addmm epilogue function, its hash string, and prefix_args set to 1.
        # Reads the alpha and beta scalars and the output dtype from kernel_inputs.
        # Raises AssertionError if op_name is not addmm or baddbmm.
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        if op_name not in ["addmm", "baddbmm"]:
            raise AssertionError(f"op_name={op_name} invalid for AddMMConfigMixin")
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        return {
            **kwargs,
            "epilogue_fn": addmm_epilogue(kernel_inputs.out_dtype(), alpha, beta),
            "epilogue_fn_hash": str(
                ["addmm_epilogue", kernel_inputs.out_dtype(), alpha, beta]
            ),
            "prefix_args": 1,
        }
