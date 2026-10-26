# [MODULE DOC] Base heuristics class shared by GEMM-family templates that run only under max-autotune. No import-time side effects.
from __future__ import annotations

from typing import TYPE_CHECKING

from ... import config as inductor_config
from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from ...kernel_inputs import KernelInputs


class GemmMaxAutotuneTemplateConfigHeuristics(TemplateConfigHeuristics):
    # [CLASS DOC] Base for GEMM template heuristics that are active only when max_autotune or max_autotune_gemm is enabled.
    def should_run(self, inputs: KernelInputs) -> bool:
        # [FUNCTION DOC] Return true if inductor config max_autotune or max_autotune_gemm is set; inputs are ignored.
        """
        simple base override for GEMM family templates that run only in max-autotune
        """
        return inductor_config.max_autotune or inductor_config.max_autotune_gemm
