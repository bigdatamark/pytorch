# [MODULE DOC] Heuristics for ATen (extern kernel) choices of mm, bmm, addmm, baddbmm and related ops, so they go through
# the same template-config flow as other templates. Import-time side effect: registers the classes below in the heuristic registry.
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch._inductor import config as inductor_config
from torch._inductor.heuristics.registry import register_template_heuristic

from ...kernel.bmm import aten_baddbmm, aten_bmm, aten_bmm_dtype
from ...kernel.mm import (
    aten__fp8_mm,
    aten__int_mm,
    aten_addmm,
    aten_bias_addmm,
    aten_mm,
    aten_mm_dtype,
)
from ...kernel.mm_plus_mm import aten_mm_plus_mm
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator

    from ...kernel_inputs import KernelInputs


# These are all labeled as device type None to indicate that they
# are valid for all device types
@register_template_heuristic(aten_mm.uid, None)
@register_template_heuristic(aten_mm_dtype.uid, "cuda")
@register_template_heuristic(aten_mm_dtype.uid, "xpu")
@register_template_heuristic(aten__fp8_mm.uid, None)
@register_template_heuristic(aten__int_mm.uid, None)
@register_template_heuristic(aten_bmm.uid, None)
@register_template_heuristic(aten_mm_plus_mm.uid, None)
# bmm dtype is only valid on cuda
@register_template_heuristic(aten_bmm_dtype.uid, "cuda")
@register_template_heuristic(aten_bmm_dtype.uid, "xpu")
class ATenConfigHeuristics(TemplateConfigHeuristics):
    # [CLASS DOC] Pseudo heuristic producing exactly one empty config, so a plain ATen choice with no kwargs is handled like a template. Subclass it for ATen choices needing kwargs.
    """
    Pseudo heuristic to make ATen choices go through the same flow as other templates

    This is a single choice without kwargs

    If you want to use this with an ATen choice that has kwargs, just subclass
    """

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        # [FUNCTION DOC] Yield a single empty config dict (one choice, no template kwargs). Inputs are ignored.
        yield dict()


# None here indicates that this is valid for all device types on that op
# Note (None, op) takes precedence over (device_type, None)
@register_template_heuristic(aten_addmm.uid, None, op_name="addmm")
@register_template_heuristic(aten_baddbmm.uid, None, op_name="baddbmm")
class ATenAddMMConfigHeuristics(ATenConfigHeuristics):
    # [CLASS DOC] ATen heuristic for addmm/baddbmm choices that need alpha and beta passed as extra kwargs.
    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        # [FUNCTION DOC] Return the base extra kwargs plus alpha and beta read as scalars from the kernel inputs.
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        return {
            **kwargs,
            "alpha": alpha,
            "beta": beta,
        }


@register_template_heuristic(aten_bias_addmm.uid, None, op_name="addmm")
class ATenBiasAddMMConfigHeuristics(
    ATenAddMMConfigHeuristics, GemmMaxAutotuneTemplateConfigHeuristics
):
    # [CLASS DOC] ATen bias-addmm (cublasLt) heuristic: only valid for a 2D bias with a zero leading stride when triton autotune_cublasLt is enabled; also gated by max-autotune via the GEMM base class.
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        # [FUNCTION DOC] Check that the first input (bias) is 2D with stride zero in dim 0 and that the autotune_cublasLt config is on,
        # raising AssertionError otherwise; then yield the parent's configs (one empty config).
        nodes = kernel_inputs.nodes()
        # for addmm, bias is the first input
        bias = nodes[0]
        if not (
            len(bias.get_size()) == 2
            and bias.get_stride()[0] == 0
            and inductor_config.triton.autotune_cublasLt
        ):
            raise AssertionError(
                "Expected 2D bias with stride[0]==0 and autotune_cublasLt enabled"
            )
        yield from super()._get_template_configs_impl(kernel_inputs, op_name)
