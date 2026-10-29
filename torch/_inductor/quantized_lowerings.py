# [MODULE DOC] Inductor lowerings for quantized ops: defines ExternKernelChoice objects for the
# int8/int4 weight-only packed matmuls, plus functions that register fallbacks for quantized ops and
# register weight-only-quantized matmul lowerings (ATen and C++ GEMM template choices autotuned).
# Import-time side effects: creates the ExternKernelChoice instances and torch.ops aliases.
import logging
from typing import Any

import torch
from torch._inductor.kernel.mm_common import mm_args

from . import config
from .codegen.cpp_gemm_template import CppGemmTemplate, CppWoqInt4GemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .mkldnn_ir import WeightInt4PackMatmul
from .select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
)
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template
from .virtualized import V


log = logging.getLogger(__name__)

aten__weight_int8pack_mm = ExternKernelChoice(
    torch._weight_int8pack_mm, "at::_weight_int8pack_mm", has_out_variant=False
)

aten__weight_int4pack_mm_cpu = ExternKernelChoice(
    torch.ops.quantized.int4mm_packed_weight_cpu,
    "at::native::_weight_int4pack_mm_cpu_tensor",
    has_out_variant=False,
    kernel_creator=WeightInt4PackMatmul.create,
)

quantized = torch.ops.quantized
_quantized = torch.ops._quantized
aten = torch.ops.aten


def register_quantized_ops() -> None:
    # [FUNCTION DOC] Marks quantized max_pool2d and the fbgemm fp16 pack/linear ops as needing
    # realized inputs and registers them as fallback (extern kernel) lowerings. Mutates lowering registries.
    from . import lowering

    lowering.add_needs_realized_inputs(
        [
            quantized.max_pool2d,
            _quantized.wrapped_fbgemm_pack_gemm_matrix_fp16,
            _quantized.wrapped_fbgemm_linear_fp16_weight,
        ]
    )
    lowering.make_fallback(quantized.max_pool2d)
    lowering.make_fallback(_quantized.wrapped_fbgemm_pack_gemm_matrix_fp16)
    lowering.make_fallback(_quantized.wrapped_fbgemm_linear_fp16_weight)


def register_woq_mm_ops() -> None:
    # [FUNCTION DOC] Registers lowerings for weight-only-quantized matmuls: the int8 pack mm and the
    # int4 pack mm for CPU (both defined as nested functions below), and fallback lowerings for the
    # dynamic 4-bit quantized matmul and weight packing ops. Mutates the global lowering registry.
    """Register weight-only quantized matmul lowerings."""
    from . import lowering
    from .lowering import expand, register_lowering

    @register_lowering(aten._weight_int8pack_mm, type_promotion_kind=None)  # type: ignore[misc]
    def int8pack_mm(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        layout: Any = None,
    ) -> Any:
        # [FUNCTION DOC] Lowering for int8 weight-only matmul. Computes matmul args, requires
        # mat1 of bfloat16/float16/float and mat2 of int8 (raises AssertionError otherwise). Builds
        # choices: the ATen extern kernel if ATen GEMM is allowed, plus a C++ GEMM template with a
        # per-channel scale multiply epilogue when applicable. Returns the autotuned selected node.
        _, _, _, layout, mat1, mat2 = mm_args(
            input, weight, layout=layout, mat2_transposed=True
        )
        if not (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.int8
        ):
            raise AssertionError(
                f"expected mat1 dtype in [bfloat16, float16, float] and mat2 dtype "
                f"int8, got {mat1.get_dtype()} and {mat2.get_dtype()}"
            )
        aten_layout = layout

        # options to tune from
        choices = (
            [aten__weight_int8pack_mm.bind((mat1, mat2, scale), aten_layout)]
            if use_aten_gemm_kernels()
            else []
        )

        # scale is applied as an epilogue, and the scale tensor is expanded (with a view op)
        # for broadcasting, as it's 1D.
        def _mul_epilogue(buf: torch.Tensor) -> Any:
            # [FUNCTION DOC] HELPER: Epilogue that multiplies the matmul buffer by the scale tensor expanded to the output size.
            return create_epilogue_with_attr(
                buf, "mul", other=realize_inputs(expand(scale, layout.size))
            )

        if use_cpp_gemm_template(aten_layout, mat1, mat2, mat2_transposed=True):
            CppGemmTemplate.add_choices(
                choices,
                aten_layout,
                [mat1, mat2, scale],
                trans_w=True,
                epilogue_creator=_mul_epilogue,  # type: ignore[arg-type]
            )

        node, _ = autotune_select_algorithm(
            "_weight_int8pack_mm", choices, [mat1, mat2, scale], aten_layout
        )
        return node

    @register_lowering(aten._weight_int4pack_mm_for_cpu, type_promotion_kind=None)  # type: ignore[misc]
    def int4pack_mm_cpu(
        input: torch.Tensor,
        weight: torch.Tensor,
        qGroupSize: int,
        qScaleAndZeros: torch.Tensor,
        *,
        layout: Any = None,
    ) -> Any:
        # [FUNCTION DOC] Lowering for int4 packed-weight matmul on CPU. Requires mat1 of
        # bfloat16/float16/float and mat2 of uint8 (raises AssertionError otherwise). Registers the
        # group size as a graph tensor constant, builds choices from the ATen kernel and, under
        # max-autotune with a suitable contiguous weight, the C++ WOQ int4 GEMM template. Autotunes
        # with custom example-input generators for the packed weight and group size; returns the node.
        _, _, _, layout, mat1, mat2 = mm_args(
            input, weight, layout=layout, use_4x2_dim=True, mat2_transposed=True
        )
        if not (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.uint8
        ):
            raise AssertionError(
                f"expected mat1 dtype in [bfloat16, float16, float] and mat2 dtype "
                f"uint8, got {mat1.get_dtype()} and {mat2.get_dtype()}"
            )
        group_size = V.graph.add_tensor_constant(
            torch.tensor(qGroupSize, dtype=torch.int64), name=None
        )
        aten_layout = layout

        # options to tune from
        choices = (
            [
                aten__weight_int4pack_mm_cpu.bind(
                    (mat1, mat2, group_size, qScaleAndZeros), aten_layout
                )
            ]
            if use_aten_gemm_kernels()
            else []
        )
        if (
            (config.max_autotune or config.max_autotune_gemm)
            and use_cpp_gemm_template(
                aten_layout,
                mat1,
                mat2,
                mat2_transposed=True,
                is_woq_int4=True,
                q_group_size=qGroupSize,
            )
            and mat2.get_layout().is_contiguous()
        ):
            # pyrefly: ignore [bad-specialization, missing-attribute, not-a-type]
            CppWoqInt4GemmTemplate[qGroupSize].add_choices(
                choices,
                aten_layout,
                [mat1, mat2, group_size, qScaleAndZeros],
            )

        # define functions to generate example inputs for weight and group size
        # otherwise, autotuner generates example inputs of all zeros for them
        def get_example_weight(x: torch._inductor.ir.IRNode) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Builds a random uint8 example tensor with the node's size and device for autotuning; raises AssertionError if the layout is not contiguous.
            if not x.get_layout().is_contiguous():
                raise AssertionError("expected x to have a contiguous layout")
            shape = x.get_size()
            device = x.get_device()
            return torch.randint(0, 255, shape, dtype=torch.uint8, device=device)

        input_gen_fns = {
            1: get_example_weight,  # packed weight
            2: lambda x: V.graph.constants[x.get_name()],  # group size
        }

        node, _ = autotune_select_algorithm(
            "_weight_int4pack_mm_for_cpu",
            choices,
            [mat1, mat2, group_size, qScaleAndZeros],
            aten_layout,
            input_gen_fns=input_gen_fns,
        )
        return node

    lowering.make_fallback(aten._dyn_quant_matmul_4bit)
    lowering.make_fallback(aten._dyn_quant_pack_4bit_weight)
