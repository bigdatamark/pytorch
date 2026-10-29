# [MODULE DOC] Inductor lowerings and autotuning for grouped matmul (_grouped_mm) and scaled grouped matmul (_scaled_grouped_mm).
# Provides Triton persistent-kernel configs and pruning, ROCm/CUDA capability checks, output layout construction, aten extern choices,
# Triton/CuteDSL/NV-universal template choices and the registered lowerings. Import-time side effects: creates templates, extern choices,
# the config list, and registers two lowerings in the global lowering table.
# mypy: allow-untyped-defs
import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
from torch._inductor.heuristics.template.cutedsl import get_groupgemm_configs
from torch._inductor.runtime.triton_compat import tl
from torch._inductor.virtualized import V
from torch.utils._triton import has_triton

from ..ir import ChoiceCaller, Layout, TensorBox
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
    TritonTemplate,
)
from ..utils import (
    get_gpu_shared_memory,
    get_num_sms,
    has_free_symbols,
    use_aten_gemm_kernels,
    use_blackwell_cutedsl_grouped_mm,
    use_nv_universal_gemm_template,
    use_triton_template,
)
from .mm_common import (
    _is_static_problem,
    check_supported_striding,
    load_kernel_template,
    persistent_grouped_mm_grid,
)


log = logging.getLogger(__name__)
aten = torch.ops.aten


@dataclass
class Config:
    # [CLASS DOC] Plain dataclass describing one Triton tuning config: kwargs (mapping of kernel constexpr names to ints such as block sizes),
    # num_stages and num_warps.
    kwargs: dict[str, int]
    num_stages: int
    num_warps: int


_NV_CONFIGS = [
    Config(
        {
            "BLOCK_M": block_size_m,
            "BLOCK_N": block_size_n,
            "BLOCK_K": block_size_k,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [16, 32, 64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
]


def grouped_mm_configs():
    # [FUNCTION DOC] Returns the module-level list of candidate Triton configs (block size, stage and warp combinations).
    return _NV_CONFIGS


def early_config_prune(g, m, dtsize, configs, named_args):
    # [FUNCTION DOC] Filters configs. With static g and m, prunes BLOCK_M by the average rows per group (small m keeps small blocks, large m drops tiny blocks);
    # drops configs needing more shared memory than the GPU has ((BLOCK_M+BLOCK_N)*BLOCK_K*stages*dtsize); for warp-specialized configs requires 4 warps and enough partition size.
    # named_args supplies A_IS_2D and B_IS_2D. Returns the list of kept configs.
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, num_consumer_groups = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            config.num_stages,
            config.num_warps,
            getattr(config, "num_consumer_groups", 0),
        )

        # 1. Prune NV configs depending on g and m.
        if not has_free_symbols((g, m)):
            a_is_2d, b_is_2d = named_args["A_IS_2D"], named_args["B_IS_2D"]
            m_avg = m // g if a_is_2d and not b_is_2d else m
            if m_avg <= 16:
                if BLOCK_M > 32:
                    continue
            elif m_avg <= 32:
                if BLOCK_M > 64:
                    continue
            elif m_avg <= 64:
                if BLOCK_M <= 16:
                    continue
            else:
                if BLOCK_M <= 32:
                    continue

        # 2. make sure we have enough smem
        max_shared_memory = get_gpu_shared_memory()

        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        use_warp_specialization = num_consumer_groups >= 1

        # 3. make sure we can partition for ws
        if use_warp_specialization:
            if num_warps != 4:
                continue

            # "tritongpu-warp-spec-data-partition"
            m_slice = BLOCK_M // num_consumer_groups
            n_slice = BLOCK_N // num_consumer_groups
            if m_slice < 64 and n_slice < 256:
                continue

        pruned_configs.append(config)

    return pruned_configs


triton_grouped_mm_template = TritonTemplate(
    name="grouped_mm",
    grid=persistent_grouped_mm_grid,
    source=load_kernel_template("triton_mm_grouped"),
)

triton_scaled_grouped_mm_template = TritonTemplate(
    name="scaled_grouped_mm",
    grid=persistent_grouped_mm_grid,
    source=load_kernel_template("triton_mm_grouped"),
)

cutedsl_grouped_mm_template = CuteDSLTemplate(
    name="grouped_gemm_cutedsl",
    source=load_kernel_template("cutedsl_mm_grouped"),
)


def has_grouped_mm_triton_support() -> bool:
    # [FUNCTION DOC] True if CUDA is available and either the build is ROCm or the device capability is at least 9.0.
    if not torch.cuda.is_available():
        return False
    if torch.version.hip:
        # The grouped GEMM Triton template is supported on ROCm too. ATen
        # remains a separate autotune choice when fallback kernels are enabled.
        return True
    return torch.cuda.get_device_capability() >= (9, 0)


def _rocm_gcn_arch() -> str:
    # [FUNCTION DOC] HELPER: Returns the GCN arch name of the current device with any feature suffix after a colon removed.
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName.split(":", 1)[0]


def has_rocm_fp8_hardware_support() -> bool:
    # [FUNCTION DOC] True only on ROCm when the arch is gfx94, or gfx120 with ROCm >= 6.3, or gfx95 with ROCm >= 6.5; mirrors the test-side FP8 platform check.
    if not torch.version.hip:
        return False

    # Keep this in sync with torch.testing._internal.common_cuda.PLATFORM_SUPPORTS_FP8;
    # this is the production-side equivalent used to gate Triton FP8 lowering.
    arch = _rocm_gcn_arch()
    rocm_version = tuple(int(v) for v in torch.version.hip.split(".")[:2])
    if arch.startswith("gfx94"):
        return True
    if arch.startswith("gfx120") and rocm_version >= (6, 3):
        return True
    if arch.startswith("gfx95") and rocm_version >= (6, 5):
        return True
    return False


def has_scaled_grouped_mm_triton_support(mat_a: TensorBox, mat_b: TensorBox) -> bool:
    # [FUNCTION DOC] True on non-ROCm. On ROCm requires FP8 hardware and both inputs in the expected FP8 dtype (fnuz for gfx94, otherwise e4m3fn).
    if not torch.version.hip:
        return True
    if not has_rocm_fp8_hardware_support():
        return False

    arch = _rocm_gcn_arch()
    # Match ATen's ROCm rowwise scaled grouped GEMM contract: gfx94 uses the
    # FNUZ FP8 encoding, while newer FP8-capable arches use OCP FP8.
    expected_dtype = (
        torch.float8_e4m3fnuz if arch.startswith("gfx94") else torch.float8_e4m3fn
    )
    return mat_a.get_dtype() == expected_dtype and mat_b.get_dtype() == expected_dtype


def grouped_mm_args(
    mat1: TensorBox,
    mat2: TensorBox,
    offs: TensorBox | None,
    layout=None,
    out_dtype=None,
):
    # [FUNCTION DOC] Normalizes inputs for grouped mm: realizes mat1, mat2 and offs, requires both mats to be 2D or 3D.
    # If layout is None builds a FixedLayout with output size by operand dims (2Dx2D gives groups x m x n and needs offs; 2Dx3D, 3Dx2D and 3Dx3D handled),
    # dtype out_dtype or mat1 dtype; strides are contiguous on ROCm and last dim padded to 16 bytes on CUDA. Raises an error if both layout and out_dtype are given.
    # Returns (mat1 size, mat2 size, layout, mat1, mat2, offs).
    mat1, mat2 = realize_inputs(mat1, mat2)
    if offs is not None:
        realize_inputs(offs)
    mat1_size = mat1.get_size()
    mat2_size = mat2.get_size()

    m1dim, m2dim = len(mat1_size), len(mat2_size)

    if m1dim not in (2, 3):
        raise AssertionError(f"Expected 2D or 3D mat1, got {m1dim}D")
    if m2dim not in (2, 3):
        raise AssertionError(f"Expected 2D or 3D mat2, got {m2dim}D")

    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()
        alignment = 16 // out_dtype.itemsize

        if m1dim == 2:
            if m2dim == 2:
                if offs is None:
                    raise AssertionError("offs must be provided for 2D x 2D grouped mm")
                out_size = [offs.get_size()[0], mat1_size[0], mat2_size[1]]
            else:
                out_size = [mat1_size[0], mat2_size[-1]]
        else:
            if m2dim == 2:
                out_size = [mat1_size[1], mat2_size[1]]
            else:
                out_size = [mat1_size[0], mat1_size[1], mat2_size[-1]]
        # Match the ATen extern output layout: CUDA pads grouped GEMM outputs for
        # TMA alignment, while ROCm's ATen path returns contiguous tensors.
        # TODO: Revisit whether 16-byte alignment would be beneficial for gfx1250.
        if torch.version.hip:
            if len(out_size) == 2:
                out_stride = [out_size[1], 1]
            else:
                out_stride = [out_size[1] * out_size[2], out_size[2], 1]
        else:
            size_padded = (out_size[-1] + alignment - 1) // alignment * alignment
            if len(out_size) == 2:
                out_stride = [size_padded, 1]
            else:
                out_stride = [out_size[1] * size_padded, size_padded, 1]

        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            out_size,
            out_stride,
        )
    else:
        if out_dtype is not None:
            raise AssertionError("out_dtype is ignored if layout is specified.")

    return (mat1_size, mat2_size, layout, mat1, mat2, offs)


aten__grouped_mm = ExternKernelChoice(
    torch._grouped_mm,
    "at::_grouped_mm",
    op_overload=aten._grouped_mm.default,
    has_out_variant=False,
)


aten__scaled_grouped_mm = ExternKernelChoice(
    torch._scaled_grouped_mm,
    "at::_scaled_grouped_mm",
    op_overload=aten._scaled_grouped_mm.default,
    has_out_variant=False,
)


def can_use_triton_kernel(
    mat_a: TensorBox,
    mat_b: TensorBox,
    offs: TensorBox | None,
    bias: TensorBox | None,
    scale_result: TensorBox | None,
) -> bool:
    # [FUNCTION DOC] True if Triton grouped mm is supported on this device, Triton is installed, bias and scale_result are None, and offs is present exactly when either operand is 2D.
    if not has_grouped_mm_triton_support():
        return False
    if not has_triton():
        return False

    # The _grouped_mm()/_scaled_grouped_mm() operator do not support
    # bias nor scale_result yet.
    if bias is not None:
        return False
    if scale_result is not None:
        return False

    if len(mat_a.get_size()) == 2 or len(mat_b.get_size()) == 2:
        return offs is not None
    else:
        return offs is None


def create_offsets(offs_box, m1_is_2d, m2_is_2d, m, n, k, alignment):
    # [FUNCTION DOC] HELPER: Builds a synthetic offsets tensor for autotuning inputs. Picks the end value (k, m or n) by operand dimensionality; returns None for 3D x 3D.
    # Spreads the group boundaries evenly using size hints, rounds all but the last to a multiple of alignment, sets the last to the end, and casts to the dtype and device of offs_box.
    if m1_is_2d:
        if m2_is_2d:
            end = k
        else:
            end = m
    else:
        if m2_is_2d:
            end = n
        else:
            return None

    end_hint = V.graph.sizevars.optimization_hint(end)
    noffs_hint = V.graph.sizevars.optimization_hint(offs_box.get_size()[0])
    offs = torch.arange(1, noffs_hint + 1, dtype=torch.float32) * (
        end_hint / noffs_hint
    )
    offs[:-1] = (offs[:-1] / alignment).round() * alignment
    offs[-1] = end_hint
    return offs.to(dtype=offs_box.get_dtype(), device=offs_box.get_device())


def _tuned_grouped_mm_common(
    operator_name: str,
    algorithm_name: str,
    extern_kernel_choice: ExternKernelChoice,
    kernel_template: TritonTemplate,
    mat_a: TensorBox,
    mat_b: TensorBox,
    scale_a: TensorBox | None = None,
    scale_b: TensorBox | None = None,
    offs: TensorBox | None = None,
    bias: TensorBox | None = None,
    scale_result: TensorBox | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool | None = None,
    layout: Layout | None = None,
) -> TensorBox:
    # [FUNCTION DOC] HELPER: Shared lowering body for grouped mm and scaled grouped mm. Validates scale args pairing, builds layout via grouped_mm_args, bumps the aten_mm_info counter and logs shapes,
    # checks striding for scaled case, builds input node list (mats, scales, offs) and the aten choice (passing use_fast_accum only if given).
    # Derives group count g and m,n,k and 2D flags from operand dims, checking dims equal. Adds Triton (pruned configs, TMA when available), Blackwell CuteDSL and NV universal choices when eligible,
    # supplies an offsets generator for autotuning inputs, and returns the node chosen by autotune_select_algorithm.
    # Raises an error on inconsistent scale arguments.
    if (scale_a is None) != (scale_b is None):
        raise AssertionError(
            "scale_a and scale_b must both be None or both be provided"
        )
    if scale_result is not None and scale_a is None:
        raise AssertionError("scale_result requires scale_a and scale_b")

    m1_size, m2_size, layout, mat_a, mat_b, offs = grouped_mm_args(
        mat_a, mat_b, offs, layout=layout, out_dtype=out_dtype
    )
    counters["aten_mm_info"][operator_name] += 1
    log_message = f"Tuned {operator_name}: mat1_shape=%s, mat2_shape=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s"
    log.info(
        log_message,
        m1_size,
        m2_size,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )

    if scale_a is not None and scale_b is not None:
        check_supported_striding(mat_a, mat_b)

    # workaround for Inductor not supporting optional tensor input arguments
    input_nodes: list[Any] = [mat_a, mat_b]
    if scale_a is not None:
        input_nodes.append(realize_inputs(scale_a))
    if scale_b is not None:
        input_nodes.append(realize_inputs(scale_b))
    if offs is not None:
        input_nodes.append(realize_inputs(offs))

    if use_fast_accum is None:
        aten_choice = extern_kernel_choice.bind(
            input_nodes,
            layout,
            out_dtype=out_dtype,
        )
    else:
        aten_choice = extern_kernel_choice.bind(
            input_nodes,
            layout,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
    if use_fast_accum is None:
        use_fast_accum = False

    choices: list[ChoiceCaller] = []
    if use_aten_gemm_kernels():
        choices.append(aten_choice)

    _, is_nonzero = _is_static_problem(layout)

    # Checking only for the equality of corresponding dims of
    # multiplicands here, relying on meta function checks for
    # everything else.
    if len(m1_size) == 2:
        if len(m2_size) == 2:
            m, k1 = m1_size
            k2, n = m2_size

            g = offs.get_size()[0]
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = True, True
        else:
            g1 = offs.layout.size[0]
            m, k1 = m1_size
            g2, k2, n = m2_size
            g = V.graph.sizevars.check_equals_and_simplify(g1, g2)
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = True, False
    else:
        if len(m2_size) == 2:
            g1 = offs.layout.size[0]
            g2, m, k1 = m1_size
            k2, n = m2_size
            g = V.graph.sizevars.check_equals_and_simplify(g1, g2)
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = False, True
        else:
            g1, m, k1 = m1_size
            g2, k2, n = m2_size
            g = V.graph.sizevars.check_equals_and_simplify(g1, g2)
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = False, False

    scaled = scale_a is not None

    if (
        is_nonzero
        and use_triton_template(layout)
        and can_use_triton_kernel(mat_a, mat_b, offs, bias, scale_result)
        and (not scaled or has_scaled_grouped_mm_triton_support(mat_a, mat_b))
    ):
        a_is_k_major = mat_a.get_stride()[-1] == 1
        b_is_k_major = mat_b.get_stride()[-2] == 1

        triton_has_make_tensor_descriptor = hasattr(tl, "make_tensor_descriptor")
        triton_has_experimental_make_tensor_descriptor = hasattr(
            tl, "_experimental_make_tensor_descriptor"
        )
        use_tma_load = (
            triton_has_make_tensor_descriptor
            or triton_has_experimental_make_tensor_descriptor
        )
        kwargs = {
            "SCALED": scaled,
            "A_IS_2D": a_is_2d,
            "B_IS_2D": b_is_2d,
            "A_IS_K_MAJOR": a_is_k_major,
            "B_IS_K_MAJOR": b_is_k_major,
            "USE_FAST_ACCUM": use_fast_accum,
            "NUM_SMS": get_num_sms(),
            "USE_TMA_LOAD": use_tma_load,
            "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR": triton_has_experimental_make_tensor_descriptor,
        }

        for config in early_config_prune(
            g, m, mat_a.dtype.itemsize, grouped_mm_configs(), kwargs
        ):
            kernel_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=layout,
                num_stages=config.num_stages,
                num_warps=config.num_warps,
                **kwargs,
                **config.kwargs,
            )

    if use_blackwell_cutedsl_grouped_mm(
        mat_a, mat_b, layout, a_is_2d, b_is_2d, offs, bias, scale_result
    ):
        for config in get_groupgemm_configs():
            kwargs = dict(
                ACC_DTYPE="cutlass.Float32",
            )

            cutedsl_grouped_mm_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=layout,
                **kwargs,
                **asdict(config),
            )

    if (
        is_nonzero
        and a_is_2d
        and not b_is_2d
        and offs is not None
        and use_nv_universal_gemm_template(layout, m, n, k, mat_a, mat_b, offs, g)
    ):
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import (
            add_nv_universal_grouped_gemm_choices,
        )

        add_nv_universal_grouped_gemm_choices(
            choices,
            layout,
            input_nodes,
            accumulator_type=torch.float32,
        )

    input_gen_fns = {}
    if offs is not None:
        input_offs_idx = 2 if scale_a is None else 4
        alignment = 16 // mat_a.dtype.itemsize
        input_gen_fns[input_offs_idx] = lambda x: create_offsets(
            x, a_is_2d, b_is_2d, m, n, k, alignment
        )
    node, _ = autotune_select_algorithm(
        algorithm_name, choices, input_nodes, layout, input_gen_fns=input_gen_fns
    )
    return node


@register_lowering(aten._grouped_mm.default, type_promotion_kind=None)
def tuned_grouped_mm(
    mat_a: TensorBox,
    mat_b: TensorBox,
    offs: TensorBox | None = None,
    bias: TensorBox | None = None,
    out_dtype: torch.dtype | None = None,
    layout: Layout | None = None,
) -> TensorBox:
    # [FUNCTION DOC] Registered lowering for aten._grouped_mm: delegates to the common tuner with the aten and Triton grouped_mm templates and no scales. Returns the tuned TensorBox.
    """Auto-tuning for _grouped_mm() operator."""

    return _tuned_grouped_mm_common(
        "aten._grouped_mm.default",
        "grouped_mm",
        aten__grouped_mm,
        triton_grouped_mm_template,
        mat_a,
        mat_b,
        None,
        None,
        offs,
        bias,
        None,
        out_dtype,
        None,
        layout,
    )


@register_lowering(aten._scaled_grouped_mm.default, type_promotion_kind=None)
def tuned_scaled_grouped_mm(
    mat_a: TensorBox,
    mat_b: TensorBox,
    scale_a: TensorBox,
    scale_b: TensorBox,
    offs: TensorBox | None = None,
    bias: TensorBox | None = None,
    scale_result: TensorBox | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
    layout: Layout | None = None,
) -> TensorBox:
    # [FUNCTION DOC] Registered lowering for aten._scaled_grouped_mm: defaults out_dtype to bfloat16 and delegates to the common tuner with scales and the scaled templates. Returns the tuned TensorBox.
    """Auto-tuning for _scaled_grouped_mm() operator."""

    # matching _scaled_grouped_mm_cuda Blas.cpp implementation
    out_dtype = out_dtype or torch.bfloat16

    return _tuned_grouped_mm_common(
        "aten._scaled_grouped_mm.default",
        "scaled_grouped_mm",
        aten__scaled_grouped_mm,
        triton_scaled_grouped_mm_template,
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        offs,
        bias,
        scale_result,
        out_dtype,
        use_fast_accum,
        layout,
    )
