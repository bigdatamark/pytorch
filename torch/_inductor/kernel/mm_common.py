# [MODULE DOC] Shared helpers for Inductor matmul-family lowerings: symbolic grid functions for Triton mm templates,
# argument normalization (mm_args), accumulator dtype choice, epilogue builders, native-matmul eligibility checks,
# striding/shape checks, and a loader for kernel template files. Import-time side effects: creates a logger and the
# load_kernel_template partial bound to the templates directory next to this file.
# mypy: allow-untyped-defs
import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch._inductor.select_algorithm import realize_inputs, SymbolicGridFn
from torch._inductor.utils import get_current_backend, sympy_product
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

from .. import config
from ..codegen.wrapper import PythonWrapperCodegen
from ..ir import _IntLike, Layout, TensorBox
from ..utils import load_template, triton_type


log = logging.getLogger(__name__)


@SymbolicGridFn
def mm_grid(m, n, meta, *, cdiv):
    # [FUNCTION DOC] Returns the 3-tuple launch grid: ceil(m/BLOCK_M)*ceil(n/BLOCK_N) programs on axis 0, 1 on the others. Uses symbolic cdiv; meta supplies BLOCK_M and BLOCK_N.
    """
    The CUDA grid size for matmul triton templates.
    """
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


@SymbolicGridFn
def persistent_mm_grid(M: int, N: int, meta: dict[str, Any], *, cdiv, min):
    # [FUNCTION DOC] Grid for persistent kernels: the smaller of NUM_SMS and the number of output tiles, as a 3-tuple with trailing ones.
    """Defines the grid for persistent kernels."""
    return (
        min(meta["NUM_SMS"], cdiv(M, meta["BLOCK_M"]) * cdiv(N, meta["BLOCK_N"])),
        1,
        1,
    )


@SymbolicGridFn
def persistent_grouped_mm_grid(*args):
    # [FUNCTION DOC] Grid for persistent grouped mm: NUM_SMS programs on axis 0. Only the last positional arg (the meta mapping) is used.
    meta = args[-1]
    return (meta["NUM_SMS"], 1, 1)


def acc_type(dtype):
    # [FUNCTION DOC] Returns the Triton accumulator type string: float32 for float16, bfloat16 and 1-byte float (fp8) dtypes, else the Triton type of dtype.
    if dtype in (torch.float16, torch.bfloat16):
        return "tl.float32"
    if dtype.is_floating_point and dtype.itemsize == 1:  # fp8 dtypes
        return "tl.float32"
    return triton_type(dtype)


def mm_args(
    mat1,
    mat2,
    *others,
    layout=None,
    out_dtype=None,
    use_4x2_dim=False,
    mat2_transposed=False,
):
    # [FUNCTION DOC] Common argument processing for mm, bmm, addmm: realizes mat1/mat2, checks batch dims and K match (K of mat2 doubled if use_4x2_dim; N/K swapped if mat2_transposed).
    # Builds a FixedLayout from mat1 device, out_dtype (default mat1 dtype) and batch+[m,n] if layout is None; raises an error if both layout and out_dtype are given.
    # Expands and realizes extra inputs to the layout size. Returns [m, n, k, layout, mat1, mat2, *others].
    """
    Common arg processing for mm,bmm,addmm,etc
    """
    mat1, mat2 = realize_inputs(mat1, mat2)
    *b1, m, k1 = mat1.get_size()
    if mat2_transposed:
        *b2, n, k2 = mat2.get_size()
    else:
        *b2, k2, n = mat2.get_size()
    b = [V.graph.sizevars.check_equals_and_simplify(a, b) for a, b in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.check_equals_and_simplify(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()

        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            [*b, m, n],
        )
    else:
        if out_dtype is not None:
            raise AssertionError("out_dtype is ignored if layout is specified.")
    from ..lowering import expand

    others = [realize_inputs(expand(x, layout.size)) for x in others]

    return [m, n, k, layout, mat1, mat2, *others]


def addmm_epilogue(dtype, alpha, beta):
    # [FUNCTION DOC] HELPER: Builds an epilogue callable computing alpha*acc + beta*bias, skipping each multiply when its factor is 1. Constants use dtype.
    def epilogue(acc, bias):
        # [FUNCTION DOC] HELPER: Applies optional scaling by alpha and beta then adds acc and bias via V.ops; returns the resulting op value.
        if alpha != 1:
            acc = V.ops.mul(acc, V.ops.constant(alpha, dtype))
        if beta != 1:
            bias = V.ops.mul(bias, V.ops.constant(beta, dtype))
        return V.ops.add(acc, bias)

    return epilogue


def scale_mm_epilogue():
    # [FUNCTION DOC] Builds an epilogue callable for scaled mm that multiplies the accumulator by the product of the two inverse scales and optionally adds a bias.
    # KNOWN ISSUE: the docstring lists dtype, scale_a, scale_b args but the function takes none.
    """
    Create an epilogue function that applies scaling to matrix multiplication result
    using the given scale factors.

    Args:
        dtype: The data type of the output
        scale_a: Scale factor for matrix A
        scale_b: Scale factor for matrix B

    Returns:
        Epilogue function that takes the accumulator and applies scaling
    """

    def epilogue(acc, inv_a_scale, inv_b_scale, bias=None):
        # [FUNCTION DOC] HELPER: Returns acc times (inv_a_scale times inv_b_scale), plus bias when bias is not None.
        # The epilogue function receives the accumulator (result of mat1 @ mat2)
        # and applies the scaling factors
        # In the original scaled_mm, we use inverse scales, so we multiply by them
        mul_scales = V.ops.mul(inv_a_scale, inv_b_scale)
        mul_acc = V.ops.mul(acc, mul_scales)
        if bias is not None:
            return V.ops.add(mul_acc, bias)
        else:
            return mul_acc

    return epilogue


def use_native_matmul(mat1, mat2):
    # [FUNCTION DOC] Decides whether to use Triton native matmul codegen. Returns False if disabled in config, device is not cuda/xpu with triton backend,
    # dtypes are unsupported (int8, uint8, fp16, bf16, fp32 only), any of m,k,n has free unbacked symbols, or any is statically <= 1; else True.
    # Raises an error if TMA persistent matmul or block pointers are enabled alongside native matmul.
    if not config.triton.native_matmul:
        return False

    # If tma matmul is on, don't do native matmul
    if (
        config.triton.enable_persistent_tma_matmul
        and torch.utils._triton.has_triton_tma_device()
    ):
        raise AssertionError("native matmul doesn't support tma codegen yet")

    # Currently only enable native matmul for default indexing
    # TODO : support block ptr
    if config.triton.use_block_ptr:
        raise AssertionError("native matmul doesn't support block_ptr codegen yet")

    # Currently only enable native matmul for triton on GPU.
    device_type = mat1.get_device().type
    if not (
        device_type in ("cuda", "xpu") and get_current_backend(device_type) == "triton"
    ):
        return False

    # Currently, tl.dot only supports following dtypes
    triton_supported_dtype = [
        torch.int8,
        torch.uint8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    if mat1.dtype not in triton_supported_dtype:
        return False
    if mat2.dtype not in triton_supported_dtype:
        return False

    # (..., M, K) @ (..., K, N)
    m, k, n = mat1.get_size()[-2], mat1.get_size()[-1], mat2.get_size()[-1]

    # If the shape has unbacked symbols, don't do native matmul.
    # This is related to the behavior of statically_known_multiple_of on unbacked symints.
    # Since statically_known_multiple_of just returns False for unbacked symbols
    # due to the expensive cost, codegen fails when there is a unbacked symbol.
    # In particular, it fails at _split_iteration_ranges in codegen/simd.py.
    # See this : https://github.com/pytorch/pytorch/pull/131649
    if any(map(has_free_unbacked_symbols, [m, k, n])):
        return False

    # Consider the shape (m,k,n) > 1
    # TODO : support when size = 1
    if (
        V.graph.sizevars.statically_known_leq(m, 1)
        or V.graph.sizevars.statically_known_leq(k, 1)
        or V.graph.sizevars.statically_known_leq(n, 1)
    ):
        return False

    return True


def _use_small_mm_pointwise(m, k, n, layout) -> bool:
    # [FUNCTION DOC] HELPER: True when max autotune is off, device is not cpu/mps, and m>=64, k<5, n<5 are statically known; small mm is then lowered to pointwise ops.
    """Check if mm should be lowered to pointwise ops for small K and N.

    For very small inner dimensions (K < 5 and N < 5) with M >= 64,
    cuBLAS launch overhead dominates and a fused pointwise kernel is
    faster (1.2-4.8x on H200).  M >= 64 excludes tiny matrices where
    pointwise kernel launch overhead negates the benefit.  K,N < 5
    avoids shapes where Triton codegen quality is unstable (K=5 regresses
    at large M due to reduction-dimension alignment).  Disabled under
    max_autotune to preserve template selection.
    See https://github.com/pytorch/pytorch/issues/186348
    """
    if config.max_autotune or config.max_autotune_gemm:
        return False
    if layout.device.type in ("cpu", "mps"):
        return False
    skt = V.graph.sizevars.statically_known_true
    return skt(m >= 64) and skt(k < 5) and skt(n < 5)


def _is_static_problem(layout: Layout) -> tuple[bool, bool]:
    # [FUNCTION DOC] HELPER: Returns (is_static, is_nonzero) for layout.size. If sizes are not all static, returns False and whether no dim is statically zero;
    # otherwise True and whether the element count is positive.
    """
    Check if input tensors and output layout have static shapes and non-zero sizes.

    Args:
        layout: Output layout object with a 'size' attribute.

    Returns:
        Tuple[bool, bool]: (is_static, is_nonzero)
            is_static: True if all shapes are statically known
            is_nonzero: True if all dimensions are non-zero
    """
    static_shape = True
    static_size = PythonWrapperCodegen.statically_known_list_of_ints_or_none(
        layout.size
    )
    if static_size is None:
        nonzero = True
        for s in layout.size:
            sz = PythonWrapperCodegen.statically_known_int_or_none(s)
            if sz is not None and sz == 0:
                nonzero = False
                break
        return False, nonzero
    numel = 1
    for dim in static_size:
        numel *= dim
    nonzero = numel > 0
    return static_shape, nonzero


def check_supported_striding(mat_a: TensorBox, mat_b: TensorBox) -> None:
    # [FUNCTION DOC] Validates that mat_a is row major (last stride 1) and mat_b is column major (second-to-last stride 1), unless the matrix has a zero dim.
    # Raises a runtime error through the check mechanism otherwise. Returns nothing.
    def is_row_major(stride: Sequence[_IntLike]) -> bool:
        # [FUNCTION DOC] HELPER: True if the last stride equals 1.
        return stride[-1] == 1

    def is_col_major(stride: Sequence[_IntLike]) -> bool:
        # [FUNCTION DOC] HELPER: True if the second-to-last stride equals 1.
        return stride[-2] == 1

    def has_zero_dim(size: Sequence[_IntLike]) -> bool:
        # [FUNCTION DOC] HELPER: True if either of the first two sizes is 0.
        return bool(size[0] == 0 or size[1] == 0)

    # Check mat_a (self) stride requirements
    torch._check(
        is_row_major(mat_a.get_stride()) or has_zero_dim(mat_a.get_size()),
        lambda: f"mat_a must be row_major, got stride {mat_a.get_stride()}",
    )

    # Check mat_b stride requirements
    torch._check(
        is_col_major(mat_b.get_stride()) or has_zero_dim(mat_b.get_size()),
        lambda: f"mat_b must be col_major, got stride {mat_b.get_stride()}",
    )


def is_batch_stride_largest_or_zero(mat1, mat2, layout) -> bool:
    # [FUNCTION DOC] True if, for mat1, mat2 and layout, the batch stride is zero or equals the product of the remaining sizes.
    # Raises an error if any of the three is not 3D.
    """
    Checking if the batch stride is the largest in the stride.
    """
    sizes = [mat1.get_size(), mat2.get_size(), layout.size]
    strides = [mat1.get_stride(), mat2.get_stride(), layout.stride]
    for size, stride in zip(sizes, strides):
        if not (len(size) == len(stride) == 3):
            raise AssertionError("Expect 3D tensors")
        if stride[0] != 0 and stride[0] != sympy_product(size[1:]):
            return False

    return True


_KERNEL_TEMPLATE_DIR = Path(__file__).parent / "templates"
load_kernel_template = partial(load_template, template_dir=_KERNEL_TEMPLATE_DIR)
