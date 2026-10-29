# [MODULE DOC] Cutlass API operator wrappers for the vendored SM100 block-scaled dense GEMM kernel.
# Defines epilogue ABI helpers (tracing, padding, kind classification), the
# VendoredDenseBlockScaledGemmKernel operator and its EFC subclass. At import time tries to import
# the kernel implementation; if present, registers both operators with the CuTeDSL provider.
"""Kernel wrapper for dense_blockscaled_gemm_persistent vendored template."""

from __future__ import annotations

import ast
import dataclasses
import functools
import itertools
import logging
from collections.abc import Callable, Generator  # noqa: TC003

import cutlass.operators
from cutlass.operators import ScaleMode, ScaleSwizzleMode
from cutlass.operators.arch import TargetSm
from cutlass.operators.arguments import GemmArguments
from cutlass.operators.artifact import CompiledArtifact  # noqa: TC002
from cutlass.operators.metadata import (
    DenseTensorConstraints,
    GemmOperandsMetadata,
    OperatorMetadata,
    ScaledOperandConstraints,
    Sm100DesignMetadata,
)
from cutlass.operators.mma import BlackwellTcgen05Mma
from cutlass.operators.providers.cutedsl import integration_utils as cutedsl_utils
from cutlass.operators.providers.cutedsl.operator import CuteDslOperator
from cutlass.operators.status import Status
from cutlass.operators.utils.common import tuple_to_string
from cutlass.operators.utils.device import to_cuda_stream
from cutlass.operators.utils.tensor import strides_to_layout_string


log = logging.getLogger(__name__)


_ONES_ALPHA: dict = {}


@dataclasses.dataclass(frozen=True)
class _EpilogueABI:
    # [CLASS DOC] HELPER: Immutable bundle of the epilogue ABI passed to the block-scaled kernel.
    # Holds up to four extra input tensors (padded with None), their kind codes (0 unused, 1 full
    # matrix, 2 per-column/row-vector broadcast along M, 3 per-row broadcast along N), up to three
    # extra output tensors, the total output count, and the index of the primary (D) output.
    # Built from GEMM arguments by from_args, choosing compile-time or runtime tensor attributes.
    """Runtime tensors and metadata derived from a traced epilogue signature."""

    inputs: tuple
    input_kinds: tuple[int, ...]
    outputs: tuple
    output_count: int
    primary_output: int

    @classmethod
    def from_args(cls, args, tensor_attr: str) -> _EpilogueABI:
        # [FUNCTION DOC] HELPER: Builds an _EpilogueABI from GEMM arguments. tensor_attr selects which
        # attribute of each epilogue tensor wrapper is used (compile_time_tensor or runtime_tensor).
        # Raises NotImplementedError (from the helpers) for unsupported input/output counts or shapes.
        outputs, output_count, primary_output = _epilogue_outputs(args, tensor_attr)
        return cls(
            _epilogue_tensors(args, tensor_attr),
            _epilogue_tensor_kinds(args),
            outputs,
            output_count,
            primary_output,
        )


@functools.lru_cache(maxsize=256)
def _epilogue_signature(epilogue_fn) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # [FUNCTION DOC] HELPER: Traces the epilogue function (cached, up to 256 entries) and returns a
    # pair (input names without the accumulator named accum, output names). Imports cutlass lazily.
    from cutlass.operators.fusion import trace_in_out

    inputs, outputs = trace_in_out(epilogue_fn)
    return (
        tuple(name for name in inputs if name != "accum"),
        tuple(outputs),
    )


def _epilogue_tensors(args, attr: str) -> tuple:
    # [FUNCTION DOC] HELPER: Collects the named epilogue input tensors' attribute attr (after padding
    # the leading dimension to a multiple of 8 via _epilogue_abi_tensor) into a tuple of length four,
    # filled with None. Returns four Nones when there is no epilogue.
    # Raises NotImplementedError if more than four inputs are traced.
    epilogue = getattr(args, "epilogue", None)
    tensors = (
        ()
        if epilogue is None
        else tuple(
            getattr(_epilogue_abi_tensor(epilogue.tensors[name]), attr)
            for name in _epilogue_signature(epilogue.epilogue_fn)[0]
        )
    )
    if len(tensors) > 4:
        raise NotImplementedError("NVGEMM scaled epilogues support up to four inputs")
    return tensors + (None,) * (4 - len(tensors))


def _epilogue_abi_tensor(tensor):
    # [FUNCTION DOC] HELPER: Returns the wrapper unchanged when the runtime tensor's leading dim is
    # a multiple of 8; otherwise allocates a zero-uninitialized padded copy (leading dim rounded up
    # to a multiple of 8) on the same device and dtype, copies the data in and returns a new
    # TensorWrapper around it. Side effect: allocates device memory on every such call.
    runtime_tensor = tensor.runtime_tensor
    leading = runtime_tensor.shape[0]
    if leading % 8 == 0:
        return tensor
    from cutlass.operators.utils.tensor import TensorWrapper

    padded_shape = (leading + 7) // 8 * 8, *runtime_tensor.shape[1:]
    import torch

    padded = torch.empty(
        padded_shape,
        dtype=runtime_tensor.dtype,
        device=runtime_tensor.device,
    )
    padded[:leading].copy_(runtime_tensor)
    return TensorWrapper(padded)


def _epilogue_tensor_kinds(args) -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Classifies each epilogue input as a kind code: 1 full matrix, 2 vector
    # broadcast along rows (1-D or second-to-last dim 1), 3 column vector (last dim 1). A 1x1 input
    # is treated as kind 2 if the output has one column, kind 3 if it has one row. Returns a tuple
    # padded with zeros to length four (all zeros without an epilogue).
    # Raises NotImplementedError for scalar inputs when the output is larger than 1x1.
    epilogue = getattr(args, "epilogue", None)
    if epilogue is None:
        return (0, 0, 0, 0)
    kinds = []
    output_m, output_n = args.out.shape[-2:]
    for name in _epilogue_signature(epilogue.epilogue_fn)[0]:
        shape = epilogue.tensors[name].shape
        if shape[-1] == 1 and (len(shape) == 1 or shape[-2] == 1):
            if output_n == 1:
                kinds.append(2)
            elif output_m == 1:
                kinds.append(3)
            else:
                raise NotImplementedError(
                    "NVGEMM scaled epilogues do not support scalar tensor inputs"
                )
        elif len(shape) == 1 or shape[-2] == 1:
            kinds.append(2)
        elif shape[-1] == 1:
            kinds.append(3)
        else:
            kinds.append(1)
    return tuple(kinds) + (0,) * (4 - len(kinds))


def _epilogue_outputs(args, attr: str) -> tuple[tuple, int, int]:
    # [FUNCTION DOC] HELPER: Returns (extra output tensors padded with None to length three, total
    # output count, index of the primary output). The primary output is the one named D if present,
    # else the first; the remaining outputs' attribute attr are the extras. Without an epilogue
    # returns three Nones, count 1, primary 0.
    # Raises NotImplementedError for zero or more than four outputs, or multiple outputs lacking D.
    epilogue = getattr(args, "epilogue", None)
    if epilogue is None:
        return (None, None, None), 1, 0
    output_names = _epilogue_signature(epilogue.epilogue_fn)[1]
    if not output_names or len(output_names) > 4:
        raise NotImplementedError("NVGEMM scaled epilogues support 1-4 outputs")
    if len(output_names) > 1 and "D" not in output_names:
        raise NotImplementedError("NVGEMM scaled multi-store requires a D output")
    primary_index = output_names.index("D") if "D" in output_names else 0
    tensors = tuple(
        getattr(epilogue.tensors[name], attr)
        for index, name in enumerate(output_names)
        if index != primary_index
    )
    return tensors + (None,) * (3 - len(tensors)), len(output_names), primary_index


def _ones_alpha():
    # [FUNCTION DOC] HELPER: Returns a cached per-device TensorWrapper of four float32 ones, used as
    # an identity global-scale alpha so the kernel signature is the same whether or not a scale is
    # fused. Keyed by the current CUDA device in the module-level cache; allocates on first use.
    """Cached per-device (4,)-ones alpha TensorWrapper (identity global scale).

    The kernel always takes an alpha arg so its signature is consistent across
    compile/run paths; when not fusing we pass ones (a no-op *1.0). Len is a
    multiple of 4 (CuTeDSL requires the operand's last dim divisible by 4).
    """
    from cutlass.operators.utils.tensor import TensorWrapper

    import torch

    dev = torch.cuda.current_device()
    tw = _ONES_ALPHA.get(dev)
    if tw is None:
        tw = TensorWrapper(torch.ones(4, dtype=torch.float32, device=f"cuda:{dev}"))
        _ONES_ALPHA[dev] = tw
    return tw


def _epilogue_op_scope(cute):
    # [FUNCTION DOC] HELPER: Builds the global namespace used to evaluate a string epilogue function:
    # a mapping from names erf, exp, gelu, relu, sigmoid, silu and tanh to cute elementwise
    # implementations (gelu is the erf form, silu is x times sigmoid).
    def relu(x):
        # [FUNCTION DOC] HELPER: Elementwise max of x and zero.
        return cute.math.max(x, cute.full_like(x, 0.0))

    def sigmoid(x):
        # [FUNCTION DOC] HELPER: Elementwise logistic function.
        return 1.0 / (1.0 + cute.math.exp(-x))

    def gelu(x):
        # [FUNCTION DOC] HELPER: Elementwise exact (erf based) GELU.
        return 0.5 * x * (1.0 + cute.math.erf(x * 0.7071067811865476))

    return {
        "erf": cute.math.erf,
        "exp": cute.math.exp,
        "gelu": gelu,
        "relu": relu,
        "sigmoid": sigmoid,
        "silu": lambda x: x * sigmoid(x),
        "tanh": cute.math.tanh,
    }


try:
    from ..dense_blockscaled_gemm_persistent import (  # pyrefly: ignore[missing-import]
        Sm100BlockScaledPersistentDenseGemmKernel as BlockScaledGemmKernelImpl,
    )
except ImportError:
    BlockScaledGemmKernelImpl = None  # type: ignore[misc, assignment]


class VendoredDenseBlockScaledGemmKernel(CuteDslOperator):
    # [CLASS DOC] Cutlass API operator wrapping the vendored SM100 persistent block-scaled dense GEMM
    # kernel (MXF8/MXF4/NVF4 inputs). Holds the kernel implementation object (impl), scale-factor
    # vector size, MMA tiler and cluster shape. Provides compile, run and support-check hooks plus
    # class methods that enumerate valid operand/design metadata and generate operator instances.
    """Wrapper for vendored dense blockscaled GEMM template for SM100 GPUs."""

    supported_args_type = GemmArguments
    designed_for_min_cc = 100

    def __init__(self, metadata: OperatorMetadata):
        # [FUNCTION DOC] Initializes from operator metadata: takes the scale-factor vector size from the
        # last element of A's scale mode, tile (M, N) and cluster (M, N) from the design, and constructs the
        # vendored kernel implementation. The environment variable TORCHINDUCTOR_NVGEMM_PREFETCH equal to
        # 1 enables the kernel's prefetch option. Fails if the implementation import was unavailable.
        super().__init__(metadata)

        self.sf_vec_size = metadata.operands.A.mode[-1]
        mma_tiler_mn = (metadata.design.tile_shape[0], metadata.design.tile_shape[1])
        cluster_shape_mn = (
            metadata.design.cluster_shape[0],
            metadata.design.cluster_shape[1],
        )

        import os

        self.impl = BlockScaledGemmKernelImpl(  # pyrefly: ignore[not-callable]
            self.sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch=os.environ.get("TORCHINDUCTOR_NVGEMM_PREFETCH", "0") == "1",
        )
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn

    @staticmethod
    def _major_modes(args):
        # [FUNCTION DOC] HELPER: Static. From the strides of A, B and out (arguments or operand metadata),
        # returns three pairs: A major mode (K-major with tag k, else MN-major tag m), B major mode (K with
        # tag k, else MN with tag n) and output layout (row-major tag n, else column-major tag m).
        # The unit-stride position in the last two dims decides each choice.
        """Extract major modes from arguments or operand metadata."""
        import cutlass.utils as utils
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

        if args.A.stride[-2:].index(1) == 1:
            a_major_mode = (OperandMajorMode.K, "k")
        else:
            a_major_mode = (OperandMajorMode.MN, "m")

        if args.B.stride[-2:].index(1) == 0:
            b_major_mode = (OperandMajorMode.K, "k")
        else:
            b_major_mode = (OperandMajorMode.MN, "n")

        if args.out.stride[-2:].index(1) == 1:
            out_layout = (utils.LayoutEnum.ROW_MAJOR, "n")
        else:
            out_layout = (utils.LayoutEnum.COL_MAJOR, "m")

        return a_major_mode, b_major_mode, out_layout

    def _compile(
        self, args: GemmArguments, target_sm: TargetSm | None = None
    ) -> CompiledArtifact:
        # [FUNCTION DOC] Compiles the kernel for the given arguments and returns a CompiledArtifact.
        # Uses a fake stream and the maximum active clusters for the cluster shape. Always passes an alpha
        # tensor (ones when the arguments have none). The epilogue op is identity by default, else the
        # epilogue function; a string is parsed for its first function definition and executed with the
        # epilogue scope (side effect: runs arbitrary source text). When local reduction output or
        # feeds-main is requested, extra reduction arguments (group, axis, type, source, feeds-main flag,
        # reduction output) are appended to the compile call. target_sm is forwarded.
        import cutlass.cute as cute

        stream = cute.runtime.make_fake_stream()
        max_active_clusters = cutedsl_utils.mma.get_max_active_clusters(
            self.cluster_shape_mn
        )

        # Fused global scale: alpha is ALWAYS threaded as a trailing kernel arg
        # (ones when not fusing) so the kernel signature is consistent across all
        # compile/run paths -- a None alpha is not reliably dropped from the
        # runtime signature. args.alpha (a TensorWrapper, len multiple-of-4) is
        # applied elementwise in the epilogue; closure capture cannot read a
        # runtime tensor there.
        alpha = getattr(args, "alpha", None)
        if alpha is None:
            alpha = _ones_alpha()

        def epilogue_op(v):
            # [FUNCTION DOC] HELPER: Identity epilogue used when the arguments carry no epilogue.
            return v

        if getattr(args, "epilogue", None) is not None:
            epilogue_op = args.epilogue.epilogue_fn
            if isinstance(epilogue_op, str):
                fn_name = next(
                    node.name
                    for node in ast.parse(epilogue_op).body
                    if isinstance(node, ast.FunctionDef)
                )
                scope = _epilogue_op_scope(cute)
                exec(epilogue_op, scope)
                epilogue_op = scope[fn_name]
        epilogue = _EpilogueABI.from_args(args, "compile_time_tensor")
        local_reduce_out = getattr(args, "local_reduce_out", None)
        local_reduce_feeds_main = getattr(args, "local_reduce_feeds_main", False)
        if local_reduce_out is not None or local_reduce_feeds_main:
            return self.cute_compile(
                self.impl,
                args.A.tensor,
                args.B.tensor,
                args.A.scale.tensor,
                args.B.scale.tensor,
                args.out.tensor,
                max_active_clusters,
                stream,
                epilogue_op,
                self.metadata.operands.out.dtype,
                alpha,
                *epilogue.inputs,
                *epilogue.input_kinds,
                *epilogue.outputs,
                epilogue.output_count,
                epilogue.primary_output,
                (
                    local_reduce_out.compile_time_tensor
                    if local_reduce_out is not None
                    else None
                ),
                args.local_reduce_group,
                args.local_reduce_axis,
                args.local_reduce_type,
                args.local_reduce_source,
                local_reduce_feeds_main,
                target_sm=target_sm,
            )
        return self.cute_compile(
            self.impl,
            args.A.tensor,
            args.B.tensor,
            args.A.scale.tensor,
            args.B.scale.tensor,
            args.out.tensor,
            max_active_clusters,
            stream,
            epilogue_op,
            self.metadata.operands.out.dtype,
            alpha,
            *epilogue.inputs,
            *epilogue.input_kinds,
            *epilogue.outputs,
            epilogue.output_count,
            epilogue.primary_output,
            target_sm=target_sm,
        )

    def _run(
        self,
        args: GemmArguments,
        compiled_artifact: CompiledArtifact,
        stream,
        workspace=None,
    ) -> None:
        # [FUNCTION DOC] Executes a compiled kernel. Converts the stream to a CUDA stream (wrapping a raw
        # integer handle in an external torch stream), builds the runtime epilogue ABI under that stream,
        # defaults alpha to ones, and invokes cute_run with A, B, their scale tensors, out, stream, alpha,
        # epilogue inputs and extra outputs, and the local reduction output tensor (None if absent).
        # The argument order must stay consistent with _compile. workspace is unused. Returns nothing.
        import torch

        stream = to_cuda_stream(stream)
        compiled_gemm = compiled_artifact.compiled_obj

        # TVM FFI needs a torch.cuda.Stream, not a raw int handle
        if isinstance(stream, int):
            stream = torch.cuda.ExternalStream(stream)

        # Runtime arg list must match _compile: alpha always trails stream.
        alpha = getattr(args, "alpha", None)
        if alpha is None:
            alpha = _ones_alpha()
        with torch.cuda.stream(stream):
            epilogue = _EpilogueABI.from_args(args, "runtime_tensor")

        local_reduce_out = getattr(args, "local_reduce_out", None)
        local_reduce_feeds_main = getattr(args, "local_reduce_feeds_main", False)
        if local_reduce_out is not None or local_reduce_feeds_main:
            self.cute_run(  # pyrefly: ignore[missing-attribute]
                compiled_gemm,
                args.A.tensor,
                args.B.tensor,
                args.A.scale.tensor,
                args.B.scale.tensor,
                args.out.tensor,
                stream,
                alpha,
                *epilogue.inputs,
                *epilogue.outputs,
                local_reduce_out.runtime_tensor
                if local_reduce_out is not None
                else None,
            )
            return

        self.cute_run(  # pyrefly: ignore[missing-attribute]
            compiled_gemm,
            args.A.tensor,
            args.B.tensor,
            args.A.scale.tensor,
            args.B.scale.tensor,
            args.out.tensor,
            stream,
            alpha,
            *epilogue.inputs,
            *epilogue.outputs,
            None,
        )

    def _supports(
        self, args: GemmArguments, target_sm: TargetSm | None = None
    ) -> Status:
        # [FUNCTION DOC] Checks whether the arguments are supported. For local reduction requests validates
        # that the axis is 0 or 1, the group size is greater than 1 and not above the limit (the tile N
        # extent for axis 1, else 16), and divides the selected dimension; the reduction output must have
        # the reduced shape and be contiguous Float32. Then verifies that scale-factor element counts for A
        # and B match the expected count for their mode and swizzle. Returns Status.fail with a message on
        # any violation, else Status.success. target_sm is unused.
        # Match the upstream block-scaled operator: validate scale-factor element
        # counts via ScaledOperand.numel_scale for the operand's mode/swizzle,
        # rather than re-inferring the layout (the old _infer_scale_swizzle_impl
        # check wrongly rejected valid NVFP4 args on the transposed B operand).
        from cutlass.operators.arguments import ScaledOperand

        local_reduce_out = getattr(args, "local_reduce_out", None)
        if local_reduce_out is not None or getattr(
            args, "local_reduce_feeds_main", False
        ):
            group = args.local_reduce_group
            axis = args.local_reduce_axis
            m, n = args.out.shape[-2:]
            selected_size = n if axis == 1 else m
            max_group = self.mma_tiler_mn[axis] if axis == 1 else 16
            if (
                axis not in (0, 1)
                or group <= 1
                or group > max_group
                or selected_size % group != 0
            ):
                return Status.fail(
                    "Grouped reduction requires a supported M- or N-axis group "
                    "that divides the selected dimension."
                )
            if local_reduce_out is not None:
                expected_shape = (m, n // group) if axis == 1 else (m // group, n)
                if local_reduce_out.shape != expected_shape:
                    return Status.fail(
                        "Grouped reduction output shape must be "
                        f"{expected_shape}; got {local_reduce_out.shape}."
                    )
                if local_reduce_out.dtype is not cutlass.Float32 or tuple(
                    local_reduce_out.stride
                ) != (expected_shape[1], 1):
                    return Status.fail(
                        "Grouped reduction output must be contiguous Float32."
                    )

        m, n = args.out.shape[-2:]
        k = args.A.shape[-1]
        L = args.A.shape[0] if len(args.A.shape) == 3 else 1

        expected_sfa = ScaledOperand.numel_scale((L, m, k), args.A.mode, args.A.swizzle)
        expected_sfb = ScaledOperand.numel_scale((L, n, k), args.B.mode, args.B.swizzle)
        if args.A.scale.numel() != expected_sfa:
            return Status.fail(
                f"Scale factor A must have {expected_sfa} elements for mat shape "
                f"{args.A.shape}; got {args.A.scale.numel()}."
            )
        if args.B.scale.numel() != expected_sfb:
            return Status.fail(
                f"Scale factor B must have {expected_sfb} elements for mat shape "
                f"{args.B.shape}; got {args.B.scale.numel()}."
            )
        return Status.success()

    @staticmethod
    def _is_valid_dtype_combo(ab_dtype, sf_dtype, sf_vec_size, out_dtype) -> bool:
        # [FUNCTION DOC] HELPER: Static predicate for allowed (input dtype, scale dtype, scale vector
        # size, output dtype) combinations: inputs fp4 e2m1 or fp8; vec size 16 or 32; scale e8m0 or
        # e4m3 (e4m3 only with 16); fp8 inputs need 32; output fp32, fp16, bf16 or fp8.
        """Validate dtype/scale-factor/vec-size combinations.

        Matches constraints from the upstream kernel:
          MXF8: Float8E5M2/Float8E4M3FN + Float8E8M0FNU, sf_vec_size=32
          MXF4: Float4E2M1FN + Float8E8M0FNU, sf_vec_size=32
          NVF4: Float4E2M1FN + Float8E8M0FNU/Float8E4M3FN, sf_vec_size=16
        """
        import cutlass

        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            return False
        if sf_vec_size not in {16, 32}:
            return False
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            return False
        # Float8E4M3FN as SF only valid with sf_vec_size=16 (NVF4)
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            return False
        # Float8 AB types require sf_vec_size=32 (MXF8)
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            return False
        if out_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            return False
        return True

    @staticmethod
    def _is_valid_layouts(ab_dtype, a_major, b_major) -> bool:
        # [FUNCTION DOC] HELPER: Static predicate; fp4 inputs require both A and B K-major, other
        # dtypes accept any layout.
        """Float4E2M1FN (MXF4/NVF4) requires row-major (K-major) A and B."""
        import cutlass

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            return False
        return True

    @staticmethod
    def _valid_operands(operands: GemmOperandsMetadata, sf_vec_size: int) -> bool:
        # [FUNCTION DOC] HELPER: Static predicate over operand metadata: A and B dtypes equal, their scale
        # dtypes equal, float32 accumulator, valid dtype combination, and valid major-mode layouts.
        import cutlass

        if operands.A.dtype != operands.B.dtype:
            return False
        if operands.A.scale.dtype != operands.B.scale.dtype:
            return False
        if operands.accumulator_type != cutlass.Float32:
            return False

        ab_dtype = operands.A.dtype
        sf_dtype = operands.A.scale.dtype
        out_dtype = operands.out.dtype

        if not VendoredDenseBlockScaledGemmKernel._is_valid_dtype_combo(
            ab_dtype, sf_dtype, sf_vec_size, out_dtype
        ):
            return False

        (_, a_major), (_, b_major), _ = VendoredDenseBlockScaledGemmKernel._major_modes(
            operands
        )
        if not VendoredDenseBlockScaledGemmKernel._is_valid_layouts(
            ab_dtype, a_major, b_major
        ):
            return False

        return True

    @staticmethod
    def _metadata_operand_combinations() -> Generator[GemmOperandsMetadata, None, None]:
        # [FUNCTION DOC] HELPER: Static generator yielding operand metadata for every combination of input
        # dtype, scale dtype, scale mode (1x16, 1x32), output dtype and row/column-major strides of A, B and
        # out that pass the dtype and layout checks. Divisibility constraints assume 16-byte alignment;
        # swizzle is fixed at 32x4x4 and accumulator at float32.
        import cutlass

        ab_dtypes = [cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN]
        out_dtypes = [
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ]
        sf_dtypes = [cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN]
        scale_modes = [ScaleMode.Blockwise1x16, ScaleMode.Blockwise1x32]

        row_major = (0, 0, 1)
        col_major = (0, 1, 0)
        alignment_bytes = 16

        def major_str_a(stride):
            # [FUNCTION DOC] HELPER: Returns k for a row-major A stride, else m.
            return "k" if stride == row_major else "m"

        def major_str_b(stride):
            # [FUNCTION DOC] HELPER: Returns n for a row-major B stride, else k.
            return "n" if stride == row_major else "k"

        for ab_dtype, sf_dtype, scale_mode, out_dtype in itertools.product(
            ab_dtypes, sf_dtypes, scale_modes, out_dtypes
        ):
            sf_vec_size = scale_mode[-1]
            if not VendoredDenseBlockScaledGemmKernel._is_valid_dtype_combo(
                ab_dtype, sf_dtype, sf_vec_size, out_dtype
            ):
                continue

            for stride_A, stride_B, stride_out in itertools.product(
                [row_major, col_major], repeat=3
            ):
                a_major = major_str_a(stride_A)
                b_major = major_str_b(stride_B)

                if not VendoredDenseBlockScaledGemmKernel._is_valid_layouts(
                    ab_dtype, a_major, b_major
                ):
                    continue

                ab_div = alignment_bytes * 8 // ab_dtype.width
                out_div = alignment_bytes * 8 // out_dtype.width
                sf_div = alignment_bytes * 8 // sf_dtype.width

                yield GemmOperandsMetadata(
                    A=ScaledOperandConstraints(
                        quantized=DenseTensorConstraints(
                            dtype=ab_dtype,
                            stride=stride_A,
                            divisibility=ab_div,
                        ),
                        scale=DenseTensorConstraints(
                            dtype=sf_dtype,
                            stride=None,
                            divisibility=sf_div,
                        ),
                        mode=scale_mode,
                        swizzle=ScaleSwizzleMode.Swizzle32x4x4,
                    ),
                    B=ScaledOperandConstraints(
                        quantized=DenseTensorConstraints(
                            dtype=ab_dtype,
                            stride=stride_B,
                            divisibility=ab_div,
                        ),
                        scale=DenseTensorConstraints(
                            dtype=sf_dtype,
                            stride=None,
                            divisibility=sf_div,
                        ),
                        mode=scale_mode,
                        swizzle=ScaleSwizzleMode.Swizzle32x4x4,
                    ),
                    out=DenseTensorConstraints(
                        dtype=out_dtype,
                        stride=stride_out,
                        divisibility=out_div,
                    ),
                    accumulator_type=cutlass.Float32,
                )

    @classmethod
    def _valid_metadata(cls, metadata: OperatorMetadata) -> bool:
        # [FUNCTION DOC] HELPER: Class method deciding whether operator metadata is acceptable: all but
        # the last scale-mode entries are 1, operands valid, design is an SM100 design, cluster dims
        # positive powers of two with product at most 16 and each at most 4, tile M in 128/256 and tile N
        # in 64/128/192/256, 256-wide M requires an even cluster M, and an epilogue is allowed only for
        # classes whose name contains EFC.
        scale_vec = metadata.operands.A.mode

        if len(scale_vec) > 1:
            for i in range(len(scale_vec) - 1):
                if scale_vec[i] != 1:
                    return False

        sf_vec_size = scale_vec[-1]
        if not VendoredDenseBlockScaledGemmKernel._valid_operands(
            metadata.operands, sf_vec_size
        ):
            return False

        design = metadata.design
        if not isinstance(design, Sm100DesignMetadata):
            return False

        cm, cn, _ = design.cluster_shape
        if cm <= 0 or cn <= 0:
            return False
        if cm * cn > 16:
            return False
        if cm & (cm - 1) != 0 or cn & (cn - 1) != 0:
            return False
        # SF multicast constraint: cluster dims <=4
        if cm > 4 or cn > 4:
            return False

        tile_m, tile_n, _ = design.tile_shape
        if tile_m not in [128, 256]:
            return False
        if tile_n not in [64, 128, 192, 256]:
            return False
        use_2cta = tile_m == 256
        if use_2cta and cm % 2 != 0:
            return False

        if metadata.epilogue is not None and "EFC" not in cls.__name__:
            return False

        return True

    @classmethod
    def _generate_operators(
        cls,
        metadata_filter: Callable[[OperatorMetadata], bool],
        epilogue_args=None,
        target_sm: TargetSm | None = None,
        args=None,
    ) -> list[VendoredDenseBlockScaledGemmKernel]:
        # [FUNCTION DOC] Class method generating operator instances. Returns an empty list if target_sm is
        # given and not compute capability 100, 101 or 103, or if epilogue_args is given. Otherwise
        # enumerates operand combinations crossed with tile shapes (M 128/256, N 64/128/192/256) and
        # cluster shapes (1/2/4 each), builds a descriptive operator name, keeps those that pass
        # _valid_metadata and metadata_filter (if given), and instantiates cls for each. Logs the count.
        if target_sm is not None and target_sm.cc not in [100, 101, 103]:
            return []
        if epilogue_args is not None:
            return []

        design_params = {
            "mma_instruction_type": [BlackwellTcgen05Mma],
            "use_2cta_mma": [True],
            "tile_shape": [
                (M, N, 256) for M in [128, 256] for N in [64, 128, 192, 256]
            ],
            "cluster_shape": [(M, N, 1) for M in [1, 2, 4] for N in [1, 2, 4]],
            "use_tma_store": [True],
        }

        param_names = list(design_params.keys())
        param_values = [design_params[name] for name in param_names]

        operator_list = []

        for operands in cls._metadata_operand_combinations():
            # pyrefly: ignore[no-matching-overload]
            for values in itertools.product(*param_values):
                design = Sm100DesignMetadata(**dict(zip(param_names, values)))

                operator_name = (
                    f"inductor_vendored.{cls.__name__}_sm100_"
                    "{layout}_A{A}_B{B}_out{out}_SFA{SFA}_SFB{SFB}_"
                    "acc{acc}_scale{scale_mode}_swizzle{scale_swizzle}_"
                    "{num_cta}cta_cluster{cluster}_tile{tile}"
                    "{_tma_store}"
                ).format(
                    layout=strides_to_layout_string(
                        operands.A.stride,
                        operands.B.stride,
                        operands.out.stride,
                    ),
                    A=operands.A.dtype,
                    B=operands.B.dtype,
                    out=operands.out.dtype,
                    SFA=operands.A.scale.dtype,
                    SFB=operands.B.scale.dtype,
                    acc=operands.accumulator_type,
                    scale_mode=operands.A.mode,
                    scale_swizzle=operands.A.swizzle,
                    num_cta="2" if design.use_2cta_mma else "1",
                    cluster=tuple_to_string(design.cluster_shape),
                    tile=tuple_to_string(design.tile_shape),
                    _tma_store="_tma_store" if design.use_tma_store else "",
                )

                metadata = OperatorMetadata(
                    operands=operands,
                    design=design,
                    operator_name=operator_name,
                    operator_class=cls,
                    supported_targets=TargetSm.get_supported_targets(design, operands),
                    epilogue=None,
                )

                if cls._valid_metadata(metadata):
                    if metadata_filter is None or metadata_filter(metadata):
                        operator_list.append(cls(metadata))

        log.debug(
            "Generated %d DenseBlockScaledGemmKernel configurations",
            len(operator_list),
        )
        return operator_list


class VendoredDenseBlockScaledGemmEFC(VendoredDenseBlockScaledGemmKernel):
    # [CLASS DOC] Variant of the block-scaled kernel operator whose name contains EFC, which makes
    # _valid_metadata accept metadata carrying an epilogue. Adds no state or methods.
    """Block-scaled provider variant using the kernel's unary epilogue hook."""

    supported_args_type = GemmArguments
    designed_for_min_cc = 100


# Only register if kernel implementation is available
if BlockScaledGemmKernelImpl is not None:
    cutlass.operators.providers.cutedsl.CuTeDSLProvider.register(
        VendoredDenseBlockScaledGemmKernel
    )
    cutlass.operators.providers.cutedsl.CuTeDSLProvider.register(
        VendoredDenseBlockScaledGemmEFC
    )
