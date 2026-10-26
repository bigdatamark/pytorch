# mypy: allow-untyped-defs
# [MODULE DOC] Runtime (eager) side of FlexGEMM on the QuACK backend: validation helpers
# for operands, layouts and captured epilogue tensors, runtime plan dataclasses for aux
# outputs, local reductions and output contractions, callback registration, the dispatch
# into the vendored QuACK gemm_act kernel, and the public gemm_epilogue entry point that
# the generated wrapper calls. No import-time side effects beyond definitions and the
# swapped-kind table.
from __future__ import annotations

import contextlib
import dataclasses
import os
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR,
    FlexGemmLocalReduceCallbacks,
    FlexGemmLocalReduceGeometry,
    FlexGemmOutputContraction,
    LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR,
    LOCAL_REDUCE_COMBINE_KEY_SUFFIX,
    local_reduce_compressed_shape,
    LOCAL_REDUCE_FINALIZE_KEY_SUFFIX,
    LOCAL_REDUCE_RUNTIME_OUT_ERROR,
    LOCAL_REDUCE_SWAP_AB_ERROR,
    output_contraction_capture_supported,
    output_contraction_config_supported,
    validate_local_reduce_feed_main_capability,
    validate_local_reduce_no_c_alpha_beta,
    validate_local_reduce_out_shape,
    validate_local_reduce_runtime_dense_mm,
    validate_local_reduce_selected_dim_divisible,
)
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._prims_common import is_expandable_to


if TYPE_CHECKING:
    from torch._inductor.heuristics.template.flex_gemm import GemmConfigKey


# swap_ab transposes the dispatched GEMM, so a row broadcast becomes a col
# broadcast (and vice versa) while tile broadcasts only transpose their data and
# scalars are orientation-invariant.
_SWAPPED_ARG_KIND = {"row": "col", "col": "row", "tile": "tile", "scalar": "scalar"}


def inductor_quack_cache_dir() -> str:
    # [FUNCTION DOC] Return the Inductor cache directory joined with a quack
    # subdirectory; used as the cache root for generated FlexGEMM work. No I/O.
    """Return the Inductor-owned QuACK cache root for generated FlexGEMM."""
    return os.path.join(cache_dir(), "quack")


def check_matrix(
    name: str, tensor: torch.Tensor, expected_ndim: int | None = None
) -> None:
    # [FUNCTION DOC] HELPER: Validate a matrix operand: raises RuntimeError if
    # expected_ndim is given and differs or the tensor is not CUDA, and
    # NotImplementedError if its rank is not 2 or 3.
    """Require a CUDA matrix operand with optional generated-op rank checking."""
    if expected_ndim is not None and tensor.ndim != expected_ndim:
        raise RuntimeError(
            f"FlexGEMM expected {expected_ndim}-D {name}, got {tensor.ndim}-D"
        )
    if tensor.ndim not in (2, 3):
        raise NotImplementedError(f"FlexGEMM currently supports only 2-D or 3-D {name}")
    if not tensor.is_cuda:
        raise RuntimeError(f"FlexGEMM requires CUDA {name}")


def check_same_device(a: torch.Tensor, b: torch.Tensor, *rest: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Raise RuntimeError unless all given tensors are on the same
    # device as the first.
    """Require all runtime tensors to live on the same CUDA device."""
    device = a.device
    if b.device != device or any(tensor.device != device for tensor in rest):
        raise RuntimeError("FlexGEMM inputs must be on the same device")


def check_broadcast_shape(
    name: str, shape: torch.Size, expected_shape: tuple[int, ...]
) -> None:
    # [FUNCTION DOC] HELPER: Raise RuntimeError unless the given shape can be expanded
    # (broadcast) to the expected shape.
    """Require a tensor shape to broadcast exactly to the GEMM output shape."""
    if not is_expandable_to(tuple(shape), expected_shape):
        raise RuntimeError(
            f"{name} shape must broadcast to {expected_shape}, got {tuple(shape)}"
        )


def check_matrix_major_layout(name: str, tensor: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Raise NotImplementedError unless the tensor has unit stride
    # in its last or second-to-last dimension (row- or column-major).
    """Require row-major or column-major dense matrix strides."""
    if tensor.stride(-1) != 1 and tensor.stride(-2) != 1:
        raise NotImplementedError(
            f"FlexGEMM requires {name} to be row- or column-major"
        )


def check_matrix_row_major_layout(name: str, tensor: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Raise NotImplementedError unless the tensor's last
    # dimension has unit stride.
    """Require last-dim-contiguous matrix strides for QuACK-written aux storage."""
    if tensor.stride(-1) != 1:
        raise NotImplementedError(f"FlexGEMM requires {name} to be row-major")


def check_epilogue_arg_kinds(epilogue_arg_kinds: tuple[str, ...]) -> None:
    # [FUNCTION DOC] HELPER: Raise NotImplementedError if any declared kind is not one
    # of tile, row, col or scalar.
    """Require each epilogue arg kind to be row, col, tile, or scalar."""
    for kind in epilogue_arg_kinds:
        if kind not in ("tile", "row", "col", "scalar"):
            raise NotImplementedError(
                f"FlexGEMM supports only tile/row/col/scalar args, got {epilogue_arg_kinds}"
            )


def infer_epilogue_arg_kind(a: torch.Tensor, b: torch.Tensor, arg: torch.Tensor) -> str:
    # [FUNCTION DOC] HELPER: Infer a captured tensor's kind from its 2-D shape relative
    # to the output (rows of a by columns of b): scalar for 1x1, tile for full, row for
    # 1xN, col for Mx1, checked in that order. Raises NotImplementedError otherwise.
    """Infer a captured epilogue tensor's broadcast kind from its shape."""
    m, n = a.shape[-2], b.shape[-1]
    if tuple(arg.shape) == (1, 1):
        return "scalar"
    if tuple(arg.shape) == (m, n):
        return "tile"
    if tuple(arg.shape) == (1, n):
        return "row"
    if tuple(arg.shape) == (m, 1):
        return "col"
    raise NotImplementedError(
        "FlexGEMM captured tensor args must match the GEMM output "
        "shape or broadcast as [1, N] / [M, 1] / [1, 1]"
    )


def validate_epilogue_arg_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    arg: torch.Tensor,
    kind: str,
) -> None:
    # [FUNCTION DOC] HELPER: Raise RuntimeError unless the captured tensor's shape
    # equals the shape its declared kind requires for the output.
    """Require a captured epilogue tensor shape to match its declared kind."""
    m, n = a.shape[-2], b.shape[-1]
    expected_shapes = {
        "tile": (m, n),
        "row": (1, n),
        "col": (m, 1),
        "scalar": (1, 1),
    }
    if tuple(arg.shape) != expected_shapes[kind]:
        raise RuntimeError(
            f"{kind} epilogue arg shape must be {expected_shapes[kind]}, "
            f"got {tuple(arg.shape)}"
        )


def resolve_epilogue_arg_kinds(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_args: tuple[torch.Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Return the epilogue arg kinds to use. Raises RuntimeError
    # if declared kinds differ in count from the args or fail shape validation, and
    # NotImplementedError for unknown kinds. If no kinds were declared, infer each from
    # the tensor shapes.
    """Validate declared epilogue arg kinds or infer them from tensor shapes."""
    if epilogue_arg_kinds and len(epilogue_arg_kinds) != len(epilogue_args):
        raise RuntimeError("epilogue_arg_kinds must match epilogue_args length")
    check_epilogue_arg_kinds(epilogue_arg_kinds)
    if not epilogue_arg_kinds:
        return tuple(infer_epilogue_arg_kind(a, b, arg) for arg in epilogue_args)
    for arg, kind in zip(epilogue_args, epilogue_arg_kinds):
        validate_epilogue_arg_shape(a, b, arg, kind)
    return epilogue_arg_kinds


# NOTE [Boolean epilogue tensor storage]
# PyTorch bool tensors are byte-addressed, but CuTeDSL models cutlass.Boolean as
# a 1-bit logical type. Keep the manual uint8 storage view
# see https://github.com/NVIDIA/cutlass/issues/3348 for details
def quack_epilogue_arg(arg: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Return the tensor viewed as uint8 if it is bool (QuACK's
    # physical ABI), otherwise unchanged.
    """Adapt logical epilogue tensors to QuACK's physical tensor ABI."""
    return arg.view(torch.uint8) if arg.dtype is torch.bool else arg


def split_epilogue_args(
    epilogue_args: tuple[torch.Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    # [FUNCTION DOC] HELPER: Partition epilogue tensors by kind into four tuples (row,
    # col, tile, scalar). Bool tensors become uint8 views; col tensors are reshaped to a
    # leading-batch vector, tile tensors get a leading batch dim, scalars are flattened
    # to one element. Returns the four tuples in that order.
    """Group epilogue tensors into QuACK row, col, tile, and scalar argument lists."""
    row_args = []
    col_args = []
    tile_args = []
    scalar_args = []
    for arg, kind in zip(epilogue_args, epilogue_arg_kinds):
        arg = quack_epilogue_arg(arg)
        match kind:
            case "row":
                row_args.append(arg)
            case "col":
                col_args.append(arg.squeeze(-1).unsqueeze(0))
            case "tile":
                tile_args.append(arg.unsqueeze(0))
            case "scalar":
                scalar_args.append(arg.reshape(1))
    return tuple(row_args), tuple(col_args), tuple(tile_args), tuple(scalar_args)


def normalize_c(
    C: torch.Tensor | None, expected_shape: tuple[int, ...], beta: float
) -> torch.Tensor | None:
    # [FUNCTION DOC] Return the C tensor QuACK should read: None if C is None or beta is
    # zero; otherwise C broadcast to the expected shape, after checking broadcastability
    # (RuntimeError), CUDA rank and major-layout (NotImplementedError).
    """Return the effective C tensor that QuACK should read for alpha/beta GEMMs."""
    if C is None:
        return None
    check_broadcast_shape("C", C.shape, expected_shape)
    if beta == 0:
        return None
    broadcast_C = torch.broadcast_to(C, expected_shape)
    check_matrix("C", broadcast_C)
    check_matrix_major_layout("C", broadcast_C)
    return broadcast_C


@dataclasses.dataclass(frozen=True)
class FlexGemmRuntimeLocalReducePlan:
    # [CLASS DOC] Frozen runtime plan for one local reduction: its geometry (group,
    # axis), an optional compressed output tensor, optional physical callbacks (combine
    # and finalize functions), and whether the reduction feeds the main output.
    # Invariants checked at construction; group and axis expose geometry fields.
    """Runtime plan for one local reduction and its output/feed-main consumers."""

    geometry: FlexGemmLocalReduceGeometry
    out: torch.Tensor | None = None
    callbacks: FlexGemmLocalReduceCallbacks | None = None
    feeds_main: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validate the plan. Raises RuntimeError if there is no output
        # tensor and it does not feed main, if a feed-main plan lacks callbacks (also
        # checks the feed-main capability for this axis and group), or if a non-feed-
        # main plan whose geometry needs physical callbacks lacks them.
        """Reject plans without the output/callback state required by their consumer."""
        if self.out is None and not self.feeds_main:
            raise RuntimeError(LOCAL_REDUCE_RUNTIME_OUT_ERROR)
        if self.feeds_main:
            if self.callbacks is None:
                raise RuntimeError(LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR)
            validate_local_reduce_feed_main_capability(self.axis, self.group)
        elif self.geometry.needs_physical_callbacks and self.callbacks is None:
            raise RuntimeError(LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR)

    @property
    def group(self) -> int:
        # [FUNCTION DOC] Property: group size from the geometry.
        return self.geometry.group

    @property
    def axis(self) -> int:
        # [FUNCTION DOC] Property: reduction axis from the geometry.
        return self.geometry.axis


@dataclasses.dataclass(frozen=True)
class FlexGemmRuntimeOutputPlan:
    # [CLASS DOC] Frozen bundle of all user-visible output consumers for the generated
    # runtime ABI: aux output tensors, an optional local-reduce plan, and an optional
    # output contraction. A contraction cannot coexist with the other two.
    """Bundle all user-visible output consumers into one generated ABI value."""

    aux_outs: tuple[torch.Tensor, ...] = ()
    local_reduce: FlexGemmRuntimeLocalReducePlan | None = None
    output_contraction: FlexGemmOutputContraction | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raise NotImplementedError if an output contraction is combined
        # with aux outputs or a local reduce.
        if self.output_contraction is not None and (
            self.aux_outs or self.local_reduce is not None
        ):
            raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR)

    def output_shape(self, physical_shape: tuple[int, ...]) -> tuple[int, ...]:
        # [FUNCTION DOC] Return the logical output shape: the physical shape unchanged
        # without a contraction, otherwise the last dim divided by the contraction
        # group. Raises RuntimeError if the last dim is not divisible by the group.
        """Return the logical shape after applying the output contraction."""
        if self.output_contraction is None:
            return physical_shape
        if physical_shape[-1] % self.output_contraction.group != 0:
            raise RuntimeError(FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR)
        return (
            *physical_shape[:-1],
            physical_shape[-1] // self.output_contraction.group,
        )


def validate_runtime_local_reduce(
    plan: FlexGemmRuntimeLocalReducePlan | None,
    a: torch.Tensor,
    expected_shape: tuple[int, ...],
    effective_C: torch.Tensor | None,
    alpha: float,
    beta: float,
) -> None:
    # [FUNCTION DOC] Validate a local-reduce plan at runtime (no-op for None): dense 2-D
    # mm only, selected dim divisible by group, no effective C or non-default
    # alpha/beta, and, if there is an output tensor, that it is CUDA, row-major and has
    # the expected compressed shape. Raises via the validators.
    """Validate local-reduce runtime tensor shapes and unsupported consumers."""
    if plan is None:
        return
    validate_local_reduce_runtime_dense_mm(a.ndim)
    validate_local_reduce_selected_dim_divisible(expected_shape, plan.group, plan.axis)
    validate_local_reduce_no_c_alpha_beta(effective_C, alpha, beta)
    local_reduce_out = plan.out
    if local_reduce_out is None:
        return
    check_matrix("local_reduce_out", local_reduce_out)
    check_matrix_row_major_layout("local_reduce_out", local_reduce_out)
    expected_local_reduce_shape = local_reduce_compressed_shape(
        expected_shape, plan.group, plan.axis
    )
    validate_local_reduce_out_shape(local_reduce_out.shape, expected_local_reduce_shape)


def local_reduce_callback_key(callback: Any, fallback_key: str) -> str:
    # [FUNCTION DOC] HELPER: Return the callback's cache key (calling it if callable)
    # when it defines one, else the fallback key. Raises RuntimeError if the key is not
    # a string.
    """Use a generated callback cache key when present, otherwise its caller key."""
    cache_key = getattr(callback, "__cache_key__", None)
    if cache_key is None:
        return fallback_key
    key = cache_key() if callable(cache_key) else cache_key
    if not isinstance(key, str):
        raise RuntimeError("local-reduce callback __cache_key__ must return a string")
    return key


def register_runtime_local_reduce_callbacks(
    local_reduce: FlexGemmRuntimeLocalReducePlan | None,
    epilogue_key: str,
) -> tuple[str | None, str | None]:
    # [FUNCTION DOC] Register generated combine/finalize callbacks with QuACK under
    # cache keys derived from the callbacks or from the epilogue key plus suffixes. Side
    # effect: updates the QuACK local-reduce function registry. Returns (combine key,
    # finalize key), or (None, None) when there are no callbacks.
    """Register generated physical callbacks and return QuACK registry keys."""
    if local_reduce is None or local_reduce.callbacks is None:
        return None, None
    callbacks = local_reduce.callbacks
    local_reduce_combine_key = local_reduce_callback_key(
        callbacks.combine_fn, f"{epilogue_key}{LOCAL_REDUCE_COMBINE_KEY_SUFFIX}"
    )
    local_reduce_finalize_key = local_reduce_callback_key(
        callbacks.finalize_fn, f"{epilogue_key}{LOCAL_REDUCE_FINALIZE_KEY_SUFFIX}"
    )
    from torch._vendor.quack.gemm_act import register_local_reduce_fns

    register_local_reduce_fns(
        local_reduce_combine_key,
        callbacks.combine_fn,
        local_reduce_finalize_key,
        callbacks.finalize_fn,
    )
    return local_reduce_combine_key, local_reduce_finalize_key


def local_reduce_gemm_act_kwargs(
    local_reduce: FlexGemmRuntimeLocalReducePlan | None,
    local_reduce_out: torch.Tensor | None,
    callback_keys: tuple[str | None, str | None],
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Build the keyword arguments for QuACK gemm_act describing
    # the local reduction (output presence, feeds-main flag, output tensor, group, axis,
    # callback keys). Returns an empty mapping when there is no plan.
    """Map a concrete runtime plan onto QuACK's public local-reduce kwargs."""
    if local_reduce is None:
        return {}
    local_reduce_combine_key, local_reduce_finalize_key = callback_keys
    return {
        "tensor_epilogue_returns_local_reduce": local_reduce_out is not None,
        "local_reduce_feeds_main": local_reduce.feeds_main,
        "local_reduce_out": local_reduce_out,
        "local_reduce_group": local_reduce.group,
        "local_reduce_axis": local_reduce.axis,
        "local_reduce_combine_key": local_reduce_combine_key,
        "local_reduce_finalize_key": local_reduce_finalize_key,
    }


def dispatch_gemm_act(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor | None,
    out: torch.Tensor,
    output_plan: FlexGemmRuntimeOutputPlan,
    local_reduce_callback_keys: tuple[str | None, str | None],
    epilogue_key: str,
    epilogue_arg_kinds: tuple[str, ...],
    row_args: tuple[torch.Tensor, ...],
    col_args: tuple[torch.Tensor, ...],
    tile_args: tuple[torch.Tensor, ...],
    scalar_args: tuple[torch.Tensor, ...],
    alpha: float,
    beta: float,
    config,
    device_capacity_override: tuple[int, int] | None = None,
) -> None:
    # [FUNCTION DOC] Call the vendored QuACK gemm_act for one dense GEMM. Uses b
    # transposed as QuACK's B operand, applies swap_ab if configured (swaps operands,
    # transposes out/C/aux views and tile args, exchanges row/col args and kinds; raises
    # NotImplementedError when combined with an output contraction or a local reduce),
    # adds a leading batch dim to 2-D operands, views aux outputs as QuACK ABI types,
    # and passes tile config, epilogue key, arg groups, local-reduce kwargs, contraction
    # group and layout, alpha, beta and device-capacity override. With aux outputs the
    # main result is D and aux go to PostAct; otherwise out is PostAct. Writes results
    # in place; returns None.
    """Dispatch one dense FlexGEMM call to the vendored QuACK GEMM kernel.

    ``config.swap_ab`` dispatches the transposed problem (only the tile schedule
    changes, not numerics): it swaps the A/B operands, writes through transposed
    ``out``/``C``/``aux_outs`` views, and swaps the row/col broadcast roles of
    captured epilogue tensors so each still aligns with the transposed accumulator.
    Tuple epilogues route the main result through QuACK ``D`` and aux outputs through
    ``PostAct``/``mAuxOut``. Output contractions instead leave ``D`` unused and use
    ``PostAct``/``mAuxOut`` as the contracted logical main store.
    """
    from torch._vendor.quack.gemm_act import gemm_act as gemm_act_dispatch

    # QuACK consumes A as (l, m, k) and B as (l, n, k); b is (k, n) so b.mT is (n, k).
    quack_a, quack_b = a, b.mT
    output_contraction = output_plan.output_contraction
    aux_outs = output_plan.aux_outs
    local_reduce = output_plan.local_reduce
    quack_out, quack_aux_outs, quack_local_reduce_out, quack_c = (
        out,
        aux_outs,
        None if local_reduce is None else local_reduce.out,
        C,
    )
    if config.swap_ab:
        if output_contraction is not None:
            raise NotImplementedError(
                "FlexGEMM output contractions do not support swap_ab configs yet"
            )
        quack_a, quack_b = quack_b, quack_a
        quack_out = out.mT
        if local_reduce is not None:
            raise NotImplementedError(LOCAL_REDUCE_SWAP_AB_ERROR)
        quack_aux_outs = tuple(aux_out.mT for aux_out in aux_outs)
        quack_c = None if C is None else C.mT
        row_args, col_args = col_args, row_args
        tile_args = tuple(tile.mT for tile in tile_args)
        epilogue_arg_kinds = tuple(
            _SWAPPED_ARG_KIND[kind] for kind in epilogue_arg_kinds
        )

    # QuACK expects a leading batch dim; 2-D (non-batched) operands get one here.
    quack_a = quack_a.unsqueeze(0) if quack_a.ndim == 2 else quack_a
    quack_b = quack_b.unsqueeze(0) if quack_b.ndim == 2 else quack_b
    quack_out = quack_out.unsqueeze(0) if quack_out.ndim == 2 else quack_out
    quack_aux_outs = tuple(quack_epilogue_arg(aux_out) for aux_out in quack_aux_outs)
    quack_aux_outs = tuple(
        aux_out.unsqueeze(0) if aux_out.ndim == 2 else aux_out
        for aux_out in quack_aux_outs
    )
    if quack_local_reduce_out is not None and quack_local_reduce_out.ndim == 2:
        quack_local_reduce_out = quack_local_reduce_out.unsqueeze(0)
    if quack_c is not None and quack_c.ndim == 2:
        quack_c = quack_c.unsqueeze(0)

    returns_aux = bool(quack_aux_outs)
    quack_d = quack_out if returns_aux else None
    quack_postact = quack_aux_outs if returns_aux else quack_out
    local_reduce_kwargs = local_reduce_gemm_act_kwargs(
        local_reduce, quack_local_reduce_out, local_reduce_callback_keys
    )

    gemm_act_dispatch(
        quack_a,
        quack_b,
        quack_d,
        quack_c,
        quack_postact,
        None,  # tile_count_semaphore
        None,  # activation
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=config.is_dynamic_persistent,
        tensor_epilogue_key=epilogue_key,
        tensor_epilogue_returns_aux=returns_aux,
        tensor_epilogue_arg_kinds=epilogue_arg_kinds,
        **local_reduce_kwargs,
        tensor_epilogue_rowvec_biases=row_args,
        tensor_epilogue_colvec_biases=col_args,
        tensor_epilogue_tile_biases=tile_args,
        tensor_epilogue_scalar_biases=scalar_args,
        main_output_transform_group=(
            None if output_contraction is None else output_contraction.group
        ),
        concat_layout=(
            () if output_contraction is None else output_contraction.concat_layout
        ),
        alpha=alpha,
        beta=beta,
        use_tma_gather=config.use_tma_gather,
        device_capacity_override=device_capacity_override,
    )


def gemm_epilogue(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_fn,
    epilogue_key: str,
    *,
    C: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    output_plan: FlexGemmRuntimeOutputPlan = FlexGemmRuntimeOutputPlan(),
    epilogue_args: tuple[torch.Tensor, ...] = (),
    epilogue_arg_kinds: tuple[str, ...] = (),
    config_key: GemmConfigKey | None = None,
    config_is_lowering_validated: bool = False,
    expected_ndim: int | None = None,
    device_capacity_override: tuple[int, int] | None = None,
    stream: int | None = None,
    quack_cache_dir: str | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Public runtime entry: run a dense (2-D or 3-D) GEMM with a CuTeDSL
    # epilogue through QuACK. Validates operands (CUDA, layouts, shape compatibility,
    # matching batch), output contraction constraints (device capability, B layout, 2-D,
    # no C and default alpha/beta), out shape and dtype, aux outputs (2-D mm only, no
    # C/alpha/beta, matching shape), local-reduce plan, and captured args (not batched,
    # no C, default alpha/beta, same device, valid kinds). Registers the epilogue and
    # callbacks, allocates out if missing, resolves the config (explicit key or first
    # device candidate; contraction-compatible choice when needed, NotImplementedError
    # if unsupported), then dispatches on the given CUDA stream and cache directory.
    # Returns the main output tensor. Raises RuntimeError for shape/device/dtype errors
    # and NotImplementedError for unsupported combinations.
    """Run a dense GEMM through QuACK with a CuTeDSL epilogue.

    Args:
        a: Left operand with shape ``[M, K]`` or ``[B, M, K]``.
        b: Right operand with shape ``[K, N]`` or ``[B, K, N]``.
        epilogue_fn: CuTeDSL epilogue callable applied to the accumulator tile.
        epilogue_key: Stable cache key component for the epilogue.
        C: Optional bias/addend broadcastable to the output shape.
        alpha: Scale applied to the GEMM accumulator.
        beta: Scale applied to ``C`` when ``C`` is present.
        out_dtype: Optional output dtype. Defaults to ``a.dtype``.
        out: Optional preallocated output tensor with shape ``[M, N]`` or ``[B, M, N]``.
        output_plan: Structural plan for auxiliary and contracted outputs.
        epilogue_args: Optional tensor args captured by the epilogue.
        epilogue_arg_kinds: Explicit ``tile``, ``row``, ``col``, or ``scalar`` kind per arg.
        config_key: Optional explicit QuACK config key selected by Inductor autotune.
        config_is_lowering_validated: Whether generated lowering already checked the
            config against the output-contraction contract.
        expected_ndim: Optional generated-op rank contract for A and B operands.
        device_capacity_override: Parent-computed capability for compile-only workers.
        stream: Optional raw CUDA stream pointer supplied by the generated wrapper.
        quack_cache_dir: Optional scoped cache root for Inductor-generated QuACK work.

    Returns:
        Tensor with shape ``[M, N]`` or ``[B, M, N]``.
    """
    check_matrix("a", a, expected_ndim)
    check_matrix("b", b, expected_ndim)
    check_matrix_major_layout("a", a)
    check_matrix_major_layout("b", b)
    if a.ndim != b.ndim:
        raise RuntimeError("FlexGEMM inputs must both be 2-D or both be 3-D")
    if a.ndim == 3 and a.shape[0] != b.shape[0]:
        raise RuntimeError("FlexGEMM batched inputs must have the same batch size")
    if a.shape[-1] != b.shape[-2]:
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied ({a.shape} and {b.shape})"
        )
    physical_output_shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
    aux_outs = output_plan.aux_outs
    local_reduce = output_plan.local_reduce
    output_contraction = output_plan.output_contraction
    logical_output_shape = output_plan.output_shape(physical_output_shape)
    if output_contraction is not None:
        device_capacity = (
            torch.cuda.get_device_capability(a.device)
            if device_capacity_override is None
            else device_capacity_override
        )
        output_contraction.validate_quack(device_capacity[0])
        if output_contraction.chunked and b.stride(-1) == 1:
            raise NotImplementedError(FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR)
        if a.ndim != 2 or C is not None or alpha != 1.0 or beta != 1.0:
            raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR)
    expected_dtype = out_dtype
    if expected_dtype is None:
        expected_dtype = out.dtype if out is not None else a.dtype
    effective_C = normalize_c(C, physical_output_shape, beta)
    if out is not None:
        check_matrix("out", out)
        check_matrix_major_layout("out", out)
        if tuple(out.shape) != logical_output_shape:
            raise RuntimeError(
                f"out shape must be {logical_output_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != expected_dtype:
            raise RuntimeError(f"out dtype must be {expected_dtype}, got {out.dtype}")
    if aux_outs:
        if a.ndim != 2:
            raise NotImplementedError(
                "FlexGEMM generic aux tuple epilogues currently support only 2-D aten.mm"
            )
        if effective_C is not None or alpha != 1.0 or beta != 1.0:
            raise NotImplementedError(
                "FlexGEMM generic aux tuple epilogues cannot be combined with C/alpha/beta yet"
            )
        for index, aux_out in enumerate(aux_outs):
            check_matrix(f"aux_outs[{index}]", aux_out)
            check_matrix_major_layout(f"aux_outs[{index}]", aux_out)
            if tuple(aux_out.shape) != physical_output_shape:
                raise RuntimeError(
                    f"aux_outs[{index}] shape must be {physical_output_shape}, "
                    f"got {tuple(aux_out.shape)}"
                )
    validate_runtime_local_reduce(
        local_reduce,
        a,
        physical_output_shape,
        effective_C,
        alpha,
        beta,
    )
    if a.ndim == 3 and epilogue_args:
        raise NotImplementedError("FlexGEMM batched args are not supported yet")
    if epilogue_args and effective_C is not None:
        # TODO: Route this through the flex frontend so validated A/B/C metadata
        # can be reused here.
        raise NotImplementedError("FlexGEMM args cannot be combined with C yet")
    if epilogue_args and (alpha != 1.0 or beta != 1.0):
        raise NotImplementedError(
            "FlexGEMM args cannot be combined with non-default alpha/beta yet"
        )
    tensors = (
        C,
        out,
        *aux_outs,
        None if local_reduce is None else local_reduce.out,
        *epilogue_args,
    )
    check_same_device(a, b, *(tensor for tensor in tensors if tensor is not None))
    inferred_arg_kinds = resolve_epilogue_arg_kinds(
        a, b, epilogue_args, epilogue_arg_kinds
    )
    if output_contraction is not None and any(
        not output_contraction_capture_supported(kind, arg.dtype is torch.bool)
        for arg, kind in zip(epilogue_args, inferred_arg_kinds, strict=True)
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR)
    for index, arg in enumerate(epilogue_args):
        check_matrix_major_layout(f"epilogue_args[{index}]", arg)
    row_args, col_args, tile_args, scalar_args = split_epilogue_args(
        epilogue_args, inferred_arg_kinds
    )

    from torch._vendor.quack.gemm_act import register_tensor_epilogue_fn

    register_tensor_epilogue_fn(epilogue_key, epilogue_fn)
    local_reduce_callback_keys = register_runtime_local_reduce_callbacks(
        local_reduce,
        epilogue_key,
    )
    out = (
        torch.empty(logical_output_shape, device=a.device, dtype=expected_dtype)
        if out is None
        else out
    )
    from torch._inductor.heuristics.template.flex_gemm import (
        candidate_gemm_configs_for_device,
        gemm_config_from_key,
    )
    from torch._vendor.quack.cache import cache_dir_override

    config = gemm_config_from_key(config_key) if config_key is not None else None
    if config is None or (
        output_contraction is not None and not config_is_lowering_validated
    ):
        candidates = candidate_gemm_configs_for_device(a.device)
        if config is not None and config not in candidates:
            raise NotImplementedError(
                "FlexGEMM explicit QUACK config is not supported on this device"
            )
        if output_contraction is None:
            config = candidates[0]
        elif config is None:
            config = next(
                (
                    candidate
                    for candidate in candidates
                    if output_contraction_config_supported(
                        candidate, physical_output_shape[-1]
                    )
                ),
                None,
            )
        elif not output_contraction_config_supported(config, physical_output_shape[-1]):
            config = None
        if config is None:
            raise NotImplementedError(
                "FlexGEMM QUACK config is incompatible with output contraction"
            )

    stream_context = (
        torch.cuda.stream(torch.cuda.ExternalStream(stream, device=a.device))
        if stream is not None
        else contextlib.nullcontext()
    )
    with cache_dir_override(quack_cache_dir), stream_context:
        dispatch_gemm_act(
            a,
            b,
            effective_C,
            out,
            output_plan,
            local_reduce_callback_keys,
            epilogue_key,
            inferred_arg_kinds,
            row_args,
            col_args,
            tile_args,
            scalar_args,
            alpha,
            beta,
            config=config,
            device_capacity_override=device_capacity_override,
        )
    return out
