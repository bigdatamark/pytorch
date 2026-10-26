# [MODULE DOC] Defines the flex_gemm higher-order operator (FlexGEMM): a GEMM (mm, addmm, bmm or
# baddbmm) fused with a user-supplied epilogue function, carried as a body graph plus the
# GEMM op, its args/kwargs and backend kernel options (backend one of NVGEMM, QUACK, TRITON;
# default TRITON). Contains the op spec tables, body-graph annotation passes, QUACK fast-math
# decompositions, a test-oriented NVFP4 packing helper, the HOP class and its public
# flex_gemm wrapper, plus dense, autograd (deferred not-implemented), fake, functionalization
# and proxy-tracing rules registered at import time on the singleton flex_gemm_hop.
# mypy: allow-untyped-defs
import dataclasses
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, cast

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._higher_order_ops.utils import (
    _check_alias_and_mutation,
    autograd_not_implemented,
    reenter_make_fx,
    register_fake,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


@dataclasses.dataclass(frozen=True)
class FlexGemmOpSpec:
    # [CLASS DOC] Immutable description of a supported GEMM op: its short name, the positional
    # indices of the two matrix operands (mat1_index, mat2_index), the expected ndim of the
    # matrix inputs (2 for mm/addmm, 3 for bmm/baddbmm) and the positional index of the
    # additive bias operand, or None when the op has no bias.
    """Canonical operand positions for a supported FlexGEMM op."""

    name: str
    mat1_index: int
    mat2_index: int
    input_ndim: int
    bias_index: int | None = None


FLEX_GEMM_OP_SPECS = {
    torch.ops.aten.mm.default: FlexGemmOpSpec("mm", 0, 1, input_ndim=2),
    torch.ops.aten.addmm.default: FlexGemmOpSpec(
        "addmm", 1, 2, input_ndim=2, bias_index=0
    ),
    torch.ops.aten.bmm.default: FlexGemmOpSpec("bmm", 0, 1, input_ndim=3),
    torch.ops.aten.baddbmm.default: FlexGemmOpSpec(
        "baddbmm", 1, 2, input_ndim=3, bias_index=0
    ),
}
FLEX_GEMM_OP_ALIASES = {
    torch.mm: torch.ops.aten.mm.default,
    torch.addmm: torch.ops.aten.addmm.default,
    torch.bmm: torch.ops.aten.bmm.default,
    torch.baddbmm: torch.ops.aten.baddbmm.default,
}
_SUPPORTED_BACKENDS = {"NVGEMM", "QUACK", "TRITON"}


_SUPPORTED_FLEX_GEMM_OP_NAMES = "/".join(
    spec.name for spec in FLEX_GEMM_OP_SPECS.values()
)
_PRESERVE_FLEX_GEMM_GEMM_OP = "preserve_flex_gemm_gemm_op"

# Note [Preserving FlexGEMM body GEMMs]
# FlexGEMM lowering materializes the captured epilogue by finding the body GEMM
# node whose target matches the HOP-carried gemm_op. Generic graph passes can
# rewrite batch-size-1 bmm into mm(...).unsqueeze(0), which removes the matching
# node. This body pass tags FlexGEMM GEMM nodes so bmm_to_mm in joint_graph.py
# skips them.


def mark_flex_gemm_body_gemm_node(
    body_graph: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> None:
    # [FUNCTION DOC] Sets a preservation flag in the node metadata of every call_function node in
    # body_graph whose target equals gemm_op, so the batch-1 bmm to mm rewrite skips them.
    # Mutates node metadata in place; returns nothing.
    """Mark body GEMMs so Inductor's batch-1 bmm rewrite keeps them matchable."""
    for node in body_graph.graph.find_nodes(op="call_function", target=gemm_op):
        node.meta[_PRESERVE_FLEX_GEMM_GEMM_OP] = True


FLEX_GEMM_BODY_GRAPH_PASSES: tuple[
    Callable[[torch.fx.GraphModule, torch._ops.OpOverload], None], ...
] = (mark_flex_gemm_body_gemm_node,)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("x",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def flex_gemm_fast_math_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Fast-math sigmoid: computes tanh of half the input, halves it and adds one half.
    # Integer inputs are promoted to float by the type promotion wrapper.
    """Use the tanh sigmoid identity selected by QUACK fast math."""
    return torch.tanh(x * 0.5) * 0.5 + 0.5


@elementwise_type_promotion_wrapper(
    type_promoting_args=("x",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def flex_gemm_fast_math_silu(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Fast-math SiLU: with half the input as h, returns h times tanh of h plus h, which
    # equals x times sigmoid of x. Uses default type promotion.
    """Use the tanh SiLU identity selected by QUACK fast math."""
    half = x * 0.5
    return half * torch.tanh(half) + half


def flex_gemm_fast_math_gelu(
    x: torch.Tensor,
    approximate: str = "none",
    *,
    fallback: Callable[..., Any],
) -> torch.Tensor:
    # [FUNCTION DOC] Fast-math GELU wrapper: if approximate is "none" it is switched to "tanh", then
    # the fallback (the original GELU decomposition) is called with the input and the
    # approximate mode. Returns the fallback result.
    """Select the standard tanh GELU approximation for exact GELU calls."""
    if approximate == "none":
        approximate = "tanh"
    return fallback(x, approximate)


FLEX_GEMM_FAST_MATH_DECOMPOSITIONS: dict[torch._ops.OpOverload, Callable[..., Any]] = {
    torch.ops.aten.sigmoid.default: flex_gemm_fast_math_sigmoid,
    torch.ops.aten.silu.default: flex_gemm_fast_math_silu,
}


def flex_gemm_body_decomposition_table(
    kernel_options: dict[str, Any],
    decomposition_table: Mapping[torch._ops.OpOverload, Callable[..., Any]],
) -> dict[torch._ops.OpOverload, Callable[..., Any]] | None:
    # [FUNCTION DOC] Returns None unless kernel_options selects the QUACK backend with fast_math set to
    # exactly true. Otherwise returns a copy of decomposition_table updated with the fast-math
    # sigmoid and SiLU decompositions, and, if the table already decomposes GELU, with that
    # entry wrapped so exact GELU becomes the tanh approximation. Input table is not mutated.
    """Override composite body decompositions enabled by QUACK fast math."""
    if (
        kernel_options.get("backend") != "QUACK"
        or kernel_options.get("fast_math") is not True
    ):
        return None
    merged_decompositions = dict(decomposition_table)
    merged_decompositions.update(FLEX_GEMM_FAST_MATH_DECOMPOSITIONS)
    gelu = torch.ops.aten.gelu.default
    if gelu in merged_decompositions:
        merged_decompositions[gelu] = partial(
            flex_gemm_fast_math_gelu, fallback=merged_decompositions[gelu]
        )
    return merged_decompositions


def nvfp4_pack(input: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Packs each trailing pair of float32 values into one E2M1x2 (FP4) byte and
    # returns a contiguous tensor viewed as float4_e2m1fn_x2. Raises ValueError unless the
    # input is float32, non-scalar, with last dimension of size 2. When compiling or on fake
    # tensors it emits inline PTX (high value first) via inline_asm_elementwise; in eager
    # mode it emulates the round-to-nearest-even satfinite conversion with integer bit
    # arithmetic (saturation at magnitude 6, denormal and normal ranges, sign bit), with
    # the second element in the high nibble.
    """Pack adjacent Float32 values into E2M1x2 storage for FlexGEMM tests.

    This private helper lets tests spell the intended epilogue with public inline
    assembly while a blessed user-facing quantization API is still being decided.
    The last dimension is one low/high pair: the PTX x2 conversion consumes both
    values and emits one byte, whose same-width dtype view supplies the logical
    ``torch.float4_e2m1fn_x2`` type without changing storage.
    """
    if input.dtype is not torch.float32 or input.ndim == 0 or input.shape[-1] != 2:
        raise ValueError(
            "NVFP4 pack input must be Float32 with innermost dimension 2, "
            f"got dtype={input.dtype}, shape={tuple(input.shape)}"
        )
    if torch.compiler.is_compiling() or is_fake(input):
        low = input[..., 0]
        high = input[..., 1]
        packed = inline_asm_elementwise(
            high,
            low,
            asm_str=(
                "{ .reg .b8 packed; "
                "cvt.rn.satfinite.e2m1x2.f32 packed, $1, $2; "
                "cvt.u32.u8 $0, packed; }"
            ),
            constraints="=r,r,r",
            dtype=torch.uint8,
        )
        return packed.contiguous().view(torch.float4_e2m1fn_x2)

    # Eager path is mostly for testing.
    input_bits = input.view(torch.int32)
    sign = input_bits & 0x80000000
    magnitude = (input_bits ^ sign).view(torch.float32)
    saturated = magnitude >= 6.0
    denormal = (~saturated) & (magnitude < 1.0)
    normal = ~(saturated | denormal)
    denormal_code = ((magnitude + 4194304.0).view(torch.int32) - 1249902592).to(
        torch.uint8
    )
    normal_bits = magnitude.view(torch.int32)
    mantissa_odd = (normal_bits >> 22) & 1
    normal_code = ((normal_bits - 1054867457 + mantissa_odd) >> 22).to(torch.uint8)
    codes = torch.full_like(magnitude, 7, dtype=torch.uint8)
    codes = torch.where(denormal, denormal_code, codes)
    codes = torch.where(normal, normal_code, codes)
    codes = codes | (((sign >> 28).to(torch.uint8)) & 8)
    packed = (codes[..., 1] << 4) | codes[..., 0]
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def apply_flex_gemm_body_graph_passes(
    body_graph: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> None:
    # [FUNCTION DOC] Runs each registered body-graph pass in FLEX_GEMM_BODY_GRAPH_PASSES on body_graph
    # for gemm_op, mutating the graph metadata in place.
    """Apply FlexGEMM body annotations before generic Inductor graph passes."""
    for graph_pass in FLEX_GEMM_BODY_GRAPH_PASSES:
        graph_pass(body_graph, gemm_op)


class FlexGemm(HigherOrderOperator):
    # [CLASS DOC] HigherOrderOperator named "flex_gemm". Stateless; __call__ validates and
    # normalizes its arguments (gemm op, body function, gemm args/kwargs, kernel options)
    # before dispatch. One singleton, flex_gemm_hop, exists.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the HigherOrderOperator with the name "flex_gemm".
        super().__init__("flex_gemm")

    def __call__(
        self,
        gemm_op: torch._ops.OpOverload,
        body_fn: Callable[..., Any],
        gemm_args: tuple[Any, ...],
        gemm_kwargs: dict[str, Any],
        kernel_options: dict[str, Any],
    ) -> Any:
        # [FUNCTION DOC] Validates and dispatches. Raises RuntimeError if gemm_op is not a supported overload,
        # gemm_args is not a tuple or list or contains anything but tensors, symbolic numbers
        # or plain numbers/bools, gemm_kwargs is not a dict or holds tensor values, or
        # kernel_options is not a dict or names a backend outside NVGEMM/QUACK/TRITON.
        # Converts gemm_args to a tuple, copies kernel_options and defaults backend to TRITON.
        # Returns the dispatched result (the epilogue output).
        if gemm_op not in FLEX_GEMM_OP_SPECS:
            raise RuntimeError(f"unsupported GEMM op for FlexGEMM: {gemm_op}")
        if not isinstance(gemm_args, (tuple, list)):
            raise RuntimeError(f"gemm_args must be a tuple/list, got {type(gemm_args)}")
        gemm_args = tuple(gemm_args)
        if not all(
            isinstance(
                arg,
                (
                    torch.Tensor,
                    torch.SymInt,
                    torch.SymFloat,
                    torch.SymBool,
                    int,
                    float,
                    bool,
                ),
            )
            for arg in gemm_args
        ):
            raise RuntimeError(
                "gemm_args must be a tuple of tensors, SymInts, SymFloats, "
                f"SymBools, ints, floats, or bools, got {gemm_args}"
            )
        if not isinstance(gemm_kwargs, dict):
            raise RuntimeError(f"gemm_kwargs must be a dict, got {type(gemm_kwargs)}")
        if any(
            isinstance(value, torch.Tensor) for value in pytree.tree_leaves(gemm_kwargs)
        ):
            raise RuntimeError(
                "gemm_kwargs must not contain tensor values; pass tensor inputs through gemm_args"
            )
        if not isinstance(kernel_options, dict):
            raise RuntimeError(
                f"kernel_options must be a dict, got {type(kernel_options)}"
            )

        kernel_options = dict(kernel_options)
        kernel_options.setdefault("backend", "TRITON")
        backend = kernel_options["backend"]
        if backend not in _SUPPORTED_BACKENDS:
            raise RuntimeError(
                f"unsupported FlexGEMM backend: {backend}; "
                f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
            )
        return super().__call__(
            gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options
        )


flex_gemm_hop = FlexGemm()


def flex_gemm(
    gemm_op: Callable[..., Any],
    gemm_args: tuple[Any, ...],
    epilogue_fn: Callable[[Any], Any],
    *,
    gemm_kwargs: dict[str, Any] | None = None,
    kernel_options: dict[str, Any] | None = None,
) -> Any:
    # [FUNCTION DOC] Public API: runs gemm_op (an overload or its torch alias such as torch.mm) on
    # gemm_args with optional gemm_kwargs, then applies epilogue_fn to the GEMM result, as
    # one fused HOP call configured by kernel_options. Missing kwargs/options default to empty
    # dicts. Returns the epilogue output. Errors come from the HOP validation.
    if gemm_kwargs is None:
        gemm_kwargs = {}
    if kernel_options is None:
        kernel_options = {}
    gemm_op = cast(torch._ops.OpOverload, FLEX_GEMM_OP_ALIASES.get(gemm_op, gemm_op))

    def body_fn(*args: Any) -> Any:
        # [FUNCTION DOC] HELPER: Body of the HOP: calls the GEMM op on the positional args with the captured
        # gemm_kwargs and passes the result to epilogue_fn, returning its output.
        # Keep the traced body positional-only; the HOP carries gemm_kwargs for lowering.
        return epilogue_fn(gemm_op(*args, **gemm_kwargs))

    return flex_gemm_hop(gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options)


@flex_gemm_hop.py_impl(DispatchKey.CompositeExplicitAutograd)
def flex_gemm_dense(gemm_op, body_fn, args, kwargs, kernel_options):
    # [FUNCTION DOC] HELPER: Eager implementation: runs body_fn on args and returns its result; the
    # other parameters are ignored.
    return body_fn(*args)


flex_gemm_hop.py_autograd_impl(
    autograd_not_implemented(flex_gemm_hop, deferred_error=True)
)


@register_fake(flex_gemm_hop)
def flex_gemm_fake_tensor_mode(gemm_op, body_fn, args, kwargs, kernel_options):
    # [FUNCTION DOC] HELPER: Fake tensor rule: runs body_fn on args and returns its (fake) result.
    return body_fn(*args)


@flex_gemm_hop.py_functionalize_impl
def flex_gemm_functionalize(ctx, gemm_op, body_fn, args, kwargs, kernel_options):
    # [FUNCTION DOC] HELPER: Functionalization rule. Unwraps the args, then under redispatch checks that
    # body_fn neither aliases nor mutates inputs (raises on violation), re-invokes the HOP
    # with a functionalized body and unwrapped kwargs/options, and re-wraps the outputs.
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        _check_alias_and_mutation(
            body_fn,
            unwrapped_args,
            "flex_gemm",
            hasattr(ctx, "mode") and ctx.mode.pre_dispatch,
        )
        return ctx.wrap_tensors(
            flex_gemm_hop(
                gemm_op,
                ctx.functionalize(body_fn),
                unwrapped_args,
                ctx.unwrap_tensors(kwargs),
                ctx.unwrap_tensors(kernel_options),
            )
        )


@flex_gemm_hop.py_impl(ProxyTorchDispatchMode)
def flex_gemm_proxy_torch_dispatch_mode(
    proxy_mode, gemm_op, body_fn, args, kwargs, kernel_options
):
    # [FUNCTION DOC] HELPER: Proxy tracing rule. When tracing is enabled: traces body_fn into a sub-graph
    # (using a decomposition table adjusted for QUACK fast math), applies body-graph
    # annotation passes, registers it under a unique "flex_gemm_body_graph" name, creates a
    # call_function node with (gemm_op, graph, args, kwargs, options), and returns the real
    # body outputs tracked to that node. When tracing is disabled, re-invokes the HOP directly.
    if proxy_mode.enable_tracing:
        flat_args = tuple(args)

        def tracing_body_fn(*flat_body_args):
            # [FUNCTION DOC] HELPER: Thin wrapper that calls body_fn on the given positional args.
            return body_fn(*flat_body_args)

        body_graph = reenter_make_fx(
            tracing_body_fn,
            subgraph_decomp_table=flex_gemm_body_decomposition_table(
                kernel_options, proxy_mode.decomposition_table
            ),
        )(*flat_args)
        apply_flex_gemm_body_graph_passes(body_graph, gemm_op)
        _, body_graph_name = unique_graph_id(proxy_mode, prefix="flex_gemm_body_graph")
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)
        proxy_args = pytree.tree_map(
            proxy_mode.tracer.unwrap_proxy,
            (gemm_op, body_graph, flat_args, kwargs, kernel_options),
        )
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function",
            flex_gemm_hop,
            proxy_args,
            {},
            name="flex_gemm",
        )
        return track_tensor_tree(
            body_fn(*flat_args),
            out_proxy,
            constant=None,
            tracer=proxy_mode.tracer,
        )
    return flex_gemm_hop(gemm_op, body_fn, args, kwargs, kernel_options)
