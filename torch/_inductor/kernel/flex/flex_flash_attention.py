# mypy: allow-untyped-defs
# [MODULE DOC] Flex attention backend that lowers to the CuTe DSL flash-attention-4 (FA4) kernels.
# Provides config selection (vector widths, packed mask intervals), availability checks for the optional
# flash_attn.cute package, a CuTe-friendly hierarchical indexer patch for FixedLayout, and the forward and
# backward kernel builders that create CuteDSL template choices and autotune them.
# Import-time side effects: creates the forward and backward CuteDSLTemplate objects (loading template sources)
# and binds the aten and prims op namespaces.
"""Call into flash-attention 4 for flexattention"""

import dataclasses
import functools
import importlib
import inspect
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, cast, Literal

import sympy
from sympy import Expr, Integer

import torch
from torch.fx import GraphModule

from ...codegen.cutedsl.aux_scalars import CuteDSLAuxScalarBindings
from ...ir import FixedLayout, ShapeAsConstantBuffer, Subgraph, TensorBox
from ...lowering import empty_strided
from ...select_algorithm import autotune_select_algorithm
from ...virtualized import V
from .aux_vectorization import (
    DEFAULT_MASK_MOD_VEC_SIZE,
    select_mask_mod_vec_size,
    select_score_mod_vec_size,
)
from .common import (
    create_indices_fake,
    create_num_blocks_fake_generator,
    infer_dense_strides,
    load_flex_template,
    SubgraphResults,
)
from .interval_mask_packing import PackedMaskInterval, select_packed_mask_intervals


@dataclasses.dataclass
class FlexFlashConfig:
    # [CLASS DOC] Autotuning configuration for one CuteDSL flex flash attention kernel candidate.
    # Fields: score_mod_vec_size (elements per thread for score_mod, None = kernel default), mask_mod_vec_size
    # (consecutive KV lanes per mask_mod call), mask_mod_packed_intervals (precomputed 32-lane packed mask
    # intervals or None). Plain mutable dataclass with no invariants beyond the defaults being None.
    """Autotuning configuration for CuteDSL flex flash attention kernels.

    score_mod_vec_size: Number of elements processed per thread in the score_mod
        application loop. Maps to score_mod.__vec_size__ in CuTe flash attention.
        None uses the kernel default. Only effective for forward; backward does
        not currently support vectorized score_mod.
    mask_mod_vec_size: Number of consecutive KV lanes evaluated per mask_mod
        call. Maps to mask_mod.__vec_size__ in CuTe flash attention and to
        the direct captured-tensor vector-load width for mask_mod.
    mask_mod_packed_intervals: Precomputed 32-lane packed mask intervals.
    """

    score_mod_vec_size: int | None = None
    mask_mod_vec_size: int | None = None
    mask_mod_packed_intervals: tuple[PackedMaskInterval, ...] | None = None


def collect_aux_scalar_symbols(
    *buffer_groups: Sequence[Any],
) -> tuple[sympy.Symbol, ...]:
    # [FUNCTION DOC] HELPER: Collect the distinct sympy symbols appearing in captured aux buffers.
    # Takes any number of groups of buffers; only entries that are sympy expressions (scalar captures) are inspected.
    # Symbols inside each expression are visited in name order; the result is a tuple in first-seen order without
    # duplicates. Pure; raises nothing.
    symbols: dict[sympy.Symbol, None] = {}
    for buffers in buffer_groups:
        for buffer in buffers:
            if isinstance(buffer, sympy.Expr):
                for symbol in sorted(buffer.free_symbols, key=lambda s: s.name):
                    symbols.setdefault(symbol, None)
    return tuple(symbols)


def get_flex_flash_fwd_configs(
    has_score_mod: bool,
    has_aux_tensors: bool,
    device: torch.device | None = None,
    score_mod_graph_module: GraphModule | None = None,
    score_mod_other_buffers: Sequence[TensorBox] = (),
    has_mask_mod: bool = False,
    has_mask_aux_tensors: bool = False,
    mask_mod_graph_module: GraphModule | None = None,
    mask_mod_other_buffers: Sequence[TensorBox] = (),
    aux_scalar_symbols: Sequence[sympy.Symbol] = (),
) -> list[FlexFlashConfig]:
    # [FUNCTION DOC] Build the list of FlexFlashConfig candidates for the forward kernel.
    # Queries the CUDA compute capability (only when CUDA is available and a mask_mod or aux-tensor score_mod exists)
    # to pick mask_mod and score_mod vector widths via the aux_vectorization selectors. On SM100/SM110 with a mask_mod
    # graph it also tries to compute packed mask intervals, and when found forces the default mask vector size.
    # Under max_autotune with an unconstrained score_mod it emits one config per power-of-two width from 1 to 128;
    # otherwise a single config. The list is truncated to the test config max_flex_configs when set. Reads global
    # config only; no mutation.
    cuda_major = None
    if torch.cuda.is_available() and (
        has_mask_mod or (has_score_mod and has_aux_tensors)
    ):
        device_index = None if device is None else device.index
        cuda_major = torch.cuda.get_device_capability(device_index)[0]
    mask_mod_vec_size = select_mask_mod_vec_size(
        has_mask_mod=has_mask_mod,
        has_mask_aux_tensors=has_mask_aux_tensors,
        supports_mask_mod_vec=cuda_major in (10, 11),
        graph_module=mask_mod_graph_module,
        other_buffers=mask_mod_other_buffers,
    )
    score_mod_vec_size = select_score_mod_vec_size(
        has_score_mod=has_score_mod,
        has_aux_tensors=has_aux_tensors,
        is_sm100_or_later=cuda_major is not None and cuda_major >= 10,
        graph_module=score_mod_graph_module,
        other_buffers=score_mod_other_buffers,
    )
    mask_mod_packed_intervals = None
    if has_mask_mod and cuda_major in (10, 11) and mask_mod_graph_module is not None:
        mask_mod_packed_intervals = select_packed_mask_intervals(
            mask_mod_graph_module,
            mask_mod_other_buffers,
            CuteDSLAuxScalarBindings(tuple(aux_scalar_symbols)).symbol_codes(),
        )
    if mask_mod_packed_intervals is not None:
        mask_mod_vec_size = DEFAULT_MASK_MOD_VEC_SIZE

    if (
        has_score_mod
        and score_mod_vec_size is None
        and torch._inductor.config.max_autotune
    ):
        # None means no captured tensor load constrained the score_mod vector width.
        # Scalar captures are lane-uniform aux_scalars, so any kernel-supported
        # power-of-two vector width is legal.
        score_mod_vec_sizes = (1, 2, 4, 8, 16, 32, 64, 128)
    else:
        score_mod_vec_sizes = (score_mod_vec_size,)
    configs = [
        FlexFlashConfig(
            score_mod_vec_size=v,
            mask_mod_vec_size=mask_mod_vec_size,
            mask_mod_packed_intervals=mask_mod_packed_intervals,
        )
        for v in score_mod_vec_sizes
    ]
    max_configs = torch._inductor.config.test_configs.max_flex_configs
    if max_configs is not None and len(configs) > max_configs:
        configs = configs[:max_configs]
    return configs


def _get_flex_flash_bwd_configs() -> list[FlexFlashConfig]:
    # [FUNCTION DOC] HELPER: Return the single default (all-None) config used for the backward kernel.
    return [FlexFlashConfig()]


aten = torch.ops.aten
prims = torch.ops.prims

FLASH_ATTENTION_INSTALL_MESSAGE = (
    "Install a compatible Flash Attention package, for example "
    '`pip install --pre flash-attn-4` (`pip install --pre "flash-attn-4[cu13]"` '
    "for CUDA 13), and see https://pypi.org/project/flash-attn-4/ "
    "for PyPI packaging details."
)


def _flash_attention_unavailable_message() -> str:
    # [FUNCTION DOC] HELPER: Build the error message used when flash_attn.cute is not importable (includes install hint).
    return (
        "CUTE flash attention library is not available. "
        f"{FLASH_ATTENTION_INSTALL_MESSAGE}"
    )


@functools.lru_cache(maxsize=1)
def ensure_flash_available() -> bool:
    # [FUNCTION DOC] Return True if the flash_attn.cute package can be found (spec lookup only, no import of it).
    # Result is cached (single entry); call cache_clear to retry after installing. Returns False on ImportError.
    """Check if flash-attn is importable; cache the result for reuse.

    Call ensure_flash_available.cache_clear() after installing flash-attn
    in the same interpreter to retry the import.
    """
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None  # type: ignore[attr-defined]
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def flash_supports_aux_scalars() -> bool:
    # [FUNCTION DOC] Return True if the installed FA4 forward and backward entry points both accept an aux_scalars
    # parameter (needed for scalar captures). Imports flash_attn.cute.interface and inspects signatures; returns False
    # if the import fails. Result is cached.
    """Check whether the installed FA4 package supports scalar captures."""
    try:
        interface = importlib.import_module("flash_attn.cute.interface")
    except ImportError:
        return False
    return (
        "aux_scalars" in inspect.signature(interface._flash_attn_fwd).parameters
        and "aux_scalars" in inspect.signature(interface._flash_attn_bwd).parameters
    )


from ...codegen.cutedsl.cutedsl_template import CuteDSLTemplate


flash_attention_cutedsl_template = CuteDSLTemplate(
    name="flash_attention_cutedsl", source=load_flex_template("flash_attention")
)
flash_attention_backward_cutedsl_template = CuteDSLTemplate(
    name="flash_attention_backward_cutedsl",
    source=load_flex_template("flash_attention_backward"),
)


class HierarchicalIndex(sympy.Function):
    # [CLASS DOC] Inert sympy Function node that carries an N-dimensional index tuple through Inductor IR.
    # It never simplifies (eval returns nothing), so CuteDSL codegen can recover per-dimension coordinates from its args
    # instead of a flattened offset. Short-lived; only interpreted by the CuteDSL kernel codegen.
    """
    Inert wrapper to carry an N-D index tuple through Inductor's SymPy-based IR.

    Inductor generally represents a tensor index as a single `sympy.Expr` (often a
    flattened linear offset in memory). CuteDSL, however, wants structured coordinates so it
    can emit `tensor[i, j, ...]` and handle strides internally. We therefore wrap
    the per-dimension indices in a `sympy.Function` node: this keeps the value a
    `sympy.Expr` for existing substitution/CSE machinery, while letting CuteDSL
    codegen pattern-match and unpack the coordinates via `index.args`.

    `eval()` returns None to keep the node inert (no simplification/flattening).

    These nodes are intended to be short-lived wrappers and are only interpreted by
    CuteDSL codegen (see `ModificationWrapperCuteDSL.load` in
    `torch/_inductor/codegen/cutedsl/cutedsl_kernel.py`).
    """

    @classmethod
    def eval(cls, *args):
        # [FUNCTION DOC] HELPER: Sympy eval hook that always returns None so the node stays unevaluated.
        return None


def _hierarchical_indexer_cute(
    size: Sequence[int],
    stride: Sequence[int] | None = None,
    offset: Expr = Integer(0),
) -> Callable[[Sequence[Expr]], Expr]:
    # [FUNCTION DOC] HELPER: Return an indexer function that preserves multi-dimensional indices for CuteDSL.
    # Takes tensor size, optional stride and offset (stride is unused). The returned indexer rejects a non-zero offset
    # or a rank mismatch (AssertionError), returns zero for no indices, the index itself for one dimension, and a
    # HierarchicalIndex wrapping all indices otherwise.
    """Return an indexer that preserves multi-dimensional indices for CuteDSL."""

    def indexer(indices: Sequence[Expr]) -> Expr:
        # [FUNCTION DOC] HELPER: Map a list of per-dimension indices to a single index expression (see enclosing function).
        # Raises AssertionError if the captured offset is non-zero or the number of indices differs from the tensor rank.
        if offset != Integer(0):
            raise AssertionError("Offset not supported for hierarchical indexing")
        if len(indices) != len(size):
            raise AssertionError(
                f"Rank mismatch: got {len(indices)} indices for tensor of rank {len(size)}"
            )
        if not indices:
            return Integer(0)
        if len(indices) == 1:
            return indices[0]
        return HierarchicalIndex(*indices)

    return indexer


@contextmanager
def patch_fixed_layout_indexer_for_cutedsl():
    # [FUNCTION DOC] Context manager that temporarily replaces FixedLayout.make_indexer with the hierarchical indexer.
    # Mutates the class attribute for the duration of the block (global, not thread safe) and always restores the
    # original on exit. Safe only because flex flash attention emits pointwise reads and no stores.
    """
    Temporarily swap FixedLayout.make_indexer so CuteDSL sees hierarchical indexing.

    Note [CuteDSL indexer patch]:
    Flex flash attention only supports a limited set of IR ops (pointwise, reads, no stores),
    so temporarily changing the indexing behavior is safe for the kernels we emit today.
    TODO(dynamic shapes): Reconfirm once flex flash attention supports dynamic shapes.
    """
    original_make_indexer = FixedLayout.make_indexer

    def cutedsl_make_indexer(self):
    # [FUNCTION DOC] HELPER: Replacement make_indexer that builds the hierarchical indexer from the layout's size, stride, offset.
        return _hierarchical_indexer_cute(self.size, self.stride, self.offset)

    FixedLayout.make_indexer = cutedsl_make_indexer  # type: ignore[assignment]
    try:
        yield
    finally:
        FixedLayout.make_indexer = original_make_indexer  # type: ignore[assignment]


def wrap_choice_render_with_cutedsl_indexer(choice: Any) -> None:
    # [FUNCTION DOC] Wrap a template choice's make_kernel_render so rendering runs under the CuteDSL indexer patch.
    # Mutates the choice in place (replaces its make_kernel_render attribute); returns None.
    """
    Wrap a template choice's kernel render to apply CuteDSL indexer patching.

    See Note [CuteDSL indexer patch]:
    CuteDSL handles tensor strides internally, so template rendering must use
    hierarchical indexing.
    """
    original_make_kernel_render = choice.make_kernel_render

    def make_kernel_render_with_patch(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Call the original make_kernel_render and return its render_kernel plus a patched render callable.
        render_kernel, render = original_make_kernel_render(*args, **kwargs)

        def render_with_patch():
        # [FUNCTION DOC] HELPER: Run the original render with the CuteDSL indexer patch active and return its result.
            with patch_fixed_layout_indexer_for_cutedsl():
                return render()

        return render_kernel, render_with_patch

    choice.make_kernel_render = make_kernel_render_with_patch


def input_buffers_require_grads(graph_module, num_score_mod_placeholders: int):
    # [FUNCTION DOC] Return True if any graph input beyond the first num_score_mod_placeholders requires gradients.
    # Walks the placeholders of the graph module; the extra ones are captured buffers, whose requires_grad flag comes
    # from tensor_meta metadata (missing metadata counts as False). Returns False when there are no extra inputs.
    """Check if any of the input buffers (beyond the score mod placeholders) require gradients."""
    inputs = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node)
    if len(inputs) <= num_score_mod_placeholders:
        return False

    def requires_grad(n):
    # [FUNCTION DOC] HELPER: Return the requires_grad flag from a node's tensor_meta, or False if absent.
        tensor_meta = n.meta.get("tensor_meta")
        return tensor_meta.requires_grad if tensor_meta is not None else False

    return any(requires_grad(n) for n in inputs[num_score_mod_placeholders:])


def is_trivial_score_graph(graph_module: GraphModule) -> bool:
    # [FUNCTION DOC] Return True if the score_mod graph is the identity (output is the first placeholder).
    # Raises AssertionError if the graph does not have exactly one output node.
    """Backwards currently doesn't support score_mods, match against identity"""
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    if len(output) != 1:
        raise AssertionError("Got graph w/ multiple outputs")
    output_val = output[0].args[0]
    # The identity graph just sends the score straight through
    return output_val == placeholders[0]


def is_trivial_mask_graph(graph_module: GraphModule) -> bool:
    # [FUNCTION DOC] Return True if the mask_mod graph is trivial: exactly four placeholders and an output produced by
    # aten.full.default. Raises AssertionError if there is not exactly one output node.
    # KNOWN ISSUE: does not check the fill value, so a full-False mask would also be treated as trivial.
    """Mask graph is trivial when it only gates via the default full op."""
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = [n for n in nodes if n.op == "output"]
    if len(output) != 1:
        raise AssertionError("Got graph w/ multiple outputs")
    output_val = output[0].args[0]

    # mask mod graph is empty if we have 4 inputs and full_default output
    return len(placeholders) == 4 and output_val.target is torch.ops.aten.full.default


def has_unsupported_cpu_scalar_tensor_captures(
    score_mod_other_buffers: Sequence[Any],
    mask_mod_other_buffers: Sequence[Any],
) -> bool:
    # [FUNCTION DOC] Return True if any captured buffer is a TensorBox on CPU with zero dimensions.
    # Such captures are not supported until scalarized. Inspects both buffer lists; pure.
    """Return True for CPU 0-d tensor captures that need scalarization first."""
    for buf in list(score_mod_other_buffers) + list(mask_mod_other_buffers):
        if isinstance(buf, TensorBox):
            device = buf.get_device()
            size = buf.get_size()
            if device is not None and device.type == "cpu" and len(size) == 0:
                return True
    return False


def _can_use_flex_flash_attention(
    subgraph: Subgraph,
    mask_graph: Subgraph,
    num_score_mod_placeholders: int,
) -> tuple[bool, str]:
    # [FUNCTION DOC] Check whether flex flash attention (forward) can run.
    # Returns (can_use, reason): False with an install message when flash_attn.cute is missing, False when captured
    # inputs of the score_mod graph require gradients, otherwise (True, empty string).
    # KNOWN ISSUE: the mask_graph argument is accepted but never used.
    """Check if flex flash attention can be used for the given inputs.

    Returns:
        tuple: (can_use, reason) where reason explains why it can't be used if can_use is False
    """
    if not ensure_flash_available():
        return False, _flash_attention_unavailable_message()

    if input_buffers_require_grads(subgraph.graph_module, num_score_mod_placeholders):
        return (
            False,
            "Input buffers require gradients (not supported by flash attention)",
        )

    return True, ""


def _use_flex_flash_attention(
    subgraph: Subgraph,
    mask_graph: Subgraph,
    kernel_options: dict[str, Any],
    num_score_mod_placeholders: int,
    backend: Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE"],
) -> bool:
    # [FUNCTION DOC] Decide whether the FLASH backend is used for the forward pass.
    # Returns False unless backend is FLASH. For FLASH, raises RuntimeError with the reason if
    # _can_use_flex_flash_attention fails; returns True otherwise.
    # KNOWN ISSUE: kernel_options is accepted but unused.
    """Determine if we should use flex flash attention for the given inputs.

    Args:
        subgraph: The score modification subgraph
        mask_graph: The mask modification subgraph
        kernel_options: Kernel configuration options
        num_score_mod_placeholders: Number of placeholders in score_mod
        backend: Implementation selector (AUTO, TRITON, FLASH, TRITON_DECODE)

    Returns:
        True if flash attention should be used, False otherwise
    """
    # Flash is experimental and must be explicitly requested
    if backend != "FLASH":
        return False

    can_use, reason = _can_use_flex_flash_attention(
        subgraph,
        mask_graph,
        num_score_mod_placeholders,
    )

    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLASH' but flash attention cannot be used: {reason}"
        )

    return True


def create_flex_flash_attention_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    block_mask: tuple[Any, ...],
    scale: float,
    kernel_options: dict[str, Any],
    subgraph_buffer: SubgraphResults,
    mask_graph_buffer: SubgraphResults,
    score_mod_other_buffers: list[TensorBox],
    mask_mod_other_buffers: list[TensorBox],
    kv_num_blocks: TensorBox | None,
    kv_indices: TensorBox | None,
    full_kv_num_blocks: TensorBox | None,
    full_kv_indices: TensorBox | None,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    mask_graph: Subgraph,
    subgraph: Subgraph | None = None,
) -> tuple[TensorBox, TensorBox]:
    # [FUNCTION DOC] Build and autotune the CuteDSL flash attention forward kernel.
    # Inputs: query/key/value, block mask pieces (kv_num_blocks, kv_indices and optional full-block versions), scale,
    # score_mod and mask_mod subgraph buffers and captured buffers, sparse block sizes, graphs. Validates equal dtypes
    # (ValueError) and flash availability (RuntimeError), allocates output (dense strides matched to query) and a
    # float32 logsumexp buffer, decides which of score_mod, mask_mod and block mask are needed, and checks aux scalar
    # support. For each config it appends a template choice (under the indexer patch); errors from the template raise
    # RuntimeError when only one config exists or no choice results. Then it builds fake input generators for block
    # metadata and selects the best choice with autotune_select_algorithm.
    # Returns (output, logsumexp). Side effects: guards sparse block sizes as ints, registers choices in the graph.
    # Note: logsumexp is passed as a mutated input of the template.
    """Create a flex flash attention kernel using CuteDSL template."""
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Mixed query, key, and value dtype is not supported on this platform, "
            f"got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype}."
        )
    if not ensure_flash_available():
        raise RuntimeError(_flash_attention_unavailable_message())

    # Get dimensions
    batch_size, num_heads, seq_len_q, head_dim = query.get_size()
    v_head_dim = value.get_size()[-1]
    device = query.get_device()
    dtype = query.get_dtype()
    if device is None:
        raise AssertionError("Device must be specified")

    # Match stride pattern from query tensor
    q_strides = query.get_stride()
    out_size = [batch_size, num_heads, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, q_strides)

    output = empty_strided(
        size=out_size,
        stride=out_strides,
        dtype=dtype,
        device=device,
    )

    lse = empty_strided(
        size=[batch_size, num_heads, seq_len_q],
        stride=None,  # LSE can be contiguous
        dtype=torch.float32,  # LSE is always fp32
        device=device,
    )

    # Create layout for primary output
    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=[batch_size, num_heads, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in output.get_stride()],
    )

    mask_graph_is_trivial = is_trivial_mask_graph(mask_graph.graph_module)
    score_graph_is_trivial = subgraph is None or is_trivial_score_graph(
        subgraph.graph_module
    )

    if kv_num_blocks is None or kv_indices is None:
        raise AssertionError("kv block metadata is required for flex FLASH")

    has_score_mod = not score_graph_is_trivial
    has_mask_mod = not mask_graph_is_trivial
    has_full_blocks = full_kv_num_blocks is not None
    if has_full_blocks and full_kv_indices is None:
        raise AssertionError("full_kv_indices must be provided with full_kv_num_blocks")
    is_noop_block_mask = not has_full_blocks and V.graph.sizevars.statically_known_true(
        sympy.And(
            sympy.Eq(sparse_q_block_size, 1 << 30),
            sympy.Eq(sparse_kv_block_size, 1 << 30),
        )
    )
    needs_block_mask = not (mask_graph_is_trivial and is_noop_block_mask)
    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)

    choices: list[Any] = []
    if flash_attention_cutedsl_template is None:
        raise AssertionError("flash_attention_cutedsl_template must not be None")

    input_nodes = [query, key, value, lse]
    if needs_block_mask:
        input_nodes.extend([kv_num_blocks, kv_indices])
        if has_full_blocks:
            input_nodes.extend([full_kv_num_blocks, full_kv_indices])

    subgraphs = []
    if has_score_mod:
        subgraphs.append(subgraph_buffer)
    if has_mask_mod:
        subgraphs.append(mask_graph_buffer)

    aux_scalar_symbols = collect_aux_scalar_symbols(
        score_mod_other_buffers, mask_mod_other_buffers
    )
    if aux_scalar_symbols and not flash_supports_aux_scalars():
        raise RuntimeError(
            "CUTE flash attention scalar captures require flash-attn-4>=4.0.0b17. "
            f"{FLASH_ATTENTION_INSTALL_MESSAGE}"
        )
    has_score_aux_tensors = any(
        not isinstance(buffer, sympy.Expr) for buffer in score_mod_other_buffers
    )
    has_mask_aux_tensors = any(
        not isinstance(buffer, sympy.Expr) for buffer in mask_mod_other_buffers
    )

    configs = get_flex_flash_fwd_configs(
        has_score_mod=has_score_mod,
        has_aux_tensors=has_score_aux_tensors,
        device=device,
        score_mod_graph_module=(
            subgraph.graph_module if has_score_mod and subgraph is not None else None
        ),
        score_mod_other_buffers=score_mod_other_buffers,
        has_mask_mod=has_mask_mod,
        has_mask_aux_tensors=has_mask_aux_tensors,
        mask_mod_graph_module=mask_graph.graph_module,
        mask_mod_other_buffers=mask_mod_other_buffers,
        aux_scalar_symbols=aux_scalar_symbols,
    )
    error: NotImplementedError | None = None
    for conf in configs:
        with patch_fixed_layout_indexer_for_cutedsl():
            error = flash_attention_cutedsl_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=output_layout,
                mutated_inputs=[lse],
                subgraphs=subgraphs,
                SM_SCALE=scale,
                HAS_SCORE_MOD=has_score_mod,
                SCORE_MOD_VEC_SIZE=conf.score_mod_vec_size,
                MASK_MOD_VEC_SIZE=conf.mask_mod_vec_size,
                MASK_MOD_PACKED_INTERVALS=conf.mask_mod_packed_intervals,
                MASK_MOD_OTHER_BUFFERS=mask_mod_other_buffers,
                AUX_SCALAR_SYMBOLS=aux_scalar_symbols,
                NEEDS_BLOCK_MASK=needs_block_mask,
                HAS_MASK_MOD=has_mask_mod,
                HAS_FULL_BLOCKS=has_full_blocks,
                SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
                SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
            )
        if error is not None and len(configs) == 1:
            raise RuntimeError(f"CuteDSL template failed: {error}")

    for choice in choices:
        wrap_choice_render_with_cutedsl_indexer(choice)

    if not choices:
        raise RuntimeError(f"CuteDSL template failed: {error}")

    input_gen_fns: dict[int, Callable] | None = None
    if needs_block_mask:
        input_gen_fns = {
            4: create_num_blocks_fake_generator(kv_indices),
            5: create_indices_fake,
        }
        if has_full_blocks:
            input_gen_fns.update(
                {
                    6: create_num_blocks_fake_generator(full_kv_indices),
                    7: create_indices_fake,
                }
            )

    template_output, _ = autotune_select_algorithm(
        "flex_flash_attention",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, lse)


def _can_use_flex_flash_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    joint_outputs: Any | None = None,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
    num_score_mod_placeholders: int = 5,
) -> tuple[bool, str]:
    # [FUNCTION DOC] HELPER: Check whether flex flash attention backward can run.
    # Returns (can_use, reason): False when flash is missing, when captured inputs require grads, or when joint outputs
    # include captured grads or mutated grads (not yet supported); otherwise (True, empty string).
    # KNOWN ISSUE: mask_graph and score_mod_other_buffers are unused.
    if not ensure_flash_available():
        return False, _flash_attention_unavailable_message()

    if input_buffers_require_grads(
        fw_subgraph.graph_module, num_score_mod_placeholders
    ):
        return (
            False,
            "Input buffers require gradients (not supported by flash attention backward)",
        )

    if joint_outputs is not None:
        if joint_outputs.captured_grads_compute:
            return (
                False,
                "NYI: Flex Flash Attention bwd doesn't support captured grads yet.",
            )
        if joint_outputs.mutated_grads:
            return (
                False,
                "NYI: Flex Flash Attention bwd doesn't support mutated grads yet.",
            )

    return True, ""


def _use_flex_flash_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    backend: Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE"],
    joint_outputs: Any | None = None,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
) -> bool:
    # [FUNCTION DOC] Decide whether the FLASH backend is used for the backward pass.
    # Returns False unless backend is FLASH; raises RuntimeError with the reason if the backward cannot be used.
    # KNOWN ISSUE: does not forward a placeholder count, so the helper's default of 5 is always used.
    """Determine if we should use flex flash attention for the given inputs.

    Args:
        fw_subgraph: The forward score modification subgraph
        mask_graph: The mask modification subgraph
        backend: Implementation selector (AUTO, TRITON, FLASH, TRITON_DECODE)
        joint_outputs: Processed joint outputs (for PR1 constraint checking)
        score_mod_other_buffers: Additional buffers used by score_mod

    Returns:
        True if flash attention should be used, False otherwise
    """
    # Flash is experimental and must be explicitly requested
    if backend != "FLASH":
        return False

    can_use, reason = _can_use_flex_flash_attention_backward(
        fw_subgraph,
        mask_graph,
        joint_outputs,
        score_mod_other_buffers,
    )

    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLASH' but flash attention cannot be used: {reason}"
        )

    return True


def create_flex_flash_attention_backward_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    out: TensorBox,
    logsumexp: TensorBox,
    grad_out: TensorBox,
    grad_logsumexp: TensorBox | None,
    scale: float,
    kernel_options: dict[str, Any],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    fw_subgraph_buffer: SubgraphResults | None = None,
    joint_subgraph_buffer: Any | None = None,
    score_mod_other_buffers: list[TensorBox] | None = None,
    mask_graph_buffer: SubgraphResults | None = None,
    mask_mod_other_buffers: list[TensorBox] | None = None,
    q_num_blocks: TensorBox | None = None,
    q_indices: TensorBox | None = None,
    full_q_num_blocks: TensorBox | None = None,
    full_q_indices: TensorBox | None = None,
    dq_write_order: TensorBox | None = None,
    dq_write_order_full: TensorBox | None = None,
    dq_kv_order: TensorBox | None = None,
    dq_kv_order_spt: bool | None = None,
) -> tuple[TensorBox | ShapeAsConstantBuffer, TensorBox, TensorBox, tuple]:
    # [FUNCTION DOC] Build and autotune the CuteDSL flash attention backward kernel.
    # Allocates grad_query/grad_key/grad_value buffers with strides matched to the inputs, uses the dq layout as the
    # template output layout, and assembles input nodes (q, k, v, out, grad_out, logsumexp, grad buffers, then block
    # mask metadata, dq write-order/kv-order tensors and optional grad_logsumexp). Checks capability of the installed
    # flash_attn (dq_kv_order / spt support) and raises NotImplementedError when requested features are missing, plus
    # NotImplementedError for dLSE with head_dim 256 on SM100/110. Handles deterministic mode: with block mask it needs
    # SM100+, dq write-order and kv-order metadata, else raises (NotImplementedError / ValueError) or, in warn-only mode,
    # disables determinism and emits a warning. Appends template choices and autotunes them.
    # Returns (dq output, grad_key, grad_value, empty tuple); grad_key/grad_value are mutated inputs of the template.
    # Raises RuntimeError when flash is unavailable, aux scalars unsupported, or no template choice could be built.
    # KNOWN ISSUE: grad_key is sized with the query head_dim, which assumes key and query share head_dim.
    """Create a CuteDSL flash attention backward kernel for the default mod path."""
    if not ensure_flash_available():
        raise RuntimeError(_flash_attention_unavailable_message())

    batch_size, num_heads, seq_len_q, head_dim = query.get_size()
    _, num_heads_kv, seq_len_kv, v_head_dim = value.get_size()
    device = query.get_device()
    dtype = query.get_dtype()
    if device is None:
        raise AssertionError("Device must not be None")

    grad_query_strides = infer_dense_strides(
        [batch_size, num_heads, seq_len_q, head_dim], query.get_stride()
    )
    grad_query = empty_strided(
        size=[batch_size, num_heads, seq_len_q, head_dim],
        stride=grad_query_strides,
        dtype=dtype,
        device=device,
    )

    grad_key_strides = infer_dense_strides(
        [batch_size, num_heads_kv, seq_len_kv, head_dim], key.get_stride()
    )
    grad_key = empty_strided(
        size=[batch_size, num_heads_kv, seq_len_kv, head_dim],
        stride=grad_key_strides,
        dtype=dtype,
        device=device,
    )

    grad_value_strides = infer_dense_strides(
        [batch_size, num_heads_kv, seq_len_kv, v_head_dim], value.get_stride()
    )
    grad_value = empty_strided(
        size=[batch_size, num_heads_kv, seq_len_kv, v_head_dim],
        stride=grad_value_strides,
        dtype=dtype,
        device=device,
    )

    # we use dq as the output layout
    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=[batch_size, num_heads, seq_len_q, head_dim],
        stride=[sympy.sympify(s) for s in grad_query.get_stride()],
    )

    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)

    has_dlse = grad_logsumexp is not None
    if (
        has_dlse
        and V.graph.sizevars.statically_known_equals(head_dim, 256)
        and V.graph.sizevars.statically_known_equals(v_head_dim, 256)
        and torch.cuda.get_device_capability(device)[0] in (10, 11)
    ):
        raise NotImplementedError(
            "FLASH backend dLSE is not supported by the SM100/SM110 dedicated "
            "head_dim=256 FA4 backward kernel. Use BACKEND='TRITON' for this "
            "configuration."
        )

    choices: list[Any] = []

    input_nodes: list[TensorBox] = [
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp,
        grad_key,
        grad_value,
    ]

    has_block_mask = mask_graph_buffer is not None
    if has_block_mask:
        if q_indices is None:
            raise AssertionError("q_indices required when block mask is present")
        if full_q_num_blocks is None:
            raise AssertionError(
                "full_q_num_blocks required when block mask is present"
            )
        if full_q_indices is None:
            raise AssertionError("full_q_indices required when block mask is present")
        input_nodes.extend(
            [
                cast(TensorBox, q_num_blocks),
                q_indices,
                full_q_num_blocks,
                full_q_indices,
            ]
        )

    has_dq_write_order = dq_write_order is not None
    if has_dq_write_order:
        input_nodes.append(dq_write_order)
        if dq_write_order_full is not None:
            input_nodes.append(dq_write_order_full)
    has_dq_kv_order = dq_kv_order is not None and has_dq_write_order
    dq_kv_order_spt_for_flash = dq_kv_order_spt if has_dq_write_order else None
    if has_dq_kv_order:
        input_nodes.append(dq_kv_order)
    if has_dlse:
        input_nodes.append(cast(TensorBox, grad_logsumexp))

    supports_dq_kv_order = False
    supports_spt = False
    if has_block_mask:
        # pyrefly: ignore[missing-import]
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

        block_sparse_fields = getattr(BlockSparseTensorsTorch, "_fields", ())
        supports_dq_kv_order = "dq_kv_order" in block_sparse_fields
        supports_spt = "spt" in block_sparse_fields
        if has_dq_kv_order and not supports_dq_kv_order:
            raise NotImplementedError(
                "Explicit tensor dq_kv_order requires flash-attn-4 with dq_kv_order support"
            )
        if dq_kv_order_spt_for_flash is not None and not (
            supports_dq_kv_order or supports_spt
        ):
            raise NotImplementedError(
                "Boolean dq_kv_order requires flash-attn-4 with dq_kv_order or spt support"
            )

    deterministic_requested = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    deterministic_backward_enabled = deterministic_requested
    if deterministic_requested and has_block_mask:
        major, _ = torch.cuda.get_device_capability(device)
        missing_dq_write_order = dq_write_order is None or (
            full_q_num_blocks is not None and dq_write_order_full is None
        )
        missing_dq_kv_order = not (
            has_dq_kv_order or dq_kv_order_spt_for_flash is not None
        )
        if major < 10:
            if warn_only:
                deterministic_backward_enabled = False
            else:
                raise NotImplementedError(
                    "Deterministic backward for flex_attention with block_mask and BACKEND='FLASH' "
                    "requires SM100+ (compute capability >= 10.0). "
                    "Use BACKEND='TRITON' for deterministic backward on older architectures."
                )
        elif missing_dq_write_order:
            if warn_only:
                deterministic_backward_enabled = False
            else:
                raise ValueError(
                    "Deterministic backward for flex_attention with block_mask and BACKEND='FLASH' "
                    "requires dQ write-order metadata. Create the block mask with "
                    "create_block_mask(..., compute_dq_write_order=True)."
                )
        elif missing_dq_kv_order:
            if warn_only:
                deterministic_backward_enabled = False
            else:
                raise ValueError(
                    "Deterministic backward for flex_attention with block_mask and BACKEND='FLASH' "
                    "requires dQ KV scheduler-order metadata. Create the block mask with "
                    "create_block_mask(..., compute_dq_write_order=True)."
                )
    if deterministic_requested and not deterministic_backward_enabled:
        warnings.warn(
            "flex_attention backward with block_mask and BACKEND='FLASH' does not have "
            "a deterministic implementation for this configuration, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'. "
            "Running non-deterministic backward.",
        )

    has_score_mod = fw_subgraph_buffer is not None and joint_subgraph_buffer is not None
    subgraphs = []
    if has_score_mod:
        subgraphs.append(fw_subgraph_buffer)
        subgraphs.append(joint_subgraph_buffer)
    if has_block_mask:
        subgraphs.append(mask_graph_buffer)

    aux_scalar_symbols = collect_aux_scalar_symbols(
        score_mod_other_buffers or (), mask_mod_other_buffers or ()
    )
    if aux_scalar_symbols and not flash_supports_aux_scalars():
        raise RuntimeError(
            "CUTE flash attention scalar captures require flash-attn-4>=4.0.0b17. "
            f"{FLASH_ATTENTION_INSTALL_MESSAGE}"
        )
    configs = _get_flex_flash_bwd_configs()

    error: NotImplementedError | None = None
    for conf in configs:
        with patch_fixed_layout_indexer_for_cutedsl():
            error = flash_attention_backward_cutedsl_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=output_layout,
                mutated_inputs=[grad_key, grad_value],
                subgraphs=subgraphs or None,
                SM_SCALE=scale,
                HAS_SCORE_MOD=has_score_mod,
                SCORE_MOD_VEC_SIZE=conf.score_mod_vec_size,
                HAS_BLOCK_MASK=has_block_mask,
                HAS_DQ_WRITE_ORDER=has_dq_write_order,
                HAS_DQ_WRITE_ORDER_FULL=dq_write_order_full is not None,
                HAS_DQ_KV_ORDER=has_dq_kv_order,
                DQ_KV_ORDER_SPT=dq_kv_order_spt_for_flash,
                HAS_DLSE=has_dlse,
                AUX_SCALAR_SYMBOLS=aux_scalar_symbols,
                SUPPORTS_DQ_KV_ORDER=supports_dq_kv_order,
                SUPPORTS_SPT=supports_spt,
                DETERMINISTIC_BACKWARD_ENABLED=deterministic_backward_enabled,
                SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
                SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
            )
        if error is not None and len(configs) == 1:
            raise RuntimeError(f"CuteDSL template failed: {error}")

    for choice in choices:
        wrap_choice_render_with_cutedsl_indexer(choice)

    if not choices:
        raise RuntimeError(f"CuteDSL template failed: {error}")

    input_gen_fns: dict[int, Callable] | None = None
    if has_block_mask:
        input_gen_fns = {
            8: create_num_blocks_fake_generator(q_indices),
            9: create_indices_fake,
            10: create_num_blocks_fake_generator(full_q_indices),
            11: create_indices_fake,
        }

    template_output, _ = autotune_select_algorithm(
        "flex_flash_attention_backward",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, grad_key, grad_value, tuple())
