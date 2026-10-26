# [MODULE DOC] Inductor post-grad pass that pads matrix dimensions of mm, bmm and addmm to dtype-specific alignment
# multiples (8 for fp16/bf16, 4 for fp32) when benchmarking or heuristics say padded matmul is faster.
# Contains: safety checks (can_pad), performance checks (should_pad, benchmark cache, autoheuristic),
# padding/replacement functions, and _pad_mm_init which registers the patterns.
# Import-time side effects: none beyond defining module globals (aten alias, _skip_do_bench_times test flag).
import functools
import itertools
import operator
import typing
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch._inductor.runtime.runtime_utils
from torch import Tensor
from torch._dynamo.utils import counters
from torch._inductor import utils
from torch._inductor.autoheuristic.autoheuristic import (
    AHContext,
    AutoHeuristic,
    LocalFeedback,
)
from torch._inductor.autoheuristic.autoheuristic_utils import (
    context_add_strides,
    context_add_using_tf32,
    pad_mm_operations,
    pad_mm_precondition,
)
from torch._inductor.runtime.caching import encoders, memoizers
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.utils._mode_utils import no_dispatch

from ...utils._triton import has_triton
from ..pattern_matcher import (
    fwd_only,
    gen_register_replacement,
    joint_fwd_bwd,
    Match,
    ReplaceFn,
    SearchFn,
)


aten = torch.ops.aten


# This flag is only used for testing purpose.
# Changing it to True will ignore comparing do_bench times
# between original pattern and padded one.
_skip_do_bench_times = False


def fetch_fake_tensors(match: Match, kwarg_names: Sequence[str]) -> list[Tensor]:
    # [FUNCTION DOC] HELPER: Collect the fake tensor values (stored in node meta under the value key) for the given kwarg names of a pattern match.
    # Returns a list in the same order as kwarg_names.
    kwargs = match.kwargs
    return [kwargs[name].meta["val"] for name in kwarg_names]


def unwrap_fake_args(
    *arg_names: str,
) -> Callable[[Callable[..., Any]], Callable[[Match], Any]]:
    # [FUNCTION DOC] HELPER: Decorator factory. The decorated function is wrapped so it takes a Match and is called with the fake tensors of the named match kwargs.
    def decorator(func: Callable[..., Any]) -> Callable[[Match], Any]:
        # [FUNCTION DOC] HELPER: Wrap func into a function of a Match.
        def wrapper(match: Match) -> Any:
            # [FUNCTION DOC] HELPER: Fetch the fake tensors for the captured arg names and call func with them, returning its result.
            fake_tensors = fetch_fake_tensors(match, arg_names)
            return func(*fake_tensors)

        return wrapper

    return decorator


def get_alignment_size(x: Tensor) -> int:
    # [FUNCTION DOC] Return the padding alignment size (in elements) for a tensor, based on its dtype.
    return get_alignment_size_dtype(x.dtype)


def get_alignment_size_dtype(dtype: torch.dtype) -> int:
    # [FUNCTION DOC] Return 8 for half/bfloat16, 4 for float32, and 0 (meaning never pad) for any other dtype.
    if dtype == torch.float16 or dtype == torch.half or dtype == torch.bfloat16:
        return 8
    elif dtype == torch.float32 or dtype == torch.float:
        return 4
    else:
        return 0


def check_device(a: Tensor, b: Tensor) -> bool:
    # [FUNCTION DOC] True if both tensors are on CUDA or both are on XPU.
    return (a.is_cuda and b.is_cuda) or (a.is_xpu and b.is_xpu)


def check_dtype(a: Tensor, b: Tensor) -> bool:
    # [FUNCTION DOC] True if both tensors are floating point.
    return a.is_floating_point() and b.is_floating_point()


def hint_symbols(
    ds: Sequence[int | torch.SymInt],
) -> list[int]:
    # [FUNCTION DOC] HELPER: Map each (possibly symbolic) dimension to a concrete optimization hint integer; returns a list of ints.
    """Helper to convert symbolic dimensions to their concrete hint values."""
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    return [optimization_hint(d) for d in ds]


def can_pad(
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    # [FUNCTION DOC] Safety check: decide whether padding is possible for an mm/addmm/bmm call.
    # Returns False if shape padding is disabled, devices are not both CUDA/XPU, dtypes are not float, any tensor has no static dim,
    # any dim is zero, the op is unsupported, no dim needs padding, deterministic mode is on without force/autoheuristic, or triton is missing.
    # Raises RuntimeError if a size is neither int nor SymInt. Reads inductor config only.
    """
    Determines if an operation CAN be padded (safety checks).
    All logic related to whether it's safe to pad should be here.
    """

    # Can't pad if there is no static dims, we pad static dims only.
    def has_one_static_dim(t: Tensor) -> bool:
        # [FUNCTION DOC] HELPER: True if the tensor has at least one plain int dimension; raises RuntimeError for an unexpected size type.
        """Return False if all dimensions are symbolic — nothing concrete to pad."""
        for x in t.size():
            if isinstance(x, int):
                return True
            elif not isinstance(x, torch.SymInt):
                raise RuntimeError("not expected size")
        return False

    # Basic safety checks
    if not torch._inductor.config.shape_padding:
        return False

    if not check_device(mat1, mat2):
        return False

    if not check_dtype(mat1, mat2):
        return False

    # For padding to be vaible each tensor should have at least one static dim.
    tensors = [t for t in (mat1, mat2, input) if t is not None]
    if not all(has_one_static_dim(t) for t in tensors):
        return False

    # Skip zero-sized dimensions — padding would be wasteful (mm on empty tensors)
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    if any(
        optimization_hint(dim) == 0 for dim in itertools.chain(mat1.shape, mat2.shape)
    ):
        return False

    # Calculate padding lengths to check if padding is needed
    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
        else:
            return False

        k_padded_length = get_padded_length(k, get_alignment_size(mat1))
        n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        m_padded_length = get_padded_length(m, get_alignment_size(mat1))

        # No padding needed - can't pad if there's nothing to pad
        if m_padded_length == k_padded_length == n_padded_length == 0:
            return False

    # In deterministic mode, we can't safely benchmark - disallow padding
    # Check this after other basic checks so force_shape_pad/autoheuristic can override
    if (
        torch._inductor.config.deterministic
        and not torch._inductor.config.force_shape_pad
        and not torch._inductor.config.use_autoheuristic("pad_mm")
    ):
        return False

    # Triton availability check - required for padding to work
    if not has_triton():
        return False

    return True


def get_padded_length(x: int | torch.SymInt, alignment_size: int) -> int:
    # [FUNCTION DOC] Number of elements to add to x to reach the next multiple of alignment_size.
    # Returns 0 for symbolic x, zero alignment, already aligned x, or x equal to 1 (squeezable).
    # we don't pad x if it is symbolic
    if isinstance(x, torch.SymInt) or alignment_size == 0 or x % alignment_size == 0:
        return 0

    # ignore dim that can be squeezed away
    if x == 1:
        return 0

    return int((x // alignment_size + 1) * alignment_size) - x


def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
    # [FUNCTION DOC] Append padded_length zeros to tensor x along dim by concatenation; returns x unchanged when padded_length is 0.
    if padded_length == 0:
        return x
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
    return torch.cat([x, pad], dim=dim)


def addmm_pattern(
    input: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float
) -> Tensor:
    # [FUNCTION DOC] HELPER: Search pattern for addmm with beta and alpha scalars; calls the aten addmm op.
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def _is_statically_expandable_to(shape: torch.Size, desired: Sequence[Any]) -> bool:
    # [FUNCTION DOC] HELPER: True if shape can be statically broadcast to desired (right-aligned; each dim equals the target or is 1, proven statically).
    if len(shape) > len(desired):
        return False
    return all(
        statically_known_true(dim == desired_dim) or statically_known_true(dim == 1)
        for dim, desired_dim in zip(reversed(shape), reversed(desired))
    )


def should_pad_addmm(match: Match) -> bool:
    # [FUNCTION DOC] Extra-check for addmm matches. Returns False for beta of zero on CUDA when the input is not statically expandable to the output shape;
    # otherwise defers to should_pad with the addmm op.
    mat1, mat2, input = fetch_fake_tensors(match, ("mat1", "mat2", "input"))
    beta = match.kwargs["beta"]
    if (
        beta == 0
        and input.is_cuda
        and not _is_statically_expandable_to(
            input.shape, (mat1.shape[0], mat2.shape[1])
        )
    ):
        return False
    return should_pad(match, mat1, mat2, torch.ops.aten.addmm, input=input)


def pad_addmm(
    input: Tensor | None,
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    beta: float = 1.0,
    alpha: float = 1.0,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    # [FUNCTION DOC] Compute addmm with padded operands then slice the result back to original size.
    # Pads mat1/mat2 (unless already pre-padded) and pads the bias along n and m when it is not broadcast (size 1),
    # runs addmm, and trims m_padded_length rows and n_padded_length columns. Returns the unpadded-shape result.
    # for paddings, dim order is reversed for some reasons
    # and for every dim, we need to specify left and right padding
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )

    # the add broadcasts, so we only pad if the dimension != 1
    if input is not None:
        if n_padded_length != 0:
            if input.dim() == 2 and input.shape[1] != 1:
                input = pad_dim(input, n_padded_length, 1)
            elif input.dim() == 1 and input.shape[0] != 1:
                input = pad_dim(input, n_padded_length, 0)
        if m_padded_length != 0 and input.dim() == 2 and input.shape[0] != 1:
            input = pad_dim(input, m_padded_length, 0)

    res = aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def addmm_replace(
    input: Tensor | None,
    mat1: Tensor,
    mat2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    # [FUNCTION DOC] Replacement function for the addmm pattern: computes per-dim padded lengths from alignment and calls pad_addmm.
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    return pad_addmm(
        input,
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
        beta,
        alpha,
    )


def is_mm_compute_bound(M: int, K: int, N: int, dtype: torch.dtype) -> bool:
    # [FUNCTION DOC] Heuristic: True if the matmul of shape MxK by KxN is compute bound, using arithmetic intensity versus half the machine balance.
    # Returns False for zero-sized problems; forces True for bf16 with large K on pre-Hopper GPUs/XPU, or if device info lookup fails.
    denominator = M * K + N * K + M * N
    if denominator == 0:
        return False
    arithmetic_intensity = (M * N * K) / denominator

    # we have experienced some large perf hits in this case, even in bandwidth bound regimes
    if (
        dtype is torch.bfloat16
        and K > M
        and K > N
        and (torch.xpu.is_available() or torch.cuda.get_device_capability() < (9, 0))
    ):  # doesn't repro on h100s:
        return True

    # Fails with AMD
    try:
        machine_balance = (
            1000 * utils.get_device_tflops(dtype)
        ) / utils.get_gpu_dram_gbps()
    except Exception:
        return True

    # dram_gbps might be underestimating bandwidth because of cache.
    # if we estimate machine balance too low we might miss some speedups,
    # if we estimate too high there will be unnecessary compilation time increase.
    # TODO - finetune coefficient here. As a reference point, Triton mm model assumes
    # 80% of reads are in cache and cache is 4x faster than dram_gbps
    machine_balance = machine_balance * 0.5

    return arithmetic_intensity > machine_balance


@functools.cache
def get_pad_cache() -> torch._inductor.codecache.LocalCache:
    # [FUNCTION DOC] HELPER: Return the process-wide (cached) LocalCache used for padding decisions and benchmark times.
    return torch._inductor.codecache.LocalCache()


def get_cached_should_pad(key: str) -> bool:
    # [FUNCTION DOC] HELPER: Look up a cached padding decision by key; None if absent.
    return get_pad_cache().lookup(key)  # type: ignore[return-value]


def set_cached_should_pad(key: str, value: bool) -> None:
    # [FUNCTION DOC] HELPER: Store a padding decision in the cache.
    return get_pad_cache().set_value(key, value=value)


def get_cached_base_mm_benchmark_time(key: str) -> float:
    # [FUNCTION DOC] HELPER: Look up a cached original-mm benchmark time by key; None if absent.
    return get_pad_cache().lookup(key)  # type: ignore[return-value]


def set_cached_base_mm_benchmark_time(key: str, value: float) -> None:
    # [FUNCTION DOC] HELPER: Store an original-mm benchmark time in the cache.
    return get_pad_cache().set_value(key, value=value)


def should_pad_bench_key(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
    is_base_time_key: bool = False,
) -> str:
    # [FUNCTION DOC] HELPER: Build the string cache key for a padding decision from operand shapes/strides/dtypes, whether padding time is excluded,
    # the op, the optional bias, and the tf32 setting. With is_base_time_key the pad-exclusion parts are omitted and a prefix is added.
    def tensor_key(t: Tensor) -> tuple[torch.Size, tuple[int, ...], torch.dtype]:
        # [FUNCTION DOC] HELPER: Tuple of shape, stride and dtype for a tensor.
        return (t.shape, t.stride(), t.dtype)

    tf32_key = (
        None
        if mat1.dtype != torch.float32
        else torch.backends.cuda.matmul.fp32_precision == "tf32"
        or torch.backends.mkldnn.fp32_precision == "tf32"
    )

    def fmt_pad(name: str) -> str | None:
        # [FUNCTION DOC] HELPER: Return the pad-exclusion key text for an operand, or None for base-time keys.
        if is_base_time_key:
            return None
        return f"exclude_pad:{should_exclude_padding_time(match, name)}"

    key = (
        tensor_key(mat1),
        tensor_key(mat2),
        fmt_pad("mat1"),
        fmt_pad("mat2"),
        op,
        input if input is None else tensor_key(input),
        tf32_key,
    )

    key = str(key)
    if is_base_time_key:
        key = f"base mm time: {key}"
    return key


def get_non_view_def(node: torch.fx.Node) -> torch.fx.Node:
    # [FUNCTION DOC] HELPER: Walk back through getitem and view nodes to the defining non-view node.
    if node.op == "call_function" and node.target is operator.getitem:
        return get_non_view_def(node.args[0])  # type: ignore[arg-type]

    if (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and utils.is_view(node.target)
    ):
        return get_non_view_def(node.all_input_nodes[0])

    return node


def should_exclude_padding_time(match: Match, arg_name: str) -> bool:
    # [FUNCTION DOC] HELPER: Decide whether the cost of padding the named operand should be excluded from benchmarking because it can be memory planned away.
    # False if the operand is not contiguous or is produced by an op whose output cannot be planned (mm, conv, bmm, addmm, attention, large cat);
    # otherwise True unless it is a graph input placeholder.
    from torch._prims_common import is_contiguous_or_false

    node_def = get_non_view_def(match.kwargs[arg_name])

    # constant padding converts tensors to contiguous so even if the input tensor
    # can be planned layout transform is not free. TODO - way to pad and preserve layout ?
    # Use is_contiguous_or_false to avoid guarding on data-dependent expressions
    # with unbacked symints - returns False instead of raising an error.
    if not is_contiguous_or_false(fetch_fake_tensors(match, (arg_name,))[0]):
        return False

    # TODO - see issue https://github.com/pytorch/pytorch/issues/128889
    # We would only able to completely plan these out if we were only doing
    # first dimension padding. non-first we would still need a copy
    # because these outputs are fixed dense.
    cannot_plan_output = [
        aten.mm.default,
        aten.convolution.default,
        aten.convolution_backward.default,
        aten.bmm.default,
        aten.addmm.default,
        aten._scaled_dot_product_flash_attention.default,
        aten._scaled_dot_product_efficient_attention.default,
    ]

    if node_def.target in cannot_plan_output:
        return False

    if (
        node_def.target is aten.cat.default
        and len(node_def.all_input_nodes)
        > torch._inductor.config.max_pointwise_cat_inputs
    ):
        return False

    # optimistically assume we should be able to memory plan away
    # all non inputs
    return node_def.op != "placeholder"


def is_padded_faster(key: str, ori_time: float, pad_time: float) -> bool:
    # [FUNCTION DOC] Decide whether padded is faster: original time must exceed padded time times a multiplier (default 1.1, overridable via fusion options).
    # Side effects: stores the result in the pad cache and bumps a counter when the multiplier option is used. Always True if the test skip flag is set.
    """
    Determines if padding is beneficial by comparing benchmark times.
    Helper function that applies a multiplier to account for memory ops overhead.
    """
    multiplier = 1.1
    # Shape padding introduces additional memory ops. Based on microbenchmarks, 1.1x represents a reasonable
    # tradeoff between performance improvement from shape padding and overhead from additional memory ops
    # TODO: Build a learned model which would be better than this heuristic
    if "shape_padding_multiplier" in torch._inductor.config.post_grad_fusion_options:
        multiplier = torch._inductor.config.post_grad_fusion_options[
            "shape_padding_multiplier"
        ].get("value", 1.1)
        counters["inductor"]["shape_padding_multiplier"] += 1
    padded_is_faster = _skip_do_bench_times or ori_time > pad_time * multiplier
    set_cached_should_pad(key, padded_is_faster)
    return padded_is_faster


def should_pad_mm_bf16(dtype: torch.dtype, M: int, N: int, K: int) -> bool:
    # [FUNCTION DOC] HELPER: True to force padding for bf16 mm with large odd N and K at or above a configured threshold on pre-Hopper GPUs/XPU.
    # always force pad for mm with bf16 when the following are satisfied to avoid perf regression
    large_k_threshold_to_pad = torch._inductor.config.post_grad_fusion_options[
        "pad_aten_mm_pass"
    ].get("k_threshold_to_pad", 8388608)
    if (
        dtype is torch.bfloat16
        and K > M
        and K > N
        and N % 2 == 1
        and K >= large_k_threshold_to_pad
        and (torch.xpu.is_available() or torch.cuda.get_device_capability() < (9, 0))
    ):  # doesn't repro on h100s:
        return True
    return False


def should_pad(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    # [FUNCTION DOC] Extra-check entry point: padding must be allowed (can_pad) and worthwhile (_should_pad).
    _can_pad = can_pad(mat1, mat2, op, input)
    # Note that if you're tempted to insert a dynamo_timed call here, this function can
    # be called enough that the dynamo_timed overhead is not negligible.
    return _can_pad and _should_pad(match, mat1, mat2, op, input)


def get_do_bench() -> Callable[[Callable[[], Any]], float]:
    # [FUNCTION DOC] HELPER: Return a benchmark callable (GPU benchmarker with 5 warmup) mapping a function to its time.
    return functools.partial(
        # pyrefly: ignore [bad-argument-type]
        torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu,
        warmup=5,
    )


@memoizers.should_pad_memoizer.memoize(
    custom_params_encoder=encoders.should_pad_params_encoder
)
def _should_pad(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    # [FUNCTION DOC] Performance check for padding (memoized). Computes padded lengths; returns True if force_shape_pad is set or the bf16 rule fires,
    # False if not compute bound or op unsupported, else uses cached decision, autoheuristic (mm only), or benchmarks original vs padded.
    # Returns False in deterministic mode if no autoheuristic decision. Side effects: allocates random tensors, runs benchmarks, updates caches and counters.
    """
    Determines if an operation SHOULD be padded (performance checks).
    All logic related to whether padding would be performant should be here.
    """
    do_bench = get_do_bench()

    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        else:
            return False

        # Force padding when explicitly requested - performance override
        if torch._inductor.config.force_shape_pad:
            return True

        # Resolve symbolic dims to concrete hints for heuristic checks below.
        # These are performance decisions, not correctness — optimization_hint is safe.
        m_concrete, k_concrete, n_concrete = hint_symbols((m, k, n))

        # Performance heuristic for bf16 large K scenarios
        if (
            "pad_aten_mm_pass" in torch._inductor.config.post_grad_fusion_options
            and should_pad_mm_bf16(mat1.dtype, m_concrete, n_concrete, k_concrete)
        ):
            return True

        # Check if operation is compute bound (performance check)
        if not is_mm_compute_bound(m_concrete, k_concrete, n_concrete, mat1.dtype):
            return False

        # We don't want to look up the cache for cases that are trivially false
        # since it does file io
        key = should_pad_bench_key(match, mat1, mat2, op, input)

        cached_pad = get_cached_should_pad(key)
        if cached_pad is not None:
            return cached_pad

        def realize_tensor(t):
            # [FUNCTION DOC] HELPER: Create a real random tensor matching a tensor's shape and strides (using size hints for fake/symbolic tensors).
            if is_fake_tensor(t):
                size_hints = hint_symbols(t.size())
                # pyrefly: ignore [bad-argument-type]
                stride_hint = hint_symbols(t.stride())
                real_size = (
                    sum((d - 1) * s for d, s in zip(size_hints, stride_hint)) + 1
                )
                real_t = torch.randn(real_size, dtype=t.dtype, device=t.device)
                return torch.as_strided(real_t, size_hints, stride_hint)
            else:
                return torch.randn_like(t)

        mat1 = realize_tensor(mat1)
        mat2 = realize_tensor(mat2)

        # since we key on whether or not the inputs can be memory planned, set cache for the
        # original time which is unaffected by whether or not the input can be planned
        ori_time_key = should_pad_bench_key(
            match, mat1, mat2, op, input, is_base_time_key=True
        )
        ori_time = get_cached_base_mm_benchmark_time(ori_time_key)
        if ori_time is None and op is torch.ops.aten.addmm and input is not None:
            # realize bias for addmm
            input = realize_tensor(input)

        mat1_pad = mat1
        mat2_pad = mat2

        is_bmm = op is torch.ops.aten.bmm

        mat1_pre_padded = should_exclude_padding_time(match, "mat1")
        fns = []
        if mat1_pre_padded and (m_padded_length or k_padded_length):
            mat1_pad = pad_mat1(
                mat1_pad,
                m_padded_length=m_padded_length,
                k_padded_length=k_padded_length,
                is_bmm=is_bmm,
            )

            def write_pad():
                # [FUNCTION DOC] HELPER: Zero the padded corner region of the pre-padded mat1 buffer (benchmarks the cost of writing pad).
                if is_bmm:
                    mat1_pad[:, -m_padded_length:, -k_padded_length:].fill_(0)
                else:
                    mat1_pad[-m_padded_length:, -k_padded_length:].fill_(0)

            fns.append(write_pad)

        mat2_pre_padded = should_exclude_padding_time(match, "mat2")
        if mat2_pre_padded and (k_padded_length or n_padded_length):
            mat2_pad = pad_mat2(
                mat2_pad,
                k_padded_length=k_padded_length,
                n_padded_length=n_padded_length,
                is_bmm=is_bmm,
            )

            def write_pad():
                # [FUNCTION DOC] HELPER: Zero the padded corner region of the pre-padded mat2 buffer.
                if is_bmm:
                    mat2_pad[:, -k_padded_length:, -n_padded_length:].fill_(0)
                else:
                    mat2_pad[-k_padded_length:, -n_padded_length:].fill_(0)

            fns.append(write_pad)

        if op is torch.ops.aten.addmm:
            input_pad = None
            if input is not None and (input.is_cuda or input.is_xpu):
                input_pad = torch.randn_like(input)
            fns.append(
                lambda: pad_addmm(
                    input_pad,
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    mat1_pre_padded=mat1_pre_padded,
                    mat2_pre_padded=mat2_pre_padded,
                )
            )
        elif op is torch.ops.aten.mm:
            fns.append(
                lambda: pad_mm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    mat1_pre_padded=mat1_pre_padded,
                    mat2_pre_padded=mat2_pre_padded,
                )
            )
        else:
            fns.append(
                lambda: pad_bmm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    mat1_pre_padded=mat1_pre_padded,
                    mat2_pre_padded=mat2_pre_padded,
                )
            )

        def orig_bench_fn():
            # [FUNCTION DOC] HELPER: Run the original unpadded op once for benchmarking.
            if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
                op(mat1, mat2)
            else:
                op(input, mat1, mat2)

        def pad_bench_fn():
            # [FUNCTION DOC] HELPER: Run all padding benchmark steps once.
            for fn in fns:
                fn()

        if (
            torch._inductor.config.run_autoheuristic("pad_mm")
            and op is torch.ops.aten.mm
        ):
            ah_should_pad = run_autoheuristic(
                mat1,
                mat2,
                orig_bench_fn,
                pad_bench_fn,
                m_padded_length,
                k_padded_length,
                n_padded_length,
                do_bench,
                mat1_pre_padded,
                mat2_pre_padded,
                ori_time,
                ori_time_key,
                key,
            )
            if ah_should_pad is not None:
                return ah_should_pad

        # AH didn't make a decision, so if we're in deterministic mode, we should return false
        if torch._inductor.config.deterministic:
            return False

        if ori_time is None:
            ori_time = do_bench(orig_bench_fn)
            set_cached_base_mm_benchmark_time(ori_time_key, ori_time)

        pad_time = do_bench(pad_bench_fn)

        counters["inductor"]["pad_mm_bench"] += 1
        return is_padded_faster(key, ori_time, pad_time)


def get_context(
    mat1: Tensor,
    mat2: Tensor,
    mat1_pre_padded: bool,
    mat2_pre_padded: bool,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
) -> AHContext:
    # [FUNCTION DOC] HELPER: Build the autoheuristic context with m/k/n, strides, padded lengths, alignment sizes, dtypes, pre-padded flags and tf32 usage.
    context = AHContext()

    context.add_feature("m", mat1.shape[0])
    context.add_feature("k", mat1.shape[1])
    context.add_feature("n", mat2.shape[1])

    context_add_strides(context, "mat1", mat1.stride())
    context_add_strides(context, "mat2", mat2.stride())

    context.add_feature("m_padded_length", m_padded_length)
    context.add_feature("k_padded_length", k_padded_length)
    context.add_feature("n_padded_length", n_padded_length)

    context.add_feature("mat1_align_size", get_alignment_size(mat1))
    context.add_feature("mat2_align_size", get_alignment_size(mat2))

    context.add_feature("mat1_dtype", mat1.dtype, is_categorical=True)
    context.add_feature("mat2_dtype", mat2.dtype, is_categorical=True)

    context.add_feature("prepadded_mat1", mat1_pre_padded, is_categorical=True)
    context.add_feature("prepadded_mat2", mat2_pre_padded, is_categorical=True)

    context_add_using_tf32(context, mat1.dtype)
    return context


def run_autoheuristic(
    mat1: Tensor,
    mat2: Tensor,
    orig_bench_fn: Callable[[], None],
    pad_bench_fn: Callable[[], None],
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    do_bench: Callable[[Callable[[], Any]], float],
    mat1_pre_padded: bool,
    mat2_pre_padded: bool,
    ori_time: float,
    ori_time_key: str,
    key: str,
) -> bool | None:
    # [FUNCTION DOC] HELPER: Ask the AutoHeuristic for pad vs original. Returns True/False for a decision, or None when it defers to autotuning.
    # Side effects: may benchmark and collect feedback; caches the decision or benchmark times.
    def feedback_fn(
        choice: str,
    ) -> float | None:
        # [FUNCTION DOC] HELPER: Benchmark feedback for a choice (original or pad); None for unknown choices.
        if choice == orig_choice:
            return do_bench(orig_bench_fn)
        elif choice == pad_choice:
            return do_bench(pad_bench_fn)
        return None

    def fallback() -> str:
        # [FUNCTION DOC] HELPER: Fallback choice name meaning autotune.
        return "autotune"

    orig_choice = "orig"
    pad_choice = "pad"
    choices = [orig_choice, pad_choice]
    feedback = LocalFeedback(feedback_fn)  # type: ignore[arg-type]
    context = get_context(
        mat1,
        mat2,
        mat1_pre_padded,
        mat2_pre_padded,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )
    name = "pad_mm"
    autoheuristic = AutoHeuristic(
        fallback=fallback,
        choices=choices,
        feedback=feedback,
        context=context,
        name=name,
        augment_context=pad_mm_operations(),
        precondition=pad_mm_precondition,
    )
    choice = autoheuristic.get_choice()
    choice2should_pad = {orig_choice: False, pad_choice: True, "autotune": None}
    ah_should_pad = choice2should_pad.get(choice)

    if torch._inductor.config.collect_autoheuristic(name):
        ah_ori_time = autoheuristic.get_collected_feedback(orig_choice)
        ah_pad_time = autoheuristic.get_collected_feedback(pad_choice)

        # if precondition is not satisfied, autoheuristic does not collect data
        if ah_ori_time is not None and ah_pad_time is not None:
            if ori_time is None:
                set_cached_base_mm_benchmark_time(ori_time_key, ah_ori_time)
            return is_padded_faster(key, ah_ori_time, ah_pad_time)
    if ah_should_pad is not None:
        set_cached_should_pad(key, ah_should_pad)
    return ah_should_pad


def mm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    # [FUNCTION DOC] HELPER: Search pattern for mm; calls the aten mm op.
    return aten.mm(mat1, mat2)


def should_pad_mm(match: Match) -> bool:
    # [FUNCTION DOC] Extra-check for mm matches: delegates to should_pad with the mm op.
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.mm)


def pad_mat1(
    mat1: Tensor, *, m_padded_length: int, k_padded_length: int, is_bmm: bool = False
) -> Tensor:
    # [FUNCTION DOC] HELPER: Pad mat1 on the right (k) and bottom (m) with zeros via constant padding; extra batch dim handled when is_bmm. Returns mat1 unchanged if no padding.
    if k_padded_length != 0 or m_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, k_padded_length, 0, m_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat1, pad_arg)
    else:
        return mat1


def pad_mat2(
    mat2: Tensor, *, k_padded_length: int, n_padded_length: int, is_bmm: bool = False
) -> Tensor:
    # [FUNCTION DOC] HELPER: Pad mat2 on the right (n) and bottom (k) with zeros via constant padding; extra batch dim handled when is_bmm. Returns mat2 unchanged if no padding.
    if k_padded_length != 0 or n_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, n_padded_length, 0, k_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat2, pad_arg)
    else:
        return mat2


def pad_mm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    # [FUNCTION DOC] Compute mm on padded operands (skipping those pre-padded) and slice off padded rows/columns of the result.
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )
    res = aten.mm(mat1, mat2)
    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def mm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    # [FUNCTION DOC] Replacement function for the mm pattern: computes padded lengths from alignment and calls pad_mm.
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    return pad_mm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


def bmm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    # [FUNCTION DOC] HELPER: Search pattern for bmm; calls the aten bmm op.
    return aten.bmm(mat1, mat2)


def should_pad_bmm(match: Match) -> bool:
    # [FUNCTION DOC] Extra-check for bmm matches: delegates to should_pad with the bmm op.
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.bmm)


def pad_bmm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    # [FUNCTION DOC] Compute bmm on padded operands (skipping those pre-padded) and slice off padded rows/columns of the result.
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1,
            m_padded_length=m_padded_length,
            k_padded_length=k_padded_length,
            is_bmm=True,
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2,
            k_padded_length=k_padded_length,
            n_padded_length=n_padded_length,
            is_bmm=True,
        )
    res = aten.bmm(mat1, mat2)
    if m_padded_length != 0:
        res = res[:, :-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :, :-n_padded_length]
    return res


def bmm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    # [FUNCTION DOC] Replacement function for the bmm pattern: computes padded lengths from alignment and calls pad_bmm.
    k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    return pad_bmm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


@functools.cache
def _pad_mm_init(input_device: torch.device | None = None) -> None:
    # [FUNCTION DOC] HELPER: One-time (cached) registration of mm, bmm and addmm padding replacements, each in training and inference variants.
    # Picks the trace device from input_device or availability (cuda, xpu, else cpu), builds dummy args, and calls gen_register_replacement
    # into the joint graph patterns. Raises AssertionError if a workaround entry is not a mapping.
    from .joint_graph import patterns

    if input_device:
        device = str(input_device)
    else:
        if torch.cuda.is_available():
            # workaround https://github.com/pytorch/pytorch/issues/97894
            device = "cuda"
        elif torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"

    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds

    dim2a = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim2b = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)

    dim3a = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim3b = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)

    dim1a = functools.partial(torch.empty, (4), device=device, requires_grad=True)

    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    rep = {"beta": 0.213377, "alpha": 0.113377}

    for pattern, replacement, args, workaround, extra_check in [
        (
            typing.cast(SearchFn, mm_pattern),
            typing.cast(ReplaceFn, mm_replace),
            [dim2a(), dim2b()],
            {},
            should_pad_mm,
        ),
        (
            typing.cast(SearchFn, bmm_pattern),
            typing.cast(ReplaceFn, bmm_replace),
            [dim3a(), dim3b()],
            {},
            should_pad_bmm,
        ),
        (
            typing.cast(SearchFn, addmm_pattern),
            typing.cast(ReplaceFn, addmm_replace),
            [dim1a(), dim2a(), dim2b()],
            rep,
            should_pad_addmm,
        ),
    ]:
        if not isinstance(
            workaround, dict
        ):  # mypy is unable to infer the type properly
            raise AssertionError(
                f"expected workaround to be a dict, got {type(workaround)}"
            )
        name = pattern.__name__

        gen_register_replacement(
            f"{name}_training",
            pattern,
            replacement,
            args,
            # pyrefly: ignore [bad-argument-type]
            joint_fwd_bwd,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
            skip_duplicates=True,
        )

        gen_register_replacement(
            f"{name}_inference",
            pattern,
            replacement,
            args,
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
            skip_duplicates=True,
        )
