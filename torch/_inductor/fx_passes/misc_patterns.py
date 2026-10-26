# mypy: allow-untyped-defs
# [MODULE DOC] Miscellaneous graph rewrite patterns for Inductor: randperm-based indexing to unsafe
# indexing, e8m0 (MX format scale) extraction to a fused conversion or bit manipulation, and a
# pass that normalizes numpy-style keyword names. Import-time side effect: creates the global
# numpy_compat_normalization pass instance. Patterns are registered lazily by _misc_patterns_init.
import functools

import torch
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import fwd_only, register_replacement
from ..utils import is_nvidia_sm100_or_later


aten = torch.ops.aten


@functools.cache
def _misc_patterns_init(input_device: torch.device | None = None):
    # [FUNCTION DOC] HELPER: Cached (per input device) one-time registration of replacement patterns into the
    # post-grad (medium priority) and joint-graph pattern matchers. Device is the given one, else cuda
    # if available, else cpu. Registers: randperm+slice+index_add to unsafe index_put; randperm full index
    # and randperm sliced index to unsafe index; on cuda, float32 e8m0 rounding-up patterns (bit
    # manipulation form only on SM100+ -> fused cvt prim; log2/ceil form on all cuda, replaced by the prim on
    # SM100+ or a bit-manipulation fallback otherwise). Side effects: global pattern registries mutated.
    from .joint_graph import patterns as joint_graph_patterns
    from .post_grad import pass_patterns as post_grad_patterns_all

    post_grad_patterns = post_grad_patterns_all[1]  # medium priority

    if input_device:
        device = str(input_device)
    else:
        if torch.cuda.is_available():
            # workaround https://github.com/pytorch/pytorch/issues/97894
            device = "cuda"
        else:
            device = "cpu"

    # These patterns do 2 things
    # 1. Since we know that index is completely unique, we can codegen it using
    # stores instead of atomic adds, which is quite a bit faster.
    # 2. Also, since we are guaranteed that they are completely within bounds,
    # we can use unsafe indexing and skip debug asserts
    def randperm_index_add_pattern(x, y):
    # [FUNCTION DOC] HELPER: Pattern: random permutation prefix used as unique index in index_add; returns result and index.
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return torch.index_add(x, dim=0, source=y, index=index), index

    def randperm_index_add_replacement(x, y):
    # [FUNCTION DOC] HELPER: Replacement: same index, but uses unsafe gather plus add and unsafe index_put without accumulation.
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return (
            torch.ops.aten._unsafe_index_put(
                x, (index,), aten._unsafe_index(x, (index,)) + y, accumulate=False
            ),
            index,
        )

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_replacement,
        [torch.empty(4, 8, device=device), torch.empty(2, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        skip_duplicates=True,
    )

    def randperm_index_full_pattern(x):
    # [FUNCTION DOC] HELPER: Pattern: indexing a tensor by a full random permutation; returns result and index.
        index = torch.randperm(x.shape[0], device=x.device)
        return torch.ops.aten.index(x, (index,)), index

    def randperm_index_full_replacement(x):
    # [FUNCTION DOC] HELPER: Replacement: same but with unsafe indexing (permutation is in bounds).
        index = torch.randperm(x.shape[0], device=x.device)
        return torch.ops.aten._unsafe_index(x, (index,)), index

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_full_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_full_replacement,
        [torch.empty(4, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        skip_duplicates=True,
    )

    def randperm_index_pattern(x, slice_shape):
    # [FUNCTION DOC] HELPER: Pattern: indexing by a sliced random permutation of length slice_shape; returns result and index.
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten.index(x, (index,)), index

    def randperm_index_replacement(x, slice_shape):
    # [FUNCTION DOC] HELPER: Replacement: same but with unsafe indexing.
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten._unsafe_index(x, (index,)), index

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_replacement,
        [torch.empty(4, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        # Keep this smaller than the example input dim so tracing preserves the slice.
        scalar_workaround={"slice_shape": 2},
        skip_duplicates=True,
    )

    # Pattern: e8m0 extraction with ceiling rounding (for MX format scaling)
    if device.startswith("cuda"):

        def e8m0_extra_check(match):
        # [FUNCTION DOC] HELPER: Extra check for e8m0 patterns: the match must have an input node whose
        # fake value is a float32 cuda tensor; returns False otherwise.
            inp = match.kwargs.get("inp")
            if inp is None:
                return False
            inp_val = inp.meta.get("val")
            return (
                inp_val is not None
                and inp_val.device.type == "cuda"
                and inp_val.dtype == torch.float32
            )

        is_sm100_plus = is_nvidia_sm100_or_later()

        if is_sm100_plus:
            from .. import inductor_prims

            # Pattern 1: Bit manipulation approach (NVIDIA SM100+ only - uses PTX instruction)
            def e8m0_rceil_pattern(inp):
            # [FUNCTION DOC] HELPER: Pattern (SM100+): reinterprets float32 bits, takes the biased exponent, adds one if the
            # mantissa is nonzero, clamps to 0..255 and casts to uint8 (e8m0 round toward positive infinity).
                inp_bits = inp.view(torch.int32)
                biased_exp = (inp_bits >> 23) & 0xFF
                mantissa = inp_bits & 0x7FFFFF
                needs_round_up = mantissa != 0
                e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
                e8m0_biased = torch.clamp(e8m0_biased, 0, 255)
                return e8m0_biased.to(torch.uint8)

            def e8m0_rceil_replacement(inp):
            # [FUNCTION DOC] HELPER: Replacement: calls the fused inductor cvt_e8m0_rceil primitive.
                return inductor_prims.cvt_e8m0_rceil(inp)

            register_replacement(
                # pyrefly: ignore [bad-argument-type]
                e8m0_rceil_pattern,
                # pyrefly: ignore [bad-argument-type]
                e8m0_rceil_replacement,
                [torch.randn(32, device="cuda", dtype=torch.float32)],
                # pyrefly: ignore [bad-argument-type]
                fwd_only,
                # pyrefly: ignore [bad-argument-type]
                [post_grad_patterns],
                extra_check=e8m0_extra_check,
                skip_duplicates=True,
            )

        # Pattern 2: log2 + ceil approach (used by torchao MX formats)
        # Matches: (clamp(ceil(log2(x)), -127, 127) + 127).to(uint8)
        #
        # Registered on ALL CUDA hardware. On NVIDIA SM100+ uses the PTX instruction;
        # on earlier hardware uses exact IEEE 754 bit-manipulation.
        #
        # The bit-manipulation replacement is preferred over the software
        # log2+ceil because Triton's fused kernels can produce inputs that
        # differ by ~1 ULP from eager mode (e.g. due to FMA or libdevice
        # polynomial differences in erf-based GELU). When such an input falls
        # exactly on a power-of-2 boundary, software log2 may round the result
        # to the integer below, making ceil return the wrong value and the
        # uint8 output differ by 1.  The bit-manipulation approach reads the
        # IEEE 754 biased exponent directly, which is always correct for any
        # representable positive normal float32 value.
        E8M0_BIAS = 127

        def e8m0_rceil_log2_pattern(inp):
        # [FUNCTION DOC] HELPER: Pattern: log2, ceil, clamp to the e8m0 bias range, add bias, cast to uint8.
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        if is_sm100_plus:

            def e8m0_rceil_log2_replacement(inp):
            # [FUNCTION DOC] HELPER: Replacement on SM100+: calls the fused cvt_e8m0_rceil primitive.
                return inductor_prims.cvt_e8m0_rceil(inp)

        else:
            # Bit-manipulation fallback: extract IEEE 754 biased exponent with
            # ceiling rounding.  Equivalent to clamp(ceil(log2(inp)), -127, 127)
            # + 127 for all positive normal float32 values, but avoids software
            # log2 imprecision near exact powers of 2.
            # Clamp to [0, 254] to match the satfinite semantics of the original
            # pattern (clamp(ceil_val, -127, 127) + 127 gives at most 254).
            def e8m0_rceil_log2_replacement(inp):
            # [FUNCTION DOC] HELPER: Replacement on older hardware: exponent-bit extraction with round-up on nonzero mantissa,
            # clamped to 0..254, cast to uint8.
                inp_bits = inp.view(torch.int32)
                biased_exp = (inp_bits >> 23) & 0xFF
                mantissa = inp_bits & 0x7FFFFF
                needs_round_up = mantissa != 0
                e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
                e8m0_biased = torch.clamp(e8m0_biased, 0, 254)
                return e8m0_biased.to(torch.uint8)

        register_replacement(
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_pattern,
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_replacement,
            [torch.randn(32, device="cuda", dtype=torch.float32).abs() + 1e-10],
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            [post_grad_patterns],
            extra_check=e8m0_extra_check,
            skip_duplicates=True,
        )

    # TODO: Add pattern for cvt.rn.bf16x2.ue8m0x2 (e8m0 -> bf16 conversion)
    # This is the inverse operation for MX format dequantization


class NumpyCompatNormalization:
    # [CLASS DOC] Graph pass that renames numpy-style keyword arguments (axis, keepdims, x, a, x1, x2) of
    # torch function calls to the canonical torch names (dim, keepdim, input, other).
    # State: numpy_compat (canonical name to numpy aliases), inverse_mapping (alias to canonical name),
    # cache (call target to set of replaceable alias names). Instances are callable on an FX graph.
    numpy_compat: dict[str, tuple[str, ...]] = {
        "dim": ("axis",),
        "keepdim": ("keepdims",),
        "input": ("x", "a", "x1"),
        "other": ("x2",),
    }
    inverse_mapping: dict[str, str]
    cache: dict["torch.fx.graph.Target", OrderedSet[str]]

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes an empty cache and builds the inverse mapping from numpy aliases to canonical names.
        # Raises an error if an alias is mapped twice.
        self.cache = {}  # callable -> tuple of replaceable args e.g. ["axis"]
        self.inverse_mapping = {}
        for actual_kwarg, numpy_kwargs in self.numpy_compat.items():
            for numpy_kwarg in numpy_kwargs:
                if numpy_kwarg in self.inverse_mapping:
                    raise AssertionError(
                        f"duplicate numpy kwarg mapping for {numpy_kwarg}"
                    )
                self.inverse_mapping[numpy_kwarg] = actual_kwarg

    def __call__(self, graph: torch.fx.Graph):
        # [FUNCTION DOC] Mutates the graph in place: for each call_function node whose target is a plain torch function
        # (skips aten op overloads and packets), finds via its signatures which parameters have numpy aliases
        # (cached per target) and rewrites matching keyword names to the canonical names, keeping values.
        # Updates node kwargs and increments the inductor numpy_compat_normalization counter for each changed node.
        # Returns None.
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if isinstance(node.target, (OpOverload, OpOverloadPacket)):
                # only applies to torch ops; e.g. torch.stack(axis=1) works, torch.ops.aten.stack(axis=1) doesn't.
                continue
            kwargs = node.kwargs

            if node.target in self.cache:
                replaceable_kwargs = self.cache[node.target]
            else:
                signatures = torch.fx.operator_schemas.get_signature_for_torch_op(
                    node.target
                )
                signatures = () if signatures is None else signatures
                replaceable_kwargs = OrderedSet()
                for sig in signatures:
                    for param_name in sig.parameters:
                        if param_name in self.numpy_compat:
                            replaceable_kwargs.update(self.numpy_compat[param_name])

                self.cache[node.target] = replaceable_kwargs

            if not replaceable_kwargs:
                continue

            new_kwargs = {}
            kwargs_changed = False
            for k, v in kwargs.items():
                if k in replaceable_kwargs:
                    kwargs_changed = True
                    new_kwargs[self.inverse_mapping[k]] = v
                else:
                    new_kwargs[k] = v

            if kwargs_changed:
                node.kwargs = torch.fx.immutable_collections.immutable_dict(new_kwargs)
                counters["inductor"]["numpy_compat_normalization"] += 1


numpy_compat_normalization = NumpyCompatNormalization()
