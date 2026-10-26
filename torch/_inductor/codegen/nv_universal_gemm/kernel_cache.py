# mypy: allow-untyped-defs
# [MODULE DOC] Process-global caches of cutlass.operators GEMM operators for the
# NVGEMM backend. Holds (a) a lazily built full manifest (operator name -> operator),
# (b) an incrementally filled operator-by-name cache fed by args-filtered queries,
# (c) a memoized partition of compatible operators per GEMM signature, and (d) a cache
# of epilogue-fusion (EFC) operators bound to a specific epilogue. Public entry points:
# partition_compatible_kernels, get_kernel_by_name, get_kernel_by_name_via_args,
# get_efc_kernel_with_epilogue, ensure_cache_initialized, clear_cache.
# Import-time side effects: creates the module locks/caches and registers a wrapper with
# clear_on_fresh_cache so clear_cache runs whenever inductor caches are reset.
"""
Global kernel cache for NVIDIA Universal GEMM.

This module provides a lazy-initialized cache for cutlass.operators kernels,
avoiding expensive manifest scans on every kernel lookup.

The first call to get_kernel_by_name() loads all kernels from cutlass.operators
(~10 seconds) and builds a name->kernel dict. Subsequent calls use the
dict for O(1) lookup (~0.1 μs).
"""

import dataclasses
import functools
import logging
import threading
from collections.abc import Callable
from typing import Any, Literal

import torch


log = logging.getLogger(__name__)


@functools.cache
def _device_target(cc: int) -> Any:
    # [FUNCTION DOC] HELPER: Memoized conversion of an integer compute capability
    # into a cutlass TargetSm for the arch-specific ("a" suffixed) target. Imports
    # cutlass lazily; raises if cutlass is not installed or the target is invalid.
    """The CUDA arch target for this device (e.g. cc=100 -> TargetSm '100a').

    Used to reject kernels whose `supported_targets` don't cover this device.
    `designed_for_min_cc` alone is insufficient: an arch-conditional sm90 kernel
    reports min_cc=90 (which is <= a 100 device) yet only lists a cc=90 target
    and fails to compile on sm100 ("expects arch sm_90a, but got sm_100a").
    """
    from cutlass.operators.arch import TargetSm

    return TargetSm.ensure(f"{cc}a")


def _epilogue_args_signature(epilogue_args: Any) -> tuple:
    # [FUNCTION DOC] HELPER: Builds a hashable cache-key component from the
    # epilogue arguments' tensors: for each named tensor, (name, "tensor", dtype,
    # shape, strides); for non-tensors, (name, type name). Returns an empty tuple
    # when there are no epilogue args or no tensors. Pure, no side effects.
    """Extract a hashable signature of epilogue args for cache keying.

    Two callers with the same `(efc_kernel_name, epilogue_source)` but
    different aux-tensor specs (dtype, shape, stride) would otherwise share
    a kernel object whose internal JIT state is mutated by each call to
    `kernel.compile(args)` — a silent miscompile, since the compiled
    artifact's launch closure reads the kernel's CURRENT JIT at launch time
    rather than the one from when the artifact was built.
    """
    if epilogue_args is None:
        return ()
    tensors = getattr(epilogue_args, "tensors", None)
    if not tensors:
        return ()
    sig: list[tuple] = []
    for name, val in tensors.items():
        if torch.is_tensor(val):
            sig.append(
                (name, "tensor", val.dtype, tuple(val.shape), tuple(val.stride()))
            )
        else:
            sig.append((name, type(val).__name__))
    return tuple(sig)


# Reentrant: get_efc_kernel_with_epilogue holds this lock and may call
# get_kernel_by_name -> _ensure_caches, which re-acquires it on the same thread.
_cache_lock = threading.RLock()

# Full kernel manifest: kernel_name -> kernel object. Built lazily (~14s, ~290K
# operators) only as a LAST-RESORT fallback -- the common paths (choice
# generation and subprocess precompile) resolve operators via cheap
# args-filtered queries and cache them in _ops_by_name.
_kernel_by_name_cache: dict[str, Any] | None = None


def _operand_dtype_str(operand: Any) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the string form of an operand's dtype, looking
    # first at the operand itself and then at its wrapped tensor attribute; returns
    # None if no dtype can be found. Pure.
    """Best-effort cutlass dtype name for a kernel operand or args operand."""
    dtype = getattr(operand, "dtype", None)
    if dtype is None:
        tensor = getattr(operand, "tensor", None)
        dtype = getattr(tensor, "dtype", None)
    return None if dtype is None else str(dtype)


def _build_kernel_cache() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds the full manifest mapping operator name to
    # operator object by enumerating every cutlass operator (slow, seconds to
    # tens of seconds). First tries to import the vendored cutedsl wrapper module
    # so its operators get registered (an import failure is only logged). Returns
    # the new mapping; does not store it globally.
    """Build the full kernel name -> kernel manifest (fallback path only)."""
    import cutlass.operators

    log.debug("Building NVGEMM kernel cache (this may take a few seconds)...")

    try:
        from torch._inductor.kernel.vendored_templates.cutedsl import (  # noqa: F401
            wrappers,
        )
    except ImportError:
        log.debug("Vendored kernel wrappers not available")

    all_kernels = cutlass.operators.get_operators()
    cache = {k.metadata.operator_name: k for k in all_kernels}
    log.debug("NVGEMM kernel manifest built: %d kernels", len(cache))
    return cache


def _ensure_caches() -> None:
    # [FUNCTION DOC] HELPER: Lazily builds the global full-manifest cache using
    # double-checked locking on the reentrant cache lock. Mutates the module
    # global manifest; no return value.
    global _kernel_by_name_cache
    if _kernel_by_name_cache is None:
        with _cache_lock:
            if _kernel_by_name_cache is None:
                _kernel_by_name_cache = _build_kernel_cache()


def _get_kernel_cache() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the full manifest mapping, building it first
    # if needed. Takes a local snapshot of the global so a concurrent clear cannot
    # invalidate it. Raises AssertionError if the cache is still missing.
    """Return the fallback manifest, building it lazily if needed.

    Snapshot to local frame: a concurrent clear_cache() rebinding the global to
    None cannot turn the caller's subsequent read into AttributeError.
    """
    _ensure_caches()
    cache = _kernel_by_name_cache
    if cache is None:
        raise AssertionError("expected cache to be not None")
    return cache


def _operand_sig(operand: Any) -> tuple | None:
    # [FUNCTION DOC] HELPER: Builds a hashable signature for a GEMM operand:
    # (dtype string, shape tuple, stride tuple, recursive signature of its scale
    # operand, scale mode name, swizzle mode name). The mode/swizzle are read from
    # the instance attribute dictionary (so dense tensors' methods are not picked up)
    # and reduced to enum names since the enums are unhashable. Returns None if the
    # operand is None or dtype/shape/stride cannot be determined.
    """Hashable (dtype, shape, stride, scale-sig, mode, swizzle) signature."""
    if operand is None:
        return None
    w = getattr(operand, "tensor", operand)
    dtype = _operand_dtype_str(operand)
    shape = getattr(w, "shape", None)
    stride = getattr(w, "stride", None)
    if dtype is None or shape is None or stride is None:
        return None
    scale = getattr(operand, "scale", None)
    scale_sig = _operand_sig(scale) if scale is not None else None
    # ScaledOperand.mode/.swizzle are distinct from the scale tensor's layout
    # and are not folded into it, yet supports(args) depends on them -- two
    # scaled GEMMs with identical operand/scale layouts but different scale or
    # swizzle modes must not collide. Read from __dict__ (only ScaledOperand
    # sets these) so a dense operand's torch.Tensor.mode method isn't captured.
    d = getattr(operand, "__dict__", {})
    mode, swizzle = d.get("mode"), d.get("swizzle")
    # mode/swizzle are cutlass.operators enums (ScaleMode/ScaleSwizzleMode) that
    # are unhashable, so key on their .name to keep the sig usable as a dict key.
    mode = getattr(mode, "name", mode)
    swizzle = getattr(swizzle, "name", swizzle)
    return (dtype, tuple(shape), tuple(stride), scale_sig, mode, swizzle)


def _partition_sig(args: Any) -> tuple | None:
    # [FUNCTION DOC] HELPER: Combines the signatures of operands A, B and out plus
    # the accumulator type into one hashable key identifying everything operator
    # support checks depend on. Returns None if any operand signature is unavailable.
    # KNOWN ISSUE: for a dense torch tensor operand, shape/stride are read via
    # getattr and the stride method itself (not its result) is passed to the tuple
    # conversion, which would raise a TypeError; this appears to work only for
    # cutlass wrapper operands whose stride is a sequence attribute.
    """Signature capturing everything `supports(args)` depends on, or None if
    it can't be fully determined (falls back to a non-memoized scan)."""
    a = _operand_sig(getattr(args, "A", None))
    b = _operand_sig(getattr(args, "B", None))
    o = _operand_sig(getattr(args, "out", None))
    if a is None or b is None or o is None:
        return None
    return (a, b, o, getattr(args, "accumulator_type", None))


# Per-process cache of instantiated operators, keyed by operator_name. Populated
# incrementally by the args-filtered queries below (choice generation and the
# subprocess precompile fast path). One args query returns the whole set of
# operators compatible with a shape (hundreds); we index all of them, so any
# later lookup -- another config of that shape, another shape sharing those
# templates, or get_kernel_by_name for a chosen
# kernel -- resolves in O(1) without rebuilding the ~14s manifest. Operator
# names embed the arch (e.g. "..._sm100_..."), so the name alone is unambiguous
# within a process. Reusing an operator across different args is safe for the
# same reason the manifest always has been: an operator is a config template
# (its args-specific state lives in the artifact returned by compile(args)), and
# every (name, args) pairing routed here was validated by supports(args).
_ops_by_name: dict[str, Any] = {}

# Memoizes partition results. `partition_compatible_kernels` is called once per
# GEMM node during choice generation, but FLUX has hundreds of nodes sharing a
# handful of shapes (e.g. 114 nodes are all 4608x3072x3072); memoizing collapses
# that to one args query per unique shape.
_partition_cache: dict[tuple, list[list[Any]]] = {}


def _args_query_candidates(args: Any, cc: int, efc_only: bool) -> list[Any]:
    # [FUNCTION DOC] HELPER: Asks cutlass for operators compatible with the given
    # runtime arguments on the arch-specific target for cc, optionally restricted
    # by a metadata filter to epilogue-fusion-capable (EFC) operator classes
    # (class name contains "EFC"). Returns the list of operators. Dense GEMM only.
    """Compatible operators via an args-filtered get_operators query (~0.05s).

    cutlass prunes by supports(args) and the device target internally. Only
    valid for the dense GEMM path: for that path this returns the identical set
    as the full-manifest scan (verified), avoiding the ~14s manifest build.
    """
    import cutlass.operators

    metadata_filter = (
        (lambda md: "EFC" in md.operator_class.__name__) if efc_only else None
    )
    return cutlass.operators.get_operators(
        args=args, target_sm=f"{cc}a", metadata_filter=metadata_filter
    )


def _filter_supported(kernels: Any, args: Any, cc: int) -> list[Any]:
    # [FUNCTION DOC] HELPER: Filters an iterable of operators down to those usable
    # for args on device cc. Skips operators needing a higher min compute capability,
    # whose A/B operand dtypes differ from the args, whose supported targets do not
    # cover the device target, or whose support check reports an error. Returns the
    # kept operators in input order.
    """Keep the kernels that support `args` on this device.

    Prunes by (A, B) operand dtype first (a cheap string compare) so the
    expensive supports() runs only on the ~hundreds that could possibly match --
    the ~290K manifest spans all dtypes, and a GEMM only matches its own.
    """
    device_target = _device_target(cc)
    dtype_a = _operand_dtype_str(getattr(args, "A", None))
    dtype_b = _operand_dtype_str(getattr(args, "B", None))
    out = []
    for kernel in kernels:
        if kernel.designed_for_min_cc > cc:
            continue
        operands = kernel.metadata.operands
        if (
            _operand_dtype_str(getattr(operands, "A", None)) != dtype_a
            or _operand_dtype_str(getattr(operands, "B", None)) != dtype_b
        ):
            continue
        if not device_target.supports_operators_from(kernel.metadata.supported_targets):
            continue
        if kernel.supports(args).error is not None:
            continue
        out.append(kernel)
    return out


def _manifest_candidates(args: Any, cc: int, efc_only: bool) -> list[Any]:
    # [FUNCTION DOC] HELPER: Fallback candidate search: scans the full manifest
    # (building it if necessary, which is slow), optionally keeping only EFC
    # operator classes, then applies the support filter. Returns the list.
    """Compatible operators via a full-manifest scan (last-resort fallback)."""
    kernels = _get_kernel_cache().values()
    if efc_only:
        kernels = (
            kernel
            for kernel in kernels
            if "EFC" in kernel.metadata.operator_class.__name__
        )
    return _filter_supported(kernels, args, cc)


def _blockscaled_provider_classes() -> list[Any]:
    # [FUNCTION DOC] HELPER: Returns the operator classes that generate
    # block-scaled GEMM operators: the cutlass built-in persistent block-scaled
    # operator and the vendored one, each included only if its import succeeds
    # (ImportError is swallowed). May return an empty list.
    """The operator classes that produce block-scaled (fp4/fp8) kernels.

    Both the cutlass built-in and the vendored provider matter: for NVFP4 the
    vendored kernels supply most of the working configs (60 of 96), so missing
    either under-generates and breaks scaled autotuning.
    """
    classes: list[Any] = []
    try:
        from cutlass.operators.providers.cutedsl.gemm.sm100_dense_blockscaled_static_persistent import (
            PersistentDenseBlockScaledGemmOperator,
        )

        classes.append(PersistentDenseBlockScaledGemmOperator)
    except ImportError:
        pass
    try:
        from torch._inductor.kernel.vendored_templates.cutedsl.wrappers.dense_blockscaled_gemm_kernel import (
            VendoredDenseBlockScaledGemmKernel,
        )

        classes.append(VendoredDenseBlockScaledGemmKernel)
    except ImportError:
        pass
    return classes


@functools.cache
def _blockscaled_operators() -> tuple:
    # [FUNCTION DOC] HELPER: Memoized. Generates, without a concrete target or
    # args, the complete block-scaled operator pool from every provider class and
    # returns it as a tuple. Cleared by clear_cache.
    """Generate the architecture-neutral block-scaled operator pool."""
    ops: list[Any] = []
    for cls in _blockscaled_provider_classes():
        ops.extend(cls.generate_operators(lambda md: True, args=None))
    return tuple(ops)


def _scaled_operand_type_signature(args: Any) -> tuple:
    # [FUNCTION DOC] HELPER: Summarizes the operand type recipe of scaled GEMM
    # runtime args as a hashable tuple: per-operand (dtype, scale dtype, scale mode,
    # swizzle mode) for A and B, the output dtype and the accumulator type.
    def operand_signature(operand: Any) -> tuple:
        # [FUNCTION DOC] HELPER: Returns (operand dtype, scale dtype, scale mode
        # string, swizzle string) for one operand; missing parts become None strings.
        scale = getattr(operand, "scale", None)
        return (
            _operand_dtype_str(operand),
            _operand_dtype_str(scale),
            str(getattr(operand, "mode", None)),
            str(getattr(operand, "swizzle", None)),
        )

    return (
        operand_signature(args.A),
        operand_signature(args.B),
        _operand_dtype_str(args.out),
        str(getattr(args, "accumulator_type", None)),
    )


def _scaled_metadata_type_signature(metadata: Any) -> tuple:
    # [FUNCTION DOC] HELPER: Same type-recipe signature as the args-based variant
    # but computed from an operator's metadata operands, so operators can be matched
    # against scaled GEMM args by equality.
    operands = metadata.operands

    def operand_signature(operand: Any) -> tuple:
        # [FUNCTION DOC] HELPER: Returns (operand dtype, scale dtype, scale mode
        # string, swizzle string) for one metadata operand.
        scale = getattr(operand, "scale", None)
        return (
            _operand_dtype_str(operand),
            _operand_dtype_str(scale),
            str(getattr(operand, "mode", None)),
            str(getattr(operand, "swizzle", None)),
        )

    return (
        operand_signature(operands.A),
        operand_signature(operands.B),
        _operand_dtype_str(operands.out),
        str(getattr(operands, "accumulator_type", None)),
    )


@functools.cache
def _blockscaled_manifest(cc: int, type_signature: tuple):
    # [FUNCTION DOC] HELPER: Memoized per (cc, type signature). Builds a cutlass
    # Manifest containing only the block-scaled pool operators whose metadata type
    # signature equals the requested one. Note cc does not affect the content (it
    # only keys the cache); target/shape filtering happens at query time.
    """Build a block-scaled manifest for one operand type recipe.

    The provider set is generated without a concrete target to preserve its
    complete design space. The manifest applies shape and target filtering when
    candidates are requested.
    """
    import cutlass.operators

    manifest = cutlass.operators.Manifest()
    manifest.add_operators(
        [
            op
            for op in _blockscaled_operators()
            if _scaled_metadata_type_signature(op.metadata) == type_signature
        ]
    )
    return manifest


def _scaled_candidates(args: Any, cc: int, efc_only: bool) -> list[Any]:
    # [FUNCTION DOC] HELPER: Finds operators for a scaled GEMM by filtering the
    # block-scaled manifest for the args' type recipe and target. For each resulting
    # vendored non-EFC operator it synthesizes a matching EFC operator (renamed,
    # with the EFC class). Returns only the EFC ones if efc_only, else originals plus
    # synthesized EFC ones; if no EFC variants exist, returns originals unless
    # efc_only. Falls back to the full-manifest scan when nothing matches.
    """Compatible operators for a scaled GEMM via direct block-scaled enumeration.

    get_operators(args=...) derives operand configs from the args and
    under-generates for scaled (e.g. ~36 of 96 valid fp4 kernels), so we scan the
    block-scaled sub-provider's full design space instead -- complete, and ~50x
    cheaper than the manifest. Falls back to the manifest if the provider is
    unavailable or nothing matches (e.g. a future non-block-scaled scaled dtype).
    """
    from torch._inductor.kernel.vendored_templates.cutedsl.wrappers.dense_blockscaled_gemm_kernel import (
        VendoredDenseBlockScaledGemmEFC,
        VendoredDenseBlockScaledGemmKernel,
    )

    manifest = _blockscaled_manifest(cc, _scaled_operand_type_signature(args))
    if manifest.operators:
        out = manifest.filter_operators(
            args=args,
            target_sm=f"{cc}a",
        )
        if out:
            efc = []
            for op in out:
                if op.metadata.operator_class is not VendoredDenseBlockScaledGemmKernel:
                    continue
                metadata = dataclasses.replace(
                    op.metadata,
                    operator_name=op.metadata.operator_name.replace(
                        "VendoredDenseBlockScaledGemmKernel",
                        "VendoredDenseBlockScaledGemmEFC",
                        1,
                    ),
                    operator_class=VendoredDenseBlockScaledGemmEFC,
                )
                efc.append(VendoredDenseBlockScaledGemmEFC(metadata))
            if efc:
                return efc if efc_only else [*out, *efc]
            if not efc_only:
                return out
    return _manifest_candidates(args, cc, efc_only)


def partition_compatible_kernels(
    args: Any,
    cc: int,
    classifier: Callable[[Any], int],
    num_buckets: int,
    efc_only: bool = False,
    candidate_source: Literal["args", "scaled", "manifest"] = "manifest",
    classifier_key: str | None = None,
) -> list[list[Any]]:
    # [FUNCTION DOC] Splits the operators compatible with args on device cc into
    # num_buckets lists using classifier (metadata -> bucket index, or negative to
    # drop). candidate_source picks the search strategy (args query, scaled
    # enumeration, or full manifest); any other value raises ValueError. Registers
    # every candidate in the by-name operator cache (side effect). Results are
    # memoized when both a signature and classifier_key are available; returns fresh
    # list copies so callers may mutate them.
    """Partition the operators compatible with `args` into N buckets.

    `classifier(metadata)` returns a bucket index in [0, num_buckets-1] or -1 to
    drop the kernel. `candidate_source` selects how the compatible set is found,
    all of which avoid the ~14s full manifest except "manifest":
      - "args"     dense GEMM: cheap args-filtered get_operators query
      - "scaled"   scaled GEMM: direct block-scaled sub-provider enumeration
      - "manifest" fallback (e.g. grouped GEMM): full-manifest scan
    `efc_only` restricts to epilogue-fusion-capable kernels (the addmm/bias
    path). Results are memoized per (shape, cc, num_buckets, efc_only, source).
    """
    sig = _partition_sig(args)
    cache_key = (
        (sig, cc, num_buckets, efc_only, candidate_source, classifier_key)
        if sig is not None and classifier_key is not None
        else None
    )
    if cache_key is not None:
        cached = _partition_cache.get(cache_key)
        if cached is not None:
            return [list(b) for b in cached]

    if candidate_source == "args":
        candidates = _args_query_candidates(args, cc, efc_only)
    elif candidate_source == "scaled":
        candidates = _scaled_candidates(args, cc, efc_only)
    elif candidate_source == "manifest":
        candidates = _manifest_candidates(args, cc, efc_only)
    else:
        raise ValueError(f"Unknown NVGEMM candidate source: {candidate_source}")

    buckets: list[list[Any]] = [[] for _ in range(num_buckets)]
    for kernel in candidates:
        # Share every compatible operator so a later get_kernel_by_name (e.g. the
        # scheduling max_active_clusters lookup for the chosen kernel) hits
        # without building the manifest.
        _ops_by_name.setdefault(kernel.metadata.operator_name, kernel)
        bucket = classifier(kernel.metadata)
        if bucket < 0:
            continue
        buckets[bucket].append(kernel)
    log.debug(
        "Partitioned %d compatible operators into buckets %s",
        len(candidates),
        [len(b) for b in buckets],
    )
    if cache_key is not None:
        _partition_cache[cache_key] = [list(b) for b in buckets]
    return buckets


def get_kernel_by_name(kernel_name: str) -> Any:
    # [FUNCTION DOC] Looks up an operator by name, first in the incrementally
    # filled by-name cache, then in the full manifest (which may trigger the slow
    # build). Returns the operator or None if unknown.
    """Get a kernel by name.

    Checks the incrementally-populated operator cache first (filled by the
    args-filtered queries during choice generation / precompile), falling back
    to the full manifest only if the name was never resolved via args.
    """
    kernel = _ops_by_name.get(kernel_name)
    if kernel is not None:
        return kernel
    return _get_kernel_cache().get(kernel_name)


def get_kernel_by_name_via_args(kernel_name: str, args: Any, cc: int) -> Any:
    # [FUNCTION DOC] Fast named lookup: returns the cached operator if present;
    # otherwise runs an args-filtered cutlass query for cc and registers all results
    # in the by-name cache (keeping existing entries). For vendored EFC block-scaled
    # names that are still missing, rebuilds plain GEMM arguments from args and
    # registers the scaled candidates. Returns the operator or None on a miss.
    """Fast single-kernel lookup via an args-filtered get_operators query.

    Passing concrete `args` (a cutlass RuntimeArguments) plus the device target
    prunes the operator space to the ~hundreds compatible with this exact GEMM
    in ~0.05s, versus ~14s to enumerate/construct the full ~294K-kernel manifest
    (`get_kernel_by_name`). Used by subprocess precompile workers, which need
    only the single named kernel and would otherwise each rebuild the whole
    manifest.

    A miss runs one args query and indexes every operator it returns into
    _ops_by_name, so the ~50ms cost is paid once per distinct operator (not once
    per lookup) and is amortized across all configs and shapes a persistent
    worker handles. Returns None if no operator matches the name (caller then
    falls back to the full manifest).
    """
    import cutlass.operators

    cached = _ops_by_name.get(kernel_name)
    if cached is not None:
        return cached

    ops = cutlass.operators.get_operators(args=args, target_sm=f"{cc}a")
    for op in ops:
        # setdefault: keep an already-cached (possibly already-compiled) object
        # rather than replacing it with a fresh instance from this query.
        _ops_by_name.setdefault(op.metadata.operator_name, op)
    if (
        kernel_name not in _ops_by_name
        and "VendoredDenseBlockScaledGemmEFC" in kernel_name
    ):
        from cutlass.operators.arguments import GemmArguments

        base_args = GemmArguments(
            args.A,
            args.B,
            args.out,
            accumulator_type=args.accumulator_type,
        )
        scaled_ops = _scaled_candidates(base_args, cc, efc_only=False)
        log.debug(
            "Scaled by-name fallback found %d candidates for %s",
            len(scaled_ops),
            kernel_name,
        )
        for op in scaled_ops:
            _ops_by_name.setdefault(op.metadata.operator_name, op)
    return _ops_by_name.get(kernel_name)


def ensure_cache_initialized() -> None:
    # [FUNCTION DOC] Forces the full manifest to be built now (side effect: fills
    # the global manifest cache); no return value.
    """Ensure the kernel cache is initialized."""
    _get_kernel_cache()


_efc_epilogue_cache: dict[tuple[str, str, tuple], Any] = {}


def clear_cache() -> None:
    # [FUNCTION DOC] Resets all module caches under the cache lock: the manifest,
    # the EFC-with-epilogue cache, the partition memo, the by-name operator cache and
    # the memoized block-scaled pool/manifest. No return value.
    """Clear all kernel caches."""
    global _kernel_by_name_cache, _efc_epilogue_cache
    with _cache_lock:
        _kernel_by_name_cache = None
        _efc_epilogue_cache = {}
        _partition_cache.clear()
        _ops_by_name.clear()
        _blockscaled_operators.cache_clear()
        _blockscaled_manifest.cache_clear()


class _NVGEMMCacheWrapper:
    # [CLASS DOC] HELPER: Stateless adapter exposing a cache_clear method so
    # clear_on_fresh_cache can register this module's clear_cache like a
    # functools-cached function.
    def cache_clear(self) -> None:
        # [FUNCTION DOC] HELPER: Delegates to clear_cache.
        clear_cache()


from torch._inductor.utils import clear_on_fresh_cache


clear_on_fresh_cache(_NVGEMMCacheWrapper())


def _meta_epilogue_metadata(epilogue_args: Any) -> Any:
    # [FUNCTION DOC] HELPER: Builds epilogue metadata for a cached EFC operator
    # from storage-free meta-device copies of the epilogue's tensors (real tensors,
    # compile-time tensor wrappers, or passthrough values), re-traces the epilogue
    # using the accumulator shape/element type from the original trace, and returns
    # the resulting metadata. Avoids retaining real tensors in the cache.
    """Build the cached kernel's epilogue metadata from meta-device tensors.

    The EFC kernel is cached in ``_efc_epilogue_cache`` and reused across every
    call; the real output/aux tensors are supplied per call via the runtime
    ``GemmArguments.epilogue``, so the cached kernel needs only the epilogue's
    shape/dtype/layout (its IR). Building it from the real ``EpilogueArguments``
    would make it retain those tensors forever -- ``EpilogueMetadata`` shares the
    arguments' ``tensors`` (wrapping the output ``D`` and aux inputs) and traced
    functor -- keeping the (often intermediate) output alive: a memory leak, and
    it breaks CUDA graph capture (a live graph-pool allocation that is not a
    tracked graph output). Re-trace the epilogue with meta tensors (carrying
    dtype/layout but no device storage; the tracer reads only their metadata) so
    the cached kernel holds nothing real.
    """
    from cutlass.operators.arguments.epilogue import EpilogueArguments
    from cutlass.operators.metadata import EpilogueMetadata
    from cutlass.operators.utils.tensor import TensorWrapper

    from .nv_universal_gemm_utils import cutlass_dtype_to_torch

    def _meta_like(t: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns a meta-device tensor with the same
        # shape, strides and dtype as t.
        return torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="meta")

    fake_kwargs = {}
    for name, val in epilogue_args.tensors.items():
        # Build a storage-free meta tensor with the same shape/stride/dtype.
        # A compile-time-only TensorWrapper (the subprocess-precompile path) has
        # no runtime tensor -- its `.runtime_tensor` property raises ValueError --
        # so read `_runtime_tensor` directly and fall back to the wrapper's own
        # shape/stride and its CUTLASS dtype.
        if isinstance(val, torch.Tensor):
            fake_kwargs[name] = _meta_like(val)
        elif isinstance(val, TensorWrapper) and isinstance(
            val._runtime_tensor, torch.Tensor
        ):
            fake_kwargs[name] = _meta_like(val._runtime_tensor)
        elif (
            isinstance(val, TensorWrapper)
            and (dtype := cutlass_dtype_to_torch(val.dtype)) is not None
        ):
            fake_kwargs[name] = torch.empty_strided(
                tuple(val.shape), tuple(val.stride), dtype=dtype, device="meta"
            )
        else:
            fake_kwargs[name] = val
    fake_args = EpilogueArguments(epilogue_args.epilogue_fn, **fake_kwargs)
    accum = epilogue_args.traced_epilogue.example_inputs["accum"]
    fake_args.trace(accum.shape, accum.element)
    return EpilogueMetadata.from_args(fake_args)


def get_efc_kernel_with_epilogue(
    efc_kernel_name: str,
    epilogue_args: Any,
    epilogue_source: str = "",
    base_kernel: Any | None = None,
) -> Any:
    # [FUNCTION DOC] Returns an epilogue-fusion operator bound to the given
    # epilogue, creating and caching it on first use. The cache key combines the
    # operator name, epilogue source text (derived from the args if empty) and the
    # epilogue tensor signature. The base operator comes from base_kernel or a
    # by-name lookup; returns None if it cannot be found. The new operator is built
    # by copying the base metadata with meta-based epilogue metadata. Holds the
    # cache lock throughout.
    """Get (or create and cache) an EFC kernel bound to a specific epilogue.

    epilogue_source is preferred over inspect.getsource — generated functions
    produce unstable source strings that can't be hashed reliably.

    base_kernel: pre-resolved base EFC kernel (e.g. from the args-filtered fast
    lookup in a subprocess worker). When None, falls back to the full manifest.
    """
    if not epilogue_source:
        epilogue_source = str(epilogue_args) if epilogue_args is not None else ""

    cache_key = (
        efc_kernel_name,
        epilogue_source,
        _epilogue_args_signature(epilogue_args),
    )

    with _cache_lock:
        if cache_key in _efc_epilogue_cache:
            log.debug("EFC kernel with epilogue found in cache: %s", efc_kernel_name)
            return _efc_epilogue_cache[cache_key]

        if base_kernel is None:
            # Prefer the args-resolved operator cache (populated by choice
            # generation); only build the full manifest if the name was never
            # seen via args.
            base_kernel = get_kernel_by_name(efc_kernel_name)
        if base_kernel is None:
            log.debug("Base EFC kernel not found: %s", efc_kernel_name)
            return None

        from cutlass.operators.metadata import OperatorMetadata

        epilogue_metadata = _meta_epilogue_metadata(epilogue_args)

        base_metadata = base_kernel.metadata
        new_metadata = OperatorMetadata(
            operands=base_metadata.operands,
            design=base_metadata.design,
            operator_name=base_metadata.operator_name,
            operator_class=base_metadata.operator_class,
            supported_targets=base_metadata.supported_targets,
            epilogue=epilogue_metadata,
        )

        kernel_class = base_metadata.operator_class
        new_kernel = kernel_class(new_metadata)

        _efc_epilogue_cache[cache_key] = new_kernel
        log.debug("Created and cached EFC kernel with epilogue: %s", efc_kernel_name)

        return new_kernel
