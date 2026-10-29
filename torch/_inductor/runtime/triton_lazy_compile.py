# [MODULE DOC] Runtime support for lazily compiling and autotuning Triton kernels when the C++
# wrapper is used without compile-time autotuning. Provides start_kernel_compile (begin async
# compile), run_triton_kernel_with_autotune (wait, autotune, return TritonKernelCompileResult),
# and helpers. No import-time side effects beyond creating the module logger.
"""
Lazy Triton kernel compilation for C++ wrapper.

This module provides functionality to compile and autotune Triton kernels at runtime
when the C++ wrapper is used without autotune_at_compile_time.

The workflow is:
1. At model initialization: Call start_kernel_compile() for all kernels to start
   parallel compilation using multi-process async_compile
2. At kernel execution time: Call run_triton_kernel_with_autotune() which waits
   for the specific kernel to be ready, then runs it with autotuning
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

from .hints import (
    TRITON_DEFAULT_BLOCK_SIZES,
    TRITON_DEFAULT_RSPLIT,
    TRITON_DEFAULT_RSPLIT_SIZE,
)
from .triton_heuristics import CachingAutotuner


log = logging.getLogger(__name__)


@dataclasses.dataclass
class TritonKernelCompileResult:
    # [CLASS DOC] Plain data record returned after lazy autotuning of a Triton kernel: binary path,
    # mangled name, warps, shared memory, per-subkernel block size lists (x, y, z, r0), reduction
    # split settings, optional precomputed grid config index and scratch sizes.
    cubin_path: str
    mangled_name: str
    num_warps: int
    shared_mem: int
    xblocks: list[int]
    yblocks: list[int]
    zblocks: list[int]
    r0blocks: list[int]
    rsplit: int
    rsplit_size: int
    config_index: int | None
    global_scratch: int | None
    profile_scratch: int | None


_async_compile: Any = None


def _get_async_compile() -> Any:
    # [FUNCTION DOC] HELPER: Returns the module-level AsyncCompile, creating it lazily on first
    # call (side effect: sets the global).
    """Get or create the shared AsyncCompile instance."""
    global _async_compile
    if _async_compile is None:
        from torch._inductor.async_compile import AsyncCompile

        _async_compile = AsyncCompile()
    return _async_compile


def _wrap_tma_args(args: list[Any], kernel_fn: CachingAutotuner) -> list[Any]:
    # [FUNCTION DOC] HELPER: Scans the kernel's triton signature (skipping constexpr entries when
    # counting positions) for tensordesc entries and replaces the matching tensor args with
    # TensorDescriptor objects built using the block shape parsed from the signature string.
    # Returns the original list if no TMA args exist, else a new list. Raises an error for the
    # experimental nvTmaDesc type or when an arg index is out of range.
    """Wrap tensor args with TMA descriptors where the signature requires them."""
    signature = kernel_fn.triton_meta.get("signature", {})
    sig_items = list(signature.items())

    # Track args index separately from sig_items index since the signature
    # may include constexpr entries that are not present in args.
    tma_indices = []
    arg_idx = 0
    for name, sig_type in sig_items:
        if isinstance(sig_type, str) and sig_type == "constexpr":
            continue
        if isinstance(sig_type, str) and sig_type == "nvTmaDesc":
            raise RuntimeError(
                f"nvTmaDesc (experimental TMA API) is not supported in lazy compile "
                f"for arg '{name}'. Use the stable tensordesc API instead."
            )
        if isinstance(sig_type, str) and sig_type.startswith("tensordesc<"):
            tma_indices.append((arg_idx, name, sig_type))
        arg_idx += 1

    if not tma_indices:
        return args

    from triton.tools.tensor_descriptor import TensorDescriptor

    wrapped = list(args)
    for arg_idx, name, sig_type in tma_indices:
        if arg_idx >= len(wrapped):
            raise RuntimeError(
                f"TMA arg index {arg_idx} for '{name}' exceeds arg count {len(wrapped)}"
            )
        tensor = wrapped[arg_idx]
        # Parse block_shape from tensordesc<dtype[dim0, dim1, ...]>
        match = re.match(r"tensordesc<[^[]*\[([^\]]*)\]", sig_type)
        if match:
            block_shape = [int(x.strip()) for x in match.group(1).split(",")]
            wrapped[arg_idx] = TensorDescriptor.from_tensor(tensor, block_shape)

    return wrapped


def start_kernel_compile(
    pending_kernels: dict[str, Any], kernel_name: str, kernel_source: str
) -> None:
    # [FUNCTION DOC] Starts background compilation of one kernel at model init. Does nothing if the
    # name is already in the pending mapping. Otherwise evaluates the kernel source text (an
    # async_compile call expression) with the shared AsyncCompile, and stores the resulting
    # future or autotuner in the pending mapping under the kernel name. Does not wait.
    # Side effects: mutates pending_kernels, may create the global AsyncCompile, evaluates code.
    """
    This function is called from C++ at model initialization time for each kernel.
    It starts the compilation in a background process but does NOT wait for it.
    The actual kernel execution happens later in run_triton_kernel_with_autotune().

    The pending_kernels dict is per-module, created in C++ and passed through
    to avoid global state collisions across compiled modules.
    """
    if kernel_name in pending_kernels:
        return

    async_compile = _get_async_compile()  # noqa: F841 (used by eval below)

    # Evaluate the kernel source to get the Future or CachingAutotuner
    # The kernel_source is like: async_compile.triton('name', '''...''', ...)
    kernel_obj = eval(kernel_source.strip())

    pending_kernels[kernel_name] = kernel_obj


def run_triton_kernel_with_autotune(
    pending_kernels: dict[str, Any],
    kernel_name: str,
    stream: Any,
    args: list[Any],
) -> TritonKernelCompileResult:
    # [FUNCTION DOC] Looks up the pending kernel by name (raises an error if missing or of an
    # unexpected type), resolves a future to a CachingAutotuner, wraps TMA args, and runs it on
    # the given stream to trigger autotuning. Then reads cubin path, mangled name, warps and
    # shared memory from CudaKernelParamCache (raises an error if absent), derives per-kernel
    # block sizes, split sizes and the precomputed grid index from the winning launcher config,
    # and returns them as a TritonKernelCompileResult. Side effects: sets store_cubin in the
    # kernel's inductor_meta, runs the kernel, emits a debug log.
    """
    Run a Triton kernel with full autotuning using actual tensor arguments.
    """
    from torch._inductor.codecache import CodeCacheFuture, CudaKernelParamCache
    from torch._inductor.runtime.triton_heuristics import config_to_dict

    if kernel_name not in pending_kernels:
        raise RuntimeError(f"Kernel {kernel_name} not found in pending kernels.")
    kernel_obj = pending_kernels[kernel_name]

    if isinstance(kernel_obj, CodeCacheFuture):
        kernel_fn = kernel_obj.result()
    elif isinstance(kernel_obj, CachingAutotuner):
        kernel_fn = kernel_obj
    else:
        raise RuntimeError(f"Unexpected kernel object type: {type(kernel_obj)}")

    if not isinstance(kernel_fn, CachingAutotuner):
        raise AssertionError(f"Expected CachingAutotuner, got {type(kernel_fn)}")

    inductor_meta = kernel_fn.inductor_meta
    inductor_meta["store_cubin"] = True

    # For TMA kernels, wrap tensor args with TMA descriptors
    args = _wrap_tma_args(args, kernel_fn)

    # Run the kernel with the provided arguments
    # This will trigger autotuning if there are multiple configs
    kernel_fn.run(*args, stream=stream)
    if not kernel_fn.launchers:
        raise RuntimeError("Kernel run did not produce any launchers")
    launcher = kernel_fn.launchers[0]

    cached_params: dict[str, Any] | None = CudaKernelParamCache.get(kernel_name)
    if cached_params is None:
        raise RuntimeError(f"Failed to get cached params for kernel {kernel_name}")

    from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name

    cubin_path_name = get_cpp_wrapper_cubin_path_name()
    for key_name in (cubin_path_name, "mangled_name", "num_warps", "shared_mem"):
        if key_name not in cached_params:
            raise RuntimeError(
                f"{key_name} not found in cached params for {kernel_name}"
            )
    cubin_path = cached_params[cubin_path_name]
    if not isinstance(cubin_path, str):
        raise AssertionError(f"expected cubin_path to be str, got {type(cubin_path)}")
    runtime_bin_path = cached_params.get("runtime_bin_path", cubin_path)
    if not isinstance(runtime_bin_path, str):
        raise AssertionError(
            f"expected runtime_bin_path to be str, got {type(runtime_bin_path)}"
        )
    mangled_name = cached_params["mangled_name"]
    num_warps = cached_params["num_warps"]
    shared_mem = cached_params["shared_mem"]

    config = config_to_dict(launcher.config) if launcher.config else {}

    # For combo/foreach kernels, the autotuned config may have empty kwargs
    # (e.g., the foreach heuristic only tunes num_warps, not XBLOCK).
    # In that case, use the default_config from combo_grid_meta
    combo_grid_meta = inductor_meta.get("combo_grid_meta") if inductor_meta else None
    default_config = combo_grid_meta.get("default_config") if combo_grid_meta else None
    if default_config:
        config = {**default_config, **config}

    # Extract per-subkernel block sizes for combo kernels, or single block sizes.
    num_kernels = combo_grid_meta.get("num_kernels", 1) if combo_grid_meta else 1
    if num_kernels > 1 and "XBLOCK_0" in config:
        xblocks = [
            config.get(f"XBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["XBLOCK"])
            for i in range(num_kernels)
        ]
        yblocks = [
            config.get(f"YBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["YBLOCK"])
            for i in range(num_kernels)
        ]
        zblocks = [
            config.get(f"ZBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["ZBLOCK"])
            for i in range(num_kernels)
        ]
        r0blocks = [
            config.get(f"R0_BLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["R0_BLOCK"])
            for i in range(num_kernels)
        ]
    else:
        xblocks = [config.get("XBLOCK", TRITON_DEFAULT_BLOCK_SIZES["XBLOCK"])]
        yblocks = [config.get("YBLOCK", TRITON_DEFAULT_BLOCK_SIZES["YBLOCK"])]
        zblocks = [config.get("ZBLOCK", TRITON_DEFAULT_BLOCK_SIZES["ZBLOCK"])]
        r0blocks = [config.get("R0_BLOCK", TRITON_DEFAULT_BLOCK_SIZES["R0_BLOCK"])]
    rsplit = config.get("RSPLIT", TRITON_DEFAULT_RSPLIT)
    rsplit_size = config.get("RSPLIT_SIZE", TRITON_DEFAULT_RSPLIT_SIZE)

    config_index = None
    grid_type = inductor_meta.get("grid_type") if inductor_meta else None
    if grid_type == "PrecomputedGrid" and inductor_meta:
        # PrecomputedGrid selects one of precomputed_grids. We use config_index
        # to remember which grid is chosen.
        precomputed_grids = inductor_meta.get("precomputed_grids", [])
        for idx, entry in enumerate(precomputed_grids):
            entry_config = entry.get("config", {})
            if all(config.get(k) == v for k, v in entry_config.items()):
                config_index = idx
                break

    global_scratch: int | None = cached_params.get("global_scratch")
    profile_scratch: int | None = cached_params.get("profile_scratch")

    log.debug(
        "Successfully autotuned Triton kernel: cubin_path=%s, "
        "runtime_bin_path=%s, mangled_name=%s, "
        "num_warps=%d, shared_mem=%d, xblocks=%s, yblocks=%s, zblocks=%s, r0blocks=%s, "
        "rsplit=%d, rsplit_size=%d, config_index=%s, global_scratch=%s, profile_scratch=%s",
        cubin_path,
        runtime_bin_path,
        mangled_name,
        num_warps,
        shared_mem,
        xblocks,
        yblocks,
        zblocks,
        r0blocks,
        rsplit,
        rsplit_size,
        config_index,
        global_scratch,
        profile_scratch,
    )

    result = TritonKernelCompileResult(
        cubin_path=runtime_bin_path,
        mangled_name=mangled_name,
        num_warps=num_warps,
        shared_mem=shared_mem,
        xblocks=xblocks,
        yblocks=yblocks,
        zblocks=zblocks,
        r0blocks=r0blocks,
        rsplit=rsplit,
        rsplit_size=rsplit_size,
        config_index=config_index,
        global_scratch=global_scratch,
        profile_scratch=profile_scratch,
    )
    return result
