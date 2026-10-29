# [MODULE DOC] rocSHMEM Triton bindings for AMD ROCm builds, the counterpart of the
# NVSHMEM module. Provides a device bitcode library finder, a kernel registry, a post-
# compile init hook (reusing the c10d module-init entry point), the requires_rocshmem
# decorator, and (when Triton is available) workgroup-scoped device functions for
# put/get, signalling, waiting, fences, PE queries and barriers. Collectives and
# signal_op are stubs that fail at compile time. Import-time side effects: module
# logger; conditionally imports Triton and defines kernels. Only intended to be
# imported on ROCm.
"""
rocSHMEM Triton integration for AMD GPU (ROCm) builds.

This module provides the rocSHMEM-specific counterpart to the NVSHMEM Triton
integration in _nvshmem_triton.py.  It is only imported on ROCm builds
(i.e. when ``torch.version.hip is not None``).
"""

import logging
import os
import sysconfig
from typing import Any

import torch
from torch.distributed._symmetric_memory._shmem_triton_utils import (
    run_shmem_init_hook,
    ShmemKernelRegistry,
)
from torch.utils._triton import has_triton


logger = logging.getLogger(__name__)


class RocshmemLibFinder:
    # [CLASS DOC] Finds the architecture-specific rocSHMEM device bitcode file (named with
    # the GPU gfx architecture). Looks first in the directory from ROCSHMEM_LIB_DIR if
    # set, else in site-packages, the ROCm install lib directory (ROCM_PATH or /opt/rocm),
    # and system library paths. The found path is cached in a class attribute.
    """
    Find the architecture-specific rocSHMEM device bitcode library.

    Environment variable:
        ``ROCSHMEM_LIB_DIR`` (Optional[str]): directory containing
        ``librocshmem_device_{arch}.bc``.  When not set, the standard
        ROCm installation at ``/opt/rocm/lib`` is searched.

    Example::
        export ROCSHMEM_LIB_DIR=/opt/rocm/lib
    """

    found_device_lib_path: str | None = None

    @classmethod
    def find_device_library(cls) -> str:
        # [FUNCTION DOC] Returns the path to the rocSHMEM device library for GPU 0. Uses the
        # cached value if present. Raises RuntimeError if no GPU is available, if the library
        # is missing from ROCSHMEM_LIB_DIR, or if it is not found in any search path. Reads
        # the device architecture name (text before the first colon of the arch string), logs
        # findings, and caches the result.
        if cls.found_device_lib_path is not None:
            return cls.found_device_lib_path

        if not torch.cuda.is_available():
            raise RuntimeError(
                "ROCm/CUDA not available — cannot detect GPU architecture"
            )

        props = torch.cuda.get_device_properties(0)
        # gcnArchName returns e.g. "gfx942:sramecc+:xnack-"
        arch = props.gcnArchName.split(":")[0]
        logger.info("Detected GPU architecture: %s", arch)

        lib_name = f"librocshmem_device_{arch}.bc"

        search_paths = [
            os.path.join(sysconfig.get_path("purelib"), "amd", "rocshmem", "lib"),
            os.path.join(os.environ.get("ROCM_PATH") or "/opt/rocm", "lib"),
            "/usr/local/lib",
            "/usr/lib",
        ]

        user_lib_dir = os.environ.get("ROCSHMEM_LIB_DIR")
        if user_lib_dir is not None:
            lib_path = os.path.join(user_lib_dir, lib_name)
            if not os.path.exists(lib_path):
                raise RuntimeError(
                    f"rocSHMEM device library not found at ROCSHMEM_LIB_DIR: {lib_path}"
                )
            logger.info("Found rocSHMEM device library: %s", lib_path)
            cls.found_device_lib_path = lib_path
            return lib_path

        lib_path = None
        for path in search_paths:
            candidate = os.path.join(path, lib_name)
            if os.path.exists(candidate):
                lib_path = candidate
                break

        if lib_path is None:
            raise RuntimeError(
                f"rocSHMEM device library '{lib_name}' not found.\n"
                f"Searched: {search_paths}\n"
                "Set ROCSHMEM_LIB_DIR to the directory containing it."
            )
        logger.info("Found rocSHMEM device library: %s", lib_path)
        cls.found_device_lib_path = lib_path
        return lib_path


class RocshmemKernelRegistry(ShmemKernelRegistry):
    # [CLASS DOC] Kernel-name registry for rocSHMEM, a subclass of the shared registry
    # with its own separate class-level mapping so it does not mix with other backends.
    """Track Triton kernels that need rocSHMEM HIP-module initialization."""

    _to_init: dict[str, Any] = {}


def _rocshmem_init_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] HELPER: Post-compile hook that initializes rocSHMEM device state in
    # the compiled module by delegating to the shared hook logic with the rocSHMEM
    # registry and the c10d module-init entry point (the same binding used for NVSHMEM).
    """
    Post-compile hook that initializes rocSHMEM device context in the
    compiled HIP module using the existing c10d NVSHMEM entrypoint.
    """
    from torch._C._distributed_c10d import _nvshmemx_cumodule_init

    run_shmem_init_hook(
        kwargs=kwargs,
        registry=RocshmemKernelRegistry,
        module_init=_nvshmemx_cumodule_init,
        logger=logger,
    )


def requires_rocshmem(  # type: ignore[no-untyped-def]
    jit_func,
):
    # [FUNCTION DOC] Decorator for a Triton JIT function that needs rocSHMEM device APIs.
    # Delegates to the shared decorator builder with the rocSHMEM library finder, extern
    # lib key and registry, and the init hook. Raises TypeError if the input is not a JIT
    # function (message prefixed with the decorator name). Side effects: registers the
    # kernel and sets the global Triton post-compile hook. Returns the wrapped kernel.
    """
    Decorator to mark a Triton kernel as requiring rocSHMEM device APIs.

    Finds the architecture-specific rocSHMEM bitcode library, registers
    the kernel for post-compile HIP-module initialization, and wraps the
    function so that ``extern_libs`` is injected automatically.

    Example::

        @requires_rocshmem
        @triton.jit
        def my_kernel(...):
            pe = rocshmem_my_pe()
            rocshmem_putmem_wg(dest, src, nbytes, target_pe)

    Set ``ROCSHMEM_LIB_DIR`` to override the default library search path.
    """
    from torch.distributed._symmetric_memory._shmem_triton_utils import (
        build_requires_shmem_decorator,
    )

    return build_requires_shmem_decorator(
        jit_func=jit_func,
        find_device_library=RocshmemLibFinder.find_device_library,
        extern_libs_key="rocshmem",
        registry=RocshmemKernelRegistry,
        init_hook=_rocshmem_init_hook,
        error_prefix="@requires_rocshmem",
    )


if has_triton():
    import triton
    import triton.language as tl
    from triton.language import core

    # -----------------------------------------------------------------------
    # rocSHMEM device API — Triton-callable device functions.
    #
    # All symbol names are the direct rocSHMEM device-bitcode names; no
    # NVSHMEM → rocSHMEM translation layer is needed here.
    #
    # RMA (block/wg-scoped):
    #   nvshmemx_putmem_block  → rocshmem_putmem_wg
    #   nvshmemx_getmem_block  → rocshmem_getmem_wg
    #   nvshmemx_getmem_nbi_block → rocshmem_getmem_nbi_wg
    #   nvshmemx_putmem_signal_block → rocshmem_putmem_signal_wg
    # Wait / signal:
    #   nvshmem_int_wait_until    → rocshmem_int_wait_until
    #   nvshmem_signal_wait_until → rocshmem_uint64_wait_until
    # Memory ordering, PE info, barriers: names are identical.
    # -----------------------------------------------------------------------

    @triton.jit
    def put(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: blocking put of nelems elements from local
        # source to dest on remote PE pe. Asserts equal tensor types, converts to bytes and
        # calls the workgroup put wrapper. Returns int32 status.
        """Put *nelems* elements from local *source* to *dest* on remote *pe*."""
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return _putmem_wg(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def _putmem_wg(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper for the rocSHMEM workgroup put (int64
        # addresses, byte count, int32 PE); int32 status. Not pure.
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int32"),
                ): ("rocshmem_putmem_wg", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def get(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: blocking get of nelems elements from remote
        # PE source into local dest. Asserts equal types, converts to bytes, calls the
        # workgroup get wrapper; int32 status.
        """Get *nelems* elements from *source* on remote *pe* into local *dest* (blocking)."""
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return _getmem_wg(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def _getmem_wg(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper for the rocSHMEM workgroup get; int32 status.
        # Not pure.
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int32"),
                ): ("rocshmem_getmem_wg", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def get_nbi(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: non-blocking get from remote PE; completion
        # requires quiet. Asserts equal types, converts to bytes, calls the non-blocking
        # workgroup get wrapper; int32 status.
        """Non-blocking get; call quiet() for completion."""
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return _getmem_nbi_wg(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def _getmem_nbi_wg(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper for the rocSHMEM non-blocking workgroup get;
        # int32 status. Not pure.
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int32"),
                ): ("rocshmem_getmem_nbi_wg", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def putmem_signal_block(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
    ):
        # [FUNCTION DOC] Triton device function: copies size_bytes to the destination on PE pe
        # and atomically updates a signal variable there with a value and operation code.
        # Widens the signal value to unsigned 64 bits and calls the workgroup put-with-signal
        # wrapper; int32 status.
        """Put data to remote PE and atomically update a signal variable."""
        sig_val = 0 << 32 | sig_val
        return _putmem_signal_wg(
            dst.to(tl.int64),
            src.to(tl.int64),
            size_bytes.to(tl.int64),
            signal.to(tl.int64),
            sig_val.to(tl.uint64),
            sig_op,
            pe,
        )

    @core.extern
    def _putmem_signal_wg(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
        _semantic=None,
    ):
        # [FUNCTION DOC] HELPER: Extern wrapper for the rocSHMEM workgroup put-with-signal
        # (addresses, size, signal address, uint64 value, operation, PE); int32 status. Not
        # pure.
        return core.extern_elementwise(
            "",
            "",
            [dst, src, size_bytes, signal, sig_val, sig_op, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("uint64"),
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("rocshmem_putmem_signal_wg", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def wait_until(ivar, cmp_op, cmp_val):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: blocks until the 32-bit integer variable at
        # ivar satisfies the comparison with cmp_val. Statically asserts a 4-byte element
        # type; int32 status.
        """Block until *ivar* satisfies the comparison condition."""
        tl.static_assert(
            ivar.type.element_ty.itemsize == 4,
            "wait_until expects a 32-bit type for the synchronization variable",
        )
        return _int_wait_until(ivar.to(tl.int64), cmp_op, cmp_val)

    @core.extern
    def _int_wait_until(ivar, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper for the rocSHMEM integer wait-until; int32
        # status. Not pure.
        return core.extern_elementwise(
            "",
            "",
            [ivar, cmp, cmp_val],
            {
                (core.dtype("int64"), core.dtype("int32"), core.dtype("int32")): (
                    "rocshmem_int_wait_until",
                    core.dtype("int32"),
                )
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def signal_wait_until(signal, cmp, cmp_val):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: blocks until a 64-bit unsigned signal
        # variable satisfies the comparison with cmp_val (widened to unsigned 64 bits); int32
        # status.
        """Block until a uint64 signal variable satisfies the comparison condition."""
        cmp_val = 0 << 32 | cmp_val
        return _uint64_wait_until(signal.to(tl.int64), cmp, cmp_val.to(tl.uint64))

    @core.extern
    def _uint64_wait_until(signal, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper for the rocSHMEM uint64 wait-until; int32
        # status. Not pure.
        return core.extern_elementwise(
            "",
            "",
            [signal, cmp, cmp_val],
            {
                (core.dtype("int64"), core.dtype("int32"), core.dtype("uint64")): (
                    "rocshmem_uint64_wait_until",
                    core.dtype("int32"),
                )
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def signal_op(sig_addr, signal, sig_op, pe):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function stub: always fails at compile time with a
        # static assertion, since rocSHMEM device bitcode has no signal operation. The message
        # suggests the atomic set or add calls instead.
        """Not available in current rocSHMEM device bitcode."""
        tl.static_assert(
            False,
            "rocshmem has no device-bitcode equivalent for signal_op. "
            "Use rocshmem_uint64_atomic_set or rocshmem_uint64_atomic_add instead.",
        )

    @core.extern
    def fence(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: fence ordering issued remote operations per
        # target PE; int32 status.
        """Ensure ordering of issued remote-memory operations to each target PE."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_fence", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def quiet(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: waits for completion of all outstanding
        # remote memory operations; int32 status.
        """Wait for completion of all outstanding remote-memory operations."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_quiet", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def my_pe(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: returns the calling PE number (int32). Pure.
        """Return the PE number of the calling PE."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_my_pe", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def n_pes(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: returns the total number of PEs (int32).
        # Pure.
        """Return the total number of PEs."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_n_pes", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def barrier_all(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: workgroup-scoped barrier across all PEs with
        # completion guarantee; int32 status.
        """Barrier across all PEs with completion guarantee (workgroup-scoped)."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_barrier_all_wg", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def sync_all(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: workgroup-scoped lightweight synchronization
        # across all PEs; int32 status.
        """Lightweight synchronization barrier across all PEs (workgroup-scoped)."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_sync_all_wg", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    # Collective stubs: rocSHMEM *_wg collectives are not exposed in current
    # device bitcode, so Triton kernels cannot call them yet.

    @triton.jit
    def alltoall(team, dest, source, nelems_per_pe):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function stub: always fails at compile time (static
        # assertion) because the workgroup all-to-all is absent from device bitcode; use the
        # host-side stream API instead. Signature mirrors the NVSHMEM backend.
        """Not available: rocshmem_alltoallmem_wg is not in current device bitcode."""
        tl.static_assert(
            False,
            "rocshmem_alltoallmem_wg is not available in current device bitcode. "
            "Use host-side rocshmem_alltoallmem_on_stream instead.",
        )

    @triton.jit
    def broadcast(team, dest, source, nelems, pe_root):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function stub: always fails at compile time because the
        # workgroup broadcast is not in device bitcode; use the host-side API. Signature
        # mirrors the NVSHMEM backend.
        """Not available: rocshmem_broadcastmem_wg is not in current device bitcode."""
        tl.static_assert(
            False,
            "rocshmem_broadcastmem_wg is not available in current device bitcode. "
            "Use host-side rocshmem_broadcastmem_on_stream instead.",
        )

    @triton.jit
    def reduce(team, dest, source, nreduce, operation: tl.constexpr):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function stub: always fails at compile time because
        # team reduce is not in device bitcode; use the host-side API. Signature mirrors the
        # NVSHMEM backend.
        """Not available: rocshmem team reduce wg ops are not in current device bitcode."""
        tl.static_assert(
            False,
            "rocshmem team reduce is not available in current device bitcode. "
            "Use host-side rocshmem reduce API instead.",
        )
