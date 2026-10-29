# [MODULE DOC] Triton device-side bindings for NVSHMEM on CUDA. Provides a finder for
# the NVSHMEM device bitcode library, a kernel-name registry plus a post-compile hook
# that initializes NVSHMEM in each compiled kernel module, the requires_nvshmem
# decorator, and (only when Triton is available) JIT functions and extern wrappers for
# one-sided put/get, signalling, waiting, fences, barriers, collectives and atomics.
# Import-time side effects: creates a module logger; conditionally imports Triton and
# defines kernels.
import logging
import os
import subprocess
import sysconfig
from typing import Any

import torch.distributed as dist
from torch.utils._triton import has_triton


logger = logging.getLogger(__name__)


class NvshmemLibFinder:
    # [CLASS DOC] Locates the libnvshmem_device bitcode file. Search order: the directory
    # in NVSHMEM_LIB_DIR if set (error if file missing there), then the Python site-
    # packages nvidia nvshmem wheel directory, common system paths, and RPATH/RUNPATH
    # entries read (via the external readelf tool) from torch libtorch_nvshmem shared
    # library. The found path is cached in a class attribute for reuse.
    """
    A class to find path to the NVSHMEM device library.

    Environment variable:

    `NVSHMEM_LIB_DIR` (Optional[str]): The directory where the NVSHMEM device
    library is located. If not provided, it will use the default path where
    NVSHMEM wheel is installed, or search for the library in common system
    paths.
    """

    # Class variable to store the found library path for reuse
    found_device_lib_path: str | None = None

    @classmethod
    def find_device_library(cls) -> str:
        # [FUNCTION DOC] Returns the full path to the NVSHMEM device library file, using the
        # cached value when present. Raises RuntimeError if the user-specified directory lacks
        # the file or if no search location contains it (message lists searched paths). Side
        # effects: reads the environment, may run readelf in a subprocess (failures and a
        # missing torch import are ignored) and caches the result.
        """
        Find the path to the NVSHMEM device library.

        Returns:
            str: The path to libnvshmem_device.bc (included).
        """
        if cls.found_device_lib_path is not None:
            # Return the cached path if it exists
            return cls.found_device_lib_path

        # First, check if the user has specified a custom library path
        user_lib_dir = os.environ.get("NVSHMEM_LIB_DIR", None)
        if user_lib_dir is not None:
            lib_path = os.path.join(user_lib_dir, "libnvshmem_device.bc")
            if not os.path.exists(lib_path):
                raise RuntimeError(
                    f"NVSHMEM device library not found at specified path: {user_lib_dir}"
                )
            cls.found_device_lib_path = lib_path
            return lib_path

        # Otherwise, search for the library in the default installation paths
        paths = [
            os.path.join(sysconfig.get_path("purelib"), "nvidia", "nvshmem", "lib")
        ]

        # Add common system installation paths
        common_paths = [
            "/usr/local/lib",
            "/usr/lib",
            "/opt/nvidia/nvshmem/lib",
        ]
        paths.extend(common_paths)

        try:
            import torch

            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            so_path = os.path.join(torch_lib, "libtorch_nvshmem.so")

            if os.path.exists(so_path):
                try:
                    result = subprocess.run(
                        ["readelf", "-d", so_path],
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    for line in result.stdout.splitlines():
                        if ("RPATH" in line or "RUNPATH" in line) and "[" in line:
                            rpath = line.split("[", 1)[1].split("]", 1)[0]
                            for p in rpath.split(":"):
                                p = p.strip().replace("$ORIGIN", torch_lib)
                                if p and p not in paths:
                                    paths.append(p)
                except subprocess.CalledProcessError:
                    pass

        except ImportError:
            pass

        for path in paths:
            device_lib = os.path.join(path, "libnvshmem_device.bc")
            if os.path.exists(device_lib):
                cls.found_device_lib_path = device_lib
                return device_lib

        raise RuntimeError(f"NVSHMEM device library not found. Searched: {paths}")


def enable_triton(lib_dir: str | None = None) -> dict[str, str]:
    # [FUNCTION DOC] Deprecated stub: always raises NotImplementedError telling users to
    # decorate kernels with requires_nvshmem instead. The lib_dir argument is unused.
    raise NotImplementedError(
        "`enable_triton` is deprecated. "
        "If you need NVSHMEM device function support for Triton, "
        "please use `@requires_nvshmem` to decorate your Triton kernel. ",
    )


class NvshmemKernelRegistry:
    # [CLASS DOC] Registry of Triton kernel function names that require NVSHMEM module
    # initialization. State is a class-level mapping; classmethods register, deregister
    # and has manage it. Duplicates the shared registry class in the utils module.
    """
    A class to register kernel functions that ** require NVSHMEM initialization **
    """

    # Class variable to store the functions to be initialized
    _to_init: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> None:
        # [FUNCTION DOC] Adds the kernel function name to the registry (no effect if present).
        """
        Register a kernel function with the given name.

        Args:
            name (str): The name of the kernel function.
        """
        cls._to_init.setdefault(name)

    @classmethod
    def deregister(cls, name: str) -> None:
        # [FUNCTION DOC] Removes the name from the registry if present; no error otherwise.
        """
        Deregister a kernel function with the given name.

        Args:
            name (str): The name of the kernel function.
        """
        cls._to_init.pop(name, None)

    @classmethod
    def has(cls, name: str) -> bool:
        # [FUNCTION DOC] Returns True if the name is registered.
        """
        Check if a kernel function with the given name is registered.

        Args:
            name (str): The name of the kernel function.

        Returns:
            bool: True if the kernel function is registered, False otherwise.
        """
        return name in cls._to_init


def _nvshmem_init_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] HELPER: Triton post-compile hook (keyword arguments from Triton). For
    # a compiled function whose name is registered, finds the compiled kernel in the
    # device cache by key, forces its loading and initializes NVSHMEM device state in its
    # CUDA module via the C++ binding; if no kernel was created, logs a warning. Does
    # nothing for unregistered functions. Duplicates the shared hook logic in the utils
    # module.
    """
    A hook function to initialize the CUModule created by `triton.jit` with
    NVSHMEM device context
    """
    from torch._C._distributed_c10d import _nvshmemx_cumodule_init

    jit_function = kwargs["fn"].jit_function
    fn_name = jit_function.fn.__name__

    # Only initialize NVSHMEM module for kernels registered via @requires_nvshmem
    if NvshmemKernelRegistry.has(fn_name):
        key = kwargs["key"]
        device = kwargs["compile"]["device"]
        jit_function = kwargs["fn"].jit_function
        kernel_cache = jit_function.device_caches[device][0]
        kernel = kernel_cache.get(key, None)
        if kernel is not None:
            _ = kernel.run
            # Initialize NVSHMEM for the CU module
            _nvshmemx_cumodule_init(kernel.module)
        else:
            logger.warning(
                f"It seems Triton hasn't created a kernel for function {fn_name}. "  # noqa: G004
                "Please report this issue to Triton."
            )


if has_triton():
    from triton.runtime.jit import JITFunction, KernelInterface

    # Create a new Callable class that follows the KernelInterface protocol so
    # that the Callable works with the subscript operator, e.g. `foo[(1, 1)]`
    class GridCallableWithExtern(KernelInterface):
    # [CLASS DOC] HELPER: Triton KernelInterface wrapper that forwards launches to the JIT
    # function run method with an extra extern-libs mapping so users need not pass it.
    # Supports the grid subscript launch syntax.
        """
        `KernelInterface` invokes `self.run` in `__getitem__`, i.e. [].  We
        implement a `run` method by directing the call to `JITFunction.run`,
        with added extern_libs kwarg, so that users don't have to pass it
        """

        def __init__(self, jit_func: JITFunction, extern_libs: dict[str, str]) -> None:
        # [FUNCTION DOC] HELPER: Stores the JIT function and extern libs mapping.
            self.jit_func = jit_func
            self.extern_libs = extern_libs

        def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Runs the wrapped JIT function with the given arguments plus
        # the stored extern libs and returns its result.
            # Call the JITFunction.run with added extern_libs kwarg
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)


def requires_nvshmem(  # type: ignore[no-untyped-def]
    jit_func,  # JITFunction created by triton.jit
):
    # [FUNCTION DOC] Decorator for a Triton JIT function that needs NVSHMEM device
    # functions. Raises TypeError if the input is not a JIT function. Finds the device
    # library (may raise RuntimeError), registers the kernel name, and installs the
    # NVSHMEM init hook as the global Triton post-compile hook (overwriting any previous
    # one; the hook filters by registry). Returns a wrapper that adds the device library
    # as an extern lib at launch. Apply above the JIT decorator.
    """
    A decorator to register a Triton kernel function that requires NVSHMEM initialization.

    Example usage:
    ```
        @requires_nvshmem
        @triton.jit
        def foo(...):
            ...
    ```

    If you would like to specify a path to the NVSHMEM device library other
    than standard search locations, you can use the following environment
    variable:
    ```
        export NVSHMEM_LIB_DIR=/path/to/nvshmem/lib
    ```
    """

    import triton
    from triton.runtime.jit import JITFunction

    if not isinstance(jit_func, JITFunction):
        raise TypeError(f"Expected a JITFunction, but got {type(jit_func)}")

    # Find the NVSHMEM device library
    lib_path = NvshmemLibFinder.find_device_library()
    extern_libs = {"libnvshmem_device": lib_path}

    # Register the JITFunction with the kernel registry as "to be initialized"
    NvshmemKernelRegistry.register(jit_func.fn.__name__)

    # Register the NVSHMEM init function as a post-compile hook.
    # [Note] This is a global setting (due to lack of Triton API exposure). To
    # avoid initializing Triton kernels that do not require NVSHMEM, filtering
    # is performed in the hook function itself by checking against
    # NvshmemKernelRegistry.
    triton.knobs.runtime.jit_post_compile_hook = _nvshmem_init_hook

    return GridCallableWithExtern(jit_func, extern_libs)


if has_triton():
    import triton
    import triton.language as tl
    from triton.language import core

    @triton.jit  # type: ignore[misc]
    def put(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: blocking put of nelems elements from local
        # source to dest on remote PE pe. Statically asserts that both tensors have the same
        # type, computes the byte size from the element size and calls the block put extern
        # wrapper. Returns the status from the extern call. Delivery is not guaranteed until
        # fence or quiet.
        """
        Put tensor data from local PE to a remote PE.

        This high-level function provides a tensor-aware interface for NVSHMEM put
        operations. It automatically handles type checking and size calculations, making
        the API more ergonomic and type-safe.

        Args:
            dest: Destination tensor on the remote PE. Type must match source.
            source: Source tensor on the local PE containing data to be copied.
            nelems: Number of elements to transfer.
            pe: PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
            - This is a blocking operation that returns after data has been copied out
              of the source array on the local PE.
            - The operation does not guarantee delivery to the destination PE.
              Use nvshmem_fence() for ordering or nvshmem_quiet() for completion.

        Example:
            ```
            # Transfer 100 elements to PE 1
            nvshmem.put(dest_tensor, src_tensor, 100, 1)
            ```
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return putmem_block_extern_wrapper(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def putmem_block_extern_wrapper(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper calling the NVSHMEM block-level put of a byte
        # count between int64 addresses to a given PE; returns an int32 status. Not pure.
        """Low-level extern wrapper for NVSHMEM put"""
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe number
                ): ("nvshmemx_putmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def get(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: blocking get of nelems elements from remote
        # PE source into local dest. Asserts equal tensor types, computes bytes, calls the
        # block get extern wrapper and returns its status. Data is usable on return.
        """
        Get tensor data from a remote PE to local PE.

        This high-level function provides a tensor-aware interface for NVSHMEM get
        operations. It automatically handles type checking and size calculations, making
        the API more ergonomic and type-safe.

        Args:
            dest: Destination tensor on the local PE. Type must match source.
            source: Source tensor on the remote PE containing data to be copied.
            nelems: Number of elements to transfer.
            pe: PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
            - This is a blocking operation that returns after data has been delivered
              to the destination array on the local PE.
            - The destination data is guaranteed to be available for use after the call returns.

        Example:
            ```
            # Get 100 elements from PE 0
            nvshmem.get(dest_tensor, src_tensor, 100, 0)
            ```
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return getmem_block_extern_wrapper(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def getmem_block_extern_wrapper(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper calling the NVSHMEM block-level get; int32
        # status. Not pure.
        """Low-level extern wrapper for NVSHMEM get"""
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe number
                ): ("nvshmemx_getmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def get_nbi(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function intended as a non-blocking get from a remote
        # PE (completion at the next quiet).
        # KNOWN ISSUE: it calls the blocking get extern wrapper instead of the non-blocking
        # wrapper defined below, so it behaves like a blocking get.
        # Asserts equal tensor types, computes bytes, returns the extern status.
        """
        Get tensor data from a remote PE to local PE, non-blocking.

        Different from the `get` function, this function returns after
        initiating the operation. The operation is considered complete after a
        subsequent call to `quiet`.

        Args:
            dest: Destination tensor on the local PE. Type must match source.
            source: Source tensor on the remote PE containing data to be copied.
            nelems: Number of elements to transfer.
            pe: PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.

        Example:
            ```
            # Get 100 elements from PE 0
            nvshmem.get_nbi(dest, src, 100, 0)
            # Some independent computation which overlaps with the get operation
            ...
            # Wait for completion of the get operation
            nvshmem.quiet()
            ```
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return getmem_block_extern_wrapper(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def getmem_nbi_block_extern_wrapper(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper for the NVSHMEM block-level non-blocking get;
        # int32 status. Currently unused by the get_nbi function. Docstring says it is for
        # get.
        """Low-level extern wrapper for NVSHMEM get"""
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe number
                ): ("nvshmemx_getmem_nbi_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def putmem_signal_block(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
    ):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: copies size_bytes from the local source to
        # the destination on PE pe, then atomically updates a signal variable on that PE with
        # a value using the given signal operation (set or add). Blocking at block scope;
        # returns an int32 status. The signal must be 8-byte aligned symmetric memory.
        # KNOWN ISSUE: the docstring lists the signal value argument as a second argument
        # named signal; the actual parameter is sig_val.
        """
        Put data to remote PE with atomic signal operation using block-scoped operation.

        This function copies data from the local PE to the remote PE and then
        atomically updates a signal variable on the remote PE to indicate completion.
        This enables efficient point-to-point synchronization between PEs.

        Args:
            dst (tensor): A tensor on calling PE symmetric to the destination tensor on remote PE.
            src (tensor): Local tensor containing the source data.
            size_bytes (int64): Number of bytes to transfer. Must be positive.
            signal (tensor): Symmetric signal pad with remote PE.
                             Must be 8-byte aligned symmetric memory.
            signal (int64): Value to be used in the signal operation.
            sig_op (int32): Signal operation type. Common values:
                           - NVSHMEM_SIGNAL_SET (0): Atomic set operation
                           - NVSHMEM_SIGNAL_ADD (5): Atomic add operation
            pe (int32): PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a blocking operation that returns after data has been copied out
              of the source array and the signal has been updated on the remote PE.
            - The signal update is performed atomically with respect to other signal
              operations and synchronization routines.
            - The signal variable must be of type uint64_t in symmetric memory.
            - Use with nvshmem_signal_wait_until() for synchronization.

        Example:
            ```
            # Transfer data and set completion flag to 1
            NVSHMEM_SIGNAL_SET = 0
            nvshmem.putmem_signal_block(
                dst_ptr, src_ptr, 1024, sig_ptr, 1, NVSHMEM_SIGNAL_SET, target_pe
            )
            ```
        """
        # Ensure sig_val is 64 bits
        sig_val = 0 << 32 | sig_val
        return putmem_signal_block_extern_wrapper(
            dst.to(tl.int64),
            src.to(tl.int64),
            size_bytes.to(tl.int64),
            signal.to(tl.int64),
            sig_val.to(tl.uint64),
            sig_op,
            pe,
        )

    @core.extern
    def putmem_signal_block_extern_wrapper(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
        _semantic=None,
    ):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper for the NVSHMEM block-level put-with-signal
        # taking destination, source, size, signal address, 64-bit signal value, signal
        # operation and PE; int32 status. Not pure.
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
                ): ("nvshmemx_putmem_signal_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # Wait and Signal Operations

    @triton.jit  # type: ignore[misc]
    def wait_until(ivar, cmp_op, cmp_val):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: blocks until the 32-bit integer in symmetric
        # memory at ivar satisfies the comparison (equal, not equal, greater, and so on, by
        # code) against cmp_val. Statically asserts a 4-byte element type. Returns the extern
        # status.
        # KNOWN ISSUE: the docstring names parameters ivar_tensor and cmp, but the actual
        # parameters are ivar, cmp_op and cmp_val.
        """
        Wait until a tensor variable meets a specified condition.

        This high-level function provides a tensor-aware interface for NVSHMEM wait_until
        operations. It automatically handles tensor address extraction, making
        the API more ergonomic and type-safe.

        Args:
            ivar_tensor: Tensor to monitor (typically int64/uint64) in symmetric memory.
            cmp: Comparison operator. Common values:
                 - NVSHMEM_CMP_EQ (0): Wait until ivar == cmp_val
                 - NVSHMEM_CMP_NE (1): Wait until ivar != cmp_val
                 - NVSHMEM_CMP_GT (2): Wait until ivar > cmp_val
                 - NVSHMEM_CMP_GE (3): Wait until ivar >= cmp_val
                 - NVSHMEM_CMP_LT (4): Wait until ivar < cmp_val
                 - NVSHMEM_CMP_LE (5): Wait until ivar <= cmp_val
            cmp_val: Value to compare against.

        Notes:
            - This is a blocking operation that will wait indefinitely until the
              condition is satisfied.
            - The tensor must be in symmetric memory and accessible from other PEs.

        Example:
            ```
            # Wait until flag tensor becomes 1 (set by another PE)
            NVSHMEM_CMP_EQ = 0
            nvshmem.wait_until_tensor(flag_tensor, NVSHMEM_CMP_EQ, 1)
            ```
        """
        tl.static_assert(
            ivar.type.element_ty.itemsize == 4,
            "wait_until expects a 32-bit type for the synchronization variable",
        )
        return wait_until_extern_wrapper(ivar.to(tl.int64), cmp_op, cmp_val)

    @core.extern
    def wait_until_extern_wrapper(ivar, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper for the NVSHMEM integer wait-until on an int64
        # address, int32 comparison code and int32 value; int32 result. Not pure.
        return core.extern_elementwise(
            "",
            "",
            [ivar, cmp, cmp_val],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("nvshmem_int_wait_until", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def signal_wait_until(signal, cmp, cmp_val):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: blocks until the 64-bit signal variable at
        # the signal address satisfies the comparison with cmp_val. The value is widened to
        # unsigned 64-bit before the extern call. Returns int32 status. Used with put-with-
        # signal for producer/consumer synchronization.
        """
        Wait until a signal variable meets a specified condition.

        This function blocks the calling thread until the value at the specified
        signal variable satisfies the given comparison condition. Signal variables
        are special uint64_t symmetric objects used for efficient synchronization
        with signal operations.

        Args:
            signal (tensor): Symmetric signal tensor with remote PE.
                             Must be 8-byte aligned symmetric memory.
            cmp (int32): Comparison operator. Common values:
                        - NVSHMEM_CMP_EQ (0): Wait until signal == cmp_val
                        - NVSHMEM_CMP_NE (1): Wait until signal != cmp_val
                        - NVSHMEM_CMP_GT (2): Wait until signal > cmp_val
                        - NVSHMEM_CMP_GE (3): Wait until signal >= cmp_val
                        - NVSHMEM_CMP_LT (4): Wait until signal < cmp_val
                        - NVSHMEM_CMP_LE (5): Wait until signal <= cmp_val
            cmp_val (int64): Value to compare against.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a blocking operation designed specifically for signal variables.
            - Signal variables are updated atomically by putmem_signal operations.
            - More efficient than wait_until for signal-based synchronization patterns.
            - Ensures the signal update is fully complete before returning.
            - Commonly used with putmem_signal_block for producer-consumer patterns.

        Example:
            ```
            # Wait for signal to be set to completion value
            NVSHMEM_CMP_EQ = 0
            nvshmem.signal_wait_until(signal_ptr, NVSHMEM_CMP_EQ, 42)
            ```
        """
        cmp_val = 0 << 32 | cmp_val
        return signal_wait_until_extern_wrapper(
            signal.to(tl.int64), cmp, cmp_val.to(tl.uint64)
        )

    @core.extern
    def signal_wait_until_extern_wrapper(signal, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Extern wrapper for the NVSHMEM signal wait-until with int64
        # address, int32 comparison code and uint64 value; int32 result. Not pure.
        return core.extern_elementwise(
            "",
            "",
            [signal, cmp, cmp_val],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int32"),
                    core.dtype("uint64"),
                ): ("nvshmem_signal_wait_until", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: atomically updates a 64-bit signal variable
        # at a symmetric address on remote PE pe, with a value and an operation code (set or
        # add). One-sided; returns int32 status. Not pure.
        """
        Perform an atomic signal operation on a remote PE.

        This function atomically updates a signal variable on the specified remote PE
        using the given operation and value. This enables efficient point-to-point
        synchronization and notification between PEs.

        Args:
            sig_addr (int64): Symmetric address of the signal variable (uint64_t) on the remote PE.
                             Must be 8-byte aligned symmetric memory.
            signal (int64): Value to be used in the signal operation.
            sig_op (int32): Signal operation type. Common values:
                           - NVSHMEM_SIGNAL_SET (0): Atomically set sig_addr = signal
                           - NVSHMEM_SIGNAL_ADD (5): Atomically set sig_addr += signal
            pe (int32): PE number of the remote PE (0 ≤ pe < nvshmem_n_pes()).
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a one-sided operation - the remote PE does not need to participate.
            - The signal operation is performed atomically on the remote PE.
            - Can be used with signal_wait_until() on the remote PE for synchronization.
            - Provides low-overhead notification mechanism between PEs.
            - The signal variable must be of type uint64_t in symmetric memory.

        Example:
            ```python
            # Atomically set remote signal to 1 to notify completion
            NVSHMEM_SIGNAL_SET = 0
            nvshmem.signal_op(remote_signal_ptr, 1, NVSHMEM_SIGNAL_SET, target_pe)
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [sig_addr, signal, sig_op, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("nvshmemx_signal_op", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # Memory Ordering Operations
    @core.extern
    def fence(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: memory fence ordering earlier puts to each
        # remote PE before later puts to the same PE (per-destination ordering only, no
        # completion guarantee). Returns int32 status. Takes no user arguments.
        """
        Ensure ordering of put operations to each remote PE.

        This function provides a memory fence that ensures point-to-point ordering
        of remote memory operations. Put operations issued before the fence are
        guaranteed to be ordered before put operations issued after the fence,
        when targeting the same remote PE.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This provides weaker ordering guarantees than quiet().
            - Operations to each PE are ordered, but operations to different PEs
              may still be reordered relative to each other.
            - Does not guarantee completion of operations, only ordering.
            - Non-blocking operations are not ordered by fence - use quiet() instead.
            - Essential for ensuring correct ordering in communication patterns.

        Memory Ordering Guarantees:
            - Put operations before fence() → ordered before → Put operations after fence()
            - Ordering is maintained per-destination-PE basis
            - Remote PEs can observe the enforced ordering

        Example:
            ```
            # Ensure first put completes before second put to same PE
            nvshmem.put(dst, src, nelems, target_pe)
            nvshmem.fence()  # Enforce ordering
            nvshmem.put(dst2, src2, nelems, target_pe)
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("nvshmem_fence", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def quiet(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: blocks until all outstanding remote memory
        # operations issued by the calling PE (including non-blocking ones) have completed.
        # Stronger than fence. Returns int32 status.
        """
        Wait for completion of all outstanding put operations.

        This function blocks until all outstanding remote memory operations issued
        by the calling PE have completed. It provides stronger guarantees than
        fence() by ensuring both ordering and completion of all operations.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a blocking operation that waits for completion.
            - Ensures all previous put operations have been delivered to their destinations.
            - Provides global ordering - operations to ALL PEs are ordered.
            - Required to complete non-blocking operations.
            - More expensive than fence() but provides stronger guarantees.

        Memory Ordering Guarantees:
            - All put operations before quiet() are completed before any operations after quiet()
            - Operations are visible to all PEs as having occurred before subsequent operations
            - Both blocking and non-blocking operations are completed

        Example:
            ```
            # Ensure all data transfers complete before setting completion flag
            nvshmem.putmem_block(data_ptr, src_ptr, data_size, target_pe)
            nvshmem.quiet()  # Wait for data transfer completion
            nvshmem.putmem_block(
                flag_ptr, flag_src_ptr, 8, target_pe
            )  # Signal completion
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("nvshmem_quiet", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # PE Information Operations
    @core.extern
    def my_pe(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: returns the PE number (int32, zero-based) of
        # the calling PE. Marked pure.
        """
        Get the PE number of the calling PE.

        This function returns the unique identifier (PE number) of the current
        processing element within the NVSHMEM job. PE numbers range from 0 to
        nvshmem_n_pes() - 1.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: PE number of the calling PE (0 ≤ pe < nvshmem_n_pes()).

        Notes:
            - This is a pure function that returns the same value throughout execution.
            - PE numbering starts from 0 and is contiguous.
            - Each PE has a unique identifier within the NVSHMEM job.
            - Can be called from both host and device code.
            - Essential for implementing PE-specific logic and communication patterns.

        Example:
            ```
            # Get current PE number for conditional logic
            pe = nvshmem.my_pe()
            if pe == 0:
                # Root PE logic
                pass
            else:
                # Non-root PE logic
                pass
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_my_pe", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def n_pes(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: returns the total number of PEs in the job
        # (int32). Marked pure.
        """
        Get the total number of PEs in the NVSHMEM job.

        This function returns the total count of processing elements (PEs)
        participating in the current NVSHMEM job. This value remains constant
        throughout the execution of the program.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Total number of PEs in the job (always ≥ 1).

        Notes:
            - This is a pure function that returns the same value throughout execution.
            - The value is determined at NVSHMEM initialization and never changes.
            - Valid PE numbers range from 0 to n_pes() - 1.
            - Can be called from both host and device code.
            - Essential for implementing collective operations and communication patterns.

        Example:
            ```
            # Broadcast from root to all other PEs
            total_pes = nvshmem.n_pes()
            my_rank = nvshmem.my_pe()

            if my_rank == 0:
                # Send to all other PEs
                for peer in range(1, total_pes):
                    nvshmem.putmem_block(dst_ptr, src_ptr, size, peer)
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_n_pes", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    # Synchronization Operations
    @core.extern
    def barrier_all(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: collective barrier across all PEs that also
        # guarantees completion of outstanding memory operations. All PEs must call it; needs
        # cooperative launch. Returns int32 status.
        """
        Synchronize all PEs with completion guarantee.

        This function creates a barrier across all PEs in the NVSHMEM job. It ensures
        that all local and remote memory updates issued before the barrier by any PE
        are completed before any PE exits the barrier. This provides both
        synchronization and memory consistency.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a collective operation - all PEs must participate.
            - Stronger guarantee than sync_all() - ensures completion of remote operations.
            - Blocks until all PEs reach the barrier AND all memory operations complete.
            - Must be called from kernels launched with cooperative launch.
            - Provides full memory consistency across all PEs.
            - More expensive than sync_all() due to completion guarantees.

        Memory Consistency Guarantees:
            - All memory updates before barrier_all() are visible to all PEs
            - All remote memory operations are completed before any PE continues
            - Provides a global synchronization point with memory ordering

        Example:
            ```
            # Ensure all PEs complete their work before proceeding
            # All PEs execute this - it's a collective operation
            nvshmem.barrier_all()
            # At this point, all previous operations are complete on all PEs
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_barrier_all", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def sync_all(_semantic=None):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extern device function: lighter collective synchronization across all
        # PEs that makes local stores visible but does not guarantee completion of outstanding
        # remote operations. Cooperative launch required. Returns int32 status.
        """
        Synchronize all PEs with local completion guarantee.

        This function creates a lightweight synchronization barrier across all PEs.
        It ensures that all local store operations issued before the sync are
        visible to other PEs, but does not guarantee completion of remote memory
        operations initiated by the calling PE.

        Args:
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a collective operation - all PEs must participate.
            - Lighter weight than barrier_all() - only ensures local store visibility.
            - Does not guarantee completion of remote memory updates initiated locally.
            - Must be called from kernels launched with cooperative launch.
            - Suitable when only synchronization (not completion) is needed.
            - More efficient than barrier_all() for synchronization-only patterns.

        Memory Consistency Guarantees:
            - Local store operations are visible to other PEs
            - Does NOT ensure completion of outgoing remote operations
            - Provides synchronization point without full completion overhead

        Example:
            ```
            # Lightweight synchronization between PEs
            # All PEs execute this - it's a collective operation
            nvshmem.sync_all()
            # Local stores are visible, but remote ops may still be in flight
            ```
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_sync_all", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    # Collective Operations (mem-based APIs - sizes in bytes)
    @triton.jit  # type: ignore[misc]
    def alltoall(team, dest, source, nelems_per_pe):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: all-to-all exchange within a team where each
        # PE sends nelems_per_pe elements to every PE and receives the same from each. Asserts
        # equal dest and source types, converts the per-PE element count to bytes and calls
        # the extern wrapper. Collective; returns int32 status. Team zero means the world
        # team.
        """
        All-to-all tensor exchange between PEs in a team.

        This high-level function provides a tensor-aware interface for NVSHMEM alltoall
        operations. Each PE sends nelems_per_pe elements to every other PE and receives
        the same amount from every other PE.

        Args:
            team: Team handle for the collective operation. Use 0 for NVSHMEM_TEAM_WORLD.
            dest: Destination tensor. Must be large enough for nelems_per_pe * n_pes elements.
            source: Source tensor containing data for all PEs. Must contain nelems_per_pe * n_pes elements.
            nelems_per_pe: Number of elements to exchange with each PE.

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
            - This is a collective operation - all PEs in the team must participate.
            - Data layout: source=[data_for_pe0, data_for_pe1, ...], dest=[data_from_pe0, data_from_pe1, ...]

        Example:
            ```
            # Each PE exchanges 10 elements with every other PE
            nvshmem.alltoall(0, dest_tensor, src_tensor, 10)
            ```
        """
        tl.static_assert(dest.type == source.type)
        size_bytes_per_pe = nelems_per_pe * dest.type.element_ty.itemsize
        return alltoallmem_block_extern_wrapper(
            team, dest.to(tl.int64), source.to(tl.int64), size_bytes_per_pe.to(tl.int64)
        )

    @core.extern  # type: ignore[misc]
    def alltoallmem_block_extern_wrapper(
        team: Any, dest: Any, source: Any, size_bytes: Any, _semantic: Any = None
    ) -> None:
        # [FUNCTION DOC] HELPER: Extern wrapper for the NVSHMEM block-level all-to-all taking
        # an int32 team, int64 dest and source addresses and per-PE byte size; int32 status.
        # Not pure.
        """Low-level extern wrapper for NVSHMEM alltoall"""
        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, size_bytes],
            {
                (
                    core.dtype("int32"),  # team handle
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                ): ("nvshmemx_alltoallmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def broadcast(team, dest, source, nelems, pe_root):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: broadcasts nelems elements from the root PE
        # source to dest on all PEs in a team. Asserts equal types, computes bytes, calls the
        # extern wrapper with the root PE. Collective with cooperative launch; returns int32
        # status.
        """
        Broadcast tensor data from a root PE to all other PEs in a team.

        This high-level function provides a tensor-aware interface for NVSHMEM broadcast
        operations. It automatically handles type checking and size calculations, making
        the API more ergonomic and type-safe.

        Args:
            team: Team handle for the collective operation. Use 0 for NVSHMEM_TEAM_WORLD.
            dest: Destination tensor with type information. All PEs receive data here.
            source: Source tensor on the root PE. Type must match dest.
            nelems: Number of elements to broadcast.
            pe_root: PE number of the root PE that provides the source data.

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
            - This is a collective operation - all PEs in the team must participate.
            - Must be called from kernels launched with cooperative launch.

        Example:
            ```
            # Broadcast 100 elements from PE 0 to all PEs
            nvshmem.broadcast(0, dest_tensor, src_tensor, 100, 0)
            ```
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return broadcastmem_block_extern_wrapper(
            team, dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe_root
        )

    @core.extern  # type: ignore[misc]
    def broadcastmem_block_extern_wrapper(
        team: Any,
        dest: Any,
        source: Any,
        size_bytes: Any,
        pe_root: Any,
        _semantic: Any = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Extern wrapper for the NVSHMEM block-level broadcast with
        # team, dest and source addresses, byte size and root PE; int32 status. Not pure.
        """Low-level extern wrapper for NVSHMEM broadcast"""
        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, size_bytes, pe_root],
            {
                (
                    core.dtype("int32"),  # team handle
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe_root
                ): ("nvshmemx_broadcastmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # Reduction Operation
    @triton.jit  # type: ignore[misc]
    def reduce(team, dest, source, nreduce, operation: tl.constexpr):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Triton device function: team-wide reduction of nreduce elements from
        # source into dest across PEs, with a compile-time operation name (sum, max, min or
        # prod). Asserts equal types, passes the destination element dtype to the extern
        # wrapper. Collective with cooperative launch; returns int32 status.
        """
        Performs a collective reduction on tensors across a team of PEs.

        This high-level function provides a tensor-aware interface for NVSHMEM
        reduction operations. It automatically infers the data type from the
        input tensors and calls the appropriate underlying NVSHMEM function.

        Args:
            team: The team handle for the collective (0 for NVSHMEM_TEAM_WORLD).
            dest: Destination tensor for the reduction results.
            source: Source tensor containing data to be reduced. Must be the same type as dest.
            nreduce: The number of elements in the source tensor to reduce.
            operation: The reduction operation to perform ("sum", "max", "min", "prod").

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - This is a collective operation that must be called by all PEs in the team.
            - Requires a cooperative grid launch.

        Example:
            ```
            # Perform a sum reduction on two tensors
            nvshmem.reduce(0, dest_tensor, src_tensor, 100, "sum")
            ```
        """
        tl.static_assert(dest.type == source.type)
        dtype = dest.type.element_ty
        return reduce_extern_wrapper(
            team,
            dest.to(tl.int64),
            source.to(tl.int64),
            nreduce.to(tl.int64),
            operation,
            dtype,
        )

    @core.extern  # type: ignore[misc]
    def reduce_extern_wrapper(
        team: Any,
        dest: Any,
        source: Any,
        nreduce: Any,
        operation: str,
        dtype: Any,
        _semantic: Any = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Extern wrapper selecting the typed NVSHMEM reduce function by
        # element type and operation name. Maps Triton dtype names (integer widths, fp16,
        # bf16, fp32, fp64) to NVSHMEM type names and accepts the operation as a plain or
        # constexpr string. Raises TypeError for an unsupported dtype and ValueError for an
        # unsupported operation. Emits an extern call with team and int64 address and count
        # arguments; int32 status. Not pure.
        """
        Low-level extern wrapper for NVSHMEM reduction operations.

        This function provides a generic interface to NVSHMEM reduction operations,
        automatically selecting the appropriate NVSHMEM function based on the data type
        and operation specified.
        Args:
            team (int64): The team handle (0 for NVSHMEM_TEAM_WORLD).
            dest (pointer): Destination pointer where reduction results are stored.
            source (pointer): Source pointer containing data to be reduced.
            nreduce (int64): Number of elements to reduce.
            operation (str): Reduction operation ("sum", "max", "min", "prod").
            dtype: Data type specification - accepts torch.dtype, tl.dtype, str, or constexpr.
            _semantic: Optional semantic information for Triton compilation.

        Raises:
            ValueError: If the operation is not supported.
            TypeError: If the data type is not supported.

        Example:
            nvshmem.reduce(0, dest_ptr, src_ptr, 100, "sum", torch.float32)
        """
        # Mapping from Triton dtype names to NVSHMEM typenames
        DTYPE_TO_NVSHMEM_MAP = {
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "uint16": "uint16",
            "uint32": "uint32",
            "uint64": "uint64",
            "fp16": "half",
            "bf16": "bfloat16",
            "fp32": "float",
            "fp64": "double",
        }

        # Triton dtype names are standardized as fp16, bf16, fp32, etc.
        dtype_name = str(dtype).replace("tl.", "")

        if dtype_name not in DTYPE_TO_NVSHMEM_MAP:
            raise TypeError(
                f"Unsupported reduction dtype: {dtype_name}. Supported dtypes: {list(DTYPE_TO_NVSHMEM_MAP.keys())}"
            )

        # Extract operation name from constexpr if needed
        op_name = operation.value if hasattr(operation, "value") else operation

        # Validate operation is supported
        supported_ops = {"sum", "max", "min", "prod"}
        if op_name not in supported_ops:
            raise ValueError(
                f"Unsupported reduction operation: '{op_name}'. Supported ops are {supported_ops}"
            )

        # Map to NVSHMEM typename and validate dtype is supported
        nvshmem_typename = DTYPE_TO_NVSHMEM_MAP.get(dtype_name)
        if nvshmem_typename is None:
            raise TypeError(
                f"Unsupported reduction dtype: {dtype_name}. Supported dtypes are {list(DTYPE_TO_NVSHMEM_MAP.keys())}"
            )

        # Generate NVSHMEM function name
        nvshmem_func = f"nvshmem_{nvshmem_typename}_{op_name}_reduce"

        # Define function signature - all parameters are int64 in Triton (they are just ptrs)
        signature = (
            core.dtype("int32"),  # team handle
            core.dtype("int64"),  # destination pointer
            core.dtype("int64"),  # source pointer
            core.dtype("int64"),  # number of elements
        )

        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, nreduce],
            {signature: (nvshmem_func, core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    # Utility for inspecting Triton kernels

    triton_kernels: dict = {}

    def _log_triton_kernel(kernel) -> None:  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] HELPER: Debug utility that records a compiled Triton kernel in the
    # module-level kernel dictionary (non-zero ranks are ignored when distributed is
    # initialized). When the first kernel is recorded it registers an exit hook that
    # writes each kernel PTX to a temporary file and logs its path. Side effects: global
    # state, exit hook, files.
        import atexit
        import tempfile

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        def on_exit() -> None:
        # [FUNCTION DOC] HELPER: Exit hook: logs a header, then for each recorded kernel
        # writes its PTX assembly to a new temporary file (not deleted) and logs the name and
        # file path.
            logger.info("PTX files:")
            for kernel in triton_kernels:
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(kernel.asm["ptx"].encode("utf-8"))
                    logger.info(f"+- {kernel.name}: {f.name}")  # noqa: G004

        if len(triton_kernels) == 0:
            atexit.register(on_exit)

        if kernel not in triton_kernels:
            triton_kernels[kernel] = None
