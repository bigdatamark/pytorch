# mypy: allow-untyped-defs
# [MODULE DOC] Inductor async compilation front end. Provides AsyncCompile, which compiles generated kernels (Triton, C++, CUDA/ROCm/XPU,
# Halide, CuteDSL, Pallas, NV universal GEMM, Metal) on thread pools or a subprocess pool and returns futures that wait() later
# swaps into the generated module's scope. Also holds the process-pool lifecycle helpers, compile-time metrics, and an in-memory
# cache of compiled Triton kernels. Import-time side effects: creates module globals for pools and metrics, registers an after-fork
# hook (where supported) and an atexit hook that shuts down compile workers.
from __future__ import annotations

import atexit
import functools
import json
import logging
import multiprocessing
import os
import re
import sys
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from time import time, time_ns
from typing import Any, TYPE_CHECKING

import torch
from torch._dynamo.device_interface import get_registered_device_interfaces
from torch._dynamo.utils import (
    counters,
    dynamo_timed,
    get_metrics_context,
    set_feature_use,
)
from torch._inductor import config
from torch._inductor.codecache import (
    _load_triton_kernel_from_source,
    code_hash,
    CodeCacheFuture,
    CppCodeCache,
    CppPythonBindingsCodeCache,
    CUDACodeCache,
    HalideCodeCache,
    LambdaFuture,
    ROCmCodeCache,
    StaticAutotunerFuture,
    torch_key,
    XPUCodeCache,
)
from torch._inductor.compile_worker.subproc_pool import (
    AnyPool,
    SubprocException,
    SubprocPool,
)
from torch._inductor.compile_worker.tracked_process_pool import (
    TrackedProcessPoolExecutor,
)
from torch._inductor.compile_worker.utils import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import (
    _set_triton_libdevice_path,
    _set_triton_ptxas_path,
    _worker_compile_pycodecache_kernel,
    _worker_compile_triton,
)
from torch._inductor.utils import clear_on_fresh_cache
from torch._inductor.virtualized import V
from torch._utils_internal import log_triton_builds
from torch.hub import _Faketqdm, tqdm
from torch.utils._ordered_set import OrderedSet
from torch.utils._triton import has_triton_package


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.hints import HalideMeta
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0.0
_t0: float | None = None

kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")

log = logging.getLogger(__name__)

_triton_kernel_metrics: dict[str, dict[str, Any]] | None = None

size_hints_regex = re.compile(
    r"size_hints=(\{.*?\})",
)


def _pycodecache_kernel_compile_env() -> dict[str, str | None]:
    # [FUNCTION DOC] HELPER: Returns a mapping of cache-directory related environment variable names (inductor cache dir, triton cache dir, cutlass dir)
    # to their current values (None if unset), so workers can reproduce the parent's settings.
    env_vars = [
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CUTLASS_DIR",
    ]
    return {v: os.environ.get(v) for v in env_vars}


def pre_fork_setup():
    # [FUNCTION DOC] Prepares state before forking/spawning compile workers: caches device properties for all available devices and
    # precomputes the Triton key (slow) when Triton is available, so children inherit the cached results. No return value.
    """
    Setup that must be done prior to forking with a process pool.
    """
    # ensure properties have been calculated before processes
    # are forked
    caching_device_properties()

    # Computing the triton key can be slow. If we call it before fork,
    # it will be cached for the forked subprocesses.
    from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

    if HAS_TRITON:
        triton_key()


def caching_device_properties():
    # [FUNCTION DOC] For each registered device interface that is available, queries and thereby caches the device properties via its Worker.
    # Called before forking so workers need not recompute them. No return value.
    for _, device_interface in get_registered_device_interfaces():
        if device_interface.is_available():
            device_interface.Worker.get_device_properties()


def _compile_start() -> None:
    # [FUNCTION DOC] HELPER: Marks the start of a compile phase: records the start timestamp if not already set and initializes the per-kernel Triton metrics
    # mapping if absent. Mutates module globals.
    global _t0, _triton_kernel_metrics
    if _t0 is None:
        _t0 = time()
    if _triton_kernel_metrics is None:
        _triton_kernel_metrics = {}


def _compile_end() -> None:
    # [FUNCTION DOC] HELPER: Ends a compile phase: adds elapsed time since the start timestamp to the cumulative compile time and clears the timestamp.
    # If Triton kernel metrics were collected, emits them (sorted by kernel name) as a JSON structured-trace artifact and resets them. Mutates module globals.
    global _cumulative_compile_time, _t0, _triton_kernel_metrics
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)
    if _triton_kernel_metrics:
        # Log triton kernel info
        sorted_info = dict(sorted(_triton_kernel_metrics.items()))
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "triton_kernel_info",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(sorted_info),
        )
        _triton_kernel_metrics = None


def _add_triton_kernel_info(kernel_name: str, info: dict[str, Any]):
    # [FUNCTION DOC] HELPER: Records info for a kernel in the metrics mapping; silently ignored if called outside a compile phase (mapping is None).
    global _triton_kernel_metrics
    # Must be called between _compile_start and _compile_end
    if _triton_kernel_metrics is not None:
        _triton_kernel_metrics[kernel_name] = info


def _emit_triton_kernel_compile_metric(
    kernel: CachingAutotuner,
    kernel_name: str,
    elapsed_us: int,
) -> None:
    # [FUNCTION DOC] HELPER: Records the compile time in microseconds for a kernel: stores it in the kernel's autotune cache info (a throwaway dict if none) and sends it to the
    # triton kernel metrics map and to the metrics context top-N list. Args: kernel, kernel_name, elapsed_us.
    """Emit per-kernel ``compile_time_us`` to both the dynamo
    ``_triton_kernel_metrics`` map and the ``MetricsContext`` top-N.

    Note: ``kernel.autotune_cache_info`` is only mutated in place when
    non-empty; when ``None`` or ``{}`` the metric still reaches
    ``_triton_kernel_metrics`` via a throwaway dict, but the kernel
    attribute stays untouched.
    """
    info = kernel.autotune_cache_info or {}
    info["compile_time_us"] = elapsed_us
    _add_triton_kernel_info(kernel_name, info)
    get_metrics_context().add_top_n(
        "triton_kernel_compile_times_us", kernel_name, elapsed_us
    )


_IS_WINDOWS = sys.platform == "win32"

log = logging.getLogger(__name__)

# Used to keep track of all process pools invoked so far.
_pool_set = OrderedSet[AnyPool]()


def shutdown_compile_workers() -> None:
    # [FUNCTION DOC] Shuts down every pool recorded in the module pool set, then resets AsyncCompile's ready future and clears pool state via after_fork.
    # Registered with atexit so workers can clean up (avoids leaked semaphore warnings).
    """Shut down all outstanding compile-worker pools."""
    for pool in _pool_set:
        pool.shutdown()
    AsyncCompile._ready_future = None
    after_fork()


def after_fork():
    # [FUNCTION DOC] Resets pool bookkeeping without shutting pools down: clears the pool set, drops the ready future and clears the cached process pool.
    # Registered as an after-fork-in-child hook so children do not reuse the parent's pools.
    """Reset pools to initial state without shutting them down"""
    _pool_set.clear()
    AsyncCompile._ready_future = None
    AsyncCompile.process_pool.cache_clear()


try:
    os.register_at_fork(after_in_child=after_fork)
except AttributeError:
    pass  # register_at_fork does not exist on windows


def get_compile_threads() -> int:
    # [FUNCTION DOC] Returns the number of compile threads, lazily computing config.compile_threads on first use and storing it back into config.
    # Side effect: mutates the config value when it was None.
    """
    Temporary for internal rollout. Assign config.compile_threads lazily and return it.
    TODO: remove after rollout.
    """
    if config.compile_threads is None:
        config.compile_threads = config.decide_compile_threads()
    return config.compile_threads


def _process_pool_allowed() -> bool:
    # [FUNCTION DOC] HELPER: Returns true if a process pool may be created: either the worker start method is subprocess (the sidecar is not affected by the
    # daemon restriction) or the current multiprocessing process is not a daemon.
    # Multiprocessing daemons are not allowed to create child processes. This
    # only applies to direct multiprocessing modes: SubprocPool starts its
    # sidecar with subprocess.Popen, so the sidecar does not inherit the
    # multiprocessing daemon flag and can own its own ProcessPoolExecutor.
    return (
        config.worker_start_method == "subprocess"
        or not multiprocessing.current_process().daemon
    )


@clear_on_fresh_cache
class CompiledTritonKernels:
    # [CLASS DOC] In-memory registry (class-level mapping from key to future) of compiled Triton kernels. Keys are hashes of the full kernel source combined
    # with the torch key. Never instantiated; all methods are static. Cleared when a fresh cache is requested (via clear_on_fresh_cache).
    """
    In memory cache for storing compiled triton kernels.

    Each triton kernel is keyed by the hash of its source code. Each value stored
    in the cache is a return value of AsyncCompile.triton().

    Currently, the cache stores Future objects, but it should be generalizable for any kernels.
    """

    _cache: dict[str, CodeCacheFuture] = {}

    @staticmethod
    def key(kernel_src: str):
        # [FUNCTION DOC] Returns the cache key for a kernel: the code hash of the source string mixed with the torch key. Pure function.
        """
        Generates a cache key given a triton kernel's full source code.
        This source includes the inductor meta, compilation metadata, the kernel itself, etc.
        `kernel_src` should be the exact string passed to async_compile.triton()'s first argument.
        """
        # Hashes the kernel source with torch_key into a single hash key
        return code_hash(kernel_src, extra=torch_key())

    @staticmethod
    def save(kernel_src: str, future: CodeCacheFuture):
        # [FUNCTION DOC] Stores the future in the class-level cache under the key computed from the source, replacing any previous entry.
        """
        Saves a compiled triton kernel to the cache.
        TODO: We store a LambdaFuture as that's the callable returned by async_compile.triton,
        but the real type we want to return here is actually an abstract triton kernel.

        TODO: Source code here is not just the kernel's source code, but also includes the inductor preamble, etc.
        so it could be less strict.
        """
        key = CompiledTritonKernels.key(kernel_src)
        CompiledTritonKernels._cache[key] = future

    @staticmethod
    def get(kernel_src: str) -> CodeCacheFuture | None:
        # [FUNCTION DOC] Returns the cached future for the given source or None if absent. No side effects.
        key = CompiledTritonKernels.key(kernel_src)
        return CompiledTritonKernels._cache.get(key, None)

    @staticmethod
    def cache_clear():
        # [FUNCTION DOC] Replaces the class-level cache with a new empty mapping.
        CompiledTritonKernels._cache = {}

    @staticmethod
    def remove_future(kernel_src: str) -> None:
        # [FUNCTION DOC] Deletes the cache entry for the given source if present; no error if absent.
        key = CompiledTritonKernels.key(kernel_src)

        # Delete the LambdaFuture if there is one
        if key in CompiledTritonKernels._cache:
            del CompiledTritonKernels._cache[key]


class AsyncCompile:
    # [CLASS DOC] Entry point used by generated code to compile kernels asynchronously. Holds class-level state: the pool-readiness future and the list of
    # registered Metal kernel sources (kernel name, source, headers). Instances are stateless apart from that. Methods per backend return
    # either a ready kernel (serial mode, compile threads <= 1) or a future; wait() resolves futures in a scope mapping and compiles Metal.
    """
    Utilities to compile in thread pools or subprocess pools (in the case of Triton).
    """

    _ready_future: Future[Any] | None = None
    _metal_sources: list[tuple[str, str, list[str]]] | None = None

    def __init__(self) -> None:
        # [FUNCTION DOC] No-op constructor.
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        # [FUNCTION DOC] Returns the cached (single instance) thread pool sized by the compile thread count. Raises AssertionError if the thread count is 1 or less.
        if get_compile_threads() <= 1:
            raise AssertionError(
                f"expected get_compile_threads() > 1, got {get_compile_threads()}"
            )
        return ThreadPoolExecutor(get_compile_threads())

    @staticmethod
    def _get_ready():
        # [FUNCTION DOC] HELPER: Trivial task submitted to the process pool; returns the string ready. Its completion indicates workers are up.
        """No-op function to help mark when the subprocess pool is ready."""
        return "ready"

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> AnyPool:
        # [FUNCTION DOC] Creates (once, cached) and returns the compile process pool: a SubprocPool for the subprocess start method, otherwise a tracked
        # ProcessPoolExecutor using the configured multiprocessing context (after pre_fork_setup; sets an env var to disable warm pools in spawned children).
        # Raises AssertionError if compile threads <= 1 and RuntimeError in daemonic multiprocessing processes. Side effects: resets the ready future,
        # logs, registers a finalizer to shut the pool down at exit, and adds the pool to the module pool set.
        if get_compile_threads() <= 1:
            raise AssertionError(
                f"expected get_compile_threads() > 1, got {get_compile_threads()}"
            )
        if not _process_pool_allowed():
            raise RuntimeError(
                "Inductor async compile process pools are disabled in daemonic "
                "multiprocessing processes. Set "
                "torch._inductor.config.worker_start_method = 'subprocess' "
                "(or TORCHINDUCTOR_WORKER_START=subprocess) to use the "
                "SubprocPool path, which is not affected by the daemon restriction."
            )
        AsyncCompile._ready_future = None
        log.info(
            "Creating '%s' pool with %d workers",
            config.worker_start_method,
            get_compile_threads(),
        )

        pool: AnyPool
        if config.worker_start_method == "subprocess":
            # Wrapper around ProcessPoolExecutor forks in a new process we control
            pool = SubprocPool(
                get_compile_threads(), quiesce=config.quiesce_async_compile_pool
            )
        else:
            if config.worker_start_method == "spawn":
                # Avoid creating pools in the spawned subprocs themselves:
                os.environ["TORCH_WARM_POOL"] = "0"
            pre_fork_setup()
            ctx = multiprocessing.get_context(config.worker_start_method)
            pool = TrackedProcessPoolExecutor(
                get_compile_threads(),
                mp_context=ctx,
                initializer=partial(_async_compile_initializer, os.getpid()),
            )

        # When this pool is created in a multiprocessing subprocess, the normal
        # atexit handler may not run, and we need to register our own handler.
        # exitpriority has to be high, because another one of the finalizers will
        # kill the worker thread that sends the shutdown message to the workers.
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)

        _pool_set.add(pool)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        # [FUNCTION DOC] Eagerly creates the process pool (inside a compile start/end bracket) unless compile threads <= 1 or process pools are not allowed.
        # For a SubprocPool only the sidecar starts; its executor starts on wakeup or first job.
        if get_compile_threads() <= 1 or not _process_pool_allowed():
            return
        _compile_start()
        # Pool is created on first access. Note for a SubprocPool, the sidecar process starts,
        # but its ProcessPoolExecutor does not initialize until a wakeup() call or the first
        # job is submitted.
        cls.process_pool()
        _compile_end()

    @classmethod
    def wait_pool_ready(cls, timeout=120) -> None:
        # [FUNCTION DOC] Calls use_process_pool (which submits the readiness probe if needed) then blocks up to timeout seconds on the ready future.
        # Propagates any timeout or pool error from waiting on the future.
        cls.use_process_pool()
        if cls._ready_future is not None:
            cls._ready_future.result(timeout=timeout)

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        # [FUNCTION DOC] Runs the task immediately and returns its result when compile threads <= 1; otherwise submits it to the thread pool and returns the future.
        if get_compile_threads() <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def use_process_pool(cls):
        # [FUNCTION DOC] Non-blocking check whether the process pool can be used. Returns False when threads <= 1, pools are disallowed or proton profiling
        # is enabled. Otherwise lazily submits a readiness probe (creating the pool) and returns whether that probe has finished.
        if get_compile_threads() <= 1 or not _process_pool_allowed():
            return False

        # Proton instrumentation backend requires compilation to happen in the main
        # process so it can instrument the Triton IR during JIT compilation.
        # Force synchronous compilation when proton profiling is enabled.
        if config.triton.proton_profiling:
            return False

        # Create a dummy job to check if the pool is ready. Submit it here instead of at
        # pool creation so we don't launch the full pool of worker subprocesses until
        # we're sure they're needed.
        if not cls._ready_future:
            cls._ready_future = cls.process_pool().submit(cls._get_ready)
        return cls._ready_future.done()

    @classmethod
    def wait_process_pool_ready(cls, timeout: float = 120) -> bool:
        # [FUNCTION DOC] Blocking variant of use_process_pool: waits up to timeout seconds for the readiness probe. Returns True if the pool is usable,
        # False if unavailable/disabled, on timeout, or if the pool is broken/closed (these cases log a warning and degrade to serial rather than raise).
        """Block (up to ``timeout`` s) until the process pool is ready, returning
        whether it's usable.

        Like use_process_pool() but blocking. Use when a backend's serial
        fallback is far costlier than the warmup wait -- e.g. NVGEMM subprocess
        precompile, where skipping the pool forces ~15x-slower lazy compilation
        at benchmark time. (use_process_pool()'s non-blocking readiness check can
        race pool warmup when little other compilation precedes the decision.)
        On timeout, degrade gracefully (return False -> serial) rather than hang
        on a stuck worker.
        """
        if get_compile_threads() <= 1 or not _process_pool_allowed():
            return False
        if config.triton.proton_profiling:
            return False
        if not cls._ready_future:
            cls._ready_future = cls.process_pool().submit(cls._get_ready)
        try:
            cls._ready_future.result(timeout=timeout)
        except FuturesTimeoutError:
            log.warning(
                "Process pool not ready after %ss; falling back to serial", timeout
            )
            return False
        except (BrokenProcessPool, RuntimeError) as e:
            # A warmup worker died or the pool was closed. The readiness probe
            # failing must degrade to serial (the documented contract), not
            # propagate and abort the caller's algorithm selection.
            log.warning("Process pool unusable (%s); falling back to serial", e)
            return False
        return True

    @classmethod
    def wakeup(cls) -> None:
        # [FUNCTION DOC] If the process pool is usable and is a SubprocPool, tells the sidecar to start its executor. Otherwise does nothing.
        """
        If using a SubprocPool, signal the sidecar process to start up its
        ProcessPoolExecutor.
        """
        if not cls.use_process_pool():
            return
        pool = cls.process_pool()
        if isinstance(pool, SubprocPool):
            pool.wakeup()

    def triton(self, kernel_name: str, source_code: str, device_str: str = "cuda"):
        # [FUNCTION DOC] Compiles a Triton kernel from source. Increments cache-miss counter and logs the source. If TRITON_INTERPRET is 1, loads the module
        # through PyCodeCache and returns the named attribute. Checks CompiledTritonKernels first: on hit returns the cached future (parallel)
        # or its result (serial). On miss with a usable process pool: submits a worker compile task (passing selected env vars and config,
        # including a filtered autotune lookup table entry), wraps it in a LambdaFuture that on result restores and precompiles the kernel
        # in the parent, records compile metrics, and stores that future in the cache. Otherwise compiles synchronously in-process (setting
        # ptxas/libdevice paths) and returns the precompiled kernel, logging triton build failures. Args: kernel_name, source_code, device_str
        # (unused). Returns a future or a kernel. Raises the underlying compile exception, or the subprocess exception renamed with the kernel name.
        # KNOWN ISSUE: device_str is accepted but never used.
        """
        Async_compile.triton is more complicated than the other backends because
        we're trying to optimize compile time as much as possible for this hot callsite.

        First of all, the function is cached by CompiledTritonKernels; if there's a kernel
        already compiled, we grab it directly from the cache and return.

        Otherwise, if we have multiple compile threads, we kick off triton compilations on each
        worker process by giving it a kernel and source code to compile. The worker initializes
        a CachingAutotuner, runs triton compilation, and pickles the kernel back to us.
        We use TritonCompileResult to represent the objects being pickled back to us by each
        worker.

        Some maybe not obvious things that are pickled back to us:
        - Most of the time, we can avoid sending back CachingAutotuner.fn and other metadata
          and do not have to pay the cost of loading the triton kernel on the parent. But certain
          cases, like coordesc tuning and dynamic_scale_rblock, require us to reload the function
          in the parent lazily when we require it.
        - The AutotuneCache, if enabled, is constructed on each worker per triton config
          and pickled by to us via `CachingAutotuner.save_cache_hook`.
        """
        load_kernel = functools.partial(
            _load_triton_kernel_from_source, kernel_name, source_code
        )

        def reload_kernel_in_parent():
            # [FUNCTION DOC] HELPER: Reloads the kernel from source in the parent process, timed under the reload_kernel_in_parent event; returns the loaded kernel.
            # Benchmark how often this happens
            with dynamo_timed("reload_kernel_in_parent"):
                return load_kernel()

        counters["inductor"]["async_compile_cache_miss"] += 1

        kernel_code_log.info("Triton Kernel:\n%s", source_code)
        _compile_start()

        if os.environ.get("TRITON_INTERPRET", "0") == "1":
            return getattr(
                torch._inductor.codecache.PyCodeCache.load(source_code), kernel_name
            )

        is_parallel = self.use_process_pool()
        set_feature_use("parallel_compile_post_warmup", is_parallel)

        compile_id = torch._guards.CompileContext.current_compile_id()
        is_backward = getattr(V.graph, "is_backward", False)

        if (future := CompiledTritonKernels.get(source_code)) is not None:
            counters["inductor"]["async_compile_cache_hit"] += 1
            # Set reload_kernel_from_src properly based on source_code
            if isinstance(future, StaticAutotunerFuture):
                # Remove the future now that we've cache hit
                CompiledTritonKernels.remove_future(source_code)
                future.reload_kernel_from_src = reload_kernel_in_parent
            if is_parallel:
                return future
            else:
                return future.result()

        # Cache miss
        if is_parallel:
            # Ensure libdevice path is set in os.environ before passing to workers
            _set_triton_libdevice_path()
            # We want to support changing these env vars after (and while) the
            # process pool is running, so pass them to the subprocess to reset.
            env_vars = [
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TRITON_LIBDEVICE_PATH",
            ]
            extra_env = {v: os.environ.get(v) for v in env_vars}
            extra_config = {
                "use_static_triton_launcher": torch._inductor.config.use_static_triton_launcher
            }

            if len(torch._inductor.config.autotune_lookup_table) > 0:
                m = size_hints_regex.search(source_code)
                if m:
                    size_hints_str = m.group(1)
                else:
                    size_hints_str = str(None)

                triton_src = source_code.split("@triton.jit\n")[1]
                from torch._inductor.runtime.triton_heuristics import (
                    generate_lookup_hash_from_source_code,
                )

                fn_hash = generate_lookup_hash_from_source_code(
                    size_hints_str, triton_src
                )

                if fn_hash in torch._inductor.config.autotune_lookup_table:
                    extra_config["autotune_lookup_table"] = {  # type: ignore[assignment]
                        fn_hash: torch._inductor.config.autotune_lookup_table[fn_hash]
                    }

            task = self.process_pool().submit(
                _worker_compile_triton,
                load_kernel,
                extra_env,
                extra_config,
            )

            def get_result() -> CachingAutotuner:
                # [FUNCTION DOC] HELPER: Waits for the worker compile, re-raising subprocess errors with the kernel name; then sets compile info, drops the cached future,
                # restores unpickled state, precompiles in the parent and emits compile-time metrics. Returns the ready kernel.
                try:
                    kernel, elapsed_us = task.result()
                except SubprocException as e:
                    raise e.with_name(kernel_name) from e

                # Now that we've compiled, we should clear the future
                # so it can't be used again
                kernel.set_compile_info(compile_id, is_backward)
                CompiledTritonKernels.remove_future(source_code)

                kernel.restore_after_unpickle(old_values=None)

                kernel.precompile(
                    warm_cache_only=False,
                    reload_kernel=reload_kernel_in_parent,
                    static_triton_bundle_key=CompiledTritonKernels.key(source_code),
                )
                _emit_triton_kernel_compile_metric(kernel, kernel_name, elapsed_us)
                return kernel

            future = LambdaFuture(get_result, future=task)
            CompiledTritonKernels.save(source_code, future)
            return future
        else:
            with dynamo_timed(
                "async_compile.precompile",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="triton_compile_time_us",
                log_waitcounter=True,
                waitcounter_name_override="compile_triton",
            ):
                # [FUNCTION DOC] Logs ROCm source and submits a task that optionally AOT-compiles an object file (recorded in the ROCm cache), optionally builds a test runner executable
                # when config requests it, then loads and returns the first loaded result. Returns the submit result.
                fail = None
                try:
                    start_ns = time_ns()
                    _set_triton_ptxas_path()
                    _set_triton_libdevice_path()
                    kernel = load_kernel()
                    kernel.set_compile_info(compile_id, is_backward)
                    kernel.precompile(
                        warm_cache_only=False,
                        static_triton_bundle_key=CompiledTritonKernels.key(source_code),
                    )
                    elapsed_us = (time_ns() - start_ns) // 1000
                    _emit_triton_kernel_compile_metric(kernel, kernel_name, elapsed_us)
                    return kernel
                except Exception as e:
                    fail = str(e)
                    raise
                finally:
                    log_triton_builds(fail=fail)

    def multi_kernel(self, *args, **kwargs) -> Any:
        # [FUNCTION DOC] Builds and returns a MultiKernelCall from the arguments (imported lazily). Not parallelized since sub-kernels are already compiled in parallel.
        from torch._inductor.codegen.multi_kernel import MultiKernelCall

        # no need to call this in parallel since the sub-kernels are already parallel tasks
        return MultiKernelCall(*args, **kwargs)

    def size_hint_multi_kernel(self, *args, **kwargs) -> Any:
        # [FUNCTION DOC] Builds and returns a SizeHintMultiKernelCall from the arguments (imported lazily).
        from torch._inductor.codegen.multi_kernel import SizeHintMultiKernelCall

        return SizeHintMultiKernelCall(*args, **kwargs)

    def cpp(self, source_code: str):
        # [FUNCTION DOC] Compiles C++ kernel source. Serial mode returns the loaded kernel entry; otherwise returns a LambdaFuture that yields the kernel entry once the async load finishes.
        kernel_code_log.info("CPP Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppCodeCache.load(source_code).kernel
        else:
            get_result = CppCodeCache.load_async(source_code, submit_fn=self.submit)
            return LambdaFuture(lambda: get_result().kernel)

    def cpp_pybinding(self, argtypes: list[str], source_code: str):
        # [FUNCTION DOC] Compiles C++ source with Python bindings for the given argument types. Serial mode returns the loaded binding directly; otherwise a LambdaFuture over the async load.
        kernel_code_log.info("CPP+Bindings Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppPythonBindingsCodeCache.load_pybinding(argtypes, source_code)
        else:
            get_result = CppPythonBindingsCodeCache.load_pybinding_async(
                argtypes, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def cutlass(self, cache_cls, source_code, dst_file_ext, aot_compile=False):
        # [FUNCTION DOC] Submits a task that, using the given code cache class, optionally (aot_compile) compiles an object file and records its path in the
        # class's aot kernel list, then loads and returns the first element of the loaded result. Returns the submit result (value or future).
        def task():
            # [FUNCTION DOC] HELPER: Worker body: optionally AOT-compiles an object file via the cache class and records its path, then loads and returns the kernel.
            if aot_compile:
                # We rely on JITInductor to compile the CUDA code,
                # so that we can load it into AOTInductor.
                output_path, *_ = cache_cls.compile(source_code, "o")
                cache_cls.aot_kernels_o.append(output_path)
            return cache_cls.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def cuda(self, source_code, dst_file_ext, aot_compile=False):
        # [FUNCTION DOC] Logs the CUDA source and compiles it through the CUDA code cache via the cutlass helper.
        kernel_code_log.info("CUDA Kernel:\n%s", source_code)
        return self.cutlass(CUDACodeCache, source_code, dst_file_ext, aot_compile)

    def xpu(self, source_code, dst_file_ext, aot_compile=False):
        # [FUNCTION DOC] Logs the XPU source and compiles it through the XPU code cache via the cutlass helper.
        kernel_code_log.info("XPU Kernel:\n%s", source_code)
        return self.cutlass(XPUCodeCache, source_code, dst_file_ext, aot_compile)

    def rocm(
        self,
        source_code,
        dst_file_ext,
        aot_compile=False,
    ):
        # [FUNCTION DOC] Compiles an NVIDIA Universal GEMM kernel. With a usable process pool, submits a worker task and returns a LambdaFuture that loads the wrapper
        # when done (re-raising subprocess errors with the kernel name). Otherwise writes and loads the source through PyCodeCache and returns an
        # NVUniversalGemmKernelWrapper. Raises RuntimeError if the main function is missing in the serial path. Args: kernel_name, source_code, precompile_metadata.
        kernel_code_log.info("ROCm Kernel:\n%s", source_code)

        def task():
            # [FUNCTION DOC] HELPER: Worker body: optionally AOT-compiles an object and builds an executable test runner, then loads and returns the kernel.
            if aot_compile:
                output_path, *_ = ROCmCodeCache.compile(source_code, dst_file_ext="o")
                ROCmCodeCache.aot_kernels_o.append(output_path)
            if config.rocm.generate_test_runner:
                _ = ROCmCodeCache.compile(source_code, dst_file_ext="exe")
            return ROCmCodeCache.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def halide(self, meta: HalideMeta, source_code: str):
        # [FUNCTION DOC] Compiles a Halide kernel. Serial mode generates and returns it directly; otherwise returns a LambdaFuture over the async generation.
        kernel_code_log.info("Halide Kernel:\n%r\n%s", meta, source_code)
        if get_compile_threads() <= 1:
            return HalideCodeCache.generate_halide(meta, source_code)
        else:
            get_result = HalideCodeCache.generate_halide_async(
                meta, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def _load_kernel_wrapper(self, kernel_name, main_suffix, wrapper_cls, key, path):
        # [FUNCTION DOC] HELPER: Loads the module from PyCodeCache by key and path, looks up the entry function named kernel name plus suffix, and returns it wrapped
        # in wrapper_cls with the kernel path.
        """Reload a kernel module from PyCodeCache and wrap the entry point."""
        mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)
        main_func_name = f"{kernel_name}_{main_suffix}"
        return wrapper_cls(getattr(mod, main_func_name), kernel_path=path)

    def cutedsl(self, kernel_name: str, source_code: str, precompile_metadata=None):
        # [FUNCTION DOC] Compiles a CuteDSL kernel. With a usable process pool, submits a worker task and returns a LambdaFuture that loads the wrapper once done
        # (re-raising subprocess errors with the kernel name). Otherwise writes the source via PyCodeCache, loads it and returns a CuteDSLKernelWrapper.
        # Raises RuntimeError if the expected main function is missing in the serial path. Brackets with compile start; logs the source.
        """
        Compile CuteDSL (CUTLASS Python DSL) kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the CuteDSL kernel, as a string
            precompile_metadata: Optional dict with shapes/dtypes for triggering
                real CuTe DSL compilation in the subprocess worker.

        Note:
            CuteDSL currently requires source files to do its compilation, there we
            use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
            CuteDSLKernelWrapper,
            MAIN_SUFFIX,
        )

        kernel_code_log.info("CuteDSL Kernel:\n%s", source_code)
        _compile_start()

        is_parallel = self.use_process_pool()

        if is_parallel:
            extra_env = _pycodecache_kernel_compile_env()

            subprocess_task = self.process_pool().submit(
                _worker_compile_pycodecache_kernel,
                kernel_name,
                source_code,
                MAIN_SUFFIX,
                extra_env,
                precompile_metadata,
            )

            def get_result() -> CuteDSLKernelWrapper:
                # [FUNCTION DOC] HELPER: Waits for the subprocess compile (re-raising errors with the kernel name), logs the elapsed time and loads the kernel wrapper from the written module.
                try:
                    key, path, elapsed_us = subprocess_task.result()
                except SubprocException as e:
                    raise e.with_name(kernel_name) from e
                log.debug(
                    "CuteDSL kernel %s compiled in subprocess in %dus",
                    kernel_name,
                    elapsed_us,
                )
                return self._load_kernel_wrapper(
                    kernel_name, MAIN_SUFFIX, CuteDSLKernelWrapper, key, path
                )

            return LambdaFuture(get_result, future=subprocess_task)
        else:
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find CuteDSL main kernel function '{main_func_name}'. Available callables: {available}"
                )

            return CuteDSLKernelWrapper(getattr(mod, main_func_name), kernel_path=path)

    def pallas(self, kernel_name: str, source_code: str):
        # [FUNCTION DOC] Compiles a Pallas kernel by writing and loading the source via PyCodeCache and wrapping the main entry in a PallasKernelWrapper.
        # Runs inline when compile threads <= 1, otherwise on the thread pool, returning a LambdaFuture. Raises RuntimeError if the main function is missing.
        """
        Compile Pallas (JAX experimental) kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the Pallas kernel, as a string

        Note:
            Pallas kernels are Python code that uses JAX and Pallas APIs.
            We use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.pallas import MAIN_SUFFIX, PallasKernelWrapper

        kernel_code_log.info("Pallas Kernel:\n%s", source_code)

        def task():
            # [FUNCTION DOC] HELPER: Worker body: writes the source via PyCodeCache, loads the module, and returns a PallasKernelWrapper; raises RuntimeError if the main function is absent.
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            # Find our special entry point named function
            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find Pallas main kernel function '{main_func_name}'. Available callables: {available}"
                )

            return PallasKernelWrapper(getattr(mod, main_func_name), kernel_path=path)

        if get_compile_threads() <= 1:
            return task()
        else:
            future = self.submit(task)
            return LambdaFuture(lambda: future.result())

    def nv_universal_gemm(
        self, kernel_name: str, source_code: str, precompile_metadata=None
    ):
        # [FUNCTION DOC] Submits NVGEMM kernel precompilation to the process pool (subprocess isolation for a thread-unsafe compile) with the given kernel/variant,
        # accumulator type, tensor metadata, scale/swizzle types, epilogue flags and CUDA context. Returns the pool future; the artifact is written to the disk cache.
        """
        Compile NVIDIA Universal GEMM kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the kernel, as a string
            precompile_metadata: Optional dict with shapes/dtypes for triggering
                real CuTe DSL compilation in the subprocess worker.

        Note:
            NVIDIA Universal GEMM kernels are Python code that calls the cutlass.operators library.
            We use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            NVUniversalGemmKernelWrapper,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_scheduling import (
            MAIN_SUFFIX,
        )

        kernel_code_log.info("NVIDIA Universal GEMM Kernel:\n%s", source_code)
        _compile_start()

        is_parallel = self.use_process_pool()

        if is_parallel:
            extra_env = _pycodecache_kernel_compile_env()

            subprocess_task = self.process_pool().submit(
                _worker_compile_pycodecache_kernel,
                kernel_name,
                source_code,
                MAIN_SUFFIX,
                extra_env,
                precompile_metadata,
            )

            def get_result() -> NVUniversalGemmKernelWrapper:
                # [FUNCTION DOC] HELPER: Waits for the subprocess compile (re-raising errors with the kernel name), logs the elapsed time and loads the kernel wrapper from the written module.
                try:
                    key, path, elapsed_us = subprocess_task.result()
                except SubprocException as e:
                    raise e.with_name(kernel_name) from e
                log.debug(
                    "NV Universal GEMM kernel %s compiled in subprocess in %dus",
                    kernel_name,
                    elapsed_us,
                )
                return self._load_kernel_wrapper(
                    kernel_name, MAIN_SUFFIX, NVUniversalGemmKernelWrapper, key, path
                )

            return LambdaFuture(get_result, future=subprocess_task)
        else:
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find NVIDIA Universal GEMM main kernel function "
                    f"'{main_func_name}'. Available callables: {available}"
                )

            return NVUniversalGemmKernelWrapper(
                getattr(mod, main_func_name), kernel_path=path
            )

    def nvgemm_precompile(
        self,
        kernel_name,
        variant_name,
        accumulator_type,
        input_tensor_meta,
        output_tensor_meta,
        cuda_ctx,
        scale_type_a=None,
        scale_type_b=None,
        swizzle_type_a=None,
        swizzle_type_b=None,
        has_bias_epilogue=False,
        swap_ab=False,
        metadata=None,
    ):
        """Submit NVGEMM kernel precompilation to the subprocess pool.

        Compiles the CuTeDSL kernel artifact in a subprocess worker so the
        thread-unsafe kernel.compile() call is process-isolated. The compiled
        artifact is saved to the disk cache; the main process loads it on the
        next cache lookup during benchmarking.
        """
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _worker_nvgemm_autotuning_precompile,
        )

        extra_env = _pycodecache_kernel_compile_env()

        return self.process_pool().submit(
            _worker_nvgemm_autotuning_precompile,
            kernel_name,
            variant_name,
            accumulator_type,
            input_tensor_meta,
            output_tensor_meta,
            extra_env,
            cuda_ctx,
            scale_type_a,
            scale_type_b,
            swizzle_type_a,
            swizzle_type_b,
            has_bias_epilogue,
            swap_ab,
            metadata,
        )

    def metal(self, kernel_name: str, source: str, headers: list[str]) -> None:
        # [FUNCTION DOC] Registers a Metal kernel (name, source, headers) in the class-level pending list, creating it if needed. Compilation happens later in wait().
        # KNOWN ISSUE: the list is assigned on the instance, not the class, so registrations are per AsyncCompile instance.
        """Register a Metal kernel body; wait() compiles all registered kernels into one library."""
        if self._metal_sources is None:
            self._metal_sources = []
        self._metal_sources.append((kernel_name, source, headers))

    def wait(self, scope: dict[str, Any]) -> None:
        # [FUNCTION DOC] Resolves all pending work: when threads > 1 waits for every future in scope (timed); then compiles all registered Metal shaders and
        # merges the resulting kernels into scope, clearing the list; finally ends the compile phase (flushes metrics). Mutates scope.
        if get_compile_threads() > 1:
            with dynamo_timed(
                "async_compile.wait",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="triton_compile_time_us",
                log_waitcounter=True,
                waitcounter_name_override="compile_triton",
            ):
                self._wait_futures(scope)

        if self._metal_sources:
            from torch._inductor.runtime.runtime_utils import compile_mps_shaders

            scope.update(compile_mps_shaders(self._metal_sources))
            self._metal_sources.clear()

        _compile_end()

    def _wait_futures(self, scope: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Replaces each future value in scope with its result, showing a progress bar. Uses config.compile_worker_wait_timeout (0 means no timeout).
        # Raises RuntimeError on timeout (hinting an env var) or when a compile subprocess died (hinting to run with one compile thread).
        kernels = {
            key: value
            for key, value in scope.items()
            if isinstance(value, (Future, CodeCacheFuture))
        }
        pbar = tqdm(
            total=len(kernels),
            desc="Inductor Compilation",
            disable=config.disable_progress,
            delay=0,
        )
        # compile_worker_wait_timeout=0 (default) means "wait forever"; map
        # it to None so both Future.result() and CodeCacheFuture.result()
        # receive the same "no timeout" sentinel.
        wait_timeout = config.compile_worker_wait_timeout or None
        for key, result in kernels.items():
            if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                pbar.set_postfix_str(key)
            try:
                kernel = result.result(timeout=wait_timeout)
                scope[key] = kernel
            except FuturesTimeoutError as e:
                # concurrent.futures.TimeoutError became an alias of the
                # builtin TimeoutError in Python 3.11; on 3.10 it is a
                # distinct class, so catch it explicitly.
                raise RuntimeError(
                    f"Inductor compile-worker future for {key!r} did not "
                    f"complete within {wait_timeout}s. Override with "
                    "TORCHINDUCTOR_COMPILE_WORKER_WAIT_TIMEOUT=<seconds>."
                ) from e
            except BrokenProcessPool as e:
                raise RuntimeError(
                    "A compilation subprocess exited unexpectedly. This "
                    "is likely due to a crash. To facilitate debugging, "
                    "you can re-run with TORCHINDUCTOR_COMPILE_THREADS=1 "
                    "to cause compilation to occur in the main process."
                ) from e
            pbar.update(1)


def maybe_warm_pool() -> None:
    # [FUNCTION DOC] Starts and wakes the compile pool early unless disabled by environment (TNT in use, warm pool turned off), Triton missing, or fbcode. No return value.
    if (
        os.environ.get("TORCH_TNT_IN_USE", "0") == "1"
        or os.environ.get("TORCH_WARM_POOL", "1") != "1"
        # The subprocess pool is only used for the Triton backend
        or not has_triton_package()
        # Skip for fbcode. We have internal reports of usages inside multiprocessing
        # pools that lead a multiplicative number of compile subprocesses.
        or config.is_fbcode()
    ):
        return

    AsyncCompile.warm_pool()
    # TODO: This starts the SubprocPool's internal process pool as early as possible at
    # the expense of creating a bunch of worker processes that might not be needed. We
    # could start them lazily if we're willing to lose a small amount of compile time.
    AsyncCompile.wakeup()


# On exit give the workers a chance to clean themselves up. Without this the
# resource_tracker can complain about leaked semaphores coming from the
# ProcessPoolExecutor:
#   UserWarning: resource_tracker: There appear to be 5 leaked semaphore objects
#   to clean up at shutdown
atexit.register(shutdown_compile_workers)
