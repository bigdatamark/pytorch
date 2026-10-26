# mypy: allow-untyped-defs
# [MODULE DOC] Support for autotuning kernel choices in separate processes. Contains the
# TuningProcess/TuningProcessPool pair that run benchmark requests in pipe-connected
# child Python processes (one per visible GPU), serializable BenchmarkRequest classes
# (Triton, CUDA, CPP, XPU/ROCm variants) with device benchmarking mixins, TensorMeta for
# describing tensors across process boundaries, and an AutotuneProcessPool for running
# generic jobs. Import-time side effects: reads the TORCHINDUCTOR_AUTOTUNE_POOL_INACTIVITY_TIMEOUT
# environment variable and creates the autotuning artifact logger.
from __future__ import annotations

import atexit
import contextvars
import ctypes
import dataclasses
import functools
import logging
import multiprocessing as mp
import os
import pickle
import queue
import selectors
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p, CDLL
from typing import Any, IO, TYPE_CHECKING
from typing_extensions import override

import torch
import torch._inductor.async_compile
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided
from torch._inductor import ir
from torch._inductor.codecache import (
    CppCodeCache,
    CUDACodeCache,
    DLLWrapper,
    get_hash,
    PyCodeCache,
    write,
    XPUCodeCache,
)
from torch._inductor.compile_worker.timer import Timer
from torch._inductor.utils import (
    apply_subprocess_env,
    clear_caches,
    clear_on_fresh_cache,
    do_bench_using_profiling,
    get_gpu_type,
    get_ld_library_path,
    is_gpu,
    python_subprocess_env,
)
from torch._logging import getArtifactLogger
from torch.utils._ordered_set import OrderedSet


# Inactivity timeout for AutotuneProcessPool in seconds.
# Default: 600 seconds (10 minutes). Set to 0 to disable.
AUTOTUNE_POOL_INACTIVITY_TIMEOUT = int(
    os.environ.get("TORCHINDUCTOR_AUTOTUNE_POOL_INACTIVITY_TIMEOUT", "600")
)

if TYPE_CHECKING:
    from types import ModuleType

    from torch._inductor.select_algorithm import (
        ChoiceCaller,
        PartialRender,
        TritonTemplateCaller,
    )

from . import config
from .runtime.benchmarking import benchmarker
from .virtualized import V


CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


_visible_device_env_var_maps = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "xpu": "ZE_AFFINITY_MASK",
}


def get_visible_devices_env_var(gpu_type: str | None = None) -> str:
    # [FUNCTION DOC] Returns the environment variable that restricts visible devices for
    # the GPU type (CUDA_VISIBLE_DEVICES for cuda, ZE_AFFINITY_MASK for xpu); the type
    # defaults to the detected GPU type. Raises ValueError for unsupported types.
    if gpu_type is None:
        gpu_type = get_gpu_type()
    if gpu_type not in _visible_device_env_var_maps:
        raise ValueError(f"Unsupported gpu_type: {gpu_type}")
    return _visible_device_env_var_maps[gpu_type]


autotuning_log = getArtifactLogger(__name__, "autotuning")


class NonzeroWorkspaceNotSupportedError(Exception):
    # [CLASS DOC] Raised when a kernel needs a nonzero workspace, which autotune benchmarking
    # does not support; benchmark() catches it and reports infinite time so the op is skipped.
    pass


def _cache_env_for_subprocess() -> dict[str, str | None]:
    # [FUNCTION DOC] HELPER: Snapshots the current values (or none) of the Inductor and
    # Triton cache directory environment variables to forward to a subprocess.
    env_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
    return {v: os.environ.get(v) for v in env_vars}


_last_applied_cache_env: dict[str, str | None] | None = None


def _apply_subprocess_env_and_clear_caches(
    extra_env: dict[str, str | None] | None,
) -> None:
    # [FUNCTION DOC] HELPER: Does nothing if extra_env is none. Otherwise, if it differs from
    # the last applied one, clears Inductor caches and records a copy; then applies the
    # environment to the current process. Side effects: global state, os.environ.
    global _last_applied_cache_env

    if extra_env is None:
        return

    if extra_env != _last_applied_cache_env:
        clear_caches()
        _last_applied_cache_env = extra_env.copy()
    apply_subprocess_env(extra_env)


def _run_with_subprocess_env(
    fn: Callable[..., Any],
    extra_env: dict[str, str | None],
    *args: Any,
    **kwargs: Any,
) -> Any:
    # [FUNCTION DOC] HELPER: Applies the environment (clearing caches if it changed), then
    # calls fn with the extra arguments and returns its result.
    _apply_subprocess_env_and_clear_caches(extra_env)
    return fn(*args, **kwargs)


class TuningProcess:
    # [CLASS DOC] Wrapper around one benchmarking child process connected by two pipes.
    # State: device (index restricting visibility, or none), process, write_pipe and
    # read_pipe, selector (for read timeouts), running flag. Messages are pickled pairs of
    # object and extra environment. A dead process is restarted lazily on put.
    """
    Class to launch and interact with a benchmarking subprocess.
    """

    @staticmethod
    def process_main(read_pipe: IO[bytes], write_pipe: IO[bytes]) -> None:
        # [FUNCTION DOC] Child process entry: loops receiving (job, extra_env); a none job ends
        # the loop. Applies the environment, runs the job, and sends back its result or the
        # raised exception. Exits quietly when the parent closes the pipe. Logs at debug level.
        """
        Entry point for the child process.
        """
        autotuning_log.debug(
            "Started autotune subprocess %s. Visible devices: %s",
            os.getpid(),
            os.environ.get(get_visible_devices_env_var()),
        )

        def workloop():
            # [FUNCTION DOC] HELPER: Receive-run-reply loop described above; runs until the
            # shutdown sentinel arrives or EOF is raised on the pipe.
            while True:
                job, extra_env = TuningProcess.recv(read_pipe)
                if job is None:
                    # None is a sentinel for the child to shut down
                    break
                try:
                    _apply_subprocess_env_and_clear_caches(extra_env)
                    result = job()
                except Exception as e:
                    result = e
                TuningProcess.send(result, write_pipe)

        try:
            workloop()
        except EOFError:
            # The parent closed the pipe
            pass

    @staticmethod
    def send(
        obj: Any, write_pipe: IO[bytes], extra_env: dict[str, str | None] | None = None
    ) -> None:
        # [FUNCTION DOC] Pickles the pair of object and extra_env onto the pipe and flushes.
        pickle.dump((obj, extra_env), write_pipe)
        write_pipe.flush()

    @staticmethod
    def recv(read_pipe: IO[bytes]) -> Any:
        # [FUNCTION DOC] Unpickles and returns the next (object, extra_env) pair from the pipe;
        # raises EOFError if the pipe is closed.
        return pickle.load(read_pipe)

    def __init__(self, device: int | None):
        # [FUNCTION DOC] Records the device and immediately starts the subprocess.
        self.device = device
        self.start()

    def start(self):
        # [FUNCTION DOC] Creates two pipes, wraps the parent ends in file objects and registers
        # the read end with a selector, then spawns the autotune main script with the current
        # Python, passing the parent pid and child fds. The child env has the standard
        # subprocess env, disabled warm pool, adjusted library path, a bandwidth profiling flag
        # and, if a device is set, the visible-devices variable. Closes the child fds in the
        # parent and sets running.
        """
        Start the benchmarking subprocess.
        """
        entry = os.path.join(os.path.dirname(__file__), "__autotune_main__.py")

        subproc_read_fd, write_fd = os.pipe()
        read_fd, subproc_write_fd = os.pipe()
        self.write_pipe = os.fdopen(write_fd, "wb")
        self.read_pipe = os.fdopen(read_fd, "rb")

        self.selector = selectors.DefaultSelector()
        self.selector.register(self.read_pipe, selectors.EVENT_READ)

        cmd = [
            sys.executable,
            entry,
            f"--parent={os.getpid()}",
            f"--read-fd={str(subproc_read_fd)}",
            f"--write-fd={str(subproc_write_fd)}",
        ]
        env = {
            **python_subprocess_env(),
            # We shouldn't be using the Triton async compile subprocess pool,
            # but as a precaution set the env var that disables its creation.
            "TORCH_WARM_POOL": "0",
            # Some internal usages need a modified LD_LIBRARY_PATH.
            "LD_LIBRARY_PATH": get_ld_library_path(),
            # This will cause the subprocs to profile using the profiler.
            "TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING": "1"
            if config.profile_bandwidth_with_do_bench_using_profiling
            else "0",
        }
        if self.device is not None:
            env[get_visible_devices_env_var()] = str(self.device)
        self.process = subprocess.Popen(
            cmd,
            env=env,
            pass_fds=(subproc_read_fd, subproc_write_fd),
        )
        os.close(subproc_read_fd)
        os.close(subproc_write_fd)

        self.running = True

    def alive(self) -> bool:
        # [FUNCTION DOC] True if started and the subprocess has not exited.
        """
        True if the subprocess is still running.
        """
        return self.running and self.process.poll() is None

    def put(self, req: Any, extra_env: dict[str, str | None] | None = None) -> None:
        # [FUNCTION DOC] Sends a work item (and optional env) to the child, restarting the
        # subprocess first if it is not alive.
        """
        Push a work item to the child process.
        """
        if not self.alive():
            self.start()
        TuningProcess.send(req, self.write_pipe, extra_env=extra_env)

    def get(self, timeout: float = 120.0) -> Any:
        # [FUNCTION DOC] Waits up to timeout seconds for a reply and returns its result; if the
        # child sent an exception, re-raises it. On timeout kills the child and raises
        # TimeoutError; on EOF closes resources and re-raises; on any other error logs, kills
        # the child and re-raises.
        """
        Get a response from the child process. Raises TimeoutError on timeout;
        raises EOFError if the subprocess crashes.
        """
        try:
            if not self.selector.select(timeout):
                raise TimeoutError(f"Timeout in autotune subprocess {self.process.pid}")
            result, _ = TuningProcess.recv(self.read_pipe)
        except TimeoutError:
            self.kill()
            raise
        except EOFError:
            # The subprocess crashed
            self.close()
            raise
        except Exception:
            autotuning_log.exception(
                "Unexpected exception in autotune subprocess %s", self.process.pid
            )
            self.kill()
            raise

        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self, wait: bool = True) -> None:
        # [FUNCTION DOC] If alive, sends the none sentinel asking the child to exit; optionally
        # waits for it.
        """
        Signal the child process to shut down gracefully.
        """
        if self.alive():
            TuningProcess.send(None, self.write_pipe)
        if wait:
            self.wait()

    def wait(self) -> None:
        # [FUNCTION DOC] Waits for the child to exit if alive, then closes resources.
        """
        Wait for the child process to exit.
        """
        if self.alive():
            self.process.wait()
        self.close()

    def close(self) -> None:
        # [FUNCTION DOC] Closes the selector and both pipes and marks the process not running.
        """
        Close resources.
        """
        self.selector.close()
        self.read_pipe.close()
        self.write_pipe.close()
        self.running = False

    def kill(self) -> None:
        # [FUNCTION DOC] If alive, logs an error and sends SIGKILL to the child; then closes
        # resources.
        """
        Send a SIGKILL to the child process.
        """
        if self.alive():
            autotuning_log.error(
                "Sending SIGKILL to autotune subprocess %d",
                self.process.pid,
            )
            self.process.kill()
        self.close()

    def restart(self) -> None:
        # [FUNCTION DOC] Shuts down gracefully (waiting) and starts a new subprocess.
        """
        Gracefully restarts the child process.
        """
        self.shutdown(wait=True)
        self.start()


class TuningProcessPool:
    # [CLASS DOC] Pool of TuningProcess objects, one per device, handed out through a queue,
    # with a thread pool of the same size to dispatch benchmarks. State: processes,
    # process_queue (idle processes), executor.
    """
    Maintains a pool of TuningProcesses to benchmark kernels in parallel
    across devices. By default, we create one TuningProcess per device and
    set the sub-process environment to make only that device visible.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Determines the device list, starts one subprocess per device, fills the
        # idle queue and creates a thread pool sized to the number of devices.
        """
        Start the child processes.
        """
        devices = self.get_device_list()
        autotuning_log.debug("Sub-process autotune device list: %s", devices)

        # Launch the child processes.
        self.processes = [TuningProcess(device=device) for device in devices]

        self.process_queue: queue.Queue[TuningProcess] = queue.Queue()
        for p in self.processes:
            self.process_queue.put(p)

        # Use a thread pool to manage distributing work to the subprocesses.
        # Threads block on an available process, so it makes sense to match
        # the number of threads with the number of devices.
        self.executor = ThreadPoolExecutor(max_workers=len(devices))

    @staticmethod
    def get_device_list() -> Sequence[int | None]:
        # [FUNCTION DOC] Returns the device indices to use: a single none entry when multi-device
        # autotune is off; otherwise the indices listed in the visible-devices env var (raises
        # if it lists more than the device count) or all device indices.
        # KNOWN ISSUE: non-integer entries in the env var (such as UUIDs) would fail to parse.
        """
        Gather the list of devices to be used in the pool.
        """
        if not config.autotune_multi_device:
            # Don't use multiple devices
            return [None]

        gpu_type = get_gpu_type()
        device_interface = get_interface_for_device(gpu_type)
        count = device_interface.device_count()
        visible_devices_env_var = get_visible_devices_env_var(gpu_type)

        # If the user specified the visible devices in the env, use those.
        if visible_devices_env_var in os.environ:
            devices = [int(d) for d in os.environ[visible_devices_env_var].split(",")]
            if not len(devices) <= count:
                raise AssertionError(
                    f"{visible_devices_env_var} specifies {len(devices)} devices, "
                    f"but only {count} devices are available"
                )
            return devices

        return list(range(count))

    def shutdown(self) -> None:
        # [FUNCTION DOC] Shuts down the thread pool, asks all child processes to exit, then
        # waits for each.
        """
        Signal all child processes to exit.
        """
        self.executor.shutdown()

        for p in self.processes:
            p.shutdown(wait=False)
        for p in self.processes:
            p.wait()

    def target(self, choice: TritonTemplateCaller) -> float:
        # [FUNCTION DOC] Thread worker: takes an idle process from the queue, sends the choice
        # benchmark request with the current cache env, and returns the measured time. On
        # timeout or any other failure warns and returns infinity so the choice is ignored;
        # for sticky CUDA launch or illegal address errors restarts the process. Always returns
        # the process to the queue. Raises if bmreq is missing.
        """
        Entry point for the thread-pool helper threads: Wait for an open TuningProcess,
        remove it from the queue, execute the benchmark in that subprocess, and return
        the TuningProcess to the queue.
        """
        if choice.bmreq is None:
            raise AssertionError(
                f"Expected choice.bmreq to be set, but got None for choice '{choice}'"
            )

        extra_env = _cache_env_for_subprocess()
        process = self.process_queue.get()
        process.put(choice.bmreq.benchmark, extra_env=extra_env)
        try:
            return process.get(
                config.max_autotune_subproc_result_timeout_seconds,
            )
        except TimeoutError:
            warnings.warn(
                f"Timed out benchmarking choice '{choice}'. It will be ignored. "
                "Please debug the root cause in case the choice can bring perf gains."
            )
            # Set to INF so this choice will be ignored
            return float("inf")
        except Exception as process_exception:
            warnings.warn(
                f"Failed to benchmark choice '{choice}'. It will be ignored. "
                "Please debug the root cause in case the choice can bring perf gains."
            )
            # Sticky CUDA errors corrupt the context, making it unrecoverable.
            # The process must be restarted to restore CUDA functionality.
            error_msg = str(process_exception)
            if (
                "cudaErrorLaunchFailure" in error_msg
                or "cudaErrorIllegalAddress" in error_msg
            ):
                process.restart()
            # Set to INF so this choice will be ignored
            return float("inf")
        finally:
            self.process_queue.put(process)

    def benchmark(
        self,
        choices: list[TritonTemplateCaller],
    ) -> dict[TritonTemplateCaller, float]:
        # [FUNCTION DOC] Benchmarks all choices in parallel using the thread pool and returns a
        # mapping of choice to time (infinity for failures).
        """
        Benchmark each choice in a separate process.
        """

        # Use a ThreadExecutorPool to spread the work across the subprocesses and
        # to grab subprocesses as soon as they're free.
        results = dict(zip(choices, self.executor.map(self.target, choices)))

        return results


LayoutOrBuffer = ir.Layout | ir.Buffer


@dataclasses.dataclass
class TensorMeta:
    # [CLASS DOC] Picklable description of a tensor (device, dtype, sizes, strides, storage
    # offset, optional name) used to recreate random inputs inside a benchmark process.
    device: torch.device
    dtype: torch.dtype
    sizes: torch._prims_common.ShapeType
    strides: torch._prims_common.StrideType
    offset: int
    name: str | None = None

    @classmethod
    def from_irnodes(
        cls, irnodes: LayoutOrBuffer | Sequence[LayoutOrBuffer]
    ) -> TensorMeta | list[TensorMeta]:
        # [FUNCTION DOC] Builds TensorMeta objects from IR layouts or buffers; sequences map to
        # lists recursively. Layouts are wrapped in a placeholder buffer. Sizes, strides and
        # offset are resolved to concrete integer hints. Raises if a node has no dtype or device.
        from torch._inductor.select_algorithm import get_strides_with_layout_constraints

        if isinstance(irnodes, Sequence):
            result: list[Any] = [cls.from_irnodes(x) for x in irnodes]
            if not all(isinstance(x, TensorMeta) for x in result):
                raise AssertionError(
                    f"Expected all elements to be TensorMeta, got types: "
                    f"{[type(x) for x in result if not isinstance(x, TensorMeta)]}"
                )
            return result

        node = irnodes
        if isinstance(node, ir.Layout):
            node = ir.Buffer(name="fake", layout=node)

        dtype = node.get_dtype()
        if dtype is None:
            raise AssertionError(
                f"Expected node to have a dtype, but get_dtype() returned None for node '{node}'"
            )
        device = node.get_device()
        if device is None:
            raise AssertionError(
                f"Expected node to have a device, but get_device() returned None for node '{node}'"
            )

        return TensorMeta(
            device=device,
            dtype=dtype,
            sizes=V.graph.sizevars.optimization_hints(node.get_size()),
            strides=V.graph.sizevars.optimization_hints(
                get_strides_with_layout_constraints(node)
            ),
            offset=V.graph.sizevars.optimization_hint(node.get_layout().offset),
            name=node.get_name(),
        )

    def to_tensor(self) -> torch.Tensor:
        # [FUNCTION DOC] Creates a random strided tensor on the device with the stored sizes,
        # strides and dtype, with extra storage of offset size.
        return rand_strided(
            self.sizes,
            self.strides,
            device=self.device,
            dtype=self.dtype,
            extra_size=self.offset,
        )


@dataclasses.dataclass
class BenchmarkRequest:
    # [CLASS DOC] Base picklable description of a kernel benchmark: kernel_name,
    # input_tensor_meta (list), output_tensor_meta (single), extra_args, and a
    # benchmark_with_cudagraphs flag. Subclasses implement make_run_fn and do_bench.
    # Must never hold CUDA tensors.
    """
    Only handle triton template benchmark for now. The extern kernel benchmark
    can be done inside the same process since they usually don't cause crash.

    Important: Instances of this class and subclasses have to be serializable
    across process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
    ) -> None:
        # [FUNCTION DOC] Stores the kernel name, normalizes input meta to a list, and reduces a
        # list of output metas to the first (raises if outputs of a grouped GEMM do not share
        # device, dtype, sizes, strides and offset). Cudagraph benchmarking defaults to off.
        # the kernel name defined in the module
        self.kernel_name = kernel_name

        if isinstance(input_tensor_meta, TensorMeta):
            self.input_tensor_meta: list[TensorMeta] = [input_tensor_meta]
        else:
            self.input_tensor_meta: list[TensorMeta] = input_tensor_meta

        if output_tensor_meta and isinstance(output_tensor_meta, (tuple, list)):
            if len(output_tensor_meta) > 1:
                # Each output with same meta for Grouped GEMM
                if not all(
                    getattr(output_tensor_meta[0], attr) == getattr(x, attr)
                    for x in output_tensor_meta
                    for attr in ["device", "dtype", "sizes", "strides", "offset"]
                ):
                    raise AssertionError(
                        "All output tensor metas in a Grouped GEMM must have matching "
                        "device, dtype, sizes, strides, and offset"
                    )
            self.output_tensor_meta = output_tensor_meta[0]
        else:
            # pyrefly: ignore [bad-assignment]
            self.output_tensor_meta: TensorMeta = output_tensor_meta

        self.extra_args = extra_args
        self.benchmark_with_cudagraphs = False

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # [FUNCTION DOC] Abstract: returns a no-argument callable that launches the kernel on
        # the inputs and writes to out. Raises NotImplementedError here.
        raise NotImplementedError

    def cleanup_run_fn(self) -> None:
        # [FUNCTION DOC] Hook to release resources after benchmarking; default does nothing.
        pass

    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        # [FUNCTION DOC] Abstract: times the callable and returns the result. Raises
        # NotImplementedError here.
        raise NotImplementedError

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        # [FUNCTION DOC] Benchmarks the kernel and returns the time. If out is none, creates
        # random inputs and output from the metas (requires no explicit inputs). Builds the run
        # function (returns infinity if a nonzero workspace is required), times it with CUDA
        # graphs or do_bench, logs timing breakdown at debug level, and always calls cleanup.
        # KNOWN ISSUE: do_bench is called with out as a positional argument, so it lands in the
        # input tensors and the out keyword stays none.
        debug = autotuning_log.isEnabledFor(logging.DEBUG)
        if debug:
            start_ts = time.time()

        # create args and out tensor
        if out is None:
            if not (self.input_tensor_meta and self.output_tensor_meta):
                raise AssertionError(
                    "Input and output tensor meta must be populated when input_tensors is empty"
                )
            if not len(input_tensors) == 0:
                raise AssertionError(
                    f"Expected no input_tensors when out is None, but got {len(input_tensors)}"
                )
            input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
            out = self.output_tensor_meta.to_tensor()

        if debug:
            create_tensor_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            start_ts = time.time()
        try:
            try:
                fn = self.make_run_fn(*input_tensors, out=out)
            except NonzeroWorkspaceNotSupportedError:
                # Skipping all ops with nonzero workspace requirements
                autotuning_log.info("Skipping op due to nonzero workspace requirement")
                return float("inf")

            if debug:
                load_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
                start_ts = time.time()

            if self.benchmark_with_cudagraphs:
                res = benchmarker.benchmark_gpu_with_cuda_graph(fn)
            else:
                res = self.do_bench(fn, *input_tensors, out)

            if debug:
                bench_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
                autotuning_log.debug(
                    "InChildProcess %s: load %f, create tensor %f, bench %f",
                    self,
                    load_elapse,  # type: ignore[possibly-undefined]
                    create_tensor_elapse,  # type: ignore[possibly-undefined]
                    bench_elapse,
                )
            return res
        finally:
            self.cleanup_run_fn()


class _TestBenchmarkRequest(BenchmarkRequest):
    # [CLASS DOC] HELPER: Test-only request that returns a canned result, optionally after a
    # sleep, raising an exception, or exiting the process, and optionally checking the device
    # visibility variable. Does not call the parent constructor.
    """
    Supports unit testing. Defined in this file instead of the test file so the
    TuningProcess sub-process can unpickle these objects.
    """

    def __init__(
        self,
        result: float = 0.0,
        device: int | None = None,
        sleep: float | None = None,
        exc: Exception | None = None,
        crash: bool = False,
    ):
        # [FUNCTION DOC] HELPER: Stores result, device, sleep, exception and crash settings.
        self.result = result
        self.device = device
        self.sleep = sleep
        self.exc = exc
        self.crash = crash

    def benchmark(
        self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None
    ) -> float:
        # [FUNCTION DOC] HELPER: Verifies the visible-devices variable equals the device (if
        # set), sleeps, raises the configured exception, exits with status 1 if crash is set,
        # else returns the canned result.
        if self.device is not None:
            visible_devices_env_var = get_visible_devices_env_var()
            if os.environ.get(visible_devices_env_var, None) != str(self.device):
                raise AssertionError(
                    f"Expected {visible_devices_env_var}='{self.device}', "
                    f"but got '{os.environ.get(visible_devices_env_var, None)}'"
                )
        if self.sleep:
            time.sleep(self.sleep)
        if self.exc:
            raise self.exc
        if self.crash:
            sys.exit(1)
        return self.result


class _TestEnvBenchmarkRequest:
    # [CLASS DOC] HELPER: Test-only request whose benchmark reports an environment variable
    # value as seen by the subprocess.
    """
    Supports unit testing subprocess environment propagation.
    """

    def __init__(self, key: str):
        # [FUNCTION DOC] HELPER: Stores the environment variable name.
        self.key = key

    def benchmark(
        self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None
    ) -> str | None:
        # [FUNCTION DOC] HELPER: Returns the value of the variable in this process, or none.
        return os.environ.get(self.key)


@clear_on_fresh_cache
class _TestCodeCache:
    # [CLASS DOC] HELPER: Test-only cache that writes source text to a file with memoization;
    # registered to be cleared on fresh cache.
    @classmethod
    def cache_clear(cls) -> None:
        # [FUNCTION DOC] HELPER: Clears the memoized write results.
        cls.write.cache_clear()

    @classmethod
    @functools.lru_cache(None)
    def write(cls, source_code: str) -> tuple[str, str]:
        # [FUNCTION DOC] HELPER: Memoized; writes the source text as a txt cache file and
        # returns the pair of hash key and path.
        return write(source_code, "txt")


class _TestCodeCacheBenchmarkRequest:
    # [CLASS DOC] HELPER: Test-only request used to check that code caches are reset in
    # subprocesses; holds fixed source text.
    """
    Supports unit testing subprocess codecache resets.
    """

    source_code = "test codecache contents\n"

    def benchmark(
        self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None
    ) -> str:
        # [FUNCTION DOC] HELPER: Writes the fixed source via the test cache and returns its path.
        _, input_path = _TestCodeCache.write(self.source_code)
        return input_path


class GPUDeviceBenchmarkMixin:
    # [CLASS DOC] Mixin providing do_bench for GPU requests by timing on the tensors device.
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        # [FUNCTION DOC] Finds the single GPU device index among inputs and out (raises if
        # several), defaulting to the current device and cuda type, then under that device
        # benchmarks fn with the benchmarker, synchronizes to surface errors, and returns time.
        device_idx_set = OrderedSet(
            tensor.device.index
            for tensor in [*input_tensors, out]
            if isinstance(tensor, torch.Tensor)
            and is_gpu(tensor.device.type)
            and tensor.device.index is not None
        )
        if not len(device_idx_set) <= 1:
            raise AssertionError(f"Can not mix devices {device_idx_set}")
        device_type = next(
            (
                tensor.device.type
                for tensor in input_tensors
                if is_gpu(tensor.device.type)
            ),
            "cuda",
        )
        device_interface = get_interface_for_device(device_type)
        if len(device_idx_set) == 1:
            device_idx = next(iter(device_idx_set))
        else:
            device_idx = device_interface.current_device()
        with device_interface.device(device_idx):  # type: ignore[attr-defined]
            res = benchmarker.benchmark(fn, device=device_type)
            device_interface.synchronize()  # shake out any CUDA errors

        return res


class CPUDeviceBenchmarkMixin:
    # [CLASS DOC] Mixin providing do_bench for CPU requests.
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        # [FUNCTION DOC] Returns the CPU benchmark time of fn; tensors are ignored.
        return benchmarker.benchmark_cpu(fn)


class TritonBenchmarkRequest(BenchmarkRequest):
    # [CLASS DOC] Picklable request to benchmark a Triton template kernel living in a cached
    # module. State: module_path and module_cache_key to locate it, launch options
    # (num_stages, num_warps, warp-specialization and ROCm tuning knobs), optional workspace
    # size and zero-fill flag, and the loaded module (_benchmark_module) while running.
    """
    Represents a standalone benchmark request for a Triton Template.

    Important: Instances of this class have to be serializable
    across process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        module_path: str,  # the path of the module defining the triton kernel
        module_cache_key: str,
        num_stages: int,
        num_warps: int,
        num_consumer_groups: int = 0,
        num_buffers_warp_spec: int = 0,
        matrix_instr_nonkdim: int = 0,  # only used for hip to choose the shape of mfma instruction.
        waves_per_eu: int = 0,  # only used for hip to schedule waves per execution unit
        kpack: int = 0,  # ROCm specific gemm parameter
        workspace_size: int | None = None,  # size of workspace buffer in bytes
        workspace_zero_fill: bool = False,  # whether to zero-fill workspace
    ) -> None:
        # [FUNCTION DOC] Stores the parent fields plus module location, launch parameters,
        # workspace settings, and an empty loaded-module slot.
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.module_path = module_path
        self.module_cache_key = module_cache_key
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.num_consumer_groups = num_consumer_groups
        self.num_buffers_warp_spec = num_buffers_warp_spec
        self.matrix_instr_nonkdim = matrix_instr_nonkdim
        self.waves_per_eu = waves_per_eu
        self.kpack = kpack
        self.workspace_size = workspace_size
        self.workspace_zero_fill = workspace_zero_fill
        self._benchmark_module: Any | None = None

    @override
    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # [FUNCTION DOC] Loads the kernel module, and prepares the kernel run method; if a
        # workspace size is set, creates a byte tensor on the output device (optionally zeroed)
        # and substitutes it for the placeholder in the extra args. Disables bandwidth info,
        # adds the warmup flag if the run method supports it, picks the raw stream by device
        # (CPU uses stream 0), and returns a callable that launches the kernel with inputs,
        # output and extra args, adding the benchmark_run flag unless it is a debug autotuner.
        # The stream is looked up at call time so CUDA graph capture works. Side effect: stores
        # the loaded module on self.
        mod = PyCodeCache.load_by_key_path(
            self.module_cache_key,
            self.module_path,
            set_sys_modules=False,
        )
        self._benchmark_module = mod
        autotuning_log.debug(
            "benchmark module key: %s, path: %s",
            self.module_cache_key,
            self.module_path,
        )

        run_method = getattr(mod, self.kernel_name).run
        extra_args = list(self.extra_args)

        # Recreate workspace tensor if needed (for TMA templates)
        # The workspace tensor couldn't be pickled, so we recreate it here
        # It should be inserted before the grid values (last 3 elements)
        if self.workspace_size is not None:
            from torch._inductor.select_algorithm import WORKSPACE_ARG_PLACEHOLDER

            workspace_tensor = torch.empty(
                (self.workspace_size,),
                dtype=torch.uint8,
                device=out.device,
            )
            if self.workspace_zero_fill:
                workspace_tensor.zero_()
            workspace_index = extra_args.index(WORKSPACE_ARG_PLACEHOLDER)
            extra_args[workspace_index] = workspace_tensor

        run_method.__self__.with_bandwidth_info = False

        # Newer version of triton add warmup argument to JITFunction.run.
        # This code handles backward-compatibility.
        warmup_arg = {}
        import inspect

        if "warmup" in inspect.signature(run_method).parameters:
            warmup_arg["warmup"] = False

        if out.device.type == "cpu":
            device_interface = None
            device_index = 0
        else:
            device_type = out.device.type
            device_interface = get_interface_for_device(device_type)
            device_index = self.output_tensor_meta.device.index

        is_debug_autotuner = isinstance(
            getattr(mod, self.kernel_name),
            torch._inductor.runtime.triton_heuristics.DebugAutotuner,
        )

        # Resolve the stream at call time (not at closure-creation time) so that
        # CUDA graph capture on a different stream works correctly.
        def run_fn() -> None:
            # [FUNCTION DOC] HELPER: Launches the kernel on the current raw stream for the device.
            stream = (
                0
                if device_interface is None
                else device_interface.get_raw_stream(device_index)
            )
            if is_debug_autotuner:
                run_method(
                    *input_tensors,
                    out,
                    *extra_args,
                    **warmup_arg,
                    stream=stream,
                )
            else:
                run_method(
                    *input_tensors,
                    out,
                    *extra_args,
                    **warmup_arg,
                    stream=stream,
                    benchmark_run=True,
                )

        return run_fn

    def cleanup_run_fn(self) -> None:
        # [FUNCTION DOC] Idempotent cleanup after one benchmark load: forgets the loaded module,
        # removes it from the PyCodeCache module-without-attr table, and asks the kernel to
        # release benchmark artifacts (or, for older kernels, static launchers or by closing
        # launchers and compile results). Then removes the module with this path from the
        # cache module list and line maps. Mutates global PyCodeCache state.
        # Authoritative cleanup for one benchmark module load. Higher-level
        # finally blocks may call this again as safety nets, so keep it idempotent.
        mod = self._benchmark_module
        self._benchmark_module = None

        cached_mod = PyCodeCache.modules_no_attr.pop(self.module_path, None)
        if mod is None:
            mod = cached_mod

        if mod is not None:
            kernel = getattr(mod, self.kernel_name, None)
            release_benchmark_artifacts = getattr(
                kernel, "release_benchmark_artifacts", None
            )
            if release_benchmark_artifacts is not None:
                release_benchmark_artifacts()
            release_static_launchers = getattr(
                kernel, "_release_static_launchers_except", None
            )
            if (
                release_benchmark_artifacts is None
                and release_static_launchers is not None
            ):
                release_static_launchers(None)
            elif release_benchmark_artifacts is None:
                for launcher in getattr(kernel, "launchers", ()) or ():
                    close = getattr(getattr(launcher, "__self__", None), "close", None)
                    if close is not None:
                        close()
                for compile_result in getattr(kernel, "compile_results", ()) or ():
                    close = getattr(
                        getattr(compile_result, "kernel", None), "close", None
                    )
                    if close is not None:
                        close()

        PyCodeCache.modules[:] = [
            module
            for module in PyCodeCache.modules
            if getattr(module, "__file__", None) != self.module_path
        ]
        PyCodeCache.linemaps.pop(self.module_path, None)

    def precompile(self):
        # [FUNCTION DOC] Loads the module, precompiles the kernel, records the register count of
        # the first launcher in n_regs, and always runs cleanup afterward.
        try:
            mod = PyCodeCache.load_by_key_path(
                self.module_cache_key,
                self.module_path,
                set_sys_modules=False,
            )
            self._benchmark_module = mod
            kernel = getattr(mod, self.kernel_name)
            kernel.precompile()

            self.n_regs = kernel.launchers[0].n_regs
        finally:
            self.cleanup_run_fn()

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with kernel name, module path and cache key.
        return f"{self.kernel_name=}, {self.module_path=}, {self.module_cache_key=}"


class TritonGPUBenchmarkRequest(GPUDeviceBenchmarkMixin, TritonBenchmarkRequest):
    # [CLASS DOC] Triton benchmark request timed with GPU device benchmarking; no extra state.
    pass


class TritonCPUBenchmarkRequest(CPUDeviceBenchmarkMixin, TritonBenchmarkRequest):
    # [CLASS DOC] Triton benchmark request timed with CPU benchmarking; no extra state.
    pass


class ExternKernelBenchmarkRequest(BenchmarkRequest):
    # [CLASS DOC] Picklable request to benchmark an extern kernel (such as an aten op) by name
    # from the extern_kernels namespace. State: callable_path (informational), kwargs for the
    # call, and has_out_variant telling whether the kernel writes into a provided output.
    """
    A class to handle extern kernel benchmark requests. This allows extern kernels
    (like aten::mm) to be benchmarked in a subprocess, similar to Triton kernels.

    Important: Instances of this class have to be serializable across
    process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        callable_path: str,  # Module path to the callable (e.g., "extern_kernels.mm")
        kwargs: dict[str, Any] | None = None,
        has_out_variant: bool = True,
    ) -> None:
        # [FUNCTION DOC] Stores parent fields plus callable path, keyword arguments (empty if
        # none) and the out-variant flag.
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.callable_path = callable_path
        self.kwargs = kwargs or {}
        self.has_out_variant = has_out_variant

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # [FUNCTION DOC] Returns a partial of the kernel callable bound to the inputs, passing
        # the output as the out keyword when the kernel has an out variant.
        fn = self.to_callable()
        if self.has_out_variant:
            # For out=variant, pass output as keyword arg
            return functools.partial(fn, *input_tensors, out=out)
        else:
            # For non-out variant, just call with inputs
            return functools.partial(fn, *input_tensors)

    def benchmark(self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None):
        # [FUNCTION DOC] Returns 0.0 for an empty output. With an out variant (or no inputs)
        # defers to the parent benchmark. Otherwise calls the kernel to get a new output, checks
        # its size and stride against out and copies it into out for correctness checks, then
        # times the kernel with CUDA graphs, the profiler, or the standard benchmarker.
        if out is not None and out.numel() == 0:
            # no need to run the kernel or do benchmarking
            return 0.0
        if self.has_out_variant or len(input_tensors) == 0:
            return super().benchmark(*input_tensors, out=out)
        else:
            algo = self.to_callable()
            out_new = algo(*input_tensors)
            if out is not None:
                torch._C._dynamo.guards.assert_size_stride(
                    out_new, tuple(out.size()), tuple(out.stride())
                )
                out.copy_(out_new)  # for correctness checking
            if self.benchmark_with_cudagraphs:
                return benchmarker.benchmark_gpu_with_cuda_graph(
                    lambda: algo(*input_tensors)
                )
            if config.profile_bandwidth_with_do_bench_using_profiling:
                return do_bench_using_profiling(lambda: algo(*input_tensors))
            return benchmarker.benchmark(algo, input_tensors, {})

    def precompile(self) -> None:
        # [FUNCTION DOC] No-op: extern kernels are already compiled.
        # Extern kernels don't need precompilation - they're already compiled
        pass

    def to_callable(self):
        # [FUNCTION DOC] Looks up the kernel by name in extern_kernels (kept picklable by not
        # using the choice object) and returns it, bound to kwargs if any.
        # While ExternKernelChoice also has a to_callable method,
        # we avoid calling the ExternKernelChoice version here to make sure
        # this is picklable
        from torch._inductor.select_algorithm import extern_kernels

        fn = getattr(extern_kernels, self.kernel_name)
        if self.kwargs:
            return functools.partial(fn, **self.kwargs)

        return fn

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string naming the callable path.
        return f"ExternKernelBenchmarkRequest({self.callable_path})"


class ExternKernelGPUBenchmarkRequest(
    GPUDeviceBenchmarkMixin, ExternKernelBenchmarkRequest
):
    # [CLASS DOC] Extern kernel request timed with GPU device benchmarking.
    pass


class ExternKernelCPUBenchmarkRequest(
    CPUDeviceBenchmarkMixin, ExternKernelBenchmarkRequest
):
    # [CLASS DOC] Extern kernel request timed with CPU benchmarking.
    pass


class SubgraphBenchmarkRequest(BenchmarkRequest):
    # [CLASS DOC] Picklable request to benchmark a subgraph choice already compiled in the
    # main process. State: module_path and module_cache_key locating the compiled module and
    # sym_input_values (integer values of symbolic inputs prepended to the call arguments).
    """
    Benchmark request for subgraph choices.

    Pre-compiles the subgraph in the main process and stores
    the module path/cache key for loading in subprocess.
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        module_path: str,
        module_cache_key: str,
        sym_input_values: list[int],
    ) -> None:
        # [FUNCTION DOC] Stores parent fields plus module location and symbolic input values.
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.module_path = module_path
        self.module_cache_key = module_cache_key
        self.sym_input_values = sym_input_values

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # [FUNCTION DOC] Loads the compiled module and returns a callable that invokes its call
        # entry with a fresh list of symbolic values followed by the inputs (fresh because the
        # entry clears its argument list). The out tensor is not passed.
        mod = PyCodeCache.load_by_key_path(
            self.module_cache_key,
            self.module_path,
            set_sys_modules=False,
        )
        sym_input_values = self.sym_input_values
        # Create a new list each call since mod.call does args.clear()
        return lambda: mod.call([*sym_input_values, *input_tensors])

    def precompile(self) -> None:
        # [FUNCTION DOC] No-op: the module is already compiled in the main process.
        # Module is already compiled in main process, no precompilation needed
        pass

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with kernel name and module path.
        return f"SubgraphBenchmarkRequest({self.kernel_name}, {self.module_path})"


class SubgraphGPUBenchmarkRequest(GPUDeviceBenchmarkMixin, SubgraphBenchmarkRequest):
    # [CLASS DOC] Subgraph request timed with GPU device benchmarking.
    pass


class SubgraphCPUBenchmarkRequest(CPUDeviceBenchmarkMixin, SubgraphBenchmarkRequest):
    # [CLASS DOC] Subgraph request timed with CPU benchmarking.
    pass


class CUTLASSBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    # [CLASS DOC] Picklable request to benchmark a CUDA/XPU (CUTLASS) kernel built from C++
    # source into a shared library. State: source_code, device_type and device interface,
    # code cache class (CUDA or XPU), hash_key and source_file from writing the source, the
    # loaded DLL, workspace_size (queried from the kernel) and the allocated workspace.
    # Pickling drops the DLL and workspace and resets the workspace-size-updated flag.
    """
    A class to handle CUDA (CUTLASS) benchmark requests. This class is for
    managing the lifecycle of a CUDA kernel benchmark, including compiling
    the source code, managing workspace memory, and executing the kernel.

    Important: Instances of this class have to be serializable across
    process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        source_code: str,
        device_type: str = "cuda",
    ) -> None:
        # [FUNCTION DOC] Stores parent fields and the source, selects the XPU or CUDA code
        # cache and device interface, and writes the source to the cache as a shared library
        # target, recording hash key and file. Side effect: writes a file.
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: torch.Tensor | None = None
        self.DLL: DLLWrapper | None = None
        self._workspace_size_updated = False
        self.hash_key: str = ""
        self.source_file: str = ""
        self.device_type = device_type
        self.codecache_cls = XPUCodeCache if device_type == "xpu" else CUDACodeCache
        self.device_interface = get_interface_for_device(device_type)
        self.hash_key, self.source_file = self.codecache_cls.write(
            self.source_code, "so"
        )

    def precompile(self):
        # [FUNCTION DOC] Compiles the source into the code cache (possibly from a thread pool)
        # and logs start and end at debug level.
        """
        Precompile the CUDA source code to populate the CUDACodeCache.
        This may happen in a separate thread pool.
        """
        autotuning_log.debug("Precompiling %s", self)
        self.codecache_cls.compile(self.source_code, "so")
        autotuning_log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # [FUNCTION DOC] Loads the library, queries the workspace size, and builds a partial of
        # the kernel entry with data pointers of inputs and output, extra args, a null
        # workspace-size pointer, the workspace pointer (a zeroed float64 tensor sized to cover
        # the bytes when needed) and the current stream. Runs it once as a sanity check; if that
        # raises RuntimeError, cleans up and returns a callable that re-raises the error.
        """
        Create a function to run the CUDA/XPU kernel with the given input and output tensors.
        """

        self.ensure_dll_loaded()
        self.update_workspace_size()
        args = [c_void_p(tensor.data_ptr()) for tensor in list(input_tensors) + [out]]
        autotuning_log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        stream_ptr = c_void_p(
            self.device_interface.get_raw_stream(self.device_interface.current_device())
        )
        run_method = getattr(self.DLL, self.kernel_name)
        workspace_ptr = c_void_p(0)
        if self.workspace_size > 0:
            self.workspace = torch.zeros(
                (self.workspace_size + 7) // 8,
                dtype=torch.float64,
                device=out.device,
            )
            workspace_ptr = c_void_p(self.workspace.data_ptr())

        # Generate partial function.
        ret = functools.partial(
            run_method,
            *args,
            *self.extra_args,
            None,  # null workspace size ptr
            workspace_ptr,  # set workspace ptr,
            stream_ptr,
        )

        # sanity check to make sure we cleanup run fn properly
        try:
            ret()
        except RuntimeError as e:
            err_msg = str(e)

            def raise_runtime_error():
                # [FUNCTION DOC] HELPER: Re-raises the captured RuntimeError message.
                raise RuntimeError(err_msg)

            self.cleanup_run_fn()
            return raise_runtime_error

        return ret

    def update_workspace_size(self) -> None:
        # [FUNCTION DOC] Once per instance, loads the library and calls the kernel with null
        # pointers and a size out-parameter to learn the workspace size in bytes, synchronizes
        # the device, stores the size and logs it.
        if self._workspace_size_updated:
            return
        self.ensure_dll_loaded()
        unique_input_count = len(
            dict.fromkeys(meta.name for meta in self.input_tensor_meta)
        )
        args = [c_void_p(None) for _ in range(unique_input_count + 1)]
        stream_ptr = c_void_p(
            self.device_interface.get_raw_stream(self.device_interface.current_device())
        )

        run_method = getattr(self.DLL, self.kernel_name)
        # Retrieve workspace_size and initialize workspace.
        c_workspace_size = c_size_t()
        run_method(
            *args,  # input ptrs and output ptrs
            *self.extra_args,
            byref(
                c_workspace_size
            ),  # set workspace size ptr to retrieve workspace size
            None,  # null workspace ptr
            stream_ptr,
        )
        self.device_interface.synchronize()  # shake out any device errors
        self.workspace_size = c_workspace_size.value
        autotuning_log.debug(
            "update_workspace_size called: new workspace size=%d, self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.workspace_size,
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        self._workspace_size_updated = True

    def ensure_dll_loaded(self):
        # [FUNCTION DOC] Loads the shared library through the code cache if not yet loaded
        # (setting hash key and source file) and then opens it.
        if self.DLL is None:
            self.DLL, self.hash_key, self.source_file = self.codecache_cls.load(
                self.source_code, "so"
            )
        self.DLL.open()

    def cleanup_run_fn(self) -> None:
        # [FUNCTION DOC] Closes and drops the library and workspace.
        if self.DLL is not None:
            self.DLL.close()
            self.DLL = None
        self.workspace = None

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with kernel name, source file and hash key.
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"

    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Pickle support: returns a copy of the state with the library and
        # workspace cleared and the workspace-size-updated flag reset.
        state = self.__dict__.copy()
        state["DLL"] = None
        state["workspace"] = None
        state["_workspace_size_updated"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # [FUNCTION DOC] Pickle support: restores the instance dict from state.
        self.__dict__.update(state)


class CppBenchmarkRequest(CPUDeviceBenchmarkMixin, BenchmarkRequest):
    # [CLASS DOC] Picklable request to benchmark a CPU C++ kernel compiled from source.
    # State: source_code, hash_key of the source and the loaded library (DLL). Extra args
    # must be ctypes unsigned 64-bit integers.
    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put Tensors in here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None:
        # [FUNCTION DOC] Stores parent fields, the source, its hash, and an empty library slot.
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.hash_key = get_hash(source_code)
        self.DLL: CDLL | ModuleType | None = None

    def precompile(self):
        # [FUNCTION DOC] Compiles and loads the source into the C++ code cache and logs it.
        # Prepopulate CppCodeCache
        # may happen in separate Threadpool
        autotuning_log.debug("Precompiling %s", self)
        CppCodeCache.load(self.source_code, device_type="cpu")
        autotuning_log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # [FUNCTION DOC] Loads the library, takes data pointers of inputs and output, verifies
        # every extra arg is a ctypes unsigned 64-bit integer (raises otherwise), sets all
        # argument types to that integer type, and returns a partial of the kernel entry bound
        # to the pointers and extra args.
        # TODO(jgong5): use CppPythonBindingsCodeCache for better binding perf
        self.DLL = CppCodeCache.load(self.source_code, device_type="cpu")
        args = [tensor.data_ptr() for tensor in list(input_tensors) + [out]]
        autotuning_log.debug(
            "make_run_fn: self.kernel_name=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.DLL,
            args,
            self.extra_args,
        )
        run_method = getattr(self.DLL, self.kernel_name)
        # Assume only size with type ctypes.c_ulonglong in extra_args
        if not all(isinstance(arg, ctypes.c_ulonglong) for arg in self.extra_args):
            raise AssertionError(
                f"Expected all extra_args to be ctypes.c_ulonglong, got types: "
                f"{[type(arg) for arg in self.extra_args if not isinstance(arg, ctypes.c_ulonglong)]}"
            )
        run_method.argtypes = [ctypes.c_ulonglong] * (
            len(args) + len(list(self.extra_args))
        )

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
        )

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with the kernel name.
        return f"{self.kernel_name=}"


class CuteDSLBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    # [CLASS DOC] Picklable request to benchmark a CuteDSL kernel. State: finalized module
    # source_code and the cache key and path of the module written from it.
    """Benchmark request for CuteDSL (CUTLASS Python DSL) kernels."""

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: tuple[Any, ...],
        source_code: PartialRender,
    ) -> None:
        # [FUNCTION DOC] Stores parent fields, finalizes the partial render into full source
        # text and writes it as a cached Python module, recording its key and path.
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)

        # Template-specific precompile hooks reuse the finalized module source in subprocesses.
        self.source_code = source_code.finalize_all()
        self.module_cache_key, self.module_path = PyCodeCache.write(self.source_code)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # [FUNCTION DOC] Loads the module and finds the main kernel function named from the
        # kernel name and the main suffix (raises RuntimeError listing callables if missing),
        # and returns a callable running it with inputs, output and extra args on the raw cuda
        # stream of the output device.
        """
        Create a function to run the CuteDSL kernel with the given input and output tensors.
        Similar to TritonBenchmarkRequest.make_run_fn but for CuteDSL kernels.
        """
        mod = PyCodeCache.load_by_key_path(
            self.module_cache_key,
            self.module_path,
            set_sys_modules=False,
        )

        # Logic replicated async_compile
        from .codegen.cutedsl.cutedsl_kernel import MAIN_SUFFIX

        main_func_name = f"{self.kernel_name}_{MAIN_SUFFIX}"

        if not hasattr(mod, main_func_name):
            available = [name for name in dir(mod) if callable(getattr(mod, name))]
            raise RuntimeError(
                f"Could not find CuteDSL main kernel function '{main_func_name}'. Available callables: {available}"
            )

        kernel_func = getattr(mod, main_func_name)

        def run_kernel():
            # [FUNCTION DOC] HELPER: Runs the kernel function on the cuda stream of out.
            device_interface = get_interface_for_device("cuda")
            stream = device_interface.get_raw_stream(out.device.index)
            return kernel_func(*input_tensors, out, *self.extra_args, stream=stream)

        return run_kernel


@functools.cache
def get_tuning_process_pool() -> TuningProcessPool:
    # [FUNCTION DOC] Creates (once, cached) the global TuningProcessPool and registers its
    # shutdown to run at interpreter exit; returns it. Spawns subprocesses on first call.
    pool = TuningProcessPool()
    atexit.register(pool.shutdown)
    return pool


def benchmark_in_sub_process(
    choices: list[TritonTemplateCaller],
) -> dict[TritonTemplateCaller, float]:
    # [FUNCTION DOC] Benchmarks the choices through the global tuning process pool and returns
    # a mapping of choice to latency (infinity for failures).
    """
    Do benchmarking in a subprocess and return the perf number (latency).
    """
    return get_tuning_process_pool().benchmark(choices)


class AutotuneProcessPool:
    # [CLASS DOC] Singleton manager of a one-worker spawn-context process pool used for
    # pipelined autotuning. State: _pool (lazily spawning executor, or none after shutdown),
    # _warmup_future and start time, _timer for inactivity shutdown, class-level _instance,
    # _lock and the _shutdown_for_inactivity flag that blocks re-creation after idle shutdown.
    """
    Singleton pool manager for running autotuning (precompilation + benchmarking)
    in a separate process.
    """

    _instance: AutotuneProcessPool | None = None
    _lock: threading.Lock = threading.Lock()
    _shutdown_for_inactivity: bool = False

    def __init__(self):
        # [FUNCTION DOC] Creates the process pool and the inactivity timer (if enabled) and
        # initializes warmup bookkeeping.
        self._pool: ProcessPoolExecutor | None = self._init_pool()
        self._warmup_future: Future[Any] | None = None
        self._warmup_start_time: float | None = None
        self._timer: Timer | None = self._init_timer()

    @classmethod
    def get_instance(cls):
        # [FUNCTION DOC] Returns the singleton, creating it with double-checked locking.
        """Get or create the singleton pool instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # num_workers=1 to avoid GPU contention during benchmarking
                    cls._instance = cls()
        return cls._instance

    @property
    def pool(self):
        # [FUNCTION DOC] Property returning the executor, recreating pool and timer if they were
        # shut down. Raises if pipelined max autotune GEMM is not enabled in config.
        """Get the process pool."""
        if not config.pipeline_max_autotune_gemm:
            raise AssertionError(
                "To use AutotuneProcessPool, pipeline_max_autotune_gemm must be enabled"
            )
        if self._pool is None:
            self._pool = self._init_pool()
            self._timer = self._init_timer()
        return self._pool

    def _init_timer(self) -> Timer | None:
        # [FUNCTION DOC] HELPER: Returns an inactivity timer that calls the timeout handler, or
        # none when the timeout is zero or negative.
        if AUTOTUNE_POOL_INACTIVITY_TIMEOUT > 0:
            return Timer(AUTOTUNE_POOL_INACTIVITY_TIMEOUT, self._on_inactivity_timeout)
        return None

    def _record_activity(self) -> None:
        # [FUNCTION DOC] HELPER: Resets the inactivity timer if one exists.
        if self._timer is not None:
            self._timer.record_call()

    def _on_inactivity_timeout(self) -> None:
        # [FUNCTION DOC] HELPER: Under the lock, shuts down the pool without waiting, clears
        # pool and timer, and sets the class flag so pipelined autotuning is not re-enabled.
        autotuning_log.info(
            "AutotuneProcessPool shutting down due to inactivity (timeout=%ds)",
            AUTOTUNE_POOL_INACTIVITY_TIMEOUT,
        )

        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = None
            self._timer = None

            # Mark that the pool was shut down for inactivity.
            # This prevents the pool from being recreated on recompiles
            # which likely do not require large amounts of autotuning.
            AutotuneProcessPool._shutdown_for_inactivity = True

    def _init_pool(self):
        # [FUNCTION DOC] HELPER: Creates a one-worker process pool using the spawn context
        # (workers start lazily), registers instance shutdown at exit, logs, and returns it.
        """
        Get or create the process pool.

        Uses ProcessPoolExecutor with 'spawn' context for CUDA safety.
        ProcessPoolExecutor is lazily initialized - workers are not spawned
        until the first submit() call, making this property non-blocking.
        """
        # Use 'spawn' context to avoid CUDA fork issues
        # Workers are spawned lazily on first submit(), not here
        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
        )
        atexit.register(self._shutdown)
        autotuning_log.info("AutotuneProcessPool created (workers spawn lazily)")

        return pool

    def warm_up(self) -> Future[Any]:
        # [FUNCTION DOC] Once (double-checked under the lock), submits the subprocess init job
        # with the current fp32 matmul precision, attaches a completion callback, and returns the
        # warmup future (the same future on later calls).
        """
        Submit a warmup job to eagerly spawn workers and initialize CUDA.

        This is optional - call it early to hide spawn latency.
        Returns the warmup future which can be ignored or awaited.
        """
        if self._warmup_future is None:
            with self._lock:
                if self._warmup_future is None:
                    self._warmup_start_time = time.perf_counter()
                    self._warmup_future = self.submit(
                        _init_autotune_subprocess,
                        fp32_precision=torch.backends.cuda.matmul.fp32_precision,
                    )
                    self._warmup_future.add_done_callback(self._on_warmup_complete)
                    autotuning_log.info("Warmup job submitted")
        # pyrefly: ignore[bad-return]
        return self._warmup_future

    def _on_warmup_complete(self, future: Future[Any]) -> None:
        # [FUNCTION DOC] HELPER: Callback that logs the warmup duration and result and records
        # activity; on failure logs an error and re-raises the exception.
        """Callback invoked when the warmup job completes."""
        warmup_elapsed_time = None
        if self._warmup_start_time is not None:
            warmup_elapsed_time = time.perf_counter() - self._warmup_start_time

        try:
            result = future.result()
            autotuning_log.info(
                "AutotuneProcessPool warmup completed successfully in %.4f seconds: %s",
                warmup_elapsed_time,
                result,
            )
            self._record_activity()
        except Exception as e:
            autotuning_log.error(
                "AutotuneProcessPool warmup failed after %.4f seconds",
                warmup_elapsed_time,
            )
            raise e

    def submit(self, fn, *args, **kwargs) -> Future[Any]:
        # [FUNCTION DOC] Submits fn with arguments to the pool along with the current cache
        # environment (applied in the worker), records activity when each job finishes if a
        # timer exists, and returns the future.
        """Submit a job to the pool and return a Future."""
        future = self.pool.submit(
            _run_with_subprocess_env,
            fn,
            _cache_env_for_subprocess(),
            *args,
            **kwargs,
        )
        if self._timer is not None:
            future.add_done_callback(lambda _: self._record_activity())
        return future

    def _shutdown(self):
        # [FUNCTION DOC] HELPER: Stops the timer and shuts down the pool without waiting.
        """Shutdown the pool on exit."""
        if self._timer is not None:
            self._timer.quit()
            self._timer = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    @classmethod
    def shutdown_instance(cls):
        # [FUNCTION DOC] Shuts down and clears the singleton, if any, under the lock.
        """Explicitly shutdown the singleton instance."""
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance._shutdown()
                    cls._instance = None


def use_pipelined_autotuning() -> bool:
    # [FUNCTION DOC] True when pipelined max autotune GEMM is configured and the pool has not
    # been shut down for inactivity.
    return (
        config.pipeline_max_autotune_gemm
        and not AutotuneProcessPool._shutdown_for_inactivity
    )


def _init_autotune_subprocess(fp32_precision: str) -> bool:
    # [FUNCTION DOC] HELPER: Runs in the worker: creates a small CUDA tensor if CUDA is
    # available to initialize the context, sets the fp32 matmul precision, returns True.
    """
    Warmup function run in the autotune subprocess.
    """
    import torch

    # Initialize dummy tensor for CUDA context
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    torch.backends.cuda.matmul.fp32_precision = fp32_precision

    return True


def run_autotune_in_subprocess(
    benchmark_request: BenchmarkRequest,
) -> float:
    # [FUNCTION DOC] Runs in the worker: calls the request benchmark and returns the time;
    # on any exception logs a warning with traceback and returns infinity.
    # KNOWN ISSUE: docstring documents an argument named picklable_choices that does not exist.
    """
    Run autotuning benchmarks in a subprocess.

    This function is submitted to AutotuneProcessPool and runs in isolation
    to prevent GPU contention with the main compilation process.

    Args:
        picklable_choices: List of picklable choice information

    Returns:
        timing
    """

    try:
        # Run the benchmark directly - bmreq is already a BenchmarkRequest
        timing = benchmark_request.benchmark()

        return timing

    except Exception:
        autotuning_log.warning(
            "Failed to benchmark choice %s",
            benchmark_request,
            exc_info=True,
        )
        # Use infinity for failed benchmarks so they're not selected
        return float("inf")


class PrecompileThreadPool:
    # [CLASS DOC] Singleton wrapper around a thread pool for background precompilation.
    # State: _executor, plus class-level _instance and _lock.
    """
    Thread pool for running precompilation asynchronously.

    This allows the main compilation process to continue while
    precompilation happens in background threads.
    """

    _instance: PrecompileThreadPool | None = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4):
        # [FUNCTION DOC] Creates the thread pool with the given worker count.
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @classmethod
    def get_instance(cls) -> PrecompileThreadPool:
        # [FUNCTION DOC] Returns the singleton, creating it sized by the configured number of
        # workers with double-checked locking.
        from torch._inductor.select_algorithm import get_num_workers

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(get_num_workers())
        return cls._instance

    def submit(self, fn, *args, **kwargs):
        # [FUNCTION DOC] Submits fn to the pool, running it inside a copy of the caller context
        # so workers see the same config settings. Returns the future.
        ctx = contextvars.copy_context()
        # Need to copy context so workers have access to the correct config settings
        fn = functools.partial(ctx.run, fn)
        return self._executor.submit(fn, *args, **kwargs)

    def _shutdown(self, wait: bool = False):
        # [FUNCTION DOC] HELPER: Shuts down the executor, optionally waiting.
        return self._executor.shutdown(wait=wait)

    @classmethod
    def shutdown_instance(cls) -> None:
        # [FUNCTION DOC] Shuts down the singleton without waiting and clears it, if present.
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance._shutdown(wait=False)
                    cls._instance = None


class AsyncAutotuner:
    # [CLASS DOC] Class-level (stateless instances) coordinator for asynchronous autotuning:
    # start submits each choice benchmark request to the AutotuneProcessPool and get_results
    # blocks for them. Shared state: choice_hash_to_future maps a choice-plus-inputs hash to
    # its future, deduplicating submissions across calls.
    """
    Handles asynchronous autotuning of kernel choices in a separate process.

    This class manages the lifecycle of autotuning:
    1. Accepts precompiled choices from the main process
    2. Submits benchmarking work to AutotuneProcessPool
    3. Returns results via a Future

    Usage:
        autotuner = AsyncAutotuner(choices)
        autotuner.start()  # Kicks off async benchmarking
        timings = autotuner.get_results()  # Blocks until complete
    """

    choice_hash_to_future = {}

    @staticmethod
    def get_choice_hash(choice: ChoiceCaller, inputs_key: str) -> str:
        # [FUNCTION DOC] Returns the choice hash key concatenated with the inputs key.
        return choice.hash_key() + inputs_key

    @classmethod
    def start(cls, choices: list[ChoiceCaller], inputs_key: str):
        # [FUNCTION DOC] For each choice not already submitted, submits its benchmark request to
        # the process pool and records the future; returns immediately. Raises an error if a
        # choice lacks a benchmark request.
        """
        Start asynchronous autotuning in a subprocess.

        This method:
        1. Extracts picklable benchmark requests from choices
        2. Submits benchmarking work to AutotuneProcessPool
        3. Returns immediately (non-blocking)
        """

        for choice in choices:
            choice_hash = AsyncAutotuner.get_choice_hash(choice, inputs_key)

            if choice_hash in AsyncAutotuner.choice_hash_to_future:
                continue

            if not getattr(choice, "bmreq", None) is not None:
                raise AssertionError("bmreq is None for choice")

            autotune_future = AutotuneProcessPool.get_instance().submit(
                run_autotune_in_subprocess,
                choice.bmreq,
            )

            AsyncAutotuner.choice_hash_to_future[choice_hash] = autotune_future

    @classmethod
    def get_results(
        cls, choices: list[ChoiceCaller], inputs_key: str
    ) -> dict[ChoiceCaller, float]:
        # [FUNCTION DOC] Blocks on each choice future and returns a mapping of choice to timing.
        # Raises KeyError if a choice was never started; future exceptions propagate.
        # KNOWN ISSUE: docstring documents a timeout argument that does not exist.
        """
        Get autotuning results, blocking until complete.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            Dict mapping ChoiceCaller to benchmark timing
        """

        timings = {}
        for choice in choices:
            choice_hash = AsyncAutotuner.get_choice_hash(choice, inputs_key)
            timings[choice] = AsyncAutotuner.choice_hash_to_future[choice_hash].result()
        return timings
