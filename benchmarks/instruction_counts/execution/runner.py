"""Run benchmarks while handling parallelism, isolation, and fault tolerance."""

# mypy: ignore-errors
# [MODULE DOC] Parallel execution of benchmark work orders with core pinning and fault handling.
# Contains a failure exception, CorePool which hands out contiguous CPU core ranges, and Runner
# which launches jobs as cores free up, polls them, reports progress and an estimate of time left,
# and shuts children down on errors or interrupts. Import-time side effect: reads the CPU count of
# the machine into a module constant.

import math
import multiprocessing
import subprocess
import textwrap
import threading
import time

from worker.main import WorkerFailure, WorkerOutput

from execution.work import InProgress, PYTHON_CMD, SHELL, WorkOrder


CPU_COUNT: int = multiprocessing.cpu_count()


class WorkerFailed(Exception):
    # [CLASS DOC] Exception raised in the main process when a worker reports failure. Attributes:
    # the command line that was run and an optional formatted trace string from the worker.
    """Raised in the main process when a worker failure is detected."""

    def __init__(self, cmd: str, wrapped_trace: str | None = None) -> None:
        # [FUNCTION DOC] Stores the command and the optional worker trace; the exception message is
        # empty.
        self.cmd: str = cmd
        self.wrapped_trace: str | None = wrapped_trace
        super().__init__()


class CorePool:
    # [CLASS DOC] Allocator that gives each task a contiguous range of CPU cores (inclusive bounds)
    # so the machine is not overloaded and wall times are less noisy. Tracks which cores are free
    # and the live reservations, guarded by a lock so it can be used from several threads.
    """Allocator style helper class to assign individual tasks to a core range.

    Pinning tasks to separate cores (or core ranges if `num_threads` > 1)
    serves two purposes. First, it prevents the machine from being overloaded,
    which can result in OOMs or Callgrind crashes. Second, it helps reduce
    noise in the wall times, which are collected as a secondary metric. For
    multi-threaded workloads, adjacency is important. Often pairs of cores
    share silicon (e.g. cache), while far away cores may lie on separate NUMA
    nodes. For this reason, CorePool will only allocate contiguous core ranges.
    This falls short of full architecture awareness, and instead tries to find
    a balance between rigor and engineering complexity.
    """

    def __init__(self, min_core_id: int, max_core_id: int) -> None:
        # [FUNCTION DOC] Validates the core bounds (AssertionError if the minimum is negative, the
        # maximum is below the minimum or not below the machine CPU count), prints the pool range,
        # and marks all cores available.
        if min_core_id < 0:
            raise AssertionError(f"min_core_id must be >= 0, got {min_core_id}")
        if max_core_id < min_core_id:
            raise AssertionError(
                f"max_core_id ({max_core_id}) must be >= min_core_id ({min_core_id})"
            )
        if max_core_id >= CPU_COUNT:
            raise AssertionError(
                f"max_core_id ({max_core_id}) must be < CPU_COUNT ({CPU_COUNT})"
            )

        self._min_core_id: int = min_core_id
        self._max_core_id: int = max_core_id
        self._num_cores = max_core_id - min_core_id + 1
        print(f"Core pool created: cores {self._min_core_id}-{self._max_core_id}")

        self._available: list[bool] = [
            True for _ in range(min_core_id, min_core_id + self._num_cores)
        ]

        self._reservations: dict[str, tuple[int, ...]] = {}
        self._lock = threading.Lock()

    def reserve(self, n: int) -> str | None:
        # [FUNCTION DOC] First-fit reservation of n contiguous cores under the lock. On success
        # marks them used, remembers the reservation and returns a taskset style string (a single
        # core id or first-last range). Returns None if no free range is large enough.
        """Simple first-fit policy.

        If successful, return a string for `taskset`. Otherwise, return None.
        """
        with self._lock:
            for lower_index in range(self._num_cores - n + 1):
                indices = tuple(range(lower_index, lower_index + n))
                if all(self._available[i] for i in indices):
                    for i in indices:
                        self._available[i] = False

                    lower_core = indices[0] + self._min_core_id
                    upper_core = indices[-1] + self._min_core_id
                    key = f"{lower_core}-{upper_core}" if n > 1 else f"{lower_core}"
                    self._reservations[key] = indices
                    return key
        return None

    def release(self, key: str) -> None:
        # [FUNCTION DOC] Frees the cores of the reservation named by the key and forgets it. Raises
        # KeyError for an unknown key.
        with self._lock:
            for i in self._reservations[key]:
                self._available[i] = True
            self._reservations.pop(key)


class Runner:
    # [CLASS DOC] Runs a tuple of work orders concurrently. State: the order queue not yet started,
    # the active jobs, results per order, durations (for ETA), the order being handled (for error
    # messages) and a polling cadence in seconds. Jobs are pinned to cores from a CorePool.
    def __init__(
        self,
        work_items: tuple[WorkOrder, ...],
        core_pool: CorePool | None = None,
        cadence: float = 1.0,
    ) -> None:
        # [FUNCTION DOC] Stores the orders, a core pool (default: cores zero up to the CPU count
        # minus four, which fails the pool check on small machines), and the polling cadence.
        # Initializes queue, active list, results and timing state. Raises ValueError when the same
        # order object appears twice.
        self._work_items: tuple[WorkOrder, ...] = work_items
        self._core_pool: CorePool = core_pool or CorePool(0, CPU_COUNT - 4)
        self._cadence: float = cadence

        # Working state.
        self._work_queue: list[WorkOrder] = list(work_items)
        self._active_jobs: list[InProgress] = []
        self._results: dict[WorkOrder, WorkerOutput] = {}

        # Debug information for ETA and error messages.
        self._start_time: float = -1
        self._durations: dict[WorkOrder, float] = {}
        self._currently_processed: WorkOrder | None = None

        if len(work_items) != len(set(work_items)):
            raise ValueError("Duplicate work items.")

    def run(self) -> dict[WorkOrder, WorkerOutput]:
        # [FUNCTION DOC] Runs everything and returns the mapping from order to worker output. On a
        # keyboard interrupt, job timeout, worker failure or any other exception it shuts the
        # children down (printing diagnostics, including the failing command and trace for worker
        # failures) and re-raises the original exception.
        try:
            return self._run()

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt (ctrl-c) detected. Shutting down children.")
            self._force_shutdown(verbose=False)
            raise

        except subprocess.TimeoutExpired:
            print("\n\nJob timed out. Shutting down children.")
            self._force_shutdown(verbose=True)
            raise

        except WorkerFailed as e:
            print("Shutting down all outstanding jobs before re-raising.")
            self._force_shutdown(verbose=True)
            print(f"Cmd: {e.cmd}")
            if e.wrapped_trace:
                print(e.wrapped_trace)
            else:
                print("Unknown failure. (Worker did not report exception contents.)")
            raise

        except BaseException:
            print("\n\nUnknown exception. Shutting down jobs before re-raising.")
            self._force_shutdown(verbose=True)
            raise

    def _run(self) -> dict[WorkOrder, WorkerOutput]:
        # [FUNCTION DOC] HELPER: Main loop. Checks the canary import, then until the queue and
        # active list are empty it updates jobs, starts new ones, prints progress and sleeps for
        # the rest of the cadence interval. Prints the total time and returns a copy of the
        # results.
        self._start_time = time.time()
        self._canary_import()
        while self._work_queue or self._active_jobs:
            t0 = time.time()
            self._update_active_jobs()
            self._enqueue_new_jobs()
            self._print_progress()
            time.sleep(max(self._cadence - (time.time() - t0), 0.0))
        print(f"\nTotal time: {time.time() - self._start_time:.0f} seconds")
        return self._results.copy()

    def _update_active_jobs(self) -> None:
        # [FUNCTION DOC] HELPER: Polls all active jobs. Unfinished ones are kept. A finished job
        # with a successful output has its result and duration stored and its cores released
        # (AssertionError if it had no cpu list). A finished job with a failure raises WorkerFailed
        # carrying the command and trace. A timeout exception from a job propagates.
        active_jobs: list[InProgress] = []
        for job in self._active_jobs:
            self._currently_processed = job.work_order
            if not job.check_finished():
                active_jobs.append(job)
                continue

            result: WorkerOutput | WorkerFailure = job.result
            if isinstance(result, WorkerOutput):
                self._results[job.work_order] = result
                if job.cpu_list is None:
                    raise AssertionError("job.cpu_list must not be None")
                self._core_pool.release(job.cpu_list)
                self._durations[job.work_order] = job.duration

            else:
                if not isinstance(result, WorkerFailure):
                    raise AssertionError(f"expected WorkerFailure, got {type(result)}")
                raise WorkerFailed(cmd=job.proc.cmd, wrapped_trace=result.failure_trace)
        self._currently_processed = None
        self._active_jobs.clear()
        self._active_jobs.extend(active_jobs)

    def _enqueue_new_jobs(self) -> None:
        # [FUNCTION DOC] HELPER: Walks the queue and tries to reserve as many cores as the order
        # has threads; orders that get cores become active jobs (with a half second pause between
        # launches to reduce contention), the rest stay queued.
        work_queue: list[WorkOrder] = []
        for i, work_order in enumerate(self._work_queue):
            self._currently_processed = work_order
            cpu_list = self._core_pool.reserve(work_order.timer_args.num_threads)

            if cpu_list is None:
                work_queue.append(work_order)
            else:
                self._active_jobs.append(InProgress(work_order, cpu_list))

                # Stagger creation. This helps with contention.
                time.sleep(0.5)
        self._currently_processed = None
        self._work_queue.clear()
        self._work_queue.extend(work_queue)

    def _print_progress(self) -> None:
        # [FUNCTION DOC] HELPER: Prints a carriage-return progress line with finished over total,
        # elapsed seconds and an ETA. The ETA is unknown until five results exist, then rounds
        # remaining work (counted in rounds over the pool core count) times the mean job duration,
        # in whole minutes.
        fraction = f"{len(self._results)} / {len(self._work_items)}"
        elapsed = f"{time.time() - self._start_time:.0f} seconds"
        if len(self._results) < 5:
            eta = "Unknown"
        else:
            remaining = len(self._work_items) - len(self._results)
            iters_remaining = math.ceil(remaining / self._core_pool._num_cores)
            mean_time = sum(self._durations.values()) / len(self._durations)
            eta_minutes = math.ceil(iters_remaining * mean_time / 60)
            eta = f"~{eta_minutes:.0f} minute{'s' if eta_minutes > 1 else ''}"
        print(f"\r{fraction} ({elapsed}), ETA: {eta}", end="")

    def _force_shutdown(self, verbose: bool = False) -> None:
        # [FUNCTION DOC] HELPER: Stops all active jobs. Sends SIGINT to each, optionally prints the
        # label, auto labels and source command of the job being processed, waits half a second,
        # and if some processes remain polls up to five times every two seconds, returning early
        # when all exit. Then terminates the stragglers.
        """Try to interrupt jobs, and kill if need be.
        We would prefer to softly terminate jobs so that they have a chance to
        clean up before shutting down.
        """
        for job in self._active_jobs:
            job.proc.interrupt()

        if verbose and self._currently_processed is not None:
            print(
                textwrap.dedent(
                    f"""
                Failed when processing the following Job:
                  Label:      {self._currently_processed.label}
                  AutoLabels: {self._currently_processed.autolabels}
                  Source cmd: {self._currently_processed.source_cmd}
            """
                ).strip()
                + "\n"
            )

        if self._active_jobs:
            time.sleep(0.5)

        remaining_jobs = [j for j in self._active_jobs if j.proc.poll() is None]
        if remaining_jobs:
            print(
                f"SIGINT sent to {len(self._active_jobs)} jobs, "
                f"{len(remaining_jobs)} have not yet exited.\n"
                "Entering short cleanup loop, after which stragglers will "
                "be forcibly terminated."
            )

            for _ in range(5):
                time.sleep(2.0)
                remaining_jobs = [j for j in remaining_jobs if j.proc.poll() is None]
                if remaining_jobs:
                    print(f"{len(remaining_jobs)} still remain.")
                else:
                    print("All remaining jobs have gracefully terminated.")
                    return

            print(f"{len(remaining_jobs)} jobs refused to exit. Forcibly terminating.")
            for j in remaining_jobs:
                j.proc.terminate()

    def _canary_import(self) -> None:
        # [FUNCTION DOC] HELPER: Runs a command importing torch once per distinct source command
        # (or once without one) before launching workers. Raises ImportError with the command and
        # output if the import exits with an error.
        """Make sure we can import torch before launching a slew of workers."""
        source_cmds: set[str] = set()
        for w in self._work_items:
            if w.source_cmd is not None:
                source_cmds.add(f"{w.source_cmd} && ")

        for source_cmd in source_cmds or {""}:
            cmd = f'{source_cmd}{PYTHON_CMD} -c "import torch"'
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                executable=SHELL,
            )

            if proc.returncode:
                raise ImportError(
                    f"Failed to import torch in subprocess: {cmd}\n{proc.stdout}"
                )
