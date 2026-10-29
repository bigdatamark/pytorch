# mypy: allow-untyped-defs
# [MODULE DOC] Launches several worker processes that each run a function with their index and arguments,
# with error propagation to the parent. Provides spawn, start_processes, ProcessContext (join/monitor) and
# exception types for raised errors and abnormal exits. Optional parallel start for forkserver is
# controlled by the TORCH_MP_PARALLEL_START environment variable. Imports a prctl helper from the package.
import logging
import multiprocessing
import multiprocessing.connection
import os
import pickle
import signal
import sys
import tempfile
import time
import warnings
from concurrent.futures import as_completed, ThreadPoolExecutor

from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]


ENV_VAR_PARALLEL_START = "TORCH_MP_PARALLEL_START"

log = logging.getLogger(__name__)

__all__ = [
    "ProcessContext",
    "ProcessException",
    "ProcessExitedException",
    "ProcessRaisedException",
    "spawn",
    "SpawnContext",
    "start_processes",
]


class ProcessException(Exception):
    # [CLASS DOC] Base error for a failed child process. Holds msg, error_index (child rank) and error_pid.
    # Picklable through __reduce__ so it can cross process boundaries.
    __slots__ = ["error_index", "error_pid"]

    def __init__(self, msg: str, error_index: int, error_pid: int):
        # [FUNCTION DOC] Stores the message, index of the failed process and its pid.
        super().__init__(msg)
        self.msg = msg
        self.error_index = error_index
        self.error_pid = error_pid

    def __reduce__(self):
        # [FUNCTION DOC] Pickle support: rebuild with message, error index and pid.
        return type(self), (self.msg, self.error_index, self.error_pid)


class ProcessRaisedException(ProcessException):
    # [CLASS DOC] A child failed because its function raised; the message holds the child's traceback text.
    """Exception raised when a process failed due to an exception raised by the code."""


class ProcessExitedException(ProcessException):
    # [CLASS DOC] A child died from a signal or exited with a nonzero code without a recorded traceback.
    # Extra state: exit_code and signal_name (None unless killed by a signal).
    # KNOWN ISSUE: slots lists only exit_code although signal_name is also set; harmless since exceptions have a dict.
    """Exception raised when a process failed due to signal or exited with a specific code."""

    __slots__ = ["exit_code"]

    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
        exit_code: int,
        signal_name: str | None = None,
    ):
        # [FUNCTION DOC] Stores base error fields plus the exit code and optional signal name.
        super().__init__(msg, error_index, error_pid)
        self.exit_code = exit_code
        self.signal_name = signal_name

    def __reduce__(self):
        # [FUNCTION DOC] Pickle support: rebuild with message, index, pid, exit code and signal name.
        return (
            type(self),
            (
                self.msg,
                self.error_index,
                self.error_pid,
                self.exit_code,
                self.signal_name,
            ),
        )


def _wrap(fn, i, args, error_file):
    # [FUNCTION DOC] HELPER: Child process entry point. Requests a parent-death SIGINT where supported, runs the user
    # function with the process index and args. A keyboard interrupt is swallowed; any other exception has its
    # formatted traceback pickled to error_file and the process exits with status 1.
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        fn(i, *args)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback

        with open(error_file, "wb") as fh:
            pickle.dump(traceback.format_exc(), fh)
        sys.exit(1)


class ProcessContext:
    # [CLASS DOC] Handle to a set of started worker processes. State: processes, error_files (one per process,
    # written only on a Python exception) and sentinels (mapping from process sentinel to index, shrinking as
    # processes are joined). Methods: pids and join.
    def __init__(self, processes, error_files):
        # [FUNCTION DOC] Stores the processes and error file names and builds the sentinel-to-index mapping.
        self.error_files = error_files
        self.processes = processes
        self.sentinels = {
            process.sentinel: index for index, process in enumerate(processes)
        }

    def pids(self):
        # [FUNCTION DOC] Returns the list of process ids as integers.
        return [int(process.pid) for process in self.processes]

    def _join_procs_with_timeout(self, timeout: float):
        # [FUNCTION DOC] HELPER: Joins every process, sharing one total timeout budget across them.
        """Attempt to join all processes with a shared timeout."""
        end = time.monotonic() + timeout
        for process in self.processes:
            # pyrefly: ignore [no-matching-overload]
            time_to_wait = max(0, end - time.monotonic())
            process.join(time_to_wait)

    def join(self, timeout: float | None = None, grace_period: float | None = None):
        # [FUNCTION DOC] Waits (up to timeout) for any process to finish. Returns True when all have been joined, False if
        # some remain, immediately True if nothing is left. If a joined process failed, optionally waits a grace
        # period, sends terminate to live processes, waits again, then kills stragglers, and raises:
        # ProcessExitedException if no traceback file exists (signal name or exit code in message), otherwise
        # ProcessRaisedException containing the child's traceback. Side effects: logs warnings, kills processes.
        r"""Join one or more processes within spawn context.

        Attempt to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes (optionally with a grace period)
        and raises an exception with the cause of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Args:
            timeout (float): Wait this long (in seconds) before giving up on waiting.
            grace_period (float): When any processes fail, wait this long (in seconds)
                for others to shutdown gracefully before terminating them. If they
                still don't exit, wait another grace period before killing them.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True

        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break

        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
        # An error occurred. Clean-up all processes before returning.
        # First, allow a grace period for processes to shutdown themselves.
        if grace_period is not None:
            self._join_procs_with_timeout(grace_period)
        # Then, terminate processes that are still alive. Try SIGTERM first.
        for process in self.processes:
            if process.is_alive():
                log.warning("Terminating process %s via signal SIGTERM", process.pid)
                process.terminate()

        # Try SIGKILL if the process isn't going down after another grace_period.
        # The reason is related to python signal handling is limited
        # to main thread and if that is in c/c++ land and stuck it won't
        # to handle it. We have seen processes getting stuck not handling
        # SIGTERM for the above reason.
        self._join_procs_with_timeout(30 if grace_period is None else grace_period)
        for process in self.processes:
            if process.is_alive():
                log.warning(
                    "Unable to shutdown process %s via SIGTERM , forcefully exiting via SIGKILL",
                    process.pid,
                )
                process.kill()
            process.join()

        # The file will only be created if the process crashed.
        failed_process = self.processes[error_index]
        if not os.access(self.error_files[error_index], os.R_OK):
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                try:
                    name = signal.Signals(-exitcode).name
                except ValueError:
                    name = f"<Unknown signal {-exitcode}>"
                raise ProcessExitedException(
                    f"process {error_index:d} terminated with signal {name}",
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name,
                )
            else:
                raise ProcessExitedException(
                    f"process {error_index:d} terminated with exit code {exitcode:d}",
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                )

        with open(self.error_files[error_index], "rb") as fh:
            original_trace = pickle.load(fh)
        msg = f"\n\n-- Process {error_index:d} terminated with the following error:\n"
        msg += original_trace
        raise ProcessRaisedException(msg, error_index, failed_process.pid)


class SpawnContext(ProcessContext):
    # [CLASS DOC] Deprecated alias subclass of ProcessContext kept for backward compatibility; warns on creation.
    def __init__(self, processes, error_files):
        # [FUNCTION DOC] Emits a warning that the class was renamed to ProcessContext, then initializes the base class.
        warnings.warn(
            "SpawnContext is renamed to ProcessContext since 1.4 release.", stacklevel=2
        )
        super().__init__(processes, error_files)


# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like IPython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def start_processes(
    fn,
    args=(),
    nprocs=1,
    join=True,
    daemon=False,
    start_method="spawn",
):
    # [FUNCTION DOC] Starts nprocs processes with the chosen start method, each running the wrapper that calls fn with
    # its index and args. Starts them in parallel threads only for forkserver when the environment variable
    # is set to 1. Each process gets a unique temporary error file name (created then removed).
    # If join is False returns the ProcessContext; otherwise blocks, joining until all succeed, and
    # returns None (join raises on child failure). Side effects: spawns processes, creates temp names.
    # To speed up performance in certain cases (see https://github.com/pytorch/pytorch/issues/133010),
    # this func will start processes in parallel if start_method is 'forkserver'.
    # Please opt in to this perf optimization by setting env var (TORCH_MP_PARALLEL_START) to 1.
    # todo: investigate why spawn does not work with threadpool and raises SIGINT
    if (
        start_method == "forkserver"
        and os.environ.get(ENV_VAR_PARALLEL_START, "0") == "1"
    ):
        log.info("Starting processes in parallel.")
        start_parallel = True
    else:
        # Set env var TORCH_MP_PARALLEL_START to 0 to disable parallel start
        start_parallel = False

    mp = multiprocessing.get_context(start_method)
    error_files = [None] * nprocs
    processes = [None] * nprocs

    def start_process(i):
        # [FUNCTION DOC] HELPER: Creates the error file name, builds and starts one process (daemon flag as given).
        # Returns the index, the process and the error file name.
        # Each process is assigned a file to write tracebacks to.  We
        # use the file being non-empty to indicate an exception
        # occurred (vs an expected shutdown).  Note: this previously
        # used a multiprocessing.Queue but that can be prone to
        # deadlocks, so we went with a simpler solution for a one-shot
        # message between processes.
        tf = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix="pytorch-errorfile-", suffix=".pickle", delete=False
        )
        tf.close()
        os.unlink(tf.name)

        process = mp.Process(  # pyrefly: ignore  # missing-attribute
            target=_wrap,
            args=(fn, i, args, tf.name),
            daemon=daemon,
        )

        process.start()
        return i, process, tf.name

    if not start_parallel:
        for i in range(nprocs):
            idx, process, tf_name = start_process(i)
            error_files[idx] = tf_name
            processes[idx] = process
    else:
        with ThreadPoolExecutor(max_workers=nprocs) as executor:
            futures = [executor.submit(start_process, i) for i in range(nprocs)]
            for fut in as_completed(futures):
                idx, process, tf_name = fut.result()
                # idx and process rank needs to be the same.
                error_files[idx] = tf_name
                processes[idx] = process
    context = ProcessContext(processes, error_files)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"):
    # [FUNCTION DOC] Public wrapper that runs fn in nprocs spawned processes. Warns (FutureWarning) if start_method is not
    # spawn, but always uses spawn. Returns what start_processes returns: None when joined, else a ProcessContext.
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Args:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (str): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    if start_method != "spawn":
        msg = (
            f"This method only supports start_method=spawn (got: {start_method}).\n"
            "To use a different start_method use:\n\t\t"
            " torch.multiprocessing.start_processes(...)"
        )
        warnings.warn(msg, FutureWarning, stacklevel=2)
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
