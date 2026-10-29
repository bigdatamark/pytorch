# mypy: disallow-untyped-defs
# [MODULE DOC]
# Meta-only CLI-based function profiler using the strobeclient tool: StrobelightCLIFunctionProfiler,
# its error type and a strobelight decorator. Import-time side effects: configures the module logger
# (console handler, INFO level, no propagation).

import functools
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from threading import Lock
from timeit import default_timer as timer
from typing import Any, TypeVar
from typing_extensions import ParamSpec


logger = logging.getLogger("strobelight_function_profiler")

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

_P = ParamSpec("_P")
_R = TypeVar("_R")


class StrobelightCLIProfilerError(Exception):
    # [CLASS DOC] Exception raised for any failure in starting, stopping or reading a strobelight profiling run.
    """
    Raised when an error happens during strobelight profiling
    """


def _pid_namespace_link(pid: int | None = None) -> str:
    # [FUNCTION DOC] HELPER: Returns the target of the proc pid namespace symlink (for example a pid bracket id string)
    # for pid, defaulting to the current process. Raises OS errors if the link cannot be read.
    """Returns the link to the process's namespace, example: pid:[4026531836]"""
    PID_NAMESPACE_PATH = "/proc/{}/ns/pid"
    pid = pid or os.getpid()
    return os.readlink(PID_NAMESPACE_PATH.format(pid))


def _pid_namespace(pid: int | None = None) -> int:
    # [FUNCTION DOC] HELPER: Returns the integer pid namespace id parsed from the namespace link of pid (default current).
    """Returns the process's namespace id"""
    pid = pid or os.getpid()
    link = _pid_namespace_link(pid)
    return int(link[link.find("[") + 1 : -1])


def _command_to_string(command: Sequence[str]) -> str:
    # [FUNCTION DOC] HELPER: Joins a command argument sequence into one space-separated string for logging.
    return " ".join(command)


class StrobelightCLIFunctionProfiler:
    # [CLASS DOC] Meta-internal profiler that runs the strobeclient CLI around a Python function call and
    # logs a strobelight result. State: configuration (stop_at_error, duration, sample interval, user name,
    # timeouts, sample tags), current_run_id and profile_result of the latest run. A class-level lock allows
    # only one profile at a time. Errors are swallowed (work function still runs) unless stop_at_error.
    # KNOWN ISSUE: recorded_env_variables, stack_max_len and async_stack_max_len are accepted but unused;
    # the two timeout settings are stored but never applied.
    """
    Note: this is a Meta only tool.

    StrobelightCLIFunctionProfiler can be used to profile a python function and
    generate a strobelight link with the results. It works on meta servers but
    does not require an fbcode target.
    When stop_at_error is false(default), error during profiling does not prevent
    the work function from running.

    Check function_profiler_example.py for an example.
    """

    # This lock is used to make sure only one thread is running the profiler at any point.
    _lock = Lock()

    def __init__(
        self,
        *,
        stop_at_error: bool = False,
        max_profile_duration_sec: int = 60 * 10,
        sample_each: float = 1e7,  # sample each sample_each cycles.
        run_user_name: str = "pytorch-strobelight-ondemand",
        timeout_wait_for_running_sec: int = 60,
        timeout_wait_for_finished_sec: int = 60,
        recorded_env_variables: list[str] | None = None,
        sample_tags: list[str] | None = None,
        stack_max_len: int = 127,
        async_stack_max_len: int = 127,
    ):
        # [FUNCTION DOC] Stores the profiling configuration and initializes current_run_id and profile_result to None.
        self.stop_at_error = stop_at_error
        self.max_profile_duration_sec = max_profile_duration_sec
        self.sample_each = sample_each
        self.run_user_name = run_user_name
        self.timeout_wait_for_running_sec = timeout_wait_for_running_sec
        self.timeout_wait_for_finished_sec = timeout_wait_for_finished_sec
        # Results of the most recent run.
        # Tracks the strobelight run id of the most recent run
        self.current_run_id: int | None = None
        self.profile_result: list[str] | None = None
        self.sample_tags = sample_tags

    def _run_async(self) -> None:
        # [FUNCTION DOC] HELPER: Starts an asynchronous pyperf cycles profile of the current process via the strobeclient
        # CLI (with sample interval, duration and optional sample tags) and stores the run id parsed from
        # its stderr in current_run_id. Raises StrobelightCLIProfilerError on nonzero exit or unparseable output.
        processId = os.getpid()
        namespace = _pid_namespace(processId)
        command = [
            "strobeclient",
            "run",
            "--profiler",
            "pyperf",
            "--event",
            "cycles",
            "--async",
            "--sample-interval",
            f"{int(self.sample_each)}",
            "--duration-ms",
            f"{int(self.max_profile_duration_sec * 1000)}",
            "--pid",
            f"{namespace}:{processId}",
        ]

        if self.sample_tags:
            command.append("--sample-tags")
            command.append(",".join(self.sample_tags))

        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in run_async:{output}"
            )

        if match := re.search(r"INFO Run Id: (-?\d+)", output):
            self.current_run_id = int(match.group(1))
            return

        raise StrobelightCLIProfilerError(
            f"failed to start strobelight profiling, unexpected result {output}"
        )

    def _wait_for_running(self, counter: int = 0) -> None:
        # [FUNCTION DOC] HELPER: Polls the run status via the CLI until RUNNING, sleeping 10 seconds and recursing while
        # PREPARING. Raises StrobelightCLIProfilerError on CLI failure, unexpected status or output, or after
        # more than 20 retries (counter is the recursion depth).
        if counter > 20:
            raise StrobelightCLIProfilerError(
                "wait_for_running called more than 20 times"
            )

        command = ["strobeclient", "getRunStatus", "--run-id", f"{self.current_run_id}"]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in wait_for_running:{output}"
            )

        if match := re.search("Profile run status: (.*)", output):
            current_status = match.group(1)
            if current_status == "RUNNING":
                return
            elif current_status == "PREPARING":
                time.sleep(10)
                self._wait_for_running(counter + 1)
                return
            else:
                raise StrobelightCLIProfilerError(f"unexpected {current_status} phase")

        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")

    def _stop_run(self) -> None:
        # [FUNCTION DOC] HELPER: Stops the current run via the CLI. Raises StrobelightCLIProfilerError if the command fails
        # or its output does not report success.
        command = ["strobeclient", "stopRun", "--run-id", str(self.current_run_id)]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to stop strobelight profiling, return code is not 0 :{output}"
            )

        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            if current_status.__contains__("Success!"):
                return
            else:
                raise StrobelightCLIProfilerError(
                    f"failed to stop strobelight profiling, got {current_status} result"
                )

        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")

    def _get_results(self) -> None:
        # [FUNCTION DOC] HELPER: Fetches run status via the CLI; if still PROCESSING, sleeps 10 seconds and retries
        # recursively. Raises on CLI failure or an unexpected status. Then collects summary lines (total
        # samples, GraphProfiler, icicle view) into profile_result and logs them.
        # KNOWN ISSUE: profile_result is extended with the characters of each matched string rather than whole
        # lines, since a string is added to a list in place.
        command = ["strobeclient", "getRunStatus", "--run-id", str(self.current_run_id)]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to extract profiling results, return code is not 0 : {output}"
            )

        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            if current_status.__contains__("Profile run status: PROCESSING"):
                time.sleep(10)
                self._get_results()
                return
            elif not current_status.__contains__("Profile run finished with SUCCESS"):
                raise StrobelightCLIProfilerError(
                    f"failed to extract profiling results, unexpected response {output}"
                )

        self.profile_result = []
        for item in re.findall(
            r"(Total samples(.*)|GraphProfiler(.*)|Icicle view \(python stack\)(.*))",
            output,
        ):
            self.profile_result += item[0]
            logger.info(item[0])

    def _stop_strobelight_no_throw(
        self,
        collect_results: bool,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stops the run and optionally collects results; any exception is logged as a warning and
        # not propagated.
        try:
            # call stop run
            self._stop_run()
            logger.info("strobelight profiling stopped")

            logger.debug("collection stopped")

            if not collect_results:
                return

            self._get_results()
        except Exception:
            logger.warning("error during stop_strobelight", exc_info=True)

    # Return true if strobelight started and is running. Never throw.
    def _start_strobelight(self) -> bool:
        # [FUNCTION DOC] HELPER: Starts a run and waits until it is running. Returns True on success; on any failure logs a
        # warning, stops a partially started run, and returns False. Never raises.
        strobelight_started = False
        try:
            self._run_async()
            strobelight_started = True
            logger.info("strobelight run id is: %s", self.current_run_id)
            self._wait_for_running()
            logger.info("strobelight profiling running")
            return True

        except Exception:
            logger.warning("error during start_strobelight:", exc_info=True)
            if strobelight_started:
                self._stop_strobelight_no_throw(collect_results=False)
            return False

    def profile(
        self, work_function: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R | None:
        # [FUNCTION DOC] Runs work_function with the given arguments under strobelight and returns its result. Resets
        # current_run_id/profile_result. Takes the class-wide lock without blocking. If profiler start fails,
        # raises when stop_at_error else runs the function unprofiled. If the function raises, profiling is
        # stopped without results, the lock released and the error re-raised. On success stops and collects
        # results, logging the elapsed time.
        # KNOWN ISSUE: if the lock is already held the function returns None without running work_function; the
        # inner concurrent-run handling (raise or run anyway) is unreachable.
        self.current_run_id = None
        self.profile_result = None

        if locked := StrobelightCLIFunctionProfiler._lock.acquire(False):
            if not locked:
                if self.stop_at_error:
                    raise StrobelightCLIProfilerError("concurrent runs not supported")

                logger.warning("concurrent runs not supported")
                return work_function(*args, **kwargs)

            started = self._start_strobelight()
            if not started:
                if self.stop_at_error:
                    StrobelightCLIFunctionProfiler._lock.release()
                    raise StrobelightCLIProfilerError(
                        "failed to start strobelight profiling"
                    )
                result = work_function(*args, **kwargs)
                StrobelightCLIFunctionProfiler._lock.release()
                return result

            try:
                logger.debug("collection started")
                start = timer()
                result = work_function(*args, **kwargs)
                end = timer()
                total_time = end - start  # Time in seconds, e.g. 5.38091952400282
                logger.info("work function took %s seconds", total_time)
                self._stop_strobelight_no_throw(collect_results=True)
                StrobelightCLIFunctionProfiler._lock.release()
                return result
            except Exception as error:
                logger.warning("work function throw exception", exc_info=True)
                self._stop_strobelight_no_throw(collect_results=False)
                StrobelightCLIFunctionProfiler._lock.release()
                raise error
        return None


# A function decorator that wraps profile, if no profiler is provided one with
# default args is created. A function can be annotated as:
# @strobelight()
# @strobelight(profiler = StrobelightFunctionProfiler(stop_at_error=True,..))
# @strobelight(stop_at_error=True,...)
def strobelight(
    profiler: StrobelightCLIFunctionProfiler | None = None, **kwargs: Any
) -> Callable[[Callable[_P, _R]], Callable[_P, _R | None]]:
    # [FUNCTION DOC] Decorator factory: wraps a function so each call goes through the profile method of the given
    # profiler, or of a new profiler built from kwargs when none is given. The decorated function returns the
    # original result, or None if profiling was skipped.
    if not profiler:
        profiler = StrobelightCLIFunctionProfiler(**kwargs)

    def strobelight_inner(
        work_function: Callable[_P, _R],
    ) -> Callable[_P, _R | None]:
        # [FUNCTION DOC] HELPER: Decorator applied to work_function; returns a wrapper preserving its metadata.
        @functools.wraps(work_function)
        def wrapper_function(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
            # [FUNCTION DOC] HELPER: Calls the profiler's profile method on the wrapped function with the call arguments.
            # pyrefly: ignore [bad-argument-type]
            return profiler.profile(work_function, *args, **kwargs)

        return wrapper_function

    return strobelight_inner
