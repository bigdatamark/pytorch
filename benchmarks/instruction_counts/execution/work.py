"""Handle the details of subprocess calls and retries for a given benchmark run."""

# mypy: ignore-errors
# [MODULE DOC] Subprocess handling for one benchmark job. Defines the shell command prefix used to
# launch workers (a threading-layer environment variable plus python, and bash as the shell), the
# frozen WorkOrder spec, a private wrapper around a worker subprocess that exchanges data through a
# pickle file in the temp dir, and InProgress which adds timeout and retry bookkeeping. Imports the
# worker package for the worker path and result types.

import dataclasses
import json
import os
import pickle
import signal
import subprocess
import time
import uuid
from typing import TYPE_CHECKING

from core.api import AutoLabels
from core.types import Label
from core.utils import get_temp_dir
from worker.main import (
    WORKER_PATH,
    WorkerFailure,
    WorkerOutput,
    WorkerTimerArgs,
    WorkerUnpickler,
)


if TYPE_CHECKING:
    PopenType = subprocess.Popen[bytes]
else:
    PopenType = subprocess.Popen


# Mitigate https://github.com/pytorch/pytorch/issues/37377
_ENV = "MKL_THREADING_LAYER=GNU"
_PYTHON = "python"
PYTHON_CMD = f"{_ENV} {_PYTHON}"

# We must specify `bash` so that `source activate ...` always works
SHELL = "/bin/bash"


@dataclasses.dataclass(frozen=True)
class WorkOrder:
    # [CLASS DOC] Frozen spec of one job for the runner: label, auto labels, worker timer
    # arguments, optional shell command to run first (such as an environment activation), optional
    # timeout in seconds, and the number of retries after a timeout. Equality compares fields but
    # hashing is by object identity, so each created order is a distinct dictionary key.
    """Spec to schedule work with the benchmark runner."""

    label: Label
    autolabels: AutoLabels
    timer_args: WorkerTimerArgs
    source_cmd: str | None = None
    timeout: float | None = None
    retries: int = 0

    def __hash__(self) -> int:
        # [FUNCTION DOC] Returns the object identity as the hash value.
        return id(self)

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns a JSON text of the label, the auto labels mapping and the thread
        # count; used as a stable human readable key for the order.
        return json.dumps(
            {
                "label": self.label,
                "autolabels": self.autolabels.as_dict,
                "num_threads": self.timer_args.num_threads,
            }
        )


class _BenchmarkProcess:
    # [CLASS DOC] HELPER: Wrapper over a worker subprocess for one work order. Holds the order, the
    # cpu list string, the Popen handle, a communication file path, start and end times, return
    # code and the collected result (a worker output or failure, filled lazily once the process
    # exits).
    """Wraps subprocess.Popen for a given WorkOrder."""

    _work_order: WorkOrder
    _cpu_list: str | None
    _proc: PopenType

    # Internal bookkeeping
    _communication_file: str
    _start_time: float
    _end_time: float | None = None
    _retcode: int | None
    _result: WorkerOutput | WorkerFailure | None = None

    def __init__(self, work_order: WorkOrder, cpu_list: str | None) -> None:
        # [FUNCTION DOC] Starts the job. Records the start time, writes the timer arguments to a
        # new randomly named pickle file in the shared temp dir, and launches the worker command in
        # a bash shell with stdout piped and stderr merged into it.
        self._work_order = work_order
        self._cpu_list = cpu_list
        self._start_time = time.time()
        self._communication_file = os.path.join(get_temp_dir(), f"{uuid.uuid4()}.pkl")
        with open(self._communication_file, "wb") as f:
            pickle.dump(self._work_order.timer_args, f)

        self._proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            executable=SHELL,
        )

    def clone(self) -> "_BenchmarkProcess":
        # [FUNCTION DOC] Returns a new wrapper that starts a fresh process for the same order and
        # cpu list.
        return _BenchmarkProcess(self._work_order, self._cpu_list)

    @property
    def cmd(self) -> str:
        # [FUNCTION DOC] Property building the shell command string: optional source command
        # followed by the and operator, the threading layer environment variable, optional CPU
        # affinity environment variable and a taskset pinned to the cpu list, then python, the
        # worker script path and the communication file argument.
        cmd: list[str] = []
        if self._work_order.source_cmd is not None:
            cmd.extend([self._work_order.source_cmd, "&&"])

        cmd.append(_ENV)

        if self._cpu_list is not None:
            cmd.extend(
                [
                    f"GOMP_CPU_AFFINITY={self._cpu_list}",
                    "taskset",
                    "--cpu-list",
                    self._cpu_list,
                ]
            )

        cmd.extend(
            [
                _PYTHON,
                WORKER_PATH,
                "--communication-file",
                self._communication_file,
            ]
        )
        return " ".join(cmd)

    @property
    def duration(self) -> float:
        # [FUNCTION DOC] Property returning seconds elapsed since start, frozen at the end time
        # once the process has been collected.
        return (self._end_time or time.time()) - self._start_time

    @property
    def result(self) -> WorkerOutput | WorkerFailure:
        # [FUNCTION DOC] Property returning the worker output or failure. Collects first; raises
        # AssertionError if the process has not finished yet.
        self._maybe_collect()
        if self._result is None:
            raise AssertionError("result is None after collection")
        return self._result

    def poll(self) -> int | None:
        # [FUNCTION DOC] Collects the result if the process has exited and returns the process
        # return code, or None while it is still running.
        self._maybe_collect()
        return self._retcode

    def interrupt(self) -> None:
        # [FUNCTION DOC] Sends SIGINT to the subprocess so it can clean up.
        """Soft interrupt. Allows subprocess to cleanup."""
        self._proc.send_signal(signal.SIGINT)

    def terminate(self) -> None:
        # [FUNCTION DOC] Sends SIGTERM to the subprocess.
        """Hard interrupt. Immediately SIGTERM subprocess."""
        self._proc.terminate()

    def _maybe_collect(self) -> None:
        # [FUNCTION DOC] HELPER: Gathers the outcome once. Does nothing if already collected or if
        # the process is still running. Otherwise reads the output from the communication file; a
        # successful output with a nonzero exit code becomes a failure saying the worker failed
        # silently; if the file still holds the original timer arguments (the worker wrote nothing)
        # the result is a failure carrying everything read from the process output. Stores the
        # result and end time and deletes the communication file. Raises AssertionError if the
        # output pipe is missing.
        if self._result is not None:
            # We've already collected the results.
            return

        self._retcode = self._proc.poll()
        if self._retcode is None:
            # `_proc` is still running
            return

        with open(self._communication_file, "rb") as f:
            result = WorkerUnpickler(f).load_output()

        if isinstance(result, WorkerOutput) and self._retcode:
            # Worker managed to complete the designated task, but worker
            # process did not finish cleanly.
            result = WorkerFailure("Worker failed silently.")

        if isinstance(result, WorkerTimerArgs):
            # Worker failed, but did not write a result so we're left with the
            # original TimerArgs. Grabbing all of stdout and stderr isn't
            # ideal, but we don't have a better way to determine what to keep.
            proc_stdout = self._proc.stdout
            if proc_stdout is None:
                raise AssertionError("proc_stdout is None")
            result = WorkerFailure(failure_trace=proc_stdout.read().decode("utf-8"))

        self._result = result
        self._end_time = time.time()

        # Release communication file.
        os.remove(self._communication_file)


class InProgress:
    # [CLASS DOC] Used by the runner to track an outstanding job: owns the subprocess wrapper and
    # implements the timeout and retry logic, counting timeouts so far.
    """Used by the benchmark runner to track outstanding jobs.
    This class handles bookkeeping and timeout + retry logic.
    """

    _proc: _BenchmarkProcess
    _timeouts: int = 0

    def __init__(self, work_order: WorkOrder, cpu_list: str | None):
        # [FUNCTION DOC] Stores the order and starts its subprocess on the given cpu list.
        self._work_order = work_order
        self._proc = _BenchmarkProcess(work_order, cpu_list)

    @property
    def work_order(self) -> WorkOrder:
        # [FUNCTION DOC] Property returning the work order of the running process.
        return self._proc._work_order

    @property
    def cpu_list(self) -> str | None:
        # [FUNCTION DOC] Property returning the cpu list string the job is pinned to (None if
        # unpinned).
        return self._proc._cpu_list

    @property
    def proc(self) -> _BenchmarkProcess:
        # [FUNCTION DOC] Property exposing the subprocess wrapper, meant for cleanup only.
        # NB: For cleanup only.
        return self._proc

    @property
    def duration(self) -> float:
        # [FUNCTION DOC] Property returning the elapsed seconds of the current subprocess.
        return self._proc.duration

    def check_finished(self) -> bool:
        # [FUNCTION DOC] Returns True once the process has exited, False while it runs. If a
        # timeout is set and exceeded, counts it; while attempts remain (retries plus one) it
        # prints a message, sends SIGINT to the old process, replaces it with a fresh clone and
        # returns False. When attempts run out raises subprocess.TimeoutExpired. The abandoned
        # process is not polled again.
        if self._proc.poll() is not None:
            return True

        timeout = self.work_order.timeout
        if timeout is None or self._proc.duration < timeout:
            return False

        self._timeouts += 1
        max_attempts = (self._work_order.retries or 0) + 1
        if self._timeouts < max_attempts:
            print(
                f"\nTimeout: {self._work_order.label}, {self._work_order.autolabels} "
                f"(Attempt {self._timeouts} / {max_attempts})"
            )
            self._proc.interrupt()
            self._proc = self._proc.clone()
            return False

        raise subprocess.TimeoutExpired(cmd=self._proc.cmd, timeout=timeout)

    @property
    def result(self) -> WorkerOutput | WorkerFailure:
        # [FUNCTION DOC] Property returning the result of the current subprocess (AssertionError if
        # not finished).
        return self._proc.result

    def __hash__(self) -> int:
        # [FUNCTION DOC] Returns the object identity as the hash value.
        return id(self)
