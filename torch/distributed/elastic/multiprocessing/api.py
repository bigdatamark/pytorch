#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Process-launching abstractions for the elastic agent: Std redirect flags, log
# destination specs (LogsSpecs, DefaultLogsSpecs, LogsDest), RunProcsResult, the PContext base
# class and its concrete MultiprocessContext (function entrypoints) and SubprocessContext
# (binary entrypoints), plus start_processes-related helpers. Import-time side effects: creates a
# module logger and platform flags IS_WINDOWS / IS_MACOS.

import abc
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntFlag
from multiprocessing import synchronize
from types import FrameType
from typing import Any, TextIO, Union

import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record
from torch.distributed.elastic.multiprocessing.redirects import (
    redirect_stderr,
    redirect_stdout,
)
from torch.distributed.elastic.multiprocessing.subprocess_handler import (
    get_subprocess_handler,
    SubprocessHandler,
)
from torch.distributed.elastic.multiprocessing.tail_log import TailLog
from torch.numa.binding import _maybe_wrap_with_numa_binding, NumaOptions


IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


logger = logging.getLogger(__name__)

__all__ = [
    "DefaultLogsSpecs",
    "SignalException",
    "Std",
    "to_map",
    "RunProcsResult",
    "PContext",
    "get_std_cm",
    "MultiprocessContext",
    "SubprocessContext",
    "LogsDest",
    "LogsSpecs",
]


class SignalException(Exception):
    # [CLASS DOC] Raised in the agent process by the termination handler when a death signal is
    # received; carries the signal value in attribute sigval. Must not be silently swallowed.
    """
    Exception is raised inside the torchelastic agent process by the termination handler
    if the death signal got received by the process.
    """

    def __init__(self, msg: str, sigval: signal.Signals) -> None:
        # [FUNCTION DOC] Initializes the exception with msg and records the signal in sigval.
        super().__init__(msg)
        self.sigval = sigval


def _terminate_process_handler(signum: int, frame: FrameType | None) -> None:
    # [FUNCTION DOC] HELPER: Signal handler that converts the received signal number to a signal
    # enum and always raises SignalException with a message naming this process id. The frame is unused.
    """Termination handler that raises exceptions on the main process.

    When the process receives death signal(SIGTERM, SIGINT), this termination handler will
    be invoked. It raises the ``SignalException`` exception that should be processed by the
    user code. Python does not terminate process after the termination handler is finished,
    so the exception should not be silently ignored, otherwise the process will never
    be terminated.
    """
    sigval = signal.Signals(signum)
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)


def _get_kill_signal() -> signal.Signals:
    # [FUNCTION DOC] HELPER: Returns the forceful kill signal: SIGKILL on unix, the Ctrl-C event on Windows.
    """Get the kill signal. SIGKILL for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined]
    else:
        return signal.SIGKILL


def _get_default_signal() -> signal.Signals:
    # [FUNCTION DOC] HELPER: Returns the default termination signal: SIGTERM on unix, the Ctrl-C event on Windows.
    """Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined]
    else:
        return signal.SIGTERM


def _validate_full_rank(d: dict[int, Any], nprocs: int, what: str):
    # [FUNCTION DOC] HELPER: Checks that the mapping keys are exactly the local ranks zero through
    # nprocs minus one; otherwise raises a runtime error naming what was validated and the
    # expected and actual key sets.
    actual_keys = set(d.keys())
    expected_keys = set(range(nprocs))

    if actual_keys != expected_keys:
        raise RuntimeError(
            f"{what}, local rank mapping mismatch,"
            f" expected: {expected_keys}, actual: {actual_keys}"
        )


_MAPPING_REGEX = r"^(\d:[0123],)*(\d:[0123])$"
_VALUE_REGEX = r"^[0123]$"


class Std(IntFlag):
    # [CLASS DOC] Bit flags selecting standard streams: NONE (0), OUT (1), ERR (2), ALL (both).
    # Flags can be combined with bitwise or; from_str parses the CLI string form.
    NONE = 0
    OUT = 1
    ERR = 2
    ALL = OUT | ERR

    @classmethod
    def from_str(cls, vm: str) -> Union["Std", dict[int, "Std"]]:
        # [FUNCTION DOC] Parses either a single digit 0-3 into one flag value, or a comma-separated
        # list of rank:digit pairs into a mapping from local rank to flag value. Raises a value
        # error for any other text.
        """
        Example:
        ::

         from_str("0") -> Std.NONE
         from_str("1") -> Std.OUT
         from_str("0:3,1:0,2:1,3:2") -> {0: Std.ALL, 1: Std.NONE, 2: Std.OUT, 3: Std.ERR}

        Any other input raises an exception
        """

        def to_std(v: str) -> Std:  # type: ignore[return]
            # [FUNCTION DOC] HELPER: Converts a digit string to a Std flag, returning it if valid.
            s = Std(int(v))
            if s in Std:
                return s
            # return None -> should NEVER reach here since we regex check input

        if re.match(_VALUE_REGEX, vm):  # vm is a number (e.g. 0)
            return to_std(vm)
        elif re.match(_MAPPING_REGEX, vm):  # vm is a mapping (e.g. 0:1,1:2)
            d: dict[int, Std] = {}
            for m in vm.split(","):
                i, v = m.split(":")
                d[int(i)] = to_std(v)
            return d
        else:
            raise ValueError(
                f"{vm} does not match: <{_VALUE_REGEX}> or <{_MAPPING_REGEX}>"
            )


def to_map(val_or_map: Std | dict[int, Std], local_world_size: int) -> dict[int, Std]:
    # [FUNCTION DOC] Expands a single Std value to a mapping giving it to every local rank from
    # zero to local_world_size minus one, or completes a partial mapping by filling missing
    # ranks with NONE (extra keys beyond the world size are dropped). Returns a new mapping.
    """
    Certain APIs take redirect settings either as a single value (e.g. apply to all
    local ranks) or as an explicit user-provided mapping. This method is a convenience
    method that converts a value or mapping into a mapping.

    Example:
    ::

     to_map(Std.OUT, local_world_size=2)  # returns: {0: Std.OUT, 1: Std.OUT}
     to_map({1: Std.OUT}, local_world_size=2)  # returns: {0: Std.NONE, 1: Std.OUT}
     to_map(
         {0: Std.OUT, 1: Std.OUT}, local_world_size=2
     )  # returns: {0: Std.OUT, 1: Std.OUT}
    """
    if isinstance(val_or_map, Std):
        return dict.fromkeys(range(local_world_size), val_or_map)
    else:
        map = {}
        for i in range(local_world_size):
            map[i] = val_or_map.get(i, Std.NONE)
        return map


@dataclass
class LogsDest:
    # [CLASS DOC] Dataclass of log destinations, each a mapping from local rank to file path:
    # stdouts, stderrs, tee_stdouts (streams to tail to console), tee_stderrs, error_files; plus
    # single paths for the filtered stdout and stderr aggregate files. An empty string in
    # stdouts/stderrs means console output.
    """
    For each log type, holds mapping of local rank ids to file paths.
    """

    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)
    tee_stdouts: dict[int, str] = field(default_factory=dict)
    tee_stderrs: dict[int, str] = field(default_factory=dict)
    error_files: dict[int, str] = field(default_factory=dict)
    filtered_stdout: str = field(default_factory=str)
    filtered_stderr: str = field(default_factory=str)


class LogsSpecs(ABC):
    # [CLASS DOC] Abstract policy for where each worker's logs go. Stores the root log directory,
    # the redirect and tee stream selections (single value or per-rank mapping) and an optional
    # set of local ranks whose console output is kept. Subclasses implement reify and root_log_dir.
    """
    Defines logs processing and redirection for each worker process.

    Args:
        log_dir:
            Base directory where logs will be written.
        redirects:
            Streams to redirect to files. Pass a single ``Std``
            enum to redirect for all workers, or a mapping keyed
            by local_rank to selectively redirect.
        tee:
            Streams to duplicate to stdout/stderr.
            Pass a single ``Std`` enum to duplicate streams for all workers,
            or a mapping keyed by local_rank to selectively duplicate.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
        local_ranks_filter: set[int] | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores log_dir, redirects, tee and local_ranks_filter without validation.
        self._root_log_dir = log_dir
        self._redirects = redirects
        self._tee = tee
        self._local_ranks_filter = local_ranks_filter

    @abstractmethod
    def reify(
        self,
        envs: dict[int, dict[str, str]],
    ) -> LogsDest:
        # [FUNCTION DOC] Abstract: given the per-local-rank environment mappings, computes the log
        # destinations for each rank and may set per-rank environment entries (such as the error file).
        """
        Given the environment variables, builds destination of log files for each of the local ranks.

        Envs parameter contains env variables dict for each of the local ranks, where entries are defined in:
        :func:`~torchelastic.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent._start_workers`.
        """

    @property
    @abstractmethod
    def root_log_dir(self) -> str:
        # [FUNCTION DOC] Abstract property: returns the root log directory as text.
        pass


class DefaultLogsSpecs(LogsSpecs):
    # [CLASS DOC] Standard LogsSpecs: lays out one directory per run and attempt, with one
    # subdirectory per local rank holding stdout, stderr and error files, and aggregate filtered
    # logs per attempt. State: the run directory is created once on first reify and reused.
    # The null device as log_dir disables all log files.
    """
    Default LogsSpecs implementation:

    - `log_dir` will be created if it doesn't exist
    - Generates nested folders for each attempt and rank.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
        local_ranks_filter: set[int] | None = None,
    ) -> None:
        # [FUNCTION DOC] Prepares the root log directory unless it is the null device: creates a
        # temporary directory if none is given, creates the directory if missing, and raises
        # NotADirectoryError if the path is an existing file. Then delegates to the base
        # constructor and clears the lazily created run directory.
        if log_dir != os.devnull:
            if not log_dir:
                log_dir = tempfile.mkdtemp(prefix="torchelastic_")
            elif not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            else:
                if os.path.isfile(log_dir):
                    raise NotADirectoryError(f"log_dir: {log_dir} is a file")
        super().__init__(log_dir, redirects, tee, local_ranks_filter)
        # initialized only once
        self._run_log_dir = None

    @property
    def root_log_dir(self) -> str:
        # [FUNCTION DOC] Returns the root log directory as text.
        # KNOWN ISSUE: if the root is None this returns the text None rather than a path.
        return str(self._root_log_dir)

    def _make_log_dir(self, log_dir: str | None, rdzv_run_id: str):
        # [FUNCTION DOC] HELPER: Ensures the base directory exists (temporary one if none given) and
        # creates a uniquely named subdirectory prefixed with the run id. Logs and returns its path.
        base_log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        os.makedirs(base_log_dir, exist_ok=True)
        dir = tempfile.mkdtemp(prefix=f"{rdzv_run_id}_", dir=base_log_dir)
        logger.info("log directory set to: %s", dir)
        return dir

    def reify(
        self,
        envs: dict[int, dict[str, str]],
    ) -> LogsDest:
        # [FUNCTION DOC] Builds LogsDest for len(envs) local ranks. Reads run id and restart count
        # from rank zero's environment (defaults for tests). Unless the root is the null device,
        # creates the run directory once, then (re)creates a fresh attempt directory and a
        # subdirectory per rank. Per rank decides stdout/stderr file paths from redirect flags
        # (tee implies redirect), tee paths, and error file path, and writes the error file path
        # into that rank's environment mapping (empty for null device). Ranks outside the filter
        # have tee dropped and unredirected streams sent to the null device. Returns LogsDest
        # including the filtered output file paths. Side effects: file system and env mutation.
        """
        Uses following scheme to build log destination paths:

        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stdout.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stderr.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/error.json`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/filtered_stdout.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/filtered_stderr.log`
        """
        nprocs = len(envs)
        global_env = {}  # use only to query properties that are not dependent on a rank
        if nprocs > 0:
            global_env = envs[0]
        else:
            logger.warning(
                "Empty envs map provided when defining logging destinations."
            )
        # Keys are always defined, but values can be missing in unit tests
        run_id = global_env.get("TORCHELASTIC_RUN_ID", "test_run_id")
        restart_count = global_env.get("TORCHELASTIC_RESTART_COUNT", "0")

        attempt_log_dir: str = ""
        if self._root_log_dir != os.devnull:
            if not self._run_log_dir:
                self._run_log_dir = self._make_log_dir(self._root_log_dir, run_id)

            attempt_log_dir = os.path.join(
                self._run_log_dir, f"attempt_{restart_count}"
            )  # type: ignore[call-overload]
            shutil.rmtree(attempt_log_dir, ignore_errors=True)
            os.makedirs(attempt_log_dir)

        if self._root_log_dir == os.devnull:
            attempt_log_dir = os.devnull

        # create subdirs for each local rank in the logs_dir
        # logs_dir
        #       |- 0
        #          |- error.json
        #          |- stdout.log
        #          |- stderr.log
        #       |- ...
        #       |- (nprocs-1)
        redirs = to_map(self._redirects, nprocs)
        ts = to_map(self._tee, nprocs)

        # to tee stdout/stderr we first redirect into a file
        # then tail -f stdout.log/stderr.log so add tee settings to redirects
        for local_rank, tee_std in ts.items():
            redirect_std = redirs[local_rank]
            redirs[local_rank] = redirect_std | tee_std

        SYS_STREAM = ""  # special case to indicate to output to console
        stdouts = dict.fromkeys(range(nprocs), SYS_STREAM)
        stderrs = dict.fromkeys(range(nprocs), SYS_STREAM)
        tee_stdouts: dict[int, str] = {}
        tee_stderrs: dict[int, str] = {}
        error_files = {}

        for local_rank in range(nprocs):
            if attempt_log_dir == os.devnull:
                tee_stdouts[local_rank] = os.devnull
                tee_stderrs[local_rank] = os.devnull
                error_files[local_rank] = os.devnull
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = ""
            else:
                clogdir = os.path.join(attempt_log_dir, str(local_rank))
                os.mkdir(clogdir)

                rd = redirs[local_rank]
                if (rd & Std.OUT) == Std.OUT:
                    stdouts[local_rank] = os.path.join(clogdir, "stdout.log")
                if (rd & Std.ERR) == Std.ERR:
                    stderrs[local_rank] = os.path.join(clogdir, "stderr.log")

                t = ts[local_rank]
                if t & Std.OUT == Std.OUT:
                    tee_stdouts[local_rank] = stdouts[local_rank]
                if t & Std.ERR == Std.ERR:
                    tee_stderrs[local_rank] = stderrs[local_rank]

                if (
                    self._local_ranks_filter
                    and local_rank not in self._local_ranks_filter
                ):
                    # If stream is tee'd, only write to file, but don't tail
                    if local_rank in tee_stdouts:
                        tee_stdouts.pop(local_rank, None)
                    if local_rank in tee_stderrs:
                        tee_stderrs.pop(local_rank, None)

                    # If stream is not redirected, don't print
                    if stdouts[local_rank] == SYS_STREAM:
                        stdouts[local_rank] = os.devnull
                    if stderrs[local_rank] == SYS_STREAM:
                        stderrs[local_rank] = os.devnull

                error_file = os.path.join(clogdir, "error.json")
                # pyrefly: ignore [unsupported-operation]
                error_files[local_rank] = error_file
                logger.info(
                    "Setting worker%s reply file to: %s", local_rank, error_file
                )
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file

        return LogsDest(
            stdouts,
            stderrs,
            tee_stdouts,
            tee_stderrs,
            # pyrefly: ignore [bad-argument-type]
            error_files,
            os.path.join(attempt_log_dir, "filtered_stdout.log"),
            os.path.join(attempt_log_dir, "filtered_stderr.log"),
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns text showing root log dir, redirects, tee and rank filter.
        return (
            f"DefaultLogsSpecs(root_log_dir={self._root_log_dir}, redirects={self._redirects}, "
            f"tee={self._tee}, local_ranks_filter={self._local_ranks_filter})"
        )

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a DefaultLogsSpecs with the same root directory,
        # redirects, tee and rank filter; false for other types.
        if not isinstance(other, DefaultLogsSpecs):
            return False

        return (
            self._root_log_dir == other._root_log_dir
            and self._redirects == other._redirects
            and self._tee == other._tee
            and self._local_ranks_filter == other._local_ranks_filter
        )


@dataclass
class RunProcsResult:
    # [CLASS DOC] Dataclass result of a finished process run, all mappings keyed by local rank:
    # return_values (functions only), failures (ProcessFailure), stdouts and stderrs file paths.
    """
    Results of a completed run of processes started with ``start_processes()``. Returned by ``PContext``.

    Note the following:

    1. All fields are mapped by local rank
    2. ``return_values`` - only populated for functions (not the binaries).
    3. ``stdouts`` - path to stdout.log (empty string if no redirect)
    4. ``stderrs`` - path to stderr.log (empty string if no redirect)

    """

    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)

    def is_failed(self) -> bool:
        # [FUNCTION DOC] Returns true if any worker failed (failures mapping is non-empty).
        return len(self.failures) > 0


class PContext(abc.ABC):
    # [CLASS DOC] Abstract context managing a set of worker processes launched by some mechanism.
    # State: name, entrypoint, per-rank args and envs, per-rank stdout/stderr/error-file paths,
    # process count, optional filtered output files, and a list of TailLog tailers (console tee
    # plus filtered copies). Subclasses implement process start, polling, pid listing and close.
    """
    The base class that standardizes operations over a set of processes that are launched via different mechanisms.

    The name ``PContext`` is intentional to disambiguate with ``torch.multiprocessing.ProcessContext``.

    .. warning:: stdouts and stderrs should ALWAYS be a superset of
                 tee_stdouts and tee_stderrs (respectively) this is b/c
                 tee is implemented as a redirect + tail -f <stdout/stderr.log>

    Args:
        duplicate_stdout_filters:
            If non-empty, duplicates stdouts specified in ``logs_specs``'s ``tee``
            to a file containing only lines that match _any_ of the filter strings.
            The log file is aggregated across all ranks selected by ``tee``.
        duplicate_stderr_filters:
            If non-empty, duplicates stderrs specified in ``logs_specs``'s ``tee``
            to a file containing only lines that match _any_ of the filter strings.
            The log file is aggregated across all ranks selected by ``tee``.
    """

    def __init__(
        self,
        name: str,
        entrypoint: Callable | str,
        args: dict[int, tuple],
        envs: dict[int, dict[str, str]],
        logs_specs: LogsSpecs,
        log_line_prefixes: dict[int, str] | None = None,
        duplicate_stdout_filters: list[str] | None = None,
        duplicate_stderr_filters: list[str] | None = None,
    ):
        # [FUNCTION DOC] Resolves log destinations via logs_specs (which may modify envs), validates
        # that stdout and stderr mappings cover exactly the ranks of args (runtime error otherwise),
        # stores the configuration, and builds tailers for tee'd stdout and stderr. For each
        # non-empty duplicate filter list also opens a line-buffered filtered output file (side
        # effect) and adds a tailer writing only lines containing any filter substring.
        self.name = name
        # validate that all mappings have the same number of keys and
        # all local ranks are accounted for
        nprocs = len(args)

        # TODO log_line_prefixes can be expanded too
        logs_dest = logs_specs.reify(envs)

        _validate_full_rank(logs_dest.stdouts, nprocs, "stdouts")
        _validate_full_rank(logs_dest.stderrs, nprocs, "stderrs")

        self.entrypoint = entrypoint
        self.args = args
        self.envs = envs
        self.stdouts = logs_dest.stdouts
        self.stderrs = logs_dest.stderrs
        self.error_files = logs_dest.error_files
        self.nprocs = nprocs
        self.filtered_stdout: TextIO | None = None
        self.filtered_stderr: TextIO | None = None

        self._tail_logs = [
            TailLog(name, logs_dest.tee_stdouts, sys.stdout, log_line_prefixes),
            TailLog(name, logs_dest.tee_stderrs, sys.stderr, log_line_prefixes),
        ]

        if duplicate_stdout_filters:
            self.filtered_stdout = open(  # noqa: SIM115
                logs_dest.filtered_stdout, mode="w", errors="replace", buffering=1
            )
            self._tail_logs.append(
                TailLog(
                    name,
                    logs_dest.tee_stdouts,
                    self.filtered_stdout,
                    log_line_prefixes,
                    log_line_filter=lambda line: any(
                        needle in line for needle in duplicate_stdout_filters
                    ),
                )
            )

        if duplicate_stderr_filters:
            self.filtered_stderr = open(  # noqa: SIM115
                logs_dest.filtered_stderr, mode="w", errors="replace", buffering=1
            )
            self._tail_logs.append(
                TailLog(
                    name,
                    logs_dest.tee_stderrs,
                    self.filtered_stderr,
                    log_line_prefixes,
                    log_line_filter=lambda line: any(
                        needle in line for needle in duplicate_stderr_filters
                    ),
                )
            )

    def start(self) -> None:
        # [FUNCTION DOC] On the main thread, registers the termination handler (which raises
        # SignalException) for each signal named in the comma-separated environment variable
        # for signals to handle (default terminate, interrupt, hangup, quit); unknown or
        # unregistrable signals are logged as warnings, and unsupported ones on Windows as info.
        # On other threads only logs a warning. Then calls the subclass start hook and starts
        # all tailers. Side effects: process-wide signal handlers, worker processes, threads.
        """Start processes using parameters defined in the constructor."""
        if threading.current_thread() is threading.main_thread():
            # Register signal handlers for the signals specified in the environment variable
            signals_to_handle = os.environ.get(
                "TORCHELASTIC_SIGNALS_TO_HANDLE", "SIGTERM,SIGINT,SIGHUP,SIGQUIT"
            )
            signal_list = signals_to_handle.split(",")

            for sig_name in signal_list:
                try:
                    sig = getattr(signal, sig_name.strip())
                    signal.signal(sig, _terminate_process_handler)
                    logger.info("Registered signal handler for %s", sig_name)
                except (AttributeError, ValueError):
                    logger.warning(
                        "Failed to register signal handler for %s",
                        sig_name,
                        exc_info=True,
                    )
                except RuntimeError:
                    if IS_WINDOWS and sig_name.strip() in [
                        "SIGHUP",
                        "SIGQUIT",
                        "SIGUSR1",
                        "SIGUSR2",
                    ]:
                        logger.info(
                            "Signal %s is not supported on Windows, skipping", sig_name
                        )
                    else:
                        logger.warning(
                            "Failed to register signal handler for %s",
                            sig_name,
                            exc_info=True,
                        )
        else:
            logger.warning(
                "Failed to register signal handlers since torchelastic is running on a child thread. "
                "This could lead to orphaned worker processes if the torchrun is terminated."
            )
        self._start()
        for tail_log in self._tail_logs:
            tail_log.start()

    @abc.abstractmethod
    def _start(self) -> None:
        # [FUNCTION DOC] Abstract: launches the worker processes using the subclass mechanism.
        """Start processes using strategy defined in a particular context."""
        raise NotImplementedError

    @abc.abstractmethod
    def _poll(self) -> RunProcsResult | None:
        # [FUNCTION DOC] Abstract: non-blocking status check, all-or-nothing. Returns a result when
        # all processes succeeded or any failed, or None while all are still running.
        """
        Poll the run status of the processes running under this context.
        This method follows an "all-or-nothing" policy and returns
        a ``RunProcessResults`` object if either all processes complete
        successfully or any process fails. Returns ``None`` if
        all processes are still running.
        """
        raise NotImplementedError

    def wait(self, timeout: float = -1, period: float = 1) -> RunProcsResult | None:
        # [FUNCTION DOC] Polls every period seconds until a result is available or timeout seconds
        # pass. Zero timeout does a single poll; negative waits effectively forever. Returns the
        # result, or None on timeout. Propagates SignalException raised by the signal handler.
        """
        Wait for the specified ``timeout`` seconds, polling every ``period`` seconds
        for the processes to be done. Returns ``None`` if the processes are still running
        on timeout expiry. Negative timeout values are interpreted as "wait-forever".
        A timeout value of zero simply queries the status of the processes (e.g. equivalent
        to a poll).

        .. note::
            Multiprocessing library registers SIGTERM and SIGINT signal handlers that raise
            ``SignalException`` when the signals received. It is up to the consumer of the code
            to properly handle the exception. It is important not to swallow the exception otherwise
            the process would not terminate. Example of the typical workflow can be:

        .. code-block:: python
            pc = start_processes(...)
            try:
                pc.wait(1)
                .. do some other work
            except SignalException as e:
                pc.shutdown(e.sigval, timeout=30)

        If SIGTERM or SIGINT occurs, the code above will try to shutdown child processes by propagating
        received signal. If child processes will not terminate in the timeout time, the process will send
        the SIGKILL.
        """
        if timeout == 0:
            return self._poll()

        if timeout < 0:
            timeout = sys.maxsize

        expiry = time.time() + timeout
        while time.time() < expiry:
            pr = self._poll()
            if pr:
                return pr
            time.sleep(period)

        return None

    @abc.abstractmethod
    def pids(self) -> dict[int, int]:
        # [FUNCTION DOC] Abstract: returns a mapping from local rank to process id.
        """Return pids of processes mapped by their respective local_ranks."""
        raise NotImplementedError

    @abc.abstractmethod
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        # [FUNCTION DOC] Abstract: sends death_sig to all workers, waits up to timeout, escalates
        # to a kill, and cleans up subclass resources.
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).
        """
        raise NotImplementedError

    def close(self, death_sig: signal.Signals | None = None, timeout: int = 30) -> None:
        # [FUNCTION DOC] Terminates workers via the subclass close hook (default signal if none
        # given), then stops all tailers and closes the filtered output files if open.
        # Side effects: signals processes, joins tail threads, closes files.
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).

        Args:
            death_sig: Death signal to terminate processes.
            timeout: Time to wait for processes to finish, if process is
                still alive after this time, it will be terminated via SIGKILL.
        """
        if not death_sig:
            death_sig = _get_default_signal()
        self._close(death_sig=death_sig, timeout=timeout)
        for tail_log in self._tail_logs:
            tail_log.stop()
        if self.filtered_stdout:
            self.filtered_stdout.close()
        if self.filtered_stderr:
            self.filtered_stderr.close()


def get_std_cm(std_rd: str, redirect_fn):
    # [FUNCTION DOC] Returns a context manager for redirecting a stream: a no-op one on Windows,
    # macOS or when std_rd is empty; otherwise the result of calling redirect_fn with std_rd.
    if IS_WINDOWS or IS_MACOS or not std_rd:
        return nullcontext()
    else:
        return redirect_fn(std_rd)


def _wrap(
    local_rank: int,
    fn: Callable,
    args: dict[int, tuple],
    envs: dict[int, dict[str, str]],
    stdout_redirects: dict[int, str],  # redirect file for stdout (to console if None)
    stderr_redirects: dict[int, str],  # redirect file for stderr (to console if None)
    ret_vals: dict[int, mp.SimpleQueue],
    queue_finished_reading_event: synchronize.Event,
    numa_options: NumaOptions | None,
) -> None:
    # [FUNCTION DOC] HELPER: Worker-process entry point. Looks up this rank's args, env, return
    # queue and redirect paths (key error if missing), applies the env to the process environment,
    # runs the function (with optional NUMA binding and error recording) inside the stdout/stderr
    # redirects, puts the return value on the queue, then blocks until the parent signals that
    # it finished reading. Side effects: mutates os.environ, redirects streams.
    # get the per-rank params up front so we fail fast if no mapping is found
    args_ = args[local_rank]
    env_ = envs[local_rank]
    ret_val_ = ret_vals[local_rank]

    stdout_rd = stdout_redirects[local_rank]
    stderr_rd = stderr_redirects[local_rank]

    stdout_cm = get_std_cm(stdout_rd, redirect_stdout)
    stderr_cm = get_std_cm(stderr_rd, redirect_stderr)

    for k, v in env_.items():
        os.environ[k] = v

    with stdout_cm, stderr_cm:
        fn = _maybe_wrap_with_numa_binding(
            fn, device_index=local_rank, numa_options=numa_options
        )
        ret = record(fn)(*args_)
    ret_val_.put(ret)
    queue_finished_reading_event.wait()


class MultiprocessContext(PContext):
    # [CLASS DOC] PContext that runs a Python function in worker processes started through
    # torch multiprocessing with the given start method. State: per-rank return value queues,
    # collected return values, the process context handle (None until started), an event telling
    # workers the parent finished reading results, and NUMA options. Fails fast if any worker fails.
    """``PContext`` holding worker processes invoked as a function."""

    def __init__(
        self,
        name: str,
        entrypoint: Callable,
        args: dict[int, tuple],
        envs: dict[int, dict[str, str]],
        start_method: str,
        logs_specs: LogsSpecs,
        log_line_prefixes: dict[int, str] | None = None,
        numa_options: NumaOptions | None = None,
        duplicate_stdout_filters: list[str] | None = None,
        duplicate_stderr_filters: list[str] | None = None,
    ):
        # [FUNCTION DOC] Initializes the base context, then creates one single-item return queue
        # per rank and the worker-finished event from the chosen start method context, and
        # stores NUMA options. Does not start any process.
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_line_prefixes,
            duplicate_stdout_filters,
            duplicate_stderr_filters,
        )

        self.start_method = start_method
        # each ret_val queue will always contain a single element.
        self._ret_vals = {
            local_rank: mp.get_context(self.start_method).SimpleQueue()
            for local_rank in range(self.nprocs)
        }

        # see comments in ``join()`` for what this is
        self._return_values: dict[int, Any] = {}
        self._pc: mp.ProcessContext | None = None
        # Note: set method should ONLY be invoked for the use case when all processes finished
        # successfully. If any process died on event.wait() calling set() method will deadlock.
        self._worker_finished_event = mp.get_context(self.start_method).Event()

        self._numa_options: NumaOptions | None = numa_options

    def _start(self):
        # [FUNCTION DOC] Starts the worker processes non-blocking and non-daemonic via torch
        # multiprocessing, running the wrapper function with the shared configuration. Raises a
        # value error if already started.
        if self._pc:
            raise ValueError(
                "The process context already initialized."
                " Most likely the start method got called twice."
            )
        self._pc = mp.start_processes(
            fn=_wrap,
            args=(
                self.entrypoint,
                self.args,
                self.envs,
                self.stdouts,
                self.stderrs,
                self._ret_vals,
                self._worker_finished_event,
                self._numa_options,
            ),
            nprocs=self.nprocs,
            join=False,
            daemon=False,
            start_method=self.start_method,
        )

    def _is_done(self) -> bool:
        # [FUNCTION DOC] HELPER: True when a return value has been collected from every rank.
        return len(self._return_values) == self.nprocs

    def _poll(self) -> RunProcsResult | None:
        # [FUNCTION DOC] Non-blocking status check. Joins the process context (raising if a worker
        # raised or exited abnormally), drains any ready return queues, and when all return values
        # are in: releases workers, waits for all to exit, validates the return values cover all
        # ranks, closes the context and returns a successful result. Returns None while work
        # remains. On a worker failure logs it, closes the context and returns a result holding
        # a single ProcessFailure for the failing rank.
        if self._pc is None:
            raise AssertionError  # assertion for mypy type checker

        try:
            # torch.mp.ProcessContext Throws an Exception if some/all of
            # worker processes failed
            # timeout < 0 checks worker status and return immediately
            # Join will never return success since we use synchronize.Event to wait
            # for all processes to finish.
            self._pc.join(-1)

            # IMPORTANT: we use multiprocessing.Queue to carry worker return values
            # back to the parent, the worker process will wait before terminating
            # until all the buffered items are fed by the feeder thread to the underlying
            # pipe. Hence to prevent deadlocks on large return values,
            # we opportunistically try queue.get on each join call
            # See: https://docs.python.org/2/library/multiprocessing.html#all-platforms
            for local_rank in range(self.nprocs):
                return_queue = self._ret_vals[local_rank]
                if not return_queue.empty():
                    # save the return values temporarily into a member var
                    self._return_values[local_rank] = return_queue.get()

            if self._is_done():
                # we should ALWAYS have ALL the return values when all the processes are done
                self._worker_finished_event.set()

                # At this point workers finished running the user function
                # But the child process might still have not exited. Wait for them.
                # pc.join() blocks [forever] until "a" proc exits. Loop until all of them exits.
                while not self._pc.join():
                    logger.debug(
                        "entrypoint fn finished, waiting for all child procs to exit..."
                    )

                _validate_full_rank(
                    self._return_values, self.nprocs, "return_value queue"
                )
                self.close()
                return RunProcsResult(
                    return_values=self._return_values,
                    stdouts=self.stdouts,
                    stderrs=self.stderrs,
                )
            else:
                return None
        except (mp.ProcessRaisedException, mp.ProcessExitedException) as e:
            failed_local_rank = e.error_index

            # entrypoint for MultiprocessContext will always be a Callable
            fn_name = self.entrypoint.__qualname__  # type: ignore[union-attr]
            failed_proc = self._pc.processes[failed_local_rank]
            error_filepath = self.error_files[failed_local_rank]

            logger.exception(
                "failed (exitcode: %s)"
                " local_rank: %s (pid: %s)"
                " of fn: %s (start_method: %s)",
                failed_proc.exitcode,
                failed_local_rank,
                e.error_pid,
                fn_name,
                self.start_method,
            )

            self.close()
            return RunProcsResult(
                failures={
                    failed_local_rank: ProcessFailure(
                        local_rank=failed_local_rank,
                        pid=e.error_pid,
                        exitcode=failed_proc.exitcode,
                        error_file=error_filepath,
                    )
                },
                stdouts=self.stdouts,
                stderrs=self.stderrs,
            )

    def pids(self) -> dict[int, int]:
        # [FUNCTION DOC] Returns rank to pid mapping from the process context; assertion error if not started.
        if self._pc is None:
            raise AssertionError  # assertion for mypy type checking
        return dict(enumerate(self._pc.pids()))

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        # [FUNCTION DOC] No-op if not started. Sends death_sig to each live worker, waits for
        # all up to a shared timeout budget, then force-kills survivors and joins each with a
        # bounded timeout, logging an error for any that cannot be reaped. Missing processes ignored.
        if not self._pc:
            return
        for proc in self._pc.processes:
            if proc.is_alive():
                logger.warning(
                    "Closing process %s via signal %s", proc.pid, death_sig.name
                )
                try:
                    os.kill(proc.pid, death_sig)
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be raised, it is safe to ignore it.
                    pass
        end = time.monotonic() + timeout
        for proc in self._pc.processes:
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            proc.join(time_to_wait)
        for proc in self._pc.processes:
            if proc.is_alive():
                logger.warning(
                    "Unable to shutdown process %s via %s, forcefully exiting via %s",
                    proc.pid,
                    death_sig,
                    _get_kill_signal(),
                )
                try:
                    os.kill(proc.pid, _get_kill_signal())
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be raised, it is safe to ignore it.
                    pass
            # Bounded join: a process stuck in uninterruptible kernel sleep
            # (Linux D-state, common with hung NCCL/GPU collectives) cannot
            # be reaped by SIGKILL. Without a timeout the agent itself would
            # wedge here and the supervising launcher could never exit.
            proc.join(timeout)
            if proc.is_alive():
                logger.error(
                    "Process %s did not exit after %s within %ss;"
                    " abandoning and continuing teardown."
                    " Worker may be in uninterruptible kernel sleep"
                    " (e.g. wedged on GPU/NIC); host may need to be recycled.",
                    proc.pid,
                    _get_kill_signal().name,
                    timeout,
                )


class SubprocessContext(PContext):
    # [CLASS DOC] PContext that runs a binary entrypoint as one subprocess per rank through
    # SubprocessHandler objects. State: set of still-running ranks, mapping of failures, the
    # handlers by rank, and NUMA options. Return values are placeholders (None) on success.
    """``PContext`` holding worker processes invoked as a binary."""

    def __init__(
        self,
        name: str,
        entrypoint: str,
        args: dict[int, tuple],
        envs: dict[int, dict[str, str]],
        logs_specs: LogsSpecs,
        log_line_prefixes: dict[int, str] | None = None,
        numa_options: NumaOptions | None = None,
        duplicate_stdout_filters: list[str] | None = None,
        duplicate_stderr_filters: list[str] | None = None,
    ):
        # [FUNCTION DOC] Initializes the base context, marks all ranks as running, and prepares
        # empty failure and handler mappings. Does not start any process.
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_line_prefixes,
            duplicate_stdout_filters,
            duplicate_stderr_filters,
        )

        # state vector; _vdone[local_rank] -> is local_rank finished or not
        self._running_local_ranks: set[int] = set(range(self.nprocs))
        self._failures: dict[int, ProcessFailure] = {}
        self.subprocess_handlers: dict[int, SubprocessHandler] = {}
        self._numa_options: NumaOptions | None = numa_options

    def _start(self):
        # [FUNCTION DOC] Launches one subprocess per rank via the handler factory using that
        # rank's args, env, output paths and NUMA options. Raises a value error if handlers
        # already exist (start called twice).
        if self.subprocess_handlers:
            raise ValueError(
                "The subprocess handlers already initialized. Most likely the start method got called twice."
            )
        self.subprocess_handlers = {
            local_rank: get_subprocess_handler(
                entrypoint=self.entrypoint,  # type: ignore[arg-type] # entrypoint is always a str
                args=self.args[local_rank],
                env=self.envs[local_rank],
                stdout=self.stdouts[local_rank],
                stderr=self.stderrs[local_rank],
                local_rank_id=local_rank,
                numa_options=self._numa_options,
            )
            for local_rank in range(self.nprocs)
        }

    def _capture_process_failures(self, done_local_ranks: set[int]):
        # [FUNCTION DOC] HELPER: For each still-running rank, polls its process; if exited, adds
        # the rank to done_local_ranks (mutated) and, for a nonzero exit code, records a
        # ProcessFailure in the failures mapping.
        for local_rank in self._running_local_ranks:
            handler = self.subprocess_handlers[local_rank]
            exitcode = handler.proc.poll()
            if exitcode is not None:
                done_local_ranks.add(local_rank)
                if exitcode != 0:  # failed or signaled
                    self._failures[local_rank] = ProcessFailure(
                        local_rank=local_rank,
                        pid=handler.proc.pid,
                        exitcode=exitcode,
                        error_file=self.error_files[local_rank],
                    )
                # else: --> succeeded; nothing to do

    def _poll(self) -> RunProcsResult | None:
        # [FUNCTION DOC] Updates which ranks finished. If all finished or any failed, closes all
        # processes, re-captures failures caused by termination, and returns a result with
        # failures (logging the earliest one) or, on success, placeholder None return values for
        # every rank. Otherwise returns None.
        done_local_ranks: set[int] = set()
        self._capture_process_failures(done_local_ranks)

        self._running_local_ranks.difference_update(done_local_ranks)

        # if ALL procs are finished or ANY have failed
        if not self._running_local_ranks or self._failures:
            self.close()  # terminate all running procs
            self._capture_process_failures(
                done_local_ranks
            )  # log sigterms and sigkill exit codes in the self._failures for bookkeeping purposes

            result = RunProcsResult(
                failures=self._failures,
                stdouts=self.stdouts,
                stderrs=self.stderrs,
            )
            if result.is_failed():
                first_failure = min(result.failures.values(), key=lambda f: f.timestamp)
                logger.error(
                    "failed (exitcode: %s) local_rank: %s (pid: %s) of binary: %s",
                    first_failure.exitcode,
                    first_failure.local_rank,
                    first_failure.pid,
                    self.entrypoint,
                )
            else:
                # Populate return with dummy values. This provides consistency with MultiprocessingHandler
                result.return_values = dict.fromkeys(range(self.nprocs))

            return result
        else:  # there are no failures and procs still running
            return None

    def pids(self) -> dict[int, int]:
        # [FUNCTION DOC] Returns rank to process id mapping from the subprocess handlers.
        return {
            local_rank: sh.proc.pid
            for local_rank, sh in self.subprocess_handlers.items()
        }

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        # [FUNCTION DOC] No-op without handlers. Signals each live process with death_sig, waits
        # for all within a shared timeout budget (ignoring expiry), force-kills survivors and waits
        # a bounded time for each, logging an error for any that will not exit.
        if not self.subprocess_handlers:
            return
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                logger.warning(
                    "Sending process %s closing signal %s",
                    handler.proc.pid,
                    death_sig.name,
                )
                handler.close(death_sig=death_sig)
        end = time.monotonic() + timeout
        for handler in self.subprocess_handlers.values():
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            try:
                handler.proc.wait(time_to_wait)
            except subprocess.TimeoutExpired:
                # Ignore the timeout expired exception, since
                # the child process will be forcefully terminated via SIGKILL
                pass
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                logger.warning(
                    "Unable to shutdown process %s via %s, forcefully exiting via %s",
                    handler.proc.pid,
                    death_sig,
                    _get_kill_signal(),
                )
                handler.close(death_sig=_get_kill_signal())
                # Bounded wait: a process stuck in uninterruptible kernel sleep
                # (Linux D-state, common with hung NCCL/GPU collectives) cannot
                # be reaped by SIGKILL. Without a timeout the agent itself would
                # wedge here and the supervising launcher could never exit.
                try:
                    handler.proc.wait(timeout)
                except subprocess.TimeoutExpired:
                    logger.error(
                        "Process %s did not exit after %s within %ss;"
                        " abandoning and continuing teardown."
                        " Worker may be in uninterruptible kernel sleep"
                        " (e.g. wedged on GPU/NIC);"
                        " host may need to be recycled.",
                        handler.proc.pid,
                        _get_kill_signal().name,
                        timeout,
                    )
