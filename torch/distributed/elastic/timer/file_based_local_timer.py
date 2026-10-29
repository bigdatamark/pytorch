# mypy: allow-untyped-defs
# [MODULE DOC] File (named pipe/FIFO) based timer implementation: FileTimerRequest, FileTimerClient (worker side producer) and FileTimerServer (agent side consumer with a watchdog thread that signals workers with expired timers).
# Requests are single-line JSON messages written atomically to the FIFO. Creates a module logger at import; no other import-time side effects.
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import os
import select
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec

from torch.distributed.elastic.timer.api import TimerClient, TimerRequest
from torch.distributed.elastic.timer.debug_info_logging import (
    log_debug_info_for_expired_timers,
)
from torch.distributed.elastic.utils.logging import get_logger


_P = ParamSpec("_P")
_R = TypeVar("_R")

__all__ = ["FileTimerClient", "FileTimerRequest", "FileTimerServer"]

logger = get_logger(__name__)


def _retry(max_retries: int, sleep_time: float) -> Callable:
# [FUNCTION DOC] HELPER: Decorator factory. Returns a decorator that wraps a function so it is called up to max_retries times, logging each exception, sleeping sleep_time seconds between attempts and re-raising the last exception.
    """
    A simple retry wrapper.

    Args:
        max_retries: int, the maximum number of retries.
        sleep_time: float, the time to sleep between retries.
    """

    def wrapper(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] HELPER: Decorator that takes func and returns the retrying wrapper.
        def wrapper(*args: _P.args, **kwargs: _P.kwargs):
        # [FUNCTION DOC] HELPER: Call func with the given arguments, retrying on any Exception per the enclosing max_retries/sleep_time; returns func's result or raises after the final attempt.
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    logger.exception("Error running %s. Retrying...", func.__name__)
                    if i < max_retries - 1:
                        time.sleep(sleep_time)
                    else:
                        raise

        return wrapper

    return wrapper


class FileTimerRequest(TimerRequest):
    # [CLASS DOC] TimerRequest specialization for the file timer protocol, identifying the worker by pid.
    # Extra state: version (always 1) and signal (signal number the server sends to reap the worker; zero or negative means do not kill). A negative expiration_time means release. Key methods: worker_pid, to_json.
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``FileTimerClient`` and ``FileTimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.
    ``signal`` is the signal to reap the worker process from the server
    process.
    """

    __slots__ = ["version", "signal"]

    def __init__(
        self, worker_pid: int, scope_id: str, expiration_time: float, signal: int = 0
    ) -> None:
        # [FUNCTION DOC] Initialize the base request with worker_pid as worker id, plus version 1 and the given signal (default 0).
        super().__init__(
            worker_id=worker_pid, scope_id=scope_id, expiration_time=expiration_time
        )
        self.version = 1
        self.signal = signal

    @property
    def worker_pid(self) -> int:
        # [FUNCTION DOC] Property returning the worker pid (the base worker_id).
        return self.worker_id

    def __eq__(self, other) -> bool:
        # [FUNCTION DOC] Equality: base fields equal and same version and signal; False if other is not a FileTimerRequest.
        if isinstance(other, FileTimerRequest):
            return (
                super().__eq__(other)
                and self.version == other.version
                and self.signal == other.signal
            )
        return False

    def to_json(self) -> str:
        # [FUNCTION DOC] Serialize to a JSON string with keys version, pid, scope_id, expiration_time and signal.
        return json.dumps(
            {
                "version": self.version,
                "pid": self.worker_pid,
                "scope_id": self.scope_id,
                "expiration_time": self.expiration_time,
                "signal": self.signal,
            },
        )


class FileTimerClient(TimerClient):
    # [CLASS DOC] Worker-side client of FileTimerServer on the same host. Identifies the worker by os pid and sends JSON requests through an existing FIFO at file_path (the server must have created it).
    # State: _file_path and signal (signal the server should use to kill this worker; default SIGKILL, or CTRL_C_EVENT on Windows; zero/negative means do not kill). Key methods: acquire, release.
    """
    Client side of ``FileTimerServer``. This client is meant to be used
    on the same host that the ``FileTimerServer`` is running on and uses
    pid to uniquely identify a worker.
    This client uses a named_pipe to send timer requests to the
    ``FileTimerServer``. This client is a producer while the
    ``FileTimerServer`` is a consumer. Multiple clients can work with
    the same ``FileTimerServer``.

    Args:

        file_path: str, the path of a FIFO special file. ``FileTimerServer``
                        must have created it by calling os.mkfifo().

        signal: signal, the signal to use to kill the process. Using a
                        negative or zero signal will not kill the process.
    """

    def __init__(
        self,
        file_path: str,
        signal=(signal.SIGKILL if sys.platform != "win32" else signal.CTRL_C_EVENT),  # type: ignore[attr-defined]
    ) -> None:
        # [FUNCTION DOC] Store the FIFO path and the kill signal.
        super().__init__()
        self._file_path = file_path
        self.signal = signal

    @_retry(max_retries=10, sleep_time=0.1)
    def _open_non_blocking(self) -> io.TextIOWrapper | None:
        # [FUNCTION DOC] HELPER: Open the FIFO for text writing in non-blocking mode so it fails with OSError instead of hanging when no server reader exists. Retried up to 10 times, 0.1s apart by the decorator. Returns the file object.
        # The server may have crashed or may haven't started yet.
        # In such case, calling open() in blocking model blocks the client.
        # To avoid such issue, open it in non-blocking mode, and an OSError will
        # be raised if the server is not there.
        fd = os.open(self._file_path, os.O_WRONLY | os.O_NONBLOCK)
        return os.fdopen(fd, "wt")

    def _send_request(self, request: FileTimerRequest) -> None:
        # [FUNCTION DOC] HELPER: Open the FIFO and write one JSON line for the request, then close it (close flushes).
        # Raises BrokenPipeError if the server pipe cannot be opened, and RuntimeError if the serialized request exceeds PIPE_BUF (needed for atomic writes).
        try:
            file = self._open_non_blocking()
        except Exception as e:
            raise BrokenPipeError(
                "Could not send the FileTimerRequest because FileTimerServer is not available."
            ) from e
        with file:
            json_request = request.to_json()
            # Write request with no greater than select.PIPE_BUF is guarantee to be atomic.
            if len(json_request) > select.PIPE_BUF:
                raise RuntimeError(
                    f"FileTimerRequest larger than {select.PIPE_BUF} bytes "
                    f"is not supported: {json_request}"
                )
            file.write(json_request + "\n")

    def acquire(self, scope_id: str, expiration_time: float) -> None:
        # [FUNCTION DOC] Send a request registering a timer for the current process pid, scope_id, expiration_time and this client's signal. May raise BrokenPipeError or RuntimeError (see _send_request).
        self._send_request(
            request=FileTimerRequest(
                worker_pid=os.getpid(),
                scope_id=scope_id,
                expiration_time=expiration_time,
                signal=self.signal,
            ),
        )

    def release(self, scope_id: str) -> None:
        # [FUNCTION DOC] Send a release request (expiration_time of minus one, signal zero) for the current pid and scope_id.
        self._send_request(
            request=FileTimerRequest(
                worker_pid=os.getpid(), scope_id=scope_id, expiration_time=-1, signal=0
            ),
        )


class FileTimerServer:
    # [CLASS DOC] Agent-side server for FileTimerClient. Creates a FIFO at file_path, runs a watchdog thread that reads requests, tracks timers and sends the requested signal to workers whose timers expired.
    # State: _timers (mapping of (pid, scope_id) to request), _run_id, _max_interval, _daemon, _watchdog_thread, _stop_signaled, _log_event callback (no-op by default), _last_progress_time, _is_client_started, plus test-only _request_count and _run_once. Key methods: start, stop, run_once, register_timers, clear_timers, get_expired_timers, get_last_progress_time.
    """
    Server that works with ``FileTimerClient``. Clients are expected to be
    running on the same host as the process that is running this server.
    Each host in the job is expected to start its own timer server locally
    and each server instance manages timers for local workers (running on
    processes on the same host).

    Args:

        file_path: str, the path of a FIFO special file to be created.

        max_interval: float, max interval in seconds for each watchdog loop.

        daemon: bool, running the watchdog thread in daemon mode or not.
                      A daemon thread will not block a process to stop.
        log_event: Callable[[Dict[str, str]], None], an optional callback for
                logging the events in JSON format.
    """

    def __init__(
        self,
        file_path: str,
        run_id: str,
        max_interval: float = 10,
        daemon: bool = True,
        log_event: Callable[[str, FileTimerRequest | None], None] | None = None,
    ) -> None:
        # [FUNCTION DOC] Store configuration and initial state, delete any existing file at file_path and create a fresh FIFO there (side effect on the filesystem). log_event is an optional callback taking an event name and an optional request. Raises OSError if the FIFO cannot be created.
        # # KNOWN ISSUE: the class docstring does not document the run_id argument.
        self._file_path = file_path
        self._run_id = run_id
        self._max_interval = max_interval
        self._daemon = daemon
        self._timers: dict[tuple[int, str], FileTimerRequest] = {}
        self._stop_signaled = False
        self._watchdog_thread: threading.Thread | None = None

        self._is_client_started = False
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        os.mkfifo(self._file_path)
        # For test only. Count the number of requests received.
        self._request_count = 0
        # For test only. Process all requests and stop the server.
        self._run_once = False
        self._log_event = (
            log_event if log_event is not None else lambda name, request: None
        )
        self._last_progress_time = int(time.time())

    def start(self) -> None:
        # [FUNCTION DOC] Start the watchdog thread (daemon per constructor) and emit a "watchdog started" event. Not guarded against double start.
        logger.info(
            "Starting %s... max_interval=%s, daemon=%s, file_path=%s",
            type(self).__name__,
            self._max_interval,
            self._daemon,
            self._file_path,
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=self._daemon
        )
        logger.info("Starting watchdog thread...")
        self._watchdog_thread.start()
        self._log_event("watchdog started", None)

    def stop(self) -> None:
        # [FUNCTION DOC] Signal the watchdog to stop, join the thread for at most max_interval seconds, remove the FIFO file if present, and emit a "watchdog stopped" event.
        logger.info("Stopping %s", type(self).__name__)
        self._stop_signaled = True
        if self._watchdog_thread:
            logger.info("Stopping watchdog thread...")
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            logger.info("No watchdog thread running, doing nothing")
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        self._log_event("watchdog stopped", None)

    def run_once(self) -> None:
        # [FUNCTION DOC] Test helper: set the run-once flag so the watchdog processes pending requests and exits, join the thread (unbounded wait), and remove the FIFO file. Logs when no thread is running.
        self._run_once = True
        if self._watchdog_thread:
            logger.info("Stopping watchdog thread...")
            self._watchdog_thread.join()
            self._watchdog_thread = None
        else:
            logger.info("No watchdog thread running, doing nothing")
        if os.path.exists(self._file_path):
            os.remove(self._file_path)

    @staticmethod
    def is_process_running(pid: int):
        # [FUNCTION DOC] Static: return True if a process with the pid exists and can be signaled (probe with signal zero), False on OSError.
        """
        function to check process is running or not
        """
        try:
            # Check if the process exists and we can send signals to it
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _watchdog_loop(self) -> None:
        # [FUNCTION DOC] HELPER: Thread body. Opens the FIFO for reading (blocks until a client opens it), marks the client as started, and repeatedly runs a watchdog pass until stop is signaled or, in run-once mode, one pass is done. Updates the last progress time after each successful pass and logs per-pass exceptions. Logs and re-raises if the pipe cannot be opened.
        # Open the pipe in blocking mode blocks the server thread.
        # This is fine for the following reasons:
        #  1. No client case usually does not happen.
        #  2. We are running the watchdog loop in a separate daemon
        #     thread, which will not block the process to stop.
        try:
            with open(self._file_path) as fd:
                self._is_client_started = True
                while not self._stop_signaled:
                    try:
                        run_once = self._run_once
                        self._run_watchdog(fd)
                        if run_once:
                            break
                        self._last_progress_time = int(time.time())
                    except Exception:
                        logger.exception("Error running watchdog")

        except Exception:
            logger.exception("Could not open the FileTimerServer pipe")
            raise

    def _run_watchdog(self, fd: io.TextIOWrapper) -> None:
        # [FUNCTION DOC] HELPER: One watchdog pass. Reads requests from the pipe, registers them, finds timers expired as of now, logs debug info and sorts them by expiration per worker.
        # For each worker picks the first expired timer with a positive signal; without one the worker is not killed. Otherwise sends that signal; on success emits a "kill worker process" event. Every worker with expired timers has its timers cleared afterwards, even if the kill failed (the log message says retry, but they are cleared).
        # If any worker was killed, the server process itself is also sent the same signal.
        # # KNOWN ISSUE: reaped_worker_pids includes workers whose reap failed, contradicting the retry-on-next-watchdog log message.
        timer_requests = self._get_requests(fd, self._max_interval)
        self.register_timers(timer_requests)
        now = time.time()
        reaped_worker_pids = set()
        kill_process = False
        reap_signal = 0

        all_expired_timers = self.get_expired_timers(now)
        log_debug_info_for_expired_timers(
            self._run_id,
            {
                pid: [expired_timer.to_json() for expired_timer in expired_timers]
                for pid, expired_timers in all_expired_timers.items()
            },
        )

        for worker_pid, expired_timers in all_expired_timers.items():
            logger.info(
                "Reaping worker_pid=[%s]. Expired timers: %s",
                worker_pid,
                self._get_scopes(expired_timers),
            )
            reaped_worker_pids.add(worker_pid)
            # In case we have multiple expired timers, we find the first timer
            # with a valid signal (>0) in the expiration time order.
            expired_timers.sort(key=lambda timer: timer.expiration_time)
            signal = 0
            expired_timer = None
            for timer in expired_timers:
                self._log_event("timer expired", timer)
                if timer.signal > 0:
                    signal = timer.signal
                    expired_timer = timer
                    break
            if signal <= 0:
                logger.info(
                    "No signal specified with worker=[%s]. Do not reap it.", worker_pid
                )
                continue
            if self._reap_worker(worker_pid, signal):
                logger.info(
                    "Successfully reaped worker=[%s] with signal=%s", worker_pid, signal
                )
                self._log_event("kill worker process", expired_timer)
                kill_process = True
                reap_signal = signal
            else:
                logger.error(
                    "Error reaping worker=[%s]. Will retry on next watchdog.",
                    worker_pid,
                )
        if kill_process and reap_signal > 0:
            logger.info(
                "Terminating the server process=[%s] because of expired timers",
                os.getpid(),
            )
            self._reap_worker(os.getpid(), reap_signal)

        self.clear_timers(reaped_worker_pids)

    def _get_scopes(self, timer_requests: list[FileTimerRequest]) -> list[str]:
        # [FUNCTION DOC] HELPER: Return the scope ids of the given requests as a list.
        return [r.scope_id for r in timer_requests]

    def _get_requests(
        self, fd: io.TextIOWrapper, max_interval: float
    ) -> list[FileTimerRequest]:
        # [FUNCTION DOC] HELPER: Read JSON lines from the FIFO and return parsed FileTimerRequest objects. Loops until stop is signaled (or run-once reached end of input) or max_interval seconds have elapsed. Sleeps up to one second on end-of-file; a malformed line raises a JSON decoding error.
        start = time.time()
        requests = []
        while not self._stop_signaled or self._run_once:
            # For named pipe, readline() is blocking when at least one writer opens.
            # It returns only when flush() is called at the writer side.
            # Note that flush() is automatically called inside close().
            # After the last writer closes, readline() is not blocking.
            # It will return an empty string when it's at end-of-file.
            # Since the client side always opens the pipe, writes a message and closes
            # the pipe immediately, the readline() call below is not blocking for long.
            json_request = fd.readline()
            if len(json_request) == 0:
                if self._run_once:
                    break
                time.sleep(min(max_interval, 1))
            else:
                request = json.loads(json_request)
                pid = request["pid"]
                scope_id = request["scope_id"]
                expiration_time = request["expiration_time"]
                signal = request["signal"]
                requests.append(
                    FileTimerRequest(
                        worker_pid=pid,
                        scope_id=scope_id,
                        expiration_time=expiration_time,
                        signal=signal,
                    )
                )
            now = time.time()
            if now - start > max_interval:
                break
        return requests

    def register_timers(self, timer_requests: list[FileTimerRequest]) -> None:
        # [FUNCTION DOC] Update the timer table from requests: a negative expiration_time removes the (pid, scope_id) entry if present, otherwise stores/replaces it. Increments the received request counter for each.
        for request in timer_requests:
            pid = request.worker_pid
            scope_id = request.scope_id
            expiration_time = request.expiration_time
            self._request_count += 1

            key = (pid, scope_id)
            # negative expiration is a proxy for a release call
            if expiration_time < 0:
                if key in self._timers:
                    del self._timers[key]
            else:
                self._timers[key] = request

    def clear_timers(self, worker_pids: set[int]) -> None:
        # [FUNCTION DOC] Remove all timers whose pid is in worker_pids or whose process is no longer running.
        for pid, scope_id in list(self._timers.keys()):
            if pid in worker_pids or not FileTimerServer.is_process_running(pid):
                del self._timers[(pid, scope_id)]

    def get_expired_timers(self, deadline: float) -> dict[int, list[FileTimerRequest]]:
        # [FUNCTION DOC] Return a mapping of pid to the list of timers with expiration_time at or before deadline. Does not remove them.
        # pid -> [timer_requests...]
        expired_timers: dict[int, list[FileTimerRequest]] = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_pid, [])
                expired_scopes.append(request)
        return expired_timers

    def _reap_worker(self, worker_pid: int, signal: int) -> bool:
        # [FUNCTION DOC] HELPER: Send the signal to the pid. Returns True on success or if the process does not exist; logs and returns False on any other error.
        try:
            os.kill(worker_pid, signal)
            return True
        except ProcessLookupError:
            logger.info("Process with pid=%s does not exist. Skipping", worker_pid)
            return True
        except Exception:
            logger.exception("Error terminating pid=%s", worker_pid)
        return False

    def get_last_progress_time(self) -> int:
        return self._last_progress_time if self._is_client_started else int(time.time())
        # [FUNCTION DOC] Return the time of the last successful watchdog pass as epoch seconds once a client has started; before that, return the current time.
