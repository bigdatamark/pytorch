# mypy: allow-untyped-defs
# [MODULE DOC] Abstract API for expiration timers: TimerRequest data object, TimerClient and RequestQueue and TimerServer abstract classes, plus the module-level configure() and expires() context manager used by worker code.
# Holds one module-global timer client set by configure(). No other import-time side effects.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any


__all__ = [
    "TimerRequest",
    "TimerClient",
    "RequestQueue",
    "TimerServer",
    "configure",
    "expires",
]

logger = logging.getLogger(__name__)


class TimerRequest:
    # [CLASS DOC] Value object describing a timer acquire or release sent from a TimerClient to a TimerServer.
    # Attributes (slots): worker_id (implementation specific identifier), scope_id (string naming the guarded scope), expiration_time (absolute epoch seconds; a negative value means release). Equality compares all three fields.
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``TimerClient`` and ``TimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.

    .. note:: the type of ``worker_id`` is implementation specific.
              It is whatever the TimerServer and TimerClient implementations
              have on to uniquely identify a worker.
    """

    __slots__ = ["worker_id", "scope_id", "expiration_time"]

    def __init__(self, worker_id: Any, scope_id: str, expiration_time: float):
        # [FUNCTION DOC] Store worker_id, scope_id and expiration_time on the request.
        self.worker_id = worker_id
        self.scope_id = scope_id
        self.expiration_time = expiration_time

    def __eq__(self, other):
        # [FUNCTION DOC] Equality: True only if other is a TimerRequest with identical worker_id, scope_id and expiration_time; False for any other type.
        if isinstance(other, TimerRequest):
            return (
                self.worker_id == other.worker_id
                and self.scope_id == other.scope_id
                and self.expiration_time == other.expiration_time
            )
        return False


class TimerClient(abc.ABC):
    # [CLASS DOC] Abstract client used by a worker to acquire and release countdown timers by talking to a TimerServer. Subclasses implement acquire and release.
    """
    Client library to acquire and release countdown timers by communicating
    with the TimerServer.
    """

    @abc.abstractmethod
    def acquire(self, scope_id: str, expiration_time: float) -> None:
        # [FUNCTION DOC] Abstract: register a timer for this client's worker for scope_id expiring at the absolute time expiration_time (typically by sending a request to the server).
        """
        Acquires a timer for the worker that holds this client object
        given the scope_id and expiration_time. Typically registers
        the timer with the TimerServer.
        """

    @abc.abstractmethod
    def release(self, scope_id: str):
        # [FUNCTION DOC] Abstract: cancel the timer for scope_id on this client's worker so it no longer expires.
        """
        Releases the timer for the ``scope_id`` on the worker this
        client represents. After this method is
        called, the countdown timer on the scope is no longer in effect.
        """


class RequestQueue(abc.ABC):
    # [CLASS DOC] Abstract consumer-side queue of TimerRequest objects used by a TimerServer. Subclasses implement size and get.
    """
    Consumer queue holding timer acquisition/release requests
    """

    @abc.abstractmethod
    def size(self) -> int:
        # [FUNCTION DOC] Abstract: return the current number of queued requests. The size must not decrease until get is called, so a following get of that size returns at least that many (up to twice that many when asked for double).
        """
        Returns the size of the queue at the time this method is called.
        Note that by the time ``get`` is called the size of the queue
        may have increased. The size of the queue should not decrease
        until the ``get`` method is called. That is, the following assertion
        should hold:

        size = q.size()
        res = q.get(size, timeout=0)
        assert size == len(res)

        -- or --

        size = q.size()
        res = q.get(size * 2, timeout=1)
        assert size <= len(res) <= size * 2
        """

    @abc.abstractmethod
    def get(self, size: int, timeout: float) -> list[TimerRequest]:
        # [FUNCTION DOC] Abstract: block for up to timeout seconds and return up to size TimerRequest objects as a list.
        """
        Gets up to ``size`` number of timer requests in a blocking fashion
        (no more than ``timeout`` seconds).
        """


class TimerServer(abc.ABC):
    # [CLASS DOC] Abstract server that runs a watchdog thread which drains the request queue, registers timers, finds expired ones and reaps the owning workers.
    # State: _request_queue, _max_interval (max wait per queue read), _daemon flag, _watchdog_thread (None when stopped) and _stop_signaled flag. Subclasses implement register_timers, clear_timers, get_expired_timers and _reap_worker. Key methods: start, stop.
    """
    Entity that monitors active timers and expires them
    in a timely fashion. This server is responsible for
    reaping workers that have expired timers.
    """

    def __init__(
        self, request_queue: RequestQueue, max_interval: float, daemon: bool = True
    ):
        # [FUNCTION DOC] Store the request queue, max_interval (seconds to wait for queue items) and daemon flag; no thread is started until start() is called.
        """
        :param request_queue: Consumer ``RequestQueue``
        :param max_interval: max time (in seconds) to wait
                             for an item in the request_queue
        :param daemon: whether to run the watchdog thread as a daemon
        """
        super().__init__()
        self._request_queue = request_queue
        self._max_interval = max_interval
        self._daemon = daemon
        self._watchdog_thread: threading.Thread | None = None
        self._stop_signaled = False

    @abc.abstractmethod
    def register_timers(self, timer_requests: list[TimerRequest]) -> None:
        # [FUNCTION DOC] Abstract: process a list of TimerRequest objects, registering acquires and removing timers for releases (negative expiration_time).
        """
        Processes the incoming timer requests and registers them with the server.
        The timer request can either be an acquire-timer or release-timer request.
        Timer requests with a negative expiration_time should be interpreted
        as a release-timer request.
        """

    @abc.abstractmethod
    def clear_timers(self, worker_ids: set[Any]) -> None:
        # [FUNCTION DOC] Abstract: drop all timers belonging to the given set of worker ids.
        """
        Clears all timers for the given ``worker_ids``.
        """

    @abc.abstractmethod
    def get_expired_timers(self, deadline: float) -> dict[str, list[TimerRequest]]:
        # [FUNCTION DOC] Abstract: return a mapping from worker id to the list of its timers whose expiration_time is at or before the given deadline.
        """
        Returns all expired timers for each worker_id. An expired timer
        is a timer for which the expiration_time is less than or equal to
        the provided deadline.
        """

    @abc.abstractmethod
    def _reap_worker(self, worker_id: Any) -> bool:
        # [FUNCTION DOC] HELPER: Abstract: kill/reap the worker. Returns True on success and False to retry on the next watchdog pass. An uncaught exception is treated by callers as reaped.
        """
        Reaps the given worker. Returns True if the worker has been
        successfully reaped, False otherwise. If any uncaught exception
        is thrown from this method, the worker is considered reaped
        and all associated timers will be removed.
        """

    def _reap_worker_no_throw(self, worker_id: Any) -> bool:
        # [FUNCTION DOC] HELPER: Call _reap_worker; log and swallow any exception, returning True (worker considered reaped) in that case. Otherwise return its result.
        """
        Wraps ``_reap_worker(worker_id)``, if an uncaught exception is
        thrown, then it considers the worker as reaped.
        """
        try:
            return self._reap_worker(worker_id)
        except Exception:
            logger.exception(
                "Uncaught exception thrown from _reap_worker(), "
                "check that the implementation correctly catches exceptions",
            )
            return True

    def _watchdog_loop(self):
        # [FUNCTION DOC] HELPER: Thread body. Repeatedly runs one watchdog pass until stop is signaled, logging (not propagating) any exception from a pass.
        while not self._stop_signaled:
            try:
                self._run_watchdog()
            except Exception:
                logger.exception("Error running watchdog")

    def _run_watchdog(self):
        # [FUNCTION DOC] HELPER: One watchdog pass. Reads a batch (at least one) of requests from the queue waiting up to max_interval, registers them, then for each worker with timers expired as of now tries to reap it (logging the outcome), and finally clears timers of successfully reaped workers. Workers that fail to reap are retried next pass.
        batch_size = max(1, self._request_queue.size())
        timer_requests = self._request_queue.get(batch_size, self._max_interval)
        self.register_timers(timer_requests)
        now = time.time()
        reaped_worker_ids = set()
        for worker_id, expired_timers in self.get_expired_timers(now).items():
            logger.info(
                "Reaping worker_id=[%s]. Expired timers: %s",
                worker_id,
                self._get_scopes(expired_timers),
            )
            if self._reap_worker_no_throw(worker_id):
                logger.info("Successfully reaped worker=[%s]", worker_id)
                reaped_worker_ids.add(worker_id)
            else:
                logger.error(
                    "Error reaping worker=[%s]. Will retry on next watchdog.", worker_id
                )
        self.clear_timers(reaped_worker_ids)

    def _get_scopes(self, timer_requests):
        # [FUNCTION DOC] HELPER: Return the scope ids of the given timer requests as a list (for logging).
        return [r.scope_id for r in timer_requests]

    def start(self) -> None:
        # [FUNCTION DOC] Log and start the watchdog thread (daemon per constructor). Does not guard against being started twice.
        logger.info(
            "Starting %s... max_interval=%s, daemon=%s",
            type(self).__name__,
            self._max_interval,
            self._daemon,
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=self._daemon
        )
        logger.info("Starting watchdog thread...")
        self._watchdog_thread.start()

    def stop(self) -> None:
        # [FUNCTION DOC] Signal the watchdog loop to stop and join the thread for at most max_interval seconds, then clear the thread reference. Logs and does nothing else if no thread is running.
        logger.info("Stopping %s", type(self).__name__)
        self._stop_signaled = True
        if self._watchdog_thread:
            logger.info("Stopping watchdog thread...")
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            logger.info("No watchdog thread running, doing nothing")


_timer_client: TimerClient | None = None


def configure(timer_client: TimerClient):
    # [FUNCTION DOC] Set the module-global timer client used by expires() when no explicit client is passed. Logs the client class name.
    """
    Configures a timer client. Must be called before using ``expires``.
    """
    global _timer_client
    _timer_client = timer_client
    logger.info("Timer client configured to: %s", type(_timer_client).__name__)


@contextmanager
def expires(after: float, scope: str | None = None, client: TimerClient | None = None):
    # [FUNCTION DOC] Context manager that acquires a timer for the guarded block and releases it on exit (also on exception).
    # after is seconds from now; scope defaults to "filename#lineno" of the caller; client defaults to the configured global client.
    # Raises RuntimeError if no client is given and none was configured. The worker becomes eligible for reaping after expiry; the server decides when.
    """
    Acquires a countdown timer that expires in ``after`` seconds from now,
    unless the code-block that it wraps is finished within the timeframe.
    When the timer expires, this worker is eligible to be reaped. The
    exact meaning of "reaped" depends on the client implementation. In
    most cases, reaping means to terminate the worker process.
    Note that the worker is NOT guaranteed to be reaped at exactly
    ``time.now() + after``, but rather the worker is "eligible" for being
    reaped and the ``TimerServer`` that the client talks to will ultimately
    make the decision when and how to reap the workers with expired timers.

    Usage::

        torch.distributed.elastic.timer.configure(LocalTimerClient())
        with expires(after=10):
            torch.distributed.all_reduce(...)
    """
    if client is None:
        if _timer_client is None:
            raise RuntimeError("Configure timer client before using countdown timers.")
        client = _timer_client
    if scope is None:
        # grab the caller file + lineno
        caller = getframeinfo(stack()[1][0])
        scope = f"{caller.filename}#{caller.lineno}"
    expiration = time.time() + after
    client.acquire(scope, expiration)
    try:
        yield
    finally:
        client.release(scope)
