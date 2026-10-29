# mypy: allow-untyped-defs
# [MODULE DOC] Multiprocessing-queue based timer implementation: LocalTimerClient (worker side), MultiprocessingRequestQueue (adapter over an mp queue) and LocalTimerServer (reaps expired workers with SIGKILL). Creates a module logger at import; no other side effects.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import multiprocessing as mp
import os
import signal
import time
from queue import Empty
from typing import Any

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer


__all__ = ["LocalTimerClient", "MultiprocessingRequestQueue", "LocalTimerServer"]

logger = logging.getLogger(__name__)


class LocalTimerClient(TimerClient):
    # [CLASS DOC] Worker-side TimerClient that puts TimerRequest objects, identified by the current os pid, onto a shared multiprocessing queue read by LocalTimerServer on the same host. State: _mp_queue. Key methods: acquire, release.
    """
    Client side of ``LocalTimerServer``. This client is meant to be used
    on the same host that the ``LocalTimerServer`` is running on and uses
    pid to uniquely identify a worker. This is particularly useful in situations
    where one spawns a subprocess (trainer) per GPU on a host with multiple
    GPU devices.
    """

    def __init__(self, mp_queue):
        # [FUNCTION DOC] Store the multiprocessing queue used to send requests.
        super().__init__()
        self._mp_queue = mp_queue

    def acquire(self, scope_id, expiration_time):
        # [FUNCTION DOC] Enqueue an acquire TimerRequest for the current pid with scope_id and expiration_time.
        pid = os.getpid()
        acquire_request = TimerRequest(pid, scope_id, expiration_time)
        self._mp_queue.put(acquire_request)

    def release(self, scope_id):
        # [FUNCTION DOC] Enqueue a release TimerRequest (expiration_time of minus one) for the current pid and scope_id.
        pid = os.getpid()
        release_request = TimerRequest(pid, scope_id, -1)
        self._mp_queue.put(release_request)


class MultiprocessingRequestQueue(RequestQueue):
    # [CLASS DOC] RequestQueue implementation backed by a multiprocessing queue. State: _mp_queue. Key methods: size, get.
    """
    A ``RequestQueue`` backed by python ``multiprocessing.Queue``
    """

    def __init__(self, mp_queue: mp.Queue):
        # [FUNCTION DOC] Store the multiprocessing queue.
        super().__init__()
        self._mp_queue = mp_queue

    def size(self) -> int:
        # [FUNCTION DOC] Return the approximate number of queued requests (the queue's qsize).
        return self._mp_queue.qsize()

    def get(self, size, timeout: float) -> list[TimerRequest]:
        # [FUNCTION DOC] Fetch up to size requests, blocking per item within a shrinking time budget of timeout seconds. Stops early when the queue is empty for the remaining wait or the budget is used up. Returns the list of requests collected (possibly empty).
        requests = []
        wait = timeout
        for _ in range(size):
            start = time.time()

            try:
                r = self._mp_queue.get(block=True, timeout=wait)
            except Empty:
                break

            requests.append(r)
            wait = wait - (time.time() - start)
            if wait <= 0:
                break

        return requests


class LocalTimerServer(TimerServer):
    # [CLASS DOC] TimerServer for LocalTimerClient workers that are subprocesses of the process running this server. State: _timers, a mapping of (worker pid, scope_id) to the latest TimerRequest. Reaping sends SIGKILL to the pid. Key methods: register_timers, clear_timers, get_expired_timers.
    """
    Server that works with ``LocalTimerClient``. Clients are expected to be
    subprocesses to the parent process that is running this server. Each host
    in the job is expected to start its own timer server locally and each
    server instance manages timers for local workers (running on processes
    on the same host).
    """

    def __init__(
        self, mp_queue: mp.Queue, max_interval: float = 60, daemon: bool = True
    ):
        # [FUNCTION DOC] Wrap the mp queue in a MultiprocessingRequestQueue, initialize the base server (max_interval default 60 seconds, daemon default True) and an empty timer table.
        super().__init__(MultiprocessingRequestQueue(mp_queue), max_interval, daemon)
        self._timers: dict[tuple[Any, str], TimerRequest] = {}

    def register_timers(self, timer_requests: list[TimerRequest]) -> None:
        # [FUNCTION DOC] For each request: a negative expiration_time removes the (pid, scope_id) timer if present, otherwise stores or replaces it.
        for request in timer_requests:
            pid = request.worker_id
            scope_id = request.scope_id
            expiration_time = request.expiration_time

            # negative expiration is a proxy for a release call
            if expiration_time < 0:
                self._timers.pop((pid, scope_id), None)
            else:
                self._timers[(pid, scope_id)] = request

    def clear_timers(self, worker_ids: set[int]) -> None:
        # [FUNCTION DOC] Remove all timers belonging to the given worker ids.
        for pid, scope_id in list(self._timers.keys()):
            if pid in worker_ids:
                self._timers.pop((pid, scope_id))

    def get_expired_timers(self, deadline: float) -> dict[Any, list[TimerRequest]]:
        # [FUNCTION DOC] Return a mapping from worker id to the list of its timers with expiration_time at or before deadline. Does not modify the table.
        # pid -> [timer_requests...]
        expired_timers: dict[Any, list[TimerRequest]] = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_id, [])
                expired_scopes.append(request)
        return expired_timers

    def _reap_worker(self, worker_id: int) -> bool:
        # [FUNCTION DOC] HELPER: Send SIGKILL to the worker pid. Returns True on success or if the process does not exist; logs and returns False on other errors.
        try:
            os.kill(worker_id, signal.SIGKILL)
            return True
        except ProcessLookupError:
            logger.info("Process with pid=%s does not exist. Skipping", worker_id)
            return True
        except Exception:
            logger.exception("Error terminating pid=%s", worker_id)
        return False
