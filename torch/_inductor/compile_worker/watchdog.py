# [MODULE DOC] Shared-memory heartbeat used by the compile worker sidecar watchdog to report which phase each worker is in.
# Holds module globals for the shared array, slot counter and this worker's slot, with sidecar-side (create, reset,
# read_heartbeats) and worker-side (init_worker_slot, set_current_job, report_phase, clear_current_job) functions.
"""Shared-memory heartbeat so the compile-worker sidecar's watchdog can report
which phase a slow/stuck worker is in (see SubprocMain._watchdog_loop).

A single multiprocessing.Array is allocated in the sidecar (create()) before it
forks its worker pool; every forked worker inherits the same buffer and writes
only its own slot -- single writer per slot, so no lock is needed there and a
rare torn read is harmless for a coarse once-a-minute diagnostic. The sidecar's
watchdog thread reads all slots.

Slots are recycled off the pool's lifecycle rather than by polling worker
liveness: the sidecar calls reset() every time it (re)builds the worker pool,
and workers then claim slots 0..nprocs-1 with a per-generation counter. A worker
death tears down and rebuilds the whole pool (ProcessPoolExecutor's behavior), so
reset() runs before the replacement generation forks and there is never more than
one live worker per slot. (If the pool were ever configured to replace workers
individually -- e.g. max_tasks_per_child -- a replacement would simply run out of
counter and report nothing, rather than collide with a live worker's slot.)

Everything is a no-op unless the sidecar allocated the buffer and the worker
claimed a slot, so the report_phase() calls sprinkled through the compile path
are free in the main process and in non-forking pools (spawn workers do not
inherit the buffer and simply degrade to duration-only reporting).
"""

from __future__ import annotations

import multiprocessing
import os
import time
from enum import IntEnum
from typing import Any


class Phase(IntEnum):
    # [CLASS DOC] Integer enum of compile phases (RUNNING, QUERYING_CACHE, COMPILING) stored in the shared buffer;
    # values must stay stable, append new ones only.
    # Persisted as ints in the shared buffer; append new phases, don't renumber.
    RUNNING = 0
    QUERYING_CACHE = 1
    COMPILING = 2


# Per-slot layout: _FIELDS longs.
_FIELDS = 4
_F_JOB_ID = 0
_F_PHASE = 1
_F_PHASE_START_NS = 2
_F_PID = 3
_EMPTY = -1  # job_id sentinel for an idle slot

# Allocated in the sidecar (create) before forking; inherited by fork workers.
# _counter's lock also serializes slot claims and reset().
_heartbeat: Any | None = None
_counter: Any | None = None
_nprocs: int = 0
# Claimed post-fork by each worker (init_worker_slot).
_slot: int | None = None


def create(nprocs: int) -> None:
    # [FUNCTION DOC] Sidecar side: allocate the shared heartbeat array (nprocs slots of several 64-bit fields, all marked idle)
    # and the shared slot counter using a fork multiprocessing context. Sets module globals. Must run before forking workers.
    """Sidecar: allocate the shared buffer before forking the worker pool."""
    global _heartbeat, _counter, _nprocs
    ctx = multiprocessing.get_context("fork")
    _nprocs = nprocs
    _counter = ctx.Value("i", 0)
    buf = ctx.Array("q", nprocs * _FIELDS, lock=False)
    for s in range(nprocs):
        buf[s * _FIELDS + _F_JOB_ID] = _EMPTY
    _heartbeat = buf


def reset() -> None:
    # [FUNCTION DOC] Sidecar side: under the counter lock, reset the slot counter to zero and mark every slot idle with pid 0.
    # No-op if the buffer was not created.
    """Sidecar: recycle all slots for a fresh worker generation. Call before
    (re)building the worker pool so the new workers claim clean slots 0..n-1."""
    if _heartbeat is None or _counter is None:
        return
    with _counter.get_lock():
        _counter.value = 0
        for s in range(_nprocs):
            base = s * _FIELDS
            _heartbeat[base + _F_JOB_ID] = _EMPTY
            _heartbeat[base + _F_PID] = 0


def init_worker_slot() -> None:
    # [FUNCTION DOC] Worker side: atomically take the next slot number from the shared counter and, if it is below nprocs,
    # remember it and write this process pid into the slot. No-op without a buffer.
    """Worker: claim this generation's next slot. No-op if the sidecar didn't
    allocate a buffer (main process / spawn pools)."""
    global _slot
    if _heartbeat is None or _counter is None or _nprocs <= 0:
        return
    with _counter.get_lock():
        slot = _counter.value
        _counter.value = slot + 1
    # Only the first nprocs workers of a generation get a slot; any beyond that
    # (which shouldn't happen without per-worker replacement) report nothing.
    if slot < _nprocs:
        _slot = slot
        _heartbeat[slot * _FIELDS + _F_PID] = os.getpid()


def set_current_job(job_id: int) -> None:
    # [FUNCTION DOC] Worker side: reset the slot phase to RUNNING, stamp the phase start time, and finally write the job id
    # so readers see a populated slot. No-op without a claimed slot.
    """Worker: mark the start of a job in this worker's slot."""
    if _heartbeat is None or _slot is None:
        return
    base = _slot * _FIELDS
    _heartbeat[base + _F_PHASE] = int(Phase.RUNNING)
    _heartbeat[base + _F_PHASE_START_NS] = time.monotonic_ns()
    # Written last so a concurrent reader sees a fully-populated slot or none.
    _heartbeat[base + _F_JOB_ID] = job_id


def report_phase(phase: Phase) -> None:
    # [FUNCTION DOC] Worker side: stamp the phase start time and store the new phase in this worker's slot. No-op without a slot.
    """Worker: record the phase this worker just entered."""
    if _heartbeat is None or _slot is None:
        return
    base = _slot * _FIELDS
    _heartbeat[base + _F_PHASE_START_NS] = time.monotonic_ns()
    _heartbeat[base + _F_PHASE] = int(phase)


def clear_current_job() -> None:
    # [FUNCTION DOC] Worker side: mark this worker's slot idle by writing the empty sentinel to the job id field.
    """Worker: mark this slot idle when its job finishes."""
    if _heartbeat is None or _slot is None:
        return
    _heartbeat[_slot * _FIELDS + _F_JOB_ID] = _EMPTY


def enabled() -> bool:
    # [FUNCTION DOC] Return True if the shared heartbeat buffer exists (fork pools only).
    """Sidecar: whether phase heartbeats are active (fork pools only)."""
    return _heartbeat is not None


def read_heartbeats() -> dict[int, tuple[int, int, int]]:
    # [FUNCTION DOC] Sidecar side: scan all slots and return a mapping from job id to a tuple of (phase, phase start in monotonic ns,
    # worker pid) for busy slots; empty mapping if disabled. Reads are unlocked, so a torn read is possible but harmless.
    """Sidecar: job_id -> (phase, phase_start_ns, pid) for currently-busy slots.

    phase_start_ns is worker monotonic_ns; on Linux CLOCK_MONOTONIC is
    system-wide so the sidecar can diff it against its own monotonic_ns.
    """
    if _heartbeat is None:
        return {}
    out: dict[int, tuple[int, int, int]] = {}
    for s in range(_nprocs):
        base = s * _FIELDS
        job_id = _heartbeat[base + _F_JOB_ID]
        if job_id != _EMPTY:
            out[job_id] = (
                _heartbeat[base + _F_PHASE],
                _heartbeat[base + _F_PHASE_START_NS],
                _heartbeat[base + _F_PID],
            )
    return out
