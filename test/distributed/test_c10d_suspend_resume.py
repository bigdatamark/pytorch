# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the c10d memory offload API (Backend.suspend /
# resume / memory_stats). Parameterized over backends; the entries are
# NCCL-based (the API needs communicator memory offload support, NCCL
# 2.29.7+), but other backends can be enabled by extending
# SUSPEND_RESUME_BACKENDS.
# [MODULE DOC] Two-process tests of the backend memory offload API (suspend, resume and
# memory_stats). Test classes are generated at import time, one per entry in
# SUSPEND_RESUME_BACKENDS (backend name, device type, whether to also create a point-to-point
# channel), and added to the module namespace; each is skipped unless the backend is available
# and, for cuda, two or more GPUs and NCCL 2.29.7 or newer are present. Import-time side effect:
# exits the process with status 0 if torch.distributed is unavailable. Running as a script calls
# run_tests.

import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


SUSPEND_RESUME_BACKENDS = [
    ("nccl-legacy", "cuda", False),
    ("nccl2", "cuda", False),
    ("nccl-lazy", "cuda", True),
]


def _has_nccl_offload():
    # [FUNCTION DOC] HELPER: Returns true if the NCCL backend is available and the NCCL version is
    # at least 2.29.7; any exception while checking yields false.
    try:
        return dist.is_nccl_available() and torch.cuda.nccl.version() >= (2, 29, 7)
    except Exception:
        return False


class AbstractSuspendResumeTest:
    # [CLASS DOC] HELPER: Mixin with the test logic, combined with MultiProcessTestCase by the
    # class factory. The factory sets the class attributes backend_name, device_type and
    # create_pair_channel. Runs 2 ranks, one per cuda device; each test initializes the group,
    # then exercises memory_stats, suspend and resume on the backend object.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property: the world size is 2.
        return 2

    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Property returning the torch device for this rank, built from the
        # device type and rank index.
        return torch.device(f"{self.device_type}:{self.rank}")

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Destroys the process group if one is initialized, runs the base
        # tearDown, then removes the store file, ignoring OSError.
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _exchange_with_peer(self):
        # [FUNCTION DOC] HELPER: Two ranks swap a one-element tensor holding their rank through
        # blocking send and recv, rank 0 sending first and rank 1 receiving first. Synchronizes
        # cuda and asserts the received value equals the peer's rank. Creates the point-to-point
        # channel as a side effect.
        send_tensor = torch.full((1,), float(self.rank), device=self.device)
        recv_tensor = torch.empty(1, device=self.device)
        peer = 1 - self.rank
        if self.rank == 0:
            dist.send(send_tensor, peer)
            dist.recv(recv_tensor, peer)
        else:
            dist.recv(recv_tensor, peer)
            dist.send(send_tensor, peer)
        torch.cuda.synchronize()
        self.assertEqual(recv_tensor, torch.full_like(recv_tensor, float(peer)))

    def _init_pg(self):
        # [FUNCTION DOC] HELPER: Selects the cuda device for the rank when the device type is
        # cuda, builds a file store, and initializes the default group with the backend name and a
        # 60 second timeout. Saves the backend object for the device and runs a large all-reduce
        # (64M floats) so the communicator allocates suspendable memory. If a pair channel is
        # requested, does the peer exchange and asserts exactly one active channel.
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        self.backend = dist.get_backend_impl(device=self.device)
        # Run a large collective so the communicator allocates internal
        # (suspendable) memory.
        dist.all_reduce(torch.zeros(1024 * 1024 * 64, device=self.device))
        torch.cuda.synchronize()
        if self.create_pair_channel:
            self._exchange_with_peer()
            self.assertEqual(self.backend._num_active_channels(), 1)

    def test_memory_stats(self):
        # [FUNCTION DOC] Initializes the group and checks memory_stats returns a mapping with the
        # keys suspend, suspended, persist and total, and that nothing is suspended (the suspended
        # value is zero).
        self._init_pg()
        stats = self.backend.memory_stats()
        self.assertIsInstance(stats, dict)
        for key in ("suspend", "suspended", "persist", "total"):
            self.assertIn(key, stats)
        self.assertEqual(stats["suspended"], 0)

    def test_suspend(self):
        # [FUNCTION DOC] Initializes the group, suspends the backend and checks the suspended
        # count from memory_stats is 2 when a pair channel exists and 1 otherwise. Always resumes
        # in a finally block.
        self._init_pg()
        self.backend.suspend()
        try:
            stats = self.backend.memory_stats()
            expected = 2 if self.create_pair_channel else 1
            self.assertEqual(stats["suspended"], expected)
        finally:
            self.backend.resume()

    def test_suspend_resume_cycle(self):
        # [FUNCTION DOC] Initializes the group, suspends then resumes, and checks the suspended
        # count is zero. Then checks the communicator still works: an all-reduce of ones equals
        # the world size, and the peer exchange succeeds when a pair channel is used.
        self._init_pg()
        self.backend.suspend()
        self.backend.resume()
        stats = self.backend.memory_stats()
        self.assertEqual(stats["suspended"], 0)

        # The communicator must still work after a suspend/resume cycle.
        tensor = torch.ones(1024, device=self.device)
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        expected = torch.full((1024,), float(self.world_size), device=self.device)
        self.assertEqual(tensor, expected)
        if self.create_pair_channel:
            self._exchange_with_peer()


def _make_suspend_resume_test_class(backend_name, device_type, create_pair_channel):
    # [FUNCTION DOC] HELPER: Class factory. Builds a test class from the mixin and
    # MultiProcessTestCase, sets its backend name, device type and pair-channel flag, and names it
    # from the backend name in title case without hyphens followed by SuspendResumeTest. Wraps it
    # with a skip when the backend is not available and, for cuda, a skip when cuda is missing,
    # fewer than 2 devices exist or NCCL is older than 2.29.7. Returns the wrapped class.
    class SuspendResumeTest(AbstractSuspendResumeTest, MultiProcessTestCase):
        # [CLASS DOC] HELPER: Empty combination of the mixin and MultiProcessTestCase; attributes
        # are set by the enclosing factory.
        pass

    class_name = backend_name.replace("-", " ").title().replace(" ", "")
    class_name = f"{class_name}SuspendResumeTest"
    SuspendResumeTest.backend_name = backend_name
    SuspendResumeTest.device_type = device_type
    SuspendResumeTest.create_pair_channel = create_pair_channel
    SuspendResumeTest.__name__ = class_name
    SuspendResumeTest.__qualname__ = SuspendResumeTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(SuspendResumeTest)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2 or not _has_nccl_offload(),
            "suspend/resume requires 2+ GPUs and NCCL 2.29.7+",
        )(cls)
    return cls


for backend_name, device_type, create_pair_channel in SUSPEND_RESUME_BACKENDS:
    test_class = _make_suspend_resume_test_class(
        backend_name, device_type, create_pair_channel
    )
    globals()[test_class.__name__] = test_class


if __name__ == "__main__":
    run_tests()
