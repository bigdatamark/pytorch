# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the FlightRecorder hook
# (torch._C._distributed_c10d.FlightRecorderHook): FlightRecorder recording
# driven by the ProcessGroup pre/post collective hooks rather than native
# backend integration, so it works for any backend routed through c10d ops.
# Modeled on torchcomms' hooks/fr FlightRecorderTest; parameterized over
# backends like test_c10d_fault_tolerance.py.

# [MODULE DOC] Backend agnostic multi-process tests for the flight recorder hook that records
# collectives through process group pre and post hooks. An abstract mixin runs 2 ranks,
# attaches the hook to the default group and inspects the flight recorder JSON dump. Concrete
# test classes are generated at import time for gloo on cpu and nccl2 on cuda and put into
# module globals, with skips when the backend or two GPUs are unavailable. Exits at import
# with status zero if torch.distributed is unavailable.
import json
import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch._C._distributed_c10d import FlightRecorderHook
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


FR_HOOK_BACKENDS = [
    ("gloo", "cpu"),
    ("nccl2", "cuda"),
]


class AbstractFlightRecorderHookTest:
    # [CLASS DOC] Test mixin combined with MultiProcessTestCase by the factory; the subclass
    # supplies backend_name and device_type. Runs 2 ranks. Sets the flight recorder buffer
    # size environment variable in setUp, initializes the default group, and filters the
    # dumped trace to entries whose profiling name starts with the hook prefix. Tests check
    # recording and retirement, tensor metadata, sequence ids, detaching and entry ordering.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property: number of ranks, always 2.
        return 2

    @property
    def device(self):
        # [FUNCTION DOC] Property: torch device for this rank; the GPU with the rank as index
        # for cuda, otherwise the device of the device type.
        if self.device_type == "cuda":
            return torch.device(f"cuda:{self.rank}")
        return torch.device(self.device_type)

    def setUp(self):
        # [FUNCTION DOC] Test fixture: calls parent setUp, sets the environment variable for
        # the flight recorder buffer size to 2000 entries (process global side effect
        # inherited by spawned workers), then spawns the worker processes.
        super().setUp()
        # Note: gloo also records natively into the same recorder; keying
        # entries by the hook's profiling_name prefix ("c10d:") keeps the
        # assertions backend-agnostic.
        os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] Test fixture: destroys the default process group if initialized,
        # calls parent tearDown and tries to delete the store file, ignoring OS errors.
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        # [FUNCTION DOC] HELPER: selects the GPU for this rank on cuda, creates a file store,
        # initializes the default process group with the backend under test and a 60 second
        # timeout, and returns the world group.
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
        return dist.group.WORLD

    def _hook_entries(self):
        # [FUNCTION DOC] HELPER: dumps the flight recorder trace as a JSON string, parses it
        # and returns the list of entries whose profiling name starts with the hook prefix, so
        # native backend records are ignored.
        trace = json.loads(torch._C._distributed_c10d._dump_fr_trace_json())
        return [
            e
            for e in trace.get("entries", [])
            if e["profiling_name"].startswith("c10d:")
        ]

    def test_records_and_retires_collectives(self):
        # [FUNCTION DOC] Test: attaches the hook, runs all_reduce, broadcast and barrier, and
        # expects new entries for allreduce, broadcast and barrier, each marked retired.
        # Removes the hook at the end.
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        dist.barrier()
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        names = [e["profiling_name"] for e in entries]
        self.assertIn("c10d:allreduce", names)
        self.assertIn("c10d:broadcast", names)
        self.assertIn("c10d:barrier", names)
        # Post-hooks fire after issue, so every recorded op must be retired.
        # With null start/end events (no GPU timing) the state stays
        # "scheduled"; retired is the completion signal, as with Gloo's
        # native FR recording.
        for e in entries:
            self.assertTrue(e["retired"], msg=str(e))
        hook.remove()

    def test_records_tensor_metadata(self):
        # [FUNCTION DOC] Test: attaches the hook, runs one all_reduce on a 4 by 8 float tensor
        # and expects exactly one allreduce entry recording input size list of one 4 by 8
        # shape and the dtype name Float.
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, 8, device=self.device)
        dist.all_reduce(t)

        entries = self._hook_entries()[before:]
        allreduce = [e for e in entries if e["profiling_name"] == "c10d:allreduce"]
        self.assertEqual(len(allreduce), 1)
        self.assertEqual(allreduce[0]["input_sizes"], [[4, 8]])
        self.assertEqual(allreduce[0]["input_dtypes"], ["Float"])
        hook.remove()

    def test_p2p_and_collective_seq_ids(self):
        # [FUNCTION DOC] Test: with the hook attached, runs an all_reduce and a send and recv
        # exchange with the peer rank (ordered by rank to avoid deadlock). Expects two point
        # to point entries whose p2p sequence ids are 1 and 2, and one collective entry with
        # collective sequence id 1.
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        send_t = torch.ones(4, device=self.device)
        recv_t = torch.empty(4, device=self.device)
        peer = 1 - self.rank
        if self.rank == 0:
            dist.send(send_t, peer)
            dist.recv(recv_t, peer)
        else:
            dist.recv(recv_t, peer)
            dist.send(send_t, peer)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        p2p = [e for e in entries if e["profiling_name"] in ("c10d:send", "c10d:recv")]
        coll = [e for e in entries if e["profiling_name"] == "c10d:allreduce"]
        self.assertEqual(len(p2p), 2)
        self.assertEqual(len(coll), 1)
        # P2P ops advance p2p_seq_id only; collectives advance
        # collective_seq_id only.
        self.assertEqual(sorted(e["p2p_seq_id"] for e in p2p), [1, 2])
        self.assertEqual(coll[0]["collective_seq_id"], 1)
        hook.remove()

    def test_remove_stops_recording(self):
        # [FUNCTION DOC] Test: records some entries while the hook is attached, removes the
        # hook, runs another all_reduce and expects the number of entries not to change.
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        count_attached = len(self._hook_entries())
        self.assertGreater(count_attached, 0)

        hook.remove()
        dist.all_reduce(t)
        self.assertEqual(len(self._hook_entries()), count_attached)

    def test_multiple_collectives_entry_order(self):
        # [FUNCTION DOC] Test: runs five all_reduce calls with the hook attached and expects
        # five allreduce entries whose collective sequence ids appear in non decreasing order.
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, device=self.device)
        for _ in range(5):
            dist.all_reduce(t)

        entries = self._hook_entries()[before:]
        seqs = [
            e["collective_seq_id"]
            for e in entries
            if e["profiling_name"] == "c10d:allreduce"
        ]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), 5)
        hook.remove()


def _make_fr_hook_test_class(backend_name, device_type):
# [FUNCTION DOC] HELPER: factory building a concrete test class for a backend name and device
# type. Sets the attributes on a class joining the mixin and MultiProcessTestCase, renames it
# with the capitalized backend name, and applies skip decorators for an unavailable backend
# and, for cuda, fewer than 2 GPUs. Returns the decorated class.
    class FlightRecorderHookTest(AbstractFlightRecorderHookTest, MultiProcessTestCase):
        # [CLASS DOC] HELPER: empty concrete test case; attributes are assigned by the
        # enclosing factory.
        pass

    FlightRecorderHookTest.backend_name = backend_name
    FlightRecorderHookTest.device_type = device_type
    FlightRecorderHookTest.__name__ = (
        f"{backend_name.capitalize()}FlightRecorderHookTest"
    )
    FlightRecorderHookTest.__qualname__ = FlightRecorderHookTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(FlightRecorderHookTest)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2,
            "FR hook CUDA tests require at least 2 GPUs",
        )(cls)
    return cls


for backend_name, device_type in FR_HOOK_BACKENDS:
    globals()[f"{backend_name.capitalize()}FlightRecorderHookTest"] = (
        _make_fr_hook_test_class(backend_name, device_type)
    )


if __name__ == "__main__":
    run_tests()
