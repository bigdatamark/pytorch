# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-process tests for the c10d Gloo backend (ProcessGroupGloo) and DDP over
# Gloo. Exits the interpreter silently (status 0) at import time when c10d or Gloo is
# unavailable. Contains reference-case generators for reductions, rendezvous/timeout tests,
# collective check/basics/stress tests (broadcast, allreduce, scatter, gather, allgather,
# alltoall, reduce, barrier, send/recv), DDP, sharded tensor, and communication-hook tests.
# Run as a script, it calls the common run_tests entry point.

import copy
import json
import logging
import math
import operator
import os
import pickle
import random
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from functools import reduce
from itertools import groupby

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_gloo_available():
    print("c10d GLOO not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import test_c10d_common
from test_c10d_common import (
    FFTModel,
    gpus_for_rank,
    LOOPBACK,
    ModuleForDdpCommHook,
    SparseGradientModule,
    Task,
)

import torch.distributed as dist
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardedTensor,
    ShardMetadata,
)
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    create_device,
    MultiProcessTestCase,
    requires_gloo,
    simple_sparse_reduce_tests,
    skip_if_lt_x_gpu,
    skip_if_win32,
    verify_ddp_error_logged,
)
from torch.testing._internal.common_utils import (
    IS_MACOS,
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle,
    skipIfRocm,
    TestCase,
)


def simple_reduce_tests(rank, world_size):
    # [FUNCTION DOC] Build the list of allreduce cases for a given rank and world size.
    # Returns tuples of (reduce op, this rank's input tensor, expected reduced tensor)
    # covering SUM, PRODUCT, MIN, MAX, AVG on float, BAND (4 cases), BOR and BXOR (4 cases
    # each, emulating larger worlds by pre-combining several values per rank) on int32, plus
    # SUM and AVG on cfloat.
    tests = [
        (
            c10d.ReduceOp.SUM,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.tensor([rank + 1.0]),
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size)]),
        ),
        (
            c10d.ReduceOp.AVG,
            torch.tensor([rank + 1.0]),
            torch.tensor([float((world_size + 1) / 2)]),
        ),
    ]

    # Generate tests for BAND.
    # The bit that is set changes in every iteration to check
    # that the output changes accordingly.
    for i in range(4):
        vin = rank | (1 << i)
        vout = 1 << i
        tests.append(
            (
                c10d.ReduceOp.BAND,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for BOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-OR'ed.
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])
        vout = reduce(operator.or_, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for XOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-XOR'ed.
    for i in range(1, 5):
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        vout = reduce(operator.xor, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BXOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Extend tests for cfloat dtype
    tests.extend(
        (
            (
                c10d.ReduceOp.SUM,
                torch.tensor([complex(rank + 1.0, rank + 1.0)], dtype=torch.cfloat),
                torch.tensor(
                    [
                        complex(
                            world_size * (world_size + 1) / 2,
                            world_size * (world_size + 1) / 2,
                        )
                    ],
                    dtype=torch.cfloat,
                ),
            ),
            (
                c10d.ReduceOp.AVG,
                torch.tensor([complex(rank + 1.0, rank + 1.0)], dtype=torch.cfloat),
                torch.tensor(
                    [complex(float((world_size + 1) / 2), float((world_size + 1) / 2))],
                    dtype=torch.cfloat,
                ),
            ),
        )
    )
    return tests


def simple_coalesced_reduce_tests(rank, world_size):
    # [FUNCTION DOC] Build allreduce_coalesced cases for a rank and world size. Returns
    # tuples of (reduce op, list of two input tensors, list of two expected outputs) for SUM
    # (rank+1 and its square), PRODUCT, MIN and MAX with closed-form expected results.
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([rank + 1.0]), torch.tensor([(rank + 1.0) ** 2])],
            [
                torch.tensor([float(world_size * (world_size + 1) / 2)]),
                torch.tensor(
                    [float(world_size * (world_size + 1) * (2 * world_size + 1) / 6)]
                ),
            ],
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([rank + 1.0]), torch.tensor([rank + 2.0])],
            [
                torch.tensor([float(math.factorial(world_size))]),
                torch.tensor([float(math.factorial(world_size + 1))]),
            ],
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([rank + x]) for x in [0.0, 1.0]],
            [torch.tensor([0.0]), torch.tensor([1.0])],
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([rank + x]) for x in [1.0, 2.0]],
            [torch.tensor([float(world_size)]), torch.tensor([world_size + 1.0])],
        ),
    ]


def simple_multi_input_reduce_tests(rank, world_size):
    # [FUNCTION DOC] Build multi-input allreduce cases for a rank and world size. Returns
    # tuples of (reduce op, list of two input tensors held by this rank, single expected
    # tensor that every input tensor should equal afterwards) for SUM, PRODUCT, MIN and MAX,
    # treating each rank as owning two logical participants.
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([2 * rank + 0.0]), torch.tensor([2 * rank + 1.0])],
            torch.tensor([float(world_size * (2 * world_size - 1))]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([float(math.factorial(2 * world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([2.0 * world_size]),
        ),
    ]


class RendezvousTCPTest(TestCase):
    # [CLASS DOC] Tests TCP rendezvous with a single process and no spawned workers.
    @retry_on_connect_failures
    def test_tcp_init(self):
        # [FUNCTION DOC] Run the tcp rendezvous with port 0, rank 0, world size 1 (retried
        # on connect failures) and check that the returned rank/world size match and the
        # store got a real port.
        rendezvous_iterator = dist.rendezvous("tcp://127.0.0.1:0", rank=0, world_size=1)
        store, rank, world_size = next(rendezvous_iterator)
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)
        # port number should get assigned
        self.assertNotEqual(store.port, "0")


class RendezvousEnvTest(TestCase):
    # [CLASS DOC] Tests env:// rendezvous side effects in a single process.
    @requires_gloo()
    @retry_on_connect_failures
    def test_logging_init(self):
        # [FUNCTION DOC] Sets WORLD_SIZE, MASTER_ADDR, MASTER_PORT (free port) and RANK in
        # os.environ (side effect, never restored), initializes a one-rank gloo group via
        # env://, and checks that the root logging handler list is unchanged by init.
        # Destroys the group at the end.
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"

        previous_handlers = logging.root.handlers

        c10d.init_process_group(backend="gloo", init_method="env://")

        current_handlers = logging.root.handlers
        self.assertEqual(len(previous_handlers), len(current_handlers))
        for current, previous in zip(current_handlers, previous_handlers):
            self.assertEqual(current, previous)

        c10d.destroy_process_group()


class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    # [CLASS DOC] Gloo specialization of the shared timeout tests from test_c10d_common.
    @requires_gloo()
    @retry_on_connect_failures
    def test_default_store_timeout_gloo(self):
        # [FUNCTION DOC] Delegates to the shared default-store-timeout check using the gloo
        # backend.
        self._test_default_store_timeout("gloo")


class ProcessGroupGlooTest(MultiProcessTestCase):
    # [CLASS DOC] Multi-process test case for ProcessGroupGloo collectives. Each test runs
    # in world_size spawned processes (spawned in setUp); the class attribute lazy_init
    # controls whether created Gloo devices initialize lazily and is overridden by
    # subclasses. Helpers build the group over a file store; test methods come in checks
    # (error paths), basics (correctness), and stress (many concurrent async ops) flavors,
    # with CUDA variants.
    lazy_init = False

    def _create_process_group_gloo(self, store, rank, world_size, opts):
        # [FUNCTION DOC] HELPER: Construct a ProcessGroupGloo from the store using self.rank
        # and self.world_size (the rank/world_size arguments are ignored), then run a
        # barrier on it so all ranks are connected. Returns the group.
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, opts)
        dist.barrier(group=pg)
        return pg

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run the base setUp, then spawn the worker processes.
        super().setUp()
        self._spawn_processes()

    def opts(self, threads=2, group_name="0"):
        # [FUNCTION DOC] HELPER: Build Gloo options with a 50 second timeout, one loopback
        # device (lazy per class setting), the given thread count and group name. Returns
        # the options.
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 50.0
        opts._devices = [create_device(interface=LOOPBACK, lazy_init=self.lazy_init)]
        opts._threads = threads
        opts.group_name = group_name
        return opts

    @requires_gloo()
    def test_multi_device_constructor(self):
        # [FUNCTION DOC] Creates a group with two loopback devices and issues four async
        # allreduces of different sizes (twice the device count) so every device is used;
        # waits on all futures.
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 5.0
        opts._devices = [
            create_device(interface=LOOPBACK, lazy_init=self.lazy_init),
            create_device(interface=LOOPBACK, lazy_init=self.lazy_init),
        ]
        pg = self._create_process_group_gloo(store, self.rank, self.world_size, opts)

        # Execute 2x the number of operations to ensure we use every device.
        for fut in [pg.allreduce(torch.ones(i + 1)).get_future() for i in range(4)]:
            fut.wait()

    @requires_gloo()
    def test_split_world_pg_created_after_another_gloo_pg(self):
        # [FUNCTION DOC] Regression test: creates an unrelated gloo group, then a world-
        # sized group with default options, then splits it over all ranks. Checks the split
        # group exists, reports all ranks as its global ranks, and that an allreduce on it
        # yields the sum over ranks.
        # Regression test: ProcessGroupGloo::groupRanks() used to derive the
        # identity rank mapping only when local_id_ == 0. local_id_ is a
        # process-global counter over every ProcessGroupGloo constructed in the
        # process, so a world-sized pg built after any other gloo pg (e.g. a
        # stateless pg, or one inherited across fork) fell through to its empty
        # global_ranks_in_group, and split() then indexed that empty vector and
        # segfaulted on a null data pointer.
        store = c10d.FileStore(self.file_name, self.world_size)
        # (1) burn local_id_ == 0 on an unrelated gloo pg.
        self._create_process_group_gloo(
            c10d.PrefixStore("prior", store),
            self.rank,
            self.world_size,
            self.opts(group_name="prior"),
        )
        # (2) a world-sized pg: default options -> global_ranks_in_group empty.
        pg = self._create_process_group_gloo(
            c10d.PrefixStore("world", store),
            self.rank,
            self.world_size,
            self.opts(group_name="world"),
        )

        ranks = list(range(self.world_size))
        split = pg.split(
            c10d.PrefixStore("split", store), ranks, self.opts(group_name="split")
        )
        self.assertIsNotNone(split)
        self.assertEqual(split.options.global_ranks_in_group, ranks)

        tensor = torch.full((4,), float(self.rank + 1))
        split.allreduce([tensor]).wait()
        expected = float(sum(r + 1 for r in ranks))
        self.assertEqual(tensor, torch.full_like(tensor, expected))

    @requires_gloo()
    def test_empty_tensors(self):
        # [FUNCTION DOC] Broadcasting a zero-element tensor completes and returns an empty
        # tensor equal to the input.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        xs = [torch.FloatTensor([])]
        fut = pg.broadcast(xs).get_future()
        fut.wait()
        output = fut.value()
        self.assertEqual(0, output[0].numel())
        self.assertEqual(xs[0], output[0])

    @requires_gloo()
    def test_broadcast_checks(self):
        # [FUNCTION DOC] Checks broadcast argument validation: invalid root rank (negative
        # or too large), invalid root tensor index (negative, out of range, or empty list),
        # mismatched dtype and mismatched size among inputs each raise RuntimeError with the
        # matching message.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = -1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t2], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t3], opts)

    def _test_broadcast_basics(self, fn):
        # [FUNCTION DOC] HELPER: Shared broadcast correctness test; fn transforms each input
        # tensor (for example clone or move to CUDA). With every rank taking a turn as root,
        # checks results for one tensor, two tensors with each as root tensor, and a complex
        # tensor, then checks the convenience overload that takes a single tensor and a
        # root.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        def broadcast(xs, rootRank, rootTensor):
            # [FUNCTION DOC] HELPER: Broadcast the list of tensors from the given root rank
            # and root tensor index, wait for completion and return the future's value.
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            fut = pg.broadcast(xs, opts).get_future()
            fut.wait()
            return fut.value()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.tensor([self.rank]))
            output = broadcast([x], i, 0)
            self.assertEqual(torch.tensor([i]), output[0])

            # Run with 2 input tensors
            num = 2
            for j in range(num):
                xs = [
                    fn(torch.tensor([self.rank * num + 0.0])),
                    fn(torch.tensor([self.rank * num + 1.0])),
                ]

                output = broadcast(xs, i, j)
                self.assertEqual(
                    torch.tensor([i * num + j], dtype=torch.float32), output[0]
                )
                self.assertEqual(
                    torch.tensor([i * num + j], dtype=torch.float32), output[1]
                )

            # Run with 1 input tensor of cfloat dtype
            x = fn(torch.tensor([complex(self.rank, self.rank)], dtype=torch.cfloat))
            output = broadcast([x], i, 0)
            self.assertEqual(
                torch.tensor([complex(i, i)], dtype=torch.cfloat), output[0]
            )

        # Test overloaded convenience function
        x = torch.tensor([self.rank + 1.0])
        fut = pg.broadcast(x, root=0).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([1.0]), result[0])

    @requires_gloo()
    def test_broadcast_basics(self):
        # [FUNCTION DOC] CPU broadcast correctness test using cloned inputs.
        self._test_broadcast_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_broadcast_basics_cuda(self):
        # [FUNCTION DOC] CUDA broadcast correctness test; inputs are cloned and moved to the
        # GPU.
        self._test_broadcast_basics(lambda t: t.clone().cuda())

    def _test_broadcast_stress(self, inputs):
        # [FUNCTION DOC] HELPER: Create a group with 8 threads, issue one async broadcast
        # per input with root cycling over ranks, then wait on each and check the tensor
        # holds the root's value.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        work_handles = [
            pg.broadcast(inputs[i], root=(i % self.world_size))
            for i in range(len(inputs))
        ]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.tensor([(i * self.world_size) + (i % self.world_size)]),
                inputs[i],
                msg=(lambda msg: f"{msg}\nMismatch in iteration {i:d}"),
            )

    @requires_gloo()
    def test_broadcast_stress(self):
        # [FUNCTION DOC] Stress test: 1000 CPU broadcasts of rank-dependent values.
        inputs = [torch.tensor([i * self.world_size + self.rank]) for i in range(1000)]
        self._test_broadcast_stress(inputs)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_broadcast_stress_cuda(self):
        # [FUNCTION DOC] Stress test: 1000 CUDA broadcasts (needs 2 GPUs; skipped on ROCm).
        inputs = [
            torch.tensor([i * self.world_size + self.rank]).cuda() for i in range(1000)
        ]
        self._test_broadcast_stress(inputs)

    @requires_gloo()
    def test_allreduce_checks(self):
        # [FUNCTION DOC] Checks allreduce validation: an empty tensor list, mixed dtypes and
        # mixed sizes raise RuntimeError with the matching messages.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t3], opts)

    @requires_gloo()
    def test_allreduce_op_timeout(self):
        # [FUNCTION DOC] Sets a 1ms timeout on the allreduce options; only rank 0 issues the
        # op so it times out. Checks wait raises RuntimeError mentioning the timeout and
        # that the work's exception() returns a typed RuntimeError (regression for a raw
        # exception pointer).
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        opts = c10d.AllreduceOptions()
        opts.timeout = timedelta(milliseconds=1)

        if self.rank == 0:
            t1 = torch.zeros([1], dtype=torch.float32)
            work = pg.allreduce([t1], opts)
            with self.assertRaisesRegex(RuntimeError, "Timed out waiting 1ms"):
                work.wait()
            # Regression for #147312: Work.exception() must hand back a typed
            # Python exception, not a raw std::exception_ptr.
            exc = work.exception()
            self.assertIsInstance(exc, RuntimeError)
            self.assertIn("Timed out", str(exc))

    @requires_gloo()
    def test_allreduce_overall_timeout(self):
        # [FUNCTION DOC] Sets a 1ms group-wide timeout via set_timeout; only rank 0 issues
        # an allreduce, which must raise RuntimeError about timing out when waited on.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        pg.set_timeout(timedelta(milliseconds=1))

        if self.rank == 0:
            t1 = torch.zeros([1], dtype=torch.float32)
            with self.assertRaisesRegex(RuntimeError, "Timed out waiting 1ms"):
                pg.allreduce([t1]).wait()

    def _test_allreduce_basics(self, fn):
        # [FUNCTION DOC] HELPER: Shared allreduce correctness test with a tensor transform
        # fn. Runs all single-input cases, all multi-input cases, the default (SUM)
        # convenience overload, and an fp16 SUM check comparing allreduce against allgather
        # followed by a local sum with loose tolerances.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for op, input, expected in tests:
            with self.subTest(op=op):
                opts = c10d.AllreduceOptions()
                opts.reduceOp = op
                tensor = fn(input)
                fut = pg.allreduce([tensor], opts).get_future()
                fut.wait()
                result = fut.value()
                self.assertEqual(expected, result[0])

        # Multi input tests
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for op, inputs, output in tests:
            with self.subTest(op=op, multi_input=True):
                opts = c10d.AllreduceOptions()
                opts.reduceOp = op
                tensors = [fn(input) for input in inputs]
                fut = pg.allreduce(tensors, opts).get_future()
                fut.wait()
                result = fut.value()
                for tensor in result:
                    self.assertEqual(output, tensor)

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.tensor([self.rank + 1.0]))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]),
            result[0],
        )

        # Test fp16 numerical correctness for all-reduce SUM.
        torch.manual_seed(self.rank)
        # TODO: when create larger sizes of tensors, numerical instability will be observed.
        # We need to investigate the root cause and ensure it is fixed.
        tensor = (
            (torch.rand(200, 1, dtype=torch.float32) * 2 - 1) * 65504 / self.world_size
        )
        opts = c10d.AllreduceOptions()
        tensor = tensor.to(torch.float16)
        output = [[torch.zeros_like(tensor) for _ in range(self.world_size)]]
        # allgather all local tensors first and then sum up.
        fut = pg.allgather(output, [tensor]).get_future()
        fut.wait()
        ag_result = fut.value()
        total = torch.stack(ag_result, dim=0).sum(dim=0)

        # result from fp16 all-reduce.
        fut = pg.allreduce([tensor], opts).get_future()
        fut.wait()
        result_fp16 = fut.value()
        # float16 has only ~11 bits of mantissa, and is sensitive to accumulation
        # order and rounding errors so we use a larger tolerance.
        self.assertEqual(total, result_fp16[0], rtol=1e-2, atol=1e-3)

    @requires_gloo()
    def test_allreduce_basics(self):
        # [FUNCTION DOC] CPU allreduce correctness test.
        self._test_allreduce_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allreduce_basics_cuda(self):
        # [FUNCTION DOC] CUDA allreduce correctness test (needs 2 GPUs).
        self._test_allreduce_basics(lambda t: t.clone().cuda())

    def _test_allreduce_stress(self, inputs):
        # [FUNCTION DOC] HELPER: Create a group with 8 threads, issue one async allreduce
        # per input and check each result equals the expected sum over ranks for that
        # iteration.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = [
            pg.allreduce(inputs[i]).get_future() for i in range(len(inputs))
        ]
        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            self.assertEqual(
                torch.tensor(
                    [
                        (i * self.world_size)
                        + (self.world_size * (self.world_size - 1) // 2)
                    ]
                ),
                future_handle.value()[0],
                msg=(lambda msg: f"{msg}\nMismatch in iteration {i:d}"),
            )

    @requires_gloo()
    def test_allreduce_stress(self):
        # [FUNCTION DOC] Stress test: 1000 CPU allreduces.
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_allreduce_stress_cuda(self):
        # [FUNCTION DOC] Stress test: 1000 CUDA allreduces (needs 2 GPUs; skipped on ROCm).
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @requires_gloo()
    def test_allreduce_coalesced_checks(self):
        # [FUNCTION DOC] Checks allreduce_coalesced validation: empty list, mixed dtypes,
        # mixing dense and sparse layouts, and all-sparse input each raise RuntimeError with
        # the matching message.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros(1, dtype=torch.float32)
        t2 = torch.zeros(1, dtype=torch.float64)
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(1,))

        with self.assertRaisesRegex(RuntimeError, "requires non-empty tensor list"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([], opts)

        with self.assertRaisesRegex(
            RuntimeError, "tensors must all have the same type"
        ):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t2], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor layout at index"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t3], opts)

        with self.assertRaisesRegex(RuntimeError, "unsupported layout"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t3, t3.clone()], opts)

    @skip_if_lt_x_gpu(1)
    @requires_gloo()
    def test_allreduce_coalesced_checks_cuda(self):
        # [FUNCTION DOC] Checks that allreduce_coalesced on CUDA tensors raises RuntimeError
        # about an unsupported device type.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros(1, dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "unsupported device type"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1.cuda(), t1.cuda()], opts)

    def _test_allreduce_coalesced_basics(self, fn):
        # [FUNCTION DOC] HELPER: Run every coalesced reduce case with inputs transformed by
        # fn and compare each result tensor to the expected tensor.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        test_cases = simple_coalesced_reduce_tests(self.rank, self.world_size)
        for op, inputs, outputs in test_cases:
            opts = c10d.AllreduceCoalescedOptions()
            opts.reduceOp = op
            tensors = [fn(x) for x in inputs]
            fut = pg.allreduce_coalesced(tensors, opts).get_future()
            fut.wait()
            result = fut.value()
            for result_tensor, expected in zip(result, outputs):
                self.assertEqual(result_tensor, expected)

    @requires_gloo()
    def test_allreduce_coalesced_basics(self):
        # [FUNCTION DOC] CPU allreduce_coalesced correctness test.
        self._test_allreduce_coalesced_basics(lambda t: t.clone())

    def _expected_output(self, i):
        # [FUNCTION DOC] HELPER: Expected output of the i-th stress iteration: a list of two
        # identical tensors holding i times world size plus the sum of ranks.
        ws = self.world_size
        return 2 * [torch.tensor([(i * ws) + (ws * (ws - 1) // 2)])]

    def _test_allreduce_coalesced_stress(self, inputs):
        # [FUNCTION DOC] HELPER: Create a group with 8 threads, issue one async
        # allreduce_coalesced per input and compare each result against the expected output
        # for its iteration.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = [
            pg.allreduce_coalesced(input).get_future() for input in inputs
        ]
        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            self.assertEqual(
                self._expected_output(i),
                result,
                msg=lambda msg: f"{msg}\nMismatch in iteration {i}",
            )

    @requires_gloo()
    def test_allreduce_coalesced_stress(self):
        # [FUNCTION DOC] Stress test: 1000 coalesced allreduces, each of two identical
        # tensors.
        inputs = [2 * [torch.tensor([i + self.rank])] for i in range(1000)]
        self._test_allreduce_coalesced_stress(inputs)

    @requires_gloo()
    def test_sparse_allreduce_checks(self):
        # [FUNCTION DOC] Checks sparse allreduce validation: empty list, mixing dense and
        # sparse, mismatched sparse sizes raise RuntimeError; PRODUCT, MIN and MAX
        # reductions on sparse tensors raise RuntimeError about unsupported reduction.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1])
        t2 = torch.sparse_coo_tensor([[0]], [1], size=(2,))
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(4,))

        with self.assertRaisesRegex(RuntimeError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor layout"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t2, t3], opts)

        # Sparse allreduce only works with c10d.ReduceOp.SUM.
        for op in [c10d.ReduceOp.PRODUCT, c10d.ReduceOp.MIN, c10d.ReduceOp.MAX]:
            with self.assertRaisesRegex(
                RuntimeError, "unsupported reduction operation"
            ):
                opts = c10d.AllreduceOptions()
                opts.reduceOp = op
                pg.allreduce([t3], opts)

    def _test_sparse_allreduce_basics(self, fn):
        # [FUNCTION DOC] HELPER: Run sparse allreduce on cases from
        # simple_sparse_reduce_tests with one and two inputs per rank (inputs transformed by
        # fn); checks both the in-place tensors and the returned result equal the expected
        # outputs.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        for num_inputs_per_rank in [1, 2]:
            tests = simple_sparse_reduce_tests(
                self.rank, self.world_size, num_inputs=num_inputs_per_rank
            )
            for inputs, outputs in tests:
                tensors = [fn(input) for input in inputs]
                fut = pg.allreduce(tensors).get_future()
                fut.wait()
                result = fut.value()
                self.assertEqual(tensors, outputs)
                self.assertEqual(result, outputs)

    @requires_gloo()
    def test_sparse_allreduce_basics(self):
        # [FUNCTION DOC] CPU sparse allreduce correctness test.
        self._test_sparse_allreduce_basics(lambda t: t)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sparse_allreduce_basics_cuda(self):
        # [FUNCTION DOC] CUDA sparse allreduce correctness test (needs 2 GPUs).
        self._test_sparse_allreduce_basics(lambda t: t.clone().cuda())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sparse_allreduce_cuda_dispatched(self):
        # [FUNCTION DOC] Initializes the default gloo process group, then runs
        # dist.all_reduce asynchronously on a sparse CUDA tensor for each case (needs 2
        # GPUs) and checks the result. Does not destroy the group.
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        tests = simple_sparse_reduce_tests(self.rank, self.world_size, num_inputs=1)
        for inputs, outputs in tests:
            tensors = inputs[-1].clone().cuda()
            work = dist.all_reduce(tensors, async_op=True)
            work.wait()
            self.assertEqual([tensors], outputs)

    @requires_gloo()
    def test_allgather_into_tensor_coalesced(self):
        # [FUNCTION DOC] Initializes the default gloo group and runs the coalesced
        # allgather-into-tensor on three differently shaped random tensors; each output must
        # equal the input concatenated world_size times. Seeds the RNG with 42.
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.manual_seed(42)
        in_shapes = [(5, 5), (10, 10), (15, 15)]
        out_shapes = [(s[0] * self.world_size,) + s[1:] for s in in_shapes]

        outputs = [torch.empty(s) for s in out_shapes]
        inputs = [torch.rand(s) for s in in_shapes]
        work = dist.group.WORLD.allgather_into_tensor_coalesced(outputs, inputs)
        work.wait()

        for output, input in zip(outputs, inputs):
            expect = torch.cat([input] * self.world_size)
            self.assertTrue(torch.allclose(output, expect))

    @requires_gloo()
    def test_reduce_scatter(self):
        # [FUNCTION DOC] Initializes the default gloo group and runs async reduce_scatter
        # for SUM, AVG, MIN, MAX and PRODUCT with list inputs (constant tensors of
        # rank+i+1). Compares each output to the value computed locally over ranks, one
        # subtest per op.
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.manual_seed(42)

        for op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.PRODUCT,
        ):
            with self.subTest(op=op):
                inputs = [
                    torch.full((3,), float(self.rank + i + 1))
                    for i in range(self.world_size)
                ]
                output = torch.empty(3)

                work = dist.reduce_scatter(output, inputs, op=op, async_op=True)
                work.wait()

                r = self.rank
                ws = self.world_size
                values = [float(r + k + 1) for k in range(ws)]

                if op == c10d.ReduceOp.SUM:
                    expected_val = sum(values)
                elif op == c10d.ReduceOp.AVG:
                    expected_val = sum(values) / ws
                elif op == c10d.ReduceOp.MIN:
                    expected_val = min(values)
                elif op == c10d.ReduceOp.MAX:
                    expected_val = max(values)
                elif op == c10d.ReduceOp.PRODUCT:
                    expected_val = reduce(operator.mul, values, 1.0)
                else:
                    raise ValueError(f"Unsupported op: {op}")

                expect = torch.full((3,), expected_val)
                self.assertTrue(
                    torch.allclose(output, expect),
                    lambda msg: f"{msg}\nop={op}, rank={self.rank}: output={output}, expected={expect}",
                )

    @requires_gloo()
    def test_reduce_scatter_tensor(self):
        # [FUNCTION DOC] Same as the list reduce_scatter test but through the single-tensor
        # reduce_scatter_single variant: input is a concatenation of per-rank chunks of size
        # 10. Checks each op's result against the locally computed reduction.
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.manual_seed(42)

        for op in [
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.PRODUCT,
        ]:
            with self.subTest(op=op):
                out_size = 10
                in_size = out_size * self.world_size

                input = torch.empty(in_size)
                for i in range(self.world_size):
                    input[i * out_size : (i + 1) * out_size] = float(self.rank + i + 1)
                output = torch.empty(out_size)

                work = dist.reduce_scatter_single(output, input, op=op, async_op=True)
                work.wait()

                r = self.rank
                ws = self.world_size
                values = [float(r + k + 1) for k in range(ws)]

                if op == c10d.ReduceOp.SUM:
                    expected_val = sum(values)
                elif op == c10d.ReduceOp.AVG:
                    expected_val = sum(values) / ws
                elif op == c10d.ReduceOp.MIN:
                    expected_val = min(values)
                elif op == c10d.ReduceOp.MAX:
                    expected_val = max(values)
                elif op == c10d.ReduceOp.PRODUCT:
                    expected_val = reduce(operator.mul, values, 1.0)
                else:
                    raise ValueError(f"Unsupported op: {op}")

                expect = torch.full((out_size,), expected_val)
                self.assertTrue(
                    torch.allclose(output, expect),
                    lambda msg: f"{msg}\nop={op}, rank={self.rank}: output={output[0]}, expected={expect[0]}",
                )

    @requires_gloo()
    def test_reduce_scatter_tensor_coalesced(self):
        # [FUNCTION DOC] Tests coalesced reduce_scatter_tensor with three tensors of
        # different chunk sizes (5, 10, 15) per op, via the world group's coalesced method
        # and ReduceScatterOptions; checks every output against the locally computed
        # reduction.
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.manual_seed(42)

        for op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.PRODUCT,
        ):
            with self.subTest(op=op):
                out_sizes = [5, 10, 15]
                in_sizes = [s * self.world_size for s in out_sizes]

                outputs = [torch.empty(s) for s in out_sizes]
                inputs = []
                for in_size, out_size in zip(in_sizes, out_sizes, strict=True):
                    inp = torch.empty(in_size)
                    for i in range(self.world_size):
                        inp[i * out_size : (i + 1) * out_size] = float(
                            self.rank + i + 1
                        )
                    inputs.append(inp)

                opts = c10d.ReduceScatterOptions()
                opts.reduceOp = op
                work = dist.group.WORLD.reduce_scatter_tensor_coalesced(
                    outputs, inputs, opts
                )
                work.wait()

                r = self.rank
                ws = self.world_size
                values = [float(r + k + 1) for k in range(ws)]

                if op == c10d.ReduceOp.SUM:
                    expected_val = sum(values)
                elif op == c10d.ReduceOp.AVG:
                    expected_val = sum(values) / ws
                elif op == c10d.ReduceOp.MIN:
                    expected_val = min(values)
                elif op == c10d.ReduceOp.MAX:
                    expected_val = max(values)
                elif op == c10d.ReduceOp.PRODUCT:
                    expected_val = reduce(operator.mul, values, 1.0)
                else:
                    raise ValueError(f"Unsupported op: {op}")

                for output, out_size in zip(outputs, out_sizes, strict=True):
                    expect = torch.full((out_size,), expected_val)
                    self.assertTrue(
                        torch.allclose(output, expect),
                        lambda msg: f"{msg}\nop={op}, rank={self.rank}: output={output[0]}, expected={expect[0]}",
                    )

    @requires_gloo()
    def test_scatter_checks(self):
        # [FUNCTION DOC] Checks scatter validation: bad root rank, output list not of length
        # one, input list not a single list on the root, wrong inner list length (too short
        # or too long), mismatched dtype or size, and non-empty input on a non-root rank
        # each raise RuntimeError with the matching message.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element output tensor list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element output tensor list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([t1, t1], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element input list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element input list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * self.world_size, [t1] * self.world_size], opts)

        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = "Incorrect input list size {}. Input list size should be {}"
        with self.assertRaisesRegex(
            RuntimeError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)

        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(
            RuntimeError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t2] * self.world_size], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t3] * self.world_size], opts)

        with self.assertRaisesRegex(RuntimeError, "requires empty input on non-root"):
            opts = c10d.ScatterOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.scatter([t1], [[t1] * self.world_size], opts)

    def _test_scatter_basics(self, fn):
        # [FUNCTION DOC] HELPER: Scatter correctness; ranks take turns as root, the root
        # supplying per-rank tensors holding its own rank. Each rank's output must equal the
        # root's rank value.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Preallocate tensors for input/output
        input = [fn(torch.tensor([self.rank])) for _ in range(self.world_size)]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the scatter root and accumulate work items
        futures = []
        for i in range(self.world_size):
            opts = c10d.ScatterOptions()
            opts.rootRank = i
            if i == self.rank:
                futures.append(pg.scatter([outputs[i]], [input], opts).get_future())
            else:
                futures.append(pg.scatter([outputs[i]], [], opts).get_future())

        # Wait for work to complete
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            self.assertEqual(torch.tensor([i]), result[0])

    @requires_gloo()
    def test_scatter_basics(self):
        # [FUNCTION DOC] CPU scatter correctness test.
        self._test_scatter_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_scatter_basics_cuda(self):
        # [FUNCTION DOC] CUDA scatter correctness test (needs 2 GPUs).
        self._test_scatter_basics(lambda t: t.clone().cuda())

    def _test_scatter_stress(self, inputs, fn):
        # [FUNCTION DOC] HELPER: Create an 8-thread group and queue one scatter per (input,
        # root) pair; afterwards each result must equal the iteration index plus the root
        # rank. Inputs are transformed by fn on the root only.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        outputs = [
            [fn(torch.tensor([-1])) for _ in range(self.world_size)]
            for _ in range(len(inputs))
        ]
        future_handles = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ScatterOptions()
                opts.rootRank = root
                if root == self.rank:
                    fut = pg.scatter(
                        [outputs[i][root]], [[fn(e) for e in inputs[i]]], opts
                    ).get_future()
                else:
                    fut = pg.scatter([outputs[i][root]], [], opts).get_future()
                future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            result = future_handle.value()

            self.assertEqual(
                torch.tensor([iter + root]),
                result[0],
                msg=(
                    lambda msg: f"{msg}\nMismatch in iteration {iter:d} for rank {root:d}"
                ),
            )

    @requires_gloo()
    def test_set_gloo_pg_timeout(self):
        # [FUNCTION DOC] Checks the private default timeout of the group options is 50
        # seconds, and that setting the default timeout to 23 seconds updates the options.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        pg.allreduce(torch.rand(10))
        self.assertEqual(pg.options._timeout, timedelta(seconds=50))
        pg._set_default_timeout(timedelta(seconds=23))
        self.assertEqual(pg.options._timeout, timedelta(seconds=23))

    @requires_gloo()
    def test_gloo_set_timeout_api(self):
        # [FUNCTION DOC] Initializes the default gloo group with a 50s timeout, checks the
        # backend options, runs an allreduce, then changes the timeout to 1ms via
        # set_timeout (checking the backend options update). Only rank 0 then issues an
        # allreduce, which must time out. Destroys the process group at the end.
        """
        Test set_timeout API for Gloo backend (issue #165422).
        This test demonstrates that dynamically changing timeout via set_timeout
        actually affects operation timeouts by:
        1. verifying operations complete successfully with normal timeout
        2. setting a very short timeout via set_timeout
        3. demonstrating that operations timeout with the new short timeout value
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=50),
        )

        pg = dist.distributed_c10d._get_default_group()
        backend = pg._get_backend(torch.device("cpu"))
        self.assertEqual(backend.options._timeout, timedelta(seconds=50))

        # operations work normally with default timeout
        tensor = torch.rand(10)
        pg.allreduce(tensor).wait()

        # change timeout to a very short value using set_timeout
        # this is the API from issue #165422
        c10d.set_timeout(timedelta(milliseconds=1), pg)
        self.assertEqual(backend.options._timeout, timedelta(milliseconds=1))

        # demonstrate that the new timeout is actually enforced
        # only rank 0 participates - this will cause a timeout
        if self.rank == 0:
            t1 = torch.zeros([1], dtype=torch.float32)
            with self.assertRaisesRegex(RuntimeError, "Timed out waiting 1ms"):
                pg.allreduce([t1]).wait()

        dist.destroy_process_group()

    @requires_gloo()
    def test_gloo_set_pg_timeout_deprecated(self):
        # [FUNCTION DOC] Checks that the deprecated private _set_pg_timeout still works: it
        # emits a FutureWarning and updates the backend timeout to 23 seconds. Destroys the
        # group afterwards.
        """
        The private `_set_pg_timeout` alias is deprecated but must remain
        functional, delegating to the public `set_timeout`.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=50),
        )

        pg = dist.distributed_c10d._get_default_group()
        backend = pg._get_backend(torch.device("cpu"))

        with self.assertWarnsRegex(FutureWarning, "_set_pg_timeout"):
            c10d.distributed_c10d._set_pg_timeout(timedelta(seconds=23), pg)
        self.assertEqual(backend.options._timeout, timedelta(seconds=23))

        dist.destroy_process_group()

    @requires_gloo()
    def test_scatter_stress(self):
        # [FUNCTION DOC] Stress test: 1000 CPU scatters of rank-independent tensors.
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone())

    @skip_but_pass_in_sandcastle(
        "Test is flaky, see https://github.com/pytorch/pytorch/issues/15963"
    )
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_scatter_stress_cuda(self):
        # [FUNCTION DOC] Stress test: 1000 CUDA scatters (needs 2 GPUs; skipped on ROCm and
        # in sandcastle runs, where it is reported as passing because it is flaky).
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()
    def test_gather_checks(self):
        # [FUNCTION DOC] Checks gather validation: bad root rank, input list not of length
        # one, output list not a single list on the root, wrong output list length (short or
        # long), mismatched dtype or size, and non-empty output on a non-root rank each
        # raise RuntimeError with the message.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element input tensor list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element input tensor list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [t1, t1], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element output list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element output list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * self.world_size, [t1] * self.world_size], [t1], opts)

        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = "Incorrect output list size {}. Output list size should be {}"
        with self.assertRaisesRegex(
            RuntimeError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)

        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(
            RuntimeError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t2] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t3] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(RuntimeError, "requires empty output on non-root"):
            opts = c10d.GatherOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.gather([[t1] * self.world_size], [t1], opts)

    def _test_gather_basics(self, fn):
        # [FUNCTION DOC] HELPER: Gather correctness; ranks take turns as root; the root's
        # output list is compared with each rank's value, transformed by fn.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Preallocate tensors for input/output
        input = [fn(torch.tensor([self.rank]))]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the gather root and accumulate work items
        futures = []
        for i in range(self.world_size):
            opts = c10d.GatherOptions()
            opts.rootRank = i
            if i == self.rank:
                futures.append(pg.gather([outputs], input, opts).get_future())
            else:
                futures.append(pg.gather([], input, opts).get_future())

        # Wait for work to complete
        expected = [fn(torch.tensor([rank])) for rank in range(self.world_size)]
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            if i == self.rank:
                self.assertEqual(expected, result)

    @requires_gloo()
    def test_gather_basics(self):
        # [FUNCTION DOC] CPU gather correctness test.
        self._test_gather_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_gather_basics_cuda(self):
        # [FUNCTION DOC] CUDA gather correctness test (needs 2 GPUs).
        self._test_gather_basics(lambda t: t.clone().cuda())

    @requires_gloo()
    def test_gather_noncontiguous_input(self):
        # [FUNCTION DOC] Gather correctness with non-contiguous, non-dense input tensors (a
        # strided column view of a two-dimensional tensor).
        # Take a column of 2D tensor, such that memory is not dense
        self._test_gather_basics(lambda t: t.expand(2, 2).tril().contiguous()[:, 0])

    def _test_gather_stress(self, inputs, fn):
        # [FUNCTION DOC] HELPER: Create an 8-thread group and queue one gather per (input,
        # root) pair; on the root, the gathered list must equal the per-rank values for that
        # iteration.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = []
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.GatherOptions()
                opts.rootRank = root
                if root == self.rank:
                    fut = pg.gather(outputs[i], [fn(inputs[i])], opts).get_future()
                else:
                    fut = pg.gather([], [fn(inputs[i])], opts).get_future()
                future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                result = future_handle.value()
                self.assertEqual(
                    expected_outputs[iter],
                    [result],
                    msg=(
                        lambda msg: f"{msg}\nMismatch in iteration {iter:d} for root {root:d}"
                    ),
                )

    @requires_gloo()
    def test_gather_stress(self):
        # [FUNCTION DOC] Stress test: 1000 CPU gathers.
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    @requires_gloo()
    def test_gather_stress_cuda(self):
        # [FUNCTION DOC] Stress test: 1000 CUDA gathers (needs 2 GPUs; skipped on ROCm).
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()
    def test_allgather_checks(self):
        # [FUNCTION DOC] Checks allgather validation: empty input/output lists, input and
        # output lists of different lengths, output list not of world size (short or long),
        # and mismatched dtype or size among inputs or among output tensors each raise
        # RuntimeError with the matching message.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(
            RuntimeError, "requires non-empty input tensor list"
        ):
            pg.allgather([], [])

        with self.assertRaisesRegex(
            RuntimeError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([], [t1])

        with self.assertRaisesRegex(
            RuntimeError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([[t1] * self.world_size, [t1] * self.world_size], [t1])

        with self.assertRaisesRegex(RuntimeError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size - 1)], [t1])

        with self.assertRaisesRegex(RuntimeError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size + 1)], [t1])

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t2]
            )

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t3]
            )

        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            pg.allgather([([t1, t2] * (self.world_size))[: self.world_size]], [t1])

        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            pg.allgather([([t1, t3] * (self.world_size))[: self.world_size]], [t1])

    def _test_allgather_basics(self, fn):
        # [FUNCTION DOC] HELPER: Allgather correctness with 1, 2 and 3 input tensors per
        # rank (values derived from the rank, transformed by fn); the gathered lists must
        # equal consecutive integers. With a single input the returned value is wrapped in a
        # list before comparison.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Run with N input tensor per rank
        for n in [1, 2, 3]:
            input = [fn(torch.tensor([n * self.rank + i])) for i in range(n)]
            output = [
                [fn(torch.tensor([-1])) for _ in range(n * self.world_size)]
                for _ in range(n)
            ]
            expected_output = [
                [fn(torch.tensor([i])) for i in range(n * self.world_size)]
                for _ in range(n)
            ]
            fut = pg.allgather(output, input).get_future()
            fut.wait()
            result = fut.value()
            if n == 1:
                result = [result]
            self.assertEqual(expected_output, result)

    @requires_gloo()
    def test_allgather_basics(self):
        # [FUNCTION DOC] CPU allgather correctness test.
        self._test_allgather_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allgather_basics_cuda(self):
        # [FUNCTION DOC] CUDA allgather correctness test (needs 2 GPUs).
        self._test_allgather_basics(lambda t: t.clone().cuda())

    @requires_gloo()
    def test_allgather_noncontiguous_input(self):
        # [FUNCTION DOC] Allgather correctness with non-contiguous, non-dense input tensors.
        # Take a column of 2D tensor, such that memory is not dense
        self._test_allgather_basics(lambda t: t.expand(2, 2).tril().contiguous()[:, 0])

    @requires_gloo()
    def test_allgather_inference_mode(self):
        # [FUNCTION DOC] Runs the CPU allgather correctness test under inference mode.
        with torch.inference_mode():
            self._test_allgather_basics(lambda t: t.clone())

    def _test_allgather_stress(self, inputs, fn):
        # [FUNCTION DOC] HELPER: Create an 8-thread group and queue one allgather per input
        # (each input kept alive in a holder mapping to avoid a known data race), then check
        # each gathered list equals the iteration value plus each rank index.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = []
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        input_holder = {}
        for i in range(len(inputs)):
            # Note that this works around the data race discussed in
            # https://github.com/pytorch/pytorch/issues/75529, but we should
            # actually be able to pass the list directly into allgather when
            # that race is fixed.
            input_holder[i] = [fn(inputs[i])]
            fut = pg.allgather(outputs[i], input_holder[i]).get_future()
            future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            self.assertEqual(
                expected_outputs[i],
                [result],
                msg=(lambda msg: f"{msg}\nMismatch in iteration {i:d}"),
            )

    @requires_gloo()
    def test_allgather_stress(self):
        # [FUNCTION DOC] Stress test: 1000 CPU allgathers.
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_allgather_stress_cuda(self):
        # [FUNCTION DOC] Stress test: 1000 CUDA allgathers (needs 2 GPUs; skipped on ROCm).
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()
    def test_allgather_coalesced_checks(self):
        # [FUNCTION DOC] Checks all_gather_coalesced validation: output tensor with wrong
        # size or wrong dtype relative to the input (RuntimeError naming index 0), output
        # list count not equal to world size (RuntimeError), and a flat tensor list instead
        # of list of lists (TypeError).
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        dummy_input = [torch.zeros([1], dtype=torch.float32)]
        dummy_output_lists = [
            [torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size)
        ]

        # One of output tensors does not match input list.
        dummy_output_lists[0] = [torch.zeros([0], dtype=torch.float32)]
        with self.assertRaisesRegex(
            RuntimeError, "invalid size of output tensor at index 0"
        ):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # One of output tensors does not match input list.
        dummy_output_lists[0] = [torch.zeros([1], dtype=torch.float64)]
        with self.assertRaisesRegex(RuntimeError, "invalid tensor type at index 0"):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # Output lists have too many elements
        dummy_output_lists = [
            [torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size + 1)
        ]
        with self.assertRaisesRegex(
            RuntimeError, "output lists should be equal to world size"
        ):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # Output is not a list of lists.
        dummy_output_lists = [torch.zeros([0], dtype=torch.float32)]
        with self.assertRaisesRegex(
            TypeError, "Invalid function argument.*output_tensor_lists"
        ):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

    @requires_gloo()
    def test_allgather_coalesced_async(self):
        # [FUNCTION DOC] Initializes the default gloo group, launches two async
        # all_gather_coalesced calls with two identical tensors each, waits for all futures
        # and checks each gathered tensor equals the corresponding rank's input. Destroys
        # the process group explicitly at the end because otherwise the Gloo group may not
        # be destructed before the interpreter exits.
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )

        xxs = [2 * [torch.tensor([i + self.rank])] for i in range(2)]
        yys = [
            [[torch.zeros_like(x) for x in xx] for _ in range(self.world_size)]
            for xx in xxs
        ]
        futs = [
            c10d.all_gather_coalesced(yy, xx, async_op=True) for xx, yy in zip(xxs, yys)
        ]

        # expected outputs
        zzs = [
            [2 * [torch.tensor([i + r])] for r in range(self.world_size)]
            for i in range(2)
        ]

        torch.futures.wait_all(futs)
        for yy, zz in zip(yys, zzs):
            # one iteration
            for y_out, z_out in zip(yy, zz):
                # one output tensor list
                for y, z in zip(y_out, z_out):
                    # one tensor in output tensor list
                    self.assertEqual(y, z)

        # Added to address https://github.com/pytorch/pytorch/issues/65231
        # In the failed tests, all assertEqual are passed on all processes.
        # However, one of the processes didn't call ProcessGroupGloo
        # destructor before exiting program. This is not surprising as the only
        # guarantee that Python makes is that garbage collection MAY happen
        # before the program exits. If GC didn't happen, the two threads in
        # ProcessGroup might be destructed before joined.
        # FIXME: it's still unclear why only this test require explicit
        # destroy_process_group()
        c10d.destroy_process_group()

    @requires_gloo()
    def test_reduce_checks(self):
        # [FUNCTION DOC] Checks reduce validation: bad root rank (negative or too large),
        # bad root tensor index, and a tensor list that is not a single element each raise
        # RuntimeError.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element tensor list"
        ):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.reduce([t1, t1], opts)

    def _test_reduce_basics(self, fn):
        # [FUNCTION DOC] HELPER: Run every simple reduce case with every rank as root,
        # inputs transformed by fn; the root rank compares its result to the expected tensor
        # (non-roots are not checked).
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        for op, input, output in simple_reduce_tests(self.rank, self.world_size):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                tmp = fn(input)
                fut = pg.reduce([tmp], opts).get_future()
                fut.wait()
                result = fut.value()
                if root == self.rank:
                    self.assertEqual(output, result[0])

    @requires_gloo()
    def test_reduce_basics(self):
        # [FUNCTION DOC] CPU reduce correctness test.
        self._test_reduce_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_reduce_basics_cuda(self):
        # [FUNCTION DOC] CUDA reduce correctness test (needs 2 GPUs).
        self._test_reduce_basics(lambda t: t.clone().cuda())

    def _test_reduce_stress(self, inputs):
        # [FUNCTION DOC] HELPER: Create an 8-thread group and queue a reduce (default SUM)
        # on a clone of each input for every root; when this rank is root, the result must
        # equal the sum over ranks.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = []
        outputs = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.rootRank = root
                tmp = inputs[i].clone()
                outputs.append(tmp)
                fut = pg.reduce([tmp], opts).get_future()
                future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                self.assertEqual(
                    torch.tensor(
                        [
                            (iter * self.world_size)
                            + (self.world_size * (self.world_size - 1) // 2)
                        ]
                    ),
                    result[0],
                    msg=(
                        lambda msg: f"{msg}\nMismatch in iteration {iter:d} with root rank {root:d}"
                    ),
                )

    @requires_gloo()
    def test_reduce_stress(self):
        # [FUNCTION DOC] Stress test: 1000 CPU reduces over all roots.
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_reduce_stress(inputs)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_reduce_stress_cuda(self):
        # [FUNCTION DOC] Stress test: 1000 CUDA reduces (needs 2 GPUs; skipped on ROCm).
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_reduce_stress(inputs)

    @unittest.skipIf(IS_MACOS, "https://github.com/pytorch/pytorch/issues/71195")
    @requires_gloo()
    def test_send_recv_all_to_all(self):
        # [FUNCTION DOC] Every rank sends its rank value to every other rank with point-to-
        # point send and recv (tag 0), waits for all work, checks it reports completed, and
        # checks received values equal the sender's rank. Skipped on macOS.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Preallocate tensors for input/output
        inputs = [torch.tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.tensor([-1]) for _ in range(self.world_size)]

        # Issue sends
        send_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            send_work.append(pg.send([inputs[i]], i, 0))

        # Issue recvs
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            recv_work.append(pg.recv([outputs[i]], i, 0))

        # Wait for sends to complete
        for work in send_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Wait for recvs to complete
        for work in recv_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Test that every output other than our own contains the respective rank
        for i in range(self.world_size):
            if i == self.rank:
                continue
            self.assertEqual(torch.tensor([i]), outputs[i])

    @requires_gloo()
    def test_barrier_implies_wait(self):
        # [FUNCTION DOC] Issues 16 allreduces whose work handles are dropped, then a
        # barrier, and checks after the barrier future completes that all tensors hold the
        # reduced values (barrier waits for prior work).
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Kick off allreduce operations
        size = (100, 100)
        num = 16
        tensors = [torch.full(size, float(i)) for i in range(num)]
        for tensor in tensors:
            # Note: leak the returned work handle
            pg.allreduce(tensor)

        # Barrier should ensure all previous work has completed
        pg.barrier().get_future().wait()

        for i, tensor in enumerate(tensors):
            self.assertEqual(torch.full(size, float(i * self.world_size)), tensor)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_block_current_stream_cuda(self):
        # [FUNCTION DOC] On a CUDA tensor, calls block_current_stream on an allreduce work,
        # synchronizes the current stream, then waits on the work. Needs 2 GPUs; skipped on
        # ROCm.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        t = torch.zeros(10, device="cuda")
        work = pg.allreduce(t)
        work.block_current_stream()
        torch.cuda.current_stream().synchronize()

        work.wait()

    @requires_gloo()
    def test_send_recv_complex(self):
        # [FUNCTION DOC] Rank 0 sends a seeded random complex tensor to rank 1, which
        # receives into a tensor and checks equality with the same locally regenerated
        # tensor.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        # Generate the same random tensor
        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, dtype=torch.cfloat)
        if self.rank == 0:
            pg.send([send_tensor], 1, 0).wait()
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, dtype=torch.cfloat)
            pg.recv([recv_tensor], 0, 0).wait()
            self.assertEqual(send_tensor, recv_tensor)

    @requires_gloo()
    def test_alltoall_checks(self):
        # [FUNCTION DOC] Checks alltoall validation: input or output list not of world size,
        # mismatched dtype or size inside the input or output lists, and input/output
        # tensors differing in dtype or size each raise RuntimeError with the matching
        # message.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        # Check input tensor list size does not match world size
        with self.assertRaisesRegex(
            RuntimeError, "input tensor list size.*does not match world size"
        ):
            pg.alltoall([t1] * self.world_size, [t1] * (self.world_size - 1))

        # Check output tensor list size does not match world size
        with self.assertRaisesRegex(
            RuntimeError, "output tensor list size.*does not match world size"
        ):
            pg.alltoall([t1] * (self.world_size - 1), [t1] * self.world_size)

        # Check invalid tensor type in inputs
        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            pg.alltoall([t1] * self.world_size, [t1, t2] + [t1] * (self.world_size - 2))

        # Check invalid tensor size in inputs
        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            pg.alltoall([t1] * self.world_size, [t1, t3] + [t1] * (self.world_size - 2))

        # Check invalid tensor type in outputs
        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            pg.alltoall([t1, t2] + [t1] * (self.world_size - 2), [t1] * self.world_size)

        # Check invalid tensor size in outputs
        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            pg.alltoall([t1, t3] + [t1] * (self.world_size - 2), [t1] * self.world_size)

        # Check input and output tensors must have the same type
        with self.assertRaisesRegex(
            RuntimeError, "input and output tensors must have the same type"
        ):
            pg.alltoall([t2] * self.world_size, [t1] * self.world_size)

        # Check input and output tensors must have the same size
        with self.assertRaisesRegex(
            RuntimeError, "input and output tensors must have the same size"
        ):
            pg.alltoall([t3] * self.world_size, [t1] * self.world_size)

    def _test_alltoall_basics(self, fn):
        # [FUNCTION DOC] HELPER: Each rank sends a tensor holding its rank value to every
        # rank; after alltoall, the output at index j must hold j. Tensors are transformed
        # by fn.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Preallocate tensors for input/output
        # Each rank sends tensor with value equal to rank to each other rank
        input_tensors = [fn(torch.tensor([self.rank])) for _ in range(self.world_size)]
        output_tensors = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Perform all-to-all
        fut = pg.alltoall(output_tensors, input_tensors).get_future()
        fut.wait()

        # After alltoall, rank i should have received tensor with value j from rank j
        # in output_tensors[j]
        for i in range(self.world_size):
            self.assertEqual(torch.tensor([i]), output_tensors[i])

    @requires_gloo()
    def test_alltoall_basics(self):
        # [FUNCTION DOC] CPU alltoall correctness test.
        self._test_alltoall_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_alltoall_basics_cuda(self):
        # [FUNCTION DOC] CUDA alltoall correctness test (needs 2 GPUs).
        self._test_alltoall_basics(lambda t: t.clone().cuda())

    def _test_alltoall_stress(self, inputs, fn):
        # [FUNCTION DOC] HELPER: Create an 8-thread group and queue one alltoall per input
        # list; each output j must hold the value sent by rank j to this rank (iteration
        # times world size plus this rank).
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )

        outputs = [
            [fn(torch.tensor([-1])) for _ in range(self.world_size)]
            for _ in range(len(inputs))
        ]

        future_handles = []
        for i in range(len(inputs)):
            input_tensors = [fn(inputs[i][j]) for j in range(self.world_size)]
            fut = pg.alltoall(outputs[i], input_tensors).get_future()
            future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()

            # Verify correctness: output_tensors[j] should contain the value
            # that rank j sent to this rank (self.rank) at position self.rank in rank j's input.
            # All ranks construct inputs[i] = [tensor([i * world_size + 0]), ..., tensor([i * world_size + world_size-1])]
            # So rank j sends inputs[i][self.rank] = tensor([i * world_size + self.rank]) to this rank.
            # This rank receives that in output_tensors[j].
            for j in range(self.world_size):
                expected_value = torch.tensor([i * self.world_size + self.rank])
                self.assertEqual(
                    expected_value,
                    outputs[i][j],
                    msg=(
                        lambda msg: f"{msg}\nMismatch in iteration {i:d} from rank {j:d}"
                    ),
                )

    @requires_gloo()
    def test_alltoall_stress(self):
        # [FUNCTION DOC] Stress test: 1000 CPU alltoalls.
        inputs = [
            [torch.tensor([i * self.world_size + j]) for j in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_alltoall_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @skipIfRocm
    def test_alltoall_stress_cuda(self):
        # [FUNCTION DOC] Stress test: 1000 CUDA alltoalls (needs 2 GPUs; skipped on ROCm).
        inputs = [
            [torch.tensor([i * self.world_size + j]) for j in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_alltoall_stress(inputs, lambda t: t.clone().cuda())

    def _test_alltoall_data_routing(self, fn):
        # [FUNCTION DOC] HELPER: Verifies routing: rank i sends value i times 100 plus j to
        # rank j, and rank r must receive j times 100 plus r from every rank j.
        """
        Test that data routing is correct: rank i sends inputTensors[j]
        to rank j
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Each rank sends unique values to every other rank
        # Rank i sends tensor with value (i * 100 + j) to rank j
        input_tensors = []
        for j in range(self.world_size):
            value = self.rank * 100 + j
            input_tensors.append(fn(torch.tensor([value], dtype=torch.float32)))

        output_tensors = [
            fn(torch.tensor([-1], dtype=torch.float32)) for _ in range(self.world_size)
        ]

        # Perform all-to-all
        fut = pg.alltoall(output_tensors, input_tensors).get_future()
        fut.wait()

        # Verify: rank i should receive tensor with value (j * 100 + i)
        # from rank j in output_tensors[j]
        for j in range(self.world_size):
            expected_value = j * 100 + self.rank
            actual_value = output_tensors[j].item()
            self.assertEqual(
                expected_value,
                actual_value,
                msg=(
                    lambda msg: f"{msg}\nRank {self.rank}: output_tensors[{j}] = "
                    f"{actual_value}, expected {expected_value}"
                ),
            )

    @requires_gloo()
    def test_alltoall_data_routing(self):
        # [FUNCTION DOC] CPU alltoall routing test.
        self._test_alltoall_data_routing(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_alltoall_data_routing_cuda(self):
        # [FUNCTION DOC] CUDA alltoall routing test (needs 2 GPUs).
        self._test_alltoall_data_routing(lambda t: t.clone().cuda())

    def _test_alltoall_multidim(self, fn):
        # [FUNCTION DOC] HELPER: alltoall with 3 by 4 tensors whose entries encode source,
        # destination, row and column; each received tensor (moved to CPU) must match what
        # its source sent here.
        """Test alltoall with multi-dimensional tensors."""
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # Each rank sends a 3x4 tensor with unique values to each destination
        # Value pattern: rank * 1000 + dest * 100 + row * 10 + col
        input_tensors = []
        for dest in range(self.world_size):
            t = torch.zeros(3, 4, dtype=torch.float32)
            for row in range(3):
                for col in range(4):
                    t[row, col] = self.rank * 1000 + dest * 100 + row * 10 + col
            input_tensors.append(fn(t))

        output_tensors = [
            fn(torch.zeros(3, 4, dtype=torch.float32)) for _ in range(self.world_size)
        ]

        fut = pg.alltoall(output_tensors, input_tensors).get_future()
        fut.wait()

        # Verify: output_tensors[src] should contain what rank src sent to us
        for src in range(self.world_size):
            expected = torch.zeros(3, 4, dtype=torch.float32)
            for row in range(3):
                for col in range(4):
                    expected[row, col] = src * 1000 + self.rank * 100 + row * 10 + col
            self.assertEqual(expected, output_tensors[src].cpu())

    @requires_gloo()
    def test_alltoall_multidim(self):
        # [FUNCTION DOC] CPU multi-dimensional alltoall test.
        self._test_alltoall_multidim(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_alltoall_multidim_cuda(self):
        # [FUNCTION DOC] CUDA multi-dimensional alltoall test (needs 2 GPUs).
        self._test_alltoall_multidim(lambda t: t.clone().cuda())


class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    # [CLASS DOC] Gloo DDP tests running in spawned processes. Mixes the shared DDP test
    # implementations from test_c10d_common into a multi-process case; tests cover device
    # layouts, unused parameters, buffers, comm hooks and related DDP options over a gloo
    # process group.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Run the base setUp, then spawn the worker processes.
        super().setUp()
        self._spawn_processes()

    def _get_process_group(self):
        # [FUNCTION DOC] HELPER: Initializes the default gloo process group from the common
        # store (side effect: global default group) and returns it.
        store = self._get_store()
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    def _test_gloo_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        # [FUNCTION DOC] HELPER: Initializes the default gloo group over a file store, takes
        # the backend for the last device in devices, creates a loopback device on it, then
        # delegates to the shared DDP-with-process-group test with the given device list,
        # device ids and options.
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = devices[-1]
        backend = process_group._get_backend(device)
        backend.create_device(interface=LOOPBACK)
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_gloo()
    def test_gloo_backend_cpu_module(self):
        # [FUNCTION DOC] Gloo DDP test on CPU module.
        self._test_gloo_backend([torch.device("cpu")], None)

    @requires_gloo()
    def test_gloo_backend_cpu_module_grad_is_view(self):
        # [FUNCTION DOC] Gloo DDP test on CPU module with gradient_as_bucket_view enabled.
        self._test_gloo_backend(
            [torch.device("cpu")], None, gradient_as_bucket_view=True
        )

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_gloo_backend_1gpu_module_device_ids_integer_list(self):
        # [FUNCTION DOC] Gloo DDP with one GPU per rank, device ids given as a list of
        # integers.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, int_devices)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_gloo_backend_1gpu_module_device_ids_torch_device_list(self):
        # [FUNCTION DOC] Gloo DDP with one GPU per rank, device ids given as torch devices.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, devices)

    @requires_gloo()
    @skip_if_lt_x_gpu(4)
    def test_gloo_backend_2gpu_module(self):
        # [FUNCTION DOC] Gloo DDP with a module spanning two GPUs per rank (multi_device).
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, None, multi_device=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(8)
    def test_gloo_backend_4gpu_module(self):
        # [FUNCTION DOC] Gloo DDP with a module spanning four GPUs per rank (multi_device).
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, None, multi_device=True)

    def _test_global_local_unused_params_grad(
        self, gradient_as_bucket_view=False, static_graph=False
    ):
        # [FUNCTION DOC] HELPER: Simulates multi-task training with a module using one of
        # two tasks depending on rank. For CPU and GPU DDP wrappers (find_unused_parameters
        # on) it checks that gradients start as None, and after backward the locally used
        # task parameters have gradients on all ranks while the globally unused task
        # parameter stays None.
        """
        By simulating a multi-task training, this test is to make sure:
        1) DDP does not touch the grad of globally unused parameters.
        2) DDP does update the grad of locally unused parameters.
        """

        class GlobalLocalUnusedParamModule(nn.Module):
            # [CLASS DOC] HELPER: Module with three Task submodules; forward uses the first
            # on rank 0 and the second otherwise, so the third is never used.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three Task submodules t0, t1 and
                # task_unused.
                super().__init__()
                self.t0 = Task()
                self.t1 = Task()
                self.task_unused = Task()

            def task_parameters(self):
                # [FUNCTION DOC] HELPER: Returns the tuple of the parameter of each of the
                # three tasks.
                return (self.t0.p, self.t1.p, self.task_unused.p)

            def forward(self, x, rank):
                # [FUNCTION DOC] HELPER: Applies task t0 to the input on rank 0, else task
                # t1.
                return self.t0(x) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            # [FUNCTION DOC] HELPER: Runs a forward and backward on the DDP model, asserting
            # gradients of all task parameters are None before backward, and afterwards the
            # two used ones are set while the unused one is still None.
            # Run forward
            output = model(8, self.rank)

            # The grads of all parameters should be None at this point.
            t0_p, t1_p, task_unused_p = model.module.task_parameters()
            self.assertIsNone(t0_p.grad)
            self.assertIsNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

            # Run backward
            output.mean().backward()

            # Now locally unused parameter should have grad updated on all ranks.
            # However the globally unused parameter should still have None grad.
            self.assertIsNotNone(t0_p.grad)
            self.assertIsNotNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

        process_group = self._get_process_group()

        # Test on CPU
        cpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        run_and_verify_grad(cpu_model)

        # Test on GPU
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        run_and_verify_grad(gpu_model)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad(self):
        # [FUNCTION DOC] Unused-parameter gradient test with default options (needs 2 GPUs).
        self._test_global_local_unused_params_grad()

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_grad_is_view(self):
        # [FUNCTION DOC] Unused-parameter gradient test with gradient_as_bucket_view (needs
        # 2 GPUs).
        self._test_global_local_unused_params_grad(gradient_as_bucket_view=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_static_graph(self):
        # [FUNCTION DOC] Unused-parameter gradient test with static_graph (needs 2 GPUs).
        self._test_global_local_unused_params_grad(static_graph=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_when_unused_parameters_empty(self):
        # [FUNCTION DOC] Checks DDP allreduces unused parameters correctly when one rank
        # (rank 0) uses all parameters, so its unused-parameter list is empty, while other
        # ranks skip some. Needs 2 GPUs.
        """
        An empty unused_parameters array does not imply find_unused_parameters =
        false. This test makes sure that DDP allreduces unused parameters
        accordingly where the forward pass in some process uses all parameters.
        This unit test creates a module that uses all parameters in rank = 0, and
        has unused parameters in other ranks.
        """

        class FindUnusedParamModule(nn.Module):
            # [CLASS DOC] HELPER: Module with two Task submodules; rank 0 applies both in
            # sequence, other ranks only the second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates Task submodules t0 and t1.
                super().__init__()
                self.t0 = Task()
                self.t1 = Task()

            def task_parameters(self):
                # [FUNCTION DOC] HELPER: Returns the tuple of both tasks' parameters.
                return (self.t0.p, self.t1.p)

            def forward(self, x, rank):
                # [FUNCTION DOC] HELPER: Rank 0 applies t0 then t1 to the input; other ranks
                # apply only t1.
                return self.t1(self.t0(x)) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            # [FUNCTION DOC] HELPER: Runs forward and backward on the DDP model; checks
            # every task parameter has no gradient before backward and has a gradient on
            # every rank afterwards.
            # Run forward
            output = model(8, self.rank)

            # The grads of all parameters should be None at this point.
            [self.assertIsNone(t_p.grad) for t_p in model.module.task_parameters()]

            # Run backward
            output.mean().backward()

            # Now locally unused parameter should have grad updated on all ranks.
            [self.assertIsNotNone(t_p.grad) for t_p in model.module.task_parameters()]

        process_group = self._get_process_group()

        # Test on CPU
        cpu_model = DistributedDataParallel(
            FindUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(cpu_model)

        # Test on GPU
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            FindUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(gpu_model)

    @requires_gloo()
    def test_ignored_output(self):
        # [FUNCTION DOC] Builds a small DDP model on the default gloo group and runs 4
        # forward passes whose outputs are discarded (no backward required), then 4
        # iterations with cross-entropy loss and backward, checking nothing hangs or errors.
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called.
        """
        process_group = self._get_process_group()

        class IgnoredOutput(nn.Module):
            # [CLASS DOC] HELPER: Two-layer bias-free linear network with ReLU after each
            # layer and a softmax over dim 1 on the output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1 (2 to 10), fc2 (10 to 4), both without
                # bias, and a ReLU.
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1, ReLU, fc2, ReLU and returns the
                # softmax over dim 1.
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        model = DistributedDataParallel(
            IgnoredOutput().float(),
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # Run a few iterations where we ignore the output.
        for _ in range(4):
            output = model(input)
            del output

        # Run a few iterations where we use the output.
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_gloo()
    def test_ignored_output_with_unused_parameters(self):
        # [FUNCTION DOC] Like the ignored-output test but with find_unused_parameters
        # enabled and a module containing an unused third linear layer; ignored forward
        # outputs must not require backward, and subsequent normal iterations must succeed.
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called, if not all model
        parameters participated in computing the model output.
        """
        process_group = self._get_process_group()

        class IgnoredOutputWithUnusedParameters(nn.Module):
            # [CLASS DOC] HELPER: Same network as the ignored-output module but also owns an
            # unused fc3 layer that forward never touches.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1, fc2, an unused fc3 (all bias-free
                # linear) and a ReLU.
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1, ReLU, fc2, ReLU and returns softmax;
                # fc3 is unused.
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        model = DistributedDataParallel(
            IgnoredOutputWithUnusedParameters().float(),
            process_group=process_group,
            find_unused_parameters=True,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # Run a few iterations where we ignore the output.
        for _ in range(4):
            output = model(input)
            del output

        # Run a few iterations where we use the output.
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_ignored_sharded_tensor(self):
        # [FUNCTION DOC] Checks that DDP construction succeeds when the module holds a
        # ShardedTensor parameter that is registered as ignored. Initializes a gloo group
        # from a file init method, builds a sharded tensor from per-rank local CUDA shards,
        # marks it ignored and constructs DDP with static graph and bucket-view grads. Needs
        # 2 GPUs.
        # KNOWN ISSUE: the device type is compared to the string gpu, which never matches
        # cuda, so device_ids is always None here.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Linear layer plus a sharded-tensor parameter; forward only
            # uses the linear layer, ReLU and softmax.
            def __init__(self, shard_tensor: ShardedTensor) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1, wraps the given sharded tensor as a
                # parameter named st, and a ReLU.
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.st = nn.Parameter(shard_tensor)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1 and ReLU and returns the softmax over
                # dim 1.
                x = self.relu(self.fc1(x))
                return F.softmax(x, dim=1)

        pg = dist.init_process_group(
            "gloo",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )
        device = torch.device(f"cuda:{self.rank}")
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank % 2) * 5, 0],
            shard_sizes=[5, 10],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )
        local_shards = [Shard(torch.randn(5, 10, device=device), local_shard_metadata)]
        st = init_from_local_shards(local_shards, [10, 10])
        m = MyModule(st)
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            module=m, params_and_buffers_to_ignore={"st"}
        )
        # test to make DDP constructor will not fail when module includes a ShardedTensor when ignored
        DistributedDataParallel(
            m,
            device_ids=[device] if device.type == "gpu" else None,
            process_group=pg,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            static_graph=True,
        )

    def _run_and_verify_sparse_gradients(self, vanilla_model, ddp_model):
        # [FUNCTION DOC] HELPER: Runs the vanilla model on the full batch and the DDP model
        # on this rank's slice of it, both with cross-entropy backward, then checks the
        # first parameter's gradients (coalesced) are equal, i.e. DDP averaged the sparse
        # gradients correctly.
        mult = 2
        batch_size = mult * self.world_size
        criterion = nn.CrossEntropyLoss()
        input = torch.randint(0, 10, [batch_size, 2])
        target = torch.randint(0, 10, [batch_size])

        # Run with entire batch against single process version
        criterion(vanilla_model(input), target).backward()

        # Run with partial batch against multi process version
        partial_input = input.split(mult)[self.rank]
        partial_target = target.split(mult)[self.rank]
        criterion(ddp_model(partial_input), partial_target).backward()

        # Check that the gradients are sparse and identical
        vanilla_parameter = next(vanilla_model.parameters())
        ddp_parameter = next(ddp_model.parameters())
        self.assertEqual(
            vanilla_parameter.grad.coalesce(), ddp_parameter.grad.coalesce()
        )

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_save_load_checkpoint(self):
        # [FUNCTION DOC] Trains three copies of a small network on GPU (a DDP model that
        # gets checkpointed, a plain copy reloaded from the same state dict with the module
        # prefix removed, and a DDP model trained without interruption), all starting from
        # zeroed parameters. Rank 0 saves the DDP state dict to a temp-directory file and
        # all ranks reload it after a barrier. After equal total steps all parameters must
        # match. Needs 2 GPUs.
        dist.init_process_group(
            "gloo",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        class TestModel(nn.Module):
            # [CLASS DOC] HELPER: Same network as in the ignored-output tests (fc1, fc2,
            # ReLU, softmax).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1, fc2 (bias-free) and a ReLU.
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1, ReLU, fc2, ReLU and returns the
                # softmax.
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        def train_loop(model, optimizer, iterations):
            # [FUNCTION DOC] HELPER: Runs the given number of SGD-style steps (zero grad,
            # forward on the enclosing input and target, loss, backward, step) on a model.
            for _ in range(iterations):
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        device_id = gpus_for_rank(self.world_size)[self.rank][0]

        model_withload = TestModel().float().to(device_id)
        model_withoutload = TestModel().float().to(device_id)

        ddp_withload = DistributedDataParallel(
            model_withload,
            device_ids=[device_id],
        )
        ddp_withoutload = DistributedDataParallel(
            model_withoutload,
            device_ids=[device_id],
        )

        # ensure that all the three models start with the same set of parameters. By default they are randomized on construction
        for p in ddp_withload.parameters():
            with torch.no_grad():
                p.zero_()
        for p in model_withload.parameters():
            with torch.no_grad():
                p.zero_()
        for p in ddp_withoutload.parameters():
            with torch.no_grad():
                p.zero_()

        batch_size = 4
        criterion = nn.CrossEntropyLoss()

        optimizer_withload = torch.optim.SGD(ddp_withload.parameters(), lr=0.001)
        optimizer_non_ddp_withload = torch.optim.SGD(
            model_withload.parameters(), lr=0.001
        )
        optimizer_withoutload = torch.optim.SGD(ddp_withoutload.parameters(), lr=0.001)

        input = torch.rand([batch_size, 2], dtype=torch.float).to(device_id)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # run the model for 6 iterations, with a checkpoint in the middle
        train_loop(ddp_withload, optimizer_withload, 3)

        # zero out parameters of both DDP and non-DDP models and reload them from the DDP state dict
        checkpoint_path = tempfile.gettempdir() + "/model.checkpoint"
        if self.rank == 0:
            torch.save(ddp_withload.state_dict(), checkpoint_path)

        dist.barrier()
        map_location = {"cuda:0": f"cuda:{self.rank:d}"}
        ddp_state_dict = torch.load(checkpoint_path, map_location=map_location)

        for model in [ddp_withload, model_withload]:
            for p in model.parameters():
                with torch.no_grad():
                    p.zero_()
        ddp_withload.load_state_dict(ddp_state_dict)
        # the non-DDP model needs to first remove the prefix of "module." from the DDP state dict
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            ddp_state_dict, "module."
        )
        model_withload.load_state_dict(ddp_state_dict)

        train_loop(ddp_withload, optimizer_withload, 3)
        train_loop(model_withload, optimizer_non_ddp_withload, 3)

        # re-run the model with the same inputs for 6 iterations with no checkpoint
        train_loop(ddp_withoutload, optimizer_withoutload, 6)

        for p_withload, p_withoutload, p_non_ddp_withload in zip(
            ddp_withload.parameters(),
            ddp_withoutload.parameters(),
            model_withload.parameters(),
        ):
            self.assertEqual(p_withload, p_withoutload)
            self.assertEqual(p_non_ddp_withload, p_withoutload)

    def _test_sparse_gradients(
        self, gradient_as_bucket_view=False, batched_grad_copy=False
    ):
        # [FUNCTION DOC] HELPER: Seeds the RNG identically on all ranks, wraps a deep copy
        # of a sparse embedding module in DDP with the given gradient_as_bucket_view and
        # batched_grad_copy options, then compares sparse gradients against the single-
        # process model.
        process_group = self._get_process_group()

        # Ensure initialized weights and inputs are identical across processes
        torch.manual_seed(1337)

        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
            batched_grad_copy=batched_grad_copy,
        )

        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

    @requires_gloo()
    def test_sparse_gradients(self):
        # [FUNCTION DOC] Sparse gradient test with default DDP options.
        self._test_sparse_gradients()

    @requires_gloo()
    def test_sparse_gradients_grad_is_view(self):
        # [FUNCTION DOC] Sparse gradient test with gradient_as_bucket_view.
        self._test_sparse_gradients(gradient_as_bucket_view=True)

    @requires_gloo()
    def test_sparse_gradients_batched_grad_copy(self):
        # [FUNCTION DOC] Sparse gradient test with batched_grad_copy.
        self._test_sparse_gradients(batched_grad_copy=True)

    @requires_gloo()
    def test_sparse_gradients_grad_is_view_batched_grad_copy(self):
        # [FUNCTION DOC] Sparse gradient test with both bucket-view grads and
        # batched_grad_copy.
        self._test_sparse_gradients(
            gradient_as_bucket_view=True, batched_grad_copy=True
        )

    @requires_gloo()
    def test_static_graph_batched_grad_copy(self):
        # [FUNCTION DOC] Trains a double-precision linear layer both locally on the full
        # batch and in DDP with static graph and batched_grad_copy on per-rank slices, for 3
        # iterations with SGD, checking gradients each step and final parameters match.
        """Test batched_grad_copy with static_graph (delay_all_reduce path)."""
        process_group = self._get_process_group()
        torch.manual_seed(42)

        vanilla_model = nn.Linear(2, 4, bias=False).double()
        model_bat = copy.deepcopy(vanilla_model)

        ddp_bat = DistributedDataParallel(
            model_bat,
            process_group=process_group,
            static_graph=True,
            batched_grad_copy=True,
        )

        opt_van = torch.optim.SGD(vanilla_model.parameters(), lr=0.01)
        opt_bat = torch.optim.SGD(ddp_bat.parameters(), lr=0.01)

        mult = 2
        # Run multiple iterations to cover both static_graph first iteration
        # (delay_all_reduce path) and subsequent iterations.
        for _ in range(3):
            batch_size = mult * self.world_size
            input = torch.rand(batch_size, 2, dtype=torch.double)

            # Vanilla model on full batch (mean normalizes by batch size,
            # matching DDP's allreduce averaging across ranks)
            opt_van.zero_grad(set_to_none=True)
            vanilla_model(input).mean().backward()

            # DDP model on partial batch — allreduce averages gradients
            partial_input = input.split(mult)[self.rank]
            opt_bat.zero_grad(set_to_none=True)
            ddp_bat(partial_input).mean().backward()

            for p_van, p_bat in zip(vanilla_model.parameters(), ddp_bat.parameters()):
                self.assertEqual(p_van.grad, p_bat.grad)

            opt_van.step()
            opt_bat.step()

        for p_van, p_bat in zip(vanilla_model.parameters(), ddp_bat.parameters()):
            self.assertEqual(p_van, p_bat)

    @requires_gloo()
    def test_ddp_comm_hook_future_passing_cpu(self):
        # [FUNCTION DOC] Registers the simple test hook (which makes the gradient bucket the
        # constant result) on a CPU DDP model and checks gradients equal the hook's output
        # rather than the default average. Creates an unused file store.
        """
        This unit test verifies whether the Future object is passed properly.
        The callback function creates a Future object and sets a value to it.
        """
        store = c10d.FileStore(self.file_name, self.world_size)  # noqa: F841

        process_group = self._get_process_group()

        # Test on CPU
        cpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().cpu(), process_group=process_group
        )

        # Register DDP Communication Hook
        cpu_model.register_comm_hook(None, self._simple_hook)

        # check whether the grads are equal to what then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(cpu_model, 8, 2 * torch.ones(2, 2))

    def _gpu_model_with_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False, state=None
    ):
        # [FUNCTION DOC] HELPER: Builds a GPU DDP model (first GPU assigned to this rank) of
        # the small hook-test module with the given process group and bucket-view option;
        # registers the hook with the given state if one is supplied. Returns the model.
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Register a DDP communication hook if any.
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_gloo(self):
        # [FUNCTION DOC] Same as the CPU future-passing test but on a GPU model (needs 2
        # GPUs).
        """
        This unit test verifies whether the Future object is passed properly using gloo backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        process_group = self._get_process_group()

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    @requires_gloo()
    def test_ddp_invalid_comm_hook_init(self):
        # [FUNCTION DOC] Checks register_comm_hook rejects a non-callable hook with
        # TypeError and a hook whose bucket parameter is annotated with a wrong type with
        # ValueError.
        """
        This unit test makes sure that register_comm_hook properly checks the format
        of hook defined by user. The Python hook must be callable. This test also
        checks whether bucket annotation checked properly if defined.
        """
        process_group = self._get_process_group()

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        with self.assertRaisesRegex(TypeError, "Communication hook must be callable."):
            model.register_comm_hook(state=None, hook=1)

        with self.assertRaisesRegex(
            ValueError, "bucket annotation should be dist.GradBucket."
        ):

            def comm_hook(
                state: object, bucket: int
            ) -> torch.futures.Future[torch.Tensor]:
                # [FUNCTION DOC] HELPER: Hook with an incorrect bucket annotation (int);
                # returns a new Future. Never called by the test, only inspected at
                # registration.
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

    @requires_gloo()
    def test_ddp_invalid_comm_hook_return_type(self):
        # [FUNCTION DOC] Checks a hook annotated to return int raises ValueError at
        # registration and the error is logged in DDP logging data; checks that an
        # unannotated hook returning a non-Future makes backward raise RuntimeError about
        # the callback return type.
        """
        This test checks whether return annotation checked properly if defined. It also
        checks whether an internal error is thrown if return type is incorrect and user
        hasn't specified any return type annotation.
        """
        process_group = self._get_process_group()

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        expected_err = (
            "Communication hook: return annotation should be torch.futures.Future"
        )
        with self.assertRaisesRegex(
            ValueError,
            expected_err,
        ):

            def comm_hook(state: object, bucket: dist.GradBucket) -> int:
                # [FUNCTION DOC] HELPER: Hook with a wrong return annotation (int) that
                # returns a new Future.
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

        verify_ddp_error_logged(model, expected_err)

        with self.assertRaisesRegex(
            RuntimeError,
            "callback must return a torch.futures.Future object, but got",
        ):

            def comm_hook(state: object, bucket: dist.GradBucket):
                # [FUNCTION DOC] HELPER: Hook without annotations that wrongly returns the
                # integer 1.
                return 1

            model.register_comm_hook(state=None, hook=comm_hook)

            # Run forward
            output = model(8, self.rank)

            # Run backward
            output.mean().backward()

    @requires_gloo()
    def test_ddp_comm_hook_register_just_once(self):
        # [FUNCTION DOC] Registers a hook once on a DDP model, then checks registering again
        # raises RuntimeError saying hooks can only be registered once.
        """
        DDP communication hook can only be registered once. This test validates whether
        the error is thrown properly when register_comm_hook is called more than once.
        """
        process_group = self._get_process_group()

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        def dummy_hook(state, bucket):
            # [FUNCTION DOC] HELPER: Returns a future already resolved with a list holding
            # the bucket buffer.
            fut = torch.futures.Future()
            fut.set_result([bucket.buffer()])
            return fut

        model.register_comm_hook(None, dummy_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "register_comm_hook or register_builtin_comm_hook can only be called once.",
        ):
            model.register_comm_hook(None, dummy_hook)

    @requires_gloo()
    def test_ddp_comm_hook_sparse_gradients(self):
        # [FUNCTION DOC] Runs the sparse gradient comparison with a DDP communication hook
        # that allreduces the bucket buffer on the process group and divides the result by
        # world size.
        """
        Runs "test_sparse_gradients" unit test with DDP communication hook. We define a
        simple hook that does allreduce and works with gloo backend for this test.
        """
        process_group = self._get_process_group()

        # Ensure initialized weights and inputs are identical across processes
        torch.manual_seed(1337)

        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
        )

        def allreduce_hook_gloo(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            # [FUNCTION DOC] HELPER: Hook that starts an async allreduce on the bucket
            # buffer and returns a future that divides the first result tensor by world
            # size.
            # KNOWN ISSUE: a comment in the body says divide by 2 times world size but the
            # code only divides by world size.
            def div_by_world_size(fut):
                # [FUNCTION DOC] HELPER: Waits on the future and returns its first tensor
                # divided by world size.
                # Divide the result by 2 * world_size.
                return fut.wait()[0] / self.world_size

            # Prepare allreduced grad bucket tensors by running an async work.
            fut = process_group.allreduce([bucket.buffer()]).get_future()
            return fut.then(div_by_world_size)

        ddp_model.register_comm_hook(None, allreduce_hook_gloo)

        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

    @requires_gloo()
    def test_ddp_complex_params(self):
        # [FUNCTION DOC] Wraps an FFT model with complex parameters in DDP, runs one
        # forward, summed loss, backward and Adam step to check complex parameters work with
        # the gloo reducer.
        process_group = self._get_process_group()
        N, C, H, W = 1, 16, 64, 64
        ddp_model = DistributedDataParallel(
            FFTModel(hin=H, win=W, n_features=C),
            process_group=process_group,
        )
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

        inp = torch.ones((N, C, H, W), dtype=torch.float32)

        # train step
        out = ddp_model(inp)
        loss = torch.sum(out)
        loss.backward()
        optimizer.step()


class ReducerModule(nn.Module):
    # [CLASS DOC] HELPER: Network with three bias-free linear layers (2 to 10, 10 to 4, 4 to
    # 4) used to exercise the reducer; the last layer can be skipped in forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates fc1, fc2, fc3 (bias-free) and a ReLU.
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 4, bias=False)
        self.fc3 = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, use_fc3=True):
        # [FUNCTION DOC] HELPER: Applies fc1 and fc2 each followed by ReLU and a cast to
        # float, then fc3 (cast to float) if use_fc3 is true, and returns softmax over dim
        # 1.
        x = self.relu(self.fc1(x)).float()
        x = self.relu(self.fc2(x)).float()
        if use_fc3:
            x = self.fc3(x).float()
        return F.softmax(x, dim=1)


class ReducerTest(TestCase):
    # [CLASS DOC] Single-process tests of the DDP Reducer on a one-rank gloo group. The
    # default process group is created in setUp from a temporary file store and stored in
    # self.process_group.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Creates a temp file, a one-rank file store on it,
        # initializes the default gloo group and stores the group.
        super().setUp()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file = f
        world_size = 1
        self.store = c10d.FileStore(self.file.name, world_size)
        c10d.init_process_group(
            backend="gloo", store=self.store, rank=0, world_size=world_size
        )
        self.process_group = c10d.distributed_c10d._get_default_group()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Destroys the process group and removes the temp file,
        # printing the error if removal fails.
        c10d.destroy_process_group()
        try:
            os.remove(self.file.name)
        except OSError as e:
            print(str(e))

    @requires_gloo()
    def test_single_dtype_single_bucket(self):
        # [FUNCTION DOC] Builds a Reducer with all parameters in one bucket (single dtype);
        # must succeed.
        model = ReducerModule()
        parameters = list(model.parameters())
        buckets = [list(range(len(parameters)))]
        dist.Reducer(
            parameters, buckets, [dist._DEFAULT_FIRST_BUCKET_BYTES], self.process_group
        )

    def _create_mixed_precision_model(self):
        # [FUNCTION DOC] HELPER: Returns the reducer module cast to float with the first
        # layer in double.
        model = ReducerModule()
        model.float()
        model.fc1.double()
        return model

    @requires_gloo()
    def test_multi_dtype_single_bucket(self):
        # [FUNCTION DOC] Checks that building a Reducer with one bucket for parameters of
        # mixed dtypes raises RuntimeError.
        model = self._create_mixed_precision_model()

        # Raise if there are multiple types per bucket.
        # In this case we create one bucket for all parameters.
        with self.assertRaises(RuntimeError):
            parameters = list(model.parameters())
            buckets = [list(range(len(parameters)))]
            dist.Reducer(
                parameters,
                buckets,
                [dist._DEFAULT_FIRST_BUCKET_BYTES],
                self.process_group,
            )

    @requires_gloo()
    def test_multi_dtype_multi_bucket(self):
        # [FUNCTION DOC] Builds a Reducer with one bucket per consecutive run of identical
        # dtype; must succeed.
        model = self._create_mixed_precision_model()
        parameters = list(model.parameters())
        group_by_dtype = groupby(
            range(len(parameters)), key=lambda i: parameters[i].dtype
        )
        buckets = [list(indices) for _, indices in group_by_dtype]
        dist.Reducer(
            parameters,
            buckets,
            [dist._DEFAULT_FIRST_BUCKET_BYTES for _ in buckets],
            self.process_group,
        )

    def _create_reducer_for_models(self, models, find_unused_parameters=False):
        # [FUNCTION DOC] HELPER: Asserts exactly one model is passed, buckets its parameters
        # by consecutive runs of the same dtype, and returns a Reducer on the test process
        # group with the given find_unused_parameters flag.
        self.assertEqual(len(models), 1)
        parameters = list(models[0].parameters())
        group_by_dtype = groupby(
            range(len(parameters)), key=lambda i: parameters[i].dtype
        )
        buckets = [list(indices) for _, indices in group_by_dtype]
        return dist.Reducer(
            parameters,
            buckets,
            [dist._DEFAULT_FIRST_BUCKET_BYTES for _ in range(len(buckets))],
            self.process_group,
            find_unused_parameters=find_unused_parameters,
        )

    @requires_gloo()
    def test_forward_backward(self):
        # [FUNCTION DOC] Mixed-precision model with a Reducer: prepare_for_forward, forward,
        # cross-entropy, prepare_for_backward and backward must run without error.
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model])
        reducer.prepare_for_forward()
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input), target)
        reducer.prepare_for_backward(output)
        output.backward()

    @requires_gloo()
    def test_forward_backward_unused_parameters(self):
        # [FUNCTION DOC] Reducer with find_unused_parameters: forward skips the last layer;
        # its gradient is None before and after backward because it is globally unused.
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        reducer.prepare_for_forward()
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input, use_fc3=False), target)

        # Check that the grad of fc3 is not set.
        self.assertEqual(None, model.fc3.weight.grad)

        # Compute and accumulate gradients.
        reducer.prepare_for_backward(output)
        output.backward()

        # The reducer will have marked the grad of fc3 as ready, because
        # it doesn't show up in the autograd graph of `output`. Since fc3.weight
        # is considered being globally unused, it will be kept untouched as None.
        self.assertEqual(None, model.fc3.weight.grad)

    @requires_gloo()
    def test_forward_backward_optimizer(self):
        # [FUNCTION DOC] Runs 3 training iterations with Adam through a Reducer (find_unused
        # on), with the last layer unused only in the first; zero_grad each time must not
        # break reducer buckets.
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        reducer.prepare_for_forward()
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        for i in range(3):
            input = torch.rand([batch_size, 2], dtype=torch.double)
            target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

            # The `zero_grad` function calls `detach_` and `zero_` on the grad
            # tensors of model parameters. If we tried to set the grad tensors
            # to a view of the reducer's bucket tensors, this would blow up.
            optimizer.zero_grad()

            # Unused parameter only in the first iteration.
            output = loss(model(input, use_fc3=(i > 0)), target)
            reducer.prepare_for_backward(output)
            output.backward()
            optimizer.step()

    def _create_reducer(
        self,
        model,
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
        batched_grad_copy=False,
    ):
        # [FUNCTION DOC] HELPER: Like the models-based reducer creator but for one model;
        # buckets parameters by consecutive dtype runs and forwards find_unused_parameters,
        # gradient_as_bucket_view and batched_grad_copy to the Reducer.
        parameters = list(model.parameters())
        group_by_dtype = groupby(
            range(len(parameters)), key=lambda i: parameters[i].dtype
        )
        buckets = [list(indices) for _, indices in group_by_dtype]
        return dist.Reducer(
            parameters,
            buckets,
            [dist._DEFAULT_FIRST_BUCKET_BYTES for _ in range(len(buckets))],
            self.process_group,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            batched_grad_copy=batched_grad_copy,
        )

    def _run_forward_backward(self, model, reducer, batch_size=10):
        # [FUNCTION DOC] HELPER: Runs one forward and backward pass with random double input
        # and random class targets through the reducer; returns the loss tensor.
        """Run one forward+backward pass, return loss."""
        loss_fn = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss_fn(model(input), target)
        reducer.prepare_for_backward(output)
        output.backward()
        return output

    @requires_gloo()
    def test_batched_grad_copy_basic(self):
        # [FUNCTION DOC] Profiles one backward with batched_grad_copy and checks the batched
        # foreach copy operator appears, the per-parameter mul_out operator does not, and
        # gradients are non-zero.
        """Verify batched_grad_copy uses _foreach_copy_ and produces grads."""
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer(model, batched_grad_copy=True)
        reducer.prepare_for_forward()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
        ) as prof:
            self._run_forward_backward(model, reducer)

        events = [e.key for e in prof.key_averages()]
        self.assertIn("aten::_foreach_copy_", events)
        # mul_out is the per-param copy+div — should NOT appear when batched
        self.assertNotIn("torch::distributed::reducer::mul_out", events)

        for p in model.parameters():
            if p.grad is not None:
                self.assertFalse(torch.all(p.grad == 0))

    @requires_gloo()
    def test_batched_grad_copy_matches_default(self):
        # [FUNCTION DOC] Builds one reducer with and one without batched grad copy over
        # identical seeded model copies, runs the same input through both, and checks all
        # gradients are equal.
        """Verify grads are numerically identical with and without batching."""
        torch.manual_seed(42)
        model_ref = self._create_mixed_precision_model()
        model_bat = copy.deepcopy(model_ref)

        input = torch.rand([10, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(10)])
        loss_fn = nn.CrossEntropyLoss()

        # Reference: no batching
        reducer_ref = self._create_reducer(model_ref)
        reducer_ref.prepare_for_forward()
        out_ref = loss_fn(model_ref(input), target)
        reducer_ref.prepare_for_backward(out_ref)
        out_ref.backward()

        # Batched
        reducer_bat = self._create_reducer(model_bat, batched_grad_copy=True)
        reducer_bat.prepare_for_forward()
        out_bat = loss_fn(model_bat(input), target)
        reducer_bat.prepare_for_backward(out_bat)
        out_bat.backward()

        for p_ref, p_bat in zip(model_ref.parameters(), model_bat.parameters()):
            self.assertEqual(p_ref.grad, p_bat.grad)

    @requires_gloo()
    def test_batched_grad_copy_with_set_to_none(self):
        # [FUNCTION DOC] Trains two identical models for 3 SGD iterations, one reducer plain
        # and one with batched grad copy, zeroing gradients to None each step; gradients
        # must match every iteration and parameters must match at the end.
        """Multi-iteration with set_to_none=True — the primary use case."""
        torch.manual_seed(42)
        model_ref = self._create_mixed_precision_model()
        model_bat = copy.deepcopy(model_ref)

        opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)
        opt_bat = torch.optim.SGD(model_bat.parameters(), lr=0.01)

        reducer_ref = self._create_reducer(model_ref)
        reducer_bat = self._create_reducer(model_bat, batched_grad_copy=True)
        loss_fn = nn.CrossEntropyLoss()

        for i in range(3):
            input = torch.rand([10, 2], dtype=torch.double)
            target = torch.LongTensor([random.randrange(4) for _ in range(10)])

            # set_to_none=True is the default
            opt_ref.zero_grad(set_to_none=True)
            opt_bat.zero_grad(set_to_none=True)

            reducer_ref.prepare_for_forward()
            out_ref = loss_fn(model_ref(input), target)
            reducer_ref.prepare_for_backward(out_ref)
            out_ref.backward()

            reducer_bat.prepare_for_forward()
            out_bat = loss_fn(model_bat(input), target)
            reducer_bat.prepare_for_backward(out_bat)
            out_bat.backward()

            for p_ref, p_bat in zip(model_ref.parameters(), model_bat.parameters()):
                self.assertEqual(p_ref.grad, p_bat.grad)

            opt_ref.step()
            opt_bat.step()

        # Params should be identical after training
        for p_ref, p_bat in zip(model_ref.parameters(), model_bat.parameters()):
            self.assertEqual(p_ref, p_bat)

    @requires_gloo()
    def test_batched_grad_copy_with_bucket_view(self):
        # [FUNCTION DOC] Same comparison as the set-to-none test but both reducers use
        # gradient_as_bucket_view, so gradients alias bucket memory after the first
        # iteration.
        """Test batched_grad_copy + gradient_as_bucket_view (alias path)."""
        torch.manual_seed(42)
        model_ref = self._create_mixed_precision_model()
        model_bat = copy.deepcopy(model_ref)

        opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)
        opt_bat = torch.optim.SGD(model_bat.parameters(), lr=0.01)

        reducer_ref = self._create_reducer(model_ref, gradient_as_bucket_view=True)
        reducer_bat = self._create_reducer(
            model_bat,
            gradient_as_bucket_view=True,
            batched_grad_copy=True,
        )
        loss_fn = nn.CrossEntropyLoss()

        for i in range(3):
            input = torch.rand([10, 2], dtype=torch.double)
            target = torch.LongTensor([random.randrange(4) for _ in range(10)])

            # First iteration: grads undefined (set_to_none default).
            # Subsequent iterations: grads alias bucket views.
            opt_ref.zero_grad(set_to_none=True)
            opt_bat.zero_grad(set_to_none=True)

            reducer_ref.prepare_for_forward()
            out_ref = loss_fn(model_ref(input), target)
            reducer_ref.prepare_for_backward(out_ref)
            out_ref.backward()

            reducer_bat.prepare_for_forward()
            out_bat = loss_fn(model_bat(input), target)
            reducer_bat.prepare_for_backward(out_bat)
            out_bat.backward()

            for p_ref, p_bat in zip(model_ref.parameters(), model_bat.parameters()):
                self.assertEqual(p_ref.grad, p_bat.grad)

            opt_ref.step()
            opt_bat.step()

        for p_ref, p_bat in zip(model_ref.parameters(), model_bat.parameters()):
            self.assertEqual(p_ref, p_bat)

    @requires_gloo()
    def test_batched_grad_copy_with_unused_parameters(self):
        # [FUNCTION DOC] Compares plain and batched reducers (find_unused_parameters on)
        # when the last layer is skipped in forward: its gradient must stay None in both and
        # other gradients match.
        """Test batched_grad_copy + find_unused_parameters with fc3 unused."""
        torch.manual_seed(42)
        model_ref = self._create_mixed_precision_model()
        model_bat = copy.deepcopy(model_ref)

        reducer_ref = self._create_reducer(model_ref, find_unused_parameters=True)
        reducer_bat = self._create_reducer(
            model_bat, find_unused_parameters=True, batched_grad_copy=True
        )
        loss_fn = nn.CrossEntropyLoss()

        input = torch.rand([10, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(10)])

        reducer_ref.prepare_for_forward()
        out_ref = loss_fn(model_ref(input, use_fc3=False), target)
        reducer_ref.prepare_for_backward(out_ref)
        out_ref.backward()

        reducer_bat.prepare_for_forward()
        out_bat = loss_fn(model_bat(input, use_fc3=False), target)
        reducer_bat.prepare_for_backward(out_bat)
        out_bat.backward()

        # fc3 is unused — its grad should remain None in both cases
        self.assertIsNone(model_ref.fc3.weight.grad)
        self.assertIsNone(model_bat.fc3.weight.grad)

        # Used parameters should have identical grads
        for (name_ref, p_ref), (name_bat, p_bat) in zip(
            model_ref.named_parameters(), model_bat.named_parameters()
        ):
            if p_ref.grad is not None:
                self.assertEqual(p_ref.grad, p_bat.grad, msg=name_ref)

    @requires_gloo()
    def test_batched_grad_copy_with_create_graph(self):
        # [FUNCTION DOC] Runs backward with create_graph on a plain reducer and on a batched
        # reducer (identically seeded), capturing any RuntimeError message from each; both
        # must behave the same (same error text or both None), since batching falls back in
        # that mode.
        """batched_grad_copy falls back to non-batched path with create_graph=True."""
        torch.manual_seed(42)

        # Run with batched_grad_copy=False (reference)
        model_ref = self._create_mixed_precision_model()
        reducer_ref = self._create_reducer(model_ref)
        loss_fn = nn.CrossEntropyLoss()
        input = torch.rand([10, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(10)])

        reducer_ref.prepare_for_forward()
        out_ref = loss_fn(model_ref(input), target)
        reducer_ref.prepare_for_backward(out_ref)
        ref_error = None
        try:
            out_ref.backward(create_graph=True)
        except RuntimeError as e:
            ref_error = str(e)

        # Run with batched_grad_copy=True — should fall back to same path
        torch.manual_seed(42)
        model_bat = self._create_mixed_precision_model()
        reducer_bat = self._create_reducer(model_bat, batched_grad_copy=True)

        reducer_bat.prepare_for_forward()
        out_bat = loss_fn(model_bat(input), target)
        reducer_bat.prepare_for_backward(out_bat)
        bat_error = None
        try:
            out_bat.backward(create_graph=True)
        except RuntimeError as e:
            bat_error = str(e)

        # Both should behave identically (either both succeed or both fail)
        self.assertEqual(ref_error, bat_error)

    @requires_gloo()
    def test_batched_grad_copy_with_comm_hook(self):
        # [FUNCTION DOC] Registers a simple hook on both a plain and a batched reducer
        # (world size one), runs the same forward and backward, and checks gradients are
        # equal.
        """batched_grad_copy works correctly when a comm_hook is registered."""
        torch.manual_seed(42)
        model_ref = self._create_mixed_precision_model()
        model_bat = copy.deepcopy(model_ref)

        reducer_ref = self._create_reducer(model_ref)
        reducer_bat = self._create_reducer(model_bat, batched_grad_copy=True)

        # A simple comm hook that just allreduces and divides by world_size.
        def simple_hook(state, bucket):
            # [FUNCTION DOC] HELPER: Hook returning a resolved future holding the bucket
            # buffer divided by the state value (no allreduce is performed despite the
            # comment).
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer() / state)
            return fut

        world_size = 1  # single-process test
        dist._register_comm_hook(reducer_ref, world_size, simple_hook)
        dist._register_comm_hook(reducer_bat, world_size, simple_hook)

        loss_fn = nn.CrossEntropyLoss()
        input = torch.rand([10, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(10)])

        reducer_ref.prepare_for_forward()
        out_ref = loss_fn(model_ref(input), target)
        reducer_ref.prepare_for_backward(out_ref)
        out_ref.backward()

        reducer_bat.prepare_for_forward()
        out_bat = loss_fn(model_bat(input), target)
        reducer_bat.prepare_for_backward(out_bat)
        out_bat.backward()

        for p_ref, p_bat in zip(model_ref.parameters(), model_bat.parameters()):
            self.assertEqual(p_ref.grad, p_bat.grad)


@skip_if_win32()
class ProcessGroupGlooLazyInitTest(ProcessGroupGlooTest):
    # [CLASS DOC] Runs every ProcessGroupGlooTest test with lazy Gloo device initialization:
    # lazy_init is true and the environment variable enabling lazy init is set for each test
    # and removed in tearDown. Skipped on Windows.
    lazy_init = True

    def setUp(self):
        # [FUNCTION DOC] HELPER: Sets the lazy-init environment variable to 1 then runs the
        # base setUp.
        os.environ["TORCH_GLOO_LAZY_INIT"] = "1"
        super().setUp()

    def tearDown(self) -> None:
        # [FUNCTION DOC] HELPER: Removes the lazy-init environment variable then runs the
        # base tearDown.
        del os.environ["TORCH_GLOO_LAZY_INIT"]
        return super().tearDown()


class ProcessGroupGlooFRTest(ProcessGroupGlooTest):
    # [CLASS DOC] Runs ProcessGroupGlooTest with the flight recorder enabled (buffer of 10
    # entries via environment variable, set in setUp and removed in tearDown) and adds tests
    # that dump and inspect the recorded trace.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Sets the flight recorder buffer size variable to 10 then
        # runs the base setUp.
        os.environ["TORCH_FR_BUFFER_SIZE"] = "10"
        super().setUp()

    def tearDown(self) -> None:
        # [FUNCTION DOC] HELPER: Removes the flight recorder buffer size variable then runs
        # the base tearDown.
        del os.environ["TORCH_FR_BUFFER_SIZE"]
        return super().tearDown()

    def _verify_trace(self, t, is_json):
        # [FUNCTION DOC] HELPER: Validates a flight recorder dump after two allreduces on
        # the default group: version string, a single process group config with
        # name/desc/ranks, group status counters (3 enqueued and completed, none started),
        # three entries whose last is a scheduled float allreduce on a 3 by 4 tensor with
        # sequence id 3, timeout 50000 ms and no duration. Stack frames are checked only for
        # the pickle format (is_json false).
        ver = t["version"]
        self.assertEqual(ver, "2.10")
        pg_config = t["pg_config"]
        self.assertEqual(len(pg_config), 1)
        default_pg_info = pg_config["0"]
        self.assertIn("name", default_pg_info)
        self.assertIn("desc", default_pg_info)
        self.assertIn("ranks", default_pg_info)
        pg_status = t["pg_status"]
        self.assertEqual(len(pg_status), 1)
        self.assertEqual(str(pg_status["0"]["last_enqueued_collective"]), "3")
        self.assertEqual(str(pg_status["0"]["last_completed_collective"]), "3")
        self.assertEqual(
            str(pg_status["0"]["last_started_collective"]),
            "-1",
        )
        global_ranks = pg_config["0"]["ranks"]
        self.assertEqual(len(json.loads(global_ranks)), self.world_size)
        self.assertEqual(len(t["entries"]), 3)
        t = t["entries"]
        last = t[-1]
        self.assertEqual(last["process_group"], ("0", ""))
        # No event recorded for Gloo.
        self.assertEqual(last["state"], "scheduled")
        # we don't collect stack traces in JSON at the moment
        if not is_json:
            self.assertIn("test_c10d_gloo.py", str(last["frames"]))
        self.assertEqual(last["input_sizes"], ((3, 4),))
        self.assertEqual(last["input_dtypes"], ["Float"])
        self.assertEqual(last["output_sizes"], ((3, 4),))
        self.assertEqual(last["output_dtypes"], ["Float"])
        self.assertEqual(last["collective_seq_id"], 3)
        # TODO: Needs verification
        self.assertEqual(last["timeout_ms"], 50000)
        self.assertTrue("duration_ms" not in last)

    @requires_gloo()
    def test_short_json(self):
        # [FUNCTION DOC] Runs two allreduces, waits a second, dumps the flight recorder
        # trace as JSON and validates it.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(group_name="0")
        )
        a = torch.full((3, 4), float(self.rank))
        for _ in range(2):
            f = pg.allreduce(a)
            f.wait()
        time.sleep(1)
        t = json.loads(
            torch._C._distributed_c10d._dump_fr_trace_json(includeCollectives=True)
        )
        self._verify_trace(t, True)

    @requires_gloo()
    def test_short_pickle(self):
        # [FUNCTION DOC] Same as the JSON trace test but dumps the pickled trace and
        # validates it including stack frames.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(group_name="0")
        )
        a = torch.full((3, 4), float(self.rank))
        for _ in range(2):
            f = pg.allreduce(a)
            f.wait()
        time.sleep(1)
        t = pickle.loads(
            torch._C._distributed_c10d._dump_fr_trace(includeCollectives=True)
        )
        self._verify_trace(
            t,
            is_json=False,
        )

    @requires_gloo()
    def test_long(self):
        # [FUNCTION DOC] Runs broadcast, allreduce, reduce, allgather, reduce_scatter and
        # another allreduce twice, then checks the pickled trace has 10 entries (9 apart in
        # sequence ids from first to last), last entry being gloo all_reduce with expected
        # metadata.
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(group_name="0")
        )
        a = torch.full((3, 4), float(self.rank))
        for _ in range(2):
            # test some other primitives to make sure
            # their strings are valid
            xs = [torch.ones(3, 4)]
            pg.broadcast(xs).wait()
            pg.allreduce(xs).wait()
            pg.reduce(xs).wait()
            ys = [[torch.empty(3, 4) for _ in range(self.world_size)]]
            pg.allgather(ys, xs).wait()
            pg.reduce_scatter(xs, ys).wait()
            f = pg.allreduce(a)
            f.wait()
        t = pickle.loads(torch._C._distributed_c10d._dump_fr_trace())
        t = t["entries"]
        self.assertEqual(len(t), 10)
        first = t[0]
        last = t[-1]
        self.assertEqual(last["profiling_name"], "gloo:all_reduce")
        self.assertEqual(last["state"], "scheduled")
        self.assertIn("test_c10d_gloo.py", str(last["frames"]))
        self.assertEqual(last["input_sizes"], ((3, 4),))
        self.assertEqual(last["input_dtypes"], ["Float"])
        self.assertEqual(last["output_sizes"], ((3, 4),))
        self.assertEqual(last["output_dtypes"], ["Float"])
        self.assertEqual(last["timeout_ms"], 50000)
        self.assertEqual(last["collective_seq_id"] - first["collective_seq_id"], 9)


class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):
    # [CLASS DOC] Gloo specialization of the shared communication tests; runs in spawned CPU
    # processes and removes the file store file in tearDown.
    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Returns the string cpu as the device for the shared tests.
        return "cpu"

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the base tearDown, then removes the store file,
        # ignoring OS errors.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        # [FUNCTION DOC] HELPER: Broadcast six groups of five chunks (a mix of fp16 or fp32
        # on CPU, fp32, fp64) with the coalesced broadcast using a small 256 buffer; the
        # root keeps real values, others start with zeros and must end up equal to the
        # target.
        half = torch.float16

        # No support for float16 for CPU tensors
        if device == torch.device("cpu"):
            half = torch.float32

        target = torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

        # The tensors to pass to broadcast are identical to the target
        # only on the process that is the root of the broadcast.
        if self.rank == root_rank:
            tensors = [tensor.clone() for tensor in target]
        else:
            tensors = [torch.zeros_like(tensor) for tensor in target]

        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)

        c10d._broadcast_coalesced(
            process_group, tensors, buffer_size=256, src=root_rank
        )

        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_gloo_cuda(self):
        # [FUNCTION DOC] Initializes the default gloo group, creates a loopback device for
        # this rank's CUDA device and runs the coalesced broadcast for every rank as root.
        # Needs 2 GPUs.
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"cuda:{self.rank:d}")
        backend = process_group._get_backend(device)
        backend.create_device(interface=LOOPBACK)
        ranks = list(range(self.world_size))
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_gloo()
    def test_broadcast_coalesced_gloo_cpu(self):
        # [FUNCTION DOC] CPU variant of the coalesced broadcast test for every root rank.
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device("cpu")
        backend = process_group._get_backend(device)
        backend.create_device(interface=LOOPBACK)
        ranks = list(range(self.world_size))
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_gloo_warn_not_in_group(self):
        # [FUNCTION DOC] Runs the shared not-in-group warning test with gloo (needs 2 GPUs).
        self._test_warn_not_in_group(backend="gloo")

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_gloo_rank_membership(self):
        # [FUNCTION DOC] Runs the shared rank-membership test with gloo (needs 2 GPUs).
        self._test_rank_membership(backend="gloo")

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_tensor_dtype_mismatch(self):
        # [FUNCTION DOC] Runs the shared tensor dtype mismatch test with gloo (needs 2
        # GPUs).
        self._test_tensor_dtype_mismatch(backend="gloo")

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_tensor_dtype_complex(self):
        # [FUNCTION DOC] Runs the shared complex dtype test with gloo (needs 2 GPUs).
        self._test_tensor_dtype_complex(backend="gloo")

    @requires_gloo()
    def test_bool_tensors(self):
        # [FUNCTION DOC] Runs the shared bool tensor test with gloo.
        self._test_bool_tensors(backend="gloo")

    @requires_gloo()
    def test_split_group_does_not_alias_parent_options(self):
        # [FUNCTION DOC] Regression test: after splitting the default group into halves, the
        # child options object must be distinct from the parent's; the parent keeps its
        # group name, 111 second timeout and empty global ranks, while the child has the 222
        # second timeout and its own ranks. A second split by rank parity must report the
        # parity ranks. Destroys the group at the end.
        # Regression test: ProcessGroup::splitGroup used to hand the parent
        # backend's live Options object to the child, so a split rewrote the
        # parent's group_name/timeout and ProcessGroupGloo::split overwrote the
        # parent's global_ranks_in_group with the child's ranks. The parent's
        # next split then indexed that subgroup-sized vector out of bounds.
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=111),
        )
        pg = c10d.distributed_c10d._get_default_group()
        cpu = torch.device("cpu")
        parent_options = pg._get_backend(cpu).options
        parent_group_name = parent_options.group_name

        half = self.world_size // 2
        halves = (
            list(range(half))
            if self.rank < half
            else list(range(half, self.world_size))
        )
        child = pg.split_group(
            halves, timeout=timedelta(seconds=222), group_name=f"halves{halves}"
        )
        child_options = child._get_backend(cpu).options
        self.assertIsNot(child_options, parent_options)
        self.assertEqual(parent_options.group_name, parent_group_name)
        self.assertEqual(parent_options._timeout, timedelta(seconds=111))
        self.assertEqual(parent_options.global_ranks_in_group, [])
        self.assertEqual(child_options._timeout, timedelta(seconds=222))
        self.assertEqual(child_options.global_ranks_in_group, halves)

        # A second split of the same parent must still see the world's ranks.
        parity = [r for r in range(self.world_size) if r % 2 == self.rank % 2]
        second = pg.split_group(
            parity, timeout=timedelta(seconds=333), group_name=f"parity{parity}"
        )
        self.assertEqual(second._get_backend(cpu).options.global_ranks_in_group, parity)

        c10d.destroy_process_group()

    @skip_if_lt_x_gpu(1)
    @requires_gloo()
    def test_split_group_keeps_gloo_options(self):
        # [FUNCTION DOC] Regression test: with a gloo world bound to a CUDA device,
        # splitting all ranks into one group with a 222 second timeout must give the CPU
        # gloo backend options carrying that timeout, the child's group name and the ranks.
        # Needs 1 GPU; destroys the group.
        # dist.split_group used to substitute a deep copy of the accelerator
        # backend's options for every device. On a gloo world that raised
        # "cannot pickle _Options" (ProcessGroupGloo._Options has no
        # __deepcopy__), and on a "cpu:gloo,cuda:nccl" world the gloo leg
        # rejected the nccl options and fell back to Options::create_default(),
        # dropping the caller's timeout and group_name.
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=torch.device("cuda", self.rank % torch.cuda.device_count()),
        )
        ranks = list(range(self.world_size))
        child = c10d.split_group(split_ranks=[ranks], timeout=timedelta(seconds=222))
        options = child._get_backend(torch.device("cpu")).options
        self.assertEqual(options._timeout, timedelta(seconds=222))
        self.assertEqual(options.group_name, child.group_name)
        self.assertEqual(options.global_ranks_in_group, ranks)
        c10d.destroy_process_group()


class GlooProcessGroupWithDispatchedCollectivesTests(
    test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
):
    # [CLASS DOC] Gloo specialization of the shared dispatched-collective tests
    # (collectives, coalesced allreduce, all-to-all single), plus gloo-only allgather
    # coalesced and monitored barrier tests.
    @requires_gloo()
    def test_collectives(self):
        # [FUNCTION DOC] Runs the shared collectives test with gloo.
        self._test_collectives(backend="gloo")

    @requires_gloo()
    def test_allreduce_coalesced(self):
        # [FUNCTION DOC] Runs the shared allreduce_coalesced test with gloo.
        self._test_allreduce_coalesced(backend="gloo")

    @requires_gloo()
    def test_all_to_all_single(self):
        # [FUNCTION DOC] Runs the shared all_to_all_single test with gloo.
        self._test_all_to_all_single(backend="gloo")

    @requires_gloo()
    def test_allgather_coalesced(self):
        # [FUNCTION DOC] Initializes the default gloo group and runs the module-level
        # all_gather_coalesced on one 10 by 10 tensor of ones; the output list entry must
        # equal it.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "gloo",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        input_tensor = torch.ones(10, 10, dtype=torch.float32)
        output_tensor_list = [torch.zeros_like(input_tensor)]
        dist.all_gather_coalesced([output_tensor_list], [input_tensor])
        self.assertEqual(output_tensor_list, [input_tensor])

    @requires_gloo()
    def test_monitored_barrier(self):
        # [FUNCTION DOC] Initializes the default gloo group and calls monitored_barrier;
        # passes if it returns without error on all ranks.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "gloo",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        dist.monitored_barrier()


class LargeCommTest(test_c10d_common.AbstractLargeCommTest, MultiProcessTestCase):
    # [CLASS DOC] Gloo specialization of the shared large-communicator tests; spawned CPU
    # processes with file cleanup in tearDown.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the base tearDown, then removes the store file,
        # ignoring OS errors.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Returns the CPU torch device for the shared tests.
        return torch.device("cpu")

    @requires_gloo()
    def test_new_group_local_sync(self):
        # [FUNCTION DOC] Runs the shared new_group local-sync test with gloo.
        self._test_new_group_local_sync(backend="gloo")

    @requires_gloo()
    def test_new_group_local_sync_sanity_check(self):
        # [FUNCTION DOC] Runs the shared new_group local-sync sanity check with gloo.
        self._test_new_group_local_sync_sanity_check(backend="gloo")

    @requires_gloo()
    def test_new_group_local_sync_duplicate_pg(self):
        # [FUNCTION DOC] Runs the shared new_group local-sync duplicate-group test with
        # gloo.
        self._test_new_group_local_sync_duplicate_pg(backend="gloo")

    @requires_gloo()
    def test_new_group_ordered(self):
        # [FUNCTION DOC] Runs the shared ordered new_group test with gloo.
        self._test_new_group_ordered(backend="gloo")


if __name__ == "__main__":
    if torch.cuda._initialized:
        raise AssertionError(
            "test_distributed must not have initialized CUDA context on main process"
        )

    run_tests()
