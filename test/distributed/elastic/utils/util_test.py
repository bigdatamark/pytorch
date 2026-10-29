#!/usr/bin/env python3
# [MODULE DOC] Tests for torch.distributed.elastic.utils.store helpers (get_all, synchronize,
# barrier, with rank tracing on timeout) using a recording MockStore and real HashStore with
# threads, plus get_logger naming tests. No import-time side effects.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from multiprocessing.pool import ThreadPool
from unittest import mock

import torch.distributed as dist
import torch.distributed.elastic.utils.store as store_util
from torch.distributed.elastic.utils.logging import get_logger
from torch.testing._internal.common_utils import run_tests, TestCase


class MockStore:
    # [CLASS DOC] HELPER: Fake key-value store that records every call in an ops list as tuples of
    # operation name and arguments, with canned return values: get returns the string value,
    # multi_get returns one value per key, add returns 3, and timeout returns a fixed timedelta of
    # 1234 seconds.
    _TEST_TIMEOUT = 1234

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes an empty ops list that records calls.
        self.ops = []

    def set_timeout(self, timeout: float) -> None:
        # [FUNCTION DOC] HELPER: Records a set_timeout operation with the given timeout.
        self.ops.append(("set_timeout", timeout))

    @property
    def timeout(self) -> datetime.timedelta:
        # [FUNCTION DOC] HELPER: Property that records a timeout read and returns a timedelta of
        # the class test timeout (1234 seconds).
        self.ops.append(("timeout",))

        return datetime.timedelta(seconds=self._TEST_TIMEOUT)

    def set(self, key: str, value: str) -> None:
        # [FUNCTION DOC] HELPER: Records a set operation with key and value.
        self.ops.append(("set", key, value))

    def get(self, key: str) -> str:
        # [FUNCTION DOC] HELPER: Records a get operation and returns the fixed string value.
        self.ops.append(("get", key))
        return "value"

    def multi_get(self, keys: list[str]) -> list[str]:
        # [FUNCTION DOC] HELPER: Records a multi_get operation and returns the string value
        # repeated once per key.
        self.ops.append(("multi_get", keys))
        return ["value"] * len(keys)

    def add(self, key: str, val: int) -> int:
        # [FUNCTION DOC] HELPER: Records an add operation and returns the constant 3.
        self.ops.append(("add", key, val))
        return 3

    def wait(self, keys: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Records a wait operation for the keys and returns immediately.
        self.ops.append(("wait", keys))


class StoreUtilTest(TestCase):
    # [CLASS DOC] Tests of store_util.get_all, synchronize and barrier, checking exact store call
    # sequences with MockStore and correctness with threads over HashStore.
    def test_get_all_rank_0(self):
        # [FUNCTION DOC] Checks rank 0 of 3 does a multi_get of all rank keys, adds to the finished
        # counter, sets the last_member key and waits on it.
        world_size = 3

        store = MockStore()

        store_util.get_all(store, 0, "test/store", world_size)

        self.assertListEqual(
            store.ops,
            [
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),
                ("add", "test/store/finished/num_members", 1),
                ("set", "test/store/finished/last_member", "<val_ignored>"),
                ("wait", ["test/store/finished/last_member"]),
            ],
        )

    def test_get_all_rank_n(self):
        # [FUNCTION DOC] Checks a non-zero rank does the multi_get, adds to the counter and sets
        # last_member but does not wait.
        store = MockStore()
        world_size = 3
        store_util.get_all(store, 1, "test/store", world_size)

        self.assertListEqual(
            store.ops,
            [
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),
                ("add", "test/store/finished/num_members", 1),
                ("set", "test/store/finished/last_member", "<val_ignored>"),
            ],
        )

    def test_synchronize(self):
        # [FUNCTION DOC] Checks synchronize on rank 0 saves the store timeout, sets it to 300
        # seconds, sets its data key, gathers all keys, runs the finish protocol with wait, then
        # restores the original timeout.
        store = MockStore()

        data = b"data0"
        store_util.synchronize(store, data, 0, 3, key_prefix="test/store")

        self.assertListEqual(
            store.ops,
            [
                ("timeout",),
                ("set_timeout", datetime.timedelta(seconds=300)),
                ("set", "test/store0", data),
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),
                ("add", "test/store/finished/num_members", 1),
                ("set", "test/store/finished/last_member", "<val_ignored>"),
                ("wait", ["test/store/finished/last_member"]),
                ("set_timeout", datetime.timedelta(seconds=store._TEST_TIMEOUT)),
            ],
        )

    def test_synchronize_hash_store(self) -> None:
        # [FUNCTION DOC] Runs synchronize on 4 threads sharing a HashStore and checks every thread
        # gets the same list of all four byte-encoded payloads.
        N = 4

        store = dist.HashStore()

        def f(i: int):
            # [FUNCTION DOC] HELPER: Thread worker calling synchronize with this index's string
            # data and returning the gathered result.
            return store_util.synchronize(
                store, f"data{i}", i, N, key_prefix="test/store"
            )

        with ThreadPool(N) as pool:
            out = pool.map(f, range(N))

        self.assertListEqual(out, [[f"data{i}".encode() for i in range(N)]] * N)

    def test_barrier(self):
        # [FUNCTION DOC] Checks barrier on a mock store sets a 300 second timeout, adds to the
        # member counter, sets last_member, waits on it and restores the original timeout.
        store = MockStore()

        store_util.barrier(store, 3, key_prefix="test/store")

        self.assertListEqual(
            store.ops,
            [
                ("timeout",),
                ("set_timeout", datetime.timedelta(seconds=300)),
                ("add", "test/store/num_members", 1),
                ("set", "test/store/last_member", "<val_ignored>"),
                ("wait", ["test/store/last_member"]),
                ("set_timeout", datetime.timedelta(seconds=store._TEST_TIMEOUT)),
            ],
        )

    def test_barrier_timeout_rank_tracing(self):
        # [FUNCTION DOC] Runs barrier on two of three ranks over a HashStore with a short timeout
        # and a rank tracing decoder, so it times out. Checks the error messages list missing ranks
        # and point to rank 0 with its decoded host.
        N = 3

        store = dist.HashStore()

        def run_barrier_for_rank(i: int):
            # [FUNCTION DOC] HELPER: Runs barrier with tracing options for the given rank
            # and returns the error message string if an exception is raised, or an empty
            # string otherwise.
            try:
                store_util.barrier(
                    store,
                    N,
                    key_prefix="test/store",
                    barrier_timeout=0.1,
                    rank=i,
                    rank_tracing_decoder=lambda x: f"Rank {x} host",
                    trace_timeout=0.01,
                )
            except Exception as e:
                return str(e)
            return ""

        with ThreadPool(N - 1) as pool:
            outputs: list[str] = pool.map(run_barrier_for_rank, range(N - 1))

        self.assertTrue(any("missing_ranks=[Rank " in msg for msg in outputs))

        self.assertTrue(
            any(
                "check rank 0 (Rank 0 host) for missing rank info" in msg
                for msg in outputs
            )
        )

    def test_barrier_timeout_operations(self):
        # [FUNCTION DOC] With wait mocked to raise DistStoreError, checks the store call sequence
        # and re-raised error for rank 0 (writes its TRACE key then the tracing gate) and for rank
        # 1 (writes only its TRACE key).
        import torch

        DistStoreError = torch._C._DistStoreError

        N = 3
        store = MockStore()

        # rank 0
        with mock.patch.object(store, "wait") as wait_mock:
            wait_mock.side_effect = [DistStoreError("test"), None, None]

            with self.assertRaises(DistStoreError):
                store_util.barrier(
                    store,
                    N,
                    key_prefix="test/store",
                    barrier_timeout=1,
                    rank=0,
                    rank_tracing_decoder=lambda x: f"Rank {x} host",
                    trace_timeout=0.1,
                )

            self.assertListEqual(
                store.ops,
                [
                    ("timeout",),
                    ("set_timeout", datetime.timedelta(seconds=1)),
                    ("add", "test/store/num_members", 1),
                    ("set", "test/store/last_member", "<val_ignored>"),
                    # wait for last member is mocked
                    ("set", "test/store0/TRACE", "<val_ignored>"),
                    # wait for each rank is mocked
                    ("set", "test/store/TRACING_GATE", "<val_ignored>"),
                ],
            )

        # rank 1
        with mock.patch.object(store, "wait") as wait_mock:
            store.ops = []

            wait_mock.side_effect = [
                DistStoreError("test"),
                None,
            ]

            with self.assertRaises(DistStoreError):
                store_util.barrier(
                    store,
                    N,
                    key_prefix="test/store",
                    barrier_timeout=1,
                    rank=1,
                    rank_tracing_decoder=lambda x: f"Rank {x} host",
                    trace_timeout=0.1,
                )

            self.assertListEqual(
                store.ops,
                [
                    ("timeout",),
                    ("set_timeout", datetime.timedelta(seconds=1)),
                    ("add", "test/store/num_members", 1),
                    ("set", "test/store/last_member", "<val_ignored>"),
                    ("set", "test/store1/TRACE", "<val_ignored>"),
                    # wait for gate is mocked
                ],
            )

    def test_barrier_hash_store(self) -> None:
        # [FUNCTION DOC] Runs barrier on 4 threads over a HashStore and checks all return None.
        N = 4

        store = dist.HashStore()

        def f(i: int):
            # [FUNCTION DOC] HELPER: Thread worker calling barrier on the shared store.
            store_util.barrier(store, N, key_prefix="test/store")

        with ThreadPool(N) as pool:
            out = pool.map(f, range(N))

        self.assertEqual(out, [None] * N)


class UtilTest(TestCase):
    # [CLASS DOC] Tests for get_logger naming.
    def test_get_logger_different(self):
        # [FUNCTION DOC] Checks loggers for different names have different names.
        logger1 = get_logger("name1")
        logger2 = get_logger("name2")
        self.assertNotEqual(logger1.name, logger2.name)

    def test_get_logger(self):
        # [FUNCTION DOC] Checks get_logger with no name returns a logger named after this module.
        logger1 = get_logger()
        self.assertEqual(__name__, logger1.name)

    def test_get_logger_none(self):
        # [FUNCTION DOC] Checks get_logger with None returns a logger named after this module.
        logger1 = get_logger(None)
        self.assertEqual(__name__, logger1.name)

    def test_get_logger_custom_name(self):
        # [FUNCTION DOC] Checks get_logger with a custom name returns a logger with that name.
        logger1 = get_logger("test.module")
        self.assertEqual("test.module", logger1.name)


if __name__ == "__main__":
    run_tests()
