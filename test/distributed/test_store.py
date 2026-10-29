# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the c10d key-value stores: FileStore, HashStore, TCPStore (legacy and
# libuv backends), PrefixStore wrappers, Python-defined Store subclasses, and the file/env/tcp
# rendezvous handlers. A shared mixin (StoreTestBase) holds the common store tests. Import-time
# side effects: exits the process with status 0 if torch.distributed is not available; builds a
# module-level accelerator device_type; TestMultiThreadedWait creates FileStore, HashStore and two
# TCPStore servers at class definition time. setUpModule and tearDownModule disable and restore
# the CUDA matmul fp32 precision setting.

import datetime
import os
import socket
import struct
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from sys import platform
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.distributed.rpc as rpc
from torch.distributed import DistError, DistNetworkError, DistStoreError
from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests


if not dist.is_available():
    print("torch.distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_distributed import (
    create_tcp_store,
    skip_if_win32,
    tp_transports,
)
from torch.testing._internal.common_utils import (
    ADDRESS_IN_USE,
    CONNECT_TIMEOUT,
    load_tests,
    retry_on_connect_failures,
    run_tests,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

if platform == "darwin":
    LOOPBACK = "lo0"
else:
    LOOPBACK = "lo"

DEFAULT_HOSTNAME = "localhost"


_PRIOR_FP32_PRECISION: str | None = None


def setUpModule():
    # [FUNCTION DOC] HELPER: Module fixture. Saves the current CUDA matmul fp32 precision in a
    # module global, then turns off tf32 matmul. Mutates global backend settings.
    global _PRIOR_FP32_PRECISION
    # Snapshot fp32_precision (not allow_tf32) so tearDownModule restores the
    # exact original; writing allow_tf32 back can't reproduce the "none" default.
    _PRIOR_FP32_PRECISION = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.allow_tf32 = False


def tearDownModule():
    # [FUNCTION DOC] HELPER: Module fixture. Restores the CUDA matmul fp32 precision saved by
    # setUpModule if one was saved, and clears the saved value.
    global _PRIOR_FP32_PRECISION
    if _PRIOR_FP32_PRECISION is not None:
        torch.backends.cuda.matmul.fp32_precision = _PRIOR_FP32_PRECISION
        _PRIOR_FP32_PRECISION = None


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class StoreTestBase:
    # [CLASS DOC] HELPER: Mixin (not a test case by itself) with tests shared by all store
    # flavours. A concrete subclass combines it with TestCase and supplies a no-argument store
    # factory and, if the key count differs from the default, num_keys_total. Several tests
    # silently return for stores without the extended API, and queue tests skip when queues are
    # unsupported.
    def _create_store(self, i):
        # [FUNCTION DOC] HELPER: Abstract store factory; raises RuntimeError in the base mixin.
        # Subclasses override it.
        # KNOWN ISSUE: the signature takes an unused index argument, but every caller and every
        # override uses no arguments.
        raise RuntimeError("not implemented")

    def _test_set_get_check(self, fs):
        # [FUNCTION DOC] HELPER: Exercises add, set, get, check, num_keys and delete_key on the
        # given store: sums counters through repeated adds, stores string values, checks the key
        # count equals num_keys_total, verifies the returned bytes, checks presence and absence of
        # keys, then sets and deletes a temporary key and confirms the count returns to the
        # earlier value.
        fs.add("key", 1)
        fs.add("key", 2)
        fs.add("key", 3)
        fs.set("key0", "value0")
        fs.add("key3", 1)
        fs.set("key1", "value1")
        fs.add("key3", 2)
        fs.set("key2", "value2")
        fs.add("key3", 3)
        fs.add("key3", 4)
        fs.add("key3", 5)
        fs.add("key3", 6)
        self.assertEqual(fs.num_keys(), self.num_keys_total)
        self.assertEqual(b"6", fs.get("key"))
        self.assertEqual(b"value0", fs.get("key0"))
        self.assertEqual(b"value1", fs.get("key1"))
        self.assertEqual(b"value2", fs.get("key2"))
        self.assertEqual(b"21", fs.get("key3"))
        self.assertTrue(fs.check(["key3"]))
        self.assertFalse(fs.check(["Randomkey3"]))

        fs.set("-key3", "7")
        self.assertEqual(b"7", fs.get("-key3"))
        fs.delete_key("-key3")
        self.assertEqual(fs.num_keys(), self.num_keys_total)

    def test_set_get_check(self):
        # [FUNCTION DOC] Runs the shared set/get/check scenario against a freshly created store.
        self._test_set_get_check(self._create_store())

    def _test_compare_set(self, store):
        # [FUNCTION DOC] HELPER: Checks compare_set semantics: for a missing key the expected
        # value is returned unchanged and nothing is stored; with a wrong expected value the
        # current value is returned and kept; with the right expected value the new value is
        # stored and returned; an empty expected value on a missing key creates the key with the
        # new value.
        missing_key_result = store.compare_set(
            "cs_key0", "wrong_old_value", "new_value0"
        )
        self.assertEqual(b"wrong_old_value", missing_key_result)

        store.set("cs_key0", "value0")
        self.assertEqual(b"value0", store.get("cs_key0"))
        old_value_result = store.compare_set("cs_key0", "wrong_old_value", "new_value0")
        self.assertEqual(b"value0", old_value_result)
        self.assertEqual(b"value0", store.get("cs_key0"))
        new_value_result = store.compare_set("cs_key0", "value0", "new_value0")
        self.assertEqual(b"new_value0", new_value_result)
        self.assertEqual(b"new_value0", store.get("cs_key0"))
        empty_old_value_result = store.compare_set("cs_key1", "", "new_value1")
        self.assertEqual(b"new_value1", empty_old_value_result)
        self.assertEqual(b"new_value1", store.get("cs_key1"))

    def test_compare_set(self):
        # [FUNCTION DOC] Runs the shared compare_set scenario against a freshly created store.
        self._test_compare_set(self._create_store())

    def _test_simple_wait(self, fs):
        # [FUNCTION DOC] HELPER: Verifies wait raises RuntimeError mentioning a timeout for a
        # missing key with a 0.25 second timeout, then that wait returns for a key made present by
        # add.
        with self.assertRaisesRegex(RuntimeError, "[t -i]imeout"):
            fs.wait(["bad_key"], timedelta(seconds=0.25))
        fs.add("good_key", 1)
        fs.wait(["good_key"])

    def test_simple_wait(self):
        # [FUNCTION DOC] Runs the shared wait scenario against a freshly created store.
        self._test_simple_wait(self._create_store())

    def _test_append(self, store):
        # [FUNCTION DOC] HELPER: Checks append concatenates values onto existing and onto new keys
        # (giving the bytes potato in both cases). Returns early if the store lacks the extended
        # API.
        if not store.has_extended_api():
            # Just return for stores that don't support extended APIs.
            return
        store.set("foo", "po")
        store.append("foo", "tato")
        store.append("bar", "po")
        store.append("bar", "tato")
        self.assertEqual(b"potato", store.get("foo"))
        self.assertEqual(b"potato", store.get("bar"))

    def test_append(self):
        # [FUNCTION DOC] Runs the shared append scenario against a freshly created store.
        self._test_append(self._create_store())

    def _create_store_or_skip_if_no_queues(self) -> dist.Store:
        # [FUNCTION DOC] HELPER: Creates a store and probes queue support by pushing one item to a
        # probe queue. Skips the test if the store raises NotImplementedError. Returns the store;
        # the probe item stays in its own queue.
        store = self._create_store()

        try:
            store.queue_push("test_queue_support", "1")
        except NotImplementedError:
            self.skipTest("Store does not support queues")

        return store

    def test_queues(self) -> None:
        # [FUNCTION DOC] Tests basic queue behaviour: an empty queue has length 0 and does not
        # count as an existing key; after two pushes the key exists, length is 2 and wait returns;
        # pops return items in FIFO order; afterwards the queue is empty again.
        store = self._create_store_or_skip_if_no_queues()

        self.assertFalse(store.check(["foo"]))
        self.assertEqual(store.queue_len("foo"), 0)

        store.queue_push("foo", "1")
        store.queue_push("foo", "2")

        self.assertTrue(store.check(["foo"]))
        self.assertEqual(store.queue_len("foo"), 2)
        store.wait(["foo"])

        self.assertEqual(store.queue_pop("foo"), b"1")
        self.assertEqual(store.queue_pop("foo"), b"2")

        self.assertFalse(store.check(["foo"]))
        self.assertEqual(store.queue_len("foo"), 0)

    def test_queues_nonblocking(self) -> None:
        # [FUNCTION DOC] Tests that a non-blocking pop on an empty queue raises QueueEmptyError
        # mentioning empty, and that a non-blocking pop returns an item that was pushed.
        store = self._create_store_or_skip_if_no_queues()

        with self.assertRaisesRegex(dist.QueueEmptyError, "empty"):
            store.queue_pop("foo", block=False)

        store.queue_push("foo", "a")
        self.assertEqual(store.queue_pop("foo", block=False), b"a")

    def test_queues_bidirectional(self) -> None:
        # [FUNCTION DOC] Tests queues across two threads, each using its own clone of the store:
        # one pushes to queue a and pops from queue b, the other does the reverse; both results
        # must succeed.
        store = self._create_store_or_skip_if_no_queues()

        def worker_a():
            # [FUNCTION DOC] HELPER: Thread body: on a cloned store pushes a1 to queue a, then
            # blocks popping queue b and asserts it receives b1.
            local_store = store.clone()

            local_store.queue_push("a", "a1")
            self.assertEqual(local_store.queue_pop("b"), b"b1")

        def worker_b():
            # [FUNCTION DOC] HELPER: Thread body: on a cloned store pops queue a expecting a1,
            # then pushes b1 to queue b.
            local_store = store.clone()

            self.assertEqual(local_store.queue_pop("a"), b"a1")
            local_store.queue_push("b", "b1")

        # test bidirectional communication
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(worker_a),
                pool.submit(worker_b),
            ]
            for fut in futures:
                fut.result()

    def test_queues_timeout(self) -> None:
        # [FUNCTION DOC] Tests that popping from a queue that never receives data fails with
        # DistStoreError mentioning a timeout once the store timeout is set to 0.01 seconds.
        store = self._create_store_or_skip_if_no_queues()

        store.set_timeout(timedelta(seconds=0.01))
        with self.assertRaisesRegex(DistStoreError, "timeout"):
            store.queue_pop("non_existant")

    def _test_multi_set(self, store):
        # [FUNCTION DOC] HELPER: Checks multi_set stores several keys with their values in one
        # call, verified with get. Returns early if the store lacks the extended API.
        if not store.has_extended_api():
            # Just return for stores that don't support extended APIs.
            return
        store.multi_set(["foo", "bar"], ["po", "tato"])
        self.assertEqual(b"po", store.get("foo"))
        self.assertEqual(b"tato", store.get("bar"))

    def test_multi_set(self):
        # [FUNCTION DOC] Runs the shared multi_set scenario against a freshly created store.
        self._test_multi_set(self._create_store())

    def _test_multi_get(self, store):
        # [FUNCTION DOC] HELPER: Checks multi_get returns the values of several keys, in order,
        # after they were set. Returns early if the store lacks the extended API.
        if not store.has_extended_api():
            # Just return for stores that don't support extended APIs.
            return
        store.set("foo", "po")
        store.set("bar", "tato")
        v0, v1 = store.multi_get(["foo", "bar"])
        self.assertEqual(b"po", v0)
        self.assertEqual(b"tato", v1)

    def test_multi_get(self):
        # [FUNCTION DOC] Runs the shared multi_get scenario against a freshly created store.
        self._test_multi_get(self._create_store())

    def test_clone(self):
        # [FUNCTION DOC] Tests clone: the clone is a Store instance and sees a value written
        # through the original.
        a = self._create_store()
        b = a.clone()

        self.assertIsInstance(b, dist.Store)

        a.set("foo", "bar")
        self.assertEqual(b.get("foo"), b"bar")

    def test_list_keys(self):
        # [FUNCTION DOC] Tests list_keys: after setting two keys, both names appear in the
        # returned collection.
        a = self._create_store()
        a.set("foo", "bar")
        a.set("baz", "qux")
        keys = a.list_keys()
        self.assertIn("foo", keys)
        self.assertIn("baz", keys)

    def _test_barrier(self, store):
        # [FUNCTION DOC] HELPER: Checks a barrier with a single participant returns immediately.
        # Barrier with world_size=1 should return immediately
        store.barrier("test_barrier_key", 1)

    def test_barrier(self):
        # [FUNCTION DOC] Runs the single-participant barrier scenario against a freshly created
        # store.
        self._test_barrier(self._create_store())

    # This is the number of keys used in test_set_get. Adding this as a class
    # property instead of hardcoding in the test since some Store
    # implementations will have differing number of keys. In the base case,
    # there will be 5 keys: key, key0, key1, key2, key3.
    @property
    def num_keys_total(self):
        # [FUNCTION DOC] HELPER: Property giving the expected number of keys after the
        # set/get/check scenario; default 5 (key, key0, key1, key2, key3). Subclasses override
        # when the store keeps extra keys.
        return 5


class FileStoreTest(TestCase, StoreTestBase):
    # [CLASS DOC] Runs the shared store tests against a FileStore on a temporary file (world size
    # 1). setUp creates the temporary file (not deleted by the test, FileStore cleans it up).
    def setUp(self):
        # [FUNCTION DOC] HELPER: Creates a named temporary file that is kept, and remembers it on
        # the instance.
        super().setUp()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file = f

    def _create_store(self):
        # [FUNCTION DOC] HELPER: Returns a FileStore on the temporary file with world size 1 and a
        # 300 second timeout.
        store = dist.FileStore(self.file.name, 1)
        store.set_timeout(timedelta(seconds=300))
        return store

    def test_init_pg_and_rpc_with_same_file(self):
        # [FUNCTION DOC] Tests that RPC (tensorpipe) and a gloo process group, both with world
        # size 1, can be initialized from the same file:// init method, and that the file still
        # exists after the process group is destroyed. Shuts RPC down and removes the file.
        with tempfile.NamedTemporaryFile(delete=False) as file:
            # Init RPC using file
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
            rpc_backend_options.init_method = f"file://{file.name}"
            rpc_backend_options._transports = tp_transports()
            rpc.init_rpc(
                "worker", rank=0, world_size=1, rpc_backend_options=rpc_backend_options
            )

            # Init PG using file
            dist.init_process_group(
                "gloo", rank=0, world_size=1, init_method=f"file://{file.name}"
            )
            dist.destroy_process_group()
            if not os.path.exists(file.name):
                raise AssertionError(f"Expected file {file.name} to exist")

            rpc.shutdown()
            os.remove(file.name)

    def test_refcount(self):
        # [FUNCTION DOC] Tests FileStore reference counting: two stores on the same file; deleting
        # one keeps the file, deleting the last one removes it.
        with tempfile.NamedTemporaryFile(delete=False) as file:
            store = dist.FileStore(file.name, 1)
            store2 = dist.FileStore(file.name, 1)

            del store
            if not os.path.exists(file.name):
                raise AssertionError(f"Expected file {file.name} to exist")
            del store2
            if os.path.exists(file.name):
                raise AssertionError(f"Expected file {file.name} to not exist")

    @property
    def num_keys_total(self):
        # [FUNCTION DOC] HELPER: Property: FileStore reports 6 keys in the set/get/check scenario,
        # not 5.
        return 6


@skip_if_win32()
class HashStoreTest(TestCase, StoreTestBase):
    # [CLASS DOC] Runs the shared store tests against an in-memory HashStore (skipped on Windows).
    def _create_store(self):
        # [FUNCTION DOC] HELPER: Returns a HashStore with a 300 second timeout.
        store = dist.HashStore()
        store.set_timeout(timedelta(seconds=300))
        return store


class PrefixStoreTest(TestCase):
    # [CLASS DOC] Tests PrefixStore behaviour that does not need the shared mixin; uses a
    # temporary file for a FileStore.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Creates a named temporary file that is kept, and remembers it on
        # the instance.
        super().setUp()
        # delete is false as FileStore will automatically clean up the file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file = f

    def test_get_underlying_store(self):
        # [FUNCTION DOC] Tests that PrefixStore exposes the wrapped store through underlying_store
        # for a TCPStore, HashStore and FileStore, and that passing None as the wrapped store
        # raises ValueError.
        tcp_store = dist.TCPStore(
            host_name=DEFAULT_HOSTNAME, port=0, world_size=1, is_master=True
        )
        hash_store = dist.HashStore()
        file_store = dist.FileStore(self.file.name, world_size=1)
        for store in [tcp_store, hash_store, file_store]:
            with self.subTest(f"Testing getting underlying_store for {type(store)}"):
                prefix_store = dist.PrefixStore("prefix", store)
                self.assertEqual(prefix_store.underlying_store, store)

        # We do not allow passing in None as the underlying store, this would cause a segfault if used
        with self.assertRaises(ValueError):
            dist.PrefixStore("prefix", None)


class PrefixFileStoreTest(TestCase, StoreTestBase):
    # [CLASS DOC] Runs the shared store tests against a PrefixStore wrapping a FileStore. State:
    # the FileStore and the prefix string.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Creates the temporary file, a FileStore on it with world size 1
        # and 300 second timeout, and the prefix string.
        super().setUp()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file = f
        self.filestore = dist.FileStore(self.file.name, 1)
        self.prefix = "test_prefix"
        self.filestore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        # [FUNCTION DOC] HELPER: Returns a PrefixStore with the test prefix over the FileStore.
        return dist.PrefixStore(self.prefix, self.filestore)

    @property
    def num_keys_total(self):
        # [FUNCTION DOC] HELPER: Property: expects 6 keys in the set/get/check scenario.
        return 6


class TCPStoreTest(TestCase, StoreTestBase):
    # [CLASS DOC] Runs the shared store tests plus TCPStore specific tests (ports, multi-tenancy,
    # workers, timeouts, barriers). The class attribute _use_libuv selects the backend; this class
    # uses the legacy non-libuv backend and the libuv subclass flips it.
    _use_libuv = False

    def _create_store(self):
        # [FUNCTION DOC] HELPER: Returns a TCP store from the test helper, using the selected
        # backend, with a 300 second timeout.
        store = create_tcp_store(use_libuv=self._use_libuv)
        store.set_timeout(timedelta(seconds=300))
        return store

    def _create_store_with_ws(self, addr, world_size):
        # [FUNCTION DOC] HELPER: Returns a TCP store for the given address and world size without
        # waiting for workers, using the selected backend.
        return create_tcp_store(
            addr, world_size, wait_for_workers=False, use_libuv=self._use_libuv
        )

    def test_address_already_in_use(self):
        # [FUNCTION DOC] Tests that creating a second master TCPStore on a port already held by
        # the first raises DistNetworkError saying the server socket failed to listen because the
        # address is in use.
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        err_msg_reg = "^The server socket has failed to listen on any local .*(?i:address already in use)"
        with self.assertRaisesRegex(dist.DistNetworkError, err_msg_reg):
            # Use noqa to silence flake8.
            # Need to store in an unused variable here to ensure the first
            # object is not destroyed before the second object is created.
            store1 = dist.TCPStore(addr, port, 1, True, use_libuv=self._use_libuv)
            store2 = dist.TCPStore(addr, port, 1, True, use_libuv=self._use_libuv)
            self.assertEqual(store1.libuvBackend, self._use_libuv)
            self.assertEqual(store2.libuvBackend, self._use_libuv)

    @retry_on_connect_failures
    def test_multitenancy(self):
        # [FUNCTION DOC] Tests that two master TCPStores can share one port when multi_tenant is
        # enabled, and that both report the selected backend. Retried on port collisions.
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        # Use noqa to silence flake8.
        # Need to store in an unused variable here to ensure the first
        # object is not destroyed before the second object is created.
        store1 = dist.TCPStore(
            addr, port, 1, True, multi_tenant=True, use_libuv=self._use_libuv
        )  # type: ignore[call-arg]
        store2 = dist.TCPStore(
            addr, port, 1, True, multi_tenant=True, use_libuv=self._use_libuv
        )  # type: ignore[call-arg]
        self.assertEqual(store1.libuvBackend, self._use_libuv)
        self.assertEqual(store2.libuvBackend, self._use_libuv)

    def test_repr(self) -> None:
        # [FUNCTION DOC] Tests the string form of a server store and of a client store: both show
        # a client with its socket details; the server shows a server with a port, the client
        # shows a null server.
        # server
        store1 = self._create_store()
        self.assertRegex(
            repr(store1),
            r"TCPStore\("
            r"client=TCPClient\(SocketImpl\(fd=\d+, addr=\[?localhost\]?:\d+, remote=\[?localhost\]?:\d+\)\), "
            r"server=TCPServer\(port=\d+\)\)",
        )

        # client
        store2 = dist.TCPStore(
            store1.host,
            store1.port,
            world_size=2,
            is_master=False,
        )
        self.assertRegex(
            repr(store2),
            r"TCPStore\("
            r"client=TCPClient\(SocketImpl\(fd=\d+, addr=\[?localhost\]?:\d+, remote=\[?localhost\]?:\d+\)\), "
            r"server=<nullptr>\)",
        )

    @skip_if_win32()
    @retry_on_connect_failures
    def test_init_pg_and_rpc_with_same_socket(self):
        # [FUNCTION DOC] Tests that a gloo process group (env:// init) and RPC (tcp init on the
        # same address and port) both initialize on one socket address in a single process, using
        # the multi-tenant store. Sets MASTER_ADDR, MASTER_PORT and USE_LIBUV in the environment
        # and removes only USE_LIBUV afterwards. Shuts down RPC and the process group.
        # KNOWN ISSUE: leaves MASTER_ADDR and MASTER_PORT set in the environment.
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        # We internally use a multi-tenant TCP store. Both PG and RPC should successfully
        # initialize even when using the same socket address.

        os.environ["USE_LIBUV"] = "1" if self._use_libuv else "0"
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=0,
            world_size=1,
        )

        backend_opts = rpc.TensorPipeRpcBackendOptions(
            init_method=f"tcp://{addr}:{port}", _transports=tp_transports()
        )
        rpc.init_rpc(
            name="worker0",
            rank=0,
            world_size=1,
            rpc_backend_options=backend_opts,
        )

        del os.environ["USE_LIBUV"]
        if "USE_LIBUV" in os.environ:
            raise AssertionError("Expected USE_LIBUV to not be in os.environ")
        rpc.shutdown()
        dist.destroy_process_group()

    @skip_if_win32()
    def test_take_over_listen_socket(self):
        # [FUNCTION DOC] Tests creating a master TCPStore from an already bound listening socket
        # descriptor (master_listen_fd), then a set and get through it.
        listen_sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.bind(("localhost", 0))
        addr, port, *_ = listen_sock.getsockname()
        listen_fd = listen_sock.detach()

        store = dist.TCPStore(
            addr,
            port,
            1,
            is_master=True,
            master_listen_fd=listen_fd,
            use_libuv=self._use_libuv,
        )

        self.assertEqual(store.libuvBackend, self._use_libuv)
        store.set("key", "value")
        self.assertEqual(b"value", store.get("key"))

    # The TCPStore has 6 keys in test_set_get. It contains the 5 keys added by
    # the user and one additional key used for coordinate all the workers.
    @property
    def num_keys_total(self):
        # [FUNCTION DOC] HELPER: Property: TCPStore reports 6 keys (5 user keys plus one internal
        # coordination key).
        return 6

    def _test_numkeys_delkeys(self, fs):
        # [FUNCTION DOC] HELPER: Checks num_keys and delete_key on a fresh store: starts at 1
        # internal key, counts added keys, drops after deletes, a get of a deleted key raises
        # RuntimeError with a 2 second timeout, and later sets are counted and readable.
        # We start off with one init key in the store to coordinate workers
        self.assertEqual(fs.num_keys(), 1)
        fs.add("key", 1)
        fs.add("key", 2)
        fs.add("key", 3)
        fs.set("key0", "value0")
        fs.add("key3", 1)
        fs.set("key1", "value1")
        self.assertEqual(fs.num_keys(), 5)
        fs.delete_key("key")
        self.assertEqual(fs.num_keys(), 4)
        fs.set_timeout(timedelta(seconds=2))
        with self.assertRaises(RuntimeError):
            fs.get("key")
        fs.delete_key("key0")
        fs.delete_key("key3")
        self.assertEqual(fs.num_keys(), 2)
        fs.set("key4", "value2")
        self.assertEqual(fs.num_keys(), 3)
        self.assertEqual(b"value1", fs.get("key1"))
        self.assertEqual(b"value2", fs.get("key4"))

    def test_numkeys_delkeys(self):
        # [FUNCTION DOC] Runs the num_keys/delete_key scenario against a freshly created store.
        self._test_numkeys_delkeys(self._create_store())

    def _create_client(self, index, addr, port, world_size):
        # [FUNCTION DOC] HELPER: Connects a client TCPStore to the given address, port and world
        # size with a 10 second timeout, reads the key set by the server, writes its own key and
        # checks compare_set returns the new value for the index.
        client_store = dist.TCPStore(
            addr,
            port,
            world_size=world_size,
            timeout=timedelta(seconds=10),
            use_libuv=self._use_libuv,
        )
        self.assertEqual(b"value", client_store.get("key"))
        client_store.set(f"new_key{index}", f"new_value{index}")
        self.assertEqual(
            f"next_value{index}".encode(),
            client_store.compare_set(
                f"new_key{index}", f"new_value{index}", f"next_value{index}"
            ),
        )

    def _multi_worker_helper(self, world_size):
        # [FUNCTION DOC] HELPER: Starts a server store for the world size (a missing world size
        # means no fixed size), sets a shared key, and runs one client per rank, or one client
        # when world size is unset.
        addr = DEFAULT_HOSTNAME
        server_store = self._create_store_with_ws(addr, world_size)
        self.assertEqual(server_store.libuvBackend, self._use_libuv)
        server_store.set("key", "value")
        port = server_store.port

        num_indices = world_size if world_size else 1
        for i in range(num_indices):
            self._create_client(i, addr, port, world_size)

    def test_multi_worker_with_fixed_world_size(self):
        # [FUNCTION DOC] Tests the multi-client scenario with a fixed world size of 5.
        self._multi_worker_helper(5)

    def test_multi_worker_with_nonfixed_world_size(self):
        # [FUNCTION DOC] Tests the multi-client scenario with no fixed world size.
        self._multi_worker_helper(None)

    def test_append(self):
        # [FUNCTION DOC] Overrides the shared test: checks the backend flag, then appends build up
        # the value potato on two keys, without the extended API guard.
        store = self._create_store()
        self.assertEqual(store.libuvBackend, self._use_libuv)
        store.set("foo", "po")
        store.append("foo", "tato")
        store.append("bar", "po")
        store.append("bar", "tato")
        self.assertEqual(b"potato", store.get("foo"))
        self.assertEqual(b"potato", store.get("bar"))

    def test_multi_set(self):
        # [FUNCTION DOC] Overrides the shared test: checks the backend flag, then multi_set stores
        # two keys.
        store = self._create_store()
        self.assertEqual(store.libuvBackend, self._use_libuv)
        store.multi_set(["foo", "bar"], ["po", "tato"])
        self.assertEqual(b"po", store.get("foo"))
        self.assertEqual(b"tato", store.get("bar"))

    def test_multi_get(self):
        # [FUNCTION DOC] Overrides the shared test: checks the backend flag, then multi_get
        # returns both values in order.
        store = self._create_store()
        self.assertEqual(store.libuvBackend, self._use_libuv)
        store.set("foo", "po")
        store.set("bar", "tato")
        v0, v1 = store.multi_get(["foo", "bar"])
        self.assertEqual(b"po", v0)
        self.assertEqual(b"tato", v1)

    def test_store_timeout_on_missing_clients(self):
        # [FUNCTION DOC] Tests that a master store with world size 2 and no clients raises
        # DistStoreError after a 2 second timeout reporting how many clients joined, and that no
        # error is raised when waiting for workers is disabled.
        with self.assertRaisesRegex(
            DistStoreError,
            r"Timed out after \d+ seconds waiting for clients. \d+/\d+ clients joined.",
        ):
            # world_size is 2 so it should timeout
            dist.TCPStore(
                "localhost",
                0,
                2,
                True,
                timeout=timedelta(seconds=2),
                use_libuv=self._use_libuv,
            )

        # when wait_for_workers is not set, then there should be no exception raised
        dist.TCPStore(
            "localhost",
            0,
            2,
            True,
            timeout=timedelta(seconds=2),
            wait_for_workers=False,
            use_libuv=self._use_libuv,
        )

    @skip_if_win32()
    def test_world_size_0_raises(self):
        # [FUNCTION DOC] Tests that creating a TCPStore client with world size 0 raises
        # ValueError.
        with self.assertRaisesRegex(ValueError, "TCPStore world size cannot be 0"):
            dist.TCPStore("localhost", 0, world_size=0, is_master=False)

    def test_agent_store(self) -> None:
        # [FUNCTION DOC] Tests agent store behaviour: a second master on the same port normally
        # raises DistNetworkError; with TORCHELASTIC_USE_AGENT_STORE set and MASTER_PORT equal to
        # the port, the second master is created and reports the same port. Sets and removes those
        # two environment variables.
        store = self._create_store()

        with self.assertRaisesRegex(
            dist.DistNetworkError,
            "The server socket has failed to listen on any local network address",
        ):
            dist.TCPStore(
                host_name="localhost",
                port=store.port,
                world_size=1,
                is_master=True,
                use_libuv=self._use_libuv,
            )

        USE_AGENT_STORE = "TORCHELASTIC_USE_AGENT_STORE"
        MASTER_PORT = "MASTER_PORT"

        os.environ[USE_AGENT_STORE] = "1"
        os.environ[MASTER_PORT] = str(store.port)
        second_server = dist.TCPStore(
            host_name="localhost",
            port=store.port,
            world_size=1,
            is_master=True,
            use_libuv=self._use_libuv,
        )
        del os.environ[USE_AGENT_STORE]
        del os.environ[MASTER_PORT]

        self.assertEqual(second_server.port, store.port)

    def test_barrier(self):
        # [FUNCTION DOC] Tests a single-participant barrier returns immediately.
        store = self._create_store()
        # Single worker barrier should return immediately
        store.barrier("test_barrier_key", 1)

    def test_barrier_with_timeout(self):
        # [FUNCTION DOC] Tests a single-participant barrier given a 10 second timeout returns
        # without error.
        store = self._create_store()
        store.barrier("test_barrier_timeout_key", 1, timedelta(seconds=10))

    def test_barrier_timeout_expires(self):
        # [FUNCTION DOC] Tests that a barrier waiting for 2 participants with only one raises
        # DistStoreError mentioning a barrier timeout after 0.1 seconds.
        store = self._create_store()
        with self.assertRaisesRegex(DistStoreError, "barrier timeout"):
            store.barrier("test_barrier_fail", 2, timedelta(seconds=0.1))

    def test_barrier_multi_worker(self):
        # [FUNCTION DOC] Tests a barrier shared by 4 threads: rank 0 uses the server store and
        # others connect their own client stores; all must pass the barrier within 10 seconds.
        server_store = self._create_store()
        world_size = 4

        def worker(rank):
            # [FUNCTION DOC] HELPER: Thread body: picks the server store for rank 0 or a new
            # client store, then enters the barrier.
            if rank == 0:
                worker_store = server_store
            else:
                worker_store = dist.TCPStore(
                    host_name=server_store.host,
                    port=server_store.port,
                    is_master=False,
                    wait_for_workers=False,
                    use_libuv=self._use_libuv,
                )
            worker_store.barrier("multi_barrier", world_size, timedelta(seconds=10))

        with ThreadPoolExecutor(max_workers=world_size) as pool:
            futures = [pool.submit(worker, i) for i in range(world_size)]
            for f in futures:
                f.result()


class LibUvTCPStoreTest(TCPStoreTest):
    # [CLASS DOC] Runs all TCPStore tests with the libuv backend selected.
    _use_libuv = True

    def _create_store(self):
        # [FUNCTION DOC] HELPER: Returns a libuv TCP store with a 300 second timeout.
        store = create_tcp_store(use_libuv=True)
        store.set_timeout(timedelta(seconds=300))
        return store

    def _create_store_with_ws(self, addr, world_size):
        # [FUNCTION DOC] HELPER: Returns a libuv TCP store for the given address and world size
        # without waiting for workers.
        return create_tcp_store(
            addr, world_size, wait_for_workers=False, use_libuv=True
        )


class PrefixTCPStoreTest(TestCase, StoreTestBase):
    # [CLASS DOC] Runs the shared store tests against a PrefixStore wrapping a TCP store. State:
    # the TCP store and the prefix string.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Creates the TCP store with a 300 second timeout and sets the
        # prefix string.
        super().setUp()
        self.tcpstore = create_tcp_store()
        self.prefix = "test_prefix"
        self.tcpstore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        # [FUNCTION DOC] HELPER: Returns a PrefixStore with the test prefix over the TCP store.
        return dist.PrefixStore(self.prefix, self.tcpstore)

    # The PrefixTCPStore has 6 keys in test_set_get. It contains the 5 keys
    # added by the user and one additional key used for coordinate all the
    # workers.
    @property
    def num_keys_total(self):
        # [FUNCTION DOC] HELPER: Property: expects 6 keys (5 user keys plus one internal key).
        return 6

    def test_underlying_non_prefix_store(self):
        # [FUNCTION DOC] Tests that the underlying non-prefix store of a prefix store, also when
        # prefix stores are nested twice, is the original TCP store.
        store = self._create_store()
        wrapped_store = dist.PrefixStore(
            self.prefix, dist.PrefixStore(self.prefix, store)
        )
        self.assertEqual(self.tcpstore, store._underlying_non_prefix_store)
        self.assertEqual(self.tcpstore, wrapped_store._underlying_non_prefix_store)


class MyPythonStore(dist.Store):
    # [CLASS DOC] HELPER: Minimal dictionary backed Python Store used to test the native
    # trampoline. State: a mapping from key to bytes. Its methods validate argument types by
    # raising AssertionError.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the base Store and an empty mapping for the data.
        super().__init__()
        self.store = {}

    def set(self, key, value):
        # [FUNCTION DOC] HELPER: Stores the value under the key; raises AssertionError unless the
        # key is a string or bytes and the value is bytes.
        if not isinstance(key, (str, bytes)):
            raise AssertionError("Expected set to be called with string key")
        if type(value) is not bytes:
            raise AssertionError("Expected set to be called with bytes value")
        self.store[key] = value

    def get(self, key):
        # [FUNCTION DOC] HELPER: Returns the stored bytes for the key, or empty bytes if missing;
        # raises AssertionError if a stored value is not bytes.
        value = self.store.get(key, b"")
        if type(value) is not bytes:
            raise AssertionError("Expected get to return bytes value")
        return value

    def add(self, key, value):
        # [FUNCTION DOC] HELPER: Adds the integer to the numeric value stored (zero if missing),
        # stores the result as decimal text bytes and returns the new integer.
        new = int(self.store.get(key, 0)) + value
        self.set(key, bytes(str(new).encode("utf-8")))
        return new

    def compare_set(self, key, expected, newValue):
        # [FUNCTION DOC] HELPER: Raises AssertionError unless both the expected and new values are
        # bytes. If the key is missing or equals expected, stores and returns the new value;
        # otherwise returns the current value.
        if type(expected) is not bytes:
            raise AssertionError("compare_set::expected not bytes")
        if type(newValue) is not bytes:
            raise AssertionError("compare_set::newValue not bytes")

        val = self.store.get(key, None)
        if expected == val or val is None:
            val = self.store[key] = newValue
        return val

    def clone(self) -> "MyPythonStore":
        # [FUNCTION DOC] HELPER: Returns the same object, so a clone shares all data.
        return self


class PythonStoreTest(TestCase):
    # [CLASS DOC] Tests a Python-defined Store through the native code path.
    def test_set_get(self):
        # [FUNCTION DOC] Runs the native set/get test helper on a MyPythonStore so calls go
        # through the C++ trampoline.
        # If we were to inherit from StoreTestBase and try to use
        # its test_set_get function, we would exercise the Python
        # API directly, instead of going through the C++ trampoline.
        # We care about testing the C++ trampoline, so run the
        # equivalent of StoreTestBase.test_set_get from C++.
        # See `torch/csrc/distributed/c10d/init.cpp` for the definition
        # of this test function.
        dist._test_python_store(MyPythonStore())


class RendezvousTest(TestCase):
    # [CLASS DOC] Tests generic errors of the rendezvous function.
    def test_unknown_handler(self):
        # [FUNCTION DOC] Tests that an unknown URL scheme raises RuntimeError saying there is no
        # rendezvous handler.
        with self.assertRaisesRegex(RuntimeError, "^No rendezvous handler"):
            dist.rendezvous("invalid://")

    def test_url_with_node_params(self):
        # [FUNCTION DOC] Tests that a URL with rank and world size parameters plus explicit rank
        # and size arguments raises AssertionError mentioning node-specific arguments.
        with self.assertRaisesRegex(AssertionError, "has node-specific arguments"):
            dist.rendezvous("file://foo?rank=12&world_size=16", 12, 16)


class RendezvousEnvTest(TestCase):
    # [CLASS DOC] Tests rendezvous through the env:// handler.
    @retry_on_connect_failures
    def test_nominal(self):
        # [FUNCTION DOC] Tests env:// rendezvous with world size 1 and rank 0 on a free port: the
        # first yielded tuple has rank 0 and size 1, and its store can set and get a value. Sets
        # WORLD_SIZE, MASTER_ADDR, MASTER_PORT and RANK in the environment and does not restore
        # them. Retried on port clashes.
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())

        # Single rank
        os.environ["RANK"] = "0"
        gen0 = dist.rendezvous("env://")
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(1, size0)

        store0.set("key0", "value0")

        # check with get
        self.assertEqual(b"value0", store0.get("key0"))


class RendezvousFileTest(TestCase):
    # [CLASS DOC] Tests rendezvous through the file:// handler.
    def test_common_errors(self):
        # [FUNCTION DOC] Tests that file:// URLs with a missing path, missing rank or missing
        # world size each raise ValueError with a matching message when the generator is advanced.
        with self.assertRaisesRegex(ValueError, "path missing"):
            gen = dist.rendezvous("file://?rank=0&world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            gen = dist.rendezvous("file:///tmp/foo?world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "size parameter missing"):
            gen = dist.rendezvous("file:///tmp/foo?rank=0")
            next(gen)

    def test_nominal(self):
        # [FUNCTION DOC] Tests two ranks of world size 2 rendezvous on the same temporary file:
        # ranks and sizes are returned and each store sees the other store's value.
        with tempfile.NamedTemporaryFile(delete=False) as file:
            url = f"file:///{file.name.replace(os.path.sep, '/')}?world_size=2"
            gen0 = dist.rendezvous(url + "&rank=0")
            store0, rank0, size0 = next(gen0)
            self.assertEqual(0, rank0)
            self.assertEqual(2, size0)
            gen1 = dist.rendezvous(url + "&rank=1")
            store1, rank1, size1 = next(gen1)
            self.assertEqual(1, rank1)
            self.assertEqual(2, size1)

            # Set value on both stores
            store0.set("key0", "value0")
            store1.set("key1", "value1")

            # Cross check with get
            self.assertEqual(b"value0", store1.get("key0"))
            self.assertEqual(b"value1", store0.get("key1"))


@skip_if_win32()
class RendezvousTCPTest(TestCase):
    # [CLASS DOC] Tests rendezvous through the tcp:// handler (skipped on Windows).
    def create_tcp_url(self):
        # [FUNCTION DOC] HELPER: Returns a tcp URL for localhost on a free port with world size 1.
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()
        url = f"tcp://{addr}:{port:d}?world_size=1"
        return url

    def test_common_errors(self):
        # [FUNCTION DOC] Tests that tcp:// URLs with a missing port, rank or world size each raise
        # ValueError with a matching message.
        with self.assertRaisesRegex(ValueError, "port number missing"):
            gen = dist.rendezvous("tcp://127.0.0.1?rank=0&world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            gen = dist.rendezvous("tcp://127.0.0.1:23456?world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "size parameter missing"):
            gen = dist.rendezvous("tcp://127.0.0.1:23456?rank=0")
            next(gen)

    def test_dns_timeout(self):
        # [FUNCTION DOC] Tests that a rank 0 rendezvous to an unresolvable host with world size 2
        # and a 1 second timeout raises DistNetworkError (a DistError) naming the host.
        with self.assertRaisesRegex(
            DistNetworkError, "client socket has timed out after.*dnsnotexist"
        ) as manager:
            gen = dist.rendezvous(
                "tcp://dnsnotexist:23456?world_size=2&rank=0",
                timeout=timedelta(seconds=1),
            )
            next(gen)
        self.assertTrue(isinstance(manager.exception, DistError))

    @retry_on_connect_failures
    def test_nominal(self):
        # [FUNCTION DOC] Tests tcp:// rendezvous with rank 0 and world size 1: returns rank 0 and
        # size 1 and a working store. Retried on port clashes.
        url = self.create_tcp_url()
        gen0 = dist.rendezvous(url + "&rank=0")
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(1, size0)

        # Set value on the single store
        store0.set("key0", "value0")

        # check with get
        self.assertEqual(b"value0", store0.get("key0"))

    @retry_on_connect_failures(connect_errors=(CONNECT_TIMEOUT, ADDRESS_IN_USE))
    def test_tcp_store_timeout_set(self):
        # [FUNCTION DOC] Tests that the timeout set on the store after rendezvous applies: a get
        # of a missing key fails with DistStoreError reporting a 100ms wait timeout, well under 10
        # seconds. Retried on connect timeout and address in use errors.
        url = self.create_tcp_url()
        test_store_timeout = timedelta(seconds=0.1)
        gen0 = dist.rendezvous(url + "&rank=0", timeout=timedelta(seconds=10))
        store0, _, _ = next(gen0)
        store0.set_timeout(test_store_timeout)
        # this should time out in 0.1s. If the timeout passed into rendezvous was
        # not respected, it will take much longer to timeout.
        start = time.time()
        with self.assertRaisesRegex(
            DistStoreError, "wait timeout after 100ms, keys: /nonexistent key"
        ):
            store0.get("nonexistent key")

        end = time.time()
        time_diff = end - start
        self.assertGreater(10, time_diff)

    def test_tcp_store_timeout_doest_break_client(self):
        # [FUNCTION DOC] Tests that after a get times out with DistStoreError after 100ms, the
        # store stays usable: a later set and get return the value, all within 10 seconds.
        url = self.create_tcp_url()
        test_store_timeout = timedelta(seconds=0.1)
        gen0 = dist.rendezvous(url + "&rank=0", timeout=timedelta(seconds=10))
        store0, _, _ = next(gen0)
        store0.set_timeout(test_store_timeout)
        # this should time out in 10s. If the timeout passed into rendezvous was
        # not respected, it will take much longer to timeout.
        start = time.time()
        with self.assertRaisesRegex(
            DistStoreError, "wait timeout after 100ms, keys: /the_key"
        ):
            store0.get("the_key")

        store0.set("the_key", "x")

        self.assertEqual(b"x", store0.get("the_key"))

        end = time.time()
        time_diff = end - start
        self.assertGreater(10, time_diff)

    def test_tcp_store_url_with_libuv(self):
        # [FUNCTION DOC] Tests that a tcp URL with the libuv option set gives a store using the
        # libuv backend.
        url = self.create_tcp_url()
        gen0 = dist.rendezvous(url + "&rank=0&use_libuv=1")
        store0, _, _ = next(gen0)
        self.assertTrue(store0.libuvBackend)

    def test_agent_store_ignored_for_other_address(self):
        # [FUNCTION DOC] Tests that with the elastic agent store environment flag set and
        # MASTER_ADDR and MASTER_PORT pointing at a different port than the rendezvous URL,
        # rendezvous still starts its own server and the store works.
        # torchrun's elastic agent exports TORCHELASTIC_USE_AGENT_STORE into
        # every worker, meaning "the agent serves a store at
        # MASTER_ADDR:MASTER_PORT". _create_c10d_store used to honor it for any
        # address it was handed, so a private store on a caller-chosen port
        # became client-only on every rank, nobody ever started a server and
        # every rank blocked in connect-retry until the store timeout.
        url = self.create_tcp_url()
        env = {
            "TORCHELASTIC_USE_AGENT_STORE": "True",
            "MASTER_ADDR": DEFAULT_HOSTNAME,
            "MASTER_PORT": str(common.find_free_port()),
        }
        with mock.patch.dict(os.environ, env):
            gen0 = dist.rendezvous(url + "&rank=0", timeout=timedelta(seconds=10))
            store0, _, _ = next(gen0)
        store0.set("key0", "value0")
        self.assertEqual(b"value0", store0.get("key0"))

    def test_agent_store_honored_for_master_address(self):
        # [FUNCTION DOC] Tests that when the URL address matches MASTER_ADDR and MASTER_PORT and
        # the agent store flag is set, rendezvous acts only as a client and raises
        # DistNetworkError (nothing listens on the port, 1 second timeout).
        # The flip side: at MASTER_ADDR:MASTER_PORT the agent is the server, so
        # every rank must stay a client. Point at a port nothing listens on and
        # check we fail to connect rather than silently starting a server.
        port = common.find_free_port()
        url = f"tcp://{DEFAULT_HOSTNAME}:{port:d}?world_size=1&rank=0"
        env = {
            "TORCHELASTIC_USE_AGENT_STORE": "True",
            "MASTER_ADDR": DEFAULT_HOSTNAME,
            "MASTER_PORT": str(port),
        }
        with mock.patch.dict(os.environ, env):
            with self.assertRaises(DistNetworkError):
                next(dist.rendezvous(url, timeout=timedelta(seconds=1)))


class DummyStore(dist.Store):
    # [CLASS DOC] HELPER: Python Store that records calls to append, multi_get and multi_set in
    # lists and claims extended API support. State: lists of calls and a queue of results for
    # multi_get.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the call record lists and result list, then the base
        # Store.
        self.appends = []
        self.multi_sets = []
        self.multi_gets = []
        self.multi_get_res = []
        super().__init__()

    def append(self, key, value):
        # [FUNCTION DOC] HELPER: Records the key and value pair; returns nothing.
        self.appends.append((key, value))

    def multi_get(self, keys):
        # [FUNCTION DOC] HELPER: Records the keys and returns (and removes) the first prepared
        # result; raises IndexError if none.
        self.multi_gets.append(keys)
        return self.multi_get_res.pop(0)

    def multi_set(self, keys, values):
        # [FUNCTION DOC] HELPER: Records the keys and values as a pair; returns nothing.
        self.multi_sets.append((keys, values))

    def has_extended_api(self):
        # [FUNCTION DOC] HELPER: Always reports that the extended API is supported.
        return True


class TestPythonStore(TestCase):
    # [CLASS DOC] Tests optional extended methods of Python-defined stores and their forwarding
    # through PrefixStore.
    def test_optional_methods_fail(self):
        # [FUNCTION DOC] Tests that a Store subclass without overrides reports no extended API and
        # that append, multi_get and multi_set raise RuntimeError saying not implemented.
        class TestStore(dist.Store):
            # [CLASS DOC] HELPER: Empty Store subclass with no optional methods.
            pass

        store = TestStore()
        self.assertFalse(store.has_extended_api())
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.append("foo", "bar")
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_get(["foo", "bar"])
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_set(["foo", "bar"], [b"v", b"v"])

    def test_has_extended_api_passthrough(self):
        # [FUNCTION DOC] Tests that a PrefixStore over a store with no extended API also reports
        # none and raises the same not implemented errors.
        class TestStore(dist.Store):
            # [CLASS DOC] HELPER: Empty Store subclass with no optional methods.
            pass

        test_store = TestStore()
        store = dist.PrefixStore("p", test_store)
        self.assertFalse(store.has_extended_api())
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.append("foo", "bar")
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_get(["foo", "bar"])
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_set(["foo", "bar"], [b"v", b"v"])

    def test_has_extended_api_roundtrip(self):
        # [FUNCTION DOC] Tests that a PrefixStore reports the extended API as available when the
        # wrapped store does.
        store = DummyStore()
        prefix = dist.PrefixStore("p", store)
        self.assertTrue(prefix.has_extended_api())

    def test_append_roundtrip(self):
        # [FUNCTION DOC] Tests that append through a PrefixStore reaches the wrapped store once
        # with the key prefixed and the value as bytes.
        store = DummyStore()
        prefix = dist.PrefixStore("p", store)
        prefix.append("foo", "bar")
        self.assertEqual(1, len(store.appends))
        self.assertEqual(("p/foo", b"bar"), store.appends[0])

    def test_multi_get_roundtrip(self):
        # [FUNCTION DOC] Tests that multi_get through a PrefixStore passes prefixed keys to the
        # wrapped store once and returns its result unchanged.
        store = DummyStore()
        prefix = dist.PrefixStore("p", store)
        store.multi_get_res.append([b"x", b"y"])
        res = prefix.multi_get(["foo", "bar"])
        self.assertEqual(1, len(store.multi_gets))
        self.assertEqual(["p/foo", "p/bar"], store.multi_gets[0])
        self.assertEqual([b"x", b"y"], res)

    def test_multi_set_roundtrip(self):
        # [FUNCTION DOC] Tests that multi_set through a PrefixStore passes prefixed keys and the
        # values unchanged to the wrapped store once.
        store = DummyStore()
        prefix = dist.PrefixStore("p", store)
        prefix.multi_set(["foo", "bar"], [b"x", b"y"])
        self.assertEqual(1, len(store.multi_sets))
        self.assertEqual(["p/foo", "p/bar"], store.multi_sets[0][0])
        self.assertEqual([b"x", b"y"], store.multi_sets[0][1])

    def test_extended_methods_fallbacks(self):
        # [FUNCTION DOC] Tests that over a store without the extended API, PrefixStore append and
        # multi_set/multi_get still work via fallbacks built on get and set, though no extended
        # API is reported.
        test_store = MyPythonStore()
        store = dist.PrefixStore("p", test_store)
        self.assertFalse(store.has_extended_api())
        store.append("foo", b"po")
        store.append("foo", b"tato")
        self.assertEqual(store.get("foo"), b"potato")

        store.multi_set(["a", "b"], [b"c", b"d"])
        self.assertEqual(store.multi_get(["a", "b", "foo"]), [b"c", b"d", b"potato"])


class TestMultiThreadedWait(MultiThreadedTestCase):
    # [CLASS DOC] Tests wait across two threads (ranks) of the in-process threaded process group
    # on several store types. State: shared stores created once at class definition time.
    file_store = dist.FileStore(tempfile.NamedTemporaryFile(delete=False).name, 1)  # noqa: SIM115
    hash_store = dist.HashStore()

    tcp_store = create_tcp_store(use_libuv=False)
    tcp_store_uv = create_tcp_store(use_libuv=True)

    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property: the number of threads, 2.
        return 2

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp then starts the rank threads.
        super().setUp()
        self._spawn_threads()

    def _test_wait(self, store):
        # [FUNCTION DOC] HELPER: Sets a 2 second store timeout. Rank 0 waits for a key and checks
        # its value; rank 1 sets the key.
        store.set_timeout(timedelta(seconds=2))
        if dist.get_rank() == 0:
            store.wait(["key1"])
            self.assertEqual(b"value1", store.get("key1"))
        if dist.get_rank() == 1:
            store.set("key1", "value1")

    def test_wait_hash_store(self):
        # [FUNCTION DOC] Runs the two rank wait scenario on the shared HashStore.
        self._test_wait(self.hash_store)

    def test_wait_file_store(self):
        # [FUNCTION DOC] Runs the two rank wait scenario on the shared FileStore.
        self._test_wait(self.file_store)

    def test_wait_prefix_file_store(self):
        # [FUNCTION DOC] Runs the two rank wait scenario on a PrefixStore over the shared
        # FileStore.
        store = dist.PrefixStore("pre", self.file_store)
        self._test_wait(store)

    def _test_wait_tcp_store(self, master_store):
        # [FUNCTION DOC] HELPER: Rank 0 uses the given master store; other ranks make a non-libuv
        # client store to it. Runs the wait scenario on the store and then on a PrefixStore over
        # it.
        store = (
            master_store
            if dist.get_rank() == 0
            else dist.TCPStore(
                host_name=master_store.host,
                port=master_store.port,
                is_master=False,
                wait_for_workers=False,
                use_libuv=False,
            )
        )
        self._test_wait(store)

        prefix_store = dist.PrefixStore("pre", store)
        self._test_wait(prefix_store)

    def test_wait_tcp_store(self):
        # [FUNCTION DOC] Runs the wait scenario on the shared legacy TCP store.
        self._test_wait_tcp_store(self.tcp_store)

    def test_wait_tcp_store_uv(self):
        # [FUNCTION DOC] Runs the wait scenario on the shared libuv TCP store.
        self._test_wait_tcp_store(self.tcp_store_uv)


instantiate_parametrized_tests(TestMultiThreadedWait)


@skip_if_win32()
class TimeoutTest(TestCase):
    # [CLASS DOC] Tests that a signal does not break a blocking store wait (skipped on Windows).
    def tearDown(self):
        # [FUNCTION DOC] HELPER: After the base teardown, sets the SIGUSR1 handler back to ignore.
        import signal

        super().tearDown()
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)

    def test_interrupt_doesnt_break_wait(self):
        # [FUNCTION DOC] Tests wait interrupted by a signal: thread 0 sleeps 4 seconds then sets a
        # key; thread 1 waits up to 10 seconds. After 1 second SIGUSR1 (with a no-op handler) is
        # sent to thread 1. Both threads must finish without error. Installs a process wide signal
        # handler.
        import signal

        rank_res = [None, None]

        def run(rank, my_store):
            # [FUNCTION DOC] HELPER: Thread body: rank 0 sleeps then sets the key; other ranks
            # wait for it. Stores True or the caught exception in the shared result list, then
            # sleeps 1 second.
            nonlocal rank_res
            try:
                if rank == 0:
                    time.sleep(4)
                    my_store.set("foo", "bar")
                else:
                    my_store.wait(["foo"], datetime.timedelta(seconds=10))
                rank_res[rank] = True
            except BaseException as e:  # noqa: E261
                rank_res[rank] = e
            time.sleep(1)

        rank0_store = dist.TCPStore(
            host_name=DEFAULT_HOSTNAME,
            port=0,
            world_size=2,
            is_master=True,
            wait_for_workers=False,
        )
        rank1_store = dist.TCPStore(
            host_name=DEFAULT_HOSTNAME,
            port=rank0_store.port,
            world_size=2,
            is_master=False,
            wait_for_workers=False,
        )

        threads = []
        for i in range(2):
            t = threading.Thread(
                target=run,
                args=(
                    i,
                    [rank0_store, rank1_store][i],
                ),
            )
            t.start()
            threads.append(t)

        def handler(a, b):
            # [FUNCTION DOC] HELPER: Signal handler that does nothing.
            pass

        signal.signal(signal.SIGUSR1, handler)
        time.sleep(1)
        signal.pthread_kill(threads[1].ident, signal.SIGUSR1)

        for t in threads:
            t.join()
        self.assertTrue(rank_res[0], "rank0")
        self.assertTrue(rank_res[1], "rank1")


class InitPgWithNonUvStore(TestCase):
    # [CLASS DOC] Tests creating a process group on the legacy non-libuv TCPStore backend selected
    # by a URL parameter or the USE_LIBUV environment variable. Teardown clears related
    # environment variables.
    """
    This test shows how to use the legacy TCPStore (non-libuv) backend since libuv is now
    the default backend.
    """

    def tearDown(self):
        # [FUNCTION DOC] HELPER: After the base teardown, removes USE_LIBUV, MASTER_ADDR and
        # MASTER_PORT from the environment.
        super().tearDown()
        os.environ.pop("USE_LIBUV", None)
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)

    def test_with_url_param(self):
        # [FUNCTION DOC] Tests gloo init with a tcp init method on a free port and the libuv
        # option off, then checks the store.
        port = common.find_free_port()
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=1,
            init_method=f"tcp://{DEFAULT_HOSTNAME}:{port}?use_libuv=0",
        )
        self._run_test()

    def test_with_env_var(self):
        # [FUNCTION DOC] Tests gloo init with env:// where USE_LIBUV is 0 and the master address
        # and port are set, then checks the store.
        port = common.find_free_port()
        os.environ["USE_LIBUV"] = "0"
        os.environ["MASTER_ADDR"] = DEFAULT_HOSTNAME
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=0, world_size=1, init_method="env://")
        self._run_test()

    def _run_test(self):
        # [FUNCTION DOC] HELPER: Checks the world group's store is a PrefixStore, unwraps nested
        # PrefixStores, and requires a TCPStore without the libuv backend. Destroys the process
        # group.
        pg = dist.group.WORLD
        store = c10d._get_process_group_store(pg)
        self.assertTrue(isinstance(store, dist.PrefixStore))
        # c10d does multiple levels of wrapping
        while isinstance(store, dist.PrefixStore):
            store = store.underlying_store
        self.assertTrue(isinstance(store, dist.TCPStore))
        self.assertFalse(store.libuvBackend)
        dist.destroy_process_group()


class TestClientProtocol(TestCase):
    # [CLASS DOC] Tests the wire protocol used by a TCPStore client when it connects.
    def test_client_connect(self) -> None:
        # [FUNCTION DOC] Runs a fake server socket thread; creates a client TCPStore with world
        # size 2 and checks the server received the handshake. Joins the thread.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

        def listen() -> None:
            # [FUNCTION DOC] HELPER: Fake server: listens, accepts one connection, checks it
            # received a validate message with the expected magic number and a ping message
            # carrying this process id, echoes the nonce, and closes.
            sock.listen()
            conn, _ = sock.accept()

            # VALIDATE
            # 0x3C85F7CE
            self.assertEqual(conn.recv(5), b"\x00\xce\xf7\x85\x3c")

            # PING
            data = conn.recv(5)
            self.assertEqual(data[0], 13)
            nonce = struct.unpack("i", data[1:])[0]
            self.assertEqual(nonce, os.getpid())

            # send PING nonce response
            conn.sendall(data[1:])

            conn.close()

        thread = threading.Thread(target=listen)
        thread.start()

        dist.TCPStore(
            host_name="localhost",
            port=port,
            world_size=2,
            is_master=False,
            timeout=timedelta(seconds=2),
            wait_for_workers=False,
        )

        thread.join()


if __name__ == "__main__":
    if device_type != "cpu":
        if getattr(torch.get_device_module(device_type), "_initialized", False):
            raise AssertionError(
                f"test_distributed must not have initialized {device_type} context on main process"
            )
    run_tests()
